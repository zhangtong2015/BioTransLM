#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文数据集加载器
支持多种格式的中文训练数据
"""

import os
import json
import csv
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

# 导入新的无损Tokenizer
from chinese_tokenizer import ChineseTokenizer

logger = logging.getLogger(__name__)

def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'cp936']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.readline()
            return enc
        except UnicodeDecodeError:
            continue
    return 'utf-8'

def load_jsonl(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """加载JSONL格式文件"""
    data = []
    enc = detect_encoding(file_path) if encoding == 'auto' else encoding
    logger.info(f"使用编码 {enc} 读取 {file_path}")
    
    with open(file_path, 'r', encoding=enc) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {i} 行解析失败: {e}")
                continue
    return data

def load_csv(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """加载CSV格式文件"""
    data = []
    enc = detect_encoding(file_path) if encoding == 'auto' else encoding
    logger.info(f"使用编码 {enc} 读取 {file_path}")
    
    with open(file_path, 'r', encoding=enc, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

class ChineseTextDataset(Dataset):
    """
    中文文本数据集通用加载器
    
    支持的数据集格式:
    - DeepSeek-R1蒸馏格式: input, content, reasoning_content
    - 对话问答格式: question, answer
    - 情感分类格式: text, label
    """
    
    def __init__(self,
                 data: Union[str, Path, List[Dict]],
                 file_type: str = 'auto',
                 max_seq_length: int = 128,
                 vocab_size: int = 21504,
                 split: str = 'train',
                 train_ratio: float = 0.9,
                 encoding: str = 'auto'):
        """
        初始化数据集
        
        Args:
            data: 数据文件路径 或 已加载的数据列表
            file_type: 文件类型: auto/jsonl/csv
            max_seq_length: 最大序列长度
            vocab_size: 词汇表大小
            split: 数据集划分: train/eval
            train_ratio: 训练集比例
            encoding: 文件编码: auto/utf-8/gbk
        """
        self.file_path = data if isinstance(data, (str, Path)) else None
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.split = split
        
        # 初始化tokenizer
        self._tokenizer = ChineseTokenizer(vocab_size=self.vocab_size)
        
        # 直接传入数据列表 或 从文件加载
        if isinstance(data, list):
            self.data = data
            self.file_type = 'list'
        else:
            # 确定文件类型
            if file_type == 'auto':
                if str(data).endswith('.jsonl'):
                    self.file_type = 'jsonl'
                elif str(data).endswith('.csv'):
                    self.file_type = 'csv'
                else:
                    self.file_type = 'jsonl'
            else:
                self.file_type = file_type
            
            # 加载数据
            self.data = self._load_data(encoding)
        
        # 数据集划分
        if split in ['train', 'eval']:
            random.seed(42)
            random.shuffle(self.data)
            split_idx = int(len(self.data) * train_ratio)
            if split == 'train':
                self.data = self.data[:split_idx]
            else:
                self.data = self.data[split_idx:]
        
        logger.info(f"加载 {split} 数据集: {len(self.data)} 条样本")
    
    def _load_data(self, encoding: str) -> List[Dict[str, Any]]:
        """加载原始数据"""
        if self.file_type == 'jsonl':
            return load_jsonl(self.file_path, encoding)
        elif self.file_type == 'csv':
            return load_csv(self.file_path, encoding)
        else:
            raise ValueError(f"不支持的文件类型: {self.file_type}")
    
    def _tokenize_simple(self, text: str) -> List[int]:
        """使用无损中文Tokenizer进行编码"""
        # 编码并确保长度
        tokens = self._tokenizer.encode(text)
        
        # 截断或填充到最大长度
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        elif len(tokens) < self.max_seq_length:
            # 使用<PAD>填充
            tokens = tokens + [0] * (self.max_seq_length - len(tokens))
        
        return tokens[:self.max_seq_length]
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理单个样本"""
        input_text = ''
        output_text = ''
        
        # DeepSeek-R1格式
        if 'input' in item:
            input_text = item.get('input', '')
        if 'content' in item:
            output_text = item.get('content', '')
        if 'reasoning_content' in item and item.get('reasoning_content'):
            output_text = item.get('reasoning_content', '') + '\n' + output_text
        
        # 问答格式
        if 'question' in item:
            input_text = item.get('question', '')
        if 'answer' in item:
            output_text = item.get('answer', '')
        
        # 分类格式
        if 'text' in item:
            input_text = item.get('text', '')
        if 'label' in item:
            output_text = item.get('label', '')
        
        full_text = input_text + ' ' + output_text
        tokens = self._tokenize_simple(full_text)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(tokens), dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long),  # CLM训练: labels = input_ids
            'raw_text': full_text[:100]
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        if isinstance(idx, slice):
            return [self._process_item(item) for item in self.data[idx]]
        item = self.data[idx]
        return self._process_item(item)

def get_dataloader(file_path: str,
                   batch_size: int = 8,
                   max_seq_length: int = 128,
                   vocab_size: int = 21504,
                   num_workers: int = 0,
                   shuffle: bool = True,
                   split: str = 'train',
                   encoding: str = 'auto') -> DataLoader:
    dataset = ChineseTextDataset(
        data=file_path,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size,
        split=split,
        encoding=encoding
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def analyze_dataset(file_path: str, num_samples: int = 5) -> Dict[str, Any]:
    dataset = ChineseTextDataset(data=file_path, max_seq_length=32, vocab_size=21504)
    
    stats = {
        'total_samples': len(dataset),
        'samples': []
    }
    
    for i in range(min(num_samples, len(dataset))):
        item = dataset.data[i]
        stats['samples'].append({
            'index': i,
            'fields': list(item.keys()),
            'preview': str(item)[:200]
        })
    
    return stats
