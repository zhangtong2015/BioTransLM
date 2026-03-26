#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理工具 - 数据集加载、格式检测、统计分析
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据集处理工具类"""
    
    # 支持的模板类型
    TEMPLATES = {
        '小说': 'novel',
        '蒸馏数据': 'distillation',
        '一问一答': 'qna'
    }
    
    def __init__(self):
        self.supported_extensions = ['.jsonl', '.csv', '.txt', '.json']
    
    # ========== 文件加载 ==========
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            DataFrame 格式的数据
        """
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()
        
        if ext == '.jsonl':
            return self._load_jsonl(file_path)
        elif ext == '.csv':
            return self._load_csv(file_path)
        elif ext == '.txt':
            return self._load_txt(file_path)
        elif ext == '.json':
            return self._load_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")
    
    def _load_jsonl(self, file_path: str) -> pd.DataFrame:
        """加载 JSONL 文件"""
        data = []
        encoding = self._detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行解析失败：{e}")
                    continue
        
        return pd.DataFrame(data)
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """加载 CSV 文件"""
        encoding = self._detect_encoding(file_path)
        
        # 尝试不同的分隔符
        for sep in [',', '\t', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                return df
            except Exception:
                continue
        
        raise ValueError("无法解析 CSV 文件")
    
    def _load_txt(self, file_path: str) -> pd.DataFrame:
        """加载 TXT 文件（假设每行是一个样本）"""
        encoding = self._detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        # 创建简单的 DataFrame
        data = [{'text': line.strip(), 'index': i} for i, line in enumerate(lines) if line.strip()]
        return pd.DataFrame(data)
    
    def _load_json(self, file_path: str) -> pd.DataFrame:
        """加载 JSON 文件"""
        encoding = self._detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # 如果是列表，直接转换
        if isinstance(data, list):
            return pd.DataFrame(data)
        # 如果是字典，尝试提取数据
        elif isinstance(data, dict):
            # 常见格式：{'data': [...]}
            if 'data' in data:
                return pd.DataFrame(data['data'])
            # 或者 {'items': [...]}
            elif 'items' in data:
                return pd.DataFrame(data['items'])
            else:
                # 将整个字典作为一行
                return pd.DataFrame([data])
        else:
            raise ValueError("无法解析 JSON 文件")
    
    def _detect_encoding(self, file_path: str) -> str:
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
    
    # ========== 格式检测 ==========
    
    def detect_format(self, df: pd.DataFrame) -> str:
        """
        检测数据格式类型
        
        Returns:
            格式类型：'deepseek', 'qna', 'classification', 'conversation', 'unknown'
        """
        if df.empty:
            return 'unknown'
        
        columns = set(df.columns)
        
        # DeepSeek-R1 蒸馏格式
        if 'input' in columns or 'content' in columns or 'reasoning_content' in columns:
            return 'deepseek'
        
        # 问答格式
        if 'question' in columns and 'answer' in columns:
            return 'qna'
        
        # 分类格式
        if 'text' in columns and 'label' in columns:
            return 'classification'
        
        # 对话格式
        if 'messages' in columns or 'conversation' in columns:
            return 'conversation'
        
        # 简单文本格式
        if 'text' in columns or 'content' in columns:
            return 'text_only'
        
        return 'unknown'
    
    def get_format_description(self, format_type: str) -> str:
        """获取格式描述"""
        descriptions = {
            'deepseek': 'DeepSeek-R1 蒸馏数据（包含 input、content、reasoning_content）',
            'qna': '问答数据（包含 question 和 answer）',
            'classification': '分类数据（包含 text 和 label）',
            'conversation': '对话数据（包含 messages 或 conversation）',
            'text_only': '纯文本数据（仅包含 text 或 content）',
            'unknown': '未知格式'
        }
        return descriptions.get(format_type, '未知格式')
    
    # ========== 数据统计 ==========
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算数据统计信息"""
        if df.empty:
            return {'error': '数据为空'}
        
        stats = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'missing_values': {},
            'text_stats': {},
            'field_stats': {}
        }
        
        # 缺失值统计
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                stats['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(df) * 100, 2)
                }
        
        # 文本字段统计
        for col in df.columns:
            if df[col].dtype == 'object':
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    text_lengths = non_null.astype(str).apply(len)
                    stats['text_stats'][col] = {
                        'avg_length': round(text_lengths.mean(), 2),
                        'min_length': int(text_lengths.min()),
                        'max_length': int(text_lengths.max()),
                        'unique_count': int(non_null.nunique())
                    }
        
        # 字段值分布（对于分类字段）
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                value_counts = df[col].value_counts()
                if len(value_counts) <= 20:  # 只显示类别数较少的字段
                    stats['field_stats'][col] = {
                        'num_categories': int(len(value_counts)),
                        'top_values': value_counts.head(5).to_dict()
                    }
        
        return stats
    
    # ========== 数据预览 ==========
    
    def preview_data(self, df: pd.DataFrame, limit: int = 100) -> List[Dict[str, Any]]:
        """获取数据预览"""
        preview_df = df.head(limit)
        
        # 转换为字典列表，处理 NaN 值
        preview = []
        for _, row in preview_df.iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (list, dict)):
                    row_dict[col] = str(val)[:200]  # 限制长度
                else:
                    row_dict[col] = val
            preview.append(row_dict)
        
        return preview
    
    # ========== 数据集划分 ==========
    
    def split_dataset(self, 
                     df: pd.DataFrame, 
                     train_ratio: float = 0.9,
                     shuffle: bool = True,
                     seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        划分训练集和评估集
        
        Args:
            df: 原始数据
            train_ratio: 训练集比例
            shuffle: 是否打乱
            seed: 随机种子
            
        Returns:
            (train_df, eval_df)
        """
        import random
        
        if shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        eval_df = df.iloc[split_idx:].reset_index(drop=True)
        
        logger.info(f"数据集划分：训练集 {len(train_df)} 条，评估集 {len(eval_df)} 条")
        
        return train_df, eval_df
    
    # ========== 内置模板 ==========
    
    def generate_template(self, template_type: str, num_samples: int = 10) -> pd.DataFrame:
        """
        生成内置模板数据
        
        Args:
            template_type: 模板类型（'小说'、'蒸馏数据'、'一问一答'）
            num_samples: 样本数量
            
        Returns:
            DataFrame 格式的模板数据
        """
        if template_type == '小说' or template_type == 'novel':
            return self._generate_novel_template(num_samples)
        elif template_type == '蒸馏数据' or template_type == 'distillation':
            return self._generate_distillation_template(num_samples)
        elif template_type == '一问一答' or template_type == 'qna':
            return self._generate_qna_template(num_samples)
        else:
            raise ValueError(f"未知的模板类型：{template_type}")
    
    def _generate_novel_template(self, num_samples: int) -> pd.DataFrame:
        """生成小说模板"""
        data = []
        for i in range(num_samples):
            data.append({
                'text': f'这是第{i+1}段小说内容示例。在这里放入您的小说文本数据...',
                'title': f'章节{i+1}',
                'source': '示例来源'
            })
        return pd.DataFrame(data)
    
    def _generate_distillation_template(self, num_samples: int) -> pd.DataFrame:
        """生成蒸馏数据模板"""
        data = []
        for i in range(num_samples):
            data.append({
                'input': f'这是第{i+1}个问题的输入示例',
                'content': f'这是对应的输出内容示例',
                'reasoning_content': f'这是推理过程示例（可选）'
            })
        return pd.DataFrame(data)
    
    def _generate_qna_template(self, num_samples: int) -> pd.DataFrame:
        """生成一问一答模板"""
        data = []
        questions = [
            "什么是人工智能？",
            "如何学习 Python 编程？",
            "解释一下机器学习的基本原理",
            "神经网络是如何工作的？",
            "深度学习与传统机器学习有什么区别？"
        ]
        answers = [
            "人工智能是模拟人类智能的科学和工程...",
            "学习 Python 编程可以从基础语法开始...",
            "机器学习的基本原理是通过数据训练模型...",
            "神经网络通过多层神经元处理信息...",
            "深度学习使用深层神经网络，而传统机器学习..."
        ]
        
        for i in range(num_samples):
            q_idx = i % len(questions)
            data.append({
                'question': questions[q_idx],
                'answer': answers[q_idx]
            })
        return pd.DataFrame(data)
    
    # ========== 保存功能 ==========
    
    def save_dataset(self, 
                    df: pd.DataFrame, 
                    save_path: str, 
                    format: str = 'jsonl') -> str:
        """
        保存数据集
        
        Args:
            df: 数据 DataFrame
            save_path: 保存路径
            format: 保存格式（'jsonl', 'csv', 'parquet'）
            
        Returns:
            实际保存的路径
        """
        save_path = str(save_path)
        
        if format == 'jsonl':
            if not save_path.endswith('.jsonl'):
                save_path += '.jsonl'
            df.to_json(save_path, orient='records', force_ascii=False, lines=True)
        
        elif format == 'csv':
            if not save_path.endswith('.csv'):
                save_path += '.csv'
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        elif format == 'parquet':
            if not save_path.endswith('.parquet'):
                save_path += '.parquet'
            df.to_parquet(save_path, index=False)
        
        else:
            raise ValueError(f"不支持的保存格式：{format}")
        
        logger.info(f"数据集已保存到：{save_path}")
        return save_path
    
    # ========== 便捷函数 ==========
    
    def load_and_analyze(self, file_path: str) -> Dict[str, Any]:
        """加载文件并返回分析报告"""
        df = self.load_file(file_path)
        
        format_type = self.detect_format(df)
        statistics = self.compute_statistics(df)
        preview = self.preview_data(df, limit=5)
        
        return {
            'format': format_type,
            'format_description': self.get_format_description(format_type),
            'statistics': statistics,
            'preview': preview,
            'columns': list(df.columns)
        }
