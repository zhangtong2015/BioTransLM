#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioTransLM 训练器模块 - 纯生物启发式训练
支持PT格式模型保存与加载
"""

import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 导入模型组件
from core.base_module import BaseModule
from generation.generator import BioGenerator, GeneratorConfig
from generation.sequence_predictor import SequencePredictorConfig
from generation.sdr_vocabulary import SDRVocabularyConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    device: str = 'auto'
    seed: int = 42
    output_dir: str = './checkpoints'
    
    # 训练超参数
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler: str = 'cosine'
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    
    # 保存策略
    save_steps: int = 100
    eval_steps: int = 50
    max_checkpoints: int = 5
    
    # 数据配置
    max_seq_length: int = 128
    vocab_size: int = 30522
    
    # 损失函数权重
    lambda_sparsity: float = 0.1
    lambda_temporal: float = 0.05
    lambda_recon: float = 1.0

class BioDataset(Dataset):
    """生物启发式训练数据集"""
    
    def __init__(self, 
                 data: Optional[List[Dict[str, Any]]] = None,
                 vocab_size: int = 30522,
                 max_seq_length: int = 128,
                 num_samples: int = 1000):
        """
        初始化数据集
        Args:
            data: 实际数据，为None时生成测试数据
            vocab_size: 词汇表大小
            max_seq_length: 最大序列长度
            num_samples: 测试数据样本数
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        if data is None:
            # 生成测试数据
            self.data = self._generate_dummy_data(num_samples)
        else:
            self.data = data
    
    def _generate_dummy_data(self, num_samples: int) -> List[Dict[str, torch.Tensor]]:
        """生成测试用的虚拟数据"""
        data = []
        for _ in range(num_samples):
            # 随机生成token ids
            seq_length = torch.randint(16, self.max_seq_length, (1,)).item()
            input_ids = torch.randint(0, self.vocab_size, (seq_length,))
            
            # padding到固定长度
            padded = torch.full((self.max_seq_length,), 0, dtype=torch.long)
            padded[:seq_length] = input_ids
            
            # attention mask
            attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
            attention_mask[:seq_length] = 1
            
            data.append({
                'input_ids': padded,
                'attention_mask': attention_mask,
                'labels': padded.clone()  # 语言模型标签
            })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

class BioTrainer:
    """生物启发式模型训练器"""
    
    def __init__(self,
                 config: TrainingConfig,
                 model: Optional[BioGenerator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None):
        """
        初始化训练器
        Args:
            config: 训练配置
            model: 模型实例
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 设置设备
        self._setup_device()
        
        # 设置随机种子
        self._set_seed()
        
        # 输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 初始化优化器和调度器
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None  # 用于混合精度训练
    
    def _setup_device(self):
        """设置计算设备"""
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"使用设备: {self.device}")
    
    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def initialize_model(self, model_config: Optional[GeneratorConfig] = None):
        """初始化模型"""
        if model_config is None:
            model_config = GeneratorConfig(
                vocab_size=self.config.vocab_size,
                hidden_dim=768,
                n_columns=2048,
                sdr_size=4096,
                max_sequence_length=self.config.max_seq_length
            )
        
        self.model = BioGenerator(model_config)
        self.model.to(self.device)
        logger.info(f"模型初始化完成，参数数量: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def initialize_optimizer(self):
        """初始化优化器和学习率调度器"""
        if self.model is None:
            raise ValueError("请先初始化模型")
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        if self.config.lr_scheduler == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio
            )
        elif self.config.lr_scheduler == 'step':
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=5,
                gamma=0.5
            )
        
        # 混合精度训练
        if self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info("优化器和学习率调度器初始化完成")
    
    def _compute_loss(self,
                     outputs: Dict[str, Any],
                     batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算生物启发式损失函数
        包含：语言模型损失 + 稀疏性损失 + 时序连续性损失
        """
        logits = outputs['generation_state']['logits']
        labels = batch['labels'].to(self.device)
        
        # 1. 基础语言模型损失
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm_loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 2. 稀疏性损失：鼓励稀疏激活
        sparsity_loss = torch.tensor(0.0, device=self.device)
        if 'sparsity_info' in outputs:
            activation_rates = outputs['sparsity_info'].get('activation_rates', [])
            for rate in activation_rates:
                if isinstance(rate, torch.Tensor):
                    sparsity_loss += torch.mean(torch.abs(rate - 0.02))  # 目标激活率2%
        
        # 3. 时序连续性损失：鼓励时序平滑
        temporal_loss = torch.tensor(0.0, device=self.device)
        if 'sequence_output' in outputs.get('generation_state', {}):
            seq_out = outputs['generation_state']['sequence_output']
            if len(seq_out.shape) == 3:
                diff = torch.mean(torch.abs(seq_out[:, 1:, :] - seq_out[:, :-1, :]))
                temporal_loss = diff
        
        # 总损失
        total_loss = (self.config.lambda_recon * lm_loss +
                     self.config.lambda_sparsity * sparsity_loss +
                     self.config.lambda_temporal * temporal_loss)
        
        # 返回详细损失信息
        loss_dict = {
            'total': total_loss.item(),
            'lm': lm_loss.item(),
            'sparsity': sparsity_loss.item(),
            'temporal': temporal_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 将数据移到设备
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # 前向传播（支持混合精度）
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    max_new_tokens=0,  # 仅计算表示，不生成新token
                    reset_state=True
                )
                loss, loss_dict = self._compute_loss(outputs, batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                max_new_tokens=0,
                reset_state=True
            )
            loss, loss_dict = self._compute_loss(outputs, batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
        
        return loss_dict
    
    def _evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步评估"""
        self.model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                max_new_tokens=0,
                reset_state=True
            )
            
            _, loss_dict = self._compute_loss(outputs, batch)
        
        return loss_dict
    
    def save_checkpoint(self, name: Optional[str] = None, is_best: bool = False):
        """
        保存检查点（PT格式）
        Args:
            name: 检查点名称，None则使用step数
            is_best: 是否为最佳模型
        """
        if name is None:
            name = f'checkpoint-step-{self.global_step}'
        
        checkpoint_path = self.output_dir / f'{name}.pt'
        
        # 保存内容
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'training_config': self.config,
            'model_config': self.model.config,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 同时保存为best模型
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已更新: {best_path}")
        
        # 清理旧检查点
        self._clean_old_checkpoints()
    
    def _clean_old_checkpoints(self):
        """清理过多的检查点"""
        checkpoints = list(self.output_dir.glob('checkpoint-step-*.pt'))
        if len(checkpoints) > self.config.max_checkpoints:
            # 按创建时间排序，删除最旧的
            checkpoints.sort(key=lambda x: x.stat().st_ctime)
            for ckpt in checkpoints[:-self.config.max_checkpoints]:
                ckpt.unlink()
                logger.info(f"已删除旧检查点: {ckpt}")
    
    def load_checkpoint(self, checkpoint_path: str, load_training_state: bool = True):
        """
        加载检查点
        Args:
            checkpoint_path: 检查点路径
            load_training_state: 是否加载训练状态（优化器、调度器等）
        """
        try:
            # PyTorch 2.6+ 需要显式声明安全的全局对象
            with torch.serialization.safe_globals([GeneratorConfig, TrainingConfig]):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except:
            # 降级方式：使用 weights_only=False（仅用于可信来源）
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载模型
        if self.model is None:
            if 'model_config' in checkpoint:
                self.initialize_model(checkpoint['model_config'])
            else:
                raise ValueError("需要model_config来初始化模型")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型参数已加载: {checkpoint_path}")
        
        # 加载训练状态
        if load_training_state:
            if self.optimizer is None:
                self.initialize_optimizer()
            
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'lr_scheduler_state_dict' in checkpoint and checkpoint['lr_scheduler_state_dict']:
                if self.lr_scheduler:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                if self.scaler:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.global_step = checkpoint.get('global_step', 0)
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            logger.info(f"训练状态已恢复: step={self.global_step}, epoch={self.current_epoch}")
    
    def export_model(self, save_path: str):
        """
        导出推理模型（仅保存模型权重和配置）
        Args:
            save_path: 保存路径
        """
        export_data = {
            'model_type': 'BioGenerator',
            'model_config': self.model.config,
            'state_dict': self.model.state_dict(),
            'vocab_size': self.config.vocab_size,
            'max_seq_length': self.config.max_seq_length,
            'export_time': time.time(),
            'version': '1.0.0'
        }
        
        torch.save(export_data, save_path)
        logger.info(f"推理模型已导出: {save_path}")
    
    def train(self):
        """开始训练循环"""
        if self.model is None:
            self.initialize_model()
        if self.optimizer is None:
            self.initialize_optimizer()
        
        # 创建数据加载器
        if self.train_dataset is None:
            logger.warning("未提供训练数据集，使用测试数据")
            self.train_dataset = BioDataset(
                vocab_size=self.config.vocab_size,
                max_seq_length=self.config.max_seq_length,
                num_samples=10000
            )
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        eval_dataloader = None
        if self.eval_dataset:
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        logger.info("开始训练...")
        logger.info(f"  训练样本数: {len(self.train_dataset)}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  总Epochs: {self.config.num_epochs}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_dataloader):
                # 训练步
                loss_dict = self._training_step(batch)
                epoch_loss += loss_dict['total']
                num_batches += 1
                self.global_step += 1
                
                # 日志
                if step % 10 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Step {step}: total_loss={loss_dict['total']:.4f}, "
                        f"lm_loss={loss_dict['lm']:.4f}, "
                        f"sparsity_loss={loss_dict['sparsity']:.4f}, "
                        f"temporal_loss={loss_dict['temporal']:.4f}, "
                        f"lr={lr:.2e}"
                    )
                
                # 定期评估
                if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                    if eval_dataloader:
                        eval_loss = self.evaluate(eval_dataloader)
                        logger.info(f"评估结果: avg_loss={eval_loss:.4f}")
                
                # 定期保存
                if self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            # Epoch结束
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1} 完成: avg_loss={avg_epoch_loss:.4f}")
            
            # 更新学习率
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Epoch结束后评估和保存
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                logger.info(f"Epoch {epoch + 1} 评估: avg_loss={eval_loss:.4f}")
                
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint(is_best=True)
            
            # 保存epoch检查点
            self.save_checkpoint(name=f'epoch-{epoch + 1}')
        
        logger.info("\n训练完成!")
        logger.info(f"最佳验证损失: {self.best_loss:.4f}")
        
        # 保存最终模型
        self.save_checkpoint(name='final')
        self.export_model(self.output_dir / 'final_model.pt')
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """评估模型"""
        logger.info("开始评估...")
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            loss_dict = self._evaluation_step(batch)
            total_loss += loss_dict['total']
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

# 便捷函数
def load_pretrained_model(
        model_path: str,
        device: str = 'auto'
) -> BioGenerator:
    """
    加载预训练模型
    
    Args:
        model_path: 模型路径
        device: 计算设备
    Returns:
        加载好的模型
    """
    # 确定设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # 加载模型 - 处理PyTorch 2.6+的序列化安全问题
    from generation import GeneratorConfig
    from training import TrainingConfig
    
    try:
        # PyTorch 2.6+ 需要显式声明安全的全局对象
        with torch.serialization.safe_globals([GeneratorConfig, TrainingConfig]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except:
        # 降级方式：使用 weights_only=False（仅用于可信来源）
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 创建模型
    model_config = checkpoint.get('model_config', checkpoint.get('config'))
    if model_config is None:
        # 尝试从导出格式加载
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            raise ValueError("无法找到模型配置")
    
    model = BioGenerator(model_config)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint.get('model_state_dict', {})))
    model.to(device)
    model.eval()
    
    logger.info(f"模型加载完成: {model_path}")
    return model

if __name__ == '__main__':
    # 快速测试
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建训练配置
    config = TrainingConfig(
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-4,
        max_seq_length=64,
        vocab_size=5000,
        save_steps=200,
        eval_steps=100
    )
    
    # 创建训练器
    trainer = BioTrainer(config)
    
    # 初始化模型
    trainer.initialize_model()
    
    # 开始训练
    trainer.train()
