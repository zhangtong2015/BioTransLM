#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型管理器 - 负责模型的创建、配置、训练、保存和加载
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging
import threading

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础配置
    model_name: str = "biotranslm_base"
    model_type: str = "bio_generator"  # bio_generator, orchestrator
    
    # 架构参数
    vocab_size: int = 50257
    hidden_dim: int = 768
    n_columns: int = 4096
    n_cells_per_col: int = 16
    sdr_size: int = 4096
    max_sequence_length: int = 512
    
    # HTM 参数
    segment_threshold: int = 12
    active_column_rate: float = 0.02
    sdr_sparsity: float = 0.02
    
    # 训练参数
    dropout_rate: float = 0.1
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建"""
        return cls(**config_dict)
    
    def save(self, save_path: str):
        """保存配置到文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"模型配置已保存：{save_path}")
    
    @classmethod
    def load(cls, config_path: str) -> 'ModelConfig':
        """从文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class ModelManager:
    """模型管理器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model_dir = Path("./models")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # 当前模型状态
        self.current_model = None
        self.current_config: Optional[ModelConfig] = None
        self.model_info = {}
        
        # 训练状态
        self.is_training = False
        self.training_thread = None
        self.stop_flag = False
        
        # 回调函数
        self.progress_callback: Optional[Callable] = None
        
        self._initialized = True
        logger.info("ModelManager 初始化完成")
    
    # ========== 模型创建 ==========
    
    def create_model(self, config: Optional[ModelConfig] = None) -> bool:
        """
        创建新模型
        
        Args:
            config: 模型配置，None 则使用默认配置
            
        Returns:
            是否成功创建
        """
        try:
            if config is None:
                config = ModelConfig()
            
            logger.info(f"创建模型：{config.model_name}")
            
            # 根据模型类型创建不同实例
            if config.model_type == "bio_generator":
                from generation.generator import BioGenerator, GeneratorConfig
                
                gen_config = GeneratorConfig(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    n_columns=config.n_columns,
                    sdr_size=config.sdr_size,
                    max_sequence_length=config.max_sequence_length,
                    dropout_rate=config.dropout_rate,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p
                )
                
                model = BioGenerator(gen_config)
                
            elif config.model_type == "orchestrator":
                from orchestrator import Orchestrator, OrchestratorConfig
                
                orch_config = OrchestratorConfig(
                    debug_mode=True,
                    device=config.device
                )
                
                model = Orchestrator(orch_config)
                
            else:
                raise ValueError(f"不支持的模型类型：{config.model_type}")
            
            # 保存当前模型
            self.current_model = model
            self.current_config = config
            
            # 更新模型信息
            self.model_info = {
                'model_name': config.model_name,
                'model_type': config.model_type,
                'created_at': time.time(),
                'vocab_size': config.vocab_size,
                'hidden_dim': config.hidden_dim,
                'n_columns': config.n_columns,
                'device': str(model._device) if hasattr(model, '_device') else 'unknown',
                'parameter_count': self._count_parameters(model)
            }
            
            logger.info(f"模型创建成功：{self.model_info}")
            return True
            
        except Exception as e:
            logger.error(f"模型创建失败：{e}", exc_info=True)
            return False
    
    def _count_parameters(self, model) -> int:
        """计算模型参数数量"""
        try:
            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            pass
        return 0
    
    # ========== 模型加载 ==========
    
    def load_model(self, model_path: str) -> bool:
        """
        加载已保存的模型
        
        Args:
            model_path: 模型路径（.pt 文件或目录）
            
        Returns:
            是否成功加载
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在：{model_path}")
            
            logger.info(f"加载模型：{model_path}")
            
            # 判断是目录还是文件
            if model_path.is_dir():
                # 从目录加载（包含 config.json 和 model.pt）
                config_path = model_path / "config.json"
                model_file = model_path / "model.pt"
                
                if not config_path.exists() or not model_file.exists():
                    raise FileNotFoundError("目录中缺少 config.json 或 model.pt")
                
                # 加载配置
                config = ModelConfig.load(str(config_path))
                
                # 先创建模型
                if not self.create_model(config):
                    raise RuntimeError("创建模型失败")
                
                # 加载权重
                checkpoint = torch.load(str(model_file), map_location='cpu', weights_only=False)
                
                if hasattr(self.current_model, 'load_state_dict'):
                    self.current_model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("模型权重已加载")
                
                self.model_info['loaded_from'] = str(model_path)
                self.model_info['loaded_at'] = time.time()
                
            else:
                # 直接从 .pt 文件加载
                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
                
                # 尝试从 checkpoint 恢复配置
                if 'model_config' in checkpoint:
                    config = ModelConfig.from_dict(checkpoint['model_config'])
                    if not self.create_model(config):
                        raise RuntimeError("从 checkpoint 创建模型失败")
                    
                    if hasattr(self.current_model, 'load_state_dict'):
                        self.current_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    raise ValueError("模型文件中没有配置信息")
            
            logger.info(f"模型加载成功：{model_path}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败：{e}", exc_info=True)
            return False
    
    # ========== 模型保存 ==========
    
    def save_model(self, save_path: Optional[str] = None, name: Optional[str] = None) -> str:
        """
        保存当前模型
        
        Args:
            save_path: 保存路径（目录或文件）
            name: 模型名称（用于自动生成路径）
            
        Returns:
            实际保存的路径
        """
        if self.current_model is None:
            raise RuntimeError("没有可保存的模型，请先创建或加载模型")
        
        try:
            # 确定保存路径
            if save_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_name = name or self.current_config.model_name or "biotranslm"
                save_path = str(self.model_dir / f"{model_name}_{timestamp}")
            
            save_path = Path(save_path)
            
            # 创建目录
            if save_path.is_absolute() and '.' not in save_path.name:
                # 是目录
                save_path.mkdir(exist_ok=True, parents=True)
                config_path = save_path / "config.json"
                model_file = save_path / "model.pt"
            else:
                # 是文件
                save_path.parent.mkdir(exist_ok=True, parents=True)
                config_path = save_path.with_suffix('.json')
                model_file = save_path
            
            # 保存配置
            if self.current_config:
                self.current_config.save(str(config_path))
            
            # 保存模型权重
            checkpoint = {
                'model_state_dict': self.current_model.state_dict(),
                'model_config': self.current_config.to_dict() if self.current_config else {},
                'model_info': self.model_info,
                'saved_at': time.time()
            }
            
            torch.save(checkpoint, str(model_file))
            
            logger.info(f"模型已保存：{model_file}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"模型保存失败：{e}", exc_info=True)
            raise
    
    # ========== 模型训练 ==========
    
    def start_training(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        开始训练
        
        Args:
            dataset_path: 数据集路径
            config: 训练配置
            progress_callback: 进度回调函数 callback(epoch, step, losses, metrics)
            
        Returns:
            是否成功启动
        """
        if self.current_model is None:
            logger.error("没有可训练的模型")
            return False
        
        if self.is_training:
            logger.warning("训练正在进行中")
            return False
        
        try:
            self.is_training = True
            self.stop_flag = False
            self.progress_callback = progress_callback
            
            # 启动训练线程
            self.training_thread = threading.Thread(
                target=self._training_loop,
                args=(dataset_path, config),
                daemon=True
            )
            self.training_thread.start()
            
            logger.info(f"训练已启动：{dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"启动训练失败：{e}", exc_info=True)
            self.is_training = False
            return False
    
    def _training_loop(self, dataset_path: str, config: Dict[str, Any]):
        """
        训练循环（在独立线程中运行）
        
        Args:
            dataset_path: 数据集路径
            config: 训练配置
        """
        try:
            from training.trainer import BioTrainer, TrainingConfig
            from training.data_loader import ChineseTextDataset
            from torch.utils.data import DataLoader
            
            # 创建训练配置
            train_config = TrainingConfig(
                batch_size=config.get('batch_size', 16),
                num_epochs=config.get('num_epochs', 10),
                learning_rate=config.get('learning_rate', 1e-4),
                max_seq_length=config.get('max_seq_length', 128),
                output_dir=str(self.model_dir / "checkpoints")
            )
            
            # 加载数据集
            logger.info(f"加载数据集：{dataset_path}")
            train_dataset = ChineseTextDataset(
                data=dataset_path,
                max_seq_length=train_config.max_seq_length,
                vocab_size=self.current_config.vocab_size
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_config.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            # 创建训练器
            trainer = BioTrainer(
                config=train_config,
                model=self.current_model,
                train_dataset=train_dataset
            )
            
            logger.info(f"开始训练：{len(train_dataset)} 样本，{train_config.num_epochs} epochs")
            
            # 训练循环
            global_step = 0
            for epoch in range(train_config.num_epochs):
                if self.stop_flag:
                    logger.info("训练已停止")
                    break
                
                epoch_loss = 0.0
                num_batches = 0
                
                for step, batch in enumerate(train_loader):
                    if self.stop_flag:
                        break
                    
                    # 执行训练步
                    loss_dict = trainer._training_step(batch)
                    epoch_loss += loss_dict['total']
                    num_batches += 1
                    global_step += 1
                    
                    # 调用进度回调
                    if self.progress_callback:
                        self.progress_callback(
                            epoch=epoch,
                            step=global_step,
                            losses=loss_dict,
                            metrics={
                                'batch_loss': loss_dict['total'],
                                'lr': config.get('learning_rate', 1e-4)
                            }
                        )
                    
                    # 定期保存检查点
                    if global_step % config.get('save_steps', 100) == 0:
                        checkpoint_path = self.model_dir / f"checkpoint_step_{global_step}.pt"
                        trainer.save_checkpoint(str(checkpoint_path))
                        logger.info(f"检查点已保存：{checkpoint_path}")
                
                # Epoch 结束
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                logger.info(f"Epoch {epoch+1}/{train_config.num_epochs} 完成，平均损失：{avg_loss:.4f}")
                
                # Epoch 结束保存
                epoch_path = self.model_dir / f"epoch_{epoch+1}.pt"
                trainer.save_checkpoint(str(epoch_path))
            
            # 训练完成
            self.is_training = False
            logger.info("训练完成")
            
            # 保存最终模型
            final_path = self.model_dir / "final_model.pt"
            trainer.export_model(str(final_path))
            logger.info(f"最终模型已导出：{final_path}")
            
        except Exception as e:
            logger.error(f"训练失败：{e}", exc_info=True)
            self.is_training = False
            
            if self.progress_callback:
                self.progress_callback(
                    epoch=-1,
                    step=0,
                    losses={'error': str(e)},
                    metrics={}
                )
    
    def stop_training(self):
        """停止训练"""
        if not self.is_training:
            return
        
        logger.info("正在停止训练...")
        self.stop_flag = True
        
        # 等待线程结束
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        
        self.is_training = False
        logger.info("训练已停止")
    
    # ========== 模型信息 ==========
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        return {
            'has_model': self.current_model is not None,
            'config': self.current_config.to_dict() if self.current_config else None,
            'info': self.model_info,
            'is_training': self.is_training
        }
    
    def list_saved_models(self) -> list:
        """列出已保存的模型"""
        models = []
        
        # 查找模型目录
        if self.model_dir.exists():
            for item in self.model_dir.iterdir():
                if item.is_dir():
                    # 目录格式的模型
                    config_file = item / "config.json"
                    model_file = item / "model.pt"
                    if config_file.exists() and model_file.exists():
                        models.append({
                            'name': item.name,
                            'path': str(item),
                            'type': 'directory'
                        })
                elif item.suffix == '.pt':
                    # 单文件模型
                    models.append({
                        'name': item.stem,
                        'path': str(item),
                        'type': 'file'
                    })
        
        return models
    
    def delete_model(self, model_path: str) -> bool:
        """删除模型"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                return False
            
            if model_path.is_dir():
                import shutil
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
            
            logger.info(f"模型已删除：{model_path}")
            return True
            
        except Exception as e:
            logger.error(f"删除模型失败：{e}")
            return False
    
    # ========== 快捷方法 ==========
    
    def has_model(self) -> bool:
        """检查是否有当前模型"""
        return self.current_model is not None
    
    def get_model(self):
        """获取当前模型实例"""
        return self.current_model
    
    def reset(self):
        """重置管理器状态"""
        self.current_model = None
        self.current_config = None
        self.model_info = {}
        self.is_training = False
        self.stop_flag = True
        logger.info("模型管理器已重置")


# 全局单例
_model_manager_instance: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """获取模型管理器单例"""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance
