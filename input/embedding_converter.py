# -*- coding: utf-8 -*-
"""
嵌入转换层 - EmbeddingConverter
将标准词嵌入转换为HTM兼容的稀疏表示

按照工程文档设计实现：
- 支持input_ids和input_embeddings两种输入模式
- 高维稀疏表示输出（默认95%稀疏度）
- 可选加载预训练嵌入或训练原生嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import BaseConfig
from core.base_module import BaseModule

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConverterConfig(BaseConfig):
    """嵌入转换器配置"""
    vocab_size: int = 30522
    input_dim: int = 768
    output_dim: int = 2048
    sparsity_target: float = 0.95
    use_pretrained: bool = False
    pretrained_embedding_path: Optional[str] = None

class SparsifyActivation(nn.Module):
    """
    稀疏激活层：只保留top-k激活值
    按照工程文档实现，确保输出稀疏度达到目标
    """
    def __init__(self, sparsity_target: float = 0.95):
        super().__init__()
        self.sparsity_target = sparsity_target
        self.k_ratio = 1 - sparsity_target  # 保留的激活比例
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, dim] 或 [batch_size, dim]
        Returns:
            稀疏化后的张量
        """
        original_shape = x.shape
        k = max(1, int(original_shape[-1] * self.k_ratio))
        
        # 找到top-k激活
        topk_values, topk_indices = torch.topk(x, k, dim=-1)
        
        # 创建稀疏掩码
        output = torch.zeros_like(x)
        output.scatter_(-1, topk_indices, topk_values)
        
        return output

class EmbeddingConverter(BaseModule):
    """
    将标准词嵌入转换为HTM兼容的稀疏表示
    
    无Transformer依赖，仅用线性层+稀疏化激活
    支持两种输入模式：
    1. input_ids: token id序列（兼容Transformer数据格式）
    2. input_embeddings: 预训练词嵌入（如BERT输出）
    """
    
    def __init__(self, config: Optional[EmbeddingConverterConfig] = None):
        self.config = config or EmbeddingConverterConfig()
        super().__init__(config=self.config, module_name="embedding_converter")
    
    def _initialize_module(self):
        """初始化嵌入转换层组件"""
        # 原生嵌入层
        self.native_embedding = nn.Embedding(
            self.config.vocab_size, 
            self.config.input_dim
        )
        
        # 加载预训练嵌入（如果配置）
        if self.config.use_pretrained and self.config.pretrained_embedding_path:
            self._load_pretrained_embeddings()
        
        # 转换层：将嵌入转换为高维稀疏表示
        self.converter = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.output_dim),
            nn.ReLU(),
            SparsifyActivation(self.config.sparsity_target)
        )
        
        logger.info(f"嵌入转换器初始化完成: 输出维度={self.config.output_dim}, "
                        f"稀疏度目标={self.config.sparsity_target:.1%}")
    
    def _load_pretrained_embeddings(self):
        """加载预训练嵌入权重"""
        try:
            pretrained_weights = torch.load(self.config.pretrained_embedding_path)
            self.native_embedding.weight = nn.Parameter(pretrained_weights)
            logger.info("预训练嵌入加载成功")
        except Exception as e:
            logger.warning(f"预训练嵌入加载失败: {e}，使用随机初始化")
    
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                input_embeddings: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_ids: token id序列 [batch_size, seq_len]
            input_embeddings: 预训练词嵌入 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            包含稀疏输出的字典：
            - sparse_output: 稀疏表示 [batch_size, seq_len, output_dim]
            - input_embeddings: 输入嵌入（用于后续处理）
            - actual_sparsity: 实际稀疏度
        """
        if input_embeddings is None:
            if input_ids is None:
                raise ValueError("input_ids 和 input_embeddings 不能同时为None")
            # 确保在正确的设备上
            input_ids = input_ids.to(self._device)
            input_embeddings = self.native_embedding(input_ids)
        else:
            # 确保在正确的设备上
            input_embeddings = input_embeddings.to(self._device)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
            input_embeddings = input_embeddings * attention_mask.unsqueeze(-1)
        
        # 转换为稀疏表示
        sparse_output = self.converter(input_embeddings)
        
        # 计算实际稀疏度
        actual_sparsity = (sparse_output == 0).float().mean().item()
        
        result = {
            'sparse_output': sparse_output,
            'input_embeddings': input_embeddings,
            'actual_sparsity': actual_sparsity,
            'sparsity_target': self.config.sparsity_target
        }
        
        return result
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.config.output_dim
    
    def get_sparsity(self) -> float:
        """获取稀疏度目标"""
        return self.config.sparsity_target
