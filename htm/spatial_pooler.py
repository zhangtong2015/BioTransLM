# -*- coding: utf-8 -*-
"""
空间池化器 - SpatialPooler

按照工程文档设计实现HTM核心算法：
- 将输入转换为稀疏分布式表示(SDR)
- Hebbian学习机制
- 动态抑制机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from config import BaseConfig
from core.base_module import BaseModule

logger = logging.getLogger(__name__)

@dataclass
class SpatialPoolerConfig(BaseConfig):
    """空间池化器配置"""
    input_dim: int = 2048           # 输入维度（嵌入转换器输出）
    n_columns: int = 4096           # 微型柱数量
    activation_threshold: int = 15  # 列激活所需突触数
    learning_threshold: int = 10     # 触发学习所需突触数
    permanence_inc: float = 0.1      # 持久度增量
    permanence_dec: float = 0.05     # 持久度减量
    permanence_thresh: float = 0.5   # 连接阈值
    activation_rate: float = 0.02    # 激活率目标（默认2%）
    potential_pool_size: int = 1000   # 每个柱的潜在连接池大小
    global_inhibition: bool = True    # 是否使用全局抑制

class SpatialPooler(BaseModule):
    """
    HTM空间池化核心算法
    
    按照工程文档实现：
    - 计算重叠分数：每个微型柱与输入的匹配程度
    - 抑制机制：只保留top-k激活列
    - Hebbian学习：加强激活连接，减弱非激活连接
    """
    
    def __init__(self, config: Optional[SpatialPoolerConfig] = None):
        self.config = config or SpatialPoolerConfig()
        super().__init__(config=self.config, module_name="spatial_pooler")
    
    def _initialize_module(self):
        """初始化空间池化器组件"""
        input_dim = self.config.input_dim
        n_columns = self.config.n_columns
        
        # 连接持久度矩阵 [n_columns, input_dim]
        # 使用0-1随机初始化
        self.permanences = nn.Parameter(
            torch.rand(n_columns, input_dim) * 0.6,
            requires_grad=False
        )
        
        # 连接权重：持久度>阈值则连接
        self.connections = nn.Parameter(
            (self.permanences > self.config.permanence_thresh).float(),
            requires_grad=False
        )
        
        # 增强因子（用于boosting机制）
        self.boost_factors = nn.Parameter(
            torch.ones(n_columns),
            requires_grad=False
        )
        
        # 激活历史追踪（用于boosting）
        self.activation_counts = nn.Parameter(
            torch.zeros(n_columns),
            requires_grad=False
        )
        
        # 重叠分数计算的偏置
        self.overlap_bias = nn.Parameter(
            torch.zeros(n_columns),
            requires_grad=False
        )
        
        logger.info(f"空间池化器初始化完成: "
                        f"列数={n_columns}, "
                        f"输入维度={input_dim}, "
                        f"激活率={self.config.activation_rate:.1%}")
    
    def compute_overlap(self, input_sdr: torch.Tensor) -> torch.Tensor:
        """
        计算每个微型柱的重叠分数
        
        Args:
            input_sdr: 输入稀疏表示 [batch, seq_len, input_dim]
            
        Returns:
            overlap: 每个柱的重叠分数 [batch, seq_len, n_columns]
        """
        batch_size, seq_len, _ = input_sdr.shape
        
        # 计算输入激活位置（二值化）
        input_active = (input_sdr.abs() > 1e-8).float()
        
        # 计算重叠度：使用矩阵乘法优化
        # input_active: [batch, seq_len, input_dim]
        # connections: [n_columns, input_dim]
        # overlap: [batch, seq_len, n_columns]
        overlap = F.linear(input_active, self.connections)
        
        # 应用boost因子和偏置
        overlap = overlap * self.boost_factors.unsqueeze(0).unsqueeze(0)
        overlap = overlap + self.overlap_bias.unsqueeze(0).unsqueeze(0)
        
        return overlap
    
    def inhibit_columns(self, overlap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用抑制，只保留top-k激活列
        
        Args:
            overlap: 重叠分数 [batch, seq_len, n_columns]
            
        Returns:
            active_mask: 激活掩码 [batch, seq_len, n_columns]
            topk_values: top-k激活值 [batch, seq_len, k]
        """
        batch_size, seq_len, n_cols = overlap.shape
        k = max(1, int(n_cols * self.config.activation_rate))
        
        # 找出每个样本top-k激活列
        topk_values, topk_indices = torch.topk(overlap, k, dim=-1)
        
        # 创建激活掩码
        active_mask = torch.zeros_like(overlap)
        active_mask.scatter_(-1, topk_indices, 1.0)
        
        # 只保留超过阈值的激活
        threshold_mask = overlap > self.config.activation_threshold
        active_mask = active_mask * threshold_mask
        
        return active_mask, topk_values
    
    def learn(self, input_sdr: torch.Tensor, active_columns: torch.Tensor):
        """
        Hebbian学习：加强激活连接，减弱非激活连接
        
        Args:
            input_sdr: 输入稀疏表示 [batch, seq_len, input_dim]
            active_columns: 激活列掩码 [batch, seq_len, n_columns]
        """
        batch_size, seq_len, _ = input_sdr.shape
        
        # 计算输入激活位置
        input_active = (input_sdr.abs() > 1e-8).float()
        
        # 合并batch和seq_len维度进行批量处理
        active_col_flat = active_columns.view(-1, self.config.n_columns)
        input_active_flat = input_active.view(-1, self.config.input_dim)
        
        # 计算每个列的激活样本数
        col_active_count = active_col_flat.sum(dim=0)  # [n_columns]
        
        # 只更新有激活的列
        active_cols = (col_active_count > 0).nonzero().squeeze(1)
        
        if len(active_cols) > 0:
            # 计算每个列的输入激活模式 [n_active_cols, input_dim]
            col_input_patterns = torch.matmul(
                active_col_flat[:, active_cols].t(),  # [n_active_cols, batch*seq_len]
                input_active_flat                     # [batch*seq_len, input_dim]
            )
            
            # 归一化（按样本数）
            norm_factors = col_active_count[active_cols].unsqueeze(1)
            col_input_patterns = col_input_patterns / (norm_factors + 1e-8)
            
            # Hebbian更新
            # 加强激活输入的连接
            inc = col_input_patterns * self.config.permanence_inc
            # 减弱非激活输入的连接
            dec = (1 - col_input_patterns) * self.config.permanence_dec
            
            # 更新持久度
            self.permanences.data[active_cols] += inc - dec
        
        # 截断到[0, 1]范围
        self.permanences.data.clamp_(0, 1)
        
        # 更新连接权重
        self.connections.data = (self.permanences > self.config.permanence_thresh).float()
        
        # 更新激活计数用于boosting
        self.activation_counts.data += col_active_count
    
    def update_boost_factors(self, target_density: float = None):
        """
        更新boost因子，确保各列公平激活
        
        Args:
            target_density: 目标激活密度
        """
        if target_density is None:
            target_density = self.config.activation_rate
        
        # 计算每个列的激活频率
        total_counts = self.activation_counts.sum()
        if total_counts > 0:
            activation_freq = self.activation_counts / total_counts
            
            # 目标频率（均匀分布）
            target_freq = torch.ones_like(activation_freq) / len(activation_freq)
            
            # 更新boost因子：低于目标频率的列获得更高boost
            freq_ratio = target_freq / (activation_freq + 1e-8)
            self.boost_factors.data = torch.exp(1 - freq_ratio)
    
    def forward(self, 
                input_sdr: torch.Tensor, 
                learn: bool = True) -> Dict[str, Any]:
        """
        前向传播+可选学习
        
        Args:
            input_sdr: 输入稀疏表示 [batch, seq_len, input_dim]
            learn: 是否执行Hebbian学习
            
        Returns:
            包含稀疏输出的字典：
            - active_columns: 激活列掩码
            - overlap_scores: 重叠分数
            - sparse_output: 稀疏输出表示
            - n_active_columns: 激活列数量
        """
        # 1. 计算重叠分数
        overlap = self.compute_overlap(input_sdr)
        
        # 2. 抑制机制
        active_columns, overlap_values = self.inhibit_columns(overlap)
        
        # 3. 学习（如果启用）
        if learn and self.training:
            self.learn(input_sdr, active_columns)
        
        # 4. 输出稀疏激活
        sparse_output = active_columns * overlap
        
        # 统计信息
        n_active = active_columns.sum(dim=-1).mean().item()
        
        result = {
            'active_columns': active_columns,
            'overlap_scores': overlap,
            'sparse_output': sparse_output,
            'n_active_columns': n_active,
            'total_columns': self.config.n_columns,
            'activation_density': n_active / self.config.n_columns
        }
        
        return result
    
    def reset_learning(self):
        """重置学习状态"""
        self.activation_counts.zero_()
        self.boost_factors.fill_(1.0)
    
    def get_sparsity(self) -> float:
        """获取当前激活稀疏度"""
        return self.config.activation_rate
    
    def get_n_columns(self) -> int:
        """获取列数"""
        return self.config.n_columns
