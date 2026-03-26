# -*- coding: utf-8 -*-
"""
HTM系统整合模块 - HTMSystem

将空间池化和时序记忆整合为完整的HTM皮层系统
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import BaseConfig
from core.base_module import BaseModule
from htm.spatial_pooler import SpatialPooler, SpatialPoolerConfig
from htm.temporal_memory import TemporalMemory, TemporalMemoryConfig

logger = logging.getLogger(__name__)

@dataclass
class HTMSystemConfig(BaseConfig):
    """HTM系统配置"""
    input_dim: int = 2048                   # 输入维度
    n_columns: int = 4096                   # 微型柱数量
    n_cells_per_col: int = 16               # 每柱细胞数
    spatial_activation_rate: float = 0.02   # 空间池化激活率
    temporal_segment_threshold: int = 15    # 时序记忆段阈值
    output_dim: int = 1024                  # 输出维度（用于降维）

class HTMSystem(BaseModule):
    """
    HTM皮层系统整合
    
    整合空间池化和时序记忆：
    1. 空间池化：将输入转换为稀疏列激活
    2. 时序记忆：学习序列模式，预测未来激活
    """
    
    def __init__(self, config: Optional[HTMSystemConfig] = None):
        self.config = config or HTMSystemConfig()
        super().__init__(config=self.config, module_name="htm_system")
    
    def _initialize_module(self):
        """初始化HTM系统组件"""
        # 空间池化器配置
        sp_config = SpatialPoolerConfig(
            input_dim=self.config.input_dim,
            n_columns=self.config.n_columns,
            activation_rate=self.config.spatial_activation_rate
        )
        self.spatial_pooler = SpatialPooler(sp_config)
        
        # 时序记忆配置
        tm_config = TemporalMemoryConfig(
            n_columns=self.config.n_columns,
            n_cells_per_col=self.config.n_cells_per_col,
            segment_threshold=self.config.temporal_segment_threshold
        )
        self.temporal_memory = TemporalMemory(tm_config)
        
        # 输出投影层：将高维HTM表示降维用于后续处理
        self.output_projection = nn.Sequential(
            nn.Linear(self.config.n_columns * self.config.n_cells_per_col, 
                     self.config.output_dim),
            nn.ReLU()
        )
        
        # 列级输出投影
        self.column_projection = nn.Sequential(
            nn.Linear(self.config.n_columns, self.config.output_dim),
            nn.ReLU()
        )
        
        logger.info(f"HTM系统初始化完成: "
                        f"输入维度={self.config.input_dim}, "
                        f"列数={self.config.n_columns}, "
                        f"每柱细胞数={self.config.n_cells_per_col}, "
                        f"输出维度={self.config.output_dim}")
    
    def forward(self, 
                input_sdr: torch.Tensor, 
                learn: bool = True,
                return_sequence: bool = False) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_sdr: 输入稀疏表示 [batch, seq_len, input_dim]
            learn: 是否执行学习
            return_sequence: 是否返回完整的细胞激活序列
            
        Returns:
            包含HTM系统输出的字典：
            - spatial_output: 空间池化输出
            - temporal_output: 时序记忆输出
            - htm_repr: HTM表示（投影到输出维度）
            - column_repr: 列级表示
        """
        batch_size, seq_len, _ = input_sdr.shape
        
        # 确保输入在正确的设备上
        input_sdr = input_sdr.to(self._device)
        
        # 1. 空间池化：处理每个时间步
        spatial_output = self.spatial_pooler(input_sdr, learn=learn)
        
        # 获取激活列 [batch, seq_len, n_columns]
        active_columns = spatial_output['active_columns']
        
        # 2. 时序记忆：处理序列
        # 准备前一时刻和当前时刻的激活列
        if seq_len > 1:
            # 多时间步序列 - 简化处理：每个时间步都与前一时间步配对
            temporal_results = []
            self.temporal_memory.reset_sequence()
            
            for t in range(seq_len):
                if t == 0:
                    # 第一个时间步，使用dummy前一状态
                    prev_step = torch.zeros(batch_size, 1, self.config.n_columns, 
                                          device=input_sdr.device)
                else:
                    prev_step = active_columns[:, t-1:t, :]
                
                curr_step = active_columns[:, t:t+1, :]
                result = self.temporal_memory(prev_step, curr_step, learn=learn)
                temporal_results.append(result)
            
            # 合并结果
            temporal_output = {
                'active_cells': torch.cat([r['active_cells'] for r in temporal_results], dim=1),
                'predicted_cells': torch.cat([r['predicted_cells'] for r in temporal_results], dim=1),
                'burst_columns': torch.cat([r['burst_columns'] for r in temporal_results], dim=1),
                'prediction_accuracy': sum(r['prediction_accuracy'] for r in temporal_results) / len(temporal_results),
                'n_bursts': sum(r['n_bursts'] for r in temporal_results),
                'n_correct_predictions': sum(r['n_correct_predictions'] for r in temporal_results),
                'total_columns': sum(r['total_columns'] for r in temporal_results)
            }
        else:
            # 单时间步（首次输入，无前序状态）
            self.temporal_memory.reset_sequence()
            dummy_prev = torch.zeros_like(active_columns)
            temporal_output = self.temporal_memory(dummy_prev, active_columns, learn=learn)
        
        # 3. 生成输出表示
        active_cells = temporal_output['active_cells']  # [batch, seq_len, n_columns, n_cells_per_col]
        
        # 确保形状匹配
        if active_cells.size(1) != seq_len:
            if active_cells.size(1) > seq_len:
                active_cells = active_cells[:, :seq_len, :, :]
            else:
                # 填充到正确长度
                pad_length = seq_len - active_cells.size(1)
                padding = torch.zeros(
                    batch_size, pad_length, 
                    active_cells.size(2), active_cells.size(3),
                    device=active_cells.device
                )
                active_cells = torch.cat([padding, active_cells], dim=1)
        
        # 展平细胞维度：[batch, seq_len, n_columns, n_cells_per_col] -> [batch, seq_len, n_columns * n_cells_per_col]
        cell_flat = active_cells.reshape(batch_size, seq_len, -1)
        
        # 验证最终形状
        expected_dim = self.config.n_columns * self.config.n_cells_per_col
        if cell_flat.size(-1) != expected_dim:
            logger.warning(f"形状警告：期望最后维度为{expected_dim}，实际为{cell_flat.size(-1)}")
            # 如果不匹配，进行简单的投影调整
            if cell_flat.size(-1) < expected_dim:
                pad = torch.zeros(batch_size, seq_len, expected_dim - cell_flat.size(-1), 
                                device=cell_flat.device)
                cell_flat = torch.cat([cell_flat, pad], dim=-1)
            else:
                cell_flat = cell_flat[..., :expected_dim]
        
        # 投影到输出维度
        htm_repr = self.output_projection(cell_flat)
        
        # 列级表示 [batch, seq_len, n_columns] -> [batch, seq_len, output_dim]
        column_repr = self.column_projection(active_columns)
        
        result = {
            'spatial_output': spatial_output,
            'temporal_output': temporal_output,
            'htm_repr': htm_repr,
            'column_repr': column_repr,
            'active_cells': active_cells,
            'active_columns': active_columns
        }
        
        if return_sequence:
            result['full_sequence'] = True
        
        return result
    
    def reset_sequence(self):
        """重置序列状态"""
        self.temporal_memory.reset_sequence()
    
    def reset_learning(self):
        """重置学习状态"""
        self.spatial_pooler.reset_learning()
        self.temporal_memory.reset_learning()
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.config.output_dim
    
    def get_spatial_sparsity(self) -> float:
        """获取空间稀疏度"""
        return self.spatial_pooler.get_sparsity()
    
    def get_total_segments(self) -> int:
        """获取总段数"""
        return self.temporal_memory.get_total_segments()
