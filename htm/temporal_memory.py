# -*- coding: utf-8 -*-
"""
时序记忆 - TemporalMemory

按照工程文档设计实现HTM核心算法：
- 学习序列模式，预测未来激活
- 基于远端段连接的上下文感知
- burst机制处理未预测的输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import random

from config import BaseConfig
from core.base_module import BaseModule

logger = logging.getLogger(__name__)

@dataclass
class DistalSegment:
    """远端段数据结构"""
    cell_idx: Tuple[int, int]  # (column_idx, cell_idx)
    connections: torch.Tensor  # [n_connected_cells] 连接的细胞索引
    permanences: torch.Tensor  # [n_connected_cells] 连接持久度
    activation_threshold: int = 15
    permanence_inc: float = 0.1
    permanence_dec: float = 0.05

@dataclass
class TemporalMemoryConfig(BaseConfig):
    """时序记忆配置"""
    n_columns: int = 4096           # 微型柱数量
    n_cells_per_col: int = 16       # 每柱细胞数
    segment_threshold: int = 15     # 段激活阈值
    learning_threshold: int = 10     # 学习阈值
    max_segments_per_cell: int = 10  # 每个细胞最多段数
    permanence_inc: float = 0.1      # 持久度增量
    permanence_dec: float = 0.05     # 持久度减量
    permanence_thresh: float = 0.5   # 连接阈值
    new_synapse_count: int = 20      # 新段初始突触数

class TemporalMemory(BaseModule):
    """
    HTM时序记忆核心算法
    
    按照工程文档实现：
    - 基于前一状态预测当前激活
    - 段激活预测细胞的激活
    - burst机制：列为预测则burst所有细胞激活
    """
    
    def __init__(self, config: Optional[TemporalMemoryConfig] = None):
        self.config = config or TemporalMemoryConfig()
        super().__init__(config=self.config, module_name="temporal_memory")
    
    def _initialize_module(self):
        """初始化时序记忆组件"""
        self.n_columns = self.config.n_columns
        self.n_cells_per_col = self.config.n_cells_per_col
        self.total_cells = self.n_columns * self.n_cells_per_col
        
        # 段阈值：激活一个段需要多少突触
        self.segment_threshold = self.config.segment_threshold
        
        # 预测与状态历史
        self.prev_predicted = None
        self.prev_active_cells = None
        
        # 细胞使用追踪（用于学习新模式时的细胞选择）
        self.cell_usage = nn.Parameter(
            torch.zeros(self.n_columns, self.n_cells_per_col).to(self._device),
            requires_grad=False
        )
        
        # 持久度增量/减量
        self.permanence_inc = self.config.permanence_inc
        self.permanence_dec = self.config.permanence_dec
        self.permanence_thresh = self.config.permanence_thresh
        
        # 存储所有远端段 [n_columns, n_cells_per_col, max_segments]
        # 使用稀疏表示：只存储有效的段索引和数据
        self.segment_count = nn.Parameter(
            torch.zeros(self.n_columns, self.n_cells_per_col, dtype=torch.int32).to(self._device),
            requires_grad=False
        )
        
        # 段数据存储（动态列表）
        self.segments_data: List[List[List[Optional[DistalSegment]]]] = [
            [[] for _ in range(self.n_cells_per_col)] 
            for _ in range(self.n_columns)
        ]
        
        # 连接矩阵稀疏表示
        self.segment_connections = {}  # segment_uid -> (connected_cells, permanences)
        self._next_segment_uid = 0
        
        logger.info(f"时序记忆初始化完成: "
                        f"列数={self.n_columns}, "
                        f"每柱细胞数={self.n_cells_per_col}, "
                        f"总细胞数={self.total_cells}")
    
    def _get_segment_uid(self, col_idx: int, cell_idx: int, seg_idx: int) -> int:
        """生成段唯一标识符"""
        return col_idx * 1000000 + cell_idx * 1000 + seg_idx
    
    def compute_prediction(self, active_cells_prev: torch.Tensor) -> torch.Tensor:
        """
        基于前一状态计算当前预测
        
        Args:
            active_cells_prev: 前一时刻激活细胞 [batch, seq_len, n_columns, n_cells_per_col]
            
        Returns:
            prediction: 当前预测细胞激活概率 [batch, seq_len, n_columns, n_cells_per_col]
        """
        batch_size, seq_len, _, _ = active_cells_prev.shape
        device = active_cells_prev.device
        
        prediction = torch.zeros(
            batch_size, seq_len, self.n_columns, self.n_cells_per_col,
            device=device
        )
        
        # 展平batch和seq_len维度便于处理
        active_flat = active_cells_prev.view(-1, self.total_cells)
        
        # 检查每个段是否被激活
        for col_idx in range(self.n_columns):
            for cell_idx in range(self.n_cells_per_col):
                segments = self.segments_data[col_idx][cell_idx]
                for seg_idx, segment in enumerate(segments):
                    if segment is None:
                        continue
                    
                    # 计算段激活分数
                    conn_cells = segment.connections.to(device)
                    if len(conn_cells) == 0:
                        continue
                    
                    # 计算连接细胞的激活和
                    act_sum = active_flat[:, conn_cells].sum(dim=1)
                    act_sum = act_sum.view(batch_size, seq_len)
                    
                    # 如果超过阈值，预测该细胞激活
                    active_mask = act_sum > self.segment_threshold
                    prediction[:, :, col_idx, cell_idx] = active_mask.float()
        
        return prediction
    
    def burst_column(self, col_idx: int) -> int:
        """
        列为预测则burst（所有细胞激活），并选择学习细胞
        
        Args:
            col_idx: 列索引
            
        Returns:
            选择的学习细胞索引
        """
        # 选择最少使用的细胞来学习新模式
        least_used = int(self.cell_usage[col_idx].argmin())
        self.cell_usage[col_idx, least_used] += 1
        return least_used
    
    def create_new_segment(self, 
                         col_idx: int, 
                         cell_idx: int,
                         active_cells_prev: torch.Tensor) -> DistalSegment:
        """
        创建一个新的远端段，连接到前一时刻激活的细胞
        
        Args:
            col_idx: 列索引
            cell_idx: 细胞索引
            active_cells_prev: 前一时刻激活细胞
            
        Returns:
            新创建的远端段
        """
        # 找出前一时刻激活的细胞
        active_flat = active_cells_prev.view(-1)
        active_indices = active_flat.nonzero().squeeze(1)
        
        if len(active_indices) == 0:
            # 如果没有前激活，随机选择一些细胞建立初始连接
            active_indices = torch.randint(0, self.total_cells, 
                                         (self.config.new_synapse_count,))
        
        # 随机选择部分连接
        n_sample = min(self.config.new_synapse_count, len(active_indices))
        if n_sample > 0:
            selected_indices = active_indices[
                torch.randperm(len(active_indices))[:n_sample]
            ]
        else:
            selected_indices = torch.randint(0, self.total_cells, 
                                          (self.config.new_synapse_count,))
        
        # 创建新段
        new_segment = DistalSegment(
            cell_idx=(col_idx, cell_idx),
            connections=selected_indices,
            permanences=torch.ones(len(selected_indices)) * 0.6,
            activation_threshold=self.segment_threshold,
            permanence_inc=self.permanence_inc,
            permanence_dec=self.permanence_dec
        )
        
        return new_segment
    
    def update_segment(self, 
                      segment: DistalSegment,
                      active_cells_prev: torch.Tensor,
                      was_correct: bool):
        """
        更新段的连接持久度
        
        Args:
            segment: 要更新的段
            active_cells_prev: 前一时刻激活细胞
            was_correct: 预测是否正确
        """
        active_flat = active_cells_prev.view(-1)
        device = active_cells_prev.device
        
        conn_cells = segment.connections.to(device)
        if len(conn_cells) == 0:
            return
        
        # 获取前一时刻这些连接细胞的激活状态
        prev_active = active_flat[conn_cells] > 0
        
        # Hebbian更新
        # 激活的连接加强，未激活的减弱
        if was_correct:
            # 正确预测：加强激活连接，减弱非激活连接
            inc = prev_active.float() * self.permanence_inc
            dec = (~prev_active).float() * self.permanence_dec
            segment.permanences += inc.cpu() - dec.cpu()
        else:
            # 错误预测：稍微减弱激活连接
            dec = prev_active.float() * (self.permanence_dec * 0.5)
            segment.permanences -= dec.cpu()
        
        # 确保持久度在[0, 1]范围
        segment.permanences.clamp_(0, 1)
    
    def forward(self, 
                active_columns_prev: torch.Tensor,
                active_columns_curr: torch.Tensor,
                learn: bool = True) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            active_columns_prev: 前一时刻激活列 [batch, seq_len, n_columns]
            active_columns_curr: 当前时刻激活列 [batch, seq_len, n_columns]
            learn: 是否执行学习
            
        Returns:
            包含时序记忆输出的字典：
            - active_cells: 当前激活细胞
            - predicted_cells: 预测细胞
            - burst_columns: burst列标识
            - prediction_accuracy: 预测准确率
        """
        batch_size, seq_len, _ = active_columns_curr.shape
        device = active_columns_curr.device
        
        # 初始化当前激活细胞 [batch, seq_len, n_columns, n_cells_per_col]
        active_cells = torch.zeros(
            batch_size, seq_len, self.n_columns, self.n_cells_per_col,
            device=device
        )
        
        # 初始化预测细胞
        predicted_cells = torch.zeros_like(active_cells)
        
        # 跟踪burst列
        burst_columns = torch.zeros(batch_size, seq_len, self.n_columns, 
                                  device=device)
        
        # 1. 基于前一状态的预测
        if self.prev_active_cells is not None:
            predicted_cells = self.compute_prediction(self.prev_active_cells)
        
        # 2. 确定当前激活细胞
        correct_predictions = 0
        total_predictions = 0
        
        for b in range(batch_size):
            for t in range(seq_len):
                # 当前时刻激活的列
                active_cols = active_columns_curr[b, t].nonzero().squeeze(1)
                
                for col in active_cols:
                    col = int(col)
                    # 检查该列是否有预测的细胞
                    col_predictions = predicted_cells[b, t, col]
                    predicted_cell_count = int(col_predictions.sum())
                    
                    if predicted_cell_count > 0:
                        # 正确预测：激活预测的细胞
                        active_cells[b, t, col] = col_predictions
                        correct_predictions += 1
                        
                        if learn and self.prev_active_cells is not None:
                            # 更新正确预测细胞的段
                            active_cell_indices = col_predictions.nonzero().squeeze(1)
                            for cell_idx in active_cell_indices:
                                cell_idx = int(cell_idx)
                                segments = self.segments_data[col][cell_idx]
                                for seg in segments:
                                    if seg is not None:
                                        self.update_segment(
                                            seg, 
                                            self.prev_active_cells[b, t],
                                            was_correct=True
                                        )
                    else:
                        # 未预测：burst（激活所有细胞）并选择学习细胞
                        active_cells[b, t, col, :] = 1.0  # burst
                        burst_columns[b, t, col] = 1.0
                        
                        if learn:
                            # 选择最少使用的细胞来学习这个新模式
                            learn_cell = self.burst_column(col)
                            
                            # 在前一激活细胞上创建新连接
                            if self.prev_active_cells is not None:
                                new_segment = self.create_new_segment(
                                    col, learn_cell, 
                                    self.prev_active_cells[b, t]
                                )
                                
                                # 添加到段数据中
                                if len(self.segments_data[col][learn_cell]) < self.config.max_segments_per_cell:
                                    self.segments_data[col][learn_cell].append(new_segment)
                                    self.segment_count[col, learn_cell] += 1
                
                total_predictions += len(active_cols)
        
        # 保存当前状态用于下一次预测
        self.prev_active_cells = active_cells.detach().clone()
        self.prev_predicted = predicted_cells.detach().clone()
        
        # 计算预测准确率
        prediction_accuracy = correct_predictions / max(1, total_predictions)
        
        result = {
            'active_cells': active_cells,
            'predicted_cells': predicted_cells,
            'burst_columns': burst_columns,
            'prediction_accuracy': prediction_accuracy,
            'n_bursts': int(burst_columns.sum().item()),
            'n_correct_predictions': correct_predictions,
            'total_columns': total_predictions
        }
        
        return result
    
    def reset_sequence(self):
        """重置序列状态"""
        self.prev_predicted = None
        self.prev_active_cells = None
    
    def reset_learning(self):
        """重置学习状态"""
        self.cell_usage.zero_()
        self.segment_count.zero_()
        self.segments_data = [
            [[] for _ in range(self.n_cells_per_col)] 
            for _ in range(self.n_columns)
        ]
        self.reset_sequence()
    
    def get_total_segments(self) -> int:
        """获取总段数"""
        total = 0
        for col in range(self.n_columns):
            for cell in range(self.n_cells_per_col):
                total += len(self.segments_data[col][cell])
        return total
