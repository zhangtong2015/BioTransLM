# -*- coding: utf-8 -*-
"""
序列预测器 - SequencePredictorTM

基于HTM时序记忆的生物启发式序列预测
完全替代Transformer解码器，无任何Transformer组件

核心特性：
1. 基于时序记忆(TM)的序列预测
2. 远端段的上下文感知
3. Winner-Take-All竞争机制
4. burst模式处理新异模式
5. SDR到token的可训练映射
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
from htm.temporal_memory import TemporalMemory, TemporalMemoryConfig

logger = logging.getLogger(__name__)

@dataclass
class SequencePredictorConfig(TemporalMemoryConfig):
    """序列预测器配置"""
    n_columns: int = 4096           # 微型柱数量
    n_cells_per_col: int = 16       # 每柱细胞数
    output_dim: int = 768           # 输出维度
    vocab_size: int = 50257         # 词汇表大小
    segment_threshold: int = 12     # 段激活阈值
    learning_threshold: int = 8      # 学习阈值
    max_segments_per_cell: int = 12  # 每个细胞最多段数
    prediction_horizon: int = 1      # 预测步数
    active_column_rate: float = 0.02 # 列激活率


class SparseWinnerTakeAll(nn.Module):
    """
    稀疏Winner-Take-All层
    只保留top-k激活的列，实现稀疏竞争
    """
    def __init__(self, n_columns: int, activation_rate: float = 0.02):
        super().__init__()
        self.n_columns = n_columns
        self.k = max(1, int(n_columns * activation_rate))
        self.activation_rate = activation_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入激活 [batch_size, seq_len, n_columns]
               或 [batch_size, n_columns]
        Returns:
            稀疏激活张量
        """
        orig_shape = x.shape
        
        # 处理二维输入
        if len(orig_shape) == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # Top-k竞争
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        
        # 创建稀疏掩码
        output = torch.zeros_like(x)
        output.scatter_(-1, topk_indices, topk_values)
        
        # 恢复原始形状
        if len(orig_shape) == 2:
            output = output.squeeze(1)
        
        return output


class SequencePredictorTM(BaseModule):
    """
    基于时序记忆的序列预测器
    
    完全生物启发的设计，替代Transformer解码器：
    1. 使用HTM时序记忆学习序列模式
    2. 远端段提供上下文感知
    3. Winner-Take-All竞争实现稀疏激活
    4. burst机制处理未预测输入
    """
    
    def __init__(self, config: Optional[SequencePredictorConfig] = None):
        self.config = config or SequencePredictorConfig()
        super().__init__(config=self.config, module_name="sequence_predictor")
    
    def _initialize_module(self):
        """初始化序列预测器组件"""
        # 1. 时序记忆核心
        self.temporal_memory = TemporalMemory(self.config)
        
        # 2. Winner-Take-All竞争层
        self.wta = SparseWinnerTakeAll(
            n_columns=self.config.n_columns,
            activation_rate=self.config.active_column_rate
        )
        
        # 3. SDR输出投影（细胞状态→连续表示）
        cell_state_dim = self.config.n_columns * self.config.n_cells_per_col
        self.cell_projection = nn.Sequential(
            nn.Linear(cell_state_dim, self.config.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. 列状态投影
        self.column_projection = nn.Sequential(
            nn.Linear(self.config.n_columns, self.config.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 细胞使用追踪（用于学习新模式时的细胞选择）
        self.cell_usage = nn.Parameter(
            torch.zeros(self.config.n_columns, self.config.n_cells_per_col).to(self._device),
            requires_grad=False
        )
        
        # 5. 状态融合门
        self.state_gate = nn.Sequential(
            nn.Linear(self.config.output_dim * 2, self.config.output_dim),
            nn.Sigmoid()
        )
        
        # 6. 内部状态追踪
        self.reset_state()
        
        logger.info(f"序列预测器初始化完成: "
                        f"列数={self.config.n_columns}, "
                        f"每柱细胞数={self.config.n_cells_per_col}, "
                        f"激活率={self.config.active_column_rate:.1%}, "
                        f"输出维度={self.config.output_dim}")
    
    def reset_state(self):
        """重置内部状态"""
        self.prev_active_cells = None
        self.prev_predicted = None
        self.prediction_history = []
    
    def _columns_to_cells(self, 
                         active_columns: torch.Tensor, 
                         predictions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        将列激活转换为细胞激活
        
        Args:
            active_columns: 激活列 [batch, seq_len, n_columns]
            predictions: 预测细胞 [batch, seq_len, n_columns, n_cells_per_col]
            
        Returns:
            active_cells: 激活细胞 [batch, seq_len, n_columns, n_cells_per_col]
        """
        batch_size, seq_len, n_cols = active_columns.shape
        device = active_columns.device
        
        active_cells = torch.zeros(
            batch_size, seq_len, n_cols, self.config.n_cells_per_col,
            device=device
        )
        
        # 找到激活列的索引
        for b in range(batch_size):
            for t in range(seq_len):
                col_indices = active_columns[b, t].nonzero().squeeze(-1)
                for col_idx in col_indices:
                    if predictions is not None:
                        # 检查该列是否有预测的细胞
                        col_pred = predictions[b, t, col_idx]
                        pred_cells = col_pred.nonzero().squeeze(-1)
                        if len(pred_cells) > 0:
                            # 激活预测细胞
                            active_cells[b, t, col_idx, pred_cells] = 1.0
                        else:
                            # 无预测：burst该列（所有细胞激活）
                            active_cells[b, t, col_idx, :] = 1.0
                    else:
                        # 无预测：burst所有细胞
                        active_cells[b, t, col_idx, :] = 1.0
        
        return active_cells
    
    def forward(self, 
                column_activations: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                learn: bool = True) -> Dict[str, Any]:
        """
        前向传播：处理列激活序列，生成预测
        
        Args:
            column_activations: 列激活输入 [batch, seq_len, n_columns]
            context: 上下文调制信号 [batch, seq_len, output_dim]
            learn: 是否执行学习
            
        Returns:
            包含预测和状态的字典
        """
        # 确保在正确的设备上
        column_activations = column_activations.to(self._device)
        if context is not None:
            context = context.to(self._device)
        
        batch_size, seq_len, n_cols = column_activations.shape
        device = self._device
        
        # 1. Winner-Take-All稀疏化
        sparse_columns = self.wta(column_activations)
        
        # 2. 准备输出张量
        all_cell_states = []
        all_column_outputs = []
        all_predictions = []
        
        # 3. 序列处理
        for t in range(seq_len):
            current_cols = sparse_columns[:, t, :]  # [batch, n_cols]
            
            if self.prev_active_cells is not None:
                # 基于前一状态计算预测
                predictions = self.temporal_memory.compute_prediction(
                    self.prev_active_cells
                )
                all_predictions.append(predictions)
            else:
                predictions = None
                all_predictions.append(torch.zeros(
                    batch_size, 1, n_cols, self.config.n_cells_per_col,
                    device=device
                ))
            
            # 确定激活细胞
            if predictions is not None:
                # 使用2D预测进行单步处理
                pred_2d = predictions[:, 0, :, :] if len(predictions.shape) == 4 else predictions
                active_cells = self._columns_to_cells(
                    current_cols.unsqueeze(1),
                    pred_2d.unsqueeze(1)
                )
            else:
                active_cells = self._columns_to_cells(current_cols.unsqueeze(1))
            
            # 4. 时序记忆学习
            if learn and self.prev_active_cells is not None:
                # 展平细胞状态用于学习
                prev_cells_flat = self.prev_active_cells.view(-1, n_cols, self.config.n_cells_per_col)
                curr_cols_flat = current_cols.view(-1, n_cols)
                self.temporal_memory.learn(prev_cells_flat, curr_cols_flat)
            
            # 5. 更新状态
            self.prev_active_cells = active_cells  # [batch, 1, n_cols, n_cells]
            
            # 6. 生成输出表示
            # 细胞状态投影
            cells_flat = active_cells.view(batch_size, -1)
            cell_output = self.cell_projection(cells_flat)  # [batch, output_dim]
            
            # 列状态投影
            col_output = self.column_projection(current_cols)  # [batch, output_dim]
            
            # 7. 上下文调制
            if context is not None:
                ctx = context[:, t, :] if len(context.shape) == 3 else context
                gate_input = torch.cat([cell_output, col_output], dim=-1)
                gate = self.state_gate(gate_input)
                cell_output = cell_output * gate
                col_output = col_output * (1 - gate)
                # 加入上下文偏置
                cell_output = cell_output + ctx * 0.1
            
            # 8. 融合状态
            fused_output = cell_output + col_output
            
            all_cell_states.append(cell_output.unsqueeze(1))
            all_column_outputs.append(col_output.unsqueeze(1))
        
        # 9. 连接序列输出
        cell_states_seq = torch.cat(all_cell_states, dim=1)  # [batch, seq_len, output_dim]
        column_outputs_seq = torch.cat(all_column_outputs, dim=1)  # [batch, seq_len, output_dim]
        
        # 10. 组合最终输出
        final_output = cell_states_seq + column_outputs_seq
        
        return {
            'sequence_output': final_output,  # [batch, seq_len, output_dim]
            'cell_states': cell_states_seq,    # [batch, seq_len, output_dim]
            'column_states': column_outputs_seq,  # [batch, seq_len, output_dim]
            'active_cells': self.prev_active_cells,  # 最新细胞状态
            'sparse_columns': sparse_columns,  # [batch, seq_len, n_columns]
            'predictions': all_predictions     # 每步预测列表
        }
    
    def predict_next(self, 
                     current_state: torch.Tensor, 
                     context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测下一时刻状态
        
        Args:
            current_state: 当前列激活 [batch, n_columns]
            context: 上下文信号 [batch, output_dim]
            
        Returns:
            next_column_activation: 预测的下一列激活 [batch, n_columns]
        """
        batch_size = current_state.shape[0]
        device = current_state.device
        
        # 1. Winner-Take-All稀疏化
        sparse_state = self.wta(current_state)
        
        # 2. 转换为细胞状态
        active_cells = self._columns_to_cells(
            sparse_state.unsqueeze(1), 
            self.prev_predicted
        )  # [batch, 1, n_cols, n_cells]
        
        # 3. 计算预测
        if self.prev_active_cells is not None:
            predictions = self.temporal_memory.compute_prediction(
                self.prev_active_cells
            )
            self.prev_predicted = predictions
        else:
            predictions = torch.zeros(
                batch_size, 1, self.config.n_columns, self.config.n_cells_per_col,
                device=device
            )
            self.prev_predicted = predictions
        
        # 4. 更新内部状态
        self.prev_active_cells = active_cells
        
        # 5. 从细胞预测计算列激活
        col_predictions = predictions.max(dim=-1)[0].squeeze(1)  # [batch, n_columns]
        
        # 6. 应用上下文调制
        if context is not None:
            # 简单的上下文偏置
            context_bias = torch.tanh(context) * 0.1
            context_proj = nn.Linear(context.shape[-1], self.config.n_columns).to(device)
            col_predictions = col_predictions + context_proj(context_bias)
        
        # 7. 稀疏化预测输出
        output = self.wta(col_predictions)
        
        return output
