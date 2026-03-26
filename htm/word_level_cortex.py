# 模块C：词级皮层 (WordLevelCortex)
# 负责人：AI 3
# 输入：word_embeds: (B, L, 768)
# 输出：{'sparse_repr': (B, L, 1024), 'prediction_error': (B, L), 'predicted_embeds': (B, L, 768)}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class WordLevelCortexConfig(BaseConfig):
    """词级皮层配置"""
    hidden_dim: int = 768
    num_columns: int = 1024  # HTM列数
    cells_per_column: int = 16  # 每列细胞数
    activation_threshold: int = 8  # 激活阈值
    learning_threshold: int = 5  # 学习阈值
    sparsity_level: float = 0.02  # 稀疏度（激活列占比）
    temporal_window: int = 5


class WordLevelCortex(BaseModule):
    """
    词级皮层模块
    
    实现空间池化(SpatialPooler)、时序记忆(TemporalMemory)和预测机制。
    使用位运算优化稀疏计算，列数: 1024，细胞数/列: 16
    """
    
    def __init__(
        self, 
        config: Optional[WordLevelCortexConfig] = None,
        module_name: str = "word_level_cortex"
    ):
        config = config or WordLevelCortexConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 空间池化：输入到列的投影矩阵
        self.spatial_pooler = nn.Linear(
            self.config.hidden_dim,
            self.config.num_columns,
            bias=False
        )
        
        # 细胞状态：每个列有多个细胞，使用2层MLP增强时序建模
        self.cell_state_encoder = nn.Sequential(
            nn.Linear(
                self.config.num_columns * self.config.cells_per_column,
                self.config.num_columns * 4
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.num_columns * 4,
                self.config.num_columns * self.config.cells_per_column
            )
        )
        
        # 前向突触连接：学习细胞间的时序依赖
        self.forward_synapses = nn.Linear(
            self.config.num_columns * self.config.cells_per_column,
            self.config.num_columns * self.config.cells_per_column,
            bias=False
        )
        
        # 预测投影：将激活状态投影回嵌入空间
        self.prediction_projection = nn.Linear(
            self.config.num_columns,
            self.config.hidden_dim
        )
        
        # 初始化稀疏连接（促进局部学习）
        self._initialize_sparse_connections()
    
    def forward(
        self,
        word_embeds: torch.Tensor,
        prev_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            word_embeds: 词级嵌入 (batch_size, seq_len, hidden_dim)
            prev_state: 之前的状态（用于序列处理）
            
        Returns:
            Dict包含：
                sparse_repr: 稀疏表示 (batch_size, seq_len, num_columns)
                prediction_error: 预测误差 (batch_size, seq_len)
                predicted_embeds: 预测嵌入 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = word_embeds.shape
        
        # 处理序列中的每个时间步
        sparse_reprs = []
        prediction_errors = []
        predicted_embeds = []
        
        current_state = prev_state or self._init_state(batch_size)
        
        for t in range(seq_len):
            step_input = word_embeds[:, t, :]
            
            # 1. 空间池化 - 生成稀疏激活列
            step_sparse = self._spatial_pooling(step_input)
            
            # 2. 时序记忆 - 基于上下文预测
            step_pred, current_state = self._temporal_memory(step_sparse, current_state)
            
            # 3. 计算预测误差
            pred_error = self._compute_prediction_error(step_sparse, step_pred)
            
            # 4. 投影回嵌入空间
            pred_embed = self.prediction_projection(step_sparse)
            
            sparse_reprs.append(step_sparse)
            prediction_errors.append(pred_error)
            predicted_embeds.append(pred_embed)
        
        # 堆叠序列结果
        sparse_repr = torch.stack(sparse_reprs, dim=1)
        prediction_error = torch.stack(prediction_errors, dim=1)
        predicted_embeds = torch.stack(predicted_embeds, dim=1)
        
        return {
            'sparse_repr': sparse_repr,
            'prediction_error': prediction_error,
            'predicted_embeds': predicted_embeds,
            'state': current_state
        }
    
    def _init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """初始化状态"""
        return {
            'prev_activation': torch.zeros(
                batch_size,
                self.config.num_columns,
                device=self._device
            ),
            'cell_states': torch.zeros(
                batch_size,
                self.config.num_columns * self.config.cells_per_column,
                device=self._device
            ),
            'active_cells': torch.zeros(
                batch_size,
                self.config.num_columns * self.config.cells_per_column,
                device=self._device
            ),
            'predictive_cells': torch.zeros(
                batch_size,
                self.config.num_columns * self.config.cells_per_column,
                device=self._device
            )
        }
    
    def _initialize_sparse_connections(self) -> None:
        """初始化稀疏连接 - 模拟HTM中的突触连接"""
        # 只有10%的连接是非零的
        with torch.no_grad():
            weight = self.forward_synapses.weight
            mask = torch.rand_like(weight) < 0.1  # 10%稀疏度
            weight.data *= mask.float()
    
    def _spatial_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """空间池化 - 生成稀疏列激活"""
        batch_size = x.shape[0]
        
        # 计算列激活分数
        activation = self.spatial_pooler(x)
        
        # 应用稀疏性：只保留前k%的激活列
        k = int(self.config.num_columns * self.config.sparsity_level)
        topk_values, topk_indices = torch.topk(activation, k, dim=-1)
        
        # 创建稀疏表示掩码 - 二值化（只有被选中的列为1）
        binary_repr = torch.zeros_like(activation)
        binary_repr.scatter_(1, topk_indices, 1.0)
        
        return binary_repr
    
    def _temporal_memory(
        self,
        sparse_input: torch.Tensor,
        prev_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """时序记忆 - 学习序列模式，模拟HTM的细胞激活和预测机制"""
        batch_size = sparse_input.shape[0]
        num_columns = self.config.num_columns
        cells_per_column = self.config.cells_per_column
        
        # 1. 确定激活列
        active_columns = sparse_input > 0.5
        
        # 2. 选择每个激活列中的获胜细胞
        active_cells = self._select_winner_cells(active_columns, prev_state)
        
        # 3. 基于上一步的激活细胞，计算预测
        # 前向突触传播
        prev_active = prev_state.get('active_cells', prev_state['cell_states'])
        predictive_signal = self.forward_synapses(prev_active)
        predictive_signal = self.cell_state_encoder(predictive_signal)
        predictive_signal = torch.sigmoid(predictive_signal)
        
        # 4. 重塑预测信号
        predictive_signal_3d = predictive_signal.view(
            batch_size, num_columns, cells_per_column
        )
        
        # 5. 列级预测（取每列最大激活）
        column_pred, _ = torch.max(predictive_signal_3d, dim=-1)
        
        # 6. 更新状态
        new_state = {
            'prev_activation': sparse_input,
            'cell_states': predictive_signal,
            'active_cells': active_cells,
            'predictive_cells': predictive_signal
        }
        
        return column_pred, new_state
    
    def _select_winner_cells(
        self,
        active_columns: torch.Tensor,
        prev_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """为每个激活列选择获胜细胞 - 向量化实现"""
        batch_size = active_columns.shape[0]
        num_columns = self.config.num_columns
        cells_per_column = self.config.cells_per_column
        
        # 获取上一步的预测细胞
        prev_predictive = prev_state.get('predictive_cells', prev_state['cell_states'])
        prev_predictive_3d = prev_predictive.view(
            batch_size, num_columns, cells_per_column
        )
        
        # 找出每个列中最活跃的预测细胞
        max_pred_values, winner_cell_indices = torch.max(prev_predictive_3d, dim=-1)
        
        # 确定哪些列有预测中的细胞（阈值0.5）
        has_prediction = max_pred_values > 0.5
        
        # 确定有效激活列：既是活跃列又有预测
        active_with_pred = active_columns & has_prediction
        
        # 确定爆发列：活跃列但没有预测
        bursting_columns = active_columns & (~has_prediction)
        
        # 创建激活细胞掩码
        active_cells = torch.zeros(
            batch_size, num_columns, cells_per_column,
            device=self._device
        )
        
        # 对于有预测的激活列，选择预测得分最高的细胞
        batch_range = torch.arange(batch_size, device=self._device)
        col_range = torch.arange(num_columns, device=self._device)
        batch_indices, col_indices = torch.meshgrid(batch_range, col_range, indexing='ij')
        
        # 只在有预测的激活列上设置获胜细胞
        mask = active_with_pred
        if mask.any():
            active_cells[
                batch_indices[mask],
                col_indices[mask],
                winner_cell_indices[mask]
            ] = 1.0
        
        # 对于爆发列（没有预测的激活列），选择列中第一个细胞作为获胜者
        bursting_mask = bursting_columns
        if bursting_mask.any():
            active_cells[
                batch_indices[bursting_mask],
                col_indices[bursting_mask],
                0  # 选择第一个细胞
            ] = 1.0
        
        return active_cells.view(batch_size, -1)
    
    def _compute_prediction_error(
        self,
        actual: torch.Tensor,
        predicted: torch.Tensor
    ) -> torch.Tensor:
        """计算预测误差"""
        # 使用MSE作为误差度量
        error = F.mse_loss(actual, predicted, reduction='none').mean(dim=-1)
        return error
