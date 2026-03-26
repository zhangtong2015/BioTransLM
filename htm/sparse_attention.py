# 模块E：动态稀疏注意力 (SparseAttention)
# 负责人：AI 1
# 输入：attention_matrix: (B, H, L, L) 或 embeddings: (B, L, D)
# 输出：{'sparse_attention': (B, L, D), 'activation_rate': float, 'attention_weights': (B, L, L)}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class SparseAttentionConfig(BaseConfig):
    """稀疏注意力配置"""
    hidden_dim: int = 768
    num_heads: int = 8
    activation_threshold: float = 0.1  # 激活阈值
    max_activation_rate: float = 0.1  # 最大激活率（<10%）
    dropout: float = 0.1
    use_stdp_approx: bool = True  # 使用STDP近似


class SparseAttention(BaseModule):
    """
    动态稀疏注意力模块
    
    实现简化版脉冲神经元模型、STDP学习规则近似、稀疏激活机制。
    激活率控制在10%以下，先用稀疏注意力矩阵近似，不实现完整SNN。
    """
    
    def __init__(
        self, 
        config: Optional[SparseAttentionConfig] = None,
        module_name: str = "sparse_attention"
    ):
        config = config or SparseAttentionConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        self.head_dim = self.config.hidden_dim // self.config.num_heads
        
        # Q/K/V投影
        self.q_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.k_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.v_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        
        # STDP近似学习的门控（可选）
        if self.config.use_stdp_approx:
            self.stdp_gate = nn.Sequential(
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            hidden_states: 输入嵌入 (batch_size, seq_len, hidden_dim)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            output_attentions: 是否输出注意力权重
            
        Returns:
            Dict包含：
                sparse_attention: 稀疏注意力输出 (batch_size, seq_len, hidden_dim)
                activation_rate: 激活率（应该 < 10%）
                attention_weights: 注意力权重 (可选)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 1. 投影到Q/K/V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 2. 重塑为多头
        q = q.view(batch_size, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # 4. 应用掩码（如果有）
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.config.num_heads, seq_len, -1)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9)
        
        # 5. 稀疏化：应用阈值和激活率约束
        sparse_weights, activation_rate = self._sparsify_attention(attn_weights)
        
        # 6. 应用softmax和dropout
        attn_probs = F.softmax(sparse_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 7. 计算输出
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # 8. STDP近似学习（可选）
        if self.config.use_stdp_approx:
            attn_output = self._apply_stdp_approx(attn_output, hidden_states)
        
        # 9. 输出投影
        attn_output = self.out_proj(attn_output)
        
        result = {
            'sparse_attention': attn_output,
            'activation_rate': activation_rate
        }
        
        if output_attentions:
            result['attention_weights'] = sparse_weights
        
        return result
    
    def _sparsify_attention(
        self,
        attn_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        稀疏化注意力权重
        确保只有不到10%的连接被激活
        """
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        # 1. 先应用阈值
        thresholded = attn_weights * (attn_weights > self.config.activation_threshold).float()
        
        # 2. 计算每个位置的非零数量
        nonzero_count = (thresholded > 0).float().sum()
        total_count = float(batch_size * num_heads * seq_len * seq_len)
        current_activation_rate = nonzero_count / total_count
        
        # 3. 如果激活率超过限制，进一步稀疏化
        if current_activation_rate > self.config.max_activation_rate:
            # 计算需要保留的top-k值
            k = int(total_count * self.config.max_activation_rate)
            k_per_batch = k // batch_size
            
            # 展平并取top-k
            flat_weights = attn_weights.view(batch_size, -1)
            topk_values, topk_indices = torch.topk(flat_weights, k_per_batch, dim=-1)
            
            # 创建掩码
            mask = torch.zeros_like(flat_weights)
            mask.scatter_(1, topk_indices, 1.0)
            mask = mask.view_as(attn_weights)
            
            # 应用掩码
            sparse_weights = attn_weights * mask
            activation_rate = self.config.max_activation_rate
        else:
            sparse_weights = thresholded
            activation_rate = current_activation_rate.item() if isinstance(current_activation_rate, torch.Tensor) else current_activation_rate
        
        return sparse_weights, activation_rate
    
    def _apply_stdp_approx(
        self,
        attn_output: torch.Tensor,
        input_states: torch.Tensor
    ) -> torch.Tensor:
        """
        应用STDP（脉冲时序依赖可塑性）的近似
        简化版本：基于输入和输出的时间相关性门控
        """
        batch_size, seq_len, hidden_dim = attn_output.shape
        
        # 拼接当前输出与前一时刻输入（简化的时序依赖）
        if seq_len > 1:
            # 当前输出
            current_out = attn_output
            
            # 前一时刻输入（右移一位）
            prev_in = torch.cat([
                torch.zeros_like(input_states[:, :1, :]),
                input_states[:, :-1, :]
            ], dim=1)
            
            # 计算STDP门控
            stdp_input = torch.cat([current_out, prev_in], dim=-1)
            stdp_gate = self.stdp_gate(stdp_input)
            
            # 应用门控
            output = attn_output * stdp_gate
        else:
            output = attn_output
        
        return output
