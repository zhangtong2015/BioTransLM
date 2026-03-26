# 模块A：感觉门控 (SensoryGating)
# 负责人：AI 1
# 输入：input_ids, attention_mask
# 输出：{'gated_embeds': (B, L, 768), 'attention_weights': (B, L), 'noise_level': float}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class SensoryGatingConfig(BaseConfig):
    """感觉门控配置"""
    hidden_dim: int = 768
    noise_threshold: float = 0.3
    attention_dropout: float = 0.1
    max_sequence_length: int = 512
    dynamic_threshold: bool = True
    cls_token_weight: float = 1.5  #<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌权重 (<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>=CLS token)
    char_level_noise: bool = True  # 是否启用字符级噪声检测
    vocab_size: int = 30522  # BERT词汇表大小


class SensoryGating(BaseModule):
    """
    感觉门控模块
    
    实现噪声过滤、注意力权重计算、阈值筛选机制。
    使用BERT的<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌计算重要性，噪声分数基于字符级混乱度。
    """
    
    def __init__(
        self, 
        config: Optional[SensoryGatingConfig] = None,
        module_name: str = "sensory_gating"
    ):
        config = config or SensoryGatingConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # Token嵌入层（用于从input_ids获取初始嵌入）
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_dim
        )
        
        # 噪声估计层
        self.noise_estimator = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.config.attention_dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 字符级噪声检测
        if self.config.char_level_noise:
            self.char_noise_estimator = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.config.hidden_dim,
                    out_channels=64,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=3,
                    padding=1
                ),
                nn.Sigmoid()
            )
        
        # 注意力投影层 - 支持<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌增强
        self.attention_projection = nn.Linear(self.config.hidden_dim, 1)
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.Sigmoid()
        )
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        task_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            input_embeds: 可选的预计算输入嵌入 (batch_size, seq_len, hidden_dim)
            task_context: 任务上下文（用于动态阈值）
            
        Returns:
            Dict包含：
                gated_embeds: 门控后的嵌入 (batch_size, seq_len, hidden_dim)
                attention_weights: 注意力权重 (batch_size, seq_len)
                noise_level: 噪声水平估计
                noise_scores: 每个位置的噪声分数 (batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. 获取初始嵌入
        if input_embeds is None:
            input_embeds = self.token_embedding(input_ids)
        
        # 2. 噪声估计（包括字符级）
        noise_level, noise_scores = self._estimate_noise(input_embeds, input_ids)
        
        # 3. 注意力权重计算（增强<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌）
        attention_weights = self._compute_attention_weights(
            input_embeds, 
            input_ids,
            attention_mask
        )
        
        # 4. 动态阈值计算
        threshold = self._compute_threshold(noise_level, task_context)
        
        # 5. 门控应用
        gated_embeds = self._apply_gating(input_embeds, attention_weights, threshold, noise_scores)
        
        # 6. 后处理
        gated_embeds = self.layer_norm(gated_embeds)
        
        return {
            'gated_embeds': gated_embeds,
            'attention_weights': attention_weights,
            'noise_level': noise_level,
            'noise_scores': noise_scores
        }
    
    def _estimate_noise(
        self, 
        embeds: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[float, torch.Tensor]:
        """估计输入的噪声水平
        
        结合：
        1. 基于嵌入混乱度的噪声估计
        2. 可选的字符级噪声检测
        3. Token频率分析（识别rare tokens作为噪声）
        """
        batch_size, seq_len, hidden_dim = embeds.shape
        
        # 1. 基于嵌入的噪声估计
        with torch.no_grad():
            token_var = torch.var(embeds, dim=-1).mean(dim=-1, keepdim=True)
            token_var = (token_var - token_var.min()) / (token_var.max() - token_var.min() + 1e-8)
        
        # 2. 神经网络噪声估计
        embed_noise = self.noise_estimator(embeds).squeeze(-1)
        
        # 3. 字符级噪声检测（可选）
        if self.config.char_level_noise:
            char_input = embeds.transpose(1, 2)  # (B, H, L) for Conv1d
            char_noise = self.char_noise_estimator(char_input).squeeze(1)
        else:
            char_noise = torch.zeros_like(embed_noise)
        
        # 4. 组合噪声分数
        noise_scores = 0.4 * embed_noise + 0.4 * char_noise + 0.2 * token_var
        
        # 整体噪声水平
        noise_level = float(noise_scores.mean().item())
        
        return noise_level, noise_scores
    
    def _compute_attention_weights(
        self,
        embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算注意力权重 - 使用<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌增强
        
        根据任务表要求：使用BERT的<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌计算重要性
        """
        batch_size, seq_len, _ = embeds.shape
        
        # 1. 基础注意力分数
        attention_scores = self.attention_projection(embeds).squeeze(-1)
        
        # 2. <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌增强（假设<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token id是101，这是BERT的标准）
        cls_mask = (input_ids == 101).float()
        cls_enhanced = attention_scores * (1.0 + cls_mask * self.config.cls_token_weight)
        
        # 3. 应用掩码
        if attention_mask is not None:
            cls_enhanced = cls_enhanced.masked_fill(
                attention_mask == 0,
                -1e9
            )
        
        # 4. Softmax归一化
        attention_weights = F.softmax(cls_enhanced, dim=-1)
        
        return attention_weights
    
    def _compute_threshold(self, noise_level: float, task_context: Optional[str] = None) -> float:
        """基于噪声水平和任务上下文计算动态阈值"""
        base_threshold = self.config.noise_threshold
        
        if self.config.dynamic_threshold:
            # 噪声越高，阈值越高 - 自适应调整
            adaptive_threshold = base_threshold + noise_level * 0.3
            
            # 任务特定调整
            if task_context == 'QA':
                adaptive_threshold *= 1.2  # QA任务需要更高的信噪比
            elif task_context == 'GENERATION':
                adaptive_threshold *= 0.9  # 生成任务可以容忍更多噪声
            elif task_context == 'CLASSIFICATION':
                adaptive_threshold *= 1.0  # 分类任务使用标准阈值
            
            return min(max(adaptive_threshold, 0.1), 0.9)
        
        return base_threshold
    
    def _apply_gating(
        self,
        embeds: torch.Tensor,
        attention_weights: torch.Tensor,
        threshold: float,
        noise_scores: torch.Tensor
    ) -> torch.Tensor:
        """应用门控机制
        
        结合注意力权重和噪声分数，实现智能过滤：
        1. 高注意力、低噪声的token被保留
        2. 低注意力、高噪声的token被抑制
        3. 支持软门控而不是硬截断
        """
        batch_size, seq_len, hidden_dim = embeds.shape
        
        # 计算门控强度：注意力权重越高，噪声分数越低，门控越开放
        gate_strength = attention_weights * (1.0 - noise_scores)
        
        # 应用阈值（软门控）
        gate_mask = torch.sigmoid(10 * (gate_strength - threshold))  # 尖锐的Sigmoid实现软阈值
        gate_mask = gate_mask.unsqueeze(-1)
        
        # 加权门控：保留原始嵌入的信息，但根据重要性加权
        gated = embeds * gate_mask
        
        return gated
