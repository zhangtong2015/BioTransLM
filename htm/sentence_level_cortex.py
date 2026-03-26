# 模块D：句级皮层 (SentenceLevelCortex)
# 负责人：AI 4
# 输入：sentence_embeds: (B, L, 768), word_level_output: Dict
# 输出：{'sentence_repr': (B, 768), 'feedback_signal': (B, L, 768), 'cross_layer_error': float}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class SentenceLevelCortexConfig(BaseConfig):
    """句级皮层配置"""
    hidden_dim: int = 768
    num_attention_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    feedback_weight: float = 0.3  # 反馈信号权重


class SentenceLevelCortex(BaseModule):
    """
    句级皮层模块
    
    实现句子级序列学习、跨层预测误差计算、反馈调节机制。
    预测误差使用MSE计算，误差信号回传给词级皮层。
    """
    
    def __init__(
        self, 
        config: Optional[SentenceLevelCortexConfig] = None,
        module_name: str = "sentence_level_cortex"
    ):
        config = config or SentenceLevelCortexConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 句子级Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.sentence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers
        )
        
        # <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌投影（句子表示）
        self.cls_projection = nn.Linear(
            self.config.hidden_dim,
            self.config.hidden_dim
        )
        
        # 反馈信号生成器 - 基于句级预测误差生成反馈到词级
        self.feedback_generator = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )
        
        # 序列预测头
        self.prediction_head = nn.Linear(
            self.config.hidden_dim,
            self.config.hidden_dim
        )
    
    def forward(
        self,
        sentence_embeds: torch.Tensor,
        word_level_output: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            sentence_embeds: 句子级嵌入 (batch_size, seq_len, hidden_dim)
            word_level_output: 词级皮层输出（用于跨层误差计算）
            attention_mask: 注意力掩码 (batch_size, seq_len)
            
        Returns:
            Dict包含：
                sentence_repr: 句子级表示 (batch_size, hidden_dim)
                feedback_signal: 反馈到词级的信号 (batch_size, seq_len, hidden_dim)
                cross_layer_error: 跨层预测误差
                token_predictions: 令牌级预测 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = sentence_embeds.shape
        
        # 1. 句子级编码
        encoded = self._encode_sentence(sentence_embeds, attention_mask)
        
        # 2. 提取<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌作为句子表示
        sentence_repr = self._extract_sentence_representation(encoded)
        
        # 3. 生成预测（用于计算误差）
        predictions = self.prediction_head(encoded)
        
        # 4. 计算跨层误差（与词级输出对比）
        cross_layer_error = self._compute_cross_layer_error(
            predictions,
            word_level_output
        )
        
        # 5. 生成反馈信号到词级皮层
        feedback_signal = self._generate_feedback_signal(
            encoded,
            cross_layer_error,
            seq_len
        )
        
        return {
            'sentence_repr': sentence_repr,
            'feedback_signal': feedback_signal,
            'cross_layer_error': cross_layer_error,
            'token_predictions': predictions
        }
    
    def _encode_sentence(
        self,
        embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码句子级表示"""
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        
        encoded = self.sentence_encoder(
            embeds,
            src_key_padding_mask=src_key_padding_mask
        )
        return encoded
    
    def _extract_sentence_representation(self, encoded: torch.Tensor) -> torch.Tensor:
        """提取句子级表示（使用<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>令牌或均值池化）"""
        # 使用第一个令牌（<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>）作为句子表示
        cls_token = encoded[:, 0, :]
        sentence_repr = self.cls_projection(cls_token)
        return sentence_repr
    
    def _compute_cross_layer_error(
        self,
        predictions: torch.Tensor,
        word_level_output: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """计算跨层预测误差"""
        batch_size = predictions.shape[0]
        
        if word_level_output is not None and 'predicted_embeds' in word_level_output:
            # 与词级预测对比计算误差
            word_preds = word_level_output['predicted_embeds']
            error = F.mse_loss(predictions, word_preds, reduction='none')
            error = error.mean(dim=[1, 2])  # (batch_size,)
        else:
            # 自预测误差：预测下一个令牌
            if predictions.shape[1] > 1:
                pred = predictions[:, :-1, :]
                target = predictions[:, 1:, :]
                error = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2])
            else:
                error = torch.zeros(batch_size, device=predictions.device)
        
        return error
    
    def _generate_feedback_signal(
        self,
        encoded: torch.Tensor,
        error: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """生成反馈信号回传给词级皮层"""
        batch_size = encoded.shape[0]
        
        # 将误差扩展到序列长度
        error_expanded = error.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, -1)
        
        # 将编码与误差拼接以生成上下文相关的反馈
        feedback_input = torch.cat([encoded, error_expanded.expand(-1, -1, encoded.shape[-1])], dim=-1)
        
        # 生成反馈信号
        feedback = self.feedback_generator(feedback_input)
        
        # 应用反馈权重
        feedback = feedback * self.config.feedback_weight
        
        return feedback
