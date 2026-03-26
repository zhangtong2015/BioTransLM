# 模块B：多粒度编码 (MultiGranularEncoder)
# 负责人：AI 2
# 输入：input_embeds, attention_mask
# 输出：{'char_embeds', 'word_embeds', 'phrase_embeds', 'sentence_embeds', 'combined_embeds'}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class MultiGranularEncoderConfig(BaseConfig):
    """多粒度编码器配置"""
    hidden_dim: int = 768
    char_kernel_size: int = 3
    word_kernel_size: int = 5
    phrase_kernel_sizes: Tuple[int, ...] = (2, 3, 4)
    dropout: float = 0.1
    num_heads: int = 8


class MultiGranularEncoder(BaseModule):
    """
    多粒度编码模块
    
    实现字符级、词级、短语级、句子级的编码，构建层次图结构。
    使用BERT-base-uncased作为基础编码器，各粒度输出维度统一为768维。
    """
    
    def __init__(
        self, 
        config: Optional[MultiGranularEncoderConfig] = None,
        module_name: str = "multigranular_encoder"
    ):
        config = config or MultiGranularEncoderConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 字符级CNN编码
        self.char_encoder = nn.Conv1d(
            in_channels=self.config.hidden_dim,
            out_channels=self.config.hidden_dim,
            kernel_size=self.config.char_kernel_size,
            padding=self.config.char_kernel_size // 2
        )
        
        # 词级BiLSTM编码
        self.word_encoder = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout
        )
        
        # 短语级多窗口CNN
        self.phrase_encoders = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.config.hidden_dim,
                out_channels=self.config.hidden_dim // len(self.config.phrase_kernel_sizes),
                kernel_size=ks,
                padding=ks // 2
            ) for ks in self.config.phrase_kernel_sizes
        ])
        
        # 句子级自注意力编码
        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Dropout(self.config.dropout)
        )
    
    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_embeds: 输入嵌入 (batch_size, seq_len, hidden_dim)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            
        Returns:
            Dict包含各粒度编码特征：
                char_embeds: 字符级编码 (B, L, 768)
                word_embeds: 词级编码 (B, L, 768)
                phrase_embeds: 短语级编码 (B, L, 768)
                sentence_embeds: 句子级编码 (B, L, 768)
                combined_embeds: 融合特征 (B, L, 768)
                hierarchy_graph: 层次图结构（各粒度间的关联）
        """
        # 确保在正确的设备上
        input_embeds = input_embeds.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        
        batch_size, seq_len, _ = input_embeds.shape
        
        # 1. 字符级编码
        char_embeds = self._char_level_encode(input_embeds)
        
        # 2. 词级编码
        word_embeds = self._word_level_encode(input_embeds)
        
        # 3. 短语级编码
        phrase_embeds = self._phrase_level_encode(input_embeds)
        
        # 4. 句子级编码
        sentence_embeds = self._sentence_level_encode(input_embeds, attention_mask)
        
        # 5. 特征融合
        combined_embeds = self._fuse_features(
            char_embeds,
            word_embeds, 
            phrase_embeds,
            sentence_embeds
        )
        
        # 6. 构建层次图结构
        hierarchy_graph = self._build_hierarchy_graph(
            char_embeds,
            word_embeds,
            phrase_embeds,
            sentence_embeds
        )
        
        return {
            'char_embeds': char_embeds,
            'word_embeds': word_embeds,
            'phrase_embeds': phrase_embeds,
            'sentence_embeds': sentence_embeds,
            'combined_embeds': combined_embeds,
            'hierarchy_graph': hierarchy_graph
        }
    
    def _char_level_encode(self, embeds: torch.Tensor) -> torch.Tensor:
        """字符级CNN编码"""
        # CNN需要输入形状: (B, C, L)
        x = embeds.transpose(1, 2)
        x = self.char_encoder(x)
        x = F.relu(x)
        return x.transpose(1, 2)
    
    def _word_level_encode(self, embeds: torch.Tensor) -> torch.Tensor:
        """词级BiLSTM编码"""
        x, _ = self.word_encoder(embeds)
        return x
    
    def _phrase_level_encode(self, embeds: torch.Tensor) -> torch.Tensor:
        """短语级多窗口CNN编码"""
        batch_size, seq_len, hidden_dim = embeds.shape
        
        # CNN需要输入形状: (B, C, L)
        x = embeds.transpose(1, 2)
        
        phrase_features = []
        for encoder in self.phrase_encoders:
            feat = encoder(x)
            feat = F.relu(feat)
            # 确保所有特征图具有相同的序列长度
            if feat.size(2) > seq_len:
                feat = feat[:, :, :seq_len]
            phrase_features.append(feat)
        
        # 拼接不同窗口的特征
        phrase_embeds = torch.cat(phrase_features, dim=1)
        return phrase_embeds.transpose(1, 2)
    
    def _sentence_level_encode(
        self,
        embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """句子级自注意力编码"""
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        x, _ = self.sentence_attention(
            query=embeds,
            key=embeds,
            value=embeds,
            key_padding_mask=key_padding_mask
        )
        return x + embeds  # 残差连接
    
    def _fuse_features(
        self,
        char_embeds: torch.Tensor,
        word_embeds: torch.Tensor,
        phrase_embeds: torch.Tensor,
        sentence_embeds: torch.Tensor
    ) -> torch.Tensor:
        """融合多粒度特征"""
        # 拼接所有粒度的特征
        all_features = torch.cat(
            [char_embeds, word_embeds, phrase_embeds, sentence_embeds],
            dim=-1
        )
        
        # 融合
        fused = self.fusion_layer(all_features)
        return fused
    
    def _build_hierarchy_graph(
        self,
        char_embeds: torch.Tensor,
        word_embeds: torch.Tensor,
        phrase_embeds: torch.Tensor,
        sentence_embeds: torch.Tensor
    ) -> Dict[str, Any]:
        """构建层次图结构
        
        构建各粒度之间的关联矩阵，实现层次图：
        - 字符到词的关联
        - 词到短语的关联
        - 短语到句子的关联
        
        Returns:
            包含关联矩阵和层次结构信息的字典
        """
        batch_size, seq_len, _ = char_embeds.shape
        
        # 计算各粒度之间的相似度作为关联权重
        # char_to_word: (B, L, L) - 每个字符对每个词的关联
        char_norm = torch.nn.functional.normalize(char_embeds, dim=-1)
        word_norm = torch.nn.functional.normalize(word_embeds, dim=-1)
        char_to_word = torch.bmm(char_norm, word_norm.transpose(1, 2))
        
        # word_to_phrase: (B, L, L)
        phrase_norm = torch.nn.functional.normalize(phrase_embeds, dim=-1)
        word_to_phrase = torch.bmm(word_norm, phrase_norm.transpose(1, 2))
        
        # phrase_to_sentence: (B, L, L)
        sentence_norm = torch.nn.functional.normalize(sentence_embeds, dim=-1)
        phrase_to_sentence = torch.bmm(phrase_norm, sentence_norm.transpose(1, 2))
        
        # 构建层次表示（每一层对上层的注意力权重
        # 对角线上的高值表示更强的关联
        hierarchy = {
            'char_to_word': char_to_word,
            'word_to_phrase': word_to_phrase,
            'phrase_to_sentence': phrase_to_sentence,
            'levels': ['char', 'word', 'phrase', 'sentence']
        }
        
        return hierarchy
