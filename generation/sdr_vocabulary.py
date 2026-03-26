# -*- coding: utf-8 -*-
"""
SDR词汇表 - SDRVocabulary

实现稀疏分布式表示(SDR)与token之间的双向映射：
1. token id → SDR表示（可训练的稀疏原型）
2. SDR激活 → token概率分布（基于重叠相似度）

核心特性：
- 语义相似的token具有重叠的SDR模式
- 可训练的原型向量支持端到端优化
- 稀疏计算保持生物合理性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import math

from config import BaseConfig
from core.base_module import BaseModule

logger = logging.getLogger(__name__)

@dataclass
class SDRVocabularyConfig(BaseConfig):
    """SDR词汇表配置"""
    vocab_size: int = 50257          # 词汇表大小
    sdr_size: int = 4096             # SDR维度
    activation_sparsity: float = 0.02 # 激活稀疏度（2%）
    similarity_temp: float = 10.0     # 相似度计算温度
    use_stable_hash: bool = True      # 使用稳定哈希初始化
    embedding_dim: int = 768          # 辅助嵌入维度
    update_prototypes: bool = True     # 是否允许原型更新


class SDRPrototypeInitializer:
    """
    SDR原型向量初始化器
    
    使用稳定哈希或随机正交初始化确保：
    1. 每个token的SDR是唯一的
    2. 语义相似的token有更多重叠
    3. 稀疏度保持在目标水平
    """
    
    @staticmethod
    def stable_hash_init(vocab_size: int, 
                         sdr_size: int, 
                         sparsity: float,
                         seed: int = 42) -> torch.Tensor:
        """
        使用稳定哈希初始化原型
        
        确保：
        - 每个token有k个激活位
        - 任意两个token的期望重叠是k^2 / sdr_size
        """
        k = max(1, int(sdr_size * sparsity))
        prototypes = torch.zeros(vocab_size, sdr_size)
        
        # 伪随机但可重现的初始化
        for token_id in range(vocab_size):
            # 使用确定性随机数生成
            random_gen = torch.Generator().manual_seed(seed + token_id)
            # 随机选择k个位置（无重复）
            indices = torch.randperm(sdr_size, generator=random_gen)[:k]
            prototypes[token_id, indices] = 1.0
            
        return prototypes
    
    @staticmethod
    def semantic_aware_init(vocab_size: int,
                            sdr_size: int,
                            sparsity: float,
                            pretrained_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        基于预训练嵌入的语义感知初始化
        
        如果提供了预训练嵌入，确保语义相似的词有更多重叠
        """
        k = max(1, int(sdr_size * sparsity))
        
        if pretrained_embeds is None:
            return SDRPrototypeInitializer.stable_hash_init(vocab_size, sdr_size, sparsity)
        
        # 使用PCA降维到SDR空间
        from sklearn.decomposition import PCA
        import numpy as np
        
        embeds_np = pretrained_embeds.numpy()
        pca = PCA(n_components=min(100, vocab_size - 1))
        reduced = pca.fit_transform(embeds_np)
        
        # 基于降维结果分配SDR位
        prototypes = torch.zeros(vocab_size, sdr_size)
        for i in range(vocab_size):
            # 基于各维度符号选择位
            pos_bits = []
            for d in range(min(100, reduced.shape[1])):
                if reduced[i, d] > 0:
                    pos = d * int(sdr_size / 100) + int(abs(reduced[i, d]) * 10) % int(sdr_size / 100)
                    pos_bits.append(pos)
            
            # 补充到k个位
            if len(pos_bits) < k:
                extra = torch.randint(0, sdr_size, (k - len(pos_bits),))
                pos_bits.extend(extra.tolist())
            elif len(pos_bits) > k:
                pos_bits = pos_bits[:k]
            
            prototypes[i, pos_bits] = 1.0
        
        return prototypes


class SDRVocabulary(BaseModule):
    """
    SDR词汇表 - 稀疏分布式表示与token的双向映射
    
    实现两种映射：
    1. token id → SDR: 每个token对应一个稀疏原型向量
    2. SDR → token概率: 基于重叠相似度的软匹配
    """
    
    def __init__(self, config: Optional[SDRVocabularyConfig] = None):
        self.config = config or SDRVocabularyConfig()
        self.k = max(1, int(self.config.sdr_size * self.config.activation_sparsity))
        super().__init__(config=self.config, module_name="sdr_vocabulary")
    
    def _initialize_module(self):
        """初始化SDR词汇表组件"""
        # SDR原型矩阵 [vocab_size, sdr_size]
        self.prototype_sdrs = nn.Parameter(
            SDRPrototypeInitializer.stable_hash_init(
                self.config.vocab_size,
                self.config.sdr_size,
                self.config.activation_sparsity
            ).to(self._device),
            requires_grad=self.config.update_prototypes
        )
        
        # 辅助嵌入（用于梯度流和附加语义）
        self.token_embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_dim
        )
        
        # SDR到嵌入的投影
        self.sdr_to_embedding = nn.Sequential(
            nn.Linear(self.config.sdr_size, self.config.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim * 2, self.config.embedding_dim)
        )
        
        # 嵌入到token logits的投影
        self.embedding_to_logits = nn.Linear(
            self.config.embedding_dim,
            self.config.vocab_size
        )
        
        # 层归一化
        self.sdr_norm = nn.LayerNorm(self.config.sdr_size)
        self.embedding_norm = nn.LayerNorm(self.config.embedding_dim)
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(
            torch.tensor(self.config.similarity_temp),
            requires_grad=True
        )
        
        logger.info(f"SDR词汇表初始化完成: "
                        f"词汇量={self.config.vocab_size}, "
                        f"SDR维度={self.config.sdr_size}, "
                        f"激活位数量={self.k}, "
                        f"稀疏度={self.config.activation_sparsity:.1%}")
    
    def token_to_sdr(self, 
                    token_ids: torch.Tensor, 
                    use_embedding: bool = False) -> torch.Tensor:
        """
        将token id转换为SDR表示
        
        Args:
            token_ids: token id张量 [batch_size, seq_len]
            use_embedding: 是否使用嵌入增强
            
        Returns:
            sdr: SDR表示 [batch_size, seq_len, sdr_size]
        """
        # 确保在正确的设备上
        token_ids = token_ids.to(self._device)
        batch_size, seq_len = token_ids.shape
        
        # 1. 获取原型SDR
        sdr = F.embedding(token_ids, self.prototype_sdrs)  # [B, T, S]
        
        # 2. 可选：使用嵌入信息增强
        if use_embedding:
            embeds = self.token_embeddings(token_ids)  # [B, T, D]
            embed_proj = torch.tanh(self.sdr_to_embedding(sdr))  # [B, T, D]
            gate = torch.sigmoid(torch.sum(embeds * embed_proj, dim=-1, keepdim=True))
            sdr = sdr * gate + sdr * (1 - gate) * 0.1
        
        # 3. 确保稀疏度（仅保留top-k激活）
        sdr_flat = sdr.view(-1, self.config.sdr_size)
        topk_values, topk_indices = torch.topk(sdr_flat, self.k, dim=-1)
        
        sparse_sdr = torch.zeros_like(sdr_flat)
        sparse_sdr.scatter_(-1, topk_indices, topk_values)
        sparse_sdr = sparse_sdr.view(batch_size, seq_len, self.config.sdr_size)
        
        return sparse_sdr
    
    def _compute_overlap_batch(self, 
                              sdr_batch: torch.Tensor, 
                              prototypes: torch.Tensor) -> torch.Tensor:
        """
        批量计算SDR与所有原型的重叠相似度
        
        Args:
            sdr_batch: SDR张量 [batch_size, seq_len, sdr_size] 或 [batch_size, sdr_size]
            prototypes: 原型矩阵 [vocab_size, sdr_size]
            
        Returns:
            相似度矩阵 [batch_size, seq_len, vocab_size] 或 [batch_size, vocab_size]
        """
        # 处理二维输入
        if len(sdr_batch.shape) == 2:
            sdr_batch = sdr_batch.unsqueeze(1)
        
        batch_size, seq_len, sdr_size = sdr_batch.shape
        device = sdr_batch.device
        
        # 二值化SDR（简化重叠计算）
        sdr_binary = (sdr_batch > 0.1).float()
        proto_binary = (prototypes > 0.1).float().to(device)
        
        # 计算每个SDR与所有原型的重叠
        # 使用矩阵乘法高效计算: [B, T, S] @ [S, V] -> [B, T, V]
        overlaps = torch.matmul(sdr_binary, proto_binary.T)
        
        # 归一化相似度（除以每个SDR的激活位数）
        sdr_nnz = sdr_binary.sum(dim=-1, keepdim=True).clamp(min=1)
        proto_nnz = proto_binary.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Jaccard相似度
        union = sdr_nnz + proto_nnz.T - overlaps
        jaccard = overlaps / union.clamp(min=1e-8)
        
        return jaccard
    
    def sdr_to_token_logits(self, 
                           sdr: torch.Tensor, 
                           temperature: Optional[float] = None) -> torch.Tensor:
        """
        将SDR表示转换为token logits概率分布
        
        Args:
            sdr: SDR张量 [batch_size, seq_len, sdr_size] 或 [batch_size, sdr_size]
            temperature: 采样温度（None使用可学习的温度参数）
            
        Returns:
            token_logits: token logits [batch_size, seq_len, vocab_size]
        """
        # 确保在正确的设备上
        sdr = sdr.to(self._device)
        batch_shape = sdr.shape[:-1]
        
        # 1. 基于重叠相似度的logits
        overlap_logits = self._compute_overlap_batch(sdr, self.prototype_sdrs)
        
        # 2. 基于嵌入的logits（用于梯度流和语义平滑）
        embeds = self.sdr_to_embedding(sdr)
        embeds = self.embedding_norm(embeds)
        embed_logits = self.embedding_to_logits(embeds)
        
        # 3. 融合两种logits
        temp = temperature if temperature is not None else self.temperature
        fused_logits = (overlap_logits * 0.7 + embed_logits * 0.3) * temp
        
        return fused_logits
    
    def sdr_to_token_probs(self,
                          sdr: torch.Tensor,
                          temperature: float = 1.0,
                          top_k: Optional[int] = None,
                          top_p: Optional[float] = None) -> torch.Tensor:
        """
        将SDR转换为token概率分布，支持采样策略
        
        Args:
            sdr: SDR张量
            temperature: 采样温度
            top_k: top-k采样（None表示不使用）
            top_p: nucleus采样（None表示不使用）
            
        Returns:
            token_probs: token概率分布
        """
        logits = self.sdr_to_token_logits(sdr, temperature=1.0)
        
        # 应用温度
        logits = logits / temperature
        
        # Top-k采样
        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        
        # Nucleus (top-p) 采样
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def sample_tokens(self,
                     sdr: torch.Tensor,
                     temperature: float = 1.0,
                     top_k: Optional[int] = 50,
                     top_p: Optional[float] = 0.9,
                     do_sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从SDR表示中采样token
        
        Args:
            sdr: SDR张量
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
            do_sample: 是否采样（False表示贪婪解码）
            
        Returns:
            token_ids: 采样的token id [batch_size, seq_len]
            token_probs: token概率 [batch_size, seq_len]
        """
        probs = self.sdr_to_token_probs(sdr, temperature, top_k, top_p)
        
        if do_sample:
            # 重新标准化概率（处理可能的数值问题）
            probs_reshaped = probs.view(-1, self.config.vocab_size)
            token_ids = torch.multinomial(probs_reshaped, num_samples=1).view(probs.shape[:-1])
        else:
            # 贪婪解码
            token_ids = torch.argmax(probs, dim=-1)
        
        # 收集采样token的概率
        token_probs = torch.gather(
            probs, -1, token_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_ids, token_probs
    
    def update_prototype(self, 
                        token_id: int, 
                        activation_pattern: torch.Tensor,
                        learning_rate: float = 0.01) -> None:
        """
        更新特定token的原型SDR
        
        用于在线学习，根据新的激活模式调整原型
        """
        if not self.config.update_prototypes:
            return
        
        # 将新激活模式二值化
        binary_pattern = (activation_pattern > 0.1).float()
        
        # 只更新top-k位置
        k = self.k
        if binary_pattern.sum() > k:
            topk_values, topk_indices = torch.topk(activation_pattern, k)
            binary_pattern = torch.zeros_like(activation_pattern)
            binary_pattern[topk_indices] = 1.0
        
        # 滑动平均更新原型
        with torch.no_grad():
            self.prototype_sdrs[token_id] = (
                self.prototype_sdrs[token_id] * (1 - learning_rate) +
                binary_pattern * learning_rate
            )
            # 保持稀疏度
            if self.prototype_sdrs[token_id].sum() > k * 1.2:
                topk = torch.topk(self.prototype_sdrs[token_id], k)[0][-1]
                self.prototype_sdrs[token_id][self.prototype_sdrs[token_id] < topk] = 0
    
    def compute_semantic_similarity(self, 
                                   token_id_1: int, 
                                   token_id_2: int) -> float:
        """
        计算两个token之间的语义相似度（基于SDR重叠）
        
        返回：Jaccard相似度 [0, 1]
        """
        proto1 = (self.prototype_sdrs[token_id_1] > 0.1).float()
        proto2 = (self.prototype_sdrs[token_id_2] > 0.1).float()
        
        intersection = (proto1 * proto2).sum()
        union = ((proto1 + proto2) > 0).sum()
        
        return float(intersection / union.clamp(min=1))
    
    def forward(self, 
                x: torch.Tensor, 
                mode: str = 'token_to_sdr',
                **kwargs) -> Dict[str, Any]:
        """
        统一前向传播接口
        
        Args:
            x: 输入张量（token_ids或SDR）
            mode: 'token_to_sdr' 或 'sdr_to_token'
            **kwargs: 其他参数
            
        Returns:
            包含输出的字典
        """
        # 确保在正确的设备上
        x = x.to(self._device)
        
        if mode == 'token_to_sdr':
            sdr = self.token_to_sdr(x, **kwargs)
            embeddings = self.token_embeddings(x)
            return {
                'sdr': sdr,
                'embeddings': embeddings
            }
        
        elif mode == 'sdr_to_token':
            logits = self.sdr_to_token_logits(x, **kwargs)
            probs = F.softmax(logits, dim=-1)
            return {
                'token_logits': logits,
                'token_probs': probs
            }
        
        else:
            raise ValueError(f"未知模式: {mode}")
