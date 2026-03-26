# 模块N：前向模型 (ForwardModel)
# 负责人：AI 3
# 输入：candidate_text: str, context: Optional[Dict], reference: Optional[str]
# 输出：{'quality_score': float, 'feedback_signals': Dict, 'metric_details': Dict}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import math

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class ForwardModelConfig(BaseConfig):
    """前向模型配置"""
    hidden_dim: int = 768
    max_sequence_length: int = 512
    num_layers: int = 3
    num_heads: int = 8
    dropout_rate: float = 0.1
    quality_threshold: float = 0.6
    fluency_weight: float = 0.3
    coherence_weight: float = 0.3
    relevance_weight: float = 0.25
    diversity_weight: float = 0.15


class QualityScorer(nn.Module):
    """质量评分器 - 多维度评估文本质量"""
    
    def __init__(self, config: ForwardModelConfig):
        super().__init__()
        self.config = config
        
        # 特征提取层
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        )
        
        # 评分头
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim // 4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算质量评分
        Args:
            x: (B, L, D) 输入嵌入
        Returns:
            score: (B, 1) 质量评分 [0, 1]
            features: (B, D/4) 提取的特征
        """
        # 平均池化获得句子级表示
        if x.dim() == 3:
            x = x.mean(dim=1)  # (B, D)
        
        features = self.feature_encoder(x)
        score = self.scorer(features)
        
        return score, features


class FluencyEvaluator:
    """流畅度评估器"""
    
    @staticmethod
    def calculate_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """计算困惑度"""
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = math.exp(min(loss.item(), 10))
        return max(0, 1 - min(perplexity / 100, 1))  # 归一化到[0, 1]
    
    @staticmethod
    def calculate_repetition_penalty(text: str, ngram_range: Tuple[int, int] = (2, 4)) -> float:
        """计算重复惩罚"""
        words = text.split()
        if len(words) < 2:
            return 1.0
        
        repetitions = 0
        total = 0
        
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = []
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                if ngram in ngrams:
                    repetitions += 1
                ngrams.append(ngram)
                total += 1
        
        if total == 0:
            return 1.0
        
        repetition_rate = repetitions / total
        return max(0, 1 - repetition_rate * 2)


class CoherenceEvaluator:
    """连贯性评估器"""
    
    @staticmethod
    def calculate_semantic_coherence(embeddings: torch.Tensor) -> float:
        """计算语义连贯性"""
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)  # (L, D)
        
        if len(embeddings) < 2:
            return 1.0
        
        # 计算相邻词嵌入的余弦相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = utils.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim.item())
        
        return float(np.mean(similarities))
    
    @staticmethod
    def calculate_topic_consistency(embeddings: torch.Tensor, window_size: int = 5) -> float:
        """计算主题一致性"""
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        
        if len(embeddings) < window_size:
            return 1.0
        
        # 滑动窗口计算主题稳定性
        topic_embeddings = []
        for i in range(len(embeddings) - window_size + 1):
            window_emb = embeddings[i:i + window_size].mean(dim=0)
            topic_embeddings.append(window_emb)
        
        topic_embeddings = torch.stack(topic_embeddings)
        similarities = []
        
        for i in range(len(topic_embeddings) - 1):
            sim = utils.cosine_similarity(topic_embeddings[i], topic_embeddings[i + 1])
            similarities.append(sim.item())
        
        return float(np.mean(similarities)) if similarities else 1.0


class RelevanceEvaluator:
    """相关性评估器"""
    
    @staticmethod
    def calculate_context_relevance(
        candidate_emb: torch.Tensor,
        context_emb: torch.Tensor
    ) -> float:
        """计算与上下文的相关性"""
        if candidate_emb.dim() == 3:
            candidate_emb = candidate_emb.mean(dim=1)  # (B, D)
        if context_emb.dim() == 3:
            context_emb = context_emb.mean(dim=1)  # (B, D)
        
        similarity = utils.cosine_similarity(candidate_emb, context_emb)
        return float(similarity.mean().item())
    
    @staticmethod
    def calculate_keyword_overlap(candidate: str, reference: str) -> float:
        """计算关键词重叠率"""
        from collections import Counter
        import re
        
        def get_keywords(text: str) -> set:
            # 简单的关键词提取
            words = re.findall(r'\w+', text.lower())
            return set(words)
        
        cand_keywords = get_keywords(candidate)
        ref_keywords = get_keywords(reference)
        
        if not ref_keywords:
            return 1.0
        
        intersection = cand_keywords & ref_keywords
        union = cand_keywords | ref_keywords
        
        return len(intersection) / len(union) if union else 0


class DiversityEvaluator:
    """多样性评估器"""
    
    @staticmethod
    def calculate_vocab_diversity(text: str) -> float:
        """计算词汇多样性"""
        words = text.split()
        if not words:
            return 0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    @staticmethod
    def calculate_ngram_diversity(text: str, n: int = 2) -> float:
        """计算n-gram多样性"""
        words = text.split()
        if len(words) < n:
            return 1.0
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i + n]))
        
        unique_ngrams = set(ngrams)
        return len(unique_ngrams) / len(ngrams)


class ForwardModel(BaseModule):
    """
    前向模型模块
    
    对生成的候选文本进行质量评估，提供多维度反馈信号，
    用于指导生成器的优化和候选选择。
    """
    
    def __init__(
        self, 
        config: Optional[ForwardModelConfig] = None,
        module_name: str = "forward_model"
    ):
        config = config or ForwardModelConfig()
        super().__init__(config=config, module_name=module_name)
        
        # 初始化评估器
        self.fluency_evaluator = FluencyEvaluator()
        self.coherence_evaluator = CoherenceEvaluator()
        self.relevance_evaluator = RelevanceEvaluator()
        self.diversity_evaluator = DiversityEvaluator()
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 质量评分网络
        self.quality_scorer = QualityScorer(self.config)
        
        # 嵌入投影层（用于将输入嵌入转换为统一维度）
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate)
        )
        
        # 反馈信号生成器
        self.feedback_generator = nn.Sequential(
            nn.Linear(self.config.hidden_dim // 4 + 4, 128),  # 特征 + 4个基础度量
            nn.GELU(),
            nn.Linear(128, 64)
        )
    
    def _compute_basic_metrics(
        self,
        candidate_text: str,
        context_embeds: Optional[torch.Tensor] = None,
        candidate_embeds: Optional[torch.Tensor] = None,
        reference_text: Optional[str] = None
    ) -> Dict[str, float]:
        """计算基础评估指标"""
        metrics = {}
        
        # 流畅度指标
        metrics['fluency_repetition'] = self.fluency_evaluator.calculate_repetition_penalty(candidate_text)
        
        # 连贯性指标
        if candidate_embeds is not None:
            metrics['coherence_semantic'] = self.coherence_evaluator.calculate_semantic_coherence(candidate_embeds)
            metrics['coherence_topic'] = self.coherence_evaluator.calculate_topic_consistency(candidate_embeds)
        else:
            metrics['coherence_semantic'] = 0.5
            metrics['coherence_topic'] = 0.5
        
        # 相关性指标
        if context_embeds is not None and candidate_embeds is not None:
            metrics['relevance_context'] = self.relevance_evaluator.calculate_context_relevance(
                candidate_embeds, context_embeds
            )
        else:
            metrics['relevance_context'] = 0.5
        
        if reference_text:
            metrics['relevance_keyword'] = self.relevance_evaluator.calculate_keyword_overlap(
                candidate_text, reference_text
            )
        else:
            metrics['relevance_keyword'] = 1.0  # 无参考时默认满分
        
        # 多样性指标
        metrics['diversity_vocab'] = self.diversity_evaluator.calculate_vocab_diversity(candidate_text)
        metrics['diversity_ngram'] = self.diversity_evaluator.calculate_ngram_diversity(candidate_text)
        
        return metrics
    
    def _compute_feedback_signals(
        self,
        metrics: Dict[str, float],
        quality_features: torch.Tensor,
        quality_score: float
    ) -> Dict[str, Any]:
        """生成反馈信号用于指导生成优化"""
        feedback = {}
        
        # 判断是否需要改进
        needs_improvement = quality_score < self.config.quality_threshold
        
        # 识别薄弱维度
        weak_dimensions = []
        if metrics.get('fluency_repetition', 1.0) < 0.7:
            weak_dimensions.append('fluency')
        if metrics.get('coherence_semantic', 1.0) < 0.6:
            weak_dimensions.append('coherence')
        if metrics.get('relevance_context', 1.0) < 0.6:
            weak_dimensions.append('relevance')
        if metrics.get('diversity_vocab', 1.0) < 0.5:
            weak_dimensions.append('diversity')
        
        # 生成调整建议
        adjustments = {}
        if 'fluency' in weak_dimensions:
            adjustments['temperature'] = -0.1  # 降低温度
            adjustments['repetition_penalty'] = 0.2
        if 'diversity' in weak_dimensions:
            adjustments['temperature'] = adjustments.get('temperature', 0) + 0.15
            adjustments['top_k'] = 10
        if 'relevance' in weak_dimensions:
            adjustments['top_p'] = -0.1
            adjustments['context_weight'] = 0.2
        
        feedback.update({
            'needs_improvement': needs_improvement,
            'weak_dimensions': weak_dimensions,
            'adjustments': adjustments,
            'confidence': quality_score,
            'acceptance_threshold': self.config.quality_threshold
        })
        
        return feedback
    
    def forward(
        self,
        candidate_text: str,
        context_embeds: Optional[torch.Tensor] = None,
        candidate_embeds: Optional[torch.Tensor] = None,
        reference_text: Optional[str] = None,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        前向传播 - 评估候选文本质量
        
        Args:
            candidate_text: 候选文本字符串
            context_embeds: 上下文嵌入 (B, L, D) 可选
            candidate_embeds: 候选文本嵌入 (B, L, D) 可选
            reference_text: 参考文本 可选
            logits: 模型输出logits (B, L, V) 可选
            labels: 标签序列 (B, L) 可选
        
        Returns:
            {
                'quality_score': float,  # 综合质量评分 [0, 1]
                'feedback_signals': Dict,  # 反馈调节信号
                'metric_details': Dict,  # 各维度详细得分
                'is_acceptable': bool  # 是否达到质量阈值
            }
        """
        # 设备适配
        if candidate_embeds is not None:
            candidate_embeds = candidate_embeds.to(self._device)
            candidate_embeds = self.embedding_projection(candidate_embeds)
        
        if context_embeds is not None:
            context_embeds = context_embeds.to(self._device)
        
        # 1. 计算基础度量
        basic_metrics = self._compute_basic_metrics(
            candidate_text,
            context_embeds,
            candidate_embeds,
            reference_text
        )
        
        # 2. 使用神经网络计算深度质量评分
        quality_score = 0.5  # 默认值
        quality_features = None
        
        if candidate_embeds is not None:
            quality_score_tensor, quality_features = self.quality_scorer(candidate_embeds)
            quality_score = float(quality_score_tensor.mean().item())
        
        # 3. 计算困惑度（如果有logits和labels）
        if logits is not None and labels is not None:
            logits = logits.to(self._device)
            labels = labels.to(self._device)
            perplexity_score = self.fluency_evaluator.calculate_perplexity(logits, labels)
            basic_metrics['fluency_perplexity'] = perplexity_score
        else:
            basic_metrics['fluency_perplexity'] = 0.7  # 默认值
        
        # 4. 加权融合所有指标
        fluency_score = np.mean([
            basic_metrics.get('fluency_repetition', 0.5),
            basic_metrics.get('fluency_perplexity', 0.5)
        ])
        
        coherence_score = np.mean([
            basic_metrics.get('coherence_semantic', 0.5),
            basic_metrics.get('coherence_topic', 0.5)
        ])
        
        relevance_score = np.mean([
            basic_metrics.get('relevance_context', 0.5),
            basic_metrics.get('relevance_keyword', 0.5)
        ])
        
        diversity_score = np.mean([
            basic_metrics.get('diversity_vocab', 0.5),
            basic_metrics.get('diversity_ngram', 0.5)
        ])
        
        # 加权综合评分
        weighted_score = (
            fluency_score * self.config.fluency_weight +
            coherence_score * self.config.coherence_weight +
            relevance_score * self.config.relevance_weight +
            diversity_score * self.config.diversity_weight
        )
        
        # 融合神经网络评分和规则评分
        final_score = float(0.6 * weighted_score + 0.4 * quality_score)
        final_score = max(0.0, min(1.0, final_score))  # 限制在[0, 1]
        
        # 5. 生成反馈信号
        feedback_signals = self._compute_feedback_signals(
            basic_metrics,
            quality_features if quality_features is not None else torch.zeros(1, self.config.hidden_dim // 4),
            final_score
        )
        
        # 6. 整理详细度量
        metric_details = {
            'overall': final_score,
            'fluency': {
                'repetition': basic_metrics.get('fluency_repetition', 0.0),
                'perplexity': basic_metrics.get('fluency_perplexity', 0.0),
                'overall': fluency_score
            },
            'coherence': {
                'semantic': basic_metrics.get('coherence_semantic', 0.0),
                'topic': basic_metrics.get('coherence_topic', 0.0),
                'overall': coherence_score
            },
            'relevance': {
                'context': basic_metrics.get('relevance_context', 0.0),
                'keyword': basic_metrics.get('relevance_keyword', 0.0),
                'overall': relevance_score
            },
            'diversity': {
                'vocab': basic_metrics.get('diversity_vocab', 0.0),
                'ngram': basic_metrics.get('diversity_ngram', 0.0),
                'overall': diversity_score
            }
        }
        
        return {
            'quality_score': final_score,
            'feedback_signals': feedback_signals,
            'metric_details': metric_details,
            'is_acceptable': final_score >= self.config.quality_threshold
        }
    
    def rank_candidates(
        self,
        candidates: List[str],
        context_embeds: Optional[torch.Tensor] = None,
        candidate_embeds_list: Optional[List[torch.Tensor]] = None,
        reference_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        对多个候选文本进行排序
        
        Args:
            candidates: 候选文本列表
            context_embeds: 上下文嵌入
            candidate_embeds_list: 每个候选的嵌入列表
            reference_text: 参考文本
        
        Returns:
            排序后的结果列表，包含评分和排名
        """
        results = []
        for i, candidate in enumerate(candidates):
            embeds = candidate_embeds_list[i] if candidate_embeds_list else None
            result = self.forward(
                candidate_text=candidate,
                context_embeds=context_embeds,
                candidate_embeds=embeds,
                reference_text=reference_text
            )
            result['candidate_text'] = candidate
            result['index'] = i
            results.append(result)
        
        # 按质量评分降序排序
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # 添加排名
        for rank, result in enumerate(results):
            result['rank'] = rank + 1
        
        return results