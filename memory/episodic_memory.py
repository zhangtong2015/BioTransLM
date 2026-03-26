# 模块K：情景记忆 (EpisodicMemory)
# 负责人：AI 4
# 输入：input_embeds: (B, L, D), context: Optional[Dict], retrieve_k: int
# 输出：{'episodes': List[Dict], 'similarity_scores': List[float], 'memory_state': torch.Tensor}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import uuid
import numpy as np

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class EpisodicMemoryConfig(BaseConfig):
    """情景记忆配置"""
    hidden_dim: int = 768
    max_episodes: int = 10000
    similarity_threshold: float = 0.6
    retrieval_top_k: int = 5
    temporal_weight: float = 0.3  # 时间权重（0-1）
    enable_sequence_matching: bool = True


class Episode:
    """情景记忆片段"""
    
    def __init__(
        self,
        content: str,
        embedding: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        episode_type: str = 'general',
        importance: float = 0.5,
        timestamp: Optional[float] = None,
        sequence_id: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.context = context or {}
        self.episode_type = episode_type
        self.importance = importance
        self.timestamp = timestamp or time.time()
        self.sequence_id = sequence_id or str(uuid.uuid4())  # 用于序列匹配
        self.position_in_sequence = self.context.get('position', 0)
        self.access_count = 0
        self.last_access = self.timestamp
    
    def access(self) -> None:
        """访问记忆片段"""
        self.access_count += 1
        self.last_access = time.time()
    
    def get_temporal_score(self, current_time: Optional[float] = None) -> float:
        """计算时间相关分数（越新分数越高）"""
        current_time = current_time or time.time()
        time_diff = current_time - self.timestamp
        # 指数衰减：1小时半衰期
        temporal_score = np.exp(-time_diff / 3600.0)
        return temporal_score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'content': self.content,
            'context': self.context,
            'episode_type': self.episode_type,
            'importance': self.importance,
            'timestamp': self.timestamp,
            'sequence_id': self.sequence_id,
            'position_in_sequence': self.position_in_sequence,
            'access_count': self.access_count,
            'last_access': self.last_access
        }


class EpisodicMemory(BaseModule):
    """
    情景记忆模块
    
    实现事件序列存储、时间戳关联检索、上下文模式补全。
    模拟人类情景记忆，存储特定时间和地点的事件或经历。
    """
    
    def __init__(
        self, 
        config: Optional[EpisodicMemoryConfig] = None,
        module_name: str = "episodic_memory"
    ):
        config = config or EpisodicMemoryConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 记忆存储
        self.episodes: List[Episode] = []
        
        # 序列索引: sequence_id -> [episode_ids]
        self.sequence_index: Dict[str, List[str]] = {}
        
        # 类型索引: episode_type -> [episode_ids]
        self.type_index: Dict[str, List[str]] = {}
        
        # 相似度匹配网络
        self.similarity_net = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim + 64, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim)
        )
        
        # 序列补全网络
        if self.config.enable_sequence_matching:
            self.sequence_completion = nn.Sequential(
                nn.Linear(self.config.hidden_dim * 3, 256),
                nn.ReLU(),
                nn.Linear(256, self.config.hidden_dim)
            )
        
        # 延迟导入faiss
        self.faiss = None
        self.index = None
        self._import_faiss()
    
    def _import_faiss(self) -> None:
        """导入faiss库（延迟导入）"""
        try:
            import faiss
            self.faiss = faiss
            self._init_faiss_index()
        except ImportError:
            print("警告: faiss未安装，将使用torch实现（速度较慢）")
            self.faiss = None
    
    def _init_faiss_index(self) -> None:
        """初始化FAISS索引"""
        if self.faiss is not None:
            self.index = self.faiss.IndexFlatIP(self.config.hidden_dim)
            if self.faiss.get_num_gpus() > 0:
                self.index = self.faiss.index_cpu_to_gpu(
                    self.faiss.StandardGpuResources(), 0, self.index
                )
    
    def forward(
        self,
        input_embeds: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        retrieve_k: Optional[int] = None,
        operation: str = 'retrieve',
        content: str = '',
        episode_type: str = 'general',
        importance: float = 0.5,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        前向传播 - 情景记忆操作
        
        Args:
            input_embeds: 输入嵌入 (batch_size, seq_len, hidden_dim) 或 (hidden_dim)
            context: 上下文信息字典
            retrieve_k: 检索返回数量
            operation: 操作类型 ['store', 'retrieve', 'retrieve_sequence', 'update', 'delete', 'clear']
            content: 存储时的内容文本
            episode_type: 情景类型
            importance: 重要性评分 (0-1)
            threshold: 相似度阈值
            
        Returns:
            Dict包含：
                episodes: 检索到的情景列表（字典形式）
                similarity_scores: 相似度分数列表
                memory_state: 聚合的记忆状态张量
                episode_count: 当前情景总数
        """
        retrieve_k = retrieve_k or self.config.retrieval_top_k
        threshold = threshold or self.config.similarity_threshold
        
        if operation == 'store':
            return self._store_episode(input_embeds, context, content, episode_type, importance)
        elif operation == 'retrieve':
            return self._retrieve_episodes(input_embeds, context, retrieve_k, threshold)
        elif operation == 'retrieve_sequence':
            return self._retrieve_by_sequence(input_embeds, context, retrieve_k)
        elif operation == 'update':
            return self._update_episode(input_embeds, context)
        elif operation == 'delete':
            return self._delete_episode(input_embeds)
        elif operation == 'clear':
            return self._clear_memory()
        else:
            return self._retrieve_episodes(input_embeds, context, retrieve_k, threshold)
    
    def _store_episode(
        self,
        input_embeds: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        content: str = '',
        episode_type: str = 'general',
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """存储新的情景记忆"""
        # 处理输入嵌入
        if len(input_embeds.shape) == 3:
            embeds = input_embeds.mean(dim=1)  # (B, D)
        elif len(input_embeds.shape) == 1:
            embeds = input_embeds.unsqueeze(0)  # (1, D)
        else:
            embeds = input_embeds
        
        batch_size = embeds.shape[0]
        new_episodes = []
        
        for i in range(batch_size):
            embed = embeds[i].detach().cpu()
            
            # 创建情景记忆
            episode = Episode(
                content=content if batch_size == 1 else f"{content}_{i}",
                embedding=embed,
                context=context,
                episode_type=episode_type,
                importance=importance
            )
            
            # 添加到记忆库
            self.episodes.append(episode)
            new_episodes.append(episode.to_dict())
            
            # 更新索引
            if episode.sequence_id not in self.sequence_index:
                self.sequence_index[episode.sequence_id] = []
            self.sequence_index[episode.sequence_id].append(episode.id)
            
            if episode.episode_type not in self.type_index:
                self.type_index[episode.episode_type] = []
            self.type_index[episode.episode_type].append(episode.id)
        
        # 检查容量限制
        if len(self.episodes) > self.config.max_episodes:
            self._prune_memory(len(self.episodes) - self.config.max_episodes)
        
        # 更新FAISS索引
        self._update_faiss_index()
        
        # 计算记忆状态
        memory_state = self._compute_memory_state()
        
        return {
            'episodes': new_episodes,
            'similarity_scores': [1.0] * len(new_episodes),
            'memory_state': memory_state,
            'episode_count': len(self.episodes),
            'stored_count': len(new_episodes)
        }
    
    def _retrieve_episodes(
        self,
        query_embeds: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        retrieve_k: int = 5,
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """根据查询检索情景记忆"""
        if not self.episodes:
            return self._empty_result()
        
        # 如果没有查询嵌入，返回最近的记忆
        if query_embeds is None:
            # 按时间排序返回最近的
            sorted_episodes = sorted(
                self.episodes,
                key=lambda x: x.timestamp,
                reverse=True
            )[:retrieve_k]
            
            scores = [ep.get_temporal_score() for ep in sorted_episodes]
            
            # 标记为已访问
            for ep in sorted_episodes:
                ep.access()
            
            memory_state = self._compute_memory_state()
            
            return {
                'episodes': [ep.to_dict() for ep in sorted_episodes],
                'similarity_scores': scores,
                'memory_state': memory_state,
                'episode_count': len(self.episodes),
                'retrieval_method': 'temporal'
            }
        
        # 处理查询嵌入
        if len(query_embeds.shape) == 3:
            query = query_embeds.mean(dim=1)  # (B, D)
        elif len(query_embeds.shape) == 1:
            query = query_embeds.unsqueeze(0)  # (1, D)
        else:
            query = query_embeds
        
        # 计算相似度
        similarities = self._compute_similarity(query, context)
        
        # 应用时间权重
        current_time = time.time()
        temporal_scores = torch.tensor(
            [ep.get_temporal_score(current_time) for ep in self.episodes],
            device=self._device
        )
        
        # 融合相似度和时间权重
        fused_scores = (
            similarities * (1 - self.config.temporal_weight) +
            temporal_scores.unsqueeze(0) * self.config.temporal_weight
        )
        
        # 应用重要性权重
        importance_weights = torch.tensor(
            [ep.importance for ep in self.episodes],
            device=self._device
        )
        fused_scores = fused_scores * importance_weights.unsqueeze(0)
        
        # 获取top-k结果
        retrieve_k = min(retrieve_k, len(self.episodes))
        top_scores, top_indices = torch.topk(fused_scores, retrieve_k, dim=-1)
        
        # 标记为已访问并收集结果
        retrieved_episodes = []
        final_scores = []
        
        for i in range(top_indices.shape[0]):
            batch_results = []
            batch_scores = []
            for j, idx in enumerate(top_indices[i]):
                idx_item = idx.item()
                if 0 <= idx_item < len(self.episodes) and top_scores[i][j] > threshold:
                    self.episodes[idx_item].access()
                    batch_results.append(self.episodes[idx_item].to_dict())
                    batch_scores.append(float(top_scores[i][j]))
            retrieved_episodes.append(batch_results)
            final_scores.append(batch_scores)
        
        memory_state = self._compute_memory_state()
        
        return {
            'episodes': retrieved_episodes[0] if len(retrieved_episodes) == 1 else retrieved_episodes,
            'similarity_scores': final_scores[0] if len(final_scores) == 1 else final_scores,
            'memory_state': memory_state,
            'episode_count': len(self.episodes),
            'retrieval_method': 'similarity'
        }
    
    def _retrieve_by_sequence(
        self,
        query_embeds: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        retrieve_k: int = 5
    ) -> Dict[str, Any]:
        """根据序列匹配检索情景记忆"""
        if not self.episodes or not self.config.enable_sequence_matching:
            return self._empty_result()
        
        # 首先找到最相似的情景
        base_result = self._retrieve_episodes(query_embeds, context, retrieve_k=1)
        if not base_result['episodes']:
            return self._empty_result()
        
        # 获取相似情景的序列ID
        sequence_id = base_result['episodes'][0]['sequence_id']
        
        # 检索同序列的所有情景
        if sequence_id in self.sequence_index:
            sequence_episode_ids = self.sequence_index[sequence_id]
            sequence_episodes = []
            
            for ep in self.episodes:
                if ep.id in sequence_episode_ids:
                    ep.access()
                    sequence_episodes.append(ep)
            
            # 按位置排序
            sequence_episodes.sort(key=lambda x: x.position_in_sequence)
            
            memory_state = self._compute_memory_state()
            
            return {
                'episodes': [ep.to_dict() for ep in sequence_episodes],
                'similarity_scores': [base_result['similarity_scores'][0]] * len(sequence_episodes),
                'memory_state': memory_state,
                'episode_count': len(self.episodes),
                'sequence_id': sequence_id,
                'sequence_length': len(sequence_episodes)
            }
        
        return self._empty_result()
    
    def _update_episode(
        self,
        input_embeds: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """更新情景记忆"""
        if not self.episodes:
            return self._empty_result()
        
        # 找到最相似的情景进行更新
        if len(input_embeds.shape) == 3:
            query = input_embeds.mean(dim=1)
        else:
            query = input_embeds
        
        similarities = self._compute_similarity(query, None)
        most_similar_idx = similarities.argmax(dim=-1).item()
        
        updated = False
        if 0 <= most_similar_idx < len(self.episodes):
            self.episodes[most_similar_idx].embedding = query[0].detach().cpu()
            if context:
                self.episodes[most_similar_idx].context.update(context)
            self.episodes[most_similar_idx].access()
            updated = True
        
        memory_state = self._compute_memory_state()
        
        return {
            'episodes': [self.episodes[most_similar_idx].to_dict()] if updated else [],
            'similarity_scores': [float(similarities[0][most_similar_idx])] if updated else [],
            'memory_state': memory_state,
            'episode_count': len(self.episodes),
            'updated_idx': most_similar_idx if updated else -1
        }
    
    def _delete_episode(
        self,
        query_embeds: torch.Tensor
    ) -> Dict[str, Any]:
        """删除情景记忆"""
        if not self.episodes:
            return self._empty_result()
        
        if len(query_embeds.shape) == 3:
            query = query_embeds.mean(dim=1)
        else:
            query = query_embeds
        
        similarities = self._compute_similarity(query, None)
        delete_idx = similarities.argmax(dim=-1).item()
        
        deleted = []
        if 0 <= delete_idx < len(self.episodes):
            deleted_ep = self.episodes.pop(delete_idx)
            deleted.append(deleted_ep.to_dict())
            
            # 更新索引
            for seq_id, ids in self.sequence_index.items():
                if deleted_ep.id in ids:
                    ids.remove(deleted_ep.id)
                    if not ids:
                        del self.sequence_index[seq_id]
                    break
            
            for typ, ids in self.type_index.items():
                if deleted_ep.id in ids:
                    ids.remove(deleted_ep.id)
                    if not ids:
                        del self.type_index[typ]
                    break
        
        self._update_faiss_index()
        memory_state = self._compute_memory_state()
        
        return {
            'episodes': deleted,
            'similarity_scores': [],
            'memory_state': memory_state,
            'episode_count': len(self.episodes),
            'deleted_count': len(deleted)
        }
    
    def _clear_memory(self) -> Dict[str, Any]:
        """清空所有情景记忆"""
        cleared_count = len(self.episodes)
        self.episodes = []
        self.sequence_index = {}
        self.type_index = {}
        
        if self.index is not None:
            self.index.reset()
        
        return {
            'episodes': [],
            'similarity_scores': [],
            'memory_state': torch.zeros(self.config.hidden_dim, device=self._device),
            'episode_count': 0,
            'cleared_count': cleared_count
        }
    
    def _compute_similarity(
        self,
        query: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """计算查询与记忆之间的相似度"""
        batch_size = query.shape[0]
        memory_size = len(self.episodes)
        
        if memory_size == 0:
            return torch.zeros(batch_size, 0, device=self._device)
        
        # 获取记忆嵌入 (M, D)
        memory_embeds = torch.stack([ep.embedding for ep in self.episodes]).to(self._device)
        
        # 使用FAISS快速检索（如果可用）
        if self.faiss is not None and self.index is not None and self.index.ntotal > 0:
            query_np = F.normalize(query, p=2, dim=-1).detach().cpu().numpy().astype('float32')
            similarities, _ = self.index.search(query_np, memory_size)
            return torch.tensor(similarities, device=self._device)
        
        # 使用PyTorch计算
        query_norm = F.normalize(query, p=2, dim=-1)
        memory_norm = F.normalize(memory_embeds, p=2, dim=-1)
        
        # 基础余弦相似度 (B, M)
        cosine_sim = torch.matmul(query_norm, memory_norm.transpose(0, 1))
        
        # 使用神经网络增强相似度计算
        query_expanded = query_norm.unsqueeze(1).expand(-1, memory_size, -1)
        memory_expanded = memory_norm.unsqueeze(0).expand(batch_size, -1, -1)
        
        combined = torch.cat([query_expanded, memory_expanded], dim=-1)
        enhanced_sim = self.similarity_net(combined).squeeze(-1)  # (B, M)
        
        # 融合两种相似度
        final_sim = cosine_sim * 0.7 + enhanced_sim * 0.3
        
        return final_sim
    
    def _compute_memory_state(self) -> torch.Tensor:
        """计算聚合的记忆状态张量"""
        if not self.episodes:
            return torch.zeros(self.config.hidden_dim, device=self._device)
        
        # 加权平均：重要性 * 时间分数
        current_time = time.time()
        weights = torch.tensor(
            [ep.importance * ep.get_temporal_score(current_time) for ep in self.episodes],
            device=self._device
        )
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        
        memory_embeds = torch.stack([ep.embedding for ep in self.episodes]).to(self._device)
        weighted_state = (memory_embeds * weights.unsqueeze(1)).sum(dim=0)
        
        return weighted_state
    
    def _update_faiss_index(self) -> None:
        """更新FAISS索引"""
        if self.faiss is not None and self.index is not None and len(self.episodes) > 0:
            embeddings = np.array([
                F.normalize(ep.embedding, p=2, dim=-1).numpy()
                for ep in self.episodes
            ]).astype('float32')
            
            self.index.reset()
            self.index.add(embeddings)
    
    def _prune_memory(self, num_to_remove: int) -> None:
        """修剪记忆，移除最旧/最不重要的"""
        if num_to_remove <= 0:
            return
        
        # 按综合评分排序
        current_time = time.time()
        self.episodes.sort(
            key=lambda x: x.importance * x.get_temporal_score(current_time)
        )
        self.episodes = self.episodes[num_to_remove:]
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'episodes': [],
            'similarity_scores': [],
            'memory_state': torch.zeros(self.config.hidden_dim, device=self._device),
            'episode_count': 0
        }
    
    def get_episode_by_id(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取情景记忆"""
        for ep in self.episodes:
            if ep.id == episode_id:
                return ep.to_dict()
        return None
    
    def get_episodes_by_type(self, episode_type: str) -> List[Dict[str, Any]]:
        """根据类型获取情景记忆"""
        return [
            ep.to_dict() for ep in self.episodes
            if ep.episode_type == episode_type
        ]
    
    def get_episodes_in_time_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[Dict[str, Any]]:
        """获取指定时间范围内的情景记忆"""
        return [
            ep.to_dict() for ep in self.episodes
            if start_time <= ep.timestamp <= end_time
        ]
