# 模块J：工作记忆 (WorkingMemory)
# 负责人：AI 4
# 输入：input_embeds: (B, L, D), operation: str, max_items: Optional[int]
# 输出：{'memory_items': List[Dict], 'memory_state': torch.Tensor, 'attention_weights': torch.Tensor}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import uuid

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class WorkingMemoryConfig(BaseConfig):
    """工作记忆配置"""
    hidden_dim: int = 768
    max_memory_items: int = 10  # Miller's law: 7±2
    memory_decay_rate: float = 0.05  # 记忆衰减率
    attention_temperature: float = 1.0
    enable_decay: bool = True


class WorkingMemoryItem:
    """工作记忆项"""
    
    def __init__(
        self,
        content: Any,
        embedding: torch.Tensor,
        item_type: str = 'general',
        importance: float = 0.5,
        timestamp: Optional[float] = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.item_type = item_type
        self.importance = importance
        self.timestamp = timestamp or time.time()
        self.activation = 1.0  # 激活程度
        self.access_count = 0  # 访问次数
    
    def access(self) -> None:
        """访问记忆项，增加访问次数和激活程度"""
        self.access_count += 1
        self.activation = min(1.0, self.activation + 0.1)
    
    def decay(self, rate: float) -> None:
        """记忆衰减"""
        self.activation *= (1.0 - rate)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'content': self.content,
            'item_type': self.item_type,
            'importance': self.importance,
            'timestamp': self.timestamp,
            'activation': self.activation,
            'access_count': self.access_count
        }


class WorkingMemory(BaseModule):
    """
    工作记忆模块
    
    实现有限容量槽存储（约7±2项）、注意力门控更新、近期性/重要性加权。
    模拟人类工作记忆的特性，支持读写操作、衰减和刷新机制。
    """
    
    def __init__(
        self, 
        config: Optional[WorkingMemoryConfig] = None,
        module_name: str = "working_memory"
    ):
        config = config or WorkingMemoryConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 记忆存储
        self.memory_items: List[WorkingMemoryItem] = []
        
        # 注意力控制器
        self.attention_controller = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 记忆更新门
        self.update_gate = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 记忆状态投影
        self.state_projection = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim)
        )
        
        # 记忆操作模式
        self.operation_modes = {
            'read': self._read_operation,
            'write': self._write_operation,
            'update': self._update_operation,
            'delete': self._delete_operation,
            'clear': self._clear_operation,
            'refresh': self._refresh_operation
        }
    
    def forward(
        self,
        input_embeds: Optional[torch.Tensor] = None,
        operation: str = 'read',
        max_items: Optional[int] = None,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        前向传播 - 工作记忆操作
        
        Args:
            input_embeds: 输入嵌入 (batch_size, seq_len, hidden_dim) 或 (hidden_dim)
            operation: 操作类型 ['read', 'write', 'update', 'delete', 'clear', 'refresh']
            max_items: 最大记忆项数（覆盖配置）
            top_k: 读取时返回top-k相关记忆
            threshold: 记忆激活阈值
            
        Returns:
            Dict包含：
                memory_items: 记忆项列表（字典形式）
                memory_state: 聚合的记忆状态张量
                attention_weights: 注意力权重
                item_count: 当前记忆项数量
        """
        max_items = max_items or self.config.max_memory_items
        
        # 执行记忆衰减
        if self.config.enable_decay:
            self._apply_decay()
        
        # 执行指定操作
        operation_func = self.operation_modes.get(operation.lower(), self._read_operation)
        result = operation_func(input_embeds, max_items, top_k, threshold)
        
        return result
    
    def _read_operation(
        self,
        query_embeds: Optional[torch.Tensor] = None,
        max_items: int = 10,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """读取操作：根据查询检索相关记忆"""
        if not self.memory_items:
            return self._empty_result()
        
        # 计算注意力权重
        if query_embeds is not None:
            attention_weights = self._compute_attention(query_embeds)
        else:
            # 没有查询时，按激活程度和重要性排序
            attention_weights = torch.tensor(
                [item.activation * item.importance for item in self.memory_items],
                device=self.config.device
            )
        
        # 过滤掉激活程度低的记忆
        active_mask = torch.tensor(
            [item.activation > threshold for item in self.memory_items],
            device=self.config.device
        )
        attention_weights = attention_weights * active_mask.float()
        
        # 获取top-k记忆
        top_k = min(top_k, len(self.memory_items))
        if top_k > 0 and attention_weights.sum() > 0:
            top_values, top_indices = torch.topk(attention_weights, top_k)
            
            # 标记这些记忆为已访问
            for idx in top_indices:
                self.memory_items[idx].access()
            
            retrieved_items = [self.memory_items[i].to_dict() for i in top_indices]
        else:
            retrieved_items = []
            top_indices = []
        
        # 计算聚合记忆状态
        memory_state = self._compute_memory_state()
        
        return {
            'memory_items': retrieved_items,
            'all_memory_items': [item.to_dict() for item in self.memory_items],
            'memory_state': memory_state,
            'attention_weights': attention_weights,
            'item_count': len(self.memory_items),
            'retrieved_indices': top_indices.tolist() if len(top_indices) > 0 else []
        }
    
    def _write_operation(
        self,
        input_embeds: torch.Tensor,
        max_items: int = 10,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """写入操作：添加新的记忆项"""
        # 处理输入
        batch_size = input_embeds.shape[0] if len(input_embeds.shape) > 2 else 1
        if len(input_embeds.shape) == 3:
            # (B, L, D) -> 平均池化到 (B, D)
            embeds = input_embeds.mean(dim=1)
        elif len(input_embeds.shape) == 2:
            embeds = input_embeds
        else:
            embeds = input_embeds.unsqueeze(0)
        
        # 检查记忆容量
        if len(self.memory_items) >= max_items:
            # 需要替换：移除激活最低的项
            self._prune_memory(len(self.memory_items) - max_items + batch_size)
        
        # 创建新记忆项
        new_items = []
        for i in range(batch_size):
            embed = embeds[i].detach().cpu()
            
            # 检查相似性（避免重复）
            is_duplicate, similar_idx = self._check_duplicate(embed)
            
            if not is_duplicate:
                new_item = WorkingMemoryItem(
                    content=f"memory_item_{len(self.memory_items) + i}",
                    embedding=embed,
                    item_type='working',
                    importance=0.7,  # 默认重要性
                    timestamp=time.time()
                )
                self.memory_items.append(new_item)
                new_items.append(new_item.to_dict())
            else:
                # 刷新现有记忆
                self.memory_items[similar_idx].access()
                self.memory_items[similar_idx].activation = min(
                    1.0, self.memory_items[similar_idx].activation + 0.2
                )
        
        # 计算聚合记忆状态
        memory_state = self._compute_memory_state()
        attention_weights = self._compute_attention(embeds)
        
        return {
            'memory_items': new_items,
            'all_memory_items': [item.to_dict() for item in self.memory_items],
            'memory_state': memory_state,
            'attention_weights': attention_weights,
            'item_count': len(self.memory_items),
            'added_count': len(new_items)
        }
    
    def _update_operation(
        self,
        input_embeds: torch.Tensor,
        max_items: int = 10,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """更新操作：更新现有记忆项"""
        if not self.memory_items:
            return self._empty_result()
        
        # 计算与输入的相似性
        if len(input_embeds.shape) == 3:
            embeds = input_embeds.mean(dim=1)
        else:
            embeds = input_embeds
        
        embeds_norm = F.normalize(embeds, p=2, dim=-1)
        memory_embeds = torch.stack([item.embedding for item in self.memory_items]).to(self.config.device)
        memory_embeds_norm = F.normalize(memory_embeds, p=2, dim=-1)
        
        similarity = torch.matmul(embeds_norm, memory_embeds_norm.transpose(0, 1))
        most_similar_idx = similarity.argmax(dim=-1).item()
        
        # 更新最相似的记忆项
        if 0 <= most_similar_idx < len(self.memory_items):
            self.memory_items[most_similar_idx].embedding = embeds[0].detach().cpu()
            self.memory_items[most_similar_idx].access()
            self.memory_items[most_similar_idx].activation = 1.0
            updated = True
        else:
            updated = False
        
        memory_state = self._compute_memory_state()
        attention_weights = self._compute_attention(embeds)
        
        return {
            'memory_items': [self.memory_items[most_similar_idx].to_dict()] if updated else [],
            'all_memory_items': [item.to_dict() for item in self.memory_items],
            'memory_state': memory_state,
            'attention_weights': attention_weights,
            'item_count': len(self.memory_items),
            'updated_idx': most_similar_idx if updated else -1
        }
    
    def _delete_operation(
        self,
        input_embeds: Optional[torch.Tensor] = None,
        max_items: int = 10,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """删除操作：删除指定或激活最低的记忆"""
        if not self.memory_items:
            return self._empty_result()
        
        deleted_items = []
        
        if input_embeds is not None:
            # 删除与输入相似的记忆
            if len(input_embeds.shape) == 3:
                embeds = input_embeds.mean(dim=1)
            else:
                embeds = input_embeds
            
            embeds_norm = F.normalize(embeds, p=2, dim=-1)
            memory_embeds = torch.stack([item.embedding for item in self.memory_items]).to(self.config.device)
            memory_embeds_norm = F.normalize(memory_embeds, p=2, dim=-1)
            
            similarity = torch.matmul(embeds_norm, memory_embeds_norm.transpose(0, 1))
            delete_idx = similarity.argmax(dim=-1).item()
            
            if 0 <= delete_idx < len(self.memory_items):
                deleted = self.memory_items.pop(delete_idx)
                deleted_items.append(deleted.to_dict())
        else:
            # 删除激活最低的记忆
            delete_idx = min(range(len(self.memory_items)), key=lambda i: self.memory_items[i].activation)
            deleted = self.memory_items.pop(delete_idx)
            deleted_items.append(deleted.to_dict())
        
        memory_state = self._compute_memory_state()
        attention_weights = torch.zeros(len(self.memory_items), device=self.config.device)
        
        return {
            'memory_items': deleted_items,
            'all_memory_items': [item.to_dict() for item in self.memory_items],
            'memory_state': memory_state,
            'attention_weights': attention_weights,
            'item_count': len(self.memory_items),
            'deleted_count': len(deleted_items)
        }
    
    def _clear_operation(
        self,
        input_embeds: Optional[torch.Tensor] = None,
        max_items: int = 10,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """清空操作：清空所有记忆"""
        cleared_count = len(self.memory_items)
        self.memory_items = []
        
        return {
            'memory_items': [],
            'all_memory_items': [],
            'memory_state': torch.zeros(self.config.hidden_dim, device=self.config.device),
            'attention_weights': torch.tensor([], device=self.config.device),
            'item_count': 0,
            'cleared_count': cleared_count
        }
    
    def _refresh_operation(
        self,
        input_embeds: Optional[torch.Tensor] = None,
        max_items: int = 10,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """刷新操作：增加所有记忆的激活程度"""
        for item in self.memory_items:
            item.activation = min(1.0, item.activation + 0.3)
            item.access()
        
        memory_state = self._compute_memory_state()
        attention_weights = torch.tensor(
            [item.activation for item in self.memory_items],
            device=self.config.device
        )
        
        return {
            'memory_items': [item.to_dict() for item in self.memory_items],
            'all_memory_items': [item.to_dict() for item in self.memory_items],
            'memory_state': memory_state,
            'attention_weights': attention_weights,
            'item_count': len(self.memory_items),
            'refreshed': True
        }
    
    def _compute_attention(self, query_embeds: torch.Tensor) -> torch.Tensor:
        """计算查询与记忆项之间的注意力权重"""
        if not self.memory_items:
            return torch.tensor([], device=self.config.device)
        
        # 准备查询
        if len(query_embeds.shape) == 3:
            query = query_embeds.mean(dim=1)  # (B, D)
        elif len(query_embeds.shape) == 1:
            query = query_embeds.unsqueeze(0)  # (1, D)
        else:
            query = query_embeds
        
        batch_size = query.shape[0]
        memory_size = len(self.memory_items)
        
        # 获取记忆嵌入 (M, D)
        memory_embeds = torch.stack([item.embedding for item in self.memory_items]).to(self.config.device)
        
        # 扩展查询以匹配记忆数量 (B, M, D)
        query_expanded = query.unsqueeze(1).expand(-1, memory_size, -1)
        memory_expanded = memory_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 计算注意力分数
        combined = torch.cat([query_expanded, memory_expanded], dim=-1)
        attention_scores = self.attention_controller(combined).squeeze(-1)  # (B, M)
        
        # 应用softmax
        attention_weights = F.softmax(
            attention_scores / self.config.attention_temperature,
            dim=-1
        )
        
        # 加权激活程度
        activations = torch.tensor(
            [item.activation for item in self.memory_items],
            device=self.config.device
        )
        importance = torch.tensor(
            [item.importance for item in self.memory_items],
            device=self.config.device
        )
        attention_weights = attention_weights * activations.unsqueeze(0) * importance.unsqueeze(0)
        
        # 归一化
        if attention_weights.sum() > 0:
            attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        return attention_weights.squeeze(0) if batch_size == 1 else attention_weights
    
    def _compute_memory_state(self) -> torch.Tensor:
        """计算聚合的记忆状态张量"""
        if not self.memory_items:
            return torch.zeros(self.config.hidden_dim, device=self.config.device)
        
        # 加权平均：激活程度 * 重要性
        weights = torch.tensor(
            [item.activation * item.importance for item in self.memory_items],
            device=self.config.device
        )
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        
        memory_embeds = torch.stack([item.embedding for item in self.memory_items]).to(self.config.device)
        weighted_state = (memory_embeds * weights.unsqueeze(1)).sum(dim=0)
        
        # 应用投影
        memory_state = self.state_projection(weighted_state.unsqueeze(0)).squeeze(0)
        
        return memory_state
    
    def _apply_decay(self) -> None:
        """应用记忆衰减"""
        for item in self.memory_items:
            item.decay(self.config.memory_decay_rate)
        
        # 移除激活程度过低的记忆
        self.memory_items = [
            item for item in self.memory_items
            if item.activation > 0.01
        ]
    
    def _prune_memory(self, num_to_remove: int) -> None:
        """修剪记忆，移除激活最低的项"""
        if num_to_remove <= 0 or not self.memory_items:
            return
        
        # 按激活程度排序，移除最低的
        self.memory_items.sort(key=lambda x: x.activation * x.importance)
        self.memory_items = self.memory_items[num_to_remove:]
    
    def _check_duplicate(self, embed: torch.Tensor, threshold: float = 0.9) -> Tuple[bool, int]:
        """检查是否有重复的记忆项"""
        if not self.memory_items:
            return False, -1
        
        embed_norm = F.normalize(embed.unsqueeze(0).to(self.config.device), p=2, dim=-1)
        memory_embeds = torch.stack([item.embedding for item in self.memory_items]).to(self.config.device)
        memory_embeds_norm = F.normalize(memory_embeds, p=2, dim=-1)
        
        similarity = torch.matmul(embed_norm, memory_embeds_norm.transpose(0, 1))
        max_sim = similarity.max().item()
        max_idx = similarity.argmax().item()
        
        return max_sim > threshold, max_idx
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'memory_items': [],
            'all_memory_items': [],
            'memory_state': torch.zeros(self.config.hidden_dim, device=self.config.device),
            'attention_weights': torch.tensor([], device=self.config.device),
            'item_count': 0
        }
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取记忆项"""
        for item in self.memory_items:
            if item.id == memory_id:
                return item.to_dict()
        return None
    
    def update_memory_importance(self, memory_id: str, new_importance: float) -> bool:
        """更新记忆项的重要性"""
        for item in self.memory_items:
            if item.id == memory_id:
                item.importance = max(0.0, min(1.0, new_importance))
                return True
        return False
