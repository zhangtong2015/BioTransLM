# 模块G：系统1 - 直觉联想 (System1Intuition)
# 负责人：AI 2
# 输入：query_embeds: (B, D), context_embeds: Optional[(B, L, D)], top_k: int
# 输出：{'retrieved_results': List[str], 'similarities': List[float], 'confidence': float, 'indices': List[int]}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class System1Config(BaseConfig):
    """系统1 - 直觉联想配置"""
    hidden_dim: int = 768
    index_dim: int = 768
    max_index_size: int = 100000
    similarity_threshold: float = 0.7
    default_top_k: int = 5
    use_gpu_index: bool = True
    index_type: str = 'Flat'  # 'Flat', 'IVF', 'HNSW'


class System1Intuition(BaseModule):
    """
    系统1 - 直觉联想模块
    
    实现快速向量检索、近似最近邻搜索、置信度评估。
    使用faiss进行快速检索，预建立常见模式索引，响应时间要求 < 100ms。
    """
    
    def __init__(
        self, 
        config: Optional[System1Config] = None,
        module_name: str = "system1_intuition"
    ):
        # 先初始化faiss导入标记
        self.faiss = None
        
        config = config or System1Config()
        super().__init__(config=config, module_name=module_name)
    
    def _import_faiss(self) -> None:
        """导入faiss库（延迟导入）"""
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            print("警告: faiss未安装，将使用torch实现（速度较慢）")
            self.faiss = None
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 导入faiss
        self._import_faiss()
        
        # FAISS索引（延迟初始化）
        self.index = None
        self.index_to_data = {}  # 索引映射到实际数据
        self.data_embeddings = []  # 存储嵌入向量
        self.data_contents = []    # 存储内容
        
        # 查询投影层
        self.query_projection = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim)
        )
        
        # 置信度评估网络
        self.confidence_net = nn.Sequential(
            nn.Linear(3, 64),  # 输入: max_sim, avg_sim, std_sim
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 初始化空索引
        self._init_index()
    
    def _init_index(self) -> None:
        """初始化FAISS索引"""
        if self.faiss is not None:
            if self.config.index_type == 'Flat':
                self.index = self.faiss.IndexFlatIP(self.config.index_dim)
            elif self.config.index_type == 'IVF':
                nlist = 100  # 聚类中心数量
                quantizer = self.faiss.IndexFlatIP(self.config.index_dim)
                self.index = self.faiss.IndexIVFFlat(
                    quantizer, self.config.index_dim, nlist, self.faiss.METRIC_INNER_PRODUCT
                )
            elif self.config.index_type == 'HNSW':
                self.index = self.faiss.IndexHNSWFlat(self.config.index_dim, 32)
            else:
                self.index = self.faiss.IndexFlatIP(self.config.index_dim)
            
            # GPU加速
            if self.config.use_gpu_index and self.faiss.get_num_gpus() > 0:
                self.index = self.faiss.index_cpu_to_gpu(
                    self.faiss.StandardGpuResources(), 0, self.index
                )
        else:
            # 使用torch作为备用
            self.index = []
    
    def forward(
        self,
        query_embeds: torch.Tensor,
        context_embeds: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播 - 快速直觉联想
        
        Args:
            query_embeds: 查询嵌入 (batch_size, hidden_dim)
            context_embeds: 上下文嵌入 (batch_size, seq_len, hidden_dim)
            top_k: 返回结果数量
            return_embeddings: 是否返回嵌入向量
            
        Returns:
            Dict包含：
                retrieved_results: 检索到的结果内容列表
                similarities: 相似度分数列表
                confidence: 检索置信度
                indices: 检索到的索引
                retrieved_embeds: 检索到的嵌入向量（可选）
        """
        top_k = top_k or self.config.default_top_k
        
        # 确保输入在正确的设备上
        query_embeds = query_embeds.to(self._device)
        if context_embeds is not None:
            context_embeds = context_embeds.to(self._device)
        
        # 1. 查询投影增强
        query = self.query_projection(query_embeds)
        
        # 2. 如果有上下文，与上下文融合
        if context_embeds is not None:
            # 简单的注意力加权融合
            context_mean = context_embeds.mean(dim=1)
            query = query + context_mean * 0.3  # 加权融合
        
        # 3. 归一化以进行内积相似度计算
        query_norm = F.normalize(query, p=2, dim=-1)
        
        # 4. 执行检索
        similarities, indices = self._search(query_norm, top_k)
        
        # 5. 获取检索结果
        retrieved_results = []
        retrieved_embeds = []
        
        for idx_list in indices:
            batch_results = []
            batch_embeds = []
            for idx in idx_list:
                if idx < len(self.data_contents):
                    batch_results.append(self.data_contents[idx])
                    if return_embeddings:
                        batch_embeds.append(self.data_embeddings[idx])
                else:
                    batch_results.append("")
                    if return_embeddings:
                        batch_embeds.append(torch.zeros(self.config.hidden_dim))
            retrieved_results.append(batch_results)
            if return_embeddings:
                retrieved_embeds.append(torch.stack(batch_embeds))
        
        # 6. 计算置信度
        confidence = self._compute_confidence(similarities)
        
        result = {
            'retrieved_results': retrieved_results,
            'similarities': similarities.tolist(),
            'confidence': confidence,
            'indices': indices.tolist()
        }
        
        if return_embeddings and retrieved_embeds:
            result['retrieved_embeds'] = torch.stack(retrieved_embeds)
        
        return result
    
    def _search(
        self,
        query_norm: torch.Tensor,
        top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行相似性搜索"""
        batch_size = query_norm.shape[0]
        
        if self.faiss is not None and self.index is not None and len(self.data_embeddings) > 0:
            # 使用FAISS快速检索
            query_np = query_norm.detach().cpu().numpy().astype('float32')
            
            if not self.index.is_trained and hasattr(self.index, 'train'):
                # 对于需要训练的索引类型
                if len(self.data_embeddings) >= 100:
                    data_np = np.array([x.cpu().numpy() for x in self.data_embeddings]).astype('float32')
                    self.index.train(data_np)
            
            similarities, indices = self.index.search(query_np, top_k)
            similarities = torch.tensor(similarities, device=query_norm.device)
            indices = torch.tensor(indices, device=query_norm.device)
            
        else:
            # 使用torch实现（备用方法）
            similarities = torch.zeros(batch_size, top_k, device=query_norm.device)
            indices = torch.zeros(batch_size, top_k, dtype=torch.long, device=query_norm.device)
            
            if len(self.data_embeddings) > 0:
                data_tensor = torch.stack(self.data_embeddings).to(query_norm.device)
                data_norm = F.normalize(data_tensor, p=2, dim=-1)
                
                # 计算相似度矩阵
                sim_matrix = torch.matmul(query_norm, data_norm.transpose(0, 1))
                
                # 获取top-k
                similarities, indices = torch.topk(sim_matrix, min(top_k, sim_matrix.shape[1]), dim=-1)
        
        return similarities, indices
    
    def _compute_confidence(self, similarities: torch.Tensor) -> List[float]:
        """计算检索置信度"""
        batch_size = similarities.shape[0]
        confidences = []
        
        for i in range(batch_size):
            sims = similarities[i]
            
            # 计算统计特征
            max_sim = sims.max().item()
            avg_sim = sims.mean().item()
            std_sim = sims.std().item() if len(sims) > 1 else 0.0
            
            # 简单规则置信度计算
            feature_tensor = torch.tensor([[max_sim, avg_sim, std_sim]], device=self._device)
            confidence = self.confidence_net(feature_tensor).item()
            
            # 应用阈值
            if max_sim < self.config.similarity_threshold:
                confidence *= 0.5
            
            confidences.append(float(confidence))
        
        return confidences
    
    def add_to_index(
        self,
        embeddings: torch.Tensor,
        contents: List[str],
        ids: Optional[List[int]] = None
    ) -> None:
        """添加向量到索引"""
        batch_size = embeddings.shape[0]
        
        # 归一化
        embeds_norm = F.normalize(embeddings, p=2, dim=-1)
        
        for i in range(batch_size):
            embed = embeds_norm[i].cpu()
            content = contents[i]
            
            self.data_embeddings.append(embed)
            self.data_contents.append(content)
            
            if ids is not None and i < len(ids):
                idx = ids[i]
            else:
                idx = len(self.data_embeddings) - 1
            
            self.index_to_data[idx] = {
                'embedding': embed,
                'content': content
            }
        
        # 更新FAISS索引
        if self.faiss is not None and self.index is not None:
            data_np = np.array([x.numpy() for x in self.data_embeddings]).astype('float32')
            self.index.reset()
            self.index.add(data_np)
    
    def clear_index(self) -> None:
        """清空索引"""
        self.data_embeddings = []
        self.data_contents = []
        self.index_to_data = {}
        
        if self.faiss is not None and self.index is not None:
            self.index.reset()
    
    def get_index_size(self) -> int:
        """获取索引大小"""
        return len(self.data_embeddings)
