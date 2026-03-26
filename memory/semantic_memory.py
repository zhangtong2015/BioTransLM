# 模块L：语义记忆 (SemanticMemory)
# 负责人：AI 2
# 输入：query: str, query_embeds: Optional[torch.Tensor], top_k: int
# 输出：{'concepts': List[Dict], 'semantic_relations': List[Dict], 'similarities': List[float]}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import uuid
import numpy as np

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class SemanticMemoryConfig(BaseConfig):
    """语义记忆配置"""
    hidden_dim: int = 768
    max_concepts: int = 100000
    similarity_threshold: float = 0.7
    retrieval_top_k: int = 5
    enable_spreading_activation: bool = True
    max_spreading_depth: int = 2


class Concept:
    """概念节点"""
    
    def __init__(
        self,
        name: str,
        embedding: torch.Tensor,
        concept_type: str = 'entity',
        definition: str = '',
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.embedding = embedding
        self.concept_type = concept_type  # entity, action, attribute, relation, etc.
        self.definition = definition
        self.attributes = attributes or {}
        self.activation = 0.0  # 用于扩散激活
        self.access_count = 0
    
    def activate(self, level: float = 1.0) -> None:
        """激活概念"""
        self.activation = min(1.0, self.activation + level)
        self.access_count += 1
    
    def deactivate(self, decay_rate: float = 0.1) -> None:
        """衰减激活"""
        self.activation *= (1.0 - decay_rate)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'concept_type': self.concept_type,
            'definition': self.definition,
            'attributes': self.attributes,
            'activation': self.activation,
            'access_count': self.access_count
        }


class SemanticRelation:
    """语义关系"""
    
    RELATION_TYPES = {
        'is_a': '是一种',
        'part_of': '是...的一部分',
        'has_property': '具有属性',
        'causes': '导致',
        'related_to': '相关于',
        'synonym': '同义词',
        'antonym': '反义词',
        'instance_of': '是...的实例',
        'subclass_of': '是...的子类',
        'has_part': '包含部分'
    }
    
    def __init__(
        self,
        subject_id: str,
        object_id: str,
        relation_type: str,
        weight: float = 1.0,
        evidence: str = ''
    ):
        self.id = str(uuid.uuid4())
        self.subject_id = subject_id
        self.object_id = object_id
        self.relation_type = relation_type
        self.weight = weight
        self.evidence = evidence
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'subject_id': self.subject_id,
            'object_id': self.object_id,
            'relation_type': self.relation_type,
            'relation_name': self.RELATION_TYPES.get(self.relation_type, self.relation_type),
            'weight': self.weight,
            'evidence': self.evidence
        }


class SemanticMemory(BaseModule):
    """
    语义记忆模块
    
    实现概念层次网络存储、类别继承推理、关系路径查询。
    模拟人类语义记忆，存储关于世界的结构化知识和概念关系。
    """
    
    def __init__(
        self, 
        config: Optional[SemanticMemoryConfig] = None,
        module_name: str = "semantic_memory"
    ):
        config = config or SemanticMemoryConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 概念存储: concept_id -> Concept
        self.concepts: Dict[str, Concept] = {}
        
        # 名称索引: name -> concept_id
        self.name_index: Dict[str, str] = {}
        
        # 类型索引: concept_type -> List[concept_id]
        self.type_index: Dict[str, List[str]] = {}
        
        # 语义关系存储: relation_id -> SemanticRelation
        self.relations: Dict[str, SemanticRelation] = {}
        
        # 邻接表: subject_id -> List[(object_id, relation_id, relation_type, weight)]
        self.adjacency: Dict[str, List[Tuple[str, str, str, float]]] = {}
        
        # 反向邻接表: object_id -> List[(subject_id, relation_id, relation_type, weight)]
        self.reverse_adjacency: Dict[str, List[Tuple[str, str, str, float]]] = {}
        
        # 相似度匹配网络
        self.similarity_net = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 概念融合网络
        self.concept_fusion = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim)
        )
        
        # 初始化基础概念知识
        self._initialize_basic_concepts()
    
    def _initialize_basic_concepts(self) -> None:
        """初始化基础概念和关系"""
        # 创建一些基础概念
        basic_concepts = [
            ("生物", "entity", "有生命的物体，包括动物、植物等"),
            ("动物", "entity", "生物的一个大类，包括人类、鸟类、鱼类等"),
            ("人类", "entity", "智人，具有高级认知能力的灵长类动物"),
            ("鸟", "entity", "有羽毛、产卵的脊椎动物"),
            ("鱼", "entity", "生活在水中的脊椎动物"),
            ("植物", "entity", "能够进行光合作用的生物"),
            ("食物", "entity", "能够提供营养的物质"),
            ("水", "entity", "生命必需的液体"),
            ("大", "attribute", "尺寸、数量或程度超过平均水平"),
            ("小", "attribute", "尺寸、数量或程度低于平均水平"),
            ("快", "attribute", "速度高的"),
            ("慢", "attribute", "速度低的"),
            ("吃", "action", "摄入食物"),
            ("飞", "action", "在空中移动"),
            ("游", "action", "在水中移动"),
            ("生长", "action", "体积、重量或程度增加"),
            ("会飞", "attribute", "具有飞行能力"),
            ("会游", "attribute", "具有游泳能力"),
            ("有羽毛", "attribute", "身体覆盖羽毛")
        ]
        
        # 创建dummy嵌入（实际使用时会被真实嵌入替换）
        dummy_embed = torch.randn(self.config.hidden_dim)
        
        for name, ctype, definition in basic_concepts:
            concept = Concept(
                name=name,
                embedding=dummy_embed.clone(),
                concept_type=ctype,
                definition=definition
            )
            self._add_concept_to_indices(concept)
        
        # 添加基础关系
        basic_relations = [
            ("人类", "动物", "is_a"),
            ("鸟", "动物", "is_a"),
            ("鱼", "动物", "is_a"),
            ("动物", "生物", "is_a"),
            ("鸟", "会飞", "has_property"),
            ("鱼", "会游", "has_property"),
            ("动物", "吃", "related_to"),
            ("鸟", "有羽毛", "has_property")
        ]
        
        for subj_name, obj_name, rel_type in basic_relations:
            if subj_name in self.name_index and obj_name in self.name_index:
                self._add_relation(
                    self.name_index[subj_name],
                    self.name_index[obj_name],
                    rel_type
                )
    
    def _add_concept_to_indices(self, concept: Concept) -> None:
        """将概念添加到各种索引"""
        self.concepts[concept.id] = concept
        self.name_index[concept.name] = concept.id
        
        if concept.concept_type not in self.type_index:
            self.type_index[concept.concept_type] = []
        self.type_index[concept.concept_type].append(concept.id)
        
        if concept.id not in self.adjacency:
            self.adjacency[concept.id] = []
        if concept.id not in self.reverse_adjacency:
            self.reverse_adjacency[concept.id] = []
    
    def _add_relation(
        self,
        subject_id: str,
        object_id: str,
        relation_type: str,
        weight: float = 1.0,
        evidence: str = ''
    ) -> Optional[str]:
        """添加语义关系"""
        if subject_id not in self.concepts or object_id not in self.concepts:
            return None
        
        # 检查是否已存在相同关系
        for obj_id, _, rel_type, _ in self.adjacency.get(subject_id, []):
            if obj_id == object_id and rel_type == relation_type:
                return None
        
        relation = SemanticRelation(subject_id, object_id, relation_type, weight, evidence)
        self.relations[relation.id] = relation
        
        # 更新邻接表
        if subject_id not in self.adjacency:
            self.adjacency[subject_id] = []
        self.adjacency[subject_id].append(
            (object_id, relation.id, relation_type, weight)
        )
        
        if object_id not in self.reverse_adjacency:
            self.reverse_adjacency[object_id] = []
        self.reverse_adjacency[object_id].append(
            (subject_id, relation.id, relation_type, weight)
        )
        
        return relation.id
    
    def forward(
        self,
        query: str = '',
        query_embeds: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        operation: str = 'retrieve',
        concept_info: Optional[Dict[str, Any]] = None,
        relation_info: Optional[Dict[str, Any]] = None,
        include_relations: bool = True,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        前向传播 - 语义记忆操作
        
        Args:
            query: 查询字符串
            query_embeds: 查询嵌入向量
            top_k: 返回结果数量
            operation: 操作类型 ['retrieve', 'retrieve_by_type', 'get_relations', 'add_concept', 'add_relation', 'spreading_activation']
            concept_info: 添加概念时的信息
            relation_info: 添加关系时的信息
            include_relations: 是否包含关系信息
            threshold: 相似度阈值
            
        Returns:
            Dict包含：
                concepts: 概念列表
                semantic_relations: 语义关系列表
                similarities: 相似度分数列表
                concept_count: 概念总数
                relation_count: 关系总数
        """
        top_k = top_k or self.config.retrieval_top_k
        threshold = threshold or self.config.similarity_threshold
        
        if operation == 'add_concept' and concept_info:
            return self._add_concept(concept_info)
        elif operation == 'add_relation' and relation_info:
            return self._add_relation_forward(relation_info)
        elif operation == 'retrieve_by_type':
            return self._retrieve_by_type(query, top_k)
        elif operation == 'get_relations':
            return self._get_concept_relations(query, top_k)
        elif operation == 'spreading_activation':
            return self._spreading_activation(query, top_k)
        else:  # retrieve
            return self._retrieve_concepts(query, query_embeds, top_k, threshold, include_relations)
    
    def _add_concept(self, concept_info: Dict[str, Any]) -> Dict[str, Any]:
        """添加新概念"""
        name = concept_info.get('name', '')
        if not name:
            return {'error': '概念名称不能为空'}
        
        # 检查是否已存在
        if name in self.name_index:
            return {
                'message': '概念已存在',
                'concept_id': self.name_index[name],
                'concept_count': len(self.concepts)
            }
        
        # 检查容量
        if len(self.concepts) >= self.config.max_concepts:
            # 移除访问最少的概念
            self._prune_concepts(len(self.concepts) - self.config.max_concepts + 1)
        
        embedding = concept_info.get('embedding', torch.randn(self.config.hidden_dim))
        
        concept = Concept(
            name=name,
            embedding=embedding,
            concept_type=concept_info.get('concept_type', 'entity'),
            definition=concept_info.get('definition', ''),
            attributes=concept_info.get('attributes', {})
        )
        
        self._add_concept_to_indices(concept)
        
        return {
            'concepts': [concept.to_dict()],
            'semantic_relations': [],
            'similarities': [1.0],
            'concept_count': len(self.concepts),
            'relation_count': len(self.relations),
            'added': True
        }
    
    def _add_relation_forward(self, relation_info: Dict[str, Any]) -> Dict[str, Any]:
        """添加关系（forward接口）"""
        subject = relation_info.get('subject', '')
        obj = relation_info.get('object', '')
        rel_type = relation_info.get('relation_type', 'related_to')
        
        if not subject or not obj:
            return {'error': '主体和客体不能为空'}
        
        # 查找概念ID
        subj_id = self.name_index.get(subject)
        obj_id = self.name_index.get(obj)
        
        if not subj_id or not obj_id:
            return {'error': '主体或客体概念不存在'}
        
        rel_id = self._add_relation(
            subj_id, obj_id, rel_type,
            weight=relation_info.get('weight', 1.0),
            evidence=relation_info.get('evidence', '')
        )
        
        if rel_id:
            return {
                'concepts': [],
                'semantic_relations': [self.relations[rel_id].to_dict()],
                'similarities': [],
                'concept_count': len(self.concepts),
                'relation_count': len(self.relations),
                'added': True
            }
        else:
            return {'message': '关系已存在或添加失败'}
    
    def _retrieve_concepts(
        self,
        query: str = '',
        query_embeds: Optional[torch.Tensor] = None,
        top_k: int = 5,
        threshold: float = 0.7,
        include_relations: bool = True
    ) -> Dict[str, Any]:
        """检索概念"""
        if not self.concepts:
            return self._empty_result()
        
        # 1. 基于名称的精确匹配和部分匹配
        name_matches = []
        if query:
            query_lower = query.lower()
            for name, cid in self.name_index.items():
                if query_lower in name.lower():
                    score = 1.0 if name.lower() == query_lower else 0.8
                    name_matches.append((cid, score))
        
        # 2. 基于嵌入的相似度匹配
        embed_matches = []
        if query_embeds is not None:
            if len(query_embeds.shape) == 3:
                query_vec = query_embeds.mean(dim=1)
            elif len(query_embeds.shape) == 1:
                query_vec = query_embeds.unsqueeze(0)
            else:
                query_vec = query_embeds
            
            # 获取所有概念的嵌入
            concept_embeds = torch.stack([
                c.embedding for c in self.concepts.values()
            ]).to(self.config.device)
            
            concept_ids = list(self.concepts.keys())
            
            # 计算相似度
            query_norm = F.normalize(query_vec, p=2, dim=-1)
            concept_norm = F.normalize(concept_embeds, p=2, dim=-1)
            
            similarities = torch.matmul(query_norm, concept_norm.transpose(0, 1))
            
            # 获取top-k
            top_scores, top_indices = torch.topk(similarities, min(top_k, len(concept_ids)), dim=-1)
            
            for i, idx in enumerate(top_indices[0]):
                cid = concept_ids[idx.item()]
                embed_matches.append((cid, float(top_scores[0][i])))
        
        # 3. 融合两种匹配结果
        all_matches = {}
        for cid, score in name_matches:
            all_matches[cid] = score
        for cid, score in embed_matches:
            if cid in all_matches:
                all_matches[cid] = max(all_matches[cid], score)
            else:
                all_matches[cid] = score
        
        # 4. 排序和过滤
        sorted_matches = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)
        sorted_matches = [(cid, score) for cid, score in sorted_matches if score >= threshold][:top_k]
        
        # 5. 收集结果
        concepts = []
        scores = []
        relations = []
        
        for cid, score in sorted_matches:
            concept = self.concepts[cid]
            concept.activate()
            concepts.append(concept.to_dict())
            scores.append(score)
            
            if include_relations:
                # 获取该概念的关系
                concept_relations = self._get_relations_for_concept(cid)
                relations.extend(concept_relations)
        
        return {
            'concepts': concepts,
            'semantic_relations': relations,
            'similarities': scores,
            'concept_count': len(self.concepts),
            'relation_count': len(self.relations)
        }
    
    def _retrieve_by_type(self, concept_type: str, top_k: int) -> Dict[str, Any]:
        """按类型检索概念"""
        if concept_type not in self.type_index:
            return self._empty_result()
        
        concept_ids = self.type_index[concept_type][:top_k]
        concepts = [self.concepts[cid].to_dict() for cid in concept_ids]
        
        return {
            'concepts': concepts,
            'semantic_relations': [],
            'similarities': [1.0] * len(concepts),
            'concept_count': len(self.concepts),
            'relation_count': len(self.relations)
        }
    
    def _get_concept_relations(self, concept_name: str, top_k: int) -> Dict[str, Any]:
        """获取概念的关系"""
        if concept_name not in self.name_index:
            return self._empty_result()
        
        cid = self.name_index[concept_name]
        relations = self._get_relations_for_concept(cid)[:top_k]
        
        return {
            'concepts': [self.concepts[cid].to_dict()],
            'semantic_relations': relations,
            'similarities': [1.0],
            'concept_count': len(self.concepts),
            'relation_count': len(self.relations)
        }
    
    def _get_relations_for_concept(self, concept_id: str) -> List[Dict[str, Any]]:
        """获取概念的所有关系"""
        relations = []
        
        # 出边关系
        for obj_id, rel_id, rel_type, weight in self.adjacency.get(concept_id, []):
            rel_dict = self.relations[rel_id].to_dict()
            rel_dict['subject_name'] = self.concepts[concept_id].name
            rel_dict['object_name'] = self.concepts.get(obj_id, {}).name if obj_id in self.concepts else '未知'
            relations.append(rel_dict)
        
        # 入边关系
        for subj_id, rel_id, rel_type, weight in self.reverse_adjacency.get(concept_id, []):
            rel_dict = self.relations[rel_id].to_dict()
            rel_dict['subject_name'] = self.concepts.get(subj_id, {}).name if subj_id in self.concepts else '未知'
            rel_dict['object_name'] = self.concepts[concept_id].name
            relations.append(rel_dict)
        
        return relations
    
    def _spreading_activation(
        self,
        start_concept: str,
        top_k: int
    ) -> Dict[str, Any]:
        """扩散激活"""
        if not self.config.enable_spreading_activation:
            return self._empty_result()
        
        if start_concept not in self.name_index:
            return self._empty_result()
        
        start_id = self.name_index[start_concept]
        
        # 重置所有激活
        for concept in self.concepts.values():
            concept.activation = 0.0
        
        # BFS扩散激活
        activation_queue = [(start_id, 1.0, 0)]  # (concept_id, activation, depth)
        activated = set()
        
        while activation_queue:
            cid, act_level, depth = activation_queue.pop(0)
            
            if cid in activated or depth >= self.config.max_spreading_depth:
                continue
            
            activated.add(cid)
            self.concepts[cid].activate(act_level)
            
            # 扩散到相邻节点
            decay_factor = 0.5
            for neighbor_id, _, _, weight in self.adjacency.get(cid, []):
                new_act_level = act_level * decay_factor * weight
                if new_act_level > 0.1:  # 最小激活阈值
                    activation_queue.append((neighbor_id, new_act_level, depth + 1))
        
        # 收集激活的概念
        activated_concepts = [
            (cid, self.concepts[cid].activation)
            for cid in activated
            if cid != start_id
        ]
        
        # 按激活程度排序
        activated_concepts.sort(key=lambda x: x[1], reverse=True)
        activated_concepts = activated_concepts[:top_k]
        
        # 准备结果
        concepts = []
        scores = []
        
        for cid, act_level in activated_concepts:
            concepts.append(self.concepts[cid].to_dict())
            scores.append(act_level)
        
        # 始终包含起始概念
        concepts.insert(0, self.concepts[start_id].to_dict())
        scores.insert(0, 1.0)
        
        return {
            'concepts': concepts,
            'semantic_relations': self._get_relations_for_concept(start_id),
            'similarities': scores,
            'concept_count': len(self.concepts),
            'relation_count': len(self.relations),
            'activated_count': len(activated)
        }
    
    def _prune_concepts(self, num_to_remove: int) -> None:
        """修剪概念，移除访问最少的"""
        if num_to_remove <= 0:
            return
        
        # 按访问次数排序
        sorted_concepts = sorted(
            self.concepts.items(),
            key=lambda x: x[1].access_count
        )
        
        for i in range(min(num_to_remove, len(sorted_concepts))):
            cid, _ = sorted_concepts[i]
            
            # 移除概念
            concept = self.concepts.pop(cid)
            
            # 从索引移除
            del self.name_index[concept.name]
            self.type_index[concept.concept_type].remove(cid)
            if not self.type_index[concept.concept_type]:
                del self.type_index[concept.concept_type]
            
            # 移除相关关系
            relations_to_remove = []
            for rel_id, rel in self.relations.items():
                if rel.subject_id == cid or rel.object_id == cid:
                    relations_to_remove.append(rel_id)
            
            for rel_id in relations_to_remove:
                del self.relations[rel_id]
            
            # 清理邻接表
            if cid in self.adjacency:
                del self.adjacency[cid]
            if cid in self.reverse_adjacency:
                del self.reverse_adjacency[cid]
            
            # 清理其他节点对该概念的引用
            for adj_id in list(self.adjacency.keys()):
                self.adjacency[adj_id] = [
                    entry for entry in self.adjacency[adj_id]
                    if entry[0] != cid
                ]
            for adj_id in list(self.reverse_adjacency.keys()):
                self.reverse_adjacency[adj_id] = [
                    entry for entry in self.reverse_adjacency[adj_id]
                    if entry[0] != cid
                ]
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'concepts': [],
            'semantic_relations': [],
            'similarities': [],
            'concept_count': len(self.concepts),
            'relation_count': len(self.relations)
        }
    
    def get_concept_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取概念"""
        if name not in self.name_index:
            return None
        cid = self.name_index[name]
        return self.concepts[cid].to_dict()
    
    def get_concept_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取概念"""
        if concept_id not in self.concepts:
            return None
        return self.concepts[concept_id].to_dict()
    
    def inference_isa_hierarchy(self, concept_name: str) -> List[str]:
        """推理is_a层次结构"""
        if concept_name not in self.name_index:
            return []
        
        cid = self.name_index[concept_name]
        hierarchy = [concept_name]
        visited = set([cid])
        
        # BFS查找所有父类
        queue = [cid]
        while queue:
            current = queue.pop(0)
            
            # 查找is_a关系
            for obj_id, _, rel_type, _ in self.adjacency.get(current, []):
                if rel_type == 'is_a' and obj_id not in visited:
                    visited.add(obj_id)
                    hierarchy.append(self.concepts[obj_id].name)
                    queue.append(obj_id)
        
        return hierarchy
    
    def infer_concept_attributes(self, concept_name: str) -> Dict[str, List[str]]:
        """
        推理概念的所有属性，包括继承自父类的属性
        使用属性继承规则：如果A是一种B，B具有属性P，则A也具有属性P
        """
        if concept_name not in self.name_index:
            return {}
        
        # 获取层次结构
        hierarchy = self.inference_isa_hierarchy(concept_name)
        attributes = {
            'direct': [],
            'inherited': [],
            'all': []
        }
        
        # 收集每个概念的属性
        for i, name in enumerate(hierarchy):
            if name not in self.name_index:
                continue
            
            cid = self.name_index[name]
            
            # 查找has_property关系
            for obj_id, _, rel_type, _ in self.adjacency.get(cid, []):
                if rel_type == 'has_property' and obj_id in self.concepts:
                    prop_name = self.concepts[obj_id].name
                    if i == 0:  # 直接属性
                        attributes['direct'].append(prop_name)
                    else:  # 继承属性
                        attributes['inherited'].append(prop_name)
                    attributes['all'].append(prop_name)
        
        # 去重
        attributes['all'] = list(set(attributes['all']))
        return attributes
    
    def find_path_between_concepts(
        self,
        start_name: str,
        end_name: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """查找两个概念之间的关系路径"""
        if start_name not in self.name_index or end_name not in self.name_index:
            return []
        
        start_id = self.name_index[start_name]
        end_id = self.name_index[end_name]
        
        # BFS查找路径
        queue = [(start_id, [])]  # (current_id, path)
        visited = set()
        paths = []
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_id:
                paths.append(path)
                continue
            
            if len(path) >= max_depth or current in visited:
                continue
            
            visited.add(current)
            
            # 遍历所有出边
            for obj_id, rel_id, rel_type, weight in self.adjacency.get(current, []):
                if obj_id not in visited:
                    rel_info = {
                        'from': self.concepts[current].name,
                        'to': self.concepts[obj_id].name,
                        'relation': rel_type,
                        'relation_name': SemanticRelation.RELATION_TYPES.get(rel_type, rel_type),
                        'weight': weight
                    }
                    queue.append((obj_id, path + [rel_info]))
        
        # 按路径长度排序
        paths.sort(key=len)
        return paths
    
    def query_by_relation(
        self,
        relation_type: str,
        target_name: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """按关系类型查询"""
        results = []
        
        for rel in self.relations.values():
            if rel.relation_type == relation_type:
                if target_name and self.concepts[rel.object_id].name != target_name:
                    continue
                
                results.append({
                    'subject': self.concepts[rel.subject_id].name,
                    'object': self.concepts[rel.object_id].name,
                    'relation': rel.relation_type,
                    'relation_name': rel.RELATION_TYPES.get(rel.relation_type, rel.relation_type),
                    'weight': rel.weight
                })
        
        results.sort(key=lambda x: x['weight'], reverse=True)
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取语义记忆统计信息"""
        type_counts = {}
        for ctype, ids in self.type_index.items():
            type_counts[ctype] = len(ids)
        
        relation_type_counts = {}
        for rel in self.relations.values():
            rtype = rel.relation_type
            relation_type_counts[rtype] = relation_type_counts.get(rtype, 0) + 1
        
        return {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'concept_types': type_counts,
            'relation_types': relation_type_counts,
            'max_capacity': self.config.max_concepts,
            'usage_percent': (len(self.concepts) / self.config.max_concepts) * 100
        }
    
    def get_semantic_path(
        self,
        start_concept: str,
        end_concept: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """查找两个概念之间的语义路径"""
        if start_concept not in self.name_index or end_concept not in self.name_index:
            return []
        
        start_id = self.name_index[start_concept]
        end_id = self.name_index[end_concept]
        
        # BFS查找路径
        queue = [(start_id, [start_id], [])]  # (current_id, path, relations)
        visited = set([start_id])
        
        while queue:
            current, path, rels = queue.pop(0)
            
            if current == end_id:
                # 构造路径信息
                path_info = []
                for i in range(len(path) - 1):
                    subj_name = self.concepts[path[i]].name
                    obj_name = self.concepts[path[i + 1]].name
                    rel = rels[i] if i < len(rels) else 'unknown'
                    path_info.append({
                        'from': subj_name,
                        'to': obj_name,
                        'relation': rel
                    })
                return path_info
            
            if len(path) >= max_depth:
                continue
            
            for neighbor_id, _, rel_type, _ in self.adjacency.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id], rels + [rel_type]))
        
        return []
