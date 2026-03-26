# 模块 H：系统 2 - 符号推理 (System2Reasoning)
# 负责人：AI 3
# 输入：query: str, context: Optional[List[str]], knowledge_graph: Optional[Dict]
# 输出：{'answer': str, 'reasoning_chain': List[str], 'confidence': float, 'facts_used': List[str]}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import re

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class System2Config(BaseConfig):
    """系统 2 - 符号推理配置"""
    hidden_dim: int = 768
    max_reasoning_steps: int = 3  # 推理步骤不超过 3 步
    confidence_threshold: float = 0.6
    enable_logging: bool = True


class System2Reasoning(BaseModule):
    """
    系统 2 - 符号推理模块
    
    实现基本逻辑运算、多步推理链、结果解释生成。
    第一天先实现简单的演绎推理，知识图谱用字典近似存储。
    """
    
    def __init__(
        self, 
        config: Optional[System2Config] = None,
        module_name: str = "system2_reasoning"
    ):
        config = config or System2Config()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 知识图谱（用字典近似存储）
        # 结构：{entity: {relation: [target_entities]}}
        self.knowledge_graph = {}
        
        # 规则库
        self.rules = []
        
        # 推理状态跟踪
        self.reasoning_trace = []
        
        # 文本编码器（用于置信度评估）
        self.text_encoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # 置信度评分网络 - 输入 80 维 (64 文本特征 + 16 推理特征)
        self.confidence_scorer = nn.Sequential(
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 初始化基础规则
        self._initialize_basic_rules()
    
    def _initialize_basic_rules(self) -> None:
        """初始化基础推理规则"""
        # 规则格式：(条件模式，结论，置信度)
        self.rules = [
            # 传递性规则
            (
                lambda facts, a, b, c: (f"{a}是{b}" in facts) and (f"{b}是{c}" in facts),
                lambda a, b, c: f"{a}是{c}",
                0.9
            ),
            # 属性继承规则
            (
                lambda facts, x, y, p: (f"{x}属于{y}" in facts) and (f"{y}的属性是{p}" in facts),
                lambda x, y, p: f"{x}的属性是{p}",
                0.85
            ),
            # 因果关系规则
            (
                lambda facts, a, b: (f"{a}导致{b}" in facts) and (f"观察到{a}" in facts),
                lambda a, b: f"可能发生{b}",
                0.7
            )
        ]
        
        # 初始化一些示例知识
        self.knowledge_graph = {
            "人类": {"是": ["动物"], "属性": ["思考", "语言"]},
            "动物": {"是": ["生物"]},
            "鸟": {"是": ["动物"], "属性": ["飞", "羽毛", "产卵"]},
            "鱼": {"是": ["动物"], "属性": ["游泳", "鳃"]},
        }
    
    def forward(
        self,
        query: str,
        context: Optional[List[str]] = None,
        knowledge_graph: Optional[Dict[str, Any]] = None,
        query_embeds: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        前向传播 - 符号推理
        
        Args:
            query: 查询字符串
            context: 上下文事实列表
            knowledge_graph: 外部知识图谱（可选，覆盖内部）
            query_embeds: 查询嵌入向量，用于置信度计算
            
        Returns:
            Dict 包含：
                answer: 推理答案
                reasoning_chain: 推理步骤链
                confidence: 推理置信度
                facts_used: 使用的事实列表
        """
        # 重置推理追踪
        self.reasoning_trace = []
        
        # 使用外部知识图谱（如果提供）
        if knowledge_graph is not None:
            kg = knowledge_graph
        else:
            kg = self.knowledge_graph
        
        # 1. 解析查询
        query_type, query_entities = self._parse_query(query)
        self.reasoning_trace.append(f"解析查询：类型={query_type}, 实体={query_entities}")
        
        # 2. 收集相关事实
        facts = self._collect_facts(query_entities, context, kg)
        self.reasoning_trace.append(f"收集事实：{facts}")
        
        # 3. 执行推理
        reasoning_chain = []
        current_facts = set(facts)
        answer = None
        confidence = 0.0
        
        for step in range(self.config.max_reasoning_steps):
            self.reasoning_trace.append(f"推理步骤 {step + 1}:")
            
            # 应用规则生成新事实
            new_facts = self._apply_rules(current_facts)
            self.reasoning_trace.append(f"  生成新事实：{new_facts}")
            
            # 检查是否找到答案
            candidate_answer = self._check_answer(query_type, query_entities, current_facts | new_facts, kg)
            if candidate_answer is not None:
                answer = candidate_answer
                reasoning_chain = list(self.reasoning_trace)
                confidence = self._compute_confidence(
                    query_embeds, 
                    len(reasoning_chain),
                    len(current_facts | new_facts)
                )
                self.reasoning_trace.append(f"找到答案：{answer} (置信度：{confidence:.2f})")
                break
            
            # 更新事实集
            if new_facts:
                current_facts.update(new_facts)
            else:
                # 没有新事实生成，提前结束
                break
        
        # 如果没有找到明确答案，生成默认回答
        if answer is None:
            answer = self._generate_default_answer(query_type, query_entities, current_facts)
            reasoning_chain = list(self.reasoning_trace)
            confidence = 0.3  # 低置信度
        
        # 提取使用的事实
        facts_used = list(current_facts)[:5]  # 限制显示数量
        
        return {
            'answer': answer,
            'reasoning_chain': reasoning_chain,
            'confidence': float(confidence),
            'facts_used': facts_used
        }
    
    def _parse_query(self, query: str) -> Tuple[str, List[str]]:
        """解析查询，确定查询类型和实体"""
        # 简单的基于关键词的查询分类
        query_lower = query.lower()
        
        # 提取实体（简单实现，实际中可用 NER）
        entities = self._extract_entities(query)
        
        # 判断查询类型
        if any(q in query_lower for q in ['什么是', '是什么', 'who is', 'what is', '是指']):
            query_type = 'definition'
        elif any(q in query_lower for q in ['为什么', '原因', 'why', '因为']):
            query_type = 'causal'
        elif any(q in query_lower for q in ['如何', '怎么', 'how to', 'how']):
            query_type = 'process'
        elif any(q in query_lower for q in ['是否', '是不是', 'is', 'are', '吗']):
            query_type = 'yesno'
        else:
            query_type = 'general'
        
        return query_type, entities
    
    def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体（简化版）"""
        # 简单实现：提取名词短语，实际中可用 NER 模型
        entities = []
        kg_entities = self.knowledge_graph.keys()
        
        for entity in kg_entities:
            if entity in text:
                entities.append(entity)
        
        # 如果没有找到已知实体，尝试提取名词
        if not entities:
            # 简单的中文名词提取（基于长度和位置）
            words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text)
            entities = [w for w in words if len(w) >= 2][:3]
        
        return entities if entities else ['未知实体']
    
    def _collect_facts(
        self,
        entities: List[str],
        context: Optional[List[str]] = None,
        knowledge_graph: Dict[str, Any] = None
    ) -> List[str]:
        """收集与实体相关的事实"""
        facts = set()
        
        # 从知识图谱收集事实
        if knowledge_graph:
            for entity in entities:
                if entity in knowledge_graph:
                    relations = knowledge_graph[entity]
                    for rel, targets in relations.items():
                        for target in targets:
                            facts.add(f"{entity}{rel}{target}")
        
        # 添加上下文事实
        if context:
            facts.update(context)
        
        return list(facts)
    
    def _apply_rules(self, facts: set) -> set:
        """应用规则生成新事实"""
        new_facts = set()
        
        # 简化版：基于模式匹配的规则应用
        # 实际中可使用更复杂的推理引擎
        
        # 检查所有可能的三元组组合
        fact_list = list(facts)
        for i in range(len(fact_list)):
            for j in range(len(fact_list)):
                if i == j:
                    continue
                
                fact1 = fact_list[i]
                fact2 = fact_list[j]
                
                # 传递性推理：A 是 B，B 是 C  A 是 C
                if "是" in fact1 and "是" in fact2:
                    parts1 = fact1.split("是", 1)
                    parts2 = fact2.split("是", 1)
                    if len(parts1) == 2 and len(parts2) == 2:
                        a, b = parts1
                        b2, c = parts2
                        if b.strip() == b2.strip() and a.strip() != c.strip():
                            new_fact = f"{a.strip()}是{c.strip()}"
                            if new_fact not in facts:
                                new_facts.add(new_fact)
        
        return new_facts
    
    def _check_answer(
        self,
        query_type: str,
        entities: List[str],
        facts: set,
        knowledge_graph: Dict[str, Any]
    ) -> Optional[str]:
        """检查是否找到答案"""
        if not entities:
            return None
        
        entity = entities[0]
        
        # 定义类查询
        if query_type == 'definition' or query_type == 'general':
            answers = []
            for fact in facts:
                if entity in fact:
                    # 提取属性和关系
                    if "是" in fact:
                        parts = fact.split("是", 1)
                        if len(parts) == 2 and parts[0].strip() == entity:
                            answers.append(parts[1].strip())
                    elif "属性" in fact:
                        answers.append(fact)
            
            if answers:
                return f"{entity}是{', '.join(answers[:3])}"
        
        # 是否类查询
        elif query_type == 'yesno':
            for fact in facts:
                if entity in fact and ("是" in fact or "不是" in fact):
                    return f"是的，{fact}" if "不是" not in fact else f"不是，{fact}"
        
        return None
    
    def _compute_confidence(
        self,
        query_embeds: Optional[torch.Tensor],
        reasoning_length: int,
        num_facts: int
    ) -> float:
        """计算推理置信度"""
        # 基于规则的置信度计算
        # 推理步骤越少，置信度越高；使用事实越多，置信度越高
        base_confidence = 0.8
        
        # 推理长度惩罚
        length_penalty = max(0.5, 1.0 - (reasoning_length - 1) * 0.15)
        
        # 事实数量奖励（最多奖励 5 个事实）
        fact_bonus = min(0.2, num_facts * 0.04)
        
        confidence = base_confidence * length_penalty + fact_bonus
        confidence = max(0.1, min(1.0, confidence))
        
        # 如果有嵌入，使用神经网络增强置信度计算
        if query_embeds is not None:
            with torch.no_grad():
                # 确保输入在正确的设备上
                query_embeds = query_embeds.to(self._device)
                # 编码查询嵌入 - query_features shape: (1, 64)
                query_features = self.text_encoder(query_embeds.mean(dim=0, keepdim=True))
                
                # 构造推理特征 - 需要填充到 16 维以匹配 confidence_scorer 输入 (64+16=80)
                reasoning_features_list = [
                    float(reasoning_length * 0.1),
                    float(num_facts * 0.1),
                    float(confidence),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]  # 总共 16 维
                reasoning_features = torch.tensor(
                    [reasoning_features_list],
                    device=self._device
                )  # shape: (1, 16)
                
                # 组合特征 - cat 后 shape: (1, 80)
                combined = torch.cat([query_features, reasoning_features], dim=1)
                
                # 计算置信度
                confidence = self.confidence_scorer(combined).item()
        
        return confidence
    
    def _generate_default_answer(
        self,
        query_type: str,
        entities: List[str],
        facts: set
    ) -> str:
        """生成默认回答"""
        entity = entities[0] if entities else "这个"
        
        if facts:
            related_info = [f for f in facts if entity in f]
            if related_info:
                return f"关于{entity}，我知道：{', '.join(related_info[:3])}"
        
        return f"关于{entity}，我目前了解有限，需要更多信息来回答。"
    
    def add_knowledge(self, entity: str, relation: str, target: str) -> None:
        """添加知识到知识图谱"""
        if entity not in self.knowledge_graph:
            self.knowledge_graph[entity] = {}
        
        if relation not in self.knowledge_graph[entity]:
            self.knowledge_graph[entity][relation] = []
        
        if target not in self.knowledge_graph[entity][relation]:
            self.knowledge_graph[entity][relation].append(target)
    
    def clear_knowledge(self) -> None:
        """清空知识图谱"""
        self.knowledge_graph = {}
