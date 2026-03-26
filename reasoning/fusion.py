# 模块I：双系统融合 (DualSystemFusion)
# 负责人：AI 1
# 输入：system1_output: Dict, system2_output: Dict, regulation_signals: Optional[Dict]
# 输出：{'final_answer': Any, 'fusion_state': Dict, 'system_weights': Dict, 'confidence': float}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import time

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class FusionConfig(BaseConfig):
    """双系统融合配置"""
    hidden_dim: int = 768
    fusion_strategy: str = 'dynamic'  # 'dynamic', 'static', 'confidence_based'
    default_system1_weight: float = 0.5
    default_system2_weight: float = 0.5
    confidence_threshold: float = 0.6
    enable_meta_learning: bool = True


class DualSystemFusion(BaseModule):
    """
    双系统融合模块
    
    实现动态权重分配、冲突检测与解决、元认知校准。
    基于两个系统的置信度、任务类型、调节信号进行智能融合。
    """
    
    def __init__(
        self, 
        config: Optional[FusionConfig] = None,
        module_name: str = "dual_system_fusion"
    ):
        config = config or FusionConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 融合策略网络
        self.fusion_net = nn.Sequential(
            nn.Linear(16, 64),  # 输入特征：置信度、相似度、任务复杂度等
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 输出：system1_weight, system2_weight
            nn.Softmax(dim=-1)  # 权重归一化
        )
        
        # 冲突检测网络
        self.conflict_detector = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 元认知校准网络
        if self.config.enable_meta_learning:
            self.meta_calibration = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
                nn.Tanh()  # 输出校准因子
            )
        
        # 融合历史记录
        self.fusion_history = []
        
        # 任务类型模式识别（简单规则）
        self.task_patterns = {
            'fast_pattern_matching': 0.7,  # 系统1权重高
            'complex_reasoning': 0.3,      # 系统1权重低
            'fact_verification': 0.5,
            'creative_generation': 0.6
        }
    
    def forward(
        self,
        system1_output: Dict[str, Any],
        system2_output: Dict[str, Any],
        regulation_signals: Optional[Dict[str, Any]] = None,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        前向传播 - 双系统融合
        
        Args:
            system1_output: 系统1输出字典
            system2_output: 系统2输出字典
            regulation_signals: 神经调节信号（可选）
            task_context: 任务上下文信息（可选）
            
        Returns:
            Dict包含：
                final_answer: 融合后的最终答案
                fusion_state: 融合状态信息
                system_weights: 系统权重分配
                confidence: 融合置信度
                conflict_resolution: 冲突解决信息（如果有冲突）
        """
        # 1. 提取系统输出特征
        s1_features = self._extract_system_features(system1_output, 'system1')
        s2_features = self._extract_system_features(system2_output, 'system2')
        
        # 2. 检测输出冲突
        conflict_info = self._detect_conflict(system1_output, system2_output)
        
        # 3. 确定融合策略和权重
        if self.config.fusion_strategy == 'static':
            weights = {
                'system1': self.config.default_system1_weight,
                'system2': self.config.default_system2_weight
            }
        elif self.config.fusion_strategy == 'confidence_based':
            weights = self._confidence_based_weighting(s1_features, s2_features)
        else:  # dynamic
            weights = self._dynamic_weighting(
                s1_features, s2_features, 
                regulation_signals, task_context
            )
        
        # 4. 应用元认知校准（如果启用）
        if self.config.enable_meta_learning:
            weights = self._apply_meta_calibration(weights, conflict_info, regulation_signals)
        
        # 5. 执行融合
        fusion_result = self._fuse_outputs(
            system1_output, system2_output,
            weights, conflict_info
        )
        
        # 6. 记录融合历史
        self._record_fusion(
            system1_output, system2_output,
            weights, fusion_result, conflict_info
        )
        
        return {
            'final_answer': fusion_result['answer'],
            'fusion_state': {
                'strategy_used': self.config.fusion_strategy,
                'conflict_detected': conflict_info['has_conflict'],
                'conflict_severity': conflict_info['severity'],
                'fusion_method': fusion_result['method']
            },
            'system_weights': weights,
            'confidence': fusion_result['confidence'],
            'conflict_resolution': conflict_info.get('resolution', {})
        }
    
    def _extract_system_features(
        self,
        output: Dict[str, Any],
        system_name: str
    ) -> Dict[str, Any]:
        """提取系统输出的特征"""
        # 处理置信度：系统1可能返回list，系统2返回float
        conf = output.get('confidence', 0.5)
        if isinstance(conf, (list, tuple)):
            conf = conf[0] if conf else 0.5
        confidence = float(conf)
        
        features = {
            'confidence': confidence,
            'has_result': 1.0 if (output.get('retrieved_results') or output.get('answer')) else 0.0,
            'result_count': float(len(output.get('retrieved_results', []))) if system_name == 'system1' else 0.0,
            'similarity_max': self._get_max_similarity(output.get('similarities', [[0]])) if system_name == 'system1' else 0.0,
            'similarity_mean': self._get_mean_similarity(output.get('similarities', [[0]])) if system_name == 'system1' else 0.0,
            'reasoning_steps': float(len(output.get('reasoning_chain', []))) if system_name == 'system2' else 0.0,
            'facts_used': float(len(output.get('facts_used', []))) if system_name == 'system2' else 0.0,
            'response_time': 0.0  # 预留字段
        }
        
        # 转换为 8 维特征向量（fusion_net 期望输入 16 维，在_dynamic_weighting 中会拼接两个系统的特征）
        feature_vector = torch.tensor([
            features['confidence'],
            features['has_result'],
            features['result_count'] * 0.1,
            features['similarity_max'],
            features['similarity_mean'],
            features['reasoning_steps'] * 0.1,
            features['facts_used'] * 0.1,
            features['response_time']
        ], device=self._device).unsqueeze(0)
        
        features['vector'] = feature_vector
        return features
    
    def _get_max_similarity(self, similarities: List[List[float]]) -> float:
        """获取最大相似度（处理嵌套列表）"""
        if not similarities:
            return 0.0
        # 展平列表
        flat = []
        for sublist in similarities:
            if isinstance(sublist, list):
                flat.extend(sublist)
            else:
                flat.append(sublist)
        return max(flat) if flat else 0.0
    
    def _get_mean_similarity(self, similarities: List[List[float]]) -> float:
        """获取平均相似度（处理嵌套列表）"""
        if not similarities:
            return 0.0
        # 展平列表
        flat = []
        for sublist in similarities:
            if isinstance(sublist, list):
                flat.extend(sublist)
            else:
                flat.append(sublist)
        return float(np.mean(flat)) if flat else 0.0
    
    def _detect_conflict(
        self,
        s1_output: Dict[str, Any],
        s2_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检测两个系统输出之间的冲突"""
        conflict_info = {
            'has_conflict': False,
            'severity': 0.0,
            'type': None,
            'resolution': {}
        }
        
        # 提取答案内容
        s1_results = s1_output.get('retrieved_results', [])
        s2_answer = s2_output.get('answer', '')
        
        if not s1_results or not s2_answer:
            return conflict_info
        
        # 简单的冲突检测：基于关键词重叠
        s1_content = ' '.join([str(r) for r in s1_results[0]]) if s1_results and s1_results[0] else ''
        s2_content = str(s2_answer)
        
        # 计算词汇重叠度
        def compute_overlap(a: str, b: str) -> float:
            words_a = set(a.split())
            words_b = set(b.split())
            if not words_a or not words_b:
                return 0.0
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            return intersection / union if union > 0 else 0.0
        
        overlap = compute_overlap(s1_content, s2_content)
        # 处理置信度格式差异
        def get_confidence(output):
            conf = output.get('confidence', 0.5)
            if isinstance(conf, (list, tuple)):
                conf = conf[0] if conf else 0.5
            return float(conf)
        
        s1_conf = get_confidence(s1_output)
        s2_conf = get_confidence(s2_output)
        confidence_diff = abs(s1_conf - s2_conf)
        
        # 冲突判断
        if overlap < 0.2 and confidence_diff > 0.3:
            conflict_info['has_conflict'] = True
            conflict_info['severity'] = min(1.0, (1.0 - overlap) * 0.5 + confidence_diff * 0.5)
            conflict_info['type'] = 'content_mismatch'
            
            # 初步解决策略：选择置信度高的（使用之前计算的）
            
            if s1_conf > s2_conf + 0.1:
                conflict_info['resolution'] = {
                    'chosen_system': 'system1',
                    'reason': '系统1置信度显著更高'
                }
            elif s2_conf > s1_conf + 0.1:
                conflict_info['resolution'] = {
                    'chosen_system': 'system2',
                    'reason': '系统2置信度显著更高'
                }
            else:
                conflict_info['resolution'] = {
                    'chosen_system': 'both',
                    'reason': '两个系统置信度相当，需要综合考虑'
                }
        
        return conflict_info
    
    def _confidence_based_weighting(
        self,
        s1_features: Dict[str, Any],
        s2_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """基于置信度的权重分配"""
        s1_conf = s1_features['confidence']
        s2_conf = s2_features['confidence']
        
        total_conf = s1_conf + s2_conf
        if total_conf == 0:
            return {'system1': 0.5, 'system2': 0.5}
        
        return {
            'system1': s1_conf / total_conf,
            'system2': s2_conf / total_conf
        }
    
    def _dynamic_weighting(
        self,
        s1_features: Dict[str, Any],
        s2_features: Dict[str, Any],
        regulation_signals: Optional[Dict[str, Any]] = None,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """动态权重分配（使用神经网络）"""
        # 构建输入特征向量
        input_features = []
        
        # 系统1特征（8维）
        input_features.extend([
            s1_features['confidence'],
            s1_features['has_result'],
            s1_features['similarity_max'],
            s1_features['similarity_mean'],
            s1_features['result_count'] * 0.1,
            0.0, 0.0, 0.0
        ])
        
        # 系统2特征（8维）
        input_features.extend([
            s2_features['confidence'],
            s2_features['has_result'],
            s2_features['reasoning_steps'] * 0.1,
            s2_features['facts_used'] * 0.1,
            0.0, 0.0, 0.0, 0.0
        ])
        
        # 转换为tensor
        feature_tensor = torch.tensor(
            input_features,
            device=self._device
        ).unsqueeze(0).float()
        
        # 应用融合网络
        with torch.no_grad():
            weights = self.fusion_net(feature_tensor).squeeze(0).cpu().numpy()
        
        return {
            'system1': float(weights[0]),
            'system2': float(weights[1])
        }
    
    def _apply_meta_calibration(
        self,
        weights: Dict[str, float],
        conflict_info: Dict[str, Any],
        regulation_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """应用元认知校准"""
        # 构建校准特征
        calib_features = [
            weights['system1'],
            weights['system2'],
            conflict_info['severity'],
            1.0 if conflict_info['has_conflict'] else 0.0
        ]
        
        # 添加神经调节信号
        if regulation_signals:
            calib_features.extend([
                regulation_signals.get('attention_strength', 1.0),
                regulation_signals.get('learning_rate', 1.0),
                regulation_signals.get('system_bias', 0.0),
                0.0
            ])
        else:
            calib_features.extend([1.0, 1.0, 0.0, 0.0])
        
        # 填充到32维
        while len(calib_features) < 32:
            calib_features.append(0.0)
        
        # 应用校准网络
        with torch.no_grad():
            feature_tensor = torch.tensor(
                calib_features,
                device=self._device
            ).unsqueeze(0).float()
            
            calibration = self.meta_calibration(feature_tensor).squeeze(0).cpu().numpy()
        
        # 应用校准因子（限制在合理范围）
        s1_calib = np.clip(calibration[0] * 0.1, -0.3, 0.3)
        s2_calib = np.clip(calibration[1] * 0.1, -0.3, 0.3)
        
        new_s1 = np.clip(weights['system1'] + s1_calib, 0.1, 0.9)
        new_s2 = np.clip(weights['system2'] + s2_calib, 0.1, 0.9)
        
        # 重新归一化
        total = new_s1 + new_s2
        return {
            'system1': new_s1 / total,
            'system2': new_s2 / total
        }
    
    def _fuse_outputs(
        self,
        s1_output: Dict[str, Any],
        s2_output: Dict[str, Any],
        weights: Dict[str, float],
        conflict_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """融合两个系统的输出"""
        s1_weight = weights['system1']
        s2_weight = weights['system2']
        
        # 辅助函数：获取置信度
        def get_confidence(output):
            conf = output.get('confidence', 0.5)
            if isinstance(conf, (list, tuple)):
                conf = conf[0] if conf else 0.5
            return float(conf)
        
        # 获取两个系统的答案
        s1_results = s1_output.get('retrieved_results', [[]])
        s1_answer = s1_results[0][0] if s1_results and len(s1_results) > 0 and len(s1_results[0]) > 0 else ""
        
        s2_answer = s2_output.get('answer', '')
        
        # 计算综合置信度
        s1_conf = get_confidence(s1_output)
        s2_conf = get_confidence(s2_output)
        combined_confidence = s1_conf * s1_weight + s2_conf * s2_weight
        
        # 根据冲突情况选择融合方法
        if conflict_info['has_conflict'] and conflict_info['severity'] > 0.5:
            # 严重冲突，选择置信度高的或提示冲突
            if s1_conf > s2_conf + 0.2:
                final_answer = s1_answer
                method = 'conflict_select_s1'
            elif s2_conf > s1_conf + 0.2:
                final_answer = s2_answer
                method = 'conflict_select_s2'
            else:
                final_answer = self._synthesize_answer(s1_answer, s2_answer)
                method = 'conflict_synthesis'
        else:
            # 无冲突或轻微冲突，综合两个答案
            final_answer = self._synthesize_answer(s1_answer, s2_answer)
            method = 'synthesis'
        
        return {
            'answer': final_answer,
            'confidence': combined_confidence,
            'method': method
        }
    
    def _synthesize_answer(self, s1_answer: str, s2_answer: str) -> str:
        """综合两个系统的答案"""
        s1_answer = str(s1_answer).strip()
        s2_answer = str(s2_answer).strip()
        
        # 如果只有一个系统有答案
        if not s1_answer or s1_answer == "[]":
            return s2_answer if s2_answer else "无法找到答案"
        if not s2_answer:
            return s1_answer
        
        # 检查内容是否相似
        def content_similarity(a: str, b: str) -> float:
            words_a = set(a.split())
            words_b = set(b.split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / max(len(words_a), len(words_b))
        
        similarity = content_similarity(s1_answer, s2_answer)
        
        if similarity > 0.5:
            # 内容相似，选择更详细的答案
            return s1_answer if len(s1_answer) > len(s2_answer) else s2_answer
        else:
            # 内容互补，组合答案
            return f"{s1_answer}。此外，{s2_answer}"
    
    def _record_fusion(
        self,
        s1_output: Dict[str, Any],
        s2_output: Dict[str, Any],
        weights: Dict[str, float],
        fusion_result: Dict[str, Any],
        conflict_info: Dict[str, Any]
    ) -> None:
        """记录融合历史用于元学习"""
        # 辅助函数：获取置信度
        def get_confidence(output):
            conf = output.get('confidence', 0.5)
            if isinstance(conf, (list, tuple)):
                conf = conf[0] if conf else 0.5
            return float(conf)
            
        record = {
            'timestamp': time.time(),
            'system1_confidence': get_confidence(s1_output),
            'system2_confidence': get_confidence(s2_output),
            'weights': weights.copy(),
            'fusion_confidence': fusion_result['confidence'],
            'has_conflict': conflict_info['has_conflict'],
            'conflict_severity': conflict_info['severity']
        }
        
        self.fusion_history.append(record)
        
        # 限制历史记录大小
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-500:]
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        if not self.fusion_history:
            return {'message': '没有融合历史记录'}
        
        avg_s1_weight = np.mean([r['weights']['system1'] for r in self.fusion_history])
        avg_s2_weight = np.mean([r['weights']['system2'] for r in self.fusion_history])
        avg_confidence = np.mean([r['fusion_confidence'] for r in self.fusion_history])
        conflict_rate = np.mean([1 if r['has_conflict'] else 0 for r in self.fusion_history])
        
        return {
            'total_fusions': len(self.fusion_history),
            'average_system1_weight': float(avg_s1_weight),
            'average_system2_weight': float(avg_s2_weight),
            'average_fusion_confidence': float(avg_confidence),
            'conflict_rate': float(conflict_rate)
        }
