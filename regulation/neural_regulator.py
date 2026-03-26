# 模块F：神经调节器 (NeuralRegulator)
# 负责人：AI 1
# 输入：errors: Dict[str, float], task_complexity: float, system_state: Dict
# 输出：{'attention_strength': float, 'learning_rate': float, 'system_bias': float, 'regulation_state': Dict}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils


@dataclass
class NeuralRegulatorConfig(BaseConfig):
    """神经调节器配置"""
    hidden_dim: int = 256
    max_attention_strength: float = 2.0
    min_attention_strength: float = 0.1
    max_learning_rate: float = 0.01
    min_learning_rate: float = 0.0001
    system_bias_range: Tuple[float, float] = (-1.0, 1.0)
    update_rate: float = 0.1  # 状态更新速率


class NeuralRegulator(BaseModule):
    """
    神经调节器模块
    
    实现状态感知、简化版调节算法、调节信号输出。
    输入信号：预测误差、任务复杂度，输出：attention_strength, learning_rate, system_bias
    第一天用规则-based，不用完整RL。
    """
    
    def __init__(
        self, 
        config: Optional[NeuralRegulatorConfig] = None,
        module_name: str = "neural_regulator"
    ):
        config = config or NeuralRegulatorConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化模块特定组件"""
        # 状态编码器：将各种状态信号编码为内部表示
        self.state_encoder = nn.Sequential(
            nn.Linear(16, self.config.hidden_dim),  # 输入特征维度
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU()
        )
        
        # 调节信号生成器
        self.regulation_generator = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出: attention_strength, learning_rate, system_bias
        )
        
        # 内部状态追踪
        self.internal_state = {
            'arousal_level': 0.5,  # 唤醒水平
            'focus_level': 0.5,    # 专注水平
            'stress_level': 0.0,   # 压力水平
            'last_update': time.time()
        }
        
        # 历史记录
        self.history = {
            'prediction_errors': [],
            'regulation_signals': [],
            'states': []
        }
    
    def forward(
        self,
        errors: Dict[str, float],
        task_complexity: float = 0.5,
        system_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        前向传播 - 生成调节信号
        
        Args:
            errors: 预测误差字典 {'layer1': error, 'layer2': error, ...}
            task_complexity: 任务复杂度评估 (0.0 - 1.0)
            system_state: 系统状态字典（可选）
            
        Returns:
            Dict包含：
                attention_strength: 注意力强度 (0.1 - 2.0)
                learning_rate: 学习率调整因子
                system_bias: 系统偏置 (-1.0 - 1.0)
                regulation_state: 调节器内部状态
        """
        # 1. 状态感知 - 整合所有输入信号
        state_features = self._perceive_state(errors, task_complexity, system_state)
        
        # 2. 更新内部状态
        self._update_internal_state(state_features)
        
        # 3. 生成调节信号（规则-based 方法，简化版）
        regulation = self._generate_regulation_signals(state_features)
        
        # 4. 记录历史
        self._record_history(errors, regulation)
        
        return {
            'attention_strength': regulation['attention_strength'],
            'learning_rate': regulation['learning_rate'],
            'system_bias': regulation['system_bias'],
            'regulation_state': self.internal_state.copy()
        }
    
    def _perceive_state(
        self,
        errors: Dict[str, float],
        task_complexity: float,
        system_state: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """感知系统状态，生成特征向量"""
        batch_size = 1  # 简化为单批次处理
        
        # 收集误差特征
        error_values = list(errors.values()) if errors else [0.0]
        avg_error = sum(error_values) / len(error_values) if error_values else 0.0
        max_error = max(error_values) if error_values else 0.0
        error_variance = torch.var(torch.tensor(error_values)).item() if len(error_values) > 1 else 0.0
        
        # 构建特征向量
        features = [
            avg_error,
            max_error,
            error_variance,
            task_complexity,
            self.internal_state['arousal_level'],
            self.internal_state['focus_level'],
            self.internal_state['stress_level'],
            len(error_values) * 0.1  # 误差源数量归一化
        ]
        
        # 如果有系统状态，添加更多特征
        if system_state:
            features.extend([
                system_state.get('memory_usage', 0.5),
                system_state.get('computation_load', 0.5),
                system_state.get('confidence', 0.5)
            ])
        
        # 填充到固定长度（16维）
        while len(features) < 16:
            features.append(0.0)
        
        return torch.tensor(features, dtype=torch.float32, device=self._device).unsqueeze(0)
    
    def _update_internal_state(self, state_features: torch.Tensor) -> None:
        """更新内部状态（唤醒、专注、压力水平）"""
        features = state_features.squeeze()
        
        # 基于预测误差更新唤醒水平
        avg_error = features[0].item()
        arousal_update = avg_error * 0.5
        self.internal_state['arousal_level'] = (
            self.internal_state['arousal_level'] * (1 - self.config.update_rate) +
            arousal_update * self.config.update_rate
        )
        self.internal_state['arousal_level'] = max(0.0, min(1.0, self.internal_state['arousal_level']))
        
        # 基于误差变化更新专注水平
        error_variance = features[2].item()
        focus_update = 1.0 - min(1.0, error_variance * 2.0)
        self.internal_state['focus_level'] = (
            self.internal_state['focus_level'] * (1 - self.config.update_rate) +
            focus_update * self.config.update_rate
        )
        self.internal_state['focus_level'] = max(0.0, min(1.0, self.internal_state['focus_level']))
        
        # 基于任务复杂度更新压力水平
        task_complexity = features[3].item()
        stress_update = task_complexity * features[1].item()  # max_error
        self.internal_state['stress_level'] = (
            self.internal_state['stress_level'] * 0.9 +
            stress_update * 0.1
        )
        self.internal_state['stress_level'] = max(0.0, min(1.0, self.internal_state['stress_level']))
        
        self.internal_state['last_update'] = time.time()
    
    def _generate_regulation_signals(self, state_features: torch.Tensor) -> Dict[str, float]:
        """基于规则生成调节信号（简化版，不使用RL）"""
        features = state_features.squeeze()
        avg_error = features[0].item()
        max_error = features[1].item()
        task_complexity = features[3].item()
        
        # 1. 注意力强度调节
        # 误差越大、任务越复杂，注意力强度越高
        base_attention = 0.5 + avg_error * 0.5 + task_complexity * 0.3
        attention_strength = base_attention * self.internal_state['arousal_level']
        attention_strength = max(
            self.config.min_attention_strength,
            min(self.config.max_attention_strength, attention_strength)
        )
        
        # 2. 学习率调节
        # 误差稳定时降低学习率，误差波动时提高学习率
        error_stability = 1.0 - min(1.0, features[2].item() * 3.0)  # error_variance
        lr_factor = 0.5 + error_stability * 0.3 + (1.0 - self.internal_state['stress_level']) * 0.2
        learning_rate = lr_factor * 0.001  # 基础学习率缩放
        
        # 3. 系统偏置调节
        # 基于唤醒水平和专注水平调整系统偏置
        # 高唤醒、高专注  正向偏置（更积极探索）
        # 低唤醒、低专注  负向偏置（更保守）
        system_bias = (
            (self.internal_state['arousal_level'] - 0.5) * 0.5 +
            (self.internal_state['focus_level'] - 0.5) * 0.3 +
            (task_complexity - 0.5) * 0.2
        )
        min_bias, max_bias = self.config.system_bias_range
        system_bias = max(min_bias, min(max_bias, system_bias))
        
        return {
            'attention_strength': float(attention_strength),
            'learning_rate': float(learning_rate),
            'system_bias': float(system_bias)
        }
    
    def _record_history(
        self,
        errors: Dict[str, float],
        regulation: Dict[str, float]
    ) -> None:
        """记录历史用于分析"""
        self.history['prediction_errors'].append(errors)
        self.history['regulation_signals'].append(regulation)
        self.history['states'].append(self.internal_state.copy())
        
        # 限制历史长度
        max_history = 1000
        if len(self.history['prediction_errors']) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]
