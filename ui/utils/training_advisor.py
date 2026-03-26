#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能训练助手 - 自动分析数据并推荐训练配置
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecommendation:
    """训练推荐配置"""
    # 数据类型
    data_type: str
    task_type: str
    
    # 推荐配置
    batch_size: int
    learning_rate: float
    epochs: int
    max_seq_length: int
    save_steps: int
    
    # 说明
    reason: str
    estimated_time: str


class IntelligentTrainingAdvisor:
    """智能训练顾问"""
    
    def __init__(self):
        # 数据类型到训练配置的映射
        self.config_templates = {
            'qna': {
                'task_type': '问答生成',
                'batch_size': 16,
                'learning_rate': 1e-4,
                'epochs': 10,
                'max_seq_length': 128,
                'save_steps': 50,
                'reason': '问答数据通常较短，适合中等批次和学习率'
            },
            'deepseek': {
                'task_type': '推理蒸馏',
                'batch_size': 8,
                'learning_rate': 5e-5,
                'epochs': 15,
                'max_seq_length': 256,
                'save_steps': 30,
                'reason': '包含推理过程，需要更长序列和更多 epoch'
            },
            'conversation': {
                'task_type': '对话生成',
                'batch_size': 32,
                'learning_rate': 2e-4,
                'epochs': 8,
                'max_seq_length': 192,
                'save_steps': 40,
                'reason': '对话数据量大，可以使用大批次快速训练'
            },
            'novel': {
                'task_type': '文本生成',
                'batch_size': 8,
                'learning_rate': 1e-4,
                'epochs': 20,
                'max_seq_length': 512,
                'save_steps': 20,
                'reason': '长文本需要大序列长度和更多训练轮数'
            },
            'classification': {
                'task_type': '文本分类',
                'batch_size': 64,
                'learning_rate': 5e-4,
                'epochs': 5,
                'max_seq_length': 64,
                'save_steps': 100,
                'reason': '分类任务简单，可以快速收敛'
            }
        }
    
    def analyze_and_recommend(
        self,
        data_stats: Dict[str, Any],
        data_format: str,
        num_samples: int
    ) -> TrainingRecommendation:
        """
        分析数据并推荐训练配置
        
        Args:
            data_stats: 数据统计信息
            data_format: 数据格式类型
            num_samples: 样本数量
            
        Returns:
            训练推荐配置
        """
        # 获取基础模板
        template = self.config_templates.get(data_format, self.config_templates['qna'])
        
        # 根据数据特征调整配置
        config = template.copy()
        
        # 根据样本数量调整 epoch
        if num_samples < 1000:
            config['epochs'] = min(config['epochs'] * 2, 50)
            config['reason'] += '；数据量少，增加训练轮数'
        elif num_samples > 100000:
            config['epochs'] = max(config['epochs'] // 2, 3)
            config['reason'] += '；数据量大，减少训练轮数'
        
        # 根据平均文本长度调整 max_seq_length
        avg_length = data_stats.get('text_stats', {}).get('avg_length', 100)
        if isinstance(avg_length, dict):
            # 如果有多个字段，取最大值
            avg_length = max(v.get('avg_length', 100) for v in avg_length.values())
        
        if avg_length > 300:
            config['max_seq_length'] = 512
            config['reason'] += '；文本较长，使用更大序列长度'
        elif avg_length < 50:
            config['max_seq_length'] = 64
            config['reason'] += '；文本较短，使用较小序列长度'
        
        # 估算训练时间
        estimated_time = self._estimate_training_time(
            num_samples=num_samples,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            seq_length=config['max_seq_length']
        )
        
        return TrainingRecommendation(
            data_type=data_format,
            task_type=template['task_type'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            max_seq_length=config['max_seq_length'],
            save_steps=config['save_steps'],
            reason=config['reason'],
            estimated_time=estimated_time
        )
    
    def _estimate_training_time(
        self,
        num_samples: int,
        batch_size: int,
        epochs: int,
        seq_length: int
    ) -> str:
        """估算训练时间"""
        # 简化估算公式
        steps_per_epoch = num_samples / batch_size
        total_steps = steps_per_epoch * epochs
        
        # 假设每步耗时与序列长度相关
        time_per_step = 0.1 + (seq_length / 512) * 0.2  # 秒
        
        total_seconds = total_steps * time_per_step
        
        # 转换为可读格式
        if total_seconds < 60:
            return f"约{int(total_seconds)}秒"
        elif total_seconds < 3600:
            return f"约{int(total_seconds / 60)}分钟"
        else:
            hours = total_seconds / 3600
            return f"约{hours:.1f}小时"
    
    def get_quick_config(self, data_format: str, num_samples: int) -> Dict[str, Any]:
        """获取快速配置（一键训练用）"""
        recommendation = self.analyze_and_recommend(
            data_stats={},
            data_format=data_format,
            num_samples=num_samples
        )
        
        return {
            'batch_size': recommendation.batch_size,
            'num_epochs': recommendation.epochs,
            'learning_rate': recommendation.learning_rate,
            'max_seq_length': recommendation.max_seq_length,
            'save_steps': recommendation.save_steps,
            'task_type': recommendation.task_type,
            'estimated_time': recommendation.estimated_time
        }


# 全局单例
_advisor_instance: Optional[IntelligentTrainingAdvisor] = None

def get_training_advisor() -> IntelligentTrainingAdvisor:
    """获取训练顾问单例"""
    global _advisor_instance
    if _advisor_instance is None:
        _advisor_instance = IntelligentTrainingAdvisor()
    return _advisor_instance
