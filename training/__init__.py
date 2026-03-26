# 训练模块 - 纯生物启发式训练框架
"""
训练模块提供完整的训练功能，包括：

1. 训练配置：
   - TrainingConfig: 训练超参数配置
   - BioTrainer: 生物启发式训练器

2. 数据集：
   - BioDataset: 训练数据集包装器
   - ChineseTextDataset: 中文文本数据集加载器
   - get_dataloader: 便捷创建DataLoader
   - analyze_dataset: 分析数据集内容
   
3. 便捷函数：
   - load_pretrained_model: 加载预训练模型

保存格式（PT）
"""

from .trainer import (
    TrainingConfig,
    BioDataset,
    BioTrainer,
    load_pretrained_model,
)
from .data_loader import (
    ChineseTextDataset,
    get_dataloader,
    analyze_dataset,
)

__all__ = [
    'TrainingConfig',
    'BioDataset',
    'BioTrainer',
    'load_pretrained_model',
    'ChineseTextDataset',
    'get_dataloader',
    'analyze_dataset',
]
