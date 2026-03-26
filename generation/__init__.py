# 生成器模块 - 完全生物启发版
# 已重构：完全替换Transformer解码器，使用HTM时序记忆和SDR表示

# 核心生成器（生物启发式，原ResponseGenerator的别名）
from .generator import ResponseGenerator as Generator, BioGenerator, GeneratorConfig

# 新的生物启发组件
from .sequence_predictor import SequencePredictorTM, SequencePredictorConfig, SparseWinnerTakeAll
from .sdr_vocabulary import SDRVocabulary, SDRVocabularyConfig, SDRPrototypeInitializer

# 前向模型
from .forward_model import ForwardModel

__all__ = [
    'Generator',           # 主生成器接口（向后兼容）
    'BioGenerator',        # 生物启发式生成器（显式名称）
    'GeneratorConfig',
    'SequencePredictorTM', # 序列预测核心
    'SequencePredictorConfig',
    'SparseWinnerTakeAll', # WTA竞争层
    'SDRVocabulary',       # SDR词汇映射
    'SDRVocabularyConfig',
    'SDRPrototypeInitializer',
    'ForwardModel',        # 前向质量模型
]
