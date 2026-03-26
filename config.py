# config.py - 核心配置定义
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class BaseConfig:
    """所有模块配置的基类"""
    hidden_dim: int = 768
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    dtype: str = 'float32'
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
