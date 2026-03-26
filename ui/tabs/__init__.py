# Tabs 包初始化
from .chat_tab import create_chat_tab
from .training_tab import create_training_tab
from .dataset_tab import create_dataset_tab
from .inspection_tab import create_inspection_tab
from .model_tab import create_model_tab

__all__ = [
    'create_chat_tab',
    'create_training_tab',
    'create_dataset_tab',
    'create_inspection_tab',
    'create_model_tab'
]
