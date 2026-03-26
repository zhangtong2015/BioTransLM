# core/base_module.py - 所有模块的基类
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

class BaseModule(nn.Module, ABC):
    """所有神经模块的抽象基类"""
    
    def __init__(
        self, 
        config: Optional[Any] = None,
        module_name: str = "base_module"
    ):
        super().__init__()
        self.config = config
        self.module_name = module_name
        
        # 自动设备管理
        self._device = self._get_device()
        
        # 初始化模块
        self._initialize_module()
        
        # 移动到设备
        self.to(self._device)
    
    def _get_device(self) -> torch.device:
        """获取设备"""
        if self.config and hasattr(self.config, 'device'):
            device_str = self.config.device
        else:
            device_str = 'auto'
        
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)
    
    @abstractmethod
    def _initialize_module(self) -> None:
        """初始化模块特定组件 - 由子类实现"""
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """前向传播 - 由子类实现
        
        Returns:
            Dict: 必须包含模块输出，键名按契约定义
        """
        pass
    
    def get_device(self) -> torch.device:
        """获取当前设备"""
        return self._device
    
    def ensure_input_device(self, *args, **kwargs):
        """
        将所有输入张量移动到模块所在设备
        
        Returns:
            tuple: (args, kwargs) 所有张量已移动到正确设备
        """
        from utils.common_utils import utils
        device = self._device
        
        args = tuple(utils.move_to_device(arg, device) for arg in args)
        kwargs = {k: utils.move_to_device(v, device) for k, v in kwargs.items()}
        
        return args, kwargs
    
    def save_state(self, path: str) -> None:
        """保存模块状态"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'module_name': self.module_name
        }, path)
    
    @classmethod
    def load_state(cls, path: str, device: Optional[str] = None) -> 'BaseModule':
        """加载模块状态"""
        # PyTorch 2.6+ weights_only兼容处理
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            # 旧版本PyTorch不支持weights_only参数
            checkpoint = torch.load(path, map_location=device)
        
        instance = cls(config=checkpoint['config'], 
                       module_name=checkpoint['module_name'])
        instance.load_state_dict(checkpoint['state_dict'])
        return instance
