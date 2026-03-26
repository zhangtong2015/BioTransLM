# utils/common_utils.py - 通用工具函数
import torch
import torch.nn.functional as F
import logging
from typing import Optional, Any, Dict, List
import numpy as np


class CommonUtils:
    """通用工具类"""
    
    _logger = None
    
    @staticmethod
    def setup_logging(level: str = 'INFO') -> None:
        """设置日志配置"""
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=getattr(logging, level.upper()),
            force=True
        )
        CommonUtils._logger = logging.getLogger('BioTransLM')
    
    @staticmethod
    def get_logger(name: str = None) -> logging.Logger:
        """获取日志器"""
        if CommonUtils._logger is None:
            CommonUtils.setup_logging()
        return logging.getLogger(name or 'BioTransLM')
    
    @staticmethod
    def normalize_tensor(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """归一化张量"""
        norm = torch.norm(tensor, dim=dim, keepdim=True)
        return tensor / (norm + eps)
    
    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """计算余弦相似度"""
        a_norm = CommonUtils.normalize_tensor(a, dim=dim)
        b_norm = CommonUtils.normalize_tensor(b, dim=dim)
        return (a_norm * b_norm).sum(dim=dim)
    
    @staticmethod
    def sparse_softmax(x: torch.Tensor, dim: int = -1, sparsity_mask: float = 0.0) -> torch.Tensor:
        """稀疏Softmax，保留一定的稀疏性"""
        if sparsity_mask > 0:
            threshold = torch.quantile(x, sparsity_mask, dim=dim, keepdim=True)
            x = torch.where(x < threshold, torch.tensor(-1e9, device=x.device), x)
        return F.softmax(x, dim=dim)
    
    @staticmethod
    def apply_sparsity_mask(x: torch.Tensor, sparsity_level: float = 0.9) -> torch.Tensor:
        """应用稀疏掩码，保留top k%的值"""
        if sparsity_level <= 0:
            return x
        k = max(1, int(x.shape[-1] * (1 - sparsity_level)))
        topk, _ = torch.topk(x, k, dim=-1)
        threshold = topk[..., -1, None]
        return torch.where(x < threshold, torch.tensor(0.0, device=x.device), x)
    
    @staticmethod
    def safe_divide(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """安全除法，避免除以零"""
        return a / (b + eps)
    
    @staticmethod
    def moving_average(x: torch.Tensor, window_size: int, dim: int = 1) -> torch.Tensor:
        """计算滑动窗口平均"""
        if window_size <= 1:
            return x
        kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (B, D, L)
            x = F.pad(x, (window_size - 1, 0))
            x = F.conv1d(x, kernel, groups=x.shape[1])
            x = x.transpose(1, 2)  # (B, L, D)
        return x
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """将PyTorch张量转换为NumPy数组"""
        return tensor.detach().cpu().numpy()
    
    @staticmethod
    def batch_index_select(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """批量索引选择"""
        batch_size = x.shape[0]
        batch_indices = torch.arange(batch_size, device=x.device)[:, None]
        return x[batch_indices, indices]
    
    @staticmethod
    def ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """确保张量在正确的设备上"""
        if tensor.device != device:
            return tensor.to(device)
        return tensor
    
    @staticmethod
    def move_to_device(obj: Any, device: torch.device) -> Any:
        """将对象（张量或张量容器）移动到指定设备"""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(CommonUtils.move_to_device(x, device) for x in obj)
        elif isinstance(obj, dict):
            return {k: CommonUtils.move_to_device(v, device) for k, v in obj.items()}
        return obj


# 全局工具实例
utils = CommonUtils()
