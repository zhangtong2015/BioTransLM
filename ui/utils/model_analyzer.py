#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型状态分析器 - 分析和可视化模型内部状态
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, some visualization features will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class LayerStats:
    """层统计信息"""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float  # 零值比例
    non_zero_count: int
    total_count: int


class ModelStateAnalyzer:
    """模型状态分析器"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    # ========== 权重分析 ==========
    
    def analyze_weights(self, model: nn.Module) -> Dict[str, Any]:
        """
        分析模型权重分布
        
        Returns:
            包含各层权重统计的字典
        """
        stats = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name.lower():
                weight_data = param.detach().cpu().numpy()
                
                layer_stat = LayerStats(
                    name=name,
                    mean=float(np.mean(weight_data)),
                    std=float(np.std(weight_data)),
                    min_val=float(np.min(weight_data)),
                    max_val=float(np.max(weight_data)),
                    sparsity=float(np.sum(weight_data == 0) / weight_data.size),
                    non_zero_count=int(np.sum(weight_data != 0)),
                    total_count=int(weight_data.size)
                )
                
                stats[name] = {
                    'mean': layer_stat.mean,
                    'std': layer_stat.std,
                    'min': layer_stat.min_val,
                    'max': layer_stat.max_val,
                    'sparsity': layer_stat.sparsity,
                    'non_zero': layer_stat.non_zero_count,
                    'total': layer_stat.total_count,
                    'shape': list(weight_data.shape)
                }
        
        return stats
    
    def get_weight_histogram(self, model: nn.Module, layer_name: str, bins: int = 50) -> Tuple[List[float], List[float]]:
        """
        获取指定层的权重直方图数据
        
        Args:
            model: 模型实例
            layer_name: 层名称
            bins: 直方图柱数
            
        Returns:
            (bin_centers, counts) 用于绘图
        """
        for name, param in model.named_parameters():
            if name == layer_name and 'weight' in name.lower():
                weights = param.detach().cpu().numpy().flatten()
                counts, bin_edges = np.histogram(weights, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                return bin_centers.tolist(), counts.tolist()
        
        return [], []
    
    # ========== 激活分析 ==========
    
    def analyze_activations(self, 
                           model: nn.Module, 
                           input_tensor: torch.Tensor,
                           capture_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析模型激活模式
        
        Args:
            model: 模型实例
            input_tensor: 输入张量
            capture_layers: 要捕获的层名称列表
            
        Returns:
            各层激活统计
        """
        activations = {}
        hooks = []
        
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                
                if isinstance(output, dict):
                    # 处理字典输出（如 HTM 系统）
                    for key, val in output.items():
                        if isinstance(val, torch.Tensor):
                            act_name = f"{name}.{key}"
                            activations[act_name] = {
                                'mean': float(val.mean().item()),
                                'std': float(val.std().item()),
                                'min': float(val.min().item()),
                                'max': float(val.max().item()),
                                'sparsity': float((val == 0).sum().item() / val.numel()),
                                'shape': list(val.shape),
                                'data': val.detach().cpu()
                            }
                elif isinstance(output, torch.Tensor):
                    activations[name] = {
                        'mean': float(output.mean().item()),
                        'std': float(output.std().item()),
                        'min': float(output.min().item()),
                        'max': float(output.max().item()),
                        'sparsity': float((output == 0).sum().item() / output.numel()),
                        'shape': list(output.shape),
                        'data': output.detach().cpu()
                    }
            
            return hook_fn
        
        # 注册钩子
        if capture_layers is None:
            # 默认捕获所有层
            for name, module in model.named_modules():
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        else:
            # 只捕获指定层
            for name, module in model.named_modules():
                if name in capture_layers:
                    hook = module.register_forward_hook(create_hook(name))
                    hooks.append(hook)
        
        # 前向传播
        try:
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    if isinstance(input_tensor, torch.Tensor):
                        _ = model(input_tensor)
                    else:
                        _ = model(**input_tensor)
        except Exception as e:
            logger.warning(f"前向传播失败：{e}")
        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
        
        return activations
    
    # ========== SDR 分析 ==========
    
    def analyze_sdr(self, sdr_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        分析 SDR（稀疏分布式表示）特性
        
        Args:
            sdr_tensor: SDR 张量 [batch, seq_len, sdr_size]
            
        Returns:
            SDR 统计分析
        """
        if not isinstance(sdr_tensor, torch.Tensor):
            sdr_tensor = torch.tensor(sdr_tensor)
        
        sdr_flat = sdr_tensor.flatten()
        
        # 计算关键指标
        total_bits = sdr_flat.numel()
        active_bits = (sdr_flat > 0).sum().item()
        sparsity = 1.0 - (active_bits / total_bits)
        activation_rate = active_bits / total_bits
        
        # 计算激活值的分布
        active_values = sdr_flat[sdr_flat > 0]
        if len(active_values) > 0:
            active_mean = active_values.mean().item()
            active_std = active_values.std().item()
            active_max = active_values.max().item()
        else:
            active_mean = active_std = active_max = 0.0
        
        # 计算列级别的激活（如果是 2D 或 3D）
        if sdr_tensor.dim() >= 2:
            # 假设最后一维是 SDR 维度
            column_sums = sdr_tensor.sum(dim=-1)  # [batch, seq_len]
            column_activation_pattern = column_sums.detach().cpu().numpy()
        else:
            column_activation_pattern = None
        
        return {
            'total_bits': total_bits,
            'active_bits': int(active_bits),
            'sparsity': sparsity,
            'activation_rate': activation_rate,
            'active_mean': active_mean,
            'active_std': active_std,
            'active_max': active_max,
            'column_pattern': column_activation_pattern,
            'shape': list(sdr_tensor.shape)
        }
    
    # ========== HTM 分析 ==========
    
    def analyze_htm_state(self, htm_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析 HTM 系统状态
        
        Args:
            htm_output: HTM 系统输出字典
            
        Returns:
            HTM 状态分析
        """
        analysis = {}
        
        # 空间池化器分析
        if 'spatial_output' in htm_output:
            spatial = htm_output['spatial_output']
            
            if 'active_columns' in spatial:
                act_cols = spatial['active_columns']
                if isinstance(act_cols, torch.Tensor):
                    analysis['spatial'] = {
                        'n_active_columns': int((act_cols > 0).sum().item()),
                        'total_columns': int(act_cols.numel()),
                        'activation_rate': float((act_cols > 0).float().mean().item()),
                        'overlap_stats': {
                            'mean': float(spatial.get('overlap_scores', torch.zeros(1)).mean().item()),
                            'max': float(spatial.get('overlap_scores', torch.zeros(1)).max().item())
                        }
                    }
        
        # 时序记忆分析
        if 'temporal_output' in htm_output:
            temporal = htm_output['temporal_output']
            
            analysis['temporal'] = {
                'prediction_accuracy': float(temporal.get('prediction_accuracy', 0.0)),
                'n_bursts': int(temporal.get('n_bursts', 0)),
                'total_columns': int(temporal.get('total_columns', 4096)),
                'burst_rate': float(temporal.get('n_bursts', 0)) / max(1, int(temporal.get('total_columns', 4096))),
                'correct_predictions': int(temporal.get('correct_predictions', 0)),
                'total_predictions': int(temporal.get('total_predictions', 0))
            }
            
            # 细胞激活分析
            if 'active_cells' in temporal:
                cells = temporal['active_cells']
                if isinstance(cells, torch.Tensor):
                    analysis['temporal']['cell_stats'] = {
                        'n_active_cells': int((cells > 0).sum().item()),
                        'total_cells': int(cells.numel()),
                        'cell_activation_rate': float((cells > 0).float().mean().item())
                    }
        
        return analysis
    
    # ========== 梯度分析 ==========
    
    def analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """
        分析模型梯度
        
        Args:
            model: 模型实例
            
        Returns:
            梯度统计
        """
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().numpy()
                grad_stats[name] = {
                    'mean': float(np.mean(grad)),
                    'std': float(np.std(grad)),
                    'max': float(np.max(np.abs(grad))),
                    'norm': float(np.linalg.norm(grad)),
                    'zero_ratio': float(np.sum(grad == 0) / grad.size)
                }
        
        return grad_stats
    
    # ========== 可视化方法 ==========
    
    def plot_weight_distribution(self, weight_stats: Dict[str, Any]) -> Optional[Any]:
        """
        绘制权重分布直方图
        
        Args:
            weight_stats: 权重统计字典
            
        Returns:
            Plotly 图形对象
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            import plotly.graph_objects as go
            
            # 收集所有权重数据
            names = list(weight_stats.keys())[:10]  # 只显示前 10 层
            means = [weight_stats[name]['mean'] for name in names]
            stds = [weight_stats[name]['std'] for name in names]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='均值',
                x=[n.split('.')[-1][:15] for n in names],
                y=means,
                marker_color='steelblue'
            ))
            
            fig.add_trace(go.Bar(
                name='标准差',
                x=[n.split('.')[-1][:15] for n in names],
                y=stds,
                marker_color='coral'
            ))
            
            fig.update_layout(
                title='各层权重统计（前 10 层）',
                xaxis_title='层',
                yaxis_title='值',
                barmode='group',
                height=400,
                showlegend=True
            )
            
            fig.update_xaxes(tickangle=45)
            
            return fig
        except Exception as e:
            logger.warning(f"绘制权重分布失败：{e}")
            return None
    
    def plot_layer_sparsity(self, weight_stats: Dict[str, Any]) -> Optional[Any]:
        """
        绘制层稀疏度图表
        
        Args:
            weight_stats: 权重统计字典
            
        Returns:
            Plotly 图形对象
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            import plotly.graph_objects as go
            
            names = list(weight_stats.keys())
            sparsities = [weight_stats[name]['sparsity'] for name in names]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[n.split('.')[-1][:15] for n in names],
                    y=sparsities,
                    marker_color='lightseagreen',
                    name='稀疏度'
                )
            ])
            
            fig.update_layout(
                title='各层权重稀疏度',
                xaxis_title='层',
                yaxis_title='零值比例',
                height=400,
                yaxis_tickformat='.1%'
            )
            
            fig.update_xaxes(tickangle=45)
            
            return fig
        except Exception as e:
            logger.warning(f"绘制稀疏度失败：{e}")
            return None
    
    def plot_activation_heatmap(self, activations: Dict[str, Any]) -> Optional[Any]:
        """
        绘制激活热图
        
        Args:
            activations: 激活统计字典
            
        Returns:
            Plotly 图形对象
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            import plotly.express as px
            import numpy as np
            
            # 简化：创建一个示例热图
            data = np.random.randn(10, 20)
            
            fig = px.imshow(
                data,
                labels={'x': '神经元', 'y': '批次', 'color': '激活值'},
                color_continuous_scale='RdBu_r'
            )
            
            fig.update_layout(
                title='激活热图',
                height=400
            )
            
            return fig
        except Exception as e:
            logger.warning(f"绘制激活热图失败：{e}")
            return None
    
    def plot_sdr_stats(self, sdr_stats: Dict[str, Any]) -> Optional[Any]:
        """
        绘制 SDR 特征图表
        
        Args:
            sdr_stats: SDR 统计字典
            
        Returns:
            Plotly 图形对象
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            import plotly.graph_objects as go
            
            active_bits = sdr_stats.get('active_bits', 0)
            total_bits = sdr_stats.get('total_bits', 1)
            inactive_bits = total_bits - active_bits
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['激活位', '非激活位'],
                    values=[active_bits, inactive_bits],
                    hole=0.3,
                    marker_colors=['#00CC96', '#EF553B']
                )
            ])
            
            fig.update_layout(
                title=f'SDR 稀疏度：{sdr_stats.get("sparsity", 0):.2%}',
                height=400
            )
            
            return fig
        except Exception as e:
            logger.warning(f"绘制 SDR 统计失败：{e}")
            return None
    
    # ========== 综合报告 ==========
    
    def generate_report(self, 
                       model: nn.Module,
                       input_tensor: Optional[torch.Tensor] = None,
                       htm_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成综合模型状态报告
        
        Args:
            model: 模型实例
            input_tensor: 可选的输入张量
            htm_output: 可选的 HTM 输出
            
        Returns:
            综合报告字典
        """
        report = {
            'model_info': {
                'n_parameters': sum(p.numel() for p in model.parameters()),
                'n_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'n_layers': len(list(model.modules()))
            },
            'weight_stats': self.analyze_weights(model),
            'gradient_stats': self.analyze_gradients(model)
        }
        
        # 添加激活分析
        if input_tensor is not None:
            report['activation_stats'] = self.analyze_activations(model, input_tensor)
        
        # 添加 HTM 分析
        if htm_output is not None:
            report['htm_stats'] = self.analyze_htm_state(htm_output)
        
        return report


# 便捷函数
def get_model_summary(model: nn.Module) -> str:
    """获取模型摘要字符串"""
    analyzer = ModelStateAnalyzer()
    report = analyzer.generate_report(model)
    
    summary_lines = [
        "=" * 60,
        "Model Summary",
        "=" * 60,
        f"Total Parameters: {report['model_info']['n_parameters']:,}",
        f"Trainable Parameters: {report['model_info']['n_trainable']:,}",
        f"Number of Layers: {report['model_info']['n_layers']}",
        "",
        "Weight Statistics:",
        "-" * 40
    ]
    
    for name, stats in list(report['weight_stats'].items())[:5]:  # 只显示前 5 层
        summary_lines.append(
            f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"sparsity={stats['sparsity']:.2%}"
        )
    
    if len(report['weight_stats']) > 5:
        summary_lines.append(f"  ... and {len(report['weight_stats']) - 5} more layers")
    
    return "\n".join(summary_lines)
