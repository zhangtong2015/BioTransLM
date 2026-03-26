#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HTM 可视化组件 - 空间池化器和时序记忆的激活可视化
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import torch
import logging

logger = logging.getLogger(__name__)


class HTMVisualizer:
    """HTM 系统可视化工具"""
    
    def __init__(self):
        self.colorscale_active = [
            [0.0, '#00008B'],      # 深蓝色 - 完全抑制
            [0.3, '#4169E1'],      # 宝蓝色 - 低激活
            [0.6, '#FFD700'],      # 金黄色 - 中等激活
            [0.8, '#FF6347'],      # 番茄红 - 高激活
            [1.0, '#FF0000']       # 红色 - 完全激活
        ]
        
        self.colorscale_temporal = [
            [0.0, '#FFFFFF'],      # 白色 - 无活动
            [0.2, '#E0FFFF'],      # 淡青色 - 预测
            [0.5, '#00CED1'],      # 暗青色 - 正确预测
            [0.8, '#FF1493'],      # 深粉红 - burst
            [1.0, '#8B0000']       # 深红色 - 高强度 burst
        ]
    
    # ========== 空间池化器可视化 ==========
    
    def plot_column_activations(
        self, 
        active_columns: torch.Tensor,
        title: str = "空间池化器列激活热图",
        batch_idx: int = 0,
        seq_idx: int = 0
    ) -> go.Figure:
        """
        绘制空间池化器列激活热图
        
        Args:
            active_columns: 列激活张量 [batch_size, seq_len, n_columns] 或 [n_columns]
            title: 图表标题
            batch_idx: batch 索引（当 batch_size > 1 时）
            seq_idx: 序列位置索引（当 seq_len > 1 时）
            
        Returns:
            Plotly Figure 对象
        """
        # 处理输入
        if isinstance(active_columns, torch.Tensor):
            data = active_columns.detach().cpu()
        else:
            data = torch.tensor(active_columns)
        
        # 提取指定位置的激活
        if data.dim() == 1:
            activations = data.numpy()
        elif data.dim() == 2:
            activations = data[batch_idx].numpy() if batch_idx < data.shape[0] else data[0].numpy()
        elif data.dim() >= 3:
            # 确保索引在范围内
            b_idx = min(batch_idx, data.shape[0] - 1)
            s_idx = min(seq_idx, data.shape[1] - 1)
            activations = data[b_idx, s_idx].numpy()
        else:
            logger.warning(f"意外的激活维度：{data.dim()}")
            return self._create_error_figure("不支持的数据维度")
        
        n_columns = len(activations)
        
        # 计算统计信息
        n_active = int(np.sum(activations > 0))
        activation_rate = n_active / n_columns * 100
        max_activation = float(np.max(activations))
        mean_activation = float(np.mean(activations[activations > 0])) if n_active > 0 else 0
        
        # 创建热力图数据
        # 将列重新排列为 2D 网格以便可视化
        grid_size = int(np.ceil(np.sqrt(n_columns)))
        grid = np.zeros((grid_size, grid_size))
        
        # 填充网格（按行优先）
        for i in range(min(n_columns, grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            grid[row, col] = activations[i]
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=grid,
            colorscale='Reds',
            showscale=True,
            hovertemplate='列索引：%{x},%{y}<br>激活强度：%{z:.4f}<extra></extra>',
            colorbar=dict(
                title="激活强度",
                thickness=15,
                len=0.8
            )
        ))
        
        # 更新布局
        fig.update_layout(
            title={
                'text': f"{title}<br><span style='font-size: 12px'>激活率：{activation_rate:.2f}% | 激活列数：{n_active}/{n_columns} | 最大激活：{max_activation:.4f}</span>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title="列索引（简化）",
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                title="行索引（简化）",
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1,
                zeroline=False
            ),
            width=600,
            height=500,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='white'
        )
        
        return fig
    
    def plot_activation_distribution(
        self,
        active_columns: torch.Tensor,
        title: str = "列激活分布直方图"
    ) -> go.Figure:
        """绘制激活分布直方图"""
        
        if isinstance(active_columns, torch.Tensor):
            data = active_columns.detach().cpu().numpy().flatten()
        else:
            data = np.array(active_columns).flatten()
        
        # 计算分位数
        percentiles = {
            'P50': float(np.percentile(data, 50)),
            'P90': float(np.percentile(data, 90)),
            'P95': float(np.percentile(data, 95)),
            'P99': float(np.percentile(data, 99))
        }
        
        # 创建直方图
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=100,
            marker_color='#4169E1',
            opacity=0.7,
            hovertemplate='激活范围：%{x:.4f}<br>频数：%{y}<extra></extra>'
        ))
        
        # 添加阈值线
        threshold = np.percentile(data, 98)
        fig.add_shape(
            type="line",
            x0=threshold,
            y0=0,
            x1=threshold,
            y1=max(np.histogram(data, bins=100)[0]),
            line=dict(color="red", width=2, dash="dash"),
            name="Top 2% 阈值"
        )
        
        fig.update_layout(
            title=f"{title}<br><span style='font-size: 12px'>中位数：{percentiles['P50']:.4f} | P90: {percentiles['P90']:.4f} | P99: {percentiles['P99']:.4f}</span>",
            xaxis_title="激活强度",
            yaxis_title="频数",
            showlegend=False,
            width=700,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    # ========== 时序记忆可视化 ==========
    
    def plot_temporal_metrics(
        self,
        temporal_output: Dict[str, Any],
        title: str = "时序记忆指标"
    ) -> go.Figure:
        """
        绘制时序记忆关键指标
        
        Args:
            temporal_output: 时序记忆输出字典
                - prediction_accuracy: 预测准确率
                - n_bursts: burst 列数
                - total_columns: 总列数
                - active_cells: 活跃细胞数
        """
        # 提取指标
        pred_acc = temporal_output.get('prediction_accuracy', 0.0)
        n_bursts = temporal_output.get('n_bursts', 0)
        total_cols = temporal_output.get('total_columns', 4096)
        burst_rate = n_bursts / max(1, total_cols)
        
        # 创建指标卡片风格的图表
        fig = go.Figure()
        
        # 添加预测准确率条
        fig.add_trace(go.Bar(
            name='预测准确率',
            x=['预测准确率'],
            y=[pred_acc * 100],
            marker_color='#00CED1',
            text=[f'{pred_acc*100:.1f}%'],
            textposition='outside',
            hovertemplate='准确率：%{y:.1f}%<extra></extra>'
        ))
        
        # 添加 burst 率条
        fig.add_trace(go.Bar(
            name='Burst 率',
            x=['Burst 率'],
            y=[burst_rate * 100],
            marker_color='#FF1493' if burst_rate > 0.1 else '#FFA500',
            text=[f'{burst_rate*100:.1f}%'],
            textposition='outside',
            hovertemplate=f'Burst 率：%{{y:.1f}}%<br>Burst 列数：{n_bursts}<extra></extra>'
        ))
        
        # 添加柱状对比
        if 'correct_predictions' in temporal_output and 'total_predictions' in temporal_output:
            correct = temporal_output['correct_predictions']
            total = temporal_output['total_predictions']
            incorrect = total - correct
            
            fig.add_trace(go.Bar(
                name='预测结果',
                x=['正确预测', '错误/未预测'],
                y=[correct, incorrect],
                marker_color=['#32CD32', '#DC143C'],
                text=[f'{correct}', f'{incorrect}'],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=title,
            barmode='group',
            yaxis_title="百分比 (%)",
            showlegend=False,
            width=600,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def plot_cell_activations(
        self,
        active_cells: torch.Tensor,
        n_columns: int = 4096,
        n_cells_per_col: int = 16,
        title: str = "时序记忆细胞激活模式"
    ) -> go.Figure:
        """
        绘制细胞级别的激活模式
        
        Args:
            active_cells: 细胞激活 [batch, seq_len, n_columns, n_cells_per_col]
            n_columns: 列数
            n_cells_per_col: 每列细胞数
        """
        if isinstance(active_cells, torch.Tensor):
            data = active_cells.detach().cpu()
        else:
            data = torch.tensor(active_cells)
        
        # 取第一个样本的第一个时间步
        if data.dim() >= 4:
            cells_2d = data[0, 0].numpy()  # [n_columns, n_cells_per_col]
        elif data.dim() == 3:
            cells_2d = data[0].numpy()
        else:
            return self._create_error_figure("细胞激活数据维度不正确")
        
        # 限制显示的列数（太多会太密集）
        max_display_cols = min(100, n_columns)
        cells_2d = cells_2d[:max_display_cols, :]
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=cells_2d.T,  # 转置使得细胞在 y 轴
            colorscale='Blues',
            showscale=True,
            hovertemplate='列：%{x}<br>细胞：%{y}<br>激活：%{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{title} (显示前{max_display_cols}列)",
            xaxis_title="列索引",
            yaxis_title="细胞索引",
            width=700,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    # ========== 多时间步动画 ==========
    
    def create_activation_animation(
        self,
        active_columns_seq: torch.Tensor,
        title: str = "列激活动态变化"
    ) -> go.Figure:
        """
        创建多个时间步的激活变化动画
        
        Args:
            active_columns_seq: 序列激活 [seq_len, n_columns] 或 [batch, seq_len, n_columns]
        """
        if isinstance(active_columns_seq, torch.Tensor):
            data = active_columns_seq.detach().cpu()
        else:
            data = torch.tensor(active_columns_seq)
        
        # 确保是 2D
        if data.dim() == 3:
            data = data[0]  # 取第一个 batch
        
        seq_len = data.shape[0]
        n_columns = data.shape[1]
        
        # 创建帧
        frames = []
        grid_size = int(np.ceil(np.sqrt(n_columns)))
        
        for t in range(seq_len):
            activations = data[t].numpy()
            grid = np.zeros((grid_size, grid_size))
            
            for i in range(min(n_columns, grid_size * grid_size)):
                row = i // grid_size
                col = i % grid_size
                grid[row, col] = activations[i]
            
            frame = go.Frame(
                data=[go.Heatmap(z=grid, colorscale='Reds', showscale=False)],
                name=str(t),
                labels={'name': f'Time step {t}'}
            )
            frames.append(frame)
        
        # 创建初始图
        fig = go.Figure(
            data=[go.Heatmap(
                z=np.zeros((grid_size, grid_size)),
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="激活强度", thickness=15)
            )],
            frames=frames
        )
        
        # 添加播放控件
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'direction': 'left',
                'pad': {'r': 10, 't': 20},
                'buttons': [
                    {
                        'label': '▶ 播放',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 200, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': '⏸ 暂停',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 14},
                    'prefix': '时间步：',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [{
                    'args': [[str(t)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(t),
                    'method': 'animate'
                } for t in range(seq_len)]
            }],
            title=title,
            width=600,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    # ========== 辅助方法 ==========
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """创建错误提示图"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=400,
            height=300
        )
        return fig
    
    def extract_htm_metrics(self, htm_output: Dict[str, Any]) -> Dict[str, Any]:
        """从 HTM 输出中提取可量化的指标"""
        metrics = {}
        
        # 空间池化器指标
        if 'spatial_output' in htm_output:
            spatial = htm_output['spatial_output']
            if 'active_columns' in spatial:
                act_cols = spatial['active_columns']
                if isinstance(act_cols, torch.Tensor):
                    metrics['sparsity'] = float((act_cols > 0).float().mean())
                    metrics['n_active_columns'] = int((act_cols > 0).sum().item())
        
        # 时序记忆指标
        if 'temporal_output' in htm_output:
            temporal = htm_output['temporal_output']
            metrics['prediction_accuracy'] = float(temporal.get('prediction_accuracy', 0.0))
            metrics['n_bursts'] = int(temporal.get('n_bursts', 0))
            metrics['total_columns'] = int(temporal.get('total_columns', 4096))
            metrics['burst_rate'] = metrics['n_bursts'] / max(1, metrics['total_columns'])
        
        return metrics
