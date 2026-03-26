#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失曲线可视化组件 - 实时显示训练损失变化
"""

import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LossChart:
    """损失曲线图表组件"""
    
    def __init__(self):
        self.loss_colors = {
            'total': '#FF0000',      # 红色 - 总损失
            'lm': '#4169E1',         # 宝蓝色 - 语言模型损失
            'sparsity': '#32CD32',   # 绿色 - 稀疏性损失
            'temporal': '#FFA500'    # 橙色 - 时序连续性损失
        }
        
        self.loss_names = {
            'total': '总损失',
            'lm': '语言模型损失',
            'sparsity': '稀疏性损失',
            'temporal': '时序连续性损失'
        }
    
    def plot_loss_curves(
        self,
        loss_history: List[Dict[str, float]],
        title: str = "训练损失曲线",
        show_all: bool = True
    ) -> go.Figure:
        """
        绘制多损失曲线图
        
        Args:
            loss_history: 损失历史记录列表
                [{'step': 1, 'epoch': 0, 'total': 2.5, 'lm': 2.3, 'sparsity': 0.1, 'temporal': 0.1}, ...]
            title: 图表标题
            show_all: 是否显示所有损失（否则只显示总损失）
            
        Returns:
            Plotly Figure 对象
        """
        if not loss_history or len(loss_history) == 0:
            return self._create_empty_figure()
        
        # 提取数据
        steps = [item.get('step', i) for i, item in enumerate(loss_history)]
        epochs = [item.get('epoch', 0) for item in loss_history]
        
        # 计算当前 epoch
        current_epoch = max(epochs) + 1 if epochs else 0
        total_steps = len(loss_history)
        
        fig = go.Figure()
        
        # 添加总损失曲线（必须显示）
        if 'total' in loss_history[0]:
            total_losses = [item.get('total') for item in loss_history]
            fig.add_trace(go.Scatter(
                x=steps,
                y=total_losses,
                mode='lines+markers',
                name=self.loss_names['total'],
                line=dict(color=self.loss_colors['total'], width=3),
                marker=dict(size=4),
                hovertemplate='Step: %{x}<br>Epoch: %{customdata}<br>总损失：%{y:.4f}<extra></extra>',
                customdata=epochs
            ))
        
        # 添加其他损失曲线
        if show_all:
            for loss_key in ['lm', 'sparsity', 'temporal']:
                if loss_key in loss_history[0]:
                    losses = [item.get(loss_key) for item in loss_history]
                    fig.add_trace(go.Scatter(
                        x=steps,
                        y=losses,
                        mode='lines',
                        name=self.loss_names.get(loss_key, loss_key),
                        line=dict(color=self.loss_colors.get(loss_key, '#999999'), width=2, dash='dash'),
                        opacity=0.7,
                        hovertemplate=f'Step: %{{x}}<br>{loss_key}: %{{y:.4f}}<extra></extra>'
                    ))
        
        # 计算统计信息
        if 'total' in loss_history[0]:
            initial_loss = total_losses[0] if total_losses else 0
            final_loss = total_losses[-1] if total_losses else 0
            min_loss = min(total_losses) if total_losses else 0
            improvement = ((initial_loss - final_loss) / max(initial_loss, 1e-6)) * 100
            
            subtitle = f"Epoch: {current_epoch} | 总步数：{total_steps} | 初始：{initial_loss:.4f} → 最终：{final_loss:.4f} | 改善：{improvement:.1f}%"
        else:
            subtitle = f"Epoch: {current_epoch} | 总步数：{total_steps}"
        
        # 更新布局
        fig.update_layout(
            title={
                'text': f"{title}<br><span style='font-size: 12px'>{subtitle}</span>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="训练步数 (Step)",
            yaxis_title="损失值 (Loss)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            width=800,
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        # 添加网格线样式
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE', type="log" if final_loss < 0.1 else "linear")
        
        return fig
    
    def plot_single_loss(
        self,
        loss_history: List[float],
        loss_name: str = "Loss",
        color: str = '#4169E1',
        title: str = "损失曲线"
    ) -> go.Figure:
        """绘制单个损失曲线"""
        
        steps = list(range(len(loss_history)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=loss_history,
            mode='lines',
            name=loss_name,
            line=dict(color=color, width=3),
            fill='tozeroy',
            fillcolor=color,
            opacity=0.3
        ))
        
        # 计算统计
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0
        min_loss = min(loss_history) if loss_history else 0
        max_loss = max(loss_history) if loss_history else 0
        
        fig.update_layout(
            title=f"{title}<br><span style='font-size: 12px'>平均：{avg_loss:.4f} | 最小：{min_loss:.4f} | 最大：{max_loss:.4f}</span>",
            xaxis_title="步数",
            yaxis_title=loss_name,
            width=700,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def plot_lr_schedule(
        self,
        lr_history: List[float],
        title: str = "学习率调度曲线"
    ) -> go.Figure:
        """绘制学习率变化曲线"""
        
        steps = list(range(len(lr_history)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=lr_history,
            mode='lines',
            name='Learning Rate',
            line=dict(color='#9370DB', width=3),
            fill='tozeroy',
            fillcolor='#9370DB',
            opacity=0.2
        ))
        
        # 找到关键节点
        max_lr = max(lr_history) if lr_history else 0
        min_lr = min(lr_history) if lr_history else 0
        
        fig.update_layout(
            title=f"{title}<br><span style='font-size: 12px'>最大 LR: {max_lr:.2e} | 最小 LR: {min_lr:.2e}</span>",
            xaxis_title="步数",
            yaxis_title="学习率",
            width=700,
            height=400,
            template='plotly_white',
            yaxis_type="log"
        )
        
        return fig
    
    def create_training_dashboard(
        self,
        loss_history: List[Dict[str, float]],
        metrics: Dict[str, Any],
        title: str = "训练监控面板"
    ) -> go.Figure:
        """
        创建综合训练监控面板
        
        Args:
            loss_history: 损失历史
            metrics: 其他指标（准确率、吞吐量等）
        """
        from plotly.subplots import make_subplots
        
        # 创建 2x2 子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('损失曲线', '损失分布', 'Epoch 进度', '关键指标'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "indicator"}, {"type": "table"}]]
        )
        
        # 1. 损失曲线（左上）
        if loss_history and 'total' in loss_history[0]:
            steps = [item['step'] for item in loss_history]
            total_losses = [item['total'] for item in loss_history]
            
            fig.add_trace(
                go.Scatter(x=steps, y=total_losses, mode='lines', name='总损失',
                          line=dict(color='#FF0000', width=2)),
                row=1, col=1
            )
        
        # 2. 损失分布（右上）
        if loss_history and 'total' in loss_history[0]:
            all_losses = [item['total'] for item in loss_history]
            fig.add_trace(
                go.Histogram(x=all_losses, nbinsx=30, name='损失分布',
                            marker_color='#4169E1'),
                row=1, col=2
            )
        
        # 3. Epoch 进度（左下）
        current_epoch = metrics.get('current_epoch', 0)
        total_epochs = metrics.get('total_epochs', 1)
        progress = current_epoch / max(total_epochs, 1) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+percent",
                value=progress / 100,
                title={'text': "训练进度"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#32CD32"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FFE4E1"},
                        {'range': [50, 80], 'color': "#FFFACD"},
                        {'range': [80, 100], 'color': "#F0FFF0"}
                    ],
                }
            ),
            row=2, col=1
        )
        
        # 4. 关键指标表格（右下）
        metric_rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in ['current_epoch', 'total_epochs']:
                    metric_rows.append([key, f"{value:.4f}" if isinstance(value, float) else str(value)])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['指标', '数值'], fill_color='#4169E1', font=dict(color='white')),
                cells=dict(values=[[row[0] for row in metric_rows[:5]], 
                                  [row[1] for row in metric_rows[:5]]]),
                name='指标'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _create_empty_figure(self) -> go.Figure:
        """创建空图表提示"""
        fig = go.Figure()
        fig.add_annotation(
            text="📊 暂无数据<br>开始训练后显示损失曲线",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#999999")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=600,
            height=400
        )
        return fig
    
    def update_chart_with_new_data(
        self,
        existing_fig: go.Figure,
        new_loss_data: Dict[str, float]
    ) -> go.Figure:
        """
        更新现有图表（用于实时刷新）
        
        Args:
            existing_fig: 现有的图表对象
            new_loss_data: 新的损失数据点
            
        Returns:
            更新后的图表
        """
        # 获取最后一个点的数据
        if existing_fig.data and len(existing_fig.data) > 0:
            first_trace = existing_fig.data[0]
            if hasattr(first_trace, 'x') and len(first_trace.x) > 0:
                last_step = int(first_trace.x[-1])
                new_step = last_step + 1
                
                # 为每个 trace 添加新数据点
                for i, trace in enumerate(existing_fig.data):
                    loss_key = list(new_loss_data.keys())[i % len(new_loss_data)]
                    trace.x.append(new_step)
                    trace.y.append(new_loss_data.get(loss_key, 0))
        
        return existing_fig
