#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理过程追踪组件 - 双系统推理可视化
"""

import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ReasoningTrace:
    """推理过程追踪器"""
    
    def visualize_system1_results(
        self,
        system1_output: Dict[str, Any],
        title: str = "系统 1 - 直觉联想检索"
    ) -> go.Figure:
        """可视化系统 1 检索结果"""
        
        if not system1_output:
            return self._create_empty_figure("无系统 1 输出")
        
        # 提取相似度和结果
        similarities = system1_output.get('similarities', [])
        retrieved = system1_output.get('retrieved_results', [])
        confidence = system1_output.get('confidence', 0.0)
        
        # 创建条形图
        fig = go.Figure()
        
        if similarities and len(similarities) > 0:
            indices = list(range(len(similarities[:10])))  # Top 10
            
            fig.add_trace(go.Bar(
                x=indices,
                y=similarities[:10],
                marker=dict(
                    color=similarities[:10],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"{s:.3f}" for s in similarities[:10]],
                textposition='outside',
                hovertemplate='排名：%{x}+1<br>相似度：%{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{title}<br><span style='font-size: 12px'>置信度：{confidence:.3f}</span>",
            xaxis_title="检索结果排名",
            yaxis_title="相似度分数",
            width=600,
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def visualize_system2_chain(
        self,
        system2_output: Dict[str, Any],
        title: str = "系统 2 - 符号推理链"
    ) -> go.Figure:
        """可视化系统 2 推理链"""
        
        if not system2_output:
            return self._create_empty_figure("无系统 2 输出")
        
        # 提取推理步骤
        reasoning_chain = system2_output.get('reasoning_chain', [])
        facts_used = system2_output.get('facts_used', [])
        answer = system2_output.get('answer', '')
        confidence = system2_output.get('confidence', 0.0)
        
        # 创建步骤流程图
        steps = []
        step_labels = []
        
        # 添加初始问题
        steps.append(("start", "问题输入", "#4169E1"))
        
        # 添加事实收集步骤
        if facts_used:
            steps.append(("facts", f"收集 {len(facts_used)} 个事实", "#32CD32"))
        
        # 添加推理步骤
        for i, step in enumerate(reasoning_chain[:5]):  # 最多 5 步
            step_type = step.get('type', 'inference')
            step_label = step.get('description', f'步骤{i+1}')
            color = '#FFA500' if step_type == 'inference' else '#9370DB'
            steps.append((f"step{i}", step_label, color))
        
        # 添加结论
        if answer:
            steps.append(("conclusion", "得出结论", "#FF6347"))
        
        # 创建流程图
        fig = go.Figure()
        
        # 添加节点
        for i, (step_id, label, color) in enumerate(steps):
            fig.add_shape(
                type="rect",
                x0=i * 1.2, y0=0.4, x1=i * 1.2 + 1.0, y1=0.6,
                fillcolor=color,
                opacity=0.7,
                line=dict(width=2, color="DarkSlateGrey"),
                xref="x", yref="y"
            )
            
            fig.add_annotation(
                x=i * 1.2 + 0.5, y=0.5,
                text=label,
                showarrow=False,
                font=dict(size=10, color="white"),
                xref="x", yref="y"
            )
            
            # 添加箭头（除了最后一个）
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=i * 1.2 + 1.05, y=0.5,
                    text="→",
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    xref="x", yref="y"
                )
        
        fig.update_layout(
            title=f"{title}<br><span style='font-size: 12px'>推理置信度：{confidence:.3f}</span>",
            xaxis=dict(showgrid=False, zeroline=False, range=[-0.5, len(steps) * 1.2]),
            yaxis=dict(showgrid=False, zeroline=False, range=[0, 1]),
            width=700,
            height=300,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def visualize_fusion_weights(
        self,
        fusion_output: Dict[str, Any],
        title: str = "双系统融合决策"
    ) -> go.Figure:
        """可视化双系统融合权重"""
        
        if not fusion_output:
            return self._create_empty_figure("无融合输出")
        
        # 提取权重
        weights = fusion_output.get('system_weights', [0.5, 0.5])
        s1_weight = weights[0] if len(weights) > 0 else 0.5
        s2_weight = weights[1] if len(weights) > 1 else 0.5
        
        # 冲突检测结果
        conflict_info = fusion_output.get('conflict_detected', False)
        conflict_severity = fusion_output.get('conflict_severity', 0.0)
        
        # 创建饼图
        fig = go.Figure(data=[go.Pie(
            labels=['系统 1 (直觉)', '系统 2 (推理)'],
            values=[s1_weight, s2_weight],
            hole=.3,
            marker_colors=['#4169E1', '#FF6347'],
            textinfo='label+percent',
            hovertemplate='%{label}<br>权重：%{percent:.1%}<extra></extra>'
        )])
        
        # 添加冲突标注
        if conflict_info:
            fig.add_annotation(
                text=f"⚠️ 检测到冲突<br>严重度：{conflict_severity:.2f}",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color="red"),
                bgcolor="yellow",
                bordercolor="red"
            )
        
        fig.update_layout(
            title=f"{title}<br><span style='font-size: 12px'>S1 权重：{s1_weight:.2f} | S2 权重：{s2_weight:.2f}</span>",
            width=500,
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_reasoning_summary(
        self,
        system1_output: Dict[str, Any],
        system2_output: Dict[str, Any],
        fusion_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成推理过程摘要（用于 JSON 显示）"""
        
        summary = {
            '系统 1 (直觉)': {},
            '系统 2 (推理)': {},
            '融合决策': {}
        }
        
        # 系统 1 摘要
        summary['系统 1 (直觉)'] = {
            '检索结果数': len(system1_output.get('retrieved_results', [])),
            '最高相似度': max(system1_output.get('similarities', [0])),
            '置信度': system1_output.get('confidence', 0.0)
        }
        
        # 系统 2 摘要
        summary['系统 2 (推理)'] = {
            '推理步数': len(system2_output.get('reasoning_chain', [])),
            '使用事实数': len(system2_output.get('facts_used', [])),
            '答案': system2_output.get('answer', '')[:100],
            '置信度': system2_output.get('confidence', 0.0)
        }
        
        # 融合摘要
        weights = fusion_output.get('system_weights', [0.5, 0.5])
        summary['融合决策'] = {
            '系统 1 权重': weights[0] if len(weights) > 0 else 0.5,
            '系统 2 权重': weights[1] if len(weights) > 1 else 0.5,
            '是否冲突': fusion_output.get('conflict_detected', False),
            '最终答案': fusion_output.get('final_answer', '')[:100]
        }
        
        return summary
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """创建空提示图"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"🤔 {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#666666")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=500,
            height=300
        )
        return fig
