#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
记忆系统可视化组件 - 工作记忆、情景记忆、语义记忆展示
"""

import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class MemoryViewer:
    """记忆系统查看器"""
    
    def visualize_working_memory(
        self,
        wm_items: List[Dict[str, Any]],
        title: str = "工作记忆内容"
    ) -> go.Figure:
        """可视化工作记忆项"""
        
        if not wm_items or len(wm_items) == 0:
            return self._create_empty_figure("工作记忆为空")
        
        # 提取数据
        items = []
        importances = []
        
        for i, item in enumerate(wm_items[:10]):  # 最多显示 10 项
            items.append(f"Item {i+1}")
            importances.append(item.get('importance', 0.5))
        
        # 创建条形图
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=items,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="重要性")
            ),
            hovertemplate='%{y}<br>重要性：%{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{title} ({len(wm_items)} 项)",
            xaxis_title="重要性分数",
            yaxis_title="记忆项",
            width=600,
            height=max(300, len(items) * 40),
            template='plotly_white'
        )
        
        return fig
    
    def visualize_episodic_retrieval(
        self,
        episodes: List[Dict[str, Any]],
        title: str = "情景记忆检索结果"
    ) -> go.Figure:
        """可视化情景记忆检索"""
        
        if not episodes or len(episodes) == 0:
            return self._create_empty_figure("无检索到的情景记忆")
        
        # 提取相似度分数
        similarities = [ep.get('similarity', 0.5) for ep in episodes[:10]]
        timestamps = [ep.get('timestamp', 0) for ep in episodes[:10]]
        
        # 创建散点图
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=range(len(episodes)),
            y=similarities,
            mode='markers+text',
            marker=dict(
                size=15,
                color=similarities,
                colorscale='Blues',
                showscale=True,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[f"#{i+1}" for i in range(len(episodes))],
            textposition="top center",
            hovertemplate='情景 #%{x}+1<br>相似度：%{y:.3f}<br>时间戳：%{customdata}<extra></extra>',
            customdata=timestamps
        ))
        
        fig.update_layout(
            title=f"{title} (Top {len(episodes)})",
            xaxis_title="检索排名",
            yaxis_title="相似度分数",
            width=600,
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def visualize_semantic_network(
        self,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        title: str = "语义记忆网络"
    ) -> go.Figure:
        """可视化语义记忆概念网络"""
        
        try:
            # 创建网络图
            G = nx.Graph()
            
            # 添加节点
            concept_positions = {}
            for i, concept in enumerate(concepts[:20]):  # 最多 20 个概念
                concept_id = concept.get('id', f'concept_{i}')
                label = concept.get('label', concept.get('name', f'C{i}'))
                G.add_node(concept_id, label=label, importance=concept.get('importance', 0.5))
                concept_positions[concept_id] = (i % 5, i // 5)
            
            # 添加边
            for rel in relations[:30]:  # 最多 30 条关系
                subj = rel.get('subject', '')
                obj = rel.get('object', '')
                rel_type = rel.get('type', 'related')
                
                if subj in G.nodes and obj in G.nodes:
                    G.add_edge(subj, obj, type=rel_type)
            
            if len(G.nodes) == 0:
                return self._create_empty_figure("无语义网络数据")
            
            # 使用 spring 布局
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # 提取节点坐标
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_labels = [G.nodes[node].get('label', node) for node in G.nodes()]
            
            # 创建节点轨迹
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_labels,
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=20,
                    colorbar=dict(
                        thickness=15,
                        title='重要性',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2)
            )
            
            # 根据重要性设置颜色
            node_colors = [G.nodes[node].get('importance', 0.5) for node in G.nodes()]
            node_trace.marker.color = node_colors
            
            # 创建边轨迹
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=title,
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               width=700,
                               height=600
                           ))
            
            return fig
            
        except Exception as e:
            logger.error(f"语义网络可视化失败：{e}")
            return self._create_error_figure(str(e))
    
    def display_memory_summary(
        self,
        memory_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成记忆系统状态摘要（用于 JSON 显示）"""
        
        summary = {
            '工作记忆': {},
            '情景记忆': {},
            '语义记忆': {}
        }
        
        # 工作记忆摘要
        wm = memory_state.get('working_memory', {})
        summary['工作记忆'] = {
            'item_count': wm.get('item_count', 0),
            'capacity_used': f"{wm.get('item_count', 0)}/10",
            'items_preview': [item.get('content', '')[:50] for item in wm.get('memory_items', [])[:3]]
        }
        
        # 情景记忆摘要
        em = memory_state.get('episodic_retrieval', {})
        summary['情景记忆'] = {
            'episode_count': em.get('episode_count', 0),
            'retrieved_episodes': len(em.get('episodes', [])),
            'avg_similarity': sum(ep.get('similarity', 0) for ep in em.get('episodes', [])) / max(1, len(em.get('episodes', [])))
        }
        
        # 语义记忆摘要
        sm = memory_state.get('semantic_retrieval', {})
        summary['语义记忆'] = {
            'concept_count': sm.get('concept_count', 0),
            'retrieved_concepts': len(sm.get('concepts', [])),
            'relation_count': len(sm.get('semantic_relations', []))
        }
        
        return summary
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """创建空提示图"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"📝 {message}",
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
    
    def _create_error_figure(self, error_msg: str) -> go.Figure:
        """创建错误提示图"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ 错误：{error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=500,
            height=300
        )
        return fig
