#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理过程查看标签页 - 展示对话的详细处理链路
"""

import gradio as gr
from typing import Dict, Any, List
import logging

from ..utils import StateManager, ModelStateAnalyzer, ModelManager
from ..components import HTMVisualizer, MemoryViewer, ReasoningTrace

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_inspection_tab(state_manager: StateManager):
    """创建处理过程查看标签页"""
    
    htm_viz = HTMVisualizer()
    memory_viewer = MemoryViewer()
    reasoning_trace = ReasoningTrace()
    model_analyzer = ModelStateAnalyzer()
    model_manager = ModelManager()
    
    with gr.Tab("🔍 处理过程查看", id="inspection_tab"):
        gr.Markdown("## 🔬 BioTransLM 处理过程可视化")
        gr.Markdown("查看最近一次对话的详细处理链路，了解模型内部工作机制")
        
        # ========== 处理流程思维导图（新增）==========
        with gr.Group():
            gr.Markdown("### 🧠 处理流程总览")
            process_map_plot = gr.Plot(label="处理流程思维导图")
            gr.Markdown("""
            **上图展示了 BioTransLM 的 7 步处理流程**：
            1. 感觉门控 → 2. 多粒度编码 → 3. HTM 系统 → 4. 神经调节 → 5. 记忆系统 → 6. 双系统推理 → 7. 响应生成
            
            💡 **提示**：点击下方各个阶段查看详细可视化
            """)
        
        # ========== 对话选择 ==========
        conversation_selector = gr.Dropdown(
            label="选择要查看的对话",
            choices=[],
            interactive=True
        )
        
        refresh_btn = gr.Button("🔄 刷新对话列表", variant="secondary")
        
        # ========== 分阶段展示处理链路 ==========
        
        # 阶段 1：感觉门控
        with gr.Accordion("1️⃣ 感觉门控 (Sensory Gating)", open=False):
            gating_stats = gr.JSON(label="门控统计信息")
            gating_plot = gr.Plot(label="门控输出分布")
        
        # 阶段 2：多粒度编码
        with gr.Accordion("2️⃣ 多粒度编码 (Multi-granular Encoding)", open=False):
            multigran_info = gr.JSON(label="编码结果")
        
        # 阶段 3：HTM 系统
        with gr.Accordion("3️⃣ HTM 系统 (Hierarchical Temporal Memory)", open=False):
            with gr.Row():
                htm_spatial_plot = gr.Plot(label="空间池化器激活热图")
                htm_temporal_plot = gr.Plot(label="时序记忆指标")
            
            htm_metrics = gr.JSON(label="HTM 详细指标")
        
        # 阶段 4：神经调节
        with gr.Accordion("4️⃣ 神经调节 (Neural Regulation)", open=False):
            regulation_signals = gr.JSON(label="调节信号")
        
        # 阶段 5：记忆系统
        with gr.Accordion("5️⃣ 记忆系统 (Memory System)", open=False):
            with gr.Row():
                wm_plot = gr.Plot(label="工作记忆")
                em_plot = gr.Plot(label="情景记忆检索")
            
            memory_summary = gr.JSON(label="记忆系统摘要")
        
        # 阶段 6：双系统推理
        with gr.Accordion("6️⃣ 双系统推理 (Dual System Reasoning)", open=False):
            with gr.Row():
                system1_plot = gr.Plot(label="系统 1 - 直觉检索")
                system2_plot = gr.Plot(label="系统 2 - 符号推理链")
            
            fusion_plot = gr.Plot(label="双系统融合")
            reasoning_summary = gr.JSON(label="推理过程摘要")
        
        # 阶段 7：响应生成
        with gr.Accordion("7️⃣ 响应生成 (Response Generation)", open=False):
            generation_stats = gr.JSON(label="生成统计")
        
        # ========== 模型状态分析 ==========
        with gr.Accordion("📊 模型状态分析 (Model State Analysis)", open=False):
            gr.Markdown("### 模型内部状态可视化")
            
            with gr.Row():
                with gr.Column(scale=1):
                    weight_dist_plot = gr.Plot(label="权重分布直方图")
                    layer_sparsity_plot = gr.Plot(label="层稀疏度")
                
                with gr.Column(scale=1):
                    activation_heatmap = gr.Plot(label="激活热图")
                    sdr_stats_plot = gr.Plot(label="SDR 特征")
            
            with gr.Row():
                htm_state_metrics = gr.JSON(label="HTM 状态指标")
                gradient_monitor = gr.JSON(label="梯度监控")
            
            model_summary = gr.Textbox(
                label="模型摘要",
                lines=5,
                interactive=False
            )
        
        # ========== 回调函数 ==========
        
        def list_conversations():
            """列出所有对话"""
            conversations = state_manager.list_conversations()
            
            if not conversations:
                return gr.update(choices=[], value=None)
            
            # 构建选项列表
            choices = []
            for conv in conversations:
                conv_id = conv['id']
                msg_count = conv['message_count']
                timestamp = conv.get('last_message_time', 0)
                
                from datetime import datetime
                time_str = datetime.fromtimestamp(timestamp).strftime('%m-%d %H:%M')
                
                label = f"对话 #{conv_id} ({msg_count}条消息) - {time_str}"
                choices.append((label, conv_id))
            
            return gr.update(choices=choices, value=choices[0][1] if choices else None)
        
        def on_conversation_selected(conv_id):
            """处理对话选择"""
            if not conv_id:
                return (
                    {}, None, {}, None, None, {},
                    None, None, None, None, None, {},
                    {}, None, None, {},
                    None, None, None, None, {}, {}, ""
                )
            
            # 获取对话和处理结果
            conv = state_manager.get_conversation(conv_id)
            if not conv or not conv.processing_results:
                gr.Info("该对话没有保存处理结果")
                return (
                    {"状态": "无处理数据"}, None,
                    {"状态": "无处理数据"}, None, None, {"状态": "无处理数据"},
                    None, None, None, None, None, {"状态": "无处理数据"},
                    {"状态": "无处理数据"}, None, None, {"状态": "无处理数据"},
                    None, None, None, None, {}, {}, "没有可用的模型"
                )
            
            results = conv.processing_results
            
            try:
                # 1. 感觉门控
                gating_info = results.get('sensory_gating', {})
                gating_fig = None  # 简化：不绘制具体图表
                
                # 2. 多粒度编码
                multigran_info = results.get('multigranular_encoding', {})
                
                # 3. HTM 系统
                htm_info = results.get('htm_system', {})
                htm_spatial_fig = None
                htm_temporal_fig = None
                
                if htm_info and 'temporal_output' in htm_info:
                    htm_temporal_fig = htm_viz.plot_temporal_metrics(htm_info['temporal_output'])
                
                if htm_info and 'spatial_output' in htm_info:
                    htm_spatial_fig = htm_viz.plot_column_activations(
                        htm_info['spatial_output'].get('active_columns')
                    )
                
                htm_metrics_info = htm_viz.extract_htm_metrics(htm_info)
                
                # 4. 神经调节
                regulation_info = results.get('neural_regulation', {})
                
                # 5. 记忆系统
                memory_info = results.get('memory_system', {})
                wm_fig = None
                em_fig = None
                
                if memory_info:
                    wm_info = memory_info.get('working_memory', {})
                    if wm_info.get('item_count', 0) > 0:
                        wm_fig = memory_viewer.visualize_working_memory(
                            wm_info.get('memory_items', [])
                        )
                    
                    em_info = memory_info.get('episodic_retrieval', {})
                    if em_info.get('episode_count', 0) > 0:
                        em_fig = memory_viewer.visualize_episodic_retrieval(
                            em_info.get('episodes', [])
                        )
                
                memory_summary_info = memory_viewer.display_memory_summary(memory_info)
                
                # 6. 双系统推理
                system1_info = results.get('system1', {})
                system2_info = results.get('system2', {})
                fusion_info = results.get('fusion', {})
                
                s1_fig = reasoning_trace.visualize_system1_results(system1_info)
                s2_fig = reasoning_trace.visualize_system2_chain(system2_info)
                fusion_fig = reasoning_trace.visualize_fusion_weights(fusion_info)
                
                reasoning_summary_info = reasoning_trace.create_reasoning_summary(
                    system1_info, system2_info, fusion_info
                )
                
                # 7. 响应生成
                generation_info = results.get('generation', {})
                
                # ========== 模型状态分析 ==========
                weight_fig = None
                sparsity_fig = None
                activation_fig = None
                sdr_fig = None
                htm_state_info = {}
                gradient_info = {}
                summary_text = "没有可用的模型"
                
                try:
                    # 检查是否有模型
                    if model_manager.has_model():
                        model = model_manager.get_model()
                        
                        # 分析权重
                        weight_stats = model_analyzer.analyze_weights(model)
                        weight_fig = model_analyzer.plot_weight_distribution(weight_stats)
                        sparsity_fig = model_analyzer.plot_layer_sparsity(weight_stats)
                        
                        # 分析 SDR 特征
                        if 'sdr_output' in results:
                            sdr_tensor = results['sdr_output']
                            sdr_stats = model_analyzer.analyze_sdr(sdr_tensor)
                            sdr_fig = model_analyzer.plot_sdr_stats(sdr_stats)
                        
                        # HTM 状态
                        if htm_info:
                            htm_state_info = model_analyzer.analyze_htm_state(htm_info)
                        
                        # 生成摘要
                        report = model_analyzer.generate_report(model, None, htm_info)
                        summary_text = report.get('summary', '无法生成模型摘要')
                        
                        logger.info("已完成模型状态分析")
                    else:
                        summary_text = "当前没有加载的模型，请先创建或加载模型"
                        
                except Exception as e:
                    logger.warning(f"模型状态分析失败：{e}")
                    summary_text = f"模型分析失败：{str(e)}"
                
                logger.info(f"已加载对话处理详情：{conv_id}")
                
                return (
                    gating_info, gating_fig,
                    multigran_info, None,  # 简化多粒度可视化
                    htm_spatial_fig, htm_temporal_fig, htm_metrics_info,
                    regulation_info,
                    wm_fig, em_fig, memory_summary_info,
                    s1_fig, s2_fig, fusion_fig, reasoning_summary_info,
                    generation_info,
                    weight_fig, sparsity_fig, activation_fig, sdr_fig,
                    htm_state_info, gradient_info, summary_text
                )
                
            except Exception as e:
                logger.error(f"加载处理详情失败：{e}", exc_info=True)
                error_result = {"错误": str(e)}
                
                return (
                    error_result, None, error_result, None, None, error_result,
                    None, None, error_result, None, None, error_result,
                    None, None, None, error_result,
                    error_result,
                    None, None, None, None, error_result, error_result, f"错误：{str(e)}"
                )
        
        # 绑定事件
        refresh_btn.click(
            fn=list_conversations,
            outputs=[conversation_selector]
        )
        
        conversation_selector.change(
            fn=on_conversation_selected,
            inputs=[conversation_selector],
            outputs=[
                gating_stats, gating_plot,
                multigran_info, multigran_info,  # 简化，使用相同输出
                htm_spatial_plot, htm_temporal_plot, htm_metrics,
                regulation_signals,
                wm_plot, em_plot, memory_summary,
                system1_plot, system2_plot, fusion_plot, reasoning_summary,
                generation_stats,
                weight_dist_plot, layer_sparsity_plot, activation_heatmap, sdr_stats_plot,
                htm_state_metrics, gradient_monitor, model_summary
            ]
        )
        
        # 页面加载时显示思维导图和刷新列表
        gr.on(
            triggers=[],
            fn=create_process_map,
            outputs=[process_map_plot]
        )
        gr.on(
            triggers=[],
            fn=list_conversations,
            outputs=[conversation_selector]
        )
