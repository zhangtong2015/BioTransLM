#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练监控标签页 - 实时显示训练进度和损失曲线
"""

import gradio as gr
from typing import Dict, Any, Optional
import logging
import threading
import time

from ..utils import StateManager
from ..components import HTMVisualizer, LossChart

logger = logging.getLogger(__name__)


def create_training_tab(state_manager: StateManager):
    """创建训练监控标签页"""
    
    htm_viz = HTMVisualizer()
    loss_chart = LossChart()
    
    with gr.Tab("🎯 训练监控", id="training_tab"):
        gr.Markdown("## 📈 训练监控面板")
        
        # ========== 第一行：控制按钮和进度 ==========
        with gr.Row():
            with gr.Column(scale=2):
                status_box = gr.Textbox(
                    label="训练状态",
                    value="就绪",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                start_btn = gr.Button("▶️ 开始训练", variant="primary")
                stop_btn = gr.Button("⏹️ 停止训练", variant="stop")
        
        # ========== 第二行：配置参数 ==========
        with gr.Accordion("⚙️ 训练配置", open=False):
            with gr.Row():
                epochs_slider = gr.Slider(1, 50, value=10, step=1, label="Epochs")
                batch_size_slider = gr.Slider(4, 64, value=16, step=4, label="Batch Size")
                
            with gr.Row():
                lr_slider = gr.Slider(1e-5, 1e-3, value=1e-4, step=1e-5, label="Learning Rate")
                max_seq_slider = gr.Slider(32, 512, value=128, step=32, label="最大序列长度")
        
        # ========== 第三行：损失曲线和 HTM 可视化 ==========
        with gr.Row():
            with gr.Column(scale=2):
                loss_plot = gr.Plot(label="损失曲线")
            
            with gr.Column(scale=1):
                htm_plot = gr.Plot(label="HTM 激活热图")
        
        # ========== 第四行：指标显示 ==========
        with gr.Row():
            epoch_display = gr.Number(label="当前 Epoch", value=0)
            step_display = gr.Number(label="Global Step", value=0)
            loss_display = gr.Number(label="当前损失", value=0.0)
        
        metrics_json = gr.JSON(label="详细指标")
        
        # ========== 回调函数 ==========
        
        training_thread = None
        stop_flag = False
        
        def mock_training_loop(epochs, batch_size, lr):
            """模拟训练循环（用于演示）"""
            nonlocal stop_flag
            
            for epoch in range(epochs):
                if stop_flag:
                    break
                
                for step in range(10):  # 每个 epoch 10 步
                    if stop_flag:
                        break
                    
                    # 模拟损失下降
                    base_loss = 2.5 * (0.9 ** (epoch * 10 + step))
                    losses = {
                        'total': base_loss,
                        'lm': base_loss * 0.9,
                        'sparsity': 0.1,
                        'temporal': 0.05
                    }
                    
                    # 模拟 HTM 指标
                    htm_metrics = {
                        'prediction_accuracy': min(0.95, 0.5 + epoch * 0.05),
                        'n_bursts': max(0, 100 - epoch * 10),
                        'total_columns': 4096
                    }
                    
                    # 更新状态
                    state_manager.update_training_state(epoch, step, losses, htm_metrics)
                    
                    time.sleep(0.1)  # 模拟训练时间
            
            state_manager.stop_training()
        
        def on_start_training(epochs, batch_size, lr, max_seq_len):
            """开始训练"""
            nonlocal training_thread, stop_flag
            
            if state_manager.is_training():
                return "⚠️ 训练正在进行中", 0, 0, 0.0, {}, None, None
            
            try:
                # 重置停止标志
                stop_flag = False
                
                # 检查数据集
                dataset_state = state_manager.get_dataset_state()
                if dataset_state['loaded_dataset'] is None:
                    gr.Warning("请先在'数据集管理'标签页加载数据")
                    return "❌ 错误：没有数据集", 0, 0, 0.0, {}, None, None
                
                # 开始训练
                state_manager.start_training(int(epochs))
                
                # 启动训练线程
                training_thread = threading.Thread(
                    target=mock_training_loop,
                    args=(int(epochs), int(batch_size), lr),
                    daemon=True
                )
                training_thread.start()
                
                logger.info(f"训练已启动：{epochs} epochs, batch_size={batch_size}, lr={lr}")
                
                return "▶️ 训练中...", 0, 0, 2.5, {"状态": "初始化完成"}, None, None
                
            except Exception as e:
                logger.error(f"启动训练失败：{e}", exc_info=True)
                return f"❌ 错误：{e}", 0, 0, 0.0, {}, None, None
        
        def on_stop_training():
            """停止训练"""
            nonlocal stop_flag, training_thread
            
            stop_flag = True
            state_manager.stop_training()
            
            if training_thread and training_thread.is_alive():
                training_thread.join(timeout=2)
            
            return "⏹️ 训练已停止", 0, 0, 0.0, {}, None, None
        
        def update_training_display():
            """实时更新训练显示"""
            training_state = state_manager.get_training_state()
            
            if not training_state['is_training']:
                return (
                    gr.update(),  # Status
                    gr.update(value=0),  # Epoch
                    gr.update(value=0),  # Step
                    gr.update(value=0.0),  # Loss
                    gr.update(value={}),  # Metrics
                    gr.update(),  # Loss plot
                    gr.update()   # HTM plot
                )
            
            # 提取最新损失
            loss_history = training_state['loss_history']
            current_loss = loss_history[-1]['total'] if loss_history else 0.0
            
            # 提取 HTM 指标
            htm_history = training_state['htm_metrics_history']
            latest_htm = htm_history[-1] if htm_history else {}
            
            # 创建图表
            loss_fig = loss_chart.plot_loss_curves(loss_history) if loss_history else None
            htm_fig = None
            
            if latest_htm and 'prediction_accuracy' in latest_htm:
                htm_fig = htm_viz.plot_temporal_metrics(latest_htm)
            
            return (
                gr.update(value=f"▶️ 训练中... {training_state['current_epoch']}/{training_state['total_epochs']}"),
                gr.update(value=training_state['current_epoch']),
                gr.update(value=training_state['global_step']),
                gr.update(value=current_loss),
                gr.update(value={
                    '当前 Epoch': training_state['current_epoch'],
                    '总步数': training_state['global_step'],
                    '预测准确率': latest_htm.get('prediction_accuracy', 0.0),
                    'Burst 率': latest_htm.get('burst_rate', 0.0)
                }),
                gr.update(value=loss_fig) if loss_fig else gr.update(),
                gr.update(value=htm_fig) if htm_fig else gr.update()
            )
        
        # 绑定事件
        start_btn.click(
            fn=on_start_training,
            inputs=[epochs_slider, batch_size_slider, lr_slider, max_seq_slider],
            outputs=[status_box, epoch_display, step_display, loss_display, metrics_json, loss_plot, htm_plot]
        )
        
        stop_btn.click(
            fn=on_stop_training,
            outputs=[status_box, epoch_display, step_display, loss_display, metrics_json, loss_plot, htm_plot]
        )
        
        # 定时刷新（每 2 秒）
        gr.Timer(2.0).tick(
            fn=update_training_display,
            outputs=[status_box, epoch_display, step_display, loss_display, metrics_json, loss_plot, htm_plot]
        )
