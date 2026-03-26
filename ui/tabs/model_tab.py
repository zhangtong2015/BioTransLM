#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型管理标签页 - 创建、配置、训练、保存和加载模型
"""

import gradio as gr
from typing import Dict, Any, List
import logging
import time

from ..utils import StateManager, get_model_manager, ModelConfig

logger = logging.getLogger(__name__)


def create_model_tab(state_manager: StateManager):
    """创建模型管理标签页"""
    
    model_manager = get_model_manager()
    
    with gr.Tab("🧠 模型管理", id="model_tab"):
        gr.Markdown("## 🧠 模型管理中心")
        gr.Markdown("创建新模型、配置参数、训练模型、保存和加载预训练权重")
        
        # ========== 第一部分：模型状态 ==========
        gr.Markdown("### 📊 当前模型状态")
        
        with gr.Row():
            model_status_box = gr.Textbox(
                label="模型状态",
                value="未加载模型",
                interactive=False,
                scale=2
            )
            
            refresh_status_btn = gr.Button("🔄 刷新状态", variant="secondary", scale=1)
        
        model_info_json = gr.JSON(
            label="模型详细信息",
            value={"状态": "请先创建或加载模型"}
        )
        
        # ========== 第二部分：创建新模型 ==========
        gr.Markdown("### ➕ 创建新模型")
        
        with gr.Accordion("⚙️ 模型配置参数", open=False):
            with gr.Row():
                model_name_input = gr.Textbox(
                    label="模型名称",
                    value="biotranslm_base",
                    placeholder="输入模型名称"
                )
                
                model_type_dropdown = gr.Dropdown(
                    choices=[
                        ("BioGenerator (生物启发式生成器)", "bio_generator"),
                        ("Orchestrator (完整编排器)", "orchestrator")
                    ],
                    value="bio_generator",
                    label="模型类型"
                )
            
            with gr.Row():
                vocab_size_slider = gr.Slider(
                    minimum=1000,
                    maximum=100000,
                    value=50257,
                    step=1000,
                    label="词汇表大小"
                )
                
                hidden_dim_slider = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=768,
                    step=64,
                    label="隐藏层维度"
                )
            
            with gr.Row():
                n_columns_slider = gr.Slider(
                    minimum=1024,
                    maximum=8192,
                    value=4096,
                    step=512,
                    label="HTM 列数量"
                )
                
                sdr_size_slider = gr.Slider(
                    minimum=1024,
                    maximum=8192,
                    value=4096,
                    step=512,
                    label="SDR 维度"
                )
            
            with gr.Row():
                max_seq_len_slider = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="最大序列长度"
                )
                
                device_radio = gr.Radio(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="计算设备"
                )
        
        with gr.Row():
            create_model_btn = gr.Button("✨ 创建新模型", variant="primary")
            load_config_btn = gr.Button("📂 加载配置文件")
        
        # ========== 第三部分：训练模型 ==========
        gr.Markdown("### 🎯 训练模型")
        
        with gr.Row():
            with gr.Column(scale=2):
                dataset_path_input = gr.Textbox(
                    label="数据集路径",
                    placeholder="请输入数据集文件路径或使用下方选择器",
                    interactive=True
                )
                
                dataset_file_upload = gr.File(
                    label="或上传数据集文件",
                    file_types=[".jsonl", ".csv", ".txt"],
                    height=100
                )
            
            with gr.Column(scale=1):
                train_config_json = gr.JSON(
                    label="训练配置",
                    value={
                        "batch_size": 16,
                        "num_epochs": 10,
                        "learning_rate": 0.0001,
                        "max_seq_length": 128,
                        "save_steps": 100
                    }
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("⚙️ 高级训练参数", open=False):
                    with gr.Row():
                        batch_size_slider = gr.Slider(4, 64, value=16, step=4, label="Batch Size")
                        epochs_slider = gr.Slider(1, 50, value=10, step=1, label="Epochs")
                    
                    with gr.Row():
                        lr_slider = gr.Slider(1e-5, 1e-3, value=1e-4, step=1e-5, label="Learning Rate")
                        max_seq_slider = gr.Slider(32, 512, value=128, step=32, label="最大序列长度")
                    
                    save_steps_slider = gr.Slider(50, 500, value=100, step=50, label="保存步数间隔")
            
            with gr.Column(scale=1):
                training_progress = gr.Textbox(
                    label="训练进度",
                    value="就绪",
                    interactive=False,
                    lines=3
                )
        
        with gr.Row():
            start_train_btn = gr.Button("▶️ 开始训练", variant="primary", scale=1)
            stop_train_btn = gr.Button("⏹️ 停止训练", variant="stop", scale=1)
            export_model_btn = gr.Button("💾 导出模型", scale=1)
        
        # ========== 第四部分：保存和加载模型 ==========
        gr.Markdown("### 💾 保存与加载模型")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 保存当前模型")
                
                save_path_input = gr.Textbox(
                    label="保存路径（可选）",
                    placeholder="留空则保存到默认目录"
                )
                
                save_model_name = gr.Textbox(
                    label="模型名称",
                    placeholder="用于自动生成文件名"
                )
                
                save_model_btn = gr.Button("💾 保存模型", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("#### 加载已有模型")
                
                saved_models_dropdown = gr.Dropdown(
                    label="选择已保存的模型",
                    choices=[],
                    interactive=True
                )
                
                load_model_btn = gr.Button("📂 加载模型", variant="secondary")
        
        saved_models_list = gr.JSON(label="已保存的模型列表")
        
        # ========== 第五部分：模型操作 ==========
        gr.Markdown("### 🛠️ 模型操作")
        
        with gr.Row():
            chat_with_model_btn = gr.Button("💬 切换到对话模式")
            view_inspection_btn = gr.Button("🔍 查看处理过程")
            reset_model_btn = gr.Button("🗑️ 重置模型", variant="stop")
        
        # ========== 回调函数 ==========
        
        def refresh_model_status():
            """刷新模型状态"""
            info = model_manager.get_model_info()
            
            if info['has_model']:
                status = f"✅ 已加载模型：{info['info'].get('model_name', 'unknown')}"
                if info['is_training']:
                    status += " (训练中...)"
            else:
                status = "❌ 未加载模型"
            
            return status, info
        
        def on_create_model(
            name, model_type, vocab_size, hidden_dim,
            n_columns, sdr_size, max_seq_len, device
        ):
            """创建新模型"""
            try:
                config = ModelConfig(
                    model_name=name,
                    model_type=model_type,
                    vocab_size=vocab_size,
                    hidden_dim=hidden_dim,
                    n_columns=n_columns,
                    sdr_size=sdr_size,
                    max_sequence_length=max_seq_len,
                    device=device
                )
                
                success = model_manager.create_model(config)
                
                if success:
                    info = model_manager.get_model_info()
                    return (
                        f"✅ 模型创建成功：{name}",
                        info,
                        info,
                        f"模型 '{name}' 已创建，可以开始训练或保存"
                    )
                else:
                    return (
                        "❌ 模型创建失败",
                        {"错误": "创建失败"},
                        {},
                        "模型创建失败，请查看日志"
                    )
                    
            except Exception as e:
                logger.error(f"创建模型失败：{e}", exc_info=True)
                return (
                    f"❌ 错误：{e}",
                    {"错误": str(e)},
                    {},
                    f"创建失败：{e}"
                )
        
        def on_update_train_config(batch_size, epochs, lr, max_seq, save_steps):
            """更新训练配置"""
            config = {
                "batch_size": int(batch_size),
                "num_epochs": int(epochs),
                "learning_rate": float(lr),
                "max_seq_length": int(max_seq),
                "save_steps": int(save_steps)
            }
            return config
        
        def on_dataset_uploaded(file):
            """处理数据集上传"""
            if file is None:
                return "", {}
            
            # 更新数据集路径
            return file.name, {"dataset_path": file.name}
        
        def on_start_training(
            dataset_path, train_config,
            batch_size, epochs, lr, max_seq, save_steps
        ):
            """开始训练"""
            if not model_manager.has_model():
                gr.Warning("请先创建或加载模型")
                return "❌ 错误：没有可用模型", {}, []
            
            if not dataset_path:
                gr.Warning("请指定数据集路径")
                return "❌ 错误：没有数据集", {}, []
            
            # 构建训练配置
            config = {
                "batch_size": int(batch_size),
                "num_epochs": int(epochs),
                "learning_rate": float(lr),
                "max_seq_length": int(max_seq),
                "save_steps": int(save_steps)
            }
            
            # 定义进度回调
            def progress_callback(epoch, step, losses, metrics):
                nonlocal training_progress
                
                msg = f"Epoch: {epoch+1}\n"
                msg += f"Step: {step}\n"
                
                if 'total' in losses:
                    msg += f"Loss: {losses['total']:.4f}\n"
                
                if 'batch_loss' in metrics:
                    msg += f"Batch Loss: {metrics['batch_loss']:.4f}\n"
                
                state_manager._training_state.update({
                    'current_epoch': epoch,
                    'global_step': step,
                    'current_loss': losses.get('total', 0),
                    'loss_history': state_manager._training_state.get('loss_history', []) + [losses]
                })
            
            # 启动训练
            success = model_manager.start_training(
                dataset_path=dataset_path,
                config=config,
                progress_callback=progress_callback
            )
            
            if success:
                return (
                    f"▶️ 训练已开始\n数据集：{dataset_path}\n配置：{config}",
                    train_config,
                    []  # 简化：不返回列表
                )
            else:
                return "❌ 训练启动失败", {}, []
        
        def on_stop_training():
            """停止训练"""
            model_manager.stop_training()
            return "⏹️ 训练已停止", {}, []
        
        def on_save_model(save_path, model_name):
            """保存模型"""
            if not model_manager.has_model():
                gr.Warning("没有可保存的模型")
                return "❌ 错误：没有模型", {}
            
            try:
                saved_path = model_manager.save_model(
                    save_path=save_path if save_path else None,
                    name=model_name if model_name else None
                )
                
                # 刷新模型列表
                models = model_manager.list_saved_models()
                
                return (
                    f"✅ 模型已保存：{saved_path}",
                    {"saved_path": saved_path, "models": models}
                )
                
            except Exception as e:
                logger.error(f"保存模型失败：{e}", exc_info=True)
                return f"❌ 保存失败：{e}", {}
        
        def on_load_model(model_path):
            """加载模型"""
            if not model_path:
                gr.Warning("请选择要加载的模型")
                return "❌ 错误：未选择模型", {}
            
            success = model_manager.load_model(model_path)
            
            if success:
                info = model_manager.get_model_info()
                return (
                    f"✅ 模型已加载：{model_path}",
                    info
                )
            else:
                return f"❌ 加载失败：{model_path}", {}
        
        def list_saved_models():
            """列出已保存的模型"""
            models = model_manager.list_saved_models()
            
            choices = [(m['name'], m['path']) for m in models]
            
            return {
                'models': models,
                'dropdown_choices': choices
            }
        
        def on_reset_model():
            """重置模型"""
            model_manager.reset()
            return "🗑️ 模型已重置", {"状态": "未加载模型"}
        
        # 绑定事件
        
        # 刷新状态
        refresh_status_btn.click(
            fn=refresh_model_status,
            outputs=[model_status_box, model_info_json]
        )
        
        # 创建模型
        create_model_btn.click(
            fn=on_create_model,
            inputs=[
                model_name_input, model_type_dropdown,
                vocab_size_slider, hidden_dim_slider,
                n_columns_slider, sdr_size_slider,
                max_seq_len_slider, device_radio
            ],
            outputs=[model_status_box, model_info_json, train_config_json, training_progress]
        )
        
        # 更新训练配置
        batch_size_slider.change(
            fn=lambda bs, ep, lr, ms, ss: on_update_train_config(bs, ep, lr, ms, ss),
            inputs=[batch_size_slider, epochs_slider, lr_slider, max_seq_slider, save_steps_slider],
            outputs=[train_config_json]
        )
        
        epochs_slider.change(
            fn=lambda bs, ep, lr, ms, ss: on_update_train_config(bs, ep, lr, ms, ss),
            inputs=[batch_size_slider, epochs_slider, lr_slider, max_seq_slider, save_steps_slider],
            outputs=[train_config_json]
        )
        
        # 数据集上传
        dataset_file_upload.change(
            fn=on_dataset_uploaded,
            inputs=[dataset_file_upload],
            outputs=[dataset_path_input, train_config_json]
        )
        
        # 开始训练
        start_train_btn.click(
            fn=on_start_training,
            inputs=[
                dataset_path_input, train_config_json,
                batch_size_slider, epochs_slider,
                lr_slider, max_seq_slider, save_steps_slider
            ],
            outputs=[training_progress, train_config_json, saved_models_list]
        )
        
        # 停止训练
        stop_train_btn.click(
            fn=on_stop_training,
            outputs=[training_progress, train_config_json, saved_models_list]
        )
        
        # 保存模型
        save_model_btn.click(
            fn=on_save_model,
            inputs=[save_path_input, save_model_name],
            outputs=[training_progress, saved_models_list]
        )
        
        # 列出已保存的模型
        saved_models_dropdown.change(
            fn=list_saved_models,
            outputs=[saved_models_list]
        )
        
        # 加载模型
        load_model_btn.click(
            fn=on_load_model,
            inputs=[saved_models_dropdown],
            outputs=[training_progress, model_info_json]
        )
        
        # 重置模型
        reset_model_btn.click(
            fn=on_reset_model,
            outputs=[model_status_box, model_info_json]
        )
        
        # 页面加载时刷新状态和模型列表
        gr.on(
            triggers=[],
            fn=refresh_model_status,
            outputs=[model_status_box, model_info_json]
        )
        gr.on(
            triggers=[],
            fn=list_saved_models,
            outputs=[saved_models_list]
        )
