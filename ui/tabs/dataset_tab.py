#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集管理标签页 v2 - 添加智能推荐和一键训练
"""

import gradio as gr
from typing import Dict, Any, Tuple
import logging

from ..utils import StateManager, DataProcessor, get_training_advisor, get_model_manager

logger = logging.getLogger(__name__)


def create_dataset_tab(state_manager: StateManager):
    """创建数据集管理标签页（v2 智能版）"""
    
    data_processor = DataProcessor()
    training_advisor = get_training_advisor()
    model_manager = get_model_manager()
    
    with gr.Tab("📊 数据集管理", id="dataset_tab"):
        gr.Markdown("## 📚 数据集管理")
        gr.Markdown("上传训练数据或选择内置模板，支持 JSONL/CSV/TXT 格式")
        
        # ========== 第一行：文件上传和模板选择 ==========
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 方式一：上传数据文件")
                file_upload = gr.File(
                    label="上传数据集文件",
                    file_types=[".jsonl", ".csv", ".txt", ".json"],
                    height=200
                )
                
                gr.Markdown("### 方式二：使用内置模板")
                template_dropdown = gr.Dropdown(
                    choices=["小说", "蒸馏数据", "一问一答"],
                    label="选择模板类型",
                    value="一问一答"
                )
                
                template_samples = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=10,
                    step=5,
                    label="生成样本数量"
                )
                
                load_template_btn = gr.Button("加载模板数据", variant="secondary")
            
            with gr.Column(scale=2):
                format_info_box = gr.Textbox(
                    label="📋 检测到的数据格式",
                    placeholder="上传文件后自动检测格式...",
                    interactive=False,
                    lines=2
                )
                
                dataset_stats = gr.JSON(
                    label="📈 数据统计信息",
                    value={"状态": "请上传数据或加载模板"}
                )
        
        # ========== 第二行：智能推荐配置（NEW!）==========
        gr.Markdown("### 🤖 智能训练推荐")
        
        with gr.Group(visible=False) as recommendation_box:
            rec_type_text = gr.Textbox(
                label="数据类型",
                interactive=False
            )
            
            rec_task_text = gr.Textbox(
                label="推荐任务类型",
                interactive=False
            )
            
            with gr.Row():
                rec_batch = gr.Number(label="Batch Size", interactive=False)
                rec_lr = gr.Number(label="Learning Rate", interactive=False)
                rec_epochs = gr.Number(label="Epochs", interactive=False)
                rec_seq_len = gr.Number(label="Max Sequence Length", interactive=False)
            
            rec_time = gr.Textbox(label="⏱️ 预计训练时间", interactive=False)
            rec_reason = gr.Textbox(
                label="💡 推荐理由",
                interactive=False,
                lines=2
            )
            
            with gr.Row():
                use_rec_btn = gr.Button("🚀 使用推荐配置并开始训练", variant="primary")
                manual_config_btn = gr.Button("⚙️ 手动调整参数", variant="secondary")
        
        # ========== 第三行：数据预览 ==========
        gr.Markdown("### 👁️ 数据预览")
        preview_table = gr.Dataframe(
            label="数据预览（前 100 条）",
            wrap=True,
            max_height=400
        )
        
        # ========== 第四行：数据集划分 ==========
        gr.Markdown("### ✂️ 数据集划分")
        with gr.Row():
            train_ratio_slider = gr.Slider(
                minimum=0.5,
                maximum=0.95,
                value=0.9,
                step=0.05,
                label="训练集比例"
            )
            
            split_btn = gr.Button("划分数据集", variant="primary")
        
        # ========== 第五行：保存和操作 ==========
        gr.Markdown("### 💾 保存与操作")
        with gr.Row():
            save_format = gr.Radio(
                choices=["jsonl", "csv"],
                value="jsonl",
                label="保存格式"
            )
            
            save_train_btn = gr.Button("💾 保存训练集", variant="primary")
            save_eval_btn = gr.Button("💾 保存评估集", variant="primary")
            clear_btn = gr.Button("🗑️ 清空数据", variant="stop")
        
        # 结果显示
        save_result = gr.Textbox(
            label="操作结果",
            interactive=False,
            visible=False
        )
        
        # ========== 回调函数 ==========
        
        def on_file_uploaded(file):
            """处理文件上传并显示智能推荐"""
            if file is None:
                return (
                    None, {"状态": "未检测到文件"}, "请上传数据或加载模板",
                    gr.update(visible=False), [], [], [], [], "", ""
                )
            
            try:
                # 加载数据
                df = data_processor.load_file(file.name)
                
                # 检测格式
                format_type = data_processor.detect_format(df)
                format_desc = data_processor.get_format_description(format_type)
                
                # 计算统计
                stats = data_processor.compute_statistics(df)
                
                # 生成预览
                preview = data_processor.preview_data(df, limit=100)
                
                # 智能推荐配置
                rec = training_advisor.analyze_and_recommend(
                    data_stats=stats,
                    data_format=format_type,
                    num_samples=len(df)
                )
                
                # 保存到状态
                state_manager.set_loaded_dataset(df, preview, stats)
                
                logger.info(f"数据文件已加载：{len(df)} 条样本，格式：{format_type}")
                
                # 返回推荐配置
                rec_values = [
                    gr.update(visible=True),
                    f"{format_type} ({format_desc})",
                    rec.task_type,
                    rec.batch_size,
                    rec.learning_rate,
                    rec.epochs,
                    rec.max_seq_length,
                    rec.estimated_time,
                    rec.reason
                ]
                
                return (
                    preview,
                    stats,
                    f"格式：{format_type}\n{format_desc}",
                    *rec_values
                )
                
            except Exception as e:
                logger.error(f"文件加载失败：{e}")
                return (
                    [],
                    {"错误": str(e)},
                    f"格式：unknown\n文件加载失败：{e}",
                    gr.update(visible=False),
                    [], [], [], [], "", ""
                )
        
        def on_template_selected(template_type, num_samples):
            """处理模板加载"""
            try:
                # 生成模板数据
                df = data_processor.generate_template(template_type, num_samples)
                
                # 检测格式
                format_type = data_processor.detect_format(df)
                format_desc = data_processor.get_format_description(format_type)
                
                # 计算统计
                stats = data_processor.compute_statistics(df)
                
                # 生成预览
                preview = data_processor.preview_data(df, limit=100)
                
                # 智能推荐配置
                rec = training_advisor.analyze_and_recommend(
                    data_stats=stats,
                    data_format=format_type,
                    num_samples=len(df)
                )
                
                # 保存到状态
                state_manager.set_loaded_dataset(df, preview, stats)
                
                logger.info(f"模板数据已加载：{template_type}, {len(df)} 条样本")
                
                rec_values = [
                    gr.update(visible=True),
                    f"{format_type} (模板)",
                    rec.task_type,
                    rec.batch_size,
                    rec.learning_rate,
                    rec.epochs,
                    rec.max_seq_length,
                    rec.estimated_time,
                    rec.reason
                ]
                
                return (
                    preview,
                    stats,
                    f"格式：{format_type}\n{format_desc} (模板)",
                    *rec_values
                )
                
            except Exception as e:
                logger.error(f"模板加载失败：{e}")
                return (
                    [],
                    {"错误": str(e)},
                    f"格式：unknown\n模板加载失败：{e}",
                    gr.update(visible=False),
                    [], [], [], [], "", ""
                )
        
        def on_start_one_click_training():
            """一键开始训练"""
            dataset_state = state_manager.get_dataset_state()
            
            if dataset_state['loaded_dataset'] is None:
                gr.Warning("请先上传数据或加载模板")
                return "❌ 错误：没有可训练的数据集"
            
            try:
                # 检查是否有模型
                if not model_manager.has_model():
                    # 自动创建默认模型
                    from ..utils import ModelConfig
                    config = ModelConfig(
                        model_name="auto_created_model",
                        model_type="bio_generator"
                    )
                    model_manager.create_model(config)
                
                # 获取推荐配置
                stats = dataset_state['statistics']
                format_type = data_processor.detect_format(dataset_state['loaded_dataset'])
                rec = training_advisor.analyze_and_recommend(
                    data_stats=stats,
                    data_format=format_type,
                    num_samples=len(dataset_state['loaded_dataset'])
                )
                
                # 保存临时数据集
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_dir = f"./datasets/temp_{timestamp}"
                os.makedirs(temp_dir, exist_ok=True)
                
                train_path = data_processor.save_dataset(
                    dataset_state['train_dataset'] if dataset_state['train_dataset'] is not None else dataset_state['loaded_dataset'],
                    f"{temp_dir}/train",
                    "jsonl"
                )
                
                # 开始训练
                train_config = {
                    'batch_size': rec.batch_size,
                    'num_epochs': rec.epochs,
                    'learning_rate': rec.learning_rate,
                    'max_seq_length': rec.max_seq_length,
                    'save_steps': 50
                }
                
                success = model_manager.start_training(
                    dataset_path=train_path,
                    config=train_config
                )
                
                if success:
                    return f"✅ 训练已开始\n配置：Batch={rec.batch_size}, LR={rec.learning_rate}, Epochs={rec.epochs}\n预计时间：{rec.estimated_time}"
                else:
                    return "❌ 训练启动失败"
                
            except Exception as e:
                logger.error(f"一键训练失败：{e}", exc_info=True)
                return f"❌ 错误：{e}"
        
        def on_split_dataset(train_ratio):
            """处理数据集划分"""
            dataset_state = state_manager.get_dataset_state()
            
            if dataset_state['loaded_dataset'] is None:
                gr.Warning("请先上传数据或加载模板")
                return "❌ 错误：没有可划分的数据集"
            
            try:
                import pandas as pd
                df = dataset_state['loaded_dataset']
                
                # 划分数据集
                train_df, eval_df = data_processor.split_dataset(df, train_ratio)
                
                # 保存划分结果
                state_manager.set_split_datasets(train_df, eval_df)
                
                train_msg = f"训练集：{len(train_df)} 条 ({train_ratio*100:.0f}%)"
                eval_msg = f"评估集：{len(eval_df)} 条 ({(1-train_ratio)*100:.0f}%)"
                
                logger.info(f"数据集已划分：{train_msg}, {eval_msg}")
                
                return f"✅ 数据集划分成功\n{train_msg}\n{eval_msg}"
                
            except Exception as e:
                logger.error(f"数据集划分失败：{e}")
                return f"❌ 错误：{e}"
        
        def on_save_dataset(save_format):
            """保存数据集"""
            dataset_state = state_manager.get_dataset_state()
            
            if dataset_state['train_dataset'] is None:
                gr.Warning("请先划分数据集")
                return "❌ 错误：请先划分数据集"
            
            try:
                import os
                from datetime import datetime
                
                # 创建保存目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = f"./datasets/{timestamp}"
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存训练集
                train_path = data_processor.save_dataset(
                    dataset_state['train_dataset'],
                    f"{save_dir}/train",
                    save_format
                )
                
                # 保存评估集
                eval_path = data_processor.save_dataset(
                    dataset_state['eval_dataset'],
                    f"{save_dir}/eval",
                    save_format
                )
                
                logger.info(f"数据集已保存：{train_path}, {eval_path}")
                
                return f"✅ 数据集已保存\n训练集：{train_path}\n评估集：{eval_path}"
                
            except Exception as e:
                logger.error(f"保存数据集失败：{e}")
                return f"❌ 错误：{e}"
        
        def on_clear_dataset():
            """清空数据集"""
            state_manager.clear_dataset()
            return (
                [],
                {"状态": "数据已清空"},
                "请上传数据或加载模板",
                gr.update(visible=False),
                "", "", 0, 0, 0, 0, "", ""
            )
        
        # 绑定事件
        file_upload.change(
            fn=on_file_uploaded,
            inputs=[file_upload],
            outputs=[
                preview_table, dataset_stats, format_info_box,
                recommendation_box,
                rec_type_text, rec_task_text, rec_batch, rec_lr, rec_epochs,
                rec_seq_len, rec_time, rec_reason
            ]
        )
        
        load_template_btn.click(
            fn=on_template_selected,
            inputs=[template_dropdown, template_samples],
            outputs=[
                preview_table, dataset_stats, format_info_box,
                recommendation_box,
                rec_type_text, rec_task_text, rec_batch, rec_lr, rec_epochs,
                rec_seq_len, rec_time, rec_reason
            ]
        )
        
        use_rec_btn.click(
            fn=on_start_one_click_training,
            outputs=[save_result]
        )
        
        split_btn.click(
            fn=on_split_dataset,
            inputs=[train_ratio_slider],
            outputs=[save_result]
        )
        
        save_train_btn.click(
            fn=lambda fmt: on_save_dataset(fmt),
            inputs=[save_format],
            outputs=[save_result]
        )
        
        save_eval_btn.click(
            fn=lambda fmt: on_save_dataset(fmt),
            inputs=[save_format],
            outputs=[save_result]
        )
        
        clear_btn.click(
            fn=on_clear_dataset,
            outputs=[preview_table, dataset_stats, format_info_box, recommendation_box, rec_type_text, rec_task_text, rec_batch, rec_lr, rec_epochs, rec_seq_len, rec_time, rec_reason]
        )
