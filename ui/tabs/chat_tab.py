#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对话界面标签页 - 聊天对话框和参数调节
"""

import gradio as gr
from typing import Dict, Any, List, Tuple
import logging
import time

from ..utils import StateManager

logger = logging.getLogger(__name__)


def create_chat_tab(state_manager: StateManager):
    """创建对话界面标签页 - 豆包风格"""
    
    with gr.Tab("💬 对话", id="chat_tab"):
        # 简洁的头部
        gr.Markdown(
            """
            # 🤖 BioTransLM 智能助手
            
            基于生物启发式 HTM 理论的 AI 对话系统
            """,
            elem_classes=["centered-header"]
        )
        
        # ========== 对话历史（占满大部分空间）==========
        chatbot = gr.Chatbot(
            height=550,
            show_label=False,
            avatar_images=(
                None, 
                "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
            ),
            placeholder="开始对话吧...",
            layout="bubble"  # 气泡布局，更像聊天软件
        )
        
        # ========== 输入区域（固定在底部）==========
        with gr.Row(elem_classes=["input-row"]):
            msg_input = gr.Textbox(
                placeholder="请输入消息...",
                show_label=False,
                container=False,
                scale=8,
                lines=1,
                max_lines=4
            )
            send_btn = gr.Button(
                "发送",
                variant="primary",
                scale=1,
                size="lg"
            )
        
        # ========== 底部工具栏 ==========
        with gr.Row(elem_classes=["toolbar"]):
            clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="sm")
            
            # 模型状态（简化显示）
            model_status = gr.Markdown("⚪ 未加载模型", elem_classes=["status-indicator"])
            
            refresh_btn = gr.Button("🔄 刷新", variant="secondary", size="sm")
        
        # ========== 高级参数（折叠，默认隐藏）==========
        with gr.Accordion("⚙️ 高级设置", open=False, elem_classes=["advanced-settings"]):
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                    label="创造性"
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1,
                    label="Top-K"
                )
            
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=10, maximum=500, value=200, step=10,
                    label="最大长度"
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.1, step=0.1,
                    label="重复惩罚"
                )
        
        # 隐藏的状态文本
        status_text = gr.Textbox(visible=False)
        
        # ========== 回调函数 ==========
        
        def refresh_model_status():
            """刷新模型状态"""
            try:
                from ui.utils import ModelManager
                manager = ModelManager()
                
                if manager.has_model():
                    info = manager.get_model_info()
                    model_name = info.get('model_name', 'Unknown')
                    model_type = info.get('model_type', 'Unknown')
                    param_count = info.get('parameter_count', 0)
                    
                    status = f"✅ 已加载：{model_name} ({model_type})\n参数量：{param_count:,}"
                    return status
                else:
                    return "❌ 未加载模型，请先在「🧠 模型管理」创建或加载模型"
            except Exception as e:
                return f"❌ 读取失败：{str(e)}"
        
        def initialize_orchestrator():
            """懒加载初始化 Orchestrator"""
            try:
                from orchestrator import Orchestrator, OrchestratorConfig
                import torch
                
                # 先检查是否有模型
                from ui.utils import ModelManager
                manager = ModelManager()
                if not manager.has_model():
                    logger.warning("没有可用模型，尝试自动加载...")
                    # 尝试加载默认模型
                    try:
                        manager.load_latest_model()
                    except:
                        pass
                
                # 自动检测设备
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                config = OrchestratorConfig(
                    debug_mode=True,
                    device=device
                )
                
                orchestrator = Orchestrator(config)
                logger.info(f"Orchestrator 初始化成功，使用设备：{device}")
                return orchestrator
                
            except Exception as e:
                logger.error(f"Orchestrator 初始化失败：{e}")
                return None
        
        # 全局变量存储 orchestrator 实例
        orchestrator_instance = None
        
        def get_orchestrator():
            """获取或创建 orchestrator 实例"""
            nonlocal orchestrator_instance
            if orchestrator_instance is None:
                orchestrator_instance = initialize_orchestrator()
            return orchestrator_instance
        
        def on_message_submit(
            message: str,
            history: List[Dict],
            temperature: float,
            top_k: int,
            max_tokens: int,
            rep_penalty: float
        ):
            """处理用户消息"""
            
            if not message or not message.strip():
                return "", history
            
            # 添加用户消息到历史
            history.append({"role": "user", "content": message})
            
            # 获取 orchestrator
            orchestrator = get_orchestrator()
            
            if orchestrator is None:
                error_msg = "❌ 模型初始化失败，请检查日志"
                history.append({"role": "assistant", "content": error_msg})
                yield "", history
                return
            
            try:
                # 调用编排器进行推理
                results = orchestrator.forward(
                    message,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=max_tokens
                )
                
                response = results.get("generated_text", "抱歉，我无法生成响应")
                
                # 保存处理结果用于后续查看
                state_manager.add_message("user", message)
                state_manager.add_message("assistant", response, metadata=results)
                state_manager.set_processing_result(results)
                
                # 添加 AI 响应到历史
                history.append({"role": "assistant", "content": response})
                
                logger.info(f"对话：用户='{message[:50]}...' → AI='{response[:50]}...'")
                
            except Exception as e:
                error_msg = f"❌ 处理失败：{str(e)}"
                logger.error(f"对话处理错误：{e}", exc_info=True)
                history.append({"role": "assistant", "content": error_msg})
            
            yield "", history
        
        def on_clear_conversation():
            """清空对话"""
            state_manager.clear_conversations()
            return [], ""
        
        def on_export_conversation():
            """导出对话"""
            import json
            from datetime import datetime
            
            conversations = state_manager.list_conversations()
            if not conversations:
                return "❌ 没有可导出的对话"
            
            try:
                # 导出为 JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"./exports/conversation_{timestamp}.json"
                
                export_data = {
                    'export_time': timestamp,
                    'conversations': []
                }
                
                for conv_summary in conversations:
                    conv = state_manager.get_conversation(conv_summary['id'])
                    if conv:
                        export_data['conversations'].append({
                            'id': conv.id,
                            'messages': [
                                {
                                    'role': msg.role,
                                    'content': msg.content,
                                    'timestamp': msg.timestamp
                                }
                                for msg in conv.messages
                            ]
                        })
                
                import os
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                return f"✅ 对话已导出到：{filename}"
                
            except Exception as e:
                logger.error(f"导出对话失败：{e}")
                return f"❌ 导出失败：{e}"
        
        # 绑定事件
        send_btn.click(
            fn=on_message_submit,
            inputs=[msg_input, chatbot, temperature_slider, top_k_slider, max_tokens_slider, repetition_penalty],
            outputs=[msg_input, chatbot]
        )
        
        # 支持回车键发送
        msg_input.submit(
            fn=on_message_submit,
            inputs=[msg_input, chatbot, temperature_slider, top_k_slider, max_tokens_slider, repetition_penalty],
            outputs=[msg_input, chatbot]
        )
        
        clear_btn.click(
            fn=on_clear_conversation,
            outputs=[chatbot, status_text]
        )
        
        # 刷新模型状态
        refresh_btn.click(
            fn=refresh_model_status,
            outputs=[model_status]
        )
        
        # 页面加载时自动刷新模型状态
        gr.on(
            triggers=[],
            fn=refresh_model_status,
            outputs=[model_status]
        )
