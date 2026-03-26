#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioTransLM 可视化训练与对话系统 - 主应用入口
"""

import gradio as gr
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui.utils import StateManager
from ui.tabs import (
    create_chat_tab,
    create_training_tab,
    create_dataset_tab,
    create_inspection_tab,
    create_model_tab
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_ui():
    """创建 Gradio 界面"""
    
    # 初始化状态管理器
    state_manager = StateManager()
    
    # 创建主题
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="slate",
        font=(
            gr.themes.GoogleFont("Noto Sans SC"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif"
        )
    )
    
    # 创建 Blocks 布局
    with gr.Blocks(
        title="BioTransLM 可视化系统",
        analytics_enabled=False
    ) as app:
        
        # ========== 页面头部 ==========
        gr.Markdown(
            """
            # 🧬 BioTransLM 可视化训练与对话系统
            
            基于生物启发式 HTM 理论的语言模型 - 完全摒弃 Transformer 架构
            
            **核心特性**:
            - 💬 智能对话交互
            - 🔍 处理过程可视化
            - 📈 实时训练监控  
            - 📊 数据集管理
            """,
            elem_classes=["main-header"]
        )
        
        gr.HTML("<hr>")
        
        # ========== 标签页区域 ==========
        with gr.Tabs(selected=0):
            # 标签页 1: 模型管理（新增，放在最前面）
            create_model_tab(state_manager)
            
            # 标签页 2: 对话
            create_chat_tab(state_manager)
            
            # 标签页 3: 处理过程查看
            create_inspection_tab(state_manager)
            
            # 标签页 4: 训练监控
            create_training_tab(state_manager)
            
            # 标签页 5: 数据集管理
            create_dataset_tab(state_manager)
        
        gr.HTML("<hr>")
        
        # ========== 页面底部 ==========
        gr.Markdown(
            """
            ### ℹ️ 使用说明
            
            1. **创建模型**: 先在「🧠 模型管理」标签页创建或加载模型
            2. **准备数据**: 在「📊 数据集管理」标签页加载训练数据或使用内置模板
            3. **训练模型**: 在「🧠 模型管理」标签页配置训练参数并启动训练
            4. **对话交互**: 在「💬 对话」标签页与模型进行自然语言交互
            5. **查看过程**: 在「🔍 处理过程查看」标签页了解模型内部工作机制
            
            ---
            
            **核心功能**:
            - 🧠 **模型管理**: 创建、配置、训练、保存和加载模型
            - 💬 **智能对话**: 基于 BioTransLM 的自然语言交互
            - 🔍 **过程可视化**: HTM 激活、记忆检索、双系统推理展示
            - 📈 **训练监控**: 实时损失曲线和训练进度跟踪
            - 📊 **数据管理**: 多格式数据集支持和自动划分
            
            ---
            
            **技术栈**: BioTransLM + HTM Theory + Gradio
            
            **版本**: v2.0.0 (带模型管理) | **License**: MIT
            """
        )
    
    logger.info("UI 创建完成")
    return app


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
    **kwargs
):
    """
    启动 Gradio 应用
    
    Args:
        server_name: 服务器地址
        server_port: 服务器端口
        share: 是否创建公开分享链接
        inbrowser: 是否自动打开浏览器
        **kwargs: 其他传递给 app.launch() 的参数
    """
    app = create_ui()
    
    logger.info(f"启动服务器：http://{server_name}:{server_port}")
    
    # 定义 CSS 样式 - 豆包风格
    custom_css = """
        .gradio-container {
            max-width: 1200px !important;
            padding: 10px !important;
        }
        
        /* 居中头部 */
        .centered-header {
            text-align: center;
            margin-bottom: 10px !important;
            padding: 10px !important;
        }
        .centered-header h1 {
            font-size: 2em !important;
            margin: 0 !important;
        }
        
        /* 输入框样式 */
        .input-row {
            margin-top: 10px !important;
            margin-bottom: 10px !important;
        }
        .input-row .gradio-textbox {
            border-radius: 20px !important;
            border: 1px solid #e0e0e0 !important;
            padding: 10px 15px !important;
            font-size: 15px !important;
        }
        .input-row .gradio-button {
            border-radius: 20px !important;
            padding: 10px 25px !important;
            font-size: 15px !important;
            height: auto !important;
        }
        
        /* 工具栏样式 */
        .toolbar {
            justify-content: space-between !important;
            align-items: center !important;
            margin-top: 5px !important;
        }
        .status-indicator {
            font-size: 14px !important;
            color: #666 !important;
            margin: 0 !important;
        }
        
        /* 气泡布局优化 */
        .chatbot {
            border-radius: 15px !important;
            border: 1px solid #f0f0f0 !important;
        }
        
        /* 隐藏不必要的标签 */
        .gradio-label {
            display: none !important;
        }
        
        /* 高级设置 */
        .advanced-settings {
            margin-top: 15px !important;
            border-top: 1px solid #e0e0e0 !important;
            padding-top: 15px !important;
        }
    """
    
    # 获取主题（从 create_ui 中重新创建）
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="slate",
        font=(
            gr.themes.GoogleFont("Noto Sans SC"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif"
        )
    )
    
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
        theme=theme,
        css=custom_css,
        **kwargs
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BioTransLM 可视化训练与对话系统")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公开分享链接")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    
    args = parser.parse_args()
    
    launch_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser
    )
