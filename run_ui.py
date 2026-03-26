#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioTransLM UI 快速启动脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.app import launch_app

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BioTransLM 可视化训练与对话系统")
    print("=" * 60)
    print()
    print("正在启动服务器...")
    print()
    print("访问地址：http://127.0.0.1:7860")
    print()
    print("功能说明:")
    print("  💬 对话 - 与 BioTransLM 进行自然语言交互")
    print("  🔍 处理过程查看 - 查看模型内部处理链路")
    print("  📈 训练监控 - 实时监控训练进度和损失曲线")
    print("  📊 数据集管理 - 上传和管理训练数据")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    launch_app(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
