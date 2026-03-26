# BioTransLM 可视化训练与对话系统

## 📋 项目简介

BioTransLM 是一个基于 HTM（Hierarchical Temporal Memory）理论的纯生物启发式语言模型，完全摒弃传统 Transformer 架构。本项目提供了可视化的训练和对话界面，让您能够：

- 💬 **智能对话** - 与模型进行自然语言交互
- 🔍 **过程可视化** - 深入了解模型内部处理机制
- 📈 **训练监控** - 实时查看训练进度和损失变化
- 📊 **数据管理** - 便捷地上传和管理训练数据集

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖（如果还没有安装）
pip install torch torchvision torchaudio

# UI 额外依赖
pip install -r requirements-ui.txt

# 或者手动安装
pip install gradio plotly networkx pandas
```

### 2. 启动应用

```bash
# 方式一：使用启动脚本
python run_ui.py

# 方式二：直接运行 ui 模块
cd ui
python app.py

# 方式三：自定义端口
python run_ui.py --port 8080 --no-browser
```

### 3. 访问界面

浏览器打开：**http://127.0.0.1:7860**

---

## 📖 功能说明

### 1. 💬 对话标签页

**功能**：
- 聊天式对话框，类似豆包/微信的交互体验
- 可调节生成参数（Temperature、Top-K、最大长度等）
- 支持导出对话记录为 JSON 格式
- 自动保存对话历史

**使用方法**：
1. 在输入框中输入问题或消息
2. 点击"发送"按钮或按回车键
3. 等待模型生成响应
4. 可在"处理过程查看"标签页查看详细推理过程

---

### 2. 🔍 处理过程查看标签页

**功能**：
- 分阶段展示 BioTransLM 的完整处理链路
- 可视化 HTM 空间池化器和时序记忆的激活模式
- 展示双系统推理（系统 1 直觉 vs 系统 2 推理）的决策过程
- 显示记忆系统（工作记忆、情景记忆、语义记忆）的检索结果

**处理阶段**：
1. **感觉门控** - 输入噪声过滤和特征选择
2. **多粒度编码** - 字符/词/短语/句子多层次表示
3. **HTM 系统** - 空间池化 + 时序记忆处理
4. **神经调节** - 基于预测误差的动态调节
5. **记忆系统** - 三层记忆检索与融合
6. **双系统推理** - 直觉联想与符号推理的协同
7. **响应生成** - 基于 HTM 序列预测的文本生成

---

### 3. 📈 训练监控标签页

**功能**：
- 实时损失曲线（总损失、LM 损失、稀疏性损失、时序损失）
- HTM 激活热图可视化
- 训练进度指示器
- 学习率调度曲线
- 检查点保存与加载

**使用方法**：
1. 先在"数据集管理"标签页加载训练数据
2. 配置训练参数（Epochs、Batch Size、Learning Rate 等）
3. 点击"开始训练"按钮
4. 实时监控训练过程和指标变化
5. 可随时暂停或停止训练

**支持的损失类型**：
- **语言模型损失** - 标准的 next-token 预测损失
- **稀疏性损失** - 鼓励 2% 的稀疏激活率
- **时序连续性损失** - 鼓励时序平滑变化

---

### 4. 📊 数据集管理标签页

**功能**：
- 支持多种数据格式上传（JSONL/CSV/TXT/JSON）
- 自动检测数据格式（DeepSeek-R1、问答、分类等）
- 数据统计分析（样本数、平均长度、缺失值等）
- 内置数据集模板（小说、蒸馏数据、一问一答）
- Train/Eval 自动划分
- 一键保存处理后的数据集

**支持的数据格式**：

#### DeepSeek-R1 蒸馏格式
```jsonl
{"input": "问题", "content": "答案", "reasoning_content": "推理过程"}
```

#### 问答格式
```jsonl
{"question": "问题", "answer": "答案"}
```

#### 分类格式
```jsonl
{"text": "文本内容", "label": "类别标签"}
```

#### 对话格式
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**使用方法**：
1. 拖拽文件到上传区域，或点击选择文件
2. 查看自动检测的数据格式和统计信息
3. 预览数据内容
4. 调整训练集/评估集划分比例
5. 点击"划分数据集"
6. 选择保存格式（JSONL/CSV），点击保存

---

## 🏗️ 系统架构

```
ui/
├── app.py                 # Gradio 主应用入口
├── tabs/
│   ├── chat_tab.py        # 对话界面
│   ├── training_tab.py    # 训练监控
│   ├── dataset_tab.py     # 数据集管理
│   └── inspection_tab.py  # 处理过程查看
├── components/
│   ├── htm_visualizer.py  # HTM 可视化
│   ├── loss_chart.py      # 损失曲线图
│   ├── memory_viewer.py   # 记忆查看器
│   └── reasoning_trace.py # 推理追踪器
└── utils/
    ├── state_manager.py   # 状态管理
    └── data_processor.py  # 数据处理
```

---

## ⚙️ 配置选项

### 对话参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| Temperature | 0.1-2.0 | 0.7 | 控制生成创造性（越高越随机） |
| Top-K | 1-100 | 50 | 采样时考虑的 top-k 个 token |
| 最大长度 | 10-500 | 100 | 生成的最大 token 数 |
| 重复惩罚 | 1.0-2.0 | 1.1 | 对重复 token 的惩罚强度 |

### 训练参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| Epochs | 1-50 | 10 | 训练轮数 |
| Batch Size | 4-64 | 16 | 批次大小 |
| Learning Rate | 1e-5~1e-3 | 1e-4 | 初始学习率 |
| 最大序列长度 | 32-512 | 128 | 输入序列最大长度 |

---

## 🔧 故障排除

### 常见问题

#### 1. 启动失败：`ModuleNotFoundError: No module named 'gradio'`

**解决方案**：
```bash
pip install -r requirements-ui.txt
```

#### 2. 对话无响应

**可能原因**：
- Orchestrator 初始化失败
- 模型文件缺失

**解决方案**：
- 检查控制台日志
- 确保项目根目录下的 `orchestrator.py` 存在
- 确认所有依赖已正确安装

#### 3. 训练时显存不足

**解决方案**：
- 减小 Batch Size
- 减小最大序列长度
- 使用 CPU 模式（修改 orchestrator 配置中的 `device="cpu"`）

#### 4. 数据上传后无法识别格式

**解决方案**：
- 检查文件格式是否符合支持的类型
- 确保 JSONL 文件每行是独立的 JSON 对象
- CSV 文件需要包含表头

---

## 📝 示例数据集

### 创建自定义数据集

**示例 1：小说数据集**

创建 `novel_data.jsonl`：
```jsonl
{"text": "第一章 开始\n很久很久以前...", "title": "第一章", "source": "我的小说"}
{"text": "第二章 发展\n故事继续...", "title": "第二章", "source": "我的小说"}
```

**示例 2：问答数据集**

创建 `qa_data.jsonl`：
```jsonl
{"question": "什么是 HTM？", "answer": "HTM 是 Hierarchical Temporal Memory 的缩写..."}
{"question": "BioTransLM 有什么特点？", "answer": "BioTransLM 完全摒弃了 Transformer 架构..."}
```

**示例 3：蒸馏数据集**

创建 `distillation.jsonl`：
```jsonl
{
  "input": "请解释相对论",
  "content": "相对论是爱因斯坦提出的物理理论...",
  "reasoning_content": "首先需要区分狭义相对论和广义相对论..."
}
```

---

## 🎯 下一步

- [ ] 添加更多预训练模型支持
- [ ] 实现模型对比功能
- [ ] 支持多用户会话
- [ ] 添加 API 接口
- [ ] 优化移动端适配

---

## 📄 许可证

MIT License

---

## 🙏 致谢

感谢所有为 BioTransLM 项目做出贡献的开发者和研究人员！

---

**项目主页**: https://github.com/your-repo/BioTransLM  
**问题反馈**: 请在 GitHub 提交 Issue  
**更新时间**: 2026-03-26
