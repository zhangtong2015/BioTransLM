# BioTransLM 可视化系统 - 快速参考卡

## 🚀 启动应用

```bash
# 安装依赖
pip install -r requirements-ui.txt

# 启动应用
python run_ui.py

# 访问：http://127.0.0.1:7860
```

---

## 📋 完整工作流程

### 1️⃣ 创建模型（🧠 模型管理）
```
模型名称 → biotranslm_base
模型类型 → BioGenerator
隐藏层 → 768
HTM 列 → 4096
点击「✨ 创建新模型」
```

### 2️⃣ 准备数据（📊 数据集管理）
```
选择模板 → 一问一答
加载模板数据
划分数据集（90% 训练）
保存训练集
```

### 3️⃣ 开始训练（🧠 模型管理）
```
数据集路径 → ./datasets/xxx/train.jsonl
Batch Size → 16
Epochs → 10
Learning Rate → 0.0001
点击「▶️ 开始训练」
```

### 4️⃣ 保存模型（🧠 模型管理）
```
训练完成后
输入模型名称 → my_model_final
点击「💾 保存模型」
```

### 5️⃣ 测试对话（💬 对话）
```
输入问题 → "你好"
发送 → 查看响应
```

---

## ⚙️ 推荐配置

### 小型模型（快速测试）
```yaml
vocab_size: 10000
hidden_dim: 512
n_columns: 2048
batch_size: 8
epochs: 5
learning_rate: 0.001
```

### 中型模型（日常使用）
```yaml
vocab_size: 50257
hidden_dim: 768
n_columns: 4096
batch_size: 16
epochs: 10
learning_rate: 0.0001
```

### 大型模型（生产级）
```yaml
vocab_size: 100000
hidden_dim: 1024
n_columns: 8192
batch_size: 32
epochs: 20
learning_rate: 0.00005
```

---

## 🔍 关键指标解读

### 损失曲线
- **总损失下降** → 训练正常
- **不下降** → 调大学习率
- **剧烈震荡** → 减小 batch size

### HTM 指标
- **预测准确率上升** → 模型在学习
- **Burst 率下降** → 时序记忆生效
- **激活率 ~2%** → 稀疏性正常

### 训练进度
- **Loss < 1.0** → 收敛良好
- **Accuracy > 80%** → 预测准确
- **稳定无震荡** → 超参数合适

---

## 💾 文件结构

```
BioTransLM/
├── ui/                          # UI 代码
│   ├── app.py                   # 主应用
│   ├── tabs/                    # 标签页
│   │   ├── model_tab.py         # 🧠 模型管理
│   │   ├── chat_tab.py          # 💬 对话
│   │   ├── inspection_tab.py    # 🔍 处理查看
│   │   └── training_tab.py      # 📈 训练监控
│   └── utils/                   # 工具类
│       ├── model_manager.py     # 模型管理器
│       └── state_manager.py     # 状态管理
├── models/                      # 保存的模型
│   ├── my_model/
│   │   ├── config.json
│   │   └── model.pt
│   └── checkpoints/
├── datasets/                    # 数据集
│   └── xxx/
│       ├── train.jsonl
│       └── eval.jsonl
└── run_ui.py                    # 启动脚本
```

---

## 🛠️ 常用命令

```bash
# 自定义端口
python run_ui.py --port 8080

# 不自动打开浏览器
python run_ui.py --no-browser

# 创建公开分享链接
python run_ui.py --share

# 运行测试
python test_ui_components.py
```

---

## 📊 数据集格式

### DeepSeek-R1 蒸馏
```jsonl
{"input": "问题", "content": "答案", "reasoning_content": "推理"}
```

### 问答
```jsonl
{"question": "什么是 AI？", "answer": "AI 是人工智能..."}
```

### 对话
```jsonl
{"messages": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"}
]}
```

---

## ⚠️ 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| 显存不足 | 减小 batch size 或 sequence length |
| 损失不下降 | 调整学习率（增大或减小） |
| 训练太慢 | 减小 max_seq_length，使用 GPU |
| 端口占用 | `python run_ui.py --port 8080` |
| 模型加载失败 | 检查文件是否完整 |

---

## 📚 文档索引

- **快速开始**: QUICKSTART.md
- **详细指南**: UI_README.md
- **模型管理**: MODEL_MANAGEMENT.md
- **项目总结**: V2_SUMMARY.md

---

## 🎯 核心功能一览

| 图标 | 功能 | 位置 |
|------|------|------|
| 🧠 | 模型管理 | 标签页 1 |
| 💬 | 智能对话 | 标签页 2 |
| 🔍 | 过程查看 | 标签页 3 |
| 📈 | 训练监控 | 标签页 4 |
| 📊 | 数据管理 | 标签页 5 |

---

**版本**: v2.0  
**更新**: 2026-03-26  
**许可**: MIT
