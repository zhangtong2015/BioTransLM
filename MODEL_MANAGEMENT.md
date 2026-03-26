# BioTransLM 模型管理完全指南

## 📖 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [创建模型](#创建模型)
4. [训练模型](#训练模型)
5. [保存和加载](#保存和加载)
6. [高级功能](#高级功能)
7. [常见问题](#常见问题)

---

## 概述

### 什么是模型管理？

模型管理模块提供了完整的模型生命周期管理：

- **创建模型** - 配置并初始化 BioTransLM 模型
- **训练模型** - 使用自定义数据集进行训练
- **保存模型** - 将训练好的模型保存到磁盘
- **加载模型** - 从磁盘加载预训练模型
- **导出模型** - 导出为推理优化的格式

### 核心组件

```python
# ModelConfig - 模型配置
config = ModelConfig(
    model_name="my_model",
    vocab_size=50257,
    hidden_dim=768,
    n_columns=4096
)

# ModelManager - 模型管理器（单例）
manager = get_model_manager()
manager.create_model(config)
manager.train_model(dataset_path, train_config)
manager.save_model("./models/my_model")
```

---

## 快速开始

### 5 分钟创建并训练第一个模型

#### 步骤 1：启动应用

```bash
python run_ui.py
```

浏览器打开 http://127.0.0.1:7860

#### 步骤 2：创建模型

1. 点击 **"🧠 模型管理"** 标签页
2. 展开 **"创建新模型"**
3. 填写配置：
   - 模型名称：`my_first_model`
   - 模型类型：`BioGenerator`
   - 隐藏层维度：`768`
   - HTM 列数量：`4096`
4. 点击 **"✨ 创建新模型"**

#### 步骤 3：准备数据

1. 切换到 **"📊 数据集管理"** 标签页
2. 选择模板：**"一问一答"**
3. 点击 **"加载模板数据"**
4. 调整训练集比例：`0.9`
5. 点击 **"划分数据集"**
6. 点击 **"💾 保存训练集"**

#### 步骤 4：开始训练

1. 回到 **"🧠 模型管理"** 标签页
2. 在 **"训练模型"** 部分：
   - 数据集路径：输入刚才保存的路径
   - Batch Size: `16`
   - Epochs: `5`（测试用，正式训练建议 10-20）
   - Learning Rate: `0.0001`
3. 点击 **"▶️ 开始训练"**
4. 观察训练进度和损失变化

#### 步骤 5：保存模型

训练完成后：
1. 输入模型名称：`my_first_model_final`
2. 点击 **"💾 保存模型"**
3. 模型保存到 `./models/my_first_model_final/`

---

## 创建模型

### 方法一：通过 UI 创建

#### 基础配置参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| 模型名称 | biotranslm_base | - | 模型的标识名称 |
| 模型类型 | bio_generator | - | BioGenerator 或 Orchestrator |
| 词汇表大小 | 50257 | 1000-100000 | Token 数量 |
| 隐藏层维度 | 768 | 256-2048 | 表示向量维度 |
| HTM 列数量 | 4096 | 1024-8192 | 空间池化器列数 |
| SDR 维度 | 4096 | 1024-8192 | 稀疏分布式表示维度 |
| 最大序列长度 | 512 | 64-1024 | 输入文本最大长度 |
| 计算设备 | auto | auto/cpu/cuda | 运行设备 |

#### 模型类型选择

**BioGenerator (推荐新手)**
- 纯生物启发式生成器
- 基于 HTM 时序记忆
- SDR 稀疏表示
- 适合文本生成任务
- 参数量适中（~50M）

**Orchestrator (完整系统)**
- 包含所有模块的完整系统
- 感觉门控 + 多粒度编码
- HTM 皮层 + 神经调节
- 双系统推理 + 记忆系统
- 适合复杂推理任务
- 参数量较大（~200M）

### 方法二：通过代码创建

```python
from ui.utils import ModelConfig, get_model_manager

# 创建配置
config = ModelConfig(
    model_name="custom_model",
    model_type="bio_generator",
    vocab_size=30000,      # 中文可以小一些
    hidden_dim=1024,       # 更大的隐藏层
    n_columns=8192,        # 更多的 HTM 列
    sdr_size=8192,
    max_sequence_length=1024,
    device="cuda"          # 使用 GPU
)

# 创建模型
manager = get_model_manager()
success = manager.create_model(config)

if success:
    print("模型创建成功！")
    info = manager.get_model_info()
    print(f"参数量：{info['parameter_count']:,}")
```

### 方法三：从配置文件加载

**config.json**
```json
{
  "model_name": "my_model",
  "model_type": "bio_generator",
  "vocab_size": 50257,
  "hidden_dim": 768,
  "n_columns": 4096,
  "sdr_size": 4096,
  "max_sequence_length": 512,
  "device": "auto"
}
```

```python
config = ModelConfig.load("config.json")
manager.create_model(config)
```

---

## 训练模型

### 训练前准备

#### 1. 数据集要求

**格式要求**：
- JSONL（推荐）：每行一个 JSON 对象
- CSV：包含表头的 CSV 文件
- TXT：每行一个样本

**内容要求**：
- 至少 1000 条样本（建议 10000+）
- 文本长度适中（平均 50-200 字）
- 质量高、噪声少

#### 2. 硬件要求

| 配置 | 最低要求 | 推荐配置 | 理想配置 |
|------|----------|----------|----------|
| CPU | 4 核 | 8 核 | 16 核 |
| 内存 | 8GB | 16GB | 32GB |
| GPU | 无 | GTX 1060 6GB | RTX 3090 |
| 存储 | 10GB | 50GB | 100GB SSD |

### 训练参数配置

#### 基础参数

```python
train_config = {
    "batch_size": 16,        # 批次大小
    "num_epochs": 10,        # 训练轮数
    "learning_rate": 0.0001, # 初始学习率
    "max_seq_length": 128,   # 序列长度
    "save_steps": 100        # 保存间隔
}
```

#### 参数调优指南

**Batch Size**
- 小数据集 ( < 10k): 8-16
- 中等数据集：16-32
- 大数据集 (> 100k): 32-64
- GPU 显存足够可以调大

**Learning Rate**
- 初始训练：1e-4 ~ 5e-4
- 微调：1e-5 ~ 5e-5
- 使用学习率调度器自动调整

**Num Epochs**
- 小数据集：20-50（容易过拟合）
- 中等数据集：10-20
- 大数据集：5-10

**Max Sequence Length**
- 短文本（微博、评论）：64-128
- 中等文本（问答、段落）：128-256
- 长文本（文章、故事）：256-512

### 监控训练过程

#### 关键指标

**损失曲线**
- `total_loss`: 总损失（应该持续下降）
- `lm_loss`: 语言模型损失（主要损失）
- `sparsity_loss`: 稀疏性损失（鼓励 2% 激活率）
- `temporal_loss`: 时序连续性损失（平滑性）

**HTM 指标**
- `prediction_accuracy`: 时序记忆预测准确率（应该上升）
- `burst_rate`: Burst 列比例（应该下降）
- `active_columns`: 激活列数（应该稳定在 2% 左右）

#### 正常训练的特征

✅ **健康训练**：
- 总损失稳步下降
- 预测准确率逐渐上升
- Burst 率从高到低然后稳定
- 没有剧烈的损失波动

❌ **异常训练**：
- 损失不下降或上升 → 学习率太高
- 损失剧烈震荡 → batch size 太小
- 预测准确率很低 → 模型容量不足
- 显存溢出 → batch size 太大

### 训练技巧

#### 1. 热身训练（Warmup）

先用小学习率训练几个 epoch：

```python
# 前 2 个 epoch
config = {
    "learning_rate": 1e-5,
    "num_epochs": 2
}

# 然后正常训练
config = {
    "learning_rate": 1e-4,
    "num_epochs": 10
}
```

#### 2. 课程学习（Curriculum Learning）

从简单到复杂：

```python
# 第一阶段：短序列
config = {
    "max_seq_length": 64,
    "num_epochs": 5
}

# 第二阶段：中等序列
config = {
    "max_seq_length": 128,
    "num_epochs": 5
}

# 第三阶段：完整序列
config = {
    "max_seq_length": 256,
    "num_epochs": 5
}
```

#### 3. 混合精度训练（GPU 用户）

如果支持 AMP：

```python
# 启用混合精度
use_amp = True  # 节省显存，加速训练
```

---

## 保存和加载

### 保存模型

#### 方法一：UI 保存

1. 在「模型管理」标签页
2. 输入保存路径（可选，留空使用默认路径）
3. 输入模型名称
4. 点击 **"💾 保存模型"**

#### 方法二：代码保存

```python
# 保存到默认目录（./models/）
saved_path = manager.save_model(name="my_model")

# 保存到指定目录
saved_path = manager.save_model(
    save_path="./exports/my_export",
    name="exported_model"
)

print(f"模型已保存到：{saved_path}")
```

#### 保存的内容

目录格式的模型包含：

```
my_model_20260326_174500/
├── config.json          # 模型配置
├── model.pt            # 模型权重
└── model_info.json     # 模型元信息
```

**config.json**:
```json
{
  "model_name": "my_model",
  "model_type": "bio_generator",
  "vocab_size": 50257,
  "hidden_dim": 768,
  ...
}
```

**model_info.json**:
```json
{
  "created_at": 1711446000,
  "parameter_count": 52428800,
  "training_dataset": "my_dataset",
  "training_epochs": 10,
  "final_loss": 0.85
}
```

### 加载模型

#### 方法一：UI 加载

1. 在「模型管理」标签页
2. 从下拉菜单选择已保存的模型
3. 点击 **"📂 加载模型"**

#### 方法二：代码加载

```python
# 从目录加载
success = manager.load_model("./models/my_model_20260326_174500")

# 从 .pt 文件加载
success = manager.load_model("./models/checkpoint_step_500.pt")

if success:
    print("模型加载成功！")
    info = manager.get_model_info()
    print(f"模型名称：{info['info']['model_name']}")
```

### 导出推理模型

导出为轻量级格式（仅用于推理）：

```python
from training.trainer import BioTrainer

# 假设已经有 trainer 实例
trainer.export_model("./inference_model.pt")
```

---

## 高级功能

### 1. 模型微调（Fine-tuning）

在预训练模型基础上微调：

```python
# 1. 加载预训练模型
manager.load_model("./models/pretrained_model")

# 2. 使用新数据集训练
manager.start_training(
    dataset_path="./my_custom_data.jsonl",
    config={
        "learning_rate": 1e-5,  # 小学习率
        "num_epochs": 5,         # 少量 epoch
        "batch_size": 16
    }
)
```

### 2. 模型合并（实验性）

合并多个模型的权重：

```python
def merge_models(model_paths, output_path, weights=None):
    """
    合并多个模型
    
    Args:
        model_paths: 模型路径列表
        output_path: 输出路径
        weights: 每个模型的权重（默认平均）
    """
    import torch
    
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    
    merged_state = None
    
    for path, weight in zip(model_paths, weights):
        checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint['model_state_dict']
        
        if merged_state is None:
            merged_state = {k: v * weight for k, v in state.items()}
        else:
            for k, v in state.items():
                merged_state[k] += v * weight
    
    # 保存合并后的模型
    torch.save({'model_state_dict': merged_state}, output_path)
    print(f"模型已合并到：{output_path}")
```

### 3. 批量评估

评估多个模型的性能：

```python
def evaluate_multiple_models(model_paths, eval_dataset):
    """批量评估模型"""
    results = []
    
    for path in model_paths:
        manager.load_model(path)
        model = manager.get_model()
        
        # 在评估集上测试
        loss = evaluate_on_dataset(model, eval_dataset)
        
        results.append({
            'model': path,
            'loss': loss
        })
    
    return results
```

### 4. 自动化超参数搜索

```python
import itertools

def grid_search_hyperparams(base_config, param_grid, dataset_path):
    """网格搜索最优超参数"""
    
    best_loss = float('inf')
    best_config = None
    
    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for i, config in enumerate(combinations):
        print(f"\n=== 测试配置 {i+1}/{len(combinations)} ===")
        print(config)
        
        # 创建新模型
        manager.reset()
        manager.create_model(base_config)
        
        # 训练
        full_config = {**base_config, **config}
        manager.start_training(dataset_path, full_config)
        
        # 等待训练完成
        while manager.is_training:
            time.sleep(10)
        
        # 获取最终损失
        final_loss = manager.get_model_info()['info'].get('final_loss', float('inf'))
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_config = config
        
        print(f"Loss: {final_loss:.4f} (当前最佳：{best_loss:.4f})")
    
    return best_config, best_loss

# 使用示例
param_grid = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size": [8, 16, 32],
    "num_epochs": [5, 10]
}

best_config, best_loss = grid_search_hyperparams(
    base_config=ModelConfig().to_dict(),
    param_grid=param_grid,
    dataset_path="./data.jsonl"
)

print(f"\n最佳配置：{best_config}")
print(f"最佳损失：{best_loss:.4f}")
```

---

## 常见问题

### Q1: 训练时显存不足怎么办？

**解决方案**：
1. 减小 batch size（最有效）
2. 减小 max_seq_length
3. 减小 hidden_dim 或 n_columns
4. 使用 CPU 模式（慢但不会 OOM）
5. 梯度累积（模拟大 batch）

```python
# CPU 模式
config = ModelConfig(device="cpu")

# 或者
config = {
    "batch_size": 4,      # 最小
    "max_seq_length": 64  # 较短
}
```

### Q2: 训练损失不下降？

**可能原因**：
- 学习率太低 → 增大 learning_rate
- 学习率太高 → 减小 learning_rate
- 数据集有问题 → 检查数据质量
- 模型容量不足 → 增大 hidden_dim

**调试步骤**：
1. 检查数据是否正确加载
2. 尝试不同的学习率（1e-5 ~ 1e-3）
3. 查看单个 batch 的过拟合情况
4. 检查梯度是否正常

### Q3: 如何中断训练？

**方法**：
1. UI 中点击 **"⏹️ 停止训练"**
2. 代码中调用 `manager.stop_training()`
3. Ctrl+C（强制终止，可能损坏模型）

**注意**：正常停止会自动保存检查点

### Q4: 模型保存在哪里？

**默认位置**：
```
./models/
├── biotranslm_base_20260326_174500/
├── my_model/
└── checkpoints/
    ├── checkpoint_step_100.pt
    └── checkpoint_step_200.pt
```

### Q5: 如何分享模型？

**打包分享**：
```bash
# 压缩整个模型目录
zip -r my_model.zip ./models/my_model/

# 或者只保存必要文件
cp ./models/my_model/config.json ./
cp ./models/my_model/model.pt ./
zip my_model_share.zip config.json model.pt
```

**上传到 HuggingFace**：
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./models/my_model",
    repo_id="your-username/biotranslm-model",
    repo_type="model"
)
```

### Q6: 训练需要多长时间？

**估算公式**：
```
时间 ≈ (样本数 / batch_size) × epochs × 每步时间
```

**经验值**：
- 10k 样本，batch=16，10 epochs: ~30 分钟（CPU）
- 10k 样本，batch=16，10 epochs: ~5 分钟（GPU）
- 100k 样本，batch=32，10 epochs: ~5 小时（CPU）
- 100k 样本，batch=32，10 epochs: ~50 分钟（GPU）

---

## 下一步

完成模型训练后，你可以：

1. **对话测试** - 在「💬 对话」标签页测试模型效果
2. **过程分析** - 在「🔍 处理过程查看」分析内部机制
3. **继续优化** - 调整参数重新训练
4. **部署应用** - 导出模型用于生产环境

---

**文档版本**: v2.0  
**最后更新**: 2026-03-26  
**适用版本**: BioTransLM UI v2.0+
