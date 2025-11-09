# Transformer从零实现

完整的Transformer模型实现，包含Encoder和Decoder，支持序列到序列任务和语言建模任务。

## 项目特点

- **完整实现**: Multi-head Self-Attention、Position-wise FFN、残差连接、Layer Normalization、位置编码
- **双架构**: 同时实现Encoder和Decoder，支持完整Seq2Seq任务
- **高级特性**: 相对位置编码、可学习位置编码、标签平滑、Warmup学习率调度
- **训练稳定性**: AdamW优化器、梯度裁剪、学习率预热、参数统计
- **完善工具**: 模型保存/加载、训练曲线可视化、消融实验
- **可重现性**: 固定随机种子，提供exact命令行参数

## 项目结构

```
transformer_project/
├── src/                          # 源代码
│   ├── attention.py              # 多头注意力机制（含相对位置编码）
│   ├── layers.py                 # 基础层（FFN、位置编码、LayerNorm等）
│   ├── encoder.py                # Transformer Encoder
│   ├── decoder.py                # Transformer Decoder
│   ├── transformer.py            # 完整Transformer模型
│   ├── dataset.py                # 数据集和词汇表
│   ├── train.py                  # 训练脚本
│   ├── utils.py                  # 工具函数
│   └── ablation_study.py         # 消融实验
├── scripts/                      # 运行脚本
│   ├── run.sh                    # Linux/Mac训练脚本
│   ├── run.bat                   # Windows训练脚本
│   └── run_ablation.bat          # 消融实验脚本
├── data/                         # 数据目录（自动生成）
├── checkpoints/                  # 模型检查点（自动生成）
├── results/                      # 实验结果（自动生成）
├── requirements.txt              # Python依赖
└── README.md                     # 本文件
```

## 环境要求

### 硬件要求

- **CPU**: 多核处理器（建议4核以上）
- **内存**: 8GB以上
- **GPU**: 可选，但强烈建议（NVIDIA GPU with CUDA support）
  - 小规模实验: GTX 1060 (6GB) 或以上
  - 完整消融实验: RTX 2060 (8GB) 或以上

### 软件要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (如使用GPU)

## 安装

```bash
# 克隆仓库（实际使用时替换为你的GitHub链接）
git clone https://github.com/yourusername/transformer_project.git
cd transformer_project

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 方法1: 使用脚本（推荐）

**Windows:**

```bash
# 训练复制任务
scripts\run.bat copy experiment_copy

# 训练序列反转任务
scripts\run.bat reverse experiment_reverse
```

**Linux/Mac:**

```bash
# 给脚本添加执行权限
chmod +x scripts/run.sh

# 训练复制任务
bash scripts/run.sh copy experiment_copy

# 训练序列反转任务
bash scripts/run.sh reverse experiment_reverse
```

### 方法2: 直接使用Python命令

```bash
cd src

# 基础训练（复制任务）
python train.py \
    --task copy \
    --num_samples 10000 \
    --seq_len 10 \
    --vocab_size 50 \
    --d_model 256 \
    --n_heads 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --d_ff 1024 \
    --dropout 0.1 \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 0.0001 \
    --warmup_steps 4000 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --seed 42 \
    --save_dir ../checkpoints/default

# 使用相对位置编码
python train.py --task copy --use_relative_position --save_dir ../checkpoints/relative_pe

# 使用可学习位置编码
python train.py --task copy --use_learnable_pe --save_dir ../checkpoints/learnable_pe
```

## 消融实验

运行完整的消融实验（测试不同配置的影响）：

**Windows:**

```bash
scripts\run_ablation.bat
```

**Linux/Mac:**

```bash
cd src
python ablation_study.py --experiments all --save_dir ../results/ablation
```

### 消融实验内容

1. **注意力头数**: 2, 4, 8头
2. **模型层数**: 2, 3, 4, 6层
3. **模型大小**: d_model=128/256/512
4. **位置编码**: Sinusoidal / Relative / Learnable

## 实验结果

训练完成后，会在保存目录生成以下文件：

```
checkpoints/experiment_name/
├── best_model.pt              # 最佳模型
├── checkpoint_epoch_N.pt      # 定期检查点
├── config.json                # 训练配置
├── history.json               # 训练历史
├── training_curves.png        # 训练曲线图
├── src_vocab.pkl             # 源词汇表
└── tgt_vocab.pkl             # 目标词汇表
```

消融实验结果：

```
results/ablation/
├── ablation_summary.csv       # 实验汇总表格
├── all_experiments.json       # 详细结果
└── *_comparison.png          # 对比图表
```

## 模型架构

### 核心组件

#### 1. Multi-Head Self-Attention

```
数学公式:
Attention(Q, K, V) = softmax(QK^T / √d_k)V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

实现要点：

- 缩放点积注意力
- 多头并行计算
- 线性投影和合并

#### 2. Position-wise Feed-Forward Network

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
或使用 GELU: FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

#### 3. 位置编码

```
正弦位置编码:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 4. Layer Normalization + 残差连接

```
LayerNorm(x + Sublayer(x))
```

### 模型参数

默认配置的参数量统计：

| 配置             | 参数量 |
| ---------------- | ------ |
| d_model=256, 3层 | ~4.5M  |
| d_model=512, 6层 | ~44M   |

## 数学推导与算法

### Scaled Dot-Product Attention

**输入**: 查询Q、键K、值V矩阵

**步骤**:

1. 计算注意力分数: `scores = QK^T / √d_k`
2. 应用softmax: `attention = softmax(scores)`
3. 加权求和: `output = attention · V`

**为什么缩放?**
当d_k很大时，点积的方差会很大，导致softmax进入梯度很小的区域，因此除以√d_k。

### 位置编码原理

Transformer没有循环或卷积结构，无法捕获序列顺序信息。位置编码通过给每个位置添加唯一的向量来解决这个问题。

**正弦位置编码的优点**:

- 可以外推到训练时未见过的序列长度
- 不同位置之间的相对位置关系可以通过三角函数性质表达

### 学习率预热（Warmup）

```python
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

**原理**:

1. 前warmup_steps步线性增长
2. 之后按step^(-0.5)衰减
3. 防止训练初期梯度过大导致不稳定

## 训练技巧

### 1. 标签平滑 (Label Smoothing)

防止模型过于自信，提高泛化能力。

```python
smoothed_label = (1 - ε) * true_label + ε / (vocab_size - 1)
```

### 2. 梯度裁剪 (Gradient Clipping)

防止梯度爆炸。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Dropout

在注意力权重和FFN输出后应用，防止过拟合。

### 4. AdamW优化器

带权重衰减的Adam，比标准Adam在Transformer上效果更好。

## 验证实验

### 复制任务 (Copy Task)

- **任务**: 输入一个序列，输出相同序列
- **难度**: 简单，用于验证模型基本功能
- **期望**: Loss应快速下降至接近0

### 序列反转任务 (Reverse Task)

- **任务**: 输入一个序列，输出其反转
- **难度**: 中等，需要模型学习长距离依赖
- **期望**: Loss应稳定下降，最终达到较低水平

### 训练曲线示例

正常训练应观察到：

- Loss持续下降
- 训练集和验证集Loss差距不大（无明显过拟合）
- Perplexity随Loss下降而降低

## 故障排查

### 问题1: Loss不下降

**可能原因**:

- 学习率过大或过小
- 模型太小无法拟合
- 数据有问题

**解决方案**:

- 调整学习率（尝试0.0001）
- 增加模型大小
- 检查数据预处理

### 问题2: Loss震荡

**可能原因**:

- 学习率过大
- Batch size过小

**解决方案**:

- 降低学习率
- 增大batch size或使用梯度累积

### 问题3: 过拟合

**可能原因**:

- 模型太大
- 训练数据太少
- Dropout太小

**解决方案**:

- 减小模型或增加数据
- 增大dropout
- 使用更强的正则化

## 重现实验的Exact命令

### 重要说明

为确保**完全可重现**的结果，所有实验使用**固定随机种子(seed=42)**。以下命令包含所有必需的超参数设置。

### 核心实验命令

#### 实验1: 复制任务基线（主实验）

**任务**: 输入随机整数序列，输出相同序列
**预期**: 训练Loss从~3.3降至<0.01，验证Loss从~2.7降至<0.01
**训练时间**: CPU约20-30分钟，GPU(RTX 3070)约3-5分钟

```bash
# Linux/Mac多行命令
python src/train.py \
    --task copy \
    --num_samples 10000 \
    --seq_len 10 \
    --vocab_size 50 \
    --d_model 256 \
    --n_heads 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --d_ff 1024 \
    --dropout 0.1 \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 0.0001 \
    --warmup_steps 4000 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --seed 42 \
    --save_dir checkpoints001

# Windows一行命令
python src/train.py --task copy --num_samples 10000 --seq_len 10 --vocab_size 50 --d_model 256 --n_heads 8 --num_encoder_layers 3 --num_decoder_layers 3 --d_ff 1024 --dropout 0.1 --batch_size 64 --num_epochs 50 --lr 0.0001 --warmup_steps 4000 --max_grad_norm 1.0 --label_smoothing 0.1 --seed 42 --save_dir checkpoints001
```

#### 实验2: 序列反转任务

**任务**: 输入随机序列，输出反转序列
**预期**: 训练Loss从~3.3降至<0.05（难度略高于复制）

```bash
# Windows一行命令
python src/train.py --task reverse --num_samples 10000 --seq_len 10 --vocab_size 50 --d_model 256 --n_heads 8 --num_encoder_layers 3 --num_decoder_layers 3 --d_ff 1024 --dropout 0.1 --batch_size 64 --num_epochs 50 --lr 0.0001 --warmup_steps 4000 --max_grad_norm 1.0 --label_smoothing 0.1 --seed 42 --save_dir checkpoints002
```

#### 实验3: 相对位置编码

**任务**: 使用相对位置编码替代标准正弦位置编码

```bash
# Windows一行命令
python src/train.py --task copy --num_samples 10000 --seq_len 10 --vocab_size 50 --d_model 256 --n_heads 8 --num_encoder_layers 3 --num_decoder_layers 3 --d_ff 1024 --dropout 0.1 --batch_size 64 --num_epochs 50 --lr 0.0001 --warmup_steps 4000 --max_grad_norm 1.0 --label_smoothing 0.1 --seed 42 --use_relative_position --save_dir checkpoints003
```

#### 实验4: 可学习位置编码

```bash
# Windows一行命令
python src/train.py --task copy --num_samples 10000 --seq_len 10 --vocab_size 50 --d_model 256 --n_heads 8 --num_encoder_layers 3 --num_decoder_layers 3 --d_ff 1024 --dropout 0.1 --batch_size 64 --num_epochs 50 --lr 0.0001 --warmup_steps 4000 --max_grad_norm 1.0 --label_smoothing 0.1 --seed 42 --use_learnable_pe --save_dir checkpoints004
```

### 消融实验命令

```bash
# 运行所有消融实验（约2-3小时GPU，6-8小时CPU）
python src/ablation_study.py --experiments all --save_dir results/ablation --seed 42

# 或分别运行各维度
python src/ablation_study.py --experiments heads --seed 42   # 注意力头数
python src/ablation_study.py --experiments layers --seed 42  # 模型层数
python src/ablation_study.py --experiments size --seed 42    # 模型大小
python src/ablation_study.py --experiments pe --seed 42      # 位置编码
```

### 超参数说明

| 参数                     | 值           | 说明                               |
| ------------------------ | ------------ | ---------------------------------- |
| `--seed`               | **42** | **随机种子（确保可重现性）** |
| `--task`               | copy/reverse | 任务类型                           |
| `--num_samples`        | 10000        | 训练样本数                         |
| `--seq_len`            | 10           | 序列长度                           |
| `--vocab_size`         | 50           | 词汇表大小                         |
| `--d_model`            | 256          | 模型维度                           |
| `--n_heads`            | 8            | 注意力头数                         |
| `--num_encoder_layers` | 3            | Encoder层数                        |
| `--num_decoder_layers` | 3            | Decoder层数                        |
| `--d_ff`               | 1024         | FFN中间层维度 (4×d_model)         |
| `--dropout`            | 0.1          | Dropout率                          |
| `--batch_size`         | 64           | 批次大小                           |
| `--num_epochs`         | 50           | 训练轮数                           |
| `--lr`                 | 0.0001       | 初始学习率                         |
| `--warmup_steps`       | 4000         | Warmup步数                         |
| `--max_grad_norm`      | 1.0          | 梯度裁剪阈值                       |
| `--label_smoothing`    | 0.1          | 标签平滑系数                       |

### 验证实验结果

```bash
# 查看最终训练Loss（应接近0）
python -c "import json; data=json.load(open('checkpoints001/history.json')); print(f'Final train loss: {data[\"train_loss\"][-1]:.4f}')"

# 查看训练曲线图
# Windows: start checkpoints001/training_curves.png
# Linux: xdg-open checkpoints001/training_curves.png
# Mac: open checkpoints001/training_curves.png
```

### 可重现性保证

本项目通过以下机制确保完全可重现：

1. **固定随机种子**: 所有实验使用seed=42
2. **确定性算法**: 设置PyTorch、NumPy、Python random种子
3. **完整配置保存**: 所有超参数保存在config.json
4. **环境记录**: 记录Python、PyTorch、CUDA版本
5. **数据顺序固定**: DataLoader使用固定seed

**注意**: GPU训练由于CUDA算法的非确定性，可能有微小差异(<0.1%)。
