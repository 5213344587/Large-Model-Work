# 数据集说明

本项目使用合成数据集进行模型验证。

## 数据集类型

### 1. 序列复制任务 (Copy Task)

**描述**: 模型需要学习复制输入序列到输出。

**示例**:
```
输入: tok5 tok12 tok8 tok15 tok3 tok20 tok9 tok14 tok6 tok11
输出: tok5 tok12 tok8 tok15 tok3 tok20 tok9 tok14 tok6 tok11
```

**用途**: 验证模型基本的序列到序列学习能力。

### 2. 序列反转任务 (Reverse Task)

**描述**: 模型需要学习反转输入序列。

**示例**:
```
输入: tok5 tok12 tok8 tok15 tok3 tok20 tok9 tok14 tok6 tok11
输出: tok11 tok6 tok14 tok9 tok20 tok3 tok15 tok8 tok12 tok5
```

**用途**: 测试模型处理长距离依赖的能力。

## 数据生成

数据集在训练时自动生成，无需手动下载。

### 参数设置

- **样本数量**: 10,000 (默认)
  - 训练集: 9,000
  - 验证集: 1,000

- **序列长度**: 10 (默认)

- **词汇表大小**: 50 (默认)
  - 特殊token: `<pad>`, `<unk>`, `<sos>`, `<eos>`
  - 实际token: tok4-tok49 (避开特殊token)

### 数据生成代码

参见 `src/dataset.py` 中的:
- `create_simple_copy_dataset()`
- `create_reverse_dataset()`

## 使用真实数据集

如果要使用真实数据集（如机器翻译数据），可以：

1. 准备平行语料（source.txt 和 target.txt）
2. 每行一个句子，source和target按行对应
3. 修改训练脚本加载自定义数据

### 示例数据格式

**source.txt**:
```
hello world
how are you
good morning
```

**target.txt**:
```
你好 世界
你 好 吗
早上 好
```

### 推荐的真实数据集

- **机器翻译**: WMT, IWSLT
- **文本摘要**: CNN/DailyMail, XSum
- **问答**: SQuAD, MS MARCO

## 数据预处理

词汇表构建和序列编码在 `Vocabulary` 类中实现：

```python
from dataset import Vocabulary, build_vocab_from_sentences

# 构建词汇表
vocab = build_vocab_from_sentences(sentences)

# 编码句子
indices = vocab.encode("hello world", add_sos=True, add_eos=True)

# 解码
sentence = vocab.decode(indices)
```

## 数据增强（可选）

可以考虑的数据增强方法：
- 随机删除token
- 随机替换token
- 回译（Back-translation）
- 噪声注入
