"""
训练工具函数
包含损失函数、学习率调度器、检查点保存等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        """
        Args:
            vocab_size: 词汇表大小
            padding_idx: padding索引
            smoothing: 平滑系数
        """
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    
    def forward(self, pred, target):
        """
        Args:
            pred: (N, vocab_size) 模型预测的logits
            target: (N,) 真实标签
        """
        # 创建平滑标签
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # 排除padding和正确类
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        # Mask掉padding位置
        mask = (target != self.padding_idx).unsqueeze(1)
        true_dist = true_dist * mask
        
        # 计算KL散度
        pred = F.log_softmax(pred, dim=-1)
        loss = self.criterion(pred, true_dist)
        
        # 归一化
        n_tokens = mask.sum().item()
        if n_tokens > 0:
            loss = loss / n_tokens
        
        return loss


class WarmupScheduler:
    """Warmup学习率调度器（参考Transformer论文）"""
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        """
        Args:
            optimizer: 优化器
            d_model: 模型维度
            warmup_steps: warmup步数
            factor: 缩放因子
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
    
    def step(self):
        """更新学习率"""
        self.step_num += 1
        lr = self.factor * (self.d_model ** (-0.5)) * min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CosineAnnealingWarmup:
    """带Warmup的余弦退火调度器"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        
        if self.step_num < self.warmup_steps:
            # Warmup阶段：线性增长
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            # 余弦退火阶段
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Perplexity曲线
    axes[0, 1].plot(history['train_perplexity'], label='Train PPL')
    axes[0, 1].plot(history['val_perplexity'], label='Val PPL')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Training and Validation Perplexity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 学习率曲线
    axes[1, 0].plot(history['learning_rates'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    
    # Loss对数曲线
    axes[1, 1].semilogy(history['train_loss'], label='Train Loss')
    axes[1, 1].semilogy(history['val_loss'], label='Val Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('Training and Validation Loss (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_attention(attention_weights, src_tokens, tgt_tokens, save_path):
    """可视化注意力权重"""
    import seaborn as sns
    
    # attention_weights: (tgt_len, src_len)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(attention_weights, xticklabels=src_tokens, yticklabels=tgt_tokens,
                cmap='viridis', ax=ax, cbar=True)
    
    ax.set_xlabel('Source Tokens')
    ax.set_ylabel('Target Tokens')
    ax.set_title('Attention Weights')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_bleu(predictions, references, max_n=4):
    """
    简单的BLEU分数计算
    predictions: list of predicted token sequences
    references: list of reference token sequences
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def bleu_n(pred, ref, n):
        pred_ngrams = Counter(get_ngrams(pred, n))
        ref_ngrams = Counter(get_ngrams(ref, n))
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        
        if total == 0:
            return 0.0
        return overlap / total
    
    # 计算各阶n-gram精度
    precisions = []
    for n in range(1, max_n + 1):
        precision_sum = 0
        for pred, ref in zip(predictions, references):
            precision_sum += bleu_n(pred, ref, n)
        precisions.append(precision_sum / len(predictions))
    
    # 几何平均
    if min(precisions) > 0:
        bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        bleu = 0.0
    
    # Brevity penalty
    pred_len = sum(len(p) for p in predictions)
    ref_len = sum(len(r) for r in references)
    
    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / pred_len)
    else:
        bp = 1.0
    
    return bp * bleu


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善幅度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_model_summary(model):
    """打印模型摘要"""
    print("\n" + "="*80)
    print("模型架构摘要")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    print("="*80 + "\n")
