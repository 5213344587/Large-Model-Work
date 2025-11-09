"""
Transformer基础层实现
包含Position-wise FFN、Layer Normalization、位置编码等
"""

import torch
import torch.nn as nn
import math


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络中间层维度，通常是d_model的4倍
            dropout: dropout比率
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """正弦位置编码 (Sinusoidal Positional Encoding)"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # 注册为buffer，不作为模型参数
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.pe(positions)
        return self.dropout(x)


class SublayerConnection(nn.Module):
    """
    残差连接 + Layer Normalization
    实现: LayerNorm(x + Sublayer(x))
    """
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        Args:
            x: 输入张量
            sublayer: 子层函数
        """
        # Pre-LN: LayerNorm(x) + x
        # 更稳定的训练
        return x + self.dropout(sublayer(self.norm(x)))


class TokenEmbedding(nn.Module):
    """Token嵌入层，带缩放"""
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 乘以sqrt(d_model)进行缩放，参考论文
        return self.embedding(x) * math.sqrt(self.d_model)


def get_padding_mask(seq, pad_idx):
    """
    生成padding mask
    Args:
        seq: (batch_size, seq_len)
        pad_idx: padding token的索引
    Returns:
        mask: (batch_size, 1, seq_len)
    """
    return (seq != pad_idx).unsqueeze(1)


def get_subsequent_mask(seq):
    """
    生成后续mask（用于decoder的自注意力）
    防止位置i注意到位置i之后的信息
    Args:
        seq: (batch_size, seq_len)
    Returns:
        mask: (1, seq_len, seq_len)
    """
    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8),
        diagonal=1
    )
    subsequent_mask = (1 - subsequent_mask).bool()
    return subsequent_mask.unsqueeze(0)


def get_attn_pad_mask(seq_q, seq_k, pad_idx):
    """
    生成attention的padding mask
    Args:
        seq_q: (batch_size, len_q)
        seq_k: (batch_size, len_k)
        pad_idx: padding索引
    Returns:
        mask: (batch_size, len_q, len_k)
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    # (batch_size, 1, len_k)
    pad_mask = seq_k.data.eq(pad_idx).unsqueeze(1)
    
    # (batch_size, len_q, len_k)
    return pad_mask.expand(batch_size, len_q, len_k)
