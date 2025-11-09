"""
Transformer Encoder实现
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention, RelativePositionAttention
from layers import PositionwiseFeedForward, SublayerConnection


class EncoderLayer(nn.Module):
    """单个Encoder层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_relative_position=False):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: dropout比率
            use_relative_position: 是否使用相对位置编码
        """
        super().__init__()
        
        # Multi-head self-attention
        if use_relative_position:
            self.self_attn = RelativePositionAttention(d_model, n_heads, dropout=dropout)
        else:
            self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 两个子层连接（残差 + LayerNorm）
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
            attention: attention权重
        """
        # Self-attention sublayer
        attn_output = None
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, mask)[0])
        
        # 保存attention权重用于可视化
        with torch.no_grad():
            _, attn_output = self.self_attn(x, x, x, mask)
        
        # Feed-forward sublayer
        x = self.sublayer2(x, self.feed_forward)
        
        return x, attn_output


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, num_layers, d_model, n_heads, d_ff, 
                 dropout=0.1, use_relative_position=False):
        """
        Args:
            num_layers: encoder层数
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: dropout比率
            use_relative_position: 是否使用相对位置编码
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, use_relative_position)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
            attentions: list of attention weights
        """
        attentions = []
        
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
        
        # 最后进行LayerNorm
        x = self.norm(x)
        
        return x, attentions
