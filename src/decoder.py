"""
Transformer Decoder实现
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention, RelativePositionAttention
from layers import PositionwiseFeedForward, SublayerConnection


class DecoderLayer(nn.Module):
    """单个Decoder层"""
    
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
        
        # Masked multi-head self-attention
        if use_relative_position:
            self.self_attn = RelativePositionAttention(d_model, n_heads, dropout=dropout)
        else:
            self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Multi-head cross-attention (encoder-decoder attention)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 三个子层连接
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: decoder输入 (batch_size, tgt_len, d_model)
            encoder_output: encoder输出 (batch_size, src_len, d_model)
            src_mask: source mask (batch_size, 1, src_len)
            tgt_mask: target mask (batch_size, tgt_len, tgt_len)
        Returns:
            output: (batch_size, tgt_len, d_model)
            self_attn: self-attention权重
            cross_attn: cross-attention权重
        """
        # Masked self-attention sublayer
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask)[0])
        
        # Cross-attention sublayer
        x = self.sublayer2(x, lambda _x: self.cross_attn(_x, encoder_output, encoder_output, src_mask)[0])
        
        # Feed-forward sublayer
        x = self.sublayer3(x, self.feed_forward)
        
        # 保存attention权重
        with torch.no_grad():
            _, self_attn = self.self_attn(x, x, x, tgt_mask)
            _, cross_attn = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        
        return x, self_attn, cross_attn


class TransformerDecoder(nn.Module):
    """Transformer Decoder"""
    
    def __init__(self, num_layers, d_model, n_heads, d_ff, 
                 dropout=0.1, use_relative_position=False):
        """
        Args:
            num_layers: decoder层数
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: dropout比率
            use_relative_position: 是否使用相对位置编码
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, use_relative_position)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch_size, tgt_len, d_model)
            encoder_output: (batch_size, src_len, d_model)
            src_mask: (batch_size, 1, src_len)
            tgt_mask: (batch_size, tgt_len, tgt_len)
        Returns:
            output: (batch_size, tgt_len, d_model)
            self_attentions: list of self-attention weights
            cross_attentions: list of cross-attention weights
        """
        self_attentions = []
        cross_attentions = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
        
        # 最后进行LayerNorm
        x = self.norm(x)
        
        return x, self_attentions, cross_attentions
