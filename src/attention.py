"""
Multi-Head Attention 实现
包含标准注意力、相对位置编码等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, n_heads, seq_len, d_k)
            K: (batch_size, n_heads, seq_len, d_k)
            V: (batch_size, n_heads, seq_len, d_v)
            mask: (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
        Returns:
            output: (batch_size, n_heads, seq_len, d_v)
            attention: (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数: QK^T / sqrt(d_k)
        # (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 加权求和
        output = torch.matmul(attention, V)
        
        return output, attention


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, seq_len, d_model)
            K: (batch_size, seq_len, d_model)
            V: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
            attention: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)
        
        # 线性投影并分割成多头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # 调整mask维度
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
        
        # 计算注意力
        output, attention = self.attention(Q, K, V, mask)
        
        # 合并多头
        # (batch_size, n_heads, seq_len, d_v) -> (batch_size, seq_len, n_heads, d_v)
        # -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性层
        output = self.W_O(output)
        output = self.dropout(output)
        
        return output, attention


class RelativePositionAttention(nn.Module):
    """带相对位置编码的多头注意力"""
    
    def __init__(self, d_model, n_heads, max_relative_position=32, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # 相对位置编码 embedding
        self.relative_positions_embeddings = nn.Embedding(
            2 * max_relative_position + 1, self.d_k
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def get_relative_positions(self, seq_len):
        """生成相对位置矩阵"""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 截断到最大相对位置
        distance_mat = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # 转换到0-based索引
        distance_mat = distance_mat + self.max_relative_position
        
        return distance_mat
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len = Q.size(0), Q.size(1)
        
        # 线性投影
        Q = self.W_Q(Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加相对位置编码
        relative_positions = self.get_relative_positions(seq_len).to(Q.device)
        relative_embeddings = self.relative_positions_embeddings(relative_positions)
        
        # 计算相对位置注意力
        # (batch_size, n_heads, seq_len, d_k) @ (seq_len, seq_len, d_k)
        relative_scores = torch.einsum('bhld,lrd->bhlr', Q, relative_embeddings)
        relative_scores = relative_scores / math.sqrt(self.d_k)
        
        scores = scores + relative_scores
        
        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)
        output = self.dropout(output)
        
        return output, attention
