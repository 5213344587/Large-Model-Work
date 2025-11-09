"""
完整的Transformer模型
包含Encoder和Decoder
"""

import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from layers import (TokenEmbedding, PositionalEncoding, LearnablePositionalEncoding,
                    get_padding_mask, get_subsequent_mask, get_attn_pad_mask)


class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        pad_idx=0,
        use_relative_position=False,
        use_learnable_pe=False
    ):
        """
        Args:
            src_vocab_size: 源词汇表大小
            tgt_vocab_size: 目标词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            num_encoder_layers: encoder层数
            num_decoder_layers: decoder层数
            d_ff: 前馈网络维度
            dropout: dropout比率
            max_len: 最大序列长度
            pad_idx: padding索引
            use_relative_position: 是否使用相对位置编码
            use_learnable_pe: 是否使用可学习位置编码
        """
        super().__init__()
        
        self.pad_idx = pad_idx
        self.d_model = d_model
        
        # Token embedding
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        if use_learnable_pe:
            self.src_pos_encoding = LearnablePositionalEncoding(d_model, max_len, dropout)
            self.tgt_pos_encoding = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
            self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, n_heads, d_ff, dropout, use_relative_position
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, n_heads, d_ff, dropout, use_relative_position
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier uniform初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        """
        Encoder forward
        Args:
            src: (batch_size, src_len)
            src_mask: (batch_size, src_len, src_len)
        Returns:
            encoder_output: (batch_size, src_len, d_model)
        """
        src_embedded = self.src_embedding(src)
        src_embedded = self.src_pos_encoding(src_embedded)
        encoder_output, _ = self.encoder(src_embedded, src_mask)
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decoder forward
        Args:
            tgt: (batch_size, tgt_len)
            encoder_output: (batch_size, src_len, d_model)
            src_mask: (batch_size, 1, src_len)
            tgt_mask: (batch_size, tgt_len, tgt_len)
        Returns:
            output: (batch_size, tgt_len, tgt_vocab_size)
        """
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        decoder_output, _, _ = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        return output
    
    def forward(self, src, tgt):
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
        Returns:
            output: (batch_size, tgt_len, tgt_vocab_size)
        """
        # 创建masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encoder
        encoder_output = self.encode(src, src_mask)
        
        # Decoder
        output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        return output
    
    def make_src_mask(self, src):
        """创建source mask"""
        # (batch_size, 1, src_len) -> unsqueeze再广播到attention
        # 或者 (batch_size, src_len, src_len)
        src_mask = (src != self.pad_idx).unsqueeze(1)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """创建target mask（包含padding和subsequent mask）"""
        batch_size, tgt_len = tgt.size()
        
        # Padding mask: (batch_size, 1, tgt_len)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1)
        
        # Subsequent mask: (tgt_len, tgt_len)
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()
        
        # 扩展padding mask: (batch_size, tgt_len, tgt_len)
        tgt_pad_mask = tgt_pad_mask.expand(batch_size, tgt_len, tgt_len)
        
        # 扩展subsequent mask并合并
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)
        
        return tgt_mask
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerForLanguageModeling(nn.Module):
    """用于语言建模的Transformer（仅decoder）"""
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        pad_idx=0,
        use_relative_position=False,
        use_learnable_pe=False
    ):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.d_model = d_model
        
        # Token embedding
        self.embedding = TokenEmbedding(vocab_size, d_model)
        
        # Positional encoding
        if use_learnable_pe:
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Decoder (用作语言模型)
        self.decoder = TransformerDecoder(
            num_layers, d_model, n_heads, d_ff, dropout, use_relative_position
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 使用encoder output作为key-value的占位符（对于纯decoder LM不需要）
        self.use_encoder = False
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, vocab_size)
        """
        # 创建mask
        mask = self.make_mask(x)
        
        # Embedding + Positional encoding
        embedded = self.embedding(x)
        embedded = self.pos_encoding(embedded)
        
        # 对于纯decoder LM，encoder_output设为embedded本身
        # 实际上cross-attention不会被使用
        encoder_output = embedded
        
        # Decoder forward
        output, _, _ = self.decoder(embedded, encoder_output, None, mask)
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def make_mask(self, x):
        """创建causal mask"""
        batch_size, seq_len = x.size()
        
        # Padding mask
        pad_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Subsequent mask (causal mask)
        subsequent_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device)
        ).bool().unsqueeze(0).unsqueeze(0)
        
        # 合并
        mask = pad_mask & subsequent_mask
        
        return mask
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
