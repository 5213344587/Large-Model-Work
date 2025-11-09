"""
模型测试脚本
用于快速验证模型能否正常运行
"""

import torch
import sys

from transformer import Transformer
from dataset import (
    create_simple_copy_dataset, 
    build_vocab_from_sentences,
    Seq2SeqDataset,
    collate_fn_seq2seq
)
from torch.utils.data import DataLoader


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 80)
    print("测试1: 模型前向传播")
    print("=" * 80)
    
    # 创建小数据集
    src_sentences, tgt_sentences = create_simple_copy_dataset(
        num_samples=100, seq_len=10, vocab_size=30
    )
    
    # 构建词汇表
    src_vocab = build_vocab_from_sentences(src_sentences)
    tgt_vocab = build_vocab_from_sentences(tgt_sentences)
    
    print(f"源词汇表大小: {len(src_vocab)}")
    print(f"目标词汇表大小: {len(tgt_vocab)}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.1
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    dataset = Seq2SeqDataset(src_sentences[:10], tgt_sentences[:10], 
                             src_vocab, tgt_vocab)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_seq2seq)
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        for batch in loader:
            src = batch['src']
            tgt = batch['tgt']
            
            output = model(src, tgt[:, :-1])
            print(f"输入形状: {src.shape}")
            print(f"目标形状: {tgt.shape}")
            print(f"输出形状: {output.shape}")
            print('[OK] 前向传播成功！')
            break
    
    print()


def test_model_training():
    """测试模型训练一个batch"""
    print("=" * 80)
    print("测试2: 模型训练")
    print("=" * 80)
    
    # 创建数据
    src_sentences, tgt_sentences = create_simple_copy_dataset(
        num_samples=100, seq_len=8, vocab_size=30
    )
    
    src_vocab = build_vocab_from_sentences(src_sentences)
    tgt_vocab = build_vocab_from_sentences(tgt_sentences)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.1
    )
    
    # 创建数据加载器
    dataset = Seq2SeqDataset(src_sentences[:50], tgt_sentences[:50],
                             src_vocab, tgt_vocab)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, 
                        collate_fn=collate_fn_seq2seq)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # 训练几个batch
    model.train()
    losses = []
    
    for i, batch in enumerate(loader):
        if i >= 5:  # 只训练5个batch
            break
        
        src = batch['src']
        tgt = batch['tgt']
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward
        output = model(src, tgt_input)
        
        # 计算损失
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Batch {i+1}/5, Loss: {loss.item():.4f}")
    
    print(f"\n平均损失: {sum(losses)/len(losses):.4f}")
    print(f"[OK] 训练测试成功！")
    print()


def test_attention_visualization():
    """测试注意力可视化"""
    print("=" * 80)
    print("测试3: 注意力机制")
    print("=" * 80)
    
    from attention import MultiHeadAttention
    
    # 创建注意力层
    attention = MultiHeadAttention(d_model=128, n_heads=4)
    
    # 测试输入
    batch_size, seq_len, d_model = 2, 10, 128
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attn_weights = attention(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"[OK] 注意力机制测试成功！")
    print()


def test_positional_encoding():
    """测试位置编码"""
    print("=" * 80)
    print("测试4: 位置编码")
    print("=" * 80)
    
    from layers import PositionalEncoding, LearnablePositionalEncoding
    
    # 测试正弦位置编码
    pe_sin = PositionalEncoding(d_model=128, max_len=100)
    x = torch.randn(2, 20, 128)
    output_sin = pe_sin(x)
    print(f"正弦位置编码 - 输入: {x.shape}, 输出: {output_sin.shape}")
    print(f"[OK] 正弦位置编码测试成功！")
    
    # 测试可学习位置编码
    pe_learn = LearnablePositionalEncoding(d_model=128, max_len=100)
    output_learn = pe_learn(x)
    print(f"可学习位置编码 - 输入: {x.shape}, 输出: {output_learn.shape}")
    print(f"[OK] 可学习位置编码测试成功！")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Transformer模型测试套件")
    print("=" * 80 + "\n")
    
    try:
        test_attention_visualization()
        test_positional_encoding()
        test_model_forward()
        test_model_training()
        
        print("=" * 80)
        print("[OK] 所有测试通过！模型可以正常运行。")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
