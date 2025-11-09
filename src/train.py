"""
Transformer训练脚本
包含训练循环、评估、模型保存等功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformer import Transformer, TransformerForLanguageModeling
from dataset import (Vocabulary, Seq2SeqDataset, LanguageModelingDataset,
                     collate_fn_seq2seq, collate_fn_lm,
                     create_simple_copy_dataset, create_reverse_dataset,
                     build_vocab_from_sentences)
from utils import (
    LabelSmoothingLoss, WarmupScheduler, count_parameters,
    save_checkpoint, load_checkpoint, plot_training_curves
)


def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Transformer训练器"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 criterion, device, config, save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.save_dir = save_dir
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rates': []
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 梯度裁剪
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            if 'src' in batch:  # Seq2Seq
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # 目标输入和输出（teacher forcing）
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward pass
                output = self.model(src, tgt_input)
                
            else:  # Language Modeling
                input_ids = batch['input'].to(self.device)
                target_ids = batch['target'].to(self.device)
                
                # Forward pass
                output = self.model(input_ids)
                tgt_output = target_ids
            
            # 计算损失
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = self.criterion(output, tgt_output)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 统计
            batch_loss = loss.item()
            batch_tokens = (tgt_output != 0).sum().item()
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'ppl': f'{np.exp(batch_loss):.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        avg_loss = total_loss / total_tokens
        avg_perplexity = np.exp(avg_loss)
        
        return avg_loss, avg_perplexity
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                if 'src' in batch:  # Seq2Seq
                    src = batch['src'].to(self.device)
                    tgt = batch['tgt'].to(self.device)
                    
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                    
                    output = self.model(src, tgt_input)
                    
                else:  # Language Modeling
                    input_ids = batch['input'].to(self.device)
                    target_ids = batch['target'].to(self.device)
                    
                    output = self.model(input_ids)
                    tgt_output = target_ids
                
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                loss = self.criterion(output, tgt_output)
                
                batch_loss = loss.item()
                batch_tokens = (tgt_output != 0).sum().item()
                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        avg_perplexity = np.exp(avg_loss)
        
        return avg_loss, avg_perplexity
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epochs")
        print(f"模型参数量: {count_parameters(self.model):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss, train_ppl = self.train_epoch(epoch)
            
            # 评估
            val_loss, val_ppl = self.evaluate()
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_perplexity'].append(train_ppl)
            self.history['val_perplexity'].append(val_ppl)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # 打印信息
            print(f'\nEpoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Perplexity: {train_ppl:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            print(f'  Time: {epoch_time:.2f}s')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    os.path.join(self.save_dir, 'best_model.pt')
                )
                print(f'  保存最佳模型 (val_loss: {val_loss:.4f})')
            
            # 定期保存检查点
            if epoch % 5 == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
                )
        
        # 保存训练历史
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 绘制训练曲线
        plot_training_curves(self.history, self.save_dir)
        
        print('\n训练完成！')
        print(f'最佳验证损失: {best_val_loss:.4f}')


def main():
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    
    # 数据参数
    parser.add_argument('--task', type=str, default='copy', choices=['copy', 'reverse', 'lm'],
                        help='任务类型')
    parser.add_argument('--num_samples', type=int, default=10000, help='样本数量')
    parser.add_argument('--seq_len', type=int, default=10, help='序列长度')
    parser.add_argument('--vocab_size', type=int, default=50, help='词汇表大小')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='encoder层数')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='decoder层数')
    parser.add_argument('--d_ff', type=int, default=1024, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('--use_relative_position', action='store_true', help='使用相对位置编码')
    parser.add_argument('--use_learnable_pe', action='store_true', help='使用可学习位置编码')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warmup步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='../checkpoints', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据集
    print("创建数据集...")
    if args.task == 'copy':
        src_sentences, tgt_sentences = create_simple_copy_dataset(
            args.num_samples, args.seq_len, args.vocab_size
        )
    elif args.task == 'reverse':
        src_sentences, tgt_sentences = create_reverse_dataset(
            args.num_samples, args.seq_len, args.vocab_size
        )
    
    # 划分训练集和验证集
    split_idx = int(0.9 * len(src_sentences))
    train_src, val_src = src_sentences[:split_idx], src_sentences[split_idx:]
    train_tgt, val_tgt = tgt_sentences[:split_idx], tgt_sentences[split_idx:]
    
    # 构建词汇表
    print("构建词汇表...")
    src_vocab = build_vocab_from_sentences(src_sentences)
    tgt_vocab = build_vocab_from_sentences(tgt_sentences)
    
    print(f"源词汇表大小: {len(src_vocab)}")
    print(f"目标词汇表大小: {len(tgt_vocab)}")
    
    # 保存词汇表
    os.makedirs(args.save_dir, exist_ok=True)
    src_vocab.save(os.path.join(args.save_dir, 'src_vocab.pkl'))
    tgt_vocab.save(os.path.join(args.save_dir, 'tgt_vocab.pkl'))
    
    # 创建数据集
    train_dataset = Seq2SeqDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = Seq2SeqDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, collate_fn=collate_fn_seq2seq)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn_seq2seq)
    
    # 创建模型
    print("创建模型...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=src_vocab.pad_idx,
        use_relative_position=args.use_relative_position,
        use_learnable_pe=args.use_learnable_pe
    ).to(args.device)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    scheduler = WarmupScheduler(
        optimizer,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps
    )
    
    # 损失函数
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        smoothing=args.label_smoothing
    )
    
    # 保存配置
    config = vars(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=args.device,
        config=config,
        save_dir=args.save_dir
    )
    
    # 训练
    trainer.train(args.num_epochs)


if __name__ == '__main__':
    main()
