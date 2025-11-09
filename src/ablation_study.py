"""
消融实验脚本
测试不同配置对模型性能的影响
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from transformer import Transformer
from dataset import (
    create_simple_copy_dataset, create_reverse_dataset,
    build_vocab_from_sentences, Seq2SeqDataset, collate_fn_seq2seq
)
from train import Trainer, set_seed
from utils import LabelSmoothingLoss, WarmupScheduler, count_parameters


def run_experiment(config, base_config, experiment_name):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"运行实验: {experiment_name}")
    print(f"{'='*80}")
    
    # 合并配置
    full_config = {**base_config, **config}
    
    # 设置随机种子
    set_seed(full_config['seed'])
    
    # 创建数据集
    if full_config['task'] == 'copy':
        src_sentences, tgt_sentences = create_simple_copy_dataset(
            full_config['num_samples'], full_config['seq_len'], full_config['vocab_size']
        )
    else:
        src_sentences, tgt_sentences = create_reverse_dataset(
            full_config['num_samples'], full_config['seq_len'], full_config['vocab_size']
        )
    
    # 划分数据
    split_idx = int(0.9 * len(src_sentences))
    train_src, val_src = src_sentences[:split_idx], src_sentences[split_idx:]
    train_tgt, val_tgt = tgt_sentences[:split_idx], tgt_sentences[split_idx:]
    
    # 构建词汇表
    src_vocab = build_vocab_from_sentences(src_sentences)
    tgt_vocab = build_vocab_from_sentences(tgt_sentences)
    
    # 创建数据集和数据加载器
    train_dataset = Seq2SeqDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = Seq2SeqDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=full_config['batch_size'],
                              shuffle=True, collate_fn=collate_fn_seq2seq)
    val_loader = DataLoader(val_dataset, batch_size=full_config['batch_size'],
                            shuffle=False, collate_fn=collate_fn_seq2seq)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=full_config['d_model'],
        n_heads=full_config['n_heads'],
        num_encoder_layers=full_config['num_encoder_layers'],
        num_decoder_layers=full_config['num_decoder_layers'],
        d_ff=full_config['d_ff'],
        dropout=full_config['dropout'],
        pad_idx=src_vocab.pad_idx,
        use_relative_position=full_config.get('use_relative_position', False),
        use_learnable_pe=full_config.get('use_learnable_pe', False)
    ).to(full_config['device'])
    
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=full_config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = WarmupScheduler(
        optimizer,
        d_model=full_config['d_model'],
        warmup_steps=full_config['warmup_steps']
    )
    
    # 损失函数
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        smoothing=full_config['label_smoothing']
    )
    
    # 创建保存目录
    save_dir = os.path.join(full_config['save_dir'], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=full_config['device'],
        config=full_config,
        save_dir=save_dir
    )
    
    trainer.train(full_config['num_epochs'])
    
    # 返回结果
    return {
        'name': experiment_name,
        'config': config,
        'final_train_loss': trainer.history['train_loss'][-1],
        'final_val_loss': trainer.history['val_loss'][-1],
        'best_val_loss': min(trainer.history['val_loss']),
        'final_train_ppl': trainer.history['train_perplexity'][-1],
        'final_val_ppl': trainer.history['val_perplexity'][-1],
        'parameters': count_parameters(model),
        'history': trainer.history
    }


def ablation_study_attention_heads():
    """注意力头数消融实验"""
    experiments = []
    
    base_config = {
        'task': 'copy',
        'num_samples': 8000,
        'seq_len': 10,
        'vocab_size': 50,
        'd_model': 256,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'd_ff': 1024,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 30,
        'lr': 0.0001,
        'warmup_steps': 2000,
        'label_smoothing': 0.1,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '../results/ablation'
    }
    
    # 测试不同的注意力头数
    for n_heads in [2, 4, 8]:
        config = {'n_heads': n_heads}
        result = run_experiment(config, base_config, f'attention_heads_{n_heads}')
        experiments.append(result)
    
    return experiments


def ablation_study_layers():
    """层数消融实验"""
    experiments = []
    
    base_config = {
        'task': 'copy',
        'num_samples': 8000,
        'seq_len': 10,
        'vocab_size': 50,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 30,
        'lr': 0.0001,
        'warmup_steps': 2000,
        'label_smoothing': 0.1,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '../results/ablation'
    }
    
    # 测试不同的层数
    for num_layers in [2, 3, 4, 6]:
        config = {
            'num_encoder_layers': num_layers,
            'num_decoder_layers': num_layers
        }
        result = run_experiment(config, base_config, f'layers_{num_layers}')
        experiments.append(result)
    
    return experiments


def ablation_study_model_size():
    """模型大小消融实验"""
    experiments = []
    
    base_config = {
        'task': 'reverse',
        'num_samples': 8000,
        'seq_len': 10,
        'vocab_size': 50,
        'n_heads': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 30,
        'lr': 0.0001,
        'warmup_steps': 2000,
        'label_smoothing': 0.1,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '../results/ablation'
    }
    
    # 测试不同的模型大小
    model_sizes = [
        {'d_model': 128, 'd_ff': 512},
        {'d_model': 256, 'd_ff': 1024},
        {'d_model': 512, 'd_ff': 2048}
    ]
    
    for config in model_sizes:
        result = run_experiment(
            config, base_config, 
            f'model_size_d{config["d_model"]}_ff{config["d_ff"]}'
        )
        experiments.append(result)
    
    return experiments


def ablation_study_position_encoding():
    """位置编码消融实验"""
    experiments = []
    
    base_config = {
        'task': 'copy',
        'num_samples': 8000,
        'seq_len': 10,
        'vocab_size': 50,
        'd_model': 256,
        'n_heads': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'd_ff': 1024,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 30,
        'lr': 0.0001,
        'warmup_steps': 2000,
        'label_smoothing': 0.1,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '../results/ablation'
    }
    
    # 测试不同的位置编码
    configs = [
        {'use_relative_position': False, 'use_learnable_pe': False},
        {'use_relative_position': True, 'use_learnable_pe': False},
        {'use_relative_position': False, 'use_learnable_pe': True}
    ]
    
    names = ['sinusoidal', 'relative', 'learnable']
    
    for config, name in zip(configs, names):
        result = run_experiment(config, base_config, f'position_encoding_{name}')
        experiments.append(result)
    
    return experiments


def plot_ablation_results(all_experiments, save_dir):
    """绘制消融实验结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 按实验类型分组
    experiment_groups = {}
    for exp in all_experiments:
        group_name = exp['name'].rsplit('_', 1)[0] if '_' in exp['name'] else exp['name']
        if group_name not in experiment_groups:
            experiment_groups[group_name] = []
        experiment_groups[group_name].append(exp)
    
    # 为每组实验绘图
    for group_name, experiments in experiment_groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 最终验证损失
        names = [exp['name'].split('_')[-1] for exp in experiments]
        val_losses = [exp['final_val_loss'] for exp in experiments]
        params = [exp['parameters'] for exp in experiments]
        
        axes[0].bar(names, val_losses)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Final Validation Loss')
        axes[0].set_title(f'{group_name}: Final Validation Loss')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 参数量 vs 性能
        axes[1].scatter(params, val_losses, s=100)
        for i, name in enumerate(names):
            axes[1].annotate(name, (params[i], val_losses[i]))
        axes[1].set_xlabel('Number of Parameters')
        axes[1].set_ylabel('Final Validation Loss')
        axes[1].set_title(f'{group_name}: Parameters vs Performance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{group_name}_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 训练曲线对比
        fig, ax = plt.subplots(figsize=(10, 6))
        for exp in experiments:
            label = exp['name'].split('_')[-1]
            ax.plot(exp['history']['val_loss'], label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'{group_name}: Training Curves')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{group_name}_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 创建汇总表格
    summary_data = []
    for exp in all_experiments:
        summary_data.append({
            'Experiment': exp['name'],
            'Parameters': f"{exp['parameters']:,}",
            'Final Val Loss': f"{exp['final_val_loss']:.4f}",
            'Best Val Loss': f"{exp['best_val_loss']:.4f}",
            'Final Val PPL': f"{exp['final_val_ppl']:.4f}"
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(save_dir, 'ablation_summary.csv'), index=False)
    print("\n消融实验汇总:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--experiments', nargs='+', 
                        choices=['attention_heads', 'layers', 'model_size', 'position_encoding', 'all'],
                        default=['all'], help='要运行的实验类型')
    parser.add_argument('--save_dir', type=str, default='../results/ablation',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    all_experiments = []
    
    if 'all' in args.experiments or 'attention_heads' in args.experiments:
        print("\n运行注意力头数消融实验...")
        all_experiments.extend(ablation_study_attention_heads())
    
    if 'all' in args.experiments or 'layers' in args.experiments:
        print("\n运行层数消融实验...")
        all_experiments.extend(ablation_study_layers())
    
    if 'all' in args.experiments or 'model_size' in args.experiments:
        print("\n运行模型大小消融实验...")
        all_experiments.extend(ablation_study_model_size())
    
    if 'all' in args.experiments or 'position_encoding' in args.experiments:
        print("\n运行位置编码消融实验...")
        all_experiments.extend(ablation_study_position_encoding())
    
    # 保存所有实验结果
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'all_experiments.json'), 'w') as f:
        # 移除history以减小文件大小
        simplified_results = []
        for exp in all_experiments:
            simplified_exp = {k: v for k, v in exp.items() if k != 'history'}
            simplified_results.append(simplified_exp)
        json.dump(simplified_results, f, indent=2)
    
    # 绘制结果
    plot_ablation_results(all_experiments, args.save_dir)
    
    print(f"\n所有实验完成！结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
