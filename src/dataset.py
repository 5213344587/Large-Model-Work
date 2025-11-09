"""
数据集和数据加载器
支持文本序列到序列任务和语言建模任务
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import pickle
import os


class Vocabulary:
    """词汇表类"""
    
    def __init__(self, pad_token='<pad>', unk_token='<unk>', 
                 sos_token='<sos>', eos_token='<eos>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # 添加特殊token
        self.add_word(pad_token)
        self.add_word(unk_token)
        self.add_word(sos_token)
        self.add_word(eos_token)
        
        self.pad_idx = self.word2idx[pad_token]
        self.unk_idx = self.word2idx[unk_token]
        self.sos_idx = self.word2idx[sos_token]
        self.eos_idx = self.word2idx[eos_token]
    
    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
        return self.word2idx[word]
    
    def add_sentence(self, sentence):
        """添加句子中的所有单词"""
        for word in sentence.split():
            self.add_word(word)
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence, add_sos=False, add_eos=False):
        """将句子编码为索引序列"""
        tokens = sentence.split()
        indices = [self.word2idx.get(word, self.unk_idx) for word in tokens]
        
        if add_sos:
            indices = [self.sos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """将索引序列解码为句子"""
        words = []
        special_tokens = {self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx}
        
        for idx in indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            if idx in self.idx2word:
                words.append(self.idx2word[idx])
        
        return ' '.join(words)
    
    def save(self, path):
        """保存词汇表"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': self.word_count,
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'sos_token': self.sos_token,
                'eos_token': self.eos_token
            }, f)
    
    @classmethod
    def load(cls, path):
        """加载词汇表"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(
            pad_token=data['pad_token'],
            unk_token=data['unk_token'],
            sos_token=data['sos_token'],
            eos_token=data['eos_token']
        )
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_count = data['word_count']
        
        return vocab


class Seq2SeqDataset(Dataset):
    """序列到序列数据集"""
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=100):
        """
        Args:
            src_sentences: 源语言句子列表
            tgt_sentences: 目标语言句子列表
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            max_len: 最大序列长度
        """
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sent = self.src_sentences[idx]
        tgt_sent = self.tgt_sentences[idx]
        
        # 编码
        src_indices = self.src_vocab.encode(src_sent, add_sos=True, add_eos=True)
        tgt_indices = self.tgt_vocab.encode(tgt_sent, add_sos=True, add_eos=True)
        
        # 截断
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long)
        }


class LanguageModelingDataset(Dataset):
    """语言建模数据集"""
    
    def __init__(self, sentences, vocab, max_len=100):
        """
        Args:
            sentences: 句子列表
            vocab: 词汇表
            max_len: 最大序列长度
        """
        self.sentences = sentences
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        # 编码
        indices = self.vocab.encode(sentence, add_sos=True, add_eos=True)
        
        # 截断
        indices = indices[:self.max_len]
        
        # 输入和目标（目标是输入右移一位）
        input_ids = indices[:-1]
        target_ids = indices[1:]
        
        return {
            'input': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target_ids, dtype=torch.long)
        }


def collate_fn_seq2seq(batch):
    """Seq2Seq数据的collate函数"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Padding
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded
    }


def collate_fn_lm(batch):
    """语言建模数据的collate函数"""
    input_batch = [item['input'] for item in batch]
    target_batch = [item['target'] for item in batch]
    
    # Padding
    input_padded = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=0)
    target_padded = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=0)
    
    return {
        'input': input_padded,
        'target': target_padded
    }


def create_simple_copy_dataset(num_samples=10000, seq_len=10, vocab_size=20):
    """
    创建简单的复制任务数据集
    输入一个序列，输出相同的序列
    """
    np.random.seed(42)
    
    src_sentences = []
    tgt_sentences = []
    
    for _ in range(num_samples):
        # 生成随机序列（使用数字作为token）
        seq = np.random.randint(4, vocab_size, size=seq_len)  # 4开始避开特殊token
        seq_str = ' '.join([f'tok{i}' for i in seq])
        
        src_sentences.append(seq_str)
        tgt_sentences.append(seq_str)
    
    return src_sentences, tgt_sentences


def create_reverse_dataset(num_samples=10000, seq_len=10, vocab_size=20):
    """
    创建序列反转任务数据集
    输入一个序列，输出其反转
    """
    np.random.seed(42)
    
    src_sentences = []
    tgt_sentences = []
    
    for _ in range(num_samples):
        seq = np.random.randint(4, vocab_size, size=seq_len)
        src_str = ' '.join([f'tok{i}' for i in seq])
        tgt_str = ' '.join([f'tok{i}' for i in reversed(seq)])
        
        src_sentences.append(src_str)
        tgt_sentences.append(tgt_str)
    
    return src_sentences, tgt_sentences


def load_text_file(file_path):
    """加载文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_vocab_from_sentences(sentences, min_freq=1):
    """从句子列表构建词汇表"""
    vocab = Vocabulary()
    
    for sentence in sentences:
        vocab.add_sentence(sentence)
    
    # 可以选择过滤低频词
    if min_freq > 1:
        filtered_vocab = Vocabulary()
        for word, count in vocab.word_count.items():
            if count >= min_freq or word in [vocab.pad_token, vocab.unk_token, 
                                              vocab.sos_token, vocab.eos_token]:
                filtered_vocab.add_word(word)
        return filtered_vocab
    
    return vocab
