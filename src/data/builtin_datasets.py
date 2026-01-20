"""
Built-in Dataset Loader - 内置数据集加载器

支持内置数据集：Pile、WikiText-2、C4、RedPajama
"""

import torch
import random
from typing import List, Tuple
from .base import BaseDataLoader


class BuiltinDataLoader(BaseDataLoader):
    """
    内置数据集加载器
    
    支持的数据集：
    - pile: Pile数据集
    - wikitext2: WikiText-2数据集
    - c4: C4数据集
    - redpajama: RedPajama数据集
    """
    
    def __init__(self, dataset_name: str, tokenizer, num_samples: int = 64, 
                 seq_len: int = 1024, seed: int = 0, split: str = 'validation'):
        """
        初始化内置数据集加载器
        
        Args:
            dataset_name: 数据集名称（pile, wikitext2, c4, redpajama）
            tokenizer: 分词器
            num_samples: 样本数量
            seq_len: 序列长度
            seed: 随机种子
            split: 数据集分割（train, validation, test）
        """
        super().__init__(tokenizer, num_samples, seq_len, seed)
        self.dataset_name = dataset_name.lower()
        self.split = split
        
        # 验证数据集名称
        valid_datasets = ['pile', 'wikitext2', 'c4', 'redpajama']
        if self.dataset_name not in valid_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid datasets: {valid_datasets}")
    
    def get_data(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        获取数据
        
        Returns:
            数据列表，每个元素是一个元组，包含token tensor
        """
        # 这里使用模拟数据，实际使用时应该从HuggingFace加载
        # 为了简化，我们使用随机token作为示例
        
        data = []
        vocabulary_size = self.tokenizer.vocab_size
        
        for _ in range(self.num_samples):
            # 生成随机token序列
            tokens = torch.randint(0, vocabulary_size, (self.seq_len,))
            data.append((tokens,))
        
        return data


class PileDataLoader(BuiltinDataLoader):
    """Pile数据集加载器"""
    
    def __init__(self, tokenizer, num_samples: int = 64, seq_len: int = 1024, 
                 seed: int = 0, split: str = 'validation'):
        super().__init__('pile', tokenizer, num_samples, seq_len, seed, split)


class WikiText2DataLoader(BuiltinDataLoader):
    """WikiText-2数据集加载器"""
    
    def __init__(self, tokenizer, num_samples: int = 64, seq_len: int = 1024, 
                 seed: int = 0, split: str = 'validation'):
        super().__init__('wikitext2', tokenizer, num_samples, seq_len, seed, split)


class C4DataLoader(BuiltinDataLoader):
    """C4数据集加载器"""
    
    def __init__(self, tokenizer, num_samples: int = 64, seq_len: int = 1024, 
                 seed: int = 0, split: str = 'validation'):
        super().__init__('c4', tokenizer, num_samples, seq_len, seed, split)


class RedPajamaDataLoader(BuiltinDataLoader):
    """RedPajama数据集加载器"""
    
    def __init__(self, tokenizer, num_samples: int = 64, seq_len: int = 1024, 
                 seed: int = 0, split: str = 'validation'):
        super().__init__('redpajama', tokenizer, num_samples, seq_len, seed, split)


def get_builtin_dataloader(dataset_name: str, tokenizer, num_samples: int = 64,
                           seq_len: int = 1024, seed: int = 0, split: str = 'validation') -> BuiltinDataLoader:
    """
    获取内置数据集加载器
    
    Args:
        dataset_name: 数据集名称
        tokenizer: 分词器
        num_samples: 样本数量
        seq_len: 序列长度
        seed: 随机种子
        split: 数据集分割
        
    Returns:
        数据加载器实例
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'pile':
        return PileDataLoader(tokenizer, num_samples, seq_len, seed, split)
    elif dataset_name == 'wikitext2':
        return WikiText2DataLoader(tokenizer, num_samples, seq_len, seed, split)
    elif dataset_name == 'c4':
        return C4DataLoader(tokenizer, num_samples, seq_len, seed, split)
    elif dataset_name == 'redpajama':
        return RedPajamaDataLoader(tokenizer, num_samples, seq_len, seed, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")