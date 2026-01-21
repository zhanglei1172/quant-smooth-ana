"""
Base Data Loader - 数据加载器基类

定义数据加载器的标准接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch


class BaseDataLoader(ABC):
    """
    数据加载器基类

    所有数据加载器都应该继承这个基类，并实现get_data方法
    """

    def __init__(
        self, tokenizer, num_samples: int = 64, seq_len: int = 1024, seed: int = 0
    ):
        """
        初始化数据加载器

        Args:
            tokenizer: 分词器
            num_samples: 样本数量
            seq_len: 序列长度
            seed: 随机种子
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.seed = seed

        # 设置随机种子
        torch.manual_seed(seed)

        # 缓存数据
        self._data_cache = None

    @abstractmethod
    def get_data(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        获取数据

        Returns:
            数据列表，每个元素是一个元组，包含token tensor等
        """
        pass

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        获取指定索引的数据

        Args:
            idx: 数据索引

        Returns:
            数据元组
        """
        if self._data_cache is None:
            self._data_cache = self.get_data()

        return self._data_cache[idx]

    def tokenize(self, text: str) -> torch.Tensor:
        """
        分词

        Args:
            text: 输入文本

        Returns:
            token tensor
        """
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.seq_len
        )
        return tokens["input_ids"].squeeze(0)

    def pad_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        填充序列到指定长度

        Args:
            tokens: 输入token tensor

        Returns:
            填充后的token tensor
        """
        if len(tokens) < self.seq_len:
            padding = torch.full(
                (self.seq_len - len(tokens),), self.tokenizer.pad_token_id
            )
            tokens = torch.cat([tokens, padding])
        elif len(tokens) > self.seq_len:
            tokens = tokens[: self.seq_len]

        return tokens

    def clear_cache(self):
        """清空数据缓存"""
        self._data_cache = None
