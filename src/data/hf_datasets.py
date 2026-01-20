"""
HuggingFace Dataset Loader - HuggingFace数据集加载器

支持从HuggingFace Hub加载任意数据集
"""

import torch
from typing import List, Tuple, Optional
from .base import BaseDataLoader


class HFDatasetLoader(BaseDataLoader):
    """
    HuggingFace数据集加载器

    支持加载HuggingFace Hub上的任意数据集
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        split: str = "validation",
        text_column: str = "text",
        subset: Optional[str] = None,
    ):
        """
        初始化HuggingFace数据集加载器

        Args:
            dataset_name: 数据集名称（如 'wikitext', 'c4'）
            tokenizer: 分词器
            num_samples: 样本数量
            seq_len: 序列长度
            seed: 随机种子
            split: 数据集分割（train, validation, test）
            text_column: 文本列名
            subset: 数据集子集（如 'wikitext-2-raw-v1'）
        """
        super().__init__(tokenizer, num_samples, seq_len, seed)
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.subset = subset

        # 延迟加载数据集
        self._dataset = None

    def _load_dataset(self):
        """延迟加载数据集"""
        if self._dataset is not None:
            return self._dataset

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is not installed. "
                "Please install it with: pip install datasets"
            )

        # 加载数据集
        if self.subset:
            self._dataset = load_dataset(
                self.dataset_name, self.subset, split=self.split
            )
        else:
            self._dataset = load_dataset(self.dataset_name, split=self.split)

        return self._dataset

    def get_data(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        获取数据

        Returns:
            数据列表，每个元素是一个元组，包含token tensor
        """
        dataset = self._load_dataset()

        # 检查文本列是否存在
        if self.text_column not in dataset.column_names:
            available_columns = ", ".join(dataset.column_names)
            raise ValueError(
                f"Column '{self.text_column}' not found in dataset. "
                f"Available columns: {available_columns}"
            )

        # 提取文本
        texts = [item[self.text_column] for item in dataset]

        # 随机采样
        import random

        random.seed(self.seed)

        if len(texts) > self.num_samples:
            indices = random.sample(range(len(texts)), self.num_samples)
            texts = [texts[i] for i in indices]

        # 分词
        data = []
        for text in texts:
            tokens = self.tokenize(text)
            tokens = self.pad_sequence(tokens)
            data.append((tokens,))

        return data

    def get_dataset_info(self) -> dict:
        """
        获取数据集信息

        Returns:
            数据集信息字典
        """
        dataset = self._load_dataset()

        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "num_samples": len(dataset),
            "columns": dataset.column_names,
            "text_column": self.text_column,
        }


def get_hf_dataloader(
    dataset_name: str,
    tokenizer,
    num_samples: int = 64,
    seq_len: int = 1024,
    seed: int = 0,
    split: str = "validation",
    text_column: str = "text",
    subset: Optional[str] = None,
) -> HFDatasetLoader:
    """
    获取HuggingFace数据集加载器

    Args:
        dataset_name: 数据集名称
        tokenizer: 分词器
        num_samples: 样本数量
        seq_len: 序列长度
        seed: 随机种子
        split: 数据集分割
        text_column: 文本列名
        subset: 数据集子集

    Returns:
        数据加载器实例
    """
    return HFDatasetLoader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        num_samples=num_samples,
        seq_len=seq_len,
        seed=seed,
        split=split,
        text_column=text_column,
        subset=subset,
    )
