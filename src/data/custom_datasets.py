"""
Custom Dataset Loader - 自定义数据集加载器

支持从本地文件加载自定义数据集（文本、JSON、CSV格式）
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .base import BaseDataLoader


class CustomDataLoader(BaseDataLoader):
    """
    自定义数据集加载器

    支持的文件格式：
    - text: 纯文本文件
    - json: JSON文件（每行一个JSON对象或JSON数组）
    - csv: CSV文件
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        format: str = "text",
        text_column: Optional[str] = None,
    ):
        """
        初始化自定义数据集加载器

        Args:
            file_path: 文件路径
            tokenizer: 分词器
            num_samples: 样本数量
            seq_len: 序列长度
            seed: 随机种子
            format: 文件格式（text, json, csv）
            text_column: 对于JSON/CSV文件，指定文本列名
        """
        super().__init__(tokenizer, num_samples, seq_len, seed)
        self.file_path = Path(file_path)
        self.format = format.lower()
        self.text_column = text_column

        # 验证文件格式
        valid_formats = ["text", "json", "csv"]
        if self.format not in valid_formats:
            raise ValueError(
                f"Unknown format: {format}. Valid formats: {valid_formats}"
            )

        # 验证文件存在
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 对于JSON/CSV，必须指定text_column
        if self.format in ["json", "csv"] and text_column is None:
            raise ValueError(f"text_column must be specified for {self.format} format")

    def get_data(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        获取数据

        Returns:
            数据列表，每个元素是一个元组，包含token tensor
        """
        # 根据格式加载数据
        if self.format == "text":
            texts = self._load_text()
        elif self.format == "json":
            texts = self._load_json()
        elif self.format == "csv":
            texts = self._load_csv()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # 随机采样
        if len(texts) > self.num_samples:
            random.seed(self.seed)
            texts = random.sample(texts, self.num_samples)

        # 分词
        data = []
        for text in texts:
            tokens = self.tokenize(text)
            tokens = self.pad_sequence(tokens)
            data.append((tokens,))

        return data

    def _load_text(self) -> List[str]:
        """加载文本文件"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按行分割
        texts = [line.strip() for line in content.split("\n") if line.strip()]
        return texts

    def _load_json(self) -> List[str]:
        """加载JSON文件"""
        texts = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            # 尝试按行读取JSON Lines格式
            try:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if isinstance(data, dict):
                            if self.text_column in data:
                                texts.append(data[self.text_column])
                        elif isinstance(data, str):
                            texts.append(data)
            except json.JSONDecodeError:
                # 如果不是JSON Lines，尝试读取整个文件
                f.seek(0)
                data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and self.text_column in item:
                            texts.append(item[self.text_column])
                        elif isinstance(item, str):
                            texts.append(item)
                elif isinstance(data, dict) and self.text_column in data:
                    texts.append(data[self.text_column])

        return texts

    def _load_csv(self) -> List[str]:
        """加载CSV文件"""
        texts = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if self.text_column not in reader.fieldnames:
                raise ValueError(f"Column '{self.text_column}' not found in CSV file")

            for row in reader:
                texts.append(row[self.text_column])

        return texts


class TextDataLoader(CustomDataLoader):
    """纯文本文件加载器"""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
    ):
        super().__init__(file_path, tokenizer, num_samples, seq_len, seed, "text")


class JSONDataLoader(CustomDataLoader):
    """JSON文件加载器"""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        text_column: str = "text",
    ):
        super().__init__(
            file_path, tokenizer, num_samples, seq_len, seed, "json", text_column
        )


class CSVDataLoader(CustomDataLoader):
    """CSV文件加载器"""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        text_column: str = "text",
    ):
        super().__init__(
            file_path, tokenizer, num_samples, seq_len, seed, "csv", text_column
        )
