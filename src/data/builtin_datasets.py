"""
Built-in Dataset Loader - 内置数据集加载器

支持内置数据集：Pile、WikiText-2、C4、RedPajama
"""

import torch
import random
from typing import List, Tuple
from .base import BaseDataLoader

from datasets import load_dataset


class BuiltinDataLoader(BaseDataLoader):
    """
    内置数据集加载器

    支持的数据集：
    - pile: Pile数据集
    - wikitext2: WikiText-2数据集
    - c4: C4数据集
    - redpajama: RedPajama数据集
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        split: str = "validation",
    ):
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
        valid_datasets = ["pile", "wikitext2", "c4", "redpajama"]
        if self.dataset_name not in valid_datasets:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Valid datasets: {valid_datasets}"
            )

    def get_data(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        获取数据

        Returns:
            数据列表，每个元素是一个元组，包含token tensor
        """
        # 这里使用模拟数据，实际使用时应该从HuggingFace加载
        # 为了简化，我们使用随机token作为示例

        return self.data


class PileDataLoader(BuiltinDataLoader):
    """Pile数据集加载器"""

    def __init__(
        self,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        split: str = "validation",
    ):
        super().__init__("pile", tokenizer, num_samples, seq_len, seed, split)
        self.data = get_pile(tokenizer, num_samples, seed, seq_len)


class WikiText2DataLoader(BuiltinDataLoader):
    """WikiText-2数据集加载器"""

    def __init__(
        self,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        split: str = "validation",
    ):
        super().__init__("wikitext2", tokenizer, num_samples, seq_len, seed, split)
        self.data = get_wikitext2(
            tokenizer, num_samples, seed, seq_len, test_only=False
        )


class C4DataLoader(BuiltinDataLoader):
    """C4数据集加载器"""

    def __init__(
        self,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        split: str = "validation",
    ):
        super().__init__("c4", tokenizer, num_samples, seq_len, seed, split)
        self.data = get_c4(tokenizer, num_samples, seed, seq_len, test_only=False)


class RedPajamaDataLoader(BuiltinDataLoader):
    """RedPajama数据集加载器"""

    def __init__(
        self,
        tokenizer,
        num_samples: int = 64,
        seq_len: int = 1024,
        seed: int = 0,
        split: str = "validation",
    ):
        super().__init__("redpajama", tokenizer, num_samples, seq_len, seed, split)


def get_builtin_dataloader(
    dataset_name: str,
    tokenizer,
    num_samples: int = 64,
    seq_len: int = 1024,
    seed: int = 0,
    split: str = "validation",
) -> BuiltinDataLoader:
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

    if dataset_name == "pile":
        return PileDataLoader(tokenizer, num_samples, seq_len, seed, split)
    elif dataset_name == "wikitext2":
        return WikiText2DataLoader(tokenizer, num_samples, seq_len, seed, split)
    elif dataset_name == "c4":
        return C4DataLoader(tokenizer, num_samples, seq_len, seed, split)
    elif dataset_name == "redpajama":
        return RedPajamaDataLoader(tokenizer, num_samples, seq_len, seed, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_pile(tokenizer, train_size, seed, seqlen):
    print("get_pile")
    traindata = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    traindata = traindata.shuffle(seed=seed)

    random.seed(seed)
    val_sample_ratio = 0.9
    trainloader = []
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]

        trainloader.append(inp)

    return trainloader


def get_wikitext2(tokenizer, train_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    val_sample_ratio = (
        0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    )
    for _ in range(train_size):
        i = random.randint(
            0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen
        )
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]

        trainloader.append(inp)

    return trainloader


def get_c4(tokenizer, train_size, seed, seqlen, test_only):
    print("get_c4")
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc

    random.seed(seed)
    trainloader = []
    val_sample_ratio = (
        0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    )
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader
