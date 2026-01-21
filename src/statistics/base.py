"""
Base Stat Calculator - 统计计算器基类

提供统计计算的基类和通用方法
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseStatCalculator(ABC):
    """
    统计计算器基类

    所有统计计算器都应该继承这个类
    """

    def __init__(self, model_adapter, layer_matcher, memory_manager):
        """
        初始化统计计算器

        Args:
            model_adapter: 模型适配器
            layer_matcher: Layer匹配器
            memory_manager: 显存管理器
        """
        self.adapter = model_adapter
        self.matcher = layer_matcher
        self.memory_manager = memory_manager

    @abstractmethod
    def calculate(self, dataloader, **kwargs) -> Any:
        """
        计算统计指标

        Args:
            dataloader: 数据加载器
            **kwargs: 其他参数

        Returns:
            统计结果
        """
        pass

    @abstractmethod
    def get_stat_name(self) -> str:
        """
        返回统计指标名称

        Returns:
            统计指标名称
        """
        pass

    def _collect_activations(self, input_data, layer_names, is_input=True):
        """
        收集激活值的通用方法

        Args:
            input_data: 输入数据
            layer_names: 需要收集激活值的layer名称列表
            is_input: True收集输入，False收集输出

        Returns:
            激活值字典 {layer_name: activation}
        """
        activation_dict = {}
        hooks = []

        def make_hook(layer_name, is_input):
            def hook(module, input, output):
                if is_input:
                    activation_dict[layer_name] = input[0].detach().cpu()
                else:
                    if isinstance(output, tuple):
                        output = output[0]
                    activation_dict[layer_name] = output.detach().cpu()

            return hook

        # 注册hooks
        for layer_name in layer_names:
            try:
                layer = self.adapter.model.get_submodule(layer_name)
                hooks.append(
                    layer.register_forward_hook(make_hook(layer_name, is_input))
                )
            except Exception as e:
                print(f"Warning: Could not register hook for layer {layer_name}: {e}")

        # 前向传播
        with torch.no_grad():
            # 获取embedding层的设备
            embeddings = self.adapter.get_embeddings()
            if embeddings and len(embeddings) > 0:
                embed_device = next(embeddings[0].parameters()).device
                # 将输入数据移动到embedding层的设备
                input_data = input_data.to(embed_device)

            # 让模型自己处理设备映射
            # 禁用KV缓存以避免旋转位置编码问题
            _ = self.adapter.model(input_data, use_cache=False)

        # 移除hooks
        for hook in hooks:
            hook.remove()

        return activation_dict
