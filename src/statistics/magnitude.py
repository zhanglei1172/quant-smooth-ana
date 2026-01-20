"""
Magnitude Calculator - Magnitude统计计算器

计算激活值的magnitude统计（Top-1/2/3、Median、Min）
"""

from typing import List, Optional

import numpy as np
import torch

from .base import BaseStatCalculator


class MagnitudeCalculator(BaseStatCalculator):
    """
    Magnitude统计计算器

    计算激活值的magnitude统计指标：
    - Top-1/2/3: 最大的1/2/3个值
    - Median: 中位数
    - Min: 最小值
    """

    def calculate(
        self,
        dataloader,
        component_type="down_proj",
        reduce_dim=None,
        is_input=True,
        **kwargs,
    ) -> np.ndarray:
        """
        计算magnitude统计

        Args:
            dataloader: 数据加载器
            component_type: 组件类型（如 'down_proj', 'q_proj'）
            reduce_dim: 是否按张量统计（False则按token统计）
            is_input: True统计输入，False统计输出

        Returns:
            stats: numpy数组，形状为 [num_samples, 5, num_layers]
                  - stats[:, 0, :] = Top-1
                  - stats[:, 1, :] = Top-2
                  - stats[:, 2, :] = Top-3
                  - stats[:, 3, :] = Median
                  - stats[:, 4, :] = Min
        """
        # 获取所有匹配的layer
        layer_names = self.matcher.match_by_component(component_type)
        num_layers = len(layer_names)

        # 调试信息
        pattern = self.adapter.get_layer_name_pattern(component_type)
        print(f"\nDebug: component_type={component_type}, pattern={pattern}")
        print(f"Debug: matched layers: {layer_names}")

        if num_layers == 0:
            raise ValueError(f"No layers found for component type: {component_type}")

        stats = []
        num_samples = len(dataloader)

        for sample_idx in range(num_samples):
            data = dataloader[sample_idx][0].reshape(1, -1)

            # 前向传播，收集激活值
            activation_dict = self._collect_activations(data, layer_names, is_input)

            # 计算统计
            sample_stats = np.zeros((5, num_layers))
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_name not in activation_dict:
                    print(f"Warning: No activation data for layer {layer_name}")
                    continue

                activation = activation_dict[layer_name]
                activation_abs = activation.abs()
                if reduce_dim:
                    activation_abs = activation_abs.max(dims=reduce_dim).values
                else:
                    activation_abs = activation_abs.max()

                # 排序
                sorted_vals = torch.sort(
                    activation_abs.flatten(), descending=True
                ).values

                if is_input:
                    # 输入统计：Top-1/2/3, Median, Min
                    sample_stats[0, layer_idx] = (
                        sorted_vals[0].cpu().float() if len(sorted_vals) > 0 else 0
                    )  # Top-1
                    sample_stats[1, layer_idx] = (
                        sorted_vals[1].cpu().float() if len(sorted_vals) > 1 else 0
                    )  # Top-2
                    sample_stats[2, layer_idx] = (
                        sorted_vals[2].cpu().float() if len(sorted_vals) > 2 else 0
                    )  # Top-3
                    sample_stats[3, layer_idx] = (
                        torch.median(activation_abs).cpu().float()
                    )  # Median
                    sample_stats[4, layer_idx] = (
                        torch.min(activation_abs).cpu().float()
                    )  # Min
                else:
                    # 输出统计：Max, Median, Min-1/2/3
                    sample_stats[0, layer_idx] = (
                        torch.max(activation_abs).cpu().float()
                    )  # Max
                    sample_stats[1, layer_idx] = (
                        torch.median(activation_abs).cpu().float()
                    )  # Median
                    sample_stats[2, layer_idx] = (
                        sorted_vals[-1].cpu().float() if len(sorted_vals) > 0 else 0
                    )  # Min-1
                    sample_stats[3, layer_idx] = (
                        sorted_vals[-2].cpu().float() if len(sorted_vals) > 1 else 0
                    )  # Min-2
                    sample_stats[4, layer_idx] = (
                        sorted_vals[-3].cpu().float() if len(sorted_vals) > 2 else 0
                    )  # Min-3

            stats.append(sample_stats)

        return np.array(stats)

    def calculate_for_components(
        self,
        dataloader,
        component_types: List[str],
        reduce_dim=None,
        is_input=True,
        **kwargs,
    ) -> dict:
        """
        为多个组件类型计算magnitude统计

        Args:
            dataloader: 数据加载器
            component_types: 组件类型列表
            reduce_dim: 是否按张量统计
            is_input: True统计输入，False统计输出

        Returns:
            统计结果字典 {component_type: stats_array}
        """
        results = {}
        for component_type in component_types:
            try:
                stats = self.calculate(
                    dataloader, component_type, reduce_dim, is_input, **kwargs
                )
                results[component_type] = stats
            except Exception as e:
                print(f"Warning: Could not calculate stats for {component_type}: {e}")

        return results

    def calculate_weight(self, component_type="down_proj", **kwargs) -> np.ndarray:
        """
        计算权重magnitude统计

        Args:
            component_type: 组件类型（如 'down_proj', 'q_proj'）

        Returns:
            stats: numpy数组，形状为 [5, num_layers]
                  - stats[0, :] = Top-1
                  - stats[1, :] = Top-2
                  - stats[2, :] = Top-3
                  - stats[3, :] = Median
                  - stats[4, :] = Min
        """
        # 获取所有匹配的layer
        layer_names = self.matcher.match_by_component(component_type)
        num_layers = len(layer_names)

        if num_layers == 0:
            raise ValueError(f"No layers found for component type: {component_type}")

        # 计算统计
        stats = np.zeros((5, num_layers))

        for layer_idx, layer_name in enumerate(layer_names):
            try:
                # 获取权重张量
                layer = self.adapter.model.get_submodule(layer_name)
                weight = layer.weight  # [out_features, in_features]

                # 计算绝对值
                weight_abs = weight.abs()

                # 按token维度统计：取每个输出维度的最大值
                # weight_abs shape: [out_features, in_features]
                # 按in_features维度取最大值，得到每个输出神经元的最大权重
                weight_abs = weight_abs.max(dim=-2).values  # [out_features]

                # 排序
                sorted_vals = torch.sort(weight_abs.flatten(), descending=True).values

                # 统计：Top-1/2/3, Median, Min（与magnitude_input相同）
                stats[0, layer_idx] = (
                    sorted_vals[0].cpu().float() if len(sorted_vals) > 0 else 0
                )  # Top-1
                stats[1, layer_idx] = (
                    sorted_vals[1].cpu().float() if len(sorted_vals) > 1 else 0
                )  # Top-2
                stats[2, layer_idx] = (
                    sorted_vals[2].cpu().float() if len(sorted_vals) > 2 else 0
                )  # Top-3
                stats[3, layer_idx] = torch.median(weight_abs).cpu().float()  # Median
                stats[4, layer_idx] = torch.min(weight_abs).cpu().float()  # Min

            except Exception as e:
                print(
                    f"Warning: Could not calculate weight stats for layer {layer_name}: {e}"
                )
                # 保持为0

        return stats

    def calculate_distribution(
        self, dataloader, component_type="down_proj", is_input=True, **kwargs
    ) -> dict:
        """
        计算激活值分布

        Args:
            dataloader: 数据加载器
            component_type: 组件类型
            is_input: True统计输入，False统计输出

        Returns:
            distribution_data: {
                'values': [num_layers, num_values],  # 每层的所有激活值
                'mean': [num_layers],
                'std': [num_layers],
                'median': [num_layers],
                'q25': [num_layers],  # 25th percentile
                'q75': [num_layers],  # 75th percentile
                'min': [num_layers],
                'max': [num_layers]
            }
        """
        # 获取所有匹配的layer
        layer_names = self.matcher.match_by_component(component_type)
        num_layers = len(layer_names)

        if num_layers == 0:
            raise ValueError(f"No layers found for component type: {component_type}")

        # 初始化存储
        all_values = [[] for _ in range(num_layers)]

        num_samples = len(dataloader)

        for sample_idx in range(num_samples):
            data = dataloader[sample_idx][0].reshape(1, -1)

            # 前向传播，收集激活值
            activation_dict = self._collect_activations(data, layer_names, is_input)

            # 收集所有激活值
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_name not in activation_dict:
                    continue

                activation = activation_dict[layer_name]
                activation_abs = activation.abs().float()  # 转换为float32

                # 添加所有值
                all_values[layer_idx].extend(
                    activation_abs.flatten().cpu().numpy().tolist()
                )

        # 计算统计
        distribution_data = {
            "values": [],
            "mean": np.zeros(num_layers),
            "std": np.zeros(num_layers),
            "median": np.zeros(num_layers),
            "q25": np.zeros(num_layers),
            "q75": np.zeros(num_layers),
            "min": np.zeros(num_layers),
            "max": np.zeros(num_layers),
        }

        for layer_idx in range(num_layers):
            values = np.array(all_values[layer_idx])
            distribution_data["values"].append(values)
            distribution_data["mean"][layer_idx] = np.mean(values)
            distribution_data["std"][layer_idx] = np.std(values)
            distribution_data["median"][layer_idx] = np.median(values)
            distribution_data["q25"][layer_idx] = np.percentile(values, 25)
            distribution_data["q75"][layer_idx] = np.percentile(values, 75)
            distribution_data["min"][layer_idx] = np.min(values)
            distribution_data["max"][layer_idx] = np.max(values)

        return distribution_data

    def calculate_heatmap_data(
        self,
        dataloader,
        component_type="hidden_state",
        sample_idx=0,
        is_input=False,
        **kwargs,
    ) -> np.ndarray:
        """
        计算热力图数据

        Args:
            dataloader: 数据加载器
            component_type: 组件类型
            sample_idx: 要分析的样本索引
            is_input: True使用输入，False使用输出

        Returns:
            heatmap_data: numpy数组，形状为 [num_layers, seq_len, hidden_dim]
        """
        # 获取所有匹配的layer
        layer_names = self.matcher.match_by_component(component_type)
        num_layers = len(layer_names)

        if num_layers == 0:
            raise ValueError(f"No layers found for component type: {component_type}")

        # 获取指定样本
        data = dataloader[sample_idx][0].reshape(1, -1)

        # 前向传播，收集激活值
        activation_dict = self._collect_activations(data, layer_names, is_input)

        # 收集每层的激活值
        heatmap_data = []

        for layer_idx, layer_name in enumerate(layer_names):
            if layer_name not in activation_dict:
                print(f"Warning: No activation data for layer {layer_name}")
                # 使用零填充
                if layer_idx == 0:
                    # 从第一个layer获取shape
                    for other_name in layer_names:
                        if other_name in activation_dict:
                            activation = activation_dict[other_name]
                            if hasattr(activation, "shape"):
                                shape = activation.shape
                            else:
                                # 可能是BaseModelOutputWithPast对象
                                shape = activation.last_hidden_state.shape
                            heatmap_data.append(np.zeros(shape))
                            break
                continue

            activation = activation_dict[layer_name]

            # 处理不同类型的输出
            if hasattr(activation, "last_hidden_state"):
                # BaseModelOutputWithPast对象
                activation_abs = activation.last_hidden_state.abs()
            else:
                activation_abs = activation.abs()

            # 转换为float32
            activation_abs = activation_abs.float()

            # shape: [batch, seq_len, hidden_dim]
            if len(activation_abs.shape) == 3:
                # 取第一个样本
                heatmap_data.append(activation_abs[0].cpu().numpy())
            elif len(activation_abs.shape) == 2:
                # [seq_len, hidden_dim]
                heatmap_data.append(activation_abs.cpu().numpy())
            else:
                print(
                    f"Warning: Unexpected activation shape {activation.shape} for layer {layer_name}"
                )
                heatmap_data.append(np.zeros((1, 1)))

        return np.array(heatmap_data)

    def get_stat_name(self) -> str:
        """返回统计指标名称"""
        return "magnitude"
