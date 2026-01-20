"""
Magnitude Calculator - Magnitude统计计算器

计算激活值的magnitude统计（Top-1/2/3、Median、Min）
"""

import torch
import numpy as np
from typing import List, Optional
from .base import BaseStatCalculator


class MagnitudeCalculator(BaseStatCalculator):
    """
    Magnitude统计计算器
    
    计算激活值的magnitude统计指标：
    - Top-1/2/3: 最大的1/2/3个值
    - Median: 中位数
    - Min: 最小值
    """
    
    def calculate(self, dataloader, component_type='down_proj', 
                  per_tensor=False, is_input=True, **kwargs) -> np.ndarray:
        """
        计算magnitude统计
        
        Args:
            dataloader: 数据加载器
            component_type: 组件类型（如 'down_proj', 'q_proj'）
            per_tensor: 是否按张量统计（False则按token统计）
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
        
        if num_layers == 0:
            raise ValueError(f"No layers found for component type: {component_type}")
        
        stats = []
        num_samples = len(dataloader)
        
        for sample_idx in range(num_samples):
            data = dataloader[sample_idx][0]
            
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
                
                if not per_tensor:
                    # 按token统计：取每个token的最大值
                    activation_abs = activation_abs.max(dim=-1).values
                
                # 排序
                sorted_vals = torch.sort(activation_abs.flatten(), descending=True).values
                
                if is_input:
                    # 输入统计：Top-1/2/3, Median, Min
                    sample_stats[0, layer_idx] = sorted_vals[0].cpu().float() if len(sorted_vals) > 0 else 0  # Top-1
                    sample_stats[1, layer_idx] = sorted_vals[1].cpu().float() if len(sorted_vals) > 1 else 0  # Top-2
                    sample_stats[2, layer_idx] = sorted_vals[2].cpu().float() if len(sorted_vals) > 2 else 0  # Top-3
                    sample_stats[3, layer_idx] = torch.median(activation_abs).cpu().float()  # Median
                    sample_stats[4, layer_idx] = torch.min(activation_abs).cpu().float()  # Min
                else:
                    # 输出统计：Max, Median, Min-1/2/3
                    sample_stats[0, layer_idx] = torch.max(activation_abs).cpu().float()  # Max
                    sample_stats[1, layer_idx] = torch.median(activation_abs).cpu().float()  # Median
                    sample_stats[2, layer_idx] = sorted_vals[-1].cpu().float() if len(sorted_vals) > 0 else 0  # Min-1
                    sample_stats[3, layer_idx] = sorted_vals[-2].cpu().float() if len(sorted_vals) > 1 else 0  # Min-2
                    sample_stats[4, layer_idx] = sorted_vals[-3].cpu().float() if len(sorted_vals) > 2 else 0  # Min-3
            
            stats.append(sample_stats)
        
        return np.array(stats)
    
    def calculate_for_components(self, dataloader, component_types: List[str],
                                 per_tensor=False, is_input=True, **kwargs) -> dict:
        """
        为多个组件类型计算magnitude统计
        
        Args:
            dataloader: 数据加载器
            component_types: 组件类型列表
            per_tensor: 是否按张量统计
            is_input: True统计输入，False统计输出
            
        Returns:
            统计结果字典 {component_type: stats_array}
        """
        results = {}
        for component_type in component_types:
            try:
                stats = self.calculate(dataloader, component_type, per_tensor, is_input, **kwargs)
                results[component_type] = stats
            except Exception as e:
                print(f"Warning: Could not calculate stats for {component_type}: {e}")
        
        return results
    
    def get_stat_name(self) -> str:
        """返回统计指标名称"""
        return "magnitude"