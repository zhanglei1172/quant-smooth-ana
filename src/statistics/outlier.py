"""
Outlier Calculator - Outlier统计计算器

计算outlier token的统计指标（数量、位置、内容）
"""

from typing import Any, Dict, List

import numpy as np
import torch

from .base import BaseStatCalculator


class OutlierCalculator(BaseStatCalculator):
    """
    Outlier统计计算器
    
    计算outlier token的统计指标：
    - 每层的outlier token数量
    - outlier token的位置分布
    - outlier token的内容
    """
    
    def calculate(self, dataloader, component_type='hidden_state',
                  outlier_threshold=64, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """
        计算outlier统计（默认方法，调用calculate_all）
        
        Args:
            dataloader: 数据加载器
            component_type: 组件类型
            outlier_threshold: outlier阈值
            tokenizer: 分词器
            **kwargs: 其他参数
            
        Returns:
            results: 包含所有统计指标的字典
        """
        return self.calculate_all(dataloader, component_type, outlier_threshold, tokenizer, **kwargs)
    
    def calculate_layer_wise_count(self, dataloader, component_type='hidden_state',
                                   outlier_threshold=64, **kwargs) -> np.ndarray:
        """
        计算每层outlier token数量
        
        Args:
            dataloader: 数据加载器
            component_type: 组件类型
            outlier_threshold: outlier阈值（相对于median的倍数）
            
        Returns:
            stats: numpy数组，形状为 [num_samples, 1, num_layers]
                  - stats[:, 0, :] = outlier token数量
        """
        # 获取所有匹配的layer
        layer_names = self.matcher.match_by_component(component_type)
        num_layers = len(layer_names)
        
        if num_layers == 0:
            raise ValueError(f"No layers found for component type: {component_type}")
        
        stats = []
        num_samples = len(dataloader)
        
        for data in dataloader:
            input_data = data[0].reshape(1, -1)
            activation_dict = self._collect_activations(input_data, layer_names, is_input=False)
            
            sample_stats = np.zeros((1, num_layers))
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_name not in activation_dict:
                    continue
                
                activation = activation_dict[layer_name]
                activation_abs = activation.abs().float()
                
                # 按token统计：取每个token的最大值
                activation_abs = activation_abs.max(dim=-1).values
                
                # 排序
                sorted_vals = torch.sort(activation_abs.flatten(), descending=True)
                median = sorted_vals.values[len(sorted_vals.values) // 2]
                
                # 计算相对于median的比率
                ratio = sorted_vals.values / (median + 1e-8)
                
                # 统计outlier数量
                num_outliers = (ratio > outlier_threshold).sum()
                sample_stats[0, layer_idx] = num_outliers.cpu()
            
            stats.append(sample_stats)
        
        return np.array(stats)
    
    def calculate_token_position(self, dataloader, component_type='hidden_state',
                                 outlier_threshold=20, **kwargs) -> List[int]:
        """
        计算outlier token位置分布
        
        Args:
            dataloader: 数据加载器
            component_type: 组件类型
            outlier_threshold: outlier阈值
            
        Returns:
            positions: outlier token位置列表
        """
        positions = []
        layer_names = self.matcher.match_by_component(component_type)
        
        for data in dataloader:
            input_data = data[0].reshape(1, -1)
            activation_dict = self._collect_activations(input_data, layer_names, is_input=False)
            
            for layer_name in layer_names:
                if layer_name not in activation_dict:
                    continue
                
                activation = activation_dict[layer_name]
                activation_abs = activation.abs()
                
                # 按token统计
                activation_abs = activation_abs.max(dim=-1).values
                
                # 排序
                sorted_vals = torch.sort(activation_abs.flatten(), descending=True)
                median = sorted_vals.values[len(sorted_vals.values) // 2]
                
                # 计算比率
                ratio = sorted_vals.values / (median + 1e-8)
                
                # 获取outlier位置
                outlier_indices = sorted_vals.indices[ratio > outlier_threshold]
                positions.extend(outlier_indices.tolist())
        
        return positions
    
    def calculate_token_content(self, dataloader, component_type='hidden_state',
                                outlier_threshold=20, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """
        计算outlier token的内容
        
        Args:
            dataloader: 数据加载器
            component_type: 组件类型
            outlier_threshold: outlier阈值
            tokenizer: 分词器（用于解码token）
            
        Returns:
            token_info: 字典，包含token内容和统计信息
        """
        token_counts = {}
        layer_names = self.matcher.match_by_component(component_type)
        
        for data in dataloader:
            input_data = data[0].reshape(1, -1)
            activation_dict = self._collect_activations(input_data, layer_names, is_input=False)
            
            for layer_name in layer_names:
                if layer_name not in activation_dict:
                    continue
                
                activation = activation_dict[layer_name]
                activation_abs = activation.abs()
                
                # 按token统计
                activation_abs = activation_abs.max(dim=-1).values
                
                # 排序
                sorted_vals = torch.sort(activation_abs.flatten(), descending=True)
                median = sorted_vals.values[len(sorted_vals.values) // 2]
                
                # 计算比率
                ratio = sorted_vals.values / (median + 1e-8)
                
                # 获取outlier token
                outlier_indices = sorted_vals.indices[ratio > outlier_threshold]
                
                # 统计token
                for idx in outlier_indices:
                    # 将flatten后的索引转换为序列位置
                    seq_pos = idx.item() // activation_abs.shape[-1]
                    token_id = input_data[0, seq_pos].item()
                    if token_id not in token_counts:
                        token_counts[token_id] = 0
                    token_counts[token_id] += 1
        
        # 解码token
        if tokenizer is not None:
            decoded_tokens = {}
            for token_id, count in token_counts.items():
                try:
                    decoded = tokenizer.decode([token_id])
                    decoded_tokens[decoded] = count
                except:
                    decoded_tokens[f"<token_{token_id}>"] = count
            
            return {
                'token_counts': token_counts,
                'decoded_tokens': decoded_tokens,
                'total_outliers': sum(token_counts.values())
            }
        else:
            return {
                'token_counts': token_counts,
                'total_outliers': sum(token_counts.values())
            }
    
    def calculate_all(self, dataloader, component_type='hidden_state',
                     outlier_threshold=64, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """
        计算所有outlier统计指标
        
        Args:
            dataloader: 数据加载器
            component_type: 组件类型
            outlier_threshold: outlier阈值
            tokenizer: 分词器
            
        Returns:
            results: 包含所有统计指标的字典
        """
        layer_wise_count = self.calculate_layer_wise_count(
            dataloader, component_type, outlier_threshold, **kwargs
        )
        
        token_position = self.calculate_token_position(
            dataloader, component_type, outlier_threshold, **kwargs
        )
        
        token_content = self.calculate_token_content(
            dataloader, component_type, outlier_threshold, tokenizer, **kwargs
        )
        
        return {
            'layer_wise_count': layer_wise_count,
            'token_position': token_position,
            'token_content': token_content,
            'outlier_threshold': outlier_threshold
        }
    
    def get_stat_name(self) -> str:
        """返回统计指标名称"""
        return "outlier"