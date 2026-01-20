"""
Outlier Visualizer - Outlier可视化器

可视化outlier统计指标
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from collections import Counter
from .base import BaseVisualizer


class OutlierVisualizer(BaseVisualizer):
    """
    Outlier可视化器
    
    可视化outlier统计：
    - 每层outlier token数量
    - outlier token位置分布
    - outlier token内容
    """
    
    def visualize_layer_wise_count(self, stats: np.ndarray, **kwargs) -> Optional[str]:
        """
        可视化每层outlier token数量
        
        Args:
            stats: 统计数据，形状为 [num_samples, 1, num_layers]
            
        Returns:
            保存的文件路径
        """
        mean_stats = np.mean(stats, axis=0)  # [1, num_layers]
        
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=self.dpi)
        x_axis = np.arange(mean_stats.shape[1]) + 1
        
        ax.plot(x_axis, mean_stats[0], color="cornflowerblue", 
               linestyle="-", marker="o", markersize=5)
        
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Avg. # of outlier tokens', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        filename = f'{self.model_name}-outlier-count'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_token_position(self, position_stats: list, **kwargs) -> Optional[str]:
        """
        可视化outlier token位置分布
        
        Args:
            position_stats: outlier token位置列表
            
        Returns:
            保存的文件路径
        """
        data = Counter(position_stats)
        total = sum(data.values())
        percentages = {k: (v / total) * 100 for k, v in data.items()}
        
        # 取前2个最频繁的位置，其余归为"Others"
        sorted_percentages = dict(sorted(percentages.items(), 
                                        key=lambda item: item[1], 
                                        reverse=True)[:2])
        sorted_percentages['Others'] = sum(v for k, v in percentages.items() 
                                          if k not in sorted_percentages)
        
        labels = list(sorted_percentages.keys())
        values = list(sorted_percentages.values())
        
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=self.dpi)
        
        # 使用数值索引作为x轴
        x_indices = np.arange(len(labels))
        bars = ax.bar(x_indices, values, color="cornflowerblue")
        
        # 设置x轴标签
        ax.set_xticks(x_indices)
        ax.set_xticklabels(labels)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, 
                   f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_xlabel('Position Index', fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=14)
        
        filename = f'{self.model_name}-outlier-position'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_token_content(self, token_content: dict, top_n: int = 10, **kwargs) -> Optional[str]:
        """
        可视化outlier token内容
        
        Args:
            token_content: token内容字典
            top_n: 显示前N个token
            
        Returns:
            保存的文件路径
        """
        if 'decoded_tokens' not in token_content:
            print("Warning: No decoded tokens available")
            return None
        
        decoded_tokens = token_content['decoded_tokens']
        
        # 按数量排序，取前N个
        sorted_tokens = sorted(decoded_tokens.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_tokens:
            print("Warning: No tokens to visualize")
            return None
        
        tokens = [item[0] for item in sorted_tokens]
        counts = [item[1] for item in sorted_tokens]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # 截断过长的token
        tokens_display = [t[:20] + '...' if len(t) > 20 else t for t in tokens]
        
        bars = ax.bar(range(len(tokens)), counts, color="cornflowerblue")
        
        ax.set_xlabel('Token', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens_display, rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, 
                   str(int(height)), ha='center', va='bottom')
        
        plt.tight_layout()
        
        filename = f'{self.model_name}-outlier-token'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def get_required_stat_type(self) -> str:
        """返回需要的统计类型"""
        return "outlier"