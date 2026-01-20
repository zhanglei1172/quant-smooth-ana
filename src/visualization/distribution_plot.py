"""
Distribution Visualizer - 分布可视化器

可视化激活值分布统计（箱型图）
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from .base import BaseVisualizer


class DistributionVisualizer(BaseVisualizer):
    """
    分布可视化器
    
    可视化激活值分布：
    - 箱型图（Boxplot）
    """
    
    def visualize_boxplot(self, stats: dict, layer_names: list, 
                         component_type: str = 'down_proj', **kwargs) -> Optional[str]:
        """
        可视化箱型图
        
        Args:
            stats: 分布统计数据
                {
                    'values': [num_layers, num_values],  # 每层的所有激活值
                    'mean': [num_layers],
                    'std': [num_layers],
                    'median': [num_layers],
                    'q25': [num_layers],
                    'q75': [num_layers],
                    'min': [num_layers],
                    'max': [num_layers]
                }
            layer_names: layer名称列表
            component_type: 组件类型
            
        Returns:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        # 准备数据
        values = stats['values']
        
        # 绘制箱型图
        bp = ax.boxplot(values, patch_artist=True, showfliers=True)
        
        # 设置颜色
        for patch in bp['boxes']:
            patch.set_facecolor('cornflowerblue')
            patch.set_alpha(0.7)
        
        for whisker in bp['whiskers']:
            whisker.set_linestyle('--')
            whisker.set_color('gray')
        
        for cap in bp['caps']:
            cap.set_color('gray')
        
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)
        
        # 设置标签
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Activation Value', fontsize=14)
        ax.set_title(f'Activation Distribution (Boxplot) - {component_type}', fontsize=14)
        
        # 设置x轴标签（显示部分）
        num_layers = len(layer_names)
        if num_layers > 10:
            # 只显示部分标签
            step = max(1, num_layers // 10)
            labels = [f'{i}' if i % step == 0 else '' for i in range(1, num_layers + 1)]
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels([f'{i}' for i in range(1, num_layers + 1)])
        
        ax.grid(True, alpha=0.3, axis='y')
        
        filename = f'{self.model_name}-{component_type}-distribution'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_histogram(self, stats: dict, layer_names: list,
                          component_type: str = 'down_proj',
                          num_layers_to_show: int = 4, **kwargs) -> Optional[str]:
        """
        可视化直方图
        
        Args:
            stats: 分布统计数据
            layer_names: layer名称列表
            component_type: 组件类型
            num_layers_to_show: 显示的层数
            
        Returns:
            保存的文件路径
        """
        # 选择要显示的层
        layers_to_show = layer_names[:num_layers_to_show]
        
        # 计算子图布局
        n_cols = 2
        n_rows = (num_layers_to_show + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), dpi=self.dpi)
        
        # 如果只有一行，确保axes是2D数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        # 绘制每个层的直方图
        for idx, layer_name in enumerate(layers_to_show):
            ax = axes[idx]
            layer_idx = layer_names.index(layer_name)
            
            # 获取该层的激活值
            values = stats['values'][layer_idx]
            
            # 绘制直方图
            ax.hist(values, bins=50, color='cornflowerblue', alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            
            # 添加统计线
            ax.axvline(stats['mean'][layer_idx], color='red', 
                      linestyle='--', linewidth=2, label='Mean')
            ax.axvline(stats['median'][layer_idx], color='green', 
                      linestyle='--', linewidth=2, label='Median')
            
            ax.set_xlabel('Activation Value', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Layer {layer_idx + 1}', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(num_layers_to_show, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        filename = f'{self.model_name}-{component_type}-histogram'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def get_required_stat_type(self) -> str:
        """返回需要的统计类型"""
        return "distribution"