"""
Magnitude Visualizer - Magnitude可视化器

可视化magnitude统计指标
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from .base import BaseVisualizer


class MagnitudeVisualizer(BaseVisualizer):
    """
    Magnitude可视化器
    
    可视化magnitude统计：
    - Top-1/2/3
    - Median
    - Min
    """
    
    def visualize(self, stats_data: np.ndarray, layer_names: list, 
                  plot_type: str = 'input', component_type: str = 'down_proj',
                  **kwargs) -> Optional[str]:
        """
        可视化magnitude统计
        
        Args:
            stats_data: 统计数据，形状为 [num_samples, num_metrics, num_layers]
            layer_names: layer名称列表
            plot_type: 'input' 或 'output'
            component_type: 组件类型
            
        Returns:
            保存的文件路径
        """
        # 计算平均值
        mean_stats = np.mean(stats_data, axis=0)  # [num_metrics, num_layers]
        
        if plot_type == 'input':
            return self._plot_input_magnitude(mean_stats, layer_names, component_type)
        elif plot_type == 'output':
            return self._plot_output_magnitude(mean_stats, layer_names, component_type)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    def _plot_input_magnitude(self, stats: np.ndarray, layer_names: list, 
                             component_type: str) -> str:
        """
        绘制输入magnitude
        
        Args:
            stats: 统计数据 [5, num_layers]
            layer_names: layer名称列表
            component_type: 组件类型
            
        Returns:
            保存的文件路径
        """
        colors = ["cornflowerblue", "mediumseagreen", "C4", "teal", "dimgrey", "gold"]
        
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=self.dpi)
        
        x_axis = np.arange(stats.shape[1]) + 1
        
        # Top-1/2/3
        for j in range(3):
            if stats.shape[0] > j:
                ax.plot(x_axis, stats[j], label=f"Top-{j+1}", 
                       color=colors[j], linestyle="-", marker="o", markersize=5)
        
        # Median
        if stats.shape[0] > 3:
            ax.plot(x_axis, stats[3], label="Median", 
                   color=colors[3], linestyle="-", marker="v", markersize=5)
        
        # Min
        if stats.shape[0] > 4:
            ax.plot(x_axis, stats[4], label="Min", 
                   color=colors[4], linestyle="-", marker="o", markersize=5)
        
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Maximum Value (token-wise)', fontsize=14)
        
        if self.show_legend:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        filename = f'{self.model_name}-{component_type}-input'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_output_magnitude(self, stats: np.ndarray, layer_names: list,
                              component_type: str) -> str:
        """
        绘制输出magnitude
        
        Args:
            stats: 统计数据 [5, num_layers]
            layer_names: layer名称列表
            component_type: 组件类型
            
        Returns:
            保存的文件路径
        """
        colors = ["cornflowerblue", "mediumseagreen", "C4", "teal", "dimgrey", "gold"]
        
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=self.dpi)
        
        x_axis = np.arange(stats.shape[1]) + 1
        
        # Max
        if stats.shape[0] > 0:
            ax.plot(x_axis, stats[0], label="Max", 
                   color=colors[0], linestyle="-", marker="o", markersize=5)
        
        # Median
        if stats.shape[0] > 1:
            ax.plot(x_axis, stats[1], label="Median", 
                   color=colors[1], linestyle="-", marker="v", markersize=5)
        
        # Min-1/2/3
        for j in range(2, min(5, stats.shape[0])):
            ax.plot(x_axis, stats[j], label=f"Min-{j-1}", 
                   color=colors[j], linestyle="-", marker="o", markersize=5)
        
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Maximum Value (token-wise)', fontsize=14)
        
        if self.show_legend:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        filename = f'{self.model_name}-{component_type}-output'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_weight(self, stats_data: np.ndarray, layer_names: list,
                      component_type: str = 'down_proj', **kwargs) -> Optional[str]:
        """
        可视化权重magnitude统计
        
        Args:
            stats_data: 统计数据，形状为 [5, num_layers]
            layer_names: layer名称列表
            component_type: 组件类型
            
        Returns:
            保存的文件路径
        """
        return self._plot_weight_magnitude(stats_data, layer_names, component_type)
    
    def _plot_weight_magnitude(self, stats: np.ndarray, layer_names: list,
                              component_type: str) -> str:
        """
        绘制权重magnitude
        
        Args:
            stats: 统计数据 [5, num_layers]
            layer_names: layer名称列表
            component_type: 组件类型
            
        Returns:
            保存的文件路径
        """
        colors = ["cornflowerblue", "mediumseagreen", "C4", "teal", "dimgrey", "gold"]
        
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=self.dpi)
        
        x_axis = np.arange(stats.shape[1]) + 1
        
        # Top-1/2/3
        for j in range(3):
            if stats.shape[0] > j:
                ax.plot(x_axis, stats[j], label=f"Top-{j+1}", 
                       color=colors[j], linestyle="-", marker="o", markersize=5)
        
        # Median
        if stats.shape[0] > 3:
            ax.plot(x_axis, stats[3], label="Median", 
                   color=colors[3], linestyle="-", marker="v", markersize=5)
        
        # Min
        if stats.shape[0] > 4:
            ax.plot(x_axis, stats[4], label="Min", 
                   color=colors[4], linestyle="-", marker="o", markersize=5)
        
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Weight Magnitude (neuron-wise)', fontsize=14)
        ax.set_title(f'Weight Magnitude Analysis - {component_type}', fontsize=14)
        
        if self.show_legend:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        filename = f'{self.model_name}-{component_type}-weight'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def get_required_stat_type(self) -> str:
        """返回需要的统计类型"""
        return "magnitude"