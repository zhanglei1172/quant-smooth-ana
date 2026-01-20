"""
Heatmap 3D Visualizer - 3D热力图可视化器

可视化3D热力图（激活值矩阵）
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from .base import BaseVisualizer


class Heatmap3DVisualizer(BaseVisualizer):
    """
    3D热力图可视化器
    
    可视化3D激活值矩阵：
    - 3D散点图
    """
    
    def visualize_3d_heatmap(self, data: np.ndarray, layer_names: list,
                             component_type: str = 'hidden_state',
                             view_angle: tuple = (30, 45),
                             max_points: int = 10000, **kwargs) -> Optional[str]:
        """
        可视化3D热力图
        
        Args:
            data: 3D数据 [num_layers, seq_len, hidden_dim]
            layer_names: layer名称列表
            component_type: 组件类型
            view_angle: 视角 (elevation, azimuth)
            max_points: 最大显示点数（避免点太多）
            
        Returns:
            保存的文件路径
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        num_layers, seq_len, hidden_dim = data.shape
        
        # 创建3D网格
        X, Y, Z = np.meshgrid(
            np.arange(num_layers),  # layers
            np.arange(seq_len),     # seq_len
            np.arange(hidden_dim)   # hidden_dim
        )
        
        # 展平数据
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        values_flat = data.flatten()
        
        # 如果点太多，进行采样
        if len(values_flat) > max_points:
            indices = np.random.choice(len(values_flat), max_points, replace=False)
            X_flat = X_flat[indices]
            Y_flat = Y_flat[indices]
            Z_flat = Z_flat[indices]
            values_flat = values_flat[indices]
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点图
        scatter = ax.scatter(X_flat, Y_flat, Z_flat,
                           c=values_flat, cmap='viridis',
                           alpha=0.6, s=1)
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Activation Value', fontsize=12)
        
        # 设置视角
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # 设置标签
        ax.set_xlabel('Layers', fontsize=12)
        ax.set_ylabel('Sequence Position', fontsize=12)
        ax.set_zlabel('Hidden Dimension', fontsize=12)
        ax.set_title(f'3D Activation Heatmap - {component_type}\n'
                    f'(Layers: {num_layers}, Seq: {seq_len}, Hidden: {hidden_dim})', 
                    fontsize=14)
        
        # 设置刻度
        ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 5)))
        ax.set_yticks(np.arange(0, seq_len, max(1, seq_len // 5)))
        ax.set_zticks(np.arange(0, hidden_dim, max(1, hidden_dim // 5)))
        
        filename = f'{self.model_name}-{component_type}-heatmap-3d'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_2d_heatmap(self, data: np.ndarray, layer_names: list,
                            component_type: str = 'hidden_state',
                            layer_idx: int = 0, **kwargs) -> Optional[str]:
        """
        可视化2D热力图（单层）
        
        Args:
            data: 3D数据 [num_layers, seq_len, hidden_dim]
            layer_names: layer名称列表
            component_type: 组件类型
            layer_idx: 要可视化的层索引
            
        Returns:
            保存的文件路径
        """
        # 选择指定层
        if layer_idx >= data.shape[0]:
            layer_idx = 0
        
        layer_data = data[layer_idx]  # [seq_len, hidden_dim]
        
        # 创建2D图形
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # 绘制热力图
        im = ax.imshow(layer_data.T, aspect='auto', cmap='viridis',
                      interpolation='nearest', origin='lower')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Activation Value', fontsize=12)
        
        # 设置标签
        ax.set_xlabel('Sequence Position', fontsize=12)
        ax.set_ylabel('Hidden Dimension', fontsize=12)
        ax.set_title(f'2D Activation Heatmap - {component_type}\n'
                    f'Layer {layer_idx + 1} (Seq: {layer_data.shape[0]}, Hidden: {layer_data.shape[1]})',
                    fontsize=14)
        
        filename = f'{self.model_name}-{component_type}-heatmap-2d-layer{layer_idx}'
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def get_required_stat_type(self) -> str:
        """返回需要的统计类型"""
        return "heatmap_3d"