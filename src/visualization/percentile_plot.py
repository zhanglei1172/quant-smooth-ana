"""
Percentile Visualizer - 百分位数范围可视化器

可视化激活值在hidden dimension上的百分位数范围分布
展示不同百分位数（25/75, 1/99, 0.01/99.99）和Min/Max的范围
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .base import BaseVisualizer


class PercentileVisualizer(BaseVisualizer):
    """
    百分位数范围可视化器

    可视化每个hidden dimension上激活值的百分位数分布：
    - 25/75 Percentile
    - 1/99 Percentile
    - 0.01/99.99 Percentile (1/9999)
    - Min/Max
    """

    def visualize_percentile_range(
        self,
        percentile_data: dict,
        layer_name: str,
        component_type: str = "q_proj",
        plot_type: str = "input",
        figsize: Tuple[int, int] = (10, 5),
        **kwargs,
    ) -> Optional[str]:
        """
        可视化单层的百分位数范围图

        Args:
            percentile_data: 百分位数统计数据，包含:
                {
                    'p25': np.ndarray,     # 25th percentile [hidden_dim]
                    'p75': np.ndarray,     # 75th percentile [hidden_dim]
                    'p1': np.ndarray,      # 1st percentile [hidden_dim]
                    'p99': np.ndarray,     # 99th percentile [hidden_dim]
                    'p0_01': np.ndarray,   # 0.01th percentile [hidden_dim]
                    'p99_99': np.ndarray,  # 99.99th percentile [hidden_dim]
                    'min': np.ndarray,     # min values [hidden_dim]
                    'max': np.ndarray,     # max values [hidden_dim]
                }
            layer_name: 层名称
            component_type: 组件类型（如 q_proj, k_proj）
            plot_type: 'input' 或 'output'
            figsize: 图片大小

        Returns:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        hidden_dim = len(percentile_data['p25'])
        x = np.arange(hidden_dim)

        # 定义颜色
        colors = {
            '25/75': 'red',
            '1/99': 'orange',
            '0.01/99.99': 'gold',
            'min/max': 'blue',
        }

        # 绘制Min/Max（最外层）
        ax.fill_between(
            x,
            percentile_data['min'],
            percentile_data['max'],
            alpha=0.3,
            color=colors['min/max'],
            label='Min/Max'
        )

        # 绘制0.01/99.99百分位数
        ax.fill_between(
            x,
            percentile_data['p0_01'],
            percentile_data['p99_99'],
            alpha=0.4,
            color=colors['0.01/99.99'],
            label='1/9999 Percentile'
        )

        # 绘制1/99百分位数
        ax.fill_between(
            x,
            percentile_data['p1'],
            percentile_data['p99'],
            alpha=0.5,
            color=colors['1/99'],
            label='1/99 Percentile'
        )

        # 绘制25/75百分位数
        ax.fill_between(
            x,
            percentile_data['p25'],
            percentile_data['p75'],
            alpha=0.7,
            color=colors['25/75'],
            label='25/75 Percentile'
        )

        # 设置标签
        ax.set_xlabel('hidden dimension index', fontsize=12)
        ax.set_ylabel('value', fontsize=12)
        
        # 生成标题
        title = f"{layer_name}_{plot_type}"
        ax.set_title(title, fontsize=14)

        # 添加图例
        if self.show_legend:
            ax.legend(loc='upper right', fontsize=10)

        ax.grid(True, alpha=0.3)

        # 保存文件
        # 清理layer_name中的特殊字符
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        filename = f"{self.model_name}-{safe_layer_name}-{component_type}-{plot_type}-percentile"
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        return save_path

    def visualize_percentile_line(
        self,
        percentile_data: dict,
        layer_name: str,
        component_type: str = "q_proj",
        plot_type: str = "input",
        figsize: Tuple[int, int] = (10, 5),
        **kwargs,
    ) -> Optional[str]:
        """
        可视化单层的百分位数范围图（线条模式）

        这种模式与用户提供的图片更加接近，使用线条而非填充

        Args:
            percentile_data: 百分位数统计数据
            layer_name: 层名称
            component_type: 组件类型
            plot_type: 'input' 或 'output'
            figsize: 图片大小

        Returns:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        hidden_dim = len(percentile_data['p25'])
        x = np.arange(hidden_dim)

        # 定义颜色和线宽
        line_configs = {
            'min/max': {'color': 'blue', 'alpha': 0.5, 'linewidth': 0.5},
            '0.01/99.99': {'color': 'gold', 'alpha': 0.7, 'linewidth': 0.8},
            '1/99': {'color': 'orange', 'alpha': 0.8, 'linewidth': 1.0},
            '25/75': {'color': 'red', 'alpha': 1.0, 'linewidth': 1.2},
        }

        # 绘制Min/Max
        ax.plot(x, percentile_data['min'], 
                color=line_configs['min/max']['color'],
                alpha=line_configs['min/max']['alpha'],
                linewidth=line_configs['min/max']['linewidth'])
        ax.plot(x, percentile_data['max'],
                color=line_configs['min/max']['color'],
                alpha=line_configs['min/max']['alpha'],
                linewidth=line_configs['min/max']['linewidth'],
                label='Min/Max')

        # 绘制0.01/99.99百分位数
        ax.plot(x, percentile_data['p0_01'],
                color=line_configs['0.01/99.99']['color'],
                alpha=line_configs['0.01/99.99']['alpha'],
                linewidth=line_configs['0.01/99.99']['linewidth'])
        ax.plot(x, percentile_data['p99_99'],
                color=line_configs['0.01/99.99']['color'],
                alpha=line_configs['0.01/99.99']['alpha'],
                linewidth=line_configs['0.01/99.99']['linewidth'],
                label='1/9999 Percentile')

        # 绘制1/99百分位数
        ax.plot(x, percentile_data['p1'],
                color=line_configs['1/99']['color'],
                alpha=line_configs['1/99']['alpha'],
                linewidth=line_configs['1/99']['linewidth'])
        ax.plot(x, percentile_data['p99'],
                color=line_configs['1/99']['color'],
                alpha=line_configs['1/99']['alpha'],
                linewidth=line_configs['1/99']['linewidth'],
                label='1/99 Percentile')

        # 绘制25/75百分位数
        ax.plot(x, percentile_data['p25'],
                color=line_configs['25/75']['color'],
                alpha=line_configs['25/75']['alpha'],
                linewidth=line_configs['25/75']['linewidth'])
        ax.plot(x, percentile_data['p75'],
                color=line_configs['25/75']['color'],
                alpha=line_configs['25/75']['alpha'],
                linewidth=line_configs['25/75']['linewidth'],
                label='25/75 Percentile')

        # 设置标签
        ax.set_xlabel('hidden dimension index', fontsize=12)
        ax.set_ylabel('value', fontsize=12)

        # 生成标题
        title = f"{layer_name}_{plot_type}"
        ax.set_title(title, fontsize=14)

        # 添加图例
        if self.show_legend:
            ax.legend(loc='upper right', fontsize=10)

        ax.grid(True, alpha=0.3)

        # 保存文件
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        filename = f"{self.model_name}-{safe_layer_name}-{component_type}-{plot_type}-percentile-line"
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        return save_path

    def visualize_multi_layer_percentile(
        self,
        all_percentile_data: List[dict],
        layer_names: List[str],
        component_type: str = "q_proj",
        plot_type: str = "input",
        cols: int = 2,
        figsize_per_subplot: Tuple[int, int] = (8, 4),
        style: str = "fill",
        **kwargs,
    ) -> Optional[str]:
        """
        可视化多层的百分位数范围图（子图模式）

        Args:
            all_percentile_data: 所有层的百分位数统计数据列表
            layer_names: 层名称列表
            component_type: 组件类型
            plot_type: 'input' 或 'output'
            cols: 每行的子图数量
            figsize_per_subplot: 每个子图的大小
            style: 'fill' 使用填充模式, 'line' 使用线条模式

        Returns:
            保存的文件路径
        """
        num_layers = len(layer_names)
        rows = (num_layers + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(figsize_per_subplot[0] * cols, figsize_per_subplot[1] * rows),
            dpi=self.dpi
        )

        # 确保axes是2D数组
        if num_layers == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # 定义颜色
        colors = {
            '25/75': 'red',
            '1/99': 'orange',
            '0.01/99.99': 'gold',
            'min/max': 'blue',
        }

        for idx, (percentile_data, layer_name) in enumerate(zip(all_percentile_data, layer_names)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            hidden_dim = len(percentile_data['p25'])
            x = np.arange(hidden_dim)

            if style == "fill":
                # 填充模式
                ax.fill_between(x, percentile_data['min'], percentile_data['max'],
                                alpha=0.3, color=colors['min/max'], label='Min/Max')
                ax.fill_between(x, percentile_data['p0_01'], percentile_data['p99_99'],
                                alpha=0.4, color=colors['0.01/99.99'], label='1/9999 Percentile')
                ax.fill_between(x, percentile_data['p1'], percentile_data['p99'],
                                alpha=0.5, color=colors['1/99'], label='1/99 Percentile')
                ax.fill_between(x, percentile_data['p25'], percentile_data['p75'],
                                alpha=0.7, color=colors['25/75'], label='25/75 Percentile')
            else:
                # 线条模式
                ax.plot(x, percentile_data['min'], color=colors['min/max'], alpha=0.5, linewidth=0.5)
                ax.plot(x, percentile_data['max'], color=colors['min/max'], alpha=0.5, linewidth=0.5, label='Min/Max')
                ax.plot(x, percentile_data['p0_01'], color=colors['0.01/99.99'], alpha=0.7, linewidth=0.8)
                ax.plot(x, percentile_data['p99_99'], color=colors['0.01/99.99'], alpha=0.7, linewidth=0.8, label='1/9999 Percentile')
                ax.plot(x, percentile_data['p1'], color=colors['1/99'], alpha=0.8, linewidth=1.0)
                ax.plot(x, percentile_data['p99'], color=colors['1/99'], alpha=0.8, linewidth=1.0, label='1/99 Percentile')
                ax.plot(x, percentile_data['p25'], color=colors['25/75'], alpha=1.0, linewidth=1.2)
                ax.plot(x, percentile_data['p75'], color=colors['25/75'], alpha=1.0, linewidth=1.2, label='25/75 Percentile')

            # 提取简化的层名称用于标题
            simple_name = self._simplify_layer_name(layer_name)
            ax.set_title(f"{simple_name}_{plot_type}", fontsize=11)
            ax.set_xlabel('hidden dimension index', fontsize=10)
            ax.set_ylabel('value', fontsize=10)
            ax.grid(True, alpha=0.3)

            # 只在第一个子图显示图例
            if idx == 0 and self.show_legend:
                ax.legend(loc='upper right', fontsize=8)

        # 隐藏多余的子图
        for idx in range(num_layers, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        # 保存文件
        filename = f"{self.model_name}-{component_type}-{plot_type}-percentile-multi"
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        return save_path

    def _simplify_layer_name(self, layer_name: str) -> str:
        """
        简化层名称用于显示

        Args:
            layer_name: 完整的层名称

        Returns:
            简化后的层名称
        """
        # 尝试提取层编号和组件名称
        # 例如 "model.layers.5.self_attn.q_proj" -> "layer5_q"
        import re
        
        # 匹配层编号
        layer_match = re.search(r'layers?\.(\d+)', layer_name)
        layer_num = layer_match.group(1) if layer_match else ""
        
        # 匹配组件名称
        component_patterns = [
            (r'q_proj', 'q'),
            (r'k_proj', 'k'),
            (r'v_proj', 'v'),
            (r'o_proj', 'o'),
            (r'up_proj', 'up'),
            (r'down_proj', 'down'),
            (r'gate_proj', 'gate'),
        ]
        
        component = ""
        for pattern, short_name in component_patterns:
            if re.search(pattern, layer_name):
                component = short_name
                break
        
        if layer_num and component:
            return f"layer{layer_num}_{component}"
        elif layer_num:
            return f"layer{layer_num}"
        else:
            # 返回最后两个部分
            parts = layer_name.split('.')
            return '_'.join(parts[-2:]) if len(parts) >= 2 else layer_name

    def get_required_stat_type(self) -> str:
        """返回需要的统计类型"""
        return "percentile"
