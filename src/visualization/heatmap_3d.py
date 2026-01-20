"""
Heatmap 3D Visualizer - 3D热力图可视化器

可视化3D热力图（激活值矩阵）
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .base import BaseVisualizer


class Heatmap3DVisualizer(BaseVisualizer):
    """
    3D热力图可视化器

    可视化3D激活值矩阵：
    - 3D散点图
    """

    def __init__(self, config: dict):
        """
        初始化可视化器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.dpi_scale = config.get("dpi_scale", 1)

    def _calculate_statistics(self, tensor: np.ndarray) -> dict:
        """
        计算统计信息

        Args:
            tensor: 输入张量

        Returns:
            统计信息字典
        """
        mean_val = np.mean(tensor)
        std_val = np.std(tensor)
        p99_val = np.percentile(tensor, 99)
        max_val = np.max(tensor)
        min_val = np.min(tensor)

        # Count outliers (values > 3 standard deviations from mean)
        outlier_threshold = mean_val + 3 * std_val
        outliers = np.sum(tensor > outlier_threshold)
        outlier_percentage = (outliers / tensor.size) * 100

        return {
            "mean": mean_val,
            "std": std_val,
            "p99": p99_val,
            "max": max_val,
            "min": min_val,
            "outliers": outliers,
            "outlier_percentage": outlier_percentage,
        }

    def _add_statistics_annotations(
        self, ax, stats: dict, tensor_shape: tuple, is_3d: bool = False
    ):
        """
        添加统计信息注释

        Args:
            ax: matplotlib轴对象
            stats: 统计信息字典
            tensor_shape: 张量形状
            is_3d: 是否为3D图
        """
        # Calculate ratio
        ratio_max_p99 = stats["max"] / stats["p99"] if stats["p99"] != 0 else float("inf")
        ratio_color = "red" if ratio_max_p99 > 64 else "green"

        # Build statistics text
        stats_text = f"Shape: {tensor_shape}\n"
        stats_text += f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}\n"
        stats_text += f"Min: {stats['min']:.3f}, Max: {stats['max']:.3f}\n"
        stats_text += f"P99: {stats['p99']:.3f}\n"
        stats_text += f"Outliers (>3σ): {stats['outliers']} ({stats['outlier_percentage']:.1f}%)"

        # Position main statistics text
        if not is_3d:
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Add prominent max/p99 ratio display
            ratio_text = f"Max/P99: {ratio_max_p99:.1f}"
            ax.text(
                0.02,
                0.20,
                ratio_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=16,
                color=ratio_color,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.9, edgecolor=ratio_color, linewidth=2
                ),
            )

    def visualize_3d_heatmap(
        self,
        data: np.ndarray,
        layer_names: list,
        component_type: str = "hidden_state",
        view_angle: tuple = (20, -45),
        max_points: int = 10000,
        **kwargs,
    ) -> Optional[str]:
        """
        可视化3D热力图（使用表面图）

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

        # 选择第一层的数据进行可视化
        layer_data = data[0]  # [seq_len, hidden_dim]

        # 计算统计信息
        stats = self._calculate_statistics(layer_data)

        # 创建3D图形
        fig = plt.figure(figsize=(12, 8), dpi=self.dpi // self.dpi_scale)
        ax = fig.add_subplot(111, projection="3d")

        # 创建网格
        X = np.arange(layer_data.shape[1])  # Channel (hidden_dim)
        Y = np.arange(layer_data.shape[0])  # Token (seq_len)
        X, Y = np.meshgrid(X, Y)

        # 计算步长以提高性能
        rstride = max(1, layer_data.shape[0] // 100)
        cstride = max(1, layer_data.shape[1] // 100)

        # 绘制表面图
        ax.plot_surface(
            X,
            Y,
            layer_data,
            cmap="coolwarm",
            antialiased=False,
            shade=True,
            linewidth=0.5,
            rstride=rstride,
            cstride=cstride,
        )

        # 添加统计信息注释
        self._add_statistics_annotations(ax, stats, layer_data.shape, is_3d=True)

        # 设置标签
        ax.set_xlabel("Channel", fontsize=14)
        ax.set_ylabel("Token", fontsize=14)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.tick_params(axis="z", labelsize=16)

        # 设置视角
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        plt.tight_layout(pad=0.1)

        filename = f"{self.model_name}-{component_type}-heatmap-3d"
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return save_path

    def visualize_2d_heatmap(
        self,
        data: np.ndarray,
        layer_names: list,
        component_type: str = "hidden_state",
        layer_idx: int = 0,
        **kwargs,
    ) -> Optional[str]:
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

        # 计算统计信息
        stats = self._calculate_statistics(layer_data)

        # 创建2D图形
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi // self.dpi_scale)

        # 绘制热力图
        im = ax.imshow(
            layer_data.T,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
        )

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Activation Value", fontsize=12)

        # 设置标签
        ax.set_xlabel("Sequence Position", fontsize=12)
        ax.set_ylabel("Hidden Dimension", fontsize=12)
        ax.set_title(
            f"2D Activation Heatmap - {component_type}\n"
            f"Layer {layer_idx + 1} (Seq: {layer_data.shape[0]}, Hidden: {layer_data.shape[1]})",
            fontsize=14,
        )

        # 添加统计信息注释
        self._add_statistics_annotations(ax, stats, layer_data.shape, is_3d=False)

        filename = f"{self.model_name}-{component_type}-heatmap-2d-layer{layer_idx}"
        save_path = self._get_save_path(filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        return save_path

    def get_required_stat_type(self) -> str:
        """返回需要的统计类型"""
        return "heatmap_3d"
