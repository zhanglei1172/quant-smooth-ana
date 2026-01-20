"""
Base Visualizer - 可视化器基类

定义可视化器的标准接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os


class BaseVisualizer(ABC):
    """
    可视化器基类

    所有可视化器都应该继承这个基类，并实现visualize方法
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化器

        Args:
            config: 配置字典
        """
        self.config = config
        self.save_dir = config.get("save_dir", "./figures")
        self.model_name = config.get("model_name", "model")
        self.dpi = config.get("dpi", 200)
        self.show_legend = config.get("show_legend", True)
        self.output_format = config.get("output_format", "png")

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)

    # @abstractmethod
    def visualize(self, data: Any, **kwargs) -> Optional[str]:
        """
        可视化数据

        Args:
            data: 要可视化的数据
            **kwargs: 其他参数

        Returns:
            保存的文件路径，如果未保存则返回None
        """
        pass

    def _get_save_path(self, filename: str, format: Optional[str] = None) -> str:
        """
        获取保存路径

        Args:
            filename: 文件名（不含扩展名）
            format: 文件格式（如png、pdf、svg），如果未指定则使用默认格式

        Returns:
            完整的保存路径
        """
        if format is None:
            format = self.output_format

        return os.path.join(self.save_dir, f"{filename}.{format}")

    @abstractmethod
    def get_required_stat_type(self) -> str:
        """
        返回需要的统计类型

        Returns:
            统计类型名称
        """
        pass
