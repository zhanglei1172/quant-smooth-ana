"""
Data Export - 数据导出工具

支持将统计数据导出为CSV和JSON格式
"""

import csv
import json
import os
import numpy as np
from typing import Dict, Any, List


class DataExporter:
    """
    数据导出器

    支持导出为CSV和JSON格式
    """

    def __init__(self, save_dir: str, model_name: str = "model"):
        """
        初始化数据导出器

        Args:
            save_dir: 保存目录
            model_name: 模型名称
        """
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(save_dir, exist_ok=True)

    def export_csv(self, data: Dict[str, Any], filename: str = "stats"):
        """
        导出数据为CSV格式

        Args:
            data: 要导出的数据字典
            filename: 文件名（不含扩展名）

        Returns:
            保存的文件路径
        """
        csv_path = os.path.join(self.save_dir, f"{self.model_name}-{filename}.csv")

        # 展平数据
        flattened = self._flatten_dict(data)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])

            for key, value in flattened.items():
                if isinstance(value, (list, np.ndarray)):
                    writer.writerow([key, str(value)])
                else:
                    writer.writerow([key, value])

        return csv_path

    def export_json(self, data: Dict[str, Any], filename: str = "stats"):
        """
        导出数据为JSON格式

        Args:
            data: 要导出的数据字典
            filename: 文件名（不含扩展名）

        Returns:
            保存的文件路径
        """
        json_path = os.path.join(self.save_dir, f"{self.model_name}-{filename}.json")

        # 转换numpy类型为Python原生类型
        converted = self._convert_numpy_types(data)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)

        return json_path

    def export_stats_csv(
        self,
        stats: np.ndarray,
        layer_names: List[str],
        metrics: List[str],
        filename: str = "stats",
    ):
        """
        导出统计数据为CSV格式

        Args:
            stats: 统计数据，形状为 [num_samples, num_metrics, num_layers]
            layer_names: layer名称列表
            metrics: 指标名称列表
            filename: 文件名（不含扩展名）

        Returns:
            保存的文件路径
        """
        csv_path = os.path.join(self.save_dir, f"{self.model_name}-{filename}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 写入表头
            header = ["Layer"] + metrics
            writer.writerow(header)

            # 计算平均值
            mean_stats = np.mean(stats, axis=0)  # [num_metrics, num_layers]

            # 写入数据
            for layer_idx, layer_name in enumerate(layer_names):
                row = [layer_name]
                for metric_idx in range(mean_stats.shape[0]):
                    row.append(mean_stats[metric_idx, layer_idx])
                writer.writerow(row)

        return csv_path

    def export_all(self, data: Dict[str, Any], filename: str = "analysis"):
        """
        导出所有数据（CSV和JSON）

        Args:
            data: 要导出的数据字典
            filename: 文件名（不含扩展名）

        Returns:
            保存的文件路径字典
        """
        paths = {}
        paths["csv"] = self.export_csv(data, filename)
        paths["json"] = self.export_json(data, filename)
        return paths

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """
        展平嵌套字典

        Args:
            d: 字典
            parent_key: 父键名
            sep: 分隔符

        Returns:
            展平后的字典
        """
        items = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        转换numpy类型为Python原生类型

        Args:
            obj: 要转换的对象

        Returns:
            转换后的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
