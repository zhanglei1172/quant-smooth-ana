"""
Config Loader - 配置加载器

加载和管理YAML配置文件
"""

import yaml
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    配置加载器

    加载YAML配置文件，并支持参数覆盖
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}

        if config_path and os.path.exists(config_path):
            self.load(config_path)

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键（支持嵌套，如 'model.name'）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        设置配置项

        Args:
            key: 配置键（支持嵌套，如 'model.name'）
            value: 配置值
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        更新配置

        Args:
            updates: 更新字典
        """

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        deep_update(self.config, updates)

    def to_dict(self) -> Dict[str, Any]:
        """
        获取配置字典

        Returns:
            配置字典
        """
        return self.config.copy()

    def save(self, save_path: str):
        """
        保存配置到文件

        Args:
            save_path: 保存路径
        """
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def parse_overrides(overrides: list) -> Dict[str, Any]:
        """
        解析命令行参数覆盖

        Args:
            overrides: 参数覆盖列表，格式如 ['model.name=llama', 'data.num_samples=128']

        Returns:
            更新字典
        """
        updates = {}

        for override in overrides:
            if "=" in override:
                key, value = override.split("=", 1)

                # 尝试转换为适当的类型
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == "true":
                            value = True
                        elif value.lower() == "false":
                            value = False

                # 设置嵌套键
                keys = key.split(".")
                current = updates

                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]

                current[keys[-1]] = value

        return updates
