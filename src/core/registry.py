"""
Model Registry - 插件式模型注册机制

这个模块实现了模型适配器的注册机制，允许通过装饰器注册新的模型适配器，
无需修改核心代码即可扩展对新模型的支持。
"""

from typing import Type, Dict, Any, Optional


class ModelRegistry:
    """
    模型注册器，实现插件式架构

    使用装饰器 @ModelRegistry.register("model_family") 来注册模型适配器
    """

    # 类变量，存储所有注册的模型适配器
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, model_family: str):
        """
        装饰器，用于注册模型适配器

        Args:
            model_family: 模型家族名称，如 "llama", "qwen", "mistral"

        Returns:
            装饰器函数

        Example:
            @ModelRegistry.register("llama")
            class LlamaAdapter(BaseModelAdapter):
                pass
        """

        def decorator(adapter_class: Type):
            cls._registry[model_family.lower()] = adapter_class
            return adapter_class

        return decorator

    @classmethod
    def get_adapter(cls, model: Any, model_family: Optional[str] = None):
        """
        根据模型实例获取对应的适配器

        Args:
            model: 模型实例
            model_family: 可选，手动指定模型家族。如果未提供，则自动检测

        Returns:
            模型适配器实例

        Raises:
            ValueError: 如果找不到对应的适配器
        """
        if model_family is None:
            model_family = cls._extract_model_family(model)

        if model_family is None or model_family not in cls._registry:
            supported = cls.list_supported_models()
            raise ValueError(
                f"No adapter registered for model family: '{model_family}'. "
                f"Supported models: {supported}"
            )

        adapter_class = cls._registry[model_family]
        return adapter_class(model)

    @classmethod
    def _extract_model_family(cls, model: Any) -> Optional[str]:
        """
        从模型实例提取模型家族

        Args:
            model: 模型实例

        Returns:
            模型家族名称，如果无法识别则返回None
        """
        # 检查模型类型
        if hasattr(model, "__class__"):
            class_name = model.__class__.__name__
            class_name_lower = class_name.lower()

            # 尝试匹配注册的模型家族
            for family in cls._registry:
                if family.lower() in class_name_lower:
                    return family.lower()

        # 检查模型配置
        if hasattr(model, "config"):
            config = model.config
            if hasattr(config, "model_type"):
                model_type = config.model_type.lower()
                for family in cls._registry:
                    if family.lower() in model_type:
                        return family.lower()

            if hasattr(config, "architectures"):
                for arch in config.architectures:
                    arch_lower = arch.lower()
                    for family in cls._registry:
                        if family.lower() in arch_lower:
                            return family.lower()

        return None

    @classmethod
    def list_supported_models(cls) -> list:
        """
        列出所有支持的模型

        Returns:
            支持的模型家族名称列表
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_family: str) -> bool:
        """
        检查模型家族是否已注册

        Args:
            model_family: 模型家族名称

        Returns:
            如果已注册返回True，否则返回False
        """
        return model_family.lower() in cls._registry

    @classmethod
    def unregister(cls, model_family: str):
        """
        注销模型适配器（主要用于测试）

        Args:
            model_family: 模型家族名称
        """
        if model_family.lower() in cls._registry:
            del cls._registry[model_family.lower()]

    @classmethod
    def clear(cls):
        """
        清空所有注册的适配器（主要用于测试）
        """
        cls._registry.clear()
