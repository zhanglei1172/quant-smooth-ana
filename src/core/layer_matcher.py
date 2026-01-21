"""
Layer Matcher - Layer名称正则匹配系统

这个模块提供了灵活的layer名称匹配功能，支持正则表达式、组件类型、范围等多种匹配方式。
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Union


class LayerMatcher:
    """
    Layer名称匹配系统，支持正则表达式

    提供多种匹配方式：
    1. 正则表达式匹配
    2. 组件类型匹配（q_proj, down_proj等）
    3. 范围匹配（指定层索引）
    4. 组合匹配
    """

    def __init__(self, model_adapter):
        """
        初始化LayerMatcher

        Args:
            model_adapter: 模型适配器实例
        """
        self.adapter = model_adapter
        self._all_layer_names = None
        self._compiled_patterns = {}
        self._selected_layers = None  # 用于存储选中的层

    def match_layers(self, pattern: str) -> List[str]:
        """
        根据正则模式匹配所有符合条件的layer名称

        Args:
            pattern: 正则表达式模式，例如：

        Returns:
            匹配的layer名称列表，按字母顺序排序
        """
        all_layer_names = self._get_all_layer_names()

        # 编译正则表达式（缓存）
        if pattern not in self._compiled_patterns:
            self._compiled_patterns[pattern] = re.compile(pattern)

        compiled_pattern = self._compiled_patterns[pattern]
        matched = [name for name in all_layer_names if compiled_pattern.search(name)]

        # 按层索引排序
        return self._sort_by_layer_index(matched)

    def match_by_component(
        self, component_type: str, layer_indices: Optional[Union[List[int], str]] = None
    ) -> List[str]:
        """
        根据组件类型匹配layer

        Args:
            component_type: 组件类型，如 'q_proj', 'down_proj', 'hidden_state'
            layer_indices: 指定layer索引，可以是：
                - None: 所有层
                - "all": 所有层
                - [0, 5, 10]: 指定的层索引列表

        Returns:
            匹配的layer名称列表，按层索引排序
        """
        # 获取组件的正则模式
        pattern = self.adapter.get_layer_name_pattern(component_type)

        if pattern is None:
            raise ValueError(f"Unknown component type: {component_type}")

        # 首先匹配所有层
        matched = self.match_layers(pattern)

        # 处理layer_indices参数
        if layer_indices is not None and layer_indices != "all":
            if isinstance(layer_indices, list):
                # 从匹配的层中过滤出指定索引的层
                layer_indices_set = set(layer_indices)
                filtered = []
                for layer_name in matched:
                    # 从 layer 名称中提取层索引
                    idx_match = re.search(r"layers\.(\d+)", layer_name)
                    if idx_match:
                        layer_idx = int(idx_match.group(1))
                        if layer_idx in layer_indices_set:
                            filtered.append(layer_name)
                matched = filtered
            else:
                raise ValueError(f"Invalid layer_indices: {layer_indices}")

        # 如果有选择的layer，进行过滤
        if hasattr(self, "_selected_layers") and self._selected_layers is not None:
            selected_set = set(self._selected_layers)
            matched = [layer for layer in matched if layer in selected_set]

        return matched

    def match_by_range(self, component_type: str, start: int, end: int) -> List[str]:
        """
        匹配指定范围的层

        Args:
            component_type: 组件类型
            start: 起始层索引（包含）
            end: 结束层索引（包含）

        Returns:
            匹配的layer名称列表
        """
        layer_indices = list(range(start, end + 1))
        return self.match_by_component(component_type, layer_indices)

    def match_by_components(
        self,
        component_types: List[str],
        layer_indices: Optional[Union[List[int], str]] = None,
    ) -> List[str]:
        """
        根据多个组件类型匹配layer

        Args:
            component_types: 组件类型列表，如 ['q_proj', 'k_proj', 'v_proj']
            layer_indices: 指定layer索引

        Returns:
            匹配的layer名称列表
        """
        matched = []
        for component_type in component_types:
            matched.extend(self.match_by_component(component_type, layer_indices))
        return sorted(matched)

    def match_by_pattern_list(self, patterns: List[str]) -> List[str]:
        """
        根据多个正则模式匹配layer

        Args:
            patterns: 正则表达式列表

        Returns:
            匹配的layer名称列表（去重）
        """
        matched = set()
        for pattern in patterns:
            matched.update(self.match_layers(pattern))
        return sorted(matched)

    def get_layer_indices(self, component_type: str) -> List[int]:
        """
        获取指定组件类型的所有层索引

        Args:
            component_type: 组件类型

        Returns:
            层索引列表
        """
        layer_names = self.match_by_component(component_type, layer_indices="all")
        indices = []

        for name in layer_names:
            # 提取层索引（假设格式为 layers.{idx}.xxx）
            match = re.search(r"layers\.(\d+)", name)
            if match:
                indices.append(int(match.group(1)))

        return sorted(set(indices))

    def _get_all_layer_names(self) -> List[str]:
        """
        获取模型中所有layer的名称

        Returns:
            所有layer名称列表
        """
        if self._all_layer_names is None:
            self._all_layer_names = []
            for name, module in self.adapter.model.named_modules():
                # 只包含有参数的模块
                if hasattr(module, "parameters") and list(module.parameters()):
                    self._all_layer_names.append(name)

        return self._all_layer_names

    def _sort_by_layer_index(self, layer_names: List[str]) -> List[str]:
        """
        按层索引排序layer名称

        Args:
            layer_names: layer名称列表

        Returns:
            按层索引排序的layer名称列表
        """

        def extract_index(name):
            # 提取层索引
            match = re.search(
                r"(?:layers|model\.layers|language_model\.model\.layers)\.(\d+)", name
            )
            if match:
                return int(match.group(1))
            return 0

        return sorted(layer_names, key=extract_index)

    def get_matching_info(self, pattern: str) -> dict:
        """
        获取匹配的详细信息

        Args:
            pattern: 正则表达式模式

        Returns:
            包含匹配信息的字典：
            {
                'matched_count': 匹配的数量,
                'matched_layers': 匹配的layer名称列表,
                'total_layers': 总layer数量
            }
        """
        all_layers = self._get_all_layer_names()
        matched = self.match_layers(pattern)

        return {
            "matched_count": len(matched),
            "matched_layers": matched,
            "total_layers": len(all_layers),
        }

    def set_selected_layers(self, layers: List[str]):
        """
        设置选中的层列表，后续 match_by_component 等方法会基于此列表过滤

        Args:
            layers: 选中的层名称列表
        """
        self._selected_layers = layers
        print(f"LayerMatcher: Set {len(layers)} selected layers")

    def clear_selected_layers(self):
        """
        清除选中的层列表，恢复默认行为（匹配所有层）
        """
        self._selected_layers = None
        print("LayerMatcher: Cleared selected layers filter")

    def get_selected_layers(self) -> Optional[List[str]]:
        """
        获取当前选中的层列表

        Returns:
            选中的层名称列表，如果未设置则返回 None
        """
        return self._selected_layers


class LayerSelector:
    """
    Layer选择器，提供高级的layer选择功能

    支持通过配置文件中的layer_selection配置来选择layer
    """

    def __init__(self, layer_matcher: LayerMatcher):
        """
        初始化LayerSelector

        Args:
            layer_matcher: LayerMatcher实例
        """
        self.matcher = layer_matcher

    def apply_config(self, layer_selection: dict) -> List[str]:
        """
        应用配置并设置LayerMatcher的选中层

        Args:
            layer_selection: 配置字典

        Returns:
            选择的layer名称列表
        """
        selected = self.select_from_config(layer_selection)
        if selected:
            self.matcher.set_selected_layers(selected)
        return selected

    def clear(self):
        """
        清除LayerMatcher的选中层设置
        """
        self.matcher.clear_selected_layers()

    def select_from_config(self, layer_selection: dict) -> List[str]:
        """
        根据配置选择layer

        Args:
            layer_selection: 配置字典，格式为：
                {
                    'components': {
                        'q_proj': 'all',  # 或 [0, 5, 10]
                        'down_proj': [0, 1, 2]
                    },
                    'patterns': [
                        'layers\\.\\d+\\.mlp\\.down_proj'
                    ],
                    'ranges': [
                        {'component': 'self_attn', 'start': 0, 'end': 10}
                    ]
                }

        Returns:
            选择的layer名称列表
        """
        selected = set()

        # 方式1: 通过组件类型
        if "components" in layer_selection:
            components = layer_selection["components"]
            for component_type, layer_indices in components.items():
                try:
                    matched = self.matcher.match_by_component(component_type, layer_indices)
                    selected.update(matched)
                except ValueError as e:
                    print(f"Warning: {e}, skipping component '{component_type}'")

        # 方式2: 通过正则表达式
        if "patterns" in layer_selection:
            patterns = layer_selection["patterns"]
            matched = self.matcher.match_by_pattern_list(patterns)
            selected.update(matched)

        # 方式3: 通过范围
        if "ranges" in layer_selection:
            for range_config in layer_selection["ranges"]:
                component = range_config["component"]
                start = range_config["start"]
                end = range_config["end"]
                try:
                    matched = self.matcher.match_by_range(component, start, end)
                    selected.update(matched)
                except ValueError as e:
                    print(f"Warning: {e}, skipping range for '{component}'")

        return sorted(selected)
