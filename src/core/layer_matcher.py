"""
Layer Matcher - Layer名称正则匹配系统

这个模块提供了灵活的layer名称匹配功能，只使用正则表达式匹配，
无需模型适配器，直接传入model实例即可。
"""

import re
from typing import List, Optional


class LayerMatcher:
    """
    Layer名称匹配系统，只使用正则表达式

    统一使用正则表达式进行layer匹配，支持任意模型架构。
    用户通过配置文件中的patterns正则表达式来选择需要分析的layer。
    """

    def __init__(self, model):
        """
        初始化LayerMatcher

        Args:
            model: PyTorch模型实例（任意模型）
        """
        self.model = model
        self._all_layer_names = None
        self._compiled_patterns = {}
        self._selected_layers = None  # 用于存储选中的层

    def match_layers(self, pattern: str) -> List[str]:
        """
        根据正则模式匹配所有符合条件的layer名称

        Args:
            pattern: 正则表达式模式，例如：
                - r"layers\\.\\d+\\.mlp\\.down_proj" 匹配所有down_proj
                - r"layers\\.[0-5]\\.self_attn\\.q_proj" 匹配0-5层的q_proj
                - r"model\\.layers\\.\\d+$" 匹配所有transformer层

        Returns:
            匹配的layer名称列表，按层索引排序
        """
        all_layer_names = self._get_all_layer_names()

        # 编译正则表达式（缓存）
        if pattern not in self._compiled_patterns:
            self._compiled_patterns[pattern] = re.compile(pattern)

        compiled_pattern = self._compiled_patterns[pattern]
        matched = [name for name in all_layer_names if compiled_pattern.search(name)]

        # 按层索引排序
        return self._sort_by_layer_index(matched)

    def match_by_patterns(self, patterns: List[str]) -> List[str]:
        """
        根据多个正则模式匹配layer

        Args:
            patterns: 正则表达式列表

        Returns:
            匹配的layer名称列表（去重，按层索引排序）
        """
        matched = set()
        for pattern in patterns:
            matched.update(self.match_layers(pattern))
        return self._sort_by_layer_index(list(matched))

    def get_layer_indices_from_names(self, layer_names: List[str]) -> List[int]:
        """
        从layer名称列表中提取层索引

        Args:
            layer_names: layer名称列表

        Returns:
            层索引列表（去重、排序）
        """
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
            for name, module in self.model.named_modules():
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
            # 提取层索引，支持多种命名格式
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
        设置选中的层列表，后续匹配时会基于此列表过滤

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

    def filter_by_selected(self, layer_names: List[str]) -> List[str]:
        """
        根据选中的层列表过滤

        Args:
            layer_names: 待过滤的层名称列表

        Returns:
            过滤后的层名称列表
        """
        if self._selected_layers is None:
            return layer_names
        selected_set = set(self._selected_layers)
        return [name for name in layer_names if name in selected_set]

    def print_all_layers(self, max_display: int = 30):
        """
        打印所有layer名称，方便用户编写正则表达式

        Args:
            max_display: 最多显示的layer数量
        """
        all_layers = self._get_all_layer_names()
        print(f"\nTotal layers found: {len(all_layers)}")
        print(f"Showing first {min(max_display, len(all_layers))} layers:")
        for name in all_layers[:max_display]:
            print(f"  {name}")
        if len(all_layers) > max_display:
            print(f"  ... and {len(all_layers) - max_display} more")


class LayerSelector:
    """
    Layer选择器，从配置文件应用正则表达式选择layer

    只支持通过正则表达式patterns来选择layer，简单统一。
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
            layer_selection: 配置字典，格式为：
                {
                    'patterns': [
                        'layers\\.\\d+\\.mlp\\.down_proj',
                        'layers\\.\\d+\\.self_attn\\.q_proj'
                    ]
                }

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
        根据配置选择layer（只支持patterns方式）

        Args:
            layer_selection: 配置字典，格式为：
                {
                    'patterns': [
                        'layers\\.\\d+\\.mlp\\.down_proj',
                        'layers\\.\\d+\\.self_attn\\.q_proj'
                    ]
                }

        Returns:
            选择的layer名称列表
        """
        selected = set()

        # 只支持通过正则表达式
        if "patterns" in layer_selection:
            patterns = layer_selection["patterns"]
            matched = self.matcher.match_by_patterns(patterns)
            selected.update(matched)
            print(
                f"LayerSelector: Matched {len(matched)} layers "
                f"from {len(patterns)} patterns"
            )
        else:
            print("Warning: No 'patterns' key found in layer_selection config")

        return sorted(selected)
