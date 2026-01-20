"""
Memory Manager - 自动显存管理器

这个模块实现了自动显存管理功能，支持动态onload/offload，在计算时将层加载到GPU，
计算完成后自动卸载到CPU，以优化显存使用。
"""

import torch
from contextlib import contextmanager
from typing import Optional, Dict, List
import warnings


class AutoMemoryManager:
    """
    自动显存管理器，实现动态onload/offload
    
    提供两种使用方式：
    1. 上下文管理器（推荐）：自动管理onload/offload
    2. 手动管理：手动调用load_layer_for_computation和offload_layer
    """
    
    def __init__(self, model, device: str = 'cuda', offload_device: str = 'cpu'):
        """
        初始化显存管理器
        
        Args:
            model: PyTorch模型
            device: 计算设备（通常是'cuda'）
            offload_device: 卸载设备（通常是'cpu'）
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.offload_device = offload_device
        
        # 存储原始设备信息
        self._original_device_map: Dict[str, torch.device] = {}
        
        # 存储当前加载的层
        self._current_layer: Optional[str] = None
        
        # 缓存层对象
        self._layer_cache: Dict[str, torch.nn.Module] = {}
        
        # 统计信息
        self._load_count = 0
        self._offload_count = 0
    
    @contextmanager
    def load_layer(self, layer_name: str, keep_cached: bool = False):
        """
        上下文管理器，自动onload指定层到GPU，结束后offload回CPU
        
        Args:
            layer_name: 层名称，如 'model.layers.0.mlp.down_proj'
            keep_cached: 是否保持缓存（不自动offload）
            
        Yields:
            加载到GPU的层模块
            
        Example:
            with memory_manager.load_layer('model.layers.0.mlp.down_proj'):
                output = layer(input)
            # 退出上下文后，layer自动offload到CPU
        """
        # 保存原始设备
        layer = self.model.get_submodule(layer_name)
        original_device = self._get_layer_device(layer)
        
        # onload到目标设备
        layer.to(self.device)
        self._load_count += 1
        
        try:
            yield layer
        finally:
            # offload回原始设备
            if not keep_cached:
                layer.to(original_device)
                self._offload_count += 1
                self._clear_cache()
                torch.cuda.empty_cache()
    
    def load_layer_for_computation(self, layer_name: str):
        """
        非上下文版本，手动管理onload/offload
        
        Args:
            layer_name: 层名称
            
        Example:
            memory_manager.load_layer_for_computation('model.layers.0.mlp.down_proj')
            output = layer(input)
            memory_manager.offload_layer('model.layers.0.mlp.down_proj')
        """
        layer = self.model.get_submodule(layer_name)
        original_device = self._get_layer_device(layer)
        
        self._original_device_map[layer_name] = original_device
        layer.to(self.device)
        self._current_layer = layer_name
        self._load_count += 1
    
    def offload_layer(self, layer_name: str):
        """
        手动offload指定层
        
        Args:
            layer_name: 层名称
        """
        if layer_name in self._original_device_map:
            layer = self.model.get_submodule(layer_name)
            original_device = self._original_device_map[layer_name]
            layer.to(original_device)
            del self._original_device_map[layer_name]
            self._offload_count += 1
            torch.cuda.empty_cache()
    
    def offload_all_layers(self):
        """
        offload所有层到CPU
        """
        for name, module in self.model.named_modules():
            if hasattr(module, 'parameters'):
                params = list(module.parameters())
                if params and params[0].device != self.offload_device:
                    module.to(self.offload_device)
        
        self._original_device_map.clear()
        self._current_layer = None
        torch.cuda.empty_cache()
        self._offload_count += 1
    
    def load_layers_batch(self, layer_names: List[str]) -> Dict[str, torch.nn.Module]:
        """
        批量加载多个层到GPU
        
        Args:
            layer_names: 层名称列表
            
        Returns:
            层名称到层模块的映射
        """
        layers = {}
        for layer_name in layer_names:
            layer = self.model.get_submodule(layer_name)
            original_device = self._get_layer_device(layer)
            self._original_device_map[layer_name] = original_device
            layer.to(self.device)
            layers[layer_name] = layer
            self._load_count += 1
        
        return layers
    
    def offload_layers_batch(self, layer_names: List[str]):
        """
        批量offload多个层
        
        Args:
            layer_names: 层名称列表
        """
        for layer_name in layer_names:
            self.offload_layer(layer_name)
    
    @contextmanager
    def load_model_embeddings(self):
        """
        上下文管理器，加载embedding层到GPU
        
        适用于需要embedding层的操作
        """
        embeddings = self._get_embeddings()
        
        # 保存原始设备
        original_devices = [self._get_layer_device(emb) for emb in embeddings]
        
        # onload到GPU
        for emb in embeddings:
            emb.to(self.device)
        self._load_count += 1
        
        try:
            yield embeddings
        finally:
            # offload回原始设备
            for emb, original_device in zip(embeddings, original_devices):
                emb.to(original_device)
            self._offload_count += 1
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前显存使用情况
        
        Returns:
            显存使用信息字典：
            {
                'allocated_gb': 已分配显存（GB）,
                'reserved_gb': 已保留显存（GB）,
                'free_gb': 可用显存（GB）,
                'total_gb': 总显存（GB）
            }
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            free = total - reserved
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': free,
                'total_gb': total
            }
        return {
            'allocated_gb': 0,
            'reserved_gb': 0,
            'free_gb': 0,
            'total_gb': 0
        }
    
    def print_memory_usage(self):
        """打印显存使用情况"""
        usage = self.get_memory_usage()
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {usage['allocated_gb']:.2f} GB")
        print(f"  Reserved: {usage['reserved_gb']:.2f} GB")
        print(f"  Free: {usage['free_gb']:.2f} GB")
        print(f"  Total: {usage['total_gb']:.2f} GB")
    
    def get_layer_device(self, layer_name: str) -> torch.device:
        """
        获取指定层的设备
        
        Args:
            layer_name: 层名称
            
        Returns:
            设备对象
        """
        layer = self.model.get_submodule(layer_name)
        return self._get_layer_device(layer)
    
    def _get_layer_device(self, layer: torch.nn.Module) -> torch.device:
        """
        获取层的设备
        
        Args:
            layer: 层模块
            
        Returns:
            设备对象
        """
        params = list(layer.parameters())
        if params:
            return params[0].device
        buffers = list(layer.buffers())
        if buffers:
            return buffers[0].device
        return torch.device('cpu')
    
    def _get_embeddings(self) -> List[torch.nn.Module]:
        """
        获取模型的embedding层
        
        Returns:
            embedding层列表
        """
        embeddings = []
        for name, module in self.model.named_modules():
            if 'embed' in name.lower():
                embeddings.append(module)
        return embeddings
    
    def _clear_cache(self):
        """清空缓存"""
        self._layer_cache.clear()
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取统计信息
        
        Returns:
            统计信息字典：
            {
                'load_count': onload次数,
                'offload_count': offload次数
            }
        """
        return {
            'load_count': self._load_count,
            'offload_count': self._offload_count
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._load_count = 0
        self._offload_count = 0
    
    def check_memory_available(self, required_gb: float) -> bool:
        """
        检查是否有足够的显存
        
        Args:
            required_gb: 需要的显存（GB）
            
        Returns:
            如果有足够显存返回True，否则返回False
        """
        usage = self.get_memory_usage()
        return usage['free_gb'] >= required_gb
    
    def auto_offload_if_needed(self, threshold_gb: float = 1.0):
        """
        如果显存不足，自动offload当前层
        
        Args:
            threshold_gb: 显存阈值（GB），低于此值时触发offload
        """
        usage = self.get_memory_usage()
        if usage['free_gb'] < threshold_gb:
            if self._current_layer:
                warnings.warn(
                    f"Low memory ({usage['free_gb']:.2f} GB < {threshold_gb} GB), "
                    f"offloading layer {self._current_layer}"
                )
                self.offload_layer(self._current_layer)
                self._current_layer = None