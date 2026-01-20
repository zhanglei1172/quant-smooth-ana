"""
Base Model Adapter - 模型适配器基类

定义模型适配器的标准接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModelAdapter(ABC):
    """
    模型适配器基类
    
    所有模型适配器都应该继承这个基类，并实现所有抽象方法
    """
    
    def __init__(self, model):
        """
        初始化模型适配器
        
        Args:
            model: PyTorch模型实例
        """
        self.model = model
    
    @abstractmethod
    def get_layers(self) -> List:
        """
        获取模型的所有transformer层
        
        Returns:
            transformer层列表
        """
        pass
    
    @abstractmethod
    def get_embeddings(self) -> List:
        """
        获取embedding层
        
        Returns:
            embedding层列表
        """
        pass
    
    @abstractmethod
    def get_layer_name_pattern(self, component_type: str) -> Optional[str]:
        """
        获取组件的layer名称模式（正则表达式）
        
        Args:
            component_type: 组件类型，如 'q_proj', 'down_proj', 'hidden_state'
            
        Returns:
            正则表达式字符串，如果组件类型不支持则返回None
        """
        pass
    
    @abstractmethod
    def get_full_layer_name(self, layer_idx: int, component_type: str) -> str:
        """
        构建完整的layer名称
        
        Args:
            layer_idx: 层索引
            component_type: 组件类型
            
        Returns:
            完整的layer名称，如 'model.layers.0.mlp.down_proj'
        """
        pass
    
    def get_supported_components(self) -> List[str]:
        """
        返回支持的组件类型列表
        
        Returns:
            组件类型列表
        """
        return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                'gate_proj', 'up_proj', 'down_proj', 'hidden_state']
    
    def get_model_family(self) -> str:
        """
        获取模型家族名称
        
        Returns:
            模型家族名称
        """
        return self.__class__.__name__.replace('Adapter', '').lower()
    
    def get_num_layers(self) -> int:
        """
        获取模型层数
        
        Returns:
            transformer层的数量
        """
        return len(self.get_layers())