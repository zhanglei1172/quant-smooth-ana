"""
Llama Adapter - LLaMA模型适配器

支持LLaMA、LLaMA-2、LLaMA-3等模型
"""

from typing import List, Optional
from core.registry import ModelRegistry
from models.base import BaseModelAdapter


@ModelRegistry.register("llama")
class LlamaAdapter(BaseModelAdapter):
    """
    LLaMA模型适配器
    
    支持LLaMA、LLaMA-2、LLaMA-3等模型
    """
    
    def get_layers(self) -> List:
        """
        获取模型的所有transformer层
        
        Returns:
            transformer层列表
        """
        return self.model.model.layers
    
    def get_embeddings(self) -> List:
        """
        获取embedding层
        
        Returns:
            embedding层列表
        """
        return [self.model.model.embed_tokens]
    
    def get_layer_name_pattern(self, component_type: str) -> Optional[str]:
        """
        获取组件的layer名称模式（正则表达式）
        
        Args:
            component_type: 组件类型
            
        Returns:
            正则表达式字符串
        """
        patterns = {
            'q_proj': r'self_attn\.q_proj',
            'k_proj': r'self_attn\.k_proj',
            'v_proj': r'self_attn\.v_proj',
            'o_proj': r'self_attn\.o_proj',
            'gate_proj': r'mlp\.gate_proj',
            'up_proj': r'mlp\.up_proj',
            'down_proj': r'mlp\.down_proj',
        }
        return patterns.get(component_type)
    
    def get_full_layer_name(self, layer_idx: int, component_type: str) -> str:
        """
        构建完整的layer名称
        
        Args:
            layer_idx: 层索引
            component_type: 组件类型
            
        Returns:
            完整的layer名称
        """
        pattern = self.get_layer_name_pattern(component_type)
        
        if pattern == r'$' or component_type == 'hidden_state':
            return f'model.layers.{layer_idx}'
        
        # 替换正则表达式中的转义点
        pattern_clean = pattern.replace(r'\.', '.')
        return f'model.layers.{layer_idx}.{pattern_clean}'