"""
CLI - 命令行入口

提供命令行接口，支持配置文件和参数覆盖
"""

import argparse
import os
import sys
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer_matcher import LayerMatcher
from core.memory_manager import AutoMemoryManager
from core.registry import ModelRegistry
from models.llama import LlamaAdapter
from models.qwen import QwenAdapter
from utils.config import ConfigLoader


def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='Quant Smooth Ana - Model Outlier Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用配置文件
  quant-ana --config configs/llama.yaml
  
  # 覆盖配置参数
  quant-ana --config configs/llama.yaml --data.num_samples 128 --outlier.threshold 100
  
  # 使用自定义数据集
  quant-ana --config configs/llama.yaml --data.source custom:/path/to/data.txt
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration file (YAML format)'
    )
    
    # 可选参数（覆盖配置文件）
    parser.add_argument(
        '--model.path',
        type=str,
        help='Model path (overrides config)'
    )
    
    parser.add_argument(
        '--data.source',
        type=str,
        help='Data source (overrides config)'
    )
    
    parser.add_argument(
        '--data.num_samples',
        type=int,
        help='Number of samples (overrides config)'
    )
    
    parser.add_argument(
        '--data.seq_len',
        type=int,
        help='Sequence length (overrides config)'
    )
    
    parser.add_argument(
        '--outlier.threshold',
        type=float,
        help='Outlier threshold (overrides config)'
    )
    
    parser.add_argument(
        '--visualization.save_dir',
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--memory.auto_offload',
        type=bool,
        help='Enable auto offload (overrides config)'
    )
    
    # 其他参数
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run (load config but don\'t execute)'
    )
    
    return parser.parse_args()


def load_model(config: dict):
    """
    加载模型
    
    Args:
        config: 配置字典
        
    Returns:
        模型实例
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = config.get('model', {}).get('path')
    
    if not model_path:
        raise ValueError("Model path not specified in config")
    
    print(f"Loading model from {model_path}...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully: {config.get('model', {}).get('name', 'unknown')}")
    
    return model, tokenizer


def load_data(config: dict, tokenizer):
    """
    加载数据
    
    Args:
        config: 配置字典
        tokenizer: 分词器
        
    Returns:
        数据加载器
    """
    from data.builtin_datasets import get_builtin_dataloader
    from data.custom_datasets import CustomDataLoader
    from data.hf_datasets import get_hf_dataloader
    
    data_config = config.get('data', {})
    source = data_config.get('source', 'pile')
    
    print(f"Loading data from {source}...")
    
    # 解析数据源
    if source.startswith('custom:'):
        # 自定义数据集
        file_path = source[8:]  # 移除 'custom:' 前缀
        format = data_config.get('custom', {}).get('format', 'text')
        text_column = data_config.get('custom', {}).get('text_column', 'text')
        
        dataloader = CustomDataLoader(
            file_path=file_path,
            tokenizer=tokenizer,
            num_samples=data_config.get('num_samples', 64),
            seq_len=data_config.get('seq_len', 1024),
            seed=data_config.get('seed', 0),
            format=format,
            text_column=text_column
        )
    elif source.startswith('hf:'):
        # HuggingFace数据集
        dataset_name = source[3:]  # 移除 'hf:' 前缀
        text_column = data_config.get('custom', {}).get('text_column', 'text')
        
        dataloader = get_hf_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            num_samples=data_config.get('num_samples', 64),
            seq_len=data_config.get('seq_len', 1024),
            seed=data_config.get('seed', 0),
            text_column=text_column
        )
    else:
        # 内置数据集
        dataloader = get_builtin_dataloader(
            dataset_name=source,
            tokenizer=tokenizer,
            num_samples=data_config.get('num_samples', 64),
            seq_len=data_config.get('seq_len', 1024),
            seed=data_config.get('seed', 0)
        )
    
    print(f"Data loaded successfully: {len(dataloader)} samples")
    
    return dataloader


def run_analysis(config: dict, model, tokenizer, dataloader):
    """
    运行分析
    
    Args:
        config: 配置字典
        model: 模型实例
        tokenizer: 分词器
        dataloader: 数据加载器
    """
    # 获取模型适配器
    adapter = ModelRegistry.get_adapter(model)
    
    # 创建layer匹配器
    layer_matcher = LayerMatcher(adapter)
    
    # 创建显存管理器
    memory_config = config.get('memory', {})
    memory_manager = AutoMemoryManager(
        model=model,
        device=memory_config.get('device', 'cuda'),
        offload_device=memory_config.get('offload_device', 'cpu')
    )
    
    # 运行统计计算
    from statistics.magnitude import MagnitudeCalculator
    from statistics.outlier import OutlierCalculator
    
    magnitude_calc = MagnitudeCalculator(adapter, layer_matcher, memory_manager)
    outlier_calc = OutlierCalculator(adapter, layer_matcher, memory_manager)
    
    # 计算magnitude统计
    viz_config = config.get('visualization', {})
    magnitude_config = viz_config.get('magnitude', {})
    
    if viz_config.get('enabled', {}).get('magnitude_input', False):
        print("Computing magnitude statistics (input)...")
        for component in magnitude_config.get('components', ['down_proj']):
            stats = magnitude_calc.calculate(
                dataloader,
                component_type=component,
                per_tensor=magnitude_config.get('per_tensor', False),
                is_input=True
            )
            print(f"  {component}: {stats.shape}")
    
    if viz_config.get('enabled', {}).get('magnitude_output', False):
        print("Computing magnitude statistics (output)...")
        for component in magnitude_config.get('components', ['down_proj']):
            stats = magnitude_calc.calculate(
                dataloader,
                component_type=component,
                per_tensor=magnitude_config.get('per_tensor', False),
                is_input=False
            )
            print(f"  {component}: {stats.shape}")
    
    # 计算outlier统计
    outlier_config = viz_config.get('outlier', {})
    
    if viz_config.get('enabled', {}).get('outlier_layer_wise', False):
        print("Computing outlier statistics...")
        stats = outlier_calc.calculate_layer_wise_count(
            dataloader,
            component_type=outlier_config.get('component_type', 'hidden_state'),
            outlier_threshold=outlier_config.get('threshold', 64)
        )
        print(f"  Layer-wise count: {stats.shape}")
    
    # 生成可视化
    from visualization.magnitude_plot import MagnitudeVisualizer
    from visualization.outlier_plot import OutlierVisualizer
    from visualization.report_generator import ReportGenerator
    
    print("Generating visualizations...")
    
    magnitude_viz = MagnitudeVisualizer(config)
    outlier_viz = OutlierVisualizer(config)
    
    # 生成报告
    report_gen = ReportGenerator(config)
    report_gen.add_section("Analysis Summary", "<p>Outlier analysis completed successfully.</p>")
    
    # 保存报告
    report_path = report_gen.generate()
    print(f"Report generated: {report_path}")
    
    # 导出数据
    output_config = config.get('visualization', {}).get('output', {})
    if 'csv' in output_config.get('formats', []):
        from utils.export import DataExporter
        exporter = DataExporter(viz_config.get('save_dir', './figures'))
        print("Exporting data...")
        # TODO: 导出统计数据
    
    print("Analysis completed successfully!")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    print(f"Loading configuration from {args.config}...")
    config_loader = ConfigLoader(args.config)
    config = config_loader.to_dict()
    
    # 解析命令行参数覆盖
    overrides = []
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'verbose', 'dry_run']:
            overrides.append(f"{key}={value}")
    
    if overrides:
        print(f"Applying overrides: {overrides}")
        updates = ConfigLoader.parse_overrides(overrides)
        config_loader.update(updates)
        config = config_loader.to_dict()
    
    if args.verbose:
        print("Configuration:")
        import yaml
        print(yaml.dump(config, default_flow_style=False))
    
    if args.dry_run:
        print("Dry run mode - skipping execution")
        return
    
    # 加载模型
    model, tokenizer = load_model(config)
    
    # 加载数据
    dataloader = load_data(config, tokenizer)
    
    # 运行分析
    run_analysis(config, model, tokenizer, dataloader)


if __name__ == '__main__':
    main()
