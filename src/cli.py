"""
CLI - 命令行入口

提供命令行接口，支持配置文件和参数覆盖
"""

import argparse
import os
import sys

import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer_matcher import LayerMatcher, LayerSelector
from core.memory_manager import AutoMemoryManager
from utils.config import ConfigLoader


def parse_args():
    """
    解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Quant Smooth Ana - Model Outlier Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用配置文件
  quant-ana --config configs/llama.yaml
  
  # 覆盖配置参数
  quant-ana --config configs/llama.yaml --data.num_samples 128 --outlier.threshold 100
  
  # 使用自定义数据集
  quant-ana --config configs/llama.yaml --data.source custom:/path/to/data.txt
        """,
    )

    # 必需参数
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration file (YAML format)",
    )

    # 可选参数（覆盖配置文件）
    parser.add_argument("--model.path", type=str, help="Model path (overrides config)")

    parser.add_argument(
        "--data.source", type=str, help="Data source (overrides config)"
    )

    parser.add_argument(
        "--data.num_samples", type=int, help="Number of samples (overrides config)"
    )

    parser.add_argument(
        "--data.seq_len", type=int, help="Sequence length (overrides config)"
    )

    parser.add_argument(
        "--outlier.threshold", type=float, help="Outlier threshold (overrides config)"
    )

    parser.add_argument(
        "--visualization.save_dir", type=str, help="Output directory (overrides config)"
    )

    parser.add_argument(
        "--memory.auto_offload",
        type=bool,
        help="Enable auto offload (overrides config)",
    )

    # 其他参数
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run (load config but don't execute)"
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

    model_path = config.get("model", {}).get("path")

    if not model_path:
        raise ValueError("Model path not specified in config")

    print(f"Loading model from {model_path}...")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",  # 改为 'cuda:0'
    )

    # 打印模型配置
    print(
        f"Model config: max_position_embeddings={model.config.max_position_embeddings}"
    )
    print(f"Model vocab size: {model.config.vocab_size}")

    # 打印模型设备映射
    print(f"Model device map: {model.hf_device_map}")

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Model loaded successfully: {config.get('model', {}).get('name', 'unknown')}"
    )

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

    data_config = config.get("data", {})
    source = data_config.get("source", "pile")

    print(f"Loading data from {source}...")

    # 解析数据源
    if source.startswith("custom:"):
        # 自定义数据集
        file_path = source[8:]  # 移除 'custom:' 前缀
        format = data_config.get("custom", {}).get("format", "text")
        text_column = data_config.get("custom", {}).get("text_column", "text")

        dataloader = CustomDataLoader(
            file_path=file_path,
            tokenizer=tokenizer,
            num_samples=data_config.get("num_samples", 64),
            seq_len=data_config.get("seq_len", 1024),
            seed=data_config.get("seed", 0),
            format=format,
            text_column=text_column,
        )
    elif source.startswith("hf:"):
        # HuggingFace数据集
        dataset_name = source[3:]  # 移除 'hf:' 前缀
        text_column = data_config.get("custom", {}).get("text_column", "text")

        dataloader = get_hf_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            num_samples=data_config.get("num_samples", 64),
            seq_len=data_config.get("seq_len", 1024),
            seed=data_config.get("seed", 0),
            text_column=text_column,
        )
    else:
        # 内置数据集
        dataloader = get_builtin_dataloader(
            dataset_name=source,
            tokenizer=tokenizer,
            num_samples=data_config.get("num_samples", 64),
            seq_len=data_config.get("seq_len", 1024),
            seed=data_config.get("seed", 0),
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
    # 直接创建layer匹配器（不再需要适配器）
    layer_matcher = LayerMatcher(model)

    # 调试：打印所有layer名称
    layer_matcher.print_all_layers(max_display=20)

    # 应用layer_selection配置
    viz_config = config.get("visualization", {})
    layer_selection_config = viz_config.get("layer_selection", {})
    if layer_selection_config and "patterns" in layer_selection_config:
        layer_selector = LayerSelector(layer_matcher)
        selected_layers = layer_selector.apply_config(layer_selection_config)
        print(
            f"\nApplied layer_selection config, selected {len(selected_layers)} layers:"
        )
        for name in selected_layers[:10]:
            print(f"  {name}")
        if len(selected_layers) > 10:
            print(f"  ... and {len(selected_layers) - 10} more")
    else:
        print("\nNo patterns found in layer_selection config, using all layers")

    # 创建显存管理器
    memory_config = config.get("memory", {})
    memory_manager = AutoMemoryManager(
        model=model,
        device=memory_config.get("device", "cuda"),
        offload_device=memory_config.get("offload_device", "cpu"),
    )

    # 运行统计计算
    from statistics.magnitude import MagnitudeCalculator
    from statistics.outlier import OutlierCalculator

    magnitude_calc = MagnitudeCalculator(model, layer_matcher, memory_manager)
    outlier_calc = OutlierCalculator(model, layer_matcher, memory_manager)

    # 计算magnitude统计
    magnitude_config = viz_config.get("magnitude", {})

    # 存储所有统计数据用于可视化
    magnitude_stats = {}
    weight_stats = {}
    distribution_stats = {}
    heatmap_stats = {}

    # 获取配置的patterns
    patterns = magnitude_config.get("patterns", [r"mlp\.down_proj"])

    if viz_config.get("enabled", {}).get("magnitude_input", False):
        print("Computing magnitude statistics (input)...")
        for pattern in patterns:
            try:
                stats = magnitude_calc.calculate(
                    dataloader,
                    pattern=pattern,
                    reduce_dim=magnitude_config.get("reduce_dim"),
                    is_input=True,
                )
                # 使用简化的名称作为key
                pattern_name = pattern.replace("\\", "").replace(".", "_")
                print(f"  {pattern}: {stats.shape}")
                magnitude_stats[f"{pattern_name}_input"] = {
                    "stats": stats,
                    "pattern": pattern,
                }
            except ValueError as e:
                print(f"  Warning: Skipping pattern {pattern} (input): {e}")

    if viz_config.get("enabled", {}).get("magnitude_output", False):
        print("Computing magnitude statistics (output)...")
        for pattern in patterns:
            try:
                stats = magnitude_calc.calculate(
                    dataloader,
                    pattern=pattern,
                    reduce_dim=magnitude_config.get("reduce_dim"),
                    is_input=False,
                )
                pattern_name = pattern.replace("\\", "").replace(".", "_")
                print(f"  {pattern}: {stats.shape}")
                magnitude_stats[f"{pattern_name}_output"] = {
                    "stats": stats,
                    "pattern": pattern,
                }
            except ValueError as e:
                print(f"  Warning: Skipping pattern {pattern} (output): {e}")

    # 计算权重magnitude统计
    if viz_config.get("enabled", {}).get("magnitude_weight", False):
        print("Computing weight magnitude statistics...")
        weight_patterns = magnitude_config.get("weight", {}).get("patterns", patterns)
        for pattern in weight_patterns:
            try:
                stats = magnitude_calc.calculate_weight(pattern=pattern)
                pattern_name = pattern.replace("\\", "").replace(".", "_")
                print(f"  {pattern}: {stats.shape}")
                weight_stats[pattern_name] = {"stats": stats, "pattern": pattern}
            except Exception as e:
                print(f"  Warning: Could not calculate weight stats for {pattern}: {e}")

    # 计算outlier统计
    outlier_config = viz_config.get("outlier", {})
    outlier_stats = {}
    outlier_pattern = outlier_config.get("pattern", patterns)

    if viz_config.get("enabled", {}).get("outlier_layer_wise", False):
        print("Computing outlier statistics...")
        stats = outlier_calc.calculate_layer_wise_count(
            dataloader,
            pattern=outlier_pattern,
            outlier_threshold=outlier_config.get("threshold", 64),
        )
        print(f"  Layer-wise count: {stats.shape}")
        outlier_stats["layer_wise_count"] = stats

    if viz_config.get("enabled", {}).get("outlier_position", False):
        print("Computing outlier token positions...")
        positions = outlier_calc.calculate_token_position(
            dataloader,
            pattern=outlier_pattern,
            outlier_threshold=outlier_config.get("threshold", 20),
        )
        print(f"  Token positions: {len(positions)} tokens")
        outlier_stats["token_position"] = positions

    if viz_config.get("enabled", {}).get("outlier_token", False):
        print("Computing outlier token content...")
        token_content = outlier_calc.calculate_token_content(
            dataloader,
            pattern=outlier_pattern,
            outlier_threshold=outlier_config.get("threshold", 20),
            tokenizer=tokenizer,
        )
        print(
            f"  Token content: {len(token_content.get('decoded_tokens', {}))} unique tokens"
        )
        outlier_stats["token_content"] = token_content

    # 计算distribution统计
    if viz_config.get("enabled", {}).get("distribution", False):
        print("Computing distribution statistics...")
        distribution_config = viz_config.get("distribution", {})
        distribution_patterns = distribution_config.get("patterns", patterns)
        for pattern in distribution_patterns:
            try:
                stats = magnitude_calc.calculate_distribution(
                    dataloader,
                    pattern=pattern,
                    is_input=distribution_config.get("is_input", True),
                )
                pattern_name = pattern.replace("\\", "").replace(".", "_")
                print(f"  {pattern}: {len(stats['values'])} layers")
                distribution_stats[pattern_name] = {"stats": stats, "pattern": pattern}
            except Exception as e:
                print(f"  Warning: Could not calculate distribution for {pattern}: {e}")

    # 计算heatmap_3d统计
    if viz_config.get("enabled", {}).get("heatmap_3d", False):
        print("Computing heatmap data...")
        heatmap_config = viz_config.get("heatmap_3d", {})
        heatmap_pattern = heatmap_config.get("pattern", patterns[0])
        heatmap_data = magnitude_calc.calculate_heatmap_data(
            dataloader,
            pattern=heatmap_pattern,
            sample_idx=heatmap_config.get("sample_idx", 0),
            is_input=heatmap_config.get("is_input", False),
        )
        print(f"  Heatmap shape: {heatmap_data.shape}")
        heatmap_stats["data"] = heatmap_data
        heatmap_stats["pattern"] = heatmap_pattern

    # 计算percentile_range统计（用于hidden dimension百分位数范围可视化）
    percentile_stats = {}
    if viz_config.get("enabled", {}).get("percentile_range", False):
        print("Computing percentile range statistics...")
        percentile_config = viz_config.get("percentile_range", {})
        percentile_patterns = percentile_config.get("patterns", patterns)
        is_input = percentile_config.get("is_input", True)
        use_abs = percentile_config.get("use_abs", False)
        
        for pattern in percentile_patterns:
            try:
                stats = magnitude_calc.calculate_percentile_range(
                    dataloader,
                    pattern=pattern,
                    is_input=is_input,
                    use_abs=use_abs,
                )
                pattern_name = pattern.replace("\\", "").replace(".", "_")
                plot_type = "input" if is_input else "output"
                print(f"  {pattern}: {len(stats)} layers")
                percentile_stats[pattern_name] = {
                    "stats": stats,
                    "pattern": pattern,
                    "is_input": is_input,
                }
            except Exception as e:
                print(f"  Warning: Could not calculate percentile range for {pattern}: {e}")

    # 生成可视化
    from visualization.magnitude_plot import MagnitudeVisualizer
    from visualization.outlier_plot import OutlierVisualizer
    from visualization.report_generator import ReportGenerator

    print("Generating visualizations...")

    magnitude_viz = MagnitudeVisualizer(viz_config)
    outlier_viz = OutlierVisualizer(viz_config)

    # 生成magnitude可视化
    magnitude_figures = []
    for key, data in magnitude_stats.items():
        stats = data["stats"]
        pattern = data["pattern"]
        plot_type = "input" if "input" in key else "output"
        layer_names = layer_matcher.match_layers(pattern)
        layer_names = layer_matcher.filter_by_selected(layer_names)
        figure = magnitude_viz.visualize(
            stats_data=stats,
            layer_names=layer_names,
            plot_type=plot_type,
            component_type=key,  # 使用key作为标识
        )
        if figure:
            magnitude_figures.append(figure)
            print(f"  Generated magnitude plot: {figure}")

    # 生成权重magnitude可视化
    weight_figures = []
    for key, data in weight_stats.items():
        stats = data["stats"]
        pattern = data["pattern"]
        layer_names = layer_matcher.match_layers(pattern)
        layer_names = layer_matcher.filter_by_selected(layer_names)
        figure = magnitude_viz.visualize_weight(
            stats_data=stats, layer_names=layer_names, component_type=key
        )
        if figure:
            weight_figures.append(figure)
            print(f"  Generated weight magnitude plot: {figure}")

    # 生成distribution可视化
    distribution_figures = []
    if distribution_stats:
        from visualization.distribution_plot import DistributionVisualizer

        distribution_viz = DistributionVisualizer(viz_config)

        distribution_config = viz_config.get("distribution", {})
        plot_types = distribution_config.get("plot_types", ["boxplot"])

        for key, data in distribution_stats.items():
            stats = data["stats"]
            pattern = data["pattern"]
            layer_names = layer_matcher.match_layers(pattern)
            layer_names = layer_matcher.filter_by_selected(layer_names)

            if "boxplot" in plot_types:
                figure = distribution_viz.visualize_boxplot(
                    stats=stats, layer_names=layer_names, component_type=key
                )
                if figure:
                    distribution_figures.append(figure)
                    print(f"  Generated distribution boxplot: {figure}")

            if "histogram" in plot_types:
                figure = distribution_viz.visualize_histogram(
                    stats=stats,
                    layer_names=layer_names,
                    component_type=key,
                    num_layers_to_show=distribution_config.get("num_layers_to_show", 4),
                )
                if figure:
                    distribution_figures.append(figure)
                    print(f"  Generated distribution histogram: {figure}")

    # 生成heatmap_3d可视化
    heatmap_figures = []
    if heatmap_stats:
        from visualization.heatmap_3d import Heatmap3DVisualizer

        heatmap_viz = Heatmap3DVisualizer(viz_config)

        heatmap_config = viz_config.get("heatmap_3d", {})
        heatmap_pattern = heatmap_stats.get("pattern", patterns)
        layer_names = layer_matcher.match_layers(heatmap_pattern)
        layer_names = layer_matcher.filter_by_selected(layer_names)

        # 3D热力图
        figure = heatmap_viz.visualize_3d_heatmap(
            data=heatmap_stats["data"],
            layer_names=layer_names,
            component_type="heatmap",
            view_angle=tuple(heatmap_config.get("view_angle", [30, 45])),
        )
        if figure:
            heatmap_figures.append(figure)
            print(f"  Generated 3D heatmap: {figure}")

        # 2D热力图（显示前几层）
        num_2d_layers = heatmap_config.get("num_2d_layers", 2)
        for layer_idx in range(min(num_2d_layers, heatmap_stats["data"].shape[0])):
            figure = heatmap_viz.visualize_2d_heatmap(
                data=heatmap_stats["data"],
                layer_names=layer_names,
                component_type="heatmap",
                layer_idx=layer_idx,
            )
            if figure:
                heatmap_figures.append(figure)
                print(f"  Generated 2D heatmap layer {layer_idx}: {figure}")

    # 生成percentile_range可视化
    percentile_figures = []
    if percentile_stats:
        from visualization.percentile_plot import PercentileVisualizer

        percentile_viz = PercentileVisualizer(viz_config)
        percentile_config = viz_config.get("percentile_range", {})
        plot_style = percentile_config.get("style", "fill")  # "fill" 或 "line"
        num_layers_to_show = percentile_config.get("num_layers_to_show", 4)

        for key, data in percentile_stats.items():
            stats = data["stats"]  # {layer_name: percentile_data}
            pattern = data["pattern"]
            is_input = data["is_input"]
            plot_type = "input" if is_input else "output"

            layer_names_list = list(stats.keys())
            percentile_data_list = [stats[name] for name in layer_names_list]

            # 限制显示的层数
            if len(layer_names_list) > num_layers_to_show:
                # 均匀采样
                indices = list(range(0, len(layer_names_list), max(1, len(layer_names_list) // num_layers_to_show)))
                indices = indices[:num_layers_to_show]
                layer_names_list = [layer_names_list[i] for i in indices]
                percentile_data_list = [percentile_data_list[i] for i in indices]

            # 生成多层对比图
            if len(layer_names_list) > 1:
                figure = percentile_viz.visualize_multi_layer_percentile(
                    all_percentile_data=percentile_data_list,
                    layer_names=layer_names_list,
                    component_type=key,
                    plot_type=plot_type,
                    style=plot_style,
                )
                if figure:
                    percentile_figures.append(figure)
                    print(f"  Generated percentile multi-layer plot: {figure}")

            # 为每层生成单独的图
            for layer_name, percentile_data in zip(layer_names_list, percentile_data_list):
                if plot_style == "line":
                    figure = percentile_viz.visualize_percentile_line(
                        percentile_data=percentile_data,
                        layer_name=layer_name,
                        component_type=key,
                        plot_type=plot_type,
                    )
                else:
                    figure = percentile_viz.visualize_percentile_range(
                        percentile_data=percentile_data,
                        layer_name=layer_name,
                        component_type=key,
                        plot_type=plot_type,
                    )
                if figure:
                    percentile_figures.append(figure)
                    print(f"  Generated percentile plot for {layer_name}: {figure}")

    # 生成outlier可视化
    outlier_figures = []
    if "layer_wise_count" in outlier_stats:
        outlier_layer_names = layer_matcher.match_layers(outlier_pattern)
        outlier_layer_names = layer_matcher.filter_by_selected(outlier_layer_names)
        figure = outlier_viz.visualize_layer_wise_count(
            stats=outlier_stats["layer_wise_count"],
            layer_names=outlier_layer_names,
        )
        if figure:
            outlier_figures.append(figure)
            print(f"  Generated outlier layer-wise plot: {figure}")

    if "token_position" in outlier_stats:
        figure = outlier_viz.visualize_token_position(
            position_stats=outlier_stats["token_position"]
        )
        if figure:
            outlier_figures.append(figure)
            print(f"  Generated outlier position plot: {figure}")

    if "token_content" in outlier_stats:
        figure = outlier_viz.visualize_token_content(
            token_content=outlier_stats["token_content"],
            top_n=outlier_config.get("top_n", 10),
        )
        if figure:
            outlier_figures.append(figure)
            print(f"  Generated outlier token plot: {figure}")

    # 生成报告
    report_gen = ReportGenerator(viz_config)

    # 添加摘要
    summary = f"""
    <h2>Analysis Summary</h2>
    <p>Outlier analysis completed successfully.</p>
    <ul>
        <li>Model: {config.get("model", {}).get("name", "unknown")}</li>
        <li>Dataset: {config.get("data", {}).get("source", "unknown")}</li>
        <li>Number of samples: {config.get("data", {}).get("num_samples", 0)}</li>
        <li>Sequence length: {config.get("data", {}).get("seq_len", 0)}</li>
    </ul>
    """
    report_gen.add_section("Analysis Summary", summary)

    # 添加magnitude图表
    if magnitude_figures:
        magnitude_html = "<h2>Magnitude Analysis</h2>"
        for fig in magnitude_figures:
            fig_name = fig.split("/")[-1]
            magnitude_html += (
                f'<div style="margin: 20px 0; text-align: center;">'
                f'<img src="{fig_name}" style="max-width:100%;">'
                f'<p style="color: #666; font-size: 14px; margin-top: 5px;">{fig_name}</p>'
                f'</div>'
            )
        report_gen.add_section("Magnitude Analysis", magnitude_html)

    # 添加权重magnitude图表
    if weight_figures:
        weight_html = "<h2>Weight Magnitude Analysis</h2>"
        for fig in weight_figures:
            fig_name = fig.split("/")[-1]
            weight_html += (
                f'<div style="margin: 20px 0; text-align: center;">'
                f'<img src="{fig_name}" style="max-width:100%;">'
                f'<p style="color: #666; font-size: 14px; margin-top: 5px;">{fig_name}</p>'
                f'</div>'
            )
        report_gen.add_section("Weight Magnitude Analysis", weight_html)

    # 添加distribution图表
    if distribution_figures:
        distribution_html = "<h2>Distribution Analysis</h2>"
        for fig in distribution_figures:
            fig_name = fig.split("/")[-1]
            distribution_html += (
                f'<div style="margin: 20px 0; text-align: center;">'
                f'<img src="{fig_name}" style="max-width:100%;">'
                f'<p style="color: #666; font-size: 14px; margin-top: 5px;">{fig_name}</p>'
                f'</div>'
            )
        report_gen.add_section("Distribution Analysis", distribution_html)

    # 添加heatmap图表
    if heatmap_figures:
        heatmap_html = "<h2>Heatmap Analysis</h2>"
        for fig in heatmap_figures:
            fig_name = fig.split("/")[-1]
            heatmap_html += (
                f'<div style="margin: 20px 0; text-align: center;">'
                f'<img src="{fig_name}" style="max-width:100%;">'
                f'<p style="color: #666; font-size: 14px; margin-top: 5px;">{fig_name}</p>'
                f'</div>'
            )
        report_gen.add_section("Heatmap Analysis", heatmap_html)

    # 添加percentile_range图表
    if percentile_figures:
        percentile_html = "<h2>Percentile Range Analysis</h2>"
        percentile_html += "<p>Shows the distribution of activation values across hidden dimensions with different percentile ranges (25/75, 1/99, 0.01/99.99, Min/Max).</p>"
        for fig in percentile_figures:
            fig_name = fig.split("/")[-1]
            percentile_html += (
                f'<div style="margin: 20px 0; text-align: center;">'
                f'<img src="{fig_name}" style="max-width:100%;">'
                f'<p style="color: #666; font-size: 14px; margin-top: 5px;">{fig_name}</p>'
                f'</div>'
            )
        report_gen.add_section("Percentile Range Analysis", percentile_html)

    # 添加outlier图表
    if outlier_figures:
        outlier_html = "<h2>Outlier Analysis</h2>"
        for fig in outlier_figures:
            fig_name = fig.split("/")[-1]
            outlier_html += (
                f'<div style="margin: 20px 0; text-align: center;">'
                f'<img src="{fig_name}" style="max-width:100%;">'
                f'<p style="color: #666; font-size: 14px; margin-top: 5px;">{fig_name}</p>'
                f'</div>'
            )
        report_gen.add_section("Outlier Analysis", outlier_html)

    # 保存报告
    report_path = report_gen.generate()
    print(f"Report generated: {report_path}")

    # 导出数据
    output_config = config.get("visualization", {}).get("output", {})
    if "csv" in output_config.get("formats", []):
        from utils.export import DataExporter

        _exporter = DataExporter(viz_config.get("save_dir", "./figures"))  # noqa: F841
        print("Exporting data...")
        # TODO: 导出统计数据

    print("Analysis completed successfully!")


@torch.no_grad()
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
        if value is not None and key not in ["config", "verbose", "dry_run"]:
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


if __name__ == "__main__":
    main()
