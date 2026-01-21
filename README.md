# Quant Smooth Ana

一个模块化、可扩展的模型激活值统计分析与可视化工具，用于分析大语言模型中的 outlier 分布、magnitude 统计等。

## 特性

- **无需模型适配器**：直接传入任意 PyTorch 模型实例，通过正则表达式匹配任意层
- **灵活的层选择**：统一使用正则表达式匹配模型层，支持任意模型架构
- **自动显存管理**：动态 GPU/CPU offload，优化显存使用
- **多种数据源**：内置数据集、自定义文件、HuggingFace 数据集
- **丰富的可视化**：Magnitude 图、Outlier 分析、3D 热力图、分布图
- **完整的报告**：自动生成包含所有可视化的 HTML 报告
- **数据导出**：支持 CSV 和 JSON 格式导出

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 创建配置文件

创建 YAML 配置文件（例如 `configs/my_model.yaml`）：

```yaml
# 模型配置
model:
  path: "/path/to/your/model"
  name: "my-model"

# 数据配置
data:
  source: "pile"  # 或 custom:/path/to/file.txt
  num_samples: 64
  seq_len: 1024
  seed: 0

# 可视化配置
visualization:
  save_dir: "./figures"
  dpi: 200
  
  # 启用的可视化
  enabled:
    magnitude_input: true
    magnitude_output: true
    magnitude_weight: false
    outlier_layer_wise: true
    outlier_position: true
    outlier_token: true
    heatmap_3d: false
    distribution: false
  
  # Magnitude 配置 - 使用正则表达式匹配层
  magnitude:
    patterns:
      - "mlp\\.down_proj"      # 匹配所有 down_proj 层
      - "mlp\\.up_proj"        # 匹配所有 up_proj 层
      - "self_attn\\.q_proj"   # 匹配所有 q_proj 层
      - "self_attn\\.o_proj"   # 匹配所有 o_proj 层
    reduce_dim: null
  
  # Outlier 配置 - 使用正则表达式匹配层
  outlier:
    threshold: 64
    pattern: "layers\\.\\d+$"  # 匹配 transformer 层输出
    decode_tokens: true
  
  # 层选择（可选）- 过滤要分析的层
  layer_selection:
    patterns:
      - "layers\\.\\d+\\.mlp\\.down_proj"
      - "layers\\.\\d+\\.self_attn\\.q_proj"

# 显存配置
memory:
  device: "cuda"
  offload_device: "cpu"
  auto_offload: true
```

### 2. 运行分析

```bash
python src/cli.py --config configs/my_model.yaml
```

### 3. 覆盖配置参数

```bash
python src/cli.py --config configs/my_model.yaml --data.num_samples 128 --outlier.threshold 100
```

## 正则表达式匹配

本工具使用正则表达式来匹配模型层，这使得它能够支持任意模型架构。

### 常用正则表达式示例

```yaml
# 匹配所有 down_proj 层
patterns:
  - "mlp\\.down_proj"

# 只匹配特定层索引（0, 5, 10, 15）
patterns:
  - "layers\\.(0|5|10|15)\\.mlp\\.down_proj"

# 匹配 0-9 层的所有 MLP 层
patterns:
  - "layers\\.[0-9]\\.mlp"

# 匹配整个 transformer 层输出
patterns:
  - "layers\\.\\d+$"

# VL 模型的语言模型部分
patterns:
  - "language_model\\.model\\.layers\\.\\d+\\.mlp"
```

### 查看模型所有层

运行分析时，工具会打印模型的前 20 个层名称，帮助你编写正则表达式：

```
Total layers found: 224
Showing first 20 layers:
  model.embed_tokens
  model.layers.0.self_attn.q_proj
  model.layers.0.self_attn.k_proj
  ...
```

## 数据源

### 内置数据集

- `pile`: Pile 数据集
- `wikitext2`: WikiText-2 数据集
- `c4`: C4 数据集
- `redpajama`: RedPajama 数据集

### 自定义数据集

```bash
# 文本文件
python src/cli.py --config config.yaml --data.source custom:/path/to/data.txt

# JSON 文件
python src/cli.py --config config.yaml --data.source custom:/path/to/data.json \
    --data.custom.format json --data.custom.text_column text

# CSV 文件
python src/cli.py --config config.yaml --data.source custom:/path/to/data.csv \
    --data.custom.format csv --data.custom.text_column content
```

### HuggingFace 数据集

```bash
python src/cli.py --config config.yaml --data.source hf:wikitext
```

## 可视化类型

### Magnitude 分析

分析模型各层激活值的 magnitude 统计（Top-1/2/3, Median, Min）：

```yaml
visualization:
  enabled:
    magnitude_input: true   # 输入激活值
    magnitude_output: true  # 输出激活值
    magnitude_weight: true  # 权重 magnitude
  
  magnitude:
    patterns:
      - "mlp\\.down_proj"
      - "self_attn\\.q_proj"
    reduce_dim: -1  # -1 跨 token，-2 跨 channel
```

### Outlier 分析

分析 outlier token 的数量、位置和内容：

```yaml
visualization:
  enabled:
    outlier_layer_wise: true  # 每层 outlier 数量
    outlier_position: true    # outlier 位置分布
    outlier_token: true       # outlier token 内容
  
  outlier:
    threshold: 64              # outlier 阈值（相对于中位数的倍数）
    pattern: "layers\\.\\d+$"  # 分析的层
    decode_tokens: true        # 解码 token 内容
```

### 分布分析

分析激活值的分布（箱线图、直方图）：

```yaml
visualization:
  enabled:
    distribution: true
  
  distribution:
    patterns:
      - "mlp\\.down_proj"
    plot_types: ["boxplot", "histogram"]
    is_input: true
```

### 3D 热力图

可视化激活值的 3D 热力图：

```yaml
visualization:
  enabled:
    heatmap_3d: true
  
  heatmap_3d:
    pattern: "layers\\.\\d+$"
    sample_idx: 0
    is_input: false
    view_angle: [30, 45]
```

## 项目结构

```
quant-smooth-ana/
├── configs/                    # YAML 配置文件
│   ├── default.yaml           # 默认配置
│   └── test.yaml              # 测试配置
├── src/
│   ├── cli.py                 # 命令行入口
│   ├── core/                  # 核心模块
│   │   ├── layer_matcher.py   # 层名称正则匹配器
│   │   └── memory_manager.py  # 显存管理器
│   ├── statistics/            # 统计计算器
│   │   ├── base.py            # 基类
│   │   ├── magnitude.py       # Magnitude 统计
│   │   └── outlier.py         # Outlier 统计
│   ├── visualization/         # 可视化模块
│   │   ├── magnitude_plot.py  # Magnitude 可视化
│   │   ├── outlier_plot.py    # Outlier 可视化
│   │   ├── distribution_plot.py  # 分布可视化
│   │   ├── heatmap_3d.py      # 3D 热力图
│   │   └── report_generator.py   # HTML 报告生成
│   ├── data/                  # 数据加载器
│   │   ├── builtin_datasets.py   # 内置数据集
│   │   ├── custom_datasets.py    # 自定义数据集
│   │   └── hf_datasets.py        # HuggingFace 数据集
│   └── utils/                 # 工具函数
│       ├── config.py          # 配置加载器
│       └── export.py          # 数据导出
├── figures/                   # 输出目录
├── requirements.txt
└── README.md
```

## 输出

分析完成后，会在 `save_dir` 目录下生成：

- `*.png`: 各种可视化图表
- `analysis_report.html`: 包含所有图表的 HTML 报告
- `*.csv` / `*.json`: 导出的统计数据（如果启用）

## 支持的模型

由于使用正则表达式匹配层，本工具理论上支持任意 PyTorch 模型，包括但不限于：

- LLaMA / LLaMA-2 / LLaMA-3
- Qwen / Qwen2 / Qwen2.5 / Qwen2.5-VL
- Mistral
- Gemma
- InternLM
- Phi
- DeepSeek
- 以及其他基于 Transformer 的模型

只需根据模型的层命名规则编写相应的正则表达式即可。

## License

MIT License
