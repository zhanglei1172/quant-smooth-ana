# Quant Smooth Ana

A modular and extensible tool for visualizing model outlier statistics and distributions.

## Features

- **Modular Architecture**: Plugin-based design for easy model extension
- **Automatic Memory Management**: Dynamic GPU/CPU offload for optimal memory usage
- **Flexible Layer Matching**: Regex-based layer name matching
- **Multiple Data Sources**: Built-in datasets, custom files, and HuggingFace datasets
- **Rich Visualizations**: Magnitude plots, outlier analysis, 3D heatmaps
- **Comprehensive Reports**: HTML reports with all visualizations
- **Data Export**: CSV and JSON export formats

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create a configuration file

Create a YAML configuration file (e.g., `configs/llama.yaml`):

```yaml
model:
  path: "/path/to/your/model"
  name: "llama-2-7b"
  family: "llama"

data:
  source: "pile"  # or custom:/path/to/file.txt
  num_samples: 64
  seq_len: 1024
  seed: 0

visualization:
  save_dir: "./figures"
  dpi: 200
  
  enabled:
    magnitude_input: true
    magnitude_output: true
    outlier_layer_wise: true
    outlier_position: true
    outlier_token: true
  
  magnitude:
    components:
      - q_proj
      - o_proj
      - down_proj
      - up_proj
    per_tensor: false
  
  outlier:
    threshold: 64
    component_type: "hidden_state"
    decode_tokens: true
  
  output:
    formats: ["png", "html", "csv", "json"]
    report_name: "analysis_report"

memory:
  device: "cuda"
  offload_device: "cpu"
  auto_offload: true
```

### 2. Run analysis

```bash
python src/cli.py --config configs/llama.yaml
```

### 3. Override configuration parameters

```bash
python src/cli.py --config configs/llama.yaml --data.num_samples 128 --outlier.threshold 100
```

## Supported Models

- LLaMA / LLaMA-2 / LLaMA-3
- Qwen / Qwen2 / Qwen2.5-VL
- Mistral
- Gemma
- InternLM
- Phi

## Data Sources

### Built-in Datasets

- `pile`: Pile dataset
- `wikitext2`: WikiText-2 dataset
- `c4`: C4 dataset
- `redpajama`: RedPajama dataset

### Custom Datasets

```bash
# Text file
python src/cli.py --config config.yaml --data.source custom:/path/to/data.txt

# JSON file
python src/cli.py --config config.yaml --data.source custom:/path/to/data.json --data.custom.format json --data.custom.text_column text

# CSV file
python src/cli.py --config config.yaml --data.source custom:/path/to/data.csv --data.custom.format csv --data.custom.text_column content
```

### HuggingFace Datasets

```bash
python src/cli.py --config config.yaml --data.source hf:wikitext
```

## Adding New Models

To add support for a new model, create a new adapter in `src/models/`:

```python
from src.models.base import BaseModelAdapter
from src.core.registry import ModelRegistry

@ModelRegistry.register("your_model")
class YourModelAdapter(BaseModelAdapter):
    def get_layers(self):
        return self.model.model.layers
    
    def get_embeddings(self):
        return [self.model.model.embed_tokens]
    
    def get_layer_name_pattern(self, component_type: str):
        patterns = {
            'q_proj': r'attention\.q_proj',
            'k_proj': r'attention\.k_proj',
            'v_proj': r'attention\.v_proj',
            'o_proj': r'attention\.o_proj',
            'gate_proj': r'mlp\.gate_proj',
            'up_proj': r'mlp\.up_proj',
            'down_proj': r'mlp\.down_proj',
            'hidden_state': r'$'
        }
        return patterns.get(component_type)
    
    def get_full_layer_name(self, layer_idx: int, component_type: str):
        if component_type == 'hidden_state':
            return f'model.layers.{layer_idx}'
        pattern = self.get_layer_name_pattern(component_type)
        return f'model.layers.{layer_idx}.{pattern.replace(r"\.", ".")}'
```

## Project Structure

```
quant-smooth-ana/
├── configs/                    # YAML configuration files
├── src/
│   ├── core/                  # Core modules
│   │   ├── registry.py        # Model registry
│   │   ├── layer_matcher.py   # Layer name matcher
│   │   └── memory_manager.py  # Memory manager
│   ├── models/                # Model adapters
│   ├── statistics/            # Statistics calculators
│   ├── visualization/         # Visualization modules
│   ├── data/                  # Data loaders
│   └── utils/                 # Utilities
├── requirements.txt
└── README.md
```

## License

MIT License