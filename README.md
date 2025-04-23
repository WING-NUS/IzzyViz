# IzzyViz

**IzzyViz** is a Python library designed to visualize attention scores in [transformer](https://jalammar.github.io/illustrated-transformer/) models. It provides flexible visualization functions that can handle various attention scenarios and model architectures. Additionally, it offers three attention heatmap variants that enable comparisons between two attention matrices, visualize model stability, and track the evolution of attention patterns over training time steps. Lastly, it includes an automatic key region highlighting function to assist users in identifying important attention areas. The output of all functions is provided in a **static** PDF format, making it suitable for direct use in research writing.

## 🚀 Quick Tour

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Self-Attention Visualization](#self-attention-visualization)
  - [Encoder-Decoder Attention](#encoder-decoder-attention)
  - [Compare Attention Maps](#compare-attention-maps)
  - [Visualize Attention Stability](#visualize-attention-stability)
  - [Attention Evolution Over Time](#attention-evolution-over-time)
  - [Detected Attention Regions](#detected-attention-regions)
- [Function Reference](#function-reference)
  - [`visualize_attention_self_attention`](#visualize_attention_self_attention)
  - [`visualize_attention_encoder_decoder`](#visualize_attention_encoder_decoder)
  - [`compare_two_attentions_with_circles`](#compare_two_attentions_with_circles)
  - [`check_stability_heatmap_with_gradient_color`](#check_stability_heatmap_with_gradient_color)
  - [`visualize_attention_evolution_sparklines`](#visualize_attention_evolution_sparklines)
  - [`visualize_attention_with_detected_regions`](#visualize_attention_with_detected_regions)
  - [`find_attention_regions_with_merging`](#find_attention_regions_with_merging)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Flexible Visualization Functions**: Supports multiple transformer architectures:
  - **Encoder-Only Models** (e.g., BERT)
  - **Decoder-Only Models** (e.g., GPT-2)
  - **Encoder-Decoder Models** (e.g., T5, BART)
- **Multiple Visualization Modes**:
  - **Self-Attention**
  - **Cross-Attention**
- **Advanced Analysis Features**:
  - **Compare attention patterns** between different heads or layers
  - **Visualize attention stability** across multiple runs
  - **Track attention evolution** over training time steps
  - **Automatic region detection** to highlight important attention patterns
- **Highlighting and Annotation**:
  - Highlights top attention scores with enlarged cells and annotations
  - Customizable region highlighting with boxes around specified areas
- **Customizable Visualization**:
  - Adjustable color mapping and normalization
  - Configurable parameters to suit different analysis needs
- **High-Quality Outputs**:
  - Generates heatmaps saved as PDF files for easy sharing and publication

## Installation

You can install **IzzyViz** via `pip`:

```bash
git clone https://github.com/lxz333/IzzyViz.git
cd IzzyViz
pip install .
```

## Dependencies

**IzzyViz** requires the following packages:

- Python 3.6 or higher
- `matplotlib>=3.0.0`
- `numpy>=1.15.0,<2.0.0`
- `torch>=1.0.0`
- `transformers>=4.0.0`
- `pandas>=1.4.0`
- `pybind11>=2.12`

These dependencies will be installed automatically when you install **IzzyViz** via `pip`.

## Quick Start

Here's a quick example of how to use **IzzyViz** to visualize self-attention in a transformer model:

```python
from transformers import BertTokenizer, BertModel
import torch
from izzyviz import visualize_attention_self_attention

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Single sentence input
sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Get attention weights
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

# Specify regions to highlight (e.g., words "fox" to "lazy")
left_top_cells = [(4, 4)]   # Starting cell (row index, column index)
right_bottom_cells = [(8, 8)]  # Ending cell (row index, column index)

# Visualize attention
visualize_attention_self_attention(
    attentions,
    tokens,
    layer=-1,
    head=8,
    top_n=4,
    mode='self_attention',
    left_top_cells=left_top_cells,
    right_bottom_cells=right_bottom_cells,
    plot_titles=["Custom Self-Attention Heatmap Title"]
)
```

This will generate a heatmap PDF file showing the self-attention patterns.
![quick_start.jpg](images/quick_start.jpg)

## Usage Examples

### Self-Attention Visualization

**Description**: Visualizes self-attention within a sequence in transformer models.

**Example**:

```python
from izzyviz import visualize_attention_self_attention
from transformers import BertTokenizer, BertModel
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Input text
sentence = "Deep learning models are revolutionizing AI."
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Get attention weights
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions

# Visualize self-attention
visualize_attention_self_attention(
    attentions=attentions,
    tokens=tokens,
    layer=-1,    # Last layer
    head=0,      # First attention head
    mode='self_attention'
)
```

### Encoder-Decoder Attention

**Description**: Visualizes cross-attention between the decoder and encoder outputs in an encoder-decoder model.

**Example**:

```python
from izzyviz import visualize_attention_encoder_decoder
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-en-de", output_attentions=True)
encoder_input_ids = tokenizer("The environmental scientists have discovered that climate change is affecting biodiversity in remote mountain regions.", return_tensors="pt", add_special_tokens=True).input_ids
with tokenizer.as_target_tokenizer():
    decoder_input_ids = tokenizer("Die Umweltwissenschaftler haben entdeckt, dass der Klimawandel die Artenvielfalt in abgelegenen Bergregionen beeinflusst.", return_tensors="pt", add_special_tokens=True).input_ids

outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
encoder_text = tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
decoder_text = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

visualize_attention_encoder_decoder(
    attention_matrix=outputs.cross_attentions[0][0, 3],
    encoder_tokens=encoder_text,
    decoder_tokens=decoder_text,
    top_n=3
)
```

### Compare Attention Maps

**Description**: Compares two attention matrices using a visualization with circles to highlight differences.

**Example**:

```python
from izzyviz import compare_two_attentions_with_circles
import numpy as np

# Create sample attention matrices
tokens = ["[CLS]", "Hello", "world", "!"]
attn1 = np.random.rand(4, 4)  # First attention matrix
attn2 = np.random.rand(4, 4)  # Second attention matrix

# Visualize the comparison
compare_two_attentions_with_circles(
    attn1=attn1,
    attn2=attn2,
    tokens=tokens,
    title="Comparing Two Attention Patterns",
    circle_scale=1.0
)
```

### Visualize Attention Stability

**Description**: Visualizes the stability of attention patterns across multiple runs or samples.

**Example**:

```python
from izzyviz import check_stability_heatmap_with_gradient_color
import numpy as np

# Create sample attention matrices (list of matrices from different runs)
tokens = ["[CLS]", "Hello", "world", "!"]
matrices = [np.random.rand(4, 4) for _ in range(5)]  # 5 different runs

# Visualize stability with gradient-colored circles
check_stability_heatmap_with_gradient_color(
    matrices=matrices,
    x_labels=tokens,
    y_labels=tokens,
    title="Attention Stability Across Runs",
    use_std_error=True,
    circle_scale=1.0
)
```

### Attention Evolution Over Time

**Description**: Visualizes how attention patterns evolve over time (e.g., training epochs).

**Example**:

```python
from izzyviz import visualize_attention_evolution_sparklines
import numpy as np

# Create sample attention matrices for different epochs
tokens = ["[CLS]", "Hello", "world", "!"]
num_epochs = 10
attentions_over_time = np.random.rand(num_epochs, 12, 12, 4, 4)  # 10 epochs, 12 layers, 12 heads, 4x4 attention matrices

# Visualize evolution with sparklines
visualize_attention_evolution_sparklines(
    attentions_over_time=attentions_over_time,
    tokens=tokens,
    layer=0,
    head=0,
    title="Attention Evolution Over Training"
)
```

### Detected Attention Regions

**Description**: Automatically detects and highlights important regions in attention maps.

**Example**:

```python
from izzyviz import visualize_attention_with_detected_regions, find_attention_regions_with_merging
import numpy as np

# Create a sample attention matrix
tokens = ["[CLS]", "Hello", "world", "!", "[SEP]"]
attention_matrix = np.random.rand(5, 5)
attention_matrix[1:3, 1:3] = 0.9  # Create a region of high attention

# Visualize with automatically detected regions
visualize_attention_with_detected_regions(
    attention_matrix=attention_matrix,
    source_tokens=tokens,
    target_tokens=tokens,
    title="Attention with Detected Regions",
    n_regions=2,
    region_color='orange'
)
```

## Function Reference

### `visualize_attention_self_attention`

**Signature**:

```python
visualize_attention_self_attention(
    attentions,
    tokens,
    layer,
    head,
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    question_end=None,
    top_n=3,
    enlarged_size=1.8,
    gamma=1.5,
    mode='self_attention',
    plot_titles=None,
    left_top_cells=None,
    right_bottom_cells=None,
    auto_detect_regions=False,
    save_path=None,
    length_threshold=64,
    if_interval=False,
    if_top_cells=True,
    interval=10,
    show_scores_in_enlarged_cells=True
)
```

**Description**:
Visualizes self-attention patterns in transformer models with various customization options for highlighting important attention scores and regions.

### `visualize_attention_encoder_decoder`

**Signature**:

```python
visualize_attention_encoder_decoder(
    attention_matrix,
    encoder_tokens,
    decoder_tokens,
    xlabel=None,
    ylabel=None,
    top_n=3,
    enlarged_size=1.8,
    gamma=1.5,
    plot_title=None,
    left_top_cells=None,
    right_bottom_cells=None,
    save_path=None,
    use_case='cross_attention'
)
```

**Description**:
Visualizes cross-attention between encoder and decoder components in encoder-decoder models, showing how decoder tokens attend to encoder tokens.

### `compare_two_attentions_with_circles`

**Signature**:

```python
compare_two_attentions_with_circles(
    attn1,
    attn2,
    tokens,
    title="Comparison with Circles",
    xlabel=None,
    ylabel=None,
    save_path=None,
    circle_scale=1.0,
    gamma=1.5,
    cmap="Blues",
    max_circle_ratio=0.45
)
```

**Description**:
Compares two attention matrices using a visualization technique that shows the base attention as a heatmap and the differences as circles of varying sizes.

### `check_stability_heatmap_with_gradient_color`

**Signature**:

```python
check_stability_heatmap_with_gradient_color(
    matrices,
    x_labels=None,
    y_labels=None,
    title="Check Stability Heatmap with Gradient Circles",
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    use_std_error=True,
    circle_scale=1.0,
    cmap="Blues",
    linecolor="white",
    linewidths=1.0,
    save_path="check_stability_heatmap_with_gradient_color.pdf",
    gamma=1.5,
    radial_resolution=100,
    use_white_center=True,
    color_contrast_scale=2.0,
    max_circle_ratio=0.45
)
```

**Description**:
Visualizes the stability (variance) of attention patterns across multiple runs or samples using gradient-colored circles to represent the mean and standard error/deviation.

### `visualize_attention_evolution_sparklines`

**Signature**:

```python
visualize_attention_evolution_sparklines(
    attentions_over_time,
    tokens=None,
    layer=None,
    head=None,
    title="Attention Evolution Over Training",
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    figsize=(12, 10),
    sparkline_color_dark="darkblue",
    sparkline_color_light="white",
    sparkline_linewidth=1.0,
    sparkline_alpha=0.8,
    gamma=1.5,
    normalize_sparklines=True,
    save_path="attention_evolution_sparklines.pdf"
)
```

**Description**:
Visualizes how attention patterns evolve over time (e.g., training epochs) using sparklines embedded in each cell of the attention matrix.

### `visualize_attention_with_detected_regions`

**Signature**:

```python
visualize_attention_with_detected_regions(
    attention_matrix,
    source_tokens,
    target_tokens,
    title="Attention with Detected Regions",
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    n_regions=3,
    min_distance=2,
    expansion_threshold=0.9,
    merge_threshold=0.6,
    region_color='orange',
    region_linewidth=2,
    region_alpha=0.7,
    label_regions=False,
    gamma=1.5,
    save_path="attention_with_detected_regions.pdf",
    ax=None,
    cmap="Blues",
    max_expansion_steps=3,
    proximity_threshold=2
)
```

**Description**:
Visualizes attention matrices with automatically detected important regions highlighted with colored boxes.

### `find_attention_regions_with_merging`

**Signature**:

```python
find_attention_regions_with_merging(
    attention_matrix,
    n_seeds=3,
    min_distance=2,
    expansion_threshold=0.8,
    merge_std_threshold=0.8,
    proximity_threshold=2,
    max_expansion_steps=3
)
```

**Description**:
Identifies important regions in an attention matrix by finding high-attention seeds and expanding them, then merging nearby regions with similar attention patterns.

## Contributing

Contributions are welcome! If you have ideas for improvements or encounter any issues, please open an issue or submit a pull request on [GitHub](https://github.com/lxz333/IzzyViz).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# Changelog

- **Updated Function Exports**: The library now exports these specialized visualization functions:
  - `visualize_attention_self_attention`
  - `visualize_attention_encoder_decoder`
  - `compare_two_attentions_with_circles`
  - `check_stability_heatmap_with_gradient_color`
  - `visualize_attention_evolution_sparklines`
  - `visualize_attention_with_detected_regions`
  - `find_attention_regions_with_merging`
- **Enhanced Visualization Capabilities**: Added support for comparing attention patterns, analyzing stability, and automatically detecting important regions.
- **Improved Documentation**: The README and function descriptions have been updated to reflect the new capabilities.

# Getting Help

If you have any questions or need assistance, feel free to open an issue on GitHub or reach out to the maintainers.

---

Thank you for using **IzzyViz**! We hope this tool aids in your exploration and understanding of attention mechanisms in transformer models.
