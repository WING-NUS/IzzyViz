# IzzyViz

**IzzyViz** is a Python library designed to visualize attention scores in [transformer](https://jalammar.github.io/illustrated-transformer/) models. It provides flexible visualization functions that can handle various attention scenarios and model architectures. Additionally, it offers three attention heatmap variants that enable comparisons between two attention matrices, visualize model stability, and track the evolution of attention patterns over training time steps. Lastly, it includes an automatic key region highlighting function to assist users in identifying important attention areas. The output of all functions is provided in a **static** PDF format, making it suitable for direct use in research writing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Self-Attention Visualization](#self-attention-visualization)
  - [Cross-Attention Visualization](#cross-attention-visualization)
  - [Comparison Heatmap](#comparison-heatmap)
  - [Stability Heatmap](#stability-heatmap)
  - [Evolution Heatmap](#evolution-heatmap)
  - [Automatic Key Region Detection](#automatic-key-region-detection)
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
    top_n=5,
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

**Description**: This function includes two modes: "self_attention" and "question_context". The "self_attention" mode is designed for models that rely primarily on self-attention mechanisms, such as encoder-only models like BERT and decoder-only models like GPT. It visualizes how tokens attend to other tokens within the same sequence. The "question_context" mode is an extension of the self-attention mode and is intended for tasks like question answering, where there is a clear distinction between the query and its context.

**Example**:

```python
from transformers import BertTokenizer, BertModel
import torch
from izzyviz.visualization import visualize_attention_self_attention

# Example usage
# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Prepare example input
sentence_A = "Where is the Eiffel Tower?"  # Question
sentence_B = "The Eiffel Tower is located in Paris, France."  # Context
inputs = tokenizer(sentence_A, sentence_B, return_tensors="pt", add_special_tokens=True)

input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids'][0]
question_end = (token_type_ids == 0).sum().item()  # Find the end of the question

# Get attention weights
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

# Visualize attention and save to PDF
visualize_attention_self_attention(
                    attentions,
                    tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                    layer=6,
                    head=5,
                    question_end=question_end,
                    top_n=3,
                    enlarged_size=2.3,
                    mode='question_context'
)
```

**Output**: [QC_attention_heatmaps.pdf]()

### Cross-Attention Visualization

**Description**: This function visualizes how decoder tokens attend to encoder tokens, revealing cross-sequence relationships essential for tasks like translation or summarization.

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

**Output**:

![cross-attention_heatmap.jpg](images/cross-attention_heatmap.jpg)

### Comparison Heatmap

**Description**: When analyzing transformer models, researchers often need to compare attention patterns between different models, layers, heads, or before and after fine-tuning. IzzyViz addresses this challenge with the comparison heatmap with circles visualization. This novel approach preserves information from both attention matrices while intuitively highlighting their differences. The cell colors represent the actual attention values from the first input matrix, preserving the complete information from the first attention pattern. Each cell contains a circle whose color represents the attention value from the second input matrix, ensuring that information from the second attention pattern is also fully preserved. The size of each circle is proportional to the magnitude of difference between the two matrices at that position. Larger circles instantly draw the user’s attention to cells with greater differences, while smaller circles indicate similar attention values.

**Example**:

```python
import torch
from transformers import BertTokenizer, BertModel
from izzyviz.visualization import compare_two_attentions_with_circles

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

text = "This is an example sentence for comparing two attention heads in BERT."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attn_layer8 = outputs.attentions[7][0]
attn_layer9 = outputs.attentions[8][0]

# Get head 9 (index 8) from each layer
attn_layer8_head9 = attn_layer8[8]
attn_layer9_head9 = attn_layer9[8]

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

compare_two_attentions_with_circles(
    attn1=attn_layer8_head9,
    attn2=attn_layer9_head9,
    tokens=tokens,
    title="Comparison: Layer 8 Head 9 vs Layer 9 Head 9"
)
```

**Output**:
![comparison_heatmap.jpg](images/comparison_heatmap.jpg)

### Stability Heatmap

**Description**: When developing transformer models, researchers often need to assess the stability of attention patterns across different training runs to gain deeper insights into model behavior. IzzyViz addresses this challenge with the stability heatmap visualization, which elegantly combines average attention values with visual representations of variability. Each cell’s background color represents the average attention score across all N training runs. The size of each circle is proportional to the confidence interval at each position. Larger circles instantly highlight positions with greater variability across training runs, indicating less stable attention patterns. Each circle features a gradient that transitions from the lower bound of all attention scores at the center to the upper bound of all attention scores at the edge.

**Example**:

```python
from izzyviz import check_stability_heatmap_with_gradient_color
import numpy as np

# Load previously saved attention matrices from five training runs
attn_array = np.load("attention_matrices_5_runs_test_sample.npy")

# Specify layer and head index
layer_idx = 6
head_idx = 9

# Step 1: extract (5, 64, 64) at specific layer and head
attn_head = attn_array[:, layer_idx, head_idx, :, :]  # shape: (5, 64, 64)

# Step 2: extract first 24 tokens (submatrix)
attn_submatrix = attn_head[:, :24, :24]  # shape: (5, 24, 24)

labels = ['[CLS]', 'starts', 'off', 'with', 'a', 'bang', ',', 'but', 'then', 'fi', '##zzle', '##s', 'like', 'a', 'wet', 'stick', 'of', 'dynamite', 'at', 'the', 'very', 'end', '.', '[SEP]']

check_stability_heatmap_with_gradient_color(
    attn_submatrix,
    x_labels=labels,
    y_labels=labels
)
```

**Output**:
![stability_heatmap.jpg](images/stability_heatmap.jpg)

### Evolution Heatmap

**Description**: While standard attention visualizations provide a snapshot of attention patterns at a single point in time, researchers often need to understand how these patterns develop and stabilize throughout the training process. IzzyViz addresses this challenge with the evolution heatmap visualization, which integrates temporal information directly into the heatmap structure. Each cell’s background color represents the average attention score across all training time steps, providing a reference point for the overall attention pattern. At the center of each cell, a sparkline (mini line chart) visualizes the trend of the attention score over training steps.

**Example**:

```python
from izzyviz import visualize_attention_evolution_sparklines
import numpy as np

# Load previously saved attention matrices from 5 training epochs
attention_matrices = np.load("attention_matrices_epochs.npy")  # shape: (5, 12, 12, 64, 64)

# Extract attention matrices across epochs
selected = attention_matrices[:, :, :, :19, :19]

visualize_attention_evolution_sparklines(
    selected,
    tokens=['[CLS]', 'as', 'they', 'come', ',', 'already', 'having', 'been', 'recycled', 'more', 'times', 'than', 'i', "'", 'd', 'care', 'to', 'count', '[SEP]'],
    layer=11,
    head=9
)
```

**Output**:
![evolution_heatmap.jpg](images/evolution_heatmap.jpg)

### Automatic Key Region Detection

**Description**: This algorithm helps to automatically identify and highlight significant attention patterns within heatmaps.

**Example**:

```python
from izzyviz import visualize_attention_with_detected_regions
from transformers import BertTokenizer, BertModel
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Single sentence input
sentence = "The quick brown fox jumps over the lazy dog while a cat watches from behind the tree."
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Get attention weights
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

visualize_attention_with_detected_regions(
    attention_matrix=attentions[0][0, 3],
    source_tokens=tokens,
    target_tokens=tokens
)
```

**Output**:
![region_detection.jpg](images/region_detection.jpg)

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
    show_scores_in_enlarged_cells=True,
    lean_more=False
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
    use_case='cross_attention',
    lean_more=False
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
    use_white_center=False,
    color_contrast_scale=2.0,
    max_circle_ratio=0.45
)
```

**Description**:
Visualizes the stability (variance) of attention patterns across multiple runs using gradient-colored circles to represent the mean and standard error/deviation.

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
    normalize_sparklines=False,
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
    expansion_threshold=0.7,
    merge_std_threshold=0.6,
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
- **Enhanced Visualization Capabilities**: Added support for comparing attention patterns, analyzing stability, illustrating the evolution of attention patterns, and automatically detecting important regions.
- **Improved Documentation**: The README and function descriptions have been updated to reflect the new capabilities.

# Getting Help

If you have any questions or need assistance, feel free to open an issue on GitHub or reach out to the maintainers.

---

Thank you for using **IzzyViz**! We hope this tool aids in your exploration and understanding of attention mechanisms in transformer models.
