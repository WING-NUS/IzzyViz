# visualization.py

from .my_seaborn import heatmap
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches  # Import patches to draw the rectangle
from .utility import find_non_overlapping_locally_maximal_rectangles
from matplotlib.patches import Wedge


# Define a function to make special tokens bold
def bold_special_tokens(label):
    special_tokens = ['[CLS]', '[SEP]', '[PAD]']
    if label in special_tokens:
        return f'$\mathbf{{{label}}}$'  # Make it bold using LaTeX math formatting
    return label

def create_tablelens_heatmap(attention_matrix, x_labels, y_labels, title, xlabel, ylabel, ax, cmap='Blues',
                             column_widths=None, row_heights=None, top_cells=None, vmin=None,
                             vmax=None, norm=None, gamma=2.0, left_top_cells=None, right_bottom_cells=None, linecolor='white', linewidths=1.0,
                             cbar=True):
    """
    Creates a heatmap with variable cell sizes and annotations for top cells.
    Returns both the axis and the plotter object for further customization.
    """

    if isinstance(attention_matrix, np.ndarray):
        data = attention_matrix  # It's already a NumPy array, no need to convert
    else:
        data = attention_matrix.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy

    # print("data: ", data.shape)

    # Create annot_data for annotations
    annot_data = np.empty_like(data, dtype=object)
    annot_data[:] = ''  # Initialize all cells to empty strings

    if top_cells is not None:
        for (row_index, col_index) in top_cells:
            value = data[row_index, col_index]
            annot_data[row_index, col_index] = f"{value:.3f}"

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    if norm is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # Create the heatmap
    ax, plotter = heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        linewidths=linewidths,
        linecolor=linecolor,
        square=True,  # Ensure non-highlighted cells are square
        # cbar_kws={"shrink": 1.0},
        cbar=False,  # Disable the default colorbar
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        ax=ax,
        column_widths=column_widths,
        row_heights=row_heights,
        annot=annot_data,
        fmt=""
    )

    # Adjust colorbar ticks
    # cbar = ax.collections[0].colorbar  # Get the colorbar

    # Define the data values for ticks (evenly spaced)
    num_ticks = 7 # Adjust the number of ticks as needed
    tick_values = np.linspace(vmin, vmax, num_ticks)

    # Compute the positions along the colorbar where ticks should be placed
    # normalized_positions = (tick_values - vmin) / (vmax - vmin)
    
    # adjusted_positions = normalized_positions ** gamma

    # Map adjusted positions back to data values
    # adjusted_tick_values = vmin + adjusted_positions * (vmax - vmin)

    # Set the ticks and labels on the colorbar
    # cbar.set_ticks(tick_values)

    # normalized_positions = norm(tick_values)
    # cbar.set_ticks(normalized_positions)
    # cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])

    if cbar:
        # Create a new axis for the colorbar that matches the heatmap's height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Add the colorbar
        im = ax.collections[0]
        cbar = plt.colorbar(im, cax=cax)

        # Remove the black border around the colorbar
        cbar.outline.set_visible(False)

        # Adjust colorbar ticks
        num_ticks = 7  # Adjust the number of ticks as needed
        tick_values = np.linspace(vmin, vmax, num_ticks)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    for label in ax.get_xticklabels():
        label.set_rotation(45)

    for label in ax.get_yticklabels():
        label.set_rotation(0)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if top_cells is not None:
        # Highlight tick labels corresponding to top_cells
        x_ticklabels = ax.get_xticklabels()
        y_ticklabels = ax.get_yticklabels()

        x_indices = set(col_index for (row_index, col_index) in top_cells)
        y_indices = set(row_index for (row_index, col_index) in top_cells)

        # Adjust x tick labels
        for idx, label in enumerate(x_ticklabels):
            if idx in x_indices:
                label.set_bbox(dict(facecolor='yellowgreen', edgecolor='yellowgreen', boxstyle='round,pad=0.2', alpha=0.5))

        # Adjust y tick labels without inversion
        for row_index in y_indices:
            if row_index < len(y_ticklabels):
                label = y_ticklabels[row_index]
                label.set_bbox(dict(facecolor='yellowgreen', edgecolor='yellowgreen', boxstyle='round,pad=0.2', alpha=0.5))
    
    # Draw red rectangles around the specified regions
    if left_top_cells is not None and right_bottom_cells is not None:
        if len(left_top_cells) != len(right_bottom_cells):
            raise ValueError("left_top_cells and right_bottom_cells must have the same length.")

        for lt_cell, rb_cell in zip(left_top_cells, right_bottom_cells):
            lt_row, lt_col = lt_cell
            rb_row, rb_col = rb_cell

            if lt_row > rb_row or lt_col > rb_col:
                print("lt_row: ", lt_row)
                print("rb_row: ", rb_row)
                print("lt_col: ", lt_col)
                print("rb_col: ", rb_col)
                raise ValueError("Invalid cell coordinates. Left-top cell must be above and to the left of the right-bottom cell.")
            
            if (lt_row < 0 or 
                lt_col < 0 or 
                rb_row < 0 or 
                rb_col < 0 or
                rb_row >= data.shape[0] or 
                rb_col >= data.shape[1] or
                lt_row >= data.shape[0] or
                lt_col >= data.shape[1]):
                raise ValueError("Invalid cell coordinates. Coordinates must be within the attention matrix.")

            # Get the positions of the cell edges
            col_positions = plotter.col_positions
            row_positions = plotter.row_positions

            # Compute the rectangle's position and size
            x = col_positions[lt_col]
            # print("col_positions: ", col_positions)
            # print("x: ", x)
            width = col_positions[rb_col + 1] - col_positions[lt_col]
            # print("width: ", width)
            y = row_positions[lt_row]
            # print("row_positions: ", row_positions)
            # print("y: ", y)
            height = row_positions[rb_row + 1] - row_positions[lt_row]
            # print("height: ", height)

            # Draw the rectangle
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

    return ax, plotter



def visualize_attention_encoder_only(attentions, tokens, layer, head, question_end=None,
                                     top_n=3, enlarged_size=1.8, gamma=1.5, mode='self_attention',
                                     plot_titles=None, left_top_cells=None, right_bottom_cells=None,
                                     auto_detect_regions=False, save_path=None):
    """
    Visualizes attention matrices for encoder-only models.

    Parameters:
    - attentions: List of attention matrices from the model.
    - tokens: List of token labels to display on the heatmaps.
    - layer: The layer number of the attention to visualize.
    - head: The head number of the attention to visualize.
    - question_end: The index where the first sentence ends in the token list (used in question-context modes).
    - top_n: The number of top attention scores to highlight.
    - enlarged_size: Factor by which to enlarge the top cells.
    - gamma: Gamma value for the power normalization of the colormap.
    - mode: The mode of visualization ('self_attention' or 'question_context').
    - plot_titles: List of titles for the subplots. If None, default titles are used.
    - left_top_cells: List of (row, col) tuples for the top-left cells of regions to highlight.
    - right_bottom_cells: List of (row, col) tuples for the bottom-right cells of regions to highlight.
    - auto_detect_regions: If True, automatically detect locally maximal attention regions.
                          This will override any manually specified left_top_cells and right_bottom_cells.
    """
    # Removes the first dimension if it is 1 (typically the batch size for a single input).
    attn = attentions[layer].squeeze(0)[head]

    if auto_detect_regions:
        # Convert attention matrix to numpy if it's a tensor
        attn_np = attn.detach().cpu().numpy() if torch.is_tensor(attn) else attn
        # Find locally maximal rectangles
        left_top_cells, right_bottom_cells = find_non_overlapping_locally_maximal_rectangles(attn_np)

    if mode == 'question_context':
        if question_end is None:
            raise ValueError("question_end must be provided for question_context mode.")
        fig, axes = plt.subplots(5, 1, figsize=(10, 40))

        attention_matrices = [
            attn[:question_end, :question_end],  # A->A
            attn[question_end:, question_end:],  # B->B
            attn[question_end:, :question_end],  # B->A
            attn[:question_end, question_end:],  # A->B
            attn                                 # All->All
        ]

        # Compute global vmin and vmax across all attention matrices
        data_list = [mat.detach().cpu().numpy() for mat in attention_matrices]
        global_vmin = min(data.min() for data in data_list)
        global_vmax = max(data.max() for data in data_list)

        # Create the normalization
        norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

        token_segment_pairs = [
            [tokens[:question_end], tokens[:question_end]],  # A->A
            [tokens[question_end:], tokens[question_end:]],  # B->B
            [tokens[question_end:], tokens[:question_end]],  # B->A
            [tokens[:question_end], tokens[question_end:]],  # A->B
            [tokens[:], tokens[:]]                           # All->All
        ]

        # Default titles
        default_titles = [
            "A -> A (Question attending to Question)",
            "B -> B (Context attending to Context)",
            "B -> A (Context attending to Question)",
            "A -> B (Question attending to Context)",
            "All -> All (All tokens attending to all tokens)"
        ]

        if plot_titles is None:
            plot_titles = default_titles
        elif len(plot_titles) != 5:
            raise ValueError("plot_titles must be a list of 5 titles for question_context mode.")

        for i, (att_matrix, title) in enumerate(zip(attention_matrices, plot_titles)):
            x_labels = [bold_special_tokens(token) for token in token_segment_pairs[i][0]]
            y_labels = [bold_special_tokens(token) for token in token_segment_pairs[i][1]]

            data = att_matrix.detach().cpu().numpy()
            # Ensure data has positive values for PowerNorm
            data_min_nonzero = data[data > 0].min() if np.any(data > 0) else 1e-6
            data[data <= 0] = data_min_nonzero / 10  # Avoid zeros or negative values

            # Find top attention cells
            top_cells = find_top_cells(data, top_n)

            # Initialize column widths and row heights
            num_rows, num_cols = data.shape
            default_width = 1
            default_height = 1
            column_widths = [default_width] * num_cols
            row_heights = [default_height] * num_rows

            for (row_index, col_index) in top_cells:
                column_widths[col_index] = enlarged_size
                row_heights[row_index] = enlarged_size

            ax, _ = create_tablelens_heatmap(
                att_matrix,
                x_labels,
                y_labels,
                title,
                "Tokens Attended to",
                "Tokens Attending",
                axes[i],
                column_widths=column_widths,
                row_heights=row_heights,
                top_cells=top_cells,
                vmin=global_vmin,
                vmax=global_vmax,
                norm=norm,
                gamma=gamma,
                left_top_cells=left_top_cells,
                right_bottom_cells=right_bottom_cells
            )

        if save_path is None:
            save_path = "QC_attention_heatmaps.pdf"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print("Attention heatmaps saved to ", save_path)

    elif mode == 'self_attention':
        # Self-Attention Mode
        # Only one plot: tokens attending to themselves
        if plot_titles is None:
            plot_titles = ["Self-Attention Heatmap"]
        elif not isinstance(plot_titles, list) or len(plot_titles) != 1:
            raise ValueError("plot_titles must be a list with one title for self_attention mode.")

        fig, ax = plt.subplots(figsize=(10, 10))

        attention_matrix = attn  # Shape: (seq_len, seq_len)
        x_labels = [bold_special_tokens(token) for token in tokens]
        y_labels = [bold_special_tokens(token) for token in tokens]
        title = plot_titles[0]

        # Prepare data
        data = attention_matrix.detach().cpu().numpy()
        global_vmin = data.min()
        global_vmax = data.max()
        norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

        # Find top attention cells
        top_cells = find_top_cells(data, top_n)

        # Initialize column widths and row heights
        num_rows, num_cols = data.shape
        default_width = 1
        default_height = 1
        column_widths = [default_width] * num_cols
        row_heights = [default_height] * num_rows

        # Enlarge top cells
        for (row_index, col_index) in top_cells:
            column_widths[col_index] = enlarged_size
            row_heights[row_index] = enlarged_size

        ax, _ = create_tablelens_heatmap(
            attention_matrix,
            x_labels,
            y_labels,
            title,
            "Tokens Attended to",
            "Tokens Attending",
            ax,
            column_widths=column_widths,
            row_heights=row_heights,
            top_cells=top_cells,
            vmin=global_vmin,
            vmax=global_vmax,
            norm=norm,
            gamma=gamma,
            left_top_cells=left_top_cells,
            right_bottom_cells=right_bottom_cells
        )

        if save_path is None:
            save_path = "self_attention_heatmap.pdf"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print("Self-attention heatmap saved to ", save_path)

    else:
        raise ValueError("Invalid mode for encoder-only visualization. Choose from 'question_context' or 'self_attention'.")

def visualize_attention_decoder_only(attentions, source_tokens, generated_tokens, layer, head,
                                     top_n=3, enlarged_size=1.8, gamma=1.5,
                                     plot_titles=None, left_top_cells=None, right_bottom_cells=None,
                                     use_case='full_sequence', save_path=None):
    """
    Visualizes attention matrices for decoder-only models.

    Parameters:
    - attentions: List of attention matrices from the model.
    - source_tokens: List of source token labels.
    - generated_tokens: List of generated token labels.
    - layer: The layer number of the attention to visualize.
    - head: The head number of the attention to visualize.
    - top_n: The number of top attention scores to highlight.
    - enlarged_size: Factor by which to enlarge the top cells.
    - gamma: Gamma value for the power normalization of the colormap.
    - plot_titles: List of titles for the subplots. If None, default titles are used.
    - left_top_cells: List of (row, col) tuples for the top-left cells of regions to highlight.
    - right_bottom_cells: List of (row, col) tuples for the bottom-right cells of regions to highlight.
    - use_case: The specific use case to visualize. Options are:
        - 'full_sequence': Input sequence attending to itself (no token generation).
        - 'self_attention_source': Self-Attention for Source Tokens (no causal masking).
        - 'generated_to_source': Generated-to-Source Attention (fully connected).
        - 'self_attention_generated': Self-Attention for Generated Tokens (causal-masked).
    """
    attn = attentions[layer].squeeze(0)[head]

    if use_case == 'full_sequence':
        # Input sequence attending to itself. The x, y labels are the same sentence.
        tokens = source_tokens
        attention_matrix = attn  # Shape: (seq_len, seq_len)
        x_labels = [bold_special_tokens(token) for token in tokens]
        y_labels = [bold_special_tokens(token) for token in tokens]
        title = plot_titles[0] if plot_titles else "Self-Attention Heatmap (Full Sequence)"

        # Prepare data
        data = attention_matrix.detach().cpu().numpy()
        global_vmin = data.min()
        global_vmax = data.max()
        norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

        # Find top attention cells
        top_cells = find_top_cells(data, top_n)

        # Initialize column widths and row heights
        num_rows, num_cols = data.shape
        default_width = 1
        default_height = 1
        column_widths = [default_width] * num_cols
        row_heights = [default_height] * num_rows

        # Enlarge top cells
        for (row_index, col_index) in top_cells:
            column_widths[col_index] = enlarged_size
            row_heights[row_index] = enlarged_size

        fig, ax = plt.subplots(figsize=(10, 10))
        ax, _ = create_tablelens_heatmap(
            attention_matrix,
            x_labels,
            y_labels,
            title,
            "Tokens Attended to",
            "Tokens Attending",
            ax,
            column_widths=column_widths,
            row_heights=row_heights,
            top_cells=top_cells,
            vmin=global_vmin,
            vmax=global_vmax,
            norm=norm,
            gamma=gamma,
            left_top_cells=left_top_cells,
            right_bottom_cells=right_bottom_cells
        )

        if save_path is None:
            save_path = "decoder_self_attention_heatmap.pdf"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print("Decoder self-attention heatmap saved to ", save_path)

    elif use_case == 'self_attention_source':
        # Self-Attention for Source Tokens (no causal masking)
        tokens = source_tokens
        seq_len = len(source_tokens)
        attention_matrix = attn[:seq_len, :seq_len]
        x_labels = [bold_special_tokens(token) for token in tokens]
        y_labels = [bold_special_tokens(token) for token in tokens]
        title = plot_titles[0] if plot_titles else "Self-Attention Heatmap (Source Tokens)"

        # Prepare data
        data = attention_matrix.detach().cpu().numpy()
        global_vmin = data.min()
        global_vmax = data.max()
        norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

        # Find top attention cells
        top_cells = find_top_cells(data, top_n)

        # Initialize column widths and row heights
        num_rows, num_cols = data.shape
        default_width = 1
        default_height = 1
        column_widths = [default_width] * num_cols
        row_heights = [default_height] * num_rows

        # Enlarge top cells
        for (row_index, col_index) in top_cells:
            column_widths[col_index] = enlarged_size
            row_heights[row_index] = enlarged_size

        fig, ax = plt.subplots(figsize=(10, 10))
        ax, _ = create_tablelens_heatmap(
            attention_matrix,
            x_labels,
            y_labels,
            title,
            "Tokens Attended to",
            "Tokens Attending",
            ax,
            column_widths=column_widths,
            row_heights=row_heights,
            top_cells=top_cells,
            vmin=global_vmin,
            vmax=global_vmax,
            norm=norm,
            gamma=gamma,
            left_top_cells=left_top_cells,
            right_bottom_cells=right_bottom_cells
        )

        if save_path is None:
            save_path = "decoder_self_attention_source_tokens_heatmap.pdf"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print("Decoder self-attention heatmap for source tokens saved to ", save_path)

    elif use_case == 'generated_to_source':
        # Generated-to-Source Attention (fully connected)
        source_seq_len = len(source_tokens)
        generated_seq_len = len(generated_tokens)
        attention_matrix = attn[source_seq_len:source_seq_len+generated_seq_len, :source_seq_len]
        x_labels = [bold_special_tokens(token) for token in source_tokens]
        y_labels = [bold_special_tokens(token) for token in generated_tokens]
        title = plot_titles[0] if plot_titles else "Generated Tokens attending to Source Tokens"

        # Prepare data
        data = attention_matrix.detach().cpu().numpy()
        global_vmin = data.min()
        global_vmax = data.max()
        norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

        # Find top attention cells
        top_cells = find_top_cells(data, top_n)

        # Initialize column widths and row heights
        num_rows, num_cols = data.shape
        default_width = 1
        default_height = 1
        column_widths = [default_width] * num_cols
        row_heights = [default_height] * num_rows

        # Enlarge top cells
        for (row_index, col_index) in top_cells:
            column_widths[col_index] = enlarged_size
            row_heights[row_index] = enlarged_size

        fig, ax = plt.subplots(figsize=(10, 10))
        ax, _ = create_tablelens_heatmap(
            attention_matrix,
            x_labels,
            y_labels,
            title,
            "Source Tokens",
            "Generated Tokens",
            ax,
            column_widths=column_widths,
            row_heights=row_heights,
            top_cells=top_cells,
            vmin=global_vmin,
            vmax=global_vmax,
            norm=norm,
            gamma=gamma,
            left_top_cells=left_top_cells,
            right_bottom_cells=right_bottom_cells
        )

        if save_path is None:
            save_path = "decoder_generated_to_source_attention_heatmap.pdf"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print("Decoder generated-to-source attention heatmap saved to ", save_path)

    elif use_case == 'self_attention_generated':
        # Self-Attention for Generated Tokens (causal-masked)
        source_seq_len = len(source_tokens)
        generated_seq_len = len(generated_tokens)
        total_seq_len = source_seq_len + generated_seq_len
        attention_matrix = attn[source_seq_len:total_seq_len, source_seq_len:total_seq_len]
        x_labels = [bold_special_tokens(token) for token in generated_tokens]
        y_labels = [bold_special_tokens(token) for token in generated_tokens]
        title = plot_titles[0] if plot_titles else "Self-Attention Heatmap (Generated Tokens)"

        # Prepare data
        data = attention_matrix.detach().cpu().numpy()
        global_vmin = data.min()
        global_vmax = data.max()
        norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

        # Find top attention cells
        top_cells = find_top_cells(data, top_n)

        # Initialize column widths and row heights
        num_rows, num_cols = data.shape
        default_width = 1
        default_height = 1
        column_widths = [default_width] * num_cols
        row_heights = [default_height] * num_rows

        # Enlarge top cells
        for (row_index, col_index) in top_cells:
            column_widths[col_index] = enlarged_size
            row_heights[row_index] = enlarged_size

        fig, ax = plt.subplots(figsize=(10, 10))
        ax, _ = create_tablelens_heatmap(
            attention_matrix,
            x_labels,
            y_labels,
            title,
            "Tokens Attended to",
            "Tokens Attending",
            ax,
            column_widths=column_widths,
            row_heights=row_heights,
            top_cells=top_cells,
            vmin=global_vmin,
            vmax=global_vmax,
            norm=norm,
            gamma=gamma,
            left_top_cells=left_top_cells,
            right_bottom_cells=right_bottom_cells
        )

        if save_path is None:
            save_path = "decoder_self_attention_generated_tokens_heatmap.pdf"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print("Decoder self-attention heatmap for generated tokens saved to ", save_path)

    else:
        raise ValueError("Invalid use_case for decoder-only visualization. Choose from 'full_sequence', 'self_attention_source', 'generated_to_source', or 'self_attention_generated'.")

def visualize_attention_encoder_decoder(attention_matrix, source_tokens, generated_tokens,
                                        top_n=3, enlarged_size=1.8, gamma=1.5,
                                        plot_title=None, left_top_cells=None, right_bottom_cells=None,
                                        save_path=None, use_case='cross_attention'):
    """
    Visualizes attention matrices for encoder-decoder models.

    Parameters:
    - attention_matrix: The attention matrix (numpy array or torch tensor).
    - source_tokens: List of source token labels.
    - generated_tokens: List of generated token labels.
    - top_n: The number of top attention scores to highlight.
    - enlarged_size: Factor by which to enlarge the top cells.
    - gamma: Gamma value for the power normalization of the colormap.
    - plot_title: Title for the plot.
    - left_top_cells: List of (row, col) tuples for the top-left cells of regions to highlight.
    - right_bottom_cells: List of (row, col) tuples for the bottom-right cells of regions to highlight.
    - save_path: File path to save the generated heatmap PDF.
    - use_case: Type of attention to visualize. Options are 'cross_attention', 'encoder_self_attention', 'decoder_self_attention'.
    """

    # Prepare data
    data = attention_matrix
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if use_case == 'cross_attention':
        # Cross-Attention: Decoder attending to Encoder outputs
        x_labels = [bold_special_tokens(token) for token in source_tokens]
        y_labels = [bold_special_tokens(token) for token in generated_tokens]
        xlabel = "Source Tokens"
        ylabel = "Generated Tokens"
        default_title = "Cross-Attention Heatmap (Decoder to Encoder)"
        expected_shape = (len(generated_tokens), len(source_tokens))

    elif use_case == 'encoder_self_attention':
        # Encoder Self-Attention
        x_labels = [bold_special_tokens(token) for token in source_tokens]
        y_labels = [bold_special_tokens(token) for token in source_tokens]
        xlabel = "Source Tokens"
        ylabel = "Source Tokens"
        default_title = "Encoder Self-Attention Heatmap"
        expected_shape = (len(source_tokens), len(source_tokens))

    elif use_case == 'decoder_self_attention':
        # Decoder Self-Attention
        x_labels = [bold_special_tokens(token) for token in generated_tokens]
        y_labels = [bold_special_tokens(token) for token in generated_tokens]
        xlabel = "Generated Tokens"
        ylabel = "Generated Tokens"
        default_title = "Decoder Self-Attention Heatmap"
        expected_shape = (len(generated_tokens), len(generated_tokens))

    else:
        raise ValueError("Invalid use_case. Choose from 'cross_attention', 'encoder_self_attention', 'decoder_self_attention'.")

    # Ensure data dimensions match tokens
    if data.shape != expected_shape:
        raise ValueError(f"Attention matrix shape {data.shape} does not match the expected shape {expected_shape} for the selected use_case.")

    global_vmin = data.min()
    global_vmax = data.max()
    norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

    # Find top attention cells
    top_cells = find_top_cells(data, top_n)

    # Initialize column widths and row heights
    num_rows, num_cols = data.shape
    default_width = 1
    default_height = 1
    column_widths = [default_width] * num_cols
    row_heights = [default_height] * num_rows

    # Enlarge top cells
    for (row_index, col_index) in top_cells:
        column_widths[col_index] = enlarged_size
        row_heights[row_index] = enlarged_size

    title = plot_title if plot_title else default_title

    fig, ax = plt.subplots(figsize=(10, 10))
    ax, _ = create_tablelens_heatmap(
        attention_matrix,
        x_labels,
        y_labels,
        title,
        xlabel,
        ylabel,
        ax,
        column_widths=column_widths,
        row_heights=row_heights,
        top_cells=top_cells,
        vmin=global_vmin,
        vmax=global_vmax,
        norm=norm,
        gamma=gamma,
        left_top_cells=left_top_cells,
        right_bottom_cells=right_bottom_cells
    )

    plt.tight_layout()
    if save_path is None:
        save_path = "attention_heatmap.pdf"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Attention heatmap saved to {save_path}")

# Helper function to find top attention cells
def find_top_cells(data, top_n):
    if top_n == 0:
        return []
    flat_data = data.flatten()
    threshold = np.partition(flat_data, -top_n)[-top_n]
    top_indices = np.where(flat_data >= threshold)[0]
    top_indices_sorted = top_indices[np.argsort(-flat_data[top_indices])]
    top_cells = [np.unravel_index(idx, data.shape) for idx in top_indices_sorted]
    return top_cells

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def difference_heatmap(
    data1,
    data2,
    base="data1",  # "data1", "data2", or "none" to choose the background
    circle_scale=1.0,
    circle_color_positive="blue",
    circle_color_negative="red",
    ax=None,
    gamma=1.5,  # Added gamma parameter for PowerNorm
    cbar=True,
    **kwargs
):
    """
    Plot a heatmap for one matrix and overlay circles
    whose size encodes the difference between data2 and data1.

    Parameters
    ----------
    data1 : ndarray or DataFrame
        First attention matrix.
    data2 : ndarray or DataFrame
        Second attention matrix, must be same shape as data1.
    base : str, optional
        Which matrix to use for the background heatmap: "data1", "data2", or "none".
        If "none", no colored background is drawn; only the circles are shown.
    circle_scale : float, optional
        A scale factor to multiply all circle radii. Adjust to increase or decrease
        the maximum circle size.
    circle_color_positive : str, optional
        Matplotlib color for circles where (data2 - data1) > 0.
    circle_color_negative : str, optional
        Matplotlib color for circles where (data2 - data1) < 0.
    ax : matplotlib Axes, optional
        Axes on which to plot. If None, uses current Axes.
    gamma : float, optional
        Gamma value for PowerNorm (default: 1.5).
    **kwargs
        Additional keyword args passed to the underlying `heatmap` function.
    """
    # 1. Check shapes
    if data1.shape != data2.shape:
        raise ValueError("Both matrices must have the same shape.")
        
    diff = np.array(data2) - np.array(data1)  # ensure ndarray
    
    # 2. Decide background data
    if base == "data1":
        bg_data = data1
    elif base == "data2":
        bg_data = data2
    else:
        # If base == "none", we just pass in zeros so the color is uniform
        bg_data = np.zeros_like(diff)

    # 3. Set up PowerNorm for background coloring
    if base != "none":
        vmin = bg_data.min()
        vmax = bg_data.max()
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        kwargs.setdefault('norm', norm)
        kwargs.setdefault('cmap', 'Blues')  # default colormap
        kwargs.setdefault('vmin', vmin)
        kwargs.setdefault('vmax', vmax)

    else:
        # For "none" base, set uniform white background with black borders
        kwargs.setdefault("cmap", plt.cm.colors.ListedColormap(['white']))
        kwargs.setdefault("linecolor", 'black')  # Set border color to black
        kwargs.setdefault("linewidths", 0.7)
        kwargs.setdefault("cbar", False)
        
    # 4. Draw the base heatmap
    if ax is None:
        ax = plt.gca()

    # Use the existing heatmap function with PowerNorm
    ax, plotter = create_tablelens_heatmap(
        bg_data,
        ax=ax,
        **kwargs
    )

    # 5. Overlay circles that show the magnitude of the difference
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers
    
    # We need the absolute maximum difference to normalize circle sizes
    max_abs_diff = np.max(np.abs(diff)) if np.any(diff != 0) else 1e-6
    
    # List to store circle patches
    patches = []
    colors = []
    
    # For each cell, add a circle whose radius is proportional to |diff|
    for i, y in enumerate(row_centers):
        for j, x in enumerate(col_centers):
            val = diff[i, j]
            if val == 0:
                continue  # no circle if there's no difference
            
            # radius is scaled by the absolute difference, relative to the global max
            radius = circle_scale * (abs(val) / max_abs_diff) * 0.5
            circ = Circle(
                (x, y),
                radius=radius
            )
            patches.append(circ)
            
            # Choose color based on sign
            if val > 0:
                colors.append(circle_color_positive)
            else:
                colors.append(circle_color_negative)

    # Create a PatchCollection and add it to the plot
    collection = PatchCollection(patches, facecolor=colors, edgecolor='none', alpha=0.7)
    ax.add_collection(collection)

    # Set axis limits to match the heatmap
    ax_autoscale = False
    if not ax_autoscale:
        ax.set_xlim(plotter.col_positions[0], plotter.col_positions[-1])
        ax.set_ylim(plotter.row_positions[0], plotter.row_positions[-1])
        ax.invert_yaxis()

    return ax


# When you run compare_two_attentions(attn1, attn2, tokens), you'll get:
# A background heatmap showing attn1.
# Circles in each cell whose radius is proportional to |attn2 - attn1|.
# Orange circles where attn2 > attn1, blue circles where attn2 < attn1 (by the default you gave).
def compare_two_attentions(attn1, attn2, tokens, title="Comparison: Matrix2 - Matrix1", base="data1", save_path=None):
    """
    Compares two attention matrices and visualizes their differences in a heatmap.
    
    Parameters:
    - attn1: First attention matrix (baseline)
    - attn2: Second attention matrix to compare against attn1
    - tokens: List of token labels for x/y axes
    - save_path: File path to save the generated heatmap PDF
    - title: Title for the plot (default: "Comparison: Matrix2 - Matrix1")
    - cmap: Matplotlib colormap for the heatmap (default: 'Blues')
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Convert tensors to numpy if needed
    if torch.is_tensor(attn1):
        attn1 = attn1.detach().cpu().numpy()
    if torch.is_tensor(attn2):
        attn2 = attn2.detach().cpu().numpy()

    # Create the difference heatmap
    difference_heatmap(
        attn1,
        attn2,
        base=base,  # draw the background using attn1; circles show how attn2 differs
        x_labels=[bold_special_tokens(token) for token in tokens],
        y_labels=[bold_special_tokens(token) for token in tokens],
        title=title,
        xlabel="Tokens Attended to",
        ylabel="Tokens Attending",
        circle_scale=1.0,            # adjust for bigger or smaller circles
        circle_color_positive="orange", # where attn2 > attn1
        circle_color_negative="blue", # where attn2 < attn1
        ax=ax
    )

    # # Set title and adjust layout
    # ax.set_title(title, fontsize=14)
    # ax.set_xlabel("Tokens", fontsize=12)
    # ax.set_ylabel("Tokens", fontsize=12)

    # # Rotate x-axis labels for better readability
    # for label in ax.get_xticklabels():
    #     label.set_rotation(45)

    plt.tight_layout()
    if save_path is None:
        save_path = "attention_comparison_heatmap.pdf"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Attention comparison heatmap saved to {save_path}")


def check_stability_heatmap(
    matrices,
    x_labels=None,
    y_labels=None,
    title=None,
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    use_std_error=False,      # If True, use SEM = std/sqrt(n); else use raw std
    circle_scale=1.0,         # Base scaling factor for circles
    cmap="Blues",            # Colormap for circle colors
    linecolor="black",        # Grid line color
    linewidths=0.5,          # Grid line width
    save_path="check_stability_heatmap.pdf",
    gamma=1.5               # Added gamma parameter for PowerNorm
):
    """
    Creates a 'circle-heatmap' given n attention matrices of the same shape.
    
    - The color of each circle encodes the mean across the n matrices.
    - The size (radius) of each circle is inversely proportional to the measure of spread
      (e.g. standard deviation or standard error), meaning more stable cells => larger circles.

    Parameters
    ----------
    matrices : list or np.ndarray
        A list of attention matrices (each shape = (R, C)) or a single 3D array of shape (n, R, C).
    x_labels : list of str, optional
        Labels for columns (x-axis).
    y_labels : list of str, optional
        Labels for rows (y-axis).
    title : str, optional
        Title of the plot.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, a new figure+axes is created.
    use_std_error : bool
        Whether to use standard error of the mean (SEM) instead of standard deviation.
    circle_scale : float
        Overall scale for circle sizes. Increase if circles are too small, or decrease if too large.
    cmap : str or matplotlib.colors.Colormap
        Colormap used to color circles by the mean value. Defaults to 'Blues'.
    linecolor : str
        Color of grid lines in the underlying table-lens heatmap.
    linewidths : float
        Width of grid lines in the underlying table-lens heatmap.
    save_path : str, optional
        If provided, saves the figure to this path (PDF, PNG, etc.). 
    gamma : float, optional
        Gamma value for PowerNorm used in circle coloring (default: 1.5).
    **kwargs : dict
        Additional arguments passed down to `create_tablelens_heatmap` for fine-tuning.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    # Convert input to numpy array
    matrices = np.array(matrices)  # shape: (n, R, C)
    if matrices.ndim != 3:
        raise ValueError(
            "Expected `matrices` to be a list or 3D array of shape (n, R, C). "
            f"Got shape: {matrices.shape}"
        )

    n, R, C = matrices.shape

    # Compute mean and spread
    mean_vals = np.mean(matrices, axis=0)  # shape (R, C)
    std_vals = np.std(matrices, axis=0)    # shape (R, C)

    if use_std_error:
        # Standard error of the mean (SEM) = std / sqrt(n)
        error_vals = std_vals / np.sqrt(n)
    else:
        # Use the plain standard deviation
        error_vals = std_vals

    # Create blank background
    blank_data = np.zeros_like(mean_vals)

    # Prepare default labels if None
    if x_labels is None:
        x_labels = [f"X{i}" for i in range(C)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(R)]

    # Create or use existing Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Create the base heatmap with white background and black borders
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=blank_data,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=plt.cm.colors.ListedColormap(['white']),
        linecolor=linecolor,
        linewidths=linewidths,
        cbar=False
    )

    # Set up PowerNorm for circle colors
    min_mean, max_mean = mean_vals.min(), mean_vals.max()
    # To avoid zero range, handle the degenerate case:
    if np.isclose(min_mean, max_mean):
        max_mean = min_mean + 1e-9

    norm = PowerNorm(gamma=gamma, vmin=min_mean, vmax=max_mean)

    # Get cell centers from plotter
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    # Calculate circle sizes
    nonzero_errors = error_vals[error_vals > 0]
    min_err = np.min(nonzero_errors) if len(nonzero_errors) > 0 else 1.0

    patches = []
    colors = []

    # For each cell, add a circle
    for i in range(R):
        for j in range(C):
            mval = mean_vals[i, j]
            err = error_vals[i, j]

            # Determine radius (max 0.5 to fit within cell)
            if err < 1e-12:
                radius = circle_scale * 0.5  # Max size that fits in cell
            else:
                radius = min(circle_scale * 0.5 * (min_err / err), 0.5)

            # Create circle at cell center
            circ = Circle(
                (col_centers[j], row_centers[i]),
                radius=radius
            )
            patches.append(circ)
            colors.append(plt.get_cmap(cmap)(norm(mval)))

    # Add circles to plot
    collection = PatchCollection(patches, facecolor=colors, edgecolor='none', alpha=0.7)
    ax.add_collection(collection)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Add the colorbar using the ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)

    # Remove the black border around the colorbar
    cbar.outline.set_visible(False)

    # Adjust colorbar ticks
    num_ticks = 7  # Adjust the number of ticks as needed
    tick_values = np.linspace(min_mean, max_mean, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])
    cbar.set_label("Mean Attention Score", rotation=90)

    # Format labels
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0, ha='right')

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Check Stability heatmap saved to {save_path}")
    else:
        plt.tight_layout()

    return ax

def compare_two_attentions_with_circles(attn1, attn2, tokens, title="Comparison with Circles", save_path=None, 
                                      circle_scale=1.0, gamma=1.5, cmap="Blues"):
    """
    Compares two attention matrices by showing the first matrix as background colors
    and the second matrix as circles with varying sizes based on their differences.
    
    Parameters:
    - attn1: First attention matrix (used for background colors)
    - attn2: Second attention matrix (used for circle colors)
    - tokens: List of token labels for x/y axes
    - title: Title for the plot
    - save_path: File path to save the generated heatmap PDF
    - circle_scale: Scale factor for circle sizes (default: 1.0)
    - gamma: Gamma value for the power normalization of the colormap (default: 1.5)
    - cmap: Colormap to use (default: 'Blues')
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Convert tensors to numpy if needed
    if torch.is_tensor(attn1):
        attn1 = attn1.detach().cpu().numpy()
    if torch.is_tensor(attn2):
        attn2 = attn2.detach().cpu().numpy()

    # Prepare data and normalization
    data1 = attn1
    data2 = attn2
    diff = np.abs(data2 - data1)
    
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # Create the base heatmap using attn1
    ax, plotter = create_tablelens_heatmap(
        data1,
        x_labels=[bold_special_tokens(token) for token in tokens],
        y_labels=[bold_special_tokens(token) for token in tokens],
        title=title,
        xlabel="Tokens Attended to",
        ylabel="Tokens Attending",
        ax=ax,
        cmap=cmap,
        norm=norm,
        gamma=gamma,
        vmax=vmax,
        vmin=vmin
    )

    # Get cell centers from plotter
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    # Calculate circle sizes based on differences
    max_diff = np.max(diff) if np.any(diff != 0) else 1e-6
    
    patches = []
    colors = []

    # For each cell, add a circle
    for i in range(len(row_centers)):
        for j in range(len(col_centers)):
            # Determine radius (max 0.5 to fit within cell)
            radius = min(circle_scale * 0.5 * (diff[i, j] / max_diff), 0.5)
            
            if radius > 0:  # Only add circles where there's a difference
                circ = Circle(
                    (col_centers[j], row_centers[i]),
                    radius=radius
                )
                patches.append(circ)
                colors.append(plt.get_cmap(cmap)(norm(data2[i, j])))

    # Add circles to plot
    collection = PatchCollection(patches, facecolor=colors, edgecolor='none', alpha=0.7)
    ax.add_collection(collection)

    if save_path is None:
        save_path = "attention_comparison_circles.pdf"

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Attention comparison heatmap with circles saved to {save_path}")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ensure you have your create_tablelens_heatmap imported or defined as in your code
# from .visualization import create_tablelens_heatmap  # or adjust import path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Make sure you have your create_tablelens_heatmap function available
# from .visualization import create_tablelens_heatmap  # or adjust the import path

def check_stability_heatmap_new(
    matrices,
    x_labels=None,
    y_labels=None,
    title=None,
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    use_std_error=False,   # If True, use SEM = std/sqrt(n); else use raw std
    circle_scale=1.0,      # Base scaling factor for circles
    cmap="Blues",          # Colormap for *square cells* (based on the mean)
    linecolor="white",     # Grid line color
    linewidths=1.0,        # Grid line width
    save_path="check_stability_heatmap.pdf",
    gamma=1.5
):
    """
    Plots an n-run stability heatmap:
      - The *background squares* are colored by the mean attention score across n matrices
        (darker = higher mean, using 'Blues').
      - A *hollow orange circle* is drawn in each cell, and its radius is
        now directly proportional to the measure of spread (std or std_error):
        more uncertainty => bigger circle.

    Parameters
    ----------
    matrices : list or np.ndarray
        A list of attention matrices (each shape = (n, R, C)) or a single 3D array
        of shape (n, R, C).
    x_labels : list of str, optional
        Column (x-axis) labels.
    y_labels : list of str, optional
        Row (y-axis) labels.
    title : str, optional
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, a new figure+axes is created.
    use_std_error : bool
        Whether to use the standard error of the mean (std/sqrt(n)) instead of raw std.
    circle_scale : float
        A factor controlling the size of the circles. Increase if circles are too small.
    cmap : str or Colormap
        The colormap for the *background squares*. Default is 'Blues'.
    linecolor : str
        Color of grid lines in the underlying table-lens heatmap.
    linewidths : float
        Width of grid lines in the underlying table-lens heatmap.
    save_path : str, optional
        If provided, the plot is saved to this path (PDF, PNG, etc.).
    gamma : float, optional
        Gamma value for PowerNorm used in coloring the background squares only.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    # 1) Convert input to numpy array (n, R, C)
    matrices = np.array(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            "Expected `matrices` to be a list or 3D array of shape (n, R, C). "
            f"Got shape: {matrices.shape}"
        )

    n, R, C = matrices.shape

    # 2) Compute mean and measure of spread
    mean_vals = np.mean(matrices, axis=0)  # shape (R, C)
    std_vals  = np.std(matrices, axis=0)   # shape (R, C)
    if use_std_error:
        # Standard error of the mean (SEM)
        error_vals = std_vals / np.sqrt(n)
    else:
        error_vals = std_vals

    # 3) Provide default labels if not given
    if x_labels is None:
        x_labels = [f"X{i}" for i in range(C)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(R)]

    # 4) Create or use existing Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # 5) Use create_tablelens_heatmap to plot squares colored by mean_vals
    vmin, vmax = mean_vals.min(), mean_vals.max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    # Apply a PowerNorm with gamma if desired
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # Plot the background squares with create_tablelens_heatmap
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=mean_vals,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=cmap,
        cbar=True,
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )

    # 6) Overlay the hollow orange circles for each cell
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    # We'll find the maximum error to normalize circle sizes
    max_err = error_vals.max()
    if max_err < 1e-12:
        max_err = 1.0  # fallback if everything is zero

    circle_patches = []
    for i in range(R):
        for j in range(C):
            err = error_vals[i, j]
            # Circle size grows with bigger error
            # radius up to 0.5 * circle_scale if err == max_err
            radius = (err / max_err) * 0.5 * circle_scale

            circ = Circle((col_centers[j], row_centers[i]), radius=radius)
            circle_patches.append(circ)

    # Make them hollow orange circles (facecolor='none', edgecolor='orange')
    circle_collection = PatchCollection(
        circle_patches,
        facecolor='none',      # hollow
        edgecolor='orange',    # orange ring
        linewidth=1.5,
        alpha=1.0
    )
    ax.add_collection(circle_collection)

    # 7) Adjust label rotations for clarity
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0, ha='right')

    # 8) Save or just show
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Check Stability heatmap saved to {save_path}")
    else:
        plt.tight_layout()

    return ax



def target_ring_heatmap(
    matrices,
    x_labels=None,
    y_labels=None,
    title=None,
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    cmap="Blues",            # Colormap for background and rings
    gamma=1.5,               # PowerNorm gamma
    linecolor="white",       # Grid line color
    linewidths=1.0,          # Grid line width
    ring_radius=0.45,        # Fraction of half-cell for outer ring radius
    save_path="check_stability_heatmap_with_target_rings.pdf",
    show_background=True     # <== New parameter
):
    """
    Creates a 'target ring' or 'bullseye' heatmap.

    When `show_background=True`:
      1) Each square cell's background is determined by the mean of all
         input matrices at that position.
      2) Each cell has n concentric rings (bullseyes), from inner to outer,
         where each ring's color is one of the n input matrices' values.

    When `show_background=False`:
      - All squares have a uniform (white) background, but the rings remain
        and share the same color scale + colorbar.

    Both modes:
      - The squares (when shown) and rings share one global PowerNorm color scale
        (with one colorbar).
      - The i-th matrix is used for the i-th ring in each cell
        (innermost ring -> matrix 0, outermost -> matrix n-1).
    
    Parameters
    ----------
    matrices : list or np.ndarray
        List of attention matrices or single 3D array of shape (n, R, C).
    x_labels : list of str, optional
        Column labels.
    y_labels : list of str, optional
        Row labels.
    title : str, optional
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    cmap : str or matplotlib.colors.Colormap
        Colormap for squares and rings. Defaults to 'Blues'.
    gamma : float
        Gamma value for the PowerNorm color scaling.
    linecolor : str
        Color of grid lines in the table-lens heatmap.
    linewidths : float
        Width of grid lines in the table-lens heatmap.
    ring_radius : float
        Maximum radius for the outermost ring (fraction of half the cell).
    save_path : str, optional
        If provided, saves the figure to this path (e.g. "my_plot.pdf").
    show_background : bool
        If True (default), color each cell by the mean of the n matrices.
        If False, use a uniform white background instead.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the drawn figure.
    """
    # 1) Convert input to a numpy array of shape (n, R, C)
    matrices = np.array(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            "Expected `matrices` to be a list or 3D np.array of shape (n, R, C). "
            f"Got shape: {matrices.shape}"
        )
    n, R, C = matrices.shape

    # 2) Compute the mean across n matrices for background (if desired)
    mean_vals = np.mean(matrices, axis=0)  # shape (R, C)

    # 3) Compute global min/max across *all* values => 1 color scale for squares & rings
    all_values = matrices.flatten()
    min_val, max_val = all_values.min(), all_values.max()
    if np.isclose(min_val, max_val):
        max_val = min_val + 1e-9

    # 4) Default labels if none provided
    if x_labels is None:
        x_labels = [f"X{j}" for j in range(C)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(R)]

    # 5) Create or use existing Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # 6) Define a PowerNorm for the color scale (for both squares & rings)
    norm = PowerNorm(gamma=gamma, vmin=min_val, vmax=max_val)

    # 7) Decide what data to pass to `create_tablelens_heatmap`
    #    If show_background=False, use uniform white squares. 
    if show_background:
        background_data = mean_vals  # color by mean
        background_cmap = cmap
    else:
        background_data = np.full_like(mean_vals, np.nan)
        # single-color colormap => uniform squares (white)
        # background_cmap = plt.cm.colors.ListedColormap(["white"])
        background_cmap = cmap

    # 8) Draw squares with create_tablelens_heatmap
    #    We'll still pass the same vmin, vmax, and norm so the rings share the colorbar
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=background_data,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=background_cmap,
        cbar=True,            # single colorbar for squares & rings
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=min_val,
        vmax=max_val,
        norm=norm
    )

    # 9) Overlay n concentric rings in each cell

    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    # Loop over each cell, drawing n rings (one per matrix)
    for row_i in range(R):
        for col_j in range(C):
            for ring_idx in range(n):
                val = matrices[ring_idx, row_i, col_j]
                color = plt.get_cmap(cmap)(norm(val))

                # ring i from radius_in to radius_out
                radius_in  = ring_radius * ( ring_idx     / n )
                radius_out = ring_radius * ((ring_idx+1.0)/ n )

                wedge = Wedge(
                    center=(col_centers[col_j], row_centers[row_i]),
                    r=radius_out,
                    theta1=0,
                    theta2=360,
                    width=(radius_out - radius_in),  # annulus thickness
                    facecolor=color,
                    edgecolor='none'
                )
                ax.add_patch(wedge)

    # 10) Adjust label rotation
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0, ha='right')

    # 11) Save or return
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Target ring heatmap saved to {save_path}")
    else:
        plt.tight_layout()

    return ax



def check_stability_heatmap_with_gradient_color(
    matrices,
    x_labels=None,
    y_labels=None,
    title="Check Stability Heatmap with Gradient Circles",
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    use_std_error=True,   # If True, use SEM = std/sqrt(n); else raw std
    circle_scale=1.0,      # Factor controlling how large the circle can get
    cmap="Blues",          # Colormap for background squares
    linecolor="white",     # Grid line color
    linewidths=1.0,        # Grid line width
    save_path="check_stability_heatmap_with_gradient_color.pdf",
    gamma=1.5,
    radial_resolution=100   # Resolution of the radial gradient image
):
    """
    Plots an n-run stability heatmap:

      1) Background squares are colored by the mean attention score across n matrices
         (darker = higher mean, using 'Blues').
      2) Each cell has a circle whose radius is proportional to the "confidence interval"
         (e.g. std or SEM). A bigger interval => a bigger circle.
      3) The circle is filled with a *radial gradient*:
         - The gradient goes from the color corresponding to the cell's 'lower bound'
           (mean - error) in the center,
           to the color of the 'upper bound' (mean + error) at the edge.
      4) Everything (squares + gradient circles) uses the same global PowerNorm scale
         and shares the same colorbar.

    Parameters
    ----------
    matrices : list or np.ndarray
        A list of (R, C) arrays or a single 3D array of shape (n, R, C).
    x_labels : list of str, optional
        Column (x-axis) labels.
    y_labels : list of str, optional
        Row (y-axis) labels.
    title : str, optional
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, a new figure + axes is created.
    use_std_error : bool
        Whether to use standard error of the mean (std/sqrt(n)) instead of raw std.
    circle_scale : float
        A factor controlling the size of circles. Increase if circles are too small.
    cmap : str or matplotlib.colors.Colormap
        Colormap for both squares & gradient circles. Default is 'Blues'.
    linecolor : str
        Color of grid lines in the table-lens heatmap.
    linewidths : float
        Width of grid lines in the table-lens heatmap.
    save_path : str
        If provided, the plot is saved to this path (PDF/PNG, etc.) and the figure is closed.
    gamma : float
        Gamma value for PowerNorm (affects both squares and gradient).
    radial_resolution : int
        Resolution used for the radial gradient images (NxN).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    """
    # 1) Convert input to np.ndarray of shape (n, R, C)
    matrices = np.array(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            f"Expected `matrices` to be a list or 3D array of shape (n, R, C). "
            f"Got shape: {matrices.shape}"
        )

    n, R, C = matrices.shape

    # 2) Compute the mean and the measure of spread (std or SEM)
    mean_vals = np.mean(matrices, axis=0)     # shape (R, C)
    std_vals  = np.std(matrices, axis=0)      # shape (R, C)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n)    # SEM
    else:
        error_vals = std_vals

    # 3) If no x_labels or y_labels given, provide default
    if x_labels is None:
        x_labels = [f"X{j}" for j in range(C)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(R)]

    # 4) Create or use existing Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # 5) We want a single colormap scale for everything.
    #    Find min/max across possible "lower" and "upper" bounds as well as means.
    #    lower bound = (mean_vals - error_vals), upper bound = (mean_vals + error_vals).
    lower_all = (mean_vals - error_vals).min()
    upper_all = (mean_vals + error_vals).max()
    vmin = min(lower_all, mean_vals.min())
    vmax = max(upper_all, mean_vals.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    
    # Create a PowerNorm
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    # 6) Plot the background squares using the mean
    #    This also adds a single colorbar that squares + circles will share.
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=mean_vals,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=cmap,
        cbar=True,          # share colorbar with circles
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )

    # 7) We'll render a radial gradient for each cell.
    #    The radius is proportional to error, and color goes from (mean-err) to (mean+err).
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    max_err = error_vals.max()
    if max_err < 1e-12:
        max_err = 1.0  # fallback if everything is zero

    # Helper to make a radial gradient NxN image
    def make_radial_gradient_image(inner_rgba, outer_rgba, N=100):
        """
        Creates an NxN RGBA array with a radial gradient.
          - center (N/2, N/2) has color=inner_rgba
          - outer edge radius ~ (N/2) has color=outer_rgba
        """
        # Ensure these are NumPy float arrays, not just Python tuples
        inner_rgba = np.array(inner_rgba, dtype=float)
        outer_rgba = np.array(outer_rgba, dtype=float)

        gradient = np.zeros((N, N, 4), dtype=np.float32)
        center = (N - 1) / 2.0
        radius = center

        for r in range(N):
            for c in range(N):
                dist = np.sqrt((r - center)**2 + (c - center)**2)
                t = min(dist / radius, 1.0)  # clamp to 1.0
                # linear interpolation in RGBA
                gradient[r, c, :] = (1 - t) * inner_rgba + t * outer_rgba

        return gradient

    # For each cell, create an image + clip it to a circle
    for i in range(R):
        for j in range(C):
            err = error_vals[i, j]
            # If there's no error, skip drawing any circle
            if err < 1e-12:
                continue

            # Circle radius in data coordinates
            # bigger error => bigger circle up to 0.5 * circle_scale
            radius = (err / max_err) * 0.5 * circle_scale

            # Find the lower/upper values for the gradient
            val_lower = mean_vals[i, j] - err
            val_upper = mean_vals[i, j] + err

            # Clamp to [vmin, vmax]
            val_lower = max(val_lower, vmin)
            val_lower = min(val_lower, vmax)
            val_upper = max(val_upper, vmin)
            val_upper = min(val_upper, vmax)

            # Convert to RGBA
            cmap_obj = plt.get_cmap(cmap)
            inner_rgba = np.array(cmap_obj(norm(val_lower)), dtype=float)
            outer_rgba = np.array(cmap_obj(norm(val_upper)), dtype=float)

            # Build a radial gradient image NxN
            gradient_img = make_radial_gradient_image(
                inner_rgba=inner_rgba,
                outer_rgba=outer_rgba,
                N=radial_resolution
            )

            x_center = col_centers[j]
            y_center = row_centers[i]
            x_left   = x_center - radius
            x_right  = x_center + radius
            y_bottom = y_center - radius
            y_top    = y_center + radius

            # Render the image in that region
            im = ax.imshow(
                gradient_img,
                extent=[x_left, x_right, y_bottom, y_top],
                origin='lower',
                zorder=3  # above the squares
            )
            # Then clip it to a circle so it's only visible inside
            circ = Circle((x_center, y_center), radius=radius, transform=ax.transData)
            im.set_clip_path(circ)

    # 8) Tidy up label rotations
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0, ha='right')

    # 9) Save or show
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Check Stability heatmap saved to {save_path}")
    else:
        plt.tight_layout()

    return ax




def half_pie_heatmap_original(
    matrices,
    x_labels=None,
    y_labels=None,
    title=None,
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    cmap="Blues",            
    gamma=1.5,               
    linecolor="white",       
    linewidths=1.0,          
    ring_radius=0.45,        
    save_path="check_stability_heatmap_with_pie_chart.pdf",
    show_background=True,
    use_std_error=False       # New parameter for using std error vs. std
):
    """
    Creates a 'half-pie' heatmap with optional background squares.

    Features:
      1) Each square cell can show a background color determined by the mean of all
         input matrices at that position (if show_background=True). Otherwise, the
         background is uniform white.
      2) We measure a confidence interval at each cell (standard deviation or SEM)
         and draw a light-grey circle whose radius is proportional to that interval.
      3) Inside that circle, we draw a 'half-pie chart' spanning 45° to 225°,
         evenly split into n slices. Each slice's color is mapped from the cell's
         value in one of the n input matrices, using the same global PowerNorm scale
         as the background squares.

    Parameters
    ----------
    matrices : list or np.ndarray
        A list of (R, C) matrices or a single 3D array of shape (n, R, C).
        The i-th matrix's value at (row,col) is shown in the i-th slice
        of the half-pie for that cell.
    x_labels : list of str, optional
        Column (x-axis) labels.
    y_labels : list of str, optional
        Row (y-axis) labels.
    title : str, optional
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure + axes is created.
    cmap : str or matplotlib.colors.Colormap
        Colormap for squares + half-pies. Defaults to 'Blues'.
    gamma : float
        Gamma value for the PowerNorm color scaling.
    linecolor : str
        Color of grid lines in the table-lens heatmap.
    linewidths : float
        Width of grid lines in the table-lens heatmap.
    ring_radius : float
        Maximum radius for the background circle (fraction of half the cell).
    save_path : str
        If provided, saves the figure to this path (PDF/PNG, etc.)
        and then closes the figure.
    show_background : bool
        If True, each cell's square is colored by the mean of the n matrices;
        if False, squares are drawn white.
    use_std_error : bool
        If True, measure the confidence interval as std/sqrt(n) (SEM).
        Otherwise, use raw std.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    # 1) Convert input to np.array (n, R, C)
    matrices = np.array(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            "Expected `matrices` to be a list or 3D np.array of shape (n, R, C). "
            f"Got {matrices.shape}"
        )

    n, R, C = matrices.shape

    # 2) Compute the mean for potential background, plus the measure of spread
    mean_vals = np.mean(matrices, axis=0)  # (R, C)
    std_vals  = np.std(matrices, axis=0)   # (R, C)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n) # SEM
    else:
        error_vals = std_vals

    # 3) Global min/max across *all* values for the color scale
    all_values = matrices.flatten()
    vmin, vmax = all_values.min(), all_values.max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    # 4) Provide default labels if needed
    if x_labels is None:
        x_labels = [f"X{j}" for j in range(C)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(R)]

    # 5) Create or use existing Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # 6) Define a PowerNorm for everything
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # 7) Decide background squares data
    if show_background:
        background_data = mean_vals
        background_cmap = cmap
    else:
        background_data = np.zeros_like(mean_vals)
        background_cmap = plt.cm.colors.ListedColormap(["white"])

    # 8) Draw squares with create_tablelens_heatmap
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=background_data,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=background_cmap,
        cbar=True,  # single colorbar shared by squares + half-pies
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )

    # 9) We will overlay a half-pie chart for each cell, plus a grey circle
    #    whose size is proportional to error. The largest error has circle_radius=ring_radius
    max_err = error_vals.max() if np.any(error_vals) else 1e-9

    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    # half-pie angles
    start_angle = -45
    total_span = 180  
    slice_angle = total_span / n

    # For each cell, draw:
    #   1) Light grey circle sized by error
    #   2) n wedges from 45->225, each wedge's color from that matrix's value
    for i in range(R):
        for j in range(C):
            err_val = error_vals[i, j]
            if err_val < 1e-12:
                # No circle + half-pie if error is near zero
                continue

            # The radius is a fraction of ring_radius
            frac = (err_val / max_err)
            circle_r = frac * ring_radius

            center_x = col_centers[j]
            center_y = row_centers[i]

            # 1) Draw the light grey background circle
            grey_circle = Circle(
                (center_x, center_y),
                radius=circle_r,
                facecolor="lightgrey",
                edgecolor="none",
                alpha=0.6
            )
            ax.add_patch(grey_circle)

            # 2) Draw the half-pie slices from 45° -> 225°
            #    evenly splitting that 180° across n slices
            for slice_i in range(n):
                val = matrices[slice_i, i, j]
                # get color from the global color scale
                wedge_color = plt.get_cmap(cmap)(norm(val))

                angle_1 = start_angle + slice_i * slice_angle
                angle_2 = start_angle + (slice_i + 1) * slice_angle

                wedge_patch = Wedge(
                    center=(center_x, center_y),
                    r=circle_r,
                    theta1=angle_1,
                    theta2=angle_2,
                    facecolor=wedge_color,
                    edgecolor="none"
                )
                ax.add_patch(wedge_patch)

    # 10) Set axis tick labels
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0, ha='right')

    # 11) Save or return
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Half-pie heatmap saved to {save_path}")
    else:
        plt.tight_layout()
 
    return ax





def half_pie_heatmap(
    matrices,
    x_labels=None,
    y_labels=None,
    title=None,
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    cmap="Blues",
    gamma=1.5,
    linecolor="white",
    linewidths=1.0,
    ring_radius=0.45, 
    save_path="check_stability_heatmap_half_pie.pdf",
    show_background=True,
    use_std_error=False
):
    """
    Creates a heatmap where each cell can have:
      1) (Optionally) a background color determined by the mean of the n matrices.
      2) A light-gray circle whose size is proportional to the local confidence interval
         (std or std_error).
      3) A fixed-size half-pie chart (arc from 45° to 225°) drawn on top of the circle,
         split evenly into n wedges. Each wedge is colored by that cell's value from one
         of the n matrices.
    
    Parameters
    ----------
    matrices : list or np.ndarray
        A list of (R, C) matrices or a single 3D array of shape (n, R, C).
        The i-th matrix's value at (row,col) is visualized in the i-th wedge
        of the half-pie for that cell.
    x_labels : list of str, optional
        Column labels.
    y_labels : list of str, optional
        Row labels.
    title : str, optional
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure + axes is created.
    cmap : str or matplotlib.colors.Colormap
        Colormap for squares + half-pies. Defaults to 'Blues'.
    gamma : float
        Gamma value for the PowerNorm color scaling.
    linecolor : str
        Color of grid lines in the table-lens heatmap.
    linewidths : float
        Width of grid lines in the table-lens heatmap.
    ring_radius : float
        Radius for the half-pie chart in each cell (fraction of half the cell).
    save_path : str
        If provided, saves the figure to this path (PDF/PNG, etc.)
        and then closes the figure.
    show_background : bool
        If True, each cell's square is colored by the mean of the n matrices.
        If False, squares are drawn white.
    use_std_error : bool
        If True, measure the confidence interval as std / sqrt(n).
        Otherwise, use raw std.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    # 1) Convert input to np.array (n, R, C)
    matrices = np.array(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            "Expected `matrices` to be a list or 3D np.array of shape (n, R, C). "
            f"Got shape: {matrices.shape}"
        )

    n, R, C = matrices.shape

    # 2) Compute means for optional background, and measure of spread
    mean_vals = np.mean(matrices, axis=0)  # (R, C)
    std_vals  = np.std(matrices, axis=0)   # (R, C)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n)  # SEM
    else:
        error_vals = std_vals

    # 3) Global min/max for color scale
    all_values = matrices.flatten()
    vmin, vmax = all_values.min(), all_values.max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    # 4) Provide default labels if needed
    if x_labels is None:
        x_labels = [f"X{j}" for j in range(C)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(R)]

    # 5) Create or use existing Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # 6) Define a PowerNorm for everything
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # 7) Decide background squares data
    if show_background:
        background_data = mean_vals
        background_cmap = cmap
    else:
        background_data = np.full_like(mean_vals, np.nan)
        # background_cmap = plt.cm.colors.ListedColormap(["white"])
        background_cmap = cmap

    # 8) Draw squares with create_tablelens_heatmap (both colorbar & lines)
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=background_data,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=background_cmap,
        cbar=True,  # single colorbar shared by squares + half-pies
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )

    # 9) We'll overlay for each cell:
    #    (a) Light-grey circle (size ~ error),
    #    (b) A half-pie chart from 45° to 225°, each slice colored by one matrix's value,
    #        always radius=ring_radius (the same for all cells).
    max_err = error_vals.max() if np.any(error_vals) else 1e-9

    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    start_angle = -45
    total_span = 180  # so end_angle = 225
    slice_angle = total_span / n

    for i in range(R):
        for j in range(C):
            err_val = error_vals[i, j]

            # 9a) Light-grey circle behind the half-pie
            #     radius is proportional to err_val; max circle = ring_radius
            circle_frac = 0 if max_err < 1e-12 else (err_val / max_err)
            circle_radius = circle_frac * ring_radius

            cx = col_centers[j]
            cy = row_centers[i]

            grey_circle = Circle(
                (cx, cy),
                radius=circle_radius,
                facecolor="#D55E00",  # Changed from "lightgrey" to "#D55E00"
                edgecolor="none",
                alpha=0.6,
                zorder=2
            )
            ax.add_patch(grey_circle)

            # 9b) Fixed-size half-pie chart on top, radius=ring_radius always
            #     each wedge covers slice_angle degrees
            for slice_i in range(n):
                val = matrices[slice_i, i, j]
                wedge_color = plt.get_cmap(cmap)(norm(val))

                angle_1 = start_angle + slice_i * slice_angle
                angle_2 = start_angle + (slice_i + 1) * slice_angle

                wedge_patch = Wedge(
                    center=(cx, cy),
                    r=ring_radius,
                    theta1=angle_1,
                    theta2=angle_2,
                    facecolor=wedge_color,
                    edgecolor="none",
                    zorder=3  # above grey circle
                )
                ax.add_patch(wedge_patch)

    # 10) Set axis tick labels
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0, ha='right')

    # 11) Save or return
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Half-pie heatmap saved to {save_path}")
    else:
        plt.tight_layout()

    return ax

def adaptive_attention_evolution_heatmap(
    matrices,
    tokens=None,
    x_labels=None,
    y_labels=None,
    title="Attention Evolution",
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    cmap="Blues",
    gamma=1.5,
    linecolor="white",
    linewidths=1.0,
    ring_radius=0.45,
    save_path=None,
    show_background=True,
    use_std_error=False,
    max_epochs_to_show=8,     # Maximum number of epochs to show individually
    max_tokens_to_show=30,    # Maximum number of tokens to show
    epoch_aggregation="bin",  # Options: "bin", "select", "trend"
    token_selection="importance", # Options: "importance", "random", "first_last", "custom"
    custom_token_indices=None,    # Custom token indices to display
    custom_epoch_indices=None,    # Custom epoch indices to display
    region_detection=False,       # Whether to detect and highlight attention regions
    region_threshold=0.8,         # Threshold for region detection
    display_mode="half_pie"       # Options: "half_pie", "line_trend", "heatmap_grid"
):
    """
    Creates an adaptive heatmap for visualizing attention evolution over many epochs and/or with large matrices.
    
    This function addresses two key scaling issues:
    1. Too many epochs → aggregates epochs or selects representative ones
    2. Too many tokens → samples tokens based on importance or user selection
    
    Parameters
    ----------
    matrices : list or np.ndarray
        A list of attention matrices or a single 3D array of shape (n_epochs, n_rows, n_cols)
    tokens : list of str, optional
        The token labels corresponding to the matrix rows/columns
    x_labels : list of str, optional
        Column labels. If None and tokens provided, will use tokens
    y_labels : list of str, optional
        Row labels. If None and tokens provided, will use tokens
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure + axes is created
    cmap : str or matplotlib.colors.Colormap
        Colormap for visualization. Defaults to 'Blues'
    gamma : float
        Gamma value for the PowerNorm color scaling
    linecolor : str
        Color of grid lines in the heatmap
    linewidths : float
        Width of grid lines in the heatmap
    ring_radius : float
        Radius for the visualizations in each cell (as fraction of half-cell size)
    save_path : str, optional
        If provided, saves the figure to this path
    show_background : bool
        If True, colors cell backgrounds by mean attention
    use_std_error : bool
        If True, uses standard error instead of standard deviation
    max_epochs_to_show : int
        Maximum number of epochs to show individually
    max_tokens_to_show : int
        Maximum number of tokens to include in visualization
    epoch_aggregation : str
        How to handle too many epochs: "bin" (group into bins), "select" (pick representative epochs),
        "trend" (show trend indicators)
    token_selection : str
        How to select tokens when there are too many: "importance" (most attended),
        "random", "first_last" (first and last n/2), "custom" (user-specified)
    custom_token_indices : list of int, optional
        When token_selection="custom", these indices are used
    custom_epoch_indices : list of int, optional
        When epoch_aggregation="select" and custom selection desired, these indices are used
    region_detection : bool
        Whether to detect and highlight attention regions
    region_threshold : float
        Threshold for region detection (0.0 to 1.0)
    display_mode : str
        Visualization type: "half_pie" (like original), "line_trend" (sparkline in each cell),
        "heatmap_grid" (small heatmap in each cell)
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    selected_indices : list of int
        The indices of tokens that were selected for visualization
    epoch_info : dict
        Information about how epochs were processed
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    from matplotlib.patches import Wedge, Rectangle, Circle
    from matplotlib.colors import PowerNorm
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.cm as cm
    import re
    
    # 1) Convert input to np.array (n_epochs, n_rows, n_cols)
    matrices = np.array(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            f"Expected matrices to be a list or 3D array of shape (n_epochs, n_rows, n_cols). "
            f"Got shape: {matrices.shape}"
        )

    n_epochs, n_rows, n_cols = matrices.shape
    
    # Handle token labels
    if tokens is not None:
        if len(tokens) != n_rows or len(tokens) != n_cols:
            raise ValueError(f"Token list length ({len(tokens)}) doesn't match matrix dimensions ({n_rows}x{n_cols})")
        if x_labels is None:
            x_labels = tokens
        if y_labels is None:
            y_labels = tokens
    
    if x_labels is None:
        x_labels = [f"X{j}" for j in range(n_cols)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(n_rows)]
    
    # 2) Handle token selection if we have too many tokens
    selected_indices = None
    epoch_indices = None
    
    if n_rows > max_tokens_to_show or n_cols > max_tokens_to_show:
        print(f"Matrix is too large ({n_rows}x{n_cols}). Selecting {max_tokens_to_show} tokens...")
        selected_indices = select_important_tokens(
            matrices, tokens, max_tokens_to_show, 
            method=token_selection, 
            custom_indices=custom_token_indices
        )
        
        # Filter matrices and labels
        matrices = matrices[:, selected_indices, :][:, :, selected_indices]
        if tokens is not None:
            tokens = [tokens[i] for i in selected_indices]
        x_labels = [x_labels[i] for i in selected_indices]
        y_labels = [y_labels[i] for i in selected_indices]
        n_rows, n_cols = len(selected_indices), len(selected_indices)
    else:
        selected_indices = list(range(n_rows))
    
    # 3) Handle epoch aggregation if we have too many epochs
    epoch_info = {"method": epoch_aggregation, "original_count": n_epochs}
    
    if n_epochs > max_epochs_to_show:
        print(f"Too many epochs ({n_epochs}). {epoch_aggregation.capitalize()}ing epochs...")
        
        if epoch_aggregation == "bin":
            # Group epochs into bins
            n_bins = max_epochs_to_show
            bin_size = n_epochs // n_bins
            binned_matrices = []
            bin_labels = []
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size if i < n_bins-1 else n_epochs
                bin_matrix = np.mean(matrices[start_idx:end_idx], axis=0)
                binned_matrices.append(bin_matrix)
                bin_labels.append(f"{start_idx}-{end_idx-1}")
            
            matrices = np.array(binned_matrices)
            n_epochs = len(binned_matrices)
            epoch_info["bins"] = bin_labels
            epoch_info["bin_size"] = bin_size
            
        elif epoch_aggregation == "select":
            # Select representative epochs
            if custom_epoch_indices is not None:
                epoch_indices = custom_epoch_indices[:max_epochs_to_show]
    else:
        # Evenly spaced epochs
        step = max(1, n_epochs // max_epochs_to_show)
        epoch_indices = list(range(0, n_epochs, step))[:max_epochs_to_show]
        # Always include the first and last epoch
        if n_epochs-1 not in epoch_indices:
            epoch_indices[-1] = n_epochs-1
            
            matrices = matrices[epoch_indices]
            n_epochs = len(epoch_indices)
            epoch_info["selected_indices"] = epoch_indices
            
        elif epoch_aggregation == "trend":
            # We'll compute min, max, and mean but still use original matrices
            # The trend indicators will be shown in the visualization
            trend_data = {
                "min": np.min(matrices, axis=0),
                "max": np.max(matrices, axis=0),
                "mean": np.mean(matrices, axis=0),
                "first": matrices[0],
                "last": matrices[-1]
            }
            epoch_info["trend_data"] = trend_data
        else:
            epoch_indices = list(range(n_epochs))
    
    # 4) Compute means and variability measures
    mean_vals = np.mean(matrices, axis=0)
    std_vals = np.std(matrices, axis=0)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n_epochs)
    else:
        error_vals = std_vals
    
    # 5) Compute global min/max for color scale
    all_values = matrices.flatten()
    vmin, vmax = all_values.min(), all_values.max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    
    # 6) Create or use existing axes
    if ax is None:
        fig_scale = max(1.0, n_rows / 10)  # Scale figure size with token count
        fig, ax = plt.subplots(figsize=(8 * fig_scale, 8 * fig_scale))
    else:
        fig = ax.figure
    
    # 7) Create PowerNorm for color scaling
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    # 8) Decide background data
    if show_background:
        background_data = mean_vals
        background_cmap = cmap
    else:
        background_data = np.full_like(mean_vals, np.nan)
        background_cmap = cmap
    
    # 9) Draw the base heatmap
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=background_data,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=background_cmap,
        cbar=True,
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )
    
    # 10) Add visualization based on display_mode
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers
    
    # Scale for error circle size
    max_err = error_vals.max() if np.any(error_vals) else 1e-9
    
    if display_mode == "half_pie":
        # Similar to original half_pie_heatmap but with better epoch handling
        start_angle = -45
        total_span = 180
        
        # If we're using trend mode, we'll use 5 segments: min, first, mean, last, max
        if epoch_aggregation == "trend":
            n_segments = 5
            slice_angle = total_span / n_segments
            
            for i in range(n_rows):
                for j in range(n_cols):
                    # Error circle
                    err_val = error_vals[i, j]
                    circle_frac = 0 if max_err < 1e-12 else (err_val / max_err)
                    circle_radius = circle_frac * ring_radius
                    
                    cx, cy = col_centers[j], row_centers[i]
                    
                    grey_circle = Circle(
                        (cx, cy),
                        radius=circle_radius,
                        facecolor="#D55E00",
                        edgecolor="none",
                        alpha=0.6,
                        zorder=2
                    )
                    ax.add_patch(grey_circle)
                    
                    # Trend half-pie
                    values = [
                        trend_data["min"][i, j],
                        trend_data["first"][i, j],
                        trend_data["mean"][i, j],
                        trend_data["last"][i, j],
                        trend_data["max"][i, j]
                    ]
                    
                    for slice_i, val in enumerate(values):
                        wedge_color = plt.get_cmap(cmap)(norm(val))
                        angle_1 = start_angle + slice_i * slice_angle
                        angle_2 = start_angle + (slice_i + 1) * slice_angle
                        
                        wedge_patch = Wedge(
                            center=(cx, cy),
                            r=ring_radius,
                            theta1=angle_1,
                            theta2=angle_2,
                            facecolor=wedge_color,
                            edgecolor="none",
                            zorder=3
                        )
                        ax.add_patch(wedge_patch)
        else:
            # Regular half-pie with binned or selected epochs
            slice_angle = total_span / n_epochs
            
            for i in range(n_rows):
                for j in range(n_cols):
                    # Error circle
                    err_val = error_vals[i, j]
                    circle_frac = 0 if max_err < 1e-12 else (err_val / max_err)
                    circle_radius = circle_frac * ring_radius
                    
                    cx, cy = col_centers[j], row_centers[i]
                    
                    grey_circle = Circle(
                        (cx, cy),
                        radius=circle_radius,
                        facecolor="#D55E00",
                        edgecolor="none",
                        alpha=0.6,
                        zorder=2
                    )
                    ax.add_patch(grey_circle)
                    
                    # Half-pie slices
                    for slice_i in range(n_epochs):
                        val = matrices[slice_i, i, j]
                        wedge_color = plt.get_cmap(cmap)(norm(val))
                        
                        angle_1 = start_angle + slice_i * slice_angle
                        angle_2 = start_angle + (slice_i + 1) * slice_angle
                        
                        wedge_patch = Wedge(
                            center=(cx, cy),
                            r=ring_radius,
                            theta1=angle_1,
                            theta2=angle_2,
                            facecolor=wedge_color,
                            edgecolor="none",
                            zorder=3
                        )
                        ax.add_patch(wedge_patch)
    
    elif display_mode == "line_trend":
        # Draw sparklines in each cell showing attention trend over time
        cell_width = 1.0  # Normalized cell width
        
        for i in range(n_rows):
            for j in range(n_cols):
                cx, cy = col_centers[j], row_centers[i]
                
                # Error circle
                err_val = error_vals[i, j]
                circle_frac = 0 if max_err < 1e-12 else (err_val / max_err)
                circle_radius = circle_frac * ring_radius / 2  # Smaller to not interfere with line
                
                grey_circle = Circle(
                    (cx, cy),
                    radius=circle_radius,
                    facecolor="#D55E00",
                    edgecolor="none",
                    alpha=0.6,
                    zorder=2
                )
                ax.add_patch(grey_circle)
                
                # Draw sparkline trend
                if epoch_aggregation == "trend":
                    # For trend mode, we connect min, first, mean, last, max with a line
                    x_points = np.linspace(cx - ring_radius*0.8, cx + ring_radius*0.8, 5)
                    values = [
                        trend_data["min"][i, j],
                        trend_data["first"][i, j],
                        trend_data["mean"][i, j],
                        trend_data["last"][i, j],
                        trend_data["max"][i, j]
                    ]
                else:
                    # Regular trend with all epochs
                    x_points = np.linspace(cx - ring_radius*0.8, cx + ring_radius*0.8, n_epochs)
                    values = matrices[:, i, j]
                
                # Normalize values to the cell height
                y_min = vmin
                y_max = vmax
                y_range = y_max - y_min
                norm_values = [(v - y_min) / y_range if y_range > 0 else 0.5 for v in values]
                
                # Convert to cell coordinates
                y_points = [cy - ring_radius*0.8 + norm_val * ring_radius*1.6 for norm_val in norm_values]
                
                # Create line segments
                points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Color segments by their values
                segment_colors = []
                for k in range(len(values) - 1):
                    val = (values[k] + values[k+1]) / 2
                    segment_colors.append(plt.get_cmap(cmap)(norm(val)))
                
                # Create a LineCollection
                lc = LineCollection(segments, colors=segment_colors, linewidth=2, zorder=3)
                ax.add_collection(lc)
                
                # Add points
                for x, y, val in zip(x_points, y_points, values):
                    color = plt.get_cmap(cmap)(norm(val))
                    ax.plot(x, y, 'o', color=color, markersize=4, zorder=4)
    
    elif display_mode == "heatmap_grid":
        # Draw a small heatmap in each cell showing evolution over time
        
        for i in range(n_rows):
            for j in range(n_cols):
                cx, cy = col_centers[j], row_centers[i]
                
                # Extract time series for this cell
                if epoch_aggregation == "trend":
                    # For trend, we'll create a small 1x5 heatmap
                    cell_data = np.array([
                        trend_data["min"][i, j],
                        trend_data["first"][i, j],
                        trend_data["mean"][i, j],
                        trend_data["last"][i, j],
                        trend_data["max"][i, j]
                    ]).reshape(1, 5)
                    cell_width = 5
                    cell_height = 1
                else:
                    # Regular matrices
                    cell_data = matrices[:, i, j].reshape(1, -1)
                    cell_width = n_epochs
                    cell_height = 1
                
                # Calculate cell dimensions
                width = ring_radius * 1.6
                height = ring_radius * 0.6
                
                # Create a mini heatmap at this cell position
                mini_extent = [
                    cx - width/2, cx + width/2,
                    cy - height/2, cy + height/2
                ]
                
                im = ax.imshow(
                    cell_data,
                    extent=mini_extent,
                    aspect='auto',
                    origin='lower',
                    cmap=cmap,
        norm=norm,
                    zorder=3
                )
    
    # 11) Detect and highlight attention regions if requested
    if region_detection:
        regions = detect_attention_regions(mean_vals, threshold=region_threshold)
        
        for region in regions:
            top, left, bottom, right = region
            top_y, left_x = row_centers[top] - 0.5, col_centers[left] - 0.5
            height = (row_centers[bottom] - row_centers[top]) + 1
            width = (col_centers[right] - col_centers[left]) + 1
            
            rect = Rectangle(
                (left_x, top_y), width, height,
                linewidth=2, edgecolor='red', facecolor='none',
                zorder=10
            )
            ax.add_patch(rect)
    
    # 12) Add a legend explaining the visualization
    if display_mode == "half_pie":
        if epoch_aggregation == "trend":
            legend_elements = [
                patches.Patch(facecolor='grey', edgecolor='none', alpha=0.6, label='Uncertainty (std)'),
                Wedge((0, 0), 0.1, -45, -9, facecolor='grey', label='Min'),
                Wedge((0, 0), 0.1, -9, 27, facecolor='grey', label='First'),
                Wedge((0, 0), 0.1, 27, 63, facecolor='grey', label='Mean'),
                Wedge((0, 0), 0.1, 63, 99, facecolor='grey', label='Last'),
                Wedge((0, 0), 0.1, 99, 135, facecolor='grey', label='Max')
            ]
        else:
            legend_title = "Epochs"
            if epoch_aggregation == "bin":
                legend_elements = [
                    patches.Patch(facecolor='grey', edgecolor='none', alpha=0.6, label='Uncertainty (std)')
                ]
                for i, label in enumerate(epoch_info["bins"]):
                    angle1 = -45 + i * slice_angle
                    angle2 = -45 + (i+1) * slice_angle
                    legend_elements.append(
                        Wedge((0, 0), 0.1, angle1, angle2, facecolor='grey', label=f'Epochs {label}')
                    )
            else:  # "select"
                legend_elements = [
                    patches.Patch(facecolor='grey', edgecolor='none', alpha=0.6, label='Uncertainty (std)')
                ]
                for i, epoch_idx in enumerate(epoch_indices):
                    angle1 = -45 + i * slice_angle
                    angle2 = -45 + (i+1) * slice_angle
                    legend_elements.append(
                        Wedge((0, 0), 0.1, angle1, angle2, facecolor='grey', label=f'Epoch {epoch_idx}')
                    )
            
        ax.legend(
            handles=legend_elements,
            title=legend_title if 'legend_title' in locals() else None,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(3, len(legend_elements))
        )
        
    elif display_mode == "line_trend":
        legend_elements = [
            patches.Patch(facecolor='grey', edgecolor='none', alpha=0.6, label='Uncertainty (std)'),
            Line2D([0], [0], color='grey', lw=2, label='Attention trend')
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=2
        )
    
    # 13) Save if requested
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Adaptive attention evolution heatmap saved to {save_path}")
    else:
        plt.tight_layout()
    
    return ax, selected_indices, epoch_info


def select_important_tokens(matrices, tokens=None, max_tokens=30, method="importance", custom_indices=None):
    """
    Select important tokens from an attention matrix sequence.
    
    Parameters
    ----------
    matrices : np.ndarray
        Attention matrices of shape (n_epochs, n_tokens, n_tokens)
    tokens : list of str, optional
        Token labels
    max_tokens : int
        Maximum number of tokens to select
    method : str
        Selection method: "importance" (most attended), "random", "first_last", "custom"
    custom_indices : list of int, optional
        Custom indices to use when method="custom"
        
    Returns
    -------
    list of int
        Selected token indices
    """
    import numpy as np
    import re
    
    n_epochs, n_tokens, _ = matrices.shape
    
    if method == "custom" and custom_indices is not None:
        return [i for i in custom_indices if i < n_tokens][:max_tokens]
    
    if method == "random":
        return np.random.choice(n_tokens, size=min(max_tokens, n_tokens), replace=False).tolist()
    
    if method == "first_last":
        half = max_tokens // 2
        first_half = list(range(min(half, n_tokens)))
        last_half = list(range(max(0, n_tokens-half), n_tokens))
        return first_half + last_half
    
    # Default: importance-based selection
    # Calculate token importance based on attention patterns
    token_importance = np.zeros(n_tokens)
    
    # Sum attention given and received across all epochs
    for matrix in matrices:
        token_importance += matrix.sum(axis=0)  # attention received
        token_importance += matrix.sum(axis=1)  # attention given
    
    # Identify special tokens if token labels are provided
    special_tokens_indices = []
    if tokens is not None:
        special_pattern = re.compile(r'^\[.*\]$|^<.*>$|^[.,!?;:"]$')
        special_tokens_indices = [i for i, t in enumerate(tokens) if special_pattern.match(t)]
    
    # Always include special tokens
    remaining_slots = max_tokens - len(special_tokens_indices)
    
    # For remaining slots, select most important non-special tokens
    if remaining_slots > 0:
        # Set importance of special tokens to -1 to exclude them from top selection
        importance_for_sorting = token_importance.copy()
        for idx in special_tokens_indices:
            importance_for_sorting[idx] = -1
            
        # Get top tokens by importance
        top_indices = np.argsort(-importance_for_sorting)[:remaining_slots]
        
        # Combine and sort all selected indices
        selected = sorted(list(set(special_tokens_indices).union(set(top_indices))))
        return selected[:max_tokens]
    else:
        # If we already have too many special tokens, just return those
        return special_tokens_indices[:max_tokens]


def detect_attention_regions(attention_matrix, threshold=0.8, min_size=2):
    """
    Detect rectangular regions of high attention in the matrix.
    
    Parameters
    ----------
    attention_matrix : np.ndarray
        2D attention matrix
    threshold : float
        Threshold for high attention (0.0 to 1.0)
    min_size : int
        Minimum size (area) of regions to detect
        
    Returns
    -------
    list of tuples
        List of (top, left, bottom, right) coordinates for each region
    """
    import numpy as np
    
    # Simple implementation based on thresholding
    high_attention = attention_matrix > np.percentile(attention_matrix, threshold * 100)
    
    # Find connected components (this is a simplified approach)
    n_rows, n_cols = attention_matrix.shape
    visited = np.zeros_like(high_attention, dtype=bool)
    regions = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            if high_attention[i, j] and not visited[i, j]:
                # Found a new region, perform region growing
                top, left, bottom, right = i, j, i, j
                stack = [(i, j)]
                visited[i, j] = True
                
                while stack:
                    r, c = stack.pop()
                    
                    # Update region bounds
                    top = min(top, r)
                    left = min(left, c)
                    bottom = max(bottom, r)
                    right = max(right, c)
                    
                    # Check neighbors
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < n_rows and 0 <= nc < n_cols and 
                            high_attention[nr, nc] and not visited[nr, nc]):
                            stack.append((nr, nc))
                            visited[nr, nc] = True
                
                # Check if region is large enough
                if (bottom - top + 1) * (right - left + 1) >= min_size:
                    regions.append((top, left, bottom, right))
    
    return regions

def selective_attention_evolution_heatmap(
    matrices,
    tokens=None,
    x_labels=None,
    y_labels=None,
    title="Selective Attention Evolution",
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    cmap="Blues",
    gamma=1.5,
    linecolor="white",
    linewidths=1.0,
    ring_radius=0.45,
    save_path=None,
    enlarged_size=1.8,        # Factor by which to enlarge important cells
    top_percent=15,           # Percentage of cells to enlarge (cells with highest mean attention)
    min_top_cells=5,          # Minimum number of cells to enlarge
    max_top_cells=15,         # Maximum number of cells to enlarge
    use_std_error=False,      # If True, use standard error instead of standard deviation
    max_epochs_to_show=5,     # Maximum number of epochs to show in trends (for epochs > max_epochs_to_show)
    region_detection=False,   # Whether to detect and highlight attention regions
    region_threshold=0.8      # Threshold for region detection
):
    """
    Creates a selective attention evolution heatmap that enlarges important cells and 
    shows trends only in those enlarged cells.
    
    Parameters
    ----------
    matrices : list or np.ndarray
        A list of attention matrices or a single 3D array of shape (n_epochs, n_rows, n_cols)
    tokens : list of str, optional
        The token labels corresponding to the matrix rows/columns
    x_labels : list of str, optional
        Column labels. If None and tokens provided, will use tokens
    y_labels : list of str, optional
        Row labels. If None and tokens provided, will use tokens
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure + axes is created
    cmap : str or matplotlib.colors.Colormap
        Colormap for visualization. Defaults to 'Blues'
    gamma : float
        Gamma value for the PowerNorm color scaling
    linecolor : str
        Color of grid lines in the heatmap
    linewidths : float
        Width of grid lines in the heatmap
    ring_radius : float
        Radius for the visualizations in each cell (as fraction of half-cell size)
    save_path : str, optional
        If provided, saves the figure to this path
    enlarged_size : float
        Factor by which to enlarge important cells
    top_percent : float
        Percentage of cells to enlarge (cells with highest mean attention)
    min_top_cells : int
        Minimum number of cells to enlarge
    max_top_cells : int
        Maximum number of cells to enlarge
    use_std_error : bool
        If True, uses standard error instead of standard deviation
    max_epochs_to_show : int
        Maximum number of epochs to show in evolution trends
    region_detection : bool
        Whether to detect and highlight attention regions
    region_threshold : float
        Threshold for region detection (0.0 to 1.0)
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    top_cells : list of tuples
        The (row, col) indices of cells that were enlarged
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    from matplotlib.patches import Wedge, Rectangle, Circle
    from matplotlib.colors import PowerNorm
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.cm as cm
    import re
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # 1) Convert input to np.array (n_epochs, n_rows, n_cols)
    matrices = np.array(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            f"Expected matrices to be a list or 3D array of shape (n_epochs, n_rows, n_cols). "
            f"Got shape: {matrices.shape}"
        )

    n_epochs, n_rows, n_cols = matrices.shape
    
    # Handle token labels
    if tokens is not None:
        if len(tokens) != n_rows or len(tokens) != n_cols:
            raise ValueError(f"Token list length ({len(tokens)}) doesn't match matrix dimensions ({n_rows}x{n_cols})")
        if x_labels is None:
            x_labels = tokens
        if y_labels is None:
            y_labels = tokens
    
    if x_labels is None:
        x_labels = [f"X{j}" for j in range(n_cols)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(n_rows)]
    
    # 2) Calculate mean attention across all epochs
    mean_vals = np.mean(matrices, axis=0)
    
    # 3) Identify important cells to be enlarged
    flat_mean = mean_vals.flatten()
    num_cells = len(flat_mean)
    num_top_cells = max(min_top_cells, min(max_top_cells, int(num_cells * top_percent / 100)))
    
    # Get indices of top cells
    top_flat_indices = np.argsort(-flat_mean)[:num_top_cells]
    top_cells = [(idx // n_cols, idx % n_cols) for idx in top_flat_indices]
    
    # 4) Create column widths and row heights arrays
    default_width = 1.0
    default_height = 1.0
    column_widths = [default_width] * n_cols
    row_heights = [default_height] * n_rows
    
    # Set enlarged sizes for important cells
    for row_idx, col_idx in top_cells:
        row_heights[row_idx] = enlarged_size
        column_widths[col_idx] = enlarged_size
    
    # 5) Create or use existing axes
    if ax is None:
        fig_scale = max(1.0, n_rows / 10)  # Scale figure size with token count
        fig, ax = plt.subplots(figsize=(8 * fig_scale, 8 * fig_scale))
    else:
        fig = ax.figure
    
    # 6) Compute global min/max for color scale
    all_values = matrices.flatten()
    vmin, vmax = all_values.min(), all_values.max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    
    # 7) Create PowerNorm for color scaling
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    # 8) Create background heatmap with only mean values
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=mean_vals,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=cmap,
        column_widths=column_widths,
        row_heights=row_heights,
        cbar=True,
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        top_cells=top_cells
    )
    
    # 9) Calculate cell centers using heatmap positions instead of plotter.get_cellsizes()
    # Get the heatmap data from the first AxesImage in the axes
    heatmap_obj = None
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.image.AxesImage):
            heatmap_obj = child
            break
            
    if heatmap_obj is None:
        raise ValueError("Could not find heatmap image in axes")
    
    # Get the extent of the heatmap
    extent = heatmap_obj.get_extent()
    x_min, x_max, y_min, y_max = extent
    
    # Calculate the width and height of cells
    # This handles the case where cells have different sizes due to enlargement
    x_positions = [x_min]
    current_pos = x_min
    for width in column_widths:
        current_pos += width
        x_positions.append(current_pos)
    
    y_positions = [y_min]
    current_pos = y_min
    for height in row_heights[::-1]:  # Reverse because y axis starts from bottom
        current_pos += height
        y_positions.append(current_pos)
    
    y_positions = y_positions[::-1]  # Reverse back to match matrix row order
    
    # Calculate centers of each cell
    col_centers = [(x_positions[i] + x_positions[i+1]) / 2 for i in range(n_cols)]
    row_centers = [(y_positions[i] + y_positions[i+1]) / 2 for i in range(n_rows)]
    
    # 10) Handle epoch aggregation if there are too many epochs
    if n_epochs > max_epochs_to_show:
        # Split epochs into equal parts and average each part
        part_size = n_epochs // max_epochs_to_show
        remaining = n_epochs % max_epochs_to_show
        
        parts = []
        start_idx = 0
        
        for i in range(max_epochs_to_show):
            # Distribute remaining elements to make parts as equal as possible
            current_part_size = part_size + (1 if i < remaining else 0)
            end_idx = start_idx + current_part_size
            
            # Average the matrices in this part
            part_avg = np.mean(matrices[start_idx:end_idx], axis=0)
            parts.append(part_avg)
            
            start_idx = end_idx
        
        # Replace original matrices with aggregated parts
        aggregated_matrices = np.array(parts)
        epoch_labels = [f"Epochs {i*part_size}-{min((i+1)*part_size-1, n_epochs-1)}" for i in range(max_epochs_to_show)]
    else:
        aggregated_matrices = matrices
        epoch_labels = [f"Epoch {i}" for i in range(n_epochs)]
    
    n_parts = len(aggregated_matrices)
    
    # 11) Calculate std/error values for uncertainty visualization
    std_vals = np.std(matrices, axis=0)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n_epochs)
    else:
        error_vals = std_vals
    
    max_err = np.max(error_vals)
    
    # 12) Add evolution trends to enlarged cells
    slice_angle = 180 / n_parts  # For half-pie visualization
    
    for i in range(n_rows):
        for j in range(n_cols):
            # Only add trends to enlarged cells
            if (i, j) in top_cells:
                cx, cy = col_centers[j], row_centers[i]
                
                # Draw uncertainty circle
                err_val = error_vals[i, j]
                circle_radius = ring_radius * (err_val / max_err) if max_err > 0 else 0
                
                grey_circle = Circle(
                    (cx, cy),
                    radius=circle_radius,
                    facecolor="grey",
                    edgecolor="none",
                    alpha=0.6,
                    zorder=1
                )
                ax.add_patch(grey_circle)
                
                # Add half-pie evolution visualization
                for k in range(n_parts):
                    val = aggregated_matrices[k, i, j]
                    color = plt.get_cmap(cmap)(norm(val))
                    
                    angle1 = -45 + k * slice_angle
                    angle2 = -45 + (k + 1) * slice_angle
                    
                    wedge = Wedge(
                        (cx, cy),
                        ring_radius,
                        angle1,
                        angle2,
                        facecolor=color,
                        edgecolor=linecolor,
                        linewidth=0.5,
                        zorder=2
                    )
                    ax.add_patch(wedge)
                    
                # Add sparkline if desired
                # Create points for the line
                x_points = np.linspace(cx - ring_radius*0.8, cx + ring_radius*0.8, n_parts)
                values = aggregated_matrices[:, i, j]
                
                # Normalize values to the cell height
                y_range = vmax - vmin
                norm_values = [(v - vmin) / y_range if y_range > 0 else 0.5 for v in values]
                
                # Convert to cell coordinates
                y_points = [cy - ring_radius*0.4 + norm_val * ring_radius*0.8 for norm_val in norm_values]
                
                # Create line segments
                points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Color segments by their values
                segment_colors = []
                for k in range(len(values) - 1):
                    val = (values[k] + values[k+1]) / 2
                    segment_colors.append(plt.get_cmap(cmap)(norm(val)))
                
                # Create a LineCollection
                lc = LineCollection(segments, colors=segment_colors, linewidth=2, zorder=3)
                ax.add_collection(lc)
                
                # Add points
                for x, y, val in zip(x_points, y_points, values):
                    color = plt.get_cmap(cmap)(norm(val))
                    ax.plot(x, y, 'o', color=color, markersize=4, zorder=4)
    
    # 13) Detect and highlight attention regions if requested
    if region_detection:
        regions = detect_attention_regions(mean_vals, threshold=region_threshold)
        
        for region in regions:
            top, left, bottom, right = region
            # Convert region to plot coordinates
            rect_x = col_centers[left] - 0.5
            rect_y = row_centers[top] - 0.5
            rect_width = col_centers[right] - col_centers[left] + 1
            rect_height = row_centers[bottom] - row_centers[top] + 1
            
            rect = Rectangle(
                (rect_x, rect_y), rect_width, rect_height,
                linewidth=2, edgecolor='red', facecolor='none',
                zorder=10
            )
            ax.add_patch(rect)
    
    # 14) Add a legend
    legend_elements = [
        patches.Patch(facecolor='grey', edgecolor='none', alpha=0.6, label='Uncertainty (std)'),
        Line2D([0], [0], color='grey', lw=2, label='Attention trend')
    ]
    
    for i in range(min(5, n_parts)):
        angle1 = -45 + i * slice_angle
        angle2 = -45 + (i+1) * slice_angle
        legend_elements.append(
            Wedge((0, 0), 0.1, angle1, angle2, facecolor='grey', label=epoch_labels[i])
        )
    
    ax.legend(
        handles=legend_elements,
        title="Epochs",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(3, len(legend_elements))
    )
    
    # 15) Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return ax, top_cells

def visualize_attention_evolution(attentions_over_time, tokens, layer, head,
                                 top_n=5, enlarged_size=1.8, gamma=1.5,
                                 title="Attention Evolution Over Time", 
                                 xlabel="Tokens Attended to", 
                                 ylabel="Tokens Attending",
                                 max_epochs_to_show=5,
                                 use_std_error=False,
                                 ring_radius=0.45,
                                 cmap="Blues",
                                 linecolor="white",
                                 linewidths=1.0,
                                 auto_detect_regions=False,
                                 region_threshold=0.8,
                                 save_path=None):
    """
    Visualizes the evolution of attention scores across training steps/epochs.
    
    Parameters:
    - attentions_over_time: List of attention matrices from different epochs/training steps.
                           Shape should be [n_epochs, n_layers, n_heads, n_tokens, n_tokens]
                           or [n_epochs, n_tokens, n_tokens] if layer and head are already selected.
    - tokens: List of token labels to display on the heatmaps.
    - layer: The layer number of the attention to visualize.
    - head: The head number of the attention to visualize.
    - top_n: The number or percentage of top cells to enlarge and show evolution trends.
            If < 1, treated as percentage (e.g., 0.1 = top 10%).
            If >= 1, treated as absolute number of cells.
    - enlarged_size: Factor by which to enlarge the top cells.
    - gamma: Gamma value for the power normalization of the colormap.
    - title: Title for the visualization.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - max_epochs_to_show: Maximum number of epochs to display in evolution trends.
                         If n_epochs > max_epochs_to_show, epochs will be aggregated.
    - use_std_error: If True, uses standard error (std/sqrt(n)) instead of standard deviation.
    - ring_radius: Radius for the evolution visualizations (as fraction of cell size).
    - cmap: Colormap for the heatmap and trends.
    - linecolor: Color of grid lines in the heatmap.
    - linewidths: Width of grid lines in the heatmap.
    - auto_detect_regions: If True, automatically detect and highlight attention regions.
    - region_threshold: Threshold for region detection (0.0 to 1.0).
    - save_path: If provided, saves the figure to this path.
    
    Returns:
    - ax: The matplotlib axes containing the visualization.
    - top_cells: The (row, col) indices of cells that were enlarged.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Wedge, Circle
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.patches as patches
    
    # 1) Extract attention matrices for the specified layer and head
    matrices = []
    for epoch_attn in attentions_over_time:
        # Handle different input formats
        if isinstance(epoch_attn, list):  # List of layer attentions
            # Extract attention for specific layer and head
            layer_attn = epoch_attn[layer].squeeze(0)
            head_attn = layer_attn[head]
        elif epoch_attn.ndim == 3:  # Already layer and head specific
            head_attn = epoch_attn
        elif epoch_attn.ndim >= 4:  # Shape includes layers and heads
            head_attn = epoch_attn[layer][head]
        else:
            raise ValueError(f"Unexpected attention shape: {epoch_attn.shape}")
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(head_attn):
            head_attn = head_attn.detach().cpu().numpy()
            
        matrices.append(head_attn)
    
    # Convert to 3D array [n_epochs, n_tokens, n_tokens]
    matrices = np.array(matrices)
    n_epochs, n_rows, n_cols = matrices.shape
    
    # 2) Set up labels
    if tokens is None:
        x_labels = [f"Token {j}" for j in range(n_cols)]
        y_labels = [f"Token {i}" for i in range(n_rows)]
    else:
        x_labels = tokens
        y_labels = tokens
    
    # 3) Calculate mean attention across all epochs
    mean_vals = np.mean(matrices, axis=0)
    
    # 4) Identify top cells to enlarge
    if top_n < 1:  # Treat as percentage
        top_percent = top_n * 100
        num_cells = mean_vals.size
        num_top_cells = max(5, min(15, int(num_cells * top_percent)))
    else:  # Treat as absolute number
        num_top_cells = int(top_n)
    
    # Find the top N cells by mean attention
    flat_mean = mean_vals.flatten()
    top_flat_indices = np.argsort(-flat_mean)[:num_top_cells]
    top_cells = [(idx // n_cols, idx % n_cols) for idx in top_flat_indices]
    
    # 5) Set up column widths and row heights for enlarged cells
    column_widths = [1.0] * n_cols
    row_heights = [1.0] * n_rows
    
    for row_idx, col_idx in top_cells:
        row_heights[row_idx] = enlarged_size
        column_widths[col_idx] = enlarged_size
    
    # 6) Create figure and axes
    fig_scale = max(1.0, n_rows / 10)  # Scale figure size with token count
    fig, ax = plt.subplots(figsize=(8 * fig_scale, 8 * fig_scale))
    
    # 7) Compute global min/max for color scale
    all_values = matrices.flatten()
    vmin, vmax = all_values.min(), all_values.max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    
    # 8) Create PowerNorm for color scaling
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    # 9) Create background heatmap with mean values
    ax, plotter = create_tablelens_heatmap(
        attention_matrix=mean_vals,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=cmap,
        column_widths=column_widths,
        row_heights=row_heights,
        cbar=True,
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        top_cells=top_cells
    )
    
    # 10) Calculate cell centers
    # Get the heatmap data from the first AxesImage in the axes
    heatmap_obj = None
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.image.AxesImage):
            heatmap_obj = child
            break
            
    if heatmap_obj is None:
        raise ValueError("Could not find heatmap image in axes")
    
    # Get the extent of the heatmap
    extent = heatmap_obj.get_extent()
    x_min, x_max, y_min, y_max = extent
    
    # Calculate positions for cell boundaries
    x_positions = [x_min]
    current_pos = x_min
    for width in column_widths:
        current_pos += width
        x_positions.append(current_pos)
    
    y_positions = [y_min]
    current_pos = y_min
    for height in row_heights[::-1]:  # Reverse because y axis starts from bottom
        current_pos += height
        y_positions.append(current_pos)
    
    y_positions = y_positions[::-1]  # Reverse back to match matrix row order
    
    # Calculate centers of each cell
    col_centers = [(x_positions[i] + x_positions[i+1]) / 2 for i in range(n_cols)]
    row_centers = [(y_positions[i] + y_positions[i+1]) / 2 for i in range(n_rows)]
    
    # 11) Handle epoch aggregation if there are too many epochs
    if n_epochs > max_epochs_to_show:
        # Split epochs into equal parts and average each part
        part_size = n_epochs // max_epochs_to_show
        remaining = n_epochs % max_epochs_to_show
        
        parts = []
        start_idx = 0
        
        for i in range(max_epochs_to_show):
            # Distribute remaining elements to make parts as equal as possible
            current_part_size = part_size + (1 if i < remaining else 0)
            end_idx = start_idx + current_part_size
            
            # Average the matrices in this part
            part_avg = np.mean(matrices[start_idx:end_idx], axis=0)
            parts.append(part_avg)
            
            start_idx = end_idx
        
        # Replace original matrices with aggregated parts
        aggregated_matrices = np.array(parts)
        epoch_labels = [f"Epochs {i*part_size}-{min((i+1)*part_size-1, n_epochs-1)}" for i in range(max_epochs_to_show)]
    else:
        aggregated_matrices = matrices
        epoch_labels = [f"Epoch {i}" for i in range(n_epochs)]
    
    n_parts = len(aggregated_matrices)
    
    # 12) Calculate std/error values for uncertainty visualization
    std_vals = np.std(matrices, axis=0)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n_epochs)
    else:
        error_vals = std_vals
    
    max_err = np.max(error_vals)
    
    # 13) Add evolution trends to enlarged cells
    slice_angle = 180 / n_parts  # For half-pie visualization
    
    for i in range(n_rows):
        for j in range(n_cols):
            # Only add trends to enlarged cells
            if (i, j) in top_cells:
                cx, cy = col_centers[j], row_centers[i]
                
                # Draw uncertainty circle
                err_val = error_vals[i, j]
                circle_radius = ring_radius * (err_val / max_err) if max_err > 0 else 0
                
                grey_circle = Circle(
                    (cx, cy),
                    radius=circle_radius,
                    facecolor="grey",
                    edgecolor="none",
                    alpha=0.6,
                    zorder=1
                )
                ax.add_patch(grey_circle)
                
                # Add half-pie evolution visualization
                for k in range(n_parts):
                    val = aggregated_matrices[k, i, j]
                    color = plt.get_cmap(cmap)(norm(val))
                    
                    angle1 = -45 + k * slice_angle
                    angle2 = -45 + (k + 1) * slice_angle
                    
                    wedge = Wedge(
                        (cx, cy),
                        ring_radius,
                        angle1,
                        angle2,
                        facecolor=color,
                        edgecolor=linecolor,
                        linewidth=0.5,
                        zorder=2
                    )
                    ax.add_patch(wedge)
                    
                # Add trend line
                x_points = np.linspace(cx - ring_radius*0.8, cx + ring_radius*0.8, n_parts)
                values = aggregated_matrices[:, i, j]
                
                # Normalize values to the cell height
                y_range = vmax - vmin
                norm_values = [(v - vmin) / y_range if y_range > 0 else 0.5 for v in values]
                
                # Convert to cell coordinates
                y_points = [cy - ring_radius*0.4 + norm_val * ring_radius*0.8 for norm_val in norm_values]
                
                # Create line segments
                points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Color segments by their values
                segment_colors = []
                for k in range(len(values) - 1):
                    val = (values[k] + values[k+1]) / 2
                    segment_colors.append(plt.get_cmap(cmap)(norm(val)))
                
                # Create a LineCollection
                lc = LineCollection(segments, colors=segment_colors, linewidth=2, zorder=3)
                ax.add_collection(lc)
                
                # Add points
                for x, y, val in zip(x_points, y_points, values):
                    color = plt.get_cmap(cmap)(norm(val))
                    ax.plot(x, y, 'o', color=color, markersize=4, zorder=4)
    
    # 14) Add region detection if requested
    if auto_detect_regions:
        regions = detect_attention_regions(mean_vals, threshold=region_threshold)
        
        for region in regions:
            top, left, bottom, right = region
            # Convert region to plot coordinates
            rect_x = x_positions[left]
            rect_y = y_positions[top]
            rect_width = x_positions[right+1] - x_positions[left]
            rect_height = y_positions[bottom+1] - y_positions[top]
            
            rect = patches.Rectangle(
                (rect_x, rect_y), rect_width, rect_height,
                linewidth=2, edgecolor='red', facecolor='none',
                zorder=10
            )
            ax.add_patch(rect)
    
    # 15) Add a legend
    legend_elements = [
        patches.Patch(facecolor='grey', edgecolor='none', alpha=0.6, label='Uncertainty'),
        Line2D([0], [0], color='grey', lw=2, label='Attention trend')
    ]
    
    for i in range(min(5, n_parts)):
        angle1 = -45 + i * slice_angle
        angle2 = -45 + (i+1) * slice_angle
        legend_elements.append(
            Wedge((0, 0), 0.1, angle1, angle2, facecolor='grey', label=epoch_labels[i])
        )
    
    ax.legend(
        handles=legend_elements,
        title="Epochs",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(3, len(legend_elements))
    )
    
    # 16) Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return ax, top_cells

