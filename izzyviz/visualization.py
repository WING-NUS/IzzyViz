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
