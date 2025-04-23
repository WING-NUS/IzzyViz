# visualization.py

from .my_seaborn import heatmap
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches 
from .utility import find_non_overlapping_locally_maximal_rectangles
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection


# Define a function to make special tokens bold
def bold_special_tokens(label):
    special_tokens = ['[CLS]', '[SEP]', '[PAD]']
    if label in special_tokens:
        return f'$\mathbf{{{label}}}$'  # Make it bold using LaTeX math formatting
    return label

def create_tablelens_heatmap(attention_matrix, x_labels, y_labels, title, xlabel, ylabel, ax, cmap='Blues',
                             column_widths=None, row_heights=None, top_cells=None, vmin=None,
                             vmax=None, norm=None, gamma=1.5, left_top_cells=None, right_bottom_cells=None, linecolor='white', linewidths=1.0,
                             cbar=True, show_scores=True, background_color=True, lean_more=False):
    """
    Creates a heatmap with variable cell sizes and annotations for top cells.
    Returns both the axis and the plotter object for further customization.
    """

    if isinstance(attention_matrix, np.ndarray):
        data = attention_matrix  # It's already a NumPy array, no need to convert
    else:
        data = attention_matrix.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy

    # print("data: ", data.shape)

    if show_scores:
        # Create annot_data for annotations
        annot_data = np.empty_like(data, dtype=object)
        annot_data[:] = ''  # Initialize all cells to empty strings

        if top_cells is not None:
            for (row_index, col_index) in top_cells:
                value = data[row_index, col_index]
                annot_data[row_index, col_index] = f"{value:.3f}"
    else:
        annot_data = None

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
    num_ticks = 7 # Adjust the number of ticks
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
        if lean_more:
            label.set_rotation(90)
        else:
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
            if idx in x_indices and background_color:
                label.set_bbox(dict(facecolor='yellowgreen', edgecolor='yellowgreen', boxstyle='round,pad=0.2', alpha=0.5))

        # Adjust y tick labels without inversion
        for row_index in y_indices:
            if row_index < len(y_ticklabels) and background_color:
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



def visualize_attention_self_attention(
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
):
    """
    Visualizes attention matrices for encoder-only and decoder-only models.
    
    Parameters:
    - attentions: List of attention matrices from the model.
    - tokens: List of token labels to display on the heatmaps.
    - layer: The layer number of the attention to visualize.
    - head: The head number of the attention to visualize.
    - question_end: The index where the first sentence ends in the token list (used in question-context modes).
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - top_n: The number of top attention scores to highlight.
    - enlarged_size: Factor by which to enlarge the top cells.
    - gamma: Gamma value for the power normalization of the colormap.
    - mode: The mode of visualization ('self_attention' or 'question_context').
    - plot_titles: List of titles for the subplots. If None, default titles are used.
    - left_top_cells: List of (row, col) tuples for the top-left cells of regions to highlight.
    - right_bottom_cells: List of (row, col) tuples for the bottom-right cells of regions to highlight.
    - auto_detect_regions: If True, automatically detect locally maximal attention regions.
                          This will override any manually specified left_top_cells and right_bottom_cells.
    - if_interval: If True, show labels at regular intervals.
    - if_top_cells: If True, show labels for tokens associated with important attention cells.
    - length_threshold: Maximum token count before switching to sparse labeling
    - interval: Show a label every N tokens in sparse mode
    - show_scores_in_enlarged_cells: Whether to display attention scores in enlarged cells
                                    (automatically disabled in sparse labeling mode)
    - lean_more: If True, rotate x-axis labels by 90 degrees
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
                xlabel,
                ylabel,
                axes[i],
                column_widths=column_widths,
                row_heights=row_heights,
                top_cells=top_cells,
                vmin=global_vmin,
                vmax=global_vmax,
                norm=norm,
                gamma=gamma,
                left_top_cells=left_top_cells,
                right_bottom_cells=right_bottom_cells,
                lean_more=lean_more
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
        title = plot_titles[0]

        # Prepare data
        if isinstance(attention_matrix, np.ndarray):
            data = attention_matrix  # It's already a NumPy array, no need to convert
        else:
            data = attention_matrix.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy 
        # data = attention_matrix.detach().cpu().numpy()
        global_vmin = data.min()
        global_vmax = data.max()
        norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

        # Find top attention cells
        top_cells = find_top_cells(data, top_n)

        # Set background_color based on whether sparse labels are used
        is_sparse = len(tokens) > length_threshold
        use_background_color = not is_sparse  # Only use background color when NOT using sparse labels
        
        if is_sparse:
            x_labels = generate_sparse_labels(tokens, top_cells, 1, interval=interval, if_interval=if_interval, if_top_cells=if_top_cells)
            y_labels = generate_sparse_labels(tokens, top_cells, 0, interval=interval, if_interval=if_interval, if_top_cells=if_top_cells)
            show_scores = False
        else:
            x_labels = [bold_special_tokens(token) for token in tokens]
            y_labels = [bold_special_tokens(token) for token in tokens]
            show_scores = show_scores_in_enlarged_cells

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

        ax, plotter = create_tablelens_heatmap(
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
            right_bottom_cells=right_bottom_cells,
            show_scores=show_scores,
            background_color=use_background_color,  # Use background color only for non-sparse labels
            lean_more=lean_more
        )

        # If using sparse labels, set custom ticks to only show where labels exist
        if is_sparse:
            # Find positions with non-empty labels
            x_tick_indices = [i for i, label in enumerate(x_labels) if label]
            y_tick_indices = [i for i, label in enumerate(y_labels) if label]
            
            # Get positions from the plotter
            if x_tick_indices:
                # Get cell centers for each position with a label
                x_positions = [plotter.col_positions[i] + 
                              (plotter.col_positions[i+1] - plotter.col_positions[i])/2 
                              for i in x_tick_indices]
                x_tick_labels = [x_labels[i] for i in x_tick_indices]
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
            
            if y_tick_indices:
                y_positions = [plotter.row_positions[i] + 
                              (plotter.row_positions[i+1] - plotter.row_positions[i])/2 
                              for i in y_tick_indices]
                y_tick_labels = [y_labels[i] for i in y_tick_indices]
                ax.set_yticks(y_positions)
                ax.set_yticklabels(y_tick_labels)
        

        if save_path is None:
            save_path = "self_attention_heatmap.pdf"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print("Self-attention heatmap saved to ", save_path)

    else:
        raise ValueError("Invalid mode for encoder-only visualization. Choose from 'question_context' or 'self_attention'.")

# def visualize_attention_decoder_only(attentions, source_tokens, generated_tokens, layer, head,
#                                      top_n=3, enlarged_size=1.8, gamma=1.5,
#                                      plot_titles=None, left_top_cells=None, right_bottom_cells=None,
#                                      use_case='full_sequence', save_path=None):
#     """
#     Visualizes attention matrices for decoder-only models.

#     Parameters:
#     - attentions: List of attention matrices from the model.
#     - source_tokens: List of source token labels.
#     - generated_tokens: List of generated token labels.
#     - layer: The layer number of the attention to visualize.
#     - head: The head number of the attention to visualize.
#     - top_n: The number of top attention scores to highlight.
#     - enlarged_size: Factor by which to enlarge the top cells.
#     - gamma: Gamma value for the power normalization of the colormap.
#     - plot_titles: List of titles for the subplots. If None, default titles are used.
#     - left_top_cells: List of (row, col) tuples for the top-left cells of regions to highlight.
#     - right_bottom_cells: List of (row, col) tuples for the bottom-right cells of regions to highlight.
#     - use_case: The specific use case to visualize. Options are:
#         - 'full_sequence': Input sequence attending to itself (no token generation).
#         - 'self_attention_source': Self-Attention for Source Tokens (no causal masking).
#         - 'generated_to_source': Generated-to-Source Attention (fully connected).
#         - 'self_attention_generated': Self-Attention for Generated Tokens (causal-masked).
#     """
#     attn = attentions[layer].squeeze(0)[head]

#     if use_case == 'full_sequence':
#         # Input sequence attending to itself. The x, y labels are the same sentence.
#         tokens = source_tokens
#         attention_matrix = attn  # Shape: (seq_len, seq_len)
#         x_labels = [bold_special_tokens(token) for token in tokens]
#         y_labels = [bold_special_tokens(token) for token in tokens]
#         title = plot_titles[0] if plot_titles else "Self-Attention Heatmap (Full Sequence)"

#         # Prepare data
#         data = attention_matrix.detach().cpu().numpy()
#         global_vmin = data.min()
#         global_vmax = data.max()
#         norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

#         # Find top attention cells
#         top_cells = find_top_cells(data, top_n)

#         # Initialize column widths and row heights
#         num_rows, num_cols = data.shape
#         default_width = 1
#         default_height = 1
#         column_widths = [default_width] * num_cols
#         row_heights = [default_height] * num_rows

#         # Enlarge top cells
#         for (row_index, col_index) in top_cells:
#             column_widths[col_index] = enlarged_size
#             row_heights[row_index] = enlarged_size

#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax, _ = create_tablelens_heatmap(
#             attention_matrix,
#             x_labels,
#             y_labels,
#             title,
#             "Tokens Attended to",
#             "Tokens Attending",
#             ax,
#             column_widths=column_widths,
#             row_heights=row_heights,
#             top_cells=top_cells,
#             vmin=global_vmin,
#             vmax=global_vmax,
#             norm=norm,
#             gamma=gamma,
#             left_top_cells=left_top_cells,
#             right_bottom_cells=right_bottom_cells
#         )

#         if save_path is None:
#             save_path = "decoder_self_attention_heatmap.pdf"

#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close(fig)
#         print("Decoder self-attention heatmap saved to ", save_path)

#     elif use_case == 'self_attention_source':
#         # Self-Attention for Source Tokens (no causal masking)
#         tokens = source_tokens
#         seq_len = len(source_tokens)
#         attention_matrix = attn[:seq_len, :seq_len]
#         x_labels = [bold_special_tokens(token) for token in tokens]
#         y_labels = [bold_special_tokens(token) for token in tokens]
#         title = plot_titles[0] if plot_titles else "Self-Attention Heatmap (Source Tokens)"

#         # Prepare data
#         data = attention_matrix.detach().cpu().numpy()
#         global_vmin = data.min()
#         global_vmax = data.max()
#         norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

#         # Find top attention cells
#         top_cells = find_top_cells(data, top_n)

#         # Initialize column widths and row heights
#         num_rows, num_cols = data.shape
#         default_width = 1
#         default_height = 1
#         column_widths = [default_width] * num_cols
#         row_heights = [default_height] * num_rows

#         # Enlarge top cells
#         for (row_index, col_index) in top_cells:
#             column_widths[col_index] = enlarged_size
#             row_heights[row_index] = enlarged_size

#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax, _ = create_tablelens_heatmap(
#             attention_matrix,
#             x_labels,
#             y_labels,
#             title,
#             "Tokens Attended to",
#             "Tokens Attending",
#             ax,
#             column_widths=column_widths,
#             row_heights=row_heights,
#             top_cells=top_cells,
#             vmin=global_vmin,
#             vmax=global_vmax,
#             norm=norm,
#             gamma=gamma,
#             left_top_cells=left_top_cells,
#             right_bottom_cells=right_bottom_cells
#         )

#         if save_path is None:
#             save_path = "decoder_self_attention_source_tokens_heatmap.pdf"

#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close(fig)
#         print("Decoder self-attention heatmap for source tokens saved to ", save_path)

#     elif use_case == 'generated_to_source':
#         # Generated-to-Source Attention (fully connected)
#         source_seq_len = len(source_tokens)
#         generated_seq_len = len(generated_tokens)
#         attention_matrix = attn[source_seq_len:source_seq_len+generated_seq_len, :source_seq_len]
#         x_labels = [bold_special_tokens(token) for token in source_tokens]
#         y_labels = [bold_special_tokens(token) for token in generated_tokens]
#         title = plot_titles[0] if plot_titles else "Generated Tokens attending to Source Tokens"

#         # Prepare data
#         data = attention_matrix.detach().cpu().numpy()
#         global_vmin = data.min()
#         global_vmax = data.max()
#         norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

#         # Find top attention cells
#         top_cells = find_top_cells(data, top_n)

#         # Initialize column widths and row heights
#         num_rows, num_cols = data.shape
#         default_width = 1
#         default_height = 1
#         column_widths = [default_width] * num_cols
#         row_heights = [default_height] * num_rows

#         # Enlarge top cells
#         for (row_index, col_index) in top_cells:
#             column_widths[col_index] = enlarged_size
#             row_heights[row_index] = enlarged_size

#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax, _ = create_tablelens_heatmap(
#             attention_matrix,
#             x_labels,
#             y_labels,
#             title,
#             "Source Tokens",
#             "Generated Tokens",
#             ax,
#             column_widths=column_widths,
#             row_heights=row_heights,
#             top_cells=top_cells,
#             vmin=global_vmin,
#             vmax=global_vmax,
#             norm=norm,
#             gamma=gamma,
#             left_top_cells=left_top_cells,
#             right_bottom_cells=right_bottom_cells
#         )

#         if save_path is None:
#             save_path = "decoder_generated_to_source_attention_heatmap.pdf"

#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close(fig)
#         print("Decoder generated-to-source attention heatmap saved to ", save_path)

#     elif use_case == 'self_attention_generated':
#         # Self-Attention for Generated Tokens (causal-masked)
#         source_seq_len = len(source_tokens)
#         generated_seq_len = len(generated_tokens)
#         total_seq_len = source_seq_len + generated_seq_len
#         attention_matrix = attn[source_seq_len:total_seq_len, source_seq_len:total_seq_len]
#         x_labels = [bold_special_tokens(token) for token in generated_tokens]
#         y_labels = [bold_special_tokens(token) for token in generated_tokens]
#         title = plot_titles[0] if plot_titles else "Self-Attention Heatmap (Generated Tokens)"

#         # Prepare data
#         data = attention_matrix.detach().cpu().numpy()
#         global_vmin = data.min()
#         global_vmax = data.max()
#         norm = PowerNorm(gamma=gamma, vmin=global_vmin, vmax=global_vmax)

#         # Find top attention cells
#         top_cells = find_top_cells(data, top_n)

#         # Initialize column widths and row heights
#         num_rows, num_cols = data.shape
#         default_width = 1
#         default_height = 1
#         column_widths = [default_width] * num_cols
#         row_heights = [default_height] * num_rows

#         # Enlarge top cells
#         for (row_index, col_index) in top_cells:
#             column_widths[col_index] = enlarged_size
#             row_heights[row_index] = enlarged_size

#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax, _ = create_tablelens_heatmap(
#             attention_matrix,
#             x_labels,
#             y_labels,
#             title,
#             "Tokens Attended to",
#             "Tokens Attending",
#             ax,
#             column_widths=column_widths,
#             row_heights=row_heights,
#             top_cells=top_cells,
#             vmin=global_vmin,
#             vmax=global_vmax,
#             norm=norm,
#             gamma=gamma,
#             left_top_cells=left_top_cells,
#             right_bottom_cells=right_bottom_cells
#         )

#         if save_path is None:
#             save_path = "decoder_self_attention_generated_tokens_heatmap.pdf"

#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close(fig)
#         print("Decoder self-attention heatmap for generated tokens saved to ", save_path)

#     else:
#         raise ValueError("Invalid use_case for decoder-only visualization. Choose from 'full_sequence', 'self_attention_source', 'generated_to_source', or 'self_attention_generated'.")

def visualize_attention_encoder_decoder(attention_matrix, encoder_tokens, decoder_tokens,
                                        xlabel=None, ylabel=None,
                                        top_n=3, enlarged_size=1.8, gamma=1.5,
                                        plot_title=None, left_top_cells=None, right_bottom_cells=None,
                                        save_path=None, use_case='cross_attention',
                                        lean_more=False):
    """
    Visualizes attention matrices for encoder-decoder models.

    Parameters:
    - attention_matrix: The attention matrix (numpy array or torch tensor).
    - encoder_tokens: List of encoder token labels.
    - decoder_tokens: List of decoder token labels.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - top_n: The number of top attention scores to highlight.
    - enlarged_size: Factor by which to enlarge the top cells.
    - gamma: Gamma value for the power normalization of the colormap.
    - plot_title: Title for the plot.
    - left_top_cells: List of (row, col) tuples for the top-left cells of regions to highlight.
    - right_bottom_cells: List of (row, col) tuples for the bottom-right cells of regions to highlight.
    - save_path: File path to save the generated heatmap PDF.
    - use_case: Type of attention to visualize. Options are 'cross_attention', 'encoder_self_attention', 'decoder_self_attention'.
    - lean_more: If True, rotate x-axis labels by 90 degrees
    """

    # Prepare data
    data = attention_matrix
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if use_case == 'cross_attention':
        # Cross-Attention: Decoder attending to Encoder outputs
        x_labels = [bold_special_tokens(token) for token in encoder_tokens]
        y_labels = [bold_special_tokens(token) for token in decoder_tokens]
        if xlabel is None:
            xlabel = "Encoder Tokens"
        if ylabel is None:
            ylabel = "Decoder Tokens"
        default_title = "Cross-Attention Heatmap (Decoder to Encoder)"
        expected_shape = (len(decoder_tokens), len(encoder_tokens))
        save_path = "cross_attention_heatmap.pdf"

    elif use_case == 'encoder_self_attention':
        # Encoder Self-Attention
        x_labels = [bold_special_tokens(token) for token in encoder_tokens]
        y_labels = [bold_special_tokens(token) for token in encoder_tokens]
        if xlabel is None:
            xlabel = "Encoder Tokens"
        if ylabel is None:
            ylabel = "Encoder Tokens"
        default_title = "Encoder Self-Attention Heatmap"
        expected_shape = (len(encoder_tokens), len(encoder_tokens))
        save_path = "encoder_self_attention_heatmap.pdf"

    elif use_case == 'decoder_self_attention':
        # Decoder Self-Attention
        x_labels = [bold_special_tokens(token) for token in decoder_tokens]
        y_labels = [bold_special_tokens(token) for token in decoder_tokens]
        if xlabel is None:
            xlabel = "Decoder Tokens"
        if ylabel is None:
            ylabel = "Decoder Tokens"
        default_title = "Decoder Self-Attention Heatmap"
        expected_shape = (len(decoder_tokens), len(decoder_tokens))
        save_path = "decoder_self_attention_heatmap.pdf"
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
        right_bottom_cells=right_bottom_cells,
        lean_more=lean_more
    )

    plt.tight_layout()
    if save_path is None:
        save_path = "attention_heatmap.pdf"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Attention heatmap saved to {save_path}")

# Helper function to generate sparse labels
def generate_sparse_labels(tokens, top_cells, axis, interval=10, if_interval=True, if_top_cells=True):
    """
    Generate sparse labels for token lists, showing only:
    1. Labels at regular intervals
    2. Labels for tokens associated with important attention cells
    
    Parameters:
    - tokens: List of token labels
    - top_cells: List of (row, col) tuples of important cells
    - axis: Which axis the labels are for (0 for rows/y-axis, 1 for columns/x-axis)
    - interval: Show a label every N tokens
    
    Returns:
    - List of labels, with empty strings for positions without labels
    """
    # Create an array of empty strings
    sparse_labels = [""] * len(tokens)
    
    if if_interval:
        # Add regular interval labels (token indices)
        for i in range(0, len(tokens), interval):
            if i < len(tokens):
                sparse_labels[i] = f"{i}"
    if if_top_cells:
        # Add labels for top cells
        for row, col in top_cells:
            idx = col if axis == 1 else row
            if 0 <= idx < len(tokens):
                sparse_labels[idx] = bold_special_tokens(tokens[idx])
    
    return sparse_labels

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
    gamma=1.5,
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

def compare_two_attentions_with_circles(attn1, attn2, tokens, title="Comparison with Circles", 
                                        xlabel=None, ylabel=None, save_path=None, 
                                        circle_scale=1.0, gamma=1.5, cmap="Blues", max_circle_ratio=0.45):
    """
    Compares two attention matrices by showing the first matrix as background colors
    and the second matrix as circles with varying sizes based on their differences.
    
    Parameters:
    - attn1: First attention matrix (used for background colors)
    - attn2: Second attention matrix (used for circle colors)
    - tokens: List of token labels for x/y axes
    - title: Title for the plot
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - save_path: File path to save the generated heatmap PDF
    - circle_scale: Scale factor for circle sizes (default: 1.0)
    - gamma: Gamma value for the power normalization of the colormap (default: 1.5)
    - cmap: Colormap to use (default: 'Blues')
    - max_circle_ratio : float, default=0.45
        Maximum radius of a circle as a fraction of half-cell width. Values < 0.5
        ensure circles don't completely fill the cell.
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
        xlabel=xlabel,
        ylabel=ylabel,
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
            radius = min(circle_scale * max_circle_ratio * (diff[i, j] / max_diff), 0.5)
            
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



# Ensure you have your create_tablelens_heatmap imported or defined as in your code
# from .visualization import create_tablelens_heatmap  # or adjust import path

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
    radial_resolution=100,   # Resolution of the radial gradient image
    use_white_center=True,   # If True, use white at center instead of (mean-err) color
    color_contrast_scale=2.0, # Factor to enhance contrast between inner and outer colors
    max_circle_ratio=0.45    # Maximum circle radius as a fraction of half-cell width (was 0.5)
):
    """
    Plots an n-run stability heatmap:

      1) Background squares are colored by the mean attention score across n matrices
         (darker = higher mean, using 'Blues').
      2) Each cell has a circle whose radius is proportional to the "confidence interval"
         (e.g. std or SEM). A bigger interval => a bigger circle.
      3) The circle is filled with a *radial gradient*:
         - When use_white_center=False: The gradient goes from the color corresponding 
           to the cell's 'lower bound' (mean - err*color_contrast_scale) in the center,
           to the color of the 'upper bound' (mean + err*color_contrast_scale) at the edge,
           creating enhanced color contrast between center and edge.
         - When use_white_center=True: The gradient goes from white in the center
           to the color of the 'upper bound' (mean + err) at the edge.
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
    use_white_center : bool
        If True, use white at center instead of (mean-err) color.
    color_contrast_scale : float
        Factor to enhance contrast between inner and outer colors of the gradient.
    max_circle_ratio : float
        Maximum radius of a circle as a fraction of half-cell width. Values < 0.5
        ensure circles don't completely fill the cell (default: 0.45).
    
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
        norm=norm,
        lean_more=True
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
            # bigger error => bigger circle up to max_circle_ratio * circle_scale
            radius = (err / max_err) * max_circle_ratio * circle_scale

            # Find the lower/upper values for the gradient
            if use_white_center:
                # For white center, use normal error for upper bound
                val_lower = mean_vals[i, j]  # Not used with white center
                val_upper = mean_vals[i, j] + err
            else:
                # Apply contrast scaling when using color gradient from lower to upper
                val_lower = mean_vals[i, j] - (err * color_contrast_scale)
                val_upper = mean_vals[i, j] + (err * color_contrast_scale)

            # Clamp to [vmin, vmax]
            val_lower = max(val_lower, vmin)
            val_lower = min(val_lower, vmax)
            val_upper = max(val_upper, vmin)
            val_upper = min(val_upper, vmax)

            # Convert to RGBA
            cmap_obj = plt.get_cmap(cmap)
            
            # Use white at center if specified, otherwise use lower bound color
            if use_white_center:
                inner_rgba = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)  # White with full opacity
            else:
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


def visualize_attention_evolution_sparklines(
    attentions_over_time,  # List/array of shape [n_epochs, ..., n_tokens, n_tokens]
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
):
    """
    Visualize the evolution of attention matrices over training epochs with sparklines.
    
    Args:
        attentions_over_time: Numpy array with shape [n_epochs, layers, heads, n_tokens, n_tokens]
        tokens: List of token labels (optional)
        layer: Layer index to extract (if needed)
        head: Head index to extract (if needed)
        title: Plot title
        xlabel, ylabel: Axis labels
        figsize: Figure size
        sparkline_color_dark: Dark color for the sparklines
        sparkline_color_light: Light color for the sparklines
        sparkline_linewidth: Width of sparkline
        sparkline_alpha: Transparency of sparklines
        gamma: For PowerNorm color scaling
        normalize_sparklines: Whether to normalize sparklines
        save_path: Path to save the figure
        
    Returns:
        matplotlib.axes.Axes: The axes containing the visualization
    """
    # Convert input to numpy array if it's not already
    if not isinstance(attentions_over_time, np.ndarray):
        try:
            # Try converting to numpy array
            if torch.is_tensor(attentions_over_time):
                attentions_over_time = attentions_over_time.detach().cpu().numpy()
            else:
                attentions_over_time = np.array(attentions_over_time)
            print(f"Converted input to numpy array with shape {attentions_over_time.shape}")
        except Exception as e:
            raise ValueError(f"Failed to convert input to numpy array: {str(e)}")
    
    # Validate dimensions after conversion
    if attentions_over_time.ndim != 5:
        raise ValueError(f"Expected attentions_over_time to have 5 dimensions [n_epochs, layers, heads, n_tokens, n_tokens], "
                         f"but got shape {attentions_over_time.shape}")
    
    # Process the attention matrices
    matrices = []
    for epoch_attn in attentions_over_time:
        # Extract layer and head
        if layer is None or head is None:
            raise ValueError("Both layer and head must be specified")
        
        attn = epoch_attn[layer][head]
        matrices.append(attn)
    
    # Stack matrices for easier processing
    attention_stack = np.stack(matrices)  # [n_epochs, n_tokens, n_tokens]
    n_epochs, n_tokens, _ = attention_stack.shape
    
    # Compute average attention for background color
    avg_attention = np.mean(attention_stack, axis=0)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create background heatmap with average attention
    # norm = PowerNorm(gamma=gamma)
    # im = ax.imshow(avg_attention, cmap=cmap, alpha=background_alpha, norm=norm)
    
    # # Add grid lines
    # ax.set_xticks(np.arange(-.5, n_tokens, 1), minor=True)
    # ax.set_yticks(np.arange(-.5, n_tokens, 1), minor=True)
    # ax.grid(which="minor", color=grid_color, linestyle='-', linewidth=grid_linewidth)
    # ax.tick_params(which="minor", size=0)
    min_val = avg_attention.min()
    max_val = avg_attention.max()
    norm = PowerNorm(gamma=1.5, vmin=min_val, vmax=max_val)

    x_labels = [bold_special_tokens(token) for token in tokens]
    y_labels = [bold_special_tokens(token) for token in tokens]

    ax, plotter = create_tablelens_heatmap(
        avg_attention,
        x_labels,
        y_labels,
        title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        vmin=min_val,
        vmax=max_val,
        norm=norm,
        gamma=gamma
    )
    
    # Get cell centers directly from plotter
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers
    
    # # Get the actual cell positions from the heatmap
    # # This fixes the coordinate mismatch between tablelens and sparklines
    # cells = ax.collections[0]  # The heatmap cells collection
    # cell_positions = []
    
    # for i in range(n_tokens):
    #     row_positions = []
    #     for j in range(n_tokens):
    #         # Get the path for this cell
    #         path = cells.get_paths()[i*n_tokens + j]
    #         # Get the bounds of the cell (xmin, ymin, xmax, ymax)
    #         cell_rect = path.get_extents()
    #         # Calculate cell center and dimensions
    #         x_center = (cell_rect.x0 + cell_rect.x1) / 2
    #         y_center = (cell_rect.y0 + cell_rect.y1) / 2
    #         width = cell_rect.width
    #         height = cell_rect.height
    #         row_positions.append((x_center, y_center, width, height))
    #     cell_positions.append(row_positions)
    
    # # Add sparklines in each cell
    # cell_height = 1.0
    # cell_width = 1.0
    
    # Function to interpolate between two colors
    def get_sparkline_color(cell_intensity):
        """Return either dark blue or white based on background intensity relative to color bar midpoint."""
        # Calculate the middle of the color range (with PowerNorm influence)
        norm_tmp = PowerNorm(gamma=1.5, vmin=min_val, vmax=max_val)
        middle_value = norm_tmp.inverse(0.5)
        
        # Compare the raw attention value to the middle value
        return sparkline_color_light if cell_intensity > middle_value else sparkline_color_dark
    
    # For global normalization (if not normalizing per cell), find global min/max
    if not normalize_sparklines:
        global_min = attention_stack.min()
        global_max = attention_stack.max()
    
    # Draw sparklines using row_centers and col_centers
    for i in range(n_tokens):
        for j in range(n_tokens):
            # Get time series for this cell
            values = attention_stack[:, i, j]
            
            # Get cell centers
            y_center = row_centers[i]
            x_center = col_centers[j]
            
            # Estimate cell dimensions based on spacing between centers
            width = col_centers[1] - col_centers[0] if len(col_centers) > 1 else 1.0
            height = row_centers[1] - row_centers[0] if len(row_centers) > 1 else 1.0
            
            # Normalize values as before
            if normalize_sparklines:
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:  # Avoid division by zero
                    norm_values = (values - min_val) / (max_val - min_val)
                else:
                    norm_values = np.ones_like(values) * 0.5
            else:
                if global_max > global_min:
                    norm_values = (values - global_min) / (global_max - global_min)
                else:
                    norm_values = np.ones_like(values) * 0.5
            
            # Create x-coordinates centered in the cell
            x = np.linspace(x_center - width*0.4, x_center + width*0.4, n_epochs)
            
            # Calculate y-coordinates (with the correct orientation)
            y = y_center - (norm_values - 0.5) * height * 0.7
            
            # Determine color and plot sparkline
            cell_intensity = avg_attention[i, j]
            sparkline_color = get_sparkline_color(cell_intensity)
            ax.plot(x, y, color=sparkline_color, linewidth=sparkline_linewidth, alpha=sparkline_alpha)

    
    # for i in range(n_tokens):
    #     for j in range(n_tokens):
    #         # Get time series for this cell
    #         values = attention_stack[:, i, j]
            
    #         # Normalize based on user preference
    #         if normalize_sparklines:
    #             # Per-cell normalization (original behavior)
    #             min_val, max_val = values.min(), values.max()
    #             if max_val > min_val:  # Avoid division by zero
    #                 norm_values = (values - min_val) / (max_val - min_val)
    #             else:
    #                 norm_values = np.ones_like(values) * 0.5
    #         else:
    #             # Global normalization (all cells share same y-axis scale)
    #             if global_max > global_min:  # Avoid division by zero
    #                 norm_values = (values - global_min) / (global_max - global_min)
    #             else:
    #                 norm_values = np.ones_like(values) * 0.5
            
    #         # Create x-coordinates for the time steps
    #         x = np.linspace(j - cell_width/2 + 0.1, j + cell_width/2 - 0.1, n_epochs)
            
    #         # Calculate y-coordinates: invert normalized values to plot within cell
    #         y = i + (1 - norm_values) * cell_height * 0.8 - cell_height * 0.4
            
    #         # Determine color based on background intensity
    #         cell_intensity = avg_attention[i, j]
    #         sparkline_color = get_sparkline_color(cell_intensity)
            
    #         # Plot the sparkline
    #         ax.plot(x, y, color=sparkline_color, linewidth=sparkline_linewidth, alpha=sparkline_alpha)

    # # Now use actual cell positions for sparklines
    # for i in range(n_tokens):
    #     for j in range(n_tokens):
    #         # Get time series for this cell
    #         values = attention_stack[:, i, j]
            
    #         # Get actual cell position and dimensions
    #         x_center, y_center, width, height = cell_positions[i][j]
            
    #         # Normalize values as before
    #         if normalize_sparklines:
    #             # Per-cell normalization
    #             min_val, max_val = values.min(), values.max()
    #             if max_val > min_val:  # Avoid division by zero
    #                 norm_values = (values - min_val) / (max_val - min_val)
    #             else:
    #                 norm_values = np.ones_like(values) * 0.5
    #         else:
    #             # Global normalization
    #             if global_max > global_min:
    #                 norm_values = (values - global_min) / (global_max - global_min)
    #             else:
    #                 norm_values = np.ones_like(values) * 0.5
            
    #         # Create x-coordinates centered in the actual cell
    #         x = np.linspace(x_center - width*0.4, x_center + width*0.4, n_epochs)
            
    #         # Calculate y-coordinates using actual cell dimensions
    #         y = y_center - (norm_values - 0.5) * height * 0.7
            
    #         # Determine color and plot sparkline
    #         cell_intensity = avg_attention[i, j]
    #         sparkline_color = get_sparkline_color(cell_intensity)
    #         ax.plot(x, y, color=sparkline_color, linewidth=sparkline_linewidth, alpha=sparkline_alpha)
    
    # # Set token labels if provided
    # if tokens is not None:
    #     if len(tokens) != n_tokens:
    #         print(f"Warning: tokens list length ({len(tokens)}) doesn't match matrix dimensions ({n_tokens})")
    #         # Truncate or pad tokens list as needed
    #         tokens = tokens[:n_tokens] if len(tokens) > n_tokens else tokens + [""] * (n_tokens - len(tokens))
        
    #     # Create formatted labels (escape special chars, etc.)
    #     formatted_tokens = []
    #     for token in tokens:
    #         if token.startswith("<") and token.endswith(">"):
    #             token = r"$\bf{" + token + "}$"  # Make special tokens bold
    #         formatted_tokens.append(token)
        
        # Set labels
        # ax.set_xticks(np.arange(n_tokens))
        # ax.set_yticks(np.arange(n_tokens))
        # ax.set_xticklabels(formatted_tokens)
        # ax.set_yticklabels(formatted_tokens)
    
    # Set title and labels
    # ax.set_title(title, fontsize=title_fontsize)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()

    # for label in ax.get_xticklabels():
    #     label.set_rotation(45)
    
    # Add colorbar for the background
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(im, cax=cax)
    # cbar.outline.set_visible(False)
    # cbar.set_label("Average Attention")
    
    # Update legend - show both dark and light sparkline colors
    legend_elements = [
        Line2D([0], [0], color=sparkline_color_dark, lw=sparkline_linewidth, 
               label="Trend (low attention)"),
        Line2D([0], [0], color=sparkline_color_light, lw=sparkline_linewidth, 
               label="Trend (high attention)")
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, -0.1))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    return ax




from collections import deque

def find_attention_regions_with_merging(attention_matrix, n_seeds=3, min_distance=2, 
                                        expansion_threshold=0.8, merge_std_threshold=0.8,
                                        proximity_threshold=2, max_expansion_steps=3):
    """
    Find rectangular regions of high attention in an attention matrix with intelligent merging.
    
    Parameters:
        attention_matrix: 2D numpy array of attention scores
        n_seeds: Number of seed points to start with and final number of rectangles
        min_distance: Minimum distance between seed points
        expansion_threshold: Threshold for region expansion (ratio of rectangle avg to boundary avg)
        merge_std_threshold: Threshold ratio of merged std dev / avg individual std dev (lower = stricter)
        proximity_threshold: Maximum distance between rectangles to consider merging (even if not overlapping)
        max_expansion_steps: Maximum number of steps to look ahead for expansion in each direction
        
    Returns:
        List of (top, left, bottom, right) tuples representing rectangles, sorted by highest attention inside
    """
    rows, cols = attention_matrix.shape
    
    # Step 1: Find potential seed positions with high attention scores
    # We'll get more than needed to have reserves for replacements after merging
    potential_seeds = []
    flat_indices = np.argsort(attention_matrix.flatten())[::-1]  # Indices sorted by decreasing value
    
    for idx in flat_indices:
        r, c = idx // cols, idx % cols
        
        # Check if this seed is far enough from existing seeds
        valid_seed = True
        for seed_r, seed_c in potential_seeds:
            if abs(seed_r - r) <= min_distance and abs(seed_c - c) <= min_distance:
                valid_seed = False
                break
                
        if valid_seed:
            potential_seeds.append((r, c))
            if len(potential_seeds) >= n_seeds * 3:  # Get 3x more seeds than needed as reserve
                break
    
    # Start with the first n_seeds
    active_seeds = potential_seeds[:n_seeds]
    seed_queue = deque(potential_seeds[n_seeds:])
    
    # Step 2: Grow rectangles from active seeds
    rectangles = [(r, c, r, c) for r, c in active_seeds]  # (top, left, bottom, right)
    rectangle_stats = [calculate_rectangle_stats(attention_matrix, rect) for rect in rectangles]
    
    # Main expansion loop
    iteration = 0
    max_iterations = 1000  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        
        # Flag to track if any expansion or merging happened this round
        any_change = False
        
        # Try expanding each rectangle
        for i in range(len(rectangles)):
            # Skip if this rectangle was already merged
            if rectangles[i] is None:
                continue
                
            top, left, bottom, right = rectangles[i]
            
            # Base directions for expansion
            base_directions = [
                (-1, 0, 0, 0),  # Top
                (0, -1, 0, 0),  # Left
                (0, 0, 1, 0),   # Bottom
                (0, 0, 0, 1)    # Right
            ]
            
            best_rect = rectangles[i]
            best_score = calculate_expansion_score(attention_matrix, best_rect)
            best_expansion = None
            
            # For each direction, try different step sizes
            for direction in base_directions:
                d_top, d_left, d_bottom, d_right = direction
                
                # Try expansions of 1, 2, and 3 steps
                for steps in range(1, max_expansion_steps + 1):
                    # Calculate the expanded rectangle coordinates
                    new_top = max(0, top + d_top * steps)
                    new_left = max(0, left + d_left * steps)
                    new_bottom = min(rows - 1, bottom + d_bottom * steps)
                    new_right = min(cols - 1, right + d_right * steps)
                    
                    # Skip if no change
                    if (new_top, new_left, new_bottom, new_right) == rectangles[i]:
                        continue
                    
                    new_rect = (new_top, new_left, new_bottom, new_right)
                    new_score = calculate_expansion_score(attention_matrix, new_rect)
                    
                    # Check if this expansion improves score beyond threshold
                    if new_score > best_score * expansion_threshold:
                        best_rect = new_rect
                        best_score = new_score
                        best_expansion = (direction, steps)
            
            # If we found a better rectangle, check for overlaps
            if best_expansion is not None:
                # Check for overlaps and nearby rectangles with the best expanded rectangle
                overlaps_with = []
                nearby = []
                
                for j, rect in enumerate(rectangles):
                    if j != i and rect is not None:
                        if rectangles_overlap(best_rect, rect):
                            overlaps_with.append(j)
                        elif rectangles_nearby(best_rect, rect, proximity_threshold):
                            nearby.append(j)
                
                # Combine overlapping and nearby rectangles for potential merging
                potential_merges = overlaps_with + nearby
                
                if not potential_merges:
                    # No overlaps or nearby rectangles, proceed with expansion
                    rectangles[i] = best_rect
                    rectangle_stats[i] = calculate_rectangle_stats(attention_matrix, best_rect)
                    any_change = True
                else:
                    # There's overlap or nearby rectangles - evaluate whether to merge
                    can_merge = True
                    for j in potential_merges:
                        if not should_merge_rectangles(attention_matrix, rectangles[i], rectangles[j], 
                                                    rectangle_stats[i], rectangle_stats[j], 
                                                    merge_std_threshold):
                            can_merge = False
                            break
                    
                    if can_merge:
                        # Merge rectangles
                        merged_rect = merge_rectangles([rectangles[i]] + [rectangles[j] for j in potential_merges])
                        merged_stats = calculate_rectangle_stats(attention_matrix, merged_rect)
                        
                        # Update the current rectangle with merged one
                        rectangles[i] = merged_rect
                        rectangle_stats[i] = merged_stats
                        
                        # Mark the other rectangles as merged (None)
                        for j in potential_merges:
                            rectangles[j] = None
                            rectangle_stats[j] = None
                        
                        # Get new seeds for the merged rectangles
                        for _ in range(len(potential_merges)):
                            if seed_queue:
                                new_seed = seed_queue.popleft()
                                new_rect = (new_seed[0], new_seed[1], new_seed[0], new_seed[1])
                                
                                # Find the first None position to replace
                                for k in range(len(rectangles)):
                                    if rectangles[k] is None:
                                        rectangles[k] = new_rect
                                        rectangle_stats[k] = calculate_rectangle_stats(attention_matrix, new_rect)
                                        break
                        
                        any_change = True
                    # else: can't merge, so don't expand in this direction
        
        # If no changes happened this iteration, we're done
        if not any_change:
            break
    
    # Remove any None entries from rectangles (result of merging)
    rectangles = [rect for rect in rectangles if rect is not None]
    
    # If we still need more rectangles (could happen if we ran out of seeds)
    while len(rectangles) < n_seeds:
        if not seed_queue:
            # No more seeds available
            break
            
        new_seed = seed_queue.popleft()
        new_rect = (new_seed[0], new_seed[1], new_seed[0], new_seed[1])
        rectangles.append(new_rect)
    
    # Sort rectangles by average attention value (from highest to lowest)
    rectangle_scores = []
    for rect in rectangles:
        stats = calculate_rectangle_stats(attention_matrix, rect)
        rectangle_scores.append((rect, stats['mean']))
    
    # Sort by the mean attention score in descending order
    rectangle_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the rectangles in the new sorted order
    sorted_rectangles = [rect for rect, score in rectangle_scores]
    
    return sorted_rectangles


def calculate_rectangle_stats(matrix, rect):
    """
    Calculate statistics for rectangle area.
    
    Parameters:
        matrix: The attention matrix
        rect: Tuple (top, left, bottom, right)
        
    Returns:
        Dict with mean, std, and sum of the rectangle area
    """
    top, left, bottom, right = rect
    rectangle = matrix[top:bottom+1, left:right+1]
    
    return {
        'mean': np.mean(rectangle),
        'std': np.std(rectangle),
        'sum': np.sum(rectangle),
        'size': rectangle.size
    }


def calculate_expansion_score(matrix, rect):
    """
    Calculate a score for a rectangle based on average attention inside vs. boundary.
    
    Parameters:
        matrix: Attention matrix
        rect: Tuple (top, left, bottom, right)
        
    Returns:
        A score value (higher is better)
    """
    top, left, bottom, right = rect
    rows, cols = matrix.shape
    
    # Extract rectangle
    rectangle = matrix[top:bottom+1, left:right+1]
    avg_inside = np.mean(rectangle)
    
    # Calculate boundary (1-cell wide) around rectangle
    boundary_cells = []
    
    # Top and bottom boundaries
    if top > 0:
        boundary_cells.extend(matrix[top-1, max(0, left-1):min(cols, right+2)].flatten())
    if bottom < rows - 1:
        boundary_cells.extend(matrix[bottom+1, max(0, left-1):min(cols, right+2)].flatten())
        
    # Left and right boundaries (excluding corners already counted)
    if left > 0:
        boundary_cells.extend(matrix[top:bottom+1, left-1].flatten())
    if right < cols - 1:
        boundary_cells.extend(matrix[top:bottom+1, right+1].flatten())
    
    # Handle case where rectangle is at edge
    if len(boundary_cells) == 0:
        avg_boundary = 0
    else:
        avg_boundary = np.mean(boundary_cells)
    
    # Score is ratio of inside vs boundary, adjusted by rectangle size
    # This rewards larger rectangles when scores are similar
    rect_size = (bottom - top + 1) * (right - left + 1)
    size_factor = np.log1p(rect_size) / 10  # Log to prevent too much size bias
    
    if avg_boundary == 0:
        score = avg_inside * (1 + size_factor)
    else:
        contrast = avg_inside / avg_boundary
        score = avg_inside * contrast * (1 + size_factor)
    
    return score


def rectangles_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.
    
    Parameters:
        rect1, rect2: Tuples (top, left, bottom, right)
        
    Returns:
        Boolean indicating whether the rectangles overlap
    """
    top1, left1, bottom1, right1 = rect1
    top2, left2, bottom2, right2 = rect2
    
    # Check for non-overlap conditions
    if right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1:
        return False
    
    return True


def should_merge_rectangles(matrix, rect1, rect2, stats1, stats2, merge_threshold):
    """
    Determine if two rectangles should be merged based on standard deviation change.
    
    Parameters:
        matrix: Attention matrix
        rect1, rect2: Tuples (top, left, bottom, right)
        stats1, stats2: Dictionaries with statistics for each rectangle
        merge_threshold: Threshold ratio for acceptable std dev increase
        
    Returns:
        Boolean indicating whether the rectangles should be merged
    """
    # Calculate the merged rectangle
    merged_rect = merge_rectangles([rect1, rect2])
    merged_stats = calculate_rectangle_stats(matrix, merged_rect)
    
    # Calculate weighted average of individual standard deviations
    total_size = stats1['size'] + stats2['size']
    weighted_std = (stats1['std'] * stats1['size'] + stats2['std'] * stats2['size']) / total_size
    
    # Calculate ratio of merged std dev to weighted individual std devs
    std_ratio = merged_stats['std'] / weighted_std if weighted_std > 0 else float('inf')
    
    # Allow merging if std dev doesn't increase too much (ratio close to 1.0 or below)
    return std_ratio <= (1.0 / merge_threshold)  # Inverted so that merge_threshold < 1.0 is stricter


def merge_rectangles(rectangles):
    """
    Merge multiple rectangles into one larger rectangle that contains all of them.
    
    Parameters:
        rectangles: List of (top, left, bottom, right) tuples
        
    Returns:
        Tuple (top, left, bottom, right) for the merged rectangle
    """
    tops, lefts, bottoms, rights = zip(*rectangles)
    return (min(tops), min(lefts), max(bottoms), max(rights))


def visualize_attention_with_detected_regions(
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
):
    """
    Visualize attention matrix with automatically detected important regions.
    
    Parameters:
        attention_matrix: 2D numpy array of attention scores
        tokens: List of token labels for x/y axes
        title: Title for the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        n_regions: Number of regions to detect
        min_distance: Minimum distance between seed points
        expansion_threshold: Threshold for region expansion
        merge_std_threshold: Threshold for merging regions
        region_color: Color of the region outlines
        region_linewidth: Line width of region outlines
        region_alpha: Alpha/transparency of region outlines
        label_regions: Whether to add region labels
        gamma: Gamma value for the power normalization of the colormap
        save_path: File path to save the generated heatmap
        ax: Matplotlib axis to plot on (if None, create new)
        cmap: Colormap to use
        max_expansion_steps: Maximum number of steps to look ahead for expansion in each direction
        proximity_threshold: Maximum distance between rectangles to consider merging (even if not overlapping)
    
    Returns:
        Matplotlib axis with the plot
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(attention_matrix):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    
    # Create new figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create norm for the colormap
    vmin = attention_matrix.min()
    vmax = attention_matrix.max()
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    # Create the base heatmap
    ax, plotter = create_tablelens_heatmap(
        attention_matrix,
        x_labels=[bold_special_tokens(token) for token in source_tokens],
        y_labels=[bold_special_tokens(token) for token in target_tokens],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=cmap,
        norm=norm,
        gamma=gamma,
        vmax=vmax,
        vmin=vmin
    )
    
    # Find regions of interest
    rectangles = find_attention_regions_with_merging(
        attention_matrix, 
        n_seeds=n_regions, 
        min_distance=min_distance,
        expansion_threshold=expansion_threshold, 
        merge_std_threshold=merge_threshold,
        max_expansion_steps=max_expansion_steps,
        proximity_threshold=proximity_threshold
    )
    
    # Get the positions of the cell edges from the plotter
    col_positions = plotter.col_positions
    row_positions = plotter.row_positions
    
    # Add rectangle patches for each detected region
    for i, (top, left, bottom, right) in enumerate(rectangles):
        # Compute the rectangle's position and size using the actual cell positions
        x = col_positions[left]
        width = col_positions[right + 1] - col_positions[left]
        y = row_positions[top]
        height = row_positions[bottom + 1] - row_positions[top]
        
        # Create rectangle with correct positioning
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=region_linewidth, 
            edgecolor=region_color, 
            facecolor='none',
            alpha=region_alpha,
            zorder=10  # Ensure rectangle is drawn on top
        )
        ax.add_patch(rect)
        
        # Add region label if requested
        if label_regions:
            ax.text(
                x + width/2,
                y + height/2,
                f"R{i+1}",
                color='white', 
                fontweight='bold', 
                ha='center', 
                va='center',
                bbox=dict(facecolor=region_color, alpha=0.5, boxstyle='round'),
                zorder=11
            )
    
    # Save if requested
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Attention heatmap with detected regions saved to {save_path}")
    
    return ax

def rectangles_nearby(rect1, rect2, threshold):
    """
    Check if two rectangles are within the specified distance threshold of each other.
    
    Parameters:
        rect1, rect2: Tuples (top, left, bottom, right)
        threshold: Maximum distance between rectangles to consider them nearby
        
    Returns:
        Boolean indicating whether the rectangles are nearby
    """
    top1, left1, bottom1, right1 = rect1
    top2, left2, bottom2, right2 = rect2
    
    # Calculate horizontal distance (positive if separated, negative if overlapping)
    h_dist = max(0, max(left1, left2) - min(right1, right2))
    
    # Calculate vertical distance (positive if separated, negative if overlapping)
    v_dist = max(0, max(top1, top2) - min(bottom1, bottom2))
    
    # Rectangles are nearby if both horizontal and vertical distances are within threshold
    return h_dist <= threshold and v_dist <= threshold