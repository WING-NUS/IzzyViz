from .my_seaborn import heatmap
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

THEME_CMAP = "Purples"
THEME_POSITIVE = "#C77DF3"
THEME_NEGATIVE = "#6A0DAD"


# Make special tokens bold
def bold_special_tokens(label):
    special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
    if label in special_tokens:
        return f"$\mathbf{{{label}}}$"  # Make it bold using LaTeX math formatting
    return label


def create_tablelens_heatmap(
    attention_matrix,
    x_labels,
    y_labels,
    title,
    xlabel,
    ylabel,
    ax,
    cmap=THEME_CMAP,
    column_widths=None,
    row_heights=None,
    top_cells=None,
    vmin=None,
    vmax=None,
    norm=None,
    left_top_cells=None,
    right_bottom_cells=None,
    linecolor="white",
    linewidths=1.0,
    cbar=True,
    show_scores=True,
    background_color=True,
    lean_more=False,
):
    """
    Creates a heatmap with variable cell sizes and annotations for top cells.
    Returns both the axis and the plotter object for further customization.
    """

    if isinstance(attention_matrix, np.ndarray):
        data = attention_matrix
    else:
        data = attention_matrix.detach().cpu().numpy()

    if show_scores:
        annot_data = np.empty_like(data, dtype=object)
        annot_data[:] = ""

        if top_cells is not None:
            for row_index, col_index in top_cells:
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

    ax, plotter = heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        linewidths=linewidths,
        linecolor=linecolor,
        square=True,  # Ensure non-highlighted cells are square
        cbar=False,  # Disable the default colorbar
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        ax=ax,
        column_widths=column_widths,
        row_heights=row_heights,
        annot=annot_data,
        fmt="",
    )

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
        num_ticks = 7
        tick_values = np.linspace(vmin, vmax, num_ticks)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for label in ax.get_xticklabels():
        if lean_more:
            label.set_rotation(90)
        else:
            label.set_rotation(45)

    for label in ax.get_yticklabels():
        label.set_rotation(0)

    ax.set_title(title, fontsize=14, fontname="DejaVu Serif", fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=15)

    if top_cells is not None:
        # Highlight tick labels corresponding to top_cells
        x_ticklabels = ax.get_xticklabels()
        y_ticklabels = ax.get_yticklabels()

        x_indices = set(col_index for (row_index, col_index) in top_cells)
        y_indices = set(row_index for (row_index, col_index) in top_cells)

        # Adjust x tick labels
        for idx, label in enumerate(x_ticklabels):
            if idx in x_indices and background_color:
                label.set_bbox(
                    dict(
                        facecolor="#f8bbd0",
                        edgecolor="#f8bbd0",
                        boxstyle="round,pad=0.2",
                        alpha=0.5,
                    )
                )

        # Adjust y tick labels without inversion
        for row_index in y_indices:
            if row_index < len(y_ticklabels) and background_color:
                label = y_ticklabels[row_index]
                label.set_bbox(
                    dict(
                        facecolor="#f8bbd0",
                        edgecolor="#f8bbd0",
                        boxstyle="round,pad=0.2",
                        alpha=0.5,
                    )
                )

    # Draw rectangles around the specified regions
    if left_top_cells is not None and right_bottom_cells is not None:
        if len(left_top_cells) != len(right_bottom_cells):
            raise ValueError(
                "left_top_cells and right_bottom_cells must have the same length."
            )

        for lt_cell, rb_cell in zip(left_top_cells, right_bottom_cells):
            lt_row, lt_col = lt_cell
            rb_row, rb_col = rb_cell

            if lt_row > rb_row or lt_col > rb_col:
                print("lt_row: ", lt_row)
                print("rb_row: ", rb_row)
                print("lt_col: ", lt_col)
                print("rb_col: ", rb_col)
                raise ValueError(
                    "Invalid cell coordinates. Left-top cell must be above and to the left of the right-bottom cell."
                )

            if (
                lt_row < 0
                or lt_col < 0
                or rb_row < 0
                or rb_col < 0
                or rb_row >= data.shape[0]
                or rb_col >= data.shape[1]
                or lt_row >= data.shape[0]
                or lt_col >= data.shape[1]
            ):
                raise ValueError(
                    "Invalid cell coordinates. Coordinates must be within the attention matrix."
                )

            # Get the positions of the cell edges
            col_positions = plotter.col_positions
            row_positions = plotter.row_positions

            # Compute the rectangle's position and size
            x = col_positions[lt_col]
            width = col_positions[rb_col + 1] - col_positions[lt_col]
            y = row_positions[lt_row]
            height = row_positions[rb_row + 1] - row_positions[lt_row]

            # Draw the rectangle
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=3,
                edgecolor="#CE93D8",
                facecolor="none",
                linestyle=":",
            )
            ax.add_patch(rect)

    return ax, plotter


def _virtual_token_extent(count):
    return len(str(count))


def _build_axis_groups(axis_length, important_indices, min_run):
    important_indices = set(important_indices)
    groups = []
    index_to_group = {}
    i = 0

    while i < axis_length:
        if i in important_indices:
            group_index = len(groups)
            groups.append({"indices": [i], "is_virtual": False})
            index_to_group[i] = group_index
            i += 1
            continue

        start = i
        while i < axis_length and i not in important_indices:
            i += 1

        run_indices = list(range(start, i))
        if len(run_indices) >= min_run:
            group_index = len(groups)
            groups.append({"indices": run_indices, "is_virtual": True})
            for idx in run_indices:
                index_to_group[idx] = group_index
        else:
            for idx in run_indices:
                group_index = len(groups)
                groups.append({"indices": [idx], "is_virtual": False})
                index_to_group[idx] = group_index

    return groups, index_to_group


def _labels_and_extents_for_axis_groups(labels, groups):
    grouped_labels = []
    extents = []
    virtual_indices = set()

    for group in groups:
        group_indices = group["indices"]
        if group["is_virtual"]:
            count = len(group_indices)
            virtual_indices.add(len(grouped_labels))
            grouped_labels.append(str(count))
            extents.append(_virtual_token_extent(count))
        else:
            grouped_labels.append(labels[group_indices[0]])
            extents.append(1)

    return grouped_labels, extents, virtual_indices


def _aggregate_matrix_by_axis_groups(data, row_groups, col_groups):
    compressed = np.empty((len(row_groups), len(col_groups)), dtype=float)

    for row_index, row_group in enumerate(row_groups):
        row_indices = row_group["indices"]
        for col_index, col_group in enumerate(col_groups):
            col_indices = col_group["indices"]
            compressed[row_index, col_index] = data[
                np.ix_(row_indices, col_indices)
            ].mean()

    return compressed


def _map_cells_to_axis_groups(cells, row_index_to_group, col_index_to_group):
    mapped_cells = []
    seen = set()

    for row_index, col_index in cells:
        mapped = (row_index_to_group[row_index], col_index_to_group[col_index])
        if mapped not in seen:
            mapped_cells.append(mapped)
            seen.add(mapped)

    return mapped_cells


def _map_region_cells_to_axis_groups(cells, row_index_to_group, col_index_to_group):
    if cells is None:
        return None

    mapped_cells = []
    for row_index, col_index in cells:
        if row_index not in row_index_to_group or col_index not in col_index_to_group:
            raise ValueError(
                "Invalid cell coordinates. Coordinates must be within the attention matrix."
            )
        mapped_cells.append(
            (row_index_to_group[row_index], col_index_to_group[col_index])
        )

    return mapped_cells


def _style_virtual_tick_labels(
    ax,
    virtual_x_indices,
    virtual_y_indices,
    x_tick_indices,
    y_tick_indices,
    color,
):
    virtual_x_indices = set(virtual_x_indices)
    virtual_y_indices = set(virtual_y_indices)

    for label, tick_index in zip(ax.get_xticklabels(), x_tick_indices):
        if tick_index in virtual_x_indices:
            label.set_color(color)
            label.set_alpha(0.75)
            label.set_fontstyle("italic")

    for label, tick_index in zip(ax.get_yticklabels(), y_tick_indices):
        if tick_index in virtual_y_indices:
            label.set_color(color)
            label.set_alpha(0.75)
            label.set_fontstyle("italic")


def visualize_attention_matrix(
    matrix,
    x_labels=None,
    y_labels=None,
    title="Attention Heat",
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    top_n=3,
    enlarged_size=1.8,
    gamma=1.5,
    cmap=THEME_CMAP,
    left_top_cells=None,
    right_bottom_cells=None,
    save_path=None,
    length_threshold=64,
    interval=10,
    if_interval=False,
    if_top_cells=True,
    show_scores_in_enlarged_cells=True,
    background_color=True,
    lean_more=False,
    merge_virtual_tokens=False,
    virtual_token_min_run=1,
    virtual_token_label_color="#9A9A9A",
    close_after_save=False,
    cbar=True,
    tight_layout=True,
    vmin=None,
    vmax=None,
    norm=None,
):
    """
    Visualize any 2D attention-like matrix.

    Parameters:
    - matrix: A 2D NumPy array or PyTorch tensor to visualize. Rows map to
      y_labels and columns map to x_labels.
    - x_labels: Labels for the matrix columns. If None, column indices are used.
    - y_labels: Labels for the matrix rows. If None, x_labels are reused for
      square matrices; otherwise row indices are used.
    - title: Title displayed above the heatmap.
    - xlabel: Label for the x-axis. In attention heatmaps, this usually represents
      tokens being attended to.
    - ylabel: Label for the y-axis. In attention heatmaps, this usually represents
      tokens attending.
    - ax: Matplotlib axes to draw on. If None, a new figure and axes are created.
    - top_n: Number of highest-value cells to highlight and optionally annotate.
    - enlarged_size: Width/height multiplier for rows and columns containing
      top cells.
    - gamma: Gamma value for PowerNorm color normalization.
    - cmap: Matplotlib colormap name or colormap object.
    - left_top_cells: Optional list of (row, col) coordinates for the top-left
      corners of rectangular regions to outline.
    - right_bottom_cells: Optional list of (row, col) coordinates for the
      bottom-right corners of rectangular regions to outline.
    - save_path: Optional path for saving the generated figure. If None, the
      figure is not saved automatically.
    - length_threshold: Label count above which sparse axis labeling is used.
    - interval: Interval for regular sparse labels when if_interval is True.
    - if_interval: If True in sparse mode, show labels at regular intervals.
    - if_top_cells: If True in sparse mode, show labels associated with top cells.
    - show_scores_in_enlarged_cells: If True, annotate top cells with their
      numeric values when labels are not sparse.
    - background_color: If True, highlight axis labels associated with top cells
      when labels are not sparse.
    - lean_more: If True, rotate x-axis labels by 90 degrees instead of
      45 degrees.
    - merge_virtual_tokens: If True, compress contiguous rows/columns that do
      not contain top cells into virtual tokens when the run length reaches
      virtual_token_min_run.
    - virtual_token_min_run: Minimum contiguous unimportant row/column count
      required before creating one virtual token. The default is 1, so all
      contiguous unimportant runs are compressed.
    - virtual_token_label_color: Color used for virtual-token count labels.
    - close_after_save: If True, close the created figure after saving.
    - cbar: If True, draw a colorbar for the heatmap.
    - tight_layout: If True, apply tight layout before returning/saving.
    - vmin: Optional lower bound for color normalization.
    - vmax: Optional upper bound for color normalization.
    - norm: Optional matplotlib normalization object. If provided, it overrides
      the default PowerNorm created from vmin/vmax.

    Returns:
    - ax: The Matplotlib axes containing the heatmap.
    - plotter: The internal heatmap plotter with row/column position metadata.
    """

    if torch.is_tensor(matrix):
        data = matrix.detach().cpu().numpy()
    else:
        data = np.asarray(matrix)

    if data.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {data.shape}")

    num_rows, num_cols = data.shape
    original_num_rows, original_num_cols = num_rows, num_cols

    if x_labels is None:
        x_labels = [str(i) for i in range(num_cols)]

    if y_labels is None:
        if num_rows == num_cols and len(x_labels) == num_cols:
            y_labels = x_labels
        else:
            y_labels = [str(i) for i in range(num_rows)]

    if len(x_labels) != num_cols:
        raise ValueError(
            f"len(x_labels) must match matrix columns: {len(x_labels)} != {num_cols}"
        )

    if len(y_labels) != num_rows:
        raise ValueError(
            f"len(y_labels) must match matrix rows: {len(y_labels)} != {num_rows}"
        )

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    if norm is None:
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    top_cells = find_top_cells(data, top_n)
    virtual_x_indices = set()
    virtual_y_indices = set()

    if merge_virtual_tokens:
        if virtual_token_min_run < 1:
            raise ValueError("virtual_token_min_run must be >= 1")

        top_rows = {row_index for row_index, _ in top_cells}
        top_cols = {col_index for _, col_index in top_cells}
        row_groups, row_index_to_group = _build_axis_groups(
            num_rows, top_rows, virtual_token_min_run
        )
        col_groups, col_index_to_group = _build_axis_groups(
            num_cols, top_cols, virtual_token_min_run
        )

        data = _aggregate_matrix_by_axis_groups(data, row_groups, col_groups)
        y_labels, row_heights, virtual_y_indices = _labels_and_extents_for_axis_groups(
            y_labels, row_groups
        )
        x_labels, column_widths, virtual_x_indices = (
            _labels_and_extents_for_axis_groups(x_labels, col_groups)
        )
        top_cells = _map_cells_to_axis_groups(
            top_cells, row_index_to_group, col_index_to_group
        )
        left_top_cells = _map_region_cells_to_axis_groups(
            left_top_cells, row_index_to_group, col_index_to_group
        )
        right_bottom_cells = _map_region_cells_to_axis_groups(
            right_bottom_cells, row_index_to_group, col_index_to_group
        )
        num_rows, num_cols = data.shape
    else:
        column_widths = [1] * num_cols
        row_heights = [1] * num_rows

    x_is_sparse = original_num_cols > length_threshold
    y_is_sparse = original_num_rows > length_threshold
    is_sparse = x_is_sparse or y_is_sparse

    if x_is_sparse:
        display_x_labels = generate_sparse_labels(
            x_labels,
            top_cells,
            axis=1,
            interval=interval,
            if_interval=if_interval,
            if_top_cells=if_top_cells,
        )
    else:
        display_x_labels = [bold_special_tokens(label) for label in x_labels]

    for idx in virtual_x_indices:
        display_x_labels[idx] = x_labels[idx]

    if y_is_sparse:
        display_y_labels = generate_sparse_labels(
            y_labels,
            top_cells,
            axis=0,
            interval=interval,
            if_interval=if_interval,
            if_top_cells=if_top_cells,
        )
    else:
        display_y_labels = [bold_special_tokens(label) for label in y_labels]

    for idx in virtual_y_indices:
        display_y_labels[idx] = y_labels[idx]

    for row_index, col_index in top_cells:
        column_widths[col_index] = enlarged_size
        row_heights[row_index] = enlarged_size

    show_scores = show_scores_in_enlarged_cells and not is_sparse
    use_background_color = background_color and not is_sparse

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    ax, plotter = create_tablelens_heatmap(
        data,
        display_x_labels,
        display_y_labels,
        title,
        xlabel,
        ylabel,
        ax,
        cmap=cmap,
        column_widths=column_widths,
        row_heights=row_heights,
        top_cells=top_cells,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        left_top_cells=left_top_cells,
        right_bottom_cells=right_bottom_cells,
        show_scores=show_scores,
        background_color=use_background_color,
        lean_more=lean_more,
        cbar=cbar,
    )

    if is_sparse:
        x_tick_indices = [i for i, label in enumerate(display_x_labels) if label]
        y_tick_indices = [i for i, label in enumerate(display_y_labels) if label]

        if x_tick_indices:
            x_positions = [
                plotter.col_positions[i]
                + (plotter.col_positions[i + 1] - plotter.col_positions[i]) / 2
                for i in x_tick_indices
            ]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [display_x_labels[i] for i in x_tick_indices],
                rotation=90 if lean_more else 45,
                ha="right",
            )
        else:
            x_tick_indices = []

        if y_tick_indices:
            y_positions = [
                plotter.row_positions[i]
                + (plotter.row_positions[i + 1] - plotter.row_positions[i]) / 2
                for i in y_tick_indices
            ]
            ax.set_yticks(y_positions)
            ax.set_yticklabels([display_y_labels[i] for i in y_tick_indices])
        else:
            y_tick_indices = []
    else:
        x_tick_indices = list(range(len(display_x_labels)))
        y_tick_indices = list(range(len(display_y_labels)))

    _style_virtual_tick_labels(
        ax,
        virtual_x_indices,
        virtual_y_indices,
        x_tick_indices,
        y_tick_indices,
        virtual_token_label_color,
    )

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        if close_after_save:
            plt.close(fig)

    return ax, plotter


def _attention_to_numpy(attention):
    if torch.is_tensor(attention):
        return attention.detach().cpu().numpy()
    return np.asarray(attention)


def _attention_layers_heads_to_numpy(attentions, batch_index=0):
    if isinstance(attentions, (list, tuple)):
        layer_arrays = []
        for layer_attention in attentions:
            layer_array = _attention_to_numpy(layer_attention)
            if layer_array.ndim == 4:
                if not 0 <= batch_index < layer_array.shape[0]:
                    raise ValueError(
                        f"batch_index {batch_index} is out of range for batch size {layer_array.shape[0]}"
                    )
                layer_array = layer_array[batch_index]
            if layer_array.ndim != 3:
                raise ValueError(
                    "Each layer attention must have shape (heads, rows, cols) "
                    "or (batch, heads, rows, cols)."
                )
            layer_arrays.append(layer_array)

        if not layer_arrays:
            raise ValueError("attentions must contain at least one layer.")

        return np.stack(layer_arrays, axis=0)

    attention_array = _attention_to_numpy(attentions)
    if attention_array.ndim == 5:
        if not 0 <= batch_index < attention_array.shape[1]:
            raise ValueError(
                f"batch_index {batch_index} is out of range for batch size {attention_array.shape[1]}"
            )
        return attention_array[:, batch_index]
    if attention_array.ndim == 4:
        return attention_array
    if attention_array.ndim == 3:
        return attention_array[np.newaxis, :]

    raise ValueError(
        "attentions must have shape (layers, heads, rows, cols), "
        "(layers, batch, heads, rows, cols), (heads, rows, cols), "
        "or be a list/tuple of per-layer attention tensors."
    )


def visualize_attention_overview(
    attentions,
    batch_index=0,
    title="Attention Overview",
    save_path=None,
    figsize=None,
    top_n=3,
    enlarged_size=1.8,
    gamma=1.5,
    cmap=THEME_CMAP,
    left_top_cells=None,
    right_bottom_cells=None,
    length_threshold=64,
    interval=10,
    if_interval=False,
    if_top_cells=True,
    show_scores_in_enlarged_cells=True,
    background_color=True,
    lean_more=False,
    merge_virtual_tokens=False,
    virtual_token_min_run=1,
    virtual_token_label_color="#9A9A9A",
    cbar=False,
    shared_color_scale=True,
    shared_cbar=True,
    shared_cbar_label="Attention Score",
    close_after_save=False,
):
    """
    Visualize an overview grid of all attention layers and heads.

    Rows are layers from top to bottom. Columns are heads from left to right.
    Each subplot is drawn by visualize_attention_matrix, with inner token labels
    hidden because overview cells are too small to read.

    Parameters:
    - attentions: HuggingFace-style attentions or an array/tensor. Supported
      shapes are (layers, batch, heads, rows, cols), (layers, heads, rows, cols),
      (heads, rows, cols), or a list/tuple of per-layer tensors shaped
      (batch, heads, rows, cols) or (heads, rows, cols).
    - batch_index: Batch item to visualize when attentions include a batch axis.
    - title: Overall figure title.
    - save_path: Optional path for saving the generated overview figure.
    - figsize: Optional matplotlib figure size. If None, a size is chosen from
      the layer/head grid dimensions.
    - top_n, enlarged_size, gamma, cmap, left_top_cells, right_bottom_cells,
      length_threshold, interval, if_interval, if_top_cells,
      show_scores_in_enlarged_cells, background_color, lean_more,
      merge_virtual_tokens, virtual_token_min_run, virtual_token_label_color,
      cbar: Passed to each visualize_attention_matrix call. Keep this False
      when using shared_cbar to avoid one colorbar per subplot.
    - shared_color_scale: If True, all subplots use the same global vmin/vmax.
      If False, each subplot uses its own color range.
    - shared_cbar: If True with shared_color_scale, draw one unified colorbar
      for the whole overview figure.
    - shared_cbar_label: Label for the unified overview colorbar.
    - close_after_save: If True, close the created figure after saving.

    Returns:
    - fig: The Matplotlib figure containing the overview.
    - axes: A 2D array of Matplotlib axes indexed as axes[layer, head].
    """

    attention_array = _attention_layers_heads_to_numpy(
        attentions, batch_index=batch_index
    )

    if attention_array.ndim != 4:
        raise ValueError(
            f"Expected attentions to resolve to 4D, got shape {attention_array.shape}"
        )

    num_layers, num_heads, _, _ = attention_array.shape
    shared_vmin = None
    shared_vmax = None
    shared_norm = None
    show_shared_cbar = shared_color_scale and shared_cbar

    if shared_color_scale:
        shared_vmin = attention_array.min()
        shared_vmax = attention_array.max()
        if np.isclose(shared_vmin, shared_vmax):
            shared_vmax = shared_vmin + 1e-9
        shared_norm = PowerNorm(gamma=gamma, vmin=shared_vmin, vmax=shared_vmax)

    if figsize is None:
        figsize = (max(2.2 * num_heads, 6), max(2.0 * num_layers, 4))

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=figsize,
        squeeze=False,
        constrained_layout=False,
    )

    for layer_index in range(num_layers):
        for head_index in range(num_heads):
            ax = axes[layer_index, head_index]
            matrix = attention_array[layer_index, head_index]
            num_rows, num_cols = matrix.shape

            visualize_attention_matrix(
                matrix,
                x_labels=[""] * num_cols,
                y_labels=[""] * num_rows,
                title="",
                xlabel="",
                ylabel="",
                ax=ax,
                top_n=top_n,
                enlarged_size=enlarged_size,
                gamma=gamma,
                cmap=cmap,
                left_top_cells=left_top_cells,
                right_bottom_cells=right_bottom_cells,
                save_path=None,
                length_threshold=length_threshold,
                interval=interval,
                if_interval=if_interval,
                if_top_cells=if_top_cells,
                show_scores_in_enlarged_cells=show_scores_in_enlarged_cells,
                background_color=background_color,
                lean_more=lean_more,
                merge_virtual_tokens=merge_virtual_tokens,
                virtual_token_min_run=virtual_token_min_run,
                virtual_token_label_color=virtual_token_label_color,
                close_after_save=False,
                cbar=cbar,
                tight_layout=False,
                vmin=shared_vmin,
                vmax=shared_vmax,
                norm=shared_norm,
            )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(
                left=False,
                bottom=False,
                top=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                labeltop=False,
            )

    fig_width, fig_height = fig.get_size_inches()
    left_margin = max(0.02, min(0.08, 0.7 / fig_width))
    bottom_margin = max(0.02, min(0.06, 0.35 / fig_height))
    top_reserved = 0.75 if title else 0.35
    right_reserved = 0.9 if show_shared_cbar else 0.15
    top_margin = max(0.65, 1 - top_reserved / fig_height)
    right_margin = max(0.75, 1 - right_reserved / fig_width)
    fig.tight_layout(rect=[left_margin, bottom_margin, right_margin, top_margin])

    fig.canvas.draw()
    heatmap_positions = [ax.get_position() for ax in axes.ravel()]
    grid_left = min(pos.x0 for pos in heatmap_positions)
    grid_right = max(pos.x1 for pos in heatmap_positions)
    grid_bottom = min(pos.y0 for pos in heatmap_positions)
    grid_top = max(pos.y1 for pos in heatmap_positions)

    title_fontsize = max(16, min(24, fig_width * 0.7))
    axis_label_fontsize = max(9, min(12, fig_width / max(num_heads, 1) * 4.5))
    label_gap_y = max(0.006, 0.12 / fig_height)
    title_y = 0.99
    title_height = title_fontsize / 72 / fig_height
    max_head_label_y = title_y - title_height - max(0.004, 0.08 / fig_height)
    head_label_y = min(grid_top + label_gap_y, max_head_label_y)

    for head_index in range(num_heads):
        col_positions = [
            axes[layer_index, head_index].get_position()
            for layer_index in range(num_layers)
        ]
        col_left = min(pos.x0 for pos in col_positions)
        col_right = max(pos.x1 for pos in col_positions)
        fig.text(
            (col_left + col_right) / 2,
            head_label_y,
            f"Head {head_index}",
            ha="center",
            va="bottom",
            fontsize=axis_label_fontsize,
        )

    layer_label_x = max(0.005, grid_left - max(0.01, 0.28 / fig_width))
    for layer_index in range(num_layers):
        row_positions = [
            axes[layer_index, head_index].get_position()
            for head_index in range(num_heads)
        ]
        row_bottom = min(pos.y0 for pos in row_positions)
        row_top = max(pos.y1 for pos in row_positions)
        fig.text(
            layer_label_x,
            (row_bottom + row_top) / 2,
            f"Layer {layer_index}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=axis_label_fontsize,
        )

    if title:
        fig.text(
            0.5,
            title_y,
            title,
            ha="center",
            va="top",
            fontsize=title_fontsize,
            fontname="DejaVu Serif",
            fontweight="bold",
        )

    if show_shared_cbar:
        cbar_pad = max(0.008, 0.14 / fig_width)
        cbar_width = max(0.01, min(0.02, 0.18 / fig_width))
        cbar_left = min(0.985 - cbar_width, grid_right + cbar_pad)
        cax = fig.add_axes(
            [cbar_left, grid_bottom, cbar_width, grid_top - grid_bottom]
        )
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=shared_norm)
        sm.set_array([])
        overview_cbar = fig.colorbar(sm, cax=cax)
        overview_cbar.outline.set_visible(False)

        num_ticks = 7
        tick_values = np.linspace(shared_vmin, shared_vmax, num_ticks)
        overview_cbar.set_ticks(tick_values)
        overview_cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])
        if shared_cbar_label:
            overview_cbar.set_label(shared_cbar_label, rotation=90)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        if close_after_save:
            plt.close(fig)

    return fig, axes


def generate_sparse_labels(
    tokens, top_cells, axis, interval=10, if_interval=True, if_top_cells=True
):
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


def find_top_cells(data, top_n):
    if top_n <= 0:
        return []

    flat_data = data.flatten()
    top_n = min(top_n, flat_data.size)

    threshold = np.partition(flat_data, -top_n)[-top_n]
    top_indices = np.where(flat_data >= threshold)[0]
    top_indices_sorted = top_indices[np.argsort(-flat_data[top_indices])]

    return [np.unravel_index(idx, data.shape) for idx in top_indices_sorted]


def difference_heatmap(
    data1,
    data2,
    base="data1",  # "data1", "data2", or "none" to choose the background
    circle_scale=1.0,
    circle_color_positive=THEME_POSITIVE,
    circle_color_negative=THEME_NEGATIVE,
    ax=None,
    gamma=1.5,
    **kwargs,
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
        kwargs.setdefault("norm", norm)
        kwargs.setdefault("cmap", THEME_CMAP)  # default colormap
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

    else:
        # For "none" base, set uniform white background with black borders
        kwargs.setdefault("cmap", plt.cm.colors.ListedColormap(["#fcfaf5"]))
        kwargs.setdefault("linecolor", "black")  # Set border color to black
        kwargs.setdefault("linewidths", 0.7)
        kwargs.setdefault("cbar", False)

    # 4. Draw the base heatmap
    if ax is None:
        ax = plt.gca()

    ax, plotter = create_tablelens_heatmap(bg_data, ax=ax, **kwargs)

    # 5. Overlay circles that show the magnitude of the difference
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    # We need the absolute maximum difference to normalize circle sizes
    max_abs_diff = np.max(np.abs(diff)) if np.any(diff != 0) else 1e-6

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
            circ = Circle((x, y), radius=radius)
            patches.append(circ)

            # Choose color based on sign
            if val > 0:
                colors.append(circle_color_positive)
            else:
                colors.append(circle_color_negative)

    # Create a PatchCollection and add it to the plot
    collection = PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.7)
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
def compare_two_attentions(
    attn1,
    attn2,
    tokens,
    title="Comparison: Matrix2 - Matrix1",
    base="data1",
    save_path=None,
):
    """
    Compares two attention matrices and visualizes their differences in a heatmap.

    Parameters:
    - attn1: First attention matrix (baseline)
    - attn2: Second attention matrix to compare against attn1
    - tokens: List of token labels for x/y axes
    - save_path: File path to save the generated heatmap PDF
    - title: Title for the plot (default: "Comparison: Matrix2 - Matrix1")
    - cmap: Matplotlib colormap for the heatmap (default: 'Purples')
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if torch.is_tensor(attn1):
        attn1 = attn1.detach().cpu().numpy()
    if torch.is_tensor(attn2):
        attn2 = attn2.detach().cpu().numpy()

    difference_heatmap(
        attn1,
        attn2,
        base=base,  # draw the background using attn1; circles show how attn2 differs
        x_labels=[bold_special_tokens(token) for token in tokens],
        y_labels=[bold_special_tokens(token) for token in tokens],
        title=title,
        xlabel="Tokens Attended to",
        ylabel="Tokens Attending",
        circle_scale=1.0,  # adjust for bigger or smaller circles
        circle_color_positive=THEME_POSITIVE,  # where attn2 > attn1
        circle_color_negative=THEME_NEGATIVE,  # where attn2 < attn1
        ax=ax,
    )

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
    use_std_error=False,  # If True, use SEM = std/sqrt(n); else use raw std
    circle_scale=1.0,  # Base scaling factor for circles
    cmap=THEME_CMAP,  # Colormap for circle colors
    linecolor="black",  # Grid line color
    linewidths=0.5,  # Grid line width
    save_path="check_stability_heatmap.pdf",
    gamma=1.5,
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
        Colormap used to color circles by the mean value. Defaults to 'Purples'.
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
    std_vals = np.std(matrices, axis=0)  # shape (R, C)

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
        cmap=plt.cm.colors.ListedColormap(["white"]),
        linecolor=linecolor,
        linewidths=linewidths,
        cbar=False,
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
            circ = Circle((col_centers[j], row_centers[i]), radius=radius)
            patches.append(circ)
            colors.append(plt.get_cmap(cmap)(norm(mval)))

    # Add circles to plot
    collection = PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.7)
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
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels, rotation=0, ha="right")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Check Stability heatmap saved to {save_path}")
    else:
        plt.tight_layout()

    return ax


def compare_two_attentions_with_circles(
    attn1,
    attn2,
    tokens,
    title="Comparison with Circles",
    xlabel=None,
    ylabel=None,
    save_path=None,
    circle_scale=1.0,
    gamma=1.5,
    cmap=THEME_CMAP,
    max_circle_ratio=0.45,
):
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
    - cmap: Colormap to use (default: 'Purples')
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
        vmin=vmin,
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
                circ = Circle((col_centers[j], row_centers[i]), radius=radius)
                patches.append(circ)
                colors.append(plt.get_cmap(cmap)(norm(data2[i, j])))

    # Add circles to plot
    collection = PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.7)
    ax.add_collection(collection)

    if save_path is None:
        save_path = "attention_comparison_circles.pdf"

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Attention comparison heatmap with circles saved to {save_path}")


def check_stability_heatmap_new(
    matrices,
    x_labels=None,
    y_labels=None,
    title=None,
    xlabel="Tokens Attended to",
    ylabel="Tokens Attending",
    ax=None,
    use_std_error=False,  # If True, use SEM = std/sqrt(n); else use raw std
    circle_scale=1.0,  # Base scaling factor for circles
    cmap=THEME_CMAP,  # Colormap for *square cells* (based on the mean)
    linecolor="white",  # Grid line color
    linewidths=1.0,  # Grid line width
    save_path="check_stability_heatmap.pdf",
    gamma=1.5,
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
        The colormap for the *background squares*. Default is 'Purples'.
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
    std_vals = np.std(matrices, axis=0)  # shape (R, C)
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
        norm=norm,
    )

    # 6) Overlay the hollow orange circles for each cell
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    # Find the maximum error to normalize circle sizes
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
        facecolor="none",  # hollow
        edgecolor="orange",  # orange ring
        linewidth=1.5,
        alpha=1.0,
    )
    ax.add_collection(circle_collection)

    # 7) Adjust label rotations for clarity
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels, rotation=0, ha="right")

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
    cmap=THEME_CMAP,  # Colormap for background and rings
    gamma=1.5,  # PowerNorm gamma
    linecolor="white",  # Grid line color
    linewidths=1.0,  # Grid line width
    ring_radius=0.45,  # Fraction of half-cell for outer ring radius
    save_path="check_stability_heatmap_with_target_rings.pdf",
    show_background=True,
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
        Colormap for squares and rings. Defaults to 'Purples'.
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
        cbar=True,  # single colorbar for squares & rings
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=min_val,
        vmax=max_val,
        norm=norm,
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
                radius_in = ring_radius * (ring_idx / n)
                radius_out = ring_radius * ((ring_idx + 1.0) / n)

                wedge = Wedge(
                    center=(col_centers[col_j], row_centers[row_i]),
                    r=radius_out,
                    theta1=0,
                    theta2=360,
                    width=(radius_out - radius_in),  # annulus thickness
                    facecolor=color,
                    edgecolor="none",
                )
                ax.add_patch(wedge)

    # 10) Adjust label rotation
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels, rotation=0, ha="right")

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
    use_std_error=True,  # If True, use SEM = std/sqrt(n); else raw std
    circle_scale=1.0,  # Factor controlling how large the circle can get
    cmap=THEME_CMAP,  # Colormap for background squares
    linecolor="white",  # Grid line color
    linewidths=1.0,  # Grid line width
    save_path="check_stability_heatmap_with_gradient_color.pdf",
    gamma=1.5,
    radial_resolution=100,  # Resolution of the radial gradient image
    use_white_center=False,  # If True, use white at center instead of (mean-err) color
    color_contrast_scale=2.0,  # Factor to enhance contrast between inner and outer colors
    max_circle_ratio=0.45,  # Maximum circle radius as a fraction of half-cell width (was 0.5)
):
    """
    Plots an n-run stability heatmap:

      1) Background squares are colored by the mean attention score across n matrices
         (darker = higher mean, using 'Purples').
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
        Colormap for both squares & gradient circles. Default is 'Purples'.
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
    mean_vals = np.mean(matrices, axis=0)  # shape (R, C)
    std_vals = np.std(matrices, axis=0)  # shape (R, C)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n)  # SEM
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
        cbar=True,  # share colorbar with circles
        linecolor=linecolor,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        lean_more=True,
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
                dist = np.sqrt((r - center) ** 2 + (c - center) ** 2)
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
                inner_rgba = np.array(
                    [1.0, 1.0, 1.0, 1.0], dtype=float
                )  # White with full opacity
            else:
                inner_rgba = np.array(cmap_obj(norm(val_lower)), dtype=float)

            outer_rgba = np.array(cmap_obj(norm(val_upper)), dtype=float)

            # Build a radial gradient image NxN
            gradient_img = make_radial_gradient_image(
                inner_rgba=inner_rgba, outer_rgba=outer_rgba, N=radial_resolution
            )

            x_center = col_centers[j]
            y_center = row_centers[i]
            x_left = x_center - radius
            x_right = x_center + radius
            y_bottom = y_center - radius
            y_top = y_center + radius

            # Render the image in that region
            im = ax.imshow(
                gradient_img,
                extent=[x_left, x_right, y_bottom, y_top],
                origin="lower",
                zorder=3,  # above the squares
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
    cmap=THEME_CMAP,
    gamma=1.5,
    linecolor="white",
    linewidths=1.0,
    ring_radius=0.45,
    save_path="check_stability_heatmap_with_pie_chart.pdf",
    show_background=True,
    use_std_error=False,  # New parameter for using std error vs. std
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
    std_vals = np.std(matrices, axis=0)  # (R, C)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n)  # SEM
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
        norm=norm,
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
            frac = err_val / max_err
            circle_r = frac * ring_radius

            center_x = col_centers[j]
            center_y = row_centers[i]

            # 1) Draw the light grey background circle
            grey_circle = Circle(
                (center_x, center_y),
                radius=circle_r,
                facecolor="lightgrey",
                edgecolor="none",
                alpha=0.6,
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
                    edgecolor="none",
                )
                ax.add_patch(wedge_patch)

    # 10) Set axis tick labels
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels, rotation=0, ha="right")

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
    cmap=THEME_CMAP,
    gamma=1.5,
    linecolor="white",
    linewidths=1.0,
    ring_radius=0.45,
    save_path="check_stability_heatmap_half_pie.pdf",
    show_background=True,
    use_std_error=False,
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
        Colormap for squares + half-pies. Defaults to 'Purples'.
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
    std_vals = np.std(matrices, axis=0)  # (R, C)
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
        norm=norm,
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
                zorder=2,
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
                    zorder=3,  # above grey circle
                )
                ax.add_patch(wedge_patch)

    # 10) Set axis tick labels
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels, rotation=0, ha="right")

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
    normalize_sparklines=False,
    save_path="attention_evolution_sparklines.pdf",
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
            print(
                f"Converted input to numpy array with shape {attentions_over_time.shape}"
            )
        except Exception as e:
            raise ValueError(f"Failed to convert input to numpy array: {str(e)}")

    # Validate dimensions after conversion
    if attentions_over_time.ndim != 5:
        raise ValueError(
            f"Expected attentions_over_time to have 5 dimensions [n_epochs, layers, heads, n_tokens, n_tokens], "
            f"but got shape {attentions_over_time.shape}"
        )

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
        gamma=gamma,
    )

    # Get cell centers directly from plotter
    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    def get_sparkline_color(cell_intensity):
        if not normalize_sparklines:
            """Return either dark blue or white based on background intensity relative to color bar midpoint."""
            # Calculate the middle of the color range (with PowerNorm influence)
            norm_tmp = PowerNorm(gamma=1.5, vmin=min_val, vmax=max_val)
            middle_value = norm_tmp.inverse(0.5)

            # Compare the raw attention value to the middle value
            return (
                sparkline_color_light
                if cell_intensity > middle_value
                else sparkline_color_dark
            )
        else:
            return sparkline_color_dark

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
            x = np.linspace(x_center - width * 0.4, x_center + width * 0.4, n_epochs)

            # Calculate y-coordinates (with the correct orientation)
            y = y_center - (norm_values - 0.5) * height * 0.7

            # Determine color and plot sparkline
            cell_intensity = avg_attention[i, j]
            sparkline_color = get_sparkline_color(cell_intensity)
            ax.plot(
                x,
                y,
                color=sparkline_color,
                linewidth=sparkline_linewidth,
                alpha=sparkline_alpha,
            )

    # Update legend - show both dark and light sparkline colors
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=sparkline_color_dark,
            lw=sparkline_linewidth,
            label="Trend (low attention)",
        ),
        Line2D(
            [0],
            [0],
            color=sparkline_color_light,
            lw=sparkline_linewidth,
            label="Trend (high attention)",
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, -0.1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    return ax


from collections import deque


def find_attention_regions_with_merging(
    attention_matrix,
    n_seeds=3,
    min_distance=2,
    expansion_threshold=0.7,
    merge_std_threshold=0.6,
    proximity_threshold=2,
    max_expansion_steps=3,
):
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
    flat_indices = np.argsort(attention_matrix.flatten())[
        ::-1
    ]  # Indices sorted by decreasing value

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
            if (
                len(potential_seeds) >= n_seeds * 3
            ):  # Get 3x more seeds than needed as reserve
                break

    # Start with the first n_seeds
    active_seeds = potential_seeds[:n_seeds]
    seed_queue = deque(potential_seeds[n_seeds:])

    # Step 2: Grow rectangles from active seeds
    rectangles = [(r, c, r, c) for r, c in active_seeds]  # (top, left, bottom, right)
    rectangle_stats = [
        calculate_rectangle_stats(attention_matrix, rect) for rect in rectangles
    ]

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
                (0, 0, 1, 0),  # Bottom
                (0, 0, 0, 1),  # Right
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
                    rectangle_stats[i] = calculate_rectangle_stats(
                        attention_matrix, best_rect
                    )
                    any_change = True
                else:
                    # There's overlap or nearby rectangles - evaluate whether to merge
                    can_merge = True
                    for j in potential_merges:
                        if not should_merge_rectangles(
                            attention_matrix,
                            rectangles[i],
                            rectangles[j],
                            rectangle_stats[i],
                            rectangle_stats[j],
                            merge_std_threshold,
                        ):
                            can_merge = False
                            break

                    if can_merge:
                        # Merge rectangles
                        merged_rect = merge_rectangles(
                            [rectangles[i]] + [rectangles[j] for j in potential_merges]
                        )
                        merged_stats = calculate_rectangle_stats(
                            attention_matrix, merged_rect
                        )

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
                                new_rect = (
                                    new_seed[0],
                                    new_seed[1],
                                    new_seed[0],
                                    new_seed[1],
                                )

                                # Find the first None position to replace
                                for k in range(len(rectangles)):
                                    if rectangles[k] is None:
                                        rectangles[k] = new_rect
                                        rectangle_stats[k] = calculate_rectangle_stats(
                                            attention_matrix, new_rect
                                        )
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
        rectangle_scores.append((rect, stats["mean"]))

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
    rectangle = matrix[top : bottom + 1, left : right + 1]

    return {
        "mean": np.mean(rectangle),
        "std": np.std(rectangle),
        "sum": np.sum(rectangle),
        "size": rectangle.size,
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
    rectangle = matrix[top : bottom + 1, left : right + 1]
    avg_inside = np.mean(rectangle)

    # Calculate boundary (1-cell wide) around rectangle
    boundary_cells = []

    # Top and bottom boundaries
    if top > 0:
        boundary_cells.extend(
            matrix[top - 1, max(0, left - 1) : min(cols, right + 2)].flatten()
        )
    if bottom < rows - 1:
        boundary_cells.extend(
            matrix[bottom + 1, max(0, left - 1) : min(cols, right + 2)].flatten()
        )

    # Left and right boundaries (excluding corners already counted)
    if left > 0:
        boundary_cells.extend(matrix[top : bottom + 1, left - 1].flatten())
    if right < cols - 1:
        boundary_cells.extend(matrix[top : bottom + 1, right + 1].flatten())

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
    total_size = stats1["size"] + stats2["size"]
    weighted_std = (
        stats1["std"] * stats1["size"] + stats2["std"] * stats2["size"]
    ) / total_size

    # Calculate ratio of merged std dev to weighted individual std devs
    std_ratio = merged_stats["std"] / weighted_std if weighted_std > 0 else float("inf")

    # Allow merging if std dev doesn't increase too much (ratio close to 1.0 or below)
    return std_ratio <= (
        1.0 / merge_threshold
    )  # Inverted so that merge_threshold < 1.0 is stricter


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
    region_color="orange",
    region_linewidth=2,
    region_alpha=0.7,
    label_regions=False,
    gamma=1.5,
    save_path="attention_with_detected_regions.pdf",
    ax=None,
    cmap=THEME_CMAP,
    max_expansion_steps=3,
    proximity_threshold=2,
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
        vmin=vmin,
    )

    # Find regions of interest
    rectangles = find_attention_regions_with_merging(
        attention_matrix,
        n_seeds=n_regions,
        min_distance=min_distance,
        expansion_threshold=expansion_threshold,
        merge_std_threshold=merge_threshold,
        max_expansion_steps=max_expansion_steps,
        proximity_threshold=proximity_threshold,
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
            facecolor="none",
            alpha=region_alpha,
            zorder=10,  # Ensure rectangle is drawn on top
        )
        ax.add_patch(rect)

        # Add region label if requested
        if label_regions:
            ax.text(
                x + width / 2,
                y + height / 2,
                f"R{i + 1}",
                color="white",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor=region_color, alpha=0.5, boxstyle="round"),
                zorder=11,
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
