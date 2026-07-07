import warnings
from typing import Any, List, Optional, Sequence, Tuple

from ._tablelens_heatmap import heatmap
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

THEME_CMAP = "Purples"
THEME_POSITIVE = "#C77DF3"
THEME_NEGATIVE = "#6A0DAD"
THEME_TOP_LABEL_BACKGROUND = "#f8bbd0"
THEME_REGION_EDGE = "#CE93D8"
THEME_VIRTUAL_TOKEN_LABEL = "#9A9A9A"
THEME_SPARKLINE_LOW = "darkblue"
THEME_SPARKLINE_HIGH = "white"
THEME_TITLE_FONT = "DejaVu Serif"
THEME_TITLE_FONTSIZE = 14
THEME_AXIS_LABEL_FONTSIZE = 12
THEME_CBAR_TICKS = 7
THEME_CBAR_SIZE = "5%"
THEME_CBAR_PAD = 0.1
THEME_CIRCLE_ALPHA = 0.7
THEME_REGION_LINEWIDTH = 3
THEME_REGION_LINESTYLE = ":"

__all__ = [
    "visualize_attention_matrix",
    "visualize_attention_overview",
    "compare_two_attentions_with_circles",
    "check_stability_heatmap_with_gradient_color",
    "visualize_attention_evolution_sparklines",
]


def _resolve_deprecated_bool_alias(
    new_value: bool,
    legacy_value: Optional[bool],
    new_name: str,
    legacy_name: str,
) -> bool:
    if legacy_value is None:
        return new_value

    warnings.warn(
        f"`{legacy_name}` is deprecated; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return legacy_value


def _bold_special_tokens(label: str) -> str:
    special_tokens = {"[CLS]", "[SEP]", "[PAD]"}
    if label in special_tokens:
        return f"$\mathbf{{{label}}}$"
    return label


def _create_tablelens_heatmap(
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
    rotate_x_labels_90=False,
):
    """
    Create the variable-cell heatmap used by IzzyViz public visualizations.

    This internal helper returns both the axis and the backend plotter because
    downstream overlays need exact row and column positions.
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

    ax, plotter = heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        linewidths=linewidths,
        linecolor=linecolor,
        square=True,
        cbar=False,
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
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=THEME_CBAR_SIZE, pad=THEME_CBAR_PAD)

        im = ax.collections[0]
        cbar = plt.colorbar(im, cax=cax)
        cbar.outline.set_visible(False)

        tick_values = np.linspace(vmin, vmax, THEME_CBAR_TICKS)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for label in ax.get_xticklabels():
        if rotate_x_labels_90:
            label.set_rotation(90)
        else:
            label.set_rotation(45)

    for label in ax.get_yticklabels():
        label.set_rotation(0)

    ax.set_title(
        title,
        fontsize=THEME_TITLE_FONTSIZE,
        fontname=THEME_TITLE_FONT,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(xlabel, fontsize=THEME_AXIS_LABEL_FONTSIZE, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=THEME_AXIS_LABEL_FONTSIZE, labelpad=15)

    if top_cells is not None:
        x_ticklabels = ax.get_xticklabels()
        y_ticklabels = ax.get_yticklabels()

        x_indices = set(col_index for (row_index, col_index) in top_cells)
        y_indices = set(row_index for (row_index, col_index) in top_cells)

        for idx, label in enumerate(x_ticklabels):
            if idx in x_indices and background_color:
                label.set_bbox(
                    dict(
                        facecolor=THEME_TOP_LABEL_BACKGROUND,
                        edgecolor=THEME_TOP_LABEL_BACKGROUND,
                        boxstyle="round,pad=0.2",
                        alpha=0.5,
                    )
                )

        for row_index in y_indices:
            if row_index < len(y_ticklabels) and background_color:
                label = y_ticklabels[row_index]
                label.set_bbox(
                    dict(
                        facecolor=THEME_TOP_LABEL_BACKGROUND,
                        edgecolor=THEME_TOP_LABEL_BACKGROUND,
                        boxstyle="round,pad=0.2",
                        alpha=0.5,
                    )
                )

    if left_top_cells is not None and right_bottom_cells is not None:
        if len(left_top_cells) != len(right_bottom_cells):
            raise ValueError(
                "left_top_cells and right_bottom_cells must have the same length."
            )

        for lt_cell, rb_cell in zip(left_top_cells, right_bottom_cells):
            lt_row, lt_col = lt_cell
            rb_row, rb_col = rb_cell

            if lt_row > rb_row or lt_col > rb_col:
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

            col_positions = plotter.col_positions
            row_positions = plotter.row_positions

            x = col_positions[lt_col]
            width = col_positions[rb_col + 1] - col_positions[lt_col]
            y = row_positions[lt_row]
            height = row_positions[rb_row + 1] - row_positions[lt_row]

            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=THEME_REGION_LINEWIDTH,
                edgecolor=THEME_REGION_EDGE,
                facecolor="none",
                linestyle=THEME_REGION_LINESTYLE,
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
    matrix: Any,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    title: str = "Attention Heat",
    xlabel: str = "Tokens Attended to",
    ylabel: str = "Tokens Attending",
    ax: Optional[Axes] = None,
    top_n: int = 3,
    enlarged_size: float = 1.8,
    gamma: float = 1.5,
    cmap: Any = THEME_CMAP,
    left_top_cells: Optional[Sequence[Tuple[int, int]]] = None,
    right_bottom_cells: Optional[Sequence[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    length_threshold: int = 64,
    interval: int = 10,
    show_interval_labels: bool = False,
    show_top_cell_labels: bool = True,
    show_scores_in_enlarged_cells: bool = True,
    background_color: bool = True,
    rotate_x_labels_90: bool = False,
    merge_virtual_tokens: bool = False,
    virtual_token_min_run: int = 1,
    virtual_token_label_color: str = THEME_VIRTUAL_TOKEN_LABEL,
    close_after_save: bool = False,
    cbar: bool = True,
    tight_layout: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[Any] = None,
    if_interval: Optional[bool] = None,
    if_top_cells: Optional[bool] = None,
    lean_more: Optional[bool] = None,
) -> Tuple[Axes, Any]:
    """
    Visualize any 2D attention-like matrix.

    Parameters
    ----------
    matrix
        A 2D NumPy array or PyTorch tensor. Rows map to ``y_labels`` and
        columns map to ``x_labels``.
    x_labels, y_labels
        Optional axis labels. If ``y_labels`` is omitted for a square matrix,
        ``x_labels`` are reused.
    top_n
        Number of highest-value cells to highlight. Ties at the Nth-highest
        value are included, so more than ``top_n`` cells may be selected.
    show_interval_labels, show_top_cell_labels
        Sparse-label controls used when an axis exceeds ``length_threshold``.
    rotate_x_labels_90
        If True, rotate x-axis labels by 90 degrees instead of 45 degrees.
    merge_virtual_tokens
        If True, compress contiguous rows or columns that do not contain top
        cells into virtual tokens.
    save_path
        Optional output path. If None, the figure is not saved.

    Returns
    -------
    tuple
        ``(ax, plotter)`` where ``plotter`` exposes row and column positions.
    """

    show_interval_labels = _resolve_deprecated_bool_alias(
        show_interval_labels, if_interval, "show_interval_labels", "if_interval"
    )
    show_top_cell_labels = _resolve_deprecated_bool_alias(
        show_top_cell_labels, if_top_cells, "show_top_cell_labels", "if_top_cells"
    )
    rotate_x_labels_90 = _resolve_deprecated_bool_alias(
        rotate_x_labels_90, lean_more, "rotate_x_labels_90", "lean_more"
    )

    if torch.is_tensor(matrix):
        data = matrix.detach().cpu().numpy()
    else:
        data = np.asarray(matrix)

    if data.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {data.shape}")
    if data.size == 0:
        raise ValueError("matrix must not be empty")
    if interval <= 0:
        raise ValueError("interval must be > 0")

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

    top_cells = _find_top_cells(data, top_n)
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
        display_x_labels = _generate_sparse_labels(
            x_labels,
            top_cells,
            axis=1,
            interval=interval,
            show_interval_labels=show_interval_labels,
            show_top_cell_labels=show_top_cell_labels,
        )
    else:
        display_x_labels = [_bold_special_tokens(label) for label in x_labels]

    for idx in virtual_x_indices:
        display_x_labels[idx] = x_labels[idx]

    if y_is_sparse:
        display_y_labels = _generate_sparse_labels(
            y_labels,
            top_cells,
            axis=0,
            interval=interval,
            show_interval_labels=show_interval_labels,
            show_top_cell_labels=show_top_cell_labels,
        )
    else:
        display_y_labels = [_bold_special_tokens(label) for label in y_labels]

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

    ax, plotter = _create_tablelens_heatmap(
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
        rotate_x_labels_90=rotate_x_labels_90,
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
                rotation=90 if rotate_x_labels_90 else 45,
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
    if isinstance(attention, (list, tuple)):
        return np.asarray([_attention_to_numpy(item) for item in attention])
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
    attentions: Any,
    batch_index: int = 0,
    title: str = "Attention Overview",
    title_x: float = 0.5,
    title_y: float = 0.99,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    top_n: int = 3,
    enlarged_size: float = 1.8,
    gamma: float = 1.5,
    cmap: Any = THEME_CMAP,
    left_top_cells: Optional[Sequence[Tuple[int, int]]] = None,
    right_bottom_cells: Optional[Sequence[Tuple[int, int]]] = None,
    length_threshold: int = 64,
    interval: int = 10,
    show_interval_labels: bool = False,
    show_top_cell_labels: bool = True,
    show_scores_in_enlarged_cells: bool = True,
    background_color: bool = True,
    rotate_x_labels_90: bool = False,
    merge_virtual_tokens: bool = False,
    virtual_token_min_run: int = 1,
    virtual_token_label_color: str = THEME_VIRTUAL_TOKEN_LABEL,
    cbar: bool = False,
    shared_color_scale: bool = True,
    shared_cbar: bool = True,
    shared_cbar_label: str = "Attention Score",
    close_after_save: bool = False,
    if_interval: Optional[bool] = None,
    if_top_cells: Optional[bool] = None,
    lean_more: Optional[bool] = None,
) -> Tuple[Figure, np.ndarray]:
    """
    Visualize an overview grid of all attention layers and heads.

    Rows are layers from top to bottom. Columns are heads from left to right.
    Each subplot is drawn by visualize_attention_matrix, with inner token labels
    hidden because overview cells are too small to read.

    Parameters
    ----------
    attentions
        HuggingFace-style attentions or an array/tensor. Supported shapes are
        ``(layers, batch, heads, rows, cols)``, ``(layers, heads, rows, cols)``,
        ``(heads, rows, cols)``, or a list of per-layer tensors.
    title_x, title_y
        Figure-coordinate title position. Increasing ``title_y`` moves the
        title upward; decreasing it moves the title closer to the heatmap grid.
    shared_color_scale
        If True, all subplots use one global color scale.
    shared_cbar
        If True with ``shared_color_scale``, draw one unified colorbar.
    save_path
        Optional output path. If None, the figure is not saved.

    Returns
    -------
    tuple
        ``(fig, axes)`` where ``axes[layer, head]`` indexes each subplot.
    """

    show_interval_labels = _resolve_deprecated_bool_alias(
        show_interval_labels, if_interval, "show_interval_labels", "if_interval"
    )
    show_top_cell_labels = _resolve_deprecated_bool_alias(
        show_top_cell_labels, if_top_cells, "show_top_cell_labels", "if_top_cells"
    )
    rotate_x_labels_90 = _resolve_deprecated_bool_alias(
        rotate_x_labels_90, lean_more, "rotate_x_labels_90", "lean_more"
    )

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
                show_interval_labels=show_interval_labels,
                show_top_cell_labels=show_top_cell_labels,
                show_scores_in_enlarged_cells=show_scores_in_enlarged_cells,
                background_color=background_color,
                rotate_x_labels_90=rotate_x_labels_90,
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
            title_x,
            title_y,
            title,
            ha="center",
            va="top",
            fontsize=title_fontsize,
            fontname=THEME_TITLE_FONT,
            fontweight="bold",
        )

    if show_shared_cbar:
        cbar_pad = max(0.008, 0.14 / fig_width)
        cbar_width = max(0.01, min(0.02, 0.18 / fig_width))
        cbar_left = min(0.985 - cbar_width, grid_right + cbar_pad)
        cax = fig.add_axes([cbar_left, grid_bottom, cbar_width, grid_top - grid_bottom])
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=shared_norm)
        sm.set_array([])
        overview_cbar = fig.colorbar(sm, cax=cax)
        overview_cbar.outline.set_visible(False)

        tick_values = np.linspace(shared_vmin, shared_vmax, THEME_CBAR_TICKS)
        overview_cbar.set_ticks(tick_values)
        overview_cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])
        if shared_cbar_label:
            overview_cbar.set_label(shared_cbar_label, rotation=90)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        if close_after_save:
            plt.close(fig)

    return fig, axes


def _generate_sparse_labels(
    tokens: Sequence[str],
    top_cells: Sequence[Tuple[int, int]],
    axis: int,
    interval: int = 10,
    show_interval_labels: bool = True,
    show_top_cell_labels: bool = True,
) -> List[str]:
    """
    Return sparse axis labels with optional interval and top-cell labels.

    Empty strings mark ticks that should not be shown.
    """
    sparse_labels = [""] * len(tokens)

    if show_interval_labels:
        for i in range(0, len(tokens), interval):
            if i < len(tokens):
                sparse_labels[i] = f"{i}"

    if show_top_cell_labels:
        for row, col in top_cells:
            idx = col if axis == 1 else row
            if 0 <= idx < len(tokens):
                sparse_labels[idx] = _bold_special_tokens(tokens[idx])

    return sparse_labels


def _find_top_cells(data: Any, top_n: int) -> List[Tuple[int, int]]:
    """Return cells at or above the Nth-highest value, including ties."""
    if top_n <= 0:
        return []

    data = np.asarray(data)
    flat_data = data.ravel()
    top_n = min(top_n, flat_data.size)

    threshold = np.partition(flat_data, -top_n)[-top_n]
    top_indices = np.where(flat_data >= threshold)[0]
    top_indices_sorted = top_indices[np.argsort(-flat_data[top_indices])]

    return [
        (int(row), int(col))
        for row, col in (
            np.unravel_index(idx, data.shape) for idx in top_indices_sorted
        )
    ]


def compare_two_attentions_with_circles(
    attn1: Any,
    attn2: Any,
    tokens: Sequence[str],
    title: str = "Comparison with Circles",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    circle_scale: float = 1.0,
    gamma: float = 1.5,
    cmap: Any = THEME_CMAP,
    max_circle_ratio: float = 0.45,
) -> Axes:
    """
    Compares two attention matrices by showing the first matrix as background colors
    and the second matrix as circles with varying sizes based on their differences.

    Parameters
    ----------
    save_path
        Optional output path. If None, the figure is not saved.
    max_circle_ratio
        Maximum radius of a circle as a fraction of half-cell width.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    data1 = _attention_to_numpy(attn1)
    data2 = _attention_to_numpy(attn2)

    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError(
            f"attn1 and attn2 must be 2D matrices, got {data1.shape} and {data2.shape}"
        )
    if data1.shape != data2.shape:
        raise ValueError(
            f"attn1 and attn2 must have the same shape, got {data1.shape} and {data2.shape}"
        )
    if data1.shape[0] != data1.shape[1]:
        raise ValueError(
            f"attention matrices must be square when one token list is used, got {data1.shape}"
        )
    if len(tokens) != data1.shape[0]:
        raise ValueError(
            f"len(tokens) must match matrix size: {len(tokens)} != {data1.shape[0]}"
        )

    diff = np.abs(data2 - data1)

    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    ax, plotter = _create_tablelens_heatmap(
        data1,
        x_labels=[_bold_special_tokens(token) for token in tokens],
        y_labels=[_bold_special_tokens(token) for token in tokens],
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        cmap=cmap,
        norm=norm,
        vmax=vmax,
        vmin=vmin,
    )

    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    max_diff = np.max(diff) if np.any(diff != 0) else 1e-6

    patches = []
    colors = []

    for i in range(len(row_centers)):
        for j in range(len(col_centers)):
            radius = min(circle_scale * max_circle_ratio * (diff[i, j] / max_diff), 0.5)

            if radius > 0:
                circ = Circle((col_centers[j], row_centers[i]), radius=radius)
                patches.append(circ)
                colors.append(plt.get_cmap(cmap)(norm(data2[i, j])))

    collection = PatchCollection(
        patches, facecolor=colors, edgecolor="none", alpha=THEME_CIRCLE_ALPHA
    )
    ax.add_collection(collection)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Attention comparison heatmap with circles saved to {save_path}")

    return ax


def check_stability_heatmap_with_gradient_color(
    matrices: Any,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    title: str = "Check Stability Heatmap with Gradient Circles",
    xlabel: str = "Tokens Attended to",
    ylabel: str = "Tokens Attending",
    ax: Optional[Axes] = None,
    use_std_error: bool = True,
    circle_scale: float = 1.0,
    cmap: Any = THEME_CMAP,
    linecolor: str = "white",
    linewidths: float = 1.0,
    save_path: Optional[str] = None,
    gamma: float = 1.5,
    radial_resolution: int = 100,
    use_white_center: bool = False,
    color_contrast_scale: float = 2.0,
    max_circle_ratio: float = 0.45,
) -> Axes:
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
    matrices
        A list of ``(R, C)`` arrays or a single 3D array shaped ``(n, R, C)``.
    use_std_error
        If True, use SEM; otherwise use raw standard deviation.
    save_path
        Optional output path. If None, the figure is not saved.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    """
    if radial_resolution < 2:
        raise ValueError("radial_resolution must be >= 2")

    matrices = _attention_to_numpy(matrices)
    if matrices.ndim != 3:
        raise ValueError(
            f"Expected `matrices` to be a list or 3D array of shape (n, R, C). "
            f"Got shape: {matrices.shape}"
        )

    n, R, C = matrices.shape

    mean_vals = np.mean(matrices, axis=0)
    std_vals = np.std(matrices, axis=0)
    if use_std_error:
        error_vals = std_vals / np.sqrt(n)
    else:
        error_vals = std_vals

    if x_labels is None:
        x_labels = [f"X{j}" for j in range(C)]
    if y_labels is None:
        y_labels = [f"Y{i}" for i in range(R)]
    if len(x_labels) != C:
        raise ValueError(
            f"len(x_labels) must match matrix columns: {len(x_labels)} != {C}"
        )
    if len(y_labels) != R:
        raise ValueError(
            f"len(y_labels) must match matrix rows: {len(y_labels)} != {R}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    lower_all = (mean_vals - error_vals).min()
    upper_all = (mean_vals + error_vals).max()
    vmin = min(lower_all, mean_vals.min())
    vmax = max(upper_all, mean_vals.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    ax, plotter = _create_tablelens_heatmap(
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
        rotate_x_labels_90=True,
    )

    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    max_err = error_vals.max()
    if max_err < 1e-12:
        max_err = 1.0

    def make_radial_gradient_image(inner_rgba, outer_rgba, N=100):
        inner_rgba = np.array(inner_rgba, dtype=float)
        outer_rgba = np.array(outer_rgba, dtype=float)

        gradient = np.zeros((N, N, 4), dtype=np.float32)
        center = (N - 1) / 2.0
        radius = center

        for r in range(N):
            for c in range(N):
                dist = np.sqrt((r - center) ** 2 + (c - center) ** 2)
                t = min(dist / radius, 1.0)
                gradient[r, c, :] = (1 - t) * inner_rgba + t * outer_rgba

        return gradient

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    for i in range(R):
        for j in range(C):
            err = error_vals[i, j]
            if err < 1e-12:
                continue

            radius = (err / max_err) * max_circle_ratio * circle_scale

            if use_white_center:
                val_lower = mean_vals[i, j]
                val_upper = mean_vals[i, j] + err
            else:
                val_lower = mean_vals[i, j] - (err * color_contrast_scale)
                val_upper = mean_vals[i, j] + (err * color_contrast_scale)

            val_lower = max(val_lower, vmin)
            val_lower = min(val_lower, vmax)
            val_upper = max(val_upper, vmin)
            val_upper = min(val_upper, vmax)

            if use_white_center:
                inner_rgba = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
            else:
                inner_rgba = np.array(cmap_obj(norm(val_lower)), dtype=float)

            outer_rgba = np.array(cmap_obj(norm(val_upper)), dtype=float)

            gradient_img = make_radial_gradient_image(
                inner_rgba=inner_rgba, outer_rgba=outer_rgba, N=radial_resolution
            )

            x_center = col_centers[j]
            y_center = row_centers[i]
            x_left = x_center - radius
            x_right = x_center + radius
            y_bottom = y_center - radius
            y_top = y_center + radius

            im = ax.imshow(
                gradient_img,
                extent=[x_left, x_right, y_bottom, y_top],
                origin="lower",
                zorder=3,
            )
            circ = Circle((x_center, y_center), radius=radius, transform=ax.transData)
            im.set_clip_path(circ)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Check Stability heatmap saved to {save_path}")

    return ax


def visualize_attention_evolution_sparklines(
    attentions_over_time: Any,
    tokens: Sequence[str],
    layer: int,
    head: int,
    title: str = "Attention Evolution Over Training",
    xlabel: str = "Tokens Attended to",
    ylabel: str = "Tokens Attending",
    figsize: Tuple[float, float] = (12, 10),
    sparkline_color_dark: str = THEME_SPARKLINE_LOW,
    sparkline_color_light: str = THEME_SPARKLINE_HIGH,
    sparkline_color_fixed: Optional[str] = None,
    sparkline_color_mode: str = "auto",
    sparkline_linewidth: float = 1.0,
    sparkline_alpha: float = 0.8,
    gamma: float = 1.5,
    normalize_sparklines: bool = False,
    save_path: Optional[str] = None,
) -> Axes:
    """
    Visualize the evolution of attention matrices over training epochs with sparklines.

    Parameters
    ----------
    attentions_over_time
        Array shaped ``(n_epochs, layers, heads, n_tokens, n_tokens)``.
    tokens
        Required token labels for both axes.
    layer, head
        Layer and head indices to extract from each epoch.
    sparkline_color_mode
        Color selection mode. ``"auto"`` chooses dark or light based on the
        cell background, ``"dark"`` always uses ``sparkline_color_dark``,
        ``"light"`` always uses ``sparkline_color_light``, and ``"fixed"``
        uses ``sparkline_color_fixed``.
    save_path
        Optional output path. If None, the figure is not saved.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the visualization.
    """
    if tokens is None:
        raise ValueError("tokens is required for sparkline axis labels.")

    valid_color_modes = {"auto", "dark", "light", "fixed"}
    if sparkline_color_mode not in valid_color_modes:
        raise ValueError(
            "sparkline_color_mode must be one of "
            f"{sorted(valid_color_modes)}, got {sparkline_color_mode!r}"
        )

    if sparkline_color_mode == "fixed" and sparkline_color_fixed is None:
        raise ValueError(
            "sparkline_color_fixed is required when sparkline_color_mode='fixed'."
        )

    if not isinstance(attentions_over_time, np.ndarray):
        try:
            if torch.is_tensor(attentions_over_time):
                attentions_over_time = attentions_over_time.detach().cpu().numpy()
            else:
                attentions_over_time = np.array(attentions_over_time)
        except Exception as e:
            raise ValueError(f"Failed to convert input to numpy array: {str(e)}")

    if attentions_over_time.ndim != 5:
        raise ValueError(
            f"Expected attentions_over_time to have 5 dimensions [n_epochs, layers, heads, n_tokens, n_tokens], "
            f"but got shape {attentions_over_time.shape}"
        )

    _, num_layers, num_heads, _, _ = attentions_over_time.shape
    if not 0 <= layer < num_layers:
        raise ValueError(f"layer {layer} is out of range for {num_layers} layers")
    if not 0 <= head < num_heads:
        raise ValueError(f"head {head} is out of range for {num_heads} heads")

    matrices = []
    for epoch_attn in attentions_over_time:
        attn = epoch_attn[layer][head]
        matrices.append(attn)

    attention_stack = np.stack(matrices)
    n_epochs, n_tokens, num_cols = attention_stack.shape

    if n_tokens != num_cols:
        raise ValueError(
            "attention matrices must be square when one token list is used, "
            f"got {(n_tokens, num_cols)}"
        )

    if len(tokens) != n_tokens:
        raise ValueError(
            f"len(tokens) must match n_tokens: {len(tokens)} != {n_tokens}"
        )

    avg_attention = np.mean(attention_stack, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    min_val = avg_attention.min()
    max_val = avg_attention.max()
    if np.isclose(min_val, max_val):
        max_val = min_val + 1e-9
    norm = PowerNorm(gamma=gamma, vmin=min_val, vmax=max_val)

    x_labels = [_bold_special_tokens(token) for token in tokens]
    y_labels = [_bold_special_tokens(token) for token in tokens]

    ax, plotter = _create_tablelens_heatmap(
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
    )

    row_centers = plotter.row_centers
    col_centers = plotter.col_centers

    def get_sparkline_color(cell_intensity):
        if sparkline_color_mode == "dark":
            return sparkline_color_dark
        if sparkline_color_mode == "light":
            return sparkline_color_light
        if sparkline_color_mode == "fixed":
            return sparkline_color_fixed

        norm_tmp = PowerNorm(gamma=gamma, vmin=min_val, vmax=max_val)
        middle_value = norm_tmp.inverse(0.5)
        return (
            sparkline_color_light
            if cell_intensity > middle_value
            else sparkline_color_dark
        )

    if not normalize_sparklines:
        global_min = attention_stack.min()
        global_max = attention_stack.max()

    for i in range(n_tokens):
        for j in range(n_tokens):
            values = attention_stack[:, i, j]
            y_center = row_centers[i]
            x_center = col_centers[j]

            width = col_centers[1] - col_centers[0] if len(col_centers) > 1 else 1.0
            height = row_centers[1] - row_centers[0] if len(row_centers) > 1 else 1.0

            if normalize_sparklines:
                cell_min = values.min()
                cell_max = values.max()
                if cell_max > cell_min:
                    norm_values = (values - cell_min) / (cell_max - cell_min)
                else:
                    norm_values = np.ones_like(values) * 0.5
            else:
                if global_max > global_min:
                    norm_values = (values - global_min) / (global_max - global_min)
                else:
                    norm_values = np.ones_like(values) * 0.5

            x = np.linspace(x_center - width * 0.4, x_center + width * 0.4, n_epochs)
            y = y_center - (norm_values - 0.5) * height * 0.7

            cell_intensity = avg_attention[i, j]
            sparkline_color = get_sparkline_color(cell_intensity)
            ax.plot(
                x,
                y,
                color=sparkline_color,
                linewidth=sparkline_linewidth,
                alpha=sparkline_alpha,
            )

    if sparkline_color_mode == "auto":
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
    else:
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=get_sparkline_color(avg_attention.max()),
                lw=sparkline_linewidth,
                label="Trend",
            )
        ]

    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, -0.1))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    return ax
