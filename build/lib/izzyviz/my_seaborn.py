import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

# Helper function implementations (simplified)
def to_utf8(obj):
    """Return a string representing a Python object.

    Strings (i.e. type ``str``) are returned unchanged.

    Byte strings (i.e. type ``bytes``) are returned as UTF-8-decoded strings.

    For other objects, the method ``__str__()`` is called, and the result is
    returned as a string.

    Parameters
    ----------
    obj : object
        Any Python object

    Returns
    -------
    s : str
        UTF-8-decoded string representation of ``obj``

    """
    if isinstance(obj, str):
        return obj
    try:
        return obj.decode(encoding="utf-8")
    except AttributeError:  # obj is not bytes-like
        return str(obj)
    
def _matrix_mask(data, mask):
    """Ensure that data and mask are compatible and add missing values.

    Values will be plotted for cells where ``mask`` is ``False``.

    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.

    """
    if mask is None:
        mask = np.zeros(data.shape, bool)

    if isinstance(mask, np.ndarray):
        # For array masks, ensure that shape matches data then convert
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")

        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=bool)

    elif isinstance(mask, pd.DataFrame):
        # For DataFrame masks, ensure that semantic labels match data
        if not mask.index.equals(data.index) \
           and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing data to the mask
    # This works around an issue where `plt.pcolormesh` doesn't represent
    # missing data properly
    mask = mask | pd.isnull(data)

    return mask


def _index_to_ticklabels(index):
    """Convert a pandas index or multiindex into ticklabels."""
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values

def _index_to_label(index):
    """Convert a pandas index or multiindex to an axis label."""
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name

def despine(fig=None, ax=None, top=True, right=True, left=False,
            bottom=False, offset=None, trim=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    offset : int or dict, optional
        Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
    trim : bool, optional
        If True, limit spines to the smallest and largest major tick
        on each non-despined axis.

    Returns
    -------
    None

    """
    # Get references to the axes we want
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            # Toggle the spine objects
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
            if offset is not None and is_visible:
                try:
                    val = offset.get(side, 0)
                except AttributeError:
                    val = offset
                ax_i.spines[side].set_position(('outward', val))

        # Potentially move the ticks
        if left and not right:
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.minorTicks
            )
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.minorTicks
            )
            ax_i.xaxis.set_ticks_position("top")
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if trim:
            # clip off the parts of the spines that extend past major ticks
            xticks = np.asarray(ax_i.get_xticks())
            if xticks.size:
                firsttick = np.compress(xticks >= min(ax_i.get_xlim()),
                                        xticks)[0]
                lasttick = np.compress(xticks <= max(ax_i.get_xlim()),
                                       xticks)[-1]
                ax_i.spines['bottom'].set_bounds(firsttick, lasttick)
                ax_i.spines['top'].set_bounds(firsttick, lasttick)
                newticks = xticks.compress(xticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_xticks(newticks)

            yticks = np.asarray(ax_i.get_yticks())
            if yticks.size:
                firsttick = np.compress(yticks >= min(ax_i.get_ylim()),
                                        yticks)[0]
                lasttick = np.compress(yticks <= max(ax_i.get_ylim()),
                                       yticks)[-1]
                ax_i.spines['left'].set_bounds(firsttick, lasttick)
                ax_i.spines['right'].set_bounds(firsttick, lasttick)
                newticks = yticks.compress(yticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_yticks(newticks)

def _draw_figure(fig):
    """Force draw of a matplotlib figure, accounting for back-compat."""
    # See https://github.com/matplotlib/matplotlib/issues/19197 for context
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass

def relative_luminance(color):
    """Calculate the relative luminance of a color according to W3C standards

    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.

    Returns
    -------
    luminance : float(s) between 0 and 1

    """
    rgb = mpl.colors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])
    try:
        return lum.item()
    except ValueError:
        return lum

def get_colormap(name):
    """Handle changes to matplotlib colormap interface in 3.6."""
    try:
        return mpl.colormaps[name]
    except AttributeError:
        return mpl.cm.get_cmap(name)


def axis_ticklabels_overlap(labels):
    """Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of matplotlib ticklabels

    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.

    """
    if not labels:
        return False
    try:
        bboxes = [l.get_window_extent() for l in labels]
        overlaps = [b.count_overlaps(bboxes) for b in bboxes]
        return max(overlaps) > 1
    except RuntimeError:
        # Issue on macos backend raises an error in the above code
        return False


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

class _HeatMapper:
    """Draw a heatmap plot of a matrix with variable cell sizes."""
    
    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cbar, cbar_kws,
                 xticklabels=True, yticklabels=True, mask=None,
                 column_widths=None, row_heights=None, norm=None):
        """Initialize the plotting object."""
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)
    
        # Validate the mask and convert to DataFrame
        mask = _matrix_mask(data, mask)
    
        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)
    
        # Get good names for the rows and columns
        if isinstance(xticklabels, int):
            self.xtickevery = xticklabels
            self.xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            self.xtickevery = 1
            self.xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            self.xtickevery = 1
            self.xticklabels = []
        elif xticklabels == "auto":
            self.xtickevery = "auto"
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xtickevery = 1
            self.xticklabels = xticklabels
    
        if isinstance(yticklabels, int):
            self.ytickevery = yticklabels
            self.yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            self.ytickevery = 1
            self.yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            self.ytickevery = 1
            self.yticklabels = []
        elif yticklabels == "auto":
            self.ytickevery = "auto"
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.ytickevery = 1
            self.yticklabels = yticklabels
    
        # Store the column widths and row heights
        self.column_widths = column_widths
        self.row_heights = row_heights
    
        # Get good names for the axis labels
        xlabel = _index_to_label(data.columns)
        ylabel = _index_to_label(data.index)
        self.xlabel = xlabel if xlabel is not None else ""
        self.ylabel = ylabel if ylabel is not None else ""

        self.col_centers = None
        self.row_centers = None

        self.norm = norm

        # self.top_5_12_threshold = annot_kws.pop('top_5_12_threshold', None)
        # self.annot_kws = {} if annot_kws is None else annot_kws.copy()
    
        # Determine good default values for the colormapping
        self._determine_cmap_params(plot_data, vmin, vmax,
                                    cmap, center, robust)
    
        # Sort out the annotations
        if annot is None or annot is False:
            annot = False
            annot_data = None
        else:
            if isinstance(annot, bool):
                annot_data = plot_data
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != plot_data.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)
            annot = True
    
        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data
    
        self.annot = annot
        self.annot_data = annot_data
    
        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws.copy()
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
    
    def _determine_cmap_params(self, plot_data, vmin, vmax,
                               cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""

        # plot_data is a np.ma.array instance
        calc_data = plot_data.astype(float).filled(np.nan)
        if vmin is None:
            if robust:
                vmin = np.nanpercentile(calc_data, 2)
            else:
                vmin = np.nanmin(calc_data)
        if vmax is None:
            if robust:
                vmax = np.nanpercentile(calc_data, 98)
            else:
                vmax = np.nanmax(calc_data)
        self.vmin, self.vmax = vmin, vmax

        # Choose default colormaps if not provided
        if cmap is None:
            if center is None:
                self.cmap = cm.rocket
            else:
                self.cmap = cm.icefire
        elif isinstance(cmap, str):
            self.cmap = get_colormap(cmap)
        elif isinstance(cmap, list):
            self.cmap = mpl.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap

        # Recenter a divergent colormap
        if center is not None:

            # Copy bad values
            # in mpl<3.2 only masked values are honored with "bad" color spec
            # (see https://github.com/matplotlib/matplotlib/pull/14257)
            bad = self.cmap(np.ma.masked_invalid([np.nan]))[0]

            # under/over values are set for sure when cmap extremes
            # do not map to the same color as +-inf
            under = self.cmap(-np.inf)
            over = self.cmap(np.inf)
            under_set = under != self.cmap(0)
            over_set = over != self.cmap(self.cmap.N - 1)

            vrange = max(vmax - center, center - vmin)
            normlize = mpl.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            self.cmap = mpl.colors.ListedColormap(self.cmap(cc))
            self.cmap.set_bad(bad)
            if under_set:
                self.cmap.set_under(under)
            if over_set:
                self.cmap.set_over(over)
    
    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        mesh.update_scalarmappable()
        xpos, ypos = np.meshgrid(self.col_centers, self.row_centers)
        for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                       mesh.get_array().flat, mesh.get_facecolors(),
                                       self.annot_data.flat):
            if m is not np.ma.masked:
                lum = relative_luminance(color)
                text_color = ".15" if lum > .408 else "w"
                annotation = ("{:" + self.fmt + "}").format(val)
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kws)
                ax.text(x, y, annotation, **text_kwargs)

    # def _annotate_heatmap(self, ax, mesh):
    # """Add textual labels with the value in each cell."""
    # mesh.update_scalarmappable()
    # xpos, ypos = np.meshgrid(self.col_centers, self.row_centers)
    
    # for x, y, m, color, val in zip(xpos.flat, ypos.flat,
    #                                mesh.get_array().flat, mesh.get_facecolors(),
    #                                self.annot_data.flat):
    #     if m is not np.ma.masked:
    #         lum = relative_luminance(color)
            
    #         # Check if the color is lighter than light blue
    #         # If the luminance is high, or the color resembles a light blue, use black text
    #         if lum > 0.6:
    #             text_color = ".15"  # Black for better contrast on light colors
    #         elif isinstance(color, (list, tuple)) and len(color) == 3 and color[2] > 0.7 and lum > 0.4:
    #             text_color = ".15"  # Black if the color has a high blue component and is light
    #         else:
    #             text_color = "w"  # White text otherwise
            
    #         annotation = ("{:" + self.fmt + "}").format(val)
    #         text_kwargs = dict(color=text_color, ha="center", va="center")
    #         text_kwargs.update(self.annot_kws)
    #         ax.text(x, y, annotation, **text_kwargs)


    # def _annotate_heatmap(self, ax, mesh):
    # """Add textual labels with the value in each cell."""
    # mesh.update_scalarmappable()
    # xpos, ypos = np.meshgrid(self.col_centers, self.row_centers)
    # for x, y, m, color, val in zip(xpos.flat, ypos.flat,
    #                                mesh.get_array().flat, mesh.get_facecolors(),
    #                                self.annot_data.flat):
    #     if m is not np.ma.masked:
    #         # Convert the annotation value to float
    #         if val != '':
    #             val_float = float(val)
    #         else:
    #             val_float = None

    #         # Determine text color based on the threshold
    #         if self.top_5_12_threshold is not None and val_float is not None:
    #             if val_float >= self.top_5_12_threshold:
    #                 text_color = "white"
    #             else:
    #                 text_color = "black"
    #         else:
    #             # Default behavior based on luminance
    #             lum = relative_luminance(color)
    #             text_color = ".15" if lum > .408 else "w"
    #         annotation = ("{:" + self.fmt + "}").format(val)
    #         text_kwargs = dict(color=text_color, ha="center", va="center")
    #         text_kwargs.update(self.annot_kws)
    #         ax.text(x, y, annotation, **text_kwargs)

    
    def _skip_ticks(self, positions, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        if tickevery == 0 or len(labels) == 0:
            ticks, labels = [], []
        elif tickevery == 1:
            ticks, labels = positions, labels
        else:
            ticks = positions[::tickevery]
            labels = labels[::tickevery]
        return ticks, labels
    
    def _auto_ticks(self, ax, labels, positions, axis):
        """Determine ticks and ticklabels that minimize overlap."""
        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]
        axis_obj = [ax.xaxis, ax.yaxis][axis]
        tick = axis_obj.get_major_ticks()[0]
        fontsize = tick.label1.get_size()
        max_ticks = int(size * 72 / fontsize)
        if max_ticks < 1:
            return [], []
        tick_every = max(len(labels) // max_ticks, 1)
        ticks = positions[::tick_every]
        labels = labels[::tick_every]
        return ticks, labels
    
    def plot(self, ax, cax, kws):
        """Draw the heatmap on the provided Axes."""
        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # Avoid setting vmin/vmax if norm is set
        if self.norm is not None:
            kws.pop('vmin', None)
            kws.pop('vmax', None)
        else:
            kws.setdefault('vmin', self.vmin)
            kws.setdefault('vmax', self.vmax)
    
        # setting vmin/vmax in addition to norm is deprecated
        # so avoid setting if norm is set
        # if kws.get("norm") is None:
        #     kws.setdefault("vmin", self.vmin)
        #     kws.setdefault("vmax", self.vmax)
    
        # Compute the positions of the cell edges
        nrows, ncols = self.data.shape
    
        if self.column_widths is not None:
            col_positions = np.concatenate([[0], np.cumsum(self.column_widths)])
        else:
            col_positions = np.arange(ncols + 1)
    
        if self.row_heights is not None:
            row_positions = np.concatenate([[0], np.cumsum(self.row_heights)])
        else:
            row_positions = np.arange(nrows + 1)
    
        X, Y = np.meshgrid(col_positions, row_positions)

        self.col_positions = col_positions
        self.row_positions = row_positions
    
        # Draw the heatmap
        mesh = ax.pcolormesh(X, Y, self.plot_data, cmap=self.cmap, norm=self.norm, **kws)
    
        # Set the axis limits
        ax.set_xlim(col_positions[0], col_positions[-1])
        ax.set_ylim(row_positions[0], row_positions[-1])
    
        # Invert the y axis to show the plot in matrix form
        ax.invert_yaxis()
    
        # Possibly add a colorbar
        if self.cbar:
            cb = ax.figure.colorbar(mesh, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            if kws.get('rasterized', False):
                cb.solids.set_rasterized(True)
    
        # Compute the centers of the columns and rows
        self.col_centers = (col_positions[:-1] + col_positions[1:]) / 2
        self.row_centers = (row_positions[:-1] + row_positions[1:]) / 2

    
        # Adjust ticks and labels
        if self.xtickevery == "auto":
            xticks, xticklabels = self._auto_ticks(ax, self.xticklabels, self.col_centers, axis=0)
        else:
            xticks, xticklabels = self._skip_ticks(self.col_centers, self.xticklabels, self.xtickevery)
    
        if self.ytickevery == "auto":
            yticks, yticklabels = self._auto_ticks(ax, self.yticklabels, self.row_centers, axis=1)
        else:
            yticks, yticklabels = self._skip_ticks(self.row_centers, self.yticklabels, self.ytickevery)
    
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation="vertical")
        plt.setp(ax.get_yticklabels(), va="center")
    
        # Possibly rotate them if they overlap
        _draw_figure(ax.figure)
    
        if axis_ticklabels_overlap(ax.get_xticklabels()):
            plt.setp(ax.get_xticklabels(), rotation="vertical")
        if axis_ticklabels_overlap(ax.get_yticklabels()):
            plt.setp(ax.get_yticklabels(), rotation="horizontal")
    
        # Add the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
    
        # Annotate the cells with the formatted values
        if self.annot:
            self._annotate_heatmap(ax, mesh)

def heatmap(
    data, *,
    vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=None, fmt=".2g", annot_kws=None,
    linewidths=0, linecolor="white",
    cbar=True, cbar_kws=None, cbar_ax=None,
    square=False, xticklabels="auto", yticklabels="auto",
    mask=None, ax=None,
    column_widths=None, row_heights=None, norm=None,
    **kwargs
):
    """Plot rectangular data as a color-encoded matrix with variable cell sizes."""
    # Initialize the plotter object
    plotter = _HeatMapper(
        data, vmin, vmax, cmap, center, robust, annot, fmt,
        annot_kws, cbar, cbar_kws, xticklabels,
        yticklabels, mask, column_widths, row_heights, norm
    )
    
    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor
    
    # Draw the plot and return the Axes and the plotter
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax, plotter  # Return both ax and plotter