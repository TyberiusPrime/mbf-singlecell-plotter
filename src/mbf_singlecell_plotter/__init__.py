import numpy as np
from typing import Optional
import collections
import pandas as pd
from natsort import natsorted
import matplotlib
from typing import List
import matplotlib.colors as mcolors
import matplotlib.pyplot as pyplot
from collections import namedtuple

default = object()


def map_to_integers(series, upper, min=None, max=None):
    """Map integers into 0...upper."""
    min = series.min() if min is None else min
    max = series.max() if max is None else max
    zero_to_one = (series - min) / (max - min)
    scaled = zero_to_one * (upper - 1)
    return scaled.astype(int)


def unmap(series, org_series, res):
    """Inverse of map_to_integers"""
    zero_to_one = series / (res - 1)
    mult = zero_to_one * (org_series.max() - org_series.min())
    shifted = mult + org_series.min()
    return shifted


ScatterParts = namedtuple("ScatterParts", ["fig", "ax", "cbar"])

default_colors = [
    "#1C86EE",
    "#008B00",
    "#FF7F00",  # orange
    "#4D4D4D",
    "#FFD700",
    "#7EC0EE",
    "#FB9A99",  # lt pink
    # "#60D060",  # "#90EE90",
    # "#0000FF",
    "#FDBF6F",  # lt orange
    # "#B3B3B3",
    # "#EEE685",
    "#B03060",
    "#FF83FA",
    # "#FF1493",
    # "#0000FF",
    "#36648B",
    "#00CED1",
    "#00FF00",
    "#8B8B00",
    "#CDCD00",
    "#F52A2A",
]


default_color_cmap = matplotlib.colors.ListedColormap(default_colors)


class ScanpyPlotter:
    """Plotting helpers for anndata objects"""

    def __init__(
        self,
        ad,
        embedding,
        cell_type_column: Optional[str] = "cell_type",
        colors=default,
        grid_size=12,
        grid_letters_on_vertical=False,
        boundary_resolution=200,
        boundary_blur=1.1,
        boundary_threshold=0.95,  # more means 'farther out / smoother'
    ):
        """
        @ad - ann addata object
        @cell_type_column - which .obs column has your cell type annotation

        @embedding - one of umap/tsne, or ('pca', 0,1) for the first two PCA components

        """
        self.ad = ad
        if embedding in ad.obsm:
            self.embedding = embedding
        elif "X_" + embedding in ad.obsm:
            self.embedding = "X_" + embedding
        else:
            raise KeyError(
                f"Embedding {embedding} not found in ad.obsm. Available "
                + ",".join(sorted(ad.obsm.keys()))
            )
        # wether or not this has "name ENGS" style variable indices
        self.has_name_and_id = ad.var.index.str.contains(" ").any()
        self.cell_type_column = cell_type_column
        if colors is default:
            cmap = default_color_cmap
        else:
            if isinstance(colors, list):
                cmap = matplotlib.colors.ListedColormap(colors)
            else:
                cmap = colors
        self.cell_type_color_map = cmap

        self.grid_size = grid_size
        if self.grid_size > 26:
            raise ValueError("grid_size max is 26")
        self.grid_letters_on_vertical = grid_letters_on_vertical
        if cell_type_column is not None:
            self.prep_boundaries(boundary_resolution, boundary_blur, boundary_threshold)

    def get_column(self, column):
        """Returns a Series with the data, and the corrected column name"""
        adata = self.ad
        if column in adata.obs:
            pdf = {column: adata.obs[column]}
            column = column
        elif column in adata.var.index:
            pdf = adata[:, adata.var.index == column].to_df()
        else:
            if self.has_name_and_id:
                name_hits = adata.var.index.str.startswith(column + " ")
                if name_hits.sum() == 1:
                    pdf = adata[:, name_hits].to_df()
                    column = pdf.columns[0]
                else:
                    id_hits = adata.var.index.str.endswith(" " + column)
                    if id_hits.sum() == 1:
                        pdf = adata[:, id_hits].to_df()
                        column = pdf.columns[0]
                    else:
                        raise KeyError("Could not find column %s (case 1)" % column)
            else:
                raise KeyError("Could not find column %s" % column)
        return pdf[column], column

    def get_column_cell_type(self):
        return self.ad.obs[self.cell_type_column]

    def get_coordinate_dataframe(self):
        cols = ["x", "y"]
        if isinstance(self.embedding, str):
            pdf = (
                pd.DataFrame(self.ad.obsm[self.embedding], columns=cols)
                .assign(index=self.ad.obs.index)
                .set_index("index")
            )
        elif (
            isinstance(self.embedding, tuple)
            and self.embedding[0] == "pca"
            and len(self.embedding) == 3
        ):
            pdf = (
                pd.DataFrame(
                    self.ad.obsm[self.embedding[0]][:, self.embedding[1:]],
                    columns=cols,
                )
                .assign(index=self.ad.obs.index)
                .set_index("index")
            )
        else:
            raise ValueError("Could not interpret embedding")
        return pdf

    def get_cell_type_categories(self, pdf=None):
        if pdf is None:
            ct = self.get_column_cell_type()
        else:
            ct = pdf["cell_type"]
        if ct.dtype == "category":
            cats = ct.cat.categories
        else:
            cats = natsorted(ct.unique())
        return cats

    def prep_boundaries(self, boundary_resolution, boundary_blur, boundary_threshold):
        import skimage

        # this image we'll use to find the boundaries
        img = np.zeros((boundary_resolution, boundary_resolution), dtype=np.uint8)
        # and this to determine their colors.
        color_img = np.zeros((boundary_resolution, boundary_resolution), dtype=object)
        problematic = {}

        pdf = self.get_coordinate_dataframe().assign(
            cell_type=self.get_column_cell_type()
        )

        cats = self.get_cell_type_categories(pdf)
        for cat_no, cat in enumerate(cats):
            sdf = pdf[pdf.cell_type == cat]
            mapped_x = map_to_integers(
                sdf["x"], boundary_resolution, pdf["x"].min(), pdf["x"].max()
            )
            mapped_y = map_to_integers(
                sdf["y"], boundary_resolution, pdf["y"].min(), pdf["y"].max()
            )
            color = self.cell_type_color_map(cat_no)
            for x, y, c in zip(mapped_x, mapped_y, pdf["cell_type"]):
                img[x][y] = 255
                # this is, of course, another overplotting issue
                # so we take the majority
                if (x, y) in problematic:
                    problematic[x, y][color] += 1
                else:
                    if color_img[x, y] == 0 or color_img[x, y] == color:
                        color_img[x, y] = color
                    else:
                        problematic[x, y] = collections.Counter()
                        problematic[x, y][color] += 1

        for (x, y), counts in problematic.items():
            color_img[x, y] = counts.most_common(1)[0][0]

        # print(np.max(img))
        flooded = skimage.segmentation.flood(img, (0, 0))
        # print(np.max(flooded), flooded.dtype)
        flooded = skimage.filters.gaussian(flooded, boundary_blur)
        # print(np.max(flooded), flooded.dtype)

        # bounds = skimage.segmentation.chan_vese(flooded)
        bounds = flooded < boundary_threshold
        # print(bounds.dtype)
        bounds = skimage.segmentation.find_boundaries(bounds)

        # now turn it into something matplotlib can use
        boundary_points = collections.defaultdict(list)

        def search_color(x, y, dist):
            for xi in range(
                max(0, x - dist), min(x + dist + 1, boundary_resolution - 1)
            ):
                for yi in range(
                    max(y - dist, 0), min(y + dist + 1, boundary_resolution - 1)
                ):
                    col = color_img[xi, yi]
                    if col != 0:
                        return col
            return 0

        for x in range(0, boundary_resolution):
            for y in range(0, boundary_resolution):
                if bounds[x][y]:
                    col = color_img[x, y]
                    if col == 0:
                        rdist = 1
                        while rdist < 100 and col == 0:
                            col = search_color(x, y, rdist)
                            rdist += 1
                    if col != 0:
                        boundary_points["x"].append(x)
                        boundary_points["y"].append(y)
                        boundary_points["color"].append(col)

                        for offset in [1]:  # (1,2,3,4):
                            boundary_points["x"].append(x + offset)
                            boundary_points["y"].append(y)
                            boundary_points["color"].append(col)
                            boundary_points["x"].append(x - offset)
                            boundary_points["y"].append(y)
                            boundary_points["color"].append(col)

                            boundary_points["x"].append(x)
                            boundary_points["y"].append(y + offset)
                            boundary_points["color"].append(col)
                            boundary_points["x"].append(x)
                            boundary_points["y"].append(y - offset)
                            boundary_points["color"].append(col)

                            boundary_points["x"].append(x + offset)
                            boundary_points["y"].append(y + offset)
                            boundary_points["color"].append(col)
                            boundary_points["x"].append(x - offset)
                            boundary_points["y"].append(y - offset)
                            boundary_points["color"].append(col)

                    else:
                        raise ValueError(
                            "Color was still 0 after looking at 20 cells in each direction? Something is not right"
                        )

        self.boundary_dataframe = pd.DataFrame(
            {
                "x": unmap(
                    pd.Series(boundary_points["x"]), pdf["x"], boundary_resolution
                ),
                "y": unmap(
                    pd.Series(boundary_points["y"]), pdf["y"], boundary_resolution
                ),
                "color": boundary_points["color"],
            }
        )

    def _plot_border_cell_types(self, ax, pdf, border_size, bg_color):
        bdf = self.boundary_dataframe
        ax.scatter(
            bdf["x"],
            bdf["y"],
            color=bdf["color"],
            s=border_size,
            alpha=1,
            edgecolors="none",
            linewidth=0,
            marker=".",
        )

    def _add_legends(
        self,
        fig,
        ax,
        include_border_legend,
        include_color_legend,
        is_numerical,
        pdf,
        plot,
        expr_name=None,
        expr_min=None,
        expr_max=None,
        over_threshold=None,
        clip_quantile=0.95,
        cmap=None,
        cmap_ticks=None,
        include_zeros_in_regular_plot=False,
        cbar_label="log2 expression",
        cbar_fontsize=12,
        expression_cmap=default,
        legend_args={},
    ):
        """Add legends to the plot"""
        # Add border cell type legend if requested
        if include_border_legend and self.cell_type_column is not None:
            cmap = self.cell_type_color_map
            cats = self.get_cell_type_categories(pdf)
            for cat_no, cat in enumerate(cats):
                color = cmap(cat_no)
                ax.scatter([], [], color=color, label=cat, s=15, marker="o")

        # Add color legend for categorical data if requested
        elif include_color_legend and not is_numerical:
            # For categorical data, create visible legend entries
            if pdf["expression"].dtype == "category":
                cats = pdf["expression"].cat.categories
            else:
                cats = natsorted(pdf["expression"].unique())
                cell_type_legend_x_pos = 1.40

            if expression_cmap is default:
                cmap = default_color_cmap
            else:
                cmap = expression_cmap

            for ii, kind in enumerate(cats):
                color = cmap.colors[ii % len(cmap.colors)]
                # Create larger, more visible markers for the legend
                ax.scatter([], [], color=color, label=str(kind), s=25, marker="o")

            # Add the variable name as legend title if provided
            if expr_name is not None:
                final_legend_args = {
                    "loc": "upper left",
                    "bbox_to_anchor": (1.05, 1),
                    "title": expr_name,
                }
                if legend_args is not None:
                    final_legend_args.update(legend_args)
                # If cmap_ticks is provided, filter the legend entries to just those values
                if cmap_ticks is not None:
                    # Create handles and labels just for the specified ticks
                    filtered_handles = []
                    filtered_labels = []

                    # Use either the categories from the data or the provided cmap_ticks
                    tick_values = (
                        cmap_ticks
                        if isinstance(cmap_ticks, list)
                        else [str(x) for x in cmap_ticks]
                    )

                    for ii, kind in enumerate(cats):
                        if str(kind) in tick_values:
                            color = cmap.colors[ii % len(cmap.colors)]
                            # Create a marker for the legend entry
                            filtered_handles.append(
                                matplotlib.lines.Line2D(
                                    [0],
                                    [0],
                                    marker="o",
                                    color=color,
                                    markersize=5,
                                    linestyle="",
                                )
                            )
                            filtered_labels.append(str(kind))

                    # Create the legend with just the filtered entries
                    legend = ax.legend(
                        filtered_handles, filtered_labels, **final_legend_args
                    )
                else:
                    # Default behavior - show all categories
                    legend = ax.legend(
                        **final_legend_args,
                    )

                # # Make title more prominent
                # pyplot.setp(legend.get_title(), fontweight="bold")
                # Return early since we've already created the legend with a title
                return

        # Add color legend for numerical data if requested
        cbar = None
        if include_color_legend and is_numerical and plot is not None:
            # try:
            # if cmap_ticks is None:
            #     # if expr_max < 20:
            #     #     ticks = list(range(0, int(np.ceil(expr_max)) + 1))
            #     # else:
            #         ticks = [
            #             0,
            #             int(expr_max * 0.25),
            #             int(expr_max * 0.5),
            #             int(expr_max * 0.75),
            #             int(expr_max),
            #         ]
            # else:
            #     ticks = cmap_ticks
            #
            # if clip_quantile < 1:
            #     ticks.append(over_threshold)

            if clip_quantile < 1 and not include_zeros_in_regular_plot:
                extend = "both"
            elif clip_quantile < 1:
                extend = "max"
            elif not include_zeros_in_regular_plot:
                extend = "min"
            else:
                extend = "neither"

            # Create appropriate and non-overlapping ticks
            if cmap_ticks is None:
                # Generate evenly spaced ticks from min to max
                # if expr_max < 20:
                #     # For smaller ranges, use integer ticks
                #     ticks = list(range(0, int(np.ceil(expr_max)) + 1))
                # else:
                # For larger ranges, use 5 evenly spaced ticks
                ticks = np.linspace(expr_min, expr_max, 5).tolist()
                ticks = [round(t, 2) for t in ticks]
            else:
                ticks = cmap_ticks

            # Ensure the quantile value is included and doesn't overlap with other ticks
            if clip_quantile < 1:
                # Round the threshold to 2 decimal places to avoid floating point issues
                rounded_threshold = round(over_threshold, 4)
                # Remove any ticks that are too close to the threshold
                ticks = [
                    t
                    for t in ticks
                    if abs(t - rounded_threshold) > (expr_max - expr_min) * 0.05
                ]
                # Add the threshold tick
                # ticks.append(rounded_threshold)
                # Sort ticks for proper display
                ticks = sorted(ticks)
            else:
                rounded_threshold = None

            def color_map_label(x, pos):
                # Round x to 2 decimal places for comparison
                rounded_x = round(x, 4)
                rounded_threshold = round(over_threshold, 4)

                if abs(rounded_x - rounded_threshold) < 0.01 and clip_quantile < 1:
                    return f">{rounded_threshold:.2f}"
                return f"{x:.2f}"

            cbar = fig.colorbar(
                plot,
                ax=ax,
                orientation="vertical",
                extend=extend,
                extendrect=True,
                format=matplotlib.ticker.FuncFormatter(color_map_label),
                ticks=ticks,
            )
            cbar.set_label(cbar_label, fontsize=cbar_fontsize)

            # Adjust colorbar tick spacing to avoid overlap
            cbar.ax.tick_params(pad=5)

            if not include_zeros_in_regular_plot:
                # Position the zero label at the bottom of the colorbar
                # Use transform=cbar.ax.transAxes to use axis coordinates (0-1) instead of data coordinates
                cbar.ax.text(
                    1.05,
                    -0.01,
                    "- 0",
                    ha="left",
                    va="top",
                    transform=cbar.ax.transAxes,
                )
            if rounded_threshold is not None:
                # Position the zero label at the bottom of the colorbar
                # Use transform=cbar.ax.transAxes to use axis coordinates (0-1) instead of data coordinates
                cbar.ax.text(
                    1.05,
                    1.04,
                    f"- {rounded_threshold:.2f}",
                    ha="left",
                    va="top",
                    transform=cbar.ax.transAxes,
                )

        # except Exception as e:
        #     print(f"Warning: Could not create colorbar: {e}")

        # Make sure legends don't overlap with plot
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:  # Only create legend if we have items for it
            # Create legend with appropriate title for border_legend and categorical data
            if include_border_legend and self.cell_type_column is not None:
                ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Cell Type")
            elif include_color_legend and not is_numerical and expr_name is not None:
                # Create handles and labels for categorical data
                handles = []
                labels = []

                # If cmap_ticks is provided, filter to just those categories
                if cmap_ticks is not None:
                    tick_values = (
                        cmap_ticks
                        if isinstance(cmap_ticks, list)
                        else [str(x) for x in cmap_ticks]
                    )
                    for ii, kind in enumerate(cats):
                        if str(kind) in tick_values:
                            color = cmap.colors[ii % len(cmap.colors)]
                            handles.append(
                                matplotlib.lines.Line2D(
                                    [0],
                                    [0],
                                    marker="o",
                                    color=color,
                                    markersize=10,
                                    linestyle="",
                                )
                            )
                            labels.append(str(kind))
                else:
                    # Default - show all categories
                    for ii, kind in enumerate(cats):
                        color = cmap.colors[ii % len(cmap.colors)]
                        handles.append(
                            matplotlib.lines.Line2D(
                                [0],
                                [0],
                                marker="o",
                                color=color,
                                markersize=10,
                                linestyle="",
                            )
                        )
                        labels.append(str(kind))

                legend = ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1),
                    title=expr_name,
                )
                # Make title more prominent
                # pyplot.setp(legend.get_title(), fontweight="bold")
            else:
                ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        return cbar

    def plot_cell_density(
        self,
        title=default,
        clip_quantile=0.99,
        border_cell_types=True,
        border_size=15,
        include_color_legend=True,
        include_cell_type_legend=True,
        cell_type_legend_x_pos=default,
        bins=200,
        cmap=default,
        zero_color="#FFFFFF",
        upper_clip_color="#FF0000",
        show_spines=True,
    ) -> (matplotlib.figure.Figure, matplotlib.axes.Axes):
        if self.cell_type_column is None:
            border_cell_types = False
        pdf = self.get_coordinate_dataframe()
        if border_cell_types:
            pdf = pdf.assign(cell_type=self.get_column_cell_type())
        hist = np.histogram2d(
            pdf["x"],
            pdf["y"],
            bins=bins,
        )
        vmax_quantile = np.percentile(hist[0], clip_quantile * 100)
        vmax = np.max(hist[0])

        fig, ax = pyplot.subplots(1)
        if border_cell_types:
            self._plot_border_cell_types(
                ax,
                pdf,
                include_cell_type_legend,
                border_size,
                zero_color,
            )
        del hist
        if cmap is default:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "mine",
                [
                    # "#7f7fFF",
                    "#BFBFFF",
                    "#0000FF",
                ],
                N=256,
            )
            cmap.set_under(zero_color)
            cmap.set_over(upper_clip_color)
        h = ax.hist2d(
            pdf["x"], pdf["y"], bins=bins, vmax=vmax_quantile, cmap=cmap, cmin=1
        )

        def color_map_label(x, pos):
            # Round x to 2 decimal places for comparison
            rounded_x = round(x, 2)
            rounded_quantile = round(vmax_quantile, 2)

            if abs(rounded_x - rounded_quantile) < 0.001 and clip_quantile < 1:
                return f">{rounded_quantile:.2f}"
            return f"{x:.2f}"

        # Create more evenly distributed ticks for the colorbar
        if vmax < 20:
            # For smaller ranges, use integer steps
            step_size = max(1, int(vmax / 5))
            ticks = list(range(1, int(np.ceil(vmax * 0.85)), step_size))
        else:
            # For larger ranges, create evenly spaced ticks but don't go all the way to max
            # This leaves room for the quantile tick
            ticks = np.linspace(1, vmax * 0.85, 4).tolist()
            ticks = [round(t, 2) for t in ticks]

        # Always add the quantile value
        if clip_quantile < 1:
            rounded_quantile = round(vmax_quantile, 2)
            # Remove any ticks that are too close to the quantile value
            ticks = [t for t in ticks if abs(t - rounded_quantile) > (vmax - 1) * 0.05]
            # Make sure the quantile value is included
            ticks.append(rounded_quantile)
            # Sort ticks for proper display
            ticks = sorted(ticks)

        # hide the box around the plot
        ax.spines["top"].set_visible(show_spines)
        ax.spines["bottom"].set_visible(show_spines)
        ax.spines["left"].set_visible(show_spines)
        ax.spines["right"].set_visible(show_spines)

        ax.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

        return fig, ax, h  # Return the histogram plot object for legend creation

    def point_to_grid(self, x_min, x_max, y_min, y_max, x, y):
        # Letters for rows (A-Z)
        x_step = (x_max - x_min) / self.grid_size
        y_step = (y_max - y_min) / self.grid_size
        x_index = int(round((x - x_min) / x_step))
        y_index = int(round((y - y_min) / y_step))
        assert x < x_max, "x outside of x_max range"
        assert y < y_max, "y outside of y_max range"

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: self.grid_size]
        non_letters = [x + 1 for x in range(self.grid_size)]
        if self.grid_letters_on_vertical:
            letters = letters[::-1]  # Reversed to match grid cell labels
            letter = letters[y_index]
            number = non_letters[x_index]
        else:
            letter = letters[x_index]
            number = non_letters[self.grid_size - 1 - y_index]
        if self.grid_letters_on_vertical:
            return (number, letter)
        else:
            return (letter, number)

    def _plot_scatter_core(
        self,
        pdf,
        fig,
        ax,
        expr_name,
        title,
        clip_quantile,
        border_cell_types,
        border_size,
        plot_zeros,
        zero_color,
        zero_dot_size,
        zero_dot_marker,
        zero_value,
        expression_cmap,
        dot_size,
        upper_clip_color,
        upper_clip_label,
        plot_data,
        bg_color,
        plot_categorical_outliers,
        categorical_outlier_quantile,
        anti_overplot,
        include_zeros_in_regular_plot,
        show_spines,
        cmap_ticks,
        cbar_label,
        draw_grid,
        edge_grid,
        label_grid,
        grid_color,
        grid_axes,
        label_axis,
        flip_order,
        x_min,
        x_max,
        y_min,
        y_max,
        default=default,
        default_color_cmap=default_color_cmap,
    ):
        """Core plotting function used by plot_scatter"""
        if fig is not None and ax is None:
            raise ValueError("If you pass fig, you must also pass ax")

        is_numerical = (
            (pdf["expression"].dtype != "object")
            and (pdf["expression"].dtype != "category")
            and (pdf["expression"].dtype != "bool")
        )

        # Initialize plot to None
        plot = None

        ax.set_facecolor(bg_color)

        # Draw grid if requested
        if draw_grid or grid_axes:
            # Limit grid size to 26 (number of letters in alphabet)

            # Create grid

            if draw_grid or edge_grid:
                x_grid = np.linspace(x_min, x_max, self.grid_size + 1)
                y_grid = np.linspace(y_min, y_max, self.grid_size + 1)
                # Draw vertical grid lines
                if draw_grid:
                    for x in x_grid:
                        ax.axvline(
                            x,
                            color=grid_color,
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.5,
                            zorder=0,
                        )

                    # Draw horizontal grid lines
                    for y in y_grid:
                        ax.axhline(
                            y,
                            color=grid_color,
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.5,
                            zorder=0,
                        )
                elif edge_grid:
                    # Only draw the outer border lines from 0..0.5
                    # and from grid_size-0.5..grid_size
                    for x in x_grid:
                        ax.axvline(
                            x,
                            color=grid_color,
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.5,
                            zorder=0,
                            ymin=0.0,
                            ymax=1 / self.grid_size / 2,
                        )
                        ax.axvline(
                            x,
                            color=grid_color,
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.5,
                            zorder=0,
                            ymin=1.0 - 1 / self.grid_size / 2,
                            ymax=1.0,
                        )
                    for y in y_grid:
                        ax.axhline(
                            y,
                            color=grid_color,
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.5,
                            zorder=0,
                            xmin=0,
                            xmax=1 / self.grid_size / 2,
                        )
                        ax.axhline(
                            y,
                            color=grid_color,
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.5,
                            zorder=0,
                            xmin=1.0 - 1 / self.grid_size / 2,
                            xmax=1,
                        )

            # Add labels if requested
            if label_grid:
                cell_width = (x_max - x_min) / self.grid_size
                cell_height = (y_max - y_min) / self.grid_size

                # Place labels in the center of each grid cell
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        cell_x = x_min + (i + 0.5) * cell_width
                        cell_y = y_min + (j + 0.5) * cell_height
                        x_label, y_label = self.point_to_grid(
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            x=cell_x,
                            y=cell_y,
                        )
                        if self.grid_letters_on_vertical:
                            label = f"{y_label}{x_label}"
                        else:
                            label = f"{x_label}{y_label}"
                        ax.text(
                            cell_x,
                            cell_y,
                            label,
                            ha="center",
                            va="center",
                            color=grid_color,
                            fontsize=8,
                            alpha=0.7,
                            zorder=1,
                        )

        if border_cell_types:
            self._plot_border_cell_types(
                ax,
                pdf,
                border_size,
                bg_color,
            )

        cbar = None
        if is_numerical:
            if plot_zeros:  # actually, plot all of them in this color first. That gives you dot sizes to play with.
                sdf = pdf  # [pdf["expression"] == 0]
                ax.scatter(
                    sdf["x"],
                    sdf["y"],
                    color=zero_color,
                    s=zero_dot_size,
                    alpha=1,
                    edgecolors="none",
                    linewidth=0,
                    marker=zero_dot_marker,
                )
            if include_zeros_in_regular_plot:
                sdf = pdf
            else:  # what you should be doing
                sdf = pdf[pdf["expression"] != zero_value]
            if anti_overplot:
                if anti_overplot == "break":
                    sdf = sdf.sort_values("expression", ascending=False)
                else:
                    sdf = sdf.sort_values("expression")
            expr_min = sdf.expression.min()
            expr_max = sdf.expression.max()
            # add these to the legend
            if expression_cmap is default:
                expression_cmap = mcolors.LinearSegmentedColormap.from_list(
                    "mine",
                    [
                        "#000000",
                        "#0000FF",
                        "#FF00FF",
                    ],
                    N=256,
                )
            cmap_limits = expression_cmap.resampled(256)
            cmap_limits.set_under(zero_color)
            cmap_limits.set_over(upper_clip_color)
            over_threshold = sdf["expression"].quantile(clip_quantile)
            # xdf = sdf[sdf["expression"] >= 0]
            if plot_data:
                plot = ax.scatter(
                    sdf["x"],
                    sdf["y"],
                    c=sdf["expression"],  # .clip(0, upper=over_threshold),
                    cmap=cmap_limits,
                    s=dot_size,
                    alpha=1,
                    vmin=expr_min,
                    vmax=over_threshold,
                    marker=".",
                )
            else:
                include_color_legend = False

            def color_map_label(x, pos):
                if x == expr_min:
                    return "<%.2f" % x
                elif x == over_threshold:
                    return upper_clip_label or (">%.2f" % x)
                else:
                    return "%.2f" % x

        else:
            if expression_cmap is default:
                cmap = default_color_cmap
            else:
                cmap = expression_cmap

            if pdf["expression"].dtype == "category":
                cats = pdf["expression"].cat.categories
            else:
                cats = natsorted(pdf["expression"].unique())
            # cats = cats[::-1]

            # Create a list to store scatter plots for legend
            scatter_plots = []
            if (
                plot_zeros and cats[0] == zero_value
            ):  # actually, plot all of them in this color first. That gives you dot sizes to play with.
                sdf = pdf[pdf["expression"] == cats[0]]
                ax.scatter(
                    sdf["x"],
                    sdf["y"],
                    color=zero_color,
                    s=zero_dot_size,
                    alpha=1,
                    edgecolors="none",
                    linewidth=0,
                    marker=".",
                )

            if plot_data:
                order = list(enumerate(cats))
                if flip_order:
                    order = order[::-1]
                for ii, kind in order:
                    if not include_zeros_in_regular_plot and kind == zero_value:
                        continue
                    sdf = pdf[pdf["expression"] == kind]
                    scatter = ax.scatter(
                        sdf["x"],
                        sdf["y"],
                        color=cmap.colors[ii % len(cmap.colors)],
                        s=dot_size,
                        alpha=1,
                        edgecolors="none",
                        linewidth=0,
                        marker=".",
                        # No label here - we'll add legend entries separately
                    )
                    scatter_plots.append(scatter)

                # Use the last scatter plot for the color legend
                if len(scatter_plots) > 0:
                    plot = scatter_plots[-1]

                # plot the outliers again, so they are on *top* of
                # the regular cell clouds
            if plot_categorical_outliers:
                for ii, kind in enumerate(cats):
                    if not include_zeros_in_regular_plot and kind == zero_value:
                        continue
                    sdf = pdf[pdf["expression"] == kind]
                    x_center = sdf["x"].mean()
                    y_center = sdf["y"].mean()
                    euclidean_distance = np.sqrt(
                        (sdf["x"] - x_center) ** 2 + (sdf["y"] - y_center) ** 2
                    )
                    threshold = euclidean_distance.quantile(
                        categorical_outlier_quantile
                    )
                    outliers = euclidean_distance > threshold
                    ax.scatter(
                        sdf["x"][outliers],
                        sdf["y"][outliers],
                        color=cmap.colors[ii % len(cmap.colors)],
                        s=dot_size,
                        # color = 'black',
                        alpha=1,
                        edgecolors="none",
                        linewidth=0,
                        marker=".",
                    )

        # Set up axis ticks and labels
        if grid_axes:
            x_positions, y_positions, x_labels, y_labels = self.grid_labels(
                x_min, x_max, y_min, y_max
            )

            ax.set_xticks(x_positions)
            ax.set_yticks(y_positions)
            # Set custom labels
            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)

            # Show tick marks on all sides
            ax.tick_params(
                axis="both",
                which="both",
                bottom=True,
                top=False,
                left=True,
                right=False,
                labelbottom=True,
                labeltop=False,
                labelleft=True,
                labelright=False,
                labelsize=8,
            )
        elif not label_axis:
            # Default behavior - hide all ticks and labels
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

        # hide x/y axis titles
        ax.set_xlabel("")
        ax.set_ylabel("")

        # hide the box around the plot
        ax.spines["top"].set_visible(show_spines)
        ax.spines["bottom"].set_visible(show_spines)
        ax.spines["left"].set_visible(show_spines)
        ax.spines["right"].set_visible(show_spines)

        # Don't reset tick parameters if we're using grid axes
        if not (draw_grid or grid_axes):
            ax.tick_params(
                reset=True,
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=True,  # ticks along the top edge are off
                labelbottom=True,
                labeltop=True,
                labelleft=True,
                left=True,
            )  # labels along the bottom edge are off

        # add a title to the figure
        if title is default:
            title = expr_name
        ax.set_title(title, fontsize=16)

        # Return the plot object for legend creation (could be None if no plot was created)
        return plot

    def grid_labels(self, x_min, x_max, y_min, y_max):
        # Create custom axis labels for the grid
        x_positions = np.linspace(x_min, x_max, self.grid_size + 1)[:-1] + (
            x_max - x_min
        ) / (self.grid_size * 2)
        y_positions = np.linspace(y_min, y_max, self.grid_size + 1)[:-1] + (
            y_max - y_min
        ) / (self.grid_size * 2)

        # Set custom ticks

        x_labels = [
            self.point_to_grid(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x=x,
                y=y_min,
            )[0]
            for x in x_positions
        ]
        y_labels = [
            self.point_to_grid(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x=x_min,
                y=y,
            )[1]
            for y in y_positions
        ]
        return (x_positions, y_positions, x_labels, y_labels)

    def get_cluster_centers(self, column):
        """
        Calculate the center (average x/y coordinates) for each category in the specified column.
        Also determines the grid coordinate for each center.

        Parameters
        ----------
        column : str
            Name of the column in .obs or gene name to use for clustering

        Returns
        -------
        pd.DataFrame
            DataFrame where the index is the category value and columns are:
            - x: x-coordinate of cluster center
            - y: y-coordinate of cluster center
            - grid: grid coordinate in format like "A1", "B2", etc.

        Raises
        ------
        ValueError
            If the column contains numeric (continuous) data
        """
        # Get the column data
        col_data, col_name = self.get_column(column)

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(
            col_data
        ) and not pd.api.types.is_categorical_dtype(col_data):
            raise ValueError(
                f"Column '{column}' contains numeric data. This function only works with categorical data."
            )

        # Get embedding coordinates
        coords_df = self.get_coordinate_dataframe()

        # Combine coordinates with category data
        merged_df = coords_df.copy()
        merged_df["category"] = col_data

        # Group by category and calculate mean coordinates
        centers = merged_df.groupby("category", observed=True).agg(
            {"x": "median", "y": "median"}
        )

        # Add grid coordinates
        centers["grid"] = centers.apply(
            lambda row: self.get_grid_coordinate(
                row["x"],
                row["y"],
            ),
            axis=1,
        )

        centers.index.name = col_name
        # Return the DataFrame with x, y, and grid coordinates
        return centers

    def get_grid_coordinate(self, embedding_x, embedding_y):
        """
        Determine which grid cell a point falls into using the same grid that would be drawn by plot_scatter.

        Parameters
        ----------
        embedding_x : float
            X coordinate in embedding space
        embedding_y : float
            Y coordinate in embedding space

        Returns
        -------
        str
            Grid coordinate in format like "A1", "B2", etc. where letters represent rows (bottom to top)
            and numbers represent columns (left to right)
        """
        pdf = self.get_coordinate_dataframe()
        x_min, x_max = pdf["x"].min(), pdf["x"].max()
        y_min, y_max = pdf["y"].min(), pdf["y"].max()
        x_label, y_label = self.point_to_grid(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x=embedding_x,
            y=embedding_y,
        )
        if self.grid_letters_on_vertical:
            label = f"{y_label}{x_label}"
        else:
            label = f"{x_label}{y_label}"
        return label

    def plot_scatter(  # noqa: C901
        self,
        gene,
        title=default,
        clip_quantile=0.95,
        border_cell_types=True,
        border_size=15,
        plot_zeros=True,
        zero_color="#D0D0D0",
        zero_dot_size=5,
        zero_dot_marker=".",
        zero_value=0.0,
        expression_cmap=default,
        dot_size=1,
        upper_clip_color="#FF0000",
        upper_clip_label=None,
        cell_type_legend_x_pos=default,
        plot_data=True,
        bg_color="#FFFFFF",
        plot_categorical_outliers=True,
        categorical_outlier_quantile=0.95,
        anti_overplot=True,
        include_zeros_in_regular_plot=None,
        show_spines=True,
        cmap_ticks=None,
        cbar_label=default,
        cbar_fontsize=12,
        include_color_legend=True,
        include_border_legend=True,
        fig=None,
        ax=None,
        draw_grid=False,
        edge_grid=False,
        label_grid=False,
        grid_color="#777777",
        grid_axes=False,
        grid_letters_on_vertical=True,
        label_axis=False,
        facet_variable=None,
        n_col=2,
        flip_order=False,
        x_limits=None,
        y_limits=None,
        legend_args=None,
        default=default,
    ) -> ScatterParts:
        """
        Create scatter plot visualization of gene expression data with optional faceting.

        Parameters
        ----------
        gene : str
            The gene or column to visualize
        facet_variable : str, optional
            Column to use for faceting the plot into subplots
        n_col : int, default=2
            Number of columns in the facet grid layout
        """
        if not isinstance(gene, str):
            raise ValueError("gene must be a single string value")
        if self.cell_type_column is None:
            border_cell_types = False
        # If we're faceting, get the facet variable and create subplots
        if facet_variable is not None:
            facet_values, facet_name = self.get_column(facet_variable)
        else:
            facet_values = True
            facet_name = None
            n_col = 1

        pdf = self.get_coordinate_dataframe()
        pdf = pdf.assign(facet=facet_values)

        if gene == "transform:cell_density":
            is_numerical = True
            expr_name = "cell_density"
            cbar_label = "cell density"
            if title is default:
                title = None

            faceted_pdf = []
            for facet_val in pdf["facet"].unique():
                fdf = pdf[pdf["facet"] == facet_val]

                # Convert histogram to dataframe with density values
                H, xedges, yedges = np.histogram2d(fdf["x"], fdf["y"], bins=200)
                # Get the centers of the bins
                x_centers = (xedges[:-1] + xedges[1:]) / 2
                y_centers = (yedges[:-1] + yedges[1:]) / 2

                # Create a mesh grid of coordinates
                X, Y = np.meshgrid(x_centers, y_centers)

                # Flatten the arrays and create DataFrame
                fdf = pd.DataFrame(
                    {
                        "x": X.flatten(),
                        "y": Y.flatten(),
                        "expression": H.T.flatten(),  # H needs to be transposed to match the x,y coordinates
                    }
                )

                # Filter out zero values for better visualization
                fdf = fdf[fdf["expression"] > 0]
                fdf = fdf.assign(facet=facet_val)
                faceted_pdf.append(fdf)
            pdf = pd.concat(faceted_pdf, ignore_index=True)

        else:
            expr, expr_name = self.get_column(gene)
            is_numerical = (
                (expr.dtype != "object")
                and (expr.dtype != "category")
                and (expr.dtype != "bool")
            )
            if not is_numerical and expr.dtype != "category":
                expr = expr.astype("category")
            if include_zeros_in_regular_plot is None:
                if is_numerical:
                    include_zeros_in_regular_plot = False
                else:
                    include_zeros_in_regular_plot = True

            if cbar_label is default and is_numerical:
                cbar_label = expr_name + ": log2 expression"
            if title is default:
                if is_numerical:
                    title = expr_name
                else:
                    title = None

            pdf = pdf.assign(expression=expr)

            if border_cell_types:
                pdf = pdf.assign(cell_type=self.get_column_cell_type())

        # Get unique facet values and determine grid dimensions
        if isinstance(pdf["facet"], pd.Categorical):
            unique_facets = pdf["facet"].cat.categories
        else:
            unique_facets = natsorted(pdf["facet"].unique())
        n_facets = len(unique_facets)
        n_row = int(np.ceil(n_facets / n_col))

        # Calculate global min/max values for consistent scaling across all facets
        if x_limits is None:
            x_min, x_max = pdf["x"].min(), pdf["x"].max()
        else:
            x_min, x_max = x_limits
        if y_limits is None:
            y_min, y_max = pdf["y"].min(), pdf["y"].max()
        else:
            y_min, y_max = y_limits

        # Create figure with subplots
        fig = pyplot.figure(figsize=(6 * n_col, 5 * n_row), layout="constrained")
        # Set overall figure title if provided
        if title is not None:
            ...
            # fig.suptitle(title, fontsize=20, y=0.98) # can't get the layout engine to actually make room??
        axes = fig.subplots(n_row, n_col, squeeze=False)

        # Plot each facet in its own subplot
        cbar = None
        for i, facet_value in enumerate(unique_facets):
            row = i // n_col
            col = i % n_col
            facet_pdf = pdf[pdf["facet"] == facet_value]

            # Get the axis for this facet
            ax = axes[row, col]

            # Create the subplot title
            if facet_name is not None:
                facet_title = f"{facet_name}: {facet_value}"
            else:
                facet_title = None

            # Only include legends in the last plot
            include_legends_here = i == n_facets - 1

            # Plot this facet
            plot_obj = self._plot_scatter_core(
                pdf=facet_pdf,
                fig=fig,
                ax=ax,
                expr_name=expr_name,
                title=facet_title,
                clip_quantile=clip_quantile,
                border_cell_types=border_cell_types,
                border_size=border_size,
                plot_zeros=plot_zeros,
                zero_color=zero_color,
                zero_dot_size=zero_dot_size,
                zero_dot_marker=zero_dot_marker,
                zero_value=zero_value,
                expression_cmap=expression_cmap,
                dot_size=dot_size,
                upper_clip_color=upper_clip_color,
                upper_clip_label=upper_clip_label,
                plot_data=plot_data,
                bg_color=bg_color,
                plot_categorical_outliers=plot_categorical_outliers,
                categorical_outlier_quantile=categorical_outlier_quantile,
                anti_overplot=anti_overplot,
                include_zeros_in_regular_plot=include_zeros_in_regular_plot,
                show_spines=show_spines,
                cmap_ticks=cmap_ticks,
                cbar_label=cbar_label,
                draw_grid=draw_grid,
                label_grid=label_grid,
                edge_grid=edge_grid,
                grid_color=grid_color,
                grid_axes=grid_axes,
                label_axis=label_axis,
                flip_order=flip_order,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )

        # Hide unused subplots
        for i in range(n_facets, n_row * n_col):
            row = i // n_col
            col = i % n_col
            axes[row, col].set_visible(False)

        # Add legends to the last plot if requested
        cbar = None
        if include_color_legend or include_border_legend:
            last_row = (n_facets - 1) // n_col
            last_col = (n_facets - 1) % n_col
            last_ax = axes[last_row, last_col]
            last_pdf = pdf[pdf["facet"] == unique_facets[-1]]

            # Get expression min/max for legend
            if is_numerical:
                expr_min = last_pdf["expression"].min()
                over_threshold = last_pdf["expression"].quantile(clip_quantile)
                expr_max = over_threshold
            else:
                expr_min = expr_max = over_threshold = None

            cbar = self._add_legends(
                fig=fig,
                ax=last_ax,
                include_border_legend=border_cell_types and include_border_legend,
                include_color_legend=include_color_legend,
                is_numerical=is_numerical,
                pdf=pdf,
                plot=plot_obj,
                expr_name=expr_name,
                expr_min=expr_min,
                expr_max=expr_max,
                over_threshold=over_threshold,
                clip_quantile=clip_quantile,
                cmap=expression_cmap,
                cmap_ticks=cmap_ticks,
                include_zeros_in_regular_plot=include_zeros_in_regular_plot,
                cbar_label=cbar_label,
                cbar_fontsize=cbar_fontsize,
                expression_cmap=expression_cmap,
                legend_args=legend_args,
            )

        return ScatterParts(fig, axes, cbar)

    def grid_local_histogram(self, key, min_cells=10, x_limits=None, y_limits=None):
        expr, expr_name = self.get_column(key)
        is_numerical = (
            (expr.dtype != "object")
            and (expr.dtype != "category")
            and (expr.dtype != "bool")
        )
        if is_numerical:
            raise ValueError("category types only")
        pdf = self.get_coordinate_dataframe()
        if x_limits is None:
            x_min, x_max = pdf["x"].min(), pdf["x"].max()
        else:
            x_min, x_max = x_limits
        if y_limits is None:
            y_min, y_max = pdf["y"].min(), pdf["y"].max()
        else:
            y_min, y_max = y_limits
        x_grid = np.linspace(x_min, x_max + 0.1, self.grid_size + 1)
        y_grid = np.linspace(y_min, y_max + 0.1, self.grid_size + 1)

        x_bins = np.digitize(pdf["x"].values, x_grid) - 1
        y_bins = np.digitize(pdf["y"].values, y_grid) - 1
        valid = (
            (x_bins >= 0)
            & (x_bins < len(x_grid) - 1)
            & (y_bins >= 0)
            & (y_bins < len(y_grid) - 1)
        )
        assert all(valid)
        try:
            df_cells = pd.DataFrame(
                {
                    "x_bin": x_bins[valid],
                    "y_bin": y_bins[valid],
                    "category": expr.loc[pdf.index[valid]].values,
                }
            )
        except ValueError as e:
            raise ValueError("Make sure your obs.keys are distinct!", e)

        # iterate over grid cells
        histogram = {"x": [], "y": [], "category": [], "frequency": [], "total": []}
        for (ix, iy), sub in df_cells.groupby(["x_bin", "y_bin"]):
            if len(sub) >= min_cells:
                freqs = sub["category"].value_counts(normalize=True)
                for cat, freq in freqs.items():
                    histogram["x"].append(ix)
                    histogram["y"].append(iy)
                    histogram["category"].append(cat)
                    histogram["frequency"].append(freq)
                    histogram["total"].append(len(sub))
        histogram = pd.DataFrame(histogram)
        return histogram, (x_min, x_max, y_min, y_max)

    def plot_grid_local_histogram(
        self,
        key,
        colors,
        min_cell_count,
    ):
        from plotnine import (
            ggplot,
            scale_fill_manual,
            scale_x_continuous,
            scale_y_continuous,
            theme_bw,
            element_blank,
            theme,
            geom_tile,
            coord_fixed,
            geom_hline,
            geom_vline,
            aes,
        )

        hdf, (x_min, x_max, y_min, y_max) = self.grid_local_histogram(
            key, min_cell_count
        )
        # ensure deterministic category order per cell
        hdf["category"] = pd.Categorical(
            hdf["category"], sorted(hdf["category"].unique())
        )
        hdf = hdf.sort_values(["x", "y", "category"])

        # cumulative offsets within each cell
        factor = 0.8
        hdf["frequency"] = hdf["frequency"] * factor

        x_offset = []
        for ignored, group in hdf.groupby(["x", "y"]):
            x_offset.extend(group["frequency"].cumsum().shift(fill_value=0))
        hdf["x_offset"] = x_offset

        # center tiles inside the unit cell

        hdf["x_plot"] = (
            hdf["x"] - hdf["frequency"] / 2 - hdf["x_offset"] - (1 - factor) / 2
        )
        hdf["y_plot"] = hdf["y"] + 0.5  # + (1-factor)/4
        # hdf["xx"] = [str(x + 1) for x in hdf["x"]]
        # hdf["xx"] = pd.Categorical(
        #     hdf["xx"], [str(x + 1) for x in range(self.grid_size)]
        # )
        # letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: self.grid_size][::-1]
        # hdf["yy"] = [letters[x] for x in hdf["y"]]
        # hdf["yy"] = pd.Categorical(hdf["yy"], [x for x in letters])
        max_total = hdf["total"].max()
        hdf["adj_total"] = (hdf["total"] / max_total).clip(0.3) * factor

        hdf = hdf[::-1]

        _x_ticks, _y_ticks, x_labels, y_labels = self.grid_labels(
            x_min, x_max, y_min, y_max
        )
        x_ticks = list(range(self.grid_size))
        y_ticks = list(range(self.grid_size))

        p = (
            ggplot(
                hdf,
                aes(
                    x="x",
                    y="y",
                    width="frequency",
                    height=factor,
                    #'adj_total',
                    fill="category",
                ),
            )
            + theme_bw()
            + geom_hline(
                aes(yintercept="xx"),
                data=pd.DataFrame({"xx": list(range(self.grid_size + 1))}),
                color="#D0D0D0",
            )
            + geom_vline(
                aes(xintercept="xx"),
                data=pd.DataFrame({"xx": [x - 1 for x in range(self.grid_size + 1)]}),
                color="#D0D0D0",
            )
            + geom_tile(aes(x="x_plot", y="y_plot"))
            + coord_fixed()
            + scale_x_continuous(
                expand=(0, 0.5, 0, 0.5),
                breaks=x_ticks,
                labels=x_labels,
            )
            + scale_y_continuous(
                expand=(0, 0.5, 0, 0.5),
                breaks=y_ticks,
                labels=y_labels,
            )
            + scale_fill_manual(colors)
            + theme(
                axis_title_x=element_blank(),
                axis_title_y=element_blank(),
                panel_grid=element_blank(),
                axis_ticks_length=3,
                axis_title=element_blank(),
            )
        )
        return p

    def get_grid_coordinates(
        self,
        x_limits=None,
        y_limits=None,
    ) -> List[str]:
        # translate all cell coordinates into grid coordinates
        pdf = self.get_coordinate_dataframe()

        out = []
        for x, y in zip(pdf["x"], pdf["y"]):
            label = self.get_grid_coordinate(x, y)
            out.append(label)
        return pd.Series(out, index=pdf.index)


# display(
#     ScanpyPlotter(ad_scrubbed).plot_scatter(
#         "umap",
#         "IL23A",
#         include_cell_type_legend=True,
#         include_color_legend=True,
#         plot_data=True,
#         plot_zeros=True,
#     )
# )


def barcode_rank_plot_umis(ad, thresholds=[]):
    """The 'how-many-cell-decision-knee plot'.
    Barcode Index (sorted by UMI count) vs umi count.
    Look for the 'cliff and knee' shape.
    Can highlight @thresholds, which need to be
    umi-value (y-value) and color tuples.

    """
    import plotnine as p9
    import dppd, dppd_plotnine  # noqa F401

    dp, X = dppd.dppd()
    umis = (
        ad.X.sum(axis=1).A.ravel()
        if hasattr(ad.X, "A")
        else np.array(ad.X.sum(axis=1)).ravel()
    )
    df = pd.DataFrame({"barcode": ad.obs.index, "umis": umis}).sort_values(
        "umis", ascending=False
    )
    df = df.reset_index(drop=True).reset_index()  # 'index' is rank

    # Keep only leftmost and rightmost barcode index for each distinct umi count
    grouped = df.groupby("umis")["index"].agg(["first", "last"]).reset_index()
    reduced = pd.concat(
        [
            grouped[["umis", "first"]].rename(columns={"first": "index"}),
            grouped[["umis", "last"]].rename(columns={"last": "index"}),
        ]
    )

    # Compute (x, y) points for thresholds
    vlines = []
    hlines = []
    for t, color in thresholds:
        idx = df[df["umis"] >= t]["index"].max()
        if pd.notna(idx):
            vlines.append({"index": idx, "color": color})
            hlines.append({"umis": t, "color": color})
    p = (
        dp(reduced)
        .p9()
        .theme_bw()
        .theme(
            panel_grid_minor=p9.element_line(color="#CCCCCC"),
            panel_grid_major=p9.element_line(color="#AAAAAA"),
        )
        .syc10(
            name="UMI count",
            breaks=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        )
        .sxc10(breaks=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6], name="barcode index")
        .add_line("index+1", "umis")
        .annotation_logticks()
        .scale_color_identity()
        .hide_legend()
    )
    if vlines:
        p = p.add_vline("index", color="color", data=pd.DataFrame(vlines))
    if hlines:
        p = p.add_hline("umis", color="color", data=pd.DataFrame(hlines))

    return p.pd
