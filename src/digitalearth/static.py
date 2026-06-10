"""StaticGlyph — the legacy static plotter (deprecated; use ``digitalearth.Map`` / ``quickmap``)."""
import warnings
from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from cleopatra.array_glyph import ArrayGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from geopandas import GeoDataFrame
from pyramids.dataset import Dataset

#: Message emitted by every StaticGlyph entry point (PD-1 / L-2).
_DEPRECATION_MSG = (
    "StaticGlyph is deprecated and will be removed in a future release; use digitalearth.Map "
    "(or digitalearth.quickmap) instead."
)


def _warn_deprecated() -> None:
    """Emit the StaticGlyph ``DeprecationWarning`` (``stacklevel=3`` to point at the caller)."""
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)


class StaticGlyph:
    """Legacy static raster / catchment plotter.

    .. deprecated::
        ``StaticGlyph`` predates the :class:`~digitalearth.scene.map.Map` scene API and is retained only for
        backward compatibility — every entry point emits a :class:`DeprecationWarning`. Prefer
        :func:`~digitalearth.api.quickmap` or :class:`~digitalearth.scene.map.Map`.
    """

    figure_default_options = dict(
        ylabel="",
        xlabel="",
        legend="",
        legend_size=10,
        figsize=(10, 8),
        labelsize=10,
        fontsize=10,
        name="hist.tif",
        color1="#3D59AB",
        color2="#DC143C",
        linewidth=3,
        Axisfontsize=15,
    )

    def __init__(self):
        _warn_deprecated()

    @staticmethod
    def plot(
        src: Union[Dataset, np.ndarray],
        band: int = 1,
        point_color: str = "red",
        point_size: Union[int, float] = 100,
        pid_color="blue",
        pid_size: Union[int, float] = 10,
        **kwargs
    ):
        """plot.

            plot an array/ gdal dataset

        Parameters
        ----------
        src : [array/Dataset]
            the array/gdal raster you want to plot.
        band: [int]
            band index. Default is 1.
        point_color : [str], optional
            color of the points. The default is 'red'.
        point_size : [integer], optional
            size of the points. The default is 100.
        pid_color : [str]
            the ID of the Point.The default is "blue".
        pid_size : [integer]
            size of the ID text. The default is 10.
        pid_color : []


        **kwargs : [dict]
            keys:
                nodataval: [int, float]
                    the no_data_value in case the first parameter is an array. Default is np.nan.
                figsize: Tuple[int, int] = (8, 8),
                title: Any = "Total Discharge",
                title_size: Union[int, float] = 15,
                cbar_length: Union[int, float] = 0.75,
                orientation: str = "vertical",
                cbar_label_size: Union[int, float] = 12,
                cbar_label: str = "Color bar label",
                rotation: Union[int, float] = -90,
                ticks_spacing: Union[int, float] = 5,
                num_size: Union[int, float] = 8,
                color_scale: int = 1,
                cmap: str = "coolwarm_r",
                gamma: Union[int, float] = 0.5,
                linscale: Union[int, float] = 0.001,
                linthresh: Union[int, float] = 0.0001,
                midpoint: int = 0,
                display_cell_value: bool = False,
                background_color_threshold=None,

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
        """
        _warn_deprecated()
        if isinstance(src, Dataset):
            arr = src.read_array()
            no_data_value = src.no_data_value[band - 1]
        else:
            arr = src
            if "no_data_value" not in kwargs.keys():
                raise ValueError(
                    "If the first parameter is a numpy.ndarray object you have to enter a kwargs 'no_data_value'"
                    "value"
                )
            else:
                # pop (not read) so the value is not forwarded to ArrayGlyph.plot, which would reject it.
                no_data_value = kwargs.pop("no_data_value")
        # convert the array to float as integer array gives error when compared to float
        arr = arr.astype(np.float32)

        if no_data_value is not None:
            arr[np.isclose(arr, no_data_value, rtol=0.001)] = np.nan

        # Pop `points` out of kwargs: cleopatra's ArrayGlyph.plot now has a
        # native `points` parameter expecting a 3-column [value, row, col]
        # ndarray, whereas here `points` is a GeoDataFrame overlaid manually
        # below. Keeping it in kwargs would clash with that native parameter.
        points = kwargs.pop("points", None)
        if points is not None:
            points["rows"] = np.nan
            points["col"] = np.nan
            # locate the points in the array (row, col indices)
            points.loc[:, ["rows", "col"]] = src.map_to_array_coordinates(points)

        # Avoid a None no_data_value reaching ArrayGlyph's exclude_value: it compares with np.isclose
        # (raises on None). np.nan is ArrayGlyph's "no exclusion" sentinel (an empty list would IndexError).
        exclude = [no_data_value] if no_data_value is not None else np.nan
        array = ArrayGlyph(arr, exclude_value=exclude)
        fig, ax = array.plot(**kwargs)

        points_ids = list()
        if points is not None:
            row = points.loc[:, "rows"].tolist()
            col = points.loc[:, "col"].tolist()
            # label each point with its "id" column when present, otherwise
            # fall back to the GeoDataFrame index (data may use "fid" or none)
            if "id" in points.columns:
                i_ds = points.loc[:, "id"].tolist()
            else:
                i_ds = points.index.tolist()
            ax.scatter(col, row, color=point_color, s=point_size)
            # TODO: Points = ax.scatter(col, rows, color=point_color, s=point_size)
            #  return the scatter plot object (Points)

            for i in range(len(row)):
                points_ids.append(
                    ax.text(
                        col[i],
                        row[i],
                        i_ds[i],
                        ha="center",
                        va="center",
                        color=pid_color,
                        fontsize=pid_size,
                    )
                )

        return fig, ax

    @staticmethod
    def plotCatchment(
        points: GeoDataFrame,
        column_name: Any,
        poly: GeoDataFrame,
        line: GeoDataFrame,
        scheme: Any = None,
        cmap: str = "viridis",
        size_limits: Tuple[float, float] = (20, 200),
        figsize: Tuple = (8, 8),
        title: Any = "title",
        title_size: int = 15,
        linewidth: float = 0.5,
        save: Union[bool, str] = False,
    ):
        """Plot a catchment: gauge points over a grey sub-catchment fill and a river network.

        Built on **cleopatra + matplotlib**. The gauge ``points`` are drawn as
        a value-coloured, value-scaled scatter (``cleopatra.scatter_glyph.ScatterGlyph``), the ``poly`` features
        as a uniform grey fill, and the ``line`` features as a river network. All three inputs are reprojected to
        the points' CRS; the projection is applied to the data, not to the axes.

        Parameters
        ----------
        points : [GeoDataFrame]
            geodataframe whose ``column_name`` holds the per-point values to plot.
        column_name : [str]
            name of the numeric column that drives both the colour and the marker size of the points.
        poly : [GeoDataFrame]
            geodataframe of polygon geometries (drawn as a uniform grey fill).
        line : [GeoDataFrame]
            geodataframe of line geometries (drawn as the river network).
        scheme : [str], optional
            categorical classification scheme passed to ``ScatterGlyph`` (e.g. ``"quantiles"`` /
            ``"fisher_jenks"``); ``None`` (default) colours on a continuous scale.
        cmap : [str], optional
            colormap for the point values. Default ``"viridis"``.
        size_limits : [tuple], optional
            ``(min, max)`` marker area (points²) mapped to the smallest/largest value. Default ``(20, 200)``.
        figsize : [tuple], optional
            size of the figure. Default ``(8, 8)``.
        title : [str], optional
            title of the figure. Default ``"title"``.
        title_size : [int], optional
            font size of the title. Default ``15``.
        linewidth : [float], optional
            edge width of the grey catchment polygons. Default ``0.5``.
        save : [bool/str], optional
            path (with extension) to save the figure to, or ``False`` (default) to skip saving.

        Returns
        -------
        fig, ax : the matplotlib figure and axes.
        """
        _warn_deprecated()
        # Unify the projection by reprojecting every layer to the points' CRS (non-mutating).
        crs = points.crs if points.crs is not None else 4326
        poly = poly.to_crs(crs)
        line = line.to_crs(crs)
        points = points.to_crs(crs)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Sub-catchment polygons: uniform grey fill (exterior rings, MultiPolygon expanded).
        poly_rings = [
            np.asarray(part.exterior.coords)
            for geom in poly.geometry
            for part in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom])
        ]
        if poly_rings:
            ax.add_collection(
                PolyCollection(
                    poly_rings, facecolors="grey", edgecolors="grey", linewidths=linewidth, zorder=0,
                )
            )

        # River network: line geometries (MultiLineString expanded).
        line_paths = [
            np.asarray(part.coords)
            for geom in line.geometry
            for part in (geom.geoms if geom.geom_type.startswith("Multi") else [geom])
        ]
        if line_paths:
            ax.add_collection(LineCollection(line_paths, colors="C0", linewidths=2.0, zorder=1))

        # Gauge points: coloured and sized by the chosen column (optionally classified by `scheme`).
        values = points[column_name].astype(float).to_numpy()
        glyph = ScatterGlyph(
            points.geometry.x.to_numpy(),
            points.geometry.y.to_numpy(),
            values=values,
            sizes=values,
            ax=ax,
            fig=fig,
            cmap=cmap,
            scheme=scheme,
            size_limits=size_limits,
            size_legend=True,
        )
        glyph.plot()

        ax.set_title(title, fontsize=title_size)
        ax.set_aspect("equal")
        ax.autoscale_view()
        if save:
            fig.savefig(save, bbox_inches="tight", transparent=True)

        return fig, ax
