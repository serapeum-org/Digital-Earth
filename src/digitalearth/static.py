""" StaticGlyph."""
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cleopatra.array_glyph import ArrayGlyph
from geopandas import GeoDataFrame
from loguru import logger
from pyramids.dataset import Dataset

# from Hapi.plot.visualizer import MidpointNormalize, Glyph


class StaticGlyph:
    """Glyph."""

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
        pass

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
                no_data_value = kwargs["no_data_value"]
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

        array = ArrayGlyph(arr, exclude_value=[no_data_value])
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
        scale_func: Any = None,
        cmap: str = "viridis",
        legend_values: List = [],
        legend_labels: List = [],
        figsize: Tuple = (8, 8),
        title: Any = "title",
        title_size: int = 500,
        linewidth: float = 0.5,
        save: Union[bool, str] = False,
    ):
        """PlotCatchment.

        Parameters
        ----------
        points:[GeoDataFrame]
            geodataframe contains values to plot in one of its columns.
        column_name: [str]
            name of the column you want to plot its values.
        poly: [GeoDataFrame]
            geodataframe contains polygon geometries.
        line: [GeoDataFrame]
            geodataframe contains linestring geometries.
        linewidth:
        title_size:
        legend_labels:
        legend_values:
        cmap:
        scale_func:
        scheme:
        figsize: [Tuple]
            fize oif the figure.
        title:[str]
            title of the figure.
        save: [bool/str]
            if you want to save the plot provide the path with the extention,
            Default is False.
        """
        import geoplot as gplt
        import geoplot.crs as gcrs

        # unify the projection
        if not poly.crs.is_geographic:
            logger.debug(
                "The coordinate system of the poly geodataframe is not geographic "
                "SO, it will be reprojected to WGS-84"
            )
            poly.to_crs(4326, inplace=True)

        epsg = poly.crs.to_json()
        line.to_crs(epsg, inplace=True)
        points.to_crs(epsg, inplace=True)

        pointplot_kwargs = {
            "edgecolor": "white",
            "linewidth": 0.9,
        }  # 'color': "crimson"

        # make sure that the plotted column is numeric
        points[column_name] = points[column_name].map(float)

        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={"projection": gcrs.AlbersEqualArea()}
        )

        if scheme:
            gplt.pointplot(
                points,
                projection=gcrs.AlbersEqualArea(),
                hue=column_name,
                cmap=cmap,
                scale=column_name,
                limits=(4, 20),
                scheme=scheme,
                legend=True,
                legend_var="scale",
                legend_kwargs={"bbox_to_anchor": (1, 0.35)},  # 'loc': 'upper right',
                ax=ax,
                **pointplot_kwargs  # ,
            )
        else:
            if scale_func:
                gplt.pointplot(
                    points,
                    projection=gcrs.AlbersEqualArea(),
                    hue=column_name,
                    cmap=cmap,
                    scale=column_name,
                    limits=(4, 20),
                    scale_func=scale_func,
                    legend=True,
                    legend_var="scale",
                    legend_values=legend_values,
                    legend_labels=legend_labels,
                    legend_kwargs={  # 'loc': 'upper right',
                        "bbox_to_anchor": (1, 0.35)
                    },
                    ax=ax,
                    **pointplot_kwargs  # ,
                )
            else:
                gplt.pointplot(
                    points,
                    projection=gcrs.AlbersEqualArea(),
                    hue=column_name,
                    cmap=cmap,
                    scale=column_name,
                    limits=(4, 20),
                    # scale_func=scale_func,
                    legend=True,
                    legend_var="scale",
                    legend_values=legend_values,
                    legend_labels=legend_labels,
                    legend_kwargs={  # 'loc': 'upper right',
                        "bbox_to_anchor": (1, 0.35)
                    },
                    ax=ax,
                    **pointplot_kwargs  # ,
                )

        gplt.polyplot(
            poly,
            ax=ax,
            edgecolor="grey",
            facecolor="grey",  # 'lightgray',
            linewidth=0.5,
            extent=poly.total_bounds,
        )  # # , zorder=0

        gplt.polyplot(line, ax=ax, linewidth=10)

        plt.title(title, fontsize=title_size)
        # plt.subplots_adjust(top=0.99999, right=0.9999, left=0.000005, bottom=0.000005)
        if save:
            plt.savefig(save, bbox_inches="tight", transparent=True)

        return fig, ax
