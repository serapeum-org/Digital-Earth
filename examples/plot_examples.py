"""Static map examples, written as `#%%` cells to be stepped through one at a time.

Run this from the **repository root** — every data path below is relative to it.

`digitalearth.Map` is the handle: a scene that owns one matplotlib figure and the layers drawn on it.
`Map.field` draws a raster band, `Map.points` / `Map.polygons` / `Map.sankey` draw vector layers, and
`Map.colorbar` / `Map.set_title` decorate the figure. The matplotlib objects are on the scene as `m.fig`
and `m.ax` when you need them, and each layer method hands back the artist it drew, so anything the
styling keywords do not cover can be set on that artist directly.

Most styling keywords (`figsize`, `title`, `cmap`, `ticks_spacing`, `cbar_*`, `color_scale`, …) are
forwarded to the cleopatra glyph underneath, which refuses a keyword it does not know rather than
dropping it.

Set `MPLBACKEND=Agg` to run the whole file with no display; otherwise it draws on TkAgg, cell by cell.
The three catchment cells at the end overwrite the figures in `examples/data/results/`.
"""

import os

import geopandas as gpd
import matplotlib
import pandas as pd
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

from digitalearth import Map

# Leave the backend alone when the environment already chose one (MPLBACKEND=Agg runs this headless);
# otherwise ask for TkAgg, which shows each cell's figure as it is drawn.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("TkAgg")

saveto = "examples/data/results"
# %% small map
acc = Dataset.read_file("examples/data/acc4000.tif")
rhine_acc = Dataset.read_file("examples/data/DEM5km_Rhine_burned_acc.tif")
cmap = "terrain"
# %%
acc_map = Map(crs=acc.epsg)
acc_map.field(acc, title="Flow Accumulation")
acc_map.colorbar(label="Flow Accumulation")
# %% an elevation raster
# The `DEM5km_Rhine_burned_fill.tif` this cell used to read was never committed to the repository, so it
# reads the elevation raster that is here instead.
dem = Dataset.read_file("examples/data/LisbonElevation.tif")
dem_map = Map(crs=dem.epsg)
dem_map.field(dem, title="Elevation", cmap=cmap)
dem_map.colorbar(label="Elevation")
# %% a boundary color scale
# `color_scale` names the normalization and each scale reads its own knob — `bounds` is the one the
# boundary scale reads. The integer codes the old API took (4 was this scale, 5 the midpoint one below)
# are gone: the names are `linear`, `power`, `lognorm`, `sym_log`, `boundary`, `midpoint`, `equalize`.
bounds = [-600, 0, 100, 300, 500, 700, 900, 1100, 2000, 2500, 3000, 3500]

bounded_map = Map(crs=rhine_acc.epsg)
bounded_map.field(
    rhine_acc,
    title="Flow Accumulation",
    ticks_spacing=500,
    color_scale="boundary",
    bounds=bounds,
)
bounded_map.colorbar(label="Flow Accumulation")
# %% manual normalization — a diverging scale centred on a value you choose
midpoint_map = Map(crs=rhine_acc.epsg)
midpoint_map.field(
    rhine_acc,
    title="Flow Accumulation",
    ticks_spacing=500,
    color_scale="midpoint",
    midpoint=20,
)
midpoint_map.colorbar(label="Flow Accumulation")
# %% gauges over the raster
# The cell-value annotations this example used to switch on (`display_cell_value` / `num_size` /
# `background_color_threshold`) are left out on purpose. The keywords are still accepted, but the labels
# they draw are placed at the array's (row, column) indices while a `Map`'s axes are in the display CRS,
# so on a projected map all of them land next to the origin, far outside the frame, and nothing shows.
points = pd.read_csv("examples/data/points.csv")
point_fc = FeatureCollection(
    points, geometry=gpd.points_from_xy(points["x"], points["y"]), crs=acc.crs
)

point_color = "blue"
point_size = 100
id_color = "yellow"
id_size = 20

gauge_map = Map(crs=acc.epsg)
gauge_map.field(acc, ticks_spacing=500, title="Flow Accumulation")
gauge_map.colorbar(layer=0, label="Flow Accumulation")
markers = gauge_map.points(point_fc, size=point_size)
# `Map.points` colors the markers by the collection's own value column (here `id`). One flat color is a
# property of the artist it returned, not a keyword of the layer.
markers.set_color(point_color)
# What `pid_color` / `pid_size` used to label each point with is map text, one call per gauge.
for gauge in point_fc.itertuples():
    gauge_map.text(
        gauge.x, gauge.y, str(gauge.id), crs=acc.epsg, color=id_color, fontsize=id_size
    )
# %%
# read the polygon , line, and point data
rhine_basin = FeatureCollection.read_file("examples/data/rhine_basin.geojson")
print(rhine_basin.loc[0, "geometry"].geom_type)
rhine_basin.plot()
rhine_river = FeatureCollection.read_file(
    "examples/data/rhine_river_centerline.geojson"
)
print(rhine_river.loc[0, "geometry"].geom_type)
rhine_river.plot()
metrix = FeatureCollection.read_file("examples/data/MetricsHM_Q_Obs.geojson")
print(metrix.loc[0, "geometry"].geom_type)
metrix.plot()

"The columns that we can plot its values in the dataframe"
print(metrix.columns)
"Check the projection"
print(rhine_river.crs)
print(rhine_basin.crs)
print(metrix.crs)


plot_column = "NSE"

metrix[plot_column] = metrix[plot_column].map(float)
"check the histogram for the values that you are going to plot using the bubble plot"
metrix.loc[:, plot_column].plot.hist(bins=7)
"plot "
ax = metrix.plot(column=plot_column)
"plot the values against the index"
ax = metrix.loc[:, plot_column].plot()
ax.set_xlabel("")
ax.set_ylabel("NSE")


filterMin = None
filterMax = None

# filter some values out
if filterMin:
    metrix.loc[metrix[plot_column] < filterMin, plot_column] = filterMin
if filterMax:
    metrix.loc[metrix[plot_column] > filterMax, plot_column] = filterMax


# %% the catchment map: sub-catchments, the river network, and the gauges on top
def catchment_map(title):
    """Draw the grey sub-catchments and the river network, and return the scene to add gauges to."""
    scene = Map(crs=metrix.epsg, figsize=(8, 8))
    catchments = scene.polygons(rhine_basin, line_width=0.2)
    # A uniform fill is not one of the layer's keywords — it is a property of the collection it drew.
    catchments.set_facecolor("grey")
    catchments.set_edgecolor("grey")
    network = scene.sankey(rhine_river)
    network.set_color("C0")
    network.set_linewidth(2.0)
    scene.set_title(title, fontsize=15)
    return scene


# %% apply a size function your self
"apply the function that calculates a size based on the values in the column"


def scale_func(x):
    if x >= 1:
        return 20
    elif 0.6 <= x < 0.8:
        return 14
    elif 0.4 <= x < 0.6:
        return 10
    elif 0.2 <= x < 0.4:
        return 6
    elif 0.1 <= x < 0.2:
        return 4
    else:
        return 1


title = f"Apply function-{plot_column}"
new_col = f"{plot_column}-size"
metrix[new_col] = metrix[plot_column].apply(scale_func)

legend_values = [20, 14, 10, 6, 4, 1]

sized = catchment_map(title)
# `size_column` names the column that drives each marker's size — here the one the function above filled.
sized.points(
    metrix,
    size_column=new_col,
    size_limits=(20, 200),
    size_legend=True,
    size_legend_values=legend_values,
    cmap="Blues",
)
sized.save(f"{saveto}/{title}.tif")
# %% classify the values instead of coloring them on a continuous ramp
# `scheme` is the classification cleopatra cuts the values with (`quantiles`, `fisher_jenks`,
# `equal_interval`, …) and `k` is how many classes it cuts. mapclassify is not a dependency.
title = f"Quantiles-{plot_column}"

classified = catchment_map(title)
classified.points(metrix, size_column=new_col, scheme="quantiles", k=3, cmap="Blues")
classified.save(f"{saveto}/{title}.tif")
# %% both at once: your own size function, and the quantile classes for the color
title = f"Enter scale function-{plot_column}"

combined = catchment_map(title)
combined.points(
    metrix,
    size_column=new_col,
    size_limits=(20, 200),
    size_legend=True,
    scheme="quantiles",
    k=3,
    cmap="Blues",
)
combined.save(f"{saveto}/{title}.tif")
