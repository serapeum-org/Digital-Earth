# Plot raster/array

`digitalearth` plots rasters and arrays through the `StaticGlyph` class. It
accepts either a [pyramids](https://github.com/Serapieum-of-alex/pyramids)
`Dataset` or a raw NumPy array.

## Read the raster

```python
from pyramids.dataset import Dataset

dataset = Dataset.read_file("examples/data/acc4000.tif")
```

## Default plot

With all the default parameters, `StaticGlyph.plot` plots the dataset directly
and returns the matplotlib `(fig, ax)`:

```python
from digitalearth.static import StaticGlyph

fig, ax = StaticGlyph.plot(dataset, title="Flow Accumulation")
```

## Plotting a NumPy array

When the first argument is a `numpy.ndarray`, you must pass `no_data_value`:

```python
import numpy as np

fig, ax = StaticGlyph.plot(arr, no_data_value=np.nan)
```

## Figure and color-bar options

Styling is forwarded to the underlying `cleopatra.array.Array.plot` through
`**kwargs`. Common options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `figsize` | tuple | `(8, 8)` | figure size |
| `title` | str | `"Total Discharge"` | plot title |
| `title_size` | int | `15` | title font size |
| `cbar_length` | float | `0.75` | ratio controlling color-bar height |
| `orientation` | str | `"vertical"` | color-bar orientation |
| `cbar_label_size` | int | `12` | color-bar label size |
| `cbar_label` | str | `"Color bar label"` | color-bar label |
| `rotation` | number | `-90` | color-bar label rotation |
| `ticks_spacing` | int | `5` | spacing of color-bar ticks |
| `cmap` | str | `"coolwarm_r"` | matplotlib colormap |

```python
fig, ax = StaticGlyph.plot(
    dataset,
    figsize=(8, 8),
    title="Flow Accumulation map",
    title_size=15,
    cmap="terrain",
    ticks_spacing=10,
)
```

## Color scales

`color_scale` selects the normalization applied to the data:

1. normal (linear) scale
2. power scale (`gamma`)
3. `SymLogNorm` scale (`linthresh`, `linscale`)
4. `PowerNorm` scale
5. `BoundaryNorm` scale (`midpoint`)

```python
fig, ax = StaticGlyph.plot(dataset, color_scale=2, gamma=0.5, cmap="terrain")
```

## Cell-value annotations

Display each cell's value as text on top of the map:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `display_cell_value` | bool | `False` | annotate cells with their value |
| `num_size` | int | `8` | font size of the cell numbers |
| `background_color_threshold` | float | `None` | threshold deciding black vs. white text; `max/2` if `None` |

```python
fig, ax = StaticGlyph.plot(
    dataset,
    display_cell_value=True,
    num_size=8,
    background_color_threshold=None,
    ticks_spacing=500,
)
```

## Plotting points

Overlay vector points on the raster. The `GeoDataFrame` must carry an `id`
column and share the raster's coordinate system:

```python
import geopandas as gpd

points = gpd.read_file("tests/data/points.geojson")

fig, ax = StaticGlyph.plot(
    dataset,
    points=points,
    point_color="blue",
    point_size=100,
    pid_color="green",
    pid_size=20,
    display_cell_value=True,
    ticks_spacing=500,
)
```

See the [API Reference](reference/static.md) for the full signature.
