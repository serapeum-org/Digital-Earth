# Plot raster/array

`digitalearth` plots rasters through the `Map` scene — one matplotlib figure that several layers render
onto — or in a single call through `quickmap`. Both read a
[pyramids](https://github.com/serapeum-org/pyramids) `Dataset`.

## Read the raster

```python
from pyramids.dataset import Dataset

dataset = Dataset.read_file("examples/data/acc4000.tif")
```

## Default plot

`quickmap` returns a finished `Map`: the raster field, a color bar, and a frame around the data.

```python
from digitalearth import quickmap

m = quickmap(dataset, crs=dataset.epsg)
m.set_title("Flow Accumulation")
```

Compose the same figure yourself when you want to decide what goes on it:

```python
from digitalearth import Map

m = Map(crs=dataset.epsg)
m.field(dataset)
m.colorbar(layer=0)
m.set_title("Flow Accumulation")
```

`Map(crs=...)` is the display CRS: every layer is reprojected to it before it is drawn, and your own
`Dataset` is left as it was.

## Figure and color-bar options

Styling is forwarded to the underlying `cleopatra.glyphs.gridded.array_glyph.ArrayGlyph` through
`**kwargs`. Common options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `figsize` | tuple | `(8, 8)` | figure size |
| `title` | str | `None` | plot title |
| `title_size` | int | `15` | title font size |
| `cbar_length` | float | `0.75` | ratio controlling color-bar height |
| `cbar_orientation` | str | `"vertical"` | color-bar orientation |
| `cbar_label_size` | int | `12` | color-bar label size |
| `cbar_label` | str | `None` | color-bar label |
| `cbar_label_rotation` | number | `None` | color-bar label rotation |
| `ticks_spacing` | int | `5` | spacing of color-bar ticks |
| `cmap` | str | `"coolwarm_r"` | matplotlib colormap |

```python
m = Map(crs=dataset.epsg)
m.field(
    dataset,
    figsize=(8, 8),
    title="Flow Accumulation map",
    title_size=15,
    cmap="terrain",
    ticks_spacing=10,
)
```

## Color scales

`color_scale` names the normalization applied to the data, and each scale reads its own knob:

| `color_scale` | Knob | Normalization |
|---------------|------|---------------|
| `"linear"` | — | the default, straight through |
| `"power"` | `gamma` | `PowerNorm` |
| `"lognorm"` | — | `LogNorm` |
| `"sym_log"` | `line_threshold`, `line_scale` | `SymLogNorm` |
| `"boundary"` | `bounds` | `BoundaryNorm` |
| `"midpoint"` | `midpoint` | a norm centred on `midpoint` |
| `"equalize"` | `samples` | histogram equalization |

```python
m = Map(crs=dataset.epsg)
m.field(dataset, color_scale="power", gamma=0.5, cmap="terrain")
```

## Cell-value annotations

Display each cell's value as text on top of the map:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `display_cell_value` | bool | `False` | annotate cells with their value |
| `num_size` | int | `8` | font size of the cell numbers |
| `background_color_threshold` | float | `None` | threshold deciding black vs. white text; `max/2` if `None` |

```python
m = Map(crs=dataset.epsg)
m.field(
    dataset,
    display_cell_value=True,
    num_size=8,
    background_color_threshold=None,
    ticks_spacing=500,
)
```

## Plotting points

Overlay vector points on the raster. `Map.points` reads a pyramids `FeatureCollection` and reprojects it to
the display CRS, so it need not already share the raster's coordinate system:

```python
from pyramids.feature import FeatureCollection

points = FeatureCollection.read_file("tests/data/points.geojson")

m = Map(crs=dataset.epsg)
m.field(dataset, display_cell_value=True, ticks_spacing=500)
m.points(points, size=100)
```

`size=` is the marker's visual size, spelled the same way on every backend. To vary it per point, name the
column that drives it with `size_column=` and pair that with `size_legend=True`.

Work through the [Maps & raster fields](examples/02_maps_and_fields.ipynb) example for the rest of the layer
methods.
