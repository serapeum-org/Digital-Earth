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

!!! warning "Does not work on a `Map` — upstream bug"

    `Map.field` accepts `display_cell_value`, `num_size` and `background_color_threshold` and forwards them to
    cleopatra, but **the labels do not appear.** cleopatra's `ArrayGlyph` places each one with
    `ax.text(col, row, ...)` — array index coordinates — while a `Map` draws the raster at its projected extent,
    so every label lands outside the axes. Measured on a projected raster: 89 text artists, **0 of 89** inside
    the frame, sitting at `(5, 1)`, `(6, 1)`, … against an xlim spanning millions of metres.

    There is a second, less visible consequence: `savefig(bbox_inches="tight")` expands the canvas to enclose
    the off-frame artists, so a saved figure is wrong even though the displayed one merely looks unannotated.

    Tracked upstream as [cleopatra#378](https://github.com/serapeum-org/cleopatra/issues/378). The keywords are
    documented here because they are accepted, not because they do anything — do not reach for them until that
    issue lands.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `display_cell_value` | bool | `False` | accepted; intended to annotate cells with their value |
| `num_size` | int | `8` | font size of the cell numbers |
| `background_color_threshold` | float | `None` | threshold deciding black vs. white text; `max/2` if `None` |

## Plotting points

Overlay vector points on the raster. `Map.points` reads a pyramids `FeatureCollection` and reprojects it to
the display CRS, so it need not already share the raster's coordinate system:

```python
from pyramids.feature import FeatureCollection

points = FeatureCollection.read_file("tests/data/points.geojson")

m = Map(crs=dataset.epsg)
m.field(dataset, ticks_spacing=500)
m.points(points, size=100)
```

`size=` is the marker's visual size, spelled the same way on every backend. To vary it per point, name the
column that drives it with `size_column=` and pair that with `size_legend=True`.

Work through the [Maps & raster fields](examples/02_maps_and_fields.ipynb) example for the rest of the layer
methods.
