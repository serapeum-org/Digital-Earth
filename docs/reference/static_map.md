# Maps (matplotlib)

`Map` is the default backend's host: a matplotlib figure with a display CRS, one axes, and a list of layers it
can re-order, hide and replace after they are drawn.

```python
from digitalearth import Map

m = Map(crs=3857)
m.field(dataset, cmap="terrain")
m.points(stations, name="obs")
m.coastlines()
m.colorbar()
m.save("map.png")
```

Every input is reprojected to the display CRS through **pyramids** and drawn on a plain axes — there is no
Cartopy, because the projection is applied to the *data* rather than to the axes. The rendering itself is
**cleopatra**'s: each builder assembles a glyph and hands the styling keywords straight to it, which is why
`cmap`, `levels`, `figsize` and the `cbar_*` family are documented there rather than here.

## What this tier is for

Scientific field rendering and anything that has to come back as a file: rasters, contours, unstructured
meshes, vector and flow fields, KDE, ensemble series and animations. It is the one tier installed without an
extra, which is all that distinguishes it: `digitalearth.interactive`, `digitalearth.web` and
`digitalearth.three_d` are peers, answering to the same method names wherever they can draw the same thing,
and each owns what the others have no primitive for.

## The layer list

A builder returns the artist it drew and records a layer beside it, addressed by the `name=` you gave it (or a
generated id). `layer_ids` lists them in draw order, and `move_layer`, `remove_layer`, `replace_layer` and
`set_visible` work on that list afterwards, so a figure can be rearranged without being rebuilt. Layers sit in
bands — `underlay`, `reference`, `data`, `overlay` — and a later call cannot slip under an earlier band, so a
basemap stays under the data and a label stays over it however the calls were ordered.

`figure_spec` is the whole figure as a value: its panels, their layers, the view each is drawn in and the
furniture anchored to them. That description is what the other tiers read back, and it is the reason a layer
records its recipe (`via`) rather than the method that drew it.

## What `field` accepts

A pyramids `Dataset`, a path or URL to a raster — and a **bare 2-D NumPy array**, for the "just show me this
grid" case. Such a layer is not georeferenced: it is placed at its own indices, nothing is reprojected, and no
georeferenced layer may share the figure with it (a basemap, a coastline and a reprojected raster are all in
the display CRS, so mixing them is refused rather than drawn with one of them as an invisible speck). Nodata in
a bare array is a masked array, since there is no sidecar to carry a `no_data_value`. Anything else is refused
by name, saying what the call takes.

::: digitalearth.static.map.Map
    options:
      inherited_members: true
      members:
        - field
        - contours
        - pcolormesh
        - block
        - rgb_composite
        - points
        - grid_points
        - grid_cells
        - polygons
        - choropleth
        - quiver
        - streamplot
        - labels
        - set_bounds
        - set_domain
        - graticule
        - basemap
        - coastlines
        - colorbar
        - legend
        - layer_ids
        - move_layer
        - remove_layer
        - replace_layer
        - set_visible
        - animate
        - figure_spec
        - viewport
        - save
        - show
        - close
