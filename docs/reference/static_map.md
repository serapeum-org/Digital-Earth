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

`colorbar()` with no argument keys **the most recent layer that is coloured by a value** — a coastline, a
basemap or a text label added after the data cannot take the key. Name a layer (`colorbar("obs")`) to key that
one instead, and `colorbars()` keys every layer that has a key to show. The request is recorded on the layer,
so the bar moves, hides and disappears with it.

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

## Hatching a mask

`contours(..., filled=True, hatches=[...])` gives each band between `levels` a hatch pattern, and `fill=False`
leaves the bands uncoloured so only the hatching draws — the usual way to mark a significance or uncertainty
mask over another field without spending its colours. `legend()` on such a layer keys the bands it marks by
their pattern; `hatch_color=` colours the strokes on the map and in the key alike.

```python
m.field(trend, cmap="RdBu_r")
m.contours(p_value, levels=[0, 0.05, 1], filled=True, hatches=["///", ""], fill=False, name="sig")
m.legend("sig", labels=["p < 0.05"])
```

## Overlays computed from the globe itself

`nightshade(when)` shades the night side of the day/night terminator at one instant, and `tissot()` draws
Tissot's indicatrices — circles of one ground radius, shown as the projection distorts them. cleopatra computes
both in lon/lat (`cleopatra.basemap.solar`) and they are projected into the display CRS through pyramids, so a
Web Mercator map shows the polar night reaching the pole and the indicatrices swelling toward it. On a globe
the night side is filled only where it faces the viewer, and a circle on the far side is left out.

```python
from datetime import datetime, timezone

m = Map(crs=3857)
m.nightshade(datetime(2026, 6, 21, 12, tzinfo=timezone.utc), alpha=0.3)
m.nightshade("2026-06-21T12:00:00", refraction=-6.0, alpha=0.15)  # civil twilight
m.tissot(edgecolor="crimson")                                      # a world grid of 500 km circles
```

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
        - nightshade
        - tissot
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
