# Interactive maps (HoloViz)

`InteractiveMap` is the pan/zoom/hover backend: a Bokeh canvas with a display CRS and a list of layers it can
re-order, hide, restyle and remove after they are drawn. It is the interactive sibling of the matplotlib
[`Map`](static_map.md) — the same method names wherever the two can draw the same thing — and needs the
`interactive` extra (`pip install 'digitalearth[interactive]'`) only when a builder or render method is
actually called; importing the module works without it.

```python
from digitalearth.interactive import InteractiveMap

m = InteractiveMap(crs=3857, tiles="CartoLight")
m.field(dataset, cmap="terrain")
m.points(stations, name="obs")
m.coastlines()
m.save("map.html")
```

Every input is reprojected to the display CRS through **pyramids** before an element is built, so the Bokeh
layers are already in the display CRS (EPSG:3857 by default — the only CRS Bokeh tile basemaps render). The
rendering itself is HoloViews/GeoViews/Datashader; styling keywords (`cmap`, `clim`, `alpha`, …) are handed
straight to the engine, which is why they are documented there rather than here.

## What this tier is for

Exploration: a dataset you pan into, a cell you click to pull a time series, an area you draw to crop, a
million points shaded server-side so they cost the same as a thousand. Anything that must come back as a file —
a vector export, a print-quality figure — belongs to the [matplotlib tier](static_map.md); this one saves
interactive HTML and headless PNG.

## The layer list

A builder records a layer addressed by the `name=` you gave it (or a generated id), and returns the map so the
calls chain. `layer_ids` lists them in draw order, and `move_layer`, `remove_layer`, `replace_layer` and
`set_visible` work on that list afterwards, so a figure can be rearranged without being rebuilt. Layers sit in
bands — `underlay`, `reference`, `data`, `overlay` — so a basemap stays under the data and a graticule stays
over it however the calls were ordered. `figure_spec` is the whole map as a value, the description the other
tiers read back.

## The layer control

`layer_control()` builds a Panel widget — one visibility toggle per layer, an opacity slider and a basemap
switch — and returns the map; the widget itself is read back through `layer_control_panel`. The control is
keyed by layer **id**, not by the position a layer held when it was built, so toggling and removal keep
working after the overlay is re-stacked. `reorder=True` adds a `▲`/`▼` button per layer that moves it within
its band live. The basemap switch is Web-Mercator-only (Bokeh renders tiles in EPSG:3857); named outright on a
non-Mercator map it is refused, left at the default it is quietly dropped.

```python
m = InteractiveMap().field(dataset).points(stations)
m.layer_control(controls=("visibility", "opacity"), reorder=True)
m.layer_control_panel        # the Panel object to display
```

Every widget event — the dashboard's and the layer control's alike — goes through one reconcile that
re-styles only the layers a value reaches and reuses every other element untouched, so a slider tick does not
rebuild the map.

## Big data

`rasterize` and `datashade` aggregate millions of rows into a server-rendered density image that
re-aggregates on every pan/zoom. `rasterize` keeps a Bokeh colorbar and client-side colour-mapping, so it can
pin a **frozen colour scale** with `clim`/`cnorm` — a rasterized layer then shares one range with a
neighbouring `field`, and `cnorm="eq_hist"` reaches Bokeh's `EqHistColorMapper` (which takes explicit limits,
where Datashader refuses a span under eq-hist).

```python
m.rasterize(gps_fixes, aggregator="mean", column="speed", clim=(0, 120), cnorm="eq_hist")
```

## Clicking, drawing and brushing

A live kernel or server is what makes these round-trip to Python; in a saved HTML they degrade to plain hover.
`on_tap`/`tap_profile` pull a value (or a `DatasetCollection` time series) at the clicked cell. `draw()` adds a
box/polygon/point/freehand tool and `drawn_geometry`/`aoi_crop` read the sketch back **in pyramids' CRS** to
drive a crop. `cross_filter` links a selection across panels and hands the selected rows back.

## Live data

`live()` adds a streaming layer backed by a HoloViews `Pipe` (each `push` **replaces** the data) or a
`Buffer` (each `push` **appends**, keeping the last `length` rows). Only the changed glyphs redraw — the IN-2
reconcile applied to a feed — so a moving vehicle or a sensor stream does not rebuild the map. Reach the
stream with `live_stream(id)` and feed it with `push(id, data)`; the data is in the display CRS, and the
repaint reaches the browser under a live kernel/server.

```python
m = InteractiveMap().live(kind="points", name="cars")
m.push("cars", positions_df)        # positions_df: x/y in the display CRS
```

## Themes

`theme()` sets one Bokeh theme for every render of the map — a built-in name (`"dark_minimal"`, `"night_sky"`,
…) or a `bokeh.themes.Theme` — instead of threading `bgcolor` through `**opts` at every call.

```python
m = InteractiveMap().field(dataset).theme("dark_minimal")
```

## Dashboards

`dashboard()` wraps the map and reactive widgets into a Panel app; `serve()` marks it servable and `save_app()`
bakes the widget states into a standalone HTML file. A pyramids-backed app must be **served**, not converted to
WASM — GDAL does not run in the browser. `share()` syncs the view into the URL under a running server.

::: digitalearth.interactive.map.InteractiveMap
    options:
      inherited_members: true
      members:
        - field
        - quadmesh
        - rgb
        - contours
        - spaghetti
        - large_image
        - points
        - lines
        - polygons
        - choropleth
        - hexbin
        - kde
        - flow
        - vectorfield
        - barbs
        - streamlines
        - trimesh
        - labels
        - text
        - timecube
        - rasterize
        - datashade
        - trajectory
        - live
        - live_stream
        - push
        - tiles
        - coastlines
        - graticule
        - set_bounds
        - projection
        - theme
        - hover
        - on_tap
        - tap_profile
        - draw
        - drawn_geometry
        - aoi_crop
        - cross_filter
        - layer_control
        - layer_control_panel
        - attribute_table
        - share
        - dashboard
        - serve
        - save_app
        - play
        - save_animation
        - colorbar
        - legend
        - layer_ids
        - get_layer
        - move_layer
        - remove_layer
        - replace_layer
        - set_visible
        - figure_spec
        - viewport
        - add_layer
        - render
        - save
        - show
        - close
