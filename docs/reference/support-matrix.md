<!-- This page is generated from each tier's Capabilities declaration. Do not edit it by hand; run
`python -m tests.test_support_matrix` to regenerate it, and tests/test_support_matrix.py fails if it
drifts. -->

# Backend support matrix

What each rendering backend can draw and style, read straight from its `Capabilities`
declaration. `✓` means the backend supports it; `—` means it does not.

## Layer kinds

| capability | matplotlib | web | interactive | 3d |
| --- | --- | --- | --- | --- |
| `basemap` | ✓ | ✓ | ✓ | — |
| `borders` | ✓ | — | ✓ | — |
| `choropleth` | ✓ | ✓ | ✓ | ✓ |
| `clusters` | — | ✓ | — | — |
| `coastlines` | ✓ | — | ✓ | ✓ |
| `contours` | ✓ | ✓ | ✓ | — |
| `custom:holoviews` | — | — | ✓ | — |
| `custom:maplibre` | — | ✓ | — | — |
| `custom:matplotlib` | ✓ | — | — | — |
| `custom:pyvista` | — | — | — | ✓ |
| `extrusion` | — | ✓ | — | ✓ |
| `filled_contours` | ✓ | ✓ | ✓ | — |
| `flow` | ✓ | — | ✓ | — |
| `graticule` | ✓ | ✓ | ✓ | — |
| `heatmap` | ✓ | ✓ | ✓ | — |
| `isosurface` | — | — | — | ✓ |
| `labels` | ✓ | ✓ | ✓ | — |
| `lakes` | ✓ | — | ✓ | — |
| `land` | ✓ | — | ✓ | — |
| `lines` | ✓ | ✓ | ✓ | ✓ |
| `mesh` | ✓ | — | ✓ | — |
| `model` | — | ✓ | — | — |
| `nightshade` | ✓ | — | — | — |
| `ocean` | ✓ | — | ✓ | — |
| `point_cloud` | — | ✓ | — | ✓ |
| `points` | ✓ | ✓ | ✓ | ✓ |
| `polygons` | ✓ | ✓ | ✓ | ✓ |
| `raster` | ✓ | ✓ | ✓ | ✓ |
| `reference_lines` | — | — | — | ✓ |
| `rgb` | ✓ | ✓ | ✓ | — |
| `rivers` | ✓ | — | ✓ | — |
| `streamlines` | ✓ | — | ✓ | ✓ |
| `terrain` | — | ✓ | — | ✓ |
| `text` | ✓ | ✓ | ✓ | ✓ |
| `tissot` | ✓ | — | — | — |
| `unstructured` | ✓ | — | ✓ | — |
| `vector_tiles` | — | ✓ | — | — |
| `vectors` | ✓ | — | ✓ | ✓ |
| `volume` | — | — | — | ✓ |

## Encoding channels

| capability | matplotlib | web | interactive | 3d |
| --- | --- | --- | --- | --- |
| `color` | — | ✓ | ✓ | ✓ |
| `height` | — | ✓ | — | ✓ |
| `opacity` | ✓ | ✓ | ✓ | ✓ |
| `rotation` | — | — | ✓ | — |
| `size` | ✓ | ✓ | ✓ | ✓ |
| `text` | — | ✓ | — | — |
| `tooltip` | — | ✓ | ✓ | — |
| `width` | — | ✓ | ✓ | — |

## Data-driven channels

| capability | matplotlib | web | interactive | 3d |
| --- | --- | --- | --- | --- |
| `color` | — | ✓ | ✓ | ✓ |
| `height` | — | ✓ | — | ✓ |
| `text` | — | ✓ | — | — |

## Classification schemes

| capability | matplotlib | web | interactive | 3d |
| --- | --- | --- | --- | --- |
| `categorical` | ✓ | ✓ | ✓ | ✓ |
| `equal_interval` | ✓ | ✓ | ✓ | ✓ |
| `fisher_jenks` | ✓ | ✓ | ✓ | ✓ |
| `natural_breaks` | ✓ | ✓ | ✓ | ✓ |
| `percentiles` | ✓ | ✓ | ✓ | ✓ |
| `quantiles` | ✓ | ✓ | ✓ | ✓ |
| `std_mean` | ✓ | ✓ | ✓ | ✓ |

## Features

| capability | matplotlib | web | interactive | 3d |
| --- | --- | --- | --- | --- |
| `animation` | ✓ | ✓ | ✓ | ✓ |
| `attribution` | — | ✓ | — | — |
| `coastline_overlay` | ✓ | — | ✓ | — |
| `colorbar` | ✓ | ✓ | ✓ | ✓ |
| `display_crs` | ✓ | ✓ | ✓ | ✓ |
| `domain` | ✓ | — | — | — |
| `export_html` | — | ✓ | ✓ | ✓ |
| `export_image` | ✓ | ✓ | ✓ | ✓ |
| `export_vector` | ✓ | — | — | — |
| `fullscreen` | — | ✓ | — | — |
| `geocoder` | — | ✓ | — | — |
| `geolocate` | — | ✓ | — | — |
| `globe_control` | — | ✓ | — | — |
| `layer_switcher` | — | ✓ | ✓ | — |
| `legend` | ✓ | ✓ | ✓ | ✓ |
| `measure` | — | ✓ | — | — |
| `navigation` | — | ✓ | — | — |
| `raster_renderer` | ✓ | — | ✓ | — |
| `scale_bar` | — | ✓ | — | — |
| `terrain_control` | — | ✓ | — | — |
| `time_slider` | — | ✓ | ✓ | — |

## What a backend omits, and why

Some entries here (for example a control a tier has no surface for) are not rows in the tables above: they are
omissions with no positive counterpart on any backend, so they appear only in this section.

### matplotlib

- **`attribution`** — a credit is text placed on the figure — `text` or `stamp` — rather than a control the tier
  draws
- **`export_html`** — the figure is written as an image, not as a page; an HTML export is every other tier's —
  interactive, web and 3-D all declare `export_html` as a feature, and this is the only tier that does not
- **`fullscreen`** — a saved image has no screen to fill; its size is the figure's
- **`height`** — an axes is flat; a layer raised by a column is the 3-D tier's extrusion or the web tier's
- **`label_collision`** — matplotlib draws every label it is given, wherever it lands; there is no collision
  index to ask which ones overlap, so `labels` takes no allow_overlap= and a crowded column is thinned by
  filtering the features before drawing them. MapLibre's symbol layer is where that decision is made for the
  tier that has one: the web tier's `labels` takes an `allow_overlap=` of its own (False by default) and hands
  it straight to `text-allow-overlap`
- **`layer_switcher`** — the tier draws no control onto the figure for a reader to toggle: visibility is set
  from code, with `set_visible` before or between draws, and a live canvas can do that from a `Scene.on_pick`
  callback — but the switch itself is a widget the interactive and web tiers' engines provide and this one does
  not. The 3-D tier has none either, and gives its own reason: its layers are switched by id through the scene's
  API rather than from a panel
- **`measure`** — there is no pointer to measure with; a distance is drawn as a layer of its own
- **`navigation`** — there is no viewport to pan: the extent is set by `set_bounds` before drawing
- **`time_slider`** — a sequence over time is written out as an animation here rather than scrubbed, which is
  what `animate` is
- **`tooltip`** — nothing follows the pointer: a saved figure is a picture, and on a live canvas the gesture
  this tier delivers is a click, not a hover — `Scene.on_pick` reports the layer a click landed on and its data
  coordinates, which a caller prints or annotates themselves. A value is otherwise read from the colorbar or
  printed into the cell

### web

- **`coastline_overlay`** — the tier draws coastlines and borders as Natural-Earth overlay layers through
  coastlines() and borders(); what is absent is the quickmap(coastlines=True) overlay kwarg, whose cross-tier
  wiring is tracked in #398
- **`domain`** — a web map pans and zooms, so it is framed by a centre and a zoom rather than by a region
- **`export_vector`** — a page is a raster canvas; a PDF of it would be a screenshot in a wrapper
- **`mesh`** — a raster is drawn as an image rather than as cells, which is what a tile pipeline expects
- **`north_arrow`** — the navigation control's compass already shows the bearing, and turns the map back to
  north when it is clicked
- **`raster_renderer`** — a raster is drawn as an image and contours are their own builder, so there is no
  choice of renderer to make on top of one
- **`streamlines`** — there is no streamline primitive to trace a field with in a browser
- **`unstructured`** — a UGRID mesh has no MapLibre layer type; it is drawn on the static tier
- **`vectors`** — MapLibre has no arrow glyph; a u/v field is drawn on the static or interactive tier

### interactive

- **`attribution`** — a tile source's credit is drawn by the tile layer itself, not by a separate control
- **`domain`** — no domain= on the constructor: a region is asked for with set_bounds(), which frames the map on
  a rectangle or on its own data, and the viewer pans on from there
- **`export_vector`** — the figure is a Bokeh canvas; a vector export is the static tier's
- **`fullscreen`** — a notebook cell or a served page is sized by its host, not by a control in the figure
- **`height`** — HoloViews has no z-height option: an extrusion is a 3-D or web layer (measured in #298)
- **`measure`** — there is no measurement tool in Bokeh's toolbar; a distance is drawn as a layer
- **`navigation`** — Bokeh's own toolbar pans and zooms, so a second set of buttons would duplicate it
- **`north_arrow`** — a Bokeh frame is drawn north-up, so an arrow would say only what the axes say
- **`scale_bar`** — Bokeh has no scale-bar tool; the axes carry the coordinates instead

### 3d

- **`attribution`** — a render window has no credit line; a caller writes one beside the image it saves
- **`basemap`** — there are no map tiles to drape under a scene drawn in three dimensions
- **`domain`** — a scene is framed by its camera, not by an extent, so there is no region to set
- **`fullscreen`** — the render window is resized by the window manager, not by a control in the scene
- **`layer_switcher`** — layers are switched by id through the scene's own API rather than from a panel
- **`measure`** — PyVista's own measurement widget is the tier's answer, and is not part of a figure
- **`navigation`** — the render window pans, zooms and rotates with the mouse rather than with buttons
- **`north_arrow`** — the scene can be looked at from any direction, so there is no fixed north on screen
- **`raster_renderer`** — a raster becomes a surface, a volume or a globe here — which one is the builder that
  was called, not a rendering choice on top of one
- **`scale_bar`** — a screen distance means nothing when the camera decides the scale of what is in front
- **`time_slider`** — frames over time are rendered by record() rather than scrubbed in the window
