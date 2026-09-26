# Web maps

`WebMap` renders pyramids rasters and vectors as MapLibre GL JS / deck.gl layers and exports the result as one
self-contained HTML page.

```python
from digitalearth.web import WebMap

(
    WebMap()
    .basemap()
    .choropleth(gdf, column="pop", name="Population")
    .legend(title="People per km²")
    .layer_control()
    .set_title("Population, 2024", subtitle="Source: CBS")
    .save("map.html")
)
```

The map frames itself on the data you gave it, so there is no `center`/`zoom` to work out by hand — pass them
only when you want a view of your own.

## What this tier is for

Thematic maps, 3-D layers, contours, and sharing the result as one file. It is deliberately **not** a peer of
the static tier for scientific field rendering: vector/flow fields, unstructured meshes and KDE have no native
MapLibre primitive and live in `digitalearth.static` and `digitalearth.interactive`.

## Naming layers, and why the name matters

py-maplibregl's layer switcher has no separate label: its JavaScript captions each row with the layer's **id**
and reads that same string back to toggle it. So a `name=` you pass becomes the layer's id, and that is what a
viewer reads in the switcher:

```python
WebMap().basemap().choropleth(gdf, column="pop", name="Population").layer_control()
# the switcher shows: Population
```

Without a name a layer gets a generated id (`fill-4`), which is what the switcher would then show. A repeated
name is suffixed (`Population-2`) because two layers cannot share a MapLibre id. `WebMap.layer_ids` lists them,
and `remove_layer(id)` drops one.

The big-data path is the exception: past `big_data_threshold` features, `points`/`polygons` render a deck.gl
overlay, which is not a MapLibre style layer and cannot be reached by the switcher — so `name=` and `visible=`
are refused there rather than silently dropped.

## Drawing order

Layers sit in three bands, and a later call cannot slip under an earlier band:

1. **basemaps** — `basemap()`, `tiles()` (`add_underlay`)
2. **reference** — `graticule()` (`add_reference`)
3. **data** — everything else (`add_layer`)

Within a band, layers keep the order you added them in. That is why `.basemap().graticule()` works: a
reference grid belongs over the ground and under the data, which neither end of a flat list can express.

A builder that draws several MapLibre layers — `cluster` (bubbles, counts, loose points) and `graticule`
(lines, degree labels) — appears once in `layer_ids`, under its main layer. `remove_layer` takes the whole
group; the layer switcher toggles only that main layer, because that is what py-maplibregl's control does.

## Sharing a page safely

- **A keyed basemap puts its credential in the saved HTML.** See [Keyed basemaps](basemaps.md) — treat such a
  file as a secret.
- **Values from your data are escaped** before they reach the legend or the title, so a hostile attribute in a
  downloaded shapefile cannot inject markup into the page you share.

## Time series

`timeslider(collection)` gives a notebook an `ipywidgets` slider. A **saved** page has no kernel, so it gets a
step picker instead — the steps are labelled with the times you passed, because those labels become the layer
ids. `save("out.gif")` writes the steps as an animation, which needs a headless browser (Playwright or
Selenium); that is deliberately not part of `digitalearth[web]`. The method behind that suffix is
`save_animation(path, fps=...)`. It was once spelled `animate`, `to_gif` and `save_gif` as well; those three
are gone.

## Names that were renamed

The web tier's marker size is `size=` (it was `radius=` on `points`/`deck_scatter` and `point_size=` on
`point_cloud`), its font size is `text_size=` (it was `size=` on `labels`/`text`), and its frame rate is
`fps=` (it was `duration=`, which held a frame for that many *seconds* — the reciprocal, so an old call needs
`fps = 1 / duration`). The methods were renamed too: `field` for `add_raster`, `set_bounds` for `fit_bounds`,
`set_title` for `title`, `save_animation` for `animate`, `terrain_tiles` for `terrain` and `projection` for
`globe` — where `globe(True)` is now `projection("globe")`.

**None of the old spellings still works.** Nothing in this package is released, so each was deleted rather
than kept as a second name: a call that uses one raises `TypeError` for an unexpected keyword, or
`AttributeError` for a method that is not there.

::: digitalearth.web.map.WebMap
    options:
      inherited_members: true
      members:
        - set_bounds
        - layer_ids
        - layer_control
        - remove_layer
        - legend
        - set_title
        - text
        - labels
        - graticule
        - contours
        - rgb_composite
        - save_animation
