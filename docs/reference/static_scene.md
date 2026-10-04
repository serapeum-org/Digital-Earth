# Scenes & multi-panel figures (matplotlib)

`Scene` is the static tier's base host: one matplotlib figure, one axes, and the list of layers drawn onto it.
[`Map`](static_map.md) is a `Scene` with a display CRS on top, so everything on this page — the layer list,
the colour keys, the title, `save`/`show`/`close` and `with` — is also how a `Map` behaves.

```python
import matplotlib.pyplot as plt
from digitalearth.static import Scene

fig, ax = plt.subplots()
with Scene(ax=ax, fig=fig) as scene:
    scene.set_title("Composed on an axes of my own")
    scene.save("scene.png")
```

## One scene per axes

A scene may be handed an axes, to place it inside a figure you are laying out yourself, but **two scenes must
not share one axes**: each keeps its own record of what it has drawn, and the second one's first render clears
what the first drew while the first goes on describing it. Give every scene its own axes — which is what
`grid` does.

## Several panels in one figure

`grid(nrows, ncols, crs=...)` builds the matplotlib subplot grid and binds a `Map` to each cell, returning
`(fig, maps)` in row-major order. The panels share one figure, so one `savefig` writes them all and
`shared_colorbar(fig, mappable, maps)` adds a single bar spanning the panels you name.

```python
from digitalearth.static import grid, shared_colorbar

fig, maps = grid(1, 2, crs=4326)
left = maps[0].field(rain_2023)
maps[1].field(rain_2024)
shared_colorbar(fig, left, maps, label="mm")
fig.savefig("rain.png")
```

Because the panels share that figure, closing one panel closes the figure for all of them — do not wrap a
single `grid` panel in `with`; save the figure and close it once.

## Skipping or raising on an empty layer

`strict=False` (the default) skips a layer that has nothing to draw — data entirely outside the view, or a
band with no finite values — and logs a warning naming it, so one bad frame does not abort a batch.
`strict=True` raises instead, for a pipeline that must not publish a figure with a layer missing.

::: digitalearth.static.scene.Scene
    options:
      docstring_options:
        warn_missing_types: false
      members:
        - layer_ids
        - figure_spec
        - get_layer
        - add_layer
        - remove_layer
        - set_visible
        - move_layer
        - replace_layer
        - viewport
        - colorbar
        - colorbars
        - legend
        - stamp
        - show
        - close

<!-- set_title and save are rendered as their own blocks: their `**kwargs` carry no annotation, and only an
     object's own block reliably applies `warn_missing_types: false`, so listing them as class members fails
     `mkdocs build --strict`. Annotating `**kwargs: Any` in static/scene.py would let them rejoin the list. -->

::: digitalearth.static.scene.Scene.set_title
    options:
      heading_level: 3
      docstring_options:
        warn_missing_types: false

::: digitalearth.static.scene.Scene.save
    options:
      heading_level: 3
      docstring_options:
        warn_missing_types: false

::: digitalearth.static.figure.grid
    options:
      docstring_options:
        warn_missing_types: false

::: digitalearth.static.figure.shared_colorbar
    options:
      docstring_options:
        warn_missing_types: false
