# digitalearth as a pyramids plot backend

pyramids renders through cleopatra (matplotlib) by default. Ask any pyramids object for another
backend and it looks for a registered renderer:

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("examples/data/acc4000.tif")

ds.plot()                        # matplotlib, through cleopatra — pyramids' own path
ds.plot(backend="interactive")   # a digitalearth InteractiveMap
ds.plot(backend="3d")            # a digitalearth Scene3D
ds.plot(backend="web")           # a digitalearth WebMap
```

digitalearth registers itself for those three names in the `pyramids.plot_backends` entry-point
group, so pyramids *discovers* it rather than importing it by name. The dependency stays one-way —
digitalearth depends on pyramids-core; pyramids never names digitalearth in its own code.

```toml
# digitalearth's pyproject.toml
[project.entry-points."pyramids.plot_backends"]
interactive = "digitalearth.pyramids_backend:interactive"
"3d" = "digitalearth.pyramids_backend:three_d"
web = "digitalearth.pyramids_backend:web"
```

!!! note "You do not install anything extra for this"
    A registered entry point simply wins over the fallback pyramids keeps for a plain digitalearth
    install. The difference is what happens to the *arguments* — see below.

## Why an adapter, and not just `quickmap`

pyramids forwards the **data-selection** arguments of the call it was given, in **its own**
vocabulary. Three of those names mean something different here, so they are translated on the way
in. Everything else passes through untouched.

| pyramids forwards | digitalearth takes | Why |
|---|---|---|
| `band=0` (0-based) | `band=1` (1-based) | 0-based upstream, 1-based here;<br>unshifted it was refused. |
| `variable_name=` | `variable=` | The `UgridDataset` facade's spelling.<br>Inert today — see below. |
| `interval=200` (ms/frame) | `fps=5.0` | The facades describe an interval; the<br>tier verbs take a rate. |

`cmap`, `variable`, a bool `colorbar` and `band` (once shifted) are forwarded to the builder as they
arrive. Three others are not simply passed along, because the tiers differ:

- **`title`** is applied through the scene, not the builder. The web and 3-D raster builders refuse
  it (`RasterMixin.field() got an unexpected keyword argument 'title'`, and on 3-D PyVista's own
  `"title" is an invalid keyword argument`), so it is set with `set_title` after the scene is built.
  The HoloViz tier has no `set_title` and does accept `title`, so there it stays with the builder.
- **`basemap`** is a verb on the scene in every tier that has one, never a constructor argument.
- **`kind`** reaches the builder, but only the HoloViz tier has alternative raster renderers. On web
  and 3-D a non-default `kind` is refused by that tier: `kind= is not supported by backend='web'`.

A `column` is forwarded, and is accepted only where that tier's render takes one — on a still raster
plot it is refused by the tier (`'column' is not an option Image takes on the bokeh backend`).

## What is refused, and why it is refused here

pyramids drops defaulted arguments before forwarding, so anything that arrives was set deliberately.
An argument this seam cannot honour is refused **by name**, in terms of the call you made. Left to
the tier it is still refused — but for the builder it happened to reach, and on the 3-D tier in
PyVista's terms rather than digitalearth's:

```text
ds.plot(backend="3d", title="DEM")
TypeError: "title" is an invalid keyword argument for `_common_arg_parser`
```

| Refused | Do this instead |
|---|---|
| `exclude_value` | No entry point here takes a nodata value;<br>mask the raster in pyramids before plotting. |
| `rgb_options` | Build the composite with `Map.rgb_composite` on the static tier. |
| `selectors`, `axes` | Choose a NetCDF slice with `variable=` and `band=`. |
| `bands`, `animation_axis_values`, `interval`<br>on a **still** plot | Call `animate(backend=…)` instead. |

The matplotlib-only styling (the typed `color` / `contour` / `cells` / `data_style` / `classify`
groups, the `points` overlay and a `ColorBar` *instance*) never reaches here at all — pyramids
rejects those before dispatch.

## Animation

`Dataset.animate(backend=…)` and friends forward `animate=True`. Two of the three tiers can serve it:

```python
from pyramids.dataset.collection import DatasetCollection

stack = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)

stack.animate(backend="interactive", interval=120)                     # a Panel player
stack.animate(backend="interactive", interval=120, path="flood.gif")   # written to a file
stack.animate(backend="web", interval=120, path="flood.gif")           # written to a file
```

!!! note "Leave `interval` out and the speed is digitalearth's, not pyramids'"
    With no `interval` forwarded, the tier uses digitalearth's shared `DEFAULT_FPS` of `3.0`, which
    is **333 ms** a frame, where a plain `stack.animate()` on matplotlib runs at pyramids' own
    default of 200 ms. Pass an `interval` when the speed matters.

    Whether a default `interval=200` reaches the backend at all depends on the facade:
    `Dataset.animate` declares the parameter and drops it when it is unchanged, so the backend
    cannot tell "asked for 200 ms" from "said nothing"; `DatasetCollection.animate` forwards
    `**kwargs` verbatim, so there a `200` you pass does arrive (round-2 L8).

!!! warning "The web animation needs a headless browser"
    `digitalearth[web]` does not install one, so the web example above raises until you add it:
    `ImportError: GIF export needs a headless browser … Install Playwright
    (pip install playwright && python -m playwright install chromium) or Selenium + a driver.`
    The interactive tier needs nothing extra for its player (round-2 L1).

- **interactive** builds a time cube and returns a Panel player, or writes a GIF/MP4 when you pass a
  `path`.
- **web** animates only to a file, so `path=` is required; without one the call is refused by name.
- **3-D** is refused by name. Its `Scene3D.record` needs a caller-supplied `update` callable and
  `Scene3D.orbit` sweeps the camera over a static scene, so neither can be driven from a plot call's
  keywords. Build the scene with digitalearth and call one of them directly.

## What you can reach through `plot(backend=…)`, and what needs an import

The call hands back the tier's **own object**, exactly as `ds.plot()` hands back a cleopatra glyph —
so every method on that object is in reach by chaining, and so is every attribute it exposes:

```python
scene = ds.plot(backend="3d", band=0)
scene.colorbar().orbit("fly.gif", n_frames=60)
```

A tier scene's construction-time settings are attributes on the built object. Adjust them there —
they cannot be passed to the call, because a keyword goes to the render, and several names mean one
thing to a scene and another to a builder (`height` is the HoloViz canvas height for a rasterized
layer and the web map's pixel height):

```python
scene = ds.plot(backend="web", band=0)
scene.style = "light"
```

!!! note "Some settings are read only while the scene is built"
    The web tier's `zoom` and `center` are among them: a raster layer fits the view to the data's
    extent, so assigning `scene.zoom` afterwards does not move the camera. Build the scene yourself
    when you need to choose the view — `WebMap(zoom=6, center=(31.0, 30.0))` — and add the layer to
    it.

Two things need `import digitalearth`, and no keyword can change that:

- **Composition across scenes.** One call returns one scene, so `grid`, `facet` and `quickpanels`
  — which build their own panels — cannot come back from it.
- **Everything that is not a scene**: the charts and statistical series (`bar`, `boxplot`,
  `envelope`, `stripes`, `quantile_band`), `TexturedGlobe`, `TimeSeries`, `Climatology`, the batch
  renderer and HTML gallery (`Batch`, `gallery`), and `to_backend`.

## A mesh is not drawn on these tiers

`UgridDataset.plot(backend=…)` is refused before any keyword is read: only the matplotlib tier has a
triangular render, which `digitalearth.api.MESH_BACKENDS` states. That is why the `variable_name`
translation above is inert today — the call carrying it never gets that far:

```text
UgridDataset.plot(backend="web")
TypeError: backend='web' has no unstructured-mesh render; draw a UgridDataset with
backend='matplotlib', which shades each of its faces (tripcolor)
```

## Checking what is registered

```python
from pyramids.base._plot_backends import plot_backends, resolve_plot_backend

sorted(plot_backends())              # ['3d', 'interactive', 'web']
resolve_plot_backend("interactive")  # the callable itself, loaded from the entry point
```

A registered entry point resolves to the **callable** — not to its `module:attr` string, and not to
the `functools.partial` over `quickmap` that the fallback would hand back. That difference is how you
tell the two apart.

!!! note "Needs pyramids-core 0.73.0 or newer"
    The `pyramids.plot_backends` group is resolved by pyramids-core's plot-backend seam, which first
    shipped in **0.73.0** — `0.72.0` was published after the seam merged but cut from a commit
    without it, so it carries neither `pyramids.base._plot_backends` nor the conformance kit, and
    its `Dataset.plot` has no `backend=` keyword. The declared floor is `>=0.73.0` for that reason.

    Verified against 0.73.0:

    ```python
    >>> from pyramids.base._plot_backends import plot_backends, resolve_plot_backend
    >>> sorted(plot_backends())
    ['3d', 'interactive', 'web']
    >>> resolve_plot_backend("interactive")
    <function interactive at ...>          # the entry point, not the fallback partial
    ```

::: digitalearth.pyramids_backend
