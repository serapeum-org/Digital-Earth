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
| `band=0` (0-based) | `band=1` (1-based) | 0-based upstream, 1-based here. Unshifted,<br>`band=0` read as index `-1`. |
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
    `interval=200` is pyramids' own default, and a facade **drops** a defaulted argument before
    forwarding it — so the backend cannot tell "the caller asked for 200 ms" from "the caller said
    nothing". With nothing forwarded, the tier uses digitalearth's shared `DEFAULT_FPS` of `3.0`,
    which is **333 ms** a frame, where a plain `stack.animate()` on matplotlib runs at pyramids'
    200 ms. Pass a non-default `interval` when the speed matters.

- **interactive** builds a time cube and returns a Panel player, or writes a GIF/MP4 when you pass a
  `path`.
- **web** animates only to a file, so `path=` is required; without one the call is refused by name.
- **3-D** is refused by name. Its `Scene3D.record` needs a caller-supplied `update` callable and
  `Scene3D.orbit` sweeps the camera over a static scene, so neither can be driven from a plot call's
  keywords. Build the scene with digitalearth and call one of them directly.

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

!!! warning "Needs a pyramids-core that ships the seam"
    The `pyramids.plot_backends` group is resolved by pyramids-core's plot-backend seam, which merged
    after `0.72.0` was cut and is therefore **in no release yet**: the installed `0.72.0` carries
    neither `pyramids.base._plot_backends` nor the conformance kit, and `Dataset.plot` has no
    `backend=` keyword. Until a release ships it, the entry points are registered and nothing reads
    them — inert, not conflicting, and `plot()` behaves exactly as before. The declared floor of
    `pyramids-core >=0.72.0` is simply the current release; it will need raising to whichever
    release first carries the seam, at which point the two skipped tests here start guarding.

::: digitalearth.pyramids_backend
