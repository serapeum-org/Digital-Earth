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
| `variable_name=` | `variable=` | The `UgridDataset` facade's spelling of<br>the mesh selector. |
| `interval=200` (ms/frame) | `fps=5.0` | The facades describe an interval; the<br>tier verbs take a rate. |

`title`, `basemap`, a bool `colorbar`, `kind`, `cmap`, `column` and `variable` need no translation and
are forwarded as they arrive.

## What is refused, and why it is not ignored

pyramids drops defaulted arguments before forwarding, so anything that arrives was set deliberately.
An argument this seam cannot honour is therefore refused **by name** rather than dropped — a
silently-ignored argument is indistinguishable from an applied one.

| Refused | Do this instead |
|---|---|
| `exclude_value` | digitalearth applies its own nodata handling; mask upstream, or plot on `backend="matplotlib"`. |
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

stack.animate(backend="interactive", interval=200)                     # a Panel player
stack.animate(backend="interactive", interval=200, path="flood.gif")   # written to a file
stack.animate(backend="web", interval=200, path="flood.gif")           # written to a file
```

- **interactive** builds a time cube and returns a Panel player, or writes a GIF/MP4 when you pass a
  `path`.
- **web** animates only to a file, so `path=` is required; without one the call is refused by name.
- **3-D** is refused by name. Its `Scene3D.record` needs a caller-supplied `update` callable and
  `Scene3D.orbit` sweeps the camera over a static scene, so neither can be driven from a plot call's
  keywords. Build the scene with digitalearth and call one of them directly.

## Checking what is registered

```python
from pyramids.base._plot_backends import plot_backends, resolve_plot_backend

sorted(plot_backends())            # ['3d', 'interactive', 'web']
resolve_plot_backend("interactive")  # digitalearth.pyramids_backend:interactive
```

!!! warning "Needs a pyramids-core that ships the seam"
    The `pyramids.plot_backends` group is resolved by pyramids-core's plot-backend seam. Until a
    pyramids-core release carries it, the entry points are registered but nothing reads them, and
    `plot(backend=…)` behaves as it did before. Nothing breaks in the meantime — the registration is
    inert, not conflicting.

::: digitalearth.pyramids_backend
