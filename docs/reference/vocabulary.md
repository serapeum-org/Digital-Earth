# The figure vocabulary

A bare `[float, float, float, float]` cannot say whether it is `[xmin, xmax, ymin, ymax]` or
`[xmin, ymin, xmax, ymax]`, and this package existed because both spellings were live in one tier at once. A
bare `band=1` cannot say what to do when a composite needs three bands, the 3-D tier needs a vertical level
and an animation needs frames. `digitalearth.base.spec` is the answer to both: a small set of **value
objects** — frozen, comparable, renderer-free — that a figure is described in.

They are engine-neutral by construction. Nothing here imports matplotlib, pyvista, holoviews or maplibre, and
`tests/test_base_is_engine_neutral.py` enforces it by reading the source, so even a lazy import inside a
function fails the build. Where the vocabulary genuinely needs a renderer's arithmetic — classification lives
in cleopatra — it declares a **seam** instead and something above `base/` fills it.

## `Bounds` — one rectangle, with its ordering named

Two orderings were live in the static tier at once, plus a third convention that hid an EPSG:4326 assumption
inside a bare tuple. `Bounds` carries its own CRS and makes each ordering a method, so a call site states
which it wants rather than leaving it to position.

::: digitalearth.base.spec.bounds.Bounds

## `Scale` — how a value becomes a colour

The limits were derived in several places, the "a constant band has no range" guard was written out five
times, and three tiers each called cleopatra's classifier with their own error handling. The copies agreed by
luck and drifted where they did not. A `Scale` derived once and reused across frames is what stops an
animation's colours flickering, and what lets a legend show the colours that were actually drawn.

::: digitalearth.base.spec.scale.Scale

## `Selection` — which slice of a dataset a layer draws

`band` was one integer being asked several different questions, with each tier adding a parameter beside it
where it needed more. Here it is one value, and `band` is a **tuple**: a composite is not a special case of a
different parameter, it is a selection of three bands.

::: digitalearth.base.spec.selection.Selection

## `DataRef` — where the data is, rather than the data

A layer that owns its data object cannot be tiled, cannot decimate, cannot hold a point cloud too large to
read, and cannot round-trip through a dict. `DataRef` names the data; the registry opens it.

::: digitalearth.base.spec.dataref.DataRef

## `Encoding` and `Symbology` — how a layer looks

Styling reached the renderers as undeclared keyword soup: 26 keys in the static tier alone, in no signature
anywhere, so a caller could not ask what a builder accepts and a typo was a silently ignored keyword rather
than an error. `Encoding` makes a visual channel a **key** rather than a parameter on every builder — which
is what makes size, width, extrusion and text cheap to add — and `Symbology` is a layer's style as a declared
value.

::: digitalearth.base.spec.encoding.Encoding

::: digitalearth.base.spec.encoding.Channel

### The declared channels

`CHANNELS` is the table a channel is a row in. It is the growth axis the whole type exists for: `height`
and `text` are already there, ahead of the extrusion and label features that will need them, because adding
a channel should cost a row plus a fold rule per backend — not a keyword on every builder on four tiers.

::: digitalearth.base.spec.style.Symbology

::: digitalearth.base.spec.style.StyleSchema

::: digitalearth.base.spec.style.StyleKey

## The shared defaults

Constants that exist so a second value cannot appear. `DEFAULT_BAND` is the band a builder reads when the
caller names none; `DEFAULT_CLASS_COUNT` is the five classes a graduated scheme cuts by default, which every
tier had written out separately.

::: digitalearth.base.spec.selection.DEFAULT_BAND

::: digitalearth.base.spec.scale.DEFAULT_CLASS_COUNT

## `ViewRequest` — how much to read, and how finely

A `DataRef` says *where* data is and a `Selection` says *which slice*; neither says how much of it to read or
at what resolution. Three of the four backend studies invented this third question independently, which is
the usual sign a type is missing rather than optional.

::: digitalearth.base.spec.viewrequest.ViewRequest

## `SourceView` — a view that remembers its own address

A `Source` carries materialised data and forgets where it came from, so the only way to get different pixels
was to start again from the caller's object. That is what blocks dynamic tiling, level of detail and point
clouds too large to hold. `SourceView` keeps the reference and the slice, so it can be read again.

::: digitalearth.base.sources.view.SourceView

## `LegendSpec` — a legend derived from what was drawn

Every tier built one its own way, and nothing structurally tied a swatch to the colour actually drawn.
Deriving the rows from the resolved `Scale` makes the agreement a construction rather than a maintenance
task.

::: digitalearth.base.spec.legend.LegendSpec

::: digitalearth.base.spec.legend.LegendEntry

## `PointArrays` — geometry as coordinate arrays

Turning a `FeatureCollection` into numpy arrays was written out 22 times across 9 files, in three spellings,
with inconsistent handling of non-finite coordinates — the ones a globe or clipped CRS produces for the far
side of the world.

::: digitalearth.base.points.PointArrays

## Getting data into the display CRS

The three helpers each tier used to define for itself.

::: digitalearth.base.display.needs_reproject

::: digitalearth.base.display.to_display_source

::: digitalearth.base.display.auto_cmap

## `LayerSpec` and `LayerTree` — a layer described, and addressed by id

Every tier kept a list called `layers`, three of them addressed it by position, and none kept what a layer is.
`LayerSpec` is that description — kind, source, slice, style — with no data object, viewport or engine handle, and
`LayerTree` orders layers bottom first and addresses them by id. Every change to a tree returns a new tree.

::: digitalearth.base.spec.layer.LayerSpec

::: digitalearth.base.spec.layer.LayerTree

::: digitalearth.base.spec.layer.LAYER_REFERENCE

## `Viewport` and `Camera` — the view as a value

The view used to be loose state on each facade, mutated in place to animate. These are values: a change of view is
a new view.

::: digitalearth.base.spec.viewport.Viewport

::: digitalearth.base.spec.viewport.Camera

::: digitalearth.base.spec.viewport.DEFAULT_VIEW_ANGLE

## `RenderTarget` — the output owns the read budget

A layer never chooses how much of its source to read; the output it is rendered to does, and `view_request` is how a
view becomes a `ViewRequest`.

::: digitalearth.base.spec.target.RenderTarget

::: digitalearth.base.spec.target.TARGET_KINDS

::: digitalearth.base.spec.target.DEFAULT_BUDGETS

## `PanelSpec` and `FigureSpec` — a figure that describes itself

The whole figure — sources, one layer tree, panels with their own views — round-trips through `to_dict` and
`from_dict` with no renderer imported, and records the schema version it was written in.

::: digitalearth.base.spec.figure.FigureSpec

::: digitalearth.base.spec.figure.PanelSpec

::: digitalearth.base.spec.figure.SCHEMA_VERSION

## The registries

How a reference becomes data, without `base/` knowing any reader — and how `Scale` reaches a classifier
without `base/` importing the renderer it lives in.

::: digitalearth.base.registry.register_resolver

::: digitalearth.base.registry.resolvers

::: digitalearth.base.registry.temporary_resolver

::: digitalearth.base.registry.register_object

::: digitalearth.base.registry.clear_objects

::: digitalearth.base.registry.resolve_uri

::: digitalearth.base.registry.register_classifier

::: digitalearth.base.registry.get_classifier

::: digitalearth.base.registry.temporary_classifier
