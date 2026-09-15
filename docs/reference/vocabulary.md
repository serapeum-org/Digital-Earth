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

::: digitalearth.base.spec.style.Symbology

::: digitalearth.base.spec.style.StyleSchema

::: digitalearth.base.spec.style.StyleKey

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
