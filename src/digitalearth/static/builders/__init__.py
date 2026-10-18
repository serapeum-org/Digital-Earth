"""maps — the capability mixins composed into `digitalearth.static.map.Map`.

`Map` is split into a protected base (`base.GeoLayerBase`, the Scene subclass that owns the display
CRS and the reproject/extract plumbing) plus six behaviour mixins grouped by data shape / concern:
`RasterMixin`, `VectorMixin`, `DecorationMixin`, `InsetMixin`, `ProjectionMixin` and `AnimationMixin` —
the order they are composed in, which is the order `Map`'s bases are declared and the order
`tests/static/test_map_composition.py` pins. The mixins are bags of methods that assume their siblings
exist on `self` (they only run inside a composed `Map`): `InsetMixin` is the clearest case, since
`inset()` builds a second `Map` for the locator and then drives it entirely through its siblings — each
name in `reference=` is resolved to a method on it (`("land", "coastlines")` by default, from the
`_REFERENCE_LAYERS` whitelist), then `set_global()` **or** `set_bounds()` depending on whether an
`extent=` was given, then `render()`.
"""
