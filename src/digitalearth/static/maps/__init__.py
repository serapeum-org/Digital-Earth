"""maps — the capability mixins composed into :class:`digitalearth.static.map.Map`.

`Map` is split into a protected base (:class:`base.GeoLayerBase`, the Scene subclass that owns the display
CRS and the reproject/extract plumbing) plus six behaviour mixins grouped by data shape / concern:
`RasterMixin`, `VectorMixin`, `DecorationMixin`, `InsetMixin`, `ProjectionMixin` and `AnimationMixin` —
the order they are composed in, which is the order `Map`'s bases are declared. The mixins are bags of
methods that assume their siblings exist on `self` (they only run inside a composed `Map`): `InsetMixin`
is the clearest case, since it builds a second `Map` and then calls `land`, `coastlines`, `set_global`
and `set_bounds` on it.
"""
