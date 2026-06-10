"""maps — the capability mixins composed into :class:`digitalearth.scene.map.Map`.

``Map`` is split into a protected base (:class:`base.GeoLayerBase`, the Scene subclass that owns the display
CRS and the reproject/extract plumbing) plus five behaviour mixins grouped by data shape / concern:
``RasterMixin``, ``VectorMixin``, ``DecorationMixin``, ``ProjectionMixin`` and ``AnimationMixin``. The mixins
are bags of methods that assume their siblings exist on ``self`` (they only run inside a composed ``Map``).
"""
