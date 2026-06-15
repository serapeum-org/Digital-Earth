"""VectorMixin — web-tier vector builders (DW.2).

Empty skeleton: DW.2 adds ``points``/``lines``/``polygons`` and ``choropleth`` (recipe W2 — GeoJSON sources
with data-driven MapLibre paint, compiling breaks from the shared classifier into step/interpolate/match
expressions). Methods reproject through the base ``_to_display_source`` and register via ``self.add_layer``.
"""


class VectorMixin:
    """Vector builders for :class:`~digitalearth.web.map.WebMap` (populated in DW.2)."""
