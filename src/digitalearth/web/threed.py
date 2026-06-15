"""ThreeDMixin — web-tier 3-D builders (DW.4).

Empty skeleton: DW.4 adds ``extrusion`` (3-D choropleth), ``point_cloud``, ``tiles_3d``, ``gltf``,
``terrain`` and ``globe`` (recipe W5 — MapLibre fill-extrusion + deck.gl layers, no VTK). Methods register
via ``self.add_layer`` (or set map-level terrain/globe flags).
"""


class ThreeDMixin:
    """3-D builders for :class:`~digitalearth.web.map.WebMap` (populated in DW.4)."""
