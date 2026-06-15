"""BigDataMixin — web-tier big-data builders (DW.3).

Empty skeleton: DW.3 adds ``heatmap``/``cluster`` (MapLibre) and ``deck_scatter``/``deck_polygons`` (deck.gl
via lonboard GeoArrow), plus a logged feature-count threshold that routes large vector through deck.gl
(recipe W3/W4). Methods register via ``self.add_layer``.
"""


class BigDataMixin:
    """Big-data builders for :class:`~digitalearth.web.map.WebMap` (populated in DW.3)."""
