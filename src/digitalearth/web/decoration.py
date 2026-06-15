"""DecorationMixin — web-tier decoration builders (DW.1/DW.2).

Empty skeleton: adds ``basemap``/``tiles``/``pmtiles`` (DW.1a) and ``legend``/``popup``/``tooltip``/controls
(DW.2). Methods register a basemap/control layer via ``self.add_layer`` or configure the map.
"""


class DecorationMixin:
    """Decoration builders for :class:`~digitalearth.web.map.WebMap` (populated in DW.1/DW.2)."""
