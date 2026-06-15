"""RasterMixin — web-tier raster builders (DW.1).

Empty skeleton: DW.1 adds ``add_raster`` (recipe W1 — small rasters as a normalised RGBA image source,
large ones routed through pyramids ``to_cog``/``to_xyz`` tiles). Methods will reproject through
:meth:`~digitalearth.web.base.WebMapBase._to_display_source` and register a layer via ``self.add_layer``.
"""


class RasterMixin:
    """Raster builders for :class:`~digitalearth.web.map.WebMap` (populated in DW.1)."""
