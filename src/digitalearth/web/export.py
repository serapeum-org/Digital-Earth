"""ExportMixin — web-tier export builders (DW.6).

Empty skeleton: the self-contained-HTML ``save`` lives on :class:`~digitalearth.web.base.WebMapBase` for
DW.0; DW.6 extends export here — true offline bundling (inline the ``maplibre-gl`` JS rather than CDN-ref it)
and an optional PNG snapshot via a headless browser (gated optional dep), per recipe W7.
"""


class ExportMixin:
    """Export builders for :class:`~digitalearth.web.map.WebMap` (populated in DW.6)."""
