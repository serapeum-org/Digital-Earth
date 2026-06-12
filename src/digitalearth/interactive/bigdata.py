"""BigDataMixin — Datashader wrappers for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: datashade / rasterize + zoom re-rasterization, categorical + trajectory aggregation (DI.2).

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class BigDataMixin:
    """Datashader wrappers mixin (stub — methods land in later DI tasks)."""
