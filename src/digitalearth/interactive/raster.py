"""RasterMixin — raster builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: image / rgb / quadmesh / contours / filled_contours / spaghetti (DI.1a) and large_image viewport loading (DI.14).

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class RasterMixin:
    """Raster builders mixin (stub — methods land in later DI tasks)."""
