"""DecorationMixin — tiles & cartographic features for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: tiles + provider catalog (DI.1c, DI.10), coastlines/features, text/labels (DI.5), legend/colorbar.

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class DecorationMixin:
    """Tiles & cartographic features mixin (stub — methods land in later DI tasks)."""
