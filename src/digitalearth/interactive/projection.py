"""ProjectionMixin — non-Mercator projections for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: projection via the HoloViews matplotlib backend + graticule (DI.9).

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class ProjectionMixin:
    """Non-Mercator projections mixin (stub — methods land in later DI tasks)."""
