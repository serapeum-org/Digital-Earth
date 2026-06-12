"""DashboardMixin — Panel dashboards for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: dashboard / serve / save_app (DI.4); layer control / attribute table / URL share (DI.13).

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class DashboardMixin:
    """Panel dashboards mixin (stub — methods land in later DI tasks)."""
