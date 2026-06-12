"""InteractionMixin — interactivity streams for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: tap-to-inspect / rich hover (DI.7), draw-AOI / linked selection (DI.8).

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class InteractionMixin:
    """Interactivity streams mixin (stub — methods land in later DI tasks)."""
