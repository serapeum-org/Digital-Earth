"""AnimationMixin — animation playback & export for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: Player playback + GIF/MP4/scrubber export (DI.11).

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class AnimationMixin:
    """Animation playback & export mixin (stub — methods land in later DI tasks)."""
