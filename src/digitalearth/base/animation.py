"""Animation defaults shared by every rendering backend.

A frame rate is pure data — a number, not a renderer — so the one rate every tier's animation entry point
starts from lives here rather than being re-declared per backend. Before this, ``DEFAULT_FPS = 3.0`` was
spelled out in :mod:`digitalearth.static.maps.animation`, :mod:`digitalearth.three_d.animation` and
:mod:`digitalearth.web.export`, and written twice more as a bare literal in
:mod:`digitalearth.interactive.animation` — five copies of one agreed number, each free to drift.

Not every ``fps`` in the package means this, and the difference matters: :data:`DEFAULT_FPS` is the rate a
caller who named none is animating *at*, whereas :data:`digitalearth.static.animation.FALLBACK_SAVE_FPS` is
the rate the encoder assumes when writing a clip whose own rate is unknown. Two concepts, two names — they
used to share one.
"""

__all__ = ["DEFAULT_FPS"]

#: Frames per second every tier's animation entry point defaults to, so one ``fps`` means one speed whichever
#: backend rendered the clip: ``Map.animate``/``Map.rotate`` (static), ``InteractiveMap.play``/
#: ``save_animation`` (interactive), ``Scene3D.orbit``/``animate`` (3-D) and ``WebMap.animate`` (web).
#:
#: Three frames a second is slow enough to read a scientific field frame by frame — the tiers previously
#: disagreed here (the 3-D tier alone defaulted to 12 for ``orbit`` and 8 for ``animate``), so the same stack
#: of rasters played at three different speeds depending on which backend drew it.
DEFAULT_FPS: float = 3.0
