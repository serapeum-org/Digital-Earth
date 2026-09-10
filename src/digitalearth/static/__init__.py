"""static — the matplotlib backend: shared-axes hosts that compose cleopatra glyphs into one figure.

One of the package's four rendering backends, and the default one (``quickmap(..., backend="matplotlib")``);
the others are :mod:`digitalearth.interactive` (HoloViz), :mod:`digitalearth.three_d` (PyVista) and
:mod:`digitalearth.web` (MapLibre + deck.gl). It is the only backend installed by default — the rest are
optional extras.

``Scene`` is the base host: a single matplotlib ``fig``/``ax`` that several cleopatra glyphs render onto so
layers stack (e.g. a filled field + line contours + points). ``Map`` (see :mod:`digitalearth.static.map`)
adds geospatial behaviour (display CRS, reprojection, basemap/coastlines). ``TexturedGlobe`` (see
:mod:`digitalearth.static.textured_globe`) is the 3-D outlier: a pyramids raster or tile basemap draped over
cleopatra's textured sphere on a matplotlib ``Axes3D``.

Alongside the scene hosts this backend owns its charts (:mod:`~digitalearth.static.charts`), statistical
series (:mod:`~digitalearth.static.series`), time-series products (:mod:`~digitalearth.static.temporal`),
geostatistical maps (:mod:`~digitalearth.static.geostatistics`) and animation
(:mod:`~digitalearth.static.animation`) — the matplotlib-rendering counterparts of what the other backends
provide for themselves.

``StaticGlyph`` (:mod:`digitalearth.static.glyph`) is the package's original entry point and is
**deprecated** — prefer ``Map``/``quickmap``. It stays importable as ``from digitalearth.static import
StaticGlyph``, but is resolved **lazily** through a PEP 562 module ``__getattr__``: the package facade
imports :mod:`digitalearth.static`, so an eager re-export would load the deprecated module on every
``import digitalearth``. Importing it emits no warning; every one of its entry points does. Removing it
later is then a one-line deletion here.
"""

from digitalearth.static.figure import grid, shared_colorbar
from digitalearth.static.map import Map

# Importable as ``from digitalearth.static import OffLimbError`` so a caller can catch it, but kept
# out of ``__all__``: it is a signal the layer methods already answer, not part of the backend's
# advertised surface, and every name in that list is also a package-root export.
from digitalearth.static.maps.base import OffLimbError  # noqa: F401
from digitalearth.static.scene import Scene
from digitalearth.static.textured_globe import TexturedGlobe

__all__ = ["Scene", "Map", "TexturedGlobe", "StaticGlyph", "grid", "shared_colorbar"]


def __getattr__(name: str):
    """Resolve ``StaticGlyph`` lazily so ``import digitalearth`` does not load the deprecated module.

    Args:
        name: The attribute being looked up on the ``digitalearth.static`` package.

    Returns:
        The :class:`~digitalearth.static.glyph.StaticGlyph` class.

    Raises:
        AttributeError: for any other name.
    """
    if name == "StaticGlyph":
        from digitalearth.static.glyph import StaticGlyph

        globals()[name] = StaticGlyph
        return StaticGlyph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
