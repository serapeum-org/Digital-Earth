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
series (:mod:`~digitalearth.static.series`), time-series products (:mod:`~digitalearth.static.temporal`) and
animation (:mod:`~digitalearth.static.animation`) — the matplotlib-rendering counterparts of what the other
backends provide for themselves.

Everything this backend exports is eager: the names below are the whole surface, and each is also a
package-root export (``from digitalearth import Map``). There is no lazily-resolved name left, so the module
has no PEP 562 ``__getattr__`` — an unknown attribute fails with Python's own message.
"""

from digitalearth.static.figure import grid, shared_colorbar
from digitalearth.static.map import Map

# Importable as ``from digitalearth.static import OffLimbError`` so a caller can catch it, but kept
# out of ``__all__``: it is a signal the layer methods already answer, not part of the backend's
# advertised surface, and the guard on that list expects every name it carries to be a package-root
# export too.
from digitalearth.static.maps.base import OffLimbError  # noqa: F401
from digitalearth.static.scene import Scene
from digitalearth.static.textured_globe import TexturedGlobe

__all__ = ["Scene", "Map", "TexturedGlobe", "grid", "shared_colorbar"]
