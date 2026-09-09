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
**deprecated** — prefer ``Map``/``quickmap``. It is re-exported here so ``from digitalearth.static import
StaticGlyph`` keeps working; importing it emits no warning, but every one of its entry points does.
"""
from digitalearth.static.figure import grid, shared_colorbar
from digitalearth.static.glyph import StaticGlyph
from digitalearth.static.map import Map
from digitalearth.static.scene import Scene
from digitalearth.static.textured_globe import TexturedGlobe

__all__ = ["Scene", "Map", "TexturedGlobe", "StaticGlyph", "grid", "shared_colorbar"]
