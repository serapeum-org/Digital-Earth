"""scene — shared-axes hosts that compose cleopatra glyphs into one figure.

``Scene`` is the base host: a single matplotlib ``fig``/``ax`` that several cleopatra glyphs render onto so
layers stack (e.g. a filled field + line contours + points). ``Map`` (see :mod:`digitalearth.scene.map`)
adds geospatial behaviour (display CRS, reprojection, basemap/coastlines). ``TexturedGlobe`` (see
:mod:`digitalearth.scene.textured_globe`) is the 3-D outlier: a pyramids raster or tile basemap draped over
cleopatra's textured sphere on a matplotlib ``Axes3D``.
"""
from digitalearth.scene.figure import grid, shared_colorbar
from digitalearth.scene.map import Map
from digitalearth.scene.scene import Scene
from digitalearth.scene.textured_globe import TexturedGlobe

__all__ = ["Scene", "Map", "TexturedGlobe", "grid", "shared_colorbar"]
