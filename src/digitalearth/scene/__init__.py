"""scene — shared-axes hosts that compose cleopatra glyphs into one figure.

``Scene`` is the base host: a single matplotlib ``fig``/``ax`` that several cleopatra glyphs render onto so
layers stack (e.g. a filled field + line contours + points). ``Map`` (see :mod:`digitalearth.scene.map`)
adds geospatial behaviour (display CRS, reprojection, basemap/coastlines).
"""
from digitalearth.scene.map import Map
from digitalearth.scene.scene import Scene

__all__ = ["Scene", "Map"]
