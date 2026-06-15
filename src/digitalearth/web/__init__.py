"""web — Digital-Earth's MapLibre + deck.gl web-map tier.

Public surface::

    from digitalearth.web import WebMap
    m = WebMap()                                  # constructing needs no engine
    m.render()                                     # builders lazy-import maplibre
    m.save("map.html")                             # standalone HTML page

MapLibre GL JS + deck.gl (via the ``maplibre`` py-maplibregl anywidget, with optional ``lonboard`` for
GeoArrow-fast big-data deck.gl) are a **renderer, not a GIS engine**: every layer is built from
pyramids-sourced numpy / GeoDataFrames — **never** ``xarray``/``rasterio``/``cartopy`` or any GIS competitor
(enforced by ``tests/test_no_competitor_imports.py``). All CRS/reproject work happens upstream in pyramids
(``Dataset.to_crs``) before a layer is built; MapLibre renders EPSG:3857 / 4326 only.

The engine import is **lazy**: this package imports without the optional ``web`` extra; calling a
builder/render method without it raises an actionable ``ImportError`` (``pip install 'digitalearth[web]'``).
"""

from digitalearth.web.map import WebMap

__all__ = ["WebMap"]
