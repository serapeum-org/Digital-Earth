"""interactive — Digital-Earth's interactive-2D web-map tier, built on the HoloViz stack.

Public surface::

    from digitalearth.interactive import InteractiveMap
    m = InteractiveMap()                       # constructing needs no engine
    m.image(dem).tiles().save("map.html")      # builders lazy-import geoviews/holoviews

HoloViz (GeoViews → HoloViews → Bokeh, + Datashader + Panel) is a **renderer, not a GIS engine**: every
element is built from pyramids-sourced numpy / GeoDataFrames (``read_array``, ``Dataset.x/.y``,
``to_geodataframe`` …) — **never** ``xarray``/``rasterio``/``cartopy`` or any GIS competitor (enforced by
``tests/test_no_competitor_imports.py``). All CRS/reproject work happens upstream in pyramids
(``Dataset.to_crs``) before an element is built.

The engine import is **lazy**: this package imports without the optional ``interactive`` extra; calling a
builder/render method without it raises an actionable ``ImportError``
(``pip install 'digitalearth[interactive]'``).
"""

from digitalearth.interactive.map import InteractiveMap

__all__ = ["InteractiveMap"]
