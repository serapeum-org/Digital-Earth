"""Engine-neutral shared logic.

Everything in this subpackage is used by two or more rendering backends and must stay free of any renderer
import (matplotlib, cleopatra, pyvista, geovista, holoviews, geoviews, hvplot, panel, datashader, bokeh,
maplibre, lonboard). Only numpy, pyramids and the standard library are allowed here.

The rule is enforced by ``tests/test_base_is_engine_neutral.py``.
"""
