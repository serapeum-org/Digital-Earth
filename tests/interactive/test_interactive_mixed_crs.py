"""Mixed-CRS guard for an element handed in directly (IN-16).

Builder inputs are pre-reprojected through pyramids, so a layer built here is always in the display CRS.
An element a caller builds and hands to ``add_layer`` / ``rasterize`` / ``datashade`` is not, and a
GeoViews element silently drawn in the wrong CRS mis-registers with the rest of the map. When the element's
CRS resolves to an EPSG code that differs from the display CRS, it is refused at the door with both named.
"""

import pytest

pytest.importorskip("geoviews")

import geoviews as gv  # noqa: E402
import holoviews as hv  # noqa: E402

from digitalearth.interactive import InteractiveMap  # noqa: E402


class TestMixedCrsGuard:
    """A resolvable CRS mismatch is refused; a match and a plain element are accepted."""

    def test_a_mismatched_element_is_refused(self):
        """An element in EPSG:3857 on a non-Mercator map is refused, naming the element's CRS."""
        other = InteractiveMap(crs=32618)
        element = gv.Points([(0.0, 0.0)], crs=gv.util.process_crs(3857))
        with pytest.raises(ValueError, match="EPSG:3857"):
            other.add_layer(element)

    def test_a_matching_element_is_accepted(self):
        """An element already in the display CRS registers and chains."""
        m = InteractiveMap(crs=3857)
        element = gv.Points([(0.0, 0.0)], crs=gv.util.process_crs(3857))
        assert m.add_layer(element) is m

    def test_a_plain_holoviews_element_is_accepted(self):
        """A plain ``hv`` element carries no CRS — it is display coordinates by contract — so it passes."""
        m = InteractiveMap(crs=3857)
        assert m.add_layer(hv.Points([(0.0, 0.0)])) is m

    def test_the_guard_reaches_the_datashade_path(self):
        """A pre-built mismatched element handed to ``datashade`` is refused the same way (``_as_element``)."""
        other = InteractiveMap(crs=32618)
        element = gv.Points([(0.0, 0.0)], crs=gv.util.process_crs(3857))
        with pytest.raises(ValueError, match="EPSG:3857"):
            other.datashade(element)
