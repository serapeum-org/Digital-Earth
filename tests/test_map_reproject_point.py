"""Tests for Map._reproject_point — single lon/lat -> display CRS, None off the globe (PA-8)."""
import matplotlib
import numpy as np

matplotlib.use("Agg")

from digitalearth.scene import Map  # noqa: E402
from digitalearth.scene import projections  # noqa: E402


class TestReprojectPoint:
    """Tests for Map._reproject_point."""

    def test_same_crs_returns_finite_point(self):
        """A point already in the display CRS reprojects to a finite (x, y).

        Test scenario:
            In EPSG:4326 the identity transform returns the same lon/lat as finite floats.
        """
        m = Map(crs=4326)
        xy = m._reproject_point(10.0, 20.0, 4326)
        assert xy is not None, "an in-domain point must not be dropped"
        assert np.isfinite(xy[0]) and np.isfinite(xy[1]), f"expected finite coords, got {xy}"
        assert xy[0] == 10.0 and xy[1] == 20.0, f"identity transform changed the point: {xy}"

    def test_near_side_point_on_globe_is_finite(self):
        """A point near the projection centre on a globe reprojects to finite coords.

        Test scenario:
            On an orthographic globe centred at (0, 0), the centre point itself is on the near side.
        """
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        xy = m._reproject_point(0.0, 0.0, 4326)
        assert xy is not None and np.isfinite(xy[0]) and np.isfinite(xy[1]), f"near-side point dropped: {xy}"

    def test_far_side_point_on_globe_returns_none(self):
        """A point on the far hemisphere of an orthographic globe returns None.

        Test scenario:
            On an orthographic globe centred at (0, 0), longitude 180 is on the far side and is dropped.
        """
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        assert m._reproject_point(180.0, 0.0, 4326) is None, "a far-side point must reproject to None"
