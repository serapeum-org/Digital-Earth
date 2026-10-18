"""`Map.field(scheme=, vmin=, vmax=)` confines a graduated raster to the value window (#378).

Before, `vmin`/`vmax` were accepted and then silently ignored once a `scheme` classified the band, so a few
outliers could stretch the classes and the limits read as supported without being so. Now they pin the classes
to the window: the classes span exactly `[vmin, vmax]` and out-of-range cells clamp into the end classes, so two
maps given the same limits are comparable. With no scheme the limits still set the continuous ramp, as before.

Edges are computed with numpy as an even split of the window, never read back from the figure under test.
"""

import numpy as np
import pytest

from digitalearth.static import Map
from digitalearth.static.maps.raster import _clip_window


def _values(dataset) -> np.ndarray:
    """Return the fixture raster's valid cell values as a flat float array."""
    return np.ma.compressed(dataset.read_array(masked=True)).astype(float)


class TestClipWindow:
    """`_clip_window` pulls `vmin`/`vmax` out of the glyph options as a window."""

    def test_both_limits_become_the_window_and_leave_opts(self):
        """`vmin`/`vmax` are removed and returned as a pair; the rest of `opts` is untouched."""
        opts = {"vmin": -100, "vmax": 100, "cmap": "viridis"}
        assert _clip_window(opts) == (-100, 100), "window is the pair"
        assert opts == {"cmap": "viridis"}, f"opts kept {opts}"

    def test_no_limits_is_no_window(self):
        """With neither limit set there is no window to apply."""
        assert _clip_window({"cmap": "viridis"}) is None


class TestFieldSchemeClip:
    """`Map.field` pins a graduated raster's classes to `vmin`/`vmax`, and leaves the ramp alone."""

    def test_vmin_vmax_pin_a_graduated_field_to_the_window(self, dataset):
        """`scheme="equal_interval"` with `vmin`/`vmax` steps at an even split of the window itself.

        Test scenario:
            A window at the 20th/80th percentiles; the drawn `BoundaryNorm` must step at the four equal
            intervals of `[vmin, vmax]`, so the colorbar spans the requested limits whatever the data holds.
        """
        vals = _values(dataset)
        lo, hi = (float(x) for x in np.percentile(vals, [20, 80]))
        m = Map().field(dataset, scheme="equal_interval", k=4, vmin=lo, vmax=hi)
        expected = list(np.linspace(lo, hi, 5))
        drawn = list(m.ax.images[-1].norm.boundaries)
        assert drawn == pytest.approx(expected), (drawn, expected)

    def test_the_window_differs_from_the_unclipped_edges(self, dataset):
        """The same field without limits steps at the whole band's edges — proving the window changed them."""
        vals = _values(dataset)
        lo, hi = (float(x) for x in np.percentile(vals, [20, 80]))
        clipped = list(
            Map()
            .field(dataset, scheme="equal_interval", k=4, vmin=lo, vmax=hi)
            .ax.images[-1]
            .norm.boundaries
        )
        whole = list(
            Map()
            .field(dataset, scheme="equal_interval", k=4)
            .ax.images[-1]
            .norm.boundaries
        )
        assert clipped != whole, (clipped, whole)

    def test_without_a_scheme_vmin_vmax_still_set_the_ramp(self, dataset):
        """With no scheme the limits are the continuous `clim`, untouched by the clip path."""
        vals = _values(dataset)
        lo, hi = float(vals.min()), float(np.percentile(vals, 50))
        norm = Map().field(dataset, vmin=lo, vmax=hi).ax.images[-1].norm
        assert (norm.vmin, norm.vmax) == pytest.approx((lo, hi)), (norm.vmin, norm.vmax)
