"""DI.14 — large-raster / COG viewport loading (large_image).

Drives the pyramids ``read_part``/``preview`` viewport surface through a fake COG dataset (so the test
needs no multi-GB file or network): the static frame uses ``preview``; a viewport ``RangeXY`` event
issues a ``read_part`` whose bbox follows the requested window. Runs in the ``interactive`` pixi env.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


class _FakeCOG:
    """A pyramids-Dataset-like COG exposing read_part / preview and recording its read windows."""

    epsg = 3857
    bbox = (-1.0e6, -1.0e6, 1.0e6, 1.0e6)

    def __init__(self):
        self.read_calls = []
        self.preview_calls = []

    def preview(self, *, max_size=1024, band=1):
        self.preview_calls.append(max_size)
        return np.random.default_rng(0).random((max_size, max_size))

    def read_part(self, *, bbox, dst_width, dst_height, bbox_crs=4326, band=1):
        self.read_calls.append((bbox, dst_width, dst_height))
        return np.random.default_rng(0).random((dst_height, dst_width))


@pytest.fixture()
def m() -> InteractiveMap:
    return InteractiveMap(crs=3857)


class TestLargeImage:
    """``large_image`` — viewport-driven decimated reads via pyramids."""

    def test_static_frame_uses_preview(self, m):
        cog = _FakeCOG()
        m.large_image(cog, dynamic=False, max_pixels=64 * 64)
        assert isinstance(m.layers[0], hv.Image), f"got {type(m.layers[0])}"
        assert (
            cog.preview_calls
        ), "the static frame must come from a cheap overview preview"

    def test_canvas_stays_under_max_pixels(self, m):
        cog = _FakeCOG()
        m.large_image(cog, dynamic=False, max_pixels=128 * 128)
        side = cog.preview_calls[-1]
        assert side * side <= 128 * 128, f"preview side {side} exceeds the pixel budget"

    def test_dynamic_returns_dynamicmap(self, m):
        cog = _FakeCOG()
        m.large_image(cog, dynamic=True)
        assert isinstance(
            m.layers[0], hv.DynamicMap
        ), "dynamic=True must wrap a RangeXY DynamicMap"

    def test_viewport_event_issues_read_part(self, m):
        """A RangeXY event must drive a pyramids read_part whose bbox follows the window."""
        from holoviews.streams import RangeXY

        cog = _FakeCOG()
        m.large_image(cog, dynamic=True, max_pixels=100 * 100)
        dmap = m.layers[0]
        stream = next(s for s in dmap.streams if isinstance(s, RangeXY))
        stream.event(x_range=(-5.0e5, 5.0e5), y_range=(-4.0e5, 4.0e5))
        dmap[()]  # materialise the current frame to trigger the callback
        assert cog.read_calls, "a viewport event must trigger a pyramids read_part"
        bbox = cog.read_calls[-1][0]
        assert bbox == pytest.approx(
            (-5.0e5, -4.0e5, 5.0e5, 4.0e5)
        ), f"bbox not the window: {bbox}"

    def test_missing_cog_surface_raises(self, m, dataset):
        """A plain Dataset without read_part/preview raises an actionable error (upstream-gated)."""

        class _Plain:
            epsg = 3857

        with pytest.raises(AttributeError, match="read_part"):
            m.large_image(_Plain())

    def test_real_dataset_static_frame_is_non_blank(self, dataset):
        """A real pyramids Dataset renders a non-blank decimated preview (band default 1 → 0-based read).

        Regression for the 1-based/0-based mismatch: pyramids ``preview``/``read_part`` are 0-based, so the
        1-based ``band`` must be converted. The fake-COG tests ignore ``band`` and so could not catch this.
        """
        m = InteractiveMap(crs=dataset.epsg)
        m.large_image(dataset, dynamic=False, max_pixels=80 * 80)
        img = m.layers[0]
        assert isinstance(img, hv.Image), f"got {type(img)}"
        z = np.asarray(img.dimension_values(2, flat=False), dtype="float64")
        finite = np.isfinite(z)
        assert finite.sum() > 0, "static frame is all-NoData (blank)"
        assert np.nanstd(z[finite]) > 0, "static frame is a single flat value (blank)"

    def test_band_below_one_raises(self, m):
        """``band`` is 1-based; a sub-1 value is rejected before any pyramids read."""
        cog = _FakeCOG()
        with pytest.raises(ValueError, match="band is 1-based"):
            m.large_image(cog, band=0)

    def test_band_passed_to_pyramids_is_zero_based(self, m):
        """The 1-based default ``band=1`` reaches pyramids ``preview`` as 0-based ``band=0``."""
        recorded = {}

        class _BandCOG(_FakeCOG):
            def preview(self, *, max_size=1024, band=1):
                recorded["band"] = band
                return np.random.default_rng(0).random((max_size, max_size))

        m.large_image(_BandCOG(), dynamic=False, max_pixels=64 * 64)
        assert recorded["band"] == 0, f"1-based band=1 must reach pyramids as 0-based 0, got {recorded['band']}"
