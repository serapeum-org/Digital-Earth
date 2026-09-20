"""DI.14 — large-raster / COG viewport loading (large_image).

Drives pyramids' windowed read through a fake COG dataset (so the test needs no multi-GB file or network).
Since #297 every frame is read through :class:`~digitalearth.base.sources.view.SourceView` with a request
from :class:`~digitalearth.base.spec.RenderTarget`, so what these assert is the *read request*: the first
frame windows the whole raster within the budget, and a ``RangeXY`` event reads the window it asks for.
Runs in the ``interactive`` pixi env.
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


@pytest.fixture
def m() -> InteractiveMap:
    return InteractiveMap(crs=3857)


class TestLargeImage:
    """``large_image`` — viewport-driven decimated reads via pyramids."""

    def test_the_static_frame_windows_the_whole_raster(self, m):
        """The first frame reads the source's own extent, within the budget.

        Args:
            m: The map under test.

        Test scenario:
            Rewritten for #297: the frame came from `preview(max_size=side)`, a square whatever the map's
            shape. It is now a windowed read of the whole raster through the data tier, so what is asserted
            is the window rather than the call that produced it.
        """
        cog = _FakeCOG()
        m.large_image(cog, dynamic=False, max_pixels=64 * 64)
        assert isinstance(m.layers[0], hv.Image), f"got {type(m.layers[0])}"
        assert cog.read_calls, (
            "the first frame must read through pyramids' windowed read"
        )
        assert cog.read_calls[-1][0] == pytest.approx(cog.bbox), (
            f"the first frame must window the whole raster, got {cog.read_calls[-1][0]}"
        )

    def test_the_canvas_stays_under_max_pixels(self, m):
        """`max_pixels` is a budget of cells, and the canvas follows the map's aspect within it.

        Args:
            m: The map under test.

        Test scenario:
            Rewritten for #297: the canvas was `max(64, sqrt(max_pixels))` square. It now comes from
            `RenderTarget.view_request`, so the two sides may differ — what must hold is that their product
            fits the budget.
        """
        cog = _FakeCOG()
        m.large_image(cog, dynamic=False, max_pixels=128 * 128)
        _, width, height = cog.read_calls[-1]
        assert width * height <= 128 * 128, (
            f"a {width}x{height} canvas exceeds the budget"
        )

    def test_dynamic_returns_dynamicmap(self, m):
        cog = _FakeCOG()
        m.large_image(cog, dynamic=True)
        assert isinstance(m.layers[0], hv.DynamicMap), (
            "dynamic=True must wrap a RangeXY DynamicMap"
        )

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
        assert bbox == pytest.approx((-5.0e5, -4.0e5, 5.0e5, 4.0e5)), (
            f"bbox not the window: {bbox}"
        )

    def test_missing_cog_surface_raises(self, m, dataset):
        """A plain Dataset without read_part/preview raises an actionable error (upstream-gated)."""

        class _Plain:
            epsg = 3857

        plain = _Plain()
        with pytest.raises(AttributeError, match="read_part"):
            m.large_image(plain)

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
        """The 1-based default ``band=1`` reaches pyramids' windowed read as 0-based ``band=0``.

        Args:
            m: The map under test.

        Test scenario:
            The frame is read through the data tier since #297, so the conversion is asserted where the read
            happens rather than on `preview`.
        """
        recorded = {}

        class _BandCOG(_FakeCOG):
            def read_part(self, *, bbox, dst_width, dst_height, bbox_crs=4326, band=1):
                recorded["band"] = band
                return super().read_part(
                    bbox=bbox,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    bbox_crs=bbox_crs,
                    band=band,
                )

        m.large_image(_BandCOG(), dynamic=False, max_pixels=64 * 64)
        assert recorded["band"] == 0, (
            f"1-based band=1 must reach pyramids as 0-based 0, got {recorded['band']}"
        )
