"""DW.1 — web-tier raster builder (``add_raster``) + basemap/tiles encoding helpers.

The PNG/coordinate encoding lives in pure helpers (``_rgba_png_datauri`` / ``_image_coordinates``) tested
without the engine (numpy/matplotlib are core deps via cleopatra). ``add_raster`` itself ``importorskip``s
maplibre and exercises the reproject → image-source → render → save path on the shared ``dataset`` fixture.
"""

import base64

import numpy as np
import pytest

from digitalearth.web import WebMap

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestRgbaPngDataUri:
    """``_rgba_png_datauri`` colour-maps a 2-D array to a transparent-NoData PNG data-URI (no engine)."""

    def test_returns_a_png_data_uri(self):
        arr = np.arange(12.0).reshape(3, 4)
        uri = WebMap()._rgba_png_datauri(arr, "viridis")
        assert uri.startswith("data:image/png;base64,"), "must be a PNG data-URI"
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert raw[:8] == _PNG_MAGIC, "decoded payload must be a real PNG"

    def test_nodata_is_transparent(self):
        """A masked / NaN cell becomes fully transparent in the encoded RGBA PNG.

        Decoded via matplotlib's PNG reader (a core dep) so the test needs no extra image library.
        """
        from matplotlib import image as mpimage

        import io

        arr = np.ma.masked_array(
            np.arange(12.0).reshape(3, 4), mask=np.zeros((3, 4), dtype=bool)
        )
        arr.mask[0, 0] = True
        uri = WebMap()._rgba_png_datauri(arr, "viridis")
        rgba = mpimage.imread(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        assert rgba[0, 0, 3] == 0.0, "masked cell must be transparent"
        assert rgba[2, 3, 3] > 0.0, "a valid cell must be opaque"

    def test_all_nonfinite_raises(self):
        with pytest.raises(ValueError, match="no finite values"):
            WebMap()._rgba_png_datauri(np.full((2, 2), np.nan), "viridis")

    def test_constant_band_does_not_crash(self):
        uri = WebMap()._rgba_png_datauri(np.full((2, 2), 5.0), "viridis")
        assert uri.startswith("data:image/png;base64,")


class TestImageCoordinates:
    """``_image_coordinates`` returns the MapLibre image corners ``[TL, TR, BR, BL]`` in ``[lng, lat]``."""

    def test_corner_order(self):
        x = np.array([10.0, 11.0, 12.0])
        y = np.array([50.0, 51.0, 52.0])
        corners = WebMap()._image_coordinates(x, y)
        assert corners == [[10.0, 52.0], [12.0, 52.0], [12.0, 50.0], [10.0, 50.0]]


class TestAddRasterNeedsEngine:
    """``add_raster`` reprojects through pyramids and builds an image source (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_add_raster_registers_layer_and_renders(self, dataset):
        from maplibre.ipywidget import MapWidget

        m = WebMap().add_raster(dataset)
        assert len(m.layers) == 1, "add_raster should register one layer"
        assert m._last_layer_id is not None
        assert isinstance(m.render(), MapWidget)

    def test_add_raster_then_basemap_saves(self, tmp_path, dataset):
        out = tmp_path / "raster.html"
        WebMap().add_raster(dataset, opacity=0.7).basemap("CartoLight").save(str(out))
        assert out.stat().st_size > 1_000
