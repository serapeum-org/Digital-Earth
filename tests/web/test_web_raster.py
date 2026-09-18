"""DW.1 — web-tier raster builder (``add_raster``) + basemap/tiles encoding helpers.

The PNG/coordinate encoding lives in pure helpers (``_rgba_png_datauri`` / ``_image_coordinates``) tested
without the engine (numpy/matplotlib are core deps via cleopatra). ``add_raster`` itself ``importorskip``s
maplibre and exercises the reproject → image-source → render → save path on the shared ``dataset`` fixture.
"""

import base64

import numpy as np
import pytest

from digitalearth.base.crs import OffLimbError
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
        import io

        from matplotlib import image as mpimage

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
        """The corners run top-left, top-right, bottom-right, bottom-left, around the cells.

        Test scenario:
            Since #301 the corners are the edges of the rectangle the cells cover, not their centres: three
            one-degree cells centred at 10/11/12 span 9.5 to 12.5, so an image source is no longer drawn half
            a cell inside its own data on every side.
        """
        x = np.array([10.0, 11.0, 12.0])
        y = np.array([50.0, 51.0, 52.0])
        corners = WebMap()._image_coordinates(x, y)
        assert corners == [[9.5, 52.5], [12.5, 52.5], [12.5, 49.5], [9.5, 49.5]], (
            corners
        )


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


def _source_with(y_values, values):
    """Build a stand-in display source exposing the coordinate and value arrays `add_raster` reads.

    Args:
        y_values: The y coordinates, ascending or descending.
        values: The 2-D band array, in the row order the source reports.

    Returns:
        An object with `.x.values`, `.y.values`, `.z.values` and the `metadata`/`units` accessors
        `auto_style` reads — a real `Source` has all five, and the builder now consults the style
        library for the band's units as well as its colormap.
    """
    axis = type("Axis", (), {})
    x, y, z = axis(), axis(), axis()
    x.values = np.array([0.0, 1.0])
    y.values = np.asarray(y_values, dtype=float)
    z.values = np.asarray(values, dtype=float)
    return type(
        "Source",
        (),
        {
            "x": x,
            "y": y,
            "z": z,
            "units": None,
            "metadata": staticmethod(lambda key=None, default=None: None),
        },
    )()


class TestAddRasterDrawsNorthFirst:
    """PNG row 0 is the northern edge, so a source whose rows run south-first has to be flipped.

    `rgb_composite` has this test; `add_raster` — the older and far more used builder — did not, and an
    upside-down overlay is silently plausible: it still lines up with its bounding box.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @pytest.mark.parametrize(
        "y_values, flipped", [([1.0, 0.0], False), ([0.0, 1.0], True)]
    )
    def test_rows_are_ordered_north_first(
        self, dataset, monkeypatch, y_values, flipped
    ):
        """An ascending y means row 0 is the south edge, which draws the image upside down.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to hand the builder a source with a chosen y order.
            y_values: The coordinate order the source reports.
            flipped: Whether the band should be reversed before encoding.

        Test scenario:
            The pixels handed to the encoder are compared against the orientation that *should* be drawn,
            so the test fails both when the flip is missing and when it happens for a north-up source.
        """
        band = np.array([[1.0, 2.0], [3.0, 4.0]])
        source = _source_with(y_values, band)
        monkeypatch.setattr(
            WebMap, "_to_display_source", lambda self, data, band=1: source
        )

        captured = {}
        encoder = WebMap._rgba_png_datauri

        def spy(values, cmap_name, vmin=None, vmax=None):
            """Record the array handed to the encoder, then encode it normally."""
            captured["values"] = np.array(values, copy=True)
            return encoder(values, cmap_name, vmin=vmin, vmax=vmax)

        monkeypatch.setattr(WebMap, "_rgba_png_datauri", staticmethod(spy))
        WebMap().basemap().add_raster(dataset, cmap="viridis")

        expected = band[::-1] if flipped else band
        assert np.allclose(captured["values"], expected), (
            f"rows are not north-first for y={y_values}"
        )


class TestARasterThatCannotBeGeoreferencedIsRefused:
    """A MapLibre image source is positioned by four lon/lat corners; there is no drawing it without them."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _call(m, builder, dataset):
        """Run `builder` on `m`, so the two parametrised cases read the same either side of the guard.

        Args:
            m: The map under test.
            builder: The raster builder name.
            dataset: The shared pyramids raster fixture.

        Returns:
            Whatever the builder returns (the map itself, since both are fluent).
        """
        return {
            "add_raster": lambda: m.add_raster(dataset),
            "rgb_composite": lambda: m.rgb_composite(dataset, bands=(1, 1, 1)),
        }[builder]()

    @pytest.mark.parametrize("builder", ["add_raster", "rgb_composite"])
    def test_unplaceable_corners_skip_the_layer_and_warn(
        self, dataset, monkeypatch, builder, warning_log
    ):
        """Falling back to the projected numbers would place the image somewhere off the planet.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher.
            builder: The raster builder under test.
            warning_log: The tier's loguru warnings, proving the skip is announced rather than silent.

        Test scenario:
            `_as_lonlat` answers `None` for a CRS that cannot be resolved or a reprojection that comes
            back non-finite; both builders must stop there — adding no layer, saying why, and leaving the
            rest of the chain to draw (C7).
        """
        monkeypatch.setattr(WebMap, "_as_lonlat", lambda self, *bounds: None)
        m = WebMap().basemap()
        before = len(m.layers)
        assert self._call(m, builder, dataset) is m, "the builder must stay chainable"
        assert len(m.layers) == before, "the unplaceable raster was added anyway"
        assert any("cannot be expressed in lon/lat" in line for line in warning_log), (
            f"the skip has to name why the layer is missing; got {warning_log!r}"
        )

    @pytest.mark.parametrize("builder", ["add_raster", "rgb_composite"])
    def test_strict_raises_instead_of_skipping(self, dataset, monkeypatch, builder):
        """`strict=True` is for a pipeline that must not publish a map with a layer quietly missing.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher.
            builder: The raster builder under test.
        """
        monkeypatch.setattr(WebMap, "_as_lonlat", lambda self, *bounds: None)
        m = WebMap(strict=True).basemap()
        # OffLimbError, not a bare ValueError: one exception type across the four tiers (M6).
        with pytest.raises(OffLimbError, match="cannot be expressed in lon/lat"):
            self._call(m, builder, dataset)
