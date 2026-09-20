"""RGB composites on the web tier (#190).

The encoder is a pure helper, tested without the engine (numpy/matplotlib are core deps via cleopatra);
``rgb_composite`` itself ``importorskip``s maplibre and runs on the shared ``dataset`` fixture. The stretch
is asserted to be the *same* one the static tier applies, which is the point of sharing
``digitalearth.base.stretch``.
"""

import base64
import io

import numpy as np
import pytest
from matplotlib import image as mpimage

from digitalearth.web import WebMap

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _decode(uri):
    """Decode a PNG data-URI into an RGBA array.

    Args:
        uri: A ``data:image/png;base64,…`` string.

    Returns:
        The decoded image as a float array in ``[0, 1]``.
    """
    assert uri.startswith("data:image/png;base64,"), uri[:40]
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == _PNG_MAGIC, "decoded payload must be a real PNG"
    return mpimage.imread(io.BytesIO(raw))


class TestTheCompositeEncoder:
    """Three stretched channels become one RGBA PNG, with NoData transparent."""

    def test_a_stretched_stack_becomes_a_png(self):
        """The encoder is what puts the composite on the map, so it must produce a real PNG."""
        stack = np.random.default_rng(1337).random((4, 5, 3))
        assert _decode(WebMap()._composite_png_datauri(stack)).shape == (4, 5, 4)

    def test_channel_order_is_preserved(self):
        """R, G and B are not interchangeable — a swap makes a false-colour image silently wrong."""
        stack = np.zeros((1, 3, 3))
        stack[0, 0] = (1.0, 0.0, 0.0)
        stack[0, 1] = (0.0, 1.0, 0.0)
        stack[0, 2] = (0.0, 0.0, 1.0)
        image = _decode(WebMap()._composite_png_datauri(stack))
        assert image[0, 0, 0] == pytest.approx(1.0, abs=0.01), "red channel moved"
        assert image[0, 1, 1] == pytest.approx(1.0, abs=0.01), "green channel moved"
        assert image[0, 2, 2] == pytest.approx(1.0, abs=0.01), "blue channel moved"

    def test_nodata_in_any_channel_is_transparent(self):
        """A pixel missing one band has no colour, so drawing it would invent one."""
        stack = np.ones((1, 2, 3))
        stack[0, 1, 1] = np.nan
        image = _decode(WebMap()._composite_png_datauri(stack))
        assert image[0, 0, 3] == pytest.approx(1.0), "a complete pixel went transparent"
        assert image[0, 1, 3] == pytest.approx(0.0), "an incomplete pixel was drawn"

    def test_a_stack_with_no_complete_pixel_is_refused(self):
        """An all-transparent image looks like a rendering failure rather than an empty input."""
        web_map = WebMap()
        full = np.full((2, 2, 3), np.nan)
        with pytest.raises(ValueError, match="no pixel finite"):
            web_map._composite_png_datauri(full)

    def test_values_outside_the_unit_range_are_clipped(self):
        """`stretch_to_unit` can overshoot with explicit limits, and a PNG has no room for it."""
        stack = np.array([[[-0.5, 0.5, 1.5]]])
        image = _decode(WebMap()._composite_png_datauri(stack))
        assert image[0, 0, 0] == pytest.approx(0.0, abs=0.01)
        assert image[0, 0, 2] == pytest.approx(1.0, abs=0.01)


class TestTheCompositeOnTheMap:
    """The builder wires the encoder to a MapLibre image source."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_three_bands_are_required(self, dataset):
        """Two or four bands is a caller error the renderer would report far less clearly.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        web_map = WebMap()
        with pytest.raises(ValueError):
            web_map.rgb_composite(dataset, bands=(1, 1))

    def test_it_registers_an_image_layer(self, dataset):
        """A composite is an image source like the single-band path, not a new kind of layer.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        html = m.to_html()
        payload = html[html.rfind("var data = ") :]
        assert '"type": "image"' in payload
        assert "data:image/png;base64," in payload

    def test_it_frames_the_map_like_any_other_layer(self, dataset):
        """A composite is data, so a map showing only one should open on it.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        assert m._data_bounds is not None

    def test_it_is_an_addressable_layer(self, dataset):
        """A viewer should be able to switch imagery off to see what is underneath.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = (
            WebMap()
            .basemap()
            .rgb_composite(dataset, bands=(1, 1, 1), name="True colour")
        )
        assert m.layer_ids == ["True colour"], m.layer_ids
        html = m.layer_control().to_html()
        payload = html[html.rfind("var data = ") :]
        assert '"layerIds": ["True colour"]' in payload, (
            "the switcher would caption this row with a generated id"
        )

    def test_the_stretch_matches_the_static_tier(self, dataset):
        """Two tiers that disagree about the same composite would be worse than one lacking it.

        Test scenario:
            Both call ``digitalearth.base.stretch.stretch_to_unit`` over the same
            ``digitalearth.base.sources.get_stack`` output, so the encoded pixels must equal a stretch
            computed independently here.
        """
        from digitalearth.base.sources import get_stack
        from digitalearth.base.stretch import stretch_to_unit

        bands = (1, 1, 1)
        m = WebMap()
        expected = stretch_to_unit(
            get_stack(m._to_display_raster(dataset), bands, mask=True)
        )
        uri = m._composite_png_datauri(expected)
        m.basemap().rgb_composite(dataset, bands=bands)
        html = m.to_html()
        # The whole base64 payload, not a prefix: a PNG's first bytes are the header and the IHDR
        # dimensions, which two different images of the same size share.
        assert uri.split(",", 1)[1] in html, (
            "the drawn composite is not the shared stretch"
        )


def _source_with(y_values):
    """Build a stand-in source exposing only the coordinate arrays the raster builders read.

    Args:
        y_values: The y coordinates, ascending or descending.

    Returns:
        An object with ``.x.values`` and ``.y.values``.
    """
    import numpy as np

    axis = type("Axis", (), {})
    x, y = axis(), axis()
    x.values = np.array([0.0, 1.0])
    y.values = np.asarray(y_values, dtype=float)
    return type("Source", (), {"x": x, "y": y})()


class TestTheCompositeSizeWarning:
    """The inline data-URI path has a ceiling, and a caller past it needs to hear about it."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _capture():
        """Attach a loguru sink for warnings.

        Returns:
            ``(records, sink_id)`` — the list the sink fills, and the handle to remove it afterwards.
        """
        from loguru import logger

        records = []
        return records, logger.add(records.append, level="WARNING")

    def test_an_oversized_composite_warns(self, dataset, monkeypatch):
        """The warning is all that stands between a caller and a hundreds-of-MB page.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to lower the ceiling rather than build a huge raster.

        Test scenario:
            Three bands inline as three times the data of one, so the check is on the stack's size.
        """
        from loguru import logger

        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        records, sink_id = self._capture()
        try:
            WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        finally:
            logger.remove(sink_id)
        assert any("rgb_composite" in str(r) for r in records), records

    def test_a_small_composite_stays_quiet(self, dataset):
        """A warning on every composite would train the caller to ignore it.

        Args:
            dataset: The shared pyramids raster fixture, far below the ceiling.
        """
        from loguru import logger

        records, sink_id = self._capture()
        try:
            WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        finally:
            logger.remove(sink_id)
        assert not [r for r in records if "rgb_composite" in str(r)], records


class TestNorthUpOrientation:
    """PNG row 0 is the northern edge, so a source whose rows run south-first has to be flipped."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @pytest.mark.parametrize(
        "y_values, flipped",
        [([1.0, 0.0], False), ([0.0, 1.0], True)],
    )
    def test_rows_are_ordered_north_first(
        self, dataset, monkeypatch, y_values, flipped
    ):
        """An ascending y means row 0 is the south edge, which renders the image upside down.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to hand the builder a source with a chosen y order.
            y_values: The coordinate order the source reports.
            flipped: Whether the stack should be reversed before encoding.

        Test scenario:
            The encoded pixels are compared against the orientation that *should* be drawn, so the test
            fails both when the flip is missing and when it happens for a north-up source.
        """
        import numpy as np

        from digitalearth.base.sources import get_stack
        from digitalearth.base.stretch import stretch_to_unit

        source = _source_with(y_values)
        monkeypatch.setattr(
            WebMap, "_to_display_source", lambda self, data, band=1: source
        )

        captured = {}
        encoder = WebMap._composite_png_datauri

        def spy(unit_stack):
            """Record the stack handed to the encoder, then encode it normally."""
            captured["stack"] = np.array(unit_stack, copy=True)
            return encoder(unit_stack)

        monkeypatch.setattr(WebMap, "_composite_png_datauri", staticmethod(spy))

        m = WebMap()
        stack = stretch_to_unit(
            get_stack(m._to_display_raster(dataset), (1, 1, 1), mask=True)
        )
        m.basemap().rgb_composite(dataset, bands=(1, 1, 1))
        expected = stack[::-1] if flipped else stack
        assert np.allclose(captured["stack"], expected, equal_nan=True), (
            f"rows are not north-first for y={y_values}"
        )
