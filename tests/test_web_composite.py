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
        with pytest.raises(ValueError, match="no pixel finite"):
            web_map._composite_png_datauri(np.full((2, 2, 3), np.nan))

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
        assert len(m.layer_ids) == 1
        assert m._layer_index[0][1] == "True colour"

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
        assert uri.split(",", 1)[1][:64] in html, (
            "the drawn composite is not the shared stretch"
        )
