"""Keyed-XYZ basemap dispatch on the interactive tier (#160).

Named ``test_interactive_*`` so the ``test-interactive`` pixi task — which globs
``tests/test_interactive_*.py`` and runs in the ``interactive`` env — actually picks it up.

Nothing here needs a real Planet key or the network.
"""

import pytest

FAKE_KEY = "FAKE-KEY-NOT-REAL"
#: The Amazon — inside the NICFI band. (west, south, east, north)
TROPICAL = (-60.0, -5.0, -55.0, 0.0)
#: The Netherlands — outside it.
TEMPERATE = (5.0, 52.0, 6.0, 53.0)


class TestInteractiveTierDispatch:
    """``InteractiveMap.tiles`` resolving a keyed preset (HoloViz / GeoViews)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the interactive extra, and supply a fake credential.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("geoviews")
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_the_preset_becomes_a_wmts_with_upper_cased_placeholders(self):
        """GeoViews wants ``{Z}/{X}/{Y}``, so the lower-case template is converted on the way in.

        Test scenario:
            Handing GeoViews the lower-case form yields a WMTS that requests literal ``{z}`` tiles.
        """
        from digitalearth.interactive import InteractiveMap

        m = InteractiveMap().tiles("Planet.NICFI", preset={"date": "2024-01"})
        url = m.layers[0].data
        assert "{Z}/{X}/{Y}" in url, f"placeholders were not upper-cased: {url}"
        assert FAKE_KEY in url, "the credential never reached the element"
        assert "planet_medres_normalized_analytic_2024-01_mosaic" in url, (
            f"wrong mosaic: {url}"
        )

    def test_preset_keywords_without_a_preset_provider_are_refused(self):
        """``preset=`` means nothing to a catalog provider, so it is an error rather than ignored.

        Test scenario:
            Silently dropping it would render CartoLight while the caller believed they got NICFI.
        """
        from digitalearth.interactive import InteractiveMap

        with pytest.raises(ValueError, match="no preset keywords"):
            InteractiveMap().tiles("CartoLight", preset={"date": "2024-01"})

    def test_ordinary_providers_are_unaffected(self):
        """A catalog name still resolves through the GeoViews tile sources as before."""
        from digitalearth.interactive import InteractiveMap

        assert InteractiveMap().tiles("CartoLight").layers, (
            "the ordinary tile path stopped working"
        )


class TestUpperPlaceholders:
    """The Bokeh placeholder casing, which lives in this backend rather than in ``base/``."""

    def test_every_copy_of_the_three_tokens_is_upper_cased(self):
        """GeoViews substitutes ``{Z}/{X}/{Y}`` case-sensitively, so each copy has to be rewritten.

        Test scenario:
            The path placeholders and a second ``{z}`` in the query string both change; the credential
            and the other query parameters do not. The name previously claimed only the tile-coordinate
            *path* was touched, which the second ``{z}`` contradicts.
        """
        from digitalearth.interactive.decoration import _upper_placeholders

        out = _upper_placeholders("https://a/{z}/{x}/{y}.png?api_key=abc&m={z}x")
        assert out == "https://a/{Z}/{X}/{Y}.png?api_key=abc&m={Z}x", out

    def test_no_other_placeholder_is_touched(self):
        """``{mosaic}`` and ``{api_key}`` are the tile service's, not Bokeh's, and must survive as they are."""
        from digitalearth.interactive.decoration import _upper_placeholders

        out = _upper_placeholders("https://a/{mosaic}/{z}/{x}/{y}?k={api_key}")
        assert out == "https://a/{mosaic}/{Z}/{X}/{Y}?k={api_key}", out


class TestAttributionIsCarried:
    """M7: NICFI is non-commercial-only, so its attribution is a licence obligation."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the interactive extra, and supply a fake credential.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("geoviews")
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_the_element_carries_the_attribution(self):
        """The other two tiers pass it to their engine; this one dropped it entirely.

        Test scenario:
            GeoViews surfaces the element label, which is where the attribution can be seen.
        """
        from digitalearth.interactive import InteractiveMap

        m = InteractiveMap().tiles("Planet.NICFI", preset={"date": "2024-01"})
        assert "Planet Labs" in m.layers[0].label, (
            f"attribution missing: {m.layers[0].label!r}"
        )
        assert "non-commercial" in m.layers[0].label, "the licence note was dropped"
