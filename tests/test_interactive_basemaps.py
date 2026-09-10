"""Keyed-XYZ basemap dispatch on the interactive tier (#160).

Named ``test_interactive_*`` so the ``test-interactive`` pixi task — which globs
``tests/test_interactive_*.py`` and runs in the ``interactive`` env — actually picks it up.

Nothing here needs a real Planet key or the network.
"""

import pytest

FAKE_KEY = "FAKE-KEY-NOT-REAL"


@pytest.fixture(autouse=True)
def _need_engine(monkeypatch):
    """Skip the module without the interactive extra, and give every test a credential that is not a real key.

    Args:
        monkeypatch: pytest's environment patcher, which restores the environment afterwards.
    """
    pytest.importorskip("geoviews")
    monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)


class TestInteractiveTierDispatch:
    """``InteractiveMap.tiles`` resolving a keyed preset (HoloViz / GeoViews)."""

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

        interactive_map = InteractiveMap()
        with pytest.raises(ValueError, match="no preset keywords"):
            interactive_map.tiles("CartoLight", preset={"date": "2024-01"})

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
    """NICFI is non-commercial-only, so its attribution is a licence obligation, not decoration."""

    @staticmethod
    def _rendered(element):
        """Render an element and return its figure title and every tile attribution in it.

        Args:
            element: A HoloViews/GeoViews element or overlay.

        Returns:
            ``(title, attributions)`` as Bokeh built them — what a viewer actually gets.
        """
        import holoviews as hv

        figure = hv.render(element)
        attributions = [
            renderer.tile_source.attribution
            for renderer in figure.renderers
            if getattr(renderer, "tile_source", None) is not None
        ]
        return figure.title.text, attributions

    def test_the_rendered_tiles_carry_the_attribution(self):
        """Bokeh's slot for this is the tile source, which is what its attribution control shows.

        Test scenario:
            Setting the element ``label`` instead put the text in the plot *title*, which is not an
            attribution and is not where anyone looks for one.
        """
        from digitalearth.interactive import InteractiveMap

        m = InteractiveMap().tiles("Planet.NICFI", preset={"date": "2024-01"})
        _, attributions = self._rendered(m.layers[0])
        assert attributions, "the rendered tiles carry no attribution at all"
        assert "Planet Labs" in attributions[0], attributions[0]
        assert "non-commercial" in attributions[0], "the licence note was dropped"

    def test_it_survives_an_overlay_with_data(self):
        """Tiles are only ever useful underneath data, so that is the case that has to work.

        Test scenario:
            An element label reaches the title only when the element *is* the plot; overlay it and the
            title comes from the Overlay, so the attribution was displayed nowhere at all.
        """
        import holoviews as hv

        from digitalearth.interactive import InteractiveMap

        m = InteractiveMap().tiles("Planet.NICFI", preset={"date": "2024-01"})
        _, attributions = self._rendered(m.layers[0] * hv.Points([(0.0, 0.0)]))
        assert any("Planet Labs" in text for text in attributions), attributions

    def test_the_plot_title_is_left_alone(self):
        """The title belongs to the caller; writing the attribution there silently overrode it."""
        from digitalearth.interactive import InteractiveMap

        m = InteractiveMap().tiles("Planet.NICFI", preset={"date": "2024-01"})
        title, _ = self._rendered(m.layers[0])
        assert title == "", f"the attribution hijacked the title: {title!r}"
        assert m.layers[0].label == "", (
            "the element key was changed, so .opts/.select stop matching"
        )

    def test_the_credential_is_not_in_the_attribution(self):
        """The attribution is displayed to every viewer; the key must not travel with it."""
        from digitalearth.interactive import InteractiveMap

        m = InteractiveMap().tiles("Planet.NICFI", preset={"date": "2024-01"})
        _, attributions = self._rendered(m.layers[0])
        assert FAKE_KEY not in "".join(attributions), (
            "the credential leaked into the attribution"
        )


class TestTheAttributionHookLeavesOtherRenderersAlone:
    """A figure holds the data renderers too, and the hook must only touch the tile sources."""

    def test_only_tile_renderers_are_written_to(self):
        """The hook walks every renderer in the plot, so it has to recognise which ones are tiles.

        Test scenario:
            A glyph renderer has no ``tile_source``; assigning an attribution to one would either fail
            or invent an attribute on someone else's object.
        """
        from digitalearth.interactive.decoration import _attribution_hook

        class TileRenderer:
            """Stands in for a Bokeh tile renderer."""

            def __init__(self):
                self.tile_source = type("Source", (), {"attribution": ""})()

        class GlyphRenderer:
            """Stands in for a Bokeh glyph renderer, which has no tile source."""

        tiles, glyphs = TileRenderer(), GlyphRenderer()
        plot = type(
            "Plot",
            (),
            {"handles": {"plot": type("Fig", (), {"renderers": [glyphs, tiles]})()}},
        )()
        _attribution_hook("ATTR")(plot, None)

        assert tiles.tile_source.attribution == "ATTR", (
            "the tile source was not written to"
        )
        assert not hasattr(glyphs, "tile_source"), (
            "the hook invented an attribute on a glyph renderer"
        )
