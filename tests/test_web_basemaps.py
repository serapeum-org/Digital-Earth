"""Keyed-XYZ basemap dispatch on the web tier (#160).

Named ``test_web_*`` so the ``test-web`` pixi task — which globs ``tests/test_web_*.py`` and runs in the
``web`` env — actually picks it up. Filed under ``test_basemaps.py`` these were skipped in every CI job,
because no tier task globbed that name and the ``dev`` env has no maplibre.

Nothing here needs a real Planet key or the network.
"""

import pytest

FAKE_KEY = "FAKE-KEY-NOT-REAL"
#: The Amazon — inside the NICFI band. (west, south, east, north)
TROPICAL = (-60.0, -5.0, -55.0, 0.0)
#: The Netherlands — outside it.
TEMPERATE = (5.0, 52.0, 6.0, 53.0)


def _first_raster_source(web_map):
    """Return the source spec the map registered, by replaying its layers against a recorder.

    Args:
        web_map: A `WebMap` a basemap has been added to.

    Returns:
        The first registered source dict — the only way to see the spec without rendering a widget.
    """
    recorded = {}

    class Recorder:
        """Captures what a layer registers instead of building a MapLibre widget."""

        def add_source(self, src_id, source):
            """Record the source spec."""
            recorded[src_id] = source

        def add_layer(self, layer):
            """Ignore the layer; only the source is under test."""

    for apply in web_map.layers:
        apply(Recorder())
    assert recorded, "no source was registered"
    return next(iter(recorded.values()))


class TestWebTierDispatch:
    """``WebMap.basemap`` resolving a keyed preset (MapLibre)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the web extra, and supply a fake credential.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("maplibre")
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_the_preset_becomes_a_raster_source_carrying_the_key(self):
        """The web tier emits the tile URL into a MapLibre raster source.

        Test scenario:
            Replaying the registered layer against a recorder is the only way to see the source spec
            without rendering a widget.
        """
        from digitalearth.web import WebMap

        recorded = {}

        class Recorder:
            def add_source(self, src_id, source):
                recorded[src_id] = source

            def add_layer(self, layer):
                pass

        m = WebMap().basemap("Planet.NICFI", date="2024-01")
        for apply in m.layers:
            apply(Recorder())
        assert recorded, "no source was registered"
        source = next(iter(recorded.values()))
        assert source["type"] == "raster", (
            f"expected a raster source, got {source['type']!r}"
        )
        assert FAKE_KEY in source["tiles"][0], "the credential never reached the source"
        assert "Planet Labs" in source["attribution"], "the attribution was not carried"

    def test_preset_keywords_on_an_ordinary_provider_are_refused(self):
        """``date=`` means nothing to CartoDark, so it is an error rather than a silent no-op.

        Test scenario:
            Silently ignoring it would render the wrong basemap and look like the preset failed.
        """
        from digitalearth.web import WebMap

        with pytest.raises(ValueError, match="no preset keywords"):
            WebMap().basemap("CartoDark", date="2024-01")

    def test_the_unknown_provider_error_lists_the_presets(self):
        """A caller who mistypes a preset should see the presets among the options.

        Test scenario:
            The message previously listed only the four token-free names.
        """
        from digitalearth.web import WebMap

        with pytest.raises(ValueError, match=r"Planet\.NICFI"):
            WebMap().basemap("NotARealBasemap")

    def test_ordinary_providers_still_work(self):
        """The four token-free basemaps are unaffected by the dispatch."""
        from digitalearth.web import WebMap

        assert WebMap().basemap("CartoDark").layers, (
            "the ordinary basemap path stopped registering layers"
        )


class TestPresetKeywordErrors:
    """M6: a typo'd style keyword must not be blamed on the preset machinery."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the web extra, and supply a fake credential.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("maplibre")
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_a_misspelled_style_keyword_is_a_type_error(self):
        """``opacty=`` is a typo for ``opacity``, not an attempt to use a preset.

        Test scenario:
            ``**preset`` swallows every unknown keyword, so before this the caller was told their
            "preset keywords" were wrong — about a preset they never mentioned.
        """
        from digitalearth.web import WebMap

        with pytest.raises(TypeError, match="opacty"):
            WebMap().basemap("CartoDark", opacty=0.5)

    def test_a_real_preset_keyword_on_an_ordinary_provider_still_says_so(self):
        """``date=`` *is* a preset keyword, so that message remains the right one."""
        from digitalearth.web import WebMap

        with pytest.raises(ValueError, match="no preset keywords"):
            WebMap().basemap("CartoDark", date="2024-01")


class TestSavedOutputCarriesTheKey:
    """H1: the exposure the module docstring warns about is pinned here, not left to be discovered."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the web extra, and supply a fake credential.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("maplibre")
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_the_html_contains_the_credential(self):
        """A saved web map is a secret, because the browser needs the key to fetch the tiles.

        Test scenario:
            This is how an XYZ service authenticates and cannot be avoided, so it is asserted rather
            than guarded — if a future change ever strips the key, the basemap silently stops loading
            and this test says why. The module docstring documents the same thing in prose.
        """
        from digitalearth.web import WebMap

        html = WebMap().basemap("Planet.NICFI", date="2024-01").to_html()
        assert FAKE_KEY in html, (
            "the key is gone from the saved page, so its tiles cannot load"
        )


class TestACredentialWithNothingToAuthenticate:
    """L6: the web tier hoisted api_key into its signature too, and dropped it just as silently."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the web extra, and supply a fake credential.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("maplibre")
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_an_api_key_on_a_token_free_provider_is_refused(self):
        """CartoDark needs no credential, so passing one is a misunderstanding worth reporting."""
        from digitalearth.web import WebMap

        with pytest.raises(ValueError, match="takes no api_key"):
            WebMap().basemap("CartoDark", api_key="x")

    def test_a_keyed_preset_still_takes_one(self):
        """The guard sits after the keyed branch returns, and must stay there."""
        from digitalearth.web import WebMap

        WebMap().basemap("Planet.NICFI", date="2024-01", api_key=FAKE_KEY)


class TestTheServiceZoomCeilingReachesMapLibre:
    """L3: `max_zoom` was inert everywhere; the web tier is the one that can act on it."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the web extra, and supply a fake credential.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("maplibre")
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_a_keyed_preset_bounds_the_raster_source(self):
        """Past the service's deepest level MapLibre should stretch tiles, not request 404s.

        Test scenario:
            Without ``maxzoom`` on the source, zooming past NICFI's ceiling asks Planet for levels it
            does not publish — a wall of failed requests, each carrying the credential.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap("Planet.NICFI", date="2024-01")
        source = _first_raster_source(m)
        assert source["maxzoom"] == 20, source

    def test_an_ordinary_provider_is_left_unbounded(self):
        """A token-free provider declares no ceiling, and inventing one would clamp a working basemap."""
        from digitalearth.web import WebMap

        source = _first_raster_source(WebMap().basemap("CartoDark"))
        assert "maxzoom" not in source, source
