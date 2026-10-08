"""The standalone-export document (WB-13/17/20): structure of the HTML our own runtime emits.

These are the no-browser checks — that :class:`~digitalearth.web.htmldoc.HtmlDocument` and
``swipe_html`` emit the containers, the CDN library, our runtime/feature scripts and the bootstrap,
and that each map's serialized state is embedded. The *behaviour* of the page in a browser (the maps
render, the divider clips) is covered by the Playwright smoke test (``test_web_export_smoke.py``),
which runs in the browser-smoke CI job.
"""

import pytest

from digitalearth.web import (
    WebMap,
    measure_html,
    minimap_html,
    save_measure,
    save_minimap,
    save_swipe,
    swipe_html,
)
from digitalearth.web.htmldoc import HtmlDocument, MapPanel


@pytest.fixture
def two_maps():
    """A before map (a basemap) and an after map (an xyz tile layer), each with a navigation control."""
    pytest.importorskip("maplibre")
    before = WebMap().basemap().navigation()
    after = WebMap().tiles("https://example.com/{z}/{x}/{y}.png").navigation()
    return before, after


class TestHtmlDocumentSwipe:
    """``HtmlDocument.swipe`` describes a two-map swipe page."""

    def test_builds_two_panels_in_the_swipe_containers(self):
        """The document carries one panel per map, keyed to the swipe's two container ids.

        Test scenario:
            The runtime builds a map into each container by id, so the panels must name exactly the
            before/after containers the body markup declares.
        """
        doc = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": []}, {"mapOptions": {}, "calls": []}
        )
        ids = [panel.container_id for panel in doc.panels]
        assert ids == ["de-map-before", "de-map-after"], ids
        assert all(isinstance(p, MapPanel) for p in doc.panels), doc.panels
        assert doc.feature_js, "a swipe needs the cross-map feature script"

    def test_render_has_both_containers_and_the_wrapper(self):
        """The page declares the positioned wrapper and the two overlaid map divs.

        Test scenario:
            The feature overlays the two maps in one wrapper and clips the second, so all three
            elements must be present with the ids the bootstrap and the feature reference.
        """
        html = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": []}, {"mapOptions": {}, "calls": []}
        ).render()
        for needle in (
            'id="de-swipe"',
            "de-swipe-wrap",
            'id="de-map-before"',
            'id="de-map-after"',
        ):
            assert needle in html, needle

    def test_render_loads_maplibre_and_ships_runtime_and_feature(self):
        """The page loads pinned maplibre-gl from the CDN and ships our runtime + feature scripts.

        Test scenario:
            The own-export page is self-contained apart from the CDN library: the runtime
            (``DE.buildMap``, the reimplemented ``DELayerSwitcher``) and the swipe feature
            (``DE.swipe``) are inlined, and the bootstrap builds both maps before starting the swipe.
        """
        html = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": []}, {"mapOptions": {}, "calls": []}
        ).render()
        assert "maplibre-gl@5.3.0/dist/maplibre-gl.js" in html, (
            "maplibre-gl JS not loaded"
        )
        assert "maplibre-gl@5.3.0/dist/maplibre-gl.css" in html, (
            "maplibre-gl CSS not loaded"
        )
        assert "DE.buildMap" in html, "runtime bootstrap missing"
        assert "DELayerSwitcher" in html, "runtime control missing"
        assert "DE.swipe" in html, "swipe feature missing"
        assert html.count("window.DE.buildMap(") == 2, "both maps must be built"

    def test_a_plain_document_omits_the_feature_script(self):
        """A document with no cross-map feature ships the runtime but not ``de_features.js``.

        Test scenario:
            ``feature_js`` gates the feature script, so a single-map page is not padded with the
            swipe/minimap/measure code it does not use.
        """
        doc = HtmlDocument(
            panels=(MapPanel("de-map", {"mapOptions": {}, "calls": []}),),
            body_html='<div id="de-map"></div>',
            bootstrap_js='window.DE.buildMap("de-map",{"mapOptions":{},"calls":[]});',
            feature_js=False,
        )
        html = doc.render()
        assert "DE.buildMap" in html, "the runtime is always needed"
        assert "DE.swipe" not in html, "no feature script for a plain document"


class TestSwipeHtml:
    """``swipe_html`` serialises two real maps into a swipe page."""

    def test_embeds_each_map_state(self, two_maps):
        """Each map's serialized layers reach the page, so the two sides draw different content.

        Args:
            two_maps: The before (basemap) and after (xyz tiles) maps.

        Test scenario:
            The before map adds a carto basemap source and the after map an xyz tile source; both
            source ids must appear in the embedded ``to_dict()`` state, one per map.
        """
        before, after = two_maps
        html = swipe_html(before, after)
        assert "tiles-1-src" in html, "the basemap source did not reach the page"
        assert "example.com" in html, "the after map's xyz tiles did not reach the page"

    def test_title_is_applied(self, two_maps):
        """The document title is the caller's.

        Args:
            two_maps: The before and after maps.
        """
        before, after = two_maps
        assert "<title>Before / After</title>" in swipe_html(
            before, after, title="Before / After"
        )

    def test_height_sizes_the_wrapper(self, two_maps):
        """``height`` sets the wrapper's CSS height, so the maps are the requested size.

        Args:
            two_maps: The before and after maps.
        """
        before, after = two_maps
        assert "height:480px" in swipe_html(before, after, height=480)


class TestSaveSwipe:
    """``save_swipe`` writes the swipe page to disk."""

    def test_writes_a_nontrivial_html_file(self, two_maps, tmp_path):
        """The written file is a real HTML document, not an empty shell.

        Args:
            two_maps: The before and after maps.
            tmp_path: pytest's per-test directory.
        """
        before, after = two_maps
        out = save_swipe(before, after, str(tmp_path / "swipe.html"))
        assert out.exists(), out
        text = out.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>"), "not an HTML document"
        assert "de-swipe-wrap" in text, "swipe wrapper missing"
        assert out.stat().st_size > 2_000, "file looks empty"


class TestHtmlDocumentMinimap:
    """``HtmlDocument.minimap`` describes a main map with an overview inset."""

    def test_builds_the_main_and_overview_panels(self):
        """The document carries a main panel and an overview panel in the minimap containers.

        Test scenario:
            The runtime builds the main map and the inset by id and the feature syncs the inset to
            the main, so the panels must name exactly the main/overview containers.
        """
        doc = HtmlDocument.minimap(
            {"mapOptions": {}, "calls": []}, {"mapOptions": {}, "calls": []}
        )
        assert [p.container_id for p in doc.panels] == ["de-map-main", "de-minimap"]
        assert doc.feature_js, "a minimap needs the cross-map feature script"

    def test_render_sizes_the_inset_and_wires_the_feature(self):
        """The inset is sized and the bootstrap starts the minimap over both built maps.

        Test scenario:
            The overview is a fixed-size inset the feature follows; both maps are built, then
            ``DE.minimap`` is started once they are ready.
        """
        html = HtmlDocument.minimap(
            {"mapOptions": {}, "calls": []},
            {"mapOptions": {}, "calls": []},
            mini_size=(240, 160),
        ).render()
        assert 'id="de-minimap"' in html
        assert "width:240px;height:160px" in html
        assert "DE.minimap" in html, "minimap feature missing"
        assert html.count("window.DE.buildMap(") == 2, "both maps must be built"


class TestMinimapHtml:
    """``minimap_html`` renders a main map with a default or supplied overview."""

    def test_default_overview_is_a_basemap(self):
        """With no overview given, the inset is a plain basemap, so the locator always has ground.

        Test scenario:
            A caller who just wants "a minimap" should get a usable overview without building one;
            the default basemap's source reaches the inset panel.
        """
        pytest.importorskip("maplibre")
        html = minimap_html(WebMap().basemap())
        # Two basemap sources appear: the main map's and the overview's.
        assert html.count("basemaps.cartocdn.com") >= 2, (
            "the overview basemap is missing"
        )

    def test_supplied_overview_is_used(self):
        """A caller's own overview map drives the inset.

        Test scenario:
            The overview is an ordinary map, so a caller can give it its own basemap/tiles; the
            supplied tiles reach the page.
        """
        pytest.importorskip("maplibre")
        overview = WebMap().tiles("https://inset.example/{z}/{x}/{y}.png")
        assert "inset.example" in minimap_html(WebMap().basemap(), overview=overview)

    def test_save_minimap_writes_a_file(self, tmp_path):
        """``save_minimap`` writes a real HTML document.

        Args:
            tmp_path: pytest's per-test directory.
        """
        pytest.importorskip("maplibre")
        out = save_minimap(WebMap().basemap(), str(tmp_path / "mini.html"))
        text = out.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>")
        assert "de-minimap-wrap" in text


class TestHtmlDocumentMeasure:
    """``HtmlDocument.measure`` describes a map with a live measure readout."""

    def test_one_panel_a_readout_and_draw_assets(self):
        """The page has the map, the readout element, the draw library, and starts ``DE.measure``.

        Test scenario:
            The readout needs the draw control and the live map, so the document carries one map
            panel, the readout div, mapbox-gl-draw (loaded only here, via ``needs_draw``), and the
            bootstrap that starts the measure feature once the map is ready.
        """
        doc = HtmlDocument.measure({"mapOptions": {}, "calls": []})
        assert [p.container_id for p in doc.panels] == ["de-map-measure"]
        assert doc.needs_draw, "measure needs mapbox-gl-draw on the page"
        html = doc.render()
        assert 'id="de-measure-readout"' in html, "readout element missing"
        assert "dist/mapbox-gl-draw" in html, "draw library not loaded"
        assert "DE.measure" in html, "measure feature not started"

    def test_a_non_measure_document_does_not_load_draw(self):
        """mapbox-gl-draw is loaded only for a measure page, not for swipe/minimap.

        Test scenario:
            ``needs_draw`` gates the draw library, so a swipe page is not padded with a library it
            does not use.
        """
        html = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": []}, {"mapOptions": {}, "calls": []}
        ).render()
        # The shared runtime mentions the library by name in the measure code; what must be absent
        # from a swipe page is the CDN library asset itself.
        assert "dist/mapbox-gl-draw" not in html, "swipe must not load the draw library"

    def test_tools_follow_the_distance_and_area_flags(self):
        """``distance``/``area`` select which tools the readout offers.

        Test scenario:
            A caller measuring only distance should get only the line tool in the started feature's
            options, so the flag reaches the bootstrap.
        """
        html = HtmlDocument.measure(
            {"mapOptions": {}, "calls": []}, distance=True, area=False
        ).render()
        assert '"distance": true' in html
        assert '"area": false' in html


class TestMeasureHtml:
    """``measure_html`` renders a live-measure page from a plain map."""

    def test_embeds_the_map_and_offers_the_readout(self):
        """The map's state reaches the page and the readout/draw are present.

        Test scenario:
            The draw control is added by the page's runtime, so a plain map (no ``measure()`` call)
            is enough; its basemap source and the readout both appear.
        """
        pytest.importorskip("maplibre")
        html = measure_html(WebMap().basemap())
        assert "basemaps.cartocdn.com" in html, "the map did not reach the page"
        assert "de-measure-readout" in html
        assert "DE.measure" in html

    def test_save_measure_writes_a_file(self, tmp_path):
        """``save_measure`` writes a real HTML document.

        Args:
            tmp_path: pytest's per-test directory.
        """
        pytest.importorskip("maplibre")
        out = save_measure(WebMap().basemap(), str(tmp_path / "measure.html"))
        text = out.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>")
        assert "de-measure-wrap" in text

    def test_a_map_that_already_has_a_draw_control_is_refused(self):
        """A map that already called ``measure()`` is refused, so the page never gets two draw controls.

        Test scenario:
            The live readout adds its own draw control; if the map was built with ``measure()`` its
            recorded ``addMapboxDraw`` would be replayed too, leaving two controls. The map must be
            passed plain, and a map carrying ``addMapboxDraw`` is refused by name.
        """
        with pytest.raises(ValueError, match="draw control"):
            HtmlDocument.measure(
                {"mapOptions": {}, "calls": [["addMapboxDraw", [{}]]]}
            )


class TestConditionalLibraries:
    """The CDN libraries (deck.gl, mapbox-gl-draw) load only when the maps' recorded calls need them."""

    def test_deck_loads_only_when_a_panel_has_a_deck_overlay(self):
        """deck.gl is loaded when a map records a deck overlay, and omitted otherwise.

        Test scenario:
            A map drawing a big-data layer records ``addDeckOverlay``; the page must load deck.gl so
            the runtime can build the overlay. A page whose maps have no deck overlay must not pull
            the (large) deck.gl bundle.
        """
        with_deck = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": [["addDeckOverlay", [[]]]]},
            {"mapOptions": {}, "calls": []},
        )
        assert "deck.gl@" in with_deck.render(), "deck.gl must load for a deck overlay"
        plain = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": []}, {"mapOptions": {}, "calls": []}
        )
        assert "deck.gl@" not in plain.render(), "no deck overlay must not load deck.gl"

    def test_draw_loads_when_a_panel_records_a_draw_control(self):
        """mapbox-gl-draw is loaded when a map records ``addMapboxDraw`` (e.g. a ``measure()`` map).

        Test scenario:
            A map carrying a draw control must have the draw library on the page so the runtime can
            rebuild it; detection is off the recorded call, not only the measure document flag.
        """
        doc = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": [["addMapboxDraw", [{}]]]},
            {"mapOptions": {}, "calls": []},
        )
        assert "dist/mapbox-gl-draw" in doc.render(), (
            "draw library must load for a draw control"
        )

    def test_a_plain_swipe_loads_neither_deck_nor_draw(self):
        """A swipe of plain maps pulls only maplibre-gl, not deck.gl or mapbox-gl-draw.

        Test scenario:
            Loading libraries a page does not use would bloat it; the plain case must stay lean.
        """
        html = HtmlDocument.swipe(
            {"mapOptions": {}, "calls": [["addSource", ["s", {}]]]},
            {"mapOptions": {}, "calls": []},
        ).render()
        assert "deck.gl@" not in html
        assert "dist/mapbox-gl-draw" not in html


class TestScriptSafety:
    """Embedded map state must not break out of the ``<script>`` block (H1)."""

    def test_a_closing_script_tag_in_data_is_escaped(self):
        """A ``</script>`` in a serialized string is escaped, so it cannot break out or inject markup.

        Test scenario:
            A source ``attribution`` of ``x</script><img onerror=...>`` would, if embedded raw,
            terminate the script and turn the rest into live HTML (a page-break on benign data, an
            XSS vector on third-party data). The rendered page must not contain the raw breakout, and
            the data must survive escaped.
        """
        evil = {
            "mapOptions": {},
            "calls": [
                ["addSource", ["s", {"attribution": "x</script><img src=q onerror=alert(1)>"}]]
            ],
        }
        html = HtmlDocument.swipe(evil, {"mapOptions": {}, "calls": []}).render()
        assert "</script><img" not in html, "data broke out of the <script> block"
        assert "\\u003c/script\\u003e" in html, "the </script> in data was not escaped"

    def test_angle_brackets_and_ampersand_in_data_are_escaped(self):
        """``<``/``>``/``&`` in embedded data are unicode-escaped, never emitted raw into the script.

        Test scenario:
            Escaping only ``</`` would still let a lone ``<script>`` or an HTML comment inside data
            confuse a parser; neutralising the angle brackets and ampersand closes that class.
        """
        data = {"mapOptions": {}, "calls": [["addSource", ["s", {"note": "a<b>c&d"}]]]}
        html = HtmlDocument.measure(data).render()
        assert "a<b>c&d" not in html, "raw angle brackets/ampersand embedded"
        assert "a\\u003cb\\u003ec\\u0026d" in html, "data not escaped as expected"


class TestOfflineInlining:
    """``offline=True`` routes each save through the CDN-asset inliner."""

    @pytest.mark.parametrize(
        "saver, extra",
        [
            ("save_swipe", True),
            ("save_minimap", False),
            ("save_measure", False),
        ],
    )
    def test_offline_inlines_the_assets(self, saver, extra, tmp_path, monkeypatch):
        """With ``offline=True`` the written page is the inlined HTML, not the CDN-referencing one.

        Args:
            saver: The save function under test, by name.
            extra: Whether the function takes a second map positionally (``save_swipe``).
            tmp_path: pytest's per-test directory.
            monkeypatch: Replaces the network inliner with a marker, so the branch is covered offline.

        Test scenario:
            The inliner fetches CDN assets, which needs a network; stubbing it proves the ``offline``
            branch routes the rendered page through it and writes the result, without a live fetch.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import export as web_export

        monkeypatch.setattr(
            web_export.ExportMixin,
            "_inline_offline_assets",
            staticmethod(lambda html: "<!-- INLINED -->" + html),
        )
        func = getattr(web_export, saver)
        out = tmp_path / "offline.html"
        base = WebMap().basemap()
        if extra:
            func(base, WebMap().basemap(), str(out), offline=True)
        else:
            func(base, str(out), offline=True)
        assert out.read_text(encoding="utf-8").startswith("<!-- INLINED -->"), (
            f"{saver}(offline=True) did not inline the assets"
        )
