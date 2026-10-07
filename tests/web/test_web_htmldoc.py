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
    minimap_html,
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
        assert "DE.buildMap" in html and "DELayerSwitcher" in html, "runtime missing"
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
        assert "de-swipe-wrap" in text and out.stat().st_size > 2_000, "looks empty"


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
        assert 'id="de-minimap"' in html and "width:240px;height:160px" in html
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
        assert text.startswith("<!DOCTYPE html>") and "de-minimap-wrap" in text
