"""The standalone-export document (WB-13/17/20): structure of the HTML our own runtime emits.

These are the no-browser checks — that :class:`~digitalearth.web.htmldoc.HtmlDocument` and
``swipe_html`` emit the containers, the CDN library, our runtime/feature scripts and the bootstrap,
and that each map's serialized state is embedded. The *behaviour* of the page in a browser (the maps
render, the divider clips) is covered by the Playwright smoke test (``test_web_export_smoke.py``),
which runs in the browser-smoke CI job.
"""

import pytest

from digitalearth.web import WebMap, save_swipe, swipe_html
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
