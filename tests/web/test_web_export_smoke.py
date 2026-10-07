"""Browser smoke test for the standalone export (WB-13 swipe).

The structure of the emitted page is checked without a browser in ``test_web_htmldoc.py``. This file
is the behaviour check: our own runtime (``de_maplibre.js``) and swipe feature (``de_features.js``)
actually build two maps and clip one in a real headless Chromium, with no uncaught JS error. It is
the swipe counterpart of ``test_web_deck_smoke.py`` and runs in the same browser-smoke CI job
(Playwright + Chromium are not in the ``[web]`` extra, so the file ``importorskip``s playwright and
skips cleanly where no browser is installed).
"""

import pathlib
import tempfile

import pytest

pytest.importorskip("playwright")
pytest.importorskip("maplibre")

from digitalearth.web import WebMap, swipe_html  # noqa: E402


def _render_and_probe(uri: str, probe_js: str):
    """Load ``uri`` in headless Chromium, run ``probe_js``, return ``(page_errors, console_errors, probe)``.

    Skips (not fails) when no Chromium binary is available, so the Playwright package without an
    installed browser does not produce a false failure.

    Args:
        uri: A ``file://`` URI of the page to load.
        probe_js: A zero-argument JS function (as a string) run after the maps render; its return
            value is handed back as ``probe``.

    Returns:
        ``(page_errors, console_errors, probe)`` — JS exceptions, console errors, and whatever
        ``probe_js`` returned.
    """
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    page_errors: list = []
    console_errors: list = []
    with sync_playwright() as play:
        try:
            browser = play.chromium.launch()
        except Exception as exc:  # browser not installed / cannot launch -> skip
            pytest.skip(f"Chromium not available for Playwright: {exc}")
        page = browser.new_page(viewport={"width": 900, "height": 600})
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on(
            "console",
            lambda msg: (
                console_errors.append(msg.text) if msg.type == "error" else None
            ),
        )
        page.goto(uri, wait_until="load")
        try:
            page.wait_for_selector("canvas", timeout=20000)
        except PWTimeout:
            pass
        page.wait_for_timeout(4000)  # let both maps finish their first render
        probe = page.evaluate(probe_js)
        browser.close()
    return page_errors, console_errors, probe


def _ignore_network_console(errors):
    """Drop console errors about fetching tiles/fonts — the smoke test is about our JS, not the CDN."""
    noise = ("tile", "font", "glyph", "sprite", "err_", "failed to load", "net::")
    return [e for e in errors if not any(n in e.lower() for n in noise)]


class TestSwipeRendersInABrowser:
    """WB-13 — the swipe page builds two maps and clips one, with no uncaught JS error."""

    def test_two_maps_render_and_the_divider_clips(self):
        """Both maps draw a canvas, the swipe is wired, and setting the split clips the after map.

        Test scenario:
            Our from-scratch runtime replaces py-maplibregl's sealed bundle, so this is the proof
            that it actually initialises maplibre, replays the recorded calls, exposes both maps, and
            that the feature clips the over-map — the behaviour no structure assertion can show.
        """
        before = WebMap().basemap().navigation()
        after = WebMap().basemap().navigation()
        html = swipe_html(before, after, title="smoke")
        probe_js = """() => {
            const DE = window.DE || {};
            let clip = null;
            if (DE.swipes && DE.swipes['de-swipe']) {
                DE.swipes['de-swipe'].setFraction(0.3);
                clip = document.getElementById('de-map-after').style.clipPath;
            }
            return {
                canvas_count: document.querySelectorAll('canvas').length,
                maps: Object.keys(DE.maps || {}),
                swipes: Object.keys(DE.swipes || {}),
                clip: clip,
            };
        }"""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "swipe.html"
            path.write_text(html, encoding="utf-8")
            page_errors, console_errors, probe = _render_and_probe(
                path.as_uri(), probe_js
            )

        assert page_errors == [], f"uncaught JS error(s): {page_errors}"
        assert _ignore_network_console(console_errors) == [], console_errors
        assert probe["canvas_count"] >= 2, f"both maps must draw a canvas: {probe}"
        assert sorted(probe["maps"]) == ["de-map-after", "de-map-before"], probe["maps"]
        assert probe["swipes"] == ["de-swipe"], probe["swipes"]
        assert probe["clip"] == "inset(0px 0px 0px 30%)", (
            f"setting the split to 0.3 must clip the after map, got {probe['clip']!r}"
        )


class TestMinimapRendersInABrowser:
    """WB-20 — the minimap page builds a main map and a synced overview that draws a view rectangle."""

    def test_overview_renders_and_tracks_the_main_map(self):
        """Both maps draw, the minimap is wired, and the overview carries the view-rectangle layer.

        Test scenario:
            The overview is a second map our runtime builds and the feature syncs to the main map,
            drawing a rectangle of the main view onto it — behaviour only a browser can show.
        """
        from digitalearth.web import minimap_html

        html = minimap_html(WebMap().basemap().navigation(), title="mini-smoke")
        probe_js = """() => {
            const DE = window.DE || {};
            const mini = DE.maps && DE.maps['de-minimap'];
            return {
                canvas_count: document.querySelectorAll('canvas').length,
                maps: Object.keys(DE.maps || {}),
                minimaps: Object.keys(DE.minimaps || {}),
                has_rect: !!(mini && mini.getSource && mini.getSource('de-view-rect')),
            };
        }"""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mini.html"
            path.write_text(html, encoding="utf-8")
            page_errors, console_errors, probe = _render_and_probe(
                path.as_uri(), probe_js
            )

        assert page_errors == [], f"uncaught JS error(s): {page_errors}"
        assert _ignore_network_console(console_errors) == [], console_errors
        assert probe["canvas_count"] >= 2, (
            f"main + overview must draw a canvas: {probe}"
        )
        assert sorted(probe["maps"]) == ["de-map-main", "de-minimap"], probe["maps"]
        assert probe["minimaps"] == ["de-minimap"], probe["minimaps"]
        assert probe["has_rect"], "the overview must draw the main map's view rectangle"


class TestMeasureRendersInABrowser:
    """WB-17 — the measure page loads the draw control and computes a drawn line's length live."""

    def test_drawing_a_line_updates_the_readout(self):
        """The draw control loads and a drawn 1-degree line reads ~111 km in the on-map readout.

        Test scenario:
            The readout runs in the browser off mapbox-gl-draw's events, so only a browser can show
            it: a line from (0,0) to (1,0) — one degree of longitude at the equator — must read about
            111 km, proving the draw control, the geodesic length, and the readout are all live.
        """
        from digitalearth.web import measure_html

        html = measure_html(WebMap().basemap(), title="measure-smoke")
        probe_js = """() => {
            const DE = window.DE || {};
            const h = DE.measures && DE.measures['de-map-measure'];
            let readout = null;
            if (h && h.draw) {
                h.draw.add({
                    type: 'Feature', properties: {},
                    geometry: { type: 'LineString', coordinates: [[0, 0], [1, 0]] },
                });
                h.update();
                readout = document.getElementById('de-measure-readout').textContent;
            }
            return {
                canvas_count: document.querySelectorAll('canvas').length,
                measures: Object.keys(DE.measures || {}),
                has_draw: typeof window.MapboxDraw !== 'undefined',
                readout: readout,
            };
        }"""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "measure.html"
            path.write_text(html, encoding="utf-8")
            page_errors, console_errors, probe = _render_and_probe(
                path.as_uri(), probe_js
            )

        assert page_errors == [], f"uncaught JS error(s): {page_errors}"
        assert _ignore_network_console(console_errors) == [], console_errors
        assert probe["canvas_count"] >= 1, f"the map must draw a canvas: {probe}"
        assert probe["measures"] == ["de-map-measure"], probe["measures"]
        assert probe["has_draw"], "mapbox-gl-draw did not load"
        assert probe["readout"] and "Distance: 111" in probe["readout"], (
            f"a 1-degree line must read ~111 km, got {probe['readout']!r}"
        )


class TestDeckAndGeocoderInABrowser:
    """The full reimplementation: deck.gl overlays and the reimplemented geocoder render in a browser."""

    def test_a_big_data_swipe_builds_deck_overlays(self):
        """Two big-data (deck.gl) maps in a swipe each build a deck overlay with no JS error.

        Test scenario:
            A ``points(big=True)`` map records a deck.gl overlay; the own-export runtime must load
            deck.gl, parse the ``@@type`` layer spec, and attach a MapboxOverlay to each map — the
            proof the deck.gl path works end to end, not only for a single map.
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"v": [1.0, 2.0, 3.0]},
            geometry=[Point(i * 0.1, i * 0.1) for i in range(3)],
            crs=4326,
        )
        before = WebMap().basemap().points(gdf, big=True)
        after = WebMap().basemap().points(gdf, big=True)
        html = swipe_html(before, after, title="deck-smoke")
        probe_js = """() => {
            const DE = window.DE || {};
            return {
                has_deck: typeof window.deck !== 'undefined',
                overlays: ['de-map-before', 'de-map-after'].map(
                    id => !!(DE.maps[id] && DE.maps[id]._deOverlay)
                ),
            };
        }"""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "deck.html"
            path.write_text(html, encoding="utf-8")
            page_errors, console_errors, probe = _render_and_probe(
                path.as_uri(), probe_js
            )

        assert page_errors == [], f"uncaught JS error(s): {page_errors}"
        assert _ignore_network_console(console_errors) == [], console_errors
        assert probe["has_deck"], "deck.gl did not load for a big-data map"
        assert probe["overlays"] == [True, True], (
            f"both maps must build a deck overlay, got {probe['overlays']}"
        )

    def test_a_geocoder_map_renders_the_search_box(self):
        """A map carrying a geocoder renders the reimplemented search box with no JS error.

        Test scenario:
            ``geocoder()`` records a ``MapTilerGeocodingControl``; through the own-export runtime it
            becomes our ``DEGeocoder`` control, so the search input must be present in the page.
        """
        before = WebMap().basemap().geocoder(api_key="demo-key")
        html = swipe_html(before, WebMap().basemap(), title="geocoder-smoke")
        probe_js = """() => ({
            has_box: !!document.querySelector('.de-geocoder-input'),
        })"""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "geo.html"
            path.write_text(html, encoding="utf-8")
            page_errors, console_errors, probe = _render_and_probe(
                path.as_uri(), probe_js
            )

        assert page_errors == [], f"uncaught JS error(s): {page_errors}"
        assert _ignore_network_console(console_errors) == [], console_errors
        assert probe["has_box"], "the reimplemented geocoder search box did not render"

    def test_a_nested_deck_accessor_is_refused(self):
        """A deck layer carrying a nested ``@@`` accessor is refused loudly, not silently mis-rendered.

        Test scenario:
            The layers are built directly (not through deck's JSONConverter), so an accessor
            expression would be copied through and silently fail. A nested ``{"@@function": …}`` on a
            layer prop must raise in the runtime (a page error), not pass through — the recursive
            refusal (M1), which the old top-level-only scan missed.
        """
        from digitalearth.web.htmldoc import HtmlDocument

        base = WebMap().basemap()._build_map_widget().to_dict()
        spec = {
            "@@type": "GeoJsonLayer",
            "id": "x",
            "data": {"type": "FeatureCollection", "features": []},
            "getFillColor": {"@@function": "interpolate"},
        }
        accessor = {**base, "calls": [*base["calls"], ["addDeckOverlay", [[spec]]]]}
        html = HtmlDocument.swipe(accessor, {"mapOptions": {}, "calls": []}).render()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "accessor.html"
            path.write_text(html, encoding="utf-8")
            page_errors, _console, _probe = _render_and_probe(path.as_uri(), "() => ({})")
        assert any("accessor" in e for e in page_errors), (
            f"a nested deck accessor must be refused with an error, got {page_errors}"
        )
