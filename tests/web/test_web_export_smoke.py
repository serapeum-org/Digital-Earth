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


def _render_and_probe(uri: str):
    """Load ``uri`` in headless Chromium and return ``(page_errors, console_errors, probe)``.

    Skips (not fails) when no Chromium binary is available, so the Playwright package without an
    installed browser does not produce a false failure.

    Args:
        uri: A ``file://`` URI of the page to load.

    Returns:
        ``(page_errors, console_errors, probe)`` — JS exceptions, console errors, and a dict of
        ``canvas_count`` / built map ids / wired swipe ids / the after map's clip after a set.
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
        probe = page.evaluate(
            """() => {
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
        )
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
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "swipe.html"
            path.write_text(html, encoding="utf-8")
            page_errors, console_errors, probe = _render_and_probe(path.as_uri())

        assert page_errors == [], f"uncaught JS error(s): {page_errors}"
        assert _ignore_network_console(console_errors) == [], console_errors
        assert probe["canvas_count"] >= 2, f"both maps must draw a canvas: {probe}"
        assert sorted(probe["maps"]) == ["de-map-after", "de-map-before"], probe["maps"]
        assert probe["swipes"] == ["de-swipe"], probe["swipes"]
        assert probe["clip"] == "inset(0px 0px 0px 30%)", (
            f"setting the split to 0.3 must clip the after map, got {probe['clip']!r}"
        )
