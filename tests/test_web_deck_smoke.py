"""Browser smoke test for the deck.gl layers (review H1).

The deck.gl builders are otherwise verified only at the spec level — the rest of the web suite asserts the layer
dict is built, not that deck.gl accepts and renders it. This test saves a ``WebMap`` with a deck.gl layer to
standalone HTML (which embeds the deck.gl JSON), loads it in a headless Chromium via Playwright, and asserts
deck.gl initialises with **no uncaught JS error and no deck.gl spec-rejection** — i.e. the ``@@type`` / ``@@=``
accessor JSON is valid.

Coverage: the local-data builders — ``deck_scatter`` / ``deck_polygons`` / ``point_cloud`` — are browser-verified
here. ``tiles_3d`` / ``gltf`` are **not** covered: they stream remote assets (a ``tileset.json`` / a ``.glb``),
which this hermetic job must not fetch; their docstrings keep a "not browser-verified" caveat.

Note: MapLibre owns the base ``<canvas>``, so the canvas count is a liveness check, not deck-specific proof — each
test therefore also asserts the saved HTML actually embedded the deck.gl layer spec (the ``@@type`` accessor), so a
regression that drops the deck JSON fails here.

Gated: Playwright and a Chromium browser are **not** part of the ``[web]`` extra, so this skips unless both
are installed. The CI ``deck-smoke`` job installs them and runs it; the default ``test-web`` suite skips it.
"""

import pytest

pytest.importorskip("maplibre")
pytest.importorskip("playwright")


def _render_and_collect(uri: str):
    """Load ``uri`` in headless Chromium, return ``(page_errors, console_errors, canvas_count)``.

    Skips the test (rather than failing) when no Chromium binary is available, so a machine with the
    Playwright Python package but no installed browser does not produce a false failure.
    """
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    page_errors: list = []
    console_errors: list = []
    with sync_playwright() as play:
        try:
            browser = play.chromium.launch()
        except Exception as exc:  # browser not installed / cannot launch -> skip, don't fail
            pytest.skip(f"Chromium not available for Playwright: {exc}")
        page = browser.new_page(viewport={"width": 900, "height": 700})
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.goto(uri, wait_until="load")
        try:
            page.wait_for_selector("canvas", timeout=20000)  # the MapLibre / deck.gl WebGL canvas
        except PWTimeout:  # no canvas in time -> the canvas_count assertion below reports it
            pass
        page.wait_for_timeout(3000)  # let deck.gl finish its first render after the libs load
        canvas_count = page.locator("canvas").count()
        browser.close()
    return page_errors, console_errors, canvas_count


@pytest.fixture()
def points_gdf():
    """A few points in lon/lat for a deck.gl scatter layer."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0]},
        geometry=[Point(i * 0.1, i * 0.1) for i in range(4)],
        crs=4326,
    )


@pytest.fixture()
def polygons_gdf():
    """A couple of polygons in lon/lat for a deck.gl polygon layer."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[
            Polygon([(0, 0), (0.1, 0), (0.1, 0.1)]),
            Polygon([(0.2, 0.2), (0.3, 0.2), (0.3, 0.3)]),
        ],
        crs=4326,
    )


#: deck-specific markers a console *error* would carry if deck.gl rejected the spec. Deliberately narrow — the
#: bare words ``layer`` / ``geojson`` match unrelated MapLibre messages, so only deck-flavoured tokens are used.
_DECK_ERROR_MARKERS = ("deck", "@@", "geojsonlayer", "pointcloudlayer", "scenegraphlayer", "tile3dlayer")


def _assert_html_has_deck_spec(html_path):
    """Fail if the saved HTML never embedded a deck.gl layer spec (guards a regression in ``save()``)."""
    html = html_path.read_text(encoding="utf-8")
    assert "@@type" in html, f"saved HTML embeds no deck.gl layer spec (no '@@type' accessor): {html_path}"


def _assert_clean_deck_render(page_errors, console_errors, canvas_count):
    """Shared assertions: no JS exception, no deck.gl spec rejection, a (MapLibre/deck) canvas was rendered.

    Note: MapLibre owns the base canvas, so ``canvas_count`` is a liveness check; deck-specific proof comes from
    the no-rejection gate plus the caller's :func:`_assert_html_has_deck_spec` check on the embedded JSON.
    """
    assert not page_errors, f"uncaught JS error rendering the deck.gl layer: {page_errors}"
    deck_errors = [e for e in console_errors if any(k in e.lower() for k in _DECK_ERROR_MARKERS)]
    assert not deck_errors, f"deck.gl rejected the layer spec: {deck_errors}"
    assert canvas_count >= 1, "no canvas rendered — deck.gl / MapLibre did not initialise"


class TestDeckRendersInBrowser:
    """The deck.gl JSON specs render in a real browser (promotes them out of 'experimental')."""

    def test_deck_scatter_renders(self, tmp_path, points_gdf):
        from digitalearth.web import WebMap

        out = tmp_path / "deck_scatter.html"
        WebMap().deck_scatter(points_gdf).save(str(out))
        _assert_html_has_deck_spec(out)
        _assert_clean_deck_render(*_render_and_collect(out.as_uri()))

    def test_deck_polygons_renders(self, tmp_path, polygons_gdf):
        from digitalearth.web import WebMap

        out = tmp_path / "deck_polygons.html"
        WebMap().deck_polygons(polygons_gdf).save(str(out))
        _assert_html_has_deck_spec(out)
        _assert_clean_deck_render(*_render_and_collect(out.as_uri()))

    def test_point_cloud_renders(self, tmp_path, points_gdf):
        """``point_cloud`` takes local coords (no network), so it is browser-verifiable like scatter/polygons."""
        from digitalearth.web import WebMap

        out = tmp_path / "deck_point_cloud.html"
        WebMap().point_cloud(points_gdf).save(str(out))
        _assert_html_has_deck_spec(out)
        _assert_clean_deck_render(*_render_and_collect(out.as_uri()))
