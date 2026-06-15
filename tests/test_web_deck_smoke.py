"""Browser smoke test for the deck.gl layers (review H1).

The deck.gl builders (`deck_scatter` / `deck_polygons` / `point_cloud` / `tiles_3d` / `gltf`) are otherwise
verified only at the spec level — the rest of the web suite asserts the layer dict is built, not that deck.gl
accepts and renders it. This test saves a ``WebMap`` with a deck.gl layer to standalone HTML (which embeds the
deck.gl JSON), loads it in a headless Chromium via Playwright, and asserts deck.gl initialises a canvas with
**no uncaught JS error and no deck.gl spec-rejection** — i.e. the ``@@type`` / ``@@=`` accessor JSON is valid.

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
        except Exception:
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


def _assert_clean_deck_render(page_errors, console_errors, canvas_count):
    """Shared assertions: no JS exception, no deck.gl spec rejection, a canvas was rendered."""
    assert not page_errors, f"uncaught JS error rendering the deck.gl layer: {page_errors}"
    deck_errors = [
        e for e in console_errors if any(k in e.lower() for k in ("deck", "layer", "geojson", "@@"))
    ]
    assert not deck_errors, f"deck.gl rejected the layer spec: {deck_errors}"
    assert canvas_count >= 1, "no canvas rendered — deck.gl / MapLibre did not initialise"


class TestDeckRendersInBrowser:
    """The deck.gl JSON specs render in a real browser (promotes them out of 'experimental')."""

    def test_deck_scatter_renders(self, tmp_path, points_gdf):
        from digitalearth.web import WebMap

        out = tmp_path / "deck_scatter.html"
        WebMap().deck_scatter(points_gdf).save(str(out))
        _assert_clean_deck_render(*_render_and_collect(out.as_uri()))

    def test_deck_polygons_renders(self, tmp_path, polygons_gdf):
        from digitalearth.web import WebMap

        out = tmp_path / "deck_polygons.html"
        WebMap().deck_polygons(polygons_gdf).save(str(out))
        _assert_clean_deck_render(*_render_and_collect(out.as_uri()))
