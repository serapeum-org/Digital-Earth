"""DW.6 — web-tier export & sharing: HTML (incl. offline inlining) and gated PNG snapshots.

The offline-asset regex/inliner is tested without the engine or network; HTML/PNG export through ``save``
``importorskip``s maplibre. PNG gating is tested deterministically by stubbing the browser renderers so the
test does not depend on Playwright/Selenium being installed.
"""

import pytest

from digitalearth.web import WebMap
from digitalearth.web.export import _ASSET_RE


@pytest.fixture()
def polygons_gdf():
    """Two triangles in lon/lat with a ``pop`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    geoms = [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 2), (3, 2), (3, 3)])]
    return gpd.GeoDataFrame({"pop": [1.0, 9.0]}, geometry=geoms, crs=4326)


class TestOfflineInliningPure:
    """The asset regex + inliner behave correctly without engine or network."""

    def test_regex_matches_cdn_script_tag(self):
        tag = '<script src="https://unpkg.com/maplibre-gl@4.0.0/dist/maplibre-gl.js"></script>'
        m = _ASSET_RE.search(tag)
        assert m is not None and m.group("url").endswith("maplibre-gl.js")

    def test_inline_leaves_assetless_html_unchanged(self):
        html = "<html><body>no external assets here</body></html>"
        assert WebMap()._inline_offline_assets(html) == html

    def test_inline_fetch_failure_is_loud(self):
        """A CDN tag pointing at an unreachable host raises a clear RuntimeError (no silent online page)."""
        html = '<link href="https://nonexistent.invalid/x.css">'
        with pytest.raises(RuntimeError, match="offline=True could not fetch"):
            WebMap()._inline_offline_assets(html)

    def test_inline_unmatched_cdn_reference_raises(self):
        """A CDN asset the regex cannot match must raise, not silently ship a still-online page (M2)."""
        # A bare URL (not inside a matchable <script src>/<link href> tag) → zero substitutions but a CDN ref.
        html = "<html><body>see https://cdn.example.com/maplibre-gl.js for the engine</body></html>"
        with pytest.raises(RuntimeError, match="inlined none"):
            WebMap()._inline_offline_assets(html)


class TestExportNeedsEngine:
    """HTML and PNG export through ``save`` (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_to_html_returns_map_document(self, polygons_gdf):
        html = WebMap().choropleth(polygons_gdf, column="pop").to_html()
        assert "maplibre" in html.lower() and len(html) > 1_000

    def test_save_html_writes_file(self, tmp_path, polygons_gdf):
        out = tmp_path / "m.html"
        WebMap().choropleth(polygons_gdf, column="pop").save(str(out))
        assert out.stat().st_size > 1_000

    def test_save_png_without_browser_raises_actionable(self, tmp_path, polygons_gdf, monkeypatch):
        """With no headless browser available, ``save(*.png)`` raises a clear, actionable ImportError."""

        def _no_browser(url, path):
            raise ImportError("stubbed: browser absent")

        monkeypatch.setattr(WebMap, "_png_via_playwright", staticmethod(_no_browser))
        monkeypatch.setattr(WebMap, "_png_via_selenium", staticmethod(_no_browser))
        m = WebMap().polygons(polygons_gdf)
        with pytest.raises(ImportError, match="headless browser"):
            m.save(str(tmp_path / "m.png"))

    def test_fmt_png_forces_png_dispatch(self, tmp_path, polygons_gdf, monkeypatch):
        """``fmt='png'`` routes to the PNG path even when the suffix is not .png."""
        captured = {}

        def _fake_png(self, path, **kwargs):
            captured["path"] = path
            return path

        monkeypatch.setattr(WebMap, "_render_png", _fake_png)
        WebMap().polygons(polygons_gdf).save(str(tmp_path / "snapshot.out"), fmt="png")
        assert captured["path"].endswith("snapshot.out")
