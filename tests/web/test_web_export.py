"""DW.6 — web-tier export & sharing: HTML (incl. offline inlining) and gated PNG snapshots.

The offline-asset regex/inliner is tested without the engine or network; HTML/PNG export through ``save``
``importorskip``s maplibre. PNG gating is tested deterministically by stubbing the browser renderers so the
test does not depend on Playwright/Selenium being installed.
"""

import pathlib

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

    def test_save_png_without_browser_raises_actionable(
        self, tmp_path, polygons_gdf, monkeypatch
    ):
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


class TestTheOfflineInlinerRewritesTags:
    """The inliner's success path: a fetched asset must come back as the right kind of tag.

    A stylesheet wrapped in `<script>` (or the reverse) produces a page that loads without error and
    renders wrong, which is the failure mode an offline export exists to avoid.
    """

    @staticmethod
    def _serve(body, monkeypatch):
        """Answer every `urlopen` with `body`.

        Args:
            body: The text the fake CDN returns.
            monkeypatch: pytest's patcher.
        """
        import contextlib
        import urllib.request

        class _Response:
            """The bytes-returning object `urlopen` hands back."""

            def read(self):
                """Return the served body as bytes."""
                return body.encode("utf-8")

        @contextlib.contextmanager
        def _urlopen(url, timeout=None):
            """Yield the canned response instead of reaching the network."""
            yield _Response()

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    def test_a_stylesheet_becomes_a_style_element(self, monkeypatch):
        """A .css asset has to be inlined as CSS, not as a script the browser would try to run.

        Args:
            monkeypatch: pytest's patcher.
        """
        self._serve(".maplibregl-map{position:relative}", monkeypatch)
        html = '<link href="https://cdn.example.com/maplibre-gl.css" rel="stylesheet">'
        out = WebMap()._inline_offline_assets(html)
        assert out.startswith("<style>"), out
        assert "position:relative" in out, out
        assert "cdn.example.com" not in out, "the page still points at the CDN"

    def test_a_script_becomes_a_script_element(self, monkeypatch):
        """A .js asset keeps its executable meaning when inlined.

        Args:
            monkeypatch: pytest's patcher.
        """
        self._serve("window.maplibregl = {};", monkeypatch)
        html = '<script src="https://cdn.example.com/maplibre-gl.js"></script>'
        out = WebMap()._inline_offline_assets(html)
        assert out.startswith("<script>"), out
        assert "window.maplibregl" in out, out
        assert "cdn.example.com" not in out, "the page still points at the CDN"


class TestTheHeadlessBrowserBackends:
    """The two screenshot wrappers, driven against fake browser modules.

    Neither Playwright nor Selenium is in the `[web]` extra, so the real drivers are never present in CI
    — but the wrappers still decide the viewport, the settle time and whether the browser is released,
    and a leaked browser process only shows up under a long batch.
    """

    def test_playwright_screenshots_the_url_and_closes_the_browser(self, monkeypatch):
        """Playwright is tried first, so its wrapper must write the file and release the browser.

        Args:
            monkeypatch: pytest's patcher.
        """
        import sys
        import types

        calls = {}

        class _Page:
            """Record what the wrapper asks the page to do."""

            def goto(self, url, wait_until=None):
                """Record the navigation."""
                calls["url"] = url

            def wait_for_timeout(self, milliseconds):
                """Record the settle time given to MapLibre's first render."""
                calls["settle"] = milliseconds

            def screenshot(self, path):
                """Record the output path."""
                calls["path"] = path

        class _Browser:
            """A browser that remembers being closed."""

            def new_page(self, viewport=None):
                """Return the recording page."""
                calls["viewport"] = viewport
                return _Page()

            def close(self):
                """Record the close."""
                calls["closed"] = True

        class _Play:
            """The object `sync_playwright()` yields."""

            chromium = types.SimpleNamespace(launch=lambda: _Browser())

            def __enter__(self):
                """Enter the context."""
                return self

            def __exit__(self, *exc):
                """Leave the context."""
                return False

        module = types.ModuleType("playwright.sync_api")
        module.sync_playwright = _Play
        monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
        monkeypatch.setitem(sys.modules, "playwright.sync_api", module)

        WebMap._png_via_playwright("file:///map.html", "out.png")
        assert calls["url"] == "file:///map.html", calls
        assert calls["path"] == "out.png", calls
        assert calls["settle"] > 0, (
            "no settle time: MapLibre is screenshotted mid-render"
        )
        assert calls["closed"] is True, "the browser was left running"


class _FakeWebDriverError(Exception):
    """Stand-in for Selenium's own error type, raised by the fake driver below."""


class TestSeleniumIsTheFallback:
    """Selenium runs only when Playwright is absent, and must clean up after itself."""

    def test_the_driver_quits_even_when_the_page_fails(self, monkeypatch):
        """`quit()` sits in a `finally` for a reason — a dead page must not leak a chromedriver.

        Args:
            monkeypatch: pytest's patcher.
        """
        import sys
        import types

        calls = {"args": []}

        class _Options:
            """Collect the flags the wrapper sets."""

            def add_argument(self, argument):
                """Record one flag."""
                calls["args"].append(argument)

        class _Driver:
            """A driver whose navigation fails."""

            def get(self, url):
                """Fail the way a broken browser binary does."""
                raise _FakeWebDriverError("no such window")

            def implicitly_wait(self, seconds):
                """Unused on this path."""

            def save_screenshot(self, path):
                """Unused on this path."""

            def quit(self):
                """Record the quit."""
                calls["quit"] = True

        webdriver = types.ModuleType("selenium.webdriver")
        webdriver.Chrome = lambda options=None: _Driver()
        options_module = types.ModuleType("selenium.webdriver.chrome.options")
        options_module.Options = _Options
        selenium = types.ModuleType("selenium")
        selenium.webdriver = webdriver
        chrome = types.ModuleType("selenium.webdriver.chrome")
        monkeypatch.setitem(sys.modules, "selenium", selenium)
        monkeypatch.setitem(sys.modules, "selenium.webdriver", webdriver)
        monkeypatch.setitem(sys.modules, "selenium.webdriver.chrome", chrome)
        monkeypatch.setitem(
            sys.modules, "selenium.webdriver.chrome.options", options_module
        )

        with pytest.raises(_FakeWebDriverError):
            WebMap._png_via_selenium("file:///map.html", "out.png")
        assert calls["quit"] is True, (
            "the driver was left running after a failed page load"
        )
        assert any("headless" in argument for argument in calls["args"]), calls["args"]

    def test_it_is_tried_when_playwright_is_absent(self, tmp_path, monkeypatch):
        """The fallback is the point of having two backends, and nothing asserted the order.

        Args:
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
        """
        pytest.importorskip("maplibre")
        order = []

        def _absent(url, path):
            """Behave like a backend that is not installed."""
            order.append("playwright")
            raise ImportError("stubbed: playwright absent")

        def _present(url, path):
            """Behave like the backend that is installed."""
            order.append("selenium")
            pathlib.Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(WebMap, "_png_via_playwright", staticmethod(_absent))
        monkeypatch.setattr(WebMap, "_png_via_selenium", staticmethod(_present))
        out = tmp_path / "m.png"
        assert WebMap().basemap()._render_png(str(out)) == out, (
            "the PNG renderer returns the pathlib.Path it wrote (C1)"
        )
        assert order == ["playwright", "selenium"], order
        assert out.read_bytes().startswith(b"\x89PNG"), "no screenshot was written"
