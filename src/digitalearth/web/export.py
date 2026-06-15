"""ExportMixin — web-tier export & sharing (DW.6, recipe W7).

The headline of the tier: turn any ``WebMap`` into a shareable artifact.

* ``to_html`` — the standalone HTML string (optionally fully offline);
* offline bundling — inline the ``maplibre-gl`` JS/CSS that ``to_html`` otherwise CDN-references, so the page
  opens with no network (best-effort: fetched once at save time);
* PNG snapshot — render the HTML in a headless browser and screenshot it. The browser is an **optional,
  gated** dependency (not in the ``[web]`` extra): ``save(*.png)`` raises an actionable ``ImportError`` when
  neither Playwright nor Selenium is installed, rather than failing obscurely.

``WebMapBase.save`` dispatches HTML vs. PNG and the ``offline`` flag here (the base sits first in the MRO, so
these are hooks it calls, not overrides). urllib / browser libs are imported lazily.
"""

import pathlib
import re
import tempfile
from typing import Any

#: CDN asset URLs (js/css) ``to_html`` references, matched for offline inlining.
_ASSET_RE = re.compile(r'<(script|link)[^>]*?(?:src|href)="(?P<url>https?://[^"]+?\.(?:js|css))"[^>]*?>(?:</script>)?')


class ExportMixin:
    """Export builders (HTML / offline HTML / PNG) for :class:`~digitalearth.web.map.WebMap`."""

    def to_html(self, *, title: str = "Digital-Earth map", offline: bool = False, **kwargs: Any) -> str:
        """Return the map as a standalone HTML string.

        Args:
            title: HTML document title.
            offline: When True, inline the ``maplibre-gl`` JS/CSS so the page needs no network (best-effort).
            **kwargs: Forwarded to ``MapWidget.to_html``.

        Returns:
            The HTML document as a string.

        Raises:
            ImportError: when the ``web`` extra is not installed.
        """
        html = self._build_map_widget().to_html(title=title, **kwargs)
        return self._inline_offline_assets(html) if offline else html

    @staticmethod
    def _inline_offline_assets(html: str) -> str:
        """Inline CDN ``<script src>`` / ``<link href>`` assets into ``html`` for offline use (best-effort).

        Fetches each referenced ``.js`` / ``.css`` URL once and replaces the tag with an inline
        ``<script>`` / ``<style>`` block, so the page opens without a network. A fetch failure (offline at
        save time) raises a clear error rather than silently shipping a still-online page.

        Args:
            html: The HTML produced by ``to_html`` (CDN-referencing).

        Returns:
            HTML with the external JS/CSS inlined.

        Raises:
            RuntimeError: when an asset cannot be fetched (e.g. no network at save time).
        """
        import urllib.request

        def _replace(match: "re.Match") -> str:
            url = match.group("url")
            try:
                with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed CDN https URLs
                    body = response.read().decode("utf-8")
            except Exception as err:  # network/offline at save time — fail loudly, don't ship a broken page
                raise RuntimeError(
                    f"offline=True could not fetch {url!r} to inline it; a network is needed at save time"
                ) from err
            if url.endswith(".css"):
                return f"<style>{body}</style>"
            return f"<script>{body}</script>"

        new_html, replaced = _ASSET_RE.subn(_replace, html)
        if replaced == 0 and re.search(r'https?://[^"\']+\.(?:js|css)', html):
            # The page still references a CDN asset our regex did not match — don't claim it is offline.
            raise RuntimeError(
                "offline=True found CDN asset references but inlined none of them; the to_html tag format "
                "may have changed. Report this so the inliner regex can be updated."
            )
        return new_html

    def _render_png(self, path: str, *, title: str = "Digital-Earth map", **kwargs: Any) -> str:
        """Render the map to a PNG via a headless browser and return ``path`` (gated optional dep).

        Tries Playwright, then Selenium; both render the standalone HTML offscreen and screenshot it. Neither
        is in the ``[web]`` extra, so this raises an actionable ``ImportError`` when no browser is available.

        Args:
            path: Output ``*.png`` file.
            title: HTML document title.
            **kwargs: Forwarded to ``to_html`` (e.g. an inline-data ``offline`` is not applied — PNG embeds
                pixels, not the live map).

        Returns:
            The ``path`` written.

        Raises:
            ImportError: when neither Playwright nor Selenium is installed.
        """
        html = self._build_map_widget().to_html(title=title, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            html_path = pathlib.Path(tmp) / "map.html"
            html_path.write_text(html, encoding="utf-8")
            url = html_path.as_uri()
            for renderer in (self._png_via_playwright, self._png_via_selenium):
                try:
                    renderer(url, path)
                    return str(path)
                except ImportError:
                    continue
        raise ImportError(
            "PNG export needs a headless browser, which is not part of digitalearth[web]. Install Playwright "
            "(`pip install playwright && python -m playwright install chromium`) or Selenium + a driver."
        )

    @staticmethod
    def _png_via_playwright(url: str, path: str) -> None:
        """Screenshot ``url`` to ``path`` with Playwright (raises ImportError if it is not installed)."""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as play:
            browser = play.chromium.launch()
            page = browser.new_page(viewport={"width": 1024, "height": 768})
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(1500)  # let MapLibre finish the first render
            page.screenshot(path=path)
            browser.close()

    @staticmethod
    def _png_via_selenium(url: str, path: str) -> None:
        """Screenshot ``url`` to ``path`` with headless Selenium (raises ImportError if not installed)."""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1024,768")
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            driver.implicitly_wait(2)
            driver.save_screenshot(path)
        finally:
            driver.quit()
