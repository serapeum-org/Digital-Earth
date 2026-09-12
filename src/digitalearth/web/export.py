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
from typing import TYPE_CHECKING, Any


def _write_gif(frames: list, path: str, *, duration: float, loop: int) -> None:
    """Encode PNG frames into an animated GIF.

    Pillow does the encoding — it arrives with matplotlib, so an animation needs no dependency the tier
    does not already have. MP4 would need ffmpeg, which it does not.

    Args:
        frames: Paths to the rendered PNG frames, in order.
        path: Where to write the GIF.
        duration: Seconds each frame is held.
        loop: Repeat count; ``0`` loops forever.

    Raises:
        ValueError: when there are no frames to write.
    """
    from PIL import Image

    if not frames:
        raise ValueError("no frames to write")
    images = []
    for frame in frames:
        # Closed before the GIF is written: PIL opens lazily, and a still-open frame keeps a Windows file
        # handle on the caller's temporary directory, which then fails to delete.
        with Image.open(frame) as handle:
            images.append(handle.convert("RGB"))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000),
        loop=int(loop),
    )


from digitalearth.web.base import DEFAULT_TITLE

#: CDN asset URLs (js/css) ``to_html`` references, matched for offline inlining.
_ASSET_RE = re.compile(
    r'<(script|link)[^>]*?(?:src|href)="(?P<url>https?://[^"]+?\.(?:js|css))"[^>]*?>(?:</script>)?'
)


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class ExportMixin(_MixinBase):
    """Export builders (HTML / offline HTML / PNG) for :class:`~digitalearth.web.map.WebMap`.

    A capability mixin of :class:`~digitalearth.web.map.WebMap`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the layer registry, the display CRS and the render/save
    lifecycle — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.web.base.WebMapBase`, the state ``WebMap`` inherits.
    At runtime that base is plain ``object``, so composing this mixin leaves the ``WebMap`` MRO exactly what it was
    before the annotation.

    See Also:
        digitalearth.web.map.WebMap: the composition that supplies the state these methods use.
        digitalearth.web.base.WebMapBase: the typing-only base declared above the class.
    """

    def to_html(
        self, *, title: str = DEFAULT_TITLE, offline: bool = False, **kwargs: Any
    ) -> str:
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

    def _render_png(
        self, path: str, *, title: str = DEFAULT_TITLE, **kwargs: Any
    ) -> str:
        """Render the map to a PNG via a headless browser and return ``path`` (gated optional dep).

        Tries Playwright, then Selenium; both render the standalone HTML offscreen and screenshot it. Neither
        is in the ``[web]`` extra, so this raises an actionable ``ImportError`` when no browser is available.

        Args:
            path: Output ``*.png`` file.
            title: HTML document title.
            **kwargs: Reserved for headless-browser options, and not forwarded to ``to_html``. The one
                recognised key is ``widget``: a pre-built map widget to render instead of building a fresh
                one, which is how :meth:`to_gif` renders a frame with one step visible.

        Returns:
            The ``path`` written.

        Raises:
            ImportError: when neither Playwright nor Selenium is installed.
        """
        # kwargs are reserved for future headless-browser options; they are NOT forwarded into to_html
        # (which would surface unknown-keyword errors from deep inside maplibre). `widget` is the one
        # exception: an animation frame hands in a widget it has already set the step visibility on.
        widget = kwargs.pop("widget", None)
        _ = kwargs
        html = (widget or self._build_map_widget()).to_html(title=title)
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

    def to_gif(
        self,
        path: str,
        *,
        duration: float = 0.8,
        loop: int = 0,
        title: str = DEFAULT_TITLE,
    ) -> str:
        """Write a temporal map's steps as an animated GIF (recipe W7).

        A time series is the case where a moving image says most, and it is also the case a web page
        carries worst — a saved page shows a step picker, not an animation, and nothing could be pasted
        into a report. The other two tiers have had ``save_animation`` all along.

        One frame is rendered per time step, by making that step the only visible one and screenshotting
        the page, so the frames are the same pixels the map draws.

        Args:
            path: Where to write the GIF.
            duration: Seconds each frame is held.
            loop: How many times to repeat; ``0`` loops forever.
            title: HTML document title used while rendering.

        Returns:
            The ``path`` written.

        Raises:
            ValueError: when the map has no time steps to animate, or fewer than two.
            ImportError: when no headless browser is installed — the same gated dependency the PNG
                snapshot needs, and deliberately not part of ``digitalearth[web]``.

        Examples:
            - Animate a raster stack:
                ```python
                >>> from digitalearth.web import WebMap                          # doctest: +SKIP
                >>> WebMap().basemap().timeslider(stack).to_gif("out.gif")       # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.temporal.TemporalMixin.timeslider: builds the steps this animates.
        """
        import tempfile

        frames = self._temporal_frames()
        with tempfile.TemporaryDirectory() as work:
            images = [
                self._frame_png(pathlib.Path(work) / f"frame{index}.png", frame, title)
                for index, frame in enumerate(frames)
            ]
            _write_gif(images, path, duration=duration, loop=loop)
        return str(path)

    def _temporal_frames(self) -> list:
        """Return the visible-layer set for each time step, oldest first.

        Args:
            None.

        Returns:
            One list of layer ids per step — the layers that must be visible in that frame.

        Raises:
            ValueError: when there is no series to animate, naming what to add.
        """
        config = getattr(self, "_temporal", None)
        layer_ids = list((config or {}).get("layer_ids") or [])
        if config is None or (config.get("mode") != "raster") or len(layer_ids) < 2:
            raise ValueError(
                "to_gif() needs a raster time series with at least two steps; add one with "
                "timeslider(collection). The vector time-slider filters a single layer, so its steps "
                "are not separately renderable."
            )
        return [[layer_id] for layer_id in layer_ids]

    def _frame_png(self, path: Any, visible: list, title: str) -> str:
        """Render one animation frame by showing only ``visible`` and screenshotting the page.

        Args:
            path: Where to write this frame.
            visible: The layer ids to show; every other step layer is hidden.
            title: HTML document title used while rendering.

        Returns:
            The frame's path, as a string.
        """
        config = self._temporal or {}
        steps = list(config.get("layer_ids") or [])
        widget = self._build_map_widget()
        for layer_id in steps:
            widget.set_visibility(layer_id, layer_id in visible)
        return self._render_png(str(path), title=title, widget=widget)

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
