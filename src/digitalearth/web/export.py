"""ExportMixin — web-tier export & sharing (DW.6, recipe W7).

The headline of the tier: turn any ``WebMap`` into a shareable artifact.

* ``to_html`` — the standalone HTML string (optionally fully offline);
* offline bundling — inline the ``maplibre-gl`` JS/CSS that ``to_html`` otherwise CDN-references, so the page
  opens with no network (best-effort: fetched once at save time);
* PNG snapshot — render the HTML in a headless browser and screenshot it. The browser is an **optional,
  gated** dependency (not in the ``[web]`` extra): ``save(*.png)`` raises an actionable ``ImportError`` when
  neither Playwright nor Selenium is installed, rather than failing obscurely.
* ``animate`` — one screenshot per time step, encoded as a GIF at ``fps`` frames per second (the rate every
  tier's animation entry point takes). ``animate`` and ``to_gif`` are its deprecated names, and
  ``duration=`` its deprecated
  seconds-per-frame spelling, converted to ``fps`` rather than reinterpreted.

``WebMapBase.save`` dispatches HTML vs. PNG and the ``offline`` flag here (the base sits first in the MRO, so
these are hooks it calls, not overrides). urllib / browser libs are imported lazily.
"""

import math
import pathlib
import re
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Optional

from digitalearth.base.animation import DEFAULT_FPS
from digitalearth.base.deprecation import renamed_method, renamed_parameter
from digitalearth.web.base import DEFAULT_TITLE

# `DEFAULT_FPS` is imported above rather than declared here: the rate every tier's animation entry point
# defaults to lives in `digitalearth.base.animation`, so one number means one speed whichever backend renders
# the series. It stays importable from this module because that is where this tier's callers and tests already
# reach for it.


def _fps_from_duration(seconds: Any) -> float:
    """Convert the deprecated ``duration=`` (seconds held per frame) into the ``fps`` it means.

    The rename is a unit change, not a spelling change, so the value is *converted* rather than
    reinterpreted: ``duration=0.5`` has always meant a half-second hold, i.e. two frames a second, and it
    still does. The positivity check lives here rather than at the call site so it runs after the
    both-spellings guard in :func:`~digitalearth.base.deprecation.renamed_parameter` — a contradictory call
    is a ``TypeError`` about the two names, not a complaint about one of the values.

    A non-finite hold is refused by name, before the arithmetic. ``nan`` compares ``False`` against every
    bound, so it slipped the positivity check here *and* the one on the resolved ``fps``, and ``1 / nan`` is
    ``nan`` — a rate that means nothing reached the encoder silently (review M10). ``inf`` is refused the
    same way: a frame held forever is not an animation, and ``1 / inf`` is a zero rate this then re-rejects
    in terms of a number the caller never wrote.

    Args:
        seconds: Seconds to hold each frame, as the caller wrote it.

    Returns:
        The equivalent frames per second.

    Raises:
        ValueError: when ``seconds`` is not a finite positive number — a zero hold has no rate and would
            divide by zero, and ``nan``/``inf`` have no rate to convert to at all.

    Examples:
        - A hold that is not a number is named as such, rather than becoming a ``nan`` frame rate:
            ```python
            >>> from digitalearth.web.export import _fps_from_duration
            >>> _fps_from_duration(float("nan"))
            Traceback (most recent call last):
                ...
            ValueError: duration= must be a finite number of seconds; got nan

            ```
    """
    value = float(seconds)
    if not math.isfinite(value):
        raise ValueError(
            f"duration= must be a finite number of seconds; got {seconds!r}"
        )
    if value <= 0:
        raise ValueError(f"duration= must be positive; got {seconds!r}")
    return 1.0 / value


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
        ValueError: when there are no frames to write. ``animate`` cannot reach this — it refuses a series
            of fewer than two steps first — but this function is the encoder for any frame list, so it
            checks rather than writing a GIF with nothing in it.
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
    ) -> pathlib.Path:
        """Render the map to a PNG via a headless browser and return its path (gated optional dep).

        Tries Playwright, then Selenium; both render the standalone HTML offscreen and screenshot it. Neither
        is in the ``[web]`` extra, so this raises an actionable ``ImportError`` when no browser is available.

        Args:
            path: Output ``*.png`` file.
            title: HTML document title.
            **kwargs: Reserved for headless-browser options, and not forwarded to ``to_html``. Two keys
                are recognised: ``widget``, a pre-built map widget to render instead of building a fresh
                one (how :meth:`animate` renders a frame with one step visible), and ``kind``, the export
                name quoted in the missing-browser error so a GIF failure does not talk about PNG.

        Returns:
            The :class:`pathlib.Path` written.

        Raises:
            ImportError: when neither Playwright nor Selenium is installed.
        """
        # kwargs are reserved for future headless-browser options; they are NOT forwarded into to_html
        # (which would surface unknown-keyword errors from deep inside maplibre). `widget` is the one
        # exception: an animation frame hands in a widget it has already set the step visibility on.
        widget = kwargs.pop("widget", None)
        kind = kwargs.pop("kind", "PNG")
        _ = kwargs
        html = (widget or self._build_map_widget(with_controls=False)).to_html(
            title=title
        )
        with tempfile.TemporaryDirectory() as tmp:
            html_path = pathlib.Path(tmp) / "map.html"
            html_path.write_text(html, encoding="utf-8")
            url = html_path.as_uri()
            for renderer in (self._png_via_playwright, self._png_via_selenium):
                try:
                    renderer(url, str(path))
                    return pathlib.Path(path)
                except ImportError:
                    continue
        raise ImportError(
            f"{kind} export needs a headless browser, which is not part of digitalearth[web]. Install "
            "Playwright (`pip install playwright && python -m playwright install chromium`) or Selenium + "
            "a driver."
        )

    def save_animation(
        self,
        path: str,
        *,
        fps: Optional[float] = None,
        loop: int = 0,
        title: str = DEFAULT_TITLE,
        duration: Optional[float] = None,
    ) -> pathlib.Path:
        """Write a temporal map's steps as an animated GIF (recipe W7).

        A time series is the case where a moving image says most, and it is also the case a web page
        carries worst — a saved page shows a step picker, not an animation, and nothing could be pasted
        into a report. The other two tiers have had ``save_animation`` all along.

        One frame is rendered per time step, by making that step the only visible one and screenshotting
        the page, so the frames are the same pixels the map draws.

        Args:
            path: Where to write the GIF.
            fps: Frames per second — the rate every tier's animation entry point takes, with the same
                default (:data:`DEFAULT_FPS`, ``3.0``, declared once in :mod:`digitalearth.base.animation`;
                the signature's ``None`` is the "not passed" sentinel the deprecated spelling is resolved
                against), so one number means one speed across the whole package. This entry point used to
                be spelled ``duration=0.8`` (one frame held 0.8 s, i.e. 1.25 fps); a call that names no rate
                now renders faster, and ``fps=1.25`` restores the previous speed.
            loop: How many times to repeat; ``0`` loops forever.
            title: HTML document title used while rendering.
            duration: **Deprecated** spelling of the frame rate, in seconds held per frame. Passing
                it warns that ``duration=`` will be removed in a future release and to write
                ``fps=`` instead. It is *converted* (``fps = 1 / duration``), never
                reinterpreted, so an old call produces the animation it always did.

        Returns:
            The :class:`pathlib.Path` written.

        Raises:
            TypeError: when both ``fps`` and the deprecated ``duration`` are passed — one rate, two
                spellings, so neither can be silently preferred.
            ValueError: when the map has no time steps to animate, or fewer than two, or when ``fps`` /
                ``duration`` is not a finite positive number — a zero rate has no frame to hold, and a
                ``nan``/``inf`` one is no rate at all.
            ImportError: when no headless browser is installed — the same gated dependency the PNG
                snapshot needs, and deliberately not part of ``digitalearth[web]``.

        Examples:
            - Animate a raster stack:
                ```python
                >>> from digitalearth.web import WebMap                           # doctest: +SKIP
                >>> from pyramids.dataset.collection import DatasetCollection   # doctest: +SKIP
                >>> stack = DatasetCollection.from_files(["jan.tif", "feb.tif"])  # doctest: +SKIP
                >>> WebMap().basemap().timeslider(stack).animate("out.gif")       # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.temporal.TemporalMixin.timeslider: builds the steps this animates.
            digitalearth.web.export.ExportMixin.to_gif: the deprecated name of this method.
        """
        fps = renamed_parameter(
            new="fps",
            value=fps,
            old="duration",
            alias=duration,
            caller="WebMap.save_animation()",
            default=DEFAULT_FPS,
            convert=_fps_from_duration,
        )
        if not math.isfinite(float(fps)):
            raise ValueError(f"fps= must be a finite rate; got {fps!r}")
        if float(fps) <= 0:
            raise ValueError(f"fps= must be positive; got {fps!r}")
        frames = self._temporal_frames()
        with tempfile.TemporaryDirectory() as work:
            images = [
                self._frame_png(pathlib.Path(work) / f"frame{index}.png", frame, title)
                for index, frame in enumerate(frames)
            ]
            _write_gif(images, path, duration=1.0 / float(fps), loop=loop)
        return pathlib.Path(path)

    #: Deprecated spellings of :meth:`save_animation`, the contract's name for writing a sequence of frames
    #: to a file (#299). `animate` means a matplotlib `FuncAnimation` on static and a callback loop in 3-D, so
    #: the file-writing meaning takes the name it already had on those tiers; `to_gif` was this tier's own
    #: older name. Both forward and warn.
    save_gif = renamed_method(new="save_animation", old="save_gif", owner="WebMap")
    animate = renamed_method(new="save_animation", old="animate", owner="WebMap")
    to_gif = renamed_method(new="save_animation", old="to_gif", owner="WebMap")

    def _temporal_frames(self) -> list:
        """Return the visible-layer set for each time step, oldest first.

        Returns:
            One list of layer ids per step — the layers that must be visible in that frame.

        Raises:
            ValueError: when there is no series to animate, naming what to add.
        """
        config = getattr(self, "_temporal", None)
        layer_ids = list((config or {}).get("layer_ids") or [])
        if config is None or (config.get("mode") != "raster") or len(layer_ids) < 2:
            raise ValueError(
                "save_animation() needs a raster time series with at least two steps; add one with "
                "timeslider(collection). The vector time-slider filters a single layer, so its steps "
                "are not separately renderable."
            )
        return [[layer_id] for layer_id in layer_ids]

    def _frame_png(self, path: Any, visible: list, title: str) -> pathlib.Path:
        """Render one animation frame by showing only ``visible`` and screenshotting the page.

        Args:
            path: Where to write this frame.
            visible: The layer ids to show; every other step layer is hidden.
            title: HTML document title used while rendering.

        Returns:
            The frame's :class:`pathlib.Path`.
        """
        config = self._temporal or {}
        steps = list(config.get("layer_ids") or [])
        widget = self._build_map_widget(with_controls=False)
        for layer_id in steps:
            widget.set_visibility(layer_id, layer_id in visible)
        return self._render_png(str(path), title=title, widget=widget, kind="GIF")

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
