"""browser — assemble rendered outputs into a static, self-contained HTML page (RP.11).

This is the "browser frame" of earthkit-plots: one standalone ``.html`` file, with no external assets, that
opens in any browser and can be emailed or archived as-is. :func:`gallery` lays the outputs
:class:`~digitalearth.ops.batch.Batch` produced into a responsive CSS grid, embedding each one by *what it is*
(TD-22): a raster (PNG/JPEG/GIF/WebP/SVG) as an ``<img>`` base64 data URI, and an HTML page — as the
web/interactive/3-D backends save — as an ``<iframe srcdoc>`` carrying the page's own markup, so a
mixed-backend batch renders every tile in place. Anything else becomes a labelled placeholder that links the
file by name rather than a broken tile.
"""

import html
import logging
from base64 import b64encode
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

__all__ = ["gallery"]

logger = logging.getLogger(__name__)

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #fafafa; color: #222; }}
  h1 {{ font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat({columns}, 1fr); gap: 1rem; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: .5rem; }}
  figure img {{ width: 100%; height: auto; display: block; }}
  figure iframe {{ width: 100%; height: 22rem; border: 0; display: block; background: #fff; }}
  figure.placeholder {{ display: flex; flex-direction: column; justify-content: center; min-height: 8rem; }}
  figure.placeholder p {{ color: #999; margin: 0 0 .4rem; font-size: .85rem; text-align: center; }}
  figcaption {{ font-size: .85rem; color: #555; margin-top: .4rem; text-align: center; word-break: break-all; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


#: Raster suffixes embedded as an ``<img>`` base64 data URI, mapped to the media type each ``data:`` URI needs.
#: SVG is embedded the same way (``data:image/svg+xml``) rather than inlined into the page: an ``<img>`` renders
#: it in a restricted image context — so a ``<script>`` inside the SVG cannot run in the gallery's own origin
#: and its element ids cannot collide with the page's — while keeping every card uniform and the file standalone.
_RASTER_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

#: HTML-page suffixes embedded as an ``<iframe srcdoc>`` (what the web/interactive/3-D backends save).
_HTML_SUFFIXES = frozenset({".html", ".htm"})

#: The ``sandbox`` token set every embedded HTML page's ``<iframe>`` carries (M4). ``allow-scripts`` lets a
#: deck.gl/MapLibre page's own scripts run — they must, to draw the map — while the absence of
#: ``allow-same-origin`` keeps the page in an opaque origin with no reach to ``window.parent`` (the gallery) or
#: its sibling frames. The two together would let the framed document escape its sandbox, so
#: ``allow-same-origin`` is never added; a page needing genuine same-origin storage cannot be framed, which map
#: tiles never need. This makes the ``<iframe>`` path as script-safe as the SVG-via-``<img>`` path already is.
_IFRAME_SANDBOX = "allow-scripts"

#: The placeholder message for a type the gallery has no inline preview for (the original default).
_NO_PREVIEW_NOTE = "No inline preview for this file type."

#: The placeholder message for a tile that could not be read or decoded (L1). The specific error goes to the
#: log — where the file name is also named — rather than into the page, so an exception string is never
#: interpolated into the gallery's markup.
_UNREADABLE_NOTE = "Could not read this file for preview."

#: Bytes per mebibyte, for the size threshold and the message it prints.
_BYTES_PER_MB = 1024 * 1024

#: The size above which an HTML page is **linked** rather than inlined into ``srcdoc`` (L2). Each HTML tile's
#: whole markup is inlined (escaping can inflate it further) and ``loading="lazy"`` defers only *rendering*,
#: not the payload — the bytes all live in the one output file — so without a ceiling a batch of rich deck.gl
#: pages grows the single gallery without bound. The gallery's worth is being one self-contained, openable
#: file: an ordinary deck.gl/MapLibre page is a few MB and still inlines (keeping that property), while a
#: page past this ceiling degrades to a placeholder that names its size and links it, so one pathological page
#: cannot bloat the whole gallery past usefulness. Linking trades the self-contained property for that one
#: tile (the linked page must travel beside the gallery to open) — the lesser cost, since inlining it would
#: degrade every tile. It is a module constant so the limit lives in one named place and tests can adjust it.
_MAX_INLINE_HTML_BYTES = 5 * _BYTES_PER_MB


def _image_card(image: Path, media_type: str, safe_caption: str) -> str:
    """Base64-embed one raster into an ``<img>`` ``<figure>`` card (no external file reference)."""
    data = b64encode(image.read_bytes()).decode("ascii")
    return (
        f'  <figure><img alt="{safe_caption}" src="data:{media_type};base64,{data}">'
        f"<figcaption>{safe_caption}</figcaption></figure>"
    )


def _page_card(page: Path, safe_caption: str) -> str:
    """Embed one HTML page into a sandboxed ``<iframe srcdoc>`` card so it renders in place.

    The page's whole markup is HTML-escaped with ``quote=True`` — its ``"`` become ``&quot;`` and its
    ``<``/``>`` become ``&lt;``/``&gt;`` — and placed in the ``srcdoc`` attribute; the browser un-escapes it
    back into the iframe's own document, so the card renders the real page while the gallery stays one
    self-contained file (a linked ``src=`` would need the page as a sibling asset and break that). A rich page
    (e.g. a deck.gl scene) is inlined in full up to :data:`_MAX_INLINE_HTML_BYTES`; a larger one is **not**
    inlined but degraded to a placeholder that names its size and links it (logged at WARNING), so one giant
    page cannot bloat the whole gallery (L2). The size is read with ``stat`` before the file is opened; a
    missing/locked file surfaces here as the ``OSError`` :func:`_card` turns into the L1 placeholder.

    The iframe carries ``sandbox="allow-scripts"`` (:data:`_IFRAME_SANDBOX`) so a deck.gl/MapLibre page's own
    scripts still run — they must, to draw the map — but the page loads into an **opaque origin**: it cannot
    reach ``window.parent`` (the gallery) or its sibling frames, so a tile whose markup derives from untrusted
    source data (feature attributes interpolated into popups/labels) cannot script the rest of the page. This
    mirrors the script-safety the SVG-via-``<img>`` path already has (see :data:`_RASTER_MEDIA_TYPES`).
    ``allow-same-origin`` is deliberately **not** granted: combined with ``allow-scripts`` it would let the
    framed document reach out of its own sandbox, defeating the isolation. The trade-off is that a page needing
    genuine same-origin storage (cookies / ``localStorage`` / IndexedDB) will not work when framed — map tiles
    do not need it, so the isolation is the right default.
    """
    size = page.stat().st_size
    if size > _MAX_INLINE_HTML_BYTES:
        logger.warning(
            "gallery: %s is %.1f MB (over the %.1f MB inline limit) — linking it instead of inlining",
            page.name,
            size / _BYTES_PER_MB,
            _MAX_INLINE_HTML_BYTES / _BYTES_PER_MB,
        )
        note = f"Page too large to inline ({size / _BYTES_PER_MB:.1f} MB) — linked instead."
        return _placeholder_card(page, safe_caption, note=note)
    srcdoc = html.escape(page.read_text(encoding="utf-8"), quote=True)
    return (
        f'  <figure><iframe title="{safe_caption}" srcdoc="{srcdoc}"'
        f' sandbox="{_IFRAME_SANDBOX}" loading="lazy"></iframe>'
        f"<figcaption>{safe_caption}</figcaption></figure>"
    )


def _placeholder_card(
    output: Path, safe_caption: str, note: str = _NO_PREVIEW_NOTE
) -> str:
    """Emit a labelled placeholder that links ``output`` by name, carrying ``note`` as its message.

    Args:
        output: The file the placeholder stands in for and links by name.
        safe_caption: The already-escaped caption shown under the card.
        note: The one-line reason shown in the card — the default "no inline preview" for a type the
            gallery cannot embed, or a read-failure / oversize message for a tile it declined to inline
            (:data:`_UNREADABLE_NOTE`, L1/L2). Escaped here so a message is never trusted as raw markup.
    """
    href = html.escape(output.name, quote=True)
    safe_note = html.escape(note, quote=True)
    return (
        f'  <figure class="placeholder"><p>{safe_note}</p>'
        f'<figcaption><a href="{href}">{safe_caption}</a></figcaption></figure>'
    )


def _card(output: Path, caption: str) -> str:
    """Render one output as a ``<figure>`` card chosen by what the file is (TD-22).

    The dispatch is by file suffix: a raster (:data:`_RASTER_MEDIA_TYPES`) embeds as an ``<img>`` base64 data
    URI, an HTML page (:data:`_HTML_SUFFIXES`) embeds as an ``<iframe srcdoc>`` so it renders in place, and any
    other type gets a placeholder card that links the file by name rather than crashing. The caption (often a
    file name) is HTML-escaped with ``quote=True`` first, so a value containing ``&``/``<``/``>``/``"`` cannot
    break the markup or inject attributes/scripts.

    A tile whose bytes cannot be read (missing/locked file, ``OSError``) or whose HTML cannot be decoded as
    UTF-8 (``UnicodeDecodeError``) does not abort the whole gallery (L1): it is logged at WARNING — naming the
    file — and degraded to a placeholder card, so one bad output never loses the tiles that were fine. This
    matches the skip-and-warn tolerance the rest of ``ops`` already uses (e.g. ``Batch.run``'s name-collision
    warning, the web tier's ``_skipped``).
    """
    safe = html.escape(caption, quote=True)
    suffix = output.suffix.lower()
    media_type = _RASTER_MEDIA_TYPES.get(suffix)
    try:
        if media_type is not None:
            return _image_card(output, media_type, safe)
        if suffix in _HTML_SUFFIXES:
            return _page_card(output, safe)
    except (OSError, UnicodeDecodeError) as error:
        logger.warning(
            "gallery: could not read %s (%s: %s) — using a placeholder card",
            output.name,
            type(error).__name__,
            error,
        )
        return _placeholder_card(output, safe, note=_UNREADABLE_NOTE)
    return _placeholder_card(output, safe)


def gallery(
    images: Iterable[Any],
    path: Any,
    *,
    title: str = "digitalearth gallery",
    columns: int = 3,
    captions: Optional[Sequence[str]] = None,
) -> Path:
    """Build a standalone HTML gallery embedding ``images`` and write it to ``path``.

    Args:
        images: Iterable of rendered-output paths (``str``/``Path``) to embed, in display order. Each is
            embedded by what it is (TD-22): a raster (``.png``/``.jpg``/``.jpeg``/``.gif``/``.webp``/``.svg``)
            as an ``<img>`` base64 data URI, an HTML page (``.html``/``.htm``, as the web/interactive/3-D
            backends save) as a sandboxed ``<iframe srcdoc>``, and any other type as a placeholder card
            linking the file by name. A rich HTML page is inlined in full up to a size limit, past which it is
            linked instead so one giant page cannot bloat the gallery (L2); a tile that cannot be read or
            decoded degrades to a placeholder rather than aborting the build (L1).
        path: Output ``.html`` file path (parent directories are created if missing).
        title: Page heading and ``<title>``.
        columns: Number of columns in the responsive grid.
        captions: Caption per image; defaults to each image's file name. Must match ``images`` in length
            when supplied.

    Returns:
        The written HTML file path.

    Raises:
        ValueError: If ``captions`` is given but its length does not match ``images``.

    Examples:
        - Embed one rendered PNG into a self-contained page:
            ```python
            >>> import matplotlib, tempfile
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from pathlib import Path
            >>> from digitalearth.ops.browser import gallery
            >>> d = Path(tempfile.mkdtemp())
            >>> fig = plt.figure(); _ = fig.subplots().plot([0, 1], [1, 0]); img = d / "a.png"
            >>> fig.savefig(img); plt.close(fig)
            >>> html = gallery([img], d / "index.html", title="demo")
            >>> html.name
            'index.html'
            >>> text = html.read_text()
            >>> "demo" in text and "data:image/png;base64," in text
            True

            ```
        - Special characters in titles/captions are HTML-escaped, so the page is safe to open in a browser:
            ```python
            >>> import matplotlib, tempfile
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from pathlib import Path
            >>> from digitalearth.ops.browser import gallery
            >>> d = Path(tempfile.mkdtemp())
            >>> fig = plt.figure(); _ = fig.subplots().plot([0, 1], [1, 0]); img = d / "a.png"
            >>> fig.savefig(img); plt.close(fig)
            >>> page = gallery([img], d / "g.html", title="Tom & Jerry", captions=['a" b'])
            >>> "Tom &amp; Jerry" in page.read_text() and 'a&quot; b' in page.read_text()
            True

            ```

    See Also:
        digitalearth.ops.batch.Batch: produces the PNG series this page typically embeds.
    """
    images = [Path(p) for p in images]
    if captions is not None and len(captions) != len(images):
        raise ValueError(
            f"captions ({len(captions)}) must match images ({len(images)})"
        )
    labels = list(captions) if captions is not None else [p.name for p in images]
    cards = "\n".join(_card(img, label) for img, label in zip(images, labels))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = _PAGE.format(
        title=html.escape(title, quote=True), columns=columns, cards=cards
    )
    out.write_text(page, encoding="utf-8")
    return out
