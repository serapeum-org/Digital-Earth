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
from base64 import b64encode
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

__all__ = ["gallery"]

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


def _image_card(image: Path, media_type: str, safe_caption: str) -> str:
    """Base64-embed one raster into an ``<img>`` ``<figure>`` card (no external file reference)."""
    data = b64encode(image.read_bytes()).decode("ascii")
    return (
        f'  <figure><img alt="{safe_caption}" src="data:{media_type};base64,{data}">'
        f"<figcaption>{safe_caption}</figcaption></figure>"
    )


def _page_card(page: Path, safe_caption: str) -> str:
    """Embed one HTML page into an ``<iframe srcdoc>`` card so it renders in place.

    The page's whole markup is HTML-escaped with ``quote=True`` — its ``"`` become ``&quot;`` and its
    ``<``/``>`` become ``&lt;``/``&gt;`` — and placed in the ``srcdoc`` attribute; the browser un-escapes it
    back into the iframe's own document, so the card renders the real page while the gallery stays one
    self-contained file (a linked ``src=`` would need the page as a sibling asset and break that). A rich page
    (e.g. a deck.gl scene) can be megabytes and is inlined in full, so a gallery of many web pages grows to
    match.
    """
    srcdoc = html.escape(page.read_text(encoding="utf-8"), quote=True)
    return (
        f'  <figure><iframe title="{safe_caption}" srcdoc="{srcdoc}" loading="lazy"></iframe>'
        f"<figcaption>{safe_caption}</figcaption></figure>"
    )


def _placeholder_card(output: Path, safe_caption: str) -> str:
    """Emit a labelled placeholder that links ``output`` by name, for a type with no inline preview."""
    href = html.escape(output.name, quote=True)
    return (
        '  <figure class="placeholder"><p>No inline preview for this file type.</p>'
        f'<figcaption><a href="{href}">{safe_caption}</a></figcaption></figure>'
    )


def _card(output: Path, caption: str) -> str:
    """Render one output as a ``<figure>`` card chosen by what the file is (TD-22).

    The dispatch is by file suffix: a raster (:data:`_RASTER_MEDIA_TYPES`) embeds as an ``<img>`` base64 data
    URI, an HTML page (:data:`_HTML_SUFFIXES`) embeds as an ``<iframe srcdoc>`` so it renders in place, and any
    other type gets a placeholder card that links the file by name rather than crashing. The caption (often a
    file name) is HTML-escaped with ``quote=True`` first, so a value containing ``&``/``<``/``>``/``"`` cannot
    break the markup or inject attributes/scripts.
    """
    safe = html.escape(caption, quote=True)
    suffix = output.suffix.lower()
    media_type = _RASTER_MEDIA_TYPES.get(suffix)
    if media_type is not None:
        return _image_card(output, media_type, safe)
    if suffix in _HTML_SUFFIXES:
        return _page_card(output, safe)
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
            backends save) as an ``<iframe srcdoc>``, and any other type as a placeholder card linking the
            file by name. A rich HTML page is inlined in full, so a gallery of many web pages can grow large.
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
