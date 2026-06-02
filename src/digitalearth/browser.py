"""browser — assemble rendered figures into a static, self-contained HTML page (RP.11).

This is the "browser frame" of earthkit-plots, deliberately kept **static**: there is no server and no
interactive backend (the interactive tier, RP.9, stays deferred). Figures are PNGs — typically produced by
:class:`~digitalearth.batch.Batch` — and :func:`gallery` base64-embeds them into one standalone ``.html``
file with a responsive CSS grid. The file has no external assets, so it opens in any browser and can be
emailed or archived as-is.
"""
from base64 import b64encode
import html
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


def _card(image: Path, caption: str) -> str:
    """Base64-embed one PNG into a ``<figure>`` card (no external file reference).

    The caption (often a file name) is HTML-escaped before interpolation so that a value containing
    ``&``/``<``/``>``/``"`` cannot break the markup or inject attributes/scripts.
    """
    data = b64encode(image.read_bytes()).decode("ascii")
    safe = html.escape(caption, quote=True)
    return (
        f'  <figure><img alt="{safe}" src="data:image/png;base64,{data}">'
        f"<figcaption>{safe}</figcaption></figure>"
    )


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
        images: Iterable of PNG file paths (``str``/``Path``) to embed, in display order.
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
            >>> from digitalearth.browser import gallery
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
    """
    images = [Path(p) for p in images]
    if captions is not None and len(captions) != len(images):
        raise ValueError(f"captions ({len(captions)}) must match images ({len(images)})")
    labels = list(captions) if captions is not None else [p.name for p in images]
    cards = "\n".join(_card(img, label) for img, label in zip(images, labels))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = _PAGE.format(title=html.escape(title, quote=True), columns=columns, cards=cards)
    out.write_text(page, encoding="utf-8")
    return out
