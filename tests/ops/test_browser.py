"""Tests for RP.11 — static HTML gallery (digitalearth.ops.browser)."""

import html
import logging
import re

import matplotlib.pyplot as plt
import pytest

import digitalearth.ops.browser as browser
from digitalearth.ops.browser import gallery


def _srcdoc(text: str) -> str:
    """Return the first ``srcdoc`` attribute value found in ``text``.

    The value is HTML-escaped page markup, so it contains no raw ``"`` (they are ``&quot;``); a
    non-greedy ``[^"]*`` therefore captures the whole attribute. Fails the test loudly if absent.

    Args:
        text: The generated gallery HTML.

    Returns:
        str: The raw (still-escaped) contents of the first ``srcdoc="..."`` attribute.
    """
    match = re.search(r'srcdoc="([^"]*)"', text)
    assert match is not None, (
        "expected an iframe with a srcdoc attribute in the gallery"
    )
    return match.group(1)


@pytest.fixture
def png(tmp_path):
    """Write a tiny real PNG and return its path.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path: A small on-disk PNG suitable for embedding.
    """
    fig = plt.figure()
    fig.subplots().plot([0, 1], [1, 0])
    out = tmp_path / "plot.png"
    fig.savefig(out)
    plt.close(fig)
    return out


class TestGallery:
    """Tests for gallery."""

    def test_writes_self_contained_page(self, tmp_path, png):
        """The page embeds the PNG as base64 (no external file reference) and carries the title."""
        page = gallery([png], tmp_path / "index.html", title="my maps")
        text = page.read_text(encoding="utf-8")
        assert page.name == "index.html"
        assert "data:image/png;base64," in text, "image should be embedded, not linked"
        assert "my maps" in text, "title should appear in the page"
        assert png.name in text, "default caption is the file name"

    def test_custom_captions_and_columns(self, tmp_path, png):
        """Custom captions replace file names and the column count reaches the CSS grid."""
        page = gallery([png], tmp_path / "g.html", columns=5, captions=["Lisbon DEM"])
        text = page.read_text(encoding="utf-8")
        assert "Lisbon DEM" in text
        assert "repeat(5, 1fr)" in text

    def test_creates_parent_directories(self, tmp_path, png):
        """gallery creates missing parent directories of the output path."""
        page = gallery([png], tmp_path / "deep" / "nested" / "index.html")
        assert page.exists(), "the nested output file should be written"

    def test_caption_length_mismatch_raises(self, tmp_path, png):
        """A captions list that does not match the images count raises ValueError."""
        with pytest.raises(ValueError, match=r"captions .* must match images") as exc:
            gallery([png], tmp_path / "x.html", captions=["a", "b"])
        assert "must match" in str(exc.value), f"unexpected message: {exc.value}"

    def test_html_special_characters_are_escaped(self, tmp_path, png):
        """Titles and captions are HTML-escaped so they cannot inject markup or attributes (M1)."""
        page = gallery(
            [png],
            tmp_path / "x.html",
            title="Tom & Jerry <hi>",
            captions=['x" onerror="alert(1)'],
        )
        text = page.read_text(encoding="utf-8")
        assert 'onerror="alert(1)' not in text, (
            "an injected attribute must not survive verbatim"
        )
        assert "Tom & Jerry <hi>" not in text, (
            "raw special characters in the title must be escaped"
        )
        assert "Tom &amp; Jerry &lt;hi&gt;" in text, (
            "the title should be present in escaped form"
        )
        assert "onerror=&quot;alert(1)" in text, (
            "the caption should be present in escaped form"
        )


@pytest.fixture
def web_page(tmp_path):
    """Write a tiny standalone HTML page (as a web/interactive/3-D backend would) and return its path.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path: An on-disk ``.html`` page carrying a known ``<h1>`` heading.
    """
    out = tmp_path / "map.html"
    out.write_text(
        "<!doctype html><html><body><h1>Lisbon deck.gl</h1></body></html>",
        encoding="utf-8",
    )
    return out


@pytest.fixture
def mystery_file(tmp_path):
    """Write a file with an extension the gallery cannot embed and return its path.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path: An on-disk ``.bin`` file whose type has no inline preview.
    """
    out = tmp_path / "scene.bin"
    out.write_bytes(b"\x00\x01\x02not an image or a page")
    return out


class TestGalleryBackendDispatch:
    """The card for each output is chosen by what the file is: raster, HTML page, or neither (TD-22)."""

    def test_png_embeds_as_img_data_uri(self, tmp_path, png):
        """A raster output still embeds as an ``<img>`` base64 data URI — not an iframe."""
        page = gallery([png], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        assert "<img " in text, "a PNG should render as an <img> element"
        assert "data:image/png;base64," in text, "the PNG should be a base64 data URI"
        assert "<iframe" not in text, "a raster must not be wrapped in an iframe"

    def test_html_output_embeds_as_iframe_srcdoc(self, tmp_path, web_page):
        """An HTML output embeds as an ``<iframe srcdoc>`` whose value is the page's escaped markup."""
        page = gallery([web_page], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        assert "<iframe" in text, "an HTML page must render in an iframe, not an <img>"
        assert "data:image/png;base64," not in text, (
            "the HTML page must not be base64-embedded as a broken PNG tile"
        )
        # The source markup is escaped into the attribute: its <h1> becomes &lt;h1&gt; there.
        assert "&lt;h1&gt;Lisbon deck.gl&lt;/h1&gt;" in text, (
            "the page's own markup should be HTML-escaped into the srcdoc attribute"
        )
        # Round trip: un-escaping the srcdoc attribute recovers the page's real markup.
        recovered = html.unescape(_srcdoc(text))
        assert "<h1>Lisbon deck.gl</h1>" in recovered, (
            "the iframe srcdoc must round-trip back to the page's own <h1>"
        )

    def test_mixed_batch_yields_both_card_types(self, tmp_path, png, web_page):
        """A batch of one PNG and one HTML page produces both an <img> card and an <iframe> card."""
        page = gallery([png, web_page], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        assert "<img " in text, "the PNG half of the batch should be an <img> card"
        assert "<iframe" in text, (
            "the HTML half of the batch should be an <iframe> card"
        )

    def test_unknown_extension_gets_placeholder_not_crash(self, tmp_path, mystery_file):
        """An unembeddable extension yields a placeholder card linking the file by name, not an error."""
        page = gallery([mystery_file], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        assert "<img " not in text, "an unknown type must not be embedded as an image"
        assert "<iframe" not in text, "an unknown type must not be wrapped in an iframe"
        assert "data:" not in text, "an unknown type must not be base64-embedded at all"
        assert 'href="scene.bin"' in text, (
            "the placeholder should link the file by name"
        )


class TestGalleryIframeSandbox:
    """The HTML iframe is sandboxed so an embedded page's scripts cannot reach the gallery or siblings (M4)."""

    def test_html_iframe_carries_a_sandbox_attribute(self, tmp_path, web_page):
        """An HTML page's ``<iframe>`` is emitted sandboxed, not as an unrestricted frame."""
        page = gallery([web_page], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        iframe = re.search(r"<iframe[^>]*>", text)
        assert iframe is not None, "the HTML page should render in an <iframe>"
        assert "sandbox=" in iframe.group(0), (
            f"the iframe must be sandboxed, got {iframe.group(0)!r}"
        )

    def test_sandbox_grants_scripts_only_not_same_origin(self, tmp_path, web_page):
        """The sandbox grants exactly ``allow-scripts`` — deck.gl/MapLibre render, but with no same-origin reach.

        ``allow-scripts`` together with ``allow-same-origin`` would let the framed page remove its own
        sandbox, so the token set must hold the former and never the latter.
        """
        page = gallery([web_page], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        match = re.search(r'<iframe[^>]*\bsandbox="([^"]*)"', text)
        assert match is not None, "the iframe must carry a sandbox attribute"
        granted = match.group(1).split()
        assert "allow-scripts" in granted, (
            "a deck.gl/MapLibre page needs scripts to render, so allow-scripts is required"
        )
        assert "allow-same-origin" not in granted, (
            "allow-scripts + allow-same-origin together would defeat the sandbox"
        )
        # Build the expected token set independently of the generated attribute (not a value == itself check).
        assert set(granted) == {"allow-scripts"}, (
            f"the sandbox should grant only allow-scripts, got {granted!r}"
        )


@pytest.fixture
def bad_utf8_html(tmp_path):
    """Write a ``.html`` file whose bytes are not valid UTF-8, so decoding it raises.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path: an on-disk ``.html`` file that ``read_text(encoding="utf-8")`` cannot decode.
    """
    out = tmp_path / "latin1.html"
    # 0xff is never a valid UTF-8 lead byte — a page saved in another encoding decodes to a UnicodeDecodeError.
    out.write_bytes(b"<html><body>caf\xe9 \xff\xfe</body></html>")
    return out


@pytest.fixture
def missing_png(tmp_path):
    """Return a ``.png`` path that does not exist, so reading its bytes raises ``OSError``.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path: a ``.png`` path naming no file on disk.
    """
    return tmp_path / "gone.png"


class TestGalleryPerTileTolerance:
    """One unreadable/undecodable tile becomes a placeholder rather than aborting the whole build (L1)."""

    def test_undecodable_html_tile_does_not_lose_the_good_tile(
        self, tmp_path, png, bad_utf8_html
    ):
        """A batch with an undecodable HTML page still renders the good raster tile beside a placeholder."""
        page = gallery([png, bad_utf8_html], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        assert "data:image/png;base64," in text, (
            "the good PNG tile must survive an undecodable sibling"
        )
        assert 'href="latin1.html"' in text, (
            "the undecodable page should degrade to a placeholder linking the file by name"
        )
        assert "srcdoc=" not in text, (
            "an undecodable page must not be inlined as an iframe srcdoc"
        )

    def test_unreadable_image_tile_does_not_lose_the_good_tile(
        self, tmp_path, web_page, missing_png
    ):
        """A batch with a missing raster still renders the good HTML tile beside a placeholder."""
        page = gallery([web_page, missing_png], tmp_path / "index.html")
        text = page.read_text(encoding="utf-8")
        assert "<iframe" in text, "the good HTML tile must survive an unreadable sibling"
        assert 'href="gone.png"' in text, (
            "the unreadable raster should degrade to a placeholder linking the file by name"
        )
        assert "data:image/png;base64," not in text, (
            "a raster that could not be read must not be embedded as a broken data URI"
        )

    def test_unreadable_tile_logs_a_warning_naming_the_file(
        self, tmp_path, bad_utf8_html, caplog
    ):
        """The skipped tile is announced with a WARNING naming the file, matching ops' skip-and-warn style."""
        with caplog.at_level(logging.WARNING, logger="digitalearth.ops.browser"):
            gallery([bad_utf8_html], tmp_path / "index.html")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "an unreadable tile should be logged, not swallowed silently"
        assert any(bad_utf8_html.name in r.getMessage() for r in warnings), (
            f"the warning should name the file; got {[r.getMessage() for r in warnings]}"
        )
