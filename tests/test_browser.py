"""Tests for RP.11 — static HTML gallery (digitalearth.browser)."""

import matplotlib.pyplot as plt
import pytest

from digitalearth.browser import gallery


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
        assert "Lisbon DEM" in text and "repeat(5, 1fr)" in text

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
            [png], tmp_path / "x.html", title="Tom & Jerry <hi>", captions=['x" onerror="alert(1)']
        )
        text = page.read_text(encoding="utf-8")
        assert 'onerror="alert(1)' not in text, "an injected attribute must not survive verbatim"
        assert "Tom & Jerry <hi>" not in text, "raw special characters in the title must be escaped"
        assert "Tom &amp; Jerry &lt;hi&gt;" in text, "the title should be present in escaped form"
        assert "onerror=&quot;alert(1)" in text, "the caption should be present in escaped form"
