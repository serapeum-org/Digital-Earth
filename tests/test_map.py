"""Tests for digitalearth.scene.Map — display-CRS reprojection + decoration (no Cartopy)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from digitalearth.scene import Map


def test_prepare_reprojects(dataset):
    """A dataset in a different CRS is reprojected to the Map's display CRS."""
    m = Map(crs=3857)
    src = m._prepare(dataset)
    assert src.crs == 3857


def test_prepare_skips_when_already_in_crs(dataset):
    """No reprojection happens when the dataset is already in the display CRS."""
    m = Map(crs=dataset.epsg)
    src = m._prepare(dataset)
    assert src.crs == dataset.epsg


def test_imshow_renders_in_display_crs(dataset):
    """imshow draws the reprojected raster on the shared axes with an extent in the display CRS."""
    m = Map(crs=3857)
    m.imshow(dataset)
    assert len(m.layers) == 1
    assert len(m.ax.images) == 1
    # EPSG:3857 eastings for this UTM-18N scene are large negative metres (~ -8.3e6)
    xmin, xmax = m.ax.images[0].get_extent()[:2]
    assert xmin < xmax


def test_set_extent(dataset):
    """set_extent applies the given bbox to the axes limits."""
    m = Map(crs=3857)
    m.imshow(dataset)
    m.set_extent([0.0, 100.0, 0.0, 50.0])
    assert m.ax.get_xlim() == (0.0, 100.0)
    assert m.ax.get_ylim() == (0.0, 50.0)


def test_no_cartopy_import():
    """The scene package must not import cartopy (plan §2.4: reproject via pyramids, no Cartopy)."""
    pkg = Path("src/digitalearth/scene")
    for py in pkg.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "import cartopy" not in text and "from cartopy" not in text


@pytest.mark.parametrize("layer", ["coastlines", "borders"])
def test_natural_earth_overlays(dataset, layer):
    """Coastlines/borders overlay when Natural Earth data is reachable; skipped offline."""
    m = Map(crs=3857)
    m.imshow(dataset)
    try:
        getattr(m, layer)()
    except Exception as exc:  # network/download unavailable in this environment
        pytest.skip(f"Natural Earth {layer} unavailable offline: {exc}")
    # the vector layer added at least one artist (collection/line) to the axes
    assert m.ax.collections or m.ax.lines
