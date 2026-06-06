"""Tests for Map.kde / api.kde — 2-D kernel-density plot from a pyramids FeatureCollection of points."""

import pytest

from digitalearth.scene import Map


@pytest.fixture
def points_fc():
    """The point fixture as a pyramids FeatureCollection (EPSG:32618)."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


def test_kde_plots(points_fc):
    """kde draws a density contour set and registers one layer."""
    m = Map(crs=points_fc.epsg)
    cs = m.kde(points_fc)
    assert len(m.layers) == 1
    assert m.ax.collections
    assert type(cs).__name__ in ("QuadContourSet", "ContourSet")


def test_kde_line_contours(points_fc):
    """kde with shade=False draws line contours (still one layer)."""
    m = Map(crs=points_fc.epsg)
    m.kde(points_fc, shade=False)
    assert len(m.layers) == 1


def test_kde_clip(points_fc):
    """A clip boundary keeps the density drawable."""
    hull = points_fc.geometry.union_all().convex_hull
    m = Map(crs=points_fc.epsg)
    m.kde(points_fc, clip=hull)
    assert len(m.layers) == 1


def test_kde_requires_points(points_fc):
    """kde rejects non-point geometry with a clear error."""
    polys = points_fc.copy()
    polys["geometry"] = polys.geometry.buffer(100.0)
    m = Map(crs=points_fc.epsg)
    with pytest.raises(ValueError):
        m.kde(polys)


def test_api_kde(points_fc):
    """api.kde builds a finished Map with the density layer."""
    from digitalearth import api

    m = api.kde(points_fc, crs=points_fc.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1
