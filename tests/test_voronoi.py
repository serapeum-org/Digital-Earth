"""Tests for Map.voronoi / api.voronoi — Voronoi diagram from a pyramids FeatureCollection of points."""

import pytest

from digitalearth.scene import Map


@pytest.fixture
def points_fc():
    """The point fixture as a pyramids FeatureCollection (EPSG:32618, with a numeric 'fid' column)."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


def test_voronoi_filled(points_fc):
    """voronoi fills the cells coloured by the chosen column and registers one layer."""
    m = Map(crs=points_fc.epsg)
    pc = m.voronoi(points_fc, column="fid")
    assert len(m.layers) == 1
    assert m.ax.collections
    assert len(pc.get_paths()) >= 1


def test_voronoi_outline(points_fc):
    """voronoi without a column draws cell outlines only (no value mapping)."""
    m = Map(crs=points_fc.epsg)
    m.voronoi(points_fc)
    assert len(m.layers) == 1
    assert m.ax.collections


def test_voronoi_one_cell_per_point(points_fc):
    """ordered tessellation in a projected CRS yields exactly one finite cell per input point."""
    m = Map(crs=points_fc.epsg)
    pc = m.voronoi(points_fc, column="fid")
    assert len(pc.get_paths()) == len(points_fc)


def test_voronoi_clip_trims_cells(points_fc):
    """A clip boundary (the points' convex hull) keeps the diagram drawable and bounded."""
    hull = points_fc.geometry.union_all().convex_hull
    m = Map(crs=points_fc.epsg)
    pc = m.voronoi(points_fc, column="fid", clip=hull)
    assert len(m.layers) == 1
    assert len(pc.get_paths()) >= 1


def test_voronoi_requires_points(points_fc):
    """voronoi rejects non-point geometry with a clear error."""
    polys = points_fc.copy()
    polys["geometry"] = polys.geometry.buffer(100.0)
    m = Map(crs=points_fc.epsg)
    with pytest.raises(ValueError):
        m.voronoi(polys)


def test_api_voronoi(points_fc):
    """api.voronoi builds a finished Map with the Voronoi layer."""
    from digitalearth import api

    m = api.voronoi(points_fc, column="fid", crs=points_fc.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1
