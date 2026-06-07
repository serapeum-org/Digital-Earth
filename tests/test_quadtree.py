"""Tests for Map.quadtree / api.quadtree — quadtree choropleth from a pyramids FeatureCollection of points."""

import numpy as np
import pytest

from digitalearth.scene import Map


@pytest.fixture
def points_fc():
    """The point fixture as a pyramids FeatureCollection (EPSG:32618, with a numeric 'fid' column)."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


def test_quadtree_density(points_fc):
    """quadtree with no column colours cells by point count and registers one layer."""
    m = Map(crs=points_fc.epsg)
    pc = m.quadtree(points_fc, nmax=1)
    assert len(m.layers) == 1
    assert m.ax.collections
    assert len(pc.get_paths()) >= 1


def test_quadtree_aggregate(points_fc):
    """quadtree aggregates a column per cell (mean) without error."""
    m = Map(crs=points_fc.epsg)
    m.quadtree(points_fc, column="fid", agg="mean", nmax=1)
    assert len(m.layers) == 1


def test_quadtree_nmax_controls_resolution(points_fc):
    """A small nmax produces more cells than an nmax that exceeds the point count (single cell)."""
    fine = Map(crs=points_fc.epsg).quadtree(points_fc, nmax=1)
    coarse = Map(crs=points_fc.epsg).quadtree(points_fc, nmax=10**6)
    assert len(coarse.get_paths()) == 1
    assert len(fine.get_paths()) > len(coarse.get_paths())


def _drawn_area(pc):
    """Total area of the polygons actually drawn by a PolyCollection (via matplotlib paths)."""
    from shapely.geometry import Polygon

    area = 0.0
    for path in pc.get_paths():
        for ring in path.to_polygons():
            if len(ring) >= 4:
                area += Polygon(ring).area
    return area


def test_quadtree_clip(points_fc):
    """A convex-hull clip actually trims the cells: clipped area < unclipped area."""
    hull = points_fc.geometry.union_all().convex_hull
    clipped = Map(crs=points_fc.epsg).quadtree(points_fc, nmax=1, clip=hull)
    unclipped = Map(crs=points_fc.epsg).quadtree(points_fc, nmax=1)
    assert len(clipped.get_paths()) >= 1
    assert _drawn_area(clipped) < _drawn_area(unclipped)


def test_quadtree_requires_points(points_fc):
    """quadtree rejects non-point geometry with a clear error."""
    polys = points_fc.copy()
    polys["geometry"] = polys.geometry.buffer(100.0)
    m = Map(crs=points_fc.epsg)
    with pytest.raises(ValueError):
        m.quadtree(polys)


def test_quadtree_unknown_agg(points_fc):
    """An unknown agg name is rejected."""
    m = Map(crs=points_fc.epsg)
    with pytest.raises(ValueError):
        m.quadtree(points_fc, column="fid", agg="bogus")


def test_quadtree_cells_respect_nmax():
    """_quadtree_cells never emits a cell holding more than nmax points (on splittable input)."""
    rng = np.random.default_rng(0)
    xs = rng.random(200)
    ys = rng.random(200)
    cells = Map._quadtree_cells(xs, ys, agg_fn=lambda idx: float(len(idx)), nmax=10, nmin=0)
    assert cells, "expected at least one cell"
    assert all(val <= 10 for *_bbox, val in cells)  # value == count when agg_fn is len


def test_api_quadtree(points_fc):
    """api.quadtree builds a finished Map with the quadtree layer."""
    from digitalearth import api

    m = api.quadtree(points_fc, nmax=1, crs=points_fc.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1


def test_quadtree_callable_agg(points_fc):
    """quadtree accepts a callable ``agg`` (custom per-cell reducer), not just named ones.

    Test scenario:
        Passing ``agg=lambda a: float(np.max(a))`` exercises the ``callable(agg)`` branch and renders a layer
        whose cell values never exceed the column max.
    """
    m = Map(crs=points_fc.epsg)
    fid_max = float(np.asarray(points_fc["fid"], dtype=float).max())
    pc = m.quadtree(points_fc, column="fid", agg=lambda a: float(np.max(a)), nmax=1)
    assert len(m.layers) == 1, "callable agg should still produce one layer"
    vals = np.ma.filled(np.asarray(pc.get_array(), dtype="float64"), np.nan)
    assert np.nanmax(vals) <= fid_max + 1e-9, f"max-agg cells exceed column max: {np.nanmax(vals)} > {fid_max}"


def test_quadtree_cells_coincident_points_terminate():
    """_quadtree_cells terminates on coincident points (the no-progress branch), not infinite recursion.

    Test scenario:
        Many identical points with nmax=1 cannot be split further; the recursion must stop and emit a single
        cell holding all of them rather than recursing forever.
    """
    xs = np.zeros(5)
    ys = np.zeros(5)
    cells = Map._quadtree_cells(xs, ys, agg_fn=lambda idx: float(len(idx)), nmax=1, nmin=0)
    assert len(cells) == 1, f"coincident points should collapse to one cell, got {len(cells)}"
    assert cells[0][4] == 5.0, f"the single cell should hold all 5 points, got value {cells[0][4]}"


def test_quadtree_cells_nmin_drops_sparse_cells():
    """_quadtree_cells drops cells with fewer than ``nmin`` points.

    Test scenario:
        With nmin larger than the point count, every candidate cell is below threshold and dropped (empty
        result), exercising the nmin guard.
    """
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 1.0, 2.0, 3.0])
    cells = Map._quadtree_cells(xs, ys, agg_fn=lambda idx: float(len(idx)), nmax=1, nmin=99)
    assert cells == [], f"all cells should be dropped below nmin, got {cells}"
