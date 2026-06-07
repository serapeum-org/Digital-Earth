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


def _drawn_area(pc):
    """Total area of the polygons actually drawn by a PolyCollection (via matplotlib paths)."""
    from shapely.geometry import Polygon

    area = 0.0
    for path in pc.get_paths():
        for ring in path.to_polygons():
            if len(ring) >= 4:
                area += Polygon(ring).area
    return area


def test_voronoi_clip_trims_cells(points_fc):
    """A convex-hull clip actually trims the cells: clipped area < unclipped area, all within the hull."""
    import numpy as np
    from shapely.geometry import MultiPoint

    hull = points_fc.geometry.union_all().convex_hull
    clipped = Map(crs=points_fc.epsg).voronoi(points_fc, column="fid", clip=hull)
    unclipped = Map(crs=points_fc.epsg).voronoi(points_fc, column="fid")
    assert _drawn_area(clipped) < _drawn_area(unclipped)
    # every clipped vertex lies within the hull (allowing a tiny numeric tolerance)
    verts = np.concatenate([p.vertices for p in clipped.get_paths()])
    tol = 1e-6 * max(hull.bounds[2] - hull.bounds[0], hull.bounds[3] - hull.bounds[1])
    assert hull.buffer(tol).covers(MultiPoint([tuple(v) for v in verts]))


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


def test_voronoi_drops_nonfinite_on_globe():
    """Far-side points (non-finite in a clipped/globe CRS) are dropped, not crashed on (M1)."""
    import geopandas as gpd
    from shapely.geometry import Point
    from pyramids.feature import FeatureCollection

    # 180/150 lon are on the far hemisphere of an orthographic centred at (0, 0) -> reproject to inf.
    gdf = gpd.GeoDataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0]},
        geometry=[Point(0, 0), Point(30, 10), Point(180, 0), Point(150, -20)],
        crs="EPSG:4326",
    )
    m = Map(crs="+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84")
    pc = m.voronoi(FeatureCollection(gdf), column="v")
    assert len(m.layers) == 1
    assert len(pc.get_paths()) >= 1


def test_voronoi_empty_raises():
    """An empty FeatureCollection raises a clear error rather than an opaque one (L1)."""
    import geopandas as gpd
    from pyramids.feature import FeatureCollection

    empty = FeatureCollection(gpd.GeoDataFrame({"fid": []}, geometry=[], crs="EPSG:4326"))
    with pytest.raises(ValueError):
        Map(crs=4326).voronoi(empty)
