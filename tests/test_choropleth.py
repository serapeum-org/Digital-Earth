"""Tests for T7.2 — Map choropleth/shapes from a pyramids FeatureCollection of polygons."""

import pytest

from digitalearth.scene import Map


@pytest.fixture
def polygons():
    """A polygon FeatureCollection built by buffering the point fixture, with a numeric 'fid' column.

    Returns:
        FeatureCollection: polygon geometries in EPSG:32618 with an 'fid' value column.
    """
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


def test_choropleth(polygons):
    """choropleth fills polygons coloured by the chosen column."""
    m = Map(crs=polygons.epsg)
    m.choropleth(polygons, column="fid")
    assert len(m.layers) == 1
    assert m.ax.collections


def test_shapes_outline(polygons):
    """shapes draws polygon outlines without a value mapping."""
    m = Map(crs=polygons.epsg)
    m.shapes(polygons)
    assert len(m.layers) == 1
    assert m.ax.collections


def test_choropleth_polygon_count(polygons):
    """choropleth draws at least one polygon per feature."""
    m = Map(crs=polygons.epsg)
    pc = m.choropleth(polygons, column="fid")
    assert len(pc.get_paths()) >= len(polygons)


def test_polygon_vertices_multipolygon():
    """_polygon_vertices expands a MultiPolygon into one ring per part."""
    from shapely.geometry import MultiPolygon, Polygon

    import geopandas as gpd

    p1 = Polygon([(0, 0), (1, 0), (1, 1)])
    p2 = Polygon([(2, 2), (3, 2), (3, 3)])
    mp = MultiPolygon([p1, p2])
    series = gpd.GeoSeries([mp])
    polys, repeats = Map._polygon_vertices(series)
    assert len(polys) == 2
    assert repeats == [2]
