"""Tests for T2.2 — Map point/cell methods (scatter/grid_points/point_cloud/grid_cells)."""

from digitalearth.scene import Map


def test_scatter_features(dataset):
    """A FeatureCollection of points renders as a coloured scatter layer."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = Map(crs=fc.epsg)
    m.scatter(fc)
    assert len(m.layers) == 1
    assert len(m.ax.collections) >= 1


def test_grid_points(dataset):
    """grid_points scatters raster cell centres coloured by value."""
    m = Map(crs=dataset.epsg)
    m.grid_points(dataset)
    assert len(m.layers) == 1
    assert m.ax.collections


def test_point_cloud_is_grid_points_alias(dataset):
    """point_cloud renders the same way as grid_points."""
    m = Map(crs=dataset.epsg)
    m.point_cloud(dataset)
    assert len(m.layers) == 1


def test_grid_cells(dataset):
    """grid_cells draws one polygon per raster cell coloured by value."""
    m = Map(crs=dataset.epsg)
    m.grid_cells(dataset)
    assert len(m.layers) == 1
    # one PolyCollection holding all cells
    assert m.ax.collections


def test_grid_cells_polygon_count(dataset):
    """grid_cells produces exactly rows*columns polygons."""
    import numpy as np

    m = Map(crs=dataset.epsg)
    pc = m.grid_cells(dataset)
    paths = pc.get_paths()
    assert len(paths) == dataset.rows * dataset.columns
