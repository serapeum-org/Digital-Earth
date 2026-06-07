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


def test_grid_cells_nulls_nodata_cells():
    """grid_cells routes band values through the shared nodata mask (PA-7): nodata cells become NaN.

    Test scenario:
        A 2x2 raster with one cell equal to the nodata sentinel still draws four polygons, but exactly
        that one cell carries a non-finite (masked/NaN) face value.
    """
    import numpy as np
    from pyramids.dataset import Dataset

    arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
    ds = Dataset.create_from_array(
        arr=arr, geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326, no_data_value=-9999.0
    )
    m = Map(crs=4326)
    pc = m.grid_cells(ds)
    values = np.ma.filled(np.asarray(pc.get_array(), dtype="float64"), np.nan)
    assert len(pc.get_paths()) == 4, f"all four cells should be drawn, got {len(pc.get_paths())}"
    assert int(np.isfinite(values).sum()) == 3, f"exactly the nodata cell should be nulled, got {values}"
