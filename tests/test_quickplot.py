"""Tests for T9.1 — quickplot/quickmap one-call entry points and module functions."""
import matplotlib

matplotlib.use("Agg")

import pytest

import digitalearth
from digitalearth import api as qp
from digitalearth.scene import Map


def test_quickplot_returns_finished_map(dataset):
    """quickplot(dataset) returns a Map with one drawn layer and a colorbar."""
    m = digitalearth.quickplot(dataset, crs=dataset.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1
    assert len(m.fig.axes) == 2  # main axes + colorbar


def test_quickmap_kind_dispatch(dataset):
    """quickmap honours an explicit raster kind."""
    m = qp.quickmap(dataset, crs=dataset.epsg, kind="contourf")
    assert m.layers


def test_quickmap_saves_png(dataset, tmp_path):
    """The finished map can be saved to a PNG file."""
    m = qp.quickmap(dataset, crs=dataset.epsg)
    out = tmp_path / "quick.png"
    m.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_quickmap_scatter_features():
    """A FeatureCollection of points is drawn as a scatter map."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = qp.quickmap(fc, crs=fc.epsg)
    assert m.ax.collections


def test_quickmap_choropleth_polygons():
    """A polygon FeatureCollection with a column is drawn as a choropleth."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.quickmap(fc, crs=fc.epsg, column="fid")
    assert m.ax.collections


def test_quickmap_rejects_unsupported_type():
    """quickmap raises on an input type it cannot draw."""
    with pytest.raises(TypeError, match="cannot draw"):
        qp.quickmap("not data")


def test_module_function_contourf(dataset):
    """The module-level contourf builds a finished Map via the contourf kind."""
    m = qp.contourf(dataset, crs=dataset.epsg)
    assert m.layers


def test_module_function_grid_cells(dataset):
    """The module-level grid_cells builds a finished Map with a polygon layer."""
    m = qp.grid_cells(dataset, crs=dataset.epsg)
    assert m.layers
