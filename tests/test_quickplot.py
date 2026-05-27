"""Tests for T9.1 — quickplot/quickmap one-call entry points and module functions."""

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


def test_quickmap_with_domain(dataset):
    """quickmap(domain=...) sets the axes extent from the named region."""
    m = qp.quickmap(dataset, crs=3857, domain="europe")
    assert m.ax.get_xlim()[1] > 1e6  # reprojected Europe bbox in metres


def test_quickmap_decorations_best_effort(dataset):
    """coastlines/basemap flags run without raising even if data/tiles are unreachable."""
    m = qp.quickmap(dataset, crs=3857, coastlines=True, basemap=True)
    assert m.layers  # the raster layer is drawn regardless of decoration availability


def test_quickmap_shapes_without_column_skips_colorbar():
    """A polygon FeatureCollection with no column draws outlines and skips the colorbar gracefully."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.quickmap(fc, crs=fc.epsg)  # no column -> shapes (outline only)
    assert m.ax.collections


def test_module_choropleth(dataset):
    """The module-level choropleth colours polygons by a column."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.choropleth(fc, column="fid", crs=fc.epsg)
    assert m.ax.collections


def test_module_scatter():
    """The module-level scatter draws a point FeatureCollection."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = qp.scatter(fc, crs=fc.epsg)
    assert m.ax.collections


def test_quickmap_colorbar_false(dataset):
    """colorbar=False leaves the figure with a single axes (no colorbar axes)."""
    m = qp.quickmap(dataset, crs=dataset.epsg, colorbar=False)
    assert len(m.fig.axes) == 1


def test_quickmap_swallows_decoration_failures(dataset, mocker):
    """When coastlines/basemap/colorbar raise, quickmap swallows the errors and still returns the map."""
    mocker.patch.object(Map, "coastlines", side_effect=RuntimeError("no net"))
    mocker.patch.object(Map, "basemap", side_effect=RuntimeError("no tiles"))
    mocker.patch.object(Map, "colorbar", side_effect=RuntimeError("bad mappable"))
    m = qp.quickmap(dataset, crs=dataset.epsg, coastlines=True, basemap=True, colorbar=True)
    assert m.layers


def test_module_grid_cells_swallows_colorbar_failure(dataset, mocker):
    """grid_cells still returns a map when its colorbar fails."""
    mocker.patch.object(Map, "colorbar", side_effect=RuntimeError("bad mappable"))
    m = qp.grid_cells(dataset, crs=dataset.epsg)
    assert m.layers
