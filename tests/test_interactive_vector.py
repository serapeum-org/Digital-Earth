"""DI.1b — interactive vector builders (points / path / polygons / choropleth).

Element-type, CRS-declaration and styling assertions plus matplotlib-backend render smokes — no
browser, no network. Runs in the ``interactive`` pixi env; every test ``importorskip``s geoviews.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


@pytest.fixture()
def point_fc():
    """The point fixture as a pyramids FeatureCollection (EPSG:32618, numeric 'fid')."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture()
def polygon_fc(point_fc):
    """A polygon FeatureCollection built by buffering the point fixture (same CRS, 'fid' column)."""
    fc = point_fc.copy()
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


class TestPoints:
    """``points`` — point layers from a FeatureCollection."""

    def test_registers_gv_points_and_chains(self, m, point_fc):
        out = m.points(point_fc)
        assert out is m, "points() must return the map for chaining"
        assert isinstance(
            m.layers[0], gv.Points
        ), f"expected gv.Points, got {type(m.layers[0])}"

    def test_crs_is_declared_as_display_crs(self, m, point_fc):
        """The element's crs must be the display CRS GeoViews built internally (no re-projection)."""
        m.points(point_fc)
        crs = m.layers[0].crs
        assert (
            "3857" in str(crs) or "Mercator" in type(crs).__name__
        ), f"element CRS should declare the 3857 display CRS, got {crs!r}"

    def test_coordinates_are_reprojected_via_pyramids(self, m, point_fc):
        """Geometry must be in Web-Mercator metres (pyramids to_crs), not the source UTM range."""
        m.points(point_fc)
        x_values = m.layers[0].dimension_values(0)
        expected = point_fc.to_crs(3857).geometry.x.to_numpy()
        assert np.allclose(
            np.sort(x_values), np.sort(expected)
        ), "point x coordinates must equal the pyramids-reprojected geometry"

    def test_value_column_drives_colour_and_hover(self, m, point_fc):
        m.points(point_fc, value_column="fid", cmap="magma")
        element = m.layers[0]
        assert "fid" in [
            d.name for d in element.vdims
        ], "value column must be a vdim for hover"
        style = hv.Store.lookup_options("bokeh", element, "style").kwargs
        assert style["color"] == "fid" and style["cmap"] == "magma"

    def test_size_is_recorded(self, m, point_fc):
        m.points(point_fc, size=12.0)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["size"] == 12.0, f"size not honoured: {style.get('size')}"

    def test_mpl_render_smoke(self, m, point_fc, tmp_path):
        out = tmp_path / "points.png"
        m.points(point_fc, value_column="fid").save(str(out))
        assert out.exists() and out.stat().st_size > 0


class TestPath:
    """``path`` — line layers."""

    def test_registers_gv_path(self, m, point_fc):
        lines = point_fc.copy()
        lines["geometry"] = point_fc.geometry.shortest_line(
            point_fc.geometry.shift(1).fillna(point_fc.geometry.iloc[0])
        )
        m.path(lines)
        assert isinstance(
            m.layers[0], gv.Path
        ), f"expected gv.Path, got {type(m.layers[0])}"


class TestPolygonsAndChoropleth:
    """``polygons`` / ``choropleth`` — polygon layers and colour-by-attribute fills."""

    def test_registers_gv_polygons(self, m, polygon_fc):
        m.polygons(polygon_fc)
        assert isinstance(m.layers[0], gv.Polygons), f"got {type(m.layers[0])}"

    def test_outline_only_when_no_column(self, m, polygon_fc):
        m.polygons(polygon_fc)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["fill_alpha"] == 0.0, "no-column polygons must draw outlines only"

    def test_choropleth_colours_by_column(self, m, polygon_fc):
        m.choropleth(polygon_fc, "fid", cmap="plasma", clim=(0.0, 10.0))
        element = m.layers[0]
        assert isinstance(element, gv.Polygons)
        assert "fid" in [
            d.name for d in element.vdims
        ], "choropleth column must be a vdim"
        style = hv.Store.lookup_options("bokeh", element, "style").kwargs
        assert style["color"] == "fid" and style["cmap"] == "plasma"
        plot = hv.Store.lookup_options("bokeh", element, "plot").kwargs
        assert plot["clim"] == (0.0, 10.0), f"clim not honoured: {plot.get('clim')}"

    def test_choropleth_missing_column_raises(self, m, polygon_fc):
        with pytest.raises(KeyError, match="nope"):
            m.choropleth(polygon_fc, "nope")

    def test_choropleth_mpl_render_smoke(self, m, polygon_fc, tmp_path):
        out = tmp_path / "choropleth.png"
        m.choropleth(polygon_fc, "fid").save(str(out))
        assert out.exists() and out.stat().st_size > 0


class TestRasterVectorCompose:
    """Raster + vector layers compose into one overlay (the DI.1 headline)."""

    def test_image_plus_choropleth_overlay(self, m, dataset, polygon_fc):
        m.image(dataset).choropleth(polygon_fc, "fid")
        overlay = m.render()
        assert isinstance(overlay, hv.Overlay)
        assert (
            len(overlay) == 2
        ), f"expected 2 layers in the overlay, got {len(overlay)}"
