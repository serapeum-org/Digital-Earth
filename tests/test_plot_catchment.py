"""Tests for StaticGlyph.plotCatchment — geoplot-free catchment plot (points + polygons + lines)."""

import geopandas as gpd
import numpy as np
import pytest
from matplotlib.figure import Figure
from shapely.geometry import LineString

from digitalearth.static import StaticGlyph


@pytest.fixture
def catchment():
    """Gauge points (with a numeric 'value'), sub-catchment polygons, and a river network."""
    pts = gpd.read_file("tests/data/points.geojson")
    pts["value"] = np.arange(1.0, len(pts) + 1.0)
    poly = pts.copy()
    poly["geometry"] = poly.geometry.buffer(800.0)
    coords = list(zip(pts.geometry.x.tolist(), pts.geometry.y.tolist()))
    lines = gpd.GeoDataFrame(
        geometry=[LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)],
        crs=pts.crs,
    )
    return pts, poly, lines


def test_plot_catchment(catchment):
    """plotCatchment draws the three layers and returns (fig, ax) without geoplot."""
    pts, poly, lines = catchment
    fig, ax = StaticGlyph.plotCatchment(pts, "value", poly, lines, title="Catchment")
    assert isinstance(fig, Figure)
    assert len(ax.collections) >= 3  # poly fill + river lines + gauge scatter
    assert ax.get_title() == "Catchment"


def test_plot_catchment_scheme(catchment):
    """plotCatchment accepts a categorical scheme for the gauge points."""
    pts, poly, lines = catchment
    _fig, ax = StaticGlyph.plotCatchment(pts, "value", poly, lines, scheme="quantiles")
    assert ax.collections


def test_plot_catchment_does_not_mutate_inputs(catchment):
    """Reprojection is non-mutating: the caller's GeoDataFrames keep their original CRS/columns."""
    pts, poly, lines = catchment
    crs_before = pts.crs
    StaticGlyph.plotCatchment(pts, "value", poly, lines)
    assert pts.crs == crs_before


def test_static_module_has_no_geoplot():
    """Regression: static.py no longer imports geoplot (the dependency was dropped)."""
    import digitalearth.static as static_mod

    text = open(static_mod.__file__, encoding="utf-8").read()
    assert "import geoplot" not in text
    assert "geoplot.crs" not in text
