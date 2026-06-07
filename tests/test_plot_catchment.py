"""Tests for StaticGlyph.plotCatchment — catchment plot (points + polygons + lines)."""

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
    """plotCatchment draws the three layers and returns (fig, ax)."""
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


def test_plot_catchment_empty_poly_and_line(catchment):
    """plotCatchment renders the gauge points even when poly/line layers are empty.

    Test scenario:
        Empty polygon and line GeoDataFrames take the ``if poly_rings:`` / ``if line_paths:`` false branches;
        only the scatter layer is drawn and a Figure is returned.
    """
    pts, _poly, _lines = catchment
    empty_poly = gpd.GeoDataFrame(geometry=[], crs=pts.crs)
    empty_line = gpd.GeoDataFrame(geometry=[], crs=pts.crs)
    fig, ax = StaticGlyph.plotCatchment(pts, "value", empty_poly, empty_line)
    assert isinstance(fig, Figure), "expected a Figure with empty poly/line layers"


def test_plot_catchment_save(catchment, tmp_path):
    """plotCatchment writes the figure to disk when ``save`` is a path.

    Test scenario:
        A truthy ``save`` path takes the save branch and the file is created on disk.
    """
    pts, poly, lines = catchment
    out = tmp_path / "catchment.png"
    StaticGlyph.plotCatchment(pts, "value", poly, lines, save=str(out))
    assert out.exists(), f"expected saved figure at {out}"


def test_static_uses_native_scatter_backend():
    """Regression: plotCatchment renders points via cleopatra ScatterGlyph (no external plotting backend)."""
    import digitalearth.static as static_mod

    text = open(static_mod.__file__, encoding="utf-8").read()
    assert "from cleopatra.scatter_glyph import ScatterGlyph" in text
    assert "import cartopy" not in text and " as gplt" not in text
