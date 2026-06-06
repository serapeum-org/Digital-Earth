"""Tests for Map.sankey / api.sankey — spatial flow map from a pyramids FeatureCollection of lines."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from digitalearth.scene import Map


@pytest.fixture
def lines_fc():
    """A line FeatureCollection built by chaining the point fixture, with 'flow' and 'w' columns."""
    from pyramids.feature import FeatureCollection

    pts = gpd.read_file("tests/data/points.geojson")
    coords = list(zip(pts.geometry.x.tolist(), pts.geometry.y.tolist()))
    lines = [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]
    gdf = gpd.GeoDataFrame(
        {"flow": np.arange(1.0, len(lines) + 1.0), "w": np.arange(1.0, len(lines) + 1.0)},
        geometry=lines,
        crs=pts.crs,
    )
    return FeatureCollection(gdf)


def test_sankey_plots(lines_fc):
    """sankey draws a LineCollection and registers one layer."""
    m = Map(crs=lines_fc.epsg)
    lc = m.sankey(lines_fc, column="flow", scale="w")
    assert len(m.layers) == 1
    assert type(lc).__name__ == "LineCollection"


def test_sankey_width_scaling(lines_fc):
    """scale maps line widths across the width_limits range (varying widths)."""
    m = Map(crs=lines_fc.epsg)
    lc = m.sankey(lines_fc, column="flow", scale="w", width_limits=(1, 8))
    lw = np.asarray(lc.get_linewidths())
    assert lw.max() > lw.min()


def test_sankey_flat(lines_fc):
    """sankey works with neither colour nor width column (uniform lines)."""
    m = Map(crs=lines_fc.epsg)
    m.sankey(lines_fc)
    assert len(m.layers) == 1


def test_sankey_requires_lines():
    """sankey rejects non-line geometry with a clear error."""
    from pyramids.feature import FeatureCollection

    pts = FeatureCollection.read_file("tests/data/points.geojson")
    m = Map(crs=pts.epsg)
    with pytest.raises(ValueError):
        m.sankey(pts, column="fid")


def test_api_sankey(lines_fc):
    """api.sankey builds a finished Map with the flow layer."""
    from digitalearth import api

    m = api.sankey(lines_fc, column="flow", scale="w", crs=lines_fc.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1
