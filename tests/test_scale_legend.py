"""Tests for value->size scaling + size legend on Map.scatter and Map.sankey (cleopatra CLEO-2)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from digitalearth.scene import Map


@pytest.fixture
def points_fc():
    """Point fixture (EPSG:32618, numeric 'fid')."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def lines_fc(points_fc):
    """A line FeatureCollection chaining the points, with a numeric 'w' column."""
    from pyramids.feature import FeatureCollection

    coords = list(zip(points_fc.geometry.x.tolist(), points_fc.geometry.y.tolist()))
    gdf = gpd.GeoDataFrame(
        {"w": np.arange(1.0, len(coords))},
        geometry=[LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)],
        crs=points_fc.crs,
    )
    return FeatureCollection(gdf)


def test_scatter_scale_sizes_span_limits(points_fc):
    """scale maps marker areas across size_limits (varying, monotone-bounded)."""
    m = Map(crs=points_fc.epsg)
    pc = m.scatter(points_fc, scale="fid", size_limits=(20, 200))
    sizes = np.asarray(pc.get_sizes())
    assert sizes.min() == pytest.approx(20)
    assert sizes.max() == pytest.approx(200)


def test_scatter_size_legend(points_fc):
    """size_legend draws a legend on the axes."""
    m = Map(crs=points_fc.epsg)
    m.scatter(points_fc, scale="fid", size_legend=True)
    assert m.ax.get_legend() is not None


def test_scatter_no_scale_is_uniform(points_fc):
    """Without scale, markers keep a single uniform size (backward compatible)."""
    m = Map(crs=points_fc.epsg)
    pc = m.scatter(points_fc)
    assert len(set(np.asarray(pc.get_sizes()).tolist())) == 1


def test_sankey_size_legend(lines_fc):
    """size_legend draws the width legend for a flow map."""
    m = Map(crs=lines_fc.epsg)
    m.sankey(lines_fc, scale="w", size_legend=True, width_limits=(1, 8))
    assert m.ax.get_legend() is not None
