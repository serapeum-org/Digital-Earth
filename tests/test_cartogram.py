"""Tests for Map.cartogram / api.cartogram — value-scaled polygons from a pyramids FeatureCollection."""

import numpy as np
import pytest

from digitalearth.scene import Map


@pytest.fixture
def polygons():
    """A polygon FeatureCollection (buffered points) in EPSG:32618 with a numeric 'fid' column."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


def test_cartogram_filled(polygons):
    """cartogram fills the scaled polygons coloured by a column and registers one layer."""
    m = Map(crs=polygons.epsg)
    pc = m.cartogram(polygons, scale="fid", column="fid")
    assert len(m.layers) == 1
    assert m.ax.collections
    assert len(pc.get_paths()) >= 1


def test_cartogram_outline(polygons):
    """cartogram without a column draws scaled outlines only."""
    m = Map(crs=polygons.epsg)
    m.cartogram(polygons, scale="fid")
    assert len(m.layers) == 1
    assert m.ax.collections


def test_cartogram_one_polygon_per_feature(polygons):
    """A simple-polygon layer yields one drawn polygon per feature."""
    m = Map(crs=polygons.epsg)
    pc = m.cartogram(polygons, scale="fid", column="fid")
    assert len(pc.get_paths()) == len(polygons)


def test_cartogram_requires_polygons():
    """cartogram rejects point geometry with a clear error."""
    from pyramids.feature import FeatureCollection

    pts = FeatureCollection.read_file("tests/data/points.geojson")
    m = Map(crs=pts.epsg)
    with pytest.raises(ValueError):
        m.cartogram(pts, scale="fid")


def test_scale_factors_monotonic():
    """_scale_factors maps the value range onto limits, increasing with value."""
    f = Map._scale_factors(np.array([0.0, 1.0, 2.0, 4.0]), (0.2, 1.0))
    assert f[0] == pytest.approx(0.2)
    assert f[-1] == pytest.approx(1.0)
    assert np.all(np.diff(f) > 0)


def test_scale_factors_constant_is_midpoint():
    """A constant column maps every feature to the midpoint factor (no divide-by-zero)."""
    f = Map._scale_factors(np.array([3.0, 3.0, 3.0]), (0.2, 1.0))
    assert np.allclose(f, 0.6)


def test_api_cartogram(polygons):
    """api.cartogram builds a finished Map with the cartogram layer."""
    from digitalearth import api

    m = api.cartogram(polygons, scale="fid", column="fid", crs=polygons.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1
