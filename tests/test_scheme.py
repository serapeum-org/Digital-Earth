"""Tests for categorical ``scheme`` on the value-colored Map methods (cleopatra classify, #154).

``choropleth`` / ``voronoi`` / ``cartogram`` / ``quadtree`` forward ``scheme``/``k`` to ``PolygonGlyph``; a set
``scheme`` must bin the values into discrete classes — i.e. the mappable carries a ``BoundaryNorm`` — while the
default (no ``scheme``) stays a continuous norm.
"""

import pytest
from matplotlib.colors import BoundaryNorm

from digitalearth.scene import Map


@pytest.fixture
def points_fc():
    """Point fixture (EPSG:32618, numeric 'fid')."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygons(points_fc):
    """Polygon fixture: buffered points with a numeric 'fid' column."""
    fc = points_fc.copy()
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


def test_choropleth_scheme_is_discrete(polygons):
    """A scheme bins choropleth fills into discrete classes (BoundaryNorm); default is continuous."""
    discrete = Map(crs=polygons.epsg).choropleth(polygons, column="fid", scheme="quantiles", k=3)
    continuous = Map(crs=polygons.epsg).choropleth(polygons, column="fid")
    assert isinstance(discrete.norm, BoundaryNorm)
    assert not isinstance(continuous.norm, BoundaryNorm)


def test_voronoi_scheme_is_discrete(points_fc):
    """voronoi honours a categorical scheme on its filled cells."""
    pc = Map(crs=points_fc.epsg).voronoi(points_fc, column="fid", scheme="quantiles", k=3)
    assert isinstance(pc.norm, BoundaryNorm)


def test_cartogram_scheme_is_discrete(polygons):
    """cartogram honours a categorical scheme on its scaled polygons."""
    pc = Map(crs=polygons.epsg).cartogram(polygons, scale="fid", column="fid", scheme="quantiles", k=3)
    assert isinstance(pc.norm, BoundaryNorm)


def test_quadtree_scheme_is_discrete(points_fc):
    """quadtree honours a categorical scheme on its aggregate cells."""
    pc = Map(crs=points_fc.epsg).quadtree(points_fc, column="fid", nmax=1, scheme="quantiles", k=3)
    assert isinstance(pc.norm, BoundaryNorm)


def test_scheme_fisher_jenks(polygons):
    """The native Fisher-Jenks scheme is accepted (no mapclassify dependency)."""
    pc = Map(crs=polygons.epsg).choropleth(polygons, column="fid", scheme="fisher_jenks", k=3)
    assert isinstance(pc.norm, BoundaryNorm)
