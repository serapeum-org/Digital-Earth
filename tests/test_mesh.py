"""Tests for T4.1 — Map unstructured methods (tricontourf/tricontour/tripcolor) via MeshGlyph."""

import pytest

from digitalearth.scene import Map


@pytest.mark.parametrize("kind", ["tricontourf", "tricontour", "tripcolor"])
def test_tri_methods_from_dataset(dataset, kind):
    """Each tri method triangulates the raster's valid cell centres and renders one layer."""
    m = Map(crs=dataset.epsg)
    getattr(m, kind)(dataset)
    assert len(m.layers) == 1
    assert m.ax.collections


def test_tricontourf_from_features():
    """tricontourf works on a FeatureCollection of points with a numeric value column."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = Map(crs=fc.epsg)
    m.tricontourf(fc)
    assert len(m.layers) == 1


def test_tri_requires_value_column():
    """A FeatureCollection without a numeric column raises a clear error."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc = fc.drop(columns=[c for c in fc.columns if c != fc.geometry.name])
    m = Map(crs=fc.epsg)
    with pytest.raises(ValueError, match="no numeric value column"):
        m.tricontourf(fc)
