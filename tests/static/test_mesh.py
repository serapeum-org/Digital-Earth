"""Tests for T4.1 — Map unstructured methods (tricontourf/tricontour/tripcolor) via MeshGlyph."""

import pytest

from digitalearth.static import Map


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


def test_tricontour_labels(dataset):
    """tricontour(labels=True) draws inline isoline labels (cleopatra 0.15.0 / cleopatra#151)."""
    m = Map(crs=dataset.epsg)
    m.tricontour(dataset, labels=True)
    assert len(m.ax.texts) > 0, (
        "labels=True should add inline tricontour-label Text artists"
    )


def test_tricontour_no_labels_by_default(dataset):
    """tricontour without labels= draws no inline labels (no behaviour change)."""
    m = Map(crs=dataset.epsg)
    m.tricontour(dataset)
    assert len(m.ax.texts) == 0, "default tricontour must not add label artists"


def test_tricontourf_labels_is_noop(dataset):
    """tricontourf(labels=True) is a no-op — filled tri-contours are not line-labelled."""
    m = Map(crs=dataset.epsg)
    m.tricontourf(dataset, labels=True)
    assert len(m.ax.texts) == 0, (
        "filled tricontourf must ignore labels= (no label artists)"
    )


@pytest.mark.parametrize("kind", ["tricontourf", "tricontour"])
def test_contour_refuses_a_mesh_whose_values_sit_on_the_faces(ugrid_mesh, kind):
    """A contour render refuses a face-located mesh and names the render that shades faces.

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        kind: The contour verb under test.

    Test scenario:
        Contours interpolate between values at the nodes, which a face-located mesh does not have.
        Averaging the faces onto the nodes to make it work would invent data, so it is refused instead.
    """
    mesh = ugrid_mesh("face")
    verb = getattr(Map(), kind)
    with pytest.raises(ValueError, match="values sit on its"):
        verb(mesh)


def test_tripcolor_skips_a_mesh_no_projection_can_place(ugrid_mesh, caplog):
    """A mesh whose nodes are all unplaceable is skipped as off-limb, not drawn half-way.

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        caplog: pytest log capture.

    Test scenario:
        Every node is non-finite, so no triangle keeps all three. That is the mesh equivalent of an
        off-limb raster: the verb records the skip and draws nothing, rather than handing matplotlib a
        triangulation full of infinities.
    """
    import logging

    import numpy as np

    unplaceable = ugrid_mesh("face", node_x=np.full(5, np.inf))
    scene = Map()
    with caplog.at_level(logging.WARNING):
        scene.tripcolor(unplaceable)
    assert not scene.ax.collections, "an unplaceable mesh should have drawn nothing"
