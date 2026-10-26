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


def test_tripcolor_keeps_a_finite_view_when_part_of_the_mesh_is_unplaceable(ugrid_mesh):
    """A partly unplaceable mesh still frames the triangles it did draw (#499 review H3).

    Test scenario:
        One node of the quad/triangle mesh is sent to infinity, which drops the one triangle leaning on it
        and keeps the other two. Leaving the infinite node in the coordinate arrays poisoned the extent, so
        the axes collapsed to about a tenth of a unit and the surviving triangles were drawn far outside the
        view — a blank figure with no warning. The view must enclose what was drawn.
    """
    import numpy as np

    partly = ugrid_mesh("face", node_x=np.array([0.0, 1.0, 1.0, 0.0, np.inf]))
    scene = Map()
    scene.tripcolor(partly)
    (x0, y0), (x1, y1) = scene.ax.dataLim.get_points()
    assert np.isfinite([x0, y0, x1, y1]).all(), f"the extent is not finite: {(x0, y0, x1, y1)}"
    assert x1 - x0 >= 0.9, f"x extent collapsed to {x1 - x0}, so the drawn triangles are off-view"
    assert y1 - y0 >= 0.9, f"y extent collapsed to {y1 - y0}, so the drawn triangles are off-view"


@pytest.mark.parametrize("display", [3857, "EPSG:3857"])
def test_a_crs_carrying_mesh_draws_in_a_coded_display_crs(ugrid_mesh, display):
    """A mesh that declares a CRS is reprojected for any display CRS that names an EPSG code (review H2).

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        display: The display CRS, as an int and as the `"EPSG:<code>"` spelling the Source contract allows.

    Test scenario:
        pyramids' `UgridDataset.to_crs` takes an EPSG **code** and builds its target as `f"EPSG:{code}"`,
        so handing it the string spelling produced `EPSG:EPSG:3857` and a raw pyproj CRSError escaped a
        visualization call. Both spellings must reach the same place.
    """
    mesh = ugrid_mesh("face", epsg=4326)
    scene = Map(crs=display)
    scene.tripcolor(mesh)
    assert scene.ax.collections, "a CRS-carrying mesh drew nothing"


def test_a_crs_carrying_mesh_refuses_a_display_crs_with_no_epsg_code(ugrid_mesh):
    """A projection with no authority code is refused in this repo's words, not pyproj's.

    Test scenario:
        An orthographic or Robinson display CRS carries no EPSG code, and pyramids' mesh warp can only take
        one. The caller should learn that from a message naming the CRS and the limit — not from
        `CRSError: could not interpret 'EPSG:+proj=robin'`.
    """
    mesh = ugrid_mesh("face", epsg=4326)
    scene = Map(crs="+proj=robin")
    with pytest.raises(ValueError, match="EPSG code"):
        scene.tripcolor(mesh)
