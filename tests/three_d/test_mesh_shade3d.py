"""Shade a MeshDataset by its variable on the 3-D tier (#557).

``ds.plot(var, backend="3d")`` colours the mesh surface by ``var`` (the 3-D sibling of static ``tripcolor``),
instead of a single flat colour. Face-located values shade each triangle; node-located values shade smoothly.
Gated on the ``3d`` extra.
"""

# ruff: noqa: E402
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth import quickmap
from digitalearth.three_d import Scene3D
from digitalearth.three_d.builders.vector import VALUE


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_quickmap_shades_a_face_located_mesh_by_its_variable(ugrid_mesh):
    """A face-located MeshDataset colours each triangle by the face value, not a flat fill.

    Test scenario:
        The two-face fixture carries ``depth=[10, 20]`` on its faces; fan-cut into three triangles the drawn
        surface's cell scalars are the per-face values spread to the triangles (two 10s and a 20), so the
        surface is shaded rather than one flat colour.
    """
    dataset = ugrid_mesh("face")
    scene = quickmap(dataset, backend="3d")
    try:
        surface = scene.layers[0][0]
        assert VALUE in surface.cell_data, list(surface.cell_data)
        assert sorted(set(np.asarray(surface.cell_data[VALUE]).tolist())) == [
            10.0,
            20.0,
        ]
    finally:
        scene.close()


def test_quickmap_shades_a_node_located_mesh_by_its_variable(ugrid_mesh):
    """A node-located MeshDataset colours its nodes by the node value (smooth shading).

    Test scenario:
        A node-located variable (``zs`` on the five nodes) reaches the surface as point scalars, so the mesh
        shades smoothly across its faces rather than taking one flat colour.
    """
    dataset = ugrid_mesh("node")
    scene = quickmap(dataset, backend="3d")
    try:
        surface = scene.layers[0][0]
        assert VALUE in surface.point_data, list(surface.point_data)
        assert len(set(np.asarray(surface.point_data[VALUE]).tolist())) > 1
    finally:
        scene.close()


def test_trimesh_without_values_stays_flat(ugrid_mesh):
    """A bare ``Mesh2d`` (no values) still draws one flat colour — the value path is opt-in (#557).

    Test scenario:
        Calling ``trimesh`` with a bare mesh and no ``values`` attaches no scalar, preserving the #546 flat
        render for a geometry-only mesh.
    """
    mesh = ugrid_mesh("face").mesh
    scene = Scene3D(off_screen=True)
    try:
        scene.trimesh(mesh)
        surface = scene.layers[0][0]
        assert VALUE not in surface.cell_data
        assert VALUE not in surface.point_data
    finally:
        scene.close()


def test_trimesh_refuses_a_wrong_length_value_array(ugrid_mesh):
    """A values array that doesn't match the mesh is refused with a clear message, not a cryptic IndexError (M1).

    Test scenario:
        The two-face fixture fan-cuts into three triangles, so a face-located ``trimesh`` needs one value per
        fan-triangle; passing one per face (two values) names both counts instead of raising deep in numpy.
    """
    mesh = ugrid_mesh("face").mesh
    scene = Scene3D(off_screen=True)
    try:
        values = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="one value per"):
            scene.trimesh(mesh, values=values, location="face")
    finally:
        scene.close()


def test_a_graduated_scheme_shades_the_mesh_by_class_index(ugrid_mesh):
    """A categorical/graduated ``scheme`` replaces the surface scalars with class indices and keys them (M2, L5).

    Test scenario:
        A node-located mesh (``zs=[1..5]``) shaded ``scheme="categorical"`` reaches the surface as class
        indices (one per distinct value, a small integer range), and the layer publishes a colour encoding
        with a shown colorbar guide. (``n_colors`` lives in the internal style dict ``draw_trimesh`` consumes,
        not on anything reachable from the scene, so it is covered by the shared classified-scalars tests.)
    """
    dataset = ugrid_mesh("node")
    scene = quickmap(dataset, backend="3d", scheme="categorical")
    try:
        layer_id = scene.layer_ids[0]
        surface = scene.renderer.drawn[layer_id][0]
        indices = np.asarray(surface.point_data[VALUE])
        assert set(np.unique(indices).tolist()) == {0.0, 1.0, 2.0, 3.0, 4.0}, indices
        encoding = scene.get_layer(layer_id).symbology.encoding("color")
        assert encoding is not None, "a shaded mesh must publish a colour encoding"
        guide = scene.get_layer(layer_id).symbology.guide()
        assert guide is not None, guide
        assert guide.show, guide
    finally:
        scene.close()


def test_a_crs_mesh_sets_the_scene_crs_without_a_basemap(ugrid_mesh):
    """Plotting a CRS-carrying mesh adopts its CRS as the scene's, even with no basemap (L1, intended).

    Test scenario:
        The seam conveys the MeshDataset's CRS as ``source_crs``, so a plain ``plot(backend="3d")`` on a 28992
        mesh comes back drawn in 28992 — the scene knows its CRS for any later layer — rather than CRS-less.
    """
    from digitalearth.base.spec.bounds import same_crs

    scene = quickmap(ugrid_mesh("node", epsg=28992), backend="3d")
    try:
        assert same_crs(scene.display_crs, 28992), scene.display_crs
    finally:
        scene.close()


def test_trimesh_refuses_a_wrong_length_node_value_array(ugrid_mesh):
    """A node-located values array that isn't one per node is refused, naming both counts (#557)."""
    mesh = ugrid_mesh("node").mesh
    scene = Scene3D(off_screen=True)
    try:
        values = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="one value per node"):
            scene.trimesh(mesh, values=values, location="node")
    finally:
        scene.close()


def test_relief_false_is_a_no_op_not_a_flatten(ugrid_mesh):
    """``relief=False``/``0`` mean "no relief", drawing flat without raising or force-flattening (review L1)."""
    mesh = ugrid_mesh("face").mesh
    scene = Scene3D(off_screen=True)
    try:
        scene.trimesh(mesh, relief=False)  # bare mesh, no values: must not raise
        assert float(scene.layers[0][0].points[:, 2].max()) == 0.0
    finally:
        scene.close()


def test_a_wrong_length_value_leaves_the_scene_crs_untouched(ugrid_mesh):
    """A bad-length shaded call raises before adopting the mesh CRS, so the scene is left untouched (review L4)."""
    mesh = ugrid_mesh("node", epsg=28992).mesh
    scene = Scene3D(off_screen=True)
    try:
        assert scene.display_crs is None
        values = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="one value per node"):
            scene.trimesh(mesh, values=values, location="node", source_crs=28992)
        assert scene.display_crs is None, scene.display_crs
    finally:
        scene.close()


def test_shaded_trimesh_refuses_a_caller_scalars_or_lighting(ugrid_mesh):
    """Passing ``scalars=``/``lighting=`` to a value-shaded ``trimesh`` is refused by name, not a raw TypeError (L3)."""
    mesh = ugrid_mesh("node").mesh
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # one per node of the fixture
    scene = Scene3D(off_screen=True)
    try:
        with pytest.raises(TypeError, match="set by the shaded path"):
            scene.trimesh(mesh, values=values, location="node", scalars="foo")
    finally:
        scene.close()
