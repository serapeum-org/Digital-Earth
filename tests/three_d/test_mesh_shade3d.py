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
        assert VALUE not in surface.cell_data and VALUE not in surface.point_data
    finally:
        scene.close()
