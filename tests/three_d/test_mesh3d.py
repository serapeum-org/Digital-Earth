"""The 3-D tier draws a UGRID mesh's faces as a PyVista surface (#546).

``quickmap(mesh, backend="3d")`` classifies a ``MeshDataset`` as the ``mesh`` family, unwraps it to its bare
``Mesh2d`` (``api._mesh_of``) and routes it to :meth:`Scene3D.trimesh`, which fan-cuts the faces into triangles
(pyramids' own connectivity, never re-triangulated) and builds one PyVista ``PolyData`` surface — the 3-D
sibling of the static ``tripcolor``, the interactive ``trimesh`` and the web ``trimesh`` renders. A bare
``Mesh2d`` carries geometry, not a per-element value table, so the surface takes one flat colour, exactly as
the interactive and web UGRID paths draw it. Runs in the ``viz3d`` pixi env; every test ``importorskip``s
pyvista.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.api import quickmap  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless so a ``Scene3D`` built inside ``quickmap`` never opens a window."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_quickmap_draws_a_ugrid_mesh_on_the_3d_tier(ugrid_mesh):
    """``quickmap`` builds a ``Scene3D`` whose one layer is the mesh's own faces as a PyVista surface.

    Test scenario:
        The two-face fixture (a quad and a triangle) fan-cuts into three triangles, so the surface carries
        three cells — the mesh's own connectivity rather than a re-triangulation. The layer is declared as
        the ``mesh`` kind and is addressable by id.
    """
    dataset = ugrid_mesh("face")
    scene = quickmap(dataset, backend="3d")
    try:
        assert isinstance(scene, Scene3D), type(scene)
        assert len(scene.layer_ids) == 1, scene.layer_ids
        assert scene.figure_spec.layers.get(scene.layer_ids[0]).kind == "mesh"
        poly, _actor = scene.layers[0]
        # One cell per fan triangle: the quad -> 2, the triangle -> 1 — the mesh's own faces, unwrapped from
        # the MeshDataset to its Mesh2d rather than triangulated afresh.
        assert poly.n_cells == len(dataset.mesh.fan_triangles) == 3
        # The nodes come through as they are: the fixture's x spans 0..2 in its own coordinates.
        assert float(poly.points[:, 0].max()) == 2.0, poly.points
        # A bare Mesh2d has no value table, so the surface sits flat on the ground plane.
        assert float(np.abs(poly.points[:, 2]).max()) == 0.0, poly.points
    finally:
        scene.close()


def test_trimesh_refuses_a_bad_kind(ugrid_mesh):
    """Routing a mesh with a non-mesh ``kind`` is refused, naming the renderer the tier does use.

    Test scenario:
        ``trimesh`` is the 3-D tier's only mesh renderer. ``quickmap`` refuses any non-``auto`` ``kind`` on
        this tier upstream (it has no ``raster_renderer``), so the in-branch guard is reached by routing
        through ``_quickmap_3d`` directly — the belt-and-braces that mirrors the interactive and web mesh
        branches. Its message names ``trimesh``.
    """
    from digitalearth.api import _quickmap_3d

    dataset = ugrid_mesh("face")
    with pytest.raises(ValueError, match="trimesh"):
        _quickmap_3d(dataset, kind="terrain", scene_kw={"off_screen": True})


def test_ds_plot_mesh_end_to_end(ugrid_mesh):
    """``ds.plot(var, backend="3d")`` renders the mesh through the pyramids plot-backend seam (#546 DoD).

    Test scenario:
        A ``MeshDataset`` plotted with ``backend="3d"`` routes through pyramids' registered digitalearth
        backend into ``quickmap`` with no direct digitalearth import on the caller's side, and comes back as
        a ``Scene3D`` carrying the drawn mesh surface.
    """
    dataset = ugrid_mesh("face")
    scene = dataset.plot("depth", backend="3d")
    try:
        assert isinstance(scene, Scene3D), type(scene)
        assert scene.figure_spec.layers.get(scene.layer_ids[0]).kind == "mesh"
        assert scene.layers[0][0].n_cells == 3
    finally:
        scene.close()
