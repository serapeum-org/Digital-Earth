"""Relief and basemap for a value-shaded 3-D mesh, reachable through the seam (#557).

``relief=`` lifts a node-located surface by its own value (the "coloured data draped in 3D" case) without a
manual ``node_z``; ``basemap=True`` builds the scene in EPSG:3857 so tiles align with no ``set_crs`` dance; and
the shaded mesh sits just above the basemap plane so the tiles do not occlude it. Gated on the ``3d`` extra.
"""

# ruff: noqa: E402
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth import quickmap
from digitalearth.three_d import Scene3D


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_relief_lifts_a_node_located_surface_by_its_value(ugrid_mesh):
    """``relief=True`` lifts each node to its own value, so the surface's z follows the data (#557).

    Test scenario:
        The node-located fixture (``zs=[1, 2, 3, 4, 5]``) drawn with ``relief=True`` reaches a maximum z of 5 —
        the surface is raised by the plotted variable, not left flat on the ground plane.
    """
    scene = quickmap(ugrid_mesh("node"), backend="3d", relief=True)
    try:
        poly = scene.layers[0][0]
        assert float(poly.points[:, 2].max()) == 5.0, poly.points[:, 2].max()
    finally:
        scene.close()


def test_relief_scales_the_lift(ugrid_mesh):
    """A numeric ``relief`` exaggerates the lift by that factor (#557)."""
    scene = quickmap(ugrid_mesh("node"), backend="3d", relief=2.0)
    try:
        poly = scene.layers[0][0]
        assert float(poly.points[:, 2].max()) == 10.0, poly.points[:, 2].max()
    finally:
        scene.close()


def test_relief_warps_a_face_located_mesh_by_face_node_averaging(ugrid_mesh):
    """``relief=True`` warps a face-located mesh by averaging its face values onto the nodes (#569).

    Test scenario:
        The face fixture (``depth=[10, 20]`` → fan-triangle values ``[10, 10, 20]``) drawn with
        ``relief=True`` no longer refuses: the lone triangle-only node carries depth 20, so the warped
        surface reaches z=20 instead of raising a ``ValueError``.
    """
    scene = quickmap(ugrid_mesh("face"), backend="3d", relief=True)
    try:
        poly = scene.layers[0][0]
        assert float(poly.points[:, 2].max()) == 20.0, poly.points[:, 2].max()
    finally:
        scene.close()


def test_face_relief_averages_incident_face_values_onto_each_node(ugrid_mesh):
    """``_face_values_to_nodes`` lifts each node to the mean of the fan-triangles that touch it (#569).

    Test scenario:
        For the quad+triangle fixture (fan-triangle values ``[10, 10, 20]``) each node's height is the mean
        of its incident fan-triangle values — a nodes-only-on-the-quad node reads 10, the triangle-only node
        reads 20, and the shared nodes fall strictly between — matching an independent incident-mean oracle.
    """
    from digitalearth.three_d.builders.mesh import _face_values_to_nodes

    mesh = ugrid_mesh("face").mesh
    # One value per fan-triangle, the shape the quad+triangle fixture (3 fan-triangles) is spread into.
    face_values = np.array([10.0, 10.0, 20.0])
    triangles = np.asarray(mesh.fan_triangles)
    expected = np.array(
        [
            face_values[np.any(triangles == node, axis=1)].mean()
            for node in range(len(mesh.node_x))
        ]
    )
    heights = _face_values_to_nodes(mesh, face_values)
    np.testing.assert_allclose(heights, expected)


def test_relief_without_values_is_still_refused(ugrid_mesh):
    """``relief=`` lifts each node by a value, so a mesh carrying none is still refused (#569)."""
    mesh = ugrid_mesh("face").mesh
    scene = Scene3D(off_screen=True)
    try:
        with pytest.raises(ValueError, match="values-carrying"):
            scene.trimesh(mesh, relief=True)
    finally:
        scene.close()


def _stub_basemap_fetch(monkeypatch):
    """Replace the tile fetch with a synthetic 3857 image so no test touches the network."""
    from digitalearth.three_d.decoration import decoration as deco

    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[..., 3] = 255
    monkeypatch.setattr(
        deco,
        "fetch_basemap_image",
        lambda bbox, provider, zoom: (image, (0.0, 0.0, 1.0e6, 1.0e6)),
    )


def _layer_bounds(scene, kind):
    """Return the drawn bounds of the first layer of ``kind`` in ``scene``."""
    for layer_id in scene.layer_ids:
        if scene.get_layer(layer_id).kind == kind:
            return scene.renderer.drawn[layer_id][0].bounds
    raise AssertionError(f"no {kind!r} layer drawn: {list(scene.layer_ids)}")


def test_basemap_builds_the_scene_in_3857_with_no_set_crs_dance(
    ugrid_mesh, monkeypatch
):
    """``basemap=True`` over a CRS-carrying mesh builds the scene in EPSG:3857 automatically (#557).

    Test scenario:
        An EPSG:28992 mesh plotted with ``basemap=True`` and no ``crs`` comes back drawn in EPSG:3857 — the
        scene the tiles need — with no manual ``set_crs``/``crs=3857`` from the caller.
    """
    from digitalearth.base.spec.bounds import same_crs

    _stub_basemap_fetch(monkeypatch)
    scene = quickmap(ugrid_mesh("node", epsg=28992), backend="3d", basemap=True)
    try:
        assert same_crs(scene.display_crs, 3857), scene.display_crs
    finally:
        scene.close()


def test_basemap_sits_below_the_mesh_so_it_is_not_occluded(ugrid_mesh, monkeypatch):
    """The tile plane is dropped below the mesh so an opaque basemap does not hide the data (#557).

    Test scenario:
        A flat (z=0) shaded mesh over a basemap: the basemap plane's whole z-range sits below the mesh's, so
        the mesh reads over the tiles instead of being occluded by them.
    """
    _stub_basemap_fetch(monkeypatch)
    scene = quickmap(ugrid_mesh("node", epsg=28992), backend="3d", basemap=True)
    try:
        mesh_bounds = _layer_bounds(scene, "mesh")
        basemap_bounds = _layer_bounds(scene, "basemap")
        assert basemap_bounds[5] < mesh_bounds[4], (basemap_bounds, mesh_bounds)
    finally:
        scene.close()
