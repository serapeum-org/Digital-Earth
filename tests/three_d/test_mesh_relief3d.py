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


def _spread_to_fan_triangles(mesh, face_values):
    """Spread one value per mesh face across its fan-triangles, the way the seam feeds ``trimesh``."""
    counts = np.clip(
        np.asarray(mesh.face_node_connectivity.nodes_per_element()) - 2, 0, None
    )
    return np.repeat(np.asarray(face_values, dtype="float64"), counts)


def _per_face_node_oracle(mesh, face_values):
    """Per-node mean over the distinct FACES touching each node (computed from connectivity, not triangles)."""
    conn = np.asarray(mesh.face_node_connectivity.data)
    face_values = np.asarray(face_values, dtype="float64")
    out = []
    for node in range(len(mesh.node_x)):
        incident = [
            face_values[f] for f, row in enumerate(conn) if node in row[row >= 0]
        ]
        out.append(float(np.mean(incident)) if incident else 0.0)
    return np.array(out)


def test_face_relief_default_is_a_per_face_mean(ugrid_mesh):
    """The default reduction lifts each node to the mean of the distinct FACES touching it (#569).

    Test scenario:
        The quad+triangle fixture (face values ``[10, 20]``) averaged per-face gives the shared nodes 15.0 and
        the corners their own face's value — matching a connectivity-based oracle and, crucially, NOT the
        fan-triangle-weighted 13.33 the per-triangle mode would give the quad/triangle-shared node.
    """
    from digitalearth.three_d.builders.mesh import _face_values_to_nodes

    mesh = ugrid_mesh("face").mesh
    face_values = np.array([10.0, 20.0])
    spread = _spread_to_fan_triangles(mesh, face_values)
    expected = _per_face_node_oracle(mesh, face_values)
    heights = _face_values_to_nodes(mesh, spread)
    np.testing.assert_allclose(heights, expected)
    assert sorted(heights.tolist()) == [10.0, 10.0, 15.0, 15.0, 20.0], heights.tolist()


def test_face_relief_triangle_mode_averages_over_fan_triangles(ugrid_mesh):
    """``relief_reduce="triangle"`` lifts each node to the mean of the fan-triangles that touch it (#569).

    Test scenario:
        The same fixture under the triangle reduction weights the quad by each of its two fan-triangles, so the
        node shared by the quad and the lone triangle reads ``(10+10+20)/3 = 13.33`` — matching a fan-triangle
        oracle, and differing from the per-face default by that fan-cut weighting.
    """
    from digitalearth.three_d.builders.mesh import _face_values_to_nodes

    mesh = ugrid_mesh("face").mesh
    spread = _spread_to_fan_triangles(mesh, np.array([10.0, 20.0]))
    triangles = np.asarray(mesh.fan_triangles)
    expected = np.array(
        [
            spread[np.any(triangles == node, axis=1)].mean()
            for node in range(len(mesh.node_x))
        ]
    )
    heights = _face_values_to_nodes(mesh, spread, how="triangle")
    np.testing.assert_allclose(heights, expected)
    assert any(abs(h - 13.333333333333334) < 1e-9 for h in heights), heights.tolist()


def test_face_relief_leaves_a_node_no_triangle_touches_at_zero():
    """A node no fan-triangle touches has no incident face to average, so it stays at 0 (#569).

    Test scenario:
        A one-triangle mesh over four nodes leaves node 3 referenced by nothing; the three touched nodes take
        the triangle's value (5) and the untouched node reads 0 rather than a divide-by-zero NaN.
    """
    from types import SimpleNamespace

    from digitalearth.three_d.builders.mesh import _face_values_to_nodes

    mesh = SimpleNamespace(fan_triangles=np.array([[0, 1, 2]]), node_x=np.zeros(4))
    heights = _face_values_to_nodes(mesh, np.array([5.0]), how="triangle")
    np.testing.assert_allclose(heights, [5.0, 5.0, 5.0, 0.0])


def test_face_relief_scales_the_averaged_lift(ugrid_mesh):
    """A numeric ``relief`` exaggerates a face-located warp by that factor, after averaging (#569).

    Test scenario:
        The face fixture (triangle-only node averages to 20) drawn with ``relief=2.0`` reaches z=40 — the
        averaged per-node heights are scaled by the factor, the same exaggeration the node-located path applies.
    """
    scene = quickmap(ugrid_mesh("face"), backend="3d", relief=2.0)
    try:
        poly = scene.layers[0][0]
        assert float(poly.points[:, 2].max()) == 40.0, poly.points[:, 2].max()
    finally:
        scene.close()


def test_face_relief_masks_a_dry_nan_face_from_its_neighbours(ugrid_mesh):
    """A non-finite (dry-cell) face contributes nothing, so it does not drag its neighbours to NaN (#569).

    Test scenario:
        The quad is marked dry (``NaN``) and the triangle wet (20). Every node's height stays finite: the
        quad-only corners get 0 (no finite face touches them), while the nodes the wet triangle touches read 20
        — the dry quad is masked out of their mean instead of poisoning it.
    """
    from digitalearth.three_d.builders.mesh import _face_values_to_nodes

    mesh = ugrid_mesh("face").mesh
    spread = _spread_to_fan_triangles(mesh, np.array([np.nan, 20.0]))
    heights = _face_values_to_nodes(mesh, spread)
    assert np.all(np.isfinite(heights)), heights.tolist()
    np.testing.assert_allclose(sorted(heights.tolist()), [0.0, 0.0, 20.0, 20.0, 20.0])


def test_face_relief_keeps_the_surface_colour_shaded(ugrid_mesh):
    """Warping a face mesh into relief does not drop its per-face colour scalars (#569).

    Test scenario:
        A face mesh drawn with ``relief=True`` still carries the ``VALUE`` cell scalars on the warped surface,
        so the mesh reads as coloured-and-lifted, not lifted-but-blank.
    """
    from digitalearth.three_d.builders.vector import VALUE

    scene = quickmap(ugrid_mesh("face"), backend="3d", relief=True)
    try:
        poly = scene.layers[0][0]
        assert VALUE in poly.cell_data, list(poly.cell_data.keys())
        assert float(poly.points[:, 2].max()) == 20.0, poly.points[:, 2].max()
    finally:
        scene.close()


def test_face_relief_draws_the_on_view_part_of_a_partly_off_view_mesh(ugrid_mesh):
    """A face-relief mesh with an off-view node still draws (and colours) the triangles that remain (#569).

    Test scenario:
        One node is pushed to a non-finite x so the triangles using it drop; the surface still builds from the
        surviving triangles, stays colour-shaded, and is warped (a positive z extent) rather than returning
        ``None`` or raising.
    """
    from digitalearth.three_d.builders.vector import VALUE

    off_view = np.array([0.0, 1.0, 1.0, 0.0, np.inf])
    scene = quickmap(ugrid_mesh("face", node_x=off_view), backend="3d", relief=True)
    try:
        poly = scene.layers[0][0]
        assert poly is not None and poly.n_cells > 0, poly
        assert VALUE in poly.cell_data, list(poly.cell_data.keys())
        assert float(poly.points[:, 2].max()) > 0.0, poly.points[:, 2].max()
    finally:
        scene.close()


def test_relief_reduce_rejects_an_unknown_mode(ugrid_mesh):
    """``relief_reduce`` only names the two reductions, so a typo is refused rather than silently ignored (#569)."""
    mesh = ugrid_mesh("face").mesh
    scene = Scene3D(off_screen=True)
    try:
        with pytest.raises(ValueError, match="relief_reduce"):
            scene.trimesh(mesh, relief_reduce="nope")
    finally:
        scene.close()


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
