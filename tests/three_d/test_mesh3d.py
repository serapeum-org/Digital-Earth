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
from digitalearth.base.crs import OffLimbError  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402
from digitalearth.three_d.builders.mesh import (  # noqa: E402
    MESH_EDGE_COLOR,
    MESH_FILL_COLOR,
    _elevation,
    _mesh_surface,
)


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless so a ``Scene3D`` built inside ``quickmap`` never opens a window."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _bare_mesh(node_x=None, node_y=None, epsg=None):
    """Build a two-face UGRID ``Mesh2d`` (a quad and a triangle) directly, bypassing the MeshDataset wrapper.

    Args:
        node_x: Node x coordinates; ``None`` uses the shared fixture's ``[0, 1, 1, 0, 2]`` (pass non-finite
            ones to make a mesh a projection drops).
        node_y: Node y coordinates; ``None`` uses ``[0, 0, 1, 1, 0.5]``.
        epsg: An EPSG code the mesh declares as its CRS; ``None`` leaves it unreferenced.

    Returns:
        A pyramids ``Mesh2d`` whose quad and triangle fan-cut into three triangles.
    """
    from pyramids.netcdf.ugrid.connectivity import Connectivity
    from pyramids.netcdf.ugrid.mesh import Mesh2d

    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 2.0])
        if node_x is None
        else np.asarray(node_x, dtype="float64"),
        node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        if node_y is None
        else np.asarray(node_y, dtype="float64"),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2, 3], [1, 4, 2, -1]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    if epsg is not None:
        from pyramids.base.crs import crs_from_user_input

        mesh.crs = crs_from_user_input(epsg)
    return mesh


@pytest.fixture
def scene():
    """Yield an off-screen 3-D scene, closed on the way out.

    Yields:
        The ``Scene3D`` under test.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


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
    rendered = dataset.plot("depth", backend="3d")
    try:
        assert isinstance(rendered, Scene3D), type(rendered)
        assert rendered.figure_spec.layers.get(rendered.layer_ids[0]).kind == "mesh"
        assert rendered.layers[0][0].n_cells == 3
    finally:
        rendered.close()


class TestElevation:
    """Tests for ``_elevation`` — how a builder's ``node_z`` becomes one z per node."""

    def test_none_is_a_flat_ground_plane(self):
        """``node_z=None`` returns all zeros, so the surface lies on the ground plane.

        Test scenario:
            The default: no elevation given, every node sits at z=0.
        """
        z = _elevation(None, 5)
        assert z.shape == (5,), z.shape
        assert float(np.abs(z).max()) == 0.0, z

    def test_scalar_lifts_every_node_to_one_height(self):
        """A scalar ``node_z`` broadcasts to one height across all nodes.

        Test scenario:
            ``node_z=7.5`` lifts the whole surface to a flat sheet at z=7.5.
        """
        z = _elevation(7.5, 4)
        assert z.tolist() == [7.5, 7.5, 7.5, 7.5], z

    def test_per_node_array_is_returned_as_the_relief(self):
        """A per-node array (one value per node) is the surface's own relief.

        Test scenario:
            A length-5 array is returned unchanged, giving each node its own z.
        """
        z = _elevation([0.0, 1.0, 2.0, 3.0, 4.0], 5)
        assert z.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0], z

    def test_length_mismatch_raises_naming_both_counts(self):
        """An array whose length is not the node count is a caller error, named before VTK.

        Test scenario:
            3 elevations for a 5-node mesh raises ``ValueError`` naming 3 and 5.
        """
        with pytest.raises(ValueError, match=r"3 values but the mesh has 5 nodes"):
            _elevation([0.0, 1.0, 2.0], 5)


class TestMeshSurface:
    """Tests for ``_mesh_surface`` / ``Scene3D.trimesh`` — geometry, CRS and the drop-and-reindex path."""

    def test_scalar_node_z_lifts_the_whole_surface(self, scene):
        """``trimesh(mesh, node_z=scalar)`` draws the surface lifted to that flat height.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            A flat sheet at z=5.0 — every point carries the scalar elevation, three cells still drawn.
        """
        scene.trimesh(_bare_mesh(), node_z=5.0)
        poly, _ = scene.layers[-1]
        assert poly.n_cells == 3, poly.n_cells
        assert (
            float(poly.points[:, 2].min()) == float(poly.points[:, 2].max()) == 5.0
        ), poly.points

    def test_per_node_node_z_gives_the_surface_relief(self, scene):
        """A per-node ``node_z`` array lifts each node to its own height.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            Node 4 is lifted to z=9; since it is used by one fan triangle, the surface's max z is 9.
        """
        scene.trimesh(_bare_mesh(), node_z=[0.0, 0.0, 0.0, 0.0, 9.0])
        poly, _ = scene.layers[-1]
        assert float(poly.points[:, 2].max()) == 9.0, poly.points

    def test_a_mesh_with_a_crs_adopts_it_as_the_scenes(self, scene):
        """The first mesh carrying a CRS sets the scene's display CRS, mirroring ``_place``.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            A scene with no display CRS draws an EPSG:4326 mesh; afterwards the scene draws in 4326, and the
            mesh's own small coordinates come through unreprojected.
        """
        assert scene.display_crs is None
        scene.trimesh(_bare_mesh(epsg=4326))
        poly, _ = scene.layers[-1]
        assert scene.display_crs == 4326, scene.display_crs
        assert float(poly.points[:, 0].max()) == 2.0, poly.points

    def test_a_mesh_already_in_the_scenes_crs_is_drawn_unreprojected(self, scene):
        """A second mesh in the CRS the scene already draws in is drawn as-is — no reprojection.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            The first EPSG:4326 mesh fixes the scene in 4326; a second EPSG:4326 mesh then matches it, so
            ``same_crs`` short-circuits the warp and its own coordinates (0..2) come through unchanged.
        """
        scene.trimesh(_bare_mesh(epsg=4326))
        scene.trimesh(_bare_mesh(epsg=4326, node_x=[0.0, 1.0, 1.0, 0.0, 2.0]))
        poly, _ = scene.layers[-1]
        assert scene.display_crs == 4326, scene.display_crs
        assert float(poly.points[:, 0].max()) == 2.0, poly.points

    def test_a_mesh_in_another_crs_is_reprojected_to_the_scenes(self, scene):
        """A mesh declaring a different CRS than the scene is warped through pyramids before it is drawn.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            The scene is first fixed in EPSG:4326 (by a lon/lat mesh), then a Web-Mercator mesh whose nodes
            are hundreds of thousands of metres is drawn — its points must come back as small lon/lat degrees,
            not raw metres on the 4326 surface.
        """
        scene.trimesh(_bare_mesh(epsg=4326))
        assert scene.display_crs == 4326
        metres = _bare_mesh(
            node_x=[500000.0, 600000.0, 600000.0, 500000.0, 700000.0],
            node_y=[6800000.0, 6800000.0, 6900000.0, 6900000.0, 6850000.0],
            epsg=3857,
        )
        scene.trimesh(metres)
        poly, _ = scene.layers[-1]
        assert float(np.abs(poly.points[:, :2]).max()) < 180.0, poly.points

    def test_a_non_finite_node_drops_its_faces_and_reindexes(self, scene):
        """A triangle with a non-finite node is dropped and the survivors are re-indexed with finite bounds.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            Node 4 is non-finite, so only the fan triangle ``[1, 4, 2]`` is dropped — the quad's two triangles
            remain, drawn from the four finite nodes, and no stray point stretches the surface's bounds.
        """
        mesh = _bare_mesh(node_x=[0.0, 1.0, 1.0, 0.0, np.inf])
        scene.trimesh(mesh)
        poly, _ = scene.layers[-1]
        assert poly.n_cells == 2, poly.n_cells
        assert poly.n_points == 4, poly.n_points
        assert bool(np.isfinite(poly.points).all()), poly.points

    def test_a_stored_figure_redraws_the_same_mesh(self, scene):
        """The advertised round trip: a mesh scene's description redraws the same surface and relief.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            A mesh drawn with a per-node ``node_z`` array and a declared CRS is rebuilt from its own
            ``figure_spec`` through ``Scene3D.from_figure``. The redraw must reproduce the three cells, the
            z relief (max 9.0, carried as a ``$ref`` in the stored props), and the display CRS the builder
            adopted — the guard behind the module/drawer docstring claim that a stored figure draws the same
            surface again.
        """
        from digitalearth.base.spec.bounds import same_crs

        scene.trimesh(_bare_mesh(epsg=4326), node_z=[0.0, 0.0, 0.0, 0.0, 9.0])
        again = Scene3D.from_figure(scene.figure_spec, off_screen=True)
        try:
            poly, _ = again.layers[-1]
            assert poly.n_cells == 3, poly.n_cells
            assert float(poly.points[:, 2].max()) == 9.0, poly.points
            assert same_crs(again.display_crs, 4326), again.display_crs
        finally:
            again.close()

    def test_a_mesh_with_no_drawable_face_returns_none(self, scene):
        """When every node is non-finite, ``_mesh_surface`` returns ``None`` (the drawer then skips).

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            All five nodes are non-finite, so no fan triangle survives — the surface cannot be built.
        """
        mesh = _bare_mesh(node_x=[np.inf] * 5)
        assert _mesh_surface(scene, mesh, None) is None


class TestTrimeshDrawer:
    """Tests for ``draw_trimesh`` — the skip/strict paths and the flat-fill defaults."""

    def test_an_undrawable_mesh_is_skipped_with_a_warning(self, scene, caplog):
        """A mesh that places no face is skipped (default, non-strict) at WARNING, drawing nothing.

        Args:
            scene: The off-screen scene fixture.
            caplog: pytest's log capture.

        Test scenario:
            An all-non-finite mesh leaves the scene with no layer, and a warning names ``trimesh``.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            out = scene.trimesh(_bare_mesh(node_x=[np.inf] * 5))
        assert out is None, out
        assert scene.layer_ids == [], scene.layer_ids
        assert any("trimesh" in record.getMessage() for record in caplog.records), (
            caplog.text
        )

    def test_strict_raises_when_the_mesh_places_no_face(self):
        """Under ``strict`` an undrawable mesh raises ``OffLimbError`` instead of being skipped.

        Test scenario:
            A pipeline that cannot lose a layer silently gets the same refusal the 2-D tiers raise.
        """
        strict = Scene3D(off_screen=True, strict=True)
        try:
            with pytest.raises(OffLimbError, match="trimesh"):
                strict.trimesh(_bare_mesh(node_x=[np.inf] * 5))
        finally:
            strict.close()

    def test_defaults_are_the_flat_fill_and_edges(self, scene):
        """With no colour given, the layer records the flat-fill colour, edges on, and the edge colour.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            The honest default for a value-less ``Mesh2d`` — one flat ``MESH_FILL_COLOR`` with its
            connectivity drawn in ``MESH_EDGE_COLOR``.
        """
        scene.trimesh(_bare_mesh())
        props = scene.figure_spec.layers.get(scene.layer_ids[-1]).symbology.props
        assert props["color"] == MESH_FILL_COLOR, props
        assert props["edge_color"] == MESH_EDGE_COLOR, props
        assert props["show_edges"] is True, props

    def test_opacity_is_carried_through_to_the_actor(self, scene):
        """A given ``opacity`` reaches ``add_mesh`` and the layer records it, without error.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            ``opacity=0.5`` draws a translucent surface; the branch that forwards it is exercised and the
            layer is still drawn (three cells).
        """
        actor = scene.trimesh(_bare_mesh(), opacity=0.5)
        poly, _ = scene.layers[-1]
        assert poly.n_cells == 3, poly.n_cells
        assert actor is not None
        props = scene.figure_spec.layers.get(scene.layer_ids[-1]).symbology.props
        assert props["opacity"] == 0.5, props

    def test_a_custom_color_is_recorded_and_drawn(self, scene):
        """A caller's own ``color`` replaces the default flat fill and is recorded on the layer.

        Args:
            scene: The off-screen scene fixture.

        Test scenario:
            ``color="tomato"`` is stored in the layer's props and drawn, so a stored figure redraws it.
        """
        scene.trimesh(_bare_mesh(), color="tomato")
        poly, _ = scene.layers[-1]
        assert poly.n_cells == 3, poly.n_cells
        props = scene.figure_spec.layers.get(scene.layer_ids[-1]).symbology.props
        assert props["color"] == "tomato", props
