"""The 3-D renderer seam: a scene described as a figure, and drawn from it (DE-23a + U-1, #295).

The tier drew straight into a PyVista plotter and kept `(mesh, actor)` pairs in a list, so a scene could be
drawn and then only looked at: no layer had an id, a kind, a source or a style once the call returned. These
cover what replaces that — a `FigureSpec` every builder writes into, a renderer that draws from it, layers
addressed by id, a camera as a value, and the tier's capability declaration.
"""

import json

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.capabilities import Capabilities  # noqa: E402
from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.base.spec import Camera, FigureSpec, LayerSpec  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402
from digitalearth.three_d.capabilities import CAPABILITIES  # noqa: E402
from digitalearth.three_d.renderer import drawer_for  # noqa: E402

DEM_PATH = "examples/data/acc4000.tif"


def _dem() -> np.ndarray:
    """Return a small ramped elevation grid.

    Returns:
        The elevation array.
    """
    return np.add.outer(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 8))


@pytest.fixture
def scene():
    """Yield an off-screen scene, closed on the way out.

    Yields:
        The scene.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


class TestTheDescription:
    """Every builder writes a layer into the scene's figure."""

    def test_a_builder_describes_what_it_drew(self, scene):
        """The layer carries its kind, its source and the keywords it was given.

        Args:
            scene: The scene under test.

        Test scenario:
            Before the seam, `scene.terrain(dem, cmap="terrain")` left a mesh and an actor and nothing else —
            the kind, the source and the colormap were gone the moment PyVista had the geometry.
        """
        scene.terrain(get_source(_dem()), cmap="magma")
        layer = scene.figure_spec.layers.get("terrain-1")
        assert (layer.kind, layer.symbology.props["cmap"]) == ("terrain", "magma"), (
            layer.to_dict()
        )

    def test_the_source_is_named_by_the_layer(self, scene):
        """A layer's source is in the figure's sources, under the layer's own id."""
        scene.terrain(get_source(_dem()))
        figure = scene.figure_spec
        assert figure.layers.get("terrain-1").source_id in figure.sources, (
            figure.sources
        )

    def test_a_named_layer_keeps_its_name(self, scene):
        """`name=` is the id the layer is addressed by, as it is on the web tier."""
        scene.terrain(get_source(_dem()), name="relief")
        assert scene.layer_ids == ["relief"], scene.layer_ids

    def test_layers_are_numbered_by_kind(self, scene):
        """Unnamed layers get readable ids rather than positions."""
        scene.terrain(get_source(_dem()))
        scene.point_cloud(np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]]))
        assert scene.layer_ids == ["terrain-1", "point_cloud-1"], scene.layer_ids

    def test_a_skipped_layer_is_not_described(self, scene):
        """A raster with nothing to draw leaves no layer behind, so the description lists what is there."""
        assert scene.terrain(get_source(np.full((4, 4), np.nan))) is None, "skipped"
        assert scene.layer_ids == [], scene.layer_ids

    def test_every_builder_writes_its_kind(self, scene):
        """The kinds the tier declares are the kinds its builders record.

        Test scenario:
            One layer per builder that needs no optional dependency, checked against the declaration so a
            builder and its capability row cannot drift apart.
        """
        scene.terrain(get_source(_dem()))
        scene.point_cloud(np.array([[0.0, 0.0, 1.0]]))
        scene.volume(np.random.default_rng(3).random((6, 6, 6)))
        scene.isosurface(np.random.default_rng(3).random((6, 6, 6)), isosurfaces=[0.5])
        scene.vectors(np.zeros((2, 3)), np.ones((2, 3)))
        kinds = {scene.figure_spec.layers.get(name).kind for name in scene.layer_ids}
        assert kinds <= CAPABILITIES.kinds, f"undeclared kinds drawn: {kinds}"

    def test_every_kind_the_tier_has_a_drawer_for_is_declared(self):
        """The builders above need no optional dependency; the tier draws more kinds than they cover.

        Test scenario:
            The check above exercises five builders, and `globe` and `extruded_polygons` are not among them
            — which is exactly where the hole was: `globe(data)` records a `coastlines` layer by default,
            of a kind the declaration did not list and its `absent` said could not exist (review M6).
            Reading the drawer table covers every kind without needing geovista installed to do it.
        """
        from digitalearth.three_d.renderer import DRAWN_KINDS, drawer_for

        undeclared = sorted(set(DRAWN_KINDS) - CAPABILITIES.kinds)
        assert undeclared == [], f"the tier draws {undeclared} without declaring them"
        unresolved = [kind for kind in DRAWN_KINDS if drawer_for(kind) is None]
        assert unresolved == [], f"{unresolved} are declared drawable with no drawer"


class TestDrawingFromTheDescription:
    """A figure is what the renderer draws, so it can be stored and drawn again."""

    def test_a_figure_drawn_by_another_scene_has_the_same_layers(self, scene):
        """The description is enough to rebuild the scene, without the objects being copied."""
        scene.terrain(get_source(_dem()), cmap="terrain")
        again = Scene3D.from_figure(scene.figure_spec, off_screen=True)
        try:
            assert again.layer_ids == ["terrain-1"], again.layer_ids
            assert (
                again.mesh_of("terrain-1").n_points
                == scene.mesh_of("terrain-1").n_points
            ), "the redrawn mesh must be the same surface"
        finally:
            again.close()

    def test_a_figure_survives_json_when_the_data_has_a_path(self):
        """`FigureSpec.from_dict(scene.figure_spec.to_dict())` renders the same picture.

        Test scenario:
            The round trip the seam exists for. The data is given as a path, since a reference into this
            process's memory is deliberately not storable (#285).
        """
        first = Scene3D(off_screen=True, window_size=(200, 150))
        try:
            first.terrain(DEM_PATH, cmap="terrain")
            stored = first.figure_spec.to_dict()
            before = first.screenshot()
        finally:
            first.close()
        second = Scene3D.from_figure(
            FigureSpec.from_dict(stored), off_screen=True, window_size=(200, 150)
        )
        try:
            after = second.screenshot()
        finally:
            second.close()
        assert pv.compare_images(before, after) < 100.0, "the redrawn figure differs"

    def test_a_kind_the_tier_cannot_draw_is_refused_by_name(self, scene):
        """A figure written for another backend names what it wanted drawn."""
        with pytest.raises(KeyError, match="does not draw 'choropleth'"):
            drawer_for("choropleth")

    def test_the_renderer_holds_what_it_drew(self, scene):
        """A layer id reaches the mesh and the actor behind it."""
        scene.terrain(get_source(_dem()))
        assert scene.actor_of("terrain-1") is scene.renderer.drawn["terrain-1"][1], (
            "actor_of must hand back what the renderer drew"
        )

    def test_an_unknown_layer_id_is_refused(self, scene):
        """The message names the ids that are there."""
        with pytest.raises(KeyError, match="no layer 'nope' in this scene"):
            scene.mesh_of("nope")


class TestAddressingLayers:
    """Remove, hide and reorder, by id, reflected in the description."""

    def test_a_layer_is_removed_by_id(self, scene):
        """The layer leaves the figure and the plotter together."""
        scene.terrain(get_source(_dem()))
        scene.remove_layer("terrain-1")
        assert (scene.layer_ids, len(scene.layers)) == ([], 0), scene.layer_ids

    def test_removing_a_layer_forgets_its_source(self, scene):
        """A removed layer's data is not kept in the figure it is no longer part of."""
        scene.terrain(get_source(_dem()))
        scene.remove_layer("terrain-1")
        assert dict(scene.figure_spec.sources) == {}, scene.figure_spec.sources

    def test_an_unknown_layer_cannot_be_removed(self, scene):
        """The refusal names the layers that are there."""
        with pytest.raises(KeyError, match="no layer 'nope' in this scene"):
            scene.remove_layer("nope")

    def test_a_layer_is_hidden_and_shown_by_id(self, scene):
        """Visibility is recorded on the layer and applied to its actor."""
        scene.terrain(get_source(_dem()))
        scene.set_visible("terrain-1", False)
        hidden = bool(scene.actor_of("terrain-1").visibility)
        scene.set_visible("terrain-1", True)
        assert (hidden, bool(scene.actor_of("terrain-1").visibility)) == (
            False,
            True,
        ), "the actor must follow the description"

    def test_a_hidden_layer_stays_in_the_description(self, scene):
        """A hidden layer is still there, which is what lets a viewer switch it back on."""
        scene.terrain(get_source(_dem()))
        scene.set_visible("terrain-1", False)
        assert scene.figure_spec.layers.is_visible("terrain-1") is False, "recorded"

    def test_layers_are_reordered_by_id(self, scene):
        """Draw order is part of the description, even where VTK composites by depth."""
        scene.terrain(get_source(_dem()), name="a")
        scene.terrain(get_source(_dem()), name="b")
        assert scene.move_layer("b", 0).layer_ids == ["b", "a"], scene.layer_ids

    def test_the_compatibility_view_follows_the_order(self, scene):
        """`scene.layers` still hands out `(mesh, actor)` pairs, in the figure's order."""
        scene.terrain(get_source(_dem()), name="a")
        scene.point_cloud(np.array([[0.0, 0.0, 1.0]]), name="b")
        scene.move_layer("b", 0)
        assert scene.layers[0][0] is scene.mesh_of("b"), "the pairs follow the tree"


class TestTheCamera:
    """The view is a `Camera`: set before a render, read back after one."""

    def test_a_camera_set_before_drawing_survives_the_render(self, scene):
        """PyVista resets its camera to fit the first mesh; a camera the caller set is applied again."""
        scene.camera = Camera((5.0, -5.0, 5.0), focal_point=(0.0, 0.0, 0.0))
        scene.terrain(get_source(_dem()))
        scene.screenshot()
        assert [round(value) for value in scene.camera.position] == [5, -5, 5], (
            scene.camera
        )

    def test_the_camera_is_read_back_from_the_scene(self, scene):
        """After a render the camera is the live view, so it can be written down and set again."""
        scene.terrain(get_source(_dem()))
        scene.screenshot()
        assert scene.camera.focal_point is not None, scene.camera

    def test_the_camera_carries_the_exaggeration_and_the_crs(self, scene):
        """Both belong to the view, so both travel with it."""
        scene.terrain(get_source(_dem()), z_exaggeration=3.0)
        camera = scene.camera
        assert camera.vertical_exaggeration == 3.0, camera

    def test_the_exaggeration_is_readable_before_anything_is_drawn(self):
        """Reading the view scale must not open a render window (#290)."""
        built = Scene3D(off_screen=True)
        try:
            built.vertical_exaggeration = 2.0
            assert built._plotter is None, "reading the view must not build a plotter"
            assert built.vertical_exaggeration == 2.0, built.vertical_exaggeration
        finally:
            built.close()

    def test_the_camera_round_trips_through_the_figure(self, scene):
        """The camera is the panel's view, so it is stored and read back with the figure."""
        scene.camera = Camera.look_at(
            (0.0, 0.0, 0.0), azimuth=225.0, elevation=30.0, distance=8.0
        )
        scene.terrain(DEM_PATH)
        stored = scene.figure_spec.to_dict()
        assert FigureSpec.from_dict(stored).panels[0].view == scene.camera, stored[
            "panels"
        ]

    def test_something_that_is_not_a_camera_is_refused(self, scene):
        """PyVista's three-tuple `camera_position` says nothing about projection or exaggeration."""
        with pytest.raises(ValueError, match="camera must be a Camera"):
            scene.camera = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]


class TestACallersOwnMesh:
    """`add_mesh` and `add_volume` are a caller's own objects, recorded as custom layers (#293)."""

    def test_a_mesh_a_caller_built_is_addressable(self, scene):
        """The object is drawn and described, so it can be removed like any other layer."""
        scene.add_mesh(pv.Sphere(radius=0.5), name="ball")
        assert scene.layer_ids == ["ball"], scene.layer_ids
        assert scene.figure_spec.layers.get("ball").kind == "custom:pyvista", "kind"

    def test_the_object_is_held_by_the_scene_not_the_figure(self, scene):
        """A figure describes a custom layer; the mesh stays with the scene that was handed it."""
        mesh = pv.Sphere(radius=0.5)
        scene.add_mesh(mesh, name="ball")
        assert scene.held_objects["ball"] is mesh, scene.held_objects
        stored = json.dumps(scene.figure_spec.layers.get("ball").to_dict())
        assert "PolyData" not in stored, (
            f"the layer must describe the object, not carry its type: {stored}"
        )
        assert "points" not in stored, (
            f"the layer must describe the object, not carry its vertices: {stored}"
        )

    def test_removing_it_forgets_the_object(self, scene):
        """The scene does not keep an object for a layer it no longer has."""
        scene.add_mesh(pv.Sphere(radius=0.5), name="ball")
        scene.remove_layer("ball")
        assert scene.held_objects == {}, scene.held_objects

    def test_a_volume_a_caller_built_is_ray_cast(self, scene):
        """`add_volume` records the same kind, and draws through PyVista's volume path."""
        grid = pv.ImageData(dimensions=(6, 6, 6))
        grid.cell_data["v"] = np.linspace(0.0, 1.0, 125)
        scene.add_volume(grid, name="cube")
        assert scene.figure_spec.layers.get("cube").symbology.props["volume"] is True, (
            "the description says how it is drawn"
        )

    def test_a_figure_whose_object_is_gone_skips_the_layer(self, scene):
        """A custom layer drawn where the object is not held follows C7: skipped with a warning."""
        scene.add_mesh(pv.Sphere(radius=0.5), name="ball")
        elsewhere = Scene3D(off_screen=True)
        try:
            elsewhere.draw_figure(scene.figure_spec)
            assert elsewhere.layer_ids == ["ball"], elsewhere.layer_ids
            assert elsewhere.renderer.drawn == {}, "nothing was drawn for it"
        finally:
            elsewhere.close()

    def test_a_strict_scene_refuses_a_missing_object(self, scene):
        """Under `strict`, the same case raises rather than warning."""
        from digitalearth.base.crs import OffLimbError

        scene.add_mesh(pv.Sphere(radius=0.5), name="ball")
        elsewhere = Scene3D(off_screen=True, strict=True)
        try:
            with pytest.raises(OffLimbError, match="does not carry"):
                elsewhere.draw_figure(scene.figure_spec)
        finally:
            elsewhere.close()


class TestTheDeclaration:
    """`capabilities.py` says what the tier can draw, and is kept honest by what it draws (#294)."""

    def test_the_declaration_is_this_backend(self):
        """The row is named as `quickmap(backend=...)` spells it."""
        assert CAPABILITIES.backend == "3d", CAPABILITIES.backend

    def test_every_declared_kind_has_a_drawer(self):
        """A declared kind a renderer could not draw would be a capability lie."""
        undrawable = [
            kind for kind in sorted(CAPABILITIES.kinds) if not drawer_for(kind)
        ]
        assert undrawable == [], f"declared but undrawable: {undrawable}"

    def test_the_declaration_loads_without_pyvista(self):
        """A dispatcher reads it before it decides which backend to build.

        Test scenario:
            Run in a subprocess, since this session has PyVista loaded already.
        """
        import subprocess
        import sys

        code = (
            "import sys; from digitalearth.three_d.capabilities import CAPABILITIES;"
            "print('pyvista' in sys.modules, CAPABILITIES.backend)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False 3d", result.stdout or result.stderr

    def test_what_is_absent_says_why(self):
        """The half that keeps the declaration honest: a decision, not a gap."""
        assert CAPABILITIES.reason("domain").startswith("a scene is framed"), (
            CAPABILITIES.absent
        )

    def test_the_dispatcher_reads_the_declaration(self):
        """`api.BACKEND_CAPABILITIES["3d"]` is derived, not written out beside the tier."""
        from digitalearth.api import BACKEND_CAPABILITIES

        assert sorted(BACKEND_CAPABILITIES["3d"]) == ["colorbar", "crs"], (
            BACKEND_CAPABILITIES["3d"]
        )

    def test_a_refusal_carries_the_declared_reason(self):
        """A caller is told what the tier does instead of what they asked for."""
        from digitalearth import quickmap

        with pytest.raises(ValueError, match="framed by its camera"):
            quickmap(np.zeros((4, 4)), backend="3d", domain="europe")

    def test_the_declaration_is_a_capabilities_value(self):
        """It is the shared type, so the support matrix can be built from every tier's row the same way."""
        assert isinstance(CAPABILITIES, Capabilities), type(CAPABILITIES)
        assert sorted(CAPABILITIES.to_dict()) == [
            "absent",
            "backend",
            "channels",
            "data_driven",
            "features",
            "kinds",
            "schemes",
        ], CAPABILITIES.to_dict()


class TestTheLayerSpecTheSceneWrites:
    """The description a builder writes is a plain `LayerSpec`, with nothing engine-shaped in it."""

    def test_the_layer_is_a_layer_spec(self, scene):
        """Nothing tier-specific stands in for it."""
        scene.terrain(get_source(_dem()))
        assert isinstance(scene.figure_spec.layers.get("terrain-1"), LayerSpec), "spec"

    def test_an_array_is_referenced_rather_than_copied(self, scene):
        """A point cloud's values are stored as a reference, so the description stays small."""
        values = np.linspace(0.0, 1.0, 3)
        scene.point_cloud(np.zeros((3, 3)), values=values)
        stored = scene.figure_spec.layers.get("point_cloud-1").symbology.props["values"]
        assert sorted(stored) == ["$ref"], stored


class TestTheRemainingArms:
    """The paths the seam's other tests reach around: a live plotter, a skipped layer, a restyle."""

    def test_the_exaggeration_reaches_a_plotter_that_already_exists(self, scene):
        """Set after the first render, the view scale is applied to the window that is open.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()))
        scene.screenshot()
        scene.vertical_exaggeration = 4.0
        assert float(scene.plotter.renderer.scale[2]) == 4.0, (
            scene.plotter.renderer.scale
        )

    def test_a_plotter_handed_to_the_scene_gets_the_scene_s_view(self, scene):
        """Swapping the window is not a reset of the scene.

        Args:
            scene: The scene under test.

        Test scenario:
            The measured defect: the view scale and camera were applied only where the scene *built* a
            plotter, so assigning one — the documented way to swap one in — left the replacement at true
            scale and PyVista's default viewpoint, with nothing to re-apply them later (review M7).
        """
        scene.vertical_exaggeration = 3.0
        scene.camera = Camera((9.0, 9.0, 9.0))
        replacement = pv.Plotter(off_screen=True)
        try:
            scene.plotter = replacement
            assert tuple(replacement.scale) == (1.0, 1.0, 3.0), replacement.scale
            assert [round(value) for value in replacement.camera.position] == [
                9,
                9,
                9,
            ], replacement.camera.position
        finally:
            replacement.close()

    def test_two_layers_of_one_kind_describe_their_own_data(self, scene):
        """A source is keyed by the layer that draws it, not by the kind of layer it is.

        Args:
            scene: The scene under test.

        Test scenario:
            The measured defect: both terrains registered under `object:terrain`, so the second replaced
            the first and the figure described one dataset twice (review H1).
        """
        low = get_source(_dem())
        high = get_source(_dem() * 10.0)
        scene.terrain(low)
        scene.terrain(high)
        sources = scene.figure_spec.sources
        drawn = [
            float(np.nanmax(sources[layer_id].open().z.values))
            for layer_id in scene.layer_ids
        ]
        assert drawn == [
            float(np.nanmax(low.z.values)),
            float(np.nanmax(high.z.values)),
        ], drawn

    def test_a_second_scene_leaves_the_first_scene_s_sources_alone(self, scene):
        """Two scenes number their layers the same way; their data must not share an entry.

        Args:
            scene: The scene under test.

        Test scenario:
            The object registry is process-global. Before the namespace, building a second scene
            retroactively re-pointed a figure the first scene had already handed out (review H1).
        """
        low = get_source(_dem())
        scene.terrain(low)
        kept = scene.figure_spec.sources[scene.layer_ids[0]]
        other = Scene3D(off_screen=True)
        try:
            other.terrain(get_source(_dem() * 10.0))
            assert float(np.nanmax(kept.open().z.values)) == float(
                np.nanmax(low.z.values)
            ), "the first scene's source must still be its own"
        finally:
            other.close()

    def test_a_figure_the_renderer_refuses_does_not_become_the_scene(self, scene):
        """A change the plotter would not accept leaves the scene as it was.

        Args:
            scene: The scene under test.

        Test scenario:
            The measured defect: the figure was installed first and `apply` raised afterwards, so a scene
            that had refused a layer still listed it — `layer_ids`, `figure_spec` and `to_dict` all
            advertising something that was never drawn and could not be (review H7).
        """
        from dataclasses import replace as with_fields

        scene.terrain(get_source(_dem()), name="a")
        figure = scene.figure_spec
        refused = with_fields(
            figure, layers=figure.layers.add(LayerSpec("chor", "choropleth"))
        )
        with pytest.raises(KeyError, match="does not draw"):
            scene._change(refused)
        assert scene.layer_ids == ["a"], scene.layer_ids

    def test_the_scene_still_works_after_a_refused_change(self, scene):
        """Nothing is stuck: the layers that were there can still be removed.

        Args:
            scene: The scene under test.
        """
        from dataclasses import replace as with_fields

        scene.terrain(get_source(_dem()), name="a")
        figure = scene.figure_spec
        refused = with_fields(
            figure, layers=figure.layers.add(LayerSpec("chor", "choropleth"))
        )
        with pytest.raises(KeyError):
            scene._change(refused)
        scene.remove_layer("a")
        assert scene.layer_ids == [], scene.layer_ids

    def test_a_viewpoint_does_not_flatten_the_relief(self, scene):
        """A camera that names no exaggeration says where to stand, not how tall the relief is.

        Args:
            scene: The scene under test.

        Test scenario:
            The measured defect: `terrain(z_exaggeration=3.0)` followed by any `Camera.look_at(...)` came
            back at true scale, with the plotter rescaled and nothing said (review H6). `Camera`'s default
            was `1.0`, and the setter read that as an instruction.
        """
        scene.terrain(get_source(_dem()), z_exaggeration=3.0)
        scene.camera = Camera.look_at(
            (0.0, 0.0, 0.0), azimuth=45, elevation=30, distance=10.0
        )
        assert scene.vertical_exaggeration == 3.0, scene.vertical_exaggeration

    def test_a_camera_that_names_one_still_sets_it(self, scene):
        """The other arm: an exaggeration written on the view is the view's to apply.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()), z_exaggeration=3.0)
        scene.camera = Camera((7.0, -7.0, 7.0), vertical_exaggeration=2.0)
        assert scene.vertical_exaggeration == 2.0, scene.vertical_exaggeration

    def test_a_camera_set_after_a_render_is_applied_at_once(self, scene):
        """There is a plotter to put it on, so it does not wait for the next render.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()))
        scene.screenshot()
        scene.camera = Camera((7.0, -7.0, 7.0))
        assert [round(value) for value in scene.plotter.camera.position] == [
            7,
            -7,
            7,
        ], scene.plotter.camera.position

    def test_a_parallel_camera_carries_its_scale(self, scene):
        """A parallel projection is zoomed by `parallel_scale`, so that is what is applied.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()))
        scene.camera = Camera((1.0, -1.0, 1.0), parallel=True, parallel_scale=5.0)
        scene.screenshot()
        assert scene.plotter.camera.parallel_scale == 5.0, scene.plotter.camera

    def test_the_camera_is_readable_before_anything_is_drawn(self):
        """A scene that has not rendered still says where it will look from."""
        built = Scene3D(off_screen=True)
        try:
            built.camera = Camera((2.0, -2.0, 2.0))
            assert built._plotter is None, "reading the camera must not build a plotter"
            assert built.camera.position == (2.0, -2.0, 2.0), built.camera
        finally:
            built.close()

    def test_a_described_but_undrawn_layer_has_no_mesh(self, scene):
        """A custom layer whose object is not here is in the figure and not on the plotter.

        Args:
            scene: The scene under test.
        """
        scene.add_mesh(pv.Sphere(radius=0.5), name="ball")
        elsewhere = Scene3D(off_screen=True)
        try:
            elsewhere.draw_figure(scene.figure_spec)
            assert elsewhere.mesh_of("ball") is None, "nothing was drawn for it"
            assert elsewhere.remove_layer("ball").layer_ids == [], "it still removes"
        finally:
            elsewhere.close()

    def test_an_unknown_layer_cannot_be_hidden(self, scene):
        """Visibility is addressed by id, and an id nobody used is refused by name.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(KeyError, match="no layer 'nope' in this scene"):
            scene.set_visible("nope", False)

    def test_a_hidden_layer_is_drawn_hidden(self, scene):
        """A figure that says a layer is off is drawn with it off, not drawn and then hidden.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()))
        scene.set_visible("terrain-1", False)
        elsewhere = Scene3D(off_screen=True)
        try:
            elsewhere.draw_figure(scene.figure_spec)
            assert not bool(elsewhere.actor_of("terrain-1").visibility), "drawn hidden"
        finally:
            elsewhere.close()

    def test_a_restyled_layer_is_drawn_again(self, scene):
        """VTK has no cheap restyle, so the layer is rebuilt from its new description.

        Args:
            scene: The scene under test.

        Test scenario:
            The diff's `restyled` arm: same id, same source, another colormap.
        """
        from dataclasses import replace

        from digitalearth.base.spec import Symbology

        scene.terrain(get_source(_dem()), cmap="terrain")
        before = scene.mesh_of("terrain-1")
        figure = scene.figure_spec
        layer = figure.layers.get("terrain-1")
        restyled = replace(
            figure,
            layers=figure.layers.replace(
                replace(
                    layer,
                    symbology=Symbology(
                        props={**dict(layer.symbology.props), "cmap": "magma"}
                    ),
                )
            ),
        )
        scene.draw_figure(restyled)
        assert scene.mesh_of("terrain-1") is not before, "the layer must be drawn again"

    def test_removing_a_layer_that_was_never_drawn_is_harmless(self, scene):
        """A described layer with nothing on the plotter still comes off the figure.

        Args:
            scene: The scene under test.
        """
        scene.add_mesh(pv.Sphere(radius=0.5), name="ball")
        elsewhere = Scene3D(off_screen=True)
        try:
            elsewhere.draw_figure(scene.figure_spec)
            elsewhere.remove_layer("ball")
            assert elsewhere.renderer.drawn == {}, elsewhere.renderer.drawn
        finally:
            elsewhere.close()

    def test_hiding_a_layer_that_was_never_drawn_is_harmless(self, scene):
        """A described layer with nothing on the plotter can still be switched off.

        Args:
            scene: The scene under test.
        """
        scene.add_mesh(pv.Sphere(radius=0.5), name="ball")
        elsewhere = Scene3D(off_screen=True)
        try:
            elsewhere.draw_figure(scene.figure_spec)
            elsewhere.set_visible("ball", False)
            assert elsewhere.figure_spec.layers.is_visible("ball") is False, "recorded"
        finally:
            elsewhere.close()

    def test_a_third_name_collision_gets_a_third_number(self, scene):
        """Ids stay unique however many times a name is reused.

        Args:
            scene: The scene under test.
        """
        for _ in range(3):
            scene.terrain(get_source(_dem()), name="relief")
        assert scene.layer_ids == ["relief", "relief-2", "relief-3"], scene.layer_ids

    def test_hiding_a_layer_through_a_figure_toggles_its_actor(self, scene):
        """The diff's `hidden` arm reaches the actor, not only the description.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()))
        hidden = scene.figure_spec.layers.set_visible("terrain-1", False)
        from dataclasses import replace

        scene.draw_figure(replace(scene.figure_spec, layers=hidden))
        assert not bool(scene.actor_of("terrain-1").visibility), "the actor must follow"

    def test_a_figure_drawn_through_a_flat_view_leaves_the_camera_alone(self, scene):
        """A figure whose panel is a flat map says nothing about where a scene looks from.

        Args:
            scene: The scene under test.

        Test scenario:
            `FigureSpec` panels hold a `Viewport` or a `Camera`; only the second is a 3-D view, and the other
            is drawn without touching the scene's own camera.
        """
        from digitalearth.base.spec import PanelSpec, Viewport

        before = scene.camera
        scene.draw_figure(FigureSpec(panels=(PanelSpec("scene", Viewport(4326)),)))
        assert scene.camera == before, scene.camera

    def test_a_repeated_name_is_suffixed(self, scene):
        """Two layers cannot share an id, so the second name is numbered.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()), name="relief")
        scene.terrain(get_source(_dem()), name="relief")
        assert scene.layer_ids == ["relief", "relief-2"], scene.layer_ids


class TestTheContractNames:
    """The Core names the 3-D tier answers to (#299)."""

    def test_a_layer_is_read_back_by_id(self, scene):
        """`get_layer` hands out the description a builder recorded.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()))
        assert scene.get_layer("terrain-1").kind == "terrain", scene.get_layer(
            "terrain-1"
        )

    def test_reading_an_unknown_layer_is_refused(self, scene):
        """The message names the layers that are there.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(KeyError, match="no layer 'nope' in this scene"):
            scene.get_layer("nope")

    def test_a_layer_is_replaced_in_place(self, scene):
        """`replace_layer` keeps the id and the place, and the renderer draws it again.

        Args:
            scene: The scene under test.
        """
        from dataclasses import replace

        from digitalearth.base.spec import Symbology

        scene.terrain(get_source(_dem()), cmap="terrain")
        held = scene.get_layer("terrain-1")
        scene.replace_layer(
            replace(
                held,
                symbology=Symbology(
                    props={**dict(held.symbology.props), "cmap": "magma"}
                ),
            )
        )
        assert scene.get_layer("terrain-1").symbology.props["cmap"] == "magma", (
            "restyled"
        )
        assert scene.layer_ids == ["terrain-1"], scene.layer_ids

    def test_replacing_an_unknown_layer_is_refused(self, scene):
        """A description whose id nobody drew names what is there instead.

        Args:
            scene: The scene under test.
        """
        absent = LayerSpec("nope", "terrain")
        with pytest.raises(KeyError, match="no layer 'nope' in this scene"):
            scene.replace_layer(absent)

    def test_render_hands_back_the_plotter(self, scene):
        """Every tier's `render` returns its own engine object; here that is the plotter.

        Args:
            scene: The scene under test.
        """
        scene.terrain(get_source(_dem()))
        assert scene.render() is scene.plotter, "render must hand back the plotter"

    def test_the_callback_loop_is_recorded_under_its_own_name(self, scene, tmp_path):
        """`record` writes the frames; `animate` still forwards, warning once (#299).

        Args:
            scene: The scene under test.
            tmp_path: Where the GIF is written.
        """
        import warnings

        scene.terrain(get_source(_dem()))
        out = tmp_path / "grow.gif"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene.animate([1.0], str(out), lambda s, frame: None)
        messages = [str(record.message) for record in caught]
        assert any("use Scene3D.record()" in message for message in messages), messages
        assert out.stat().st_size > 0, "the alias must still write the file"
