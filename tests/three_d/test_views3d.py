"""The named views a 3-D scene can be looked from — the table, and the drift guard over it (order 26, TD-2).

The tier already round-trips a whole :class:`~digitalearth.base.spec.Camera`; what it had no way to say was
"look at this from the top". These hold the table that closes that to three things at once: the names it ships,
the PyVista methods behind them, and **where the camera actually ends up** — read off the live
:class:`pyvista.Plotter`, not off the record the scene keeps, because a preset that writes a camera nobody
applies is the failure this repo keeps finding.

The guard fails both ways on purpose: a name added to the table without being documented here fails
:meth:`TestTheTableIsHeldToPyVista.test_the_table_ships_exactly_the_documented_names`, and a name removed from
the table fails the same assertion from the other side.
"""

# The package imports have to follow pytest.importorskip("pyvista") — importing digitalearth.three_d
# without pyvista is the very thing the skip exists to avoid — so E402 is expected throughout.
# ruff: noqa: E402
import inspect

import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.spec import Camera
from digitalearth.three_d import Scene3D
from digitalearth.three_d.views import (
    NAMED_VIEWS,
    NamedView,
    forget_view,
    named_view,
    register_view,
    view_names,
)

#: The names the tier ships, written here independently of the table so the guard fails both ways.
SHIPPED = ("back", "bottom", "front", "isometric", "left", "right", "top")

#: The table as it stands at collection time. The per-row checks run over **this**, not over
#: :data:`SHIPPED`, so a row added to the table is held to PyVista even before anybody documents it here.
REGISTERED = view_names()

#: Where each shipped view stands, as the sign of ``position - focal_point`` per axis. Measured from
#: PyVista 0.48.4: ``view_xy`` looks down ``-z`` from ``+z``, ``view_xz`` from ``-y``, ``view_yz`` from ``+x``,
#: and each takes ``negative=True`` for the opposite side.
DIRECTIONS = {
    "top": (0, 0, 1),
    "bottom": (0, 0, -1),
    "front": (0, -1, 0),
    "back": (0, 1, 0),
    "right": (1, 0, 0),
    "left": (-1, 0, 0),
    "isometric": (1, 1, 1),
}


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test so no window opens in CI."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


@pytest.fixture
def table_restored():
    """Put the view table back after a test has registered into it.

    The table is process-global, as the object registry and the clause table are, so a test that extends it
    would otherwise leak a name into every test that runs later. The same dict object is refilled rather than
    replaced, because :data:`NAMED_VIEWS` is a read-only view *of that object*.

    Yields:
        None; the table is snapshotted before the test and restored after it.
    """
    from digitalearth.three_d import views

    before = dict(views._VIEWS)
    yield
    views._VIEWS.clear()
    views._VIEWS.update(before)


def _axis_signs(camera):
    """Return the sign of ``position - focal_point`` on each axis.

    Args:
        camera: A `Camera`, or PyVista's own camera — anything with `position` and `focal_point`.

    Returns:
        A three-tuple of ``-1``, ``0`` or ``1``. A component within 1e-6 of the focal point counts as ``0``,
        which is what "straight down" means for the two axes the camera is centred on.
    """
    offsets = [float(p) - float(f) for p, f in zip(camera.position, camera.focal_point)]
    return tuple(
        0 if abs(value) < 1e-6 else (1 if value > 0 else -1) for value in offsets
    )


def _scene_with_something_to_frame():
    """Return an off-screen scene holding one cube, so the presets have bounds to frame.

    Returns:
        The scene; the caller closes it.
    """
    scene = Scene3D(off_screen=True)
    scene.add_mesh(pv.Cube())
    return scene


class TestTheTableIsHeldToPyVista:
    """The drift guard: the table is data, and this is what stops it rotting."""

    def test_the_table_ships_exactly_the_documented_names(self):
        """A preset added without a word here, or removed from under one, is a failure either way."""
        assert tuple(sorted(view_names())) == SHIPPED

    @pytest.mark.parametrize("name", REGISTERED)
    def test_every_view_names_a_real_plotter_method(self, name):
        """The table is built over PyVista's own `view_*` methods; a typo in one is a broken preset."""
        method = getattr(pv.Plotter, named_view(name).method, None)
        assert callable(method), (
            f"{name!r} names {named_view(name).method!r}, which pyvista.Plotter does not offer"
        )

    @pytest.mark.parametrize("name", REGISTERED)
    def test_every_views_options_bind_to_that_method(self, name):
        """An option PyVista's method does not take would raise only when the view was first used."""
        preset = named_view(name)
        signature = inspect.signature(getattr(pv.Plotter, preset.method))
        signature.bind(None, **dict(preset.options))

    @pytest.mark.parametrize("name", REGISTERED)
    def test_every_view_describes_itself(self, name):
        """A name with no description is a table row a reader cannot use."""
        assert named_view(name).describes.strip()


class TestAViewMovesTheRealCamera:
    """What the engine received — the plotter's own camera, after the preset was applied."""

    @pytest.mark.parametrize(("name", "expected"), sorted(DIRECTIONS.items()))
    def test_the_preset_puts_the_plotter_camera_where_its_name_says(
        self, name, expected
    ):
        """Read off `pyvista.Plotter.camera`, so a preset that records a view it never applies fails."""
        scene = _scene_with_something_to_frame()
        scene.view(name)
        signs = _axis_signs(scene.plotter.camera)
        scene.close()
        assert signs == expected

    def test_the_scene_reports_the_view_as_a_camera(self):
        """The preset lands in the tier's own vocabulary, so it can be stored and set again."""
        scene = _scene_with_something_to_frame()
        camera = scene.view("top").camera
        scene.close()
        assert isinstance(camera, Camera)

    def test_a_named_view_survives_the_next_render(self):
        """`_apply_camera` rewrites the stored camera before every render, so the preset must be stored."""
        scene = _scene_with_something_to_frame()
        scene.view("top")
        scene.render()
        signs = _axis_signs(scene.plotter.camera)
        scene.close()
        assert signs == (0, 0, 1)

    def test_the_view_reaches_the_figures_panel(self):
        """A figure carries the live camera, so a scene looked at from the top describes one."""
        scene = _scene_with_something_to_frame()
        scene.view("front")
        signs = _axis_signs(scene.figure_spec.panels[0].view)
        scene.close()
        assert signs == (0, -1, 0)

    def test_two_presets_in_a_row_leave_the_second_one_showing(self):
        """The presets are absolute viewpoints, not increments, so the last one wins."""
        scene = _scene_with_something_to_frame()
        scene.view("top").view("left")
        signs = _axis_signs(scene.plotter.camera)
        scene.close()
        assert signs == (-1, 0, 0)

    def test_the_call_returns_the_scene_so_it_chains(self):
        """Every view-setting verb on this tier returns `Self`; a preset is not an exception."""
        scene = _scene_with_something_to_frame()
        returned = scene.view("isometric")
        same = returned is scene
        scene.close()
        assert same

    def test_a_view_does_not_reset_the_scenes_vertical_exaggeration(self):
        """A viewpoint says where to stand, not how tall the relief is (the H6 rule on `Camera`)."""
        scene = _scene_with_something_to_frame()
        scene.vertical_exaggeration = 4.0
        scene.view("top")
        factor = scene.vertical_exaggeration
        scene.close()
        assert factor == 4.0


class TestAnUnknownViewIsRefused:
    """A name the table does not hold."""

    def test_the_refusal_names_the_views_that_do_exist(self):
        """The way out of the error is the list of names, so the message carries it."""
        with pytest.raises(KeyError) as refusal:
            named_view("birdseye")
        assert "isometric" in refusal.value.args[0], refusal.value.args[0]

    def test_the_scene_refuses_it_too(self):
        """`view()` does not invent a preset for a name nobody registered."""
        scene = Scene3D(off_screen=True)
        with pytest.raises(KeyError):
            scene.view("birdseye")
        scene.close()


class TestACallerCanExtendTheTable:
    """Registration — the half that makes the table a table rather than seven methods."""

    def test_a_registered_view_is_applied_by_name(self, table_restored):
        """The point of the registry: a caller's own name reaches the plotter like a shipped one."""
        register_view(
            "underside",
            method="view_xy",
            describes="from below, for a cave roof",
            negative=True,
        )
        scene = _scene_with_something_to_frame()
        scene.view("underside")
        signs = _axis_signs(scene.plotter.camera)
        scene.close()
        assert signs == (0, 0, -1)

    def test_a_registered_view_is_listed(self, table_restored):
        """It is in the same table as the shipped ones, not a second mechanism beside them."""
        register_view(
            "underside", method="view_xy", describes="from below", negative=True
        )
        assert "underside" in view_names()

    def test_registration_returns_the_record(self, table_restored):
        """The caller gets the row back, so it can be held onto rather than looked up again."""
        made = register_view("underside", method="view_xy", describes="from below")
        assert isinstance(made, NamedView)

    def test_a_method_pyvista_does_not_have_is_refused(self, table_restored):
        """The mutation the guard exists for: a view over a method that is not there."""
        with pytest.raises(ValueError, match="view_from_behind"):
            register_view("nonsense", method="view_from_behind", describes="nowhere")

    def test_a_method_that_is_not_a_view_is_refused(self, table_restored):
        """A real `Plotter` method that does not set a viewpoint is not a view either."""
        with pytest.raises(ValueError, match="add_mesh"):
            register_view("nonsense", method="add_mesh", describes="not a viewpoint")

    def test_an_option_the_method_does_not_take_is_refused(self, table_restored):
        """Bound at registration, so a bad keyword fails on the line that wrote it."""
        with pytest.raises(ValueError, match="upside_down"):
            register_view(
                "nonsense", method="view_xy", describes="nowhere", upside_down=True
            )

    def test_a_name_already_in_the_table_is_refused(self, table_restored):
        """Two libraries registering `top` would otherwise silently fight over it."""
        with pytest.raises(ValueError, match="top"):
            register_view("top", method="view_xy", describes="a second top")

    def test_replace_is_how_a_caller_overrides_a_shipped_view(self, table_restored):
        """Overriding is allowed, but it has to be asked for in as many words."""
        register_view("top", method="view_xy", describes="a parallel top", replace=True)
        assert named_view("top").describes == "a parallel top"

    def test_a_blank_name_is_refused(self, table_restored):
        """A name that is not a name would sit in the table unreachable."""
        with pytest.raises(ValueError, match="name"):
            register_view("   ", method="view_xy", describes="nowhere")

    def test_a_forgotten_view_leaves_the_table(self, table_restored):
        """The table is process-global, so what goes in has to be able to come out again."""
        register_view(
            "underside", method="view_xy", describes="from below", negative=True
        )
        forget_view("underside")
        assert "underside" not in view_names()

    def test_forgetting_gives_the_row_back(self, table_restored):
        """Returned so a caller can put it back, which is how a fixture restores the table."""
        made = register_view("underside", method="view_xy", describes="from below")
        assert forget_view("underside") == made

    def test_forgetting_a_name_nothing_registered_is_refused(self, table_restored):
        """The same refusal as looking one up, since it is the same question about the same table."""
        with pytest.raises(KeyError, match="birdseye"):
            forget_view("birdseye")

    def test_the_read_only_view_shows_a_registered_row(self, table_restored):
        """`NAMED_VIEWS` is a view *of* the live table, not a copy taken at import."""
        register_view("underside", method="view_xy", describes="from below")
        assert "underside" in NAMED_VIEWS
