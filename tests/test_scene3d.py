"""Tests for digitalearth.three_d.Scene3D — the PyVista shared-plotter host.

Gated on the optional ``3d`` extra: when pyvista is not installed these are skipped, so the default suite stays
green; install ``digitalearth[3d]`` (or run the ``viz3d`` pixi env) to exercise them for real.
"""
# The package imports have to follow pytest.importorskip("pyvista") — importing digitalearth.three_d
# without pyvista is the very thing the skip exists to avoid — so E402 is expected throughout.
# ruff: noqa: E402
import sys
import types

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D, base
from digitalearth.three_d.base import Scene3DBase, house_theme


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test so no window opens in CI."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _dem_grid(n: int = 12) -> "pv.ImageData":
    """Build a small ramped DEM as a warped ImageData (a non-flat surface to render)."""
    arr = np.add.outer(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
    grid = pv.ImageData(dimensions=(n, n, 1))
    grid.point_data["z"] = arr.ravel(order="F")
    return grid.warp_by_scalar("z", factor=3.0)


def test_scene_starts_empty():
    """A fresh Scene3D owns a plotter and no layers."""
    scene = Scene3D(off_screen=True)
    assert scene.plotter is not None
    assert scene.layers == []
    scene.close()


def test_add_mesh_registers_one_layer():
    """add_mesh adds the actor to the plotter and registers exactly one (mesh, actor) layer."""
    scene = Scene3D(off_screen=True)
    actor = scene.add_mesh(_dem_grid(), scalars="z", cmap="terrain")
    assert actor is not None
    assert len(scene.layers) == 1
    mesh, registered = scene.layers[0]
    assert registered is actor
    scene.close()


def test_two_layers_compose_on_one_plotter():
    """Two meshes stack on the same plotter and both register as layers."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z")
    scene.add_mesh(pv.Sphere(radius=0.5, center=(5, 5, 5)))
    assert len(scene.layers) == 2
    scene.close()


def test_screenshot_is_a_nonempty_rgb_frame():
    """Off-screen screenshot returns a real (H, W, 3) RGB array with content."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z", cmap="terrain")
    img = scene.screenshot()
    assert img.ndim == 3 and img.shape[-1] == 3
    assert bool(img.any())
    scene.close()


def test_screenshot_writes_png(tmp_path):
    """screenshot(path=...) writes a PNG file to disk."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z")
    out = tmp_path / "frame.png"
    scene.screenshot(path=str(out))
    assert out.exists() and out.stat().st_size > 0
    scene.close()


def test_save_dispatches_png_vs_html(tmp_path):
    """save() writes a PNG for image paths and a self-contained HTML page for .html paths."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z")
    png, html = tmp_path / "s.png", tmp_path / "s.html"
    png_result = scene.save(str(png))
    html_result = scene.save(str(html))
    assert png.exists() and png.stat().st_size > 0
    assert html.exists() and html.stat().st_size > 0
    # save() returns the RGB frame for a raster path and None for the HTML path
    assert png_result is not None and png_result.ndim == 3
    assert html_result is None
    scene.close()


def test_add_volume_registers_layer():
    """add_volume() ray-casts a scalar field and registers it as a layer."""
    scene = Scene3D(off_screen=True)
    grid = pv.ImageData(dimensions=(6, 6, 6))
    grid.cell_data["v"] = np.linspace(0.0, 1.0, 5 * 5 * 5)
    actor = scene.add_volume(grid, cmap="viridis")
    assert actor is not None and len(scene.layers) == 1
    scene.close()


def test_show_off_screen_does_not_block():
    """show() renders a frame off-screen and returns without opening a blocking window."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid())
    scene.show()  # must return (off-screen); no assertion beyond "did not hang/raise"
    scene.close()


def test_context_manager_closes_plotter():
    """The context manager closes the plotter on exit and does not suppress exceptions."""
    with Scene3D(off_screen=True) as scene:
        scene.add_mesh(_dem_grid(), scalars="z")
        assert len(scene.layers) == 1
    # after exit the plotter is closed; re-closing is a no-op (must not raise)
    scene.close()

    with pytest.raises(ValueError):
        with Scene3D(off_screen=True):
            raise ValueError("propagates")


class _RecordingComponent:
    """Stand-in for the `trame` plotter component pyvista >=0.49 registers, recording its export calls."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def export_html(self, path):
        """Record the destination (or raise the configured error) the way trame-pyvista's component would."""
        self.calls.append(path)
        if self.error is not None:
            raise self.error


class _RecordingPlotter:
    """Minimal `pyvista.Plotter` stand-in that records `export_html` calls and may carry a `trame` component.

    `Scene3DBase.export_html` picks its branch off the presence of a `trame` attribute, which is exactly the
    difference between pyvista 0.48 and >=0.49. A stub carries both shapes in one process, so each branch is
    reachable whichever pyvista is installed — the real plotter can only ever exercise one of them.
    """

    def __init__(self, component=None, error=None):
        self.calls = []
        self.error = error
        if component is not None:
            self.trame = component

    def export_html(self, path):
        """Record the destination (or raise the configured error) the way `pyvista.Plotter` 0.48 would."""
        self.calls.append(path)
        if self.error is not None:
            raise self.error


def _stub_scene(component=None, error=None):
    """Build a Scene3DBase around a recording stub, with no real render window to leak.

    `export_html` and `save` only ever touch `self.plotter`, so bypassing `__init__` keeps these tests free of
    the VTK context a real `Scene3DBase(off_screen=True)` would open and never close.
    """
    scene = Scene3DBase.__new__(Scene3DBase)
    scene.plotter = _RecordingPlotter(component=component, error=error)
    scene.layers = []
    return scene


class TestHouseTheme:
    """Tests for house_theme."""

    def test_returns_the_documented_document_theme(self):
        """house_theme() returns a DocumentTheme carrying every setting its docstring promises.

        Test scenario:
            The returned theme is a `pyvista.themes.DocumentTheme` with a white background, the `viridis`
            colormap, SSAA anti-aliasing and a black font — the publication-grade defaults the tier renders with.
        """
        theme = house_theme()
        assert isinstance(theme, pv.themes.DocumentTheme), f"Expected a DocumentTheme, got {type(theme).__name__}"
        assert theme.background.hex_rgb == "#ffffff", f"Expected a white background, got {theme.background}"
        assert theme.cmap == "viridis", f"Expected the viridis cmap, got {theme.cmap!r}"
        assert theme.anti_aliasing == "ssaa", f"Expected ssaa anti-aliasing, got {theme.anti_aliasing!r}"
        assert theme.font.color.hex_rgb == "#000000", f"Expected a black font, got {theme.font.color}"

    def test_returns_an_independent_theme_each_call(self):
        """house_theme() hands back a fresh theme, so one scene's tweaks cannot leak into the next.

        Test scenario:
            Mutating the first theme's background leaves a second call's theme at the documented white.
        """
        first = house_theme()
        first.background = "red"
        second = house_theme()
        assert second.background.hex_rgb == "#ffffff", f"A later theme inherited a mutation: {second.background}"


class TestScene3DBaseInit:
    """Tests for Scene3DBase.__init__."""

    def test_defaults_to_the_house_theme_and_window_size(self):
        """A default scene renders at 1024x768 with the house theme applied.

        Test scenario:
            No `theme` and no `window_size` means `house_theme()` and the documented (1024, 768) default reach
            the wrapped plotter.
        """
        scene = Scene3DBase(off_screen=True)
        assert scene.plotter.theme.cmap == "viridis", f"Expected the house theme, got cmap {scene.plotter.theme.cmap!r}"
        assert list(scene.plotter.window_size) == [1024, 768], f"Unexpected window size: {scene.plotter.window_size}"
        assert scene.layers == [], f"A fresh scene must own no layers, got {scene.layers}"
        scene.close()

    def test_honours_an_explicit_theme(self):
        """A caller-supplied theme is used instead of the house theme.

        Test scenario:
            Passing a DarkTheme leaves the plotter on that theme's settings, not the house theme's — proving the
            `theme or house_theme()` fallback only fires when the argument is None. pyvista copies the theme it
            is handed, so the settings are compared rather than the object identity.
        """
        theme = pv.themes.DarkTheme()
        scene = Scene3DBase(off_screen=True, theme=theme)
        applied = scene.plotter.theme
        assert applied.background.hex_rgb == theme.background.hex_rgb, (
            f"Expected the DarkTheme background {theme.background}, got {applied.background}"
        )
        assert applied.background.hex_rgb != house_theme().background.hex_rgb, (
            "An explicit theme must not be overridden by the house theme"
        )
        scene.close()

    def test_honours_an_explicit_window_size(self):
        """A caller-supplied window_size reaches the plotter, so frames come back at that size.

        Test scenario:
            `window_size=(320, 240)` reaches the plotter and it still renders an RGB frame. The frame's own
            dimensions are not asserted: VTK does not always honour the requested size exactly (HiDPI).
        """
        scene = Scene3DBase(off_screen=True, window_size=(320, 240))
        assert list(scene.plotter.window_size) == [320, 240], f"Unexpected window size: {scene.plotter.window_size}"
        frame = scene.screenshot()
        assert frame.ndim == 3 and frame.shape[-1] == 3, f"Expected an RGB frame, got shape {frame.shape}"
        scene.close()

    def test_off_screen_none_follows_the_pyvista_global(self):
        """`off_screen=None` defers to `pyvista.OFF_SCREEN` rather than forcing a window open.

        Test scenario:
            The autouse fixture pins `pv.OFF_SCREEN` True, so a scene built with the default renders headless.
        """
        scene = Scene3DBase()
        assert scene.plotter.off_screen is True, "off_screen=None must follow pv.OFF_SCREEN, which is True here"
        scene.close()

    def test_forwards_extra_plotter_kwargs(self):
        """Unrecognised keywords pass straight through to `pyvista.Plotter`.

        Test scenario:
            `shape=(1, 2)` builds a two-renderer plotter, showing **plotter_kwargs is forwarded verbatim.
        """
        scene = Scene3DBase(off_screen=True, shape=(1, 2))
        count = len(scene.plotter.renderers)
        assert count == 2, f"Expected 2 renderers from shape=(1, 2), got {count}"
        scene.close()


class TestExportHtml:
    """Tests for Scene3DBase.export_html — in particular both sides of the trame-component capability switch."""

    def test_routes_to_the_trame_component_when_registered(self, tmp_path):
        """With a `trame` component present the export goes through it, not the plotter's own method.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            The component receives the path and the deprecated `Plotter.export_html` is never called — which is
            what keeps the PyVistaDeprecationWarning off pyvista >=0.49.
        """
        component = _RecordingComponent()
        scene = _stub_scene(component=component)
        out = str(tmp_path / "s.html")
        scene.export_html(out)
        assert component.calls == [out], f"The trame component should have received {out!r}, got {component.calls}"
        assert scene.plotter.calls == [], f"Plotter.export_html is deprecated, not to be called: {scene.plotter.calls}"

    def test_falls_back_to_the_plotter_without_a_component(self, tmp_path):
        """With no `trame` attribute (pyvista 0.48 without trame-pyvista) the export uses `Plotter.export_html`.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            The plotter's own method receives the path, so the pre-0.49 arrangement keeps working after the
            component branch was added.
        """
        scene = _stub_scene()
        out = str(tmp_path / "s.html")
        scene.export_html(out)
        assert scene.plotter.calls == [out], f"Expected {out!r} on the plotter, got {scene.plotter.calls}"

    @pytest.mark.parametrize("with_component", [True, False], ids=["trame-component", "plotter-fallback"])
    def test_returns_the_destination_it_wrote(self, tmp_path, with_component):
        """Both branches return the same path they handed to the exporter.

        Args:
            tmp_path: Supplies the destination path.
            with_component: Whether the stub plotter carries a `trame` component.

        Test scenario:
            Neither branch forwards the underlying return value — trame-pyvista's own export returns None, so a
            bare `return self.plotter...` would hand the caller None instead of the path.
        """
        component = _RecordingComponent() if with_component else None
        scene = _stub_scene(component=component)
        out = str(tmp_path / "s.html")
        recorded = component.calls if with_component else scene.plotter.calls
        assert scene.export_html(out) == out, "export_html must return the path it wrote"
        assert recorded == [out], f"The returned path must be the one exported, got {recorded}"

    @pytest.mark.parametrize("with_component", [True, False], ids=["trame-component", "plotter-fallback"])
    @pytest.mark.parametrize(
        "given, written", [("s.HTML", "s.html"), ("s.htm", "s.html"), ("s", "s.html")],
        ids=["uppercase", "htm", "suffixless"],
    )
    def test_normalises_the_suffix_on_both_branches(self, tmp_path, given, written, with_component):
        """A non-`.html` suffix is rewritten before export, identically on either branch.

        Args:
            tmp_path: Supplies the destination path.
            given: The suffix the caller asks for.
            written: The suffix that must actually be exported.
            with_component: Whether the stub plotter carries a `trame` component.

        Test scenario:
            trame-pyvista rewrites a non-`.html` suffix itself while pyvista 0.48's native export honours the
            name it is given, so the same call would write two different files depending on the installed
            pyvista. Normalising up front makes both agree, and the returned path names the file that exists.
        """
        component = _RecordingComponent() if with_component else None
        scene = _stub_scene(component=component)
        expected = str(tmp_path / written)
        returned = scene.export_html(tmp_path / given)
        recorded = component.calls if with_component else scene.plotter.calls
        assert returned == expected, f"Expected the normalised path {expected!r}, got {returned!r}"
        assert recorded == [expected], f"Expected {expected!r} to be exported, got {recorded}"

    def test_propagates_a_missing_stack_error(self, tmp_path):
        """A missing trame stack surfaces pyvista's own actionable ImportError rather than a swallowed failure.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            pyvista >=0.49 without trame-pyvista raises `The "trame" plotter component is not registered`. The
            fallback branch must let that reach the caller with its message intact — it names what to install.
        """
        message = 'The "trame" plotter component is not registered. Install trame-pyvista: pip install trame-pyvista'
        scene = _stub_scene(error=ImportError(message))
        with pytest.raises(ImportError, match="trame-pyvista") as exc_info:
            scene.export_html(str(tmp_path / "s.html"))
        assert "not registered" in str(exc_info.value), f"The actionable message was lost: {exc_info.value}"


class TestPyvistaVtkRoot:
    """Tests for the public derivation of the VTK package pyvista is bound to."""

    def test_reads_the_root_off_a_pyvista_type(self):
        """The root comes from the MRO of a pyvista type, giving `vtkmodules` for a stock build.

        Test scenario:
            Deriving it this way avoids `pyvista._vtk._VTK_ROOT`, which is private and moved between pyvista
            0.48 and 0.49 — the module path does not even exist on 0.48.
        """
        root = base._pyvista_vtk_root()
        ancestry = {klass.__module__.split(".")[0] for klass in pv.PolyData.__mro__}
        assert root in ancestry, f"{root!r} is not among pyvista's own ancestry {sorted(ancestry)}"
        assert root.startswith("vtk"), f"Expected a VTK package, got {root!r}"
        assert root in sys.modules, f"{root!r} names a package that was never imported"

    def test_falls_back_when_no_vtk_ancestor_is_found(self, monkeypatch):
        """A pyvista type with no VTK ancestry yields the stock default rather than raising.

        Args:
            monkeypatch: Replaces `pyvista.PolyData` with a plain class.

        Test scenario:
            The derivation walks an MRO it does not control. If a future pyvista stops exposing a VTK base
            there, the check must degrade to the stock package name instead of breaking every export.
        """
        monkeypatch.setattr(pv, "PolyData", type("NotVtkBacked", (), {}))
        assert base._pyvista_vtk_root() == "vtkmodules", "the fallback must be the stock VTK package"


class TestVtkBuildReconciliation:
    """Tests for the VTK-build check `export_html` runs before using the trame component."""

    def test_matching_builds_export_normally(self, monkeypatch, tmp_path):
        """When trame and pyvista resolve the same VTK build the export proceeds untouched.

        Args:
            monkeypatch: Points `VTK_MODULE_NAME` at pyvista's own build.
            tmp_path: Supplies the destination path.

        Test scenario:
            The stock arrangement — both on `vtkmodules` — must not trip the guard.
        """
        monkeypatch.delitem(sys.modules, "vtk_module", raising=False)
        monkeypatch.setenv("VTK_MODULE_NAME", base._pyvista_vtk_root())
        component = _RecordingComponent()
        scene = _stub_scene(component=component)
        out = str(tmp_path / "s.html")
        scene.export_html(out)
        assert component.calls == [out], f"A matching build must export normally, got {component.calls}"

    def test_mismatched_builds_raise_naming_the_variable(self, monkeypatch, tmp_path):
        """Two VTK builds in one process raise a RuntimeError naming `VTK_MODULE_NAME`, before exporting.

        Args:
            monkeypatch: Points `VTK_MODULE_NAME` at a different build than pyvista's.
            tmp_path: Supplies the destination path.

        Test scenario:
            Objects cannot be shared between VTK builds. pyvista makes this check inside its deprecated
            `Plotter.export_html`; the component branch skips that, so the check is reproduced here — as a
            `RuntimeError` rather than an `ImportError`, so it cannot be mistaken for a missing package. The
            component must not be reached.
        """
        monkeypatch.delitem(sys.modules, "vtk_module", raising=False)
        monkeypatch.setenv("VTK_MODULE_NAME", "vtk_a_different_build")
        component = _RecordingComponent()
        scene = _stub_scene(component=component)
        with pytest.raises(RuntimeError, match="VTK_MODULE_NAME") as exc_info:
            scene.export_html(str(tmp_path / "s.html"))
        assert "vtk_a_different_build" in str(exc_info.value), f"The resolved build was not named: {exc_info.value}"
        assert component.calls == [], f"The export must not run on a mismatched build, got {component.calls}"

    def test_the_fallback_branch_skips_the_check(self, monkeypatch, tmp_path):
        """Without a component the check is irrelevant — pyvista runs its own inside `Plotter.export_html`.

        Args:
            monkeypatch: Points `VTK_MODULE_NAME` at a different build than pyvista's.
            tmp_path: Supplies the destination path.

        Test scenario:
            Duplicating the guard on the fallback branch would raise before pyvista could produce its own,
            better-placed error, so a mismatch must still reach `Plotter.export_html`.
        """
        monkeypatch.delitem(sys.modules, "vtk_module", raising=False)
        monkeypatch.setenv("VTK_MODULE_NAME", "vtk_a_different_build")
        scene = _stub_scene()
        out = str(tmp_path / "s.html")
        scene.export_html(out)
        assert scene.plotter.calls == [out], f"The fallback must defer to pyvista's check, got {scene.plotter.calls}"

    def test_a_loaded_vtk_module_wins_over_the_environment(self, monkeypatch, tmp_path):
        """An already-imported `vtk_module` decides trame's build, whatever the environment says.

        Args:
            monkeypatch: Installs a fake resolved `vtk_module` that disagrees with `VTK_MODULE_NAME`.
            tmp_path: Supplies the destination path.

        Test scenario:
            trame caches its resolved binding as `sys.modules["vtk_module"]`; once that exists the environment
            variable no longer decides, so the check must read the loaded module rather than the variable.
        """
        monkeypatch.setitem(sys.modules, "vtk_module", types.ModuleType("vtk_some_other_build"))
        monkeypatch.setenv("VTK_MODULE_NAME", base._pyvista_vtk_root())
        scene = _stub_scene(component=_RecordingComponent())
        with pytest.raises(RuntimeError, match="vtk_some_other_build"):
            scene.export_html(str(tmp_path / "s.html"))


class TestSave:
    """Tests for Scene3DBase.save — the PNG/HTML dispatch."""

    @pytest.mark.parametrize("name", ["s.html", "s.HTML", "s.Html"], ids=["lower", "upper", "mixed"])
    def test_html_dispatch_is_case_insensitive(self, tmp_path, name):
        """Any casing of the `.html` suffix takes the HTML branch and lands on the normalised name.

        Args:
            tmp_path: Supplies the destination path.
            name: File name whose suffix casing is under test.

        Test scenario:
            `save()` lowercases before matching, so `s.HTML` exports a page instead of silently writing a PNG,
            and `export_html` normalises the suffix so every casing writes `s.html`.
        """
        scene = _stub_scene()
        assert scene.save(str(tmp_path / name)) is None, "The HTML branch returns None, not a frame"
        expected = str(tmp_path / "s.html")
        assert scene.plotter.calls == [expected], f"Expected {expected!r} for {name!r}, got {scene.plotter.calls}"

    def test_accepts_a_path_object(self, tmp_path):
        """A `pathlib.Path` destination dispatches on suffix just as a string does.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            `save()` calls `str(path)` before matching, so a Path ending in `.html` still takes the HTML branch,
            and the exporter is handed a plain string.
        """
        scene = _stub_scene()
        out = tmp_path / "s.html"
        assert scene.save(out) is None, "A Path ending in .html must take the HTML branch"
        assert scene.plotter.calls == [str(out)], f"The exporter should receive a str, got {scene.plotter.calls}"

    def test_png_branch_forwards_screenshot_kwargs(self, tmp_path):
        """Non-HTML paths screenshot, forwarding extra keywords to `Plotter.screenshot`.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            `transparent_background=True` reaches the screenshot call, yielding a 4-channel RGBA frame instead
            of the usual 3-channel RGB one.
        """
        scene = Scene3DBase(off_screen=True)
        scene.add_mesh(_dem_grid(), scalars="z")
        frame = scene.save(str(tmp_path / "s.png"), transparent_background=True)
        assert frame is not None, "The PNG branch must return the rendered frame"
        assert frame.shape[-1] == 4, f"Expected an RGBA frame from transparent_background=True, got {frame.shape}"
        scene.close()


class TestContextManager:
    """Tests for Scene3DBase.__enter__ / __exit__."""

    def test_enter_returns_the_scene(self):
        """`with Scene3DBase(...) as scene` binds the scene itself, not a wrapper.

        Test scenario:
            The context manager yields the same object it was called on, so layers added inside are visible
            outside the block.
        """
        scene = Scene3DBase(off_screen=True)
        with scene as entered:
            assert entered is scene, "__enter__ must return the scene itself"

    def test_exit_returns_false_so_exceptions_propagate(self):
        """`__exit__` reports False, which is what stops it swallowing an exception.

        Test scenario:
            Called directly with no exception it still returns False — the flag that
            `test_context_manager_closes_plotter` observes indirectly by letting a ValueError escape the block.
        """
        scene = Scene3DBase(off_screen=True)
        assert scene.__exit__(None, None, None) is False, "__exit__ must return False so exceptions propagate"
