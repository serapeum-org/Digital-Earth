"""Tests for digitalearth.three_d.Scene3D — the PyVista shared-plotter host.

Gated on the optional ``3d`` extra: when pyvista is not installed these are skipped, so the default suite stays
green; install ``digitalearth[3d]`` (or run the ``viz3d`` pixi env) to exercise them for real.
"""

# The package imports have to follow pytest.importorskip("pyvista") — importing digitalearth.three_d
# without pyvista is the very thing the skip exists to avoid — so E402 is expected throughout.
# ruff: noqa: E402
import json
import sys
import types

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D, base
from digitalearth.three_d.base import (
    IMAGE_SUFFIXES,
    SCENE_EXPORTERS,
    Scene3DBase,
    house_theme,
    supported_destinations,
)


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
    assert img.ndim == 3
    assert img.shape[-1] == 3
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
    # C1: save() returns the Path it wrote on every branch — the RGB frame moved to screenshot().
    assert png_result == png, f"save() must return the png path, got {png_result!r}"
    assert html_result == html, f"save() must return the html path, got {html_result!r}"
    scene.close()


def test_add_volume_registers_layer():
    """add_volume() ray-casts a scalar field and registers it as a layer."""
    scene = Scene3D(off_screen=True)
    grid = pv.ImageData(dimensions=(6, 6, 6))
    grid.cell_data["v"] = np.linspace(0.0, 1.0, 5 * 5 * 5)
    actor = scene.add_volume(grid, cmap="viridis")
    assert actor is not None
    assert len(scene.layers) == 1
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

    `export_html` and `save` only ever touch `self.plotter`. A scene builds its own plotter only when one is
    first asked for, so assigning the stub straight after construction means no VTK context is ever opened.
    """
    scene = Scene3DBase(off_screen=True)
    scene.plotter = _RecordingPlotter(component=component, error=error)
    return scene


def test_stub_scene_mirrors_the_real_attribute_set():
    """`_stub_scene` carries the same attributes `Scene3DBase.__init__` builds.

    Test scenario:
        The helper once bypassed `__init__` to avoid opening a render window, which was only safe while it
        copied every attribute by hand. It now goes through `__init__`, and this keeps it from drifting back.
    """
    real = Scene3DBase(off_screen=True)
    try:
        assert set(vars(_stub_scene())) == set(vars(real)), (
            "the stub scene must carry the real attribute set"
        )
    finally:
        real.close()


class TestHouseTheme:
    """Tests for house_theme."""

    def test_returns_the_documented_document_theme(self):
        """house_theme() returns a DocumentTheme carrying every setting its docstring promises.

        Test scenario:
            The returned theme is a `pyvista.themes.DocumentTheme` with a white background, the `viridis`
            colormap, SSAA anti-aliasing and a black font — the publication-grade defaults the tier renders with.
        """
        theme = house_theme()
        assert isinstance(theme, pv.themes.DocumentTheme), (
            f"Expected a DocumentTheme, got {type(theme).__name__}"
        )
        assert theme.background.hex_rgb == "#ffffff", (
            f"Expected a white background, got {theme.background}"
        )
        assert theme.cmap == "viridis", f"Expected the viridis cmap, got {theme.cmap!r}"
        assert theme.anti_aliasing == "ssaa", (
            f"Expected ssaa anti-aliasing, got {theme.anti_aliasing!r}"
        )
        assert theme.font.color.hex_rgb == "#000000", (
            f"Expected a black font, got {theme.font.color}"
        )

    def test_returns_an_independent_theme_each_call(self):
        """house_theme() hands back a fresh theme, so one scene's tweaks cannot leak into the next.

        Test scenario:
            Mutating the first theme's background leaves a second call's theme at the documented white.
        """
        first = house_theme()
        first.background = "red"
        second = house_theme()
        assert second.background.hex_rgb == "#ffffff", (
            f"A later theme inherited a mutation: {second.background}"
        )


class TestScene3DBaseInit:
    """Tests for Scene3DBase.__init__."""

    def test_defaults_to_the_house_theme_and_window_size(self):
        """A default scene renders at 1024x768 with the house theme applied.

        Test scenario:
            No `theme` and no `window_size` means `house_theme()` and the documented (1024, 768) default reach
            the wrapped plotter.
        """
        scene = Scene3DBase(off_screen=True)
        assert scene.plotter.theme.cmap == "viridis", (
            f"Expected the house theme, got cmap {scene.plotter.theme.cmap!r}"
        )
        assert list(scene.plotter.window_size) == [1024, 768], (
            f"Unexpected window size: {scene.plotter.window_size}"
        )
        assert scene.layers == [], (
            f"A fresh scene must own no layers, got {scene.layers}"
        )
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
        assert list(scene.plotter.window_size) == [320, 240], (
            f"Unexpected window size: {scene.plotter.window_size}"
        )
        frame = scene.screenshot()
        assert frame.ndim == 3, f"Expected a 3-D frame, got shape {frame.shape}"
        assert frame.shape[-1] == 3, (
            f"Expected three colour channels, got shape {frame.shape}"
        )
        scene.close()

    def test_off_screen_none_follows_the_pyvista_global(self):
        """`off_screen=None` defers to `pyvista.OFF_SCREEN` rather than forcing a window open.

        Test scenario:
            The autouse fixture pins `pv.OFF_SCREEN` True, so a scene built with the default renders headless.
        """
        scene = Scene3DBase()
        assert scene.plotter.off_screen is True, (
            "off_screen=None must follow pv.OFF_SCREEN, which is True here"
        )
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
        assert component.calls == [out], (
            f"The trame component should have received {out!r}, got {component.calls}"
        )
        assert scene.plotter.calls == [], (
            f"Plotter.export_html is deprecated, not to be called: {scene.plotter.calls}"
        )

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
        assert scene.plotter.calls == [out], (
            f"Expected {out!r} on the plotter, got {scene.plotter.calls}"
        )

    @pytest.mark.parametrize(
        "with_component", [True, False], ids=["trame-component", "plotter-fallback"]
    )
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
        assert scene.export_html(out) == out, (
            "export_html must return the path it wrote"
        )
        assert recorded == [out], (
            f"The returned path must be the one exported, got {recorded}"
        )

    @pytest.mark.parametrize(
        "with_component", [True, False], ids=["trame-component", "plotter-fallback"]
    )
    @pytest.mark.parametrize(
        "given, written",
        [("s.HTML", "s.html"), ("s.htm", "s.html"), ("s", "s.html")],
        ids=["uppercase", "htm", "suffixless"],
    )
    def test_normalises_the_suffix_on_both_branches(
        self, tmp_path, given, written, with_component
    ):
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
        assert returned == expected, (
            f"Expected the normalised path {expected!r}, got {returned!r}"
        )
        assert recorded == [expected], (
            f"Expected {expected!r} to be exported, got {recorded}"
        )

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
        destination = str(tmp_path / "s.html")
        with pytest.raises(ImportError, match="trame-pyvista") as exc_info:
            scene.export_html(destination)
        assert "not registered" in str(exc_info.value), (
            f"The actionable message was lost: {exc_info.value}"
        )


class TestPyvistaVtkRoot:
    """Tests for how the VTK package pyvista is bound to gets resolved."""

    def test_prefers_the_value_pyvista_resolved(self, monkeypatch):
        """`pyvista._vtk._VTK_ROOT` wins when present — it is the value pyvista itself uses.

        Args:
            monkeypatch: Pins the resolved root to a name the MRO walk could never produce.

        Test scenario:
            pyvista 0.49 caches its resolved backend there, and it can be any distribution name. Reading it
            first is what makes the comparison exact on the versions where the component branch is reachable.
        """
        monkeypatch.setattr(pv._vtk, "_VTK_ROOT", "cvista", raising=False)
        assert base._pyvista_vtk_root() == "cvista", (
            "the resolved root must win over the MRO walk"
        )

    def test_reads_the_root_off_a_pyvista_type(self):
        """Without a resolved root the MRO of a pyvista type gives the same answer for a stock build.

        Test scenario:
            pyvista 0.48 ships `pyvista._vtk` but not `_VTK_ROOT`, so the walk is the fallback there.
        """
        root = base._pyvista_vtk_root()
        ancestry = {klass.__module__.split(".")[0] for klass in pv.PolyData.__mro__}
        assert root in ancestry, (
            f"{root!r} is not among pyvista's own ancestry {sorted(ancestry)}"
        )
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
        assert base._pyvista_vtk_root() == "vtkmodules", (
            "the fallback must be the stock VTK package"
        )


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
        assert component.calls == [out], (
            f"A matching build must export normally, got {component.calls}"
        )

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
        destination = str(tmp_path / "s.html")
        with pytest.raises(RuntimeError, match="VTK_MODULE_NAME") as exc_info:
            scene.export_html(destination)
        assert "vtk_a_different_build" in str(exc_info.value), (
            f"The resolved build was not named: {exc_info.value}"
        )
        assert component.calls == [], (
            f"The export must not run on a mismatched build, got {component.calls}"
        )

    def test_the_fallback_branch_is_guarded_too(self, monkeypatch, tmp_path):
        """The check runs before either branch, so the no-component path is guarded as well.

        Args:
            monkeypatch: Points `VTK_MODULE_NAME` at a different build than pyvista's.
            tmp_path: Supplies the destination path.

        Test scenario:
            pyvista 0.49 makes this check inside the deprecated `Plotter.export_html`, but 0.48 makes none
            anywhere — so leaving the fallback to pyvista would leave it unguarded on exactly the version the
            fallback exists for.
        """
        monkeypatch.delitem(sys.modules, "vtk_module", raising=False)
        monkeypatch.setenv("VTK_MODULE_NAME", "vtk_a_different_build")
        scene = _stub_scene()
        destination = str(tmp_path / "s.html")
        with pytest.raises(RuntimeError, match="VTK_MODULE_NAME"):
            scene.export_html(destination)
        assert scene.plotter.calls == [], (
            f"A mismatched build must not export, got {scene.plotter.calls}"
        )

    def test_a_non_vtk_prefixed_backend_is_compared_correctly(
        self, monkeypatch, tmp_path
    ):
        """A backend whose name is not `vtk*` — pyvista's `cvista`, or any PYVISTA_VTK_BACKEND — still compares.

        Args:
            monkeypatch: Pins pyvista's resolved root to a non-`vtk` name and points trame at the stock one.
            tmp_path: Supplies the destination path.

        Test scenario:
            pyvista resolves its VTK root to `cvista` merely because that package is importable, and
            `PYVISTA_VTK_BACKEND` can name any fork. A check that only recognises `vtk`-prefixed names would
            report both sides as `vtkmodules` and pass a genuinely mismatched pair.
        """
        monkeypatch.setattr(pv._vtk, "_VTK_ROOT", "cvista", raising=False)
        monkeypatch.delitem(sys.modules, "vtk_module", raising=False)
        monkeypatch.delenv("VTK_MODULE_NAME", raising=False)
        scene = _stub_scene(component=_RecordingComponent())
        destination = str(tmp_path / "s.html")
        with pytest.raises(RuntimeError, match="cvista"):
            scene.export_html(destination)

    def test_a_matched_non_vtk_backend_is_not_blocked(self, monkeypatch, tmp_path):
        """A correctly matched non-`vtk*` backend must export, not trip the guard.

        Args:
            monkeypatch: Pins both sides to the same non-`vtk` name.
            tmp_path: Supplies the destination path.

        Test scenario:
            This is the working single-build setup. Blocking it would be worse than not checking at all — the
            error would tell the user to switch to the build they deliberately are not using.
        """
        monkeypatch.setattr(pv._vtk, "_VTK_ROOT", "cvista", raising=False)
        monkeypatch.delitem(sys.modules, "vtk_module", raising=False)
        monkeypatch.setenv("VTK_MODULE_NAME", "cvista")
        component = _RecordingComponent()
        scene = _stub_scene(component=component)
        out = str(tmp_path / "s.html")
        scene.export_html(out)
        assert component.calls == [out], (
            f"A matched cvista build must export, got {component.calls}"
        )

    def test_a_loaded_vtk_module_wins_over_the_environment(self, monkeypatch, tmp_path):
        """An already-imported `vtk_module` decides trame's build, whatever the environment says.

        Args:
            monkeypatch: Installs a fake resolved `vtk_module` that disagrees with `VTK_MODULE_NAME`.
            tmp_path: Supplies the destination path.

        Test scenario:
            trame caches its resolved binding as `sys.modules["vtk_module"]`; once that exists the environment
            variable no longer decides, so the check must read the loaded module rather than the variable.
        """
        monkeypatch.setitem(
            sys.modules, "vtk_module", types.ModuleType("vtk_some_other_build")
        )
        monkeypatch.setenv("VTK_MODULE_NAME", base._pyvista_vtk_root())
        scene = _stub_scene(component=_RecordingComponent())
        destination = str(tmp_path / "s.html")
        with pytest.raises(RuntimeError, match="vtk_some_other_build"):
            scene.export_html(destination)


class TestSave:
    """Tests for Scene3DBase.save — the PNG/HTML dispatch."""

    @pytest.mark.parametrize(
        "name", ["s.html", "s.HTML", "s.Html"], ids=["lower", "upper", "mixed"]
    )
    def test_html_dispatch_is_case_insensitive(self, tmp_path, name):
        """Any casing of the `.html` suffix takes the HTML branch and lands on the normalised name.

        Args:
            tmp_path: Supplies the destination path.
            name: File name whose suffix casing is under test.

        Test scenario:
            `save()` lowercases before matching, so `s.HTML` exports a page instead of silently writing a PNG,
            and `export_html` normalises the suffix so every casing writes `s.html`. C1: the returned Path is
            the normalised name that was actually written, not the casing that was asked for.
        """
        scene = _stub_scene()
        written = scene.save(str(tmp_path / name))
        assert written == tmp_path / "s.html", (
            f"The HTML branch must return the normalised path it wrote, got {written!r}"
        )
        expected = str(tmp_path / "s.html")
        assert scene.plotter.calls == [expected], (
            f"Expected {expected!r} for {name!r}, got {scene.plotter.calls}"
        )

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
        assert scene.save(out) == out, (
            "A Path ending in .html must take the HTML branch and return the path written"
        )
        assert scene.plotter.calls == [str(out)], (
            f"The exporter should receive a str, got {scene.plotter.calls}"
        )

    def test_png_branch_forwards_screenshot_kwargs(self, tmp_path):
        """Non-HTML paths screenshot, forwarding extra keywords to `Plotter.screenshot`.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            `transparent_background=True` reaches the screenshot call, so the file written carries an alpha
            channel. C1 moved the frame off `save()`, so the proof is the PNG on disk rather than a returned
            array: a 4-channel PNG declares colour type 6 in its IHDR chunk (byte 25), a 3-channel one type 2.
        """
        scene = Scene3DBase(off_screen=True)
        scene.add_mesh(_dem_grid(), scalars="z")
        out = tmp_path / "s.png"
        written = scene.save(str(out), transparent_background=True)
        scene.close()
        assert written == out, (
            f"The PNG branch must return the path written, got {written!r}"
        )
        assert out.read_bytes()[25] == 6, (
            "transparent_background=True must reach Plotter.screenshot and write an RGBA PNG"
        )

    def test_a_scene_export_suffix_writes_that_format(self, tmp_path):
        """A `.gltf` destination exports glTF through PyVista, instead of dying in the screenshot branch.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            Regression guard for the "everything that is not .html is a screenshot" dispatch: `.gltf` used to
            reach `Plotter.screenshot`, which rejected it as a non-image extension. It must now reach
            `Plotter.export_gltf` and leave a real glTF document (JSON carrying an `asset` block) behind, with
            `None` returned because no frame was rendered.
        """
        scene = Scene3DBase(off_screen=True, window_size=(120, 90))
        scene.add_mesh(pv.Sphere())
        out = tmp_path / "scene.gltf"
        try:
            assert scene.save(str(out)) == out, (
                "An export branch returns the path it wrote (C1)"
            )
        finally:
            scene.close()
        assert out.exists(), f"export_gltf must write the file it was given, {out!r}"
        assert out.stat().st_size > 0, f"export_gltf must not leave {out!r} empty"
        assert "asset" in json.loads(out.read_text(encoding="utf-8")), (
            "The file must be a glTF document, not a renamed screenshot"
        )

    def test_an_obj_destination_reaches_the_obj_exporter(self, tmp_path):
        """A `.obj` destination exports OBJ geometry (plus the `.mtl` VTK writes beside it).

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            The second scene-export family member, proving the dispatch table is consulted rather than one
            suffix being special-cased.
        """
        scene = Scene3DBase(off_screen=True, window_size=(120, 90))
        scene.add_mesh(pv.Sphere())
        out = tmp_path / "scene.obj"
        try:
            assert scene.save(str(out)) == out, (
                "An export branch returns the path it wrote (C1)"
            )
        finally:
            scene.close()
        assert out.exists(), f"export_obj must write the .obj it was given, {out!r}"
        assert out.stat().st_size > 0, f"export_obj must not leave {out!r} empty"

    def test_a_suffix_less_path_raises_instead_of_inventing_a_png(self, tmp_path):
        """`save("scene")` raises and writes nothing, rather than quietly producing `scene.png`.

        Args:
            tmp_path: Supplies the destination directory.

        Test scenario:
            Regression guard for the silent-format case: pyvista appends `.png` to a suffix-less screenshot
            path, so the caller got a file they never named. The error must say the suffix is missing, and the
            directory must be left empty.
        """
        scene = _stub_scene()
        destination_value = str(tmp_path / "scene")
        with pytest.raises(ValueError, match="has no suffix"):
            scene.save(destination_value)
        assert list(tmp_path.iterdir()) == [], (
            "Nothing may be written for a destination save() cannot honour"
        )

    def test_an_unknown_suffix_names_the_supported_ones(self, tmp_path):
        """An unsupported suffix raises a Digital-Earth error listing every suffix `save` does dispatch.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            `.xyz` is neither a raster frame nor a scene export. The message must name both families so the
            caller can fix the call without reading the source.
        """
        scene = _stub_scene()
        destination = str(tmp_path / "scene.xyz")
        with pytest.raises(ValueError) as raised:
            scene.save(destination)
        message = str(raised.value)
        assert supported_destinations() in message, (
            "The error must name the supported suffixes"
        )
        for suffix in (*IMAGE_SUFFIXES, *SCENE_EXPORTERS):
            assert suffix in message, (
                f"{suffix} is dispatched but missing from the error message"
            )

    def test_a_pyvista_without_the_exporter_is_reported_clearly(self, tmp_path):
        """A supported scene suffix whose pyvista exporter is absent raises, naming the missing method.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            The exporters are a pyvista capability, so the dispatcher looks each one up rather than assuming
            it. The stub plotter has no `export_vtksz`, which must surface as a clear ValueError instead of an
            AttributeError from deep inside the call.
        """
        scene = _stub_scene()
        destination = str(tmp_path / "scene.vtksz")
        with pytest.raises(ValueError, match="export_vtksz"):
            scene.save(destination)

    def test_a_tif_destination_really_writes_a_tiff(self, tmp_path):
        """`.tif` screenshots to a TIFF — the format the suffix names, not a PNG under another name.

        Args:
            tmp_path: Supplies the destination path.

        Test scenario:
            The old docstring promised "anything else saves a PNG screenshot"; it never did for `.tif`. The
            written file must start with a TIFF magic number, not the PNG one.
        """
        scene = Scene3DBase(off_screen=True, window_size=(120, 90))
        scene.add_mesh(_dem_grid(), scalars="z")
        out = tmp_path / "scene.tif"
        try:
            written = scene.save(str(out))
        finally:
            scene.close()
        assert written == out, (
            f"The raster branch returns the path it wrote (C1), got {written!r}"
        )
        assert out.read_bytes()[:2] in (b"II", b"MM"), (
            "A .tif destination must hold a TIFF, not a PNG"
        )


class TestVerticalExaggeration:
    """Tests for Scene3DBase.vertical_exaggeration — exaggeration as a view scale, not baked geometry."""

    def test_a_fresh_scene_is_at_true_scale(self):
        """A scene with no exaggeration set renders at 1.0, and says so.

        Test scenario:
            The property reads the renderer's z scale, which starts at true scale.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            assert scene.vertical_exaggeration == 1.0, (
                "A fresh scene must be at true scale"
            )
        finally:
            scene.close()

    def test_the_value_set_reads_back_off_the_scene(self):
        """The exaggeration is scene state a caller can read, not a number lost inside a mesh.

        Test scenario:
            Regression guard: with the factor multiplied into the mesh points there was nothing to read back.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            scene.vertical_exaggeration = 2.5
            assert scene.vertical_exaggeration == 2.5, (
                "The scene must report the exaggeration it was set to"
            )
        finally:
            scene.close()

    def test_a_scale_set_before_the_window_exists_reaches_it(self):
        """The exaggeration belongs to the scene, so the window built later is born carrying it.

        Test scenario:
            The setter has nothing to set when `_plotter` is None — it stores the value — so the *only*
            place that value reaches a render window is the builder in the `plotter` property. Nothing
            covered that path before, which is how a guard on it could be changed unnoticed.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            scene.vertical_exaggeration = 2.0
            assert list(scene.plotter.scale) == [1.0, 1.0, 2.0], scene.plotter.scale
        finally:
            scene.close()

    def test_a_scene_nobody_exaggerated_builds_an_unscaled_window(self):
        """The other arm of the same builder: true scale is what an untouched scene renders at.

        Test scenario:
            The builder applies the scene's scale unconditionally, so this is what says the identity case
            is left exactly as a fresh plotter is.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            assert list(scene.plotter.scale) == [1.0, 1.0, 1.0], scene.plotter.scale
        finally:
            scene.close()

    def test_it_scales_actors_added_afterwards(self):
        """The view scale propagates to every actor, including ones added after it was set.

        Test scenario:
            That propagation is what lets one value govern a whole scene: the mesh keeps its own geometry
            while the actor rendering it carries the z scale.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            scene.vertical_exaggeration = 3.0
            sphere = pv.Sphere()
            actor = scene.add_mesh(sphere)
            assert actor.scale[2] == pytest.approx(3.0), (
                "A later actor must pick up the scene's z scale"
            )
            assert (sphere.bounds[5] - sphere.bounds[4]) == pytest.approx(1.0), (
                "The mesh itself must be untouched - only the view is scaled"
            )
        finally:
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
        assert scene.__exit__(None, None, None) is False, (
            "__exit__ must return False so exceptions propagate"
        )
