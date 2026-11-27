"""digitalearth owns the 3-D off-screen decision; the static-image path is self-sufficient (#573).

Before #573, inline 3-D output silently required the user to set ``pyvista.OFF_SCREEN`` (or its env var):
without it, ``ds.plot(var, backend="3d")`` rendered a blank cell. digitalearth now decides off-screen through
its own ``three_d.off_screen`` option, and the screenshot/``_repr_png_`` path forces off-screen on its own.
Gated on the ``3d`` extra.
"""

# ruff: noqa: E402
import os
import sys

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

import digitalearth
from digitalearth.three_d import Scene3D
from digitalearth.three_d.core.base import (
    _auto_off_screen,
    _headless,
    _resolve_off_screen,
)


@pytest.fixture(autouse=True)
def _restore_option():
    """Reset the option store after each test so one test's writes do not leak into another."""
    from digitalearth.base.options import _reset_options

    yield
    _reset_options()


@pytest.fixture
def mesh():
    """A tiny two-face UGRID mesh to draw (quad + triangle)."""
    from pyramids.netcdf.ugrid.connectivity import Connectivity
    from pyramids.netcdf.ugrid.mesh import Mesh2d

    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 2.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2, 3], [1, 4, 2, -1]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


@pytest.mark.parametrize(
    "env, platform, pv_off, expected",
    [
        ({}, "linux", False, True),  # Linux, no display server -> headless
        ({"DISPLAY": ":0"}, "linux", False, False),  # X display present -> on-screen
        (
            {"WAYLAND_DISPLAY": "wayland-0"},
            "linux",
            False,
            False,
        ),  # Wayland -> on-screen
        (
            {},
            "darwin",
            False,
            False,
        ),  # macOS desktop (Aqua, no DISPLAY) -> on-screen (H2)
        ({}, "win32", False, False),  # desktop Windows -> on-screen
        (
            {"CI": "true"},
            "win32",
            False,
            True,
        ),  # CI runner (even on Windows) -> headless
        (
            {"CI": "false", "DISPLAY": ":0"},
            "linux",
            False,
            False,
        ),  # CI=false not forced (M2)
        (
            {"PYVISTA_OFF_SCREEN": "0"},
            "win32",
            False,
            False,
        ),  # explicit opt-out honoured (M2)
        ({}, "win32", True, True),  # power user set pyvista.OFF_SCREEN -> honoured
        ({"PYVISTA_OFF_SCREEN": "1"}, "win32", False, True),  # env opt-in honoured
    ],
)
def test_headless_decides_from_environment(env, platform, pv_off, expected):
    """``_headless`` decides off-screen purely from the environment and platform it is handed (#573)."""
    assert _headless(env, platform, pv_off) is expected


def test_auto_off_screen_keeps_a_macos_desktop_on_screen(monkeypatch):
    """``"auto"`` must not disable interactive windows on a Mac desktop, which has no ``DISPLAY`` (#573, H2).

    Test scenario:
        A macOS desktop (``os.name == "posix"``, ``sys.platform == "darwin"``, no display vars, not CI, PyVista
        global off) resolves to on-screen — restoring the pre-PR behaviour the display-only heuristic removed.
    """
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(pv, "OFF_SCREEN", False)
    assert _auto_off_screen() is False


@pytest.mark.parametrize("explicit", [True, False])
def test_resolve_off_screen_prefers_an_explicit_argument(explicit):
    """An explicit ``off_screen=`` wins over the option (#573)."""
    digitalearth.set_option("three_d.off_screen", not explicit)
    assert _resolve_off_screen(explicit) is explicit


@pytest.mark.parametrize("setting", [True, False])
def test_resolve_off_screen_reads_the_option_when_not_explicit(setting):
    """With no explicit argument, a boolean ``three_d.off_screen`` option decides (#573)."""
    digitalearth.set_option("three_d.off_screen", setting)
    assert _resolve_off_screen(None) is setting


def test_repr_png_returns_bytes_for_a_drawn_scene(mesh, monkeypatch):
    """The headline fix: ``_repr_png_`` yields a PNG, not the blank ``None`` the shipped path returned (#573).

    Test scenario:
        A drawn off-screen scene (the mode a notebook resolves to) embeds real PNG bytes with ``pyvista``'s own
        global left off — the inline image no longer depends on ``pyvista.OFF_SCREEN``.
    """
    monkeypatch.setattr(pv, "OFF_SCREEN", False)
    scene = Scene3D(off_screen=True)
    try:
        scene.trimesh(mesh)
        png = scene._repr_png_()
        assert png is not None, "expected PNG bytes, got None (blank cell)"
        assert png[:4] == b"\x89PNG", png[:8]
    finally:
        scene.close()


def test_off_screen_is_resolved_lazily_at_plotter_build(monkeypatch):
    """The off-screen decision is read when the plotter is built, so a late option change is honoured (#573, M1).

    Test scenario:
        A scene constructed while the option is ``"auto"`` has resolved nothing yet; setting
        ``three_d.off_screen`` to ``True`` *after* construction but before the first render still makes the
        built plotter off-screen — the decision is not frozen at ``__init__`` (which also keeps construction
        renderer-free).
    """
    monkeypatch.setattr(pv, "OFF_SCREEN", False)
    scene = Scene3D()
    try:
        digitalearth.set_option("three_d.off_screen", True)
        assert scene.plotter.off_screen is True, (
            "late option change was not honoured — decision frozen early"
        )
    finally:
        scene.close()


def test_offscreen_snapshot_carries_a_custom_add_mesh_layer():
    """The off-screen snapshot redraws held ``add_mesh`` objects rather than silently dropping them (#573, H1).

    Test scenario:
        A scene holding a raw ``add_mesh(pv.Sphere())`` layer — whose object lives only in ``held_objects``,
        not in ``figure_spec`` — snapshots to an image that differs from the same scene with no custom layer.
        The pre-fix rebuild via ``from_figure`` lost the held object, so the sphere and empty frames matched.
    """
    empty = Scene3D(off_screen=True)
    with_sphere = Scene3D(off_screen=True)
    try:
        with_sphere.add_mesh(pv.Sphere())
        baseline = empty._offscreen_snapshot(None)
        sphere_frame = with_sphere._offscreen_snapshot(None)
        assert sphere_frame.ndim == 3, sphere_frame.shape
        assert sphere_frame.shape[-1] == 3, sphere_frame.shape
        assert float(sphere_frame.std()) > 0.0, (
            "snapshot is uniform — the custom sphere layer was dropped"
        )
        assert not np.array_equal(baseline, sphere_frame), (
            "the custom sphere layer did not change the image"
        )
    finally:
        empty.close()
        with_sphere.close()


def test_live_scene_survives_an_offscreen_snapshot():
    """Taking an off-screen snapshot must not free or corrupt the live scene's shared objects (#573, M2).

    Test scenario:
        A scene holding a custom ``add_mesh`` layer still renders a non-uniform frame *after* an
        ``_offscreen_snapshot`` has built and closed a copy that shared that same pyvista object — proving the
        snapshot's ``close()`` left the live scene untouched.
    """
    scene = Scene3D(off_screen=True)
    try:
        scene.add_mesh(pv.Sphere())
        scene._offscreen_snapshot(None)
        frame = scene.screenshot()
        assert float(frame.std()) > 0.0, (
            "live scene lost its sphere after the snapshot closed a shared copy"
        )
    finally:
        scene.close()


def test_screenshot_routes_an_on_screen_scene_through_the_snapshot(monkeypatch):
    """An on-screen scene takes the off-screen-snapshot path, without opening or needing a window (#573).

    Test scenario:
        ``screenshot()`` on an on-screen scene (no plotter built yet) calls ``_offscreen_snapshot`` instead of
        screenshotting a windowless live plotter — verified by spying on the method, so the test needs no
        display and runs headless.
    """
    scene = Scene3D(off_screen=False)
    calls = {}
    sentinel = np.zeros((2, 2, 3), dtype=np.uint8)

    def _spy(path, **kwargs):
        calls["hit"] = True
        return sentinel

    monkeypatch.setattr(scene, "_offscreen_snapshot", _spy)
    try:
        result = scene.screenshot()
        assert calls.get("hit"), (
            "on-screen scene did not route through _offscreen_snapshot"
        )
        assert result is sentinel, "screenshot did not return the snapshot's frame"
    finally:
        scene.close()


def test_the_option_drives_a_scene_built_after_it_is_set(mesh, monkeypatch):
    """Setting ``three_d.off_screen`` before building a scene resolves its plotter off-screen (#573).

    Test scenario:
        With ``pyvista.OFF_SCREEN`` False, ``set_option("three_d.off_screen", True)`` makes a later
        ``Scene3D()`` build an off-screen plotter — digitalearth's own setting, not the PyVista global.
    """
    monkeypatch.setattr(pv, "OFF_SCREEN", False)
    digitalearth.set_option("three_d.off_screen", True)
    scene = Scene3D()
    try:
        assert scene.plotter.off_screen is True, (
            "scene did not honour the three_d.off_screen option"
        )
    finally:
        scene.close()
