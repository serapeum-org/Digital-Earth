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
    from digitalearth.base.options import reset_options

    yield
    reset_options()


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


def test_offscreen_snapshot_carries_a_custom_add_mesh_layer():
    """The off-screen snapshot redraws held ``add_mesh`` objects rather than silently dropping them (#573, H1).

    Test scenario:
        A scene holding a raw ``add_mesh(pv.Sphere())`` layer — whose object lives only in ``held_objects``,
        not in ``figure_spec`` — snapshots to a non-uniform frame (the sphere is present). The pre-fix rebuild
        via ``from_figure`` lost the held object and produced a blank, uniform frame.
    """
    scene = Scene3D(off_screen=True)
    try:
        scene.add_mesh(pv.Sphere())
        frame = scene._offscreen_snapshot(None)
        assert frame.ndim == 3, frame.shape
        assert frame.shape[-1] == 3, frame.shape
        assert float(frame.std()) > 0.0, (
            "snapshot is uniform — the custom sphere layer was dropped"
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
    monkeypatch.setattr(
        scene,
        "_offscreen_snapshot",
        lambda path, **k: calls.setdefault("hit", True) or sentinel,
    )
    try:
        scene.screenshot()
        assert calls.get("hit"), (
            "on-screen scene did not route through _offscreen_snapshot"
        )
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
