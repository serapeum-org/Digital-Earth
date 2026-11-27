"""digitalearth owns the 3-D off-screen decision; the static-image path is self-sufficient (#573).

Before #573, inline 3-D output silently required the user to set ``pyvista.OFF_SCREEN`` (or its env var):
without it, ``ds.plot(var, backend="3d")`` rendered a blank cell. digitalearth now decides off-screen through
its own ``three_d.off_screen`` option, and the screenshot/``_repr_png_`` path forces off-screen on its own.
Gated on the ``3d`` extra.
"""

# ruff: noqa: E402
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

import digitalearth
from digitalearth.three_d import Scene3D
from digitalearth.three_d.core.base import _headless, _resolve_off_screen


@pytest.fixture(autouse=True)
def _restore_option():
    """Save and restore the option store so one test's writes do not leak into another."""
    from digitalearth.base import options

    saved = dict(options._VALUES)
    yield
    options._VALUES.clear()
    options._VALUES.update(saved)


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
    "env, os_name, pv_off, expected",
    [
        ({}, "posix", False, True),  # POSIX, no display -> headless
        ({"DISPLAY": ":0"}, "posix", False, False),  # X display present -> on-screen
        (
            {"WAYLAND_DISPLAY": "wayland-0"},
            "posix",
            False,
            False,
        ),  # Wayland -> on-screen
        ({"CI": "true"}, "nt", False, True),  # CI runner (even on Windows) -> headless
        ({}, "nt", False, False),  # desktop Windows -> on-screen
        ({}, "nt", True, True),  # power user set pyvista.OFF_SCREEN -> honoured
        (
            {"PYVISTA_OFF_SCREEN": "1"},
            "nt",
            False,
            True,
        ),  # power user set the env var -> honoured
    ],
)
def test_headless_decides_from_environment(env, os_name, pv_off, expected):
    """``_headless`` decides off-screen purely from the environment it is handed (#573)."""
    assert _headless(env, os_name, pv_off) is expected


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


def test_repr_png_renders_without_pyvista_off_screen(mesh, monkeypatch):
    """The headline fix: inline PNG renders with ``pyvista.OFF_SCREEN`` unset (#573).

    Test scenario:
        An interactive-style kernel (``pyvista.OFF_SCREEN`` False) that never configured PyVista still gets a
        PNG from ``_repr_png_`` instead of the blank ``None`` the shipped behaviour returned.
    """
    monkeypatch.setattr(pv, "OFF_SCREEN", False)
    scene = Scene3D()
    try:
        scene.trimesh(mesh)
        png = scene._repr_png_()
        assert png is not None, "expected PNG bytes, got None (blank cell)"
        assert png[:4] == b"\x89PNG", png[:8]
    finally:
        scene.close()


def test_static_path_forces_off_screen_even_for_an_on_screen_scene(mesh, monkeypatch):
    """``screenshot`` forces off-screen on its lazy build regardless of the scene's mode (#573).

    Test scenario:
        A scene explicitly built ``off_screen=False`` with ``pyvista.OFF_SCREEN`` also False still screenshots
        — the static-image path builds its plotter off-screen rather than failing with "nothing to screenshot".
    """
    monkeypatch.setattr(pv, "OFF_SCREEN", False)
    scene = Scene3D(off_screen=False)
    try:
        scene.trimesh(mesh)
        frame = scene.screenshot()
        assert frame.ndim == 3, frame.shape
        assert frame.shape[-1] == 3, frame.shape
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
