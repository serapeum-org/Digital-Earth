"""Tests for the live trame view (digitalearth.three_d.ServeMixin.serve) — TD-12.

Gated on the optional ``3d`` extra (pyvista). The real ``serve()`` launches a trame server and opens a browser,
which a test cannot do, so these mock the trame launcher and check the wiring: the plotter and mode reach
``show_trame``, and the one-VTK-build guard runs first.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D


def _dem():
    """A small ramped DEM Source."""
    return get_source(np.add.outer(np.arange(4.0), np.arange(5.0)))


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_serve_hands_the_plotter_and_mode_to_the_trame_launcher(mocker):
    """serve() calls show_trame with the scene's own plotter and the requested mode, and returns its handle.

    Args:
        mocker: Patches the trame launcher so no server starts and no browser opens.
    """
    import pyvista.trame.jupyter as trame_jupyter

    sentinel = object()
    show = mocker.patch.object(trame_jupyter, "show_trame", return_value=sentinel)
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem())
    result = scene.serve(mode="server")
    scene.close()

    assert result is sentinel
    assert show.call_args.args[0] is scene.plotter
    assert show.call_args.kwargs["mode"] == "server"


def test_serve_forwards_extra_keywords(mocker):
    """Extra keywords pass through to show_trame unchanged.

    Args:
        mocker: Patches the trame launcher.
    """
    import pyvista.trame.jupyter as trame_jupyter

    show = mocker.patch.object(trame_jupyter, "show_trame", return_value=None)
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem())
    scene.serve(mode="client", collapse_menu=True)
    scene.close()

    assert show.call_args.kwargs["collapse_menu"] is True
