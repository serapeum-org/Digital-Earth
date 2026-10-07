"""Tests for the 3-D multi-panel compositor (digitalearth.three_d.grid) — TD-16.

Gated on the optional ``3d`` extra (pyvista). Checks that several scenes compose into one subplot window, that
each cell is drawn, and that the layout is validated.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D, grid


def _scene(flip: bool = False) -> Scene3D:
    """A headless scene with one DEM terrain (optionally mirrored, so two cells differ)."""
    ramp = np.add.outer(np.arange(6.0), np.arange(6.0))
    scene = Scene3D(off_screen=True)
    scene.terrain(get_source(ramp[::-1] if flip else ramp))
    return scene


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_grid_composes_scenes_into_subplots():
    """#TD-16: grid() places each scene in its own cell and renders the whole window."""
    plotter = grid([_scene(), _scene(flip=True)], shape=(1, 2), off_screen=True)
    assert len(plotter.renderers) == 2
    image = plotter.screenshot()
    assert image.ndim == 3 and bool(image.any())
    plotter.close()


def test_grid_defaults_to_a_single_row():
    """With no shape the scenes line up in one row."""
    plotter = grid([_scene(), _scene(), _scene()], off_screen=True)
    assert len(plotter.renderers) == 3
    plotter.close()


def test_grid_links_cameras_by_default():
    """link=True ties the cells' cameras together so the comparison moves as one."""
    plotter = grid([_scene(), _scene(flip=True)], shape=(1, 2), off_screen=True)
    # Linked renderers share one camera object across the cells.
    cameras = {id(renderer.camera) for renderer in plotter.renderers}
    assert len(cameras) == 1, "linked views must share one camera"
    plotter.close()


def test_grid_refuses_an_empty_sequence():
    """grid() needs at least one scene."""
    with pytest.raises(ValueError, match="at least one scene"):
        grid([], off_screen=True)


def test_grid_refuses_a_shape_too_small():
    """A shape with fewer cells than scenes is refused, naming the shape."""
    scenes = [_scene(), _scene(), _scene()]
    with pytest.raises(ValueError, match="cells for"):
        grid(scenes, shape=(1, 2), off_screen=True)
