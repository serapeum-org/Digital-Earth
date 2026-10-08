"""Tests for 3-D reference geography (digitalearth.three_d.DecorationMixin.coastlines/borders) — TD-5b, #205.

Gated on the optional ``3d`` extra (pyvista). Coastlines/borders are Natural-Earth line overlays available in
any scene, not only on the globe; the land/ocean polygon *fills* half of #205 stays blocked upstream
(cleopatra#384) and is not tested here.
"""

import logging

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_coastlines_draw_reference_lines_in_a_flat_scene():
    """#205: coastlines() draws Natural-Earth lines in a non-globe scene, as a named layer."""
    scene = Scene3D(off_screen=True)
    actor = scene.coastlines()
    assert actor is not None
    assert "coastlines" in scene.layer_ids
    assert scene.layers[0][0].n_points > 0
    assert bool(scene.screenshot().any())
    scene.close()


def test_borders_draw_their_own_layer():
    """borders() is the same path over the borders dataset, numbered under its own id."""
    scene = Scene3D(off_screen=True)
    scene.coastlines()
    scene.borders()
    assert "coastlines" in scene.layer_ids
    assert "borders" in scene.layer_ids
    scene.close()


def test_coastlines_reject_an_unknown_resolution():
    """An unknown Natural-Earth resolution is refused by name."""
    scene = Scene3D(off_screen=True)
    with pytest.raises(ValueError, match="resolution="):
        scene.coastlines(resolution="1m")
    scene.close()


def test_coastlines_over_a_projected_scene_warn_about_the_crs(caplog):
    """Drawn in EPSG:4326 over a projected display CRS, the lines warn rather than silently misplace.

    Args:
        caplog: Captures the warning the drawer logs.
    """
    scene = Scene3D(off_screen=True)
    scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))  # no CRS
    scene.display_crs = 32618  # a projected CRS, as a UTM terrain would set
    with caplog.at_level(logging.WARNING):
        scene.coastlines()
    scene.close()
    assert "EPSG:4326" in caplog.text


def test_coastlines_and_borders_take_a_custom_name():
    """A caller-supplied name is used as the layer id, for both coastlines and borders."""
    scene = Scene3D(off_screen=True)
    scene.coastlines(name="shore")
    scene.borders(name="frontier")
    assert "shore" in scene.layer_ids
    assert "frontier" in scene.layer_ids
    scene.close()


def test_coastlines_resolution_is_recorded_on_the_layer():
    """The dataset and resolution travel on the layer description so the figure replays."""
    scene = Scene3D(off_screen=True)
    scene.coastlines(resolution="110m", color="#112233", width=2.0)
    props = scene.figure_spec.layers.get("coastlines").symbology.props
    assert props["dataset"] == "coastline"
    assert props["resolution"] == "110m"
    assert props["color"] == "#112233"
    scene.close()
