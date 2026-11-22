"""Tests for the 3-D interaction surface (digitalearth.three_d.InteractionMixin) — TD-9 widgets, TD-10 picking.

Gated on the optional ``3d`` extra (pyvista). The widgets and pickers are live callbacks below the figure seam,
so these check that each method instantiates its PyVista widget/picker headless, returns a handle, hides the
plain layer it acts on, and — for picking — maps a picked actor back to its layer id.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D


def _dem():
    """A small ramped DEM Source with real relief."""
    return get_source(
        np.add.outer(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    )


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_clip_plane_adds_a_widget_and_hides_the_plain_layer():
    """clip_plane() returns an actor and hides the layer's plain actor so nothing overdraws."""
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem(), name="dem")
    actor = scene.clip_plane("dem")
    assert actor is not None
    assert scene.renderer.is_visible("dem") is False  # plain layer hidden
    scene.close()


# The mesh widgets whose VTK filter cuts/extracts a *volume* into a normal-less surface that a strict GL draws
# blank (#538), so they default to unlit over a volume. `isovalue` (contour) is excluded: it raises on a
# cell-data volume ("Contour filter only works on Point data") before any draw, a pre-existing limitation.
_VOLUME_CUT_WIDGETS = ["clip_plane", "slice_planes", "clip_box", "threshold"]


@pytest.mark.parametrize("method", _VOLUME_CUT_WIDGETS)
def test_a_mesh_widget_over_a_volume_defaults_to_unlit(method):
    """A widget over a volume layer defaults to ``lighting=False`` so its cut surface shows on a strict GL (#538).

    Args:
        method: The widget method under test.

    Test scenario:
        A volume's cut/extracted surface carries no normals, which Mesa 26's llvmpipe drew blank; each widget
        defaults a volume layer to unlit so the mapped colour shows flat.
    """
    scene = Scene3D(off_screen=True)
    scene.volume(np.random.default_rng(0).random((6, 10, 10)), name="vol")
    actor = getattr(scene, method)("vol")
    assert actor.prop.lighting is False, f"{method} over a volume must default to unlit"
    scene.close()


@pytest.mark.parametrize("method", _VOLUME_CUT_WIDGETS)
def test_a_mesh_widget_over_a_surface_stays_lit(method):
    """A widget over a surface (terrain) layer keeps its lighting — the volume carve-out does not touch it (#538).

    Args:
        method: The widget method under test.

    Test scenario:
        A terrain surface carries normals, so its cut surface still lights correctly; the unlit default is
        narrowed to volume layers only.
    """
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem(), name="dem")
    actor = getattr(scene, method)("dem")
    assert actor.prop.lighting is True, f"{method} over a surface must stay lit"
    scene.close()


def test_a_mesh_widget_over_a_volume_lets_a_caller_force_lighting():
    """A caller's own ``lighting`` wins over the volume unlit default (#538)."""
    scene = Scene3D(off_screen=True)
    scene.volume(np.random.default_rng(0).random((6, 10, 10)), name="vol")
    actor = scene.clip_plane("vol", normal=(0, 0, 1), lighting=True)
    assert actor.prop.lighting is True, (
        "an explicit lighting= must override the volume default"
    )
    scene.close()


@pytest.mark.parametrize(
    "method", ["slice_planes", "threshold", "isovalue", "clip_box"]
)
def test_each_mesh_widget_returns_a_handle(method):
    """Every mesh widget instantiates headless over a terrain layer and returns a handle."""
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem(), name="dem")
    handle = getattr(scene, method)("dem")
    assert handle is not None
    scene.close()


def test_a_widget_on_a_missing_layer_is_refused():
    """A widget asked for a layer the scene has no mesh for is refused, naming the id."""
    scene = Scene3D(off_screen=True)
    with pytest.raises(KeyError, match="nope"):
        scene.clip_plane("nope")
    scene.close()


def test_slider_drives_its_callback_range():
    """slider() builds a vtkSliderWidget over the given range (headless, no interaction needed)."""
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem())
    seen = []
    widget = scene.slider(
        lambda value: seen.append(value), (0.0, 10.0), value=5.0, title="h"
    )
    assert widget is not None
    scene.close()


def test_enable_picking_rejects_an_unknown_mode():
    """enable_picking() names the valid gestures when given a bad mode."""
    scene = Scene3D(off_screen=True)
    with pytest.raises(ValueError, match="mode="):
        scene.enable_picking(mode="wiggle")
    scene.close()


def test_picked_layer_maps_an_actor_back_to_its_id(monkeypatch):
    """picked_layer() resolves the plotter's picked actor to the layer id it belongs to.

    Args:
        monkeypatch: Patches the plotter's read-only ``picked_actor`` to stand in for a real click.
    """
    scene = Scene3D(off_screen=True)
    actor = scene.terrain(_dem(), name="dem")
    # picked_actor is a read-only property; patch it on the class to return this layer's actor, as a pick would.
    monkeypatch.setattr(
        type(scene.plotter), "picked_actor", property(lambda _self: actor)
    )
    assert scene.picked_layer() == "dem"
    scene.close()


def test_picked_layer_is_none_without_a_pick():
    """picked_layer() is None when nothing has been picked."""
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem())
    assert scene.picked_layer() is None
    scene.close()


def test_enable_then_disable_picking():
    """enable_picking() turns a point pick on through PyVista, and disable_picking() turns it off."""
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem())
    scene.enable_picking(mode="point")
    scene.disable_picking()  # no error: the mode can be switched after disabling
    scene.enable_picking(mode="cell")
    scene.close()


def test_picked_layer_none_for_a_foreign_actor(monkeypatch):
    """picked_layer() is None when the picked actor is not one of the scene's layers.

    Args:
        monkeypatch: Patches picked_actor to a stray actor that belongs to no layer.
    """
    stray_plotter = pv.Plotter(off_screen=True)
    stray = stray_plotter.add_mesh(pv.Sphere())
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem(), name="dem")
    monkeypatch.setattr(
        type(scene.plotter), "picked_actor", property(lambda _self: stray)
    )
    assert scene.picked_layer() is None  # stray is not in the scene's drawn layers
    scene.close()
    stray_plotter.close()


def test_a_widget_on_a_described_but_undrawn_layer_is_refused():
    """A widget on a layer whose mesh the renderer no longer holds is refused by name."""
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem(), name="dem")
    scene.renderer._drawn.clear()  # the layer stays described, but nothing is drawn for it
    with pytest.raises(ValueError, match="nothing drawn"):
        scene.clip_plane("dem")
    scene.close()
