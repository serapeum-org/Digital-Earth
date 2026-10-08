"""Tests for 3-D reference-geography fills (digitalearth.three_d.DecorationMixin.land/ocean/lakes) — TD-5b.

Gated on the optional ``3d`` extra (pyvista). The fill half of #205 was blocked while cleopatra's
``natural_earth`` dropped polygon holes; cleopatra 0.42.0's ``natural_earth_polygons`` is hole-aware, so
``land()``/``ocean()``/``lakes()`` now draw filled Natural-Earth polygons on the ground plane, the continents
carved out of the ocean rather than painted over (cleopatra#384). The hole-carving itself is
``vector._cap_with_holes`` (covered by the extrusion/#199 tests); these tests cover the builders and the drawer.
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


def test_land_draws_a_fill_in_a_flat_scene():
    """TD-5b: land() draws a Natural-Earth polygon fill in a non-globe scene, as a named layer."""
    scene = Scene3D(off_screen=True)
    actor = scene.land()
    assert actor is not None
    assert "land" in scene.layer_ids
    assert scene.layers[0][0].n_points > 0
    assert bool(scene.screenshot().any())
    scene.close()


def test_ocean_and_lakes_draw_their_own_layers():
    """ocean() and lakes() are the same fill path over their own datasets, each its own id."""
    scene = Scene3D(off_screen=True)
    scene.ocean()
    scene.lakes()
    assert "ocean" in scene.layer_ids
    assert "lakes" in scene.layer_ids
    scene.close()


def test_the_ocean_fill_carves_its_holes_through_the_drawer():
    """The drawer (``scene.ocean()``) produces a mesh with less area than an exterior-only fill.

    Test scenario:
        This goes through ``draw_reference_fill`` end-to-end — ``scene.ocean()`` reads the hole-aware rings,
        extracts ``rings[1:]`` as holes and triangulates them — then compares the DRAWN mesh's area against
        an exterior-only triangulation of the same parts. The drawn (holed) area is strictly smaller, which
        only holds when the drawer wires the continents through as holes; the cleopatra#384 bug (a solid
        exterior fill) would make the two equal. Exercising the drawer, not ``_cap_with_holes`` directly,
        guards the drawer's own ``rings[1:]`` handling.
    """
    from cleopatra.basemap.reference import natural_earth_polygons

    from digitalearth.three_d.vector import _cap_with_holes

    scene = Scene3D(off_screen=True)
    scene.ocean()
    drawn = scene.layers[0][0]
    solid_area = sum(
        _cap_with_holes(np.asarray(part[0], dtype="float64"), []).area
        for part in natural_earth_polygons("ocean", "110m")
        if len(part[0]) >= 3
    )
    assert drawn.area < solid_area, (
        f"the drawer must carve the ocean's holes: drawn {drawn.area} vs exterior-only {solid_area}"
    )
    scene.close()


def test_a_non_finite_opacity_is_refused():
    """A NaN opacity is refused at the call, as the web tier does — a figure cannot hold a non-finite value."""
    scene = Scene3D(off_screen=True)
    nan = float("nan")
    with pytest.raises(ValueError, match="opacity"):
        scene.ocean(opacity=nan)
    scene.close()


def test_a_degenerate_polygon_skips_the_layer(monkeypatch):
    """A part with too few vertices is dropped; with nothing left to fill, the layer is skipped.

    Test scenario:
        Natural Earth always returns real polygons, so the degenerate/empty path is exercised with a stub:
        a single two-vertex part is not an area, so it is dropped and the non-strict scene skips the layer
        (returns ``None``, nothing added) rather than drawing an empty mesh.

    Args:
        monkeypatch: Replaces ``natural_earth_polygons`` with the degenerate stub.
    """
    import cleopatra.basemap.reference as ref

    monkeypatch.setattr(
        ref, "natural_earth_polygons", lambda dataset, resolution: [[np.zeros((2, 2))]]
    )
    scene = Scene3D(off_screen=True)
    assert scene.ocean() is None
    assert "ocean" not in scene.layer_ids
    scene.close()


def test_the_fill_bands_underlay_below_the_lines_in_the_layer_tree():
    """The fill bands ``underlay`` — below the reference lines in the LAYER TREE (not 3-D occlusion).

    Test scenario:
        ``reference_fill`` is ground cover, so it bands ``underlay`` like the static/interactive
        ``land``/``ocean``/``lakes`` kinds, whereas the reference *lines* band ``overlay``. This governs the
        scene's **layer-tree order** (the layer list and switcher): the fill is ordered before an overlay
        line. It does **not** govern 3-D occlusion — the renderer composites actors by depth, so what is in
        front is decided by ``z``/opacity, not by band (coplanar ``z=0`` layers are separated by opacity).
        This pins the band value and its tree order so it cannot regress to ``overlay``.
    """
    from digitalearth.base.spec.layer import _layer_band

    scene = Scene3D(off_screen=True)
    scene.ocean()
    scene.coastlines()
    figure = scene.figure_spec
    assert _layer_band(figure.layers.get("ocean")) == "underlay"
    assert _layer_band(figure.layers.get("coastlines")) == "overlay"
    ids = scene.layer_ids
    assert ids.index("ocean") < ids.index("coastlines"), (
        f"the underlay fill must be ordered before the overlay line in the tree: {ids}"
    )
    scene.close()


def test_an_unknown_resolution_is_refused():
    """An unknown Natural-Earth resolution is refused by name, before any download."""
    scene = Scene3D(off_screen=True)
    with pytest.raises(ValueError, match="resolution="):
        scene.ocean(resolution="1m")
    scene.close()


def test_a_fill_over_a_projected_scene_warns_about_the_crs(caplog):
    """Drawn in EPSG:4326 over a projected display CRS, the fill warns rather than silently misplace.

    Args:
        caplog: Captures the warning the drawer logs.
    """
    scene = Scene3D(off_screen=True)
    scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))  # no CRS
    scene.display_crs = 32618  # a projected CRS, as a UTM terrain would set
    with caplog.at_level(logging.WARNING):
        scene.ocean()
    scene.close()
    assert "EPSG:4326" in caplog.text


def test_a_custom_name_is_used():
    """A caller-supplied name is used as the layer id."""
    scene = Scene3D(off_screen=True)
    scene.land(name="landmass")
    assert "landmass" in scene.layer_ids
    scene.close()


def test_the_dataset_and_resolution_are_recorded():
    """The dataset, resolution and colour travel on the layer description so the figure replays."""
    scene = Scene3D(off_screen=True)
    scene.ocean(resolution="110m", color="#123456")
    props = scene.figure_spec.layers.get("ocean").symbology.props
    assert props["dataset"] == "ocean"
    assert props["resolution"] == "110m"
    assert props["color"] == "#123456"
    scene.close()
