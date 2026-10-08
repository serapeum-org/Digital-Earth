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


def test_the_ocean_fill_is_hole_aware():
    """The ocean fill triangulates its hole-aware rings into a real mesh, not an empty one.

    Test scenario:
        The 110m ocean is one exterior ring plus a continent-shaped hole per landmass. The drawer feeds
        those ``[exterior, *holes]`` rings to the hole-carving triangulator, so the drawn mesh has cells; a
        fill built from the exterior alone would paint over the continents (the cleopatra#384 bug).
    """
    scene = Scene3D(off_screen=True)
    scene.ocean()
    mesh = scene.layers[0][0]
    assert mesh.n_cells > 0, "the ocean fill triangulated to no cells"
    assert mesh.n_points > 0
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


def test_the_fill_is_banded_as_ground_cover_under_the_data():
    """A fill is ground cover: it bands ``underlay`` (below the data), not ``overlay`` like the lines.

    Test scenario:
        ``reference_fill`` is the opposite end of the stack from the reference *lines*: an opaque area fill
        belongs under the data, a thin line over it. Banding the fill ``overlay`` (as ``reference_lines`` is)
        would draw an opaque ocean/land over a flat data layer coincident at ``z=0``. This pins the band so it
        cannot silently regress to ``overlay``.
    """
    from digitalearth.base.spec.layer import _layer_band

    scene = Scene3D(off_screen=True)
    scene.ocean()
    assert _layer_band(scene.figure_spec.layers.get("ocean")) == "underlay"
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
