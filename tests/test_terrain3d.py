"""Tests for the D3.1 terrain renderer (digitalearth.three_d.TerrainMixin.terrain).

Gated on the optional ``3d`` extra (pyvista). Includes the orientation guard the plan calls for: a DEM with a
known high corner must render with its peak at the geographically-correct location (not mirrored/upside-down).
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.terrain import ELEVATION, _terrain_mesh


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_terrain_registers_a_layer_and_renders():
    """terrain() adds one layer and produces a non-empty off-screen frame."""
    dem = np.add.outer(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 12))
    scene = Scene3D(off_screen=True)
    actor = scene.terrain(get_source(dem), z_exaggeration=3.0)
    assert actor is not None and len(scene.layers) == 1
    img = scene.screenshot()
    assert img.ndim == 3 and bool(img.any())
    scene.close()


def test_terrain_mesh_is_not_upside_down():
    """A DEM peak at the north-east corner stays at the north-east point of the mesh.

    Guards the VTK Fortran-order gotcha: C-order ravel silently mirrors the surface. x ascending, y descending
    (north→south, as pyramids hands rasters); z high at row 0 (north) / last col (east).
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([30.0, 20.0, 10.0, 0.0])  # descending: first row = north
    z = np.zeros((4, 4))
    z[0, 3] = 100.0  # north-east peak → expect the elevation max at point (x=3, y=30)

    mesh = _terrain_mesh(z, x, y, z_exaggeration=1.0)
    peak = int(np.argmax(mesh.point_data[ELEVATION]))
    px, py, _ = mesh.points[peak]
    assert (px, py) == (3.0, 30.0)


def test_z_exaggeration_scales_relief():
    """A larger z_exaggeration produces a taller surface (bigger z extent)."""
    dem = np.add.outer(np.linspace(0.0, 10.0, 6), np.zeros(6))
    flat = _terrain_mesh(dem, np.arange(6.0), np.arange(6.0), z_exaggeration=1.0)
    tall = _terrain_mesh(dem, np.arange(6.0), np.arange(6.0), z_exaggeration=5.0)
    flat_h = flat.bounds[5] - flat.bounds[4]
    tall_h = tall.bounds[5] - tall.bounds[4]
    assert tall_h == pytest.approx(5.0 * flat_h)


def test_terrain_handles_nan_nodata():
    """NaN (masked nodata) cells do not crash the mesh build and are filled to the surface floor."""
    dem = np.add.outer(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    dem[0, 0] = np.nan
    mesh = _terrain_mesh(dem, np.arange(5.0), np.arange(5.0), z_exaggeration=1.0)
    assert np.isfinite(mesh.points).all()  # geometry has no NaN coordinates
