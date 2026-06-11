"""Tests for the D3.1 terrain renderer (digitalearth.three_d.TerrainMixin.terrain).

Gated on the optional ``3d`` extra (pyvista). Includes the orientation guard the plan calls for: a DEM with a
known high corner must render with its peak at the geographically-correct location (not mirrored/upside-down).
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.terrain import (
    ELEVATION,
    _METRES_PER_DEGREE,
    _terrain_mesh,
    _vertical_unit_scale,
)


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


def test_vertical_unit_scale_geographic_vs_projected():
    """A geographic CRS rescales metre elevation into degrees; a projected CRS leaves it alone."""
    assert _vertical_unit_scale(4326) == pytest.approx(1.0 / _METRES_PER_DEGREE)  # WGS84 lon/lat
    assert _vertical_unit_scale(3857) == 1.0  # Web Mercator (metres)
    assert _vertical_unit_scale(None) == 1.0  # unknown CRS → no rescaling


def test_geographic_dem_is_not_an_invisible_needle():
    """A geographic DEM (degrees x/y, metre z) builds relief comparable to its footprint, not a spike.

    Regression guard via the exact math terrain() composes (``_vertical_unit_scale`` x ``_terrain_mesh``): left
    unscaled the metre elevation dwarfs the ~degree-wide footprint ~100 000:1, so the surface is an invisible
    vertical needle. The geographic rescale must bring the relief into the footprint's order of magnitude.
    """
    # ~0.1 deg footprint, ~1000 m relief — the realistic geographic-DEM mismatch.
    x = np.linspace(-9.2, -9.1, 16)
    y = np.linspace(38.8, 38.7, 16)
    dem = np.add.outer(np.linspace(0.0, 1000.0, 16), np.zeros(16))

    scaled = _vertical_unit_scale(4326) * 1.0  # z_exaggeration=1.0, geographic CRS
    mesh = _terrain_mesh(dem, x, y, z_exaggeration=scaled)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    horizontal = max(xmax - xmin, ymax - ymin)
    vertical = zmax - zmin
    # Relief is brought into the same order of magnitude as the footprint (not ~10000x taller).
    assert vertical < horizontal * 5.0

    # And without the rescale it really is a needle — proving the guard is meaningful.
    needle = _terrain_mesh(dem, x, y, z_exaggeration=1.0)
    assert (needle.bounds[5] - needle.bounds[4]) > horizontal * 1000.0
