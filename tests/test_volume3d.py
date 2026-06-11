"""Tests for the D3.3 volumetric renderer (digitalearth.three_d.VolumeMixin).

Gated on the optional ``3d`` extra (pyvista). Covers volume ray-casting, isosurface extraction, the
``DatasetCollection``-style ``.values`` duck-typed input, and the 3-D shape guard.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D
from digitalearth.three_d.volume import FIELD, _cube


def _gaussian_cube(n: int = 14) -> np.ndarray:
    """A synthetic 3-D Gaussian blob — a smooth, bounded scalar field to render."""
    ax = np.linspace(-2.0, 2.0, n)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.exp(-(xx**2 + yy**2 + zz**2))


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_volume_registers_and_renders():
    """volume() ray-casts a cube, registers one layer, and produces a non-empty frame."""
    scene = Scene3D(off_screen=True)
    actor = scene.volume(_gaussian_cube())
    assert actor is not None and len(scene.layers) == 1
    img = scene.screenshot()
    assert img.ndim == 3 and bool(img.any())
    scene.close()


def test_isosurface_extracts_shells():
    """isosurface() builds a non-empty contour mesh carrying the field scalar."""
    scene = Scene3D(off_screen=True)
    scene.isosurface(_gaussian_cube(16), isosurfaces=[0.3, 0.6])
    mesh = scene.layers[0][0]
    assert mesh.n_points > 0
    assert FIELD in mesh.point_data
    scene.close()


def test_cube_reads_datasetcollection_values():
    """A DatasetCollection-style object (exposing a 3-D ``.values``) is duck-typed without importing it."""

    class _FakeCollection:
        values = _gaussian_cube(8)

    cube = _cube(_FakeCollection())
    assert cube.shape == (8, 8, 8)


def test_non_3d_input_raises():
    """A 2-D array is rejected (volume needs a cube)."""
    with pytest.raises(ValueError):
        _cube(np.zeros((4, 4)))


def test_isosurface_auto_levels():
    """With no explicit iso-values, PyVista picks levels and still yields a surface."""
    scene = Scene3D(off_screen=True)
    scene.isosurface(_gaussian_cube(12))
    assert scene.layers[0][0].n_points > 0
    scene.close()


def test_cube_axes_map_lon_to_world_x():
    """A `[level, lat, lon]` cube renders with lon→world-X, lat→world-Y (not axis-transposed).

    Guards L1: a feature placed only at the max-lon / max-lat cell must land at the high-x / high-y corner of
    the point grid, not on the vertical (z) axis.
    """
    from digitalearth.three_d.volume import _point_grid

    cube = np.zeros((2, 3, 5))  # (nz=level, ny=lat, nx=lon)
    cube[:, 2, 4] = 1.0  # max lat (idx 2) and max lon (idx 4)
    grid = _point_grid(cube)
    feature = grid.points[grid.point_data["field"] > 0.5]
    assert set(feature[:, 0]) == {4.0}  # lon → world X
    assert set(feature[:, 1]) == {2.0}  # lat → world Y
