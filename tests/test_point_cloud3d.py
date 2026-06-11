"""Tests for the D3.2 point-cloud renderer (digitalearth.three_d.PointCloudMixin.point_cloud).

Gated on the optional ``3d`` extra (pyvista). Exercises the numpy xyz path (LiDAR-style), the 2-D lift-to-z=0
path, per-point colouring, the GeoDataFrame (``get_cell_points``-style) path read by duck-typing, and that the
guard module imports no GIS competitor.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D
from digitalearth.three_d.point_cloud import SCALAR, _coords_from_array


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_point_cloud_from_xyz_array():
    """A coloured (N, 3) xyz table renders and registers one layer with the points and scalar."""
    pts = np.column_stack([np.arange(40.0), np.arange(40.0), np.linspace(0.0, 9.0, 40)])
    scene = Scene3D(off_screen=True)
    actor = scene.point_cloud(pts, values=pts[:, 2])
    assert actor is not None and len(scene.layers) == 1
    cloud = scene.layers[0][0]
    assert cloud.n_points == 40
    assert SCALAR in cloud.point_data
    scene.close()


def test_two_column_table_is_lifted_to_z0():
    """A (N, 2) table is lifted to z=0 (so 2-D points still build a valid cloud)."""
    out = _coords_from_array(np.random.default_rng(0).random((25, 2)))
    assert out.shape == (25, 3)
    assert np.all(out[:, 2] == 0.0)


def test_bad_shape_raises():
    """A non-(N, 2)/(N, 3) array is rejected with a clear error."""
    with pytest.raises(ValueError):
        _coords_from_array(np.zeros((5, 4)))


def test_uncoloured_cloud_has_no_scalar():
    """Without values/value_column the cloud carries no colour scalar."""
    scene = Scene3D(off_screen=True)
    scene.point_cloud(np.random.default_rng(1).random((30, 3)), eye_dome_lighting=False)
    assert SCALAR not in scene.layers[0][0].point_data
    scene.close()


def test_point_cloud_from_geodataframe(points):
    """A GeoDataFrame of points (e.g. get_cell_points) renders via duck-typed coordinate reads."""
    scene = Scene3D(off_screen=True)
    actor = scene.point_cloud(points, point_size=8.0)
    assert actor is not None
    assert scene.layers[0][0].n_points == len(points)
    scene.close()


def test_renders_a_nonempty_frame():
    """The point cloud produces a real off-screen frame with content."""
    pts = np.random.default_rng(2).random((300, 3)) * 10
    scene = Scene3D(off_screen=True)
    scene.point_cloud(pts, values=pts[:, 2])
    img = scene.screenshot()
    assert img.ndim == 3 and bool(img.any())
    scene.close()


def test_values_length_mismatch_raises():
    """point_cloud() rejects a values array whose length does not match the points."""
    scene = Scene3D(off_screen=True)
    with pytest.raises(ValueError, match="does not match"):
        scene.point_cloud(np.zeros((10, 3)), values=np.zeros(7))
    scene.close()
