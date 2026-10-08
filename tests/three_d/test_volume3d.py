"""Tests for the D3.3 volumetric renderer (digitalearth.three_d.VolumeMixin).

Gated on the optional ``3d`` extra (pyvista). Covers volume ray-casting, isosurface extraction, the
``DatasetCollection``-style ``.values`` duck-typed input, and the 3-D shape guard.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D
from digitalearth.three_d.volume import FIELD, _cube, _grid_geo


class _GeoCube:
    """A minimal georeferenced cube stand-in: a `.values` stack plus a GDAL `.geotransform`.

    Duck-types the shape `volume`/`isosurface` read — enough to exercise the #198 geo-derivation without a
    pyramids DatasetCollection.
    """

    def __init__(self, values, geotransform):
        self.values = values
        self.geotransform = geotransform


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


def test_volume_defaults_to_the_identity_placement():
    """#198: a bare numpy cube keeps origin (0, 0, 0) and unit spacing — exactly as before."""
    scene = Scene3D(off_screen=True)
    scene.volume(_gaussian_cube(6))
    grid = scene.layers[0][0]
    assert tuple(grid.origin) == (0.0, 0.0, 0.0)
    assert tuple(grid.spacing) == (1.0, 1.0, 1.0)
    scene.close()


def test_volume_takes_explicit_origin_and_spacing():
    """#198: a volume can be placed in real-world coordinates so it shares a scene with terrain."""
    scene = Scene3D(off_screen=True)
    scene.volume(_gaussian_cube(6), origin=(100.0, 200.0, 0.0), spacing=(2.0, 3.0, 5.0))
    grid = scene.layers[0][0]
    assert tuple(grid.origin) == (100.0, 200.0, 0.0)
    assert tuple(grid.spacing) == (2.0, 3.0, 5.0)
    scene.close()


def test_volume_derives_placement_from_a_geotransform():
    """#198: a georeferenced cube's footprint is read from its pyramids geotransform."""
    cube = _gaussian_cube(6)  # (6, 6, 6) → rows = 6
    # GDAL geotransform: x0=10, dx=2, y0(top)=40, dy=-4 → lower-left y = 40 + (-4)*6 = 16.
    geo = _GeoCube(cube, (10.0, 2.0, 0.0, 40.0, 0.0, -4.0))
    scene = Scene3D(off_screen=True)
    scene.volume(geo)
    grid = scene.layers[0][0]
    assert tuple(grid.origin) == (10.0, 16.0, 0.0)
    assert tuple(grid.spacing) == (2.0, 4.0, 1.0)
    scene.close()


def test_grid_geo_explicit_overrides_the_geotransform():
    """An explicit origin/spacing wins over a geotransform, each side independently."""
    cube = _gaussian_cube(6)
    geo = _GeoCube(cube, (10.0, 2.0, 0.0, 40.0, 0.0, -4.0))
    origin, spacing = _grid_geo(geo, cube, (0.0, 0.0, 0.0), None)
    assert origin == (0.0, 0.0, 0.0)  # explicit origin wins
    assert spacing == (2.0, 4.0, 1.0)  # spacing still derived


def test_isosurface_takes_explicit_placement():
    """#198: isosurface() places its grid in real-world coordinates too."""
    scene = Scene3D(off_screen=True)
    scene.isosurface(_gaussian_cube(10), isosurfaces=[0.3], origin=(5.0, 5.0, 0.0))
    mesh = scene.layers[0][0]
    # The extracted shell sits inside the placed grid, so its x/y bounds start at or above the origin.
    assert mesh.bounds[0] >= 5.0
    assert mesh.bounds[2] >= 5.0
    scene.close()


def test_volume_rejects_an_unknown_blending_mode():
    """#TD-11: an unknown blending mode is refused by name, naming the choices."""
    scene = Scene3D(off_screen=True)
    cube = _gaussian_cube(6)
    with pytest.raises(ValueError, match="blending="):
        scene.volume(cube, blending="bogus")
    scene.close()


def test_volume_accepts_blending_clim_and_shade():
    """#TD-11: the transfer-function controls are exposed, recorded on the layer, and render."""
    scene = Scene3D(off_screen=True)
    actor = scene.volume(
        _gaussian_cube(8), blending="maximum", clim=(0.0, 1.0), shade=True
    )
    assert actor is not None
    props = scene.figure_spec.layers.get(scene.layer_ids[-1]).symbology.props
    assert props["blending"] == "maximum"
    assert tuple(props["clim"]) == (0.0, 1.0)
    assert props["shade"] is True
    assert bool(scene.screenshot().any())
    scene.close()


def test_volume_registers_and_renders():
    """volume() ray-casts a cube, registers one layer, and produces a non-empty frame."""
    scene = Scene3D(off_screen=True)
    actor = scene.volume(_gaussian_cube())
    assert actor is not None
    assert len(scene.layers) == 1
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
    zeros = np.zeros((4, 4))
    with pytest.raises(ValueError):
        _cube(zeros)


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
