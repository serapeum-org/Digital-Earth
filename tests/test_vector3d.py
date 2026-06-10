"""Tests for the D3.4 vector renderers (digitalearth.three_d.VectorMixin).

Gated on the optional ``3d`` extra (pyvista). Covers arrow glyphs from a (u, v, w) field, polygon extrusion
into 3-D prisms (uniform + per-feature height + colour-by-attribute), and the MultiPolygon ring reader.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import MultiPolygon, Polygon

from digitalearth.three_d import Scene3D
from digitalearth.three_d.vector import MAGNITUDE, VALUE, _exterior_rings, _extrude_ring


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _squares():
    """Two unit-square polygons with population values (a tiny extrusion fixture)."""
    return gpd.GeoDataFrame(
        {"pop": [10.0, 20.0]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
    )


def test_vectors_glyphs_render():
    """vectors() builds magnitude-coloured arrow glyphs and registers one layer."""
    ax = np.linspace(0, 1, 6)
    xx, yy = np.meshgrid(ax, ax)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    vec = np.column_stack([np.ones(pts.shape[0]), np.zeros(pts.shape[0]), np.zeros(pts.shape[0])])
    scene = Scene3D(off_screen=True)
    scene.vectors(pts, vec, factor=0.1)
    assert len(scene.layers) == 1
    assert MAGNITUDE in scene.layers[0][0].point_data
    assert bool(scene.screenshot().any())
    scene.close()


def test_extruded_polygons_uniform_height():
    """A uniform height extrudes every footprint to the same z; the merged mesh has cells."""
    scene = Scene3D(off_screen=True)
    scene.extruded_polygons(_squares(), height=2.0)
    mesh = scene.layers[0][0]
    assert mesh.n_cells > 0
    assert mesh.bounds[5] == pytest.approx(2.0)
    scene.close()


def test_extruded_polygons_height_from_column_and_colour():
    """height='pop' extrudes per-feature; column='pop' attaches the colour scalar."""
    scene = Scene3D(off_screen=True)
    scene.extruded_polygons(_squares(), height="pop", column="pop")
    mesh = scene.layers[0][0]
    assert mesh.bounds[5] == pytest.approx(20.0)  # tallest = max pop
    assert VALUE in mesh.cell_data
    scene.close()


def test_extrude_ring_height():
    """_extrude_ring lifts a flat ring into a prism of the requested height."""
    ring = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    prism = _extrude_ring(ring, 0.5)
    assert prism.n_cells > 0
    assert prism.bounds[5] == pytest.approx(0.5)


def test_multipolygon_yields_one_ring_per_part():
    """_exterior_rings walks every part of a MultiPolygon."""
    mp = MultiPolygon(
        [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(5, 5), (6, 5), (6, 6)])]
    )
    rings = list(_exterior_rings(mp))
    assert len(rings) == 2
    assert all(r.shape[1] == 2 for r in rings)


def test_vectors_length_mismatch_raises():
    """vectors() rejects mismatched points/vectors shapes."""
    scene = Scene3D(off_screen=True)
    with pytest.raises(ValueError, match="same shape"):
        scene.vectors(np.zeros((5, 3)), np.zeros((4, 3)))
    scene.close()


def test_extruded_polygons_empty_raises():
    """extruded_polygons() rejects an empty geometry set."""
    empty = gpd.GeoDataFrame({"pop": []}, geometry=[])
    scene = Scene3D(off_screen=True)
    with pytest.raises(ValueError, match="no polygon"):
        scene.extruded_polygons(empty)
    scene.close()
