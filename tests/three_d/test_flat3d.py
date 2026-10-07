"""Tests for the flat Core builders (digitalearth.three_d.FlatMixin) — TD-24/25/26.

Gated on the optional ``3d`` extra (pyvista). Covers flat points, flat polygon fills, and the choropleth fill,
each drawn on the ground plane (z=0) and distinct from the positioned builders (point_cloud / extruded_polygons).
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, Polygon

from digitalearth.three_d import Scene3D
from digitalearth.three_d.vector import VALUE


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _squares():
    """Two unit-square polygons with a population value."""
    return gpd.GeoDataFrame(
        {"pop": [10.0, 20.0]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
    )


def test_points_are_drawn_flat_on_the_ground():
    """#TD-24: points() lays markers at z=0 whatever z the input carried."""
    table = np.column_stack([np.arange(10.0), np.arange(10.0), np.full(10, 7.0)])
    scene = Scene3D(off_screen=True)
    scene.points(table)
    cloud = scene.layers[0][0]
    assert cloud.n_points == 10
    assert cloud.points[:, 2].tolist() == pytest.approx([0.0] * 10)
    scene.close()


def test_points_colour_by_column_classifies():
    """points() over a GeoDataFrame colours by a classified column."""
    gdf = gpd.GeoDataFrame(
        {"v": [1.0, 2.0, 30.0, 40.0]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
    )
    scene = Scene3D(off_screen=True)
    scene.points(gdf, column="v", scheme="quantiles", k=2)
    assert sorted({int(v) for v in scene.layers[0][0]["scalar"]}) == [0, 1]
    scene.close()


def test_polygons_fill_flat_not_extruded():
    """#TD-25: polygons() fills footprints flat at z=0 (no extrusion)."""
    scene = Scene3D(off_screen=True)
    scene.polygons(_squares())
    mesh = scene.layers[0][0]
    assert mesh.n_cells > 0
    assert mesh.bounds[4] == pytest.approx(0.0)  # z min
    assert mesh.bounds[5] == pytest.approx(0.0)  # z max — flat
    scene.close()


def test_choropleth_classifies_the_fill():
    """#TD-26: choropleth() colours a flat fill by a classified column."""
    scene = Scene3D(off_screen=True)
    scene.choropleth(_squares(), column="pop", scheme="quantiles", k=2)
    mesh = scene.layers[0][0]
    assert VALUE in mesh.cell_data
    assert sorted({int(v) for v in mesh.cell_data[VALUE]}) == [0, 1]
    assert mesh.bounds[5] == pytest.approx(0.0)  # still flat
    scene.close()


def test_choropleth_needs_a_column():
    """A choropleth with no column is refused — there is nothing to colour by."""
    scene = Scene3D(off_screen=True)
    squares = _squares()
    with pytest.raises(ValueError, match="needs a column"):
        scene.choropleth(squares)
    scene.close()


def test_polygons_carve_holes():
    """A flat fill carves interior rings, like the extrusion does."""
    holed = gpd.GeoDataFrame(
        {"v": [1.0]},
        geometry=[
            Polygon(
                [(0, 0), (4, 0), (4, 4), (0, 4)],
                holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
            )
        ],
    )
    scene = Scene3D(off_screen=True)
    scene.polygons(holed)
    # Cap area is 16 - 4 = 12 with the hole carved, not 16.
    assert float(scene.layers[0][0].area) == pytest.approx(12.0)
    scene.close()
