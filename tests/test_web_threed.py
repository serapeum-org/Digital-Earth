"""DW.4 — web-tier 3-D builders: extrusion, point cloud, 3D tiles, glTF, terrain, globe.

``_point_cloud_data`` is pure and tested without the engine; the builders that construct ``maplibre`` /
deck.gl layers ``importorskip`` maplibre and assert the registered layer / deck spec.
"""

import numpy as np
import pytest

from digitalearth.web import WebMap


@pytest.fixture()
def polygons_gdf():
    """Three triangles in lon/lat with a ``pop`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    geoms = [
        Polygon([(0, 0), (1, 0), (1, 1)]),
        Polygon([(2, 2), (3, 2), (3, 3)]),
        Polygon([(4, 4), (5, 4), (5, 5)]),
    ]
    return gpd.GeoDataFrame({"pop": [10.0, 50.0, 90.0]}, geometry=geoms, crs=4326)


@pytest.fixture()
def points_gdf():
    """Four points in lon/lat with an ``elev`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(x, x) for x in range(4)]
    return gpd.GeoDataFrame({"elev": [5.0, 15.0, 25.0, 35.0]}, geometry=geoms, crs=4326)


class TestPointCloudData:
    """``_point_cloud_data`` builds deck position rows (no engine)."""

    def test_from_xyz_array(self):
        rows = WebMap()._point_cloud_data(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), None)
        assert rows[0]["position"] == [1.0, 2.0, 3.0]
        assert rows[1]["position"] == [4.0, 5.0, 6.0]

    def test_from_xy_array_defaults_z_zero(self):
        rows = WebMap()._point_cloud_data(np.array([[1.0, 2.0], [3.0, 4.0]]), None)
        assert rows[0]["position"] == [1.0, 2.0, 0.0]

    def test_from_geodataframe_uses_z_column(self, points_gdf):
        rows = WebMap()._point_cloud_data(points_gdf, "elev")
        assert len(rows) == 4
        assert rows[0]["position"][2] == 5.0 and rows[3]["position"][2] == 35.0


class TestThreeDNeedEngine:
    """Extrusion / terrain / globe / deck 3-D builders (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_extrusion_registers_fill_extrusion(self, polygons_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().extrusion(polygons_gdf, height="pop", column="pop")
        assert len(m.layers) == 1 and m._last_layer_id is not None
        assert isinstance(m.render(), MapWidget)

    def test_extrusion_constant_height(self, polygons_gdf):
        m = WebMap().extrusion(polygons_gdf, height=100.0)
        assert len(m.layers) == 1

    def test_terrain_and_globe_chain_and_render(self, polygons_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().extrusion(polygons_gdf, height="pop").terrain(exaggeration=1.5).globe(True)
        assert len(m.layers) == 3
        assert isinstance(m.render(), MapWidget)

    def test_point_cloud_accumulates_deck(self, points_gdf):
        m = WebMap().point_cloud(points_gdf, z_column="elev")
        assert m._deck_layers is not None
        assert m._deck_layers[0]["@@type"] == "PointCloudLayer"

    def test_tiles_3d_layer(self):
        m = WebMap().tiles_3d("https://example.com/tileset.json")
        assert m._deck_layers[0]["@@type"] == "Tile3DLayer"

    def test_gltf_layer(self):
        m = WebMap().gltf("https://example.com/model.glb", 8.0, 47.0)
        layer = m._deck_layers[0]
        assert layer["@@type"] == "ScenegraphLayer"
        assert layer["data"][0]["position"] == [8.0, 47.0]

    def test_save_writes_a_file(self, tmp_path, polygons_gdf):
        out = tmp_path / "extrusion.html"
        WebMap().extrusion(polygons_gdf, height="pop", column="pop").save(str(out))
        assert out.stat().st_size > 1_000
