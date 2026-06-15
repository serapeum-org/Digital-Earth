"""DW.3 — web-tier big-data builders: heatmap, clustering, deck.gl, and the auto-threshold routing.

The routing decision (``_route_big``) is pure and tested without the engine; the heatmap/cluster/deck
builders construct ``maplibre`` objects and so ``importorskip`` maplibre.
"""

import pytest

from digitalearth.web import WebMap


@pytest.fixture()
def points_gdf():
    """Five points in lon/lat with a ``value`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(x, x) for x in range(5)]
    return gpd.GeoDataFrame({"value": [0.0, 1.0, 2.0, 3.0, 4.0]}, geometry=geoms, crs=4326)


@pytest.fixture()
def polygons_gdf():
    """Four triangles in lon/lat with a ``pop`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    geoms = [
        Polygon([(0, 0), (1, 0), (1, 1)]),
        Polygon([(2, 2), (3, 2), (3, 3)]),
        Polygon([(4, 4), (5, 4), (5, 5)]),
        Polygon([(6, 6), (7, 6), (7, 7)]),
    ]
    return gpd.GeoDataFrame({"pop": [1.0, 5.0, 9.0, 3.0]}, geometry=geoms, crs=4326)


class TestRouteBig:
    """``_route_big`` decides (and logs) GPU routing from the feature count (no engine)."""

    def test_over_threshold_is_true(self, points_gdf):
        m = WebMap()
        m.big_data_threshold = 2
        assert m._route_big(points_gdf, "points") is True

    def test_at_or_below_threshold_is_false(self, points_gdf):
        m = WebMap()
        m.big_data_threshold = 100
        assert m._route_big(points_gdf, "points") is False


class TestBigDataBuildersNeedEngine:
    """heatmap / cluster / deck builders build and render (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_heatmap_registers_layer(self, points_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().heatmap(points_gdf, weight="value")
        assert len(m.layers) == 1 and m._last_layer_id is not None
        assert isinstance(m.render(), MapWidget)

    def test_heatmap_rejects_polygons(self, polygons_gdf):
        """heatmap is point-only — polygons must raise, not render an empty layer (M4)."""
        with pytest.raises(TypeError, match="point geometries"):
            WebMap().heatmap(polygons_gdf)

    def test_cluster_rejects_polygons(self, polygons_gdf):
        with pytest.raises(TypeError, match="point geometries"):
            WebMap().cluster(polygons_gdf)

    def test_cluster_registers_layer(self, points_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().cluster(points_gdf)
        assert len(m.layers) == 1
        assert isinstance(m.render(), MapWidget)

    def test_deck_scatter_accumulates_into_one_applier(self, points_gdf):
        m = WebMap().deck_scatter(points_gdf).deck_scatter(points_gdf)
        assert len(m.layers) == 1, "all deck layers share a single add_deck_layers applier"
        assert m._deck_layers is not None and len(m._deck_layers) == 2
        assert m._deck_layers[0]["@@type"] == "GeoJsonLayer"

    def test_deck_polygons_layer_shape(self, polygons_gdf):
        m = WebMap().deck_polygons(polygons_gdf)
        assert m._deck_layers[0]["@@type"] == "GeoJsonLayer"
        assert m._deck_layers[0]["filled"] is True

    def test_points_auto_routes_to_deck_above_threshold(self, points_gdf):
        m = WebMap()
        m.big_data_threshold = 2  # 5 points > 2 → route to deck
        m.points(points_gdf)
        assert m._deck_layers is not None, "large point sets must route to a deck.gl layer"

    def test_points_big_false_keeps_circles(self, points_gdf):
        m = WebMap()
        m.big_data_threshold = 2
        m.points(points_gdf, big=False)
        assert m._deck_layers is None, "big=False must keep per-feature circles"
        assert len(m.layers) == 1

    def test_points_with_column_over_threshold_keeps_styling(self, points_gdf):
        """A column choropleth over the threshold must NOT silently auto-route to a flat deck layer (M1)."""
        m = WebMap()
        m.big_data_threshold = 2  # 5 points > 2, but a column is set
        m.points(points_gdf, column="value")
        assert m._deck_layers is None, "column styling must be preserved (no auto-route)"
        assert len(m.layers) == 1  # a per-feature MapLibre circle layer, not a flat deck layer

    def test_points_forced_big_with_column_warns_and_routes(self, points_gdf):
        """Forcing big=True with a column routes to deck but is the explicit, logged opt-in (M1)."""
        m = WebMap()
        m.points(points_gdf, column="value", big=True)
        assert m._deck_layers is not None, "big=True forces the deck path even with a column"

    def test_save_writes_a_file(self, tmp_path, points_gdf):
        out = tmp_path / "heat.html"
        WebMap().heatmap(points_gdf).basemap("CartoDark").save(str(out))
        assert out.stat().st_size > 1_000
