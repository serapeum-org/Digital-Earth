"""DW.3 — web-tier big-data builders: heatmap, clustering, deck.gl, and the auto-threshold routing.

The routing decision (``_route_big``) is pure and tested without the engine; the heatmap/cluster/deck
builders construct ``maplibre`` objects and so ``importorskip`` maplibre.
"""

import pytest

from digitalearth.web import WebMap


@pytest.fixture
def points_gdf():
    """Five points in lon/lat with a ``value`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(x, x) for x in range(5)]
    return gpd.GeoDataFrame(
        {"value": [0.0, 1.0, 2.0, 3.0, 4.0]}, geometry=geoms, crs=4326
    )


@pytest.fixture
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
        assert len(m.layers) == 1
        assert m._last_layer_id is not None
        assert isinstance(m.render(), MapWidget)

    def test_heatmap_rejects_polygons(self, polygons_gdf):
        """heatmap is point-only — polygons must raise, not render an empty layer (M4)."""
        webMap = WebMap()
        with pytest.raises(TypeError, match="point geometries"):
            webMap.heatmap(polygons_gdf)

    def test_cluster_rejects_polygons(self, polygons_gdf):
        webMap = WebMap()
        with pytest.raises(TypeError, match="point geometries"):
            webMap.cluster(polygons_gdf)

    def test_cluster_registers_layer(self, points_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().cluster(points_gdf)
        assert len(m.layers) == 1
        assert isinstance(m.render(), MapWidget)

    def test_deck_scatter_registers_an_addressable_layer(self, points_gdf):
        """Each deck.gl layer owns a tree slot since #396, so two calls make two addressable layers."""
        m = WebMap().deck_scatter(points_gdf, name="a").deck_scatter(points_gdf)
        assert m.layer_ids == ["a", "deck-scatter-1"], m.layer_ids
        assert m._deck_layers is None, "the described route no longer accumulates here"
        assert m.layers[0]["@@type"] == "GeoJsonLayer"
        assert m.get_layer("a").kind == "deck_scatter"

    def test_deck_scatter_layer_is_removable(self, points_gdf):
        """A big-data deck layer comes off the map by id like any other (#396)."""
        m = (
            WebMap()
            .deck_scatter(points_gdf, name="big")
            .points(points_gdf, name="small", big=False)
        )
        assert m.layer_ids == ["big", "small"], m.layer_ids
        m.remove_layer("big")
        assert m.layer_ids == ["small"], m.layer_ids

    def test_deck_polygons_layer_shape(self, polygons_gdf):
        m = WebMap().deck_polygons(polygons_gdf, name="areas")
        drawn = m.layers[0]
        assert drawn["@@type"] == "GeoJsonLayer"
        assert drawn["filled"] is True
        assert m.get_layer("areas").kind == "deck_polygons"

    def test_points_auto_routes_to_deck_above_threshold(self, points_gdf):
        m = WebMap()
        m.big_data_threshold = 2  # 5 points > 2 → route to deck
        m.points(points_gdf)
        assert m.layer_ids == ["deck-scatter-1"], (
            "large point sets must route to an addressable deck.gl layer"
        )
        assert m._deck_layers is None

    def test_points_big_route_honours_name_and_visible(self, points_gdf):
        """The old refusal is gone: the big-data route names and hides the layer like the small one (#396)."""
        m = WebMap()
        m.big_data_threshold = 2
        m.points(points_gdf, name="crowd", visible=False)
        assert m.layer_ids == ["crowd"], m.layer_ids
        assert m._renderer.is_visible("crowd") is False

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
        assert m._deck_layers is None, (
            "column styling must be preserved (no auto-route)"
        )
        # a per-feature MapLibre circle layer (kind 'points'), not a flat deck layer
        assert m.get_layer(m.layer_ids[0]).kind == "points"

    def test_points_forced_big_with_column_warns_and_routes(self, points_gdf):
        """Forcing big=True with a column routes to deck but is the explicit, logged opt-in (M1)."""
        m = WebMap()
        m.points(points_gdf, column="value", big=True)
        assert m.layer_ids == ["deck-scatter-1"], (
            "big=True forces the deck path even with a column"
        )
        assert m.get_layer("deck-scatter-1").kind == "deck_scatter"

    def test_save_writes_a_file(self, tmp_path, points_gdf):
        out = tmp_path / "heat.html"
        WebMap().heatmap(points_gdf).basemap("CartoDark").save(str(out))
        assert out.stat().st_size > 1_000


def _applied_overlay(web_map):
    """Return the deck.gl layers the built page composes into its one ``addDeckOverlay`` call.

    Args:
        web_map: The map to build.

    Returns:
        The deck.gl JSON layers handed to the single overlay call, or an empty list when the page composes
        none.
    """
    widget = web_map._build_map_widget()
    for method, args in list(widget._message_queue):
        if method == "addDeckOverlay":
            return list(args[0])
    return []


class TestBigDataDeckLayerHasATreeSlot:
    """#396 — each big-data deck.gl layer owns a `LayerSpec`, so it is addressable like a small one."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_it_appears_in_layer_ids_with_the_callers_name(self, points_gdf):
        """(a) A named big-data deck layer is listed under that name, not off the tree."""
        m = WebMap().deck_scatter(points_gdf, name="crowd")
        assert m.layer_ids == ["crowd"], m.layer_ids
        assert m.get_layer("crowd").kind == "deck_scatter"

    def test_the_big_data_points_route_is_addressable_too(self, points_gdf):
        """The auto-route through `points()` above the threshold is the same addressable layer."""
        m = WebMap()
        m.big_data_threshold = 2
        m.points(points_gdf, name="crowd")
        assert m.layer_ids == ["crowd"], m.layer_ids
        assert m.get_layer("crowd").kind == "deck_scatter"

    def test_it_is_removable(self, points_gdf, polygons_gdf):
        """(b) `remove_layer` takes the big-data deck layer off by id and frees the name."""
        m = (
            WebMap()
            .deck_scatter(points_gdf, name="pts")
            .deck_polygons(polygons_gdf, name="areas")
        )
        assert m.layer_ids == ["pts", "areas"], m.layer_ids
        m.remove_layer("pts")
        assert m.layer_ids == ["areas"], m.layer_ids
        assert "pts" not in {layer.get("id") for layer in _applied_overlay(m)}

    def test_it_is_hideable_and_left_out_of_the_applied_overlay(self, points_gdf):
        """(c) `visible=False`/`set_visible` leaves the deck layer out of what the overlay draws.

        A deck.gl layer carries deck's own ``visible`` flag rather than a MapLibre layout, so a hidden one
        stays in the overlay array with ``visible: False`` — which deck.gl does not draw. The page helper
        reads it as off, so a switcher that reads the same flag leaves it out.
        """
        hidden_at_build = WebMap().deck_scatter(points_gdf, name="pts", visible=False)
        drawn = {
            layer["id"]: layer.get("visible", True)
            for layer in _applied_overlay(hidden_at_build)
        }
        assert drawn.get("pts") is False, drawn
        assert hidden_at_build._renderer.is_visible("pts") is False

        toggled = WebMap().deck_scatter(points_gdf, name="pts")
        toggled.set_visible("pts", False)
        assert toggled._renderer.is_visible("pts") is False
        drawn = {
            layer["id"]: layer.get("visible", True)
            for layer in _applied_overlay(toggled)
        }
        assert drawn.get("pts") is False, drawn

    def test_it_still_renders_into_the_overlay(self, points_gdf):
        """(d) The layer still reaches the page: the built overlay carries its deck.gl GeoJsonLayer spec."""
        m = WebMap().deck_scatter(points_gdf, name="pts")
        overlay = _applied_overlay(m)
        spec = next(layer for layer in overlay if layer["id"] == "pts")
        assert spec["@@type"] == "GeoJsonLayer", spec
        assert spec["pointType"] == "circle"

    def test_it_can_be_reordered_against_a_style_layer(self, points_gdf):
        """`move_layer` orders the described deck layer against MapLibre style layers (review H6)."""
        m = (
            WebMap()
            .deck_scatter(points_gdf, name="deck")
            .points(points_gdf, name="circles", big=False)
        )
        assert m.layer_ids == ["deck", "circles"], m.layer_ids
        m.move_layer("deck", 1)
        assert m.layer_ids == ["circles", "deck"], m.layer_ids
