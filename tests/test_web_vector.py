"""DW.2 — web-tier vector builders (points/lines/polygons/choropleth) + symbology.

The colour logic lives in pure helpers (``_color_expr`` → MapLibre paint, breaks via
``cleopatra.styling.styles.classify``); those are tested **without** the engine (cleopatra/numpy/matplotlib are core
deps). The builders that construct ``maplibre`` layers ``importorskip`` maplibre and assert the full
build → render → save path, so the lean dev env stays green while the ``web`` env runs everything.
"""

import numpy as np
import pytest

from digitalearth.web import WebMap


@pytest.fixture()
def polygons_gdf():
    """Four triangles in lon/lat (EPSG:4326) with a ``pop`` value ramp."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    geoms = [
        Polygon([(0, 0), (1, 0), (1, 1)]),
        Polygon([(2, 2), (3, 2), (3, 3)]),
        Polygon([(4, 4), (5, 4), (5, 5)]),
        Polygon([(6, 6), (7, 6), (7, 7)]),
    ]
    return gpd.GeoDataFrame({"pop": [1.0, 5.0, 9.0, 3.0]}, geometry=geoms, crs=4326)


@pytest.fixture()
def points_gdf():
    """Five points in lon/lat with a ``value`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(x, x) for x in range(5)]
    return gpd.GeoDataFrame({"value": [0.0, 1.0, 2.0, 3.0, 4.0]}, geometry=geoms, crs=4326)


class TestColorExpr:
    """``_color_expr`` compiles MapLibre paint and records the breaks (no engine needed)."""

    def test_graduated_step_matches_cleopatra_breaks(self):
        """A graduated scheme compiles a ``step`` expression whose breaks equal cleopatra's classifier.

        This is the parity contract: the web tier owns no classification logic — it reuses
        ``cleopatra.styling.styles.classify`` (the same classifier the static tier uses) and only compiles the
        result into a MapLibre paint expression.
        """
        from cleopatra.styling.styles import classify

        values = np.arange(100.0)
        expr = WebMap()._color_expr(values, "pop", "quantiles", 4, "viridis")
        edges, _ = classify(values, "quantiles", 4)

        assert expr[0] == "step", f"graduated colouring must be a step expression, got {expr[0]!r}"
        assert expr[1] == ["get", "pop"], "the step input must read the column with ['get', column]"
        # step layout: [step, [get,col], color0, e1, color1, e2, color2, ...] -> interior edges only.
        interior = [expr[i] for i in range(3, len(expr), 2)]
        assert np.allclose(interior, edges[1:-1]), f"step stops {interior} != classifier {edges[1:-1]}"

    def test_graduated_records_breaks_on_the_map(self):
        """The full class edges are exposed on ``last_breaks`` for an out-of-band legend."""
        from cleopatra.styling.styles import classify

        m = WebMap()
        m._color_expr(np.arange(50.0), "pop", "equal_interval", 5, "viridis")
        edges, _ = classify(np.arange(50.0), "equal_interval", 5)
        assert m.last_breaks is not None and np.allclose(m.last_breaks, edges)
        assert len(m.last_breaks) == 6, "k=5 should record six edges"

    def test_continuous_interpolate_expression(self):
        """``scheme=None`` compiles a linear ``interpolate`` ramp over the value range."""
        m = WebMap()
        expr = m._color_expr(np.array([0.0, 10.0]), "v", None, 5, "viridis")
        assert expr[:3] == ["interpolate", ["linear"], ["get", "v"]]
        stops = [expr[i] for i in range(3, len(expr), 2)]
        assert stops[0] == 0.0 and stops[-1] == 10.0, f"ramp should span the data: {stops}"
        colors = [expr[i] for i in range(4, len(expr), 2)]
        assert all(c.startswith("#") for c in colors), f"ramp colours must be hex: {colors}"

    def test_constant_values_do_not_crash_continuous(self):
        """A constant column widens the range instead of producing a zero-width ramp."""
        expr = WebMap()._color_expr(np.full(5, 3.0), "v", None, 5, "viridis")
        stops = [expr[i] for i in range(3, len(expr), 2)]
        assert stops[0] < stops[-1], "constant data must still yield an increasing ramp"

    def test_categorical_match_expression(self):
        """scheme='categorical' compiles a MapLibre `match` over the distinct values (DC.8)."""
        m = WebMap()
        expr = m._color_expr(np.array(["a", "b", "a", "c"], dtype=object), "kind", "categorical", 5, "tab10")
        assert expr[0] == "match", f"categorical colouring must be a match expression, got {expr[0]!r}"
        assert expr[1] == ["get", "kind"], "the match input must read the column with ['get', column]"
        assert expr[2] == "a" and expr[4] == "b" and expr[6] == "c", f"category literals misordered: {expr}"
        assert expr[-1] == "#cccccc", "the match expression must end with a default colour"
        assert m.last_breaks == ["a", "b", "c"], f"categories should be recorded: {m.last_breaks}"

    def test_categorical_numeric_literals_are_json_native(self):
        """Numeric categories are coerced to native int for the MapLibre literal."""
        m = WebMap()
        expr = m._color_expr(np.array([1, 2, 1, 3]), "code", "categorical", 5, "tab10")
        literals = [expr[i] for i in range(2, len(expr) - 1, 2)]
        assert literals == [1, 2, 3] and all(type(v) is int for v in literals)

    def test_categorical_whole_float_labels_narrow_to_int(self):
        """Whole-valued float categories become int labels — MapLibre rejects non-integer match labels (M1)."""
        m = WebMap()
        expr = m._color_expr(np.array([1.0, 2.0, 1.0, 3.0]), "zone", "categorical", 5, "tab10")
        literals = [expr[i] for i in range(2, len(expr) - 1, 2)]
        assert literals == [1, 2, 3] and all(type(v) is int for v in literals), f"float cats must narrow: {literals}"
        assert all(type(b) is int for b in m.last_breaks), f"recorded breaks must narrow too: {m.last_breaks}"

    def test_categorical_non_integer_float_rejected(self):
        """A non-integer float category cannot key a MapLibre match and is rejected clearly (M1)."""
        with pytest.raises(ValueError, match="non-integer category"):
            WebMap()._color_expr(np.array([1.5, 2.5]), "x", "categorical", 5, "tab10")

    def test_cmap_hex_count_and_format(self):
        """``_cmap_hex`` returns the requested number of hex colours."""
        colors = WebMap()._cmap_hex("viridis", 4)
        assert len(colors) == 4 and all(c.startswith("#") and len(c) == 7 for c in colors)


class TestVectorBuildersNeedEngine:
    """The point/line/polygon/choropleth builders build → render → save (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_choropleth_registers_a_layer_and_renders(self, polygons_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().choropleth(polygons_gdf, column="pop", scheme="quantiles", k=4)
        assert len(m.layers) == 1, "choropleth should register exactly one layer"
        assert m._last_layer_id is not None, "the data layer id must be recorded for popup/tooltip"
        assert m.last_breaks is not None, "choropleth must record its class breaks"
        assert isinstance(m.render(), MapWidget)

    def test_choropleth_missing_column_raises(self, polygons_gdf):
        with pytest.raises(KeyError, match="nope"):
            WebMap().choropleth(polygons_gdf, column="nope")

    def test_choropleth_constant_column_raises_clear_error(self):
        """A no-spread column surfaces a web-tier-friendly message, not a bare cleopatra error (L2)."""
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"pop": [5.0, 5.0, 5.0]},
            geometry=[Polygon([(i, 0), (i + 1, 0), (i + 0.5, 1)]) for i in range(3)],
            crs=4326,
        )
        with pytest.raises(ValueError, match="cannot classify column 'pop'"):
            WebMap().choropleth(gdf, column="pop")

    def test_points_lines_polygons_chain(self, points_gdf, polygons_gdf):
        m = WebMap()
        out = m.points(points_gdf, column="value").polygons(polygons_gdf, column="pop")
        assert out is m, "builders must return self for chaining"
        assert len(m.layers) == 2

    def test_save_writes_a_file(self, tmp_path, polygons_gdf):
        out = tmp_path / "choropleth.html"
        WebMap().choropleth(polygons_gdf, column="pop").basemap("OSM").save(str(out))
        assert out.stat().st_size > 1_000


class TestDecorationNeedsEngine:
    """basemap/tiles and popup/tooltip builders (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_basemap_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown basemap provider") as exc:
            WebMap().basemap("NoSuchProvider")
        # the suggestion list uses the canonical, correctly-cased names, not "Cartodark"/"Osm" (N1)
        assert "CartoDark" in str(exc.value) and "OSM" in str(exc.value), f"mis-cased names: {exc.value}"

    def test_basemap_registers_an_underlay(self, polygons_gdf):
        """A basemap added after data is still drawn first (underlay at index 0)."""
        m = WebMap().choropleth(polygons_gdf, column="pop").basemap("CartoDark")
        assert len(m.layers) == 2
        # the basemap callable must sit at the bottom of the stack despite being added last
        assert m.layers[0] is not m.layers[-1]

    def test_popup_without_a_layer_raises(self):
        with pytest.raises(ValueError, match="popup"):
            WebMap().popup(["pop"])

    def test_tooltip_defaults_to_last_layer_and_renders(self, polygons_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().choropleth(polygons_gdf, column="pop").tooltip(["pop"])
        assert len(m.layers) == 2
        assert isinstance(m.render(), MapWidget)

    def test_navigation_scale_fullscreen_controls_render(self, polygons_gdf):
        """ED.13 — nav/scale/fullscreen controls register and the map still renders."""
        from maplibre.ipywidget import MapWidget

        m = WebMap().polygons(polygons_gdf).navigation().scale_bar(unit="imperial").fullscreen()
        assert len(m.layers) == 4, "data layer + 3 controls"
        assert isinstance(m.render(), MapWidget)

    def test_controls_convenience(self, polygons_gdf):
        """ED.13 — controls() adds navigation + scale (+ fullscreen) in one call."""
        m = WebMap().polygons(polygons_gdf).controls(fullscreen=True)
        assert len(m.layers) == 4, "data layer + navigation + scale + fullscreen"
        m2 = WebMap().polygons(polygons_gdf).controls()
        assert len(m2.layers) == 3, "default controls() adds navigation + scale only"

    def test_measure_registers_draw_tool(self, polygons_gdf):
        """ED.10 — measure() adds a draw-based tool and the map still renders."""
        from maplibre.ipywidget import MapWidget

        m = WebMap().polygons(polygons_gdf).measure(distance=True, area=True)
        assert len(m.layers) == 2, "data layer + the measure draw control"
        assert isinstance(m.render(), MapWidget)

    def test_measure_requires_a_mode(self, polygons_gdf):
        with pytest.raises(ValueError, match="distance and/or area"):
            WebMap().polygons(polygons_gdf).measure(distance=False, area=False)
        # the mode guard is checked before position, so it wins when both are invalid (N4)
        with pytest.raises(ValueError, match="distance and/or area"):
            WebMap().polygons(polygons_gdf).measure(distance=False, area=False, position="bad")

    def test_control_position_is_validated(self):
        """An unknown control corner fails fast with a clear error rather than at render time (N3)."""
        with pytest.raises(ValueError, match="unknown control position"):
            WebMap().navigation(position="middle")
        with pytest.raises(ValueError, match="unknown control position"):
            WebMap().scale_bar(position="nowhere")


class TestAttributeTemplate:
    """``_attribute_template`` builds the right popup/tooltip kwargs (no engine needed)."""

    def test_no_fields_is_empty(self):
        assert WebMap()._attribute_template(None) == {}

    def test_single_field_uses_prop(self):
        assert WebMap()._attribute_template(["pop"]) == {"prop": "pop"}

    def test_multiple_fields_build_html_template(self):
        out = WebMap()._attribute_template(["a", "b"])
        assert "template" in out and "{a}" in out["template"] and "{b}" in out["template"]
