"""DW.2 — web-tier vector builders (points/lines/polygons/choropleth) + symbology.

The colour logic lives in pure helpers (``_color_expr`` → MapLibre paint, breaks via
``cleopatra.styling.styles.classify``); those are tested **without** the engine (cleopatra/numpy/matplotlib are core
deps). The builders that construct ``maplibre`` layers ``importorskip`` maplibre and assert the full
build → render → save path, so the lean dev env stays green while the ``web`` env runs everything.
"""

import numpy as np
import pytest

from digitalearth.base.symbology import MISSING_COLOR
from digitalearth.web import WebMap


@pytest.fixture
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


@pytest.fixture
def points_gdf():
    """Five points in lon/lat with a ``value`` column."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(x, x) for x in range(5)]
    return gpd.GeoDataFrame(
        {"value": [0.0, 1.0, 2.0, 3.0, 4.0]}, geometry=geoms, crs=4326
    )


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

        # The step is wrapped in a `case` that sends a non-numeric (missing) value to MISSING_COLOR, because
        # MapLibre's `step` has no null arm — so a feature with no value reads as missing rather than as the
        # lowest class, which is how every other tier draws it.
        assert expr[0] == "case", (
            f"a graduated expression must guard missing values, got {expr[0]!r}"
        )
        assert expr[1] == ["==", ["typeof", ["get", "pop"]], "number"], (
            f"the guard must test the column's type, got {expr[1]!r}"
        )
        assert expr[3] == MISSING_COLOR, (
            f"an unclassifiable feature must take the shared missing colour, got {expr[3]!r}"
        )

        step = expr[2]
        assert step[0] == "step", (
            f"graduated colouring must be a step expression, got {step[0]!r}"
        )
        assert step[1] == ["get", "pop"], (
            "the step input must read the column with ['get', column]"
        )
        # step layout: [step, [get,col], color0, e1, color1, e2, color2, ...] -> interior edges only.
        interior = [step[i] for i in range(3, len(step), 2)]
        assert np.allclose(interior, edges[1:-1]), (
            f"step stops {interior} != classifier {edges[1:-1]}"
        )

    def test_a_missing_value_is_not_coloured_as_the_lowest_class(self):
        """A feature whose column is null takes the shared missing colour, not the first class.

        Test scenario:
            MapLibre evaluates ``step`` left to right and has no null arm, so a missing value used to fall
            into the first bucket — rendering as "smallest", which on a choropleth reads as data rather than
            as absence. Every other tier draws it neutral grey; this pins that the web tier now does too.
        """
        expr = WebMap()._color_expr(np.arange(20.0), "pop", "quantiles", 4, "viridis")
        first_class = expr[2][2]
        assert expr[3] == MISSING_COLOR != first_class, (
            f"missing must differ from the lowest class: missing={expr[3]!r} lowest={first_class!r}"
        )

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
        assert stops[0] == 0.0, f"ramp should span the data: {stops}"
        assert stops[-1] == 10.0, f"ramp should span the data: {stops}"
        colors = [expr[i] for i in range(4, len(expr), 2)]
        assert all(c.startswith("#") for c in colors), (
            f"ramp colours must be hex: {colors}"
        )

    @pytest.mark.parametrize(
        "lo, hi",
        [
            (-3.7, 12.9),
            (3.5, 91.25),
            (0.1, 0.7),
            (-1.0, 1.0),
            (2.2, 7.7),
            (-273.15, 100.0),
        ],
    )
    def test_the_continuous_legend_holds_the_stops_that_were_drawn(self, lo, hi):
        """``last_legend["values"]`` is ``last_breaks``, not a recomputation that usually agrees.

        Test scenario:
            The ramp stops come from `np.linspace`, which pins its final element to `stop` exactly. The
            legend recomputed them as `lo + (hi - lo) * i / (stops - 1)`, which does not: for
            `lo=-3.7, hi=12.9` the top swatch read `12.900000000000002` while the MapLibre `interpolate`
            stop and `last_breaks` both read `12.9`. A legend value that is not a value the layer draws is
            the exact disagreement `LegendSpec` was introduced to remove.

            Six pairs, because the drift depends on the arithmetic of the particular limits — five of these
            agree either way, which is how a single hand-picked sample passed while the property did not
            hold.
        """
        m = WebMap()
        expr = m._color_expr(np.array([lo, hi]), "v", None, 5, "viridis")
        drawn = [expr[i] for i in range(3, len(expr), 2)]
        assert m.last_legend["values"] == m.last_breaks, (
            f"legend {m.last_legend['values']} must be the recorded breaks {m.last_breaks}"
        )
        assert m.last_legend["values"] == drawn, (
            f"and the stops the expression draws {drawn}"
        )

    def test_constant_values_do_not_crash_continuous(self):
        """A constant column widens the range instead of producing a zero-width ramp."""
        expr = WebMap()._color_expr(np.full(5, 3.0), "v", None, 5, "viridis")
        stops = [expr[i] for i in range(3, len(expr), 2)]
        assert stops[0] < stops[-1], "constant data must still yield an increasing ramp"

    def test_categorical_match_expression(self):
        """scheme='categorical' compiles a MapLibre `match` over the distinct values (DC.8)."""
        m = WebMap()
        expr = m._color_expr(
            np.array(["a", "b", "a", "c"], dtype=object),
            "kind",
            "categorical",
            5,
            "tab10",
        )
        assert expr[0] == "match", (
            f"categorical colouring must be a match expression, got {expr[0]!r}"
        )
        assert expr[1] == ["get", "kind"], (
            "the match input must read the column with ['get', column]"
        )
        assert expr[2] == "a", f"category literals misordered: {expr}"
        assert expr[4] == "b", f"category literals misordered: {expr}"
        assert expr[6] == "c", f"category literals misordered: {expr}"
        assert expr[-1] == "#cccccc", (
            "the match expression must end with a default colour"
        )
        assert m.last_breaks == ["a", "b", "c"], (
            f"categories should be recorded: {m.last_breaks}"
        )

    def test_categorical_numeric_literals_are_json_native(self):
        """Numeric categories are coerced to native int for the MapLibre literal."""
        m = WebMap()
        expr = m._color_expr(np.array([1, 2, 1, 3]), "code", "categorical", 5, "tab10")
        literals = [expr[i] for i in range(2, len(expr) - 1, 2)]
        assert literals == [1, 2, 3]
        assert all(type(v) is int for v in literals)

    def test_categorical_whole_float_labels_narrow_to_int(self):
        """Whole-valued float categories become int labels — MapLibre rejects non-integer match labels (M1)."""
        m = WebMap()
        expr = m._color_expr(
            np.array([1.0, 2.0, 1.0, 3.0]), "zone", "categorical", 5, "tab10"
        )
        literals = [expr[i] for i in range(2, len(expr) - 1, 2)]
        assert literals == [1, 2, 3], f"float cats must narrow: {literals}"
        assert all(type(v) is int for v in literals), (
            f"float cats must narrow: {literals}"
        )
        assert all(type(b) is int for b in m.last_breaks), (
            f"recorded breaks must narrow too: {m.last_breaks}"
        )

    def test_categorical_non_integer_float_rejected(self):
        """A non-integer float category cannot key a MapLibre match and is rejected clearly (M1)."""
        array = np.array([1.5, 2.5])
        webMap = WebMap()
        with pytest.raises(ValueError, match="non-integer category"):
            webMap._color_expr(array, "x", "categorical", 5, "tab10")

    def test_cmap_hex_count_and_format(self):
        """``_cmap_hex`` returns the requested number of hex colours."""
        colors = WebMap()._cmap_hex("viridis", 4)
        assert len(colors) == 4 and all(
            c.startswith("#") and len(c) == 7 for c in colors
        )


class TestVectorBuildersNeedEngine:
    """The point/line/polygon/choropleth builders build → render → save (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_choropleth_registers_a_layer_and_renders(self, polygons_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().choropleth(polygons_gdf, column="pop", scheme="quantiles", k=4)
        assert len(m.layers) == 1, "choropleth should register exactly one layer"
        assert m._last_layer_id is not None, (
            "the data layer id must be recorded for popup/tooltip"
        )
        assert m.last_breaks is not None, "choropleth must record its class breaks"
        assert isinstance(m.render(), MapWidget)

    def test_choropleth_missing_column_raises(self, polygons_gdf):
        webMap = WebMap()
        with pytest.raises(KeyError, match="nope"):
            webMap.choropleth(polygons_gdf, column="nope")

    def test_choropleth_constant_column_raises_clear_error(self):
        """A no-spread column surfaces a web-tier-friendly message, not a bare cleopatra error (L2)."""
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"pop": [5.0, 5.0, 5.0]},
            geometry=[Polygon([(i, 0), (i + 1, 0), (i + 0.5, 1)]) for i in range(3)],
            crs=4326,
        )
        webMap = WebMap()
        with pytest.raises(ValueError, match="cannot classify column 'pop'"):
            # An explicit scheme: `choropleth` is a continuous ramp by default now (C4), and a ramp over
            # a constant column has a range to widen rather than classes to cut.
            webMap.choropleth(gdf, column="pop", scheme="quantiles")

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
        webMap = WebMap()
        with pytest.raises(ValueError, match="unknown basemap provider") as exc:
            webMap.basemap("NoSuchProvider")
        # the suggestion list uses the canonical, correctly-cased names, not "Cartodark"/"Osm" (N1)
        assert "CartoDark" in str(exc.value), f"mis-cased names: {exc.value}"
        assert "OSM" in str(exc.value), f"mis-cased names: {exc.value}"

    def test_basemap_registers_an_underlay(self, polygons_gdf):
        """A basemap added after data is still drawn first (underlay at index 0)."""
        m = WebMap().choropleth(polygons_gdf, column="pop").basemap("CartoDark")
        assert len(m.layers) == 2
        # the basemap callable must sit at the bottom of the stack despite being added last
        assert m.layers[0] is not m.layers[-1]

    def test_popup_without_a_layer_raises(self):
        webMap = WebMap()
        with pytest.raises(ValueError, match="popup"):
            webMap.popup(["pop"])

    def test_popup_defaults_to_the_last_layer_and_queues(self, polygons_gdf):
        """A popup binds to the layer just drawn, and is queued like any other closure.

        Args:
            polygons_gdf: The features the popup is bound to.

        Test scenario:
            Only the refusal was covered; the path that binds a popup went through the builders' queue
            unexercised, which is where a drawn layer is placed among the bands (#292).
        """
        from maplibre.ipywidget import MapWidget

        m = WebMap().choropleth(polygons_gdf, column="pop").popup(["pop"])
        assert m._layer_tree.get(m.layer_ids[0]).kind == "choropleth", m.layer_ids
        assert isinstance(m.render(), MapWidget), (
            "the popup closure must replay onto the widget"
        )

    def test_tooltip_defaults_to_last_layer_and_renders(self, polygons_gdf):
        from maplibre.ipywidget import MapWidget

        m = WebMap().choropleth(polygons_gdf, column="pop").tooltip(["pop"])
        assert len(m.layers) == 2
        assert isinstance(m.render(), MapWidget)

    def test_navigation_scale_fullscreen_controls_render(self, polygons_gdf):
        """ED.13 — nav/scale/fullscreen controls register and the map still renders."""
        from maplibre.ipywidget import MapWidget

        m = (
            WebMap()
            .polygons(polygons_gdf)
            .navigation()
            .scale_bar(unit="imperial")
            .fullscreen()
        )
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
        drawn = WebMap().polygons(polygons_gdf)
        with pytest.raises(ValueError, match="distance and/or area"):
            drawn.measure(distance=False, area=False)
        # the mode guard is checked before position, so it wins when both are invalid (N4)
        with_bad_position = WebMap().polygons(polygons_gdf)
        with pytest.raises(ValueError, match="distance and/or area"):
            with_bad_position.measure(distance=False, area=False, position="bad")

    def test_control_position_is_validated(self):
        """An unknown control corner fails fast with a clear error rather than at render time (N3)."""
        webMap = WebMap()
        with pytest.raises(ValueError, match="unknown control position"):
            webMap.navigation(position="middle")
        webMap = WebMap()
        with pytest.raises(ValueError, match="unknown control position"):
            webMap.scale_bar(position="nowhere")


class TestAttributeTemplate:
    """``_attribute_template`` builds the right popup/tooltip kwargs (no engine needed)."""

    def test_no_fields_is_empty(self):
        assert WebMap()._attribute_template(None) == {}

    def test_single_field_uses_prop(self):
        assert WebMap()._attribute_template(["pop"]) == {"prop": "pop"}

    def test_multiple_fields_build_html_template(self):
        out = WebMap()._attribute_template(["a", "b"])
        assert "template" in out
        assert "{a}" in out["template"]
        assert "{b}" in out["template"]


class TestVectorBuilderRasterGuard:
    """Every vector builder rejects a raster at the shared ``_display_gdf`` choke point.

    Before the guard each builder failed differently on the same input — ``choropleth`` raised
    ``TypeError: argument of type 'int' is not iterable`` (issue #152's exact message), ``points`` and
    ``polygons`` raised ``has no len()``, ``heatmap`` and ``cluster`` raised ``AttributeError``, and
    ``lines`` accepted the raster and drew nothing at all.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    @pytest.mark.parametrize(
        "method, kwargs",
        [
            ("points", {}),
            ("lines", {}),
            ("polygons", {}),
            ("choropleth", {"column": "value"}),
            ("heatmap", {}),
            ("cluster", {}),
            ("deck_scatter", {}),
            ("deck_polygons", {}),
            ("extrusion", {"height": "value"}),
            ("point_cloud", {}),
        ],
    )
    def test_raster_is_rejected_by_name(self, dataset, method, kwargs):
        """Each builder names itself and the raster alternatives instead of failing incidentally.

        Args:
            dataset: The sample pyramids raster fixture.
            method: The builder under test.
            kwargs: Extra required arguments for that builder.

        Test scenario:
            A ``Dataset`` reaches the shared guard and is rejected with a message quoting the builder that
            was called, so the caller knows both what they did and what to do instead.
        """
        builder = getattr(WebMap(), method)
        with pytest.raises(TypeError) as excinfo:
            builder(dataset, **kwargs)
        message = str(excinfo.value)
        assert message.startswith(f"{method}()"), (
            f"{method} did not name itself: {message}"
        )
        assert "field" in message, (
            f"{method} did not name the raster builder: {message}"
        )
        assert "argument of type" not in message, (
            f"{method} still leaks the incidental error: {message}"
        )
        assert hasattr(WebMap, method), (
            f"the guard label {method!r} is not a real WebMap method"
        )

    def test_lines_no_longer_silently_accepts_a_raster(self, dataset):
        """``lines`` used to register a layer for a raster and render nothing.

        Args:
            dataset: The sample pyramids raster fixture.

        Test scenario:
            The worst of the six failures: no exception at all, so a caller had no signal that the map
            would come out empty.
        """
        m = WebMap()
        with pytest.raises(TypeError):
            m.lines(dataset)
        assert m.layers == [], "a rejected raster must not leave a layer registered"

    def test_a_vector_layer_still_passes_every_builder(self, points_gdf):
        """The guard must not disturb the supported path for any point-taking builder.

        Args:
            points_gdf: The existing five-point fixture.

        Test scenario:
            Point geometry is valid for points/heatmap/cluster, so none of them may raise.
        """
        for method in ("points", "heatmap", "cluster"):
            m = getattr(WebMap(), method)(points_gdf)
            assert m.layers, f"{method} accepted the layer but registered nothing"

    def test_a_table_without_active_geometry_is_told_to_set_geometry(self):
        """A geometry-less table gets the ``set_geometry`` message, not raster advice.

        Test scenario:
            The shared guard separates a real table missing its active geometry from an actual raster,
            so the two get different, accurate instructions.
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        table = gpd.GeoDataFrame({"value": [1.0], "geom": [Point(0, 0)]})
        webMap = WebMap()
        with pytest.raises(TypeError, match="set_geometry"):
            webMap.points(table)

    def test_point_cloud_still_accepts_a_raw_xyz_sequence(self):
        """``point_cloud`` takes bare coordinate triples, so it must not get the full vector guard.

        Test scenario:
            The reason this builder uses the raster-only rejection rather than ``_require_vector``: a
            plain sequence has no ``geometry`` and is still valid input.
        """
        m = WebMap().point_cloud([(0.0, 0.0, 1.0), (1.0, 1.0, 2.0)])
        assert m.layers, "a raw xyz sequence must still register a layer"


class TestAContinuousRampNeedsSomethingToScale:
    """A column with nothing finite in it cannot produce stops, and must say so."""

    def test_an_all_nan_column_is_refused(self):
        """Scaling over NaN would compile `NaN` stops into the paint expression.

        Test scenario:
            MapLibre accepts such a layer and draws nothing, so the failure surfaces as a blank map
            rather than as an error about the data.
        """
        import numpy as np

        web_map = WebMap()
        full = np.full(4, np.nan)
        with pytest.raises(ValueError, match="no finite values to colour"):
            web_map._color_expr(full, "depth", None, 5, "viridis")


class TestForcedBigPolygonsBehaveLikeForcedBigPoints:
    """`big=True` routes to deck.gl, which carries no per-feature symbology and no registry entry.

    The points half of this was covered; the polygons half compiled the same two branches and was not,
    so a caller could have lost their choropleth colouring to a silent deck route.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _capture():
        """Attach a loguru sink for warnings.

        Returns:
            `(records, sink_id)` — the list the sink fills, and the handle to remove it afterwards.
        """
        from loguru import logger

        records = []
        return records, logger.add(records.append, level="WARNING")

    def test_a_forced_route_warns_that_the_colouring_is_dropped(self, polygons_gdf):
        """Losing a choropleth's classes to a routing decision must not be silent.

        Args:
            polygons_gdf: The fixture frame.
        """
        from loguru import logger

        records, sink = self._capture()
        try:
            m = WebMap().polygons(polygons_gdf, column="pop", big=True)
        finally:
            logger.remove(sink)
        assert any("styling is dropped" in str(record) for record in records), records
        assert m.layer_ids == [], "a deck overlay is not a addressable MapLibre layer"

    def test_a_forced_route_still_draws(self, polygons_gdf, tmp_path):
        """The warning is not a refusal — the features still have to reach the page.

        Args:
            polygons_gdf: The fixture frame.
            tmp_path: pytest's per-test directory.
        """
        out = tmp_path / "deck.html"
        WebMap().basemap().polygons(polygons_gdf, big=True).save(str(out))
        assert out.stat().st_size > 1_000, "the forced deck route wrote nothing"
