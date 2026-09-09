"""DW.5 — web-tier time-slider (``timeslider``).

The bare state helper (``_temporal_times``) is tested without the engine; the slider build (which constructs
maplibre + ipywidgets) ``importorskip``s maplibre.
"""

import pytest

from digitalearth.web import WebMap


@pytest.fixture()
def timed_polygons():
    """Six triangles tagged with one of three time steps and a ``pop`` value."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    geoms = [Polygon([(i, 0), (i + 1, 0), (i + 0.5, 1)]) for i in range(6)]
    times = [2000, 2000, 2010, 2010, 2020, 2020]
    pop = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    return gpd.GeoDataFrame({"time": times, "pop": pop}, geometry=geoms, crs=4326)


def test_temporal_times_empty_without_slider():
    assert WebMap()._temporal_times() == []


class TestTimeSliderNeedsEngine:
    """``timeslider`` draws the layer once and renders a slider composite (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_records_distinct_time_steps(self, timed_polygons):
        m = WebMap().timeslider(timed_polygons, kdim="time", column="pop")
        assert m._temporal_times() == [2000, 2010, 2020]
        assert m._temporal["layer_id"] is not None

    def test_missing_kdim_raises(self, timed_polygons):
        with pytest.raises(KeyError, match="when"):
            WebMap().timeslider(timed_polygons, kdim="when")

    def test_render_returns_slider_plus_map(self, timed_polygons):
        import ipywidgets

        composite = WebMap().timeslider(timed_polygons, kdim="time", column="pop").render()
        assert isinstance(composite, ipywidgets.VBox)
        slider, _map = composite.children
        assert isinstance(slider, ipywidgets.SelectionSlider)
        assert len(slider.options) == 3, "one slider stop per distinct time step"

    def test_large_polygon_series_stays_filterable(self, timed_polygons):
        """A large temporal polygon set must keep a per-feature layer id for the filter (M3)."""
        m = WebMap()
        m.big_data_threshold = 2  # 6 polygons > 2; must NOT auto-route to deck (no layer id to filter)
        m.timeslider(timed_polygons, kdim="time")
        assert m._deck_layers is None, "temporal layers must not auto-route to deck.gl"
        assert m._temporal["layer_id"] is not None, "the slider needs a filterable layer id"

    def test_render_without_slider_is_bare_map(self, timed_polygons):
        from maplibre.ipywidget import MapWidget

        m = WebMap().choropleth(timed_polygons, column="pop")
        assert isinstance(m.render(), MapWidget), "no timeslider → bare map, not a composite"

    def test_save_writes_a_file(self, tmp_path, timed_polygons):
        out = tmp_path / "temporal.html"
        WebMap().timeslider(timed_polygons, kdim="time", column="pop").save(str(out))
        assert out.stat().st_size > 1_000


class TestTimeSliderRejectsNonVector:
    """Raster input is turned away with an actionable ``TypeError`` instead of dying inside the guard.

    A pyramids raster exposes ``columns`` as an ``int`` (the grid width in cells), so the attribute
    membership test used to raise ``TypeError: argument of type 'int' is not iterable`` — a message with no
    hint that ``timeslider`` is vector-only. Both raster types are covered: a ``Dataset`` reaches the check
    through ``_display_gdf``'s reproject branch, a ``DatasetCollection`` through its fall-through.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    @pytest.fixture()
    def collection(self, dataset):
        """The sample raster wrapped as a single-member ``DatasetCollection``."""
        from pyramids.dataset.collection import DatasetCollection

        return DatasetCollection.from_files(["examples/data/acc4000.tif"])

    def test_dataset_is_rejected(self, dataset):
        with pytest.raises(TypeError, match=r"timeslider\(\) renders a time-stepped vector layer"):
            WebMap().timeslider(dataset)

    def test_dataset_collection_is_rejected(self, collection):
        with pytest.raises(TypeError, match=r"timeslider\(\) renders a time-stepped vector layer"):
            WebMap().timeslider(collection)

    def test_message_names_the_input_and_the_raster_capable_tiers(self, dataset):
        """The error has to say what arrived and where a raster time stack belongs instead."""
        with pytest.raises(TypeError) as excinfo:
            WebMap().timeslider(dataset)
        message = str(excinfo.value)
        assert type(dataset).__name__ in message, "the rejected type is not named"
        assert "Map.animate" in message, "the matplotlib alternative is not named"
        assert "InteractiveMap.timecube" in message, "the interactive alternative is not named"

    def test_guard_runs_before_the_reprojection(self, dataset, monkeypatch):
        """A raster in another CRS must not be warped on its way to being rejected."""
        assert dataset.epsg != 4326, "fixture must not already be in the display CRS for this to bite"
        warps = []
        monkeypatch.setattr(type(dataset), "to_crs", lambda self, *a, **k: warps.append(a))
        with pytest.raises(TypeError):
            WebMap().timeslider(dataset)
        assert warps == [], "the raster was reprojected before being rejected"

    def test_bare_array_is_rejected_too(self):
        """Non-pyramids, non-vector input used to fall through to a misleading ``KeyError``."""
        np = pytest.importorskip("numpy")
        with pytest.raises(TypeError, match="ndarray"):
            WebMap().timeslider(np.zeros((4, 5)))

    def test_vector_input_still_passes_the_guard(self, timed_polygons):
        """The guard must not disturb the supported path."""
        m = WebMap().timeslider(timed_polygons, kdim="time", column="pop")
        assert m._temporal_times() == [2000, 2010, 2020]


@pytest.fixture()
def timed_points():
    """Six points tagged with one of three time steps and a ``pop`` value."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(i, i) for i in range(6)]
    return gpd.GeoDataFrame(
        {"time": [2000, 2000, 2010, 2010, 2020, 2020], "pop": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        geometry=geoms,
        crs=4326,
    )


@pytest.fixture()
def fake_widget():
    """A recording stand-in for the MapLibre ``MapWidget``, usable without the ``web`` extra.

    It subclasses a real ``ipywidgets`` widget because ``_wrap_temporal`` puts it in a ``VBox``, whose
    ``children`` trait rejects anything that is not a ``Widget``.

    Returns:
        A widget instance whose ``filters`` list records every ``(layer_id, expression)`` applied.
    """
    ipywidgets = pytest.importorskip("ipywidgets")

    class _RecordingWidget(ipywidgets.Output):
        """An ``Output`` widget that records ``set_filter`` calls instead of talking to MapLibre."""

        def __init__(self):
            """Start with no recorded filters."""
            super().__init__()
            self.filters = []

        def set_filter(self, layer_id, expression):
            """Record a ``(layer_id, expression)`` filter application.

            Args:
                layer_id: The MapLibre layer the filter targets.
                expression: The MapLibre filter expression.
            """
            self.filters.append((layer_id, expression))

    return _RecordingWidget()


class TestRequireFeatures:
    """Tests for ``TemporalMixin._require_features`` — the vector-input guard, exercised directly."""

    @pytest.mark.parametrize(
        "value, expected_name",
        [
            (object(), "object"),
            ("a string", "str"),
            (42, "int"),
            ({"time": [1]}, "dict"),
            (None, "NoneType"),
        ],
    )
    def test_rejects_anything_without_a_geometry(self, value, expected_name):
        """Every non-vector input is rejected, and the message names its concrete type.

        Args:
            value: The non-vector input handed to the guard.
            expected_name: The type name the error message must quote.

        Test scenario:
            The guard keys off ``geometry`` alone, so any object lacking it — including the ones that would
            otherwise reach the slider and fail obscurely — raises ``TypeError`` naming what arrived.
        """
        with pytest.raises(TypeError) as excinfo:
            WebMap._require_features(value, "timeslider")
        assert expected_name in str(excinfo.value), (
            f"error should name the rejected type {expected_name!r}, got: {excinfo.value}"
        )

    def test_accepts_anything_exposing_a_geometry(self):
        """A duck-typed vector layer passes silently.

        Test scenario:
            The guard is structural, not nominal — an object with a ``geometry`` attribute is accepted
            without importing geopandas or pyramids.
        """
        stub = type("VectorStub", (), {"geometry": ()})()
        assert WebMap._require_features(stub, "timeslider") is None, "a vector-like input must pass the guard"

    def test_quotes_the_calling_method_name(self):
        """The ``method`` argument is interpolated so the error names the builder the caller invoked.

        Test scenario:
            Mirrors ``BigDataMixin._require_points``, which quotes its caller the same way.
        """
        with pytest.raises(TypeError, match=r"^somebuilder\(\) renders"):
            WebMap._require_features(object(), "somebuilder")

    def test_points_at_both_raster_capable_tiers(self):
        """The message routes the caller to the tiers that do take a raster time stack.

        Test scenario:
            The raster/vector split is the thing this error exists to document, so both alternatives must
            be named: ``Map.animate`` (matplotlib) and ``InteractiveMap.timecube`` (interactive).
        """
        with pytest.raises(TypeError) as excinfo:
            WebMap._require_features(object(), "timeslider")
        message = str(excinfo.value)
        assert "Map.animate" in message, f"matplotlib alternative missing from: {message}"
        assert "InteractiveMap.timecube" in message, f"interactive alternative missing from: {message}"


class TestTimeSliderGeometryRouting:
    """Tests for the geometry-dependent builder routing inside ``timeslider``."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_polygons_without_column_take_the_plain_fill_path(self, timed_polygons, mocker):
        """Polygon input and no ``column`` renders plain fills, not a choropleth.

        Test scenario:
            The ``elif is_polygon`` branch — ``polygons`` is called with ``big=False`` so the layer keeps a
            per-feature id the slider can filter, and ``choropleth`` is not used.
        """
        polygons = mocker.spy(WebMap, "polygons")
        choropleth = mocker.spy(WebMap, "choropleth")
        WebMap().timeslider(timed_polygons, kdim="time")
        assert polygons.call_count == 1, "polygon input without a column must render plain fills"
        assert choropleth.call_count == 0, "no column means no choropleth"
        assert polygons.call_args.kwargs["big"] is False, "the temporal layer must stay filterable"

    def test_polygons_with_column_take_the_choropleth_path(self, timed_polygons, mocker):
        """Polygon input plus a ``column`` renders a graduated choropleth.

        Test scenario:
            The ``is_polygon and column is not None`` branch, with the classification kwargs forwarded.
        """
        choropleth = mocker.spy(WebMap, "choropleth")
        WebMap().timeslider(timed_polygons, kdim="time", column="pop", k=3, cmap="magma")
        assert choropleth.call_count == 1, "a column on polygons must render a choropleth"
        assert choropleth.call_args.kwargs["column"] == "pop", "the value column must be forwarded"
        assert choropleth.call_args.kwargs["k"] == 3, "the class count must be forwarded"
        assert choropleth.call_args.kwargs["cmap"] == "magma", "the colormap must be forwarded"

    def test_point_input_takes_the_circle_path(self, timed_points, mocker):
        """Non-polygon input renders circles via ``points``.

        Test scenario:
            The ``else`` branch — point geometries fall through to ``points`` with ``big=False``.
        """
        points = mocker.spy(WebMap, "points")
        WebMap().timeslider(timed_points, kdim="time", column="pop")
        assert points.call_count == 1, "point input must render circles"
        assert points.call_args.kwargs["big"] is False, "the temporal layer must stay filterable"

    def test_mixed_geometry_is_not_treated_as_polygonal(self, timed_polygons, timed_points, mocker):
        """A layer mixing polygons and points is not routed to the polygon path.

        Test scenario:
            ``is_polygon`` is a subset test over the distinct geometry types, so any non-polygon member
            makes the whole layer take the ``points`` branch.
        """
        pandas = pytest.importorskip("pandas")
        gpd = pytest.importorskip("geopandas")
        stacked = pandas.concat([timed_polygons, timed_points], ignore_index=True)
        mixed = gpd.GeoDataFrame(stacked, geometry=stacked.geometry, crs=4326)
        points = mocker.spy(WebMap, "points")
        polygons = mocker.spy(WebMap, "polygons")
        WebMap().timeslider(mixed, kdim="time")
        assert points.call_count == 1, "mixed geometry must not take the polygon path"
        assert polygons.call_count == 0, "mixed geometry is not polygonal"

    def test_returns_self_for_chaining(self, timed_polygons):
        """``timeslider`` returns the map so builders can be chained.

        Test scenario:
            The documented contract — the return value is the same object, not a copy.
        """
        m = WebMap()
        assert m.timeslider(timed_polygons, kdim="time") is m, "timeslider must return the same map"

    def test_records_sorted_distinct_times_and_the_layer_id(self, timed_polygons):
        """The recorded temporal config carries the kdim, the layer id and sorted distinct times.

        Test scenario:
            Six features over three time steps collapse to three sorted slider stops.
        """
        m = WebMap().timeslider(timed_polygons, kdim="time")
        assert m._temporal["kdim"] == "time", "the scrubbed field must be recorded"
        assert m._temporal["times"] == [2000, 2010, 2020], "times must be distinct and sorted"
        assert m._temporal["layer_id"] == m._last_layer_id, "the slider must target the layer just built"

    def test_a_non_default_kdim_is_honoured(self, timed_polygons):
        """A caller-supplied ``kdim`` other than ``"time"`` drives the slider.

        Test scenario:
            The column is renamed to ``year``; the slider must scrub that field instead of the default.
        """
        renamed = timed_polygons.rename(columns={"time": "year"})
        m = WebMap().timeslider(renamed, kdim="year")
        assert m._temporal["kdim"] == "year", "the caller's time field must be used"
        assert m._temporal_times() == [2000, 2010, 2020], "slider stops come from the renamed field"


class TestWrapTemporal:
    """Tests for ``TemporalMixin._wrap_temporal`` — the slider composite and its filter wiring."""

    @pytest.fixture()
    def configured(self):
        """A map with a temporal config set by hand (no engine needed)."""
        m = WebMap()
        m._temporal = {"layer_id": "layer-1", "kdim": "time", "times": [2000, 2010, 2020]}
        return m

    def test_builds_a_vbox_of_slider_then_map(self, configured, fake_widget):
        """The composite is a ``VBox`` holding the slider above the map widget.

        Test scenario:
            Child order matters — the slider is rendered above the map it drives.
        """
        ipywidgets = pytest.importorskip("ipywidgets")
        composite = configured._wrap_temporal(fake_widget)
        assert isinstance(composite, ipywidgets.VBox), f"expected a VBox, got {type(composite).__name__}"
        slider, mapped = composite.children
        assert isinstance(slider, ipywidgets.SelectionSlider), "the first child must be the slider"
        assert mapped is fake_widget, "the second child must be the map widget itself"

    def test_slider_is_labelled_with_the_time_field(self, configured, fake_widget):
        """The slider description is the scrubbed field name.

        Test scenario:
            A reader of the notebook has to know which attribute the slider scrubs.
        """
        pytest.importorskip("ipywidgets")
        slider, _ = configured._wrap_temporal(fake_widget).children
        assert slider.description == "time", f"expected the kdim as label, got {slider.description!r}"
        assert [label for label, _value in slider.options] == ["2000", "2010", "2020"], (
            "one slider stop per distinct time step, labelled by its value"
        )

    def test_first_time_step_is_filtered_in_immediately(self, configured, fake_widget):
        """Building the composite applies the filter for the first time step.

        Test scenario:
            Without an initial filter the map would open showing every frame at once.
        """
        configured._wrap_temporal(fake_widget)
        assert fake_widget.filters == [("layer-1", ["==", ["get", "time"], 2000])], (
            f"expected exactly the first-step filter, got {fake_widget.filters}"
        )

    def test_moving_the_slider_refilters_the_layer(self, configured, fake_widget):
        """The observer pushes a new MapLibre filter for the selected step.

        Test scenario:
            Setting ``slider.value`` fires the ``value`` observer, which must call ``set_filter`` with the
            equality expression for the new step.
        """
        slider, _ = configured._wrap_temporal(fake_widget).children
        slider.value = 2020
        assert fake_widget.filters[-1] == ("layer-1", ["==", ["get", "time"], 2020]), (
            f"the slider must refilter to the selected step, got {fake_widget.filters[-1]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "ipywidgets.SelectionSlider rejects empty options, so an empty series raises TraitError at "
            "slider construction - before the `if times:` guard below it can take effect. Reachable via "
            "timeslider(empty_gdf).render(). Tracked separately from the #152 guard."
        ),
    )
    def test_no_initial_filter_when_there_are_no_time_steps(self, fake_widget):
        """An empty series should yield a slider with no stops and apply no filter.

        Args:
            fake_widget: The recording widget fixture.

        Test scenario:
            The ``if times:`` guard exists for this case, but the ``SelectionSlider`` built just above it
            raises first, so the guard is currently unreachable and an empty layer crashes on render.
        """
        m = WebMap()
        m._temporal = {"layer_id": "layer-1", "kdim": "time", "times": []}
        composite = m._wrap_temporal(fake_widget)
        assert fake_widget.filters == [], "an empty series must not apply a filter"
        assert composite.children[0].options == (), "an empty series yields a slider with no stops"


class TestTemporalTimes:
    """Tests for ``TemporalMixin._temporal_times`` — the read-only view of the slider stops."""

    def test_returns_the_recorded_steps(self):
        """The recorded time steps are returned in order.

        Test scenario:
            Reads straight from the config ``timeslider`` wrote, without needing the engine.
        """
        m = WebMap()
        m._temporal = {"layer_id": "layer-1", "kdim": "time", "times": [1, 2, 3]}
        assert m._temporal_times() == [1, 2, 3], "the recorded steps must be returned as-is"

    def test_returns_a_copy_not_the_internal_list(self):
        """Mutating the returned list must not corrupt the slider config.

        Test scenario:
            The method wraps the stored sequence in ``list(...)``; callers get a snapshot they own.
        """
        m = WebMap()
        m._temporal = {"layer_id": "layer-1", "kdim": "time", "times": [1, 2, 3]}
        returned = m._temporal_times()
        returned.append(999)
        assert m._temporal["times"] == [1, 2, 3], f"internal state was mutated: {m._temporal['times']}"
