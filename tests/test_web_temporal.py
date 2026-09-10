"""DW.5 — web-tier time-slider (``timeslider``).

The bare state helper (``_temporal_times``) is tested without the engine; the slider build (which constructs
maplibre + ipywidgets) ``importorskip``s maplibre. Both halves of recipe W6 are covered: the vector layer
filtered per time step, and the ``DatasetCollection`` stack whose members are swapped by visibility.
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


@pytest.fixture()
def timed_points():
    """Six points tagged with one of three time steps and a ``pop`` value."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(i, i) for i in range(6)]
    return gpd.GeoDataFrame(
        {
            "time": [2000, 2000, 2010, 2010, 2020, 2020],
            "pop": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        geometry=geoms,
        crs=4326,
    )


@pytest.fixture()
def raster_stack(tmp_path):
    """A 3-member ``DatasetCollection``; member ``k`` spans ``[10k, 10k + 2.4]`` so the stack spans 0-22.4."""
    pytest.importorskip("pyramids")
    import numpy as np
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset
    from pyramids.dataset.collection import DatasetCollection

    geo_ref = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.02, epsg=4326)
    paths = []
    for step in range(3):
        _, xx = np.mgrid[0:20, 0:25]
        values = (10.0 * step + xx / 10.0).astype("float32")
        path = tmp_path / f"step{step}.tif"
        Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0).to_file(
            str(path)
        )
        paths.append(str(path))
    return DatasetCollection.from_files(paths)


@pytest.fixture()
def fake_widget():
    """A recording stand-in for the MapLibre ``MapWidget``, usable without the ``web`` extra.

    It subclasses a real ``ipywidgets`` widget because ``_wrap_temporal`` puts it in a ``VBox``, whose
    ``children`` trait rejects anything that is not a ``Widget``.

    Returns:
        A widget whose ``filters`` and ``visibility`` lists record what the slider applied.
    """
    ipywidgets = pytest.importorskip("ipywidgets")

    class _RecordingWidget(ipywidgets.Output):
        """An ``Output`` widget that records slider calls instead of talking to MapLibre."""

        def __init__(self):
            """Start with nothing recorded."""
            super().__init__()
            self.filters = []
            self.visibility = []

        def set_filter(self, layer_id, filter_):
            """Record a filter application (vector mode)."""
            self.filters.append((layer_id, filter_))

        def set_visibility(self, layer_id, visible=True):
            """Record a visibility change (raster mode)."""
            self.visibility.append((layer_id, visible))

    return _RecordingWidget()


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

        composite = (
            WebMap().timeslider(timed_polygons, kdim="time", column="pop").render()
        )
        assert isinstance(composite, ipywidgets.VBox)
        slider, _map = composite.children
        assert isinstance(slider, ipywidgets.SelectionSlider)
        assert len(slider.options) == 3, "one slider stop per distinct time step"

    def test_large_polygon_series_stays_filterable(self, timed_polygons):
        """A large temporal polygon set must keep a per-feature layer id for the filter (M3)."""
        m = WebMap()
        m.big_data_threshold = (
            2  # 6 polygons > 2; must NOT auto-route to deck (no layer id to filter)
        )
        m.timeslider(timed_polygons, kdim="time")
        assert m._deck_layers is None, "temporal layers must not auto-route to deck.gl"
        assert m._temporal["layer_id"] is not None, (
            "the slider needs a filterable layer id"
        )

    def test_render_without_slider_is_bare_map(self, timed_polygons):
        from maplibre.ipywidget import MapWidget

        m = WebMap().choropleth(timed_polygons, column="pop")
        assert isinstance(m.render(), MapWidget), (
            "no timeslider → bare map, not a composite"
        )

    def test_save_writes_a_file(self, tmp_path, timed_polygons):
        out = tmp_path / "temporal.html"
        WebMap().timeslider(timed_polygons, kdim="time", column="pop").save(str(out))
        assert out.stat().st_size > 1_000


class TestTimeSliderRejectsUnsupportedInput:
    """Input that is neither a vector layer nor a raster stack is turned away with an actionable error.

    A pyramids raster exposes ``columns`` as an ``int`` (the grid width in cells), so before the guard the
    attribute-membership test raised ``TypeError: argument of type 'int' is not iterable`` — a message with
    no hint of what ``timeslider`` accepts. A ``DatasetCollection`` is *not* covered here: it is a supported
    input (see ``TestTimeSliderRasterStack``). A single ``Dataset`` has no time dimension, so it stays a
    rejection, and the message routes it to ``add_raster``.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_single_dataset_is_rejected(self, dataset):
        with pytest.raises(TypeError, match=r"timeslider\(\) does not take a raster"):
            WebMap().timeslider(dataset)

    def test_message_names_the_input_and_the_single_raster_builder(self, dataset):
        """The error has to say what arrived and where a lone raster belongs instead."""
        with pytest.raises(TypeError) as excinfo:
            WebMap().timeslider(dataset)
        message = str(excinfo.value)
        assert type(dataset).__name__ in message, "the rejected type is not named"
        assert "add_raster" in message, "the single-raster builder is not named"
        assert "DatasetCollection" in message, "the supported stack form is not named"

    def test_guard_runs_before_the_reprojection(self, dataset, monkeypatch):
        """A raster in another CRS must not be warped on its way to being rejected."""
        assert dataset.epsg != 4326, (
            "fixture must not already be in the display CRS for this to bite"
        )
        warps = []
        monkeypatch.setattr(
            type(dataset), "to_crs", lambda self, *a, **k: warps.append(a)
        )
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


class TestRequireVector:
    """Tests for ``TemporalMixin._require_vector`` — the input guard, exercised directly."""

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
            The guard keys off ``geometry`` alone, so any object lacking it raises ``TypeError`` naming
            what arrived.
        """
        with pytest.raises(TypeError) as excinfo:
            WebMap._require_vector(value, "timeslider")
        assert expected_name in str(excinfo.value), (
            f"error should name the rejected type {expected_name!r}, got: {excinfo.value}"
        )

    def test_accepts_anything_exposing_a_geometry(self):
        """A duck-typed vector layer passes without raising.

        Test scenario:
            The guard is structural, not nominal — an object with a ``geometry`` attribute is accepted
            without importing geopandas or pyramids.
        """
        stub = type("VectorStub", (), {"geometry": ()})()
        WebMap._require_vector(stub, "timeslider")

    def test_quotes_the_calling_method_name(self):
        """The ``method`` argument is interpolated so the error names the builder the caller invoked.

        Test scenario:
            Mirrors ``BigDataMixin._require_points``, which quotes its caller the same way.
        """
        with pytest.raises(TypeError, match=r"^somebuilder\(\) needs"):
            WebMap._require_vector(object(), "somebuilder")

    def test_a_table_without_an_active_geometry_is_told_to_set_geometry(self):
        """A GeoDataFrame whose geometry was never activated is not misreported as a raster.

        Test scenario:
            ``hasattr(gdf, "geometry")`` is False for a GeoDataFrame built without ``geometry=``, so a
            geometry-only check would send a perfectly good table off to the raster tier. A raster reports
            ``columns`` as an ``int``; a table reports an ``Index``, which is what separates them.
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        table = gpd.GeoDataFrame({"time": [1, 2], "geom": [Point(0, 0), Point(1, 1)]})
        with pytest.raises(TypeError) as excinfo:
            WebMap._require_vector(table, "timeslider")
        message = str(excinfo.value)
        assert "set_geometry" in message, (
            f"the fix for a geometry-less table is not named: {message}"
        )
        assert "add_raster" not in message, (
            f"a table must not be given raster advice: {message}"
        )

    def test_a_raster_still_gets_the_raster_message(self, dataset):
        """The int-``columns`` branch must not swallow the raster case it was added beside.

        Args:
            dataset: The sample pyramids raster fixture.

        Test scenario:
            A ``Dataset`` also has ``columns``, but as an ``int``, so it keeps the raster-tier advice.
        """
        with pytest.raises(TypeError) as excinfo:
            WebMap._require_vector(dataset, "timeslider")
        message = str(excinfo.value)
        assert "add_raster" in message, f"the raster branch lost its advice: {message}"
        assert "set_geometry" not in message, (
            f"a raster must not be told to set_geometry: {message}"
        )

    def test_names_both_accepted_forms(self):
        """The message documents the two inputs recipe W6 supports.

        Test scenario:
            The raster/vector split is what this error exists to explain, so both a vector layer with a
            time attribute and a ``DatasetCollection`` must be named.
        """
        with pytest.raises(TypeError) as excinfo:
            WebMap._require_vector(object(), "timeslider")
        message = str(excinfo.value)
        assert "vector layer" in message, f"vector form missing from: {message}"
        assert "DatasetCollection" in message, f"stack form missing from: {message}"


class TestTimeSliderRasterStack:
    """Tests for the ``DatasetCollection`` half of recipe W6 — one layer per member, swapped by the slider."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_registers_one_layer_per_member(self, raster_stack):
        """Each time step becomes its own image layer, so the slider has something to swap.

        Test scenario:
            A 3-member stack registers 3 distinct layer ids and 3 map layers.
        """
        m = WebMap().timeslider(raster_stack)
        ids = m._temporal["layer_ids"]
        assert len(ids) == 3, f"expected one layer per member, got {len(ids)}"
        assert len(set(ids)) == 3, f"layer ids must be unique, got {ids}"
        assert len(m.layers) == 3, (
            f"expected 3 registered map layers, got {len(m.layers)}"
        )

    def test_records_raster_mode_and_integer_steps(self, raster_stack):
        """Without labels the slider steps are the member indices, and the mode is recorded as raster.

        Test scenario:
            ``_wrap_temporal`` branches on ``mode``, so it has to be set for the stack path.
        """
        m = WebMap().timeslider(raster_stack)
        assert m._temporal["mode"] == "raster", "the stack path must record raster mode"
        assert m._temporal_times() == [0, 1, 2], (
            "unlabelled steps are the member indices"
        )

    def test_labels_replace_the_indices(self, raster_stack):
        """Caller-supplied labels drive the slider instead of the integer index.

        Test scenario:
            The plan calls for real datetimes on the slider; any unique sequence works.
        """
        m = WebMap().timeslider(
            raster_stack, labels=["Jan", "Feb", "Mar"], kdim="month"
        )
        assert m._temporal_times() == ["Jan", "Feb", "Mar"], (
            "labels must become the slider stops"
        )
        assert m._temporal["kdim"] == "month", "the slider label must be recorded"

    def test_colour_range_is_frozen_across_the_whole_stack(self, raster_stack, mocker):
        """Every member is drawn with one ``(vmin, vmax)`` so the scale is stable across frames.

        Test scenario:
            Member ``k`` spans ``[10k, 10k+2.4]``, so a per-member range would differ on every frame; the
            acceptance criterion is that it does not.
        """
        add_raster = mocker.spy(WebMap, "add_raster")
        WebMap().timeslider(raster_stack)
        limits = {
            (c.kwargs["vmin"], c.kwargs["vmax"]) for c in add_raster.call_args_list
        }
        assert len(limits) == 1, (
            f"every frame must share one colour range, got {limits}"
        )
        vmin, vmax = limits.pop()
        assert vmin == pytest.approx(0.0), f"stack minimum should be 0.0, got {vmin}"
        assert vmax == pytest.approx(22.4, abs=0.05), (
            f"stack maximum should be ~22.4, got {vmax}"
        )

    def test_explicit_clim_skips_the_whole_stack_scan(self, raster_stack, mocker):
        """An explicit ``clim`` is used verbatim and the eager global scan is not run.

        Test scenario:
            The scan reads every member, so a caller with a large stack must be able to bypass it.
        """
        scan = mocker.spy(WebMap, "_global_clim")
        add_raster = mocker.spy(WebMap, "add_raster")
        WebMap().timeslider(raster_stack, clim=(-5.0, 5.0))
        assert scan.call_count == 0, "an explicit clim must skip the global scan"
        assert {
            (c.kwargs["vmin"], c.kwargs["vmax"]) for c in add_raster.call_args_list
        } == {(-5.0, 5.0)}

    def test_band_and_cmap_reach_every_member(self, raster_stack, mocker):
        """The styling arguments are forwarded to each member's image layer.

        Test scenario:
            A stack must not silently colour frames differently from what the caller asked for.
        """
        add_raster = mocker.spy(WebMap, "add_raster")
        WebMap().timeslider(raster_stack, band=1, cmap="magma", opacity=0.5)
        assert all(c.kwargs["cmap"] == "magma" for c in add_raster.call_args_list), (
            "cmap not forwarded"
        )
        assert all(c.kwargs["band"] == 1 for c in add_raster.call_args_list), (
            "band not forwarded"
        )
        assert all(c.kwargs["opacity"] == 0.5 for c in add_raster.call_args_list), (
            "opacity not forwarded"
        )

    def test_returns_self_for_chaining(self, raster_stack):
        """The stack path is chainable like every other builder.

        Test scenario:
            The documented contract — the return value is the same map object.
        """
        m = WebMap()
        assert m.timeslider(raster_stack) is m, "timeslider must return the same map"

    def test_empty_collection_is_rejected(self):
        """A stack with no members cannot drive a slider, so it fails loudly at build time.

        Test scenario:
            Previously an empty series only blew up later, inside ipywidgets, at render time.
        """
        stub = type("EmptyCollection", (), {"datasets": []})()
        with pytest.raises(ValueError, match="at least one time step"):
            WebMap().timeslider(stub)

    @pytest.mark.parametrize(
        "labels, expected",
        [
            (["a", "b"], "labels has 2 entries"),
            (["a", "a", "b"], "must be unique"),
        ],
    )
    def test_bad_labels_are_rejected(self, raster_stack, labels, expected):
        """Labels must match the member count and be unique.

        Args:
            labels: The rejected label sequence.
            expected: A fragment the error message must contain.

        Test scenario:
            A wrong count silently mislabels frames; a duplicate collapses two slider stops into one and
            makes a frame unreachable.
        """
        with pytest.raises(ValueError, match=expected):
            WebMap().timeslider(raster_stack, labels=labels)

    def test_only_the_first_member_is_built_visible(self, raster_stack):
        """All members are registered, but only the first starts visible.

        Test scenario:
            The slider toggles from this state. Without it every frame would be piled up at once and the
            last member would simply cover the rest.
        """
        m = WebMap().timeslider(raster_stack)
        visibilities = [
            layer.layout.get("visibility") for layer in self._raster_layers(m)
        ]
        assert visibilities == ["visible", "none", "none"], (
            f"expected only the first frame visible, got {visibilities}"
        )

    def test_an_undrawable_member_aborts_before_any_layer_is_registered(self, tmp_path):
        """A member with no finite values fails the whole call, leaving the map untouched.

        Args:
            tmp_path: pytest's per-test directory, holding the two written rasters.

        Test scenario:
            ``add_raster`` raises for an all-nodata band — routine in EO, where a step can be entirely
            cloud-masked. Discovering it midway used to leave the already-added members registered.
        """
        import numpy as np
        from pyramids.base.georeference import GeoReference
        from pyramids.dataset import Dataset
        from pyramids.dataset.collection import DatasetCollection

        geo_ref = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.02, epsg=4326)
        good, empty = tmp_path / "good.tif", tmp_path / "allnodata.tif"
        Dataset.from_array(
            np.arange(500, dtype="float32").reshape(20, 25),
            geo_ref=geo_ref,
            no_data_value=-9999.0,
        ).to_file(str(good))
        Dataset.from_array(
            np.full((20, 25), -9999.0, dtype="float32"),
            geo_ref=geo_ref,
            no_data_value=-9999.0,
        ).to_file(str(empty))

        m = WebMap()
        with pytest.raises(ValueError, match="no finite values"):
            m.timeslider(DatasetCollection.from_files([str(good), str(empty)]))
        assert m.layers == [], (
            "a failed stack must not leave half its layers registered"
        )

    def test_unhashable_labels_get_the_actionable_message(self, raster_stack):
        """A list-valued label fails with the builder's own error, not a bare ``set()`` TypeError.

        Test scenario:
            The uniqueness check hashes the labels, so an unhashable one used to surface as
            ``TypeError: unhashable type: 'list'`` from deep inside the validation.
        """
        with pytest.raises(ValueError, match="hashable"):
            WebMap().timeslider(raster_stack, labels=[["a"], ["b"], ["c"]])

    def test_save_shows_exactly_one_frame(self, tmp_path, raster_stack):
        """A saved page carries no slider, so it must not show the whole stack at once.

        Test scenario:
            ``save`` serialises the map without the slider composite. Asserting only that the file is
            non-empty let a page through that inlined every frame and showed whichever landed on top.
        """
        out = tmp_path / "stack.html"
        WebMap().timeslider(raster_stack).save(str(out))
        html = out.read_text(encoding="utf-8", errors="replace")
        assert out.stat().st_size > 1_000, "the saved stack map looks empty"
        assert html.count('"visibility": "visible"') == 1, (
            "a saved stack must show exactly one frame"
        )
        assert html.count('"visibility": "none"') == 2, (
            "the other frames must be serialised hidden"
        )

    @staticmethod
    def _raster_layers(m):
        """Return the ``maplibre`` Layer objects the map registered, in add order.

        Args:
            m: The built :class:`WebMap`.

        Returns:
            The layers captured by replaying each registered ``apply`` callable against a recorder.
        """

        class _Recorder:
            def __init__(self):
                self.added = []

            def add_source(self, *args, **kwargs):
                pass

            def add_layer(self, layer):
                self.added.append(layer)

        recorder = _Recorder()
        for apply in m.layers:
            apply(recorder)
        return recorder.added


class TestTimeSliderGeometryRouting:
    """Tests for the geometry-dependent builder routing inside ``timeslider``."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_polygons_without_column_take_the_plain_fill_path(
        self, timed_polygons, mocker
    ):
        """Polygon input and no ``column`` renders plain fills, not a choropleth.

        Test scenario:
            The ``elif is_polygon`` branch — ``polygons`` is called with ``big=False`` so the layer keeps a
            per-feature id the slider can filter, and ``choropleth`` is not used.
        """
        polygons = mocker.spy(WebMap, "polygons")
        choropleth = mocker.spy(WebMap, "choropleth")
        WebMap().timeslider(timed_polygons, kdim="time")
        assert polygons.call_count == 1, (
            "polygon input without a column must render plain fills"
        )
        assert choropleth.call_count == 0, "no column means no choropleth"
        assert polygons.call_args.kwargs["big"] is False, (
            "the temporal layer must stay filterable"
        )

    def test_polygons_with_column_take_the_choropleth_path(
        self, timed_polygons, mocker
    ):
        """Polygon input plus a ``column`` renders a graduated choropleth.

        Test scenario:
            The ``is_polygon and column is not None`` branch, with the classification kwargs forwarded.
        """
        choropleth = mocker.spy(WebMap, "choropleth")
        WebMap().timeslider(
            timed_polygons, kdim="time", column="pop", k=3, cmap="magma"
        )
        assert choropleth.call_count == 1, (
            "a column on polygons must render a choropleth"
        )
        assert choropleth.call_args.kwargs["column"] == "pop", (
            "the value column must be forwarded"
        )
        assert choropleth.call_args.kwargs["k"] == 3, (
            "the class count must be forwarded"
        )
        assert choropleth.call_args.kwargs["cmap"] == "magma", (
            "the colormap must be forwarded"
        )

    def test_point_input_takes_the_circle_path(self, timed_points, mocker):
        """Non-polygon input renders circles via ``points``.

        Test scenario:
            The ``else`` branch — point geometries fall through to ``points`` with ``big=False``.
        """
        points = mocker.spy(WebMap, "points")
        WebMap().timeslider(timed_points, kdim="time", column="pop")
        assert points.call_count == 1, "point input must render circles"
        assert points.call_args.kwargs["big"] is False, (
            "the temporal layer must stay filterable"
        )

    def test_mixed_geometry_currently_falls_through_to_circles(
        self, timed_polygons, timed_points, mocker
    ):
        """Documents current routing for a mixed-geometry layer: it is drawn as circles, not fills.

        Test scenario:
            ``is_polygon`` is a subset test over the distinct geometry types, so one non-polygon member
            sends the whole layer down the ``points`` branch. This pins today's behaviour; it is not an
            endorsement — ``BigDataMixin._require_points`` rejects mixed input outright, so the two
            builders disagree about what mixed geometry means.
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
        assert m.timeslider(timed_polygons, kdim="time") is m, (
            "timeslider must return the same map"
        )

    def test_records_the_vector_config_shape(self, timed_polygons):
        """The recorded config carries the mode, the scrubbed field and the layer the slider filters.

        Test scenario:
            Complements ``test_records_distinct_time_steps``, which covers the step values themselves.
        """
        m = WebMap().timeslider(timed_polygons, kdim="time")
        assert m._temporal["mode"] == "vector", (
            "the vector path must record vector mode"
        )
        assert m._temporal["kdim"] == "time", "the scrubbed field must be recorded"
        assert m._temporal["layer_id"] == m._last_layer_id, (
            "the slider must target the layer just built"
        )

    def test_a_non_default_kdim_is_honoured(self, timed_polygons):
        """A caller-supplied ``kdim`` other than ``"time"`` drives the slider.

        Test scenario:
            The column is renamed to ``year``; the slider must scrub that field instead of the default.
        """
        renamed = timed_polygons.rename(columns={"time": "year"})
        m = WebMap().timeslider(renamed, kdim="year")
        assert m._temporal["kdim"] == "year", "the caller's time field must be used"
        assert m._temporal_times() == [2000, 2010, 2020], (
            "slider stops come from the renamed field"
        )

    def test_a_series_with_no_time_steps_is_rejected(self, timed_polygons):
        """An empty series fails at build time rather than inside ipywidgets at render time.

        Test scenario:
            ``SelectionSlider`` rejects empty options, so without this guard the failure surfaced as an
            opaque ``TraitError`` only once the map was rendered.
        """
        with pytest.raises(ValueError, match="at least one time step"):
            WebMap().timeslider(timed_polygons.iloc[0:0], kdim="time")


class TestWrapTemporal:
    """Tests for ``TemporalMixin._wrap_temporal`` — the slider composite and its per-mode wiring."""

    @pytest.fixture()
    def configured(self):
        """A map with a vector temporal config set by hand (no engine needed)."""
        m = WebMap()
        m._temporal = {
            "mode": "vector",
            "layer_id": "layer-1",
            "kdim": "time",
            "times": [2000, 2010, 2020],
        }
        return m

    @pytest.fixture()
    def configured_stack(self):
        """A map with a raster temporal config set by hand (no engine needed)."""
        m = WebMap()
        m._temporal = {
            "mode": "raster",
            "layer_id": "img-0",
            "layer_ids": ["img-0", "img-1", "img-2"],
            "kdim": "time",
            "times": [0, 1, 2],
        }
        return m

    def test_builds_a_vbox_of_slider_then_map(self, configured, fake_widget):
        """The composite is a ``VBox`` holding the slider above the map widget.

        Test scenario:
            Child order matters — the slider is rendered above the map it drives.
        """
        ipywidgets = pytest.importorskip("ipywidgets")
        composite = configured._wrap_temporal(fake_widget)
        assert isinstance(composite, ipywidgets.VBox), (
            f"expected a VBox, got {type(composite).__name__}"
        )
        slider, mapped = composite.children
        assert isinstance(slider, ipywidgets.SelectionSlider), (
            "the first child must be the slider"
        )
        assert mapped is fake_widget, "the second child must be the map widget itself"

    def test_slider_is_labelled_with_the_time_field(self, configured, fake_widget):
        """The slider description is the scrubbed field name.

        Test scenario:
            A reader of the notebook has to know which attribute the slider scrubs.
        """
        slider, _ = configured._wrap_temporal(fake_widget).children
        assert slider.description == "time", (
            f"expected the kdim as label, got {slider.description!r}"
        )
        assert [label for label, _value in slider.options] == [
            "2000",
            "2010",
            "2020",
        ], "one slider stop per distinct time step, labelled by its value"

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

    def test_opening_on_the_first_frame_needs_no_visibility_calls(
        self, configured_stack, fake_widget
    ):
        """The layers are built with the first frame already visible, so wiring changes nothing.

        Test scenario:
            ``_timeslider_stack`` sets ``layout.visibility`` per layer at build time, which is what makes
            a *saved* page (no slider) show one frame. Re-asserting it here would be redundant work on
            every render.
        """
        configured_stack._wrap_temporal(fake_widget)
        assert fake_widget.visibility == [], (
            "opening on frame 0 must not re-toggle anything"
        )
        assert fake_widget.filters == [], (
            "the raster mode must not set MapLibre filters"
        )

    def test_moving_the_slider_swaps_only_the_two_affected_layers(
        self, configured_stack, fake_widget
    ):
        """The observer hides the outgoing frame and shows the incoming one, and touches nothing else.

        Test scenario:
            This is the "swaps the active source" behaviour recipe W6 specifies. Toggling all N layers
            would send O(stack) widget messages per slider step.
        """
        slider, _ = configured_stack._wrap_temporal(fake_widget).children
        fake_widget.visibility.clear()
        slider.value = 2
        assert fake_widget.visibility == [("img-0", False), ("img-2", True)], (
            f"expected only the outgoing and incoming frames to change, got {fake_widget.visibility}"
        )

    def test_stepping_again_hides_the_previously_shown_frame(
        self, configured_stack, fake_widget
    ):
        """The tracked "currently showing" frame follows the slider across successive moves.

        Test scenario:
            A stale cursor would hide the wrong layer on the second move and leave two frames visible.
        """
        slider, _ = configured_stack._wrap_temporal(fake_widget).children
        slider.value = 2
        fake_widget.visibility.clear()
        slider.value = 1
        assert fake_widget.visibility == [("img-2", False), ("img-1", True)], (
            f"the outgoing frame must be the one actually showing, got {fake_widget.visibility}"
        )

    def test_the_fake_widget_matches_the_real_widget_api(self, fake_widget):
        """The recording widget's methods match ``MapWidget``'s, so upstream drift cannot hide behind it.

        Test scenario:
            ``fake_widget`` stands in for the engine; if maplibre renamed a parameter these tests would
            still pass while the real map broke. Comparing the signatures closes that gap.
        """
        pytest.importorskip("maplibre")
        import inspect

        from maplibre.ipywidget import MapWidget

        expected = {
            "set_filter": ["self", "layer_id", "filter_"],
            "set_visibility": ["self", "layer_id", "visible"],
        }
        for name, params in expected.items():
            real = list(inspect.signature(getattr(MapWidget, name)).parameters)
            assert real == params, (
                f"{name} signature drifted: expected {params}, got {real}"
            )
            fake = list(inspect.signature(getattr(fake_widget, name)).parameters)
            assert fake == params[1:], (
                f"the recording widget's {name} no longer matches: {fake}"
            )


class TestTemporalTimes:
    """Tests for ``TemporalMixin._temporal_times`` — the read-only view of the slider stops."""

    def test_returns_the_recorded_steps(self):
        """The recorded time steps are returned in order.

        Test scenario:
            Reads straight from the config ``timeslider`` wrote, without needing the engine.
        """
        m = WebMap()
        m._temporal = {"layer_id": "layer-1", "kdim": "time", "times": [1, 2, 3]}
        assert m._temporal_times() == [1, 2, 3], (
            "the recorded steps must be returned as-is"
        )

    def test_returns_a_copy_not_the_internal_list(self):
        """Mutating the returned list must not corrupt the slider config.

        Test scenario:
            The method wraps the stored sequence in ``list(...)``; callers get a snapshot they own.
        """
        m = WebMap()
        m._temporal = {"layer_id": "layer-1", "kdim": "time", "times": [1, 2, 3]}
        returned = m._temporal_times()
        returned.append(999)
        assert m._temporal["times"] == [1, 2, 3], (
            f"internal state was mutated: {m._temporal['times']}"
        )


class TestASavedTemporalMapIsSteppable:
    """#187 — `render` wraps the map in a slider that only exists in a live kernel."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _payload(html):
        """Return the page's call payload — what this map does, not what the library contains.

        Args:
            html: A page from ``to_html``.

        Returns:
            The substring from ``var data =`` to the end of the page.
        """
        marker = html.rfind("var data = ")
        assert marker != -1, "the exported page carries no call payload"
        return html[marker:]

    def test_every_step_is_reachable_in_the_saved_page(self, raster_stack):
        """A shared page used to show one frozen frame with no way to move.

        Args:
            raster_stack: The 3-member collection fixture.

        Test scenario:
            The frames are already in the page, one layer each — a switcher over them is what makes them
            reachable without a kernel.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().timeslider(raster_stack, labels=["2020", "2021", "2022"])
        payload = self._payload(m.to_html())
        assert "LayerSwitcherControl" in payload, (
            "no way to change step in the saved page"
        )
        for layer_id in m._temporal["layer_ids"]:
            assert layer_id in payload, f"step layer {layer_id} is unreachable"

    def test_the_steps_are_labelled_with_their_times(self, raster_stack):
        """A switch listing raster-7/raster-9 tells a viewer nothing about which year it is.

        Args:
            raster_stack: The 3-member collection fixture.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().timeslider(raster_stack, labels=["2020", "2021", "2022"])
        m.to_html()
        assert [label for _, label in m._layer_index] == ["2020", "2021", "2022"]

    def test_building_twice_does_not_stack_controls(self, raster_stack):
        """`render` and `save` both build the widget, and a page with three switchers is a bug.

        Args:
            raster_stack: The 3-member collection fixture.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().timeslider(raster_stack)
        m.to_html()
        assert self._payload(m.to_html()).count("LayerSwitcherControl") == 1

    def test_a_map_with_no_time_dimension_gets_no_control(self, dataset):
        """The switcher is for steps; an ordinary raster map must not sprout one.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        from digitalearth.web import WebMap

        payload = self._payload(WebMap().basemap().add_raster(dataset).to_html())
        assert "LayerSwitcherControl" not in payload
