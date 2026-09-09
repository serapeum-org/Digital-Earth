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
