"""The web tier's half of the cross-backend contract (Wave 0, batch B).

One class per contract point, each proving the tier behaves the way every other tier does — and, for each
rename, that the old spelling still works **and** warns, so no caller is broken in this batch, **and**
that passing both spellings at once is a ``TypeError`` naming both (the shared
:func:`~digitalearth.base.deprecation.renamed_parameter` rule; this tier used to prefer the old one
silently):

* **C1** ``save()`` returns :class:`pathlib.Path`;
* **C2** ``animate(fps=3.0)`` is canonical, ``to_gif`` and ``duration=`` deprecated (``duration`` converted);
* **C3** ``size`` is marker size, ``text_size`` is text size, ``radius``/``point_size`` deprecated;
* **C4** ``scheme=None`` (continuous) with ``k=5`` wherever a layer classifies;
* **C5** ``cmap=None`` resolves through ``auto_style``;
* **C6** ``auto_style``'s ``levels`` and ``units`` are consumed, never guessed;
* **C7** an unplaceable layer is skipped with a warning, and raises only under ``strict=True``;
* **C8** ``big_data_threshold`` as an instance attribute **and** a per-call override;
* **C9** the basemap default comes from ``base/basemaps.py``;
* **C10** a keyed provider's coverage reaches the MapLibre source's ``bounds``;
* **C13** the display CRS is EPSG:4326 and anything else is refused.

Engine-dependent tests ``importorskip`` maplibre, so this file runs whole in the ``web`` env
(``pixi run -e web test-web``).
"""

import inspect
import pathlib
import warnings

import numpy as np
import pytest

from digitalearth.base.crs import OffLimbError
from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.source import Source
from digitalearth.web import WebMap
from digitalearth.web.base import DEFAULT_BIG_DATA_THRESHOLD, DISPLAY_CRS
from digitalearth.web.decoration import DEFAULT_BASEMAP_PROVIDER
from digitalearth.web.export import DEFAULT_FPS


def _source(variable: str) -> Source:
    """Build a minimal display-CRS source whose variable drives the autostyle lookup.

    Args:
        variable: The variable name recorded in the source metadata.

    Returns:
        A 2x2 :class:`Source` carrying that variable.
    """
    return Source(
        DimensionInfo(np.zeros((2, 2)), "z"),
        DimensionInfo(np.array([0.0, 1.0]), "x"),
        DimensionInfo(np.array([1.0, 0.0]), "y"),
        crs=DISPLAY_CRS,
        metadata={"variable": variable},
    )


def _points_frame(count: int):
    """Build a small point frame in lon/lat.

    Args:
        count: How many points to place.

    Returns:
        A GeoDataFrame of ``count`` points with a ``value`` column.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"value": [float(i) for i in range(count)]},
        geometry=[Point(float(i) / 10.0, 0.0) for i in range(count)],
        crs=DISPLAY_CRS,
    )


def _labelled_frame(count: int):
    """Build a point frame carrying a ``name`` column, ready to label.

    Args:
        count: How many points to place.

    Returns:
        A GeoDataFrame of ``count`` points with ``value`` and ``name`` columns.
    """
    frame = _points_frame(count)
    frame["name"] = [f"p{index}" for index in range(count)]
    return frame


def _polygon_frame():
    """Build a three-polygon frame with a spread ``pop`` column.

    Returns:
        A GeoDataFrame of triangles in lon/lat.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"pop": [1.0, 5.0, 9.0]},
        geometry=[Polygon([(i, 0), (i + 1, 0), (i + 0.5, 1)]) for i in range(3)],
        crs=DISPLAY_CRS,
    )


def _paint(web_map, key):
    """Return a paint property from the map's most recently registered layer.

    Args:
        web_map: The map whose layers are replayed.
        key: The MapLibre paint property to read.

    Returns:
        The property's value on the last registered layer.
    """
    return _last_layer(web_map).paint[key]


def _layout(web_map, key):
    """Return a layout property from the map's most recently registered layer.

    Args:
        web_map: The map whose layers are replayed.
        key: The MapLibre layout property to read.

    Returns:
        The property's value on the last registered layer.
    """
    return _last_layer(web_map).layout[key]


def _last_layer(web_map):
    """Replay the map's layers against a recorder and return the last MapLibre layer registered.

    Args:
        web_map: The map whose layers are replayed.

    Returns:
        The last ``maplibre`` ``Layer`` object the registry built.
    """
    built = []

    class Recorder:
        """Stands in for the MapLibre widget, recording the layers a registry entry builds."""

        def add_source(self, src_id, source):
            """Ignore the source; only the layer is under test."""

        def add_layer(self, layer):
            """Record the built layer."""
            built.append(layer)

    for apply in web_map.layers:
        apply(Recorder())
    assert built, "no layer was registered"
    return built[-1]


def _sources_of(web_map) -> list:
    """Replay a map's registered layers against a recorder and return the source specs.

    Args:
        web_map: The map whose layers are replayed.

    Returns:
        Every source spec the layers registered, in registration order.
    """
    specs = []

    class Recorder:
        """Stands in for the MapLibre widget, recording what a layer registers."""

        def add_source(self, src_id, source):
            """Record the source spec."""
            specs.append(source)

        def add_layer(self, layer):
            """Ignore the layer; only the source is under test."""

    for apply in web_map.layers:
        apply(Recorder())
    return specs


class TestC1SaveReturnsAPath:
    """Every tier's ``save`` hands back the :class:`pathlib.Path` it wrote."""

    def test_saving_html_returns_the_path_object(self, tmp_path):
        """A string return made a caller re-wrap it before they could stat or read it.

        Args:
            tmp_path: pytest's per-test directory.
        """
        pytest.importorskip("maplibre")
        out = tmp_path / "m.html"
        written = WebMap().basemap().save(str(out))
        assert isinstance(written, pathlib.Path), type(written)
        assert written == out and written.stat().st_size > 1_000


class TestC2FrameRateIsFps:
    """``fps`` is the rate on every tier; ``to_gif`` and ``duration=`` survive as deprecated spellings."""

    def test_the_default_frame_rate_is_the_shared_one(self, monkeypatch, tmp_path):
        """One default across the package, so a series animates at one speed whoever renders it.

        Args:
            monkeypatch: pytest's patcher, standing in for the GIF encoder.
            tmp_path: pytest's per-test directory.

        Test scenario:
            The signature default is the ``None`` "not passed" sentinel the deprecated ``duration=`` is
            resolved against (the same shape the 3-D tier uses), so the effective default is asserted where
            it is actually applied: at the encoder.
        """
        assert DEFAULT_FPS == 3.0
        assert inspect.signature(WebMap.animate).parameters["fps"].default is None, (
            "fps must default to the not-passed sentinel so both spellings can be told apart"
        )
        recorded = {}

        def fake_write(frames, path, *, duration, loop):
            """Record the per-frame hold the encoder was handed."""
            recorded["duration"] = duration
            pathlib.Path(path).write_bytes(b"GIF89a")

        from digitalearth.web import export as web_export

        monkeypatch.setattr(web_export, "_write_gif", fake_write)
        monkeypatch.setattr(WebMap, "_temporal_frames", lambda self: [["a"], ["b"]])
        monkeypatch.setattr(
            WebMap, "_frame_png", lambda self, path, visible, title: path
        )
        WebMap().animate(str(tmp_path / "series.gif"))
        assert recorded["duration"] == pytest.approx(1.0 / DEFAULT_FPS), (
            "omitting fps must animate at the shared default rate"
        )

    def test_both_spellings_of_the_rate_at_once_is_refused(self, tmp_path):
        """``fps`` and the deprecated ``duration`` together is a ``TypeError`` naming both.

        Args:
            tmp_path: pytest's per-test directory.

        Test scenario:
            This tier used to resolve every alias silently in favour of the **old** spelling, so a caller
            who migrated to ``fps=`` but left ``duration=`` behind got the stale value and no warning that
            their new keyword was ignored. It is now the same ``TypeError`` static, interactive and 3-D
            raise, from the shared :func:`~digitalearth.base.deprecation.renamed_parameter`.
        """
        with pytest.raises(TypeError) as excinfo:
            WebMap().animate(str(tmp_path / "series.gif"), fps=4.0, duration=0.5)
        message = str(excinfo.value)
        assert "both fps= and the deprecated duration=" in message, message
        assert "pass only fps=" in message, message

    def test_duration_is_converted_to_fps_and_warns(self, monkeypatch, tmp_path):
        """``duration`` is seconds per frame, so it *converts* — reinterpreting it would double the speed.

        Args:
            monkeypatch: pytest's patcher, standing in for the GIF encoder.
            tmp_path: pytest's per-test directory.
        """
        recorded = {}

        def fake_write(frames, path, *, duration, loop):
            """Record the per-frame hold the encoder was handed."""
            recorded["duration"] = duration
            pathlib.Path(path).write_bytes(b"GIF89a")

        from digitalearth.web import export as web_export

        monkeypatch.setattr(web_export, "_write_gif", fake_write)
        monkeypatch.setattr(WebMap, "_temporal_frames", lambda self: [["a"], ["b"]])
        monkeypatch.setattr(
            WebMap, "_frame_png", lambda self, path, visible, title: path
        )

        out = tmp_path / "series.gif"
        with pytest.warns(DeprecationWarning, match="duration= is deprecated"):
            WebMap().animate(str(out), duration=0.5)
        assert recorded["duration"] == pytest.approx(0.5), (
            "duration=0.5 must mean fps=2, i.e. the same half-second hold it always did"
        )

    def test_fps_sets_the_hold_directly(self, monkeypatch, tmp_path):
        """The canonical spelling is the inverse of the hold the encoder wants.

        Args:
            monkeypatch: pytest's patcher.
            tmp_path: pytest's per-test directory.
        """
        recorded = {}

        def fake_write(frames, path, *, duration, loop):
            """Record the per-frame hold the encoder was handed."""
            recorded["duration"] = duration
            pathlib.Path(path).write_bytes(b"GIF89a")

        from digitalearth.web import export as web_export

        monkeypatch.setattr(web_export, "_write_gif", fake_write)
        monkeypatch.setattr(WebMap, "_temporal_frames", lambda self: [["a"], ["b"]])
        monkeypatch.setattr(
            WebMap, "_frame_png", lambda self, path, visible, title: path
        )

        WebMap().animate(str(tmp_path / "series.gif"), fps=4.0)
        assert recorded["duration"] == pytest.approx(0.25)

    def test_to_gif_still_works_and_warns(self, monkeypatch, tmp_path):
        """The old name keeps working for one release, and says what replaces it.

        Args:
            monkeypatch: pytest's patcher.
            tmp_path: pytest's per-test directory.
        """
        seen = {}

        def fake_animate(self, path, **kwargs):
            """Stand in for the real animation, recording that it was reached."""
            seen["path"] = path
            return pathlib.Path(path)

        monkeypatch.setattr(WebMap, "animate", fake_animate)
        out = tmp_path / "series.gif"
        with pytest.warns(DeprecationWarning, match="use WebMap.animate"):
            assert WebMap().to_gif(str(out)) == out
        assert seen["path"] == str(out)

    @pytest.mark.parametrize("kwargs", [{"fps": 0}, {"duration": 0}])
    def test_a_non_positive_rate_is_refused(self, kwargs, tmp_path):
        """A zero rate has no frame to hold, and would divide by zero on the way to the encoder.

        Args:
            kwargs: The offending rate, in either spelling.
            tmp_path: pytest's per-test directory.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="must be positive"):
                WebMap().animate(str(tmp_path / "series.gif"), **kwargs)


class TestC3SizeIsMarkerSizeAndTextSizeIsText:
    """``size`` means one thing — the visual size of a marker — so text got its own name."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip without the web extra."""
        pytest.importorskip("maplibre")

    def test_points_size_is_the_circle_radius(self):
        """``size`` reaches MapLibre's ``circle-radius``, which is what a marker's size is there."""
        m = WebMap().points(_points_frame(3), size=9.0)
        assert _paint(m, "circle-radius") == 9.0

    def test_points_radius_still_works_and_warns(self):
        """The old spelling forwards unchanged, so an existing call renders identically."""
        with pytest.warns(DeprecationWarning, match="radius= is deprecated"):
            m = WebMap().points(_points_frame(3), radius=9.0)
        assert _paint(m, "circle-radius") == 9.0

    def test_labels_take_text_size(self):
        """A label's size is text, not a marker, so it is named for what it scales."""
        frame = _points_frame(2)
        frame["name"] = ["a", "b"]
        m = WebMap().labels(frame, "name", text_size=21.0)
        assert _layout(m, "text-size") == 21.0

    def test_labels_size_still_works_and_warns(self):
        """#211 shipped ``size`` for text; it keeps working while it is deprecated."""
        frame = _points_frame(2)
        frame["name"] = ["a", "b"]
        with pytest.warns(DeprecationWarning, match="use text_size="):
            m = WebMap().labels(frame, "name", size=21.0)
        assert _layout(m, "text-size") == 21.0

    def test_text_annotation_takes_text_size(self):
        """The single-coordinate annotation is text too, and follows the same rename."""
        m = WebMap().text(4.9, 52.4, "Amsterdam", text_size=18.0)
        assert _layout(m, "text-size") == 18.0

    def test_text_annotation_size_still_works_and_warns(self):
        """And its old spelling warns rather than silently doing nothing."""
        with pytest.warns(DeprecationWarning, match="use text_size="):
            m = WebMap().text(4.9, 52.4, "Amsterdam", size=18.0)
        assert _layout(m, "text-size") == 18.0

    def test_deck_scatter_takes_size(self):
        """The GPU path a large points layer routes to takes the same word."""
        m = WebMap().deck_scatter(_points_frame(3), size=7.0)
        assert m._deck_layers[0]["getPointRadius"] == 7.0

    def test_deck_scatter_radius_still_works_and_warns(self):
        """Its old spelling forwards unchanged."""
        with pytest.warns(DeprecationWarning, match="radius= is deprecated"):
            m = WebMap().deck_scatter(_points_frame(3), radius=7.0)
        assert m._deck_layers[0]["getPointRadius"] == 7.0

    def test_point_cloud_takes_size(self):
        """A 3-D point is a marker as much as a 2-D one."""
        m = WebMap().point_cloud([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], size=4.0)
        assert m._deck_layers[0]["pointSize"] == 4.0

    def test_point_cloud_point_size_still_works_and_warns(self):
        """``point_size`` was the third spelling of one idea; it is deprecated, not dropped."""
        with pytest.warns(DeprecationWarning, match="point_size= is deprecated"):
            m = WebMap().point_cloud([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], point_size=4.0)
        assert m._deck_layers[0]["pointSize"] == 4.0

    @pytest.mark.parametrize(
        "call, new_name, old_name",
        [
            (
                lambda: WebMap().points(_points_frame(3), size=9.0, radius=7.0),
                "size",
                "radius",
            ),
            (
                lambda: WebMap().labels(
                    _labelled_frame(2), "name", text_size=21.0, size=9.0
                ),
                "text_size",
                "size",
            ),
            (
                lambda: WebMap().text(4.9, 52.4, "Amsterdam", text_size=18.0, size=9.0),
                "text_size",
                "size",
            ),
            (
                lambda: WebMap().deck_scatter(_points_frame(3), size=7.0, radius=9.0),
                "size",
                "radius",
            ),
            (
                lambda: WebMap().point_cloud(
                    [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], size=4.0, point_size=8.0
                ),
                "size",
                "point_size",
            ),
        ],
        ids=["points", "labels", "text", "deck_scatter", "point_cloud"],
    )
    def test_both_spellings_at_once_is_refused(self, call, new_name, old_name):
        """Every renamed size keyword refuses its pair, naming both and the one to keep.

        Args:
            call: A builder call that sets one parameter under both of its spellings.
            new_name: The spelling the caller should keep.
            old_name: The deprecated spelling they should drop.

        Test scenario:
            This tier used to prefer the **old** spelling whenever both were given — silently, so a caller
            who had already migrated saw their new keyword overwritten by the one they forgot to delete.
            All four tiers now raise the same ``TypeError`` from
            :func:`~digitalearth.base.deprecation.renamed_parameter`.
        """
        with pytest.raises(TypeError) as excinfo:
            call()
        message = str(excinfo.value)
        assert f"both {new_name}= and the deprecated {old_name}=" in message, message
        assert f"pass only {new_name}=" in message, message


class TestC4SchemeAndK:
    """``scheme=None`` means a continuous ramp, and ``k=5`` is the class count when a scheme is given."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip without the web extra."""
        pytest.importorskip("maplibre")

    @pytest.mark.parametrize(
        "builder", ["points", "lines", "polygons", "choropleth", "timeslider"]
    )
    def test_every_classifying_builder_defaults_to_a_continuous_ramp(self, builder):
        """The web ``choropleth`` was the odd one out at ``"quantiles"``; now the whole tier agrees.

        Args:
            builder: The builder whose signature is inspected.
        """
        params = inspect.signature(getattr(WebMap, builder)).parameters
        assert params["scheme"].default is None, builder
        assert params["k"].default == 5, builder

    def test_choropleth_without_a_scheme_is_continuous(self):
        """The default now colours by a ramp, which is what the static and interactive tiers draw."""
        m = WebMap().choropleth(_polygon_frame(), column="pop")
        assert m.last_legend["kind"] == "continuous", m.last_legend

    def test_a_named_scheme_still_classifies_into_k_classes(self):
        """Changing the default must not change what a scheme computes."""
        m = WebMap().choropleth(_polygon_frame(), column="pop", scheme="quantiles", k=3)
        assert m.last_legend["kind"] == "graduated"
        assert len(m.last_legend["colors"]) == 3, m.last_legend


class TestC5CmapResolvesThroughAutostyle:
    """``cmap=None`` is a lookup, never a bare literal in the signature."""

    @pytest.mark.parametrize("builder", ["add_raster", "contours"])
    def test_the_raster_builders_default_to_none(self, builder):
        """A string default would hide the variable's own colormap behind a generic one.

        Args:
            builder: The raster builder whose signature is inspected.
        """
        assert (
            inspect.signature(getattr(WebMap, builder)).parameters["cmap"].default
            is None
        )

    def test_a_known_variable_picks_up_its_colormap(self):
        """The same lookup the static and interactive tiers use, so one field looks the same."""
        assert WebMap()._auto_cmap(_source("t2m"), None) == "coolwarm"

    def test_an_unknown_variable_falls_back_behind_the_lookup(self):
        """The old literal moved behind ``auto_style`` rather than in front of it."""
        assert WebMap()._auto_cmap(_source("mystery"), None) == "viridis"

    def test_a_caller_supplied_colormap_always_wins(self):
        """The lookup is a default, not an override."""
        assert WebMap()._auto_cmap(_source("t2m"), "magma") == "magma"


class TestC6LevelsAndUnitsAreConsumed:
    """``auto_style`` carries more than a colormap, and the tier now reads the rest of it."""

    def test_levels_come_from_the_library_when_the_caller_gave_none(self):
        """An operational field carries the levels its community draws it with."""
        levels = WebMap()._auto_levels(_source("msl"), None)
        assert levels and levels[0] == 960, levels

    def test_caller_supplied_levels_win(self):
        """A caller who named levels gets exactly those."""
        assert WebMap()._auto_levels(_source("msl"), [1.0, 2.0]) == [1.0, 2.0]

    def test_an_unknown_variable_yields_no_levels(self):
        """A guessed set of levels would be worse than asking the caller for them."""
        assert WebMap()._auto_levels(_source("mystery"), None) is None

    def test_units_come_from_the_library_when_the_caller_gave_none(self):
        """The units hint is what lets a key say what its numbers are measured in."""
        assert WebMap()._auto_units(_source("msl"), None) == "hPa"

    def test_caller_supplied_units_win(self):
        """A caller who named the units gets exactly those, library hint or not.

        Test scenario:
            The autostyle hint fills a gap; it never overwrites a unit the caller wrote. That is the same
            rule ``_auto_levels`` and ``_auto_cmap`` follow, so a key labelled by hand keeps its label
            even for a variable the library happens to know.
        """
        assert WebMap()._auto_units(_source("msl"), "kPa") == "kPa", (
            "a caller-supplied unit must win over the library's own"
        )

    def test_an_unknown_variable_yields_no_units(self):
        """Never guess a unit: without one the label is built exactly as it was before."""
        assert WebMap()._auto_units(_source("mystery"), None) is None

    @pytest.mark.parametrize("builder", ["add_raster", "contours"])
    def test_the_raster_builders_expose_the_units_override(self, builder):
        """The caller half of ``_auto_units`` is a real argument, not dead code (review L3).

        Args:
            builder: The raster builder whose signature is inspected.

        Test scenario:
            ``_auto_cmap`` and ``_auto_levels`` each sit behind a public ``cmap=``/``levels=``; ``units``
            had no such parameter, so the branch honouring a caller's value could never be reached from
            outside the class. Both builders now take it, defaulting to ``None`` (auto-resolve).
        """
        parameter = inspect.signature(getattr(WebMap, builder)).parameters["units"]
        assert parameter.default is None, (
            f"{builder}(units=) must default to the autostyle lookup, got {parameter.default!r}"
        )
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{builder}(units=) must be keyword-only like the rest of the styling arguments"
        )

    def test_add_raster_records_the_caller_supplied_unit(self, dataset):
        """``add_raster(units=)`` reaches ``last_units``, so a key built from the band can name it.

        Args:
            dataset: The shared pyramids raster fixture.

        Test scenario:
            The autostyle hint is canonical rather than measured, so a band in something else needs a way
            to say so. Nothing consumed the caller's unit before this, because nothing could pass one.
        """
        pytest.importorskip("maplibre")
        assert WebMap().add_raster(dataset, units="Pa").last_units == "Pa"

    def test_a_caller_supplied_unit_replaces_the_library_one_in_the_key(
        self, monkeypatch
    ):
        """``contours(units=)`` wins over the library hint all the way through to the legend heading.

        Args:
            monkeypatch: pytest's patcher, standing in for pyramids' contour tracer.

        Test scenario:
            The same field as the test above, whose library hint is ``hPa``. Correcting it through
            ``legend(title=...)`` would mean rewriting the whole heading, throwing away the ``level``
            column name the builder derived; the argument corrects only the unit.
        """
        pytest.importorskip("maplibre")
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import LineString

        class FakeRaster:
            """A raster that traces two levels, enough for a continuous key."""

            @staticmethod
            def contour(**kwargs):
                """Return two contours carrying their level."""
                return gpd.GeoDataFrame(
                    {"level": [960.0, 1000.0]},
                    geometry=[
                        LineString([(0, 0), (1, 1)]),
                        LineString([(0, 1), (1, 2)]),
                    ],
                    crs=DISPLAY_CRS,
                )

        monkeypatch.setattr(
            WebMap, "_display_raster_or_skip", lambda self, dataset, layer: FakeRaster()
        )
        monkeypatch.setattr(
            WebMap, "_to_display_source", lambda self, data, band=1: _source("msl")
        )
        m = WebMap().contours(object(), units="Pa").legend()
        assert "level (Pa)" in m._panels["legend"][0], m._panels["legend"][0]
        assert "hPa" not in m._panels["legend"][0], (
            "the library hint must not survive an override"
        )

    def test_contours_trace_the_library_levels_when_none_were_given(self, monkeypatch):
        """``contours()`` used to demand ``interval=``/``levels=``; a known field now supplies them.

        Args:
            monkeypatch: pytest's patcher, standing in for pyramids' contour tracer.
        """
        pytest.importorskip("maplibre")
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import LineString

        recorded = {}

        class FakeRaster:
            """A raster whose only job is to record how it was asked to contour."""

            @staticmethod
            def contour(**kwargs):
                """Record the trace request and hand back one drawable contour."""
                recorded.update(kwargs)
                return gpd.GeoDataFrame(
                    {"level": [960.0, 1000.0]},
                    geometry=[
                        LineString([(0, 0), (1, 1)]),
                        LineString([(0, 1), (1, 2)]),
                    ],
                    crs=DISPLAY_CRS,
                )

        monkeypatch.setattr(
            WebMap, "_display_raster_or_skip", lambda self, dataset, layer: FakeRaster()
        )
        monkeypatch.setattr(
            WebMap, "_to_display_source", lambda self, data, band=1: _source("msl")
        )
        WebMap().contours(object())
        assert recorded["fixed_levels"][0] == 960, recorded
        assert recorded["interval"] is None, recorded

    def test_the_legend_labels_the_column_with_its_units(self, monkeypatch):
        """A key of pressure contours that does not say ``hPa`` makes the reader guess.

        Args:
            monkeypatch: pytest's patcher, standing in for pyramids' contour tracer.
        """
        pytest.importorskip("maplibre")
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import LineString

        class FakeRaster:
            """A raster that traces two levels, enough for a continuous key."""

            @staticmethod
            def contour(**kwargs):
                """Return two contours carrying their level."""
                return gpd.GeoDataFrame(
                    {"level": [960.0, 1000.0]},
                    geometry=[
                        LineString([(0, 0), (1, 1)]),
                        LineString([(0, 1), (1, 2)]),
                    ],
                    crs=DISPLAY_CRS,
                )

        monkeypatch.setattr(
            WebMap, "_display_raster_or_skip", lambda self, dataset, layer: FakeRaster()
        )
        monkeypatch.setattr(
            WebMap, "_to_display_source", lambda self, data, band=1: _source("msl")
        )
        m = WebMap().contours(object()).legend()
        assert "level (hPa)" in m._panels["legend"][0], m._panels["legend"][0]

    def test_a_caller_supplied_title_still_wins(self, monkeypatch):
        """The units fill a gap; they never override what the caller wrote.

        Args:
            monkeypatch: pytest's patcher.
        """
        pytest.importorskip("maplibre")
        m = WebMap().choropleth(_polygon_frame(), column="pop")
        m.last_legend["units"] = "hPa"
        m.legend(title="Population")
        assert "Population" in m._panels["legend"][0]
        assert "hPa" not in m._panels["legend"][0]

    def test_a_classification_without_units_is_labelled_as_before(self):
        """No unit in the library means the bare column name, exactly as it always was."""
        pytest.importorskip("maplibre")
        m = WebMap().choropleth(_polygon_frame(), column="pop").legend()
        assert ">pop<" in m._panels["legend"][0], m._panels["legend"][0]


class TestC7OffLimbSkipsAndWarns:
    """Data the display CRS cannot place is a skipped layer, not a lost map."""

    def test_strict_defaults_to_off(self):
        """Skipping is the default because a builder chain may have a dozen layers in it."""
        assert inspect.signature(WebMap.__init__).parameters["strict"].default is False
        assert WebMap().strict is False

    def test_an_off_limb_raster_is_skipped_with_a_warning(
        self, monkeypatch, warning_log
    ):
        """The layer goes, the map stays, and the log says which layer and why.

        Args:
            monkeypatch: pytest's patcher, standing in for an off-limb warp.
            warning_log: The tier's loguru warnings.
        """
        pytest.importorskip("maplibre")

        def _off_limb(self, data, band=1):
            """Behave like a warp that placed none of the data."""
            raise OffLimbError("the data lies outside what 4326 can show")

        monkeypatch.setattr(WebMap, "_to_display_source", _off_limb)
        m = WebMap().basemap()
        before = len(m.layers)
        assert m.add_raster(object()) is m, "the builder must stay chainable"
        assert len(m.layers) == before
        assert any("add_raster" in line for line in warning_log), warning_log

    def test_an_off_limb_composite_is_skipped_with_a_warning(
        self, monkeypatch, warning_log
    ):
        """A composite whose dataset cannot be placed is dropped, and the log names the builder.

        Args:
            monkeypatch: pytest's patcher, standing in for a warp that placed none of the data.
            warning_log: The tier's loguru warnings.

        Test scenario:
            ``rgb_composite`` needs the dataset itself rather than one band, so it goes through the
            raster-shaped guard instead of the source-shaped one. Without its own guard an off-limb
            composite raised where the very same data drawn as a single band would have been skipped.
        """
        pytest.importorskip("maplibre")

        def _off_limb(self, dataset):
            """Behave like a warp that placed none of the data."""
            raise OffLimbError("the data lies outside what 4326 can show")

        monkeypatch.setattr(WebMap, "_to_display_raster", _off_limb)
        m = WebMap().basemap()
        before = len(m.layers)
        assert m.rgb_composite(object(), bands=(3, 2, 1)) is m, (
            "the builder must stay chainable"
        )
        assert len(m.layers) == before, "an off-limb composite must add no layer"
        assert any("rgb_composite" in line for line in warning_log), warning_log

    def test_an_off_limb_contour_layer_is_skipped_with_a_warning(
        self, monkeypatch, warning_log
    ):
        """Contours over data the display CRS cannot place are skipped, not traced.

        Args:
            monkeypatch: pytest's patcher, standing in for a warp that placed none of the data.
            warning_log: The tier's loguru warnings.

        Test scenario:
            ``contours`` reads the dataset through the same raster-shaped guard the composites use, so it
            has to answer an unplaceable input the way every other builder does — a skipped layer and a
            warning naming it, with the rest of the chain still drawable.
        """
        pytest.importorskip("maplibre")

        def _off_limb(self, dataset):
            """Behave like a warp that placed none of the data."""
            raise OffLimbError("the data lies outside what 4326 can show")

        monkeypatch.setattr(WebMap, "_to_display_raster", _off_limb)
        m = WebMap().basemap()
        before = len(m.layers)
        assert m.contours(object(), interval=100.0) is m, (
            "the builder must stay chainable"
        )
        assert len(m.layers) == before, "an off-limb contour layer must add no layer"
        assert any("contours" in line for line in warning_log), warning_log

    def test_strict_re_raises_the_off_limb_error(self, monkeypatch):
        """``strict=True`` is for a pipeline that must not publish a map with a layer missing.

        Args:
            monkeypatch: pytest's patcher.
        """
        pytest.importorskip("maplibre")

        def _off_limb(self, data, band=1):
            """Behave like a warp that placed none of the data."""
            raise OffLimbError("the data lies outside what 4326 can show")

        monkeypatch.setattr(WebMap, "_to_display_source", _off_limb)
        with pytest.raises(OffLimbError):
            WebMap(strict=True).add_raster(object())

    @staticmethod
    def _off_limb_features():
        """A FeatureCollection whose orthographic coordinates lie off the projection's disc.

        Every coordinate is outside what an orthographic CRS can invert, so the warp into the tier's
        lon/lat display CRS really does hand back infinities — the condition the contract turns on, driven
        here rather than monkeypatched, because a patched ``reproject`` cannot notice a builder that never
        calls it (which is exactly what this tier did).

        Returns:
            FeatureCollection: point geometries carrying a numeric ``v``.
        """
        import geopandas as gpd
        from pyramids.feature import FeatureCollection
        from shapely.geometry import Point

        ortho = "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        return FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[Point(5e7, 5e7), Point(6e7, 6e7)],
                crs=ortho,
            )
        )

    def test_an_off_limb_vector_layer_reaches_the_policy(self, warning_log):
        """A vector warp that places nothing is announced, naming the builder (H3).

        Test scenario:
            ``_display_gdf`` called ``to_crs`` directly, and a vector warp does not raise when it places
            nothing — it returns ``inf`` coordinates and says so nowhere. The layer was built from those
            infinities in silence, so the tier's whole skip-and-warn policy never applied to vector data.
        """
        pytest.importorskip("maplibre")
        m = WebMap().basemap()
        m.points(self._off_limb_features())
        assert any("points" in line for line in warning_log), (
            f"the skip has to name the builder; got {warning_log!r}"
        )

    def test_strict_raises_off_limb_for_a_vector_layer(self):
        """Under ``strict=True`` the same vector warp raises, as it already did for a raster (H3)."""
        pytest.importorskip("maplibre")
        with pytest.raises(OffLimbError):
            WebMap(strict=True).points(self._off_limb_features())

    def test_a_skip_with_no_exception_behind_it_still_raises_off_limb(self):
        """Every strict refusal is one type, whatever the reason behind it (M6).

        Test scenario:
            Four builders skip for reasons that carry no exception — corners that will not express as
            lon/lat (twice), a contour trace with no level in range, an unplaceable time step. Under
            ``strict`` they raised a bare ``ValueError`` while the other three tiers raised ``OffLimbError``,
            so ``except OffLimbError`` around a strict map silently missed them.
        """
        with pytest.raises(OffLimbError, match="add_raster: nothing to place"):
            WebMap(strict=True)._skipped("add_raster", "nothing to place")

    def test_the_shared_exception_type_is_not_a_value_error(self):
        """Naming the change: ``OffLimbError`` derives from ``RuntimeError``, so it is not a ``ValueError``.

        Test scenario:
            A caller who wrote ``except ValueError`` against this tier's old bare raise no longer catches
            these four skips. That break is the point — one type across the tiers — and it is pinned here
            so the base class cannot drift back without this saying so.
        """
        assert issubclass(OffLimbError, RuntimeError), (
            "the shared signal is a RuntimeError on every tier"
        )
        assert not issubclass(OffLimbError, ValueError), (
            "an `except ValueError` no longer catches the web tier's strict refusals"
        )


class TestC8TheBigDataCutoff:
    """One name, two reaches: the map's attribute and a single call's override."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip without the web extra."""
        pytest.importorskip("maplibre")

    def test_the_default_is_the_shared_one(self):
        """The cutoff is a documented number, not a literal buried in the router."""
        assert DEFAULT_BIG_DATA_THRESHOLD == 50_000
        assert WebMap().big_data_threshold == 50_000

    def test_the_attribute_moves_the_cutoff_for_every_later_layer(self):
        """Setting it once is how a caller changes the whole map, not one builder at a time."""
        m = WebMap()
        m.big_data_threshold = 2
        m.points(_points_frame(5)).points(_points_frame(5))
        assert m.layer_ids == [], "both layers should have routed to deck.gl"
        assert len(m._deck_layers) == 2, m._deck_layers

    def test_a_per_call_override_moves_it_for_that_call_only(self):
        """The same name per call, so a caller does not have to learn a second one."""
        m = WebMap()
        m.points(_points_frame(5), big_data_threshold=2)
        assert m.layer_ids == [], "the overridden call should have routed to deck.gl"
        m.points(_points_frame(5))
        assert m.layer_ids, "the map's own threshold was left at 50 000"
        assert m.big_data_threshold == 50_000, "the override leaked onto the map"

    def test_polygons_honour_the_override_too(self):
        """Both routed builders take it, or the override would depend on the geometry kind."""
        m = WebMap()
        m.polygons(_polygon_frame(), big_data_threshold=1)
        assert m.layer_ids == [] and m._deck_layers, "polygons ignored the override"

    def test_a_negative_threshold_is_refused(self):
        """A cutoff below zero routes an empty layer, which is never what the caller meant."""
        with pytest.raises(ValueError, match="must not be negative"):
            WebMap().points(_points_frame(3), big_data_threshold=-1)


class TestC9TheBasemapDefaultIsShared:
    """The default provider is one constant, not a literal per tier."""

    def test_basemap_defaults_to_the_shared_constant(self):
        """``CartoDark`` was hard-coded here while the interactive tier hard-coded ``CartoLight``."""
        default = inspect.signature(WebMap.basemap).parameters["provider"].default
        assert default == DEFAULT_BASEMAP_PROVIDER

    def test_the_constant_comes_from_base_basemaps(self):
        """It lives in ``base/`` because both tiers read it; this fails if the tier re-hard-codes it."""
        from digitalearth.base import basemaps

        shared = getattr(basemaps, "DEFAULT_BASEMAP_PROVIDER", None)
        assert shared is not None, (
            "base/basemaps.py must declare DEFAULT_BASEMAP_PROVIDER — it is the one place the tiers agree on "
            "a default, and skipping here would let its removal pass as a green run"
        )
        assert DEFAULT_BASEMAP_PROVIDER == shared

    def test_the_default_provider_resolves_to_a_real_tile_source(self):
        """A default that is not one of the known names would raise on every bare ``basemap()``."""
        pytest.importorskip("maplibre")
        assert WebMap().basemap().layers, "the default provider added no basemap"


class TestC10KeyedCoverageReachesTheSource:
    """Where a keyed provider declares its coverage, MapLibre is told."""

    @pytest.fixture(autouse=True)
    def _need_engine(self, monkeypatch):
        """Skip without the web extra, and supply a credential that is not a real key.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        pytest.importorskip("maplibre")
        monkeypatch.setenv("PLANET_API_KEY", "FAKE-KEY-NOT-REAL")

    def test_a_keyed_presets_bounds_become_the_source_bounds(self):
        """NICFI covers the tropics; without this MapLibre asks for tiles across the whole world."""
        from digitalearth.base.basemaps import get_keyed_basemap

        expected = list(get_keyed_basemap("Planet.NICFI", date="2024-01").bounds)
        source = _sources_of(
            WebMap().basemap("Planet.NICFI", preset={"date": "2024-01"})
        )[0]
        assert source["bounds"] == expected, source

    def test_a_global_provider_declares_no_bounds(self):
        """Declaring a box for a global source would clamp a basemap that covers everything."""
        source = _sources_of(WebMap().basemap("OSM"))[0]
        assert "bounds" not in source, source

    def test_a_malformed_bounds_is_refused(self):
        """MapLibre ignores a malformed ``bounds``, so the coverage would go quietly undeclared."""
        with pytest.raises(ValueError, match="west, south, east, north"):
            WebMap().tiles("https://a/{z}/{x}/{y}.png", bounds=(1.0, 2.0, 3.0))


class TestC13TheDisplayCrsIsDeclared:
    """Inline data is placed by lon/lat degrees, so 4326 is the only display CRS that works."""

    def test_the_default_is_4326(self):
        """The tier has always reprojected to lon/lat; now it says so."""
        assert WebMap().crs == 4326

    @pytest.mark.parametrize("spelling", [4326, "4326", "EPSG:4326"])
    def test_the_accepted_spellings_normalise_to_the_int(self, spelling):
        """``_needs_reproject`` compares against an int, so the spelling is normalised on the way in.

        Args:
            spelling: One of the accepted ways to write EPSG:4326.
        """
        assert WebMap(crs=spelling).crs == 4326

    @pytest.mark.parametrize("crs", [3857, "EPSG:3857", "ESRI:54009"])
    def test_any_other_crs_is_refused_with_a_clear_error(self, crs):
        """Silently mis-placing the data was the alternative, and it looks like a rendering bug.

        Args:
            crs: A display CRS MapLibre cannot place inline data in.
        """
        with pytest.raises(ValueError, match="renders in EPSG:4326 only"):
            WebMap(crs=crs)

    def test_the_error_says_what_to_do_instead(self):
        """An error that only says no leaves the caller to guess which tier can project."""
        with pytest.raises(ValueError, match="static tier"):
            WebMap(crs=3857)
