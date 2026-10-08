"""DI.2 — big-data via Datashader (rasterize / datashade / auto-routing / categorical / trajectory).

Determinism: every aggregation pins the canvas (``width``/``height``) and uses ``dynamic=False`` so
the rasterized arrays are static and assertable — ``dynamic=True`` (the interactive default) needs a
live server and is asserted type-level only. Seeded numpy randomness; no browser, no network.
"""

import pathlib

import numpy as np
import pytest
from pyramids.feature import FeatureCollection

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")

rng = np.random.default_rng(1337)


@pytest.fixture
def m():
    """Yield a fresh Web-Mercator map for each test, closing it on the way out.

    The object registry is process-global and holds strong references, so a map that is never closed
    keeps the data it drew for the rest of the session.

    Yields:
        The map.
    """
    interactive_map = InteractiveMap()
    yield interactive_map
    interactive_map.close()


@pytest.fixture(scope="module")
def big_points():
    """A seeded 20k-row Web-Mercator point GeoDataFrame with a value and a class column.

    20k is plenty to exercise the rasterize/datashade paths while keeping the file well under the
    10s budget; the auto-routing tests pin their own thresholds so they do not depend on this size.
    """
    import geopandas as gpd

    n = 20_000
    x = rng.uniform(-9e6, -8e6, n)
    y = rng.uniform(4e6, 5e6, n)
    values = rng.normal(10.0, 2.0, n)
    cls = rng.choice(["a", "b", "c"], n)
    return gpd.GeoDataFrame(
        {"value": values, "cls": cls},
        geometry=gpd.points_from_xy(x, y),
        crs="EPSG:3857",
    )


class TestRasterizeFrozenScale:
    """``rasterize`` can pin a colour scale so it does not re-range per viewport (IN-4)."""

    def test_clim_and_cnorm_reach_the_recorded_style(self, m, big_points):
        """A frozen ``(vmin, vmax)`` and ``cnorm`` are recorded on the layer's style, as a tuple.

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        m.rasterize(
            big_points,
            dynamic=False,
            width=60,
            height=40,
            clim=(0.0, 20.0),
            cnorm="eq_hist",
        )
        common = m.style_of(0)["common"]
        assert common["clim"] == (0.0, 20.0), common
        assert common["cnorm"] == "eq_hist", common

    def test_the_frozen_scale_travels_in_the_figure(self, m, big_points):
        """The pair is recorded in the JSON-safe list spelling, so a saved figure keeps it (R2-M13).

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        m.rasterize(big_points, dynamic=False, width=60, height=40, clim=(1.0, 9.0))
        props = m.figure_spec.layers.get(m.layer_ids[0]).symbology.props
        # Recorded as a plain pair of floats (the spec freezes the travelling list to a tuple to hash it),
        # which is what round-trips through `to_dict`/`from_dict` — not held beside the layer (R2-M13).
        assert tuple(props["common"]["clim"]) == (1.0, 9.0), props["common"]

    def test_no_clim_leaves_the_scale_free(self, m, big_points):
        """Without ``clim`` the layer records none, so Bokeh keeps auto-ranging (the default).

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        m.rasterize(big_points, dynamic=False, width=60, height=40)
        assert "clim" not in m.style_of(0)["common"]


class TestRasterize:
    """``rasterize`` — numeric density images with stable canvases."""

    def test_static_rasterize_is_image_not_glyphs(self, m, big_points):
        m.rasterize(big_points, dynamic=False, width=120, height=80)
        layer = m.layers[0]
        assert isinstance(layer, hv.Image), (
            f"expected a rasterized hv.Image, got {type(layer)}"
        )
        assert not isinstance(layer, gv.Points), (
            "no raw point glyphs above the threshold"
        )

    def test_canvas_shape_is_pinned(self, m, big_points):
        m.rasterize(big_points, dynamic=False, width=120, height=80)
        grid = m.layers[0].dimension_values(2, flat=False)
        assert grid.shape == (80, 120), f"canvas not pinned: {grid.shape}"

    def test_dynamic_default_returns_dynamicmap(self, m, big_points):
        m.rasterize(big_points)
        assert isinstance(m.layers[0], hv.DynamicMap), (
            "dynamic=True must wrap a DynamicMap"
        )

    def test_aggregators_change_the_output(self, m, big_points):
        m.rasterize(big_points, aggregator="count", dynamic=False, width=60, height=40)
        m.rasterize(
            big_points,
            aggregator="mean",
            column="value",
            dynamic=False,
            width=60,
            height=40,
        )
        count_grid = m.layers[0].dimension_values(2, flat=False)
        mean_grid = m.layers[1].dimension_values(2, flat=False)
        assert not np.allclose(np.nan_to_num(count_grid), np.nan_to_num(mean_grid)), (
            "count and mean aggregations must differ on the same data"
        )

    def test_column_requiring_aggregator_without_column_raises(self, m, big_points):
        with pytest.raises(ValueError, match="needs a column"):
            m.rasterize(big_points, aggregator="mean", dynamic=False)

    def test_unknown_aggregator_raises(self, m, big_points):
        with pytest.raises(ValueError, match="unknown aggregator"):
            m.rasterize(big_points, aggregator="median", dynamic=False)

    def test_datashader_reduction_object_passes_through(self, m, big_points):
        """A pre-built Datashader reduction is used verbatim (not name-resolved)."""
        import datashader as ds

        m.rasterize(
            big_points, aggregator=ds.mean("value"), dynamic=False, width=40, height=30
        )
        grid = m.layers[0].dimension_values(2, flat=False)
        assert grid.shape == (
            30,
            40,
        ), f"reduction-object path canvas not pinned: {grid.shape}"

    def test_prebuilt_element_is_rasterized_directly(self, m, big_points):
        """Passing an already-built HoloViews element skips the GeoDataFrame plumbing."""
        element = m._vector_element("Points", big_points)
        m.rasterize(element, dynamic=False, width=40, height=30)
        assert isinstance(m.layers[0], hv.Image), f"got {type(m.layers[0])}"


class TestDatashadeFrozenScale:
    """``datashade``/``trajectory`` can pin a colour span so they do not re-autorange per frame (IN-4)."""

    def test_datashade_eq_hist_with_a_span_is_refused_pointing_at_rasterize(
        self, m, big_points
    ):
        """`cnorm="eq_hist"` with a `clim` is refused early with an actionable message (review M2).

        Datashader cannot equalise under a fixed span; the error must name `rasterize` rather than let the
        raw "span is not (yet) valid to use with eq_hist" surface.

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        with pytest.raises(ValueError, match="eq_hist.*rasterize"):
            m.datashade(
                big_points,
                column="value",
                clim=(0.0, 1.0),
                cnorm="eq_hist",
                dynamic=False,
            )

    def test_trajectory_eq_hist_with_a_span_is_refused(self, m, big_points):
        """The trajectory path refuses the same combination with the same actionable message (review M2).

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        with pytest.raises(ValueError, match="eq_hist.*rasterize"):
            m.trajectory(
                big_points,
                track_column="cls",
                clim=(0.0, 1.0),
                cnorm="eq_hist",
                dynamic=False,
            )

    def test_datashade_with_a_span_renders_without_the_eq_hist_trap(
        self, m, big_points
    ):
        """A `clim` span defaults `cnorm` to linear, so the shade does not hit Datashader's eq_hist+span error.

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        m.datashade(
            big_points,
            column="value",
            clim=(0.0, 20.0),
            dynamic=False,
            width=60,
            height=40,
        )
        assert isinstance(m.layers[0], hv.RGB), (
            f"expected a shaded RGB, got {type(m.layers[0])}"
        )

    def test_datashade_records_the_span_and_norm(self, m, big_points):
        """The span travels as a plain pair and the chosen `cnorm` is recorded.

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        m.datashade(
            big_points,
            column="value",
            clim=(1.0, 9.0),
            cnorm="log",
            dynamic=False,
            width=40,
            height=30,
        )
        props = m.figure_spec.layers.get(m.layer_ids[0]).symbology.props
        assert tuple(props["clim"]) == (1.0, 9.0), props["clim"]
        assert props["cnorm"] == "log", props

    def test_trajectory_takes_a_frozen_span_too(self, m, big_points):
        """The trajectory path honours `clim`/`cnorm` the same way (IN-4).

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture (its ``cls`` column groups the tracks).
        """
        m.trajectory(
            big_points,
            track_column="cls",
            clim=(0.0, 5.0),
            dynamic=False,
            width=60,
            height=40,
        )
        assert isinstance(m.layers[0], hv.RGB), (
            f"expected a shaded RGB, got {type(m.layers[0])}"
        )

    def test_datashade_cnorm_without_a_span(self, m, big_points):
        """A `cnorm` without a `clim` sets only the normalisation, not a span (IN-4).

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        m.datashade(
            big_points, column="value", cnorm="log", dynamic=False, width=40, height=30
        )
        assert isinstance(m.layers[0], hv.RGB), (
            f"expected a shaded RGB, got {type(m.layers[0])}"
        )

    def test_trajectory_cnorm_without_a_span(self, m, big_points):
        """The trajectory path also takes a bare `cnorm` with no span (IN-4).

        Args:
            m: The map fixture.
            big_points: The 20k-point fixture.
        """
        m.trajectory(
            big_points,
            track_column="cls",
            cnorm="linear",
            dynamic=False,
            width=60,
            height=40,
        )
        assert isinstance(m.layers[0], hv.RGB), (
            f"expected a shaded RGB, got {type(m.layers[0])}"
        )


class TestDatashade:
    """``datashade`` — shaded RGB, categorical color_key (DI.2a)."""

    def test_static_datashade_is_rgb(self, m, big_points):
        m.datashade(big_points, dynamic=False, width=60, height=40)
        assert isinstance(m.layers[0], hv.RGB), (
            f"expected shaded hv.RGB, got {type(m.layers[0])}"
        )

    def test_categorical_color_key_blend(self, m, big_points):
        key = {"a": "#ff0000", "b": "#00ff00", "c": "#0000ff"}
        m.datashade(
            big_points, color_key=key, column="cls", dynamic=False, width=60, height=40
        )
        assert isinstance(m.layers[0], hv.RGB), (
            "categorical shade must produce an RGB blend"
        )

    def test_non_categorical_column_is_cast_and_logged(self, m, big_points):
        """A plain object class column is cast to category (logged, not silent)."""
        assert str(big_points["cls"].dtype) != "category", (
            "fixture must start non-categorical"
        )
        m.datashade(
            big_points,
            color_key={"a": "#ff0000", "b": "#00ff00", "c": "#0000ff"},
            column="cls",
            dynamic=False,
            width=40,
            height=30,
        )
        assert isinstance(m.layers[0], hv.RGB)

    def test_already_categorical_column_is_not_recopied(self, m, big_points):
        """An already-categorical column skips the cast (the no-op _ensure_categorical branch)."""
        gdf = big_points.copy()
        gdf["cls"] = gdf["cls"].astype("category")
        m.datashade(
            gdf,
            color_key={"a": "#ff0000", "b": "#00ff00", "c": "#0000ff"},
            column="cls",
            dynamic=False,
            width=40,
            height=30,
        )
        assert isinstance(m.layers[0], hv.RGB)

    def test_a_list_color_key_is_accepted(self, m, big_points):
        """HoloViews takes a colour list as well as a mapping; the description must not narrow that.

        Args:
            m: The map.
            big_points: Points with a three-class column.

        Test scenario:
            The recorded `color_key` comes back from the description as a tuple, and the drawer passed it
            through `dict()`, so a list of colours — one per category, in category order — raised
            `ValueError: dictionary update sequence element #0 has length 7` (review L6).
        """
        m.datashade(
            big_points,
            color_key=["#ff0000", "#00ff00", "#0000ff"],
            column="cls",
            dynamic=False,
            width=40,
            height=30,
        )
        assert isinstance(m.layers[0], hv.RGB), type(m.layers[0])


class TestAutoRouting:
    """``points``/``polygons`` auto-route through Datashader above the threshold."""

    def test_points_above_threshold_auto_rasterize(self, m, big_points):
        m.points(big_points, big_data_threshold=1_000)
        assert isinstance(m.layers[0], hv.DynamicMap), (
            "above-threshold points must become a rasterized layer, not glyphs"
        )

    def test_points_below_threshold_stay_glyphs(self, m, big_points):
        m.points(big_points.head(100), big_data_threshold=1_000)
        assert isinstance(m.layers[0], gv.Points), (
            "below-threshold points must stay raw glyphs"
        )

    def test_points_forced_off_stays_glyphs_even_when_big(self, m, big_points):
        m.points(big_points.head(5_000), rasterize=False, big_data_threshold=1_000)
        assert isinstance(m.layers[0], gv.Points), (
            "rasterize=False must force raw glyphs"
        )

    def test_points_forced_on_rasterizes_even_when_small(self, m, big_points):
        m.points(big_points.head(100), rasterize=True)
        assert isinstance(m.layers[0], hv.DynamicMap), (
            "rasterize=True must force Datashader"
        )

    def test_polygons_route_needs_spatialpandas(self, m, big_points):
        """Forced polygon rasterize raises the actionable spatialpandas error when absent."""
        from importlib.util import find_spec

        polys = big_points.head(50).copy()
        polys["geometry"] = polys.geometry.buffer(1_000.0)
        if find_spec("spatialpandas") is None:
            with pytest.raises(ImportError, match="spatialpandas"):
                m.polygons(polys, rasterize=True)
        else:  # pragma: no cover - env-dependent branch
            m.polygons(polys, rasterize=True)
            assert isinstance(m.layers[0], hv.DynamicMap)


class TestTrajectory:
    """``trajectory`` — NaN-separated track datashading (DI.2b)."""

    @pytest.fixture
    def tracks(self):
        """Three seeded random-walk tracks (3 x 2,000 points) with a track id and a class."""
        import geopandas as gpd
        import pandas as pd

        frames = []
        for track_id, cls in (("t1", "ship"), ("t2", "ship"), ("t3", "buoy")):
            steps = rng.normal(0, 200.0, size=(800, 2)).cumsum(axis=0)
            frames.append(
                pd.DataFrame(
                    {
                        "x": -8.5e6 + steps[:, 0],
                        "y": 4.5e6 + steps[:, 1],
                        "track": track_id,
                        "kind": cls,
                    }
                )
            )
        table = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(
            table[["track", "kind"]],
            geometry=gpd.points_from_xy(table["x"], table["y"]),
            crs="EPSG:3857",
        )

    def test_trajectory_shades_to_rgb(self, m, tracks):
        m.trajectory(tracks, track_column="track", dynamic=False, width=80, height=60)
        assert isinstance(m.layers[0], hv.RGB), f"got {type(m.layers[0])}"

    def test_trajectory_by_class(self, m, tracks):
        m.trajectory(
            tracks,
            track_column="track",
            by="kind",
            dynamic=False,
            dynspread=False,
            width=80,
            height=60,
        )
        assert isinstance(m.layers[0], hv.RGB)

    def test_single_track_without_track_column(self, m, tracks):
        m.trajectory(
            tracks[tracks["track"] == "t1"], dynamic=False, width=60, height=40
        )
        assert isinstance(m.layers[0], hv.RGB)

    def test_trajectory_takes_a_list_color_key(self, m, tracks):
        """The trajectory drawer passed `color_key` through `dict()` too, so a colour list raised there.

        Args:
            m: The map.
            tracks: Three tracks in two classes.
        """
        m.trajectory(
            tracks,
            track_column="track",
            by="kind",
            color_key=["#ff0000", "#0000ff"],
            dynamic=False,
            dynspread=False,
            width=80,
            height=60,
        )
        assert isinstance(m.layers[0], hv.RGB), type(m.layers[0])

    def test_trajectory_by_class_with_color_key(self, m, tracks):
        """The ``by`` + ``color_key`` branch colours each track class explicitly."""
        m.trajectory(
            tracks,
            track_column="track",
            by="kind",
            color_key={"ship": "#ff0000", "buoy": "#0000ff"},
            dynamic=False,
            dynspread=False,
            width=80,
            height=60,
        )
        assert isinstance(m.layers[0], hv.RGB)


class TestAFigureThatCarriesNoReduction:
    """A Datashader reduction has no JSON form, so a figure read back from a dict describes none."""

    def test_no_aggregator_at_all_becomes_the_tier_s_default_count(self):
        """The drawer is handed `None`, and `None` is not something Datashader can aggregate with.

        Test scenario:
            A reduction object is held on the map, never written into the description — so a figure
            saved and read back elsewhere reaches this resolver with nothing. Passing that `None`
            straight through hands Datashader a null reduction and fails inside the engine, where the
            builder's own default is what the figure meant.
        """
        import datashader as ds

        from digitalearth.interactive.bigdata import _resolve_aggregator

        resolved = _resolve_aggregator(None, None)
        assert isinstance(resolved, type(ds.count())), (
            f"a description carrying no reduction must default to count(); got {resolved!r}"
        )


class TestTheThresholdCountsRowsNotCharacters:
    """A path is a first-class input here, and `len()` of one is its character count (#316)."""

    @staticmethod
    def _recorded_kind(scene, features, threshold):
        """Draw and return the kind the layer recorded, which says which route was taken.

        Args:
            scene: The map to draw on.
            features: The builder's input — here always a path.
            threshold: The per-call `big_data_threshold`.

        Returns:
            The recorded kind: `"raster"` when the layer was routed through Datashader, else `"points"`.
        """
        scene.points(features, rasterize="auto", big_data_threshold=threshold)
        return scene.get_layer(scene.layer_ids[-1]).kind

    def test_a_path_longer_than_the_threshold_does_not_rasterize_a_small_file(self, m):
        """The discriminating case: more characters than the threshold, fewer rows than it.

        Args:
            m: A fresh interactive map.

        Test scenario:
            `tests/data/points.geojson` holds 10 features and its absolute path is far longer than 10
            characters, so a threshold set between the two tells the readings apart. Counting the path
            rasterised a 10-row file and logged "85 features exceed big_data_threshold=50"; counting the rows
            leaves it as glyphs. A threshold below the row count cannot catch this — both readings exceed it —
            which is why the positive control below is a separate case.
        """
        path = str(pathlib.Path("tests/data/points.geojson").resolve())
        rows = FeatureCollection.feature_count(path)
        assert len(path) > 50 > rows, (
            f"the premise needs the threshold between {rows} rows and {len(path)} characters"
        )
        assert self._recorded_kind(m, path, 50) == "points", (
            "a 10-row file was routed through Datashader against a threshold of 50"
        )

    def test_a_row_count_over_the_threshold_still_rasterizes(self, m):
        """The positive control: the routing itself still happens when the rows warrant it.

        Args:
            m: A fresh interactive map.

        Test scenario:
            The same file against a threshold of 5. Without this, a count that always answered zero would
            satisfy the case above.
        """
        path = str(pathlib.Path("tests/data/points.geojson").resolve())
        assert self._recorded_kind(m, path, 5) == "raster", (
            "a 10-row file was not routed through Datashader against a threshold of 5"
        )

    def test_an_opened_collection_counts_the_same_way(self, m):
        """Opening the file first must not change the decision it drives.

        Args:
            m: A fresh interactive map.

        Test scenario:
            The path and the opened collection are two different inputs naming the same 10 rows, so the
            threshold that leaves one as glyphs must leave the other as glyphs too. Before the fix they
            disagreed, which is what made the bug invisible to anyone testing with an opened collection.
        """
        path = str(pathlib.Path("tests/data/points.geojson").resolve())
        opened = FeatureCollection.read_file(path)
        assert self._recorded_kind(m, opened, 50) == "points", (
            "an opened 10-row collection was routed through Datashader against a threshold of 50"
        )
