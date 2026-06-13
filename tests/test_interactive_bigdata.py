"""DI.2 — big-data via Datashader (rasterize / datashade / auto-routing / categorical / trajectory).

Determinism: every aggregation pins the canvas (``width``/``height``) and uses ``dynamic=False`` so
the rasterized arrays are static and assertable — ``dynamic=True`` (the interactive default) needs a
live server and is asserted type-level only. Seeded numpy randomness; no browser, no network.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")

rng = np.random.default_rng(1337)


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


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


class TestRasterize:
    """``rasterize`` — numeric density images with stable canvases."""

    def test_static_rasterize_is_image_not_glyphs(self, m, big_points):
        m.rasterize(big_points, dynamic=False, width=120, height=80)
        layer = m.layers[0]
        assert isinstance(
            layer, hv.Image
        ), f"expected a rasterized hv.Image, got {type(layer)}"
        assert not isinstance(
            layer, gv.Points
        ), "no raw point glyphs above the threshold"

    def test_canvas_shape_is_pinned(self, m, big_points):
        m.rasterize(big_points, dynamic=False, width=120, height=80)
        grid = m.layers[0].dimension_values(2, flat=False)
        assert grid.shape == (80, 120), f"canvas not pinned: {grid.shape}"

    def test_dynamic_default_returns_dynamicmap(self, m, big_points):
        m.rasterize(big_points)
        assert isinstance(
            m.layers[0], hv.DynamicMap
        ), "dynamic=True must wrap a DynamicMap"

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
        assert not np.allclose(
            np.nan_to_num(count_grid), np.nan_to_num(mean_grid)
        ), "count and mean aggregations must differ on the same data"

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


class TestDatashade:
    """``datashade`` — shaded RGB, categorical color_key (DI.2a)."""

    def test_static_datashade_is_rgb(self, m, big_points):
        m.datashade(big_points, dynamic=False, width=60, height=40)
        assert isinstance(
            m.layers[0], hv.RGB
        ), f"expected shaded hv.RGB, got {type(m.layers[0])}"

    def test_categorical_color_key_blend(self, m, big_points):
        key = {"a": "#ff0000", "b": "#00ff00", "c": "#0000ff"}
        m.datashade(
            big_points, color_key=key, column="cls", dynamic=False, width=60, height=40
        )
        assert isinstance(
            m.layers[0], hv.RGB
        ), "categorical shade must produce an RGB blend"

    def test_non_categorical_column_is_cast_and_logged(self, m, big_points):
        """A plain object class column is cast to category (logged, not silent)."""
        assert (
            str(big_points["cls"].dtype) != "category"
        ), "fixture must start non-categorical"
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


class TestAutoRouting:
    """``points``/``polygons`` auto-route through Datashader above the threshold."""

    def test_points_above_threshold_auto_rasterize(self, m, big_points):
        m.points(big_points, rasterize_threshold=1_000)
        assert isinstance(
            m.layers[0], hv.DynamicMap
        ), "above-threshold points must become a rasterized layer, not glyphs"

    def test_points_below_threshold_stay_glyphs(self, m, big_points):
        m.points(big_points.head(100), rasterize_threshold=1_000)
        assert isinstance(
            m.layers[0], gv.Points
        ), "below-threshold points must stay raw glyphs"

    def test_points_forced_off_stays_glyphs_even_when_big(self, m, big_points):
        m.points(big_points.head(5_000), rasterize=False, rasterize_threshold=1_000)
        assert isinstance(
            m.layers[0], gv.Points
        ), "rasterize=False must force raw glyphs"

    def test_points_forced_on_rasterizes_even_when_small(self, m, big_points):
        m.points(big_points.head(100), rasterize=True)
        assert isinstance(
            m.layers[0], hv.DynamicMap
        ), "rasterize=True must force Datashader"

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

    @pytest.fixture()
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
