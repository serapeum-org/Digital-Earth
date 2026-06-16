"""Tests for digitalearth.charts — non-map x–y charts (line, bar, histogram)."""
import numpy as np
import pytest
from matplotlib.axes import Axes

from digitalearth import charts


def test_top_level_exports():
    """line/bar/histogram (and the layout helpers) are importable from the top-level package."""
    import digitalearth

    for name in ("line", "bar", "histogram", "grid", "shared_colorbar"):
        assert hasattr(digitalearth, name), f"digitalearth.{name} should be exported"
        assert name in digitalearth.__all__, f"{name} should be in digitalearth.__all__"
    assert digitalearth.line is charts.line, "top-level line must be the charts.line function"
    assert digitalearth.histogram is charts.histogram, "top-level histogram must be charts.histogram"


class TestLine:
    """Tests for charts.line."""

    def test_single_series_adds_one_line(self):
        """A 1-D y draws exactly one Line2D and returns the Axes."""
        ax = charts.line([0, 1, 2, 3], [0, 1, 4, 9])
        assert isinstance(ax, Axes), f"expected an Axes, got {type(ax)}"
        assert len(ax.lines) == 1, f"one series should add one line, got {len(ax.lines)}"

    def test_multi_series_one_line_per_column(self):
        """A 2-D y draws one line per column (shared x)."""
        y = np.column_stack([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
        ax = charts.line([0, 1, 2], y)
        assert len(ax.lines) == 3, f"three columns should add three lines, got {len(ax.lines)}"

    def test_label_appears_in_legend(self):
        """A passed label is discoverable through the axes legend handles."""
        ax = charts.line([0, 1, 2], [1, 2, 3], label="series-A")
        _, labels = ax.get_legend_handles_labels()
        assert "series-A" in labels, f"label not registered on the axes: {labels}"

    def test_draws_on_supplied_axes(self):
        """When an axes is supplied, the series is drawn on it (no new figure)."""
        import matplotlib.pyplot as plt

        fig, ax0 = plt.subplots()
        ax = charts.line([0, 1], [0, 1], ax=ax0)
        assert ax is ax0, "should draw on the supplied axes"
        assert ax0.get_figure() is fig

    def test_color_passthrough(self):
        """An explicit colour reaches the drawn line."""
        from matplotlib.colors import to_rgba

        ax = charts.line([0, 1, 2], [0, 1, 0], color="red")
        assert ax.lines[0].get_color() in ("red", to_rgba("red")), f"colour not applied: {ax.lines[0].get_color()}"


class TestBar:
    """Tests for charts.bar."""

    def test_bar_count_matches_x(self):
        """A bar chart adds one bar per element of x and returns the Axes."""
        ax = charts.bar([0, 1, 2, 3], [3, 1, 4, 1])
        assert isinstance(ax, Axes), f"expected an Axes, got {type(ax)}"
        assert len(ax.containers[0]) == 4, f"expected 4 bars, got {len(ax.containers[0])}"

    def test_bar_heights_match_input(self):
        """The drawn bar heights equal the supplied heights, in order."""
        ax = charts.bar([0, 1, 2], [2.0, 5.0, 3.0])
        heights = [round(rect.get_height(), 1) for rect in ax.containers[0]]
        assert heights == [2.0, 5.0, 3.0], f"heights mismatch: {heights}"

    def test_bar_color_passthrough(self):
        """An explicit colour reaches the drawn bars."""
        from matplotlib.colors import to_rgba

        ax = charts.bar([0, 1], [1, 2], color="green")
        assert ax.containers[0][0].get_facecolor() == to_rgba("green"), "bar colour not applied"

    def test_bar_on_supplied_axes(self):
        """When an axes is supplied, the bars are drawn on it."""
        import matplotlib.pyplot as plt

        fig, ax0 = plt.subplots()
        ax = charts.bar([0, 1], [1, 2], ax=ax0)
        assert ax is ax0, "should draw on the supplied axes"


class TestHistogram:
    """Tests for charts.histogram and the _as_finite_array helper."""

    def test_array_histogram_bins(self):
        """A 1-D array histogram honours the requested bin count."""
        fig, ax, hist = charts.histogram([1, 1, 2, 3, 3, 3], bins=3)
        assert len(ax.patches) == 3, f"expected 3 bins, got {len(ax.patches)}"

    def test_dataset_input_drops_nodata(self):
        """A pyramids Dataset is histogrammed over its first band with nodata excluded."""
        from pyramids.dataset import Dataset

        arr = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
        finite = charts._as_finite_array(ds)
        assert sorted(finite.tolist()) == [1.0, 2.0, 3.0], f"nodata not dropped: {finite}"
        fig, ax, hist = charts.histogram(ds, bins=3)
        assert len(ax.patches) == 3

    def test_2d_values_overlaid(self):
        """A 2-D array draws an overlaid histogram per column (one colour per series, as cleopatra needs)."""
        vals = np.column_stack([[1, 2, 3, 4], [2, 3, 4, 5]])
        out = charts._as_finite_array(vals)
        assert out.shape == (4, 2), f"2-D input should be preserved, got {out.shape}"
        fig, ax, hist = charts.histogram(vals, bins=4, color=["#1f77b4", "#ff7f0e"])
        assert ax.patches, "overlaid histograms should add patches"

    def test_as_finite_array_passthrough_drops_nonfinite_only_for_dataset(self):
        """A raw list is returned as-is by np.asarray (no nodata/finite filtering)."""
        out = charts._as_finite_array([1.0, np.nan, 3.0])
        assert out.shape == (3,) and np.isnan(out[1]), "raw arrays must pass through unfiltered"

    def test_as_finite_array_dataset_without_nodata(self):
        """A dataset declaring no nodata keeps every finite cell (skips the nodata filter)."""
        from types import SimpleNamespace

        ds = SimpleNamespace(read_array=lambda band=0: np.array([[1.0, 2.0], [3.0, 4.0]]),
                             no_data_value=[None])
        out = charts._as_finite_array(ds)
        assert sorted(out.tolist()) == [1.0, 2.0, 3.0, 4.0], f"all cells should be kept, got {out}"


class TestStatistics:
    """Tests for charts.statistics (DC.5) and the _field_values helper."""

    def test_basic_summary(self):
        """count/min/max/mean and the default quartiles are computed over the finite values."""
        s = charts.statistics([1, 2, 3, 4])
        assert (s["count"], s["min"], s["max"], s["mean"]) == (4, 1.0, 4.0, 2.5)
        assert s["q50"] == 2.5, f"median should be q50=2.5, got {s.get('q50')}"
        assert set(s) == {"count", "min", "max", "mean", "std", "q25", "q50", "q75"}

    def test_custom_quantiles(self):
        """Requested quantiles appear under q<pct> keys."""
        s = charts.statistics(range(101), quantiles=(0.1, 0.9))
        assert s["q10"] == 10.0 and s["q90"] == 90.0
        assert "q50" not in s, "only the requested quantiles should be present"

    def test_drops_nonfinite(self):
        """NaN/inf are excluded from the summary."""
        s = charts.statistics([1.0, np.nan, 3.0, np.inf])
        assert s["count"] == 2 and s["mean"] == 2.0

    def test_geodataframe_column(self):
        """A GeoDataFrame column is summarised when `column` is given."""
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"pop": [10.0, 20.0, 30.0]},
            geometry=[Point(i, i) for i in range(3)],
            crs=4326,
        )
        assert charts.statistics(gdf, column="pop")["mean"] == 20.0

    def test_dataset_band(self):
        """A pyramids Dataset is summarised over its first band, nodata dropped."""
        from pyramids.dataset import Dataset

        arr = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
        s = charts.statistics(ds)
        assert s["count"] == 3 and s["min"] == 1.0 and s["max"] == 3.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no finite values"):
            charts.statistics([np.nan, np.inf])

    def test_missing_column_raises(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame({"pop": [1.0]}, geometry=[Point(0, 0)], crs=4326)
        with pytest.raises(KeyError, match="nope"):
            charts.statistics(gdf, column="nope")

    def test_exported_top_level(self):
        import digitalearth

        assert digitalearth.statistics is charts.statistics
        assert "statistics" in digitalearth.__all__


class TestFigOf:
    """Tests for the _fig_of helper."""

    def test_none_returns_none(self):
        """_fig_of(None) is None (a fresh figure will be created downstream)."""
        assert charts._fig_of(None) is None

    def test_axes_returns_its_figure(self):
        """_fig_of(ax) returns the figure owning that axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        assert charts._fig_of(ax) is fig
