"""Tests for digitalearth.interactive.charts — HoloViz/Bokeh charts (DC.6).

The lazy-import contract runs without the engine; the chart builders ``importorskip`` holoviews and run in the
``interactive`` pixi env.
"""
import sys

import pytest

from digitalearth.interactive import charts as icharts


class TestLazyImport:
    """The module imports without holoviews; builders fail actionably without it."""

    def test_module_imports_without_engine(self):
        for name in ("histogram", "scatter", "bar", "line", "bar_by", "line_by"):
            assert hasattr(icharts, name), f"interactive.charts.{name} should exist"

    def test_missing_engine_raises_actionable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "holoviews", None)  # makes `import holoviews` raise
        with pytest.raises(ImportError, match=r"digitalearth\[interactive\]"):
            icharts.histogram([1, 2, 3])


class TestInteractiveCharts:
    """The chart builders return the expected HoloViews elements (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("holoviews")

    @pytest.fixture()
    def gdf(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        return gpd.GeoDataFrame(
            {"cat": ["a", "a", "b"], "year": [2000, 2000, 2010], "a_val": [1.0, 2.0, 3.0], "b_val": [3.0, 2.0, 1.0]},
            geometry=[Point(i, i) for i in range(3)],
            crs=4326,
        )

    def test_histogram(self):
        import holoviews as hv

        assert isinstance(icharts.histogram([1, 1, 2, 3, 3, 3], bins=3), hv.Histogram)

    def test_histogram_by_column(self, gdf):
        import holoviews as hv

        assert isinstance(icharts.histogram(gdf, column="a_val", bins=3), hv.Histogram)

    def test_scatter_arrays(self):
        import holoviews as hv

        assert isinstance(icharts.scatter([1, 2, 3], [4, 5, 6]), hv.Scatter)

    def test_scatter_by_column(self, gdf):
        import holoviews as hv

        assert isinstance(icharts.scatter("a_val", "b_val", data=gdf), hv.Scatter)

    def test_bar_and_line(self):
        import holoviews as hv

        assert isinstance(icharts.bar([0, 1, 2], [3, 1, 4]), hv.Bars)
        assert isinstance(icharts.line([0, 1, 2], [0, 1, 4]), hv.Curve)

    def test_bar_by_and_line_by(self, gdf):
        import holoviews as hv

        assert isinstance(icharts.bar_by(gdf, "cat", "a_val", agg="sum"), hv.Bars)
        assert isinstance(icharts.line_by(gdf, "year", "a_val", agg="mean"), hv.Curve)
