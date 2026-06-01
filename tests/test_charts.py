"""Tests for digitalearth.charts — non-map x–y charts (line, bar, histogram)."""
import numpy as np
import pytest
from matplotlib.axes import Axes

from digitalearth import charts


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
