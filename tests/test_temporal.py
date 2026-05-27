"""Tests for T8.1 — TimeSeries and Climatology over a pyramids DatasetCollection."""

import numpy as np
import pytest

from digitalearth.temporal import Climatology, TimeSeries


@pytest.fixture
def collection():
    """A 6-member DatasetCollection built by repeating the acc4000 raster.

    Returns:
        DatasetCollection: six identical members (enough to exercise reduction/grouping).
    """
    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(["examples/data/acc4000.tif"] * 6)


class TestTimeSeries:
    """Tests for TimeSeries."""

    def test_values_length_matches_members(self, collection):
        """One reduced value is produced per collection member."""
        ts = TimeSeries(collection)
        assert ts.values().shape == (6,)

    def test_identical_members_give_constant_series(self, collection):
        """Repeating the same raster yields a constant mean series."""
        vals = TimeSeries(collection, reducer="mean").values()
        assert np.allclose(vals, vals[0])

    def test_unknown_reducer_raises(self, collection):
        """An unsupported reducer name is rejected."""
        with pytest.raises(ValueError, match="unknown reducer"):
            TimeSeries(collection, reducer="median")

    def test_plot_returns_axes(self, collection):
        """plot draws the series and returns a (fig, ax, ...) result on a line."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        result = TimeSeries(collection).plot(ax=ax)
        assert result[1] is ax
        assert ax.lines


class TestClimatology:
    """Tests for Climatology."""

    def test_climatology_groups(self, collection):
        """Grouping by a 3-label cycle collapses to three group means."""
        labels = ["djf", "jja", "son", "djf", "jja", "son"]
        groups, mean, low, high = Climatology(collection, labels).climatology()
        assert groups == ["djf", "jja", "son"]
        assert mean.shape == (3,)

    def test_label_length_mismatch_raises(self, collection):
        """Mismatched label length is rejected."""
        with pytest.raises(ValueError, match="labels length"):
            Climatology(collection, ["a", "b"])

    def test_plot_draws_line_and_plume(self, collection):
        """plot draws a mean line plus a min/max spread plume."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        labels = ["q1", "q2", "q3", "q1", "q2", "q3"]
        Climatology(collection, labels).plot(ax=ax)
        assert ax.lines
        assert ax.collections  # the envelope plume
