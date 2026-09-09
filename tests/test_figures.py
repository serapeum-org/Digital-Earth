"""Tests for digitalearth.static.figures — the static backend's axes-to-figure helper.

Split out of ``tests/test_arrays.py`` when ``digitalearth._arrays`` was divided: the numpy helpers went to
:mod:`digitalearth.base.arrays`, and ``fig_of`` — the only part needing matplotlib — went to the static
backend.
"""
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from digitalearth.static.figures import fig_of  # noqa: E402


class TestFigOf:
    """Tests for fig_of."""

    def test_returns_none_for_none(self):
        """fig_of(None) returns None.

        Test scenario:
            No axes means no owning figure, so the helper short-circuits to None.
        """
        assert fig_of(None) is None, "fig_of(None) should be None"

    def test_returns_owning_figure(self):
        """fig_of(ax) returns the figure that owns ax.

        Test scenario:
            An axes created from a figure should report that exact figure object.
        """
        fig, ax = plt.subplots()
        try:
            assert fig_of(ax) is fig, "fig_of(ax) should return the owning figure"
        finally:
            plt.close(fig)
