"""Tests for digitalearth.static.figures — the static backend's axes-to-figure helper.

Split out of ``tests/base/test_arrays.py`` when ``digitalearth._arrays`` was divided: the numpy helpers went to
:mod:`digitalearth.base.arrays`, and ``fig_of`` — the only part needing matplotlib — went to the static
backend. ``fig_of`` is a one-liner, so the coverage here is about its *contract* rather than its branches:
it must work for every kind of axes the backend actually creates (plain, gridded, inset, 3-D), it must be a
pure lookup that creates nothing, and ``None`` must be the only input that yields ``None``.
"""

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from digitalearth.static.figures import fig_of  # noqa: E402


@pytest.fixture
def closed_figures():
    """Close every figure a test leaves behind.

    Yields:
        None: the fixture only performs teardown, so a leaked figure cannot influence the next test's
        ``plt.get_fignums()`` assertions.
    """
    yield
    plt.close("all")


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

    @pytest.mark.parametrize("maker", ["add_subplot", "add_axes", "subplot_mosaic"])
    def test_every_axes_construction_reports_its_figure(self, maker, closed_figures):
        """Axes built by any of the figure's own factories report that figure.

        Args:
            maker: Name of the Figure method used to create the axes.
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            The static backend creates axes through several matplotlib entry points (subplots, explicit
            add_axes for colorbars, mosaics for grids); ``fig_of`` must resolve all of them identically.
        """
        fig = plt.figure()
        if maker == "add_subplot":
            ax = fig.add_subplot(1, 1, 1)
        elif maker == "add_axes":
            ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        else:
            ax = fig.subplot_mosaic("A")["A"]

        assert fig_of(ax) is fig, f"axes made with {maker} should report its own figure"

    def test_gridded_axes_all_share_one_figure(self, closed_figures):
        """Every axes of a multi-panel figure resolves to the same figure.

        Test scenario:
            A 2x2 grid — the shape ``grid()`` produces — yields four axes that all report the one parent
            figure, which is what lets ``shared_colorbar`` attach to a single canvas.
        """
        fig, axes = plt.subplots(2, 2)
        figures = {id(fig_of(ax)) for ax in axes.ravel()}
        assert figures == {id(fig)}, (
            "all four axes should report the single parent figure"
        )

    def test_distinct_figures_are_not_confused(self, closed_figures):
        """Axes from two different figures resolve to their own figures.

        Test scenario:
            The helper is a per-axes lookup, not a global "current figure" read, so a second figure created
            afterwards must not shadow the first axes' owner.
        """
        first, first_ax = plt.subplots()
        second, second_ax = plt.subplots()
        assert fig_of(first_ax) is first, (
            "the first axes should still report the first figure"
        )
        assert fig_of(second_ax) is second, (
            "the second axes should report the second figure"
        )

    def test_three_dimensional_axes_supported(self, closed_figures):
        """A 3-D axes reports its figure too.

        Test scenario:
            ``TexturedGlobe`` renders onto an Axes3D; the helper must not assume a 2-D axes.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        assert fig_of(ax) is fig, "an Axes3D should report its owning figure"

    def test_creates_no_figure(self, closed_figures):
        """fig_of never creates a figure as a side effect.

        Test scenario:
            Calling it with None while no figure exists must leave the pyplot figure registry empty — the
            helper is a pure lookup, so a chart helper can call it before deciding to draw.
        """
        plt.close("all")
        assert fig_of(None) is None, "fig_of(None) should be None"
        assert plt.get_fignums() == [], (
            f"fig_of must not create a figure, found {plt.get_fignums()}"
        )

    def test_result_is_stable_across_calls(self, closed_figures):
        """Repeated calls return the identical figure object.

        Test scenario:
            The lookup is pure, so two calls on the same axes give the same object — no copy, no rebuild.
        """
        fig, ax = plt.subplots()
        assert fig_of(ax) is fig_of(ax), (
            "repeated lookups should return the same object"
        )
