"""`clip=(vmin, vmax)` pins a graduated scheme's classes to a value window (#378).

Without a window a few outliers stretch an `equal_interval`/`natural_breaks` scheme so every real value bunches
into one class. `Scale.breaks_of(..., clip=)` and `classify_band(..., clip=)` instead make the classes span the
window exactly: every cell is clamped into `[vmin, vmax]` and the bounds are added to the sample, so the edges
reach `vmin` and `vmax` even when no cell does, and an outlier lands in an end class rather than moving the
edges. Two bands given the same window therefore cut the same class domain — a comparable scale across maps. The
window applies to a named graduated scheme alone — categorical codes and an explicit edge list have no window to
pin, so the clip is left off there and never raises, which lets a pooled-scale `facet` pass `vmin`/`vmax`
without conflict.

Every expected value is computed independently of the classifier under test: `equal_interval` edges split the
window by hand, and the quantile case asserts only the pinned outer edges and the window bound, not the
classifier's interior cut.
"""

import numpy as np

from digitalearth.base.raster_classes import classify_band
from digitalearth.base.spec import Scale


class TestBreaksOfPinsTheWindow:
    """`Scale.breaks_of(values, scheme, k, clip=...)` makes the classes span the window, not the data."""

    def test_equal_interval_splits_the_window_not_the_data(self):
        """Data well inside `(0, 100)` still cuts two equal halves of 0..100, not of its own 5..40 extent."""
        edges = Scale.breaks_of(
            [5, 10, 15, 20, 40], "equal_interval", 2, clip=(0.0, 100.0)
        )
        assert edges == (0.0, 50.0, 100.0), edges

    def test_an_outlier_lands_in_the_top_class_without_stretching(self):
        """A 999 outlier is clamped into the window, so the edges stay at the window bounds."""
        edges = Scale.breaks_of(
            [5, 10, 15, 20, 40, 999], "equal_interval", 2, clip=(0.0, 100.0)
        )
        assert edges == (0.0, 50.0, 100.0), edges

    def test_without_the_window_the_outlier_stretches_the_classes(self):
        """The same data with no clip lets 999 pull the top edge — the #378 complaint, reproduced."""
        edges = Scale.breaks_of([5, 10, 15, 20, 40, 999.0], "equal_interval", 2)
        assert edges[-1] == 999.0 and edges[1] > 40.0, edges

    def test_an_open_upper_bound_pins_only_the_floor(self):
        """`clip=(0, None)` pins the floor to 0 and leaves the top at the data's own max (40)."""
        edges = Scale.breaks_of(
            [5, 10, 15, 20, 40.0], "equal_interval", 2, clip=(0.0, None)
        )
        assert edges == (0.0, 20.0, 40.0), edges

    def test_a_window_excluding_every_cell_pins_the_window_and_does_not_raise(self):
        """Every cell above the window clamps into the top class; the edges still span the window, no error."""
        edges = Scale.breaks_of([5000, 6000.0], "equal_interval", 2, clip=(0.0, 100.0))
        assert edges == (0.0, 50.0, 100.0), edges

    def test_quantiles_pin_the_outer_edges_to_the_window(self):
        """A quantile cut's outer edges are the window bounds, and no edge escapes the window.

        Test scenario:
            0, 100 are the pinned ends; the interior cut is the classifier's business, but nothing may sit
            outside `[0, 100]`, which is what makes the colorbar span the requested limits.
        """
        edges = Scale.breaks_of(
            [5, 10, 15, 20, 40, 999.0], "quantiles", 3, clip=(0.0, 100.0)
        )
        assert edges[0] == 0.0 and edges[-1] == 100.0, edges
        assert all(0.0 <= edge <= 100.0 for edge in edges), edges

    def test_masked_cells_are_dropped_before_clamping(self):
        """A masked outlier is not classified, so it cannot reach the window as a clamped value."""
        band = np.ma.masked_array([1, 2, 3, 999.0], mask=[False, False, False, True])
        edges = Scale.breaks_of(band, "equal_interval", 2, clip=(0.0, 10.0))
        assert edges == (0.0, 5.0, 10.0), edges


class TestClassifyBandClip:
    """`classify_band(values, scheme, k, cmap, clip=...)` pins the edges and the recorded scale domain."""

    def test_a_graduated_scheme_spans_the_window(self):
        """`clip=(0, 100)` gives the band edges 0, 50, 100, the same as `breaks_of`."""
        classes = classify_band(
            np.array([5, 10, 15, 20, 40, 999.0]),
            "equal_interval",
            2,
            "viridis",
            clip=(0.0, 100.0),
        )
        assert classes.edges == (0.0, 50.0, 100.0), classes.edges

    def test_the_recorded_scale_domain_is_the_window(self):
        """The recorded scale spans the requested limits — the colorbar reads 0..100, comparable across maps.

        Test scenario:
            Data 5..40 inside a 0..100 window: the legend domain is the window, not the surviving 5..40.
        """
        classes = classify_band(
            np.array([5, 10, 15, 20, 40.0]),
            "equal_interval",
            2,
            "viridis",
            clip=(0.0, 100.0),
        )
        assert (classes.scale.vmin, classes.scale.vmax) == (0.0, 100.0), (
            classes.scale.vmin,
            classes.scale.vmax,
        )

    def test_categorical_ignores_the_window_and_does_not_raise(self):
        """Categorical has no window to pin, so a clip is ignored rather than refused (facet relies on this).

        Test scenario:
            Codes 1, 2, 3 with a window of 0..1 still key all three codes — the window touches nothing.
        """
        classes = classify_band(
            np.array([1.0, 2.0, 2.0, 3.0]), "categorical", None, None, clip=(0.0, 1.0)
        )
        assert list(classes.scale.categories) == [1, 2, 3], classes.scale.categories

    def test_explicit_edges_ignore_the_window(self):
        """An explicit edge list already fixes the domain; a clip leaves the edges untouched."""
        classes = classify_band(
            np.arange(25.0), [0, 10, 20], None, "viridis", clip=(0.0, 5.0)
        )
        assert classes.edges == (0.0, 10.0, 20.0), classes.edges
