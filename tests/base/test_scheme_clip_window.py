"""`clip=(vmin, vmax)` pins a graduated scheme's classes to a value window (#378).

Without a window a few outliers stretch a scheme so every real value bunches into one class. With one, the
recorded domain is the window and a cell outside it clamps into an end class. The *interior* is the scheme's
own: `equal_interval` splits the window itself into equal parts (so the bins are equal over `[vmin, vmax]`
whatever the data holds), while a data-driven scheme (`quantiles`, `natural_breaks`, …) cuts its interior over
the clamped data and only has its outer edges pinned to the bounds — the bounds are never mixed into the
statistic. A reversed window is refused and a zero-width one is widened, as the continuous path treats explicit
limits. The window applies to a named graduated scheme alone — categorical codes and an explicit edge list have
no window to pin, so the clip is left off there and never raises, which lets a pooled-scale `facet` pass
`vmin`/`vmax` without conflict.

Expected values are computed independently of the classifier under test: `equal_interval` edges split the
window by hand, and the data-driven cases compare a clipped cut against the classifier run on the clamped data
without the window, so a difference can only be anchor pollution, not the classifier's own interior.
"""

import numpy as np
import pytest

from digitalearth.base.raster_classes import classify_band
from digitalearth.base.spec import Scale


class TestEqualIntervalPinsTheWindow:
    """`equal_interval` under a clip splits the window itself, not the surviving data."""

    def test_splits_the_window_not_the_data(self):
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
        assert edges[-1] == 999.0, edges
        assert edges[1] > 40.0, edges

    def test_an_open_upper_bound_pins_only_the_floor(self):
        """`clip=(0, None)` pins the floor to 0 and leaves the top at the data's own max (40)."""
        edges = Scale.breaks_of(
            [5, 10, 15, 20, 40.0], "equal_interval", 2, clip=(0.0, None)
        )
        assert edges == (0.0, 20.0, 40.0), edges

    def test_a_window_excluding_every_cell_keeps_the_class_count(self):
        """Every cell above the window clamps into the top class; the k classes still span the window."""
        edges = Scale.breaks_of([5000, 6000.0], "equal_interval", 4, clip=(0.0, 100.0))
        assert edges == (0.0, 25.0, 50.0, 75.0, 100.0), edges

    def test_both_bounds_open_is_a_no_op_window(self):
        """`clip=(None, None)` pins nothing — the edges match an unclipped cut."""
        data = [5, 10, 15, 20, 40.0]
        assert Scale.breaks_of(
            data, "equal_interval", 2, clip=(None, None)
        ) == Scale.breaks_of(data, "equal_interval", 2)


class TestDataDrivenSchemesKeepTheirInterior:
    """A data-driven scheme pins only its outer edges; its interior stays the statistic it names (M2)."""

    def test_quantile_interior_is_untouched_by_the_window(self):
        """The interior breaks equal the quantiles of the clamped data; only the outer edges are the bounds.

        Test scenario:
            A clipped cut is compared against the classifier on the clamped data with no window — the two
            synthetic bound points must not shift the interior, so only edges[0]/edges[-1] may differ.
        """
        windowed = Scale.breaks_of(
            [5, 10, 15, 20, 40], "quantiles", 3, clip=(0.0, 100.0)
        )
        clamped_only = Scale.breaks_of(
            np.clip([5, 10, 15, 20, 40], 0.0, 100.0), "quantiles", 3
        )
        assert windowed[1:-1] == clamped_only[1:-1], (windowed, clamped_only)
        assert (windowed[0], windowed[-1]) == (0.0, 100.0), windowed

    def test_the_class_count_is_preserved(self):
        """`quantiles k=4` under a window still cuts four classes — the bounds do not collapse them."""
        edges = Scale.breaks_of(
            [5, 10, 15, 20, 40, 60, 80], "quantiles", 4, clip=(0.0, 100.0)
        )
        assert len(edges) - 1 == 4, edges

    def test_a_window_holding_fewer_than_two_values_is_refused_by_name(self):
        """Every cell outside the window leaves one clamped value, which a data-driven scheme cannot cut."""
        with pytest.raises(ValueError, match=r"window \[0.0, 100.0\]"):
            Scale.breaks_of([5000, 6000.0], "quantiles", 4, clip=(0.0, 100.0))


class TestClipWindowValidation:
    """A reversed window is refused and a zero-width one is widened, as the continuous path does (M1, L1)."""

    def test_a_reversed_window_is_refused(self):
        """`clip=(80, 20)` is a transposed pair, refused with the same message explicit limits get."""
        with pytest.raises(ValueError, match="vmin <= vmax"):
            Scale.breaks_of([30.0, 50.0], "equal_interval", 2, clip=(80.0, 20.0))

    def test_a_zero_width_window_is_widened_for_equal_interval(self):
        """`clip=(5, 5)` widens to a unit span rather than raising, matching `from_limits`."""
        edges = Scale.breaks_of([1, 2, 3, 50.0], "equal_interval", 2, clip=(5.0, 5.0))
        assert edges == (5.0, 5.5, 6.0), edges

    def test_a_one_sided_window_degenerate_after_clamping_is_widened(self):
        """`clip=(5, None)` on data all below 5 clamps to one value, widened rather than left zero-width."""
        edges = Scale.breaks_of([1, 2, 3.0], "equal_interval", 2, clip=(5.0, None))
        assert edges == (5.0, 5.5, 6.0), edges


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

    def test_masked_cells_are_dropped_before_clamping(self):
        """A masked outlier is dropped, not clamped — a data-driven cut proves it took no part.

        Test scenario:
            The same band with and without the masked 999 must give the same quantiles; if the mask leaked,
            the clamped 999 (→10) would shift them.
        """
        band = np.ma.masked_array(
            [1, 2, 3, 4, 5, 999.0], mask=[False, False, False, False, False, True]
        )
        with_mask = classify_band(
            band, "quantiles", 2, "viridis", clip=(0.0, 10.0)
        ).edges
        without = classify_band(
            np.array([1, 2, 3, 4, 5.0]), "quantiles", 2, "viridis", clip=(0.0, 10.0)
        ).edges
        assert with_mask == without, (with_mask, without)

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
