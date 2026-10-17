"""`clip=(vmin, vmax)` confines a graduated scheme to a value window (#378).

A few outliers can stretch an `equal_interval`/`natural_breaks` scheme so every real value bunches into one
class. `Scale.breaks_of(..., clip=)` and `classify_band(..., clip=)` cut the classes over only the in-range
values instead; out-of-range cells clamp into the end classes at draw time. The window applies to a named
graduated scheme alone — categorical codes and an explicit edge list have no window to cut, so the clip is
left off there and never raises, which is what lets a pooled-scale `facet` pass `vmin`/`vmax` without conflict.

Every expected value here is computed independently of the classifier under test (numpy over the in-range
subset), so the assertions cannot agree with the code by construction.
"""

import numpy as np
import pytest

from digitalearth.base.raster_classes import classify_band
from digitalearth.base.spec import Scale

OUTLIER = [0.0, 1.0, 2.0, 3.0, 4.0, 999.0]


class TestBreaksOfClip:
    """`Scale.breaks_of(values, scheme, k, clip=...)` cuts the classes inside the window only."""

    def test_the_window_bounds_equal_interval_edges(self):
        """`clip=(0, 4)` cuts two equal intervals over 0..4, ignoring the 999 outlier.

        Test scenario:
            Edges computed by hand are 0, 2, 4; the classifier must return exactly those.
        """
        edges = Scale.breaks_of(OUTLIER, "equal_interval", 2, clip=(0.0, 4.0))
        assert edges == (0.0, 2.0, 4.0), edges

    def test_without_the_window_the_outlier_stretches_the_classes(self):
        """The same data with no clip lets 999 pull the top edge — the #378 complaint, reproduced."""
        edges = Scale.breaks_of(OUTLIER, "equal_interval", 2)
        assert edges[-1] == 999.0 and edges[1] > 4.0, edges

    def test_an_open_lower_bound_is_a_ceiling_only(self):
        """`clip=(None, 4)` keeps every value up to 4 and still drops the outlier."""
        edges = Scale.breaks_of(OUTLIER, "equal_interval", 2, clip=(None, 4.0))
        assert edges == (0.0, 2.0, 4.0), edges

    def test_quantiles_are_cut_over_the_in_range_values_only(self):
        """A windowed quantile cut equals quantiles of the in-range subset — clamped outliers do not skew it.

        Test scenario:
            0..10 with a 10000 outlier, three quantile classes, window 0..10: the edges must match the
            classifier run on just the in-range values, proving the outlier took no part in the quantiles.
        """
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10000.0])
        windowed = Scale.breaks_of(data, "quantiles", 3, clip=(0.0, 10.0))
        in_range = Scale.breaks_of(data[data <= 10.0], "quantiles", 3)
        assert windowed == in_range, (windowed, in_range)

    def test_a_window_holding_no_value_is_refused_by_name(self):
        """A window excluding every cell leaves nothing to classify, and the classifier says so."""
        with pytest.raises(ValueError, match="classify"):
            Scale.breaks_of([0.0, 1.0, 2.0], "equal_interval", 2, clip=(100.0, 200.0))


class TestClassifyBandClip:
    """`classify_band(values, scheme, k, cmap, clip=...)` threads the window to the graduated classifier."""

    def test_a_graduated_scheme_honours_the_window(self):
        """`clip=(0, 4)` gives the band edges 0, 2, 4, the same as `breaks_of`."""
        classes = classify_band(
            np.array(OUTLIER), "equal_interval", 2, "viridis", clip=(0.0, 4.0)
        )
        assert classes.edges == (0.0, 2.0, 4.0), classes.edges

    def test_categorical_ignores_the_window_and_does_not_raise(self):
        """Categorical has no window to cut, so a clip is ignored rather than refused (facet relies on this).

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
