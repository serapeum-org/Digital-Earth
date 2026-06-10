"""Tests for digitalearth._arrays — shared array helpers (PA-1).

Covers every public/private helper in the module: ``fig_of``, ``mask_nodata``, ``finite``, ``read_masked_band``
and ``_band_nodata``. The dataset-reading helpers are exercised against a small in-memory fake so no real raster
or filesystem access is needed.
"""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from digitalearth._arrays import (  # noqa: E402
    NAN_REDUCERS,
    _band_nodata,
    finite,
    fig_of,
    mask_nodata,
    read_masked_band,
)


class _FakeDataset:
    """Minimal pyramids-Dataset stand-in: one 2-D band plus a per-band nodata tuple.

    Args:
        array: The 2-D band returned by ``read_array`` regardless of the requested index.
        no_data_value: The per-band nodata tuple/list (or ``None`` to omit the attribute behaviour).
    """

    def __init__(self, array, no_data_value):
        self._array = np.asarray(array)
        self.no_data_value = no_data_value
        self.requested_band = None

    def read_array(self, band=0):
        """Record the requested 0-based band index and return the stored array."""
        self.requested_band = band
        return self._array


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


class TestMaskNodata:
    """Tests for mask_nodata."""

    def test_replaces_exact_match_with_nan(self):
        """mask_nodata nulls cells exactly equal to the sentinel.

        Test scenario:
            -9999 sentinel becomes NaN; neighbouring real values are untouched.
        """
        out = mask_nodata(np.array([1.0, -9999.0, 3.0]), -9999.0)
        assert np.isnan(out[1]), "sentinel cell should be NaN"
        assert out[0] == 1.0 and out[2] == 3.0, f"non-sentinel cells changed: {out}"

    def test_none_nodata_is_float_passthrough(self):
        """mask_nodata(arr, None) returns a float64 copy unchanged.

        Test scenario:
            With no sentinel, the only effect is the dtype cast to float64.
        """
        out = mask_nodata(np.array([1, 2, 3]), None)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_exact_compare_keeps_near_sentinel_values(self):
        """mask_nodata uses exact (not tolerant) comparison.

        Test scenario:
            A value 0.01% away from the sentinel must survive — proving exact, not isclose, semantics.
        """
        near = -9999.0 * (1 + 1e-4)
        out = mask_nodata(np.array([near, -9999.0]), -9999.0)
        assert not np.isnan(out[0]), "value near the sentinel must be preserved (exact compare)"
        assert np.isnan(out[1]), "exact sentinel must be masked"

    def test_2d_shape_preserved(self):
        """mask_nodata preserves array shape.

        Test scenario:
            A 2-D grid keeps its shape; only matching cells flip to NaN.
        """
        out = mask_nodata(np.array([[1.0, 0.0], [0.0, 2.0]]), 0.0)
        assert out.shape == (2, 2), f"shape changed: {out.shape}"
        assert np.isnan(out[0, 1]) and np.isnan(out[1, 0]), "zero cells should be NaN"


class TestFinite:
    """Tests for finite."""

    def test_drops_nan_and_inf_and_flattens(self):
        """finite removes NaN/inf and returns a 1-D array.

        Test scenario:
            A 2-D array with NaN and inf yields only the finite values, flattened in row-major order.
        """
        out = finite(np.array([[1.0, np.nan], [np.inf, 4.0]]))
        np.testing.assert_array_equal(out, [1.0, 4.0])
        assert out.ndim == 1, f"expected 1-D output, got {out.ndim}-D"

    def test_all_finite_passthrough(self):
        """finite returns all values when nothing is dropped.

        Test scenario:
            A fully-finite input is returned (flattened) with the same values.
        """
        out = finite(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_empty_when_all_nonfinite(self):
        """finite returns an empty array when every value is non-finite.

        Test scenario:
            All-NaN/inf input collapses to an empty 1-D array.
        """
        out = finite(np.array([np.nan, np.inf, -np.inf]))
        assert out.size == 0, f"expected empty result, got {out}"


class TestBandNodata:
    """Tests for _band_nodata."""

    def test_reads_indexed_value(self):
        """_band_nodata returns the per-band sentinel at the given 0-based index.

        Test scenario:
            Index 1 of a two-band nodata tuple returns the second entry.
        """
        ds = _FakeDataset([[0.0]], no_data_value=(-1.0, -2.0))
        assert _band_nodata(ds, 1) == -2.0, "should read the second band's nodata"

    def test_missing_attribute_returns_none(self):
        """_band_nodata returns None when no_data_value is empty/falsey.

        Test scenario:
            An empty tuple yields None rather than raising.
        """
        ds = _FakeDataset([[0.0]], no_data_value=())
        assert _band_nodata(ds, 0) is None, "empty nodata tuple should give None"

    def test_out_of_range_returns_none(self):
        """_band_nodata tolerates an out-of-range index.

        Test scenario:
            Asking for band index 5 of a one-element tuple returns None, not IndexError.
        """
        ds = _FakeDataset([[0.0]], no_data_value=(-1.0,))
        assert _band_nodata(ds, 5) is None, "out-of-range index should give None"

    def test_none_entry_returns_none(self):
        """_band_nodata returns a stored None entry as-is.

        Test scenario:
            A band whose nodata is None returns None.
        """
        ds = _FakeDataset([[0.0]], no_data_value=(None,))
        assert _band_nodata(ds, 0) is None, "None entry should return None"


class TestNanReducers:
    """Tests for the NAN_REDUCERS registry (PA-6)."""

    def test_exact_key_set(self):
        """NAN_REDUCERS exposes exactly the six NaN-aware reducer names.

        Test scenario:
            The registry's keys are mean/sum/median/min/max/std — no more, no less.
        """
        assert set(NAN_REDUCERS) == {"mean", "sum", "median", "min", "max", "std"}, (
            f"unexpected reducer keys: {sorted(NAN_REDUCERS)}"
        )

    @pytest.mark.parametrize("name, expected", [
        ("mean", 2.0),
        ("sum", 4.0),
        ("median", 2.0),
        ("min", 1.0),
        ("max", 3.0),
        ("std", 1.0),
    ])
    def test_reducers_are_nan_aware(self, name, expected):
        """Each reducer ignores NaN values.

        Args:
            name: Reducer key under test.
            expected: Result of applying it to [1, nan, 3].

        Test scenario:
            Reducing [1, nan, 3] yields the same value as reducing [1, 3], proving NaN is skipped.
        """
        result = float(NAN_REDUCERS[name](np.array([1.0, np.nan, 3.0])))
        assert result == pytest.approx(expected), f"{name} on [1, nan, 3] gave {result}, expected {expected}"

    def test_timeseries_reducers_are_a_subset(self):
        """TimeSeries._REDUCERS draws its functions from NAN_REDUCERS.

        Test scenario:
            Every TimeSeries reducer name is in NAN_REDUCERS and maps to the same callable object.
        """
        from digitalearth.temporal.timeseries import TimeSeries

        for name, func in TimeSeries._REDUCERS.items():
            assert func is NAN_REDUCERS[name], f"{name} not sourced from the shared registry"

    def test_quadtree_agg_is_superset_with_count(self):
        """map._QUADTREE_AGG is NAN_REDUCERS plus a special 'count'.

        Test scenario:
            Every NAN_REDUCERS entry appears (same object) in _QUADTREE_AGG, which adds only 'count'=len.
        """
        from digitalearth.scene.maps.vector import _QUADTREE_AGG

        for name, func in NAN_REDUCERS.items():
            assert _QUADTREE_AGG[name] is func, f"{name} differs from the shared registry"
        assert set(_QUADTREE_AGG) - set(NAN_REDUCERS) == {"count"}, "quadtree should add only 'count'"
        assert _QUADTREE_AGG["count"] is len, "'count' should be the builtin len"


class TestReadMaskedBand:
    """Tests for read_masked_band."""

    def test_reads_one_based_band(self):
        """read_masked_band converts 1-based band to the 0-based read_array index.

        Test scenario:
            band=1 must request internal index 0 from the dataset.
        """
        ds = _FakeDataset([[1.0, 2.0]], no_data_value=(None,))
        read_masked_band(ds, band=1)
        assert ds.requested_band == 0, f"band=1 should read index 0, read {ds.requested_band}"

    def test_masks_band_nodata(self):
        """read_masked_band nulls the band's nodata cells.

        Test scenario:
            A grid with a -1 sentinel returns that cell as NaN, others preserved, as float64.
        """
        ds = _FakeDataset([[5.0, -1.0], [-1.0, 8.0]], no_data_value=(-1.0,))
        out = read_masked_band(ds, band=1)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        assert np.isnan(out[0, 1]) and np.isnan(out[1, 0]), "sentinel cells should be NaN"
        assert out[0, 0] == 5.0 and out[1, 1] == 8.0, f"real values changed: {out}"

    def test_no_nodata_leaves_values(self):
        """read_masked_band leaves values intact when the band has no sentinel.

        Test scenario:
            A None nodata band returns the float cast of the array, nothing masked.
        """
        ds = _FakeDataset([[1.0, 2.0, 3.0]], no_data_value=(None,))
        out = read_masked_band(ds, band=1)
        np.testing.assert_array_equal(out, [[1.0, 2.0, 3.0]])
