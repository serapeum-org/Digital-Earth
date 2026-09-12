"""Tests for digitalearth.base.arrays — shared array helpers (PA-1).

Covers every public/private helper in the module: ``ring_runs``, ``mask_nodata``, ``finite``,
``read_masked_band`` and ``_band_nodata``. ``fig_of`` moved to the static backend with the module split; see
``tests/test_figures.py``. The dataset-reading helpers are exercised against a small in-memory fake so no
real raster or filesystem access is needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from digitalearth.base.arrays import (  # noqa: E402
    NAN_REDUCERS,
    _band_nodata,
    finite,
    mask_nodata,
    read_masked_band,
    ring_runs,
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


class TestRingRuns:
    """Tests for ring_runs — the seam-safe visible-run splitter both globe tiers clip rings with."""

    def test_separate_stretches_come_back_in_ring_order(self):
        """Two visible stretches either side of a hidden one are two runs, in the order they occur.

        Test scenario:
            Neither stretch touches the seam, so this is the plain split every caller relied on before.
        """
        runs = [
            run.tolist() for run in ring_runs([False, True, True, False, True, False])
        ]
        assert runs == [[1, 2], [4]], f"expected two runs, got {runs}"

    def test_a_stretch_crossing_the_seam_is_one_run(self):
        """A visible stretch that runs off the end of the array and on at its start stays whole.

        Test scenario:
            The seam case both globes got wrong: split as a straight line, the stretch came back as two
            runs, and each was closed out to the horizon on its own — a spur from the first vertex.
        """
        runs = [
            run.tolist() for run in ring_runs([True, True, False, False, True, True])
        ]
        assert runs == [[4, 5, 0, 1]], (
            f"the wrapped stretch should be one run, got {runs}"
        )

    def test_every_run_is_flanked_by_hidden_vertices(self):
        """The vertex just before and just after each run, round the ring, is hidden.

        Test scenario:
            Callers find a run's two limb crossings from those neighbours, so none may be visible.
        """
        visible = np.array(
            [True, False, True, True, False, True, True, True, False, True]
        )
        count = visible.size
        for run in ring_runs(visible):
            assert not visible[(run[0] - 1) % count], (
                f"run {run.tolist()} has a visible predecessor"
            )
            assert not visible[(run[-1] + 1) % count], (
                f"run {run.tolist()} has a visible successor"
            )

    @pytest.mark.parametrize(
        "visible, expected",
        [
            pytest.param([True, True, True], [[0, 1, 2]], id="all-visible"),
            pytest.param([False, False, False], [], id="all-hidden"),
            pytest.param([], [], id="empty"),
            pytest.param([True], [[0]], id="one-visible-vertex"),
        ],
    )
    def test_boundary_masks(self, visible, expected):
        """A wholly visible ring is one run, a hidden or empty one has none.

        Args:
            visible: The visibility mask.
            expected: The runs it must yield.
        """
        runs = [run.tolist() for run in ring_runs(visible)]
        assert runs == expected, f"expected {expected}, got {runs}"

    def test_any_sized_boolean_sequence_is_accepted(self):
        """A numpy mask and a plain list of 0/1 give the same runs."""
        mask = np.array([1, 0, 1, 1], dtype=bool)
        from_array = [run.tolist() for run in ring_runs(mask)]
        from_list = [run.tolist() for run in ring_runs([1, 0, 1, 1])]
        assert from_array == from_list == [[2, 3, 0]], (
            f"both inputs should wrap the same run, got {from_array} and {from_list}"
        )


class TestMaskNodata:
    """Tests for mask_nodata."""

    def test_replaces_exact_match_with_nan(self):
        """mask_nodata nulls cells exactly equal to the sentinel.

        Test scenario:
            -9999 sentinel becomes NaN; neighbouring real values are untouched.
        """
        out = mask_nodata(np.array([1.0, -9999.0, 3.0]), -9999.0)
        assert np.isnan(out[1]), "sentinel cell should be NaN"
        assert out[0] == 1.0, f"the first non-sentinel cell changed: {out}"
        assert out[2] == 3.0, f"the last non-sentinel cell changed: {out}"

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
        assert not np.isnan(out[0]), (
            "value near the sentinel must be preserved (exact compare)"
        )
        assert np.isnan(out[1]), "exact sentinel must be masked"

    def test_2d_shape_preserved(self):
        """mask_nodata preserves array shape.

        Test scenario:
            A 2-D grid keeps its shape; only matching cells flip to NaN.
        """
        out = mask_nodata(np.array([[1.0, 0.0], [0.0, 2.0]]), 0.0)
        assert out.shape == (2, 2), f"shape changed: {out.shape}"
        assert np.isnan(out[0, 1]), f"the first zero cell should be NaN: {out}"
        assert np.isnan(out[1, 0]), f"the second zero cell should be NaN: {out}"

    def test_accepts_array_like_input(self):
        """mask_nodata coerces any array-like, not just ndarrays.

        Test scenario:
            A plain Python list of ints with an int sentinel is coerced to float64 and masked, so callers
            need not pre-convert whatever pyramids handed them.
        """
        out = mask_nodata([1, 2, 3], 2)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        assert np.isnan(out[1]), (
            "the integer sentinel should be masked after the float cast"
        )
        assert out[0] == 1.0, f"leading non-sentinel cell changed: {out}"
        assert out[2] == 3.0, f"trailing non-sentinel cell changed: {out}"

    def test_nan_sentinel_matches_nothing(self):
        """A NaN sentinel masks nothing, because NaN != NaN under an exact comparison.

        Test scenario:
            The documented consequence of exact-compare semantics: a dataset declaring NaN as its nodata
            leaves finite values untouched (its NaN cells are already NaN), so no real value is lost.
        """
        out = mask_nodata(np.array([1.0, np.nan, 3.0]), np.nan)
        assert out[0] == 1.0, f"leading finite value must survive a NaN sentinel: {out}"
        assert out[2] == 3.0, (
            f"trailing finite value must survive a NaN sentinel: {out}"
        )
        assert np.isnan(out[1]), "an already-NaN cell stays NaN"

    def test_empty_input_stays_empty(self):
        """mask_nodata on an empty array returns an empty float64 array.

        Test scenario:
            A zero-length band must not raise; the boundary case returns size 0 with the cast applied.
        """
        out = mask_nodata(np.array([]), -9999.0)
        assert out.size == 0, f"expected an empty result, got {out}"
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"

    def test_returns_a_copy_not_a_view(self):
        """mask_nodata does not mutate its input.

        Test scenario:
            The helper is called on arrays owned by a caller's Dataset, so masking must leave the original
            untouched — proven by checking the sentinel is still there afterwards.
        """
        source = np.array([1.0, -9999.0, 3.0])
        mask_nodata(source, -9999.0)
        assert source[1] == -9999.0, f"input array was mutated: {source}"


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

    def test_empty_input_returns_empty(self):
        """finite on an empty array returns an empty 1-D float64 array.

        Test scenario:
            The zero-length boundary must not raise; callers rely on ``size == 0`` to detect "nothing to
            plot" rather than on an exception.
        """
        out = finite(np.array([]))
        assert out.shape == (0,), f"expected an empty 1-D result, got shape {out.shape}"
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"

    def test_scalar_becomes_one_element_array(self):
        """finite promotes a scalar to a 1-element 1-D array.

        Test scenario:
            A 0-d input is ravelled to shape (1,), so downstream code can index it uniformly.
        """
        out = finite(3.0)
        assert out.shape == (1,), f"expected one value, got shape {out.shape}"
        assert out[0] == 3.0, f"expected [3.0], got {out}"

    def test_integer_input_is_cast_to_float(self):
        """finite casts integer input to float64.

        Test scenario:
            An int array has no NaN/inf to drop, but the dtype contract still applies so reducers behave
            identically for integer and float bands.
        """
        out = finite(np.array([[1, 2], [3, 4]], dtype="int32"))
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])


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

    def test_absent_attribute_returns_none(self):
        """_band_nodata returns None for an object with no no_data_value attribute at all.

        Test scenario:
            A duck-typed dataset that never declares nodata must not raise AttributeError — the getattr
            default is what makes the helper safe for the loose duck-typing the callers use.
        """
        assert _band_nodata(SimpleNamespace(), 0) is None, (
            "a missing attribute should give None"
        )

    def test_none_attribute_returns_none(self):
        """_band_nodata returns None when no_data_value itself is None.

        Test scenario:
            ``no_data_value=None`` is falsey, so the helper short-circuits before subscripting.
        """
        assert _band_nodata(SimpleNamespace(no_data_value=None), 0) is None, (
            "None nodata should give None"
        )

    def test_unsubscriptable_nodata_returns_none(self):
        """_band_nodata swallows the TypeError from a non-subscriptable nodata value.

        Test scenario:
            A dataset exposing a bare number instead of a per-band sequence would raise TypeError on
            ``ndv[index]``; the helper degrades to None rather than propagating it.
        """
        assert _band_nodata(SimpleNamespace(no_data_value=5), 0) is None, (
            "a scalar nodata should give None"
        )

    def test_mapping_nodata_missing_key_returns_none(self):
        """_band_nodata swallows the KeyError from a mapping without the requested band.

        Test scenario:
            A dict-shaped nodata keyed by band index returns None for an absent key instead of raising.
        """
        ds = SimpleNamespace(no_data_value={0: -1.0})
        assert _band_nodata(ds, 3) is None, "a missing mapping key should give None"

    def test_mapping_nodata_present_key_is_read(self):
        """_band_nodata reads a mapping-shaped nodata by band index.

        Test scenario:
            The lookup is a plain subscript, so a dict keyed by band index works as well as a tuple.
        """
        ds = SimpleNamespace(no_data_value={0: -1.0})
        assert _band_nodata(ds, 0) == -1.0, "an existing mapping key should be returned"

    def test_list_nodata_is_read(self):
        """_band_nodata accepts a list as well as a tuple.

        Test scenario:
            pyramids may hand back either sequence type; both index identically.
        """
        ds = _FakeDataset([[0.0]], no_data_value=[-3.0, -4.0])
        assert _band_nodata(ds, 1) == -4.0, "a list nodata should index like a tuple"


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

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("mean", 2.0),
            ("sum", 4.0),
            ("median", 2.0),
            ("min", 1.0),
            ("max", 3.0),
            ("std", 1.0),
        ],
    )
    def test_reducers_are_nan_aware(self, name, expected):
        """Each reducer ignores NaN values.

        Args:
            name: Reducer key under test.
            expected: Result of applying it to [1, nan, 3].

        Test scenario:
            Reducing [1, nan, 3] yields the same value as reducing [1, 3], proving NaN is skipped.
        """
        result = float(NAN_REDUCERS[name](np.array([1.0, np.nan, 3.0])))
        assert result == pytest.approx(expected), (
            f"{name} on [1, nan, 3] gave {result}, expected {expected}"
        )

    def test_timeseries_reducers_are_a_subset(self):
        """TimeSeries._REDUCERS draws its functions from NAN_REDUCERS.

        Test scenario:
            Every TimeSeries reducer name is in NAN_REDUCERS and maps to the same callable object.
        """
        from digitalearth.static.temporal.timeseries import TimeSeries

        for name, func in TimeSeries._REDUCERS.items():
            assert func is NAN_REDUCERS[name], (
                f"{name} not sourced from the shared registry"
            )

    def test_quadtree_agg_is_superset_with_count(self):
        """map._QUADTREE_AGG is NAN_REDUCERS plus a special 'count'.

        Test scenario:
            Every NAN_REDUCERS entry appears (same object) in _QUADTREE_AGG, which adds only 'count'=len.
        """
        from digitalearth.static.maps.vector import _QUADTREE_AGG

        for name, func in NAN_REDUCERS.items():
            assert _QUADTREE_AGG[name] is func, (
                f"{name} differs from the shared registry"
            )
        assert set(_QUADTREE_AGG) - set(NAN_REDUCERS) == {"count"}, (
            "quadtree should add only 'count'"
        )
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
        assert ds.requested_band == 0, (
            f"band=1 should read index 0, read {ds.requested_band}"
        )

    def test_masks_band_nodata(self):
        """read_masked_band nulls the band's nodata cells.

        Test scenario:
            A grid with a -1 sentinel returns that cell as NaN, others preserved, as float64.
        """
        ds = _FakeDataset([[5.0, -1.0], [-1.0, 8.0]], no_data_value=(-1.0,))
        out = read_masked_band(ds, band=1)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        assert np.isnan(out[0, 1]), f"the first sentinel cell should be NaN: {out}"
        assert np.isnan(out[1, 0]), f"the second sentinel cell should be NaN: {out}"
        assert out[0, 0] == 5.0, f"the first real value changed: {out}"
        assert out[1, 1] == 8.0, f"the second real value changed: {out}"

    def test_no_nodata_leaves_values(self):
        """read_masked_band leaves values intact when the band has no sentinel.

        Test scenario:
            A None nodata band returns the float cast of the array, nothing masked.
        """
        ds = _FakeDataset([[1.0, 2.0, 3.0]], no_data_value=(None,))
        out = read_masked_band(ds, band=1)
        np.testing.assert_array_equal(out, [[1.0, 2.0, 3.0]])

    def test_band_defaults_to_the_first(self):
        """read_masked_band defaults to band 1 (internal index 0).

        Test scenario:
            Called with no band argument, the helper must request index 0 — the default every single-band
            caller relies on.
        """
        ds = _FakeDataset([[1.0, 2.0]], no_data_value=(None,))
        read_masked_band(ds)
        assert ds.requested_band == 0, (
            f"the default band should read index 0, read {ds.requested_band}"
        )

    def test_second_band_uses_its_own_sentinel(self):
        """read_masked_band picks the sentinel matching the requested band.

        Test scenario:
            With per-band nodata (-1, -2), band=2 must mask only -2 and leave -1 alone — the off-by-one the
            1-based/0-based conversion exists to prevent.
        """
        ds = _FakeDataset([[-1.0, -2.0]], no_data_value=(-1.0, -2.0))
        out = read_masked_band(ds, band=2)
        assert ds.requested_band == 1, (
            f"band=2 should read index 1, read {ds.requested_band}"
        )
        assert out[0, 0] == -1.0, (
            f"band 1's sentinel must not be applied to band 2: {out}"
        )
        assert np.isnan(out[0, 1]), "band 2's own sentinel should be masked"

    def test_preserves_stored_shape(self):
        """read_masked_band returns the band with its stored 2-D shape.

        Test scenario:
            No flattening happens here — that is ``finite``'s job — so a 2x3 band comes back 2x3.
        """
        ds = _FakeDataset(
            np.arange(6, dtype="float64").reshape(2, 3), no_data_value=(None,)
        )
        out = read_masked_band(ds, band=1)
        assert out.shape == (2, 3), f"shape changed: {out.shape}"
