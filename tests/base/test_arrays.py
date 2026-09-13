"""Tests for digitalearth.base.arrays — shared array helpers (PA-1).

Covers every helper the module still exposes: ``ring_runs``, ``finite``, ``NAN_REDUCERS`` and
``read_masked_band``. ``fig_of`` moved to the static backend with the module split (see
``tests/static/test_figures.py``); ``mask_nodata`` and ``_band_nodata`` were removed with the masking change
that routed every raster read through pyramids, so there is nothing left of them to test. The dataset-reading
helper is exercised against a small in-memory fake so no real raster or filesystem access is needed.
"""

import numpy as np
import pytest

from digitalearth.base.arrays import (  # noqa: E402
    NAN_REDUCERS,
    finite,
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

    def read_array(self, band=0, masked=False):
        """Record the requested 0-based band index and return the stored array.

        ``masked=True`` returns a masked array built from the requested band's own sentinel, the way
        pyramids builds one — that is the call :func:`read_masked_band` makes, and the reason the sentinel
        never has to be compared against a value here.
        """
        self.requested_band = band
        if not masked:
            return self._array
        try:
            nodata = self.no_data_value[band]
        except (IndexError, TypeError, KeyError):
            nodata = None
        if nodata is None:
            return np.ma.asarray(self._array)
        return np.ma.masked_equal(self._array, nodata)


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


class TestReadMaskedBandOnPackedData:
    """Regression tests for #234 — a CF-packed band's nodata was never masked.

    ``read_array`` unpacks ``scale_factor``/``add_offset`` to physical units while ``no_data_value`` stays a
    stored sentinel, so the old ``arr == nodata`` comparison matched nothing: a stored ``-9999`` reads as
    ``-98.49`` at ``scale=0.01, offset=1.5``. These tests use a genuinely packed band, which is the only
    shape that tells the two readings apart.
    """

    @staticmethod
    def _packed_dataset():
        """Build a 2x2 int16 raster packed at ``scale=0.01, offset=1.5`` with a stored ``-9999`` nodata cell.

        Returns:
            pyramids.dataset.Dataset: the packed raster, whose physical values are
            ``[[2.5, 3.5], [4.5, -98.49]]`` and whose last cell is nodata.
        """
        from pyramids.dataset import Dataset, GeoReference

        ds = Dataset.from_array(
            np.array([[100, 200], [300, -9999]], dtype="int16"),
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
            no_data_value=-9999,
        )
        ds.scale, ds.offset = [0.01], [1.5]
        return ds

    def test_packed_nodata_cell_is_nan(self):
        """The nodata cell of a packed band comes back NaN, not as its unpacked sentinel.

        Test scenario:
            Before the fix this cell held -98.49 — the stored sentinel run through the packing recipe — and
            every downstream statistic, colour scale and colorbar counted it as data.
        """
        out = read_masked_band(self._packed_dataset(), band=1)
        assert np.isnan(out[1, 1]), (
            f"the packed band's nodata cell should be NaN, got {out[1, 1]!r}"
        )

    def test_packed_real_values_stay_physical(self):
        """The surviving cells keep their unpacked (physical) values.

        Test scenario:
            Masking through pyramids must not cost the unpacking — the values are the ones ``read_array``
            reports, only with the masked cell nulled.
        """
        out = read_masked_band(self._packed_dataset(), band=1)
        np.testing.assert_allclose(
            [out[0, 0], out[0, 1], out[1, 0]],
            [2.5, 3.5, 4.5],
            err_msg=f"physical values changed: {out}",
        )

    def test_matches_pyramids_own_masked_read(self):
        """The result equals ``read_array(masked=True)`` filled with NaN — the upstream-prescribed reading.

        Test scenario:
            Pins the helper to pyramids' own answer rather than to a hand-rolled mask, which is the whole
            point of the change.
        """
        ds = self._packed_dataset()
        expected = np.ma.filled(
            ds.read_array(band=0, masked=True).astype("float64"), np.nan
        )
        np.testing.assert_array_equal(read_masked_band(ds, band=1), expected)
