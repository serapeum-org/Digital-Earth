"""Direct tests for digitalearth.base.chartdata — the backend-neutral chart data preparation helpers.

These four helpers were lifted out of the static backend's ``charts`` module precisely because the
interactive backend was reaching across the backend boundary to import them. ``tests/test_charts.py`` still
exercises them *through* the matplotlib renderer; this file pins the contract at the unit level instead —
return types, dtypes, key ordering and the two deliberately different non-finite policies
(:func:`column_or_array` keeps non-finite values so paired inputs stay index-aligned, :func:`field_values`
drops them) — because those are the parts every backend depends on and no renderer test can assert.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from digitalearth.base.chartdata import (
    as_finite_array,
    column_or_array,
    field_values,
    grouped_series,
)


@pytest.fixture(scope="function")
def frame():
    """A small DataFrame with numeric, non-finite, categorical and string columns.

    Returns:
        pandas.DataFrame: columns ``v`` (float with a NaN and an inf), ``i`` (int), ``cat`` (category
        labels) and ``s`` (non-numeric strings).
    """
    return pd.DataFrame(
        {
            "v": [1.0, np.nan, 3.0, np.inf],
            "i": [1, 2, 3, 4],
            "cat": ["b", "a", "b", "a"],
            "s": ["x", "y", "z", "w"],
        }
    )


class TestColumnOrArray:
    """Tests for column_or_array."""

    @pytest.mark.parametrize(
        "data", [None, pd.DataFrame({"v": [1.0]})], ids=["no-data", "with-data"]
    )
    def test_none_value_passes_through(self, data):
        """A None value returns None whether or not data is supplied.

        Args:
            data: The optional (Geo)DataFrame context.

        Test scenario:
            Optional paired inputs (``color_by``/``size_by``) stay absent rather than becoming an empty
            array, so the caller can test them with ``is None``.
        """
        assert column_or_array(data, None) is None, "a None value must stay None"

    def test_column_name_resolves_to_float64(self, frame):
        """A string naming a column reads that column as float64.

        Test scenario:
            An integer column requested by name comes back as float64, so downstream numeric code never has
            to branch on dtype.
        """
        out = column_or_array(frame, "i")
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_keeps_non_finite_values_index_aligned(self, frame):
        """Column resolution preserves NaN/inf and row order.

        Test scenario:
            This is the documented difference from field_values: paired scatter inputs must keep every row
            so x/y/color_by stay index-aligned, even where a value is NaN or inf.
        """
        out = column_or_array(frame, "v")
        assert out.shape == (4,), f"every row must survive, got shape {out.shape}"
        assert np.isnan(out[1]), "NaN must be preserved, not dropped"
        assert np.isposinf(out[3]), "inf must be preserved, not dropped"

    def test_missing_column_raises_keyerror(self, frame):
        """A column name absent from data raises KeyError naming the column.

        Test scenario:
            The message quotes the missing name and says where it was looked for.
        """
        with pytest.raises(KeyError, match="nope") as exc_info:
            column_or_array(frame, "nope")
        assert "feature attributes" in str(exc_info.value), (
            f"message should say where the column was looked for: {exc_info.value}"
        )

    def test_string_without_data_is_coerced_not_resolved(self):
        """Without data, a string is array-like input rather than a column name.

        Test scenario:
            ``column_or_array(None, "abc")`` cannot resolve a column, so numpy.asarray wins and yields a
            0-d string array — the branch that keeps the helper usable outside the data= plumbing.
        """
        out = column_or_array(None, "abc")
        assert out.shape == (), (
            f"a bare string should coerce to a 0-d array, got shape {out.shape}"
        )
        assert out.dtype.kind == "U", f"expected a unicode dtype, got {out.dtype}"

    def test_data_without_columns_attribute_falls_back_to_asarray(self):
        """Data lacking a ``columns`` attribute is not treated as a frame.

        Test scenario:
            A dict has no ``.columns``, so the string is coerced with numpy.asarray instead of being looked
            up as a column — the guard that stops mappings from being mistaken for DataFrames.
        """
        out = column_or_array({"a": [1.0, 2.0]}, "a")
        assert out.shape == (), (
            f"the dict should not be indexed as a frame, got shape {out.shape}"
        )

    def test_array_like_is_coerced_verbatim(self):
        """A list value is coerced to an array without filtering or reordering.

        Test scenario:
            Values, order and length are preserved exactly, including a NaN in the middle.
        """
        out = column_or_array(None, [3.0, np.nan, 1.0])
        assert out.shape == (3,), f"expected three elements, got shape {out.shape}"
        assert out[0] == 3.0 and out[2] == 1.0, f"order changed: {out}"
        assert np.isnan(out[1]), "NaN must survive coercion"


class TestFieldValues:
    """Tests for field_values."""

    def test_column_is_reduced_to_finite_1d(self, frame):
        """A named column returns only its finite values as a 1-D float64 array.

        Test scenario:
            The NaN and the inf are dropped — the documented difference from column_or_array — and the
            result is flattened float64.
        """
        out = field_values(frame, "v")
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        assert out.ndim == 1, f"expected 1-D output, got {out.ndim}-D"
        np.testing.assert_array_equal(out, [1.0, 3.0])

    def test_missing_column_raises_keyerror(self, frame):
        """A column absent from a frame raises KeyError naming the column.

        Test scenario:
            Same actionable message as column_or_array, so both entry points fail identically.
        """
        with pytest.raises(KeyError, match="nope"):
            field_values(frame, "nope")

    def test_frame_without_column_flattens_every_value(self):
        """With column=None a frame is flattened to its finite values.

        Test scenario:
            A two-column numeric frame collapses to a single 1-D array of its finite entries.
        """
        out = field_values(pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]}))
        assert out.ndim == 1, f"expected 1-D output, got {out.ndim}-D"
        assert sorted(out.tolist()) == [1.0, 3.0, 4.0], f"unexpected values: {out}"

    def test_column_ignored_when_data_has_no_columns(self):
        """A column name is ignored for data that is not a frame.

        Test scenario:
            A plain array has no ``.columns``, so the helper falls through to as_finite_array rather than
            raising — the branch that lets ``histogram(array, column="x")`` degrade gracefully.
        """
        out = field_values(np.array([1.0, np.nan, 5.0]), "not-a-column")
        np.testing.assert_array_equal(out, [1.0, 5.0])

    def test_dataset_reads_its_first_band(self):
        """A Dataset duck-type contributes band 1 with nodata and non-finite cells dropped.

        Test scenario:
            A fake Dataset whose band holds a -9999 sentinel and a NaN yields only the real values, 1-D.
        """
        dataset = SimpleNamespace(
            no_data_value=(-9999.0,),
            read_array=lambda band=0: np.array([[1.0, -9999.0], [np.nan, 4.0]]),
        )
        out = field_values(dataset)
        assert sorted(out.tolist()) == [1.0, 4.0], f"nodata/NaN not dropped: {out}"

    def test_all_non_finite_column_returns_empty(self):
        """A column with no finite values returns an empty array rather than raising.

        Test scenario:
            An all-NaN column collapses to size 0, leaving the "empty" decision to the caller.
        """
        out = field_values(pd.DataFrame({"v": [np.nan, np.inf]}), "v")
        assert out.size == 0, f"expected an empty result, got {out}"


class TestGroupedSeries:
    """Tests for grouped_series."""

    def test_returns_sorted_keys_and_float_values(self, frame):
        """Aggregation returns a key list in pandas' ascending order and a float64 value array.

        Test scenario:
            Summing ``i`` by ``cat`` yields keys ["a", "b"] (sorted, not first-seen) and their sums as
            float64, which is what every backend's bar/line renderer consumes.
        """
        keys, values = grouped_series(frame, "cat", "i", "sum")
        assert keys == ["a", "b"], f"keys should be sorted ascending, got {keys}"
        assert isinstance(keys, list), f"keys should be a list, got {type(keys)}"
        assert values.dtype == np.float64, (
            f"expected float64 values, got {values.dtype}"
        )
        np.testing.assert_array_equal(values, [6.0, 4.0])

    def test_none_column_counts_rows(self, frame):
        """With column=None the group sizes are returned.

        Test scenario:
            Two rows per category yield [2.0, 2.0] regardless of the agg name, since size() is used.
        """
        keys, values = grouped_series(frame, "cat", None, "sum")
        assert keys == ["a", "b"], f"unexpected keys: {keys}"
        np.testing.assert_array_equal(values, [2.0, 2.0])

    @pytest.mark.parametrize(
        "agg, expected",
        [
            ("mean", [3.0, 2.0]),
            ("sum", [6.0, 4.0]),
            ("min", [2.0, 1.0]),
            ("max", [4.0, 3.0]),
            ("count", [2.0, 2.0]),
        ],
    )
    def test_named_aggregations(self, frame, agg, expected):
        """Each pandas aggregation name is forwarded and returned as floats.

        Args:
            frame: The fixture frame.
            agg: Aggregation name passed through to pandas.
            expected: Expected per-group values for column ``i`` grouped by ``cat``.

        Test scenario:
            Group "a" holds i=2,4 and group "b" holds i=1,3, so each reducer has a distinct answer.
        """
        _, values = grouped_series(frame, "cat", "i", agg)
        np.testing.assert_allclose(values, expected)

    def test_non_dataframe_raises_typeerror(self):
        """Data without a groupby method raises a TypeError naming the requirement.

        Test scenario:
            A list has no groupby, so the helper refuses up front instead of failing deep inside pandas.
        """
        with pytest.raises(TypeError, match="groupby") as exc_info:
            grouped_series([1, 2, 3], "cat", None, "sum")
        assert "(Geo)DataFrame" in str(exc_info.value), (
            f"message should name the expected input type: {exc_info.value}"
        )

    @pytest.mark.parametrize("agg", ["mean", "median", "std", "sum", "min"])
    def test_non_numeric_column_raises_actionable_typeerror(self, frame, agg):
        """Non-numeric aggregation surfaces one clear message for both failure shapes.

        Args:
            frame: The fixture frame.
            agg: Aggregation applied to the string column ``s``.

        Test scenario:
            mean/median/std are rejected by pandas outright while sum/min succeed on strings and then fail
            the float cast; both paths must raise TypeError naming the column and the agg.
        """
        with pytest.raises(TypeError, match=r"cannot aggregate column 's'") as exc_info:
            grouped_series(frame, "cat", "s", agg)
        assert f"agg={agg!r}" in str(exc_info.value), (
            f"message should name the agg: {exc_info.value}"
        )

    def test_original_error_is_chained(self, frame):
        """The underlying pandas/numpy error is kept as the cause.

        Test scenario:
            ``raise ... from err`` must preserve __cause__ so the real failure is still debuggable.
        """
        with pytest.raises(TypeError) as exc_info:
            grouped_series(frame, "cat", "s", "mean")
        assert exc_info.value.__cause__ is not None, (
            "the pandas error should be chained as __cause__"
        )

    def test_grouping_by_a_time_like_key_preserves_order(self):
        """Grouping by an ordered numeric key returns the keys ascending.

        Test scenario:
            Years supplied out of order come back sorted, which is what a line-by-time chart needs.
        """
        data = pd.DataFrame(
            {"year": [2010, 2000, 2010, 2000], "v": [4.0, 1.0, 6.0, 3.0]}
        )
        keys, values = grouped_series(data, "year", "v", "mean")
        assert keys == [2000, 2010], f"keys should be sorted ascending, got {keys}"
        np.testing.assert_allclose(values, [2.0, 5.0])


class TestAsFiniteArray:
    """Tests for as_finite_array."""

    def test_dataset_duck_type_is_masked_and_flattened(self):
        """An object with both read_array and no_data_value is read as a masked band.

        Test scenario:
            Both duck-type attributes present, so the nodata sentinel is nulled and the result flattened.
        """
        dataset = SimpleNamespace(
            no_data_value=(-1.0,),
            read_array=lambda band=0: np.array([[1.0, -1.0], [2.0, 3.0]]),
        )
        out = as_finite_array(dataset)
        assert out.ndim == 1, f"a Dataset band should be flattened, got {out.ndim}-D"
        assert sorted(out.tolist()) == [1.0, 2.0, 3.0], f"nodata not dropped: {out}"

    @pytest.mark.parametrize(
        "attrs, label",
        [
            ({"read_array": lambda band=0: np.array([1.0])}, "read_array only"),
            ({"no_data_value": (-1.0,)}, "no_data_value only"),
        ],
    )
    def test_half_a_duck_is_not_a_dataset(self, attrs, label):
        """An object exposing only one of the two attributes takes the asarray path.

        Args:
            attrs: The single duck-type attribute the stand-in exposes.
            label: Human-readable description of the half-duck under test.

        Test scenario:
            The guard is an ``and`` of both attributes; with only one present the object is handed to
            numpy.asarray, which wraps it as a 0-d object array rather than reading a band.
        """
        out = as_finite_array(SimpleNamespace(**attrs))
        assert out.dtype == object, (
            f"{label} should fall through to asarray, got dtype {out.dtype}"
        )

    def test_two_dimensional_array_keeps_its_shape(self):
        """A raw 2-D array is passed through unchanged for overlaid histograms.

        Test scenario:
            Shape (2, 2) survives, and the NaN is *not* dropped — only the Dataset branch filters.
        """
        out = as_finite_array(np.array([[1.0, np.nan], [3.0, 4.0]]))
        assert out.shape == (2, 2), f"2-D input should keep its shape, got {out.shape}"
        assert np.isnan(out[0, 1]), "the asarray path must not filter non-finite values"

    def test_list_is_coerced_without_filtering(self):
        """A plain list is coerced verbatim, non-finite values included.

        Test scenario:
            Three elements in, three elements out, with the NaN still in place.
        """
        out = as_finite_array([1.0, np.nan, 3.0])
        assert out.shape == (3,), f"expected three elements, got shape {out.shape}"
        assert np.isnan(out[1]), "the asarray path must not filter non-finite values"
