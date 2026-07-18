"""Tests for digitalearth._symbology — categorical (distinct-value → colour) mapping (DC.8)."""
import numpy as np
import pandas as pd
import pytest

from digitalearth._symbology import (
    categorical_colors,
    is_null,
    nulls_to_none,
    resolve_categorical_cmap,
)


class TestIsNull:
    """Tests for is_null — the scalar-null predicate shared by every tier's categorical path."""

    @pytest.mark.parametrize("value", [None, np.nan, pd.NA, pd.NaT, float("nan")])
    def test_missing_values_are_null(self, value):
        """Every spelling of a missing scalar reads as null."""
        assert is_null(value) is True, f"{value!r} should be null"

    @pytest.mark.parametrize("value", ["urban", 0, False, 1.5, "", np.int64(3)])
    def test_real_values_are_not_null(self, value):
        """A genuine category — including falsy 0/False/'' — is never null."""
        assert is_null(value) is False, f"{value!r} should not be null"

    @pytest.mark.parametrize("value", [[1, 2], np.array([1, 2])])
    def test_non_scalar_is_not_null(self, value):
        """A list-like value whose `pd.isna` is elementwise flows on as a (non-null) value, not a null."""
        assert is_null(value) is False, f"non-scalar {value!r} should flow on as a value"


class TestNullsToNone:
    """Tests for nulls_to_none — normalizing a column's nulls to the one spelling cleopatra recognises."""

    def test_pd_na_becomes_none(self):
        """A pandas nullable-dtype null is rewritten to None so cleopatra drops it."""
        out = nulls_to_none(pd.array(["urban", pd.NA, "rural"], dtype="string"))
        assert out.tolist() == ["urban", None, "rural"], f"pd.NA must normalize to None, got {out.tolist()}"

    def test_no_nulls_is_unchanged(self):
        """A column with no nulls comes back with its values intact."""
        out = nulls_to_none(np.array(["a", "b"], dtype=object))
        assert out.tolist() == ["a", "b"], f"non-null values must survive untouched, got {out.tolist()}"

    def test_two_dimensional_input(self):
        """A 2-D array normalizes nulls elementwise and keeps its shape."""
        out = nulls_to_none(np.array([["a", None], ["b", "c"]], dtype=object))
        assert out.tolist() == [["a", None], ["b", "c"]], f"2-D nulls must normalize in place, got {out.tolist()}"

    def test_list_cell_is_kept_as_a_value(self):
        """A list-like cell reads as a non-null value (vectorized `pd.isna` stays elementwise), only None drops."""
        out = nulls_to_none(np.array(["a", [1, 2], None], dtype=object))
        assert out[0] == "a" and out[1] == [1, 2] and out[2] is None, f"list cell must survive, got {out.tolist()}"


def test_distinct_categories_get_distinct_colors():
    cats, colors = categorical_colors(["a", "b", "a", "c"])
    assert cats == ["a", "b", "c"], f"categories should be sorted-unique, got {cats}"
    assert len(colors) == 3 and all(c.startswith("#") for c in colors)
    assert len(set(colors)) == 3, "each category should get a distinct colour"


def test_colors_cycle_when_more_categories_than_cmap():
    cats, colors = categorical_colors(list(range(12)), cmap="tab10")
    assert len(cats) == 12 and len(colors) == 12, "colours must cycle, never run out"


def test_drops_null_and_nan():
    cats, _ = categorical_colors(["a", None, "b", float("nan")])
    assert cats == ["a", "b"], f"None/NaN should be dropped, got {cats}"


def test_numeric_categories_sorted():
    cats, colors = categorical_colors([3, 1, 2, 1])
    assert cats == [1, 2, 3] and len(colors) == 3


def test_unsortable_mixed_keeps_first_seen_order():
    cats, _ = categorical_colors(["b", 1, "a"])  # str vs int → not mutually comparable
    assert set(cats) == {"b", 1, "a"} and len(cats) == 3


def test_empty_raises():
    with pytest.raises(ValueError, match="no non-null"):
        categorical_colors([None, float("nan")])


def test_resolve_categorical_cmap_swaps_continuous_default():
    """The continuous default is swapped for a qualitative map; an explicit cmap is honoured (N1)."""
    assert resolve_categorical_cmap("viridis") == "tab10", "continuous default → qualitative default"
    assert resolve_categorical_cmap("Set2") == "Set2", "an explicit cmap must be honoured"


def test_drops_pandas_nullable_na():
    """`pd.NA`/`pd.NaT` are nulls, not categories — a nullable dtype must classify like an object one (M5)."""
    import pandas as pd

    cats, colors = categorical_colors(pd.array(["urban", pd.NA, "rural"], dtype="string"))
    assert cats == ["rural", "urban"], f"pd.NA must not become a category, got {cats}"
    assert len(colors) == 2
    object_cats, _ = categorical_colors(["urban", None, "rural"])
    assert cats == object_cats, "the same logical data must classify identically across dtypes"


@pytest.mark.parametrize(
    "cmap",
    ["tab10", "Set2", "coolwarm", "RdBu", "jet", "viridis"],
    ids=["listed-tab10", "listed-set2", "linseg-coolwarm", "linseg-rdbu", "linseg-jet", "listed-viridis"],
)
def test_matches_cleopatra_categorize_across_colormap_kinds(cmap):
    """Colours match cleopatra for both colormap kinds — a LinearSegmentedColormap must sample evenly, not
    collapse to the first-n near-identical LUT entries (the cross-tier crack this closes)."""
    from cleopatra.styles import categorize

    values = ["a", "b", "c"]
    _, ours = categorical_colors(values, cmap)
    _, upstream = categorize(np.asarray(values, dtype=object), cmap)
    assert [c.lower() for c in ours] == [c.lower() for c in upstream], f"colours must match cleopatra for {cmap}"


@pytest.mark.parametrize(
    "values",
    [
        ["urban", "rural", "park", "urban"],  # strings, sortable
        [3, 1, 2, 1],  # numbers, sortable
        ["b", None, "a", float("nan")],  # nulls dropped
    ],
    ids=["strings", "numbers", "with-nulls"],
)
def test_matches_cleopatra_categorize(values):
    """This helper must agree with cleopatra's ``categorize`` — same classes, same colours (CAT-5).

    The static tier maps categories via cleopatra while the web/interactive tiers use this helper, so the two
    must derive an identical category→colour table or the same data would render different colours per tier.
    """
    from cleopatra.styles import categorize

    ours_cats, ours_colors = categorical_colors(values)
    upstream_cats, upstream_colors = categorize(np.asarray(values, dtype=object))
    assert list(ours_cats) == list(upstream_cats), "categories must match cleopatra's (order and content)"
    assert [c.lower() for c in ours_colors] == [c.lower() for c in upstream_colors], "colours must match"
