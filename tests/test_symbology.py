"""Tests for digitalearth._symbology — categorical (distinct-value → colour) mapping (DC.8)."""
import numpy as np
import pytest

from digitalearth._symbology import categorical_colors, resolve_categorical_cmap


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
