"""Tests for digitalearth._symbology — categorical (distinct-value → colour) mapping (DC.8)."""
import numpy as np
import pytest

from digitalearth._symbology import categorical_colors


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
