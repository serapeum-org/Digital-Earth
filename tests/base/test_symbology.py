"""Tests for digitalearth.base.symbology — categorical (distinct-value → colour) mapping (DC.8)."""

import numpy as np
import pandas as pd
import pytest

from digitalearth.base.symbology import (
    categorical_colors,
    is_null,
    nulls_to_none,
    resolve_categorical_cmap,
    sample_cmap,
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
        assert is_null(value) is False, (
            f"non-scalar {value!r} should flow on as a value"
        )


class TestNullsToNone:
    """Tests for nulls_to_none — normalizing a column's nulls to the one spelling cleopatra recognises."""

    def test_pd_na_becomes_none(self):
        """A pandas nullable-dtype null is rewritten to None so cleopatra drops it."""
        out = nulls_to_none(pd.array(["urban", pd.NA, "rural"], dtype="string"))
        assert out.tolist() == ["urban", None, "rural"], (
            f"pd.NA must normalize to None, got {out.tolist()}"
        )

    def test_no_nulls_is_unchanged(self):
        """A column with no nulls comes back with its values intact."""
        out = nulls_to_none(np.array(["a", "b"], dtype=object))
        assert out.tolist() == ["a", "b"], (
            f"non-null values must survive untouched, got {out.tolist()}"
        )

    def test_two_dimensional_input(self):
        """A 2-D array normalizes nulls elementwise and keeps its shape."""
        out = nulls_to_none(np.array([["a", None], ["b", "c"]], dtype=object))
        assert out.tolist() == [["a", None], ["b", "c"]], (
            f"2-D nulls must normalize in place, got {out.tolist()}"
        )

    def test_list_cell_is_kept_as_a_value(self):
        """A list-like cell reads as a non-null value (vectorized `pd.isna` stays elementwise), only None drops."""
        out = nulls_to_none(np.array(["a", [1, 2], None], dtype=object))
        assert out[0] == "a", f"list cell must survive, got {out.tolist()}"
        assert out[1] == [1, 2], f"list cell must survive, got {out.tolist()}"
        assert out[2] is None, f"list cell must survive, got {out.tolist()}"


def test_distinct_categories_get_distinct_colors():
    cats, colors = categorical_colors(["a", "b", "a", "c"])
    assert cats == ["a", "b", "c"], f"categories should be sorted-unique, got {cats}"
    assert len(colors) == 3 and all(c.startswith("#") for c in colors)
    assert len(set(colors)) == 3, "each category should get a distinct colour"


def test_colors_cycle_when_more_categories_than_cmap():
    cats, colors = categorical_colors(list(range(12)), cmap="tab10")
    assert len(cats) == 12, "colours must cycle, never run out"
    assert len(colors) == 12, "colours must cycle, never run out"


def test_drops_null_and_nan():
    cats, _ = categorical_colors(["a", None, "b", float("nan")])
    assert cats == ["a", "b"], f"None/NaN should be dropped, got {cats}"


def test_numeric_categories_sorted():
    cats, colors = categorical_colors([3, 1, 2, 1])
    assert cats == [1, 2, 3]
    assert len(colors) == 3


def test_unsortable_mixed_keeps_first_seen_order():
    cats, _ = categorical_colors(["b", 1, "a"])  # str vs int → not mutually comparable
    assert set(cats) == {"b", 1, "a"}
    assert len(cats) == 3


def test_empty_raises():
    not_a_number = float("nan")
    with pytest.raises(ValueError, match="no non-null"):
        categorical_colors([None, not_a_number])


def test_resolve_categorical_cmap_swaps_continuous_default():
    """The continuous default is swapped for a qualitative map; an explicit cmap is honoured (N1)."""
    assert resolve_categorical_cmap("viridis") == "tab10", (
        "continuous default → qualitative default"
    )
    assert resolve_categorical_cmap("Set2") == "Set2", (
        "an explicit cmap must be honoured"
    )


def test_drops_pandas_nullable_na():
    """`pd.NA`/`pd.NaT` are nulls, not categories — a nullable dtype must classify like an object one (M5)."""
    import pandas as pd

    cats, colors = categorical_colors(
        pd.array(["urban", pd.NA, "rural"], dtype="string")
    )
    assert cats == ["rural", "urban"], f"pd.NA must not become a category, got {cats}"
    assert len(colors) == 2
    object_cats, _ = categorical_colors(["urban", None, "rural"])
    assert cats == object_cats, (
        "the same logical data must classify identically across dtypes"
    )


@pytest.mark.parametrize(
    "cmap",
    ["tab10", "Set2", "coolwarm", "RdBu", "jet", "viridis"],
    ids=[
        "listed-tab10",
        "listed-set2",
        "linseg-coolwarm",
        "linseg-rdbu",
        "linseg-jet",
        "listed-viridis",
    ],
)
def test_matches_cleopatra_categorize_across_colormap_kinds(cmap):
    """Colours match cleopatra for both colormap kinds — a LinearSegmentedColormap must sample evenly, not
    collapse to the first-n near-identical LUT entries (the cross-tier crack this closes)."""
    from cleopatra.styling.styles import categorize

    values = ["a", "b", "c"]
    _, ours = categorical_colors(values, cmap)
    _, upstream = categorize(np.asarray(values, dtype=object), cmap)
    assert [c.lower() for c in ours] == [c.lower() for c in upstream], (
        f"colours must match cleopatra for {cmap}"
    )


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
    from cleopatra.styling.styles import categorize

    ours_cats, ours_colors = categorical_colors(values)
    upstream_cats, upstream_colors = categorize(np.asarray(values, dtype=object))
    assert list(ours_cats) == list(upstream_cats), (
        "categories must match cleopatra's (order and content)"
    )
    assert [c.lower() for c in ours_colors] == [c.lower() for c in upstream_colors], (
        "colours must match"
    )


class TestAColormapObjectIsAcceptedWhereItsNameIs:
    """A `Colormap` and its name must mean the same thing to every helper that classifies (#315)."""

    def test_sampling_a_colormap_object_matches_sampling_its_name(self):
        """A caller holding a colormap should not have to know to pass its name instead.

        Test scenario:
            `sample_cmap` branched on `isinstance(cmap, str)` and treated everything else as an
            already-built sequence of colours, so a `Colormap` fell through to `list(cmap)` and raised
            `TypeError: 'ListedColormap' object is not iterable`. The unclassified builders accept the
            object, so whether a colormap worked depended on whether a scheme was given.
        """
        from matplotlib import colormaps

        by_name = sample_cmap("magma", 4)
        by_object = sample_cmap(colormaps["magma"], 4)
        assert by_object == by_name, (
            f"a colormap and its name must sample alike: {by_object} vs {by_name}"
        )

    def test_colouring_categories_from_a_colormap_object_matches_its_name(self):
        """The categorical path had its own wall: an unhashable object used as a dict key.

        Test scenario:
            `categorical_colors` looked the colormap up with `colormaps[cmap]`. A `Colormap` defines
            `__eq__` without `__hash__`, so the lookup raised `TypeError: unhashable type: 'ListedColormap'`
            before anything could name the argument that was wrong.
        """
        from matplotlib import colormaps

        values = ["a", "b", "c"]
        _, by_name = categorical_colors(values, "tab10")
        _, by_object = categorical_colors(values, colormaps["tab10"])
        assert by_object == by_name, (
            f"a colormap and its name must colour alike: {by_object} vs {by_name}"
        )

    def test_a_sequence_of_colours_is_still_taken_as_given(self):
        """Accepting an object must not disturb the spelling that was already accepted."""
        given = ["#ff0000", "#00ff00"]
        assert sample_cmap(given, 5) == given, (
            "an already-built sequence of colours is returned untouched, whatever n is"
        )


class TestTheTwoHelpersReadASequenceTheSameWay:
    """`sample_cmap` takes a sequence of colours as a palette; its neighbour raised on one (review L9)."""

    def test_colouring_categories_from_a_sequence_of_colours_uses_them(self):
        """One argument, read one way, whether the caller classifies or not.

        Test scenario:
            `sample_cmap(['#f00', '#0f0'], 2)` answers `['#f00', '#0f0']` — a sequence is a palette there —
            while `categorical_colors(values, cmap=['#f00', '#0f0'])` reached `as_colormap`, which used the
            list as a dict key and raised `TypeError: unhashable type: 'list'`. Two neighbours reading the
            same argument two ways is a trap for the tier code that passes a caller's `cmap` down both.
        """
        _, colours = categorical_colors(["a", "b"], ["#ff0000", "#00ff00"])
        assert colours == ["#ff0000", "#00ff00"], (
            f"a sequence of colours must be the palette, as it is for sample_cmap; got {colours}"
        )

    def test_a_palette_shorter_than_the_categories_cycles(self):
        """A palette is cycled the way a `ListedColormap`'s own colours are."""
        _, colours = categorical_colors(["a", "b", "c"], ["#ff0000", "#00ff00"])
        assert colours == ["#ff0000", "#00ff00", "#ff0000"], (
            f"a short palette must cycle, as a ListedColormap's does; got {colours}"
        )

    def test_an_empty_palette_is_refused_by_name(self):
        """Cycling an empty palette divides by zero, which names neither the argument nor the helper."""
        values = ["a", "b"]
        with pytest.raises(ValueError, match="cmap"):
            categorical_colors(values, [])

    def test_a_colormap_name_is_unaffected(self):
        """Reading a sequence as a palette must not change what a name means."""
        _, named = categorical_colors(["a", "b", "c"], "tab10")
        assert len(named) == 3, (
            f"a name still samples one colour per category; got {named}"
        )
