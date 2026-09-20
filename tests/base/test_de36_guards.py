"""The four validating guards in `base/spec` that no test reached (DE-36, re-measured 2026-09-20).

DE-36 recorded six such guards after Wave 1 and named them by file and line. Wave 4 edited every one of
those files, so the line numbers pointed at unrelated code and the record could not be acted on. Re-measured
by running `tests/base` under coverage and reading each `raise` in `dataref.py`, `encoding.py`, `scale.py`
and `style.py`: **two of the six are now reached** — `dataref.py`'s uri guards both are — and four were not.

These are those four. Each guard exists because the failure it prevents surfaces far from its cause: a
`TypeError` about ints rather than about `output_range`, an `IndexError` from a colour lookup rather than
from the mismatched constructor, an ordering silently chosen by hash, or one style keyword quietly
overwriting another. A guard whose message nobody has ever read is a guard nobody knows is wrong.
"""

import pytest

from digitalearth.base.spec.encoding import Encoding
from digitalearth.base.spec.scale import Scale
from digitalearth.base.spec.style import StyleKey, StyleSchema


class TestAnOutputRangeIsAPairOrIsRefused:
    """`Encoding.output_range` is `(low, high)`; anything else is named as such."""

    @pytest.mark.parametrize("given", ["ab", 5, 5.0, object()])
    def test_something_that_is_not_a_pair_names_the_channel_and_the_value(self, given):
        """`tuple(5)` raises about ints and names neither the argument nor the channel.

        Args:
            given: What the caller passed for `output_range`.
        """
        with pytest.raises(
            ValueError, match="needs output_range as a .low, high. pair"
        ):
            Encoding(channel="size", field="depth", output_range=given)

    def test_the_channel_is_in_the_message(self):
        """The message has to say which encoding is wrong, since a symbology holds several."""
        with pytest.raises(ValueError, match="'size'"):
            Encoding(channel="size", field="depth", output_range="ab")

    def test_a_pair_is_accepted_and_stored_as_a_tuple(self):
        """The guard must not refuse the shape it exists to require."""
        encoding = Encoding(channel="size", field="depth", output_range=[2.0, 8.0])
        assert encoding.output_range == (2.0, 8.0), encoding.output_range


class TestACategoricalScaleNeedsOneColourPerCategory:
    """The public constructor is reachable without going through `Scale.categorical`."""

    def test_fewer_colours_than_categories_is_refused_at_construction(self):
        """`color_for` indexes the colours by a category's position, so a short list fails far away.

        Test scenario:
            A two-colour scale over three categories raised `IndexError` from a lookup at draw time,
            with nothing pointing back at the constructor that made it.
        """
        with pytest.raises(ValueError, match="one colour per category"):
            Scale(0.0, 1.0, categories=("a", "b", "c"), _colors=("#111111", "#222222"))

    def test_the_message_counts_both_sides(self):
        """So the caller can see which of the two they got wrong."""
        with pytest.raises(ValueError, match="3 categories and 2 colours"):
            Scale(0.0, 1.0, categories=("a", "b", "c"), _colors=("#111111", "#222222"))

    def test_matching_counts_are_accepted(self):
        """The guard must not refuse a well-formed categorical scale."""
        scale = Scale(0.0, 1.0, categories=("a", "b"), _colors=("#111111", "#222222"))
        assert scale.categories == ("a", "b"), scale.categories


class TestClassEdgesCannotArriveAsASet:
    """Edges are ordered data; a set would have its order chosen by hash."""

    def test_a_set_of_edges_is_refused_rather_than_ordered_by_hash(self):
        """`tuple({3, 1, 2})` picks an order nobody asked for, and the breaks would be wrong quietly."""
        with pytest.raises(ValueError, match="a set has no ordering"):
            Scale(0.0, 4.0, scheme={1.0, 2.0, 3.0})

    def test_the_message_says_what_to_pass_instead(self):
        """A refusal that does not say what would work sends the caller to the source."""
        with pytest.raises(
            ValueError, match="list or tuple of edges, or a scheme name"
        ):
            Scale(0.0, 4.0, scheme={1.0, 2.0, 3.0})

    def test_a_list_of_edges_is_accepted_and_frozen(self):
        """A list is the documented input on three tiers, so it must still pass."""
        scale = Scale(0.0, 4.0, scheme=[1.0, 2.0, 3.0])
        assert scale.scheme == (1.0, 2.0, 3.0), scale.scheme


class TestOneChannelIsDrivenByOneKeyword:
    """Two keywords on one channel would make routing depend on kwargs order."""

    def test_two_keys_on_one_channel_are_refused(self):
        """`route()` writes `encodings[channel]`, so the second keyword would overwrite the first.

        Test scenario:
            Which of the two won depended on the order the caller's kwargs happened to iterate in —
            so the same call could style differently between runs.
        """
        keys = (
            StyleKey("color", "the fill colour", channel="color"),
            StyleKey("fill", "the fill colour, again", channel="color"),
        )
        with pytest.raises(ValueError, match="both drive the 'color' channel"):
            StyleSchema.of(*keys)

    def test_the_message_names_both_keywords(self):
        """The caller has to know which pair collided to remove one of them."""
        keys = (
            StyleKey("color", "the fill colour", channel="color"),
            StyleKey("fill", "the fill colour, again", channel="color"),
        )
        with pytest.raises(ValueError, match="'color'.*'fill'|'fill'.*'color'"):
            StyleSchema.of(*keys)

    def test_two_keys_on_different_channels_are_accepted(self):
        """The guard is about a collision, not about declaring more than one key."""
        schema = StyleSchema.of(
            StyleKey("color", "the fill colour", channel="color"),
            StyleKey("size", "the marker size", channel="size"),
        )
        assert sorted(schema.keys) == ["color", "size"], sorted(schema.keys)
