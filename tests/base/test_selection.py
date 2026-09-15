"""`Selection` — which slice of a dataset a layer draws (DE-13, #272).

36 builders take ``band: int = 1``, and a composite takes a separate ``bands`` sequence beside it: one question
asked two ways. These cover the type that asks it once.
"""

import numpy as np
import pytest

from digitalearth.base.spec import DEFAULT_BAND, Selection


class TestBandIsATuple:
    """The load-bearing change: a composite is a selection of bands, not a different parameter."""

    def test_a_scalar_band_becomes_a_one_tuple(self):
        """One code path serves a single band and a composite.

        Test scenario:
            Every existing builder passes a scalar. If that did not normalise, the shared path would need a
            branch at every call site, which is the duplication this type removes.
        """
        assert Selection.of(2).band == (2,), (
            "a scalar band must normalise to a one-tuple"
        )

    def test_a_composite_keeps_its_channel_order(self):
        """Band order is channel order for an RGB composite, so it must survive.

        Test scenario:
            Sorting or de-duplicating the bands would silently swap the red and blue channels.
        """
        assert Selection.of((3, 2, 1)).band == (3, 2, 1), (
            "channel order must be preserved"
        )

    def test_both_spellings_produce_the_same_value(self):
        """A scalar and a one-element sequence are the same selection.

        Test scenario:
            `band=1` and `bands=[1]` are the two spellings live in the code today; they must converge.
        """
        assert Selection.of(1) == Selection.of([1]), "the two spellings must converge"

    def test_the_default_band_is_the_shared_constant(self):
        """An unspecified band reads band 1, as every public builder already does.

        Test scenario:
            Changing the implicit default would silently redraw every existing call.
        """
        assert Selection().band == (DEFAULT_BAND,), "the default must stay band 1"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_zero_or_negative_band_is_refused(self, bad):
        """Bands are 1-based, and a 0 is a caller expecting 0-based indexing.

        Args:
            bad: The out-of-range band.

        Test scenario:
            Reading band 1 for a caller who wrote 0 draws the wrong data with no error — the exact
            silent-wrongness Wave 0 spent itself removing.
        """
        with pytest.raises(ValueError, match="1-based"):
            Selection.of(bad)

    @pytest.mark.parametrize("bad", [True, "x", 1.5])
    def test_a_band_that_is_not_a_whole_number_is_refused(self, bad):
        """A flag, a string or a float is not a band index.

        Args:
            bad: The mistyped band.

        Test scenario:
            `True` is the trap: it is an `int` in Python, so a naive check reads it as band 1.
        """
        with pytest.raises(ValueError, match="whole 1-based band numbers"):
            Selection.of(bad)

    def test_an_empty_selection_is_refused(self):
        """A selection naming no band cannot be read.

        Test scenario:
            It would otherwise reach a reader as an empty list and return nothing, silently.
        """
        with pytest.raises(ValueError, match="at least one band"):
            Selection.of([])

    def test_numpy_integers_are_accepted_and_stored_as_python_ints(self):
        """A numpy index is a whole 1-based band number, and is coerced to what the reader takes.

        Test scenario:
            `np.array([3, 2, 1])` is the natural way to name composite bands beside numpy data, and
            `isinstance(np.int64(1), int)` is False — so a plain int check refused it while saying it was not
            a whole number. Worse, GDAL's SWIG binding rejects numpy integers outright, so passing one
            through would have failed deep inside the reader. Coercing here is what makes it work at all.
        """
        selection = Selection.of(np.array([3, 2, 1]))
        assert selection.band == (3, 2, 1), "a numpy array of indices must be accepted"
        assert [type(band) for band in selection.band] == [int, int, int], (
            "and stored as Python ints, which is what the reader takes"
        )

    def test_a_generator_of_bands_is_accepted(self):
        """Bands may be produced lazily.

        Test scenario:
            `get_stack` iterated its bands before this type existed, so a generator worked. Testing for
            `Sequence` rather than `Iterable` silently wrapped the generator object itself as a one-tuple
            and then refused it as not a number.
        """
        assert Selection.of(band for band in (1, 2)).band == (1, 2), (
            "a generator must be consumed, not wrapped"
        )

    def test_a_float_that_happens_to_be_whole_is_still_refused(self):
        """`2.0` is not an index, even though it equals one.

        Test scenario:
            Accepting floats would mean accepting `2.5` too, or silently truncating it. The rule is whole
            numbers, and a float says the caller computed the band rather than named it.
        """
        with pytest.raises(ValueError, match="whole 1-based band numbers"):
            Selection.of(2.0)


class TestTheOtherAxes:
    """The fields each tier used to carry in a parameter of its own."""

    def test_every_axis_is_carried_together(self):
        """Time, level, member, overview and budget travel with the bands.

        Test scenario:
            These are exactly the extras the 3-D, temporal and large-data paths add beside `band` today.
        """
        sel = Selection.of(
            1, time="2024-01", level=850, member=4, overview=2, budget=10_000
        )
        assert (sel.time, sel.level, sel.member, sel.overview, sel.budget) == (
            "2024-01",
            850,
            4,
            2,
            10_000,
        ), "every axis must be carried on the one value"

    def test_narrowing_the_band_keeps_the_other_axes(self):
        """`with_band` changes only the bands.

        Test scenario:
            A composite iterating its channels must not lose the level or budget chosen once for the layer —
            losing them is how a per-channel read silently reads full resolution.
        """
        narrowed = Selection.of((1, 2, 3), level=850, budget=99).with_band(2)
        assert narrowed.band == (2,), "the band must change"
        assert (narrowed.level, narrowed.budget) == (850, 99), "the other axes must not"

    def test_narrowing_does_not_mutate_the_original(self):
        """The original selection is unchanged.

        Test scenario:
            A shared selection edited in place would change what the *other* channels read.
        """
        original = Selection.of((1, 2, 3))
        original.with_band(2)
        assert original.band == (1, 2, 3), "with_band must return a copy"


class TestFrames:
    """A composite's per-channel reads, derived rather than hand-rolled."""

    def test_a_composite_yields_one_selection_per_band_in_order(self):
        """Three bands give three single-band reads, in channel order.

        Test scenario:
            The loop every composite builder writes by hand today.
        """
        assert [s.first_band for s in Selection.of((3, 2, 1)).frames()] == [3, 2, 1], (
            "frames must preserve channel order"
        )

    def test_each_frame_carries_the_shared_axes(self):
        """A per-channel read keeps the level and budget the layer chose.

        Test scenario:
            The same loss `with_band` guards against, checked through the path a composite actually uses.
        """
        frames = Selection.of((1, 2), level=500, budget=7).frames()
        assert all(f.level == 500 and f.budget == 7 for f in frames), (
            "every frame must keep the shared axes"
        )

    def test_a_single_band_selection_yields_itself(self):
        """The non-composite case is one frame, not zero or a special case.

        Test scenario:
            Lets a caller loop over `frames()` unconditionally instead of branching on `is_composite`.
        """
        sel = Selection.of(4)
        assert [f.band for f in sel.frames()] == [(4,)], "one band means one frame"

    def test_is_composite_distinguishes_the_two(self):
        """The flag a renderer branches on.

        Test scenario:
            A composite renderer needs three bands; a scalar one needs to know it has not been handed three.
        """
        assert Selection.of((1, 2)).is_composite, "two bands is a composite"
        assert not Selection.of(1).is_composite, "one band is not"


class TestItStaysAValue:
    """Frozen and comparable, like the rest of the vocabulary."""

    def test_it_cannot_be_mutated(self):
        """Assigning to a field raises.

        Test scenario:
            A selection handed down to a reader must not change underneath it.
        """
        sel = Selection.of(1)
        with pytest.raises(AttributeError):
            sel.band = (9,)

    def test_two_equal_selections_compare_equal(self):
        """Equality is by value.

        Test scenario:
            What lets a test assert two paths resolved the same slice.
        """
        assert Selection.of(1, level=850) == Selection.of(1, level=850), (
            "equal axes must compare equal"
        )

    def test_a_list_band_is_stored_as_a_tuple(self):
        """A caller reaching the constructor directly still gets an immutable selection.

        Test scenario:
            `Selection.of` normalises, but the dataclass constructor is public too — and a list stored as
            given leaves the selection unhashable and editable from outside.
        """
        bands = [1, 2]
        selection = Selection(band=bands)
        bands.append(3)
        assert selection.band == (1, 2), (
            "the selection must not track the caller's list"
        )

    def test_a_selection_can_key_a_cache(self):
        """`Selection` hashes, so a read can be memoised by what it selects.

        Test scenario:
            Caching a materialised read on its selection is what Wave 2's data tier is for; an unhashable
            selection would rule that out.
        """
        assert len({Selection.of((1, 2)), Selection.of([1, 2])}) == 1, (
            "two equal selections must collapse to one entry"
        )
