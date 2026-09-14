"""The shared stack colour-range rule (#174 / DE-9).

Three tiers used to derive the colour range of a raster time stack independently — static capped at 24 by an
stride, web at 50 by a head slice, interactive not at all — so one collection could come out on three
different scales. These cover the one rule they now share.
"""

import numpy as np
import pytest

from digitalearth.base.clim import (
    DEFAULT_CLIM_SCAN_CAP,
    measure_clim,
    sample_evenly,
    stack_clim,
)


class TestSampleEvenly:
    """``sample_evenly`` bounds how many frames are read without biasing which part of the series they cover."""

    def test_a_stack_shorter_than_the_cap_is_returned_whole(self):
        """Nothing is dropped when the stack already fits under the cap.

        Test scenario:
            The cap exists to bound cost, not to thin a stack that costs little. A 3-member collection must be
            measured in full or the range is worse than it needs to be for no saving.
        """
        assert sample_evenly([0, 1, 2], cap=24) == [0, 1, 2], (
            "a stack under the cap must be returned unchanged"
        )

    def test_a_long_stack_is_thinned_rather_than_truncated(self):
        """The sample spans the series instead of stopping partway through it.

        Test scenario:
            This is the divergence #174 exists for. A head slice measures only the beginning; an
            index-selected sample reaches the far end at the same cost.
        """
        picked = sample_evenly(list(range(60)), cap=24)
        assert len(picked) == 24, (
            f"a stack over the cap must spend the whole budget, got {len(picked)}"
        )
        assert picked[0] == 0, (
            f"the sample must start at the first frame, got {picked[0]}"
        )
        assert picked[-1] > 49, (
            f"the sample must reach past a 50-frame head slice, but stopped at {picked[-1]}"
        )

    def test_the_final_frame_is_always_sampled(self):
        """The last member of the series is read, whatever the stack length.

        Test scenario:
            The reason the sample is index-selected rather than strided. A plain stride starts at 0 and runs
            out before the end — 60 frames stopped at 57, 200 stopped at 198 — and that is the wrong end to
            drop: a rising series (accumulated rainfall, a cumulative anomaly, a flood crest) holds its
            maximum in the final frames, so those tiers saturated exactly where the data peaked.
        """
        for count in (25, 30, 47, 60, 100, 200):
            picked = sample_evenly(list(range(count)), cap=24)
            assert picked[-1] == count - 1, (
                f"a {count}-frame stack must sample its final frame, but stopped at {picked[-1]}"
            )
            assert picked[0] == 0, (
                f"a {count}-frame stack must still start at the first frame, got {picked[0]}"
            )

    def test_the_sample_still_skips_frames_in_between(self):
        """Capping still means most frames go unread — the guarantee is about the ends, not every peak.

        Test scenario:
            Worth pinning so the guarantee is not over-read. A peak parked on an interior frame the sample
            steps over is still missed; that is inherent to reading 24 of 60 and is the cost the cap buys.
        """
        picked = sample_evenly(list(range(60)), cap=24)
        assert len(picked) == 24, (
            f"the cap must still bound the read, got {len(picked)}"
        )
        assert set(picked) != set(range(60)), (
            "a capped sample must not read the whole stack"
        )

    def test_no_cap_scans_every_member(self):
        """``cap=None`` keeps the interactive tier's old whole-stack behaviour available.

        Test scenario:
            The cap is a default, not a law. A caller who wants an exact range over a small stack must still
            be able to ask for one.
        """
        assert len(sample_evenly(list(range(60)), cap=None)) == 60, (
            "cap=None must scan the whole stack"
        )

    @pytest.mark.parametrize("cap", [0, -1])
    def test_a_nonsensical_cap_is_refused(self, cap):
        """A cap of zero or less raises, since ``None`` is how "no cap" is written.

        Args:
            cap: The nonsensical cap under test.

        Test scenario:
            Reading 0 as "unbounded" is the opposite of its natural meaning and duplicates what ``None``
            already says, so a caller who computed a cap and got 0 would silently scan a whole cube.
        """
        with pytest.raises(ValueError, match="positive frame count"):
            sample_evenly([1, 2, 3], cap=cap)

    def test_an_empty_stack_samples_to_nothing(self):
        """An empty stack is returned as-is rather than raising on the step computation.

        Test scenario:
            The step divides by the cap; an empty sequence must short-circuit before that.
        """
        assert sample_evenly([], cap=24) == [], "an empty stack must sample to nothing"

    def test_a_stack_exactly_at_the_cap_is_returned_whole(self):
        """The boundary case reads every frame rather than striding by one and dropping the last.

        Test scenario:
            ``len(seq) <= cap`` is the short-circuit, so a stack of exactly ``cap`` members is the frame
            where an off-by-one would show: selecting every index is harmless, skipping one halves it.
        """
        picked = sample_evenly(
            list(range(DEFAULT_CLIM_SCAN_CAP)), cap=DEFAULT_CLIM_SCAN_CAP
        )
        assert picked == list(range(DEFAULT_CLIM_SCAN_CAP)), (
            f"a stack of exactly {DEFAULT_CLIM_SCAN_CAP} must be read whole, got {len(picked)} frames"
        )

    @pytest.mark.parametrize("count", [25, 47, 48, 49, 500])
    def test_the_cap_is_always_honoured_exactly(self, count):
        """No stack length sneaks past the cap, which a floor-divided stride would allow.

        Args:
            count: Stack length under test, chosen around the cap's multiples where the rounding decides.

        Test scenario:
            The rule the module's own comment names: a floor divide returns a stride of 1 for anything under
            twice the cap, so a 47-frame stack would read all 47 while claiming a cap of 24 — the cap would
            be advisory rather than a bound, and a 47-member cube would pay twice the warps it budgeted for.
        """
        picked = sample_evenly(list(range(count)), cap=DEFAULT_CLIM_SCAN_CAP)
        assert len(picked) <= DEFAULT_CLIM_SCAN_CAP, (
            f"a {count}-frame stack sampled {len(picked)} frames, over the cap of {DEFAULT_CLIM_SCAN_CAP}"
        )

    def test_the_callers_sequence_is_neither_mutated_nor_aliased(self):
        """Sampling is a read: the caller's stack comes back untouched and the result is a separate list.

        Test scenario:
            Every tier hands over ``collection.datasets`` — the collection's own list. Returning it by
            reference would let a later ``.append`` on the sample reach into the collection, and the scan has
            no business owning the caller's stack.
        """
        stack = [0, 1, 2]
        picked = sample_evenly(stack, cap=DEFAULT_CLIM_SCAN_CAP)
        picked.append(99)
        assert stack == [0, 1, 2], f"the caller's stack was mutated: {stack}"

    def test_a_single_frame_budget_reads_the_last(self):
        """``cap=1`` spends its one read on the final frame.

        Test scenario:
            With one frame to measure, the end of a series is the more informative choice: a rising series
            holds its maximum there. The branch is reachable from any caller passing a computed cap, so it
            needs a test rather than only a code comment.
        """
        assert sample_evenly(list(range(10)), cap=1) == [9], (
            f"one read must be spent on the last frame, got {sample_evenly(list(range(10)), cap=1)}"
        )

    @pytest.mark.parametrize("cap", [True, False])
    def test_a_boolean_cap_is_refused(self, cap):
        """A ``bool`` cap is a mis-typed flag, not a budget of one frame.

        Args:
            cap: The boolean under test.

        Test scenario:
            ``bool`` subclasses ``int`` in Python, so ``True`` slips past the positive-count guard and reads
            as ``cap=1`` — silently measuring a single frame of a whole cube. ``False`` would be caught by
            the positive-count guard, but as a ``ValueError`` about a frame count, which tells a caller who
            passed a flag nothing about what they actually did wrong; both spellings must name the real
            mistake.
        """
        with pytest.raises(TypeError, match="not a bool"):
            sample_evenly([1, 2, 3], cap=cap)


class TestSampleEvenlyProperties:
    """The index arithmetic holds for every stack length and cap, not only the ones a case test names.

    ``sample_evenly`` computes positions from a float step and rounds them, so its correctness is arithmetic
    rather than behavioural: a change to the step, the rounding, or an off-by-one in the endpoints would keep
    every hand-picked example passing while breaking some other length. These sweep the whole small-input
    space and assert the five invariants the callers actually depend on, so the arithmetic cannot silently
    regress.
    """

    #: Stack lengths swept. Well past four times the default cap, so the sweep covers both the
    #: short-circuit region and the region where the step is large enough for rounding to matter.
    _LENGTHS = range(1, 400)

    @pytest.mark.parametrize("cap", range(1, 40))
    def test_the_sample_is_a_strictly_increasing_subsequence_within_the_cap(self, cap):
        """Every ``(length, cap)`` pair yields distinct, in-order frames and never more than ``cap``.

        Args:
            cap: The frame budget under test, swept from 1 to 39 — through the default of 24 and out the
                other side.

        Test scenario:
            Three invariants at once, over every stack length from 1 to 399:

            * **Never over the cap.** The cap is a bound on how many warps the scan pays for; a sample that
              exceeds it makes the budget advisory, which is the bug the old floor-divided stride had.
            * **No duplicates.** Rounded positions can collide when the step falls below 1, and a collision
              spends a frame's budget re-reading a frame already read — the sample would claim ``cap``
              frames while measuring fewer.
            * **Ascending order.** Frames are handed on in series order, and every tier zips them back
              against timestamps or member order; a reordered sample would mislabel frames downstream.

            The items are ``range(length)``, so each sampled value *is* its own index and the assertions read
            directly as index arithmetic.
        """
        for length in self._LENGTHS:
            picked = sample_evenly(list(range(length)), cap=cap)
            assert len(picked) <= cap, (
                f"length={length} cap={cap} sampled {len(picked)} frames, over the cap"
            )
            assert len(set(picked)) == len(picked), (
                f"length={length} cap={cap} sampled a duplicate frame: {picked}"
            )
            assert picked == sorted(picked), (
                f"length={length} cap={cap} sampled out of series order: {picked}"
            )

    @pytest.mark.parametrize("cap", range(1, 40))
    def test_the_whole_budget_is_spent(self, cap):
        """The sample is exactly ``min(length, cap)`` frames — never short, never over.

        Args:
            cap: The frame budget under test.

        Test scenario:
            The budget is paid for whether or not it is used, so returning fewer frames than the cap allows
            is a strictly worse range for the same cost. A stride-based rule cannot promise this: the number
            of frames a stride returns depends on where it runs out. Under the cap the count is the stack's
            own length, since the short-circuit returns it whole.
        """
        for length in self._LENGTHS:
            picked = sample_evenly(list(range(length)), cap=cap)
            expected = min(length, cap)
            assert len(picked) == expected, (
                f"length={length} cap={cap} must sample {expected} frames, got {len(picked)}"
            )

    @pytest.mark.parametrize("cap", range(2, 40))
    def test_both_endpoints_are_sampled_whenever_the_stack_is_thinned(self, cap):
        """A thinned stack always contributes its first and its last frame.

        Args:
            cap: The frame budget under test, from 2 up — ``cap=1`` has only one frame to spend and spends
                it on the last, which :meth:`TestSampleEvenly.test_a_single_frame_budget_reads_the_last`
                pins separately.

        Test scenario:
            The guarantee the index-selected rule exists to provide, asserted across the whole space rather
            than at the handful of lengths a case test can name. The last frame is the one a stride drops,
            and it is the one that matters: a rising series (accumulated rainfall, a cumulative anomaly, a
            flood crest) holds its maximum there, so a sample that stops short saturates the ramp exactly
            where the data peaks. The first matters too — a series usually starts at its baseline, which is
            the other end of the range.
        """
        for length in self._LENGTHS:
            if length <= cap:
                continue
            picked = sample_evenly(list(range(length)), cap=cap)
            assert picked[0] == 0, (
                f"length={length} cap={cap} must start at the first frame, got {picked[0]}"
            )
            assert picked[-1] == length - 1, (
                f"length={length} cap={cap} must reach the final frame, got {picked[-1]}"
            )

    @pytest.mark.parametrize("cap", range(1, 40))
    def test_a_stack_under_the_cap_comes_back_unchanged(self, cap):
        """Nothing is dropped, reordered or thinned while the stack still fits the budget.

        Args:
            cap: The frame budget under test.

        Test scenario:
            The short-circuit half of the contract, swept rather than sampled: for every length up to and
            including the cap the answer must be the stack itself. The boundary ``length == cap`` is where an
            off-by-one in the ``<=`` would show — it would push a full-budget stack into the arithmetic path,
            and while that path happens to be an identity map there, the check is what keeps the two halves
            from disagreeing.
        """
        for length in range(1, cap + 1):
            stack = list(range(length))
            assert sample_evenly(stack, cap=cap) == stack, (
                f"length={length} cap={cap} fits the budget and must be returned whole"
            )


class TestMeasureClim:
    """``measure_clim`` reduces frames to one range, and says so when there is nothing to measure."""

    def test_it_reduces_several_frames_to_one_range(self):
        """The range spans every frame, not just the first.

        Test scenario:
            The whole point of a frozen scale: the colour ramp must cover the series, so a later frame's
            extreme has to widen the range.
        """
        assert measure_clim([np.array([1.0, 5.0]), np.array([-2.0, 3.0])]) == (
            -2.0,
            5.0,
        ), "the range must span every frame"

    def test_nodata_and_non_finite_values_are_ignored(self):
        """``NaN`` and infinities do not become the colour limits.

        Test scenario:
            A nodata sentinel reaching the ramp collapses the real data into one end of it.
        """
        assert measure_clim([np.array([np.nan, 4.0, np.inf])]) == (4.0, 4.0), (
            "non-finite values must not set the limits"
        )

    def test_a_masked_array_agrees_with_an_already_nan_filled_one(self):
        """A masked frame measures the same as the NaN-filled frame a pyramids extractor returns.

        Test scenario:
            This was divergence 3 in #174 — three different nodata exclusions. ``np.asarray`` drops a mask and
            would let the fill value (-9999) through as a real number, so the two spellings disagreed for a
            caller passing a masked array directly.
        """
        masked = np.ma.masked_array([1.0, -9999.0, 3.0], mask=[False, True, False])
        filled = np.array([1.0, np.nan, 3.0])
        assert measure_clim([masked]) == measure_clim([filled]) == (1.0, 3.0), (
            f"a masked frame must measure as its NaN-filled twin, got {measure_clim([masked])}"
        )

    def test_nothing_measurable_answers_none(self):
        """``None`` distinguishes "nothing to measure" from a range that happens to be 0-1.

        Test scenario:
            The static tier unions one result per swept projection; a view showing no data must contribute
            nothing, or the union floors at zero and the real data collapses into the top of the ramp.
        """
        assert measure_clim([np.array([np.nan])]) is None, (
            "an unmeasurable stack must answer None, not a placeholder"
        )

    def test_an_unmeasurable_frame_does_not_veto_the_measurable_ones(self):
        """A mixed stack is measured from the frames that hold data, and the rest contribute nothing.

        Test scenario:
            The realistic stack, and the case the all-finite and all-nodata tests between them never reach:
            a series with a cloud-masked or off-view frame in the middle. Such a frame must neither widen the
            range (a ``NaN`` reaching ``min``/``max`` poisons both) nor collapse the answer to ``None``.
        """
        frames = [
            np.array([np.nan, np.nan]),
            np.array([3.0, 7.0]),
            np.array([np.inf, -np.inf]),
            np.array([-1.0, 4.0]),
        ]
        assert measure_clim(frames) == (-1.0, 7.0), (
            f"the measurable frames must set the range alone, got {measure_clim(frames)}"
        )

    def test_a_fully_masked_frame_contributes_nothing_rather_than_its_fill_value(self):
        """A frame masked end to end drops out instead of handing its ``-9999`` sentinel to the ramp.

        Test scenario:
            The masked half of the mixed case above. ``np.asarray`` drops a mask, so a fully masked frame
            that is not NaN-filled first would contribute ``-9999`` as a real minimum and flatten every real
            value into the top of the colour ramp.
        """
        masked = np.ma.masked_array([-9999.0, -9999.0], mask=[True, True])
        measured = measure_clim([masked, np.array([2.0, 6.0])])
        assert measured == (2.0, 6.0), (
            f"a fully masked frame must contribute nothing, got {measured}"
        )

    def test_an_empty_frame_contributes_nothing(self):
        """A zero-size array is skipped rather than raising on ``min`` of an empty reduction.

        Test scenario:
            A warp that lands entirely outside the view returns no cells. ``numpy`` raises on ``min()`` of an
            empty array, so the size check is what keeps a legitimate empty frame from failing the scan.
        """
        measured = measure_clim([np.array([]), np.array([5.0])])
        assert measured == (5.0, 5.0), (
            f"an empty frame must not affect the range, got {measured}"
        )

    def test_frames_may_arrive_as_a_generator(self):
        """The reduction consumes any iterable, which is how all three tiers actually call it.

        Test scenario:
            Each tier passes a generator expression over its sampled members, so that the frames are warped
            and read one at a time rather than materialised as a list of whole rasters. A signature that
            silently needed a sequence would undo the cap's whole point.
        """
        measured = measure_clim(np.array([float(index)]) for index in range(4))
        assert measured == (0.0, 3.0), (
            f"a generator of frames must reduce like a list, got {measured}"
        )

    def test_an_integer_masked_array_does_not_raise(self):
        """A nodata sentinel is usually an integer, and filling an int array with ``NaN`` raises.

        Test scenario:
            The consolidation took the web tier's ``.filled(np.nan)`` and dropped the ``.astype(float)`` the
            interactive tier did first, so ``measure_clim`` crashed on exactly the dtype its own comment
            names: ``-9999`` in an ``int16`` band. The cast has to come before the fill.
        """
        masked = np.ma.masked_array(
            np.array([1, -9999, 3], dtype="int16"), mask=[False, True, False]
        )
        assert measure_clim([masked]) == (1.0, 3.0), (
            f"an integer masked frame must measure as its NaN-filled twin, got {measure_clim([masked])}"
        )


class TestStackClim:
    """``stack_clim`` is the public form: a range a colormap can take without a ``None`` check."""

    def test_it_returns_the_measured_range(self):
        """A measurable stack reduces to its own range.

        Test scenario:
            The fallback must not shadow a real measurement.
        """
        assert stack_clim([np.array([2.0, 9.0])]) == (2.0, 9.0), (
            "the measured range must win"
        )

    def test_an_unmeasurable_stack_falls_back_to_zero_one(self):
        """Nothing finite anywhere yields ``(0.0, 1.0)`` rather than raising.

        Test scenario:
            Every frame off the view draws nothing, and a builder still needs limits to hand the colormap.
        """
        assert stack_clim([]) == (0.0, 1.0), "an empty stack must fall back to (0, 1)"

    def test_a_stack_of_frames_with_no_finite_value_falls_back_too(self):
        """Frames that exist but hold nothing measurable reach the same fallback as no frames at all.

        Test scenario:
            The case a caller actually hits — an all-nodata cube, or a stack every frame of which is off the
            view — as distinct from the empty-list case above. Both must yield limits a colormap can take,
            and neither may raise on the way there.
        """
        assert stack_clim([np.array([np.nan, np.nan]), np.array([])]) == (0.0, 1.0), (
            "an unmeasurable stack must fall back to (0, 1) rather than raise"
        )

    def test_the_fallback_does_not_shadow_a_real_range_that_happens_to_be_zero_one(
        self,
    ):
        """Data genuinely spanning 0 to 1 is returned as a measurement, not mistaken for the fallback.

        Test scenario:
            ``stack_clim`` collapses "nothing measurable" and "0 to 1" into the same tuple, which is exactly
            why :func:`measure_clim` exists alongside it. This pins that the collapse is one-way: a real
            ``(0, 1)`` still measures, so a normalised raster is not quietly treated as unmeasurable.
        """
        frames = [np.array([0.0, 1.0])]
        assert stack_clim(frames) == (0.0, 1.0), (
            f"a real 0-1 span must come back unchanged, got {stack_clim(frames)}"
        )
        assert measure_clim(frames) == (0.0, 1.0), (
            f"the same span must measure rather than read as unmeasurable, got {measure_clim(frames)}"
        )


class TestTheTiersShareOneRule:
    """The cap is one number, so the three tiers cannot drift apart again."""

    def test_the_static_tier_reads_the_shared_cap(self):
        """The static tier's cap is the shared constant, not a second literal.

        Test scenario:
            #174's root cause was three literals free to drift — 24, 50 and none. This is the half that can
            be checked from the ``dev`` env; the web tier's identical guard lives in
            ``tests/web/test_web_stack_clim.py`` and the interactive tier, which passes no cap at all, is
            pinned by the bounded-scan test in ``tests/interactive/test_interactive_stack_clim.py`` — each
            in the env that can import it.
        """
        from digitalearth.static.maps.animation import _CLIM_SCAN_CAP as static_cap

        assert static_cap == DEFAULT_CLIM_SCAN_CAP, (
            f"the static tier must read the shared cap, got {static_cap}"
        )
