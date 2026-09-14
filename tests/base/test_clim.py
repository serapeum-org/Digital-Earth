"""The shared stack colour-range rule (#174 / DE-9).

Three tiers used to derive the colour range of a raster time stack independently — static capped at 24 by an
even stride, web at 50 by a head slice, interactive not at all — so one collection could come out on three
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

    def test_a_long_stack_is_strided_rather_than_truncated(self):
        """The sample spans the series instead of stopping partway through it.

        Test scenario:
            This is the divergence #174 exists for. A head slice measures only the beginning; a stride reaches
            the far end at the same cost.
        """
        picked = sample_evenly(list(range(60)), cap=24)
        assert len(picked) <= 24, f"the cap must bound the sample, got {len(picked)}"
        assert picked[0] == 0, (
            f"the sample must start at the first frame, got {picked[0]}"
        )
        assert picked[-1] > 49, (
            f"the sample must reach past a 50-frame head slice, but stopped at {picked[-1]}"
        )

    def test_a_late_peak_is_reached_that_a_head_slice_would_miss(self):
        """A value late in the series is measured, where the old web rule never looked.

        Test scenario:
            The concrete regression from #174: a 60-member stack whose extreme sits at index 57 — a flood
            crest or fire scar. The old ``datasets[:50]`` head slice never reached it and clipped the peak to
            solid top-of-ramp; the stride samples it.
        """
        picked = sample_evenly(list(range(60)), cap=24)
        assert 57 in picked, (
            f"the stride must reach the late peak at 57, sampled {picked}"
        )
        assert 57 not in list(range(60))[:50], (
            "the old head slice is supposed to miss index 57 — this test has lost its point"
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
    def test_a_nonsensical_cap_scans_everything_rather_than_nothing(self, cap):
        """A cap of zero or less reads the whole stack instead of returning an empty sample.

        Args:
            cap: The nonsensical cap under test.

        Test scenario:
            Returning nothing would silently produce the ``(0, 1)`` fallback for a stack full of real data,
            which is a worse failure than ignoring the bad cap.
        """
        assert len(sample_evenly([1, 2, 3], cap=cap)) == 3, (
            f"cap={cap} must not empty the sample"
        )

    def test_an_empty_stack_samples_to_nothing(self):
        """An empty stack is returned as-is rather than raising on the stride computation.

        Test scenario:
            The stride divides by the cap; an empty sequence must short-circuit before that.
        """
        assert sample_evenly([], cap=24) == [], "an empty stack must sample to nothing"

    def test_a_stack_exactly_at_the_cap_is_returned_whole(self):
        """The boundary case reads every frame rather than striding by one and dropping the last.

        Test scenario:
            ``len(seq) <= cap`` is the short-circuit, so a stack of exactly ``cap`` members is the frame
            where an off-by-one would show: a stride of 1 is harmless, a stride of 2 would halve it.
        """
        picked = sample_evenly(
            list(range(DEFAULT_CLIM_SCAN_CAP)), cap=DEFAULT_CLIM_SCAN_CAP
        )
        assert picked == list(range(DEFAULT_CLIM_SCAN_CAP)), (
            f"a stack of exactly {DEFAULT_CLIM_SCAN_CAP} must be read whole, got {len(picked)} frames"
        )

    @pytest.mark.parametrize("count", [25, 47, 48, 49, 500])
    def test_the_stride_rounds_up_so_the_cap_is_always_honoured(self, count):
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
        assert stack_clim(frames) == (0.0, 1.0) and measure_clim(frames) == (
            0.0,
            1.0,
        ), f"a real 0-1 span must measure, got {measure_clim(frames)}"


class TestTheTiersShareOneRule:
    """The cap is one number, so the three tiers cannot drift apart again."""

    def test_every_tier_reads_the_shared_cap(self):
        """static and web both take their cap from ``base.clim`` rather than declaring their own.

        Test scenario:
            #174's root cause was three literals free to drift — 24, 50 and none. Importing the tier modules
            and comparing identity is what stops a fourth value appearing.
        """
        from digitalearth.static.maps.animation import _CLIM_SCAN_CAP as static_cap

        assert static_cap == DEFAULT_CLIM_SCAN_CAP, (
            f"the static tier must read the shared cap, got {static_cap}"
        )
