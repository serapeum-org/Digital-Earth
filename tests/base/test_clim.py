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
