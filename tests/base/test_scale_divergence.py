"""Whether a scale's domain runs on both sides of a value (ST-3).

A diverging colormap reads correctly only when the data actually straddles the value its neutral centre sits
on. Centred outside the domain it is worse than a sequential ramp, not better: half the colours are never
drawn, so the half that is draws every value in one arm and the picture loses the contrast it was chosen for.

That question is about the **domain**, which is what `Scale` holds, and the answer has to be available to every
tier — so it is one reader here rather than a `vmin < center < vmax` written out per backend. The arithmetic is
the same comparison cleopatra validates a diverging `center` with, which is deliberate: the two must agree
about what counts as centred, or this tier would offer a centre cleopatra then refuses.
"""

import pytest

from digitalearth.base.spec import Scale


class TestStraddlingTheDefaultCentre:
    """Zero is the centre a diverging scale is asked about when nothing names another."""

    def test_a_domain_running_either_side_of_zero_straddles_it(self):
        """An anomaly field with both signs is the case divergence exists for."""
        assert Scale.from_values([-2.0, -1.0, 3.0, 6.0]).straddles(), (
            "a domain with values on both sides of zero must straddle it"
        )

    def test_an_all_positive_domain_does_not(self):
        """A rainfall accumulation has no negative arm, so a diverging ramp wastes half its colours."""
        assert not Scale.from_values([12.0, 40.0, 88.0]).straddles(), (
            "a domain entirely above zero must not straddle it"
        )

    def test_an_all_negative_domain_does_not(self):
        """And the mirror case."""
        assert not Scale.from_values([-88.0, -40.0, -12.0]).straddles(), (
            "a domain entirely below zero must not straddle it"
        )

    def test_zero_is_the_default_centre(self):
        """Asking with no argument asks about zero.

        Test scenario:
            Read against an explicit `straddles(0.0)` on a domain the two have to agree about, so the default
            is pinned as a value rather than by repeating the no-argument call.
        """
        scale = Scale.from_values([-5.0, 5.0])
        assert scale.straddles() == scale.straddles(0.0), (
            "the default centre must be zero"
        )


class TestStraddlingAStatedCentre:
    """A centre need not be zero — a departure from a long-term mean sits wherever that mean does."""

    def test_a_domain_around_the_stated_centre_straddles_it(self):
        """A 10-degree band around 288 K straddles 288 K."""
        assert Scale.from_limits(283.0, 293.0).straddles(288.0), (
            "a domain around the stated centre must straddle it"
        )

    def test_a_domain_entirely_above_the_stated_centre_does_not(self):
        """The same band does not straddle freezing."""
        assert not Scale.from_limits(283.0, 293.0).straddles(273.15), (
            "a domain above the stated centre must not straddle it"
        )

    @pytest.mark.parametrize("centre", [0.0, 10.0])
    def test_a_domain_that_only_reaches_the_centre_does_not_straddle_it(self, centre):
        """Touching an end is not straddling: one arm of the ramp would hold a single value.

        Args:
            centre: A centre sitting exactly on one end of the domain under test.

        Test scenario:
            `Scale(0.0, 10.0)` ends at both of these. The strict comparison is also the one cleopatra
            validates with — it raises "diverging 'center' must lie strictly between vmin and vmax" — so a
            centre this reader called straddling would be refused downstream.
        """
        assert not Scale.from_limits(0.0, 10.0).straddles(centre), (
            f"a centre on the domain's own edge ({centre}) must not count as straddled"
        )

    def test_a_centre_that_is_not_a_number_straddles_nothing(self):
        """A non-finite centre is answered rather than compared.

        Test scenario:
            `nan` fails every comparison silently, so without this the reader would answer `False` by
            accident where the caller needs it to answer `False` on purpose.
        """
        assert not Scale.from_limits(-10.0, 10.0).straddles(float("nan")), (
            "a nan centre must not count as straddled"
        )
