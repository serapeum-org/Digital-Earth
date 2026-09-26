"""The interval-to-levels arithmetic ``contours(interval=)`` needs, held to one answer for every tier.

``interval=`` is one of the three keywords the Tier-2 contract declares on ``contours``, and it has to mean
the same thing wherever it appears: one level every N, in the band's own units. Where the levels are
*computed* differs per tier — the web tier hands the spacing to pyramids, which walks the band itself, while
the static and interactive tiers have to hand their engine a finished list — so the arithmetic is the part
that can drift, and :func:`~digitalearth.base.levels.levels_every` is the one place it lives.

Every check below fixes a property of the *arithmetic*, not of a picture: which multiples are inside, which
arguments have no levels to give, and what each refusal says. They run in the lean ``dev`` environment, which
is the point of the function living in ``base/`` at all.
"""

import math
import time

import numpy as np
import pytest

from digitalearth.base.levels import MAX_LEVELS, levels_every

#: A band running 0-399, the same range the static tier's contour checks use, so the two agree on the
#: measured answer for one interval rather than on two independently chosen fixtures.
BAND = np.arange(400, dtype="float32").reshape(20, 20)


class TestOnlyTheMultiplesInsideTheBand:
    """A level at or beyond an extreme traces the frame's edge or nothing, so it is not a level."""

    def test_an_interval_of_a_hundred_over_nought_to_399(self):
        """The measured answer, which is also what the web tier produces for the same call.

        Test scenario:
            0 is a multiple of 100 and sits exactly on the minimum; 400 is the next one and sits past the
            maximum. Both are excluded, so three levels are left — and that exclusion is the whole claim.
        """
        assert levels_every(BAND, 100.0) == [100.0, 200.0, 300.0]

    def test_the_gap_between_levels_is_the_interval(self):
        """The spacing is the argument's own property, whatever range it is applied to."""
        levels = levels_every(BAND, 37.0)
        gaps = {round(later - earlier, 9) for earlier, later in zip(levels, levels[1:])}
        assert gaps == {37.0}, levels

    def test_an_offset_range_still_lands_on_the_multiples(self):
        """The levels are multiples of the interval, not offsets from the band's minimum.

        Test scenario:
            A band running 250-1250 at an interval of 500 crosses 500 and 1000 only. Counting from the
            minimum instead would have traced 750 and 1250 — a different set, from the same spacing.
        """
        offset = np.linspace(250.0, 1250.0, 64)
        assert levels_every(offset, 500.0) == [500.0, 1000.0]

    def test_the_levels_come_back_as_plain_floats(self):
        """A figure is written down with these, so a ``numpy.float32`` cannot be one of them."""
        assert {type(level) for level in levels_every(BAND, 100.0)} == {float}

    def test_the_values_may_be_any_shape(self):
        """The band arrives as the drawer read it — 2-D here, and a flat sequence is the same band."""
        assert levels_every(BAND, 100.0) == levels_every(BAND.ravel().tolist(), 100.0)

    def test_a_band_with_gaps_is_measured_on_what_it_does_have(self):
        """``NaN`` is a hole in the data, not a value the range has to reach."""
        holed = BAND.astype("float64").copy()
        holed[holed > 150.0] = math.nan
        assert levels_every(holed, 100.0) == [100.0]


class TestWhatHasNoLevelsToGive:
    """Each of these would otherwise surface from inside the engine, as a complaint about a level list."""

    @pytest.mark.parametrize("interval", [0.0, -100.0, math.nan, math.inf])
    def test_a_spacing_that_cannot_space_is_refused(self, interval):
        """Zero, negative and non-finite spacings each trace nothing.

        Args:
            interval: The spacing under test.
        """
        with pytest.raises(ValueError) as refused:
            levels_every(BAND, interval)
        assert "positive spacing" in str(refused.value), refused.value

    def test_an_empty_band_is_refused(self):
        """No finite value means no range, and a guessed level would draw a line nothing supports."""
        with pytest.raises(ValueError) as refused:
            levels_every(np.full((4, 4), math.nan), 100.0)
        assert "no values to space levels through" in str(refused.value), refused.value

    def test_a_constant_band_is_refused(self):
        """One value is not a range: no multiple can lie strictly between it and itself."""
        with pytest.raises(ValueError) as refused:
            levels_every(np.full((4, 4), 5.0), 100.0)
        assert "crosses no level inside the band's range" in str(refused.value), (
            refused.value
        )

    def test_an_interval_coarser_than_the_band_is_refused_by_its_range(self):
        """The refusal names the range it could not cross, which is what the caller has to change."""
        with pytest.raises(ValueError) as refused:
            levels_every(BAND, 10_000.0)
        assert "0.0 to 399.0" in str(refused.value), refused.value


class TestAnIntervalTooFineToDrawIsRefused:
    """L6 — the walk is bounded, so a mistyped unit cannot allocate until the process dies.

    ``interval`` was validated as positive and finite but never against the band's range, so the intermediate
    ``arange`` was as long as the caller's typo made it: ``levels_every(dem, 1e-3)`` on a 0-8848 m DEM builds
    ~8.8 M float64s and then walks them in a Python comprehension. A metres-versus-kilometres slip therefore
    turned a contour call into a multi-second stall, or a ``MemoryError`` raised from inside ``base/`` with
    nothing naming the argument that caused it.
    """

    #: A band spanning the height of Everest, which is where the metres/kilometres slip actually happens.
    DEM = np.linspace(0.0, 8848.0, 256)

    def test_a_count_above_the_ceiling_is_refused(self):
        """An interval that would cut more than :data:`MAX_LEVELS` levels raises instead of allocating.

        Test scenario:
            The refusal has to come before the array, or the guard is a report on a cost already paid.
        """
        with pytest.raises(ValueError) as excinfo:
            levels_every(self.DEM, 1e-3)
        message = str(excinfo.value)
        assert "0.001" in message, (
            f"the refusal must name the interval that caused it, got: {message}"
        )
        assert str(MAX_LEVELS) in message, (
            f"the refusal must name the ceiling it exceeded, got: {message}"
        )

    def test_the_refusal_names_how_many_levels_were_asked_for(self):
        """The count is the number that tells a caller they meant kilometres, so it has to be in the message."""
        with pytest.raises(ValueError) as excinfo:
            levels_every(self.DEM, 1e-3)
        asked = len(str(8_847_999))
        assert any(len(token) >= asked for token in str(excinfo.value).split()), (
            f"no token in the refusal is as long as the level count it should name: {excinfo.value}"
        )

    def test_it_is_refused_before_the_array_is_built(self):
        """The guard reads two floats, so the interval that would exhaust memory costs nothing to refuse.

        Test scenario:
            A ceiling checked after `np.arange` would still allocate what it then complains about. Timing is
            the only way to tell the two apart, and the margin is enormous: the unguarded walk over this band
            at this interval took ~2 s, against a refusal that reads `low`, `high` and divides.
        """
        started = time.perf_counter()
        with pytest.raises(ValueError):
            levels_every(self.DEM, 1e-6)
        spent = time.perf_counter() - started
        assert spent < 0.5, (
            f"refusing an interval of 1e-06 took {spent:.3f}s, so the array was built first"
        )

    def test_a_count_at_the_ceiling_still_draws(self):
        """The ceiling is inclusive: the largest legal walk is not refused with the illegal ones.

        Test scenario:
            An off-by-one here is a silent loss of a legitimate figure, which is worse than the stall the
            guard exists to prevent.
        """
        band = np.array([0.0, float(MAX_LEVELS + 1)])
        cut = levels_every(band, 1.0)
        assert len(cut) == MAX_LEVELS, (
            f"a band asking for exactly {MAX_LEVELS} levels cut {len(cut)}"
        )

    def test_one_more_than_the_ceiling_is_refused(self):
        """And the first illegal walk is the one past it, built the other way round from the test above."""
        band = np.array([0.0, MAX_LEVELS + 2.0])
        with pytest.raises(ValueError) as excinfo:
            levels_every(band, 1.0)
        assert "interval=1.0" in str(excinfo.value), (
            f"the refusal must name the interval, got: {excinfo.value}"
        )
