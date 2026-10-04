"""One ``contours(filled=)`` on the static tier, and the two spellings it replaced (#262).

The Tier-2 contract declares ``contours`` with ``{levels, interval, filled}`` and says a name that appears on
more than one tier must mean one thing wherever it appears. This tier had **two** methods instead —
``contour`` for lines and ``contourf`` for filled bands — so a caller moving a script from the web or
interactive tier had to know which of the two to write, and ``filled=`` meant nothing here. Both are deleted:
``filled=`` is the argument that picks the render.

Every check reads what landed on the axes, because the two renders differ in the artist they produce: lines
are a ``QuadContourSet`` of paths, filled bands one of polygons. A ``filled=`` recorded and not acted on
would leave the wrong one there.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map

#: The levels the checks trace at, chosen inside the fixture's range so every one draws something.
LEVELS = [100.0, 200.0, 300.0]


def _raster():
    """Return a small single-band raster whose values run 0-399.

    Returns:
        A pyramids ``Dataset`` in EPSG:4326.
    """
    return Dataset.from_array(
        np.arange(400, dtype="float32").reshape(20, 20),
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
    )


@pytest.fixture
def drawn():
    """Yield an empty map in EPSG:4326, closed on the way out.

    Yields:
        The tier's ``Map``.
    """
    scene = Map(crs=4326)
    yield scene
    scene.close()


def _filled(artist) -> bool:
    """Say whether a contour set draws filled bands rather than lines.

    Args:
        artist: The ``QuadContourSet`` a contour render produced.

    Returns:
        ``True`` for filled bands. Read off the set's own ``filled`` flag, which is matplotlib's record of
        which of its two renders ran — not off anything this package wrote down.
    """
    return bool(artist.filled)


class TestOneMethodDrawsBothRenders:
    """``filled=`` is the switch, and it has to reach matplotlib."""

    def test_lines_by_default(self, drawn):
        """Unfilled is the default, matching the Tier-2 declaration and the other tiers.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), levels=LEVELS)
        assert _filled(drawn.artist()) is False

    def test_filled_draws_bands(self, drawn):
        """``filled=True`` draws the bands between the levels.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), levels=LEVELS, filled=True)
        assert _filled(drawn.artist()) is True

    def test_the_levels_asked_for_are_the_levels_drawn(self, drawn):
        """``levels=`` reaches the engine, which is the only place it can have an effect.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), levels=LEVELS)
        artist = drawn.artist()
        assert [float(level) for level in artist.levels] == LEVELS, artist.levels


class TestAnIntervalIsSpacing:
    """``interval=`` is the Tier-2 keyword for "one level every N", and web already takes it."""

    def test_an_interval_traces_evenly_spaced_levels(self, drawn):
        """The spacing between successive levels is the interval asked for.

        Args:
            drawn: The map under test.

        Test scenario:
            The fixture runs 0-399, so an interval of 100 traces 100, 200 and 300 — the multiples that fall
            inside the data. Asserted as the gap between drawn levels rather than as a list, because which
            multiples fall inside is a property of the data and the gap is the property of the argument.
        """
        drawn.contours(_raster(), interval=100.0)
        artist = drawn.artist()
        levels = [float(level) for level in artist.levels]
        gaps = {round(b - a, 9) for a, b in zip(levels, levels[1:])}
        assert gaps == {100.0}, levels

    def test_an_interval_and_levels_together_is_refused(self, drawn):
        """Two ways of asking for one thing, so one of them has to go.

        Args:
            drawn: The map under test.
        """
        raster = _raster()
        with pytest.raises(ValueError) as refused:
            drawn.contours(raster, levels=LEVELS, interval=100.0)
        assert "at most one of interval= or levels=" in str(refused.value), (
            refused.value
        )


class TestAnIntervalTooFineToDrawIsRefused:
    """A spacing that would cut more levels than a figure can carry is refused before the walk runs."""

    def test_the_static_tier_refuses_by_count_like_the_tier_it_shares_the_walk_with(
        self, drawn
    ):
        """An interval fine enough to cut millions of levels is refused, naming the ceiling.

        Args:
            drawn: The map under test.

        Test scenario:
            The fixture runs 0-399, so `interval=1e-5` crosses ~39.9 million levels. Walking them costs
            seconds and gigabytes of `float64`, and a finer interval costs proportionally more, so the
            spacing is refused by count rather than walked. Asserted against the shared `MAX_LEVELS`
            rather than a literal, so this tier and the interval walk cannot drift apart again.
        """
        from digitalearth.base.levels import MAX_LEVELS

        raster = _raster()
        with pytest.raises(ValueError) as refused:
            drawn.contours(raster, interval=1e-5)
        message = str(refused.value)
        assert str(MAX_LEVELS) in message, (
            f"the refusal should name the ceiling {MAX_LEVELS}; got {message}"
        )

    def test_a_spacing_the_band_can_carry_still_draws(self, drawn):
        """The ceiling refuses only the unreasonable: an ordinary spacing is untouched.

        Args:
            drawn: The map under test.

        Test scenario:
            The same fixture at `interval=100.0` traces three levels, far under the ceiling. This is the
            positive control for the refusal above — a guard that refused everything would pass without it.
        """
        drawn.contours(_raster(), interval=100.0)
        artist = drawn.artist()
        assert len(artist.levels) >= 3, (
            f"an ordinary interval should still draw; got {artist.levels}"
        )
