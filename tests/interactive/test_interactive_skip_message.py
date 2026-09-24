"""What the interactive tier's lenient skip tells a caller it skipped a layer for (review R2-L9).

`MissingObject` was given an `OffLimbError` base (#325), which put "this layer's engine object is not in
this process" inside every ``except OffLimbError`` already written in the package — including the guard this
tier puts on its builders, `_skips_off_limb`. `tests/base/test_undrawable_layer.py` establishes that the
*catch* is right: both are a layer with nothing to draw, and `digitalearth.base.custom` already prescribes
leniency with ``strict=True`` as the way out.

What the catch did not bring with it is a message. `_skip_off_limb` was written for one cause and stated it
as a fact, so a missing engine object was reported as a projection failure:

    custom: none of the data could be placed in crs=3857, so the layer was skipped (layer 'wells' is a
            holoviews object this figure does not carry; add it again on the map that draws it). Build the
            map with InteractiveMap(strict=True) to raise instead.

A caller reading that goes and checks their CRS, which is fine. These hold the message to the cause: the two
situations arrive as two exception types, so the warning asks the error what happened rather than assuming
the warp.

Latent in `src/` today — no builder that draws a custom layer carries the guard — so the guard is run over
builders declared here, which is where the two halves meet.

The tier logs through **loguru**, which pytest's `caplog` does not see, so these attach a sink of their own
the way `tests/web/conftest.py` does for the web tier.
"""

import pytest

pytest.importorskip(
    "holoviews", reason="the interactive tier needs the interactive environment"
)
pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)

from digitalearth.base.crs import OffLimbError  # noqa: E402
from digitalearth.base.custom import MissingObject, held_object  # noqa: E402
from digitalearth.interactive import InteractiveMap  # noqa: E402
from digitalearth.interactive.base import _skips_off_limb  # noqa: E402

#: The layer both refusals name, so the two messages are read side by side.
LAYER = "wells"


@pytest.fixture
def warning_log():
    """Collect the tier's loguru warnings for the duration of one test.

    Yields:
        A list that grows with every WARNING-or-worse record emitted while the test runs, as strings.
    """
    from loguru import logger

    records = []
    sink = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink)


@pytest.fixture
def lenient():
    """Yield a map that skips an undrawable layer rather than raising.

    Yields:
        An `InteractiveMap` with `strict` left at its default.
    """
    built = InteractiveMap()
    yield built
    built.close()


def _meets_a_missing_object(tier):
    """Run the tier's off-limb guard over a builder whose custom object is not here.

    The guard rather than a real builder, because no builder that draws a custom layer carries it today.
    Declared as a function so the decorator wraps something with a name the guard can log, exactly as it
    logs a public builder's name.

    Args:
        tier: The map to run it on.

    Returns:
        Whatever the guard answers — the map itself when the layer was skipped.
    """

    @_skips_off_limb
    def custom(self):
        """Reach for a custom layer's held object, the way a drawer does.

        Args:
            self: The map the builder is bound to.

        Raises:
            MissingObject: always — nothing is held under this id.
        """
        held_object(
            LAYER, "custom:holoviews", {}, engine="holoviews", backend="interactive"
        )

    return custom(tier)


def _meets_data_the_warp_cannot_place(tier):
    """Run the same guard over a builder whose data the display CRS cannot place.

    Args:
        tier: The map to run it on.

    Returns:
        Whatever the guard answers — the map itself when the layer was skipped.
    """

    @_skips_off_limb
    def image(self):
        """Refuse the way a warp that placed none of the data refuses.

        Args:
            self: The map the builder is bound to.

        Raises:
            OffLimbError: always.
        """
        raise OffLimbError(
            f"layer {LAYER!r}: the reprojection placed none of the data on the map"
        )

    return image(tier)


class TestTheSkipNamesTheCauseItActuallyMet:
    """One guard, two causes, two messages — the catch is shared and the wording is not."""

    def test_the_missing_object_skip_is_announced_at_all(self, lenient, warning_log):
        """First, that there is a warning to read — the rest of the class asserts what is in it.

        Args:
            lenient: The map under test.
            warning_log: The loguru sink.

        Test scenario:
            Written because the wording checks below are negative or substring-shaped, and every one of
            them passes against an empty log. This is what stops the class going quietly green the day the
            tier stops warning at all.
        """
        _meets_a_missing_object(lenient)
        assert len(warning_log) == 1, warning_log

    def test_a_missing_engine_object_is_not_reported_as_a_projection_failure(
        self, lenient, warning_log
    ):
        """The defect: the warning blamed the display CRS for an object that is simply not here.

        Args:
            lenient: The map under test.
            warning_log: The loguru sink.

        Test scenario:
            Measured before the fix: `custom: none of the data could be placed in crs=3857, so the layer
            was skipped (layer 'wells' is a holoviews object this figure does not carry; add it again on
            the map that draws it)`. The CRS named there is correct and irrelevant — it is the one the
            caller chose, and nothing was ever warped.
        """
        _meets_a_missing_object(lenient)
        assert "could be placed in crs=" not in warning_log[0], warning_log[0]

    def test_it_says_the_object_is_not_in_this_process(self, lenient, warning_log):
        """Saying what it is not is half an answer; the message has to say what it is.

        Args:
            lenient: The map under test.
            warning_log: The loguru sink.
        """
        _meets_a_missing_object(lenient)
        assert "is not in this process" in warning_log[0], warning_log[0]

    def test_the_underlying_refusal_is_still_quoted(self, lenient, warning_log):
        """The cause line is a summary; the exception's own message is what names the layer.

        Args:
            lenient: The map under test.
            warning_log: The loguru sink.

        Test scenario:
            `held_object`'s message is the only place the layer id and the engine appear, so a rewording
            that dropped it would leave the caller with a warning naming the builder and nothing else.
        """
        _meets_a_missing_object(lenient)
        assert LAYER in warning_log[0], warning_log[0]

    def test_an_off_limb_warp_still_says_the_display_crs_placed_nothing(
        self, lenient, warning_log
    ):
        """The other half: the original wording is right for the cause it was written for.

        Args:
            lenient: The map under test.
            warning_log: The loguru sink.

        Test scenario:
            A branch that answered both causes the same way would be no better than the branch that
            answered neither. The CRS belongs in this message — it is the thing the caller would change.
        """
        _meets_data_the_warp_cannot_place(lenient)
        assert "could be placed in crs=" in warning_log[0], warning_log[0]

    def test_both_still_point_at_the_way_out(self, lenient, warning_log):
        """`strict=True` is how a caller turns either skip into a refusal, so both must say so.

        Args:
            lenient: The map under test.
            warning_log: The loguru sink.
        """
        _meets_a_missing_object(lenient)
        _meets_data_the_warp_cannot_place(lenient)
        pointed = sum(
            1 for line in warning_log if "InteractiveMap(strict=True)" in line
        )
        assert pointed == 2, warning_log


class TestStrictStillRefusesEither:
    """The wording change must not touch what `strict=True` does with either cause."""

    def test_a_strict_map_raises_the_missing_object(self):
        """The layer is fatal on a strict map, and the class the caller catches is unchanged.

        Test scenario:
            The message branch sits inside the lenient half; a branch written above the `strict` check
            would have swallowed the refusal this pins.
        """
        strict = InteractiveMap(strict=True)
        try:
            with pytest.raises(MissingObject):
                _meets_a_missing_object(strict)
        finally:
            strict.close()

    def test_a_strict_map_still_raises_an_off_limb_warp(self):
        """The same for the cause the guard was written for.

        Test scenario:
            Asserting both halves is what says the branch was placed below the `strict` check and not
            above it.
        """
        strict = InteractiveMap(strict=True)
        try:
            with pytest.raises(OffLimbError):
                _meets_data_the_warp_cannot_place(strict)
        finally:
            strict.close()
