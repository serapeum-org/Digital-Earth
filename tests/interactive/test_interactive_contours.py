"""One ``contours(levels=, interval=, filled=)`` on the interactive tier (#262).

The Tier-2 contract declares ``contours`` with ``{levels, interval, filled}`` and says a name that appears on
more than one tier must mean one thing wherever it appears. This tier took ``levels=`` alone: ``filled=`` was
a second method (``filled_contours``) and ``interval=`` was nothing at all — it fell through ``**opts`` to
HoloViews, which refused it as an unknown *style* option, so a caller moving a working web or static call
here got ``Unexpected option 'interval' for Contours type``.

Every check reads what the **engine** produced, not what the figure recorded: HoloViews' ``contours``
operation returns a ``Contours`` element for lines and a ``Polygons`` one for filled bands, and the traced
levels are the element's own value dimension. A ``filled=`` or an ``interval=`` written down and not acted on
would leave the wrong element, or the wrong levels, there.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.interactive import InteractiveMap

pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)

#: The levels the checks trace at, chosen inside the fixture's range so every one draws something. The same
#: three the static tier's contour checks use, and the same three an interval of 100 finds there.
LEVELS = [100.0, 200.0, 300.0]


def _raster(values=None):
    """Return a small single-band raster, running 0-399 unless given other values.

    Args:
        values: A 20x20 array to use instead of the 0-399 ramp — a constant band, for the checks that need
            one with no range to space levels through.

    Returns:
        A pyramids ``Dataset`` in EPSG:4326.
    """
    band = np.arange(400, dtype="float32").reshape(20, 20) if values is None else values
    return Dataset.from_array(
        band,
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
    )


def _traced(element) -> list:
    """Return the iso-values the engine actually traced, ascending.

    Args:
        element: The HoloViews element a contour render produced.

    Returns:
        The finite values of the element's value dimension, de-duplicated and sorted. The engine writes one
        value per traced path and separates the paths with ``NaN``, so the finite set *is* the level set —
        read off the element rather than off anything this package wrote down.
    """
    name = element.vdims[0].name
    return sorted(
        {float(value) for value in element.dimension_values(name) if np.isfinite(value)}
    )


@pytest.fixture
def drawn():
    """Yield an empty interactive map in EPSG:4326, closed on the way out.

    Yields:
        The tier's ``InteractiveMap``.
    """
    scene = InteractiveMap(crs=4326)
    yield scene
    scene.close()


class TestOneMethodDrawsBothRenders:
    """``filled=`` is the switch, and it has to reach the contour operation."""

    def test_lines_by_default(self, drawn):
        """Unfilled is the default, matching the Tier-2 declaration and the other tiers.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), levels=LEVELS)
        assert type(drawn.layers[0]).__name__ == "Contours"

    def test_filled_draws_bands(self, drawn):
        """``filled=True`` draws the bands between the levels — a polygon element, not paths.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), levels=LEVELS, filled=True)
        assert type(drawn.layers[0]).__name__ == "Polygons"

    def test_the_levels_asked_for_are_the_levels_traced(self, drawn):
        """``levels=`` reaches the engine, which is the only place it can have an effect.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), levels=LEVELS)
        assert _traced(drawn.layers[0]) == LEVELS

    def test_the_separate_filled_method_still_fills(self, drawn):
        """``filled_contours`` is a promise to every script already written against this tier.

        Args:
            drawn: The map under test.
        """
        drawn.filled_contours(_raster(), levels=LEVELS)
        assert type(drawn.layers[0]).__name__ == "Polygons"


class TestAnIntervalIsSpacing:
    """``interval=`` is the Tier-2 keyword for "one level every N", and the other two tiers take it."""

    def test_an_interval_traces_evenly_spaced_levels(self, drawn):
        """The spacing between successive traced levels is the interval asked for.

        Args:
            drawn: The map under test.

        Test scenario:
            The fixture runs 0-399, so an interval of 100 crosses 100, 200 and 300 — the multiples strictly
            inside the data, which is the answer the web tier's pyramids-side walk gives for the same call.
            Asserted as the gap between traced levels rather than as a list, because which multiples fall
            inside is a property of the data while the gap is the property of the argument.
        """
        drawn.contours(_raster(), interval=100.0)
        levels = _traced(drawn.layers[0])
        gaps = {round(later - earlier, 9) for earlier, later in zip(levels, levels[1:])}
        assert gaps == {100.0}, levels

    def test_an_interval_crosses_the_multiples_inside_the_band(self, drawn):
        """The measured set, which is the one the web tier produces for the same band and spacing.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), interval=100.0)
        assert _traced(drawn.layers[0]) == [100.0, 200.0, 300.0]

    def test_an_interval_fills_too(self, drawn):
        """The two new keywords compose: a spacing can describe filled bands as well as lines.

        Args:
            drawn: The map under test.
        """
        drawn.contours(_raster(), interval=100.0, filled=True)
        assert type(drawn.layers[0]).__name__ == "Polygons"

    def test_an_interval_and_levels_together_is_refused(self, drawn):
        """Two ways of asking for one thing, so one of them has to go — the web tier's own wording.

        Args:
            drawn: The map under test.
        """
        with pytest.raises(ValueError) as refused:
            drawn.contours(_raster(), levels=LEVELS, interval=100.0)
        assert "at most one of interval= or levels=" in str(refused.value), (
            refused.value
        )

    def test_an_interval_over_a_constant_band_is_refused(self, drawn):
        """A band with no range has no levels to space through, and guessing one draws a lie.

        Args:
            drawn: The map under test.
        """
        flat = np.full((20, 20), 5.0, dtype="float32")
        with pytest.raises(ValueError) as refused:
            drawn.contours(_raster(flat), interval=100.0)
        assert "crosses no level inside the band's range" in str(refused.value), (
            refused.value
        )

    def test_a_refused_interval_leaves_no_layer_behind(self, drawn):
        """The refusal comes from the drawer, so the description must not outlive it.

        Args:
            drawn: The map under test.
        """
        flat = np.full((20, 20), 5.0, dtype="float32")
        with pytest.raises(ValueError):
            drawn.contours(_raster(flat), interval=100.0)
        assert drawn.layer_ids == []
