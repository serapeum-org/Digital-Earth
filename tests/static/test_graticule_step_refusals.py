"""A graticule step that is not a positive, finite number of degrees is refused by name (R2-L4).

Every numeric keyword ``graticule()`` takes — ``spacing``, ``lon_step``, ``lat_step``; there is no count or
``n`` argument beside them — was measured against ``0``, a negative, ``nan``, ``inf`` and a non-integral
value on this branch's parent (``204d5749``), on ``Map(crs=4326)`` framed on :data:`WINDOW`:

```
keyword    case          outcome
spacing    zero          ZeroDivisionError: float division by zero
spacing    negative      drew: polylines=0 colls=1 segments=0 texts=0 warnings=[]
spacing    nan           ValueError: arange: cannot compute length
spacing    inf           ValueError: arange: cannot compute length
spacing    non-integral  drew: polylines=72 colls=1 segments=72 texts=18 warnings=[]
lon_step   zero          ZeroDivisionError: float division by zero
lon_step   negative      drew: polylines=5 colls=1 segments=5 texts=3 warnings=[]
lon_step   nan           ValueError: arange: cannot compute length
lon_step   inf           ValueError: arange: cannot compute length
lon_step   non-integral  drew: polylines=54 colls=1 segments=54 texts=12 warnings=[]
lat_step   zero          ZeroDivisionError: float division by zero
lat_step   negative      drew: polylines=13 colls=1 segments=13 texts=3 warnings=[]
lat_step   nan           ValueError: arange: cannot compute length
lat_step   inf           ValueError: arange: cannot compute length
lat_step   non-integral  drew: polylines=36 colls=1 segments=36 texts=12 warnings=[]
```

Two defects in one table. A **negative** step drew a described grid with nothing in it — ``spacing=-30.0``
put an empty ``LineCollection`` on the axes and warned nothing, which is the figure naming a layer that
draws nothing, the shape this tier refuses by name everywhere else. ``0``, ``nan`` and ``inf`` did refuse,
but in the projection's own words (``arange: cannot compute length``, ``float division by zero``), naming
neither the method nor the keyword that was wrong.

The **non-integral** row is the one that must not change: ``spacing=7.5`` is documented as drawing here and
raising on the interactive tier, so it is a legal step and is asserted as one below.

A later probe found the row this table missed — a step that is finite, positive and so passes every guard
above, yet is small enough to build a grid that never finishes. ``projections.graticule`` runs its
meridians across the whole ``-180..180`` and its parallels across ``-90..90``, never consulting the view,
so the polyline count is ``~360/lon_step + ~180/lat_step`` whatever the map is looking at, and each line is
a 200-point pyramids reprojection at ~0.6 ms. Measured on ``Map(crs=4326)`` framed to the world:

```
keyword    case          outcome
spacing    tiny (1e-4)   ~5,400,000 lines + ~1,400,000 Text, 0 warnings, ran past 120s (never returned)
lon_step   tiny (1e-4)   ~3,600,000 meridians, same hang
lat_step   tiny (1e-4)   ~1,800,000 parallels, same hang
```

So the step now carries a lower bound as well, by the same by-keyword ``ValueError`` — a ceiling on the
whole-world line count (:data:`~digitalearth.static.maps.projection.MAX_GRATICULE_LINES`), which a step
below :data:`MERIDIAN_FLOOR` / :data:`PARALLEL_FLOOR` would exceed. The bound is a function of the step
alone, because the line count is: the view never enters it. The refusal is pure arithmetic, so it lands
before a single line is projected — which is what the hang-guarded tests below prove, by refusing in
milliseconds where the unbounded call ran for minutes.
"""

import math
import threading
from collections.abc import Callable

import pytest

from digitalearth.static import Map
from digitalearth.static.maps.projection import MAX_GRATICULE_LINES

#: Every numeric keyword ``graticule()`` takes. There is no third kind — no count, no ``n`` — so these three
#: are the whole surface a bad number can arrive through.
STEP_KEYWORDS = ("spacing", "lon_step", "lat_step")

#: The values a step may not take, by case name. ``nan`` and ``inf`` sit beside the two non-positive numbers
#: because the refusal is about a step being a usable number of degrees, and neither of those is one.
REFUSED_STEPS = {"zero": 0.0, "negative": -30.0, "nan": math.nan, "inf": math.inf}

#: A window, as ``(west, south, east, north)`` in EPSG:4326, the measurements above were taken on.
WINDOW = [-35.0, -5.0, 35.0, 65.0]

#: The step a caller who names none gets, and the one the untouched half of a one-sided call keeps.
DEFAULT_STEP = 30.0

#: A legal fractional step: documented as drawing on this tier and raising on the interactive one.
FRACTIONAL_STEP = 7.5

#: A step that is finite, positive and non-integral — so it clears every sign/finiteness guard — and tiny
#: enough that the unbounded grid is ~5.4M world-spanning lines, which never returns. The gap this row
#: closes.
TINY_STEP = 1e-4

#: The coarsest meridian step the line budget refuses, in degrees: a longitude spans ``360``, so a step
#: below ``360 / MAX`` would draw more than :data:`MAX_GRATICULE_LINES` meridians. ``spacing`` is held to
#: this one too, since it sets both and the meridian axis is the denser.
MERIDIAN_FLOOR = 360.0 / MAX_GRATICULE_LINES

#: The coarsest parallel step the budget refuses: a latitude spans ``180``, so half the meridian floor.
PARALLEL_FLOOR = 180.0 / MAX_GRATICULE_LINES

#: A step comfortably **above** :data:`MERIDIAN_FLOOR`, whose whole-world grid (``~1800`` lines) draws in
#: about a second. The finest step the bound still allows, used to pin that it does not refuse a dense but
#: drawable grid.
FINE_BUT_LEGAL_STEP = 0.2

#: A step **below** :data:`MERIDIAN_FLOOR` but far above the hang — ``~3600`` meridians, which the bound
#: refuses. Coarse enough that an *unbounded* build (the regression case) still returns in a few seconds
#: rather than hanging the suite, so the red state of this bound is observable without a timeout guard.
OVER_BUDGET_STEP = 0.1


def _raises_within(
    call: Callable[[], object], timeout: float
) -> tuple[bool, BaseException | None]:
    """Run ``call`` in a daemon thread and report whether it finished, and with what.

    The refusal this guards must land *before* any line is projected, so a regression that drops the
    lower bound does not merely draw a slow grid — it builds millions of lines and never returns. A plain
    elapsed-time assertion would only fail once that build finished, i.e. minutes later or never. Running
    the call in a daemon thread and joining with a timeout makes the regression fail *fast*: the worker is
    abandoned (it dies with the process), and the test sees that the call did not finish.

    Args:
        call: The zero-argument call under test, expected to raise at once.
        timeout: How long to wait, in seconds, before declaring the call hung.

    Returns:
        A ``(finished, raised)`` pair: whether the call returned or raised within ``timeout``, and the
        exception it raised (``None`` if it returned a value or is still running).
    """
    captured: dict = {}

    def _run() -> None:
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - re-asserted on the main thread
            captured["error"] = error
        captured["done"] = True

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    return captured.get("done", False), captured.get("error")


def _line_count(meridian_step: float, parallel_step: float) -> int:
    """Return how many lines a grid cut at these two steps holds.

    Built from the degrees the steps reach rather than read back off a map, so a step that was quietly
    swallowed could not satisfy an assertion against it.

    Args:
        meridian_step: Meridian spacing in degrees.
        parallel_step: Parallel spacing in degrees.

    Returns:
        The meridians, which include both datelines, plus the parallels, which exclude both poles.

    Examples:
        - The default grid is 13 meridians and 5 parallels:
            ```python
            >>> _line_count(30.0, 30.0)
            18

            ```
    """
    meridians = 2 * int(180.0 // meridian_step) + 1
    parallels = sum(
        1 for index in range(-360, 361) if abs(index * parallel_step) < 90.0
    )
    return meridians + parallels


@pytest.fixture
def framed():
    """Yield a flat lon/lat map framed on :data:`WINDOW`, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` with the view the steps are measured against.
    """
    scene = Map(crs=4326)
    scene.set_bounds(WINDOW)
    yield scene
    scene.close()


@pytest.mark.parametrize("keyword", STEP_KEYWORDS)
@pytest.mark.parametrize("value", list(REFUSED_STEPS.values()), ids=list(REFUSED_STEPS))
class TestEveryStepKeywordRefusesEveryUnusableNumber:
    """The 12 cells of the matrix above, each a ``ValueError`` that names what was wrong."""

    def test_it_is_refused_as_a_value_error(self, framed, keyword, value):
        """A step that is not a positive finite number raises, rather than drawing an empty grid.

        Args:
            framed: A flat lon/lat map framed on :data:`WINDOW`.
            keyword: Which of the three numeric keywords carries the value.
            value: The unusable step.

        Test scenario:
            Three of the twelve cells raised nothing at all before this — the negative row drew a described
            grid with no lines in it — and the other nine raised numpy's or Python's arithmetic message.
        """
        with pytest.raises(ValueError):
            framed.graticule(**{keyword: value})

    def test_the_refusal_names_the_keyword_that_was_wrong(self, framed, keyword, value):
        """The message says which argument to fix, which the projection's own message cannot.

        Args:
            framed: A flat lon/lat map framed on :data:`WINDOW`.
            keyword: Which of the three numeric keywords carries the value.
            value: The unusable step.

        Test scenario:
            ``float division by zero`` and ``arange: cannot compute length`` name neither ``graticule()``
            nor the keyword, so a caller reading either has to find the argument themselves.
        """
        with pytest.raises(ValueError, match=rf"graticule\(\) was given {keyword}="):
            framed.graticule(**{keyword: value})

    def test_nothing_is_described_after_the_refusal(self, framed, keyword, value):
        """The refusal lands before the layer is described, so the figure names no grid.

        Args:
            framed: A flat lon/lat map framed on :data:`WINDOW`.
            keyword: Which of the three numeric keywords carries the value.
            value: The unusable step.

        Test scenario:
            The negative row is what this is about: it described ``graticule-1`` and drew an empty
            collection for it, so the figure claimed a grid the axes did not hold.
        """
        with pytest.raises(ValueError):
            framed.graticule(**{keyword: value})
        assert framed.layer_ids == [], (
            f"a refused {keyword}={value!r} left {framed.layer_ids} described"
        )

    def test_nothing_is_drawn_after_the_refusal(self, framed, keyword, value):
        """And the axes holds no collection either, which is where the empty grid went.

        Args:
            framed: A flat lon/lat map framed on :data:`WINDOW`.
            keyword: Which of the three numeric keywords carries the value.
            value: The unusable step.

        Test scenario:
            Counted on the axes rather than in the record, because the defect was an artist the figure
            described: ``spacing=-30.0`` measured ``colls=1`` with ``segments=0``.
        """
        with pytest.raises(ValueError):
            framed.graticule(**{keyword: value})
        assert len(framed.ax.collections) == 0, (
            f"a refused {keyword}={value!r} left {len(framed.ax.collections)} collection(s)"
        )


@pytest.mark.parametrize("keyword", STEP_KEYWORDS)
class TestANonIntegralStepIsStillLegal:
    """The row of the matrix that must not change: a fractional step is this tier's own, and draws."""

    def test_it_draws_every_line_it_reaches(self, framed, keyword):
        """``spacing=7.5`` draws here and raises on the interactive tier, as the docstring says.

        Args:
            framed: A flat lon/lat map framed on :data:`WINDOW`.
            keyword: Which of the three numeric keywords carries the value.

        Test scenario:
            The count comes from :func:`_line_count`, built from the degrees the steps reach, so a refusal
            that rejected the fraction could not satisfy it. A ``lon_step`` alone leaves the parallels at
            the default and a ``lat_step`` alone the meridians, so the three keywords expect three counts.
        """
        meridian = FRACTIONAL_STEP if keyword != "lat_step" else DEFAULT_STEP
        parallel = FRACTIONAL_STEP if keyword != "lon_step" else DEFAULT_STEP
        expected = _line_count(meridian, parallel)
        framed.graticule(**{keyword: FRACTIONAL_STEP})
        (collection,) = framed.ax.collections
        drawn = len(collection.get_segments())
        assert drawn == expected, (
            f"{keyword}=7.5 should draw {expected} lines; got {drawn}"
        )


class TestARefusedReplacementLeavesTheGridItHad:
    """A step refused on a *replacing* call must not disturb the grid the map is still drawing."""

    def test_the_description_still_carries_the_first_step(self, framed):
        """The figure keeps naming the 30-degree grid the axes holds.

        Args:
            framed: A flat lon/lat map framed on :data:`WINDOW`.

        Test scenario:
            The refusal lands before the description is edited, so there is nothing to put back — the same
            guarantee the rollback gave, reached one step earlier.
        """
        framed.graticule(lon_step=DEFAULT_STEP)
        held = framed.layer_ids[0]
        with pytest.raises(ValueError):
            framed.graticule(lon_step=-45.0)
        props = framed.figure_spec.layers.get(held).symbology.props
        assert props["lon_step"] == DEFAULT_STEP, (
            f"the refused replacement rewrote the step to {props['lon_step']}"
        )

    def test_the_lines_on_the_axes_are_the_ones_it_was_drawing(self, framed):
        """And the collection still holds the 30-degree grid, counted on the axes.

        Args:
            framed: A flat lon/lat map framed on :data:`WINDOW`.

        Test scenario:
            ``lon_step=-45.0`` used to re-cut the collection to 5 segments without warning, so the grid on
            the axes disagreed with the 18 lines the figure described.
        """
        framed.graticule(lon_step=DEFAULT_STEP)
        with pytest.raises(ValueError):
            framed.graticule(lon_step=-45.0)
        (collection,) = framed.ax.collections
        drawn = len(collection.get_segments())
        assert drawn == _line_count(DEFAULT_STEP, DEFAULT_STEP), (
            f"the refused replacement left {drawn} lines on the axes"
        )


@pytest.mark.parametrize("keyword", STEP_KEYWORDS)
class TestATinyPositiveStepIsRefusedBeforeItCanHang:
    """The row the first L4 pass missed: a step small enough to build a grid that never returns.

    ``1e-4`` is finite, positive and non-integral, so every guard the other cases trip is silent on it —
    yet the whole-world grid at that step is millions of lines, each a pyramids reprojection, and the call
    ran past two minutes without returning. These tests run the call in a daemon thread
    (:func:`_raises_within`) so a regression that drops the bound fails *fast* — by the join timeout —
    rather than hanging the suite, and they never let the unbounded build run to completion.
    """

    def test_it_finishes_rather_than_hanging(self, keyword):
        """The call returns control almost at once, because the refusal precedes any projection.

        Args:
            keyword: Which of the three numeric keywords carries the tiny step.

        Test scenario:
            Without the bound the worker builds ~5.4M lines and the join times out, so ``finished`` is
            ``False`` and this fails in a second or two instead of after the minutes-long build.
        """
        finished, _ = _raises_within(
            lambda: Map(crs=4326).graticule(**{keyword: TINY_STEP}), timeout=5.0
        )
        assert finished, (
            f"graticule({keyword}={TINY_STEP!r}) did not return within the timeout: it is building the "
            "grid rather than refusing the step"
        )

    def test_it_refuses_with_a_value_error_naming_the_keyword(self, keyword):
        """And what it returns with is the by-keyword refusal, not a drawn grid.

        Args:
            keyword: Which of the three numeric keywords carries the tiny step.

        Test scenario:
            The companion test pins that it returned; this one pins *how*. A fixed build would finish with
            no exception, which this reads as a failure to refuse.
        """
        _, raised = _raises_within(
            lambda: Map(crs=4326).graticule(**{keyword: TINY_STEP}), timeout=5.0
        )
        assert isinstance(raised, ValueError), (
            f"graticule({keyword}={TINY_STEP!r}) should raise a ValueError, not hang; got {raised!r}"
        )
        assert keyword in str(raised), (
            f"the refusal of graticule({keyword}={TINY_STEP!r}) should name {keyword!r}; got {raised!r}"
        )


class TestTheLineBudgetBracketsTheFloor:
    """A step just over the budget is refused; one just under it draws, so the floor is where it says."""

    def test_a_step_over_the_budget_is_refused(self):
        """A meridian step below :data:`MERIDIAN_FLOOR` draws more than the budget and is refused.

        Test scenario:
            ``0.1`` degrees is ~3600 meridians, past the ``MAX_GRATICULE_LINES`` ceiling. The refusal is
            arithmetic on the step, so it does not build the grid to find out.
        """
        scene = Map(crs=4326)
        with pytest.raises(ValueError, match="gridlines"):
            scene.graticule(spacing=OVER_BUDGET_STEP)
        scene.close()

    def test_a_step_under_the_budget_still_draws(self):
        """A step just above the floor is a dense but drawable grid, and the bound leaves it alone.

        Test scenario:
            ``0.2`` degrees is ~1800 meridians, under the ceiling, so it draws — which brackets the floor
            between it and :data:`OVER_BUDGET_STEP` and proves the bound does not refuse a legal dense grid.
        """
        scene = Map(crs=4326)
        scene.graticule(spacing=FINE_BUT_LEGAL_STEP, labels=False)
        drawn = len(scene.artist("graticule-1"))
        scene.close()
        assert drawn > MAX_GRATICULE_LINES // 2, (
            f"a legal dense grid should still draw its lines; got {drawn}"
        )

    def test_the_meridian_floor_is_the_budget_over_the_longitude_span(self):
        """The floor is stated as a count ceiling, and its degree value follows from the 360-degree span.

        Test scenario:
            A pure arithmetic check that the published floor is exactly ``360 / MAX_GRATICULE_LINES`` — no
            map built — so the constant and the refusal boundary cannot drift apart.
        """
        assert MERIDIAN_FLOOR == pytest.approx(360.0 / MAX_GRATICULE_LINES), (
            MERIDIAN_FLOOR
        )

    def test_the_parallel_floor_is_half_the_meridian_floor(self):
        """A latitude spans half a longitude, so its floor is half — the other half of the same rule.

        Test scenario:
            Also arithmetic: the two floors are the one ceiling applied to the two spans.
        """
        assert PARALLEL_FLOOR == pytest.approx(MERIDIAN_FLOOR / 2.0), PARALLEL_FLOOR
