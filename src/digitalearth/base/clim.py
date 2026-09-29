"""Which frames of a raster time stack are read, and what they measure — for every rendering backend.

A stack rendered frame by frame has to freeze one domain across the whole series, or the scale jumps between
frames and the animation reads as data that is not there. Deriving that range is pure arithmetic over arrays —
no renderer needed — so it lives here rather than three times over.

**What is this module's and what is `Scale`'s.** This module answers two questions no
:class:`~digitalearth.base.spec.scale.Scale` builder can: *which* frames to open (:func:`sample_evenly`, a
cost decision, taken before anything is read) and *what a sequence of frames measures* (:func:`measure_clim`,
which unions per-frame extremes, fills a nodata mask before reducing, and can answer "nothing at all"). Every
`Scale` builder takes a single array, drops a mask through ``np.asarray``, and must return a scale rather than
nothing. What the measurement then *means as a domain* — the fallback for an unmeasurable stack, and the
widening of a constant one — is `Scale`'s, and :func:`stack_scale` is where the two meet. Until DE-40 this
module decided that too, in a ``stack_clim`` that fell back to ``(0.0, 1.0)`` and did **not** widen, so the
stack paths were the one family of call sites running a second, weaker copy of the domain rule.

**What DE-40 took out of this module's public surface, and what the replacement returns.**
``stack_clim(arrays) -> (float, float)`` was in ``__all__`` here and is **gone**, with no alias behind it:
`hasattr(digitalearth.base.clim, "stack_clim")` is `False`, and every caller moved in the same commit — no
call site in ``src/``, ``tests/`` or ``docs/`` names it, only prose recording that it went — a word-boundary
grep finds eight mentions, all of them in docstrings, in **three files**: five here, two in
``tests/base/test_clim.py`` and one in ``tests/web/test_web_stack_clim.py`` (the underscored
``AnimationMixin._stack_clim`` is a different, live method, and is not among them). A shim would keep a
second spelling of the domain rule alive for nobody. Its work is
:func:`stack_scale`, and
the **return type is not the same**: a :class:`~digitalearth.base.spec.scale.Scale` rather than a pair, so
what used to read ``lo, hi = stack_clim(...)`` reads ``scale.vmin``/``scale.vmax``. :func:`frozen_scale` is
the same step taken from an already-measured pair, and :func:`measure_clim` still answers in
``(min, max)``/``None`` as it always did. Measured:

| call | answer |
|---|---|
| `stack_scale([np.array([2.0, 9.0])])` | `Scale(vmin=2.0, vmax=9.0, …)` |
| `stack_scale([])` | `Scale(vmin=0.0, vmax=1.0, …)` — the old fallback, now `Scale`'s |
| `frozen_scale(None)` | `Scale(vmin=0.0, vmax=1.0, …)` |
| `frozen_scale((5.0, 5.0))` | `Scale(vmin=5.0, vmax=6.0, …)` — the widening `stack_clim` never did |

Before this, each tier had its own copy and the three had drifted on the two things that decide the answer:

| Tier | Cap | Sampling |
|---|---|---|
| static | 24 | evenly-spaced stride |
| web | 50 | `datasets[:50]` — a head slice |
| interactive | uncapped | every member |

The sampling difference is the one that shows. A head slice measures only the *beginning* of a series, so a
60-member stack whose peak sits near the end — a flood crest, a fire scar — came back with a ``vmax`` that
clipped the peak to solid top-of-ramp on the web tier while the interactive tier rendered it correctly.

The rule here spreads its sample across the whole series **including the final frame**. A plain stride does
not: starting at 0 and stepping, it runs out before the end (200 frames capped at 24 stopped at index 198),
which is precisely the wrong end to drop for time-series data, where a rising series holds its maximum in the
last frames.

See Also:
    digitalearth.base.arrays.finite: drops the non-finite values this reduction must ignore.
    digitalearth.base.spec.scale.Scale: owns what a pair of limits means as a colour domain.
"""

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from digitalearth.base.arrays import finite
from digitalearth.base.spec import Scale

__all__ = [
    "DEFAULT_CLIM_SCAN_CAP",
    "frozen_scale",
    "measure_clim",
    "sample_evenly",
    "stack_scale",
]

#: How many stack frames are read to derive a shared colour range, unless a caller overrides it. Reading every
#: frame of a long series costs one warp each and buys very little: the range of ~24 frames spread across the
#: stack is the range of the stack, to the precision a colour ramp can show. Every tier reads this one number,
#: so the same collection is read at the same members whichever backend draws it. Note that this is a claim
#: about *which frames*, not about the resulting numbers: each tier warps to its own display CRS before
#: measuring, and resampling moves the extremes, so two tiers can still report slightly different ranges.
DEFAULT_CLIM_SCAN_CAP: int = 24


def sample_evenly(
    items: Sequence[Any], cap: Optional[int] = DEFAULT_CLIM_SCAN_CAP
) -> List[Any]:
    """Return at most ``cap`` items, spread evenly across the whole of ``items``.

    Sampling happens *before* the frames are read, which is the point: the cap is there to bound how many
    warps the scan pays for, and that only works if the frames it skips are never opened.

    Args:
        items: The stack members, in series order.
        cap: Greatest number of items to return; ``None`` returns every item.

    Raises:
        TypeError: if ``cap`` is a bool — ``True`` would otherwise read as ``cap=1``.
        ValueError: if ``cap`` is zero or negative. ``None`` is how "no cap" is spelled.

    Returns:
        A list of at most ``cap`` items spread across the series, **always including both the first and the
        last** (for ``cap >= 2``; ``cap=1`` returns the last item alone, since a series' extreme is far more
        often at its end than its start). Both ends matter: the first because a series often starts at its
        baseline, the last because a rising one ends at its maximum.

    Examples:
        - A short stack is returned whole:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> sample_evenly([0, 1, 2], cap=24)
            [0, 1, 2]

            ```
        - A long stack is thinned to the cap, and the sample still reaches the final frame:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> picked = sample_evenly(list(range(60)), cap=24)
            >>> len(picked), picked[0], picked[-1]
            (24, 0, 59)

            ```
        - ``cap=None`` scans everything:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> len(sample_evenly(list(range(60)), cap=None))
            60

            ```
        - One frame to spend goes on the last, not the first:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> sample_evenly(list(range(60)), cap=1)
            [59]

            ```
        - A zero or negative cap is refused rather than read as "no cap":
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> sample_evenly([0, 1, 2], cap=0)
            Traceback (most recent call last):
                ...
            ValueError: cap must be a positive frame count or None for no cap; got 0

            ```
        - A bool is refused too, since ``True`` would otherwise measure a single frame:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> sample_evenly([0, 1, 2], cap=True)
            Traceback (most recent call last):
                ...
            TypeError: cap must be a frame count or None, not a bool; got True

            ```
    """
    seq = list(items)
    if isinstance(cap, bool):
        # `True` slips past every numeric guard and reads as `cap=1`, silently measuring one frame.
        raise TypeError(f"cap must be a frame count or None, not a bool; got {cap!r}")
    if cap is not None and cap <= 0:
        # `None` already says 'no cap'. Reading 0 as unbounded is the opposite of its natural meaning,
        # and silently over-scanning a 5000-frame cube is not a kindness.
        raise ValueError(
            f"cap must be a positive frame count or None for no cap; got {cap!r}"
        )
    if not seq or cap is None or len(seq) <= cap:
        return seq
    if cap == 1:
        # One frame to spend: the last. A series' extreme is far more often at its end than its start.
        return [seq[-1]]
    # Spread `cap` positions across [0, len-1] inclusive rather than striding from 0 and stopping wherever
    # the stride runs out. A plain stride drops the tail -- 200 frames at cap 24 stopped at index 198, and
    # 100 stopped at 95 -- which is the wrong end to lose for time-series data, where a rising series puts
    # its maximum in the final frames. `len(seq) > cap >= 2` here, so the step exceeds 1 and the rounded
    # positions stay distinct: exactly `cap` frames come back.
    step = (len(seq) - 1) / (cap - 1)
    return [seq[round(i * step)] for i in range(cap)]


def measure_clim(arrays: Iterable[Any]) -> Optional[Tuple[float, float]]:
    """Return the ``(min, max)`` across ``arrays``, ignoring nodata and non-finite values.

    Args:
        arrays: The per-frame value arrays. A masked array is filled with ``NaN`` before it is measured, so a
            caller handing over a masked array directly gets the same answer as one handing over the
            already-NaN-filled array a pyramids extractor returns — the three tiers used to disagree here.

    Returns:
        The ``(min, max)`` across every array that held at least one finite value, or ``None`` when none did.
        ``None`` rather than a placeholder because a caller that unions several of these — the static tier
        sweeping one dataset under several projections — must be able to contribute *nothing* for a view that
        shows no data. A ``(0, 1)`` placeholder there would floor the union at zero.

    Examples:
        - Two frames reduce to one range:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.clim import measure_clim
            >>> measure_clim([np.array([1.0, 5.0]), np.array([-2.0, 3.0])])
            (-2.0, 5.0)

            ```
        - Nodata and non-finite values are ignored:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.clim import measure_clim
            >>> measure_clim([np.array([np.nan, 4.0, np.inf])])
            (4.0, 4.0)

            ```
        - Nothing finite anywhere answers ``None``:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.clim import measure_clim
            >>> measure_clim([np.array([np.nan])]) is None
            True

            ```
    """
    lows: List[float] = []
    highs: List[float] = []
    for arr in arrays:
        values = arr
        if np.ma.isMaskedArray(values):
            # `finite` goes through np.asarray, which drops a mask and would let the fill value (-9999)
            # through as a real number. Filling first is what makes a masked input agree with a NaN-filled
            # one -- and the cast has to come first, because a nodata sentinel is usually an *integer* one
            # and `filled(nan)` on an int array raises rather than widening it.
            values = np.ma.filled(values.astype("float64"), np.nan)
        values = finite(values)
        if values.size:
            lows.append(float(values.min()))
            highs.append(float(values.max()))
    return (min(lows), max(highs)) if lows else None


def frozen_scale(measured: Optional[Tuple[float, float]]) -> Scale:
    """Turn a stack's measured range — or nothing measurable — into the domain every frame shares.

    The seam between this module and :class:`~digitalearth.base.spec.scale.Scale`: the measurement is this
    module's, and everything that happens to it afterwards is `Scale`'s. Take this form when the frames have
    already been reduced by something other than :func:`measure_clim` — the static tier unions one
    measurement per swept projection — and :func:`stack_scale` when they have not.

    Args:
        measured: The ``(min, max)`` the frames yielded, or ``None`` when none of them held a finite value.

    Returns:
        A frozen scale. Its domain is the measurement, widened if it has no width; ``None`` becomes the unit
        range, because a builder still needs limits to hand a colormap and a stack whose every frame lies
        off the view has no range to report.

    Examples:
        - A measured range becomes that domain:
            ```python
            >>> from digitalearth.base.clim import frozen_scale
            >>> frozen_scale((2.0, 9.0)).as_limits()
            (2.0, 9.0)

            ```
        - Nothing measurable falls back rather than raising:
            ```python
            >>> from digitalearth.base.clim import frozen_scale
            >>> frozen_scale(None).as_limits()
            (0.0, 1.0)

            ```
        - A stack of identical frames is widened, so the domain has a width a ramp can show:
            ```python
            >>> from digitalearth.base.clim import frozen_scale
            >>> frozen_scale((7.0, 7.0)).as_limits()
            (7.0, 8.0)

            ```
    """
    lo, hi = (0.0, 1.0) if measured is None else measured
    # `freeze()` is what this call site is *for*: the point of a stack range is that it is derived once and
    # every frame is drawn against that one value, so nothing later re-derives it per frame. `Scale` is an
    # immutable dataclass, so the guarantee is already structural -- `freeze()` returns self, and a frame
    # that tried to adjust the range would raise rather than quietly put the later frames on another scale.
    return Scale.from_limits(lo, hi).freeze()


def stack_scale(arrays: Iterable[Any]) -> Scale:
    """Measure ``arrays`` and return the frozen domain every frame of the stack is drawn against.

    The form every tier's public path wants: a domain it can hand to a colormap without a ``None`` check.
    Use :func:`measure_clim` instead when "nothing was measurable" has to stay distinguishable from "the data
    happens to span 0 to 1" — a caller unioning several measurements needs that difference.

    Args:
        arrays: The per-frame value arrays, as for :func:`measure_clim`.

    Returns:
        The frozen scale over the measured range, by :func:`frozen_scale`'s rules.

    Examples:
        - A measurable stack reduces to its range:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.clim import stack_scale
            >>> stack_scale([np.array([2.0, 9.0])]).as_limits()
            (2.0, 9.0)

            ```
        - An empty stack falls back rather than raising:
            ```python
            >>> from digitalearth.base.clim import stack_scale
            >>> stack_scale([]).as_limits()
            (0.0, 1.0)

            ```
    """
    return frozen_scale(measure_clim(arrays))
