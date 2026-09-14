"""The one colour-range rule for a raster time stack, shared by every rendering backend.

A stack rendered frame by frame has to freeze one ``(vmin, vmax)`` across the whole series, or the scale jumps
between frames and the animation reads as data that is not there. Deriving that range is pure arithmetic over
arrays — no renderer needed — so it lives here rather than three times over.

Before this, each tier had its own copy and the three had drifted on the two things that decide the answer:

============  ===========  =========================================================
Tier          Cap          Sampling
============  ===========  =========================================================
static        24           evenly-spaced stride
web           50           ``datasets[:50]`` — a head slice
interactive   uncapped     every member
============  ===========  =========================================================

The sampling difference is the one that shows. A head slice measures only the *beginning* of a series, so a
60-member stack whose peak sits at index 58 — a flood crest, a fire scar — comes back with a ``vmax`` that
clips the peak to solid top-of-ramp on the web tier while the interactive tier renders it correctly. Striding
evenly spans the whole series at the same cost, which is why it is the rule kept here.

See Also:
    digitalearth.base.arrays.finite: drops the non-finite values this reduction must ignore.
"""

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from digitalearth.base.arrays import finite

__all__ = ["DEFAULT_CLIM_SCAN_CAP", "measure_clim", "sample_evenly", "stack_clim"]

#: How many stack frames are read to derive a shared colour range, unless a caller overrides it. Reading every
#: frame of a long series costs one warp each and buys very little: the range of ~24 frames spread across the
#: stack is the range of the stack, to the precision a colour ramp can show. Every tier reads this one number,
#: so the same collection gets the same scale whichever backend draws it.
DEFAULT_CLIM_SCAN_CAP: int = 24


def sample_evenly(
    items: Sequence[Any], cap: Optional[int] = DEFAULT_CLIM_SCAN_CAP
) -> List[Any]:
    """Return at most ``cap`` items, spread evenly across the whole of ``items``.

    Sampling happens *before* the frames are read, which is the point: the cap is there to bound how many
    warps the scan pays for, and that only works if the frames it skips are never opened.

    Args:
        items: The stack members, in series order.
        cap: Greatest number of items to return. ``None`` (or a non-positive cap) returns every item.

    Returns:
        A list of at most ``cap`` items whose first element is always ``items[0]``, taken at a fixed stride so
        the sample spans the series rather than its beginning.

    Examples:
        - A short stack is returned whole:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> sample_evenly([0, 1, 2], cap=24)
            [0, 1, 2]

            ```
        - A long stack is strided, and the sample reaches the end rather than stopping partway:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> picked = sample_evenly(list(range(60)), cap=24)
            >>> len(picked), picked[0], picked[-1]
            (20, 0, 57)

            ```
        - ``cap=None`` scans everything:
            ```python
            >>> from digitalearth.base.clim import sample_evenly
            >>> len(sample_evenly(list(range(60)), cap=None))
            60

            ```
    """
    seq = list(items)
    if not seq or cap is None or cap <= 0 or len(seq) <= cap:
        return seq
    # Round the stride UP: a floor divide returns 1 for anything under twice the cap, so a 47-frame stack
    # would scan all 47 while claiming a cap of 24.
    return seq[:: -(-len(seq) // cap)]


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
            # through as a real number. Filling first is what makes a masked input agree with a NaN-filled one.
            values = values.filled(np.nan)
        values = finite(values)
        if values.size:
            lows.append(float(values.min()))
            highs.append(float(values.max()))
    return (min(lows), max(highs)) if lows else None


def stack_clim(arrays: Iterable[Any]) -> Tuple[float, float]:
    """Return the ``(min, max)`` across ``arrays``, falling back to ``(0.0, 1.0)``.

    The form every tier's public path wants: a range it can hand to a colormap without a ``None`` check. Use
    :func:`measure_clim` instead when "nothing was measurable" has to stay distinguishable from "the data
    happens to span 0 to 1".

    Args:
        arrays: The per-frame value arrays, as for :func:`measure_clim`.

    Returns:
        The measured ``(min, max)``, or ``(0.0, 1.0)`` when no array held a finite value — which includes a
        stack whose every frame lies off the view, since a frame that cannot be warped draws nothing and so
        contributes no colour range.

    Examples:
        - A measurable stack reduces to its range:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.clim import stack_clim
            >>> stack_clim([np.array([2.0, 9.0])])
            (2.0, 9.0)

            ```
        - An empty stack falls back rather than raising:
            ```python
            >>> from digitalearth.base.clim import stack_clim
            >>> stack_clim([])
            (0.0, 1.0)

            ```
    """
    measured = measure_clim(arrays)
    return measured if measured is not None else (0.0, 1.0)
