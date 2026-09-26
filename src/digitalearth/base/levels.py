"""levels — the arithmetic behind ``contours(interval=)``, in the one place every tier can reach it.

``interval=`` is one of the three keywords the Tier-2 contract declares on ``contours``, and it means one
thing wherever it appears: one level every N, in the band's own units. Where those levels are *computed*
differs per tier, which is what makes the arithmetic worth sharing rather than repeating:

* the **web** tier hands the spacing to pyramids, which walks the band itself;
* the **static** and **interactive** tiers have to hand their engine a finished level list, so each has to do
  the walk — and two hand-written walks is two places for "which multiples are inside" to drift.

So the walk lives here, where a tier reaches it without reaching a sibling backend. The module is numpy and
:mod:`math` only, which is what lets it sit in ``base/`` at all
(``tests/test_base_is_engine_neutral.py``).

**Strictly inside, on purpose.** A level sitting exactly on the band's minimum or maximum traces the frame's
edge or nothing at all, so it is a contour nobody can read. Only the multiples *between* the two extremes are
returned, which is also what the web tier's pyramids-side walk produces for the same call.
"""

from math import isfinite
from typing import Any, List

import numpy as np

__all__ = ["levels_every"]


def levels_every(values: Any, interval: float) -> List[float]:
    """Return the multiples of ``interval`` that fall strictly inside a band's finite range.

    Args:
        values: The band's values, as the drawer read them — any shape, and ``NaN`` is a hole rather than a
            value the range has to reach.
        interval: The spacing between successive levels, in the band's own units.

    Returns:
        The levels, ascending, as plain floats so a figure can be written down with them.

    Raises:
        ValueError: when ``interval`` is not a positive, finite number; when the band holds no finite value;
            or when no multiple of ``interval`` lies inside the range. Each is refused here, with the
            argument named, because each otherwise surfaces from inside the rendering engine as a complaint
            about an empty level list — which points at the wrong thing entirely.

    Examples:
        - A band running 0-399 at an interval of 100 crosses three levels: 0 sits on the minimum and 400
          past the maximum, so neither is one.
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.levels import levels_every
            >>> levels_every(np.arange(400.0), 100.0)
            [100.0, 200.0, 300.0]

            ```
        - The levels are multiples of the spacing, not offsets from the band's own minimum:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.levels import levels_every
            >>> levels_every(np.linspace(250.0, 1250.0, 64), 500.0)
            [500.0, 1000.0]

            ```
        - A spacing too coarse to cross the band has no levels to give, and says so rather than handing the
          engine an empty list:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.levels import levels_every
            >>> levels_every(np.arange(400.0), 10000.0)   # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: contours(interval=10000.0) crosses no level inside the band's range (0.0 to 399.0)...

            ```
    """
    if not isfinite(interval) or interval <= 0:
        raise ValueError(
            f"contours() takes interval= as a positive spacing in the band's units; got {interval!r}"
        )
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(
            "contours(interval=) has no values to space levels through: the band is empty"
        )
    low, high = float(finite.min()), float(finite.max())
    first = np.floor(low / interval) + 1.0
    last = np.ceil(high / interval) - 1.0
    steps = np.arange(first, last + 1.0) * interval
    inside = [float(level) for level in steps if low < level < high]
    if not inside:
        raise ValueError(
            f"contours(interval={interval!r}) crosses no level inside the band's range "
            f"({low!r} to {high!r}); pass a smaller interval, or levels= instead"
        )
    return inside
