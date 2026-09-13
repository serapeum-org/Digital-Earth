"""The big-data cutoff shared by every rendering backend.

Above some feature count a vector builder stops emitting one glyph per row and routes the layer to a
density/GPU renderer instead — Datashader on the interactive tier, a deck.gl layer on the web tier. *Where*
that happens is one agreed number and one agreed rule, so both live here: the cutoff is pure data (an ``int``,
no renderer needed to produce it), and the rule that rejects a nonsensical one is a plain guard over that
``int``.

Before this, ``50_000`` was written out in :mod:`digitalearth.interactive.base` and :mod:`digitalearth.web.base`
— two literals free to drift — and only the web tier refused a negative override, so the same bad call was an
error on one backend and silently routed everything on the other.
"""

from typing import Any

__all__ = ["DEFAULT_BIG_DATA_THRESHOLD", "validate_big_data_threshold"]

#: Row/face/feature count above which a vector builder auto-routes to its tier's big-data renderer, unless the
#: map or the call overrides it. Every tier reads this one number, so a layer of a given size takes the same
#: route whichever backend draws it; each tier exposes it as a ``big_data_threshold`` attribute plus a per-call
#: override of the same name.
DEFAULT_BIG_DATA_THRESHOLD: int = 50_000


def validate_big_data_threshold(threshold: Any, *, caller: str) -> int:
    """Return ``threshold`` as an ``int``, refusing a negative cutoff.

    The one rule every tier applies to a per-call ``big_data_threshold=``. A cutoff below zero routes *every*
    layer — an empty one included — to the big-data renderer, which is never what the caller meant; it is far
    more likely a sentinel (``-1``) borrowed from an API where negative means "unlimited". Answering that the
    same way on every backend is the point of sharing the check.

    Args:
        threshold: The cutoff as the caller wrote it, already known not to be ``None``.
        caller: The builder or map method the keyword was written on, named in the error so the message
            points at the call rather than at this helper.

    Returns:
        The validated cutoff.

    Raises:
        ValueError: when ``threshold`` is negative.

    Examples:
        - A legal cutoff comes back as an ``int``, zero included (it routes everything non-empty):
            ```python
            >>> from digitalearth.base.bigdata import validate_big_data_threshold
            >>> validate_big_data_threshold(1000, caller="WebMap.points()")
            1000
            >>> validate_big_data_threshold(0, caller="WebMap.points()")
            0

            ```
        - A negative one is refused, and the message names the call:
            ```python
            >>> from digitalearth.base.bigdata import validate_big_data_threshold
            >>> validate_big_data_threshold(-1, caller="InteractiveMap.points()")
            Traceback (most recent call last):
                ...
            ValueError: InteractiveMap.points(): big_data_threshold must not be negative; got -1

            ```
    """
    value = int(threshold)
    if value < 0:
        raise ValueError(
            f"{caller}: big_data_threshold must not be negative; got {threshold!r}"
        )
    return value
