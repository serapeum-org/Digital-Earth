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
    """Return ``threshold`` as an ``int``, refusing anything that is not a whole, non-negative row count.

    The one rule every tier applies to a per-call ``big_data_threshold=``. A cutoff below zero routes *every*
    layer — an empty one included — to the big-data renderer, which is never what the caller meant; it is far
    more likely a sentinel (``-1``) borrowed from an API where negative means "unlimited". Answering that the
    same way on every backend is the point of sharing the check.

    A cutoff is a count of rows, so a value that is not one is refused rather than coerced into one. ``int()``
    alone truncated: ``1.9`` became ``1`` and cut over one row *earlier* than the caller asked, silently
    (review L6). It also let every other bad value surface as a bare ``ValueError`` from ``int`` — ``"abc"``,
    ``nan``, ``inf``, ``None`` — with no mention of the parameter or the call. Both now report the same way,
    naming the keyword and the call. An integral ``float`` (``1000.0``) still passes: it *is* a whole count.
    ``True``/``False`` do not — a boolean is a flag, not a row count, and ``int(True)`` silently meant ``1``.

    Args:
        threshold: The cutoff as the caller wrote it, already known not to be ``None``.
        caller: The builder or map method the keyword was written on, named in the error so the message
            points at the call rather than at this helper.

    Returns:
        The validated cutoff.

    Raises:
        ValueError: when ``threshold`` is negative, or when it is not a whole number of rows — a fraction,
            a boolean, a non-finite float, or anything that is not a number at all.

    Examples:
        - A legal cutoff comes back as an ``int``, zero included (it routes everything non-empty):
            ```python
            >>> from digitalearth.base.bigdata import validate_big_data_threshold
            >>> validate_big_data_threshold(1000, caller="WebMap.points()")
            1000
            >>> validate_big_data_threshold(0, caller="WebMap.points()")
            0

            ```
        - A whole count written as a float is the same count, so it passes:
            ```python
            >>> from digitalearth.base.bigdata import validate_big_data_threshold
            >>> validate_big_data_threshold(1000.0, caller="WebMap.points()")
            1000

            ```
        - A negative one is refused, and the message names the call:
            ```python
            >>> from digitalearth.base.bigdata import validate_big_data_threshold
            >>> validate_big_data_threshold(-1, caller="InteractiveMap.points()")
            Traceback (most recent call last):
                ...
            ValueError: InteractiveMap.points(): big_data_threshold must not be negative; got -1

            ```
        - A fraction is refused instead of being truncated to the row before it:
            ```python
            >>> from digitalearth.base.bigdata import validate_big_data_threshold
            >>> validate_big_data_threshold(1.9, caller="InteractiveMap.points()")
            Traceback (most recent call last):
                ...
            ValueError: InteractiveMap.points(): big_data_threshold must be a whole number of rows; got 1.9

            ```
    """
    if isinstance(threshold, bool):
        raise ValueError(
            f"{caller}: big_data_threshold must be a whole number of rows; got {threshold!r}"
        )
    try:
        value = int(threshold)
        whole = value == threshold
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{caller}: big_data_threshold must be a whole number of rows; got {threshold!r}"
        ) from error
    if not whole:
        raise ValueError(
            f"{caller}: big_data_threshold must be a whole number of rows; got {threshold!r}"
        )
    if value < 0:
        raise ValueError(
            f"{caller}: big_data_threshold must not be negative; got {threshold!r}"
        )
    return value
