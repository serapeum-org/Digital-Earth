"""Chart data preparation — turning a column name or array-like into plottable values.

Shared by every backend that draws charts. Lifted out of the static backend's ``charts`` module, which is
where these lived while :mod:`digitalearth.interactive.charts` imported them across the backend boundary —
the clearest sign they were shared logic in the wrong place. Pure numpy/pandas: no renderer here.
"""
from typing import Any, Optional, Sequence

import numpy as np

from digitalearth.base.arrays import finite, read_masked_band


def column_or_array(data: Any, value: Any) -> Optional[np.ndarray]:
    """Resolve ``value`` to an array — a column of ``data`` when it names one, else array-like as-is.

    The field-vs-field plumbing: with a (Geo)DataFrame ``data`` and a string ``value`` naming a column, the
    column is read as ``float64``; otherwise ``value`` is coerced with ``numpy.asarray`` (and ``None`` passes
    through, so optional ``color_by``/``size_by`` stay absent).

    Use this for *paired* inputs that must stay index-aligned and may legitimately contain non-finite values
    (scatter ``x``/``y``/``color_by``/``size_by``). For a *single* field reduced to its finite values (histogram
    / :func:`statistics`), use :func:`field_values` instead.

    Args:
        data: A (Geo)DataFrame whose columns ``value`` may name, or ``None``.
        value: A column name (resolved against ``data``) or an array-like, or ``None``.

    Returns:
        The resolved array, or ``None`` when ``value`` is ``None``.

    Raises:
        KeyError: if ``value`` names a column absent from ``data``.
    """
    if value is None:
        return None
    if data is not None and isinstance(value, str) and hasattr(data, "columns"):
        if value not in data.columns:
            raise KeyError(f"column {value!r} not found in the feature attributes")
        return np.asarray(data[value], dtype=float)
    return np.asarray(value)


def field_values(data: Any, column: Optional[str] = None) -> np.ndarray:
    """Return the finite, flattened 1-D values of a *field*.

    The single field-extraction recipe the column-aware chart helpers share: a GeoDataFrame/DataFrame column
    (when ``column`` is given), a pyramids ``Dataset`` band, or a plain array — always reduced to its finite
    (non-``NaN``/non-``inf``) values as ``float64``.

    Use this for a *single* field that is summarised or binned on its own (histogram, :func:`statistics`). For
    *paired* inputs that must keep their (possibly non-finite) entries index-aligned, use
    :func:`column_or_array` instead.

    Args:
        data: A GeoDataFrame/DataFrame (with ``column``), a pyramids ``Dataset``, or an array-like.
        column: Attribute/column name to read when ``data`` is a (Geo)DataFrame.

    Returns:
        numpy.ndarray: the finite values as a 1-D ``float64`` array.

    Raises:
        KeyError: if ``column`` is given but absent from ``data``.
    """
    if column is not None and hasattr(data, "columns"):
        if column not in data.columns:
            raise KeyError(f"column {column!r} not found in the feature attributes")
        return finite(np.asarray(data[column], dtype=float))
    return finite(as_finite_array(data))


def grouped_series(data: Any, by: str, column: Optional[str], agg: str):
    """Group ``data`` by ``by`` and aggregate ``column`` (or count rows) — the bar/line-by recipe.

    Args:
        data: A (Geo)DataFrame to group.
        by: Column to group on (a category or time field).
        column: Value column to aggregate; ``None`` counts rows per group.
        agg: Aggregation name applied to ``column`` (``"sum"``/``"mean"``/``"count"``/``"min"``/…).

    Returns:
        tuple: ``(keys, values)`` — the group keys (sorted ascending by pandas) and the aggregated values as
        a ``float64`` array.

    Raises:
        TypeError: if ``data`` has no ``groupby`` (not a DataFrame).
    """
    if not hasattr(data, "groupby"):
        raise TypeError("bar_by/line_by need a (Geo)DataFrame with a groupby method")
    grouped = data.groupby(by)
    try:
        # The agg itself can raise for a non-numeric column (mean/median/std reject strings), and a
        # numeric-looking agg that succeeds (sum/min/max concatenate/compare strings) fails the float cast —
        # wrap both so every non-numeric aggregation surfaces the same actionable message.
        series = grouped.size() if column is None else grouped[column].agg(agg)
        values = np.asarray(series.to_numpy(), dtype=float)
    except (TypeError, ValueError) as err:
        raise TypeError(
            f"cannot aggregate column {column!r} with agg={agg!r} into numbers — is the column numeric? "
            f"({err})"
        ) from err
    return list(series.index), values


def as_finite_array(values: Any) -> np.ndarray:
    """Coerce ``values`` to a numpy array for histogramming.

    A pyramids ``Dataset`` (duck-typed by ``read_array``/``no_data_value``) contributes its first band with
    the nodata fill and non-finite cells dropped — flattened to 1-D. Anything else is passed straight to
    ``numpy.asarray`` (so a raw 2-D array stays 2-D for overlaid histograms).
    """
    if hasattr(values, "read_array") and hasattr(values, "no_data_value"):
        return finite(read_masked_band(values, band=1))
    return np.asarray(values)
