"""Chart data preparation — turning a column name or array-like into plottable values.

Shared by every backend that draws charts. Lifted out of the static backend's ``charts`` module, which is
where these lived while :mod:`digitalearth.interactive.charts` imported them across the backend boundary —
the clearest sign they were shared logic in the wrong place. Pure numpy/pandas: no renderer here.

The four helpers split by what the chart needs from the data:

* :func:`column_or_array` — *paired*, index-aligned inputs that may keep non-finite entries (scatter
  ``x``/``y``/``color_by``/``size_by``).
* :func:`field_values` — a *single* field reduced to its finite values (histogram, summary statistics).
* :func:`grouped_series` — group-and-aggregate for the ``bar_by``/``line_by`` charts.
* :func:`as_finite_array` — the raw coercion :func:`field_values` builds on, kept separate because
  histograms need a 2-D array to stay 2-D.

They lost their leading underscore when they moved here: as module-private helpers of ``charts`` they were
already being imported by a second backend, so the shared surface is now spelled as public.
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
    / :func:`digitalearth.static.charts.statistics`), use :func:`field_values` instead.

    Args:
        data: A (Geo)DataFrame whose columns ``value`` may name, or ``None``.
        value: A column name (resolved against ``data``) or an array-like, or ``None``.

    Returns:
        The resolved array, or ``None`` when ``value`` is ``None``.

    Raises:
        KeyError: if ``value`` names a column absent from ``data``.

    Examples:
        - A string names a column of the frame, read as ``float64``:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.base.chartdata import column_or_array
            >>> gdf = gpd.GeoDataFrame(
            ...     {"pop": [10, 20, 30]},
            ...     geometry=[Point(i, i) for i in range(3)],
            ...     crs=4326,
            ... )
            >>> column_or_array(gdf, "pop").tolist()
            [10.0, 20.0, 30.0]

            ```
        - An array-like is coerced as-is, and non-finite entries are **kept** so paired inputs stay aligned:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.chartdata import column_or_array
            >>> column_or_array(None, [1.0, np.nan, 3.0]).tolist()
            [1.0, nan, 3.0]

            ```
        - ``None`` passes through, so an optional ``color_by``/``size_by`` simply stays absent:
            ```python
            >>> from digitalearth.base.chartdata import column_or_array
            >>> column_or_array(None, None) is None
            True

            ```
        - A name the frame does not carry raises ``KeyError``:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.base.chartdata import column_or_array
            >>> gdf = gpd.GeoDataFrame({"pop": [1.0]}, geometry=[Point(0, 0)], crs=4326)
            >>> column_or_array(gdf, "population")
            Traceback (most recent call last):
                ...
            KeyError: "column 'population' not found in the feature attributes"

            ```

    See Also:
        field_values: The single-field counterpart, which drops non-finite values.
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

    Use this for a *single* field that is summarised or binned on its own (histogram,
    :func:`digitalearth.static.charts.statistics`). For *paired* inputs that must keep their (possibly
    non-finite) entries index-aligned, use :func:`column_or_array` instead.

    Args:
        data: A GeoDataFrame/DataFrame (with ``column``), a pyramids ``Dataset``, or an array-like.
        column: Attribute/column name to read when ``data`` is a (Geo)DataFrame.

    Returns:
        numpy.ndarray: the finite values as a 1-D ``float64`` array.

    Raises:
        KeyError: if ``column`` is given but absent from ``data``.

    Examples:
        - An array is flattened and its ``NaN``/``inf`` cells dropped:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.chartdata import field_values
            >>> field_values(np.array([[1.0, np.nan], [3.0, np.inf]])).tolist()
            [1.0, 3.0]

            ```
        - A named column of a (Geo)DataFrame comes back ready to summarise:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.base.chartdata import field_values
            >>> gdf = gpd.GeoDataFrame(
            ...     {"pop": [10.0, 20.0, 30.0]},
            ...     geometry=[Point(i, i) for i in range(3)],
            ...     crs=4326,
            ... )
            >>> float(field_values(gdf, column="pop").mean())
            20.0

            ```
        - A raster band contributes its data cells, with the nodata fill dropped:
            ```python
            >>> import numpy as np
            >>> from types import SimpleNamespace
            >>> from digitalearth.base.chartdata import field_values
            >>> ds = SimpleNamespace(
            ...     no_data_value=(-9999.0,),
            ...     read_array=lambda band=0: np.array([[1.0, -9999.0], [3.0, 4.0]]),
            ... )
            >>> field_values(ds).tolist()
            [1.0, 3.0, 4.0]

            ```

    See Also:
        as_finite_array: The coercion underneath, which leaves a plain array's shape alone.
        column_or_array: The paired-input counterpart, which keeps non-finite entries.
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
        TypeError: if ``data`` has no ``groupby`` (not a DataFrame), or if ``column`` cannot be aggregated
            into numbers with ``agg`` (a non-numeric column).

    Examples:
        - Sum a value column per group; the keys come back sorted:
            ```python
            >>> import pandas as pd
            >>> from digitalearth.base.chartdata import grouped_series
            >>> df = pd.DataFrame({"region": ["a", "b", "a"], "pop": [1.0, 2.0, 3.0]})
            >>> keys, values = grouped_series(df, "region", "pop", "sum")
            >>> keys
            ['a', 'b']
            >>> values.tolist()
            [4.0, 2.0]

            ```
        - Leave ``column`` as ``None`` to count the rows in each group instead:
            ```python
            >>> import pandas as pd
            >>> from digitalearth.base.chartdata import grouped_series
            >>> df = pd.DataFrame({"region": ["a", "b", "a"], "pop": [1.0, 2.0, 3.0]})
            >>> keys, values = grouped_series(df, "region", None, "count")
            >>> dict(zip(keys, values.tolist()))
            {'a': 2.0, 'b': 1.0}

            ```
        - Anything without a ``groupby`` is rejected up front:
            ```python
            >>> from digitalearth.base.chartdata import grouped_series
            >>> grouped_series([1, 2, 3], "region", "pop", "sum")
            Traceback (most recent call last):
                ...
            TypeError: bar_by/line_by need a (Geo)DataFrame with a groupby method

            ```
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

    That asymmetry is the whole point of keeping this separate from :func:`field_values`: a histogram wants
    one series per column of a 2-D array, so the shape must survive, whereas a summary reduces to 1-D.

    Args:
        values: A pyramids ``Dataset`` (first band is read), or any array-like.

    Returns:
        numpy.ndarray: the dataset's finite band values as a 1-D ``float64`` array, or ``values`` coerced by
        ``numpy.asarray`` with its shape and dtype untouched.

    Examples:
        - A plain 2-D array keeps its shape (one histogram series per column):
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.chartdata import as_finite_array
            >>> as_finite_array(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist()
            [[1.0, 2.0], [3.0, 4.0]]

            ```
        - A raster is flattened to its finite band values, nodata dropped:
            ```python
            >>> import numpy as np
            >>> from types import SimpleNamespace
            >>> from digitalearth.base.chartdata import as_finite_array
            >>> ds = SimpleNamespace(
            ...     no_data_value=(-1.0,),
            ...     read_array=lambda band=0: np.array([[5.0, -1.0], [7.0, 9.0]]),
            ... )
            >>> as_finite_array(ds).tolist()
            [5.0, 7.0, 9.0]

            ```

    See Also:
        field_values: Reduces the same inputs to 1-D finite values, for a single summarised field.
    """
    if hasattr(values, "read_array") and hasattr(values, "no_data_value"):
        return finite(read_masked_band(values, band=1))
    return np.asarray(values)
