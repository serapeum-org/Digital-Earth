"""charts — non-map x–y charts (line, bar, histogram) wired onto cleopatra glyphs.

These are the earthkit-plots chart primitives that are **not** maps: a line/marker series, a bar chart, and a
histogram. The rendering lives in cleopatra (``LineGlyph``, ``StatisticalGlyph``); this module only adapts
numpy arrays (or a pyramids ``Dataset`` band, with nodata dropped) into the inputs those glyphs expect.
Parallel to :mod:`digitalearth.series` (the ensemble/statistical series plots).
"""
from typing import Any, Optional, Sequence

import numpy as np
from cleopatra.line_glyph import LineGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.statistical_glyph import StatisticalGlyph
from matplotlib.axes import Axes

from digitalearth._arrays import fig_of as _fig_of
from digitalearth._arrays import finite, read_masked_band

__all__ = ["line", "bar", "bar_by", "line_by", "histogram", "scatter", "statistics"]


def _column_or_array(data: Any, value: Any) -> Optional[np.ndarray]:
    """Resolve ``value`` to an array — a column of ``data`` when it names one, else array-like as-is.

    The field-vs-field plumbing: with a (Geo)DataFrame ``data`` and a string ``value`` naming a column, the
    column is read as ``float64``; otherwise ``value`` is coerced with ``numpy.asarray`` (and ``None`` passes
    through, so optional ``color_by``/``size_by`` stay absent).

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


def _as_finite_array(values: Any) -> np.ndarray:
    """Coerce ``values`` to a numpy array for histogramming.

    A pyramids ``Dataset`` (duck-typed by ``read_array``/``no_data_value``) contributes its first band with
    the nodata fill and non-finite cells dropped — flattened to 1-D. Anything else is passed straight to
    ``numpy.asarray`` (so a raw 2-D array stays 2-D for overlaid histograms).
    """
    if hasattr(values, "read_array") and hasattr(values, "no_data_value"):
        return finite(read_masked_band(values, band=1))
    return np.asarray(values)


def _field_values(data: Any, column: Optional[str] = None) -> np.ndarray:
    """Return the finite, flattened 1-D values of a *field*.

    The single field-extraction recipe the column-aware chart helpers share: a GeoDataFrame/DataFrame column
    (when ``column`` is given), a pyramids ``Dataset`` band, or a plain array — always reduced to its finite
    (non-``NaN``/non-``inf``) values as ``float64``.

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
    return finite(_as_finite_array(data))


def statistics(
    data: Any,
    *,
    column: Optional[str] = None,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict:
    """Summarise a field — count/min/max/mean/std plus the requested quantiles (DC.5).

    A thin numpy reduction (not a GIS op): the input is coerced to its finite values and summarised. Accepts a
    GeoDataFrame/DataFrame with ``column``, a pyramids ``Dataset`` (first band, nodata dropped), or a plain
    array. Quantiles are returned under ``q<pct>`` keys (e.g. ``q50`` for the median).

    Args:
        data: A (Geo)DataFrame (with ``column``), a pyramids ``Dataset``, or an array-like of numbers.
        column: Attribute/column name to summarise when ``data`` is a (Geo)DataFrame.
        quantiles: Quantiles in ``[0, 1]`` to include (default the quartiles ``0.25``/``0.5``/``0.75``).

    Returns:
        dict: ``count`` (int) plus ``min``/``max``/``mean``/``std`` and one ``q<pct>`` entry per quantile,
        all floats.

    Raises:
        ValueError: if there are no finite values to summarise.
        KeyError: if ``column`` is given but absent from ``data``.

    Examples:
        - Summarise a plain sequence (median is ``q50``):
            ```python
            >>> from digitalearth.charts import statistics
            >>> s = statistics([1, 2, 3, 4])
            >>> (s["count"], s["min"], s["max"], s["mean"], s["q50"])
            (4, 1.0, 4.0, 2.5, 2.5)

            ```
        - Summarise a GeoDataFrame column:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.charts import statistics
            >>> gdf = gpd.GeoDataFrame(
            ...     {"pop": [10.0, 20.0, 30.0]},
            ...     geometry=[Point(i, i) for i in range(3)],
            ...     crs=4326,
            ... )
            >>> statistics(gdf, column="pop")["mean"]
            20.0

            ```
        - Pick custom quantiles:
            ```python
            >>> from digitalearth.charts import statistics
            >>> sorted(statistics(range(101), quantiles=(0.1, 0.9)))
            ['count', 'max', 'mean', 'min', 'q10', 'q90', 'std']

            ```
    """
    arr = _field_values(data, column)
    if arr.size == 0:
        raise ValueError("no finite values to summarise")
    summary: dict = {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    for q in quantiles:
        summary[f"q{int(round(q * 100))}"] = float(np.quantile(arr, q))
    return summary


def line(x: Any, y: Any, *, ax: Optional[Axes] = None, label: Any = None, color: Any = None,
         **kwargs) -> Axes:
    """Draw an x–y line/marker series (cleopatra ``LineGlyph.line``); returns the Axes.

    Args:
        x: 1-D sequence of x-coordinates.
        y: y-values — 1-D for a single series, or 2-D ``(len(x), n_series)`` for several series sharing ``x``.
        ax: Existing axes to draw on; a new figure/axes is created when ``None``.
        label: Legend label(s) — a string for one series, or a list matching the columns of a 2-D ``y``.
        color: Line colour(s) passed through to the glyph.
        **kwargs: Forwarded to ``LineGlyph.line`` / ``Axes.plot`` (e.g. ``marker``, ``linestyle``, ``alpha``).

    Returns:
        The :class:`matplotlib.axes.Axes` the series was drawn on (one ``Line2D`` per series).

    Examples:
        - A single series adds one line:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.charts import line
            >>> ax = line([0, 1, 2, 3], [0, 1, 4, 9], label="y = x²")
            >>> len(ax.lines)
            1

            ```
        - A 2-D ``y`` draws one line per column (e.g. ensemble members):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from digitalearth.charts import line
            >>> y = np.column_stack([[0, 1, 2], [0, 2, 4]])
            >>> ax = line([0, 1, 2], y)
            >>> len(ax.lines)
            2

            ```
    """
    glyph = LineGlyph(np.asarray(x), np.asarray(y), ax=ax, fig=_fig_of(ax))
    _, ax, _ = glyph.line(ax=ax, label=label, color=color, **kwargs)
    return ax


def bar(x: Any, heights: Any, *, ax: Optional[Axes] = None, color: Any = None, **kwargs) -> Axes:
    """Draw a bar chart of a single series (cleopatra ``LineGlyph.bar``); returns the Axes.

    Args:
        x: 1-D sequence of bar positions / categories.
        heights: 1-D sequence of bar heights, the same length as ``x``.
        ax: Existing axes to draw on; a new figure/axes is created when ``None``.
        color: Bar colour(s) passed through to the glyph.
        **kwargs: Forwarded to ``LineGlyph.bar`` / ``Axes.bar`` (e.g. ``width``, ``alpha``).

    Returns:
        The :class:`matplotlib.axes.Axes` the bars were drawn on (one bar per element of ``x``).

    Examples:
        - A four-category bar chart adds four bars:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.charts import bar
            >>> ax = bar([0, 1, 2, 3], [3, 1, 4, 1])
            >>> len(ax.containers[0])
            4

            ```
        - The drawn heights match the input:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.charts import bar
            >>> ax = bar([0, 1, 2], [2.0, 5.0, 3.0])
            >>> [round(float(rect.get_height()), 1) for rect in ax.containers[0]]
            [2.0, 5.0, 3.0]

            ```
    """
    glyph = LineGlyph(np.asarray(x), np.asarray(heights), ax=ax, fig=_fig_of(ax))
    _, ax, _ = glyph.bar(ax=ax, color=color, **kwargs)
    return ax


def histogram(values: Any, *, column: Optional[str] = None, bins: int = 15,
              ax: Optional[Axes] = None, **kwargs):
    """Draw a histogram of array, raster, or field values (cleopatra ``StatisticalGlyph.histogram``).

    Args:
        values: A 1-D/2-D sequence of numbers, a pyramids ``Dataset`` (its first band is used, with nodata
            and non-finite cells dropped), or a GeoDataFrame/DataFrame when ``column`` is given (the one-call
            ``histogram(gdf, column="pop")`` field path, DC.1). A 2-D array draws one overlaid histogram per
            column.
        column: Attribute/column name to histogram when ``values`` is a (Geo)DataFrame; its finite values are
            used.
        bins: Number of histogram bins.
        ax: Existing axes to draw on; a new figure/axes is created when ``None``.
        **kwargs: Forwarded to ``StatisticalGlyph.histogram`` (e.g. ``color``, ``alpha``, ``rwidth``).

    Returns:
        ``(fig, ax, hist)`` — the figure, the axes, and the histogram info dict from cleopatra.

    Examples:
        - A histogram of 1-D values returns the figure/axes and counts that sum to the sample size:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.charts import histogram
            >>> fig, ax, hist = histogram([1, 1, 2, 3, 3, 3], bins=3)
            >>> len(ax.patches)
            3

            ```
        - A pyramids Dataset is histogrammed over its first band (nodata dropped):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.charts import histogram
            >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
            >>> ds = Dataset.create_from_array(arr=arr, geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
            >>> fig, ax, hist = histogram(ds, bins=4)
            >>> len(ax.patches)
            4

            ```
        - A GeoDataFrame column is histogrammed in one call (DC.1):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.charts import histogram
            >>> gdf = gpd.GeoDataFrame(
            ...     {"pop": [1.0, 1.0, 2.0, 3.0]},
            ...     geometry=[Point(i, i) for i in range(4)],
            ...     crs=4326,
            ... )
            >>> fig, ax, hist = histogram(gdf, column="pop", bins=3)
            >>> len(ax.patches)
            3

            ```
    """
    arr = _field_values(values, column) if column is not None else _as_finite_array(values)
    glyph = StatisticalGlyph(arr, ax=ax, fig=_fig_of(ax))
    return glyph.histogram(bins=bins, **kwargs)


def scatter(x: Any, y: Any, *, data: Any = None, color_by: Any = None, size_by: Any = None,
            ax: Optional[Axes] = None, **kwargs) -> Axes:
    """Draw a field-vs-field scatter (cleopatra ``ScatterGlyph``); returns the Axes (DC.3).

    Plots ``y`` against ``x`` as a point cloud, optionally colouring points by ``color_by`` and sizing them by
    ``size_by``. With a (Geo)DataFrame ``data``, ``x``/``y``/``color_by``/``size_by`` may be column names;
    otherwise they are array-likes.

    Args:
        x: x values, or a column name of ``data``.
        y: y values (same length as ``x``), or a column name of ``data``.
        data: Optional (Geo)DataFrame; when given, the other arguments may name its columns.
        color_by: Optional values (or column name) mapped to point colour (adds a colorbar).
        size_by: Optional magnitudes (or column name) mapped to point size.
        ax: Existing axes to draw on; a new figure/axes is created when ``None``.
        **kwargs: Forwarded to ``ScatterGlyph`` (e.g. ``cmap``, ``point_size``, ``vmin``, ``vmax``, ``alpha``).

    Returns:
        The :class:`matplotlib.axes.Axes` the scatter was drawn on (one ``PathCollection``).

    Raises:
        KeyError: if a column name is given but absent from ``data``.
        ValueError: if ``x`` and ``y`` differ in shape (from ``ScatterGlyph``).

    Examples:
        - Two arrays draw one scatter collection:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.charts import scatter
            >>> ax = scatter([1, 2, 3], [4, 5, 6])
            >>> len(ax.collections)
            1

            ```
        - Field vs field from a GeoDataFrame by column name:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.charts import scatter
            >>> gdf = gpd.GeoDataFrame(
            ...     {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]},
            ...     geometry=[Point(i, i) for i in range(3)],
            ...     crs=4326,
            ... )
            >>> ax = scatter("a", "b", data=gdf)
            >>> len(ax.collections)
            1

            ```
    """
    xs = _column_or_array(data, x)
    ys = _column_or_array(data, y)
    values = _column_or_array(data, color_by)
    sizes = _column_or_array(data, size_by)
    glyph = ScatterGlyph(xs, ys, values=values, sizes=sizes, ax=ax, fig=_fig_of(ax), **kwargs)
    _, ax, _ = glyph.plot(ax=ax)
    return ax


def _grouped_series(data: Any, by: str, column: Optional[str], agg: str):
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
    series = grouped.size() if column is None else grouped[column].agg(agg)
    return list(series.index), np.asarray(series.to_numpy(), dtype=float)


def bar_by(data: Any, by: str, column: Optional[str] = None, *, agg: str = "sum",
           ax: Optional[Axes] = None, **kwargs) -> Axes:
    """Bar chart of an aggregate per category (DC.4).

    Groups ``data`` by the ``by`` column and draws one bar per group of ``column`` aggregated with ``agg``
    (or the row count when ``column`` is ``None``). Bars sit at integer positions labelled with the category
    names, so non-numeric categories render correctly.

    Args:
        data: A (Geo)DataFrame.
        by: Category column to group on.
        column: Value column to aggregate; ``None`` counts rows per category.
        agg: Aggregation name (``"sum"``/``"mean"``/``"count"``/``"min"``/``"max"``/``"median"``).
        ax: Existing axes to draw on; a new figure/axes is created when ``None``.
        **kwargs: Forwarded to :func:`bar` (e.g. ``color``, ``width``).

    Returns:
        The :class:`matplotlib.axes.Axes` with one bar per category, x-tick-labelled by category.

    Examples:
        - Sum a value column per category:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.charts import bar_by
            >>> gdf = gpd.GeoDataFrame(
            ...     {"cat": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]},
            ...     geometry=[Point(i, i) for i in range(3)],
            ...     crs=4326,
            ... )
            >>> ax = bar_by(gdf, "cat", "v", agg="sum")
            >>> [round(rect.get_height(), 1) for rect in ax.containers[0]]
            [3.0, 3.0]

            ```
        - Count rows per category (no value column):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.charts import bar_by
            >>> gdf = gpd.GeoDataFrame(
            ...     {"cat": ["a", "a", "b"]},
            ...     geometry=[Point(i, i) for i in range(3)],
            ...     crs=4326,
            ... )
            >>> ax = bar_by(gdf, "cat")
            >>> [int(rect.get_height()) for rect in ax.containers[0]]
            [2, 1]

            ```
    """
    keys, values = _grouped_series(data, by, column, agg)
    ax = bar(range(len(keys)), values, ax=ax, **kwargs)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(key) for key in keys])
    return ax


def line_by(data: Any, by: str, column: Optional[str] = None, *, agg: str = "sum",
            ax: Optional[Axes] = None, **kwargs) -> Axes:
    """Line chart of an aggregate per ordered key — e.g. a value summed by year (DC.4).

    Groups ``data`` by the ``by`` column (typically a time/ordered field), aggregates ``column`` with ``agg``
    (or counts rows when ``column`` is ``None``), and draws a line over the sorted keys.

    Args:
        data: A (Geo)DataFrame.
        by: Ordered/time column to group on (kept in ascending key order).
        column: Value column to aggregate; ``None`` counts rows per key.
        agg: Aggregation name (``"sum"``/``"mean"``/``"count"``/…).
        ax: Existing axes to draw on; a new figure/axes is created when ``None``.
        **kwargs: Forwarded to :func:`line` (e.g. ``label``, ``color``, ``marker``).

    Returns:
        The :class:`matplotlib.axes.Axes` with the aggregated series as one line.

    Examples:
        - Mean of a value by year:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.charts import line_by
            >>> gdf = gpd.GeoDataFrame(
            ...     {"year": [2000, 2000, 2010], "v": [1.0, 3.0, 5.0]},
            ...     geometry=[Point(i, i) for i in range(3)],
            ...     crs=4326,
            ... )
            >>> ax = line_by(gdf, "year", "v", agg="mean")
            >>> len(ax.lines)
            1

            ```
    """
    keys, values = _grouped_series(data, by, column, agg)
    return line(keys, values, ax=ax, **kwargs)
