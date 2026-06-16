"""Interactive (HoloViz / Bokeh) charts — the interactive-tier counterpart of :mod:`digitalearth.charts` (DC.6).

``digitalearth.charts`` renders static matplotlib charts via cleopatra; these return **HoloViews elements**
(``hv.Histogram`` / ``hv.Scatter`` / ``hv.Bars`` / ``hv.Curve``) for pan/zoom/hover charts in the interactive
tier — surfacing DC.1 (histogram), DC.3 (scatter) and DC.4 (aggregate bar/line) on the HoloViz stack. The
field-extraction and aggregation logic is shared with :mod:`digitalearth.charts` (one source of truth); only
the rendering engine differs.

holoviews is imported lazily (the optional ``interactive`` extra): importing this module needs no engine; only
calling a chart builder does, and a missing extra raises an actionable :class:`ImportError`.
"""

from typing import Any, Optional, Sequence

import numpy as np

from digitalearth.charts import _as_finite_array, _column_or_array, _field_values, _grouped_series

__all__ = ["histogram", "scatter", "bar", "line", "bar_by", "line_by"]

_INSTALL_HINT = (
    "the interactive charts need the HoloViz stack (holoviews). "
    "Install it with `pip install 'digitalearth[interactive]'` (or, in this repo, `pixi install -e interactive`)."
)


def _require_holoviews():
    """Import and return ``holoviews`` (Bokeh backend registered), raising an actionable error when absent."""
    try:
        import holoviews as hv
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err
    if "bokeh" not in hv.Store.renderers:
        hv.renderer("bokeh")
    return hv


def histogram(data: Any, *, column: Optional[str] = None, bins: int = 15, **opts: Any) -> Any:
    """Interactive histogram of a field (``hv.Histogram``) — the HoloViz DC.1 (DC.6).

    Args:
        data: A 1-D array, a pyramids ``Dataset`` (first band, nodata dropped), or a (Geo)DataFrame with
            ``column``.
        column: Attribute column to histogram when ``data`` is a (Geo)DataFrame.
        bins: Number of histogram bins.
        **opts: HoloViews options applied to the element (e.g. ``color``, ``width``, ``height``, ``alpha``).

    Returns:
        An ``hv.Histogram`` of the binned counts.

    Raises:
        ImportError: when the ``interactive`` extra is not installed.
    """
    hv = _require_holoviews()
    values = _field_values(data, column) if column is not None else _as_finite_array(data).ravel()
    element = hv.Histogram(np.histogram(values, bins=bins))
    return element.opts(**opts) if opts else element


def scatter(x: Any, y: Any, *, data: Any = None, **opts: Any) -> Any:
    """Interactive field-vs-field scatter (``hv.Scatter``) — the HoloViz DC.3 (DC.6).

    Args:
        x: x values, or a column name of ``data``.
        y: y values (same length as ``x``), or a column name of ``data``.
        data: Optional (Geo)DataFrame; when given, ``x``/``y`` may name its columns.
        **opts: HoloViews options applied to the element (e.g. ``size``, ``color``, ``tools``).

    Returns:
        An ``hv.Scatter`` of ``y`` against ``x``.

    Raises:
        ImportError: when the ``interactive`` extra is not installed.
    """
    hv = _require_holoviews()
    xs = _column_or_array(data, x)
    ys = _column_or_array(data, y)
    element = hv.Scatter((xs, ys))
    return element.opts(**opts) if opts else element


def bar(x: Any, heights: Any, **opts: Any) -> Any:
    """Interactive bar chart of a single series (``hv.Bars``).

    Args:
        x: 1-D bar positions / categories.
        heights: Bar heights, the same length as ``x``.
        **opts: HoloViews options applied to the element.

    Returns:
        An ``hv.Bars`` element.

    Raises:
        ImportError: when the ``interactive`` extra is not installed.
    """
    hv = _require_holoviews()
    element = hv.Bars((list(x), np.asarray(heights)))
    return element.opts(**opts) if opts else element


def line(x: Any, y: Any, **opts: Any) -> Any:
    """Interactive x–y line series (``hv.Curve``).

    Args:
        x: 1-D x coordinates.
        y: 1-D y values, the same length as ``x``.
        **opts: HoloViews options applied to the element.

    Returns:
        An ``hv.Curve`` element.

    Raises:
        ImportError: when the ``interactive`` extra is not installed.
    """
    hv = _require_holoviews()
    element = hv.Curve((np.asarray(x), np.asarray(y)))
    return element.opts(**opts) if opts else element


def bar_by(data: Any, by: str, column: Optional[str] = None, *, agg: str = "sum", **opts: Any) -> Any:
    """Interactive bar chart of an aggregate per category (``hv.Bars``) — the HoloViz DC.4 (DC.6).

    Args:
        data: A (Geo)DataFrame.
        by: Category column to group on.
        column: Value column to aggregate; ``None`` counts rows per category.
        agg: Aggregation name (``"sum"``/``"mean"``/``"count"``/…).
        **opts: HoloViews options applied to the element.

    Returns:
        An ``hv.Bars`` of the aggregate per category.

    Raises:
        ImportError: when the ``interactive`` extra is not installed.
    """
    hv = _require_holoviews()
    keys, values = _grouped_series(data, by, column, agg)
    element = hv.Bars(([str(k) for k in keys], values))
    return element.opts(**opts) if opts else element


def line_by(data: Any, by: str, column: Optional[str] = None, *, agg: str = "sum", **opts: Any) -> Any:
    """Interactive line of an aggregate per ordered/time key (``hv.Curve``) — the HoloViz DC.4 (DC.6).

    Args:
        data: A (Geo)DataFrame.
        by: Ordered/time column to group on (kept in ascending key order).
        column: Value column to aggregate; ``None`` counts rows per key.
        agg: Aggregation name (``"sum"``/``"mean"``/``"count"``/…).
        **opts: HoloViews options applied to the element.

    Returns:
        An ``hv.Curve`` of the aggregate over the sorted keys.

    Raises:
        ImportError: when the ``interactive`` extra is not installed.
    """
    hv = _require_holoviews()
    keys, values = _grouped_series(data, by, column, agg)
    element = hv.Curve((keys, values))
    return element.opts(**opts) if opts else element
