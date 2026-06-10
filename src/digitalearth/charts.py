"""charts — non-map x–y charts (line, bar, histogram) wired onto cleopatra glyphs.

These are the earthkit-plots chart primitives that are **not** maps: a line/marker series, a bar chart, and a
histogram. The rendering lives in cleopatra (``LineGlyph``, ``StatisticalGlyph``); this module only adapts
numpy arrays (or a pyramids ``Dataset`` band, with nodata dropped) into the inputs those glyphs expect.
Parallel to :mod:`digitalearth.series` (the ensemble/statistical series plots).
"""
from typing import Any, Optional

import numpy as np
from cleopatra.line_glyph import LineGlyph
from cleopatra.statistical_glyph import StatisticalGlyph
from matplotlib.axes import Axes

from digitalearth._arrays import fig_of as _fig_of
from digitalearth._arrays import finite, read_masked_band

__all__ = ["line", "bar", "histogram"]


def _as_finite_array(values: Any) -> np.ndarray:
    """Coerce ``values`` to a numpy array for histogramming.

    A pyramids ``Dataset`` (duck-typed by ``read_array``/``no_data_value``) contributes its first band with
    the nodata fill and non-finite cells dropped — flattened to 1-D. Anything else is passed straight to
    ``numpy.asarray`` (so a raw 2-D array stays 2-D for overlaid histograms).
    """
    if hasattr(values, "read_array") and hasattr(values, "no_data_value"):
        return finite(read_masked_band(values, band=1))
    return np.asarray(values)


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


def histogram(values: Any, *, bins: int = 15, ax: Optional[Axes] = None, **kwargs):
    """Draw a histogram of array or raster values (cleopatra ``StatisticalGlyph.histogram``).

    Args:
        values: A 1-D/2-D sequence of numbers, or a pyramids ``Dataset`` (its first band is used, with nodata
            and non-finite cells dropped). A 2-D array draws one overlaid histogram per column.
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
    """
    arr = _as_finite_array(values)
    glyph = StatisticalGlyph(arr, ax=ax, fig=_fig_of(ax))
    return glyph.histogram(bins=bins, **kwargs)
