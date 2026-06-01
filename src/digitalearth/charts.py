"""charts — non-map x–y charts (line, bar, histogram) wired onto cleopatra glyphs.

These are the earthkit-plots chart primitives that are **not** maps: a line/marker series, a bar chart, and a
histogram. The rendering lives in cleopatra (``LineGlyph``, ``StatisticalGlyph``); this module only adapts
numpy arrays (or a pyramids ``Dataset`` band, with nodata dropped) into the inputs those glyphs expect.
Parallel to :mod:`digitalearth.series` (the ensemble/statistical series plots).
"""
from typing import Any, Optional

import numpy as np
from cleopatra.line_glyph import LineGlyph
from matplotlib.axes import Axes

__all__ = ["line", "bar"]


def _fig_of(ax: Optional[Axes]):
    """Return the figure owning ``ax`` (or ``None`` when ``ax`` is ``None``)."""
    return ax.get_figure() if ax is not None else None


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
