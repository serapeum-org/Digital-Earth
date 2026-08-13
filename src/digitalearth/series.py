"""series — ensemble / statistical series plots wired onto cleopatra LineGlyph and HistogramGlyph.

These are the non-map specialty plots from earthkit-plots (T7.3): envelope/quantile bands, box and
multi-box plots, and warming stripes. The rendering primitives live in cleopatra; this module only turns
numpy ensembles/series into the arrays those glyphs expect and draws them onto a (optionally shared) axes.
"""
from typing import Any, Optional, Sequence

import numpy as np
from cleopatra.glyphs.primitives.line_glyph import LineGlyph
from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph
from matplotlib.axes import Axes

from digitalearth._arrays import fig_of as _fig_of


def envelope(
    x: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    *,
    ax: Optional[Axes] = None,
    color: Any = None,
    alpha: float = 0.3,
) -> Any:
    """Shade the band between ``low`` and ``high`` over ``x`` (``LineGlyph.fill_between``).

    Args:
        x: 1-D x coordinates.
        low: Lower bound per x.
        high: Upper bound per x.
        ax: Axes to draw on (created when ``None``).
        color: Fill colour.
        alpha: Fill opacity.

    Returns:
        ``(fig, ax, PolyCollection)`` as produced by ``LineGlyph.fill_between``.
    """
    glyph = LineGlyph(np.asarray(x), np.asarray(high), ax=ax, fig=_fig_of(ax))
    return glyph.fill_between(y2=np.asarray(low), ax=ax, color=color, alpha=alpha)


def quantile_band(
    ensemble: np.ndarray,
    x: Optional[Sequence[float]] = None,
    lower: float = 0.1,
    upper: float = 0.9,
    *,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Any:
    """Shade the inter-quantile band of an ensemble across members.

    Args:
        ensemble: 2-D array ``(n_members, n_points)``.
        x: Optional x coordinates (defaults to ``range(n_points)``).
        lower: Lower quantile in ``[0, 1]``.
        upper: Upper quantile in ``[0, 1]``.
        ax: Axes to draw on (created when ``None``).
        **kwargs: Forwarded to :func:`envelope` (e.g. ``color``, ``alpha``).

    Returns:
        ``(fig, ax, PolyCollection)`` of the shaded band.
    """
    ens = np.asarray(ensemble)
    lo = np.quantile(ens, lower, axis=0)
    hi = np.quantile(ens, upper, axis=0)
    if x is None:
        x = np.arange(ens.shape[1])
    return envelope(x, lo, hi, ax=ax, **kwargs)


def boxplot(values: Any, *, ax: Optional[Axes] = None, **kwargs) -> Any:
    """Draw a box-and-whisker plot of ``values`` (``HistogramGlyph.boxplot``)."""
    return HistogramGlyph(values, ax=ax, fig=_fig_of(ax)).boxplot(ax=ax, **kwargs)


def multiboxplot(groups: Sequence[Sequence[float]], *, ax: Optional[Axes] = None, **kwargs) -> Any:
    """Draw grouped box plots, one box per group (``HistogramGlyph.multiboxplot``)."""
    return HistogramGlyph(groups, ax=ax, fig=_fig_of(ax)).multiboxplot(ax=ax, **kwargs)


def stripes(values: Sequence[float], *, ax: Optional[Axes] = None, **kwargs) -> Any:
    """Draw a warming-stripes bar strip of a 1-D series (``HistogramGlyph.stripes``)."""
    return HistogramGlyph(np.asarray(values), ax=ax, fig=_fig_of(ax)).stripes(ax=ax, **kwargs)
