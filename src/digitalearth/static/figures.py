"""The figure that owns an axes — the one matplotlib chore the static backend shares.

Split out of the old flat ``digitalearth._arrays`` when that module was divided: everything else in it is plain
numpy and now lives in :mod:`digitalearth.base.arrays`, while this needs :mod:`matplotlib` and so belongs to
the static backend. Keeping it out of :mod:`digitalearth.base` is what lets that subpackage stay free of any
renderer import. Only :mod:`digitalearth.static.charts` and :mod:`digitalearth.static.series` use it.
"""

from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def fig_of(ax: Optional[Axes]) -> Optional[Figure]:
    """Return the figure that owns ``ax``, or ``None`` when ``ax`` is ``None``.

    Args:
        ax: A matplotlib axes, or ``None``.

    Returns:
        The owning :class:`~matplotlib.figure.Figure`, or ``None`` when ``ax`` is ``None``.

    Examples:
        - An axes reports the figure that owns it:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from digitalearth.static.figures import fig_of
            >>> fig, ax = plt.subplots()
            >>> fig_of(ax) is fig
            True

            ```
        - ``None`` short-circuits to ``None``:
            ```python
            >>> from digitalearth.static.figures import fig_of
            >>> fig_of(None) is None
            True

            ```
    """
    return ax.get_figure() if ax is not None else None
