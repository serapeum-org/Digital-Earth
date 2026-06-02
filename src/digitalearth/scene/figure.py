"""figure — multi-panel layout: a grid of :class:`~digitalearth.scene.map.Map` panels sharing one figure.

earthkit-plots models a figure as ``Figure → Subplot/Map → Layer``. Digital-Earth keeps the panel itself
(``Map``) as the unit and adds a thin :func:`grid` that creates the matplotlib figure + axes grid and binds a
``Map`` to each axes, plus :func:`shared_colorbar` for one colorbar spanning the panels. This is orchestration
only — the rendering stays in each ``Map`` (pyramids + cleopatra).
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from digitalearth.scene.map import Map

__all__ = ["grid", "shared_colorbar"]


def grid(nrows: int, ncols: int, *, crs: Any = 3857, globe: bool = False,
         figsize: Optional[Tuple[float, float]] = None, **kwargs) -> Tuple[Figure, List[Map]]:
    """Create an ``nrows`` × ``ncols`` grid of :class:`Map` panels sharing one figure.

    Each cell of a ``matplotlib`` subplot grid is wrapped in a ``Map`` (all the same ``crs``/``globe``), so
    panels can be drawn on independently while sharing one figure for a single ``savefig`` / colorbar / title.

    Args:
        nrows: Number of panel rows.
        ncols: Number of panel columns.
        crs: Display CRS for every panel (passed to each ``Map``).
        globe: When True, every panel is a globe (``Map(globe=True)``).
        figsize: Figure size in inches; ``None`` uses the matplotlib default.
        **kwargs: Forwarded to each ``Map`` (e.g. ``domain``).

    Returns:
        ``(fig, maps)`` — the shared :class:`~matplotlib.figure.Figure` and the list of ``Map`` panels in
        row-major (left-to-right, top-to-bottom) order, length ``nrows * ncols``.

    Examples:
        - A 2×2 grid yields four Maps sharing one figure:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.scene.figure import grid
            >>> fig, maps = grid(2, 2, crs=4326)
            >>> len(maps)
            4
            >>> all(m.fig is fig for m in maps)
            True

            ```
        - Draw on each panel independently (they share the figure):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.scene.figure import grid
            >>> fig, maps = grid(1, 2, crs=4326)
            >>> maps[0].set_title("left")
            >>> maps[1].set_title("right")
            >>> [m.ax.get_title() for m in maps]
            ['left', 'right']

            ```
    """
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axs).ravel()
    maps = [Map(crs=crs, globe=globe, ax=ax, fig=fig, **kwargs) for ax in axes]
    return fig, maps


def shared_colorbar(fig: Figure, mappable: Any, maps: Optional[List[Map]] = None, *,
                    label: Optional[str] = None, **kwargs) -> Any:
    """Add one colorbar to ``fig`` spanning the given panels (or every axes when ``maps`` is ``None``).

    Args:
        fig: The figure created by :func:`grid`.
        mappable: A drawn mappable (e.g. an ``AxesImage`` / ``QuadMesh`` returned by a panel's ``imshow``)
            whose colour scale the bar represents.
        maps: Panels the colorbar should steal space from; ``None`` spans all of the figure's axes.
        label: Optional colorbar label.
        **kwargs: Forwarded to ``Figure.colorbar`` (e.g. ``orientation``, ``shrink``, ``fraction``).

    Returns:
        The :class:`~matplotlib.colorbar.Colorbar` added to the figure.
    """
    axes = [m.ax for m in maps] if maps is not None else None
    cbar = fig.colorbar(mappable, ax=axes, **kwargs)
    if label is not None:
        cbar.set_label(label)
    return cbar
