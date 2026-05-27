"""Scene — a shared matplotlib axes that cleopatra glyphs render onto so layers compose.

A ``Scene`` owns one ``fig``/``ax``. Each plot method (added by subclasses / mixins in later tasks) builds a
cleopatra glyph with ``ax=self.ax``/``fig=self.fig``, calls its ``plot(...)``, and registers the returned
mappable via :meth:`_add_layer`. The Scene then owns figure-level decoration: one aggregated colorbar
(:meth:`colorbar`) or a categorical legend (:meth:`legend`), delegating to cleopatra's builders.

Because every cleopatra 0.10.0 glyph accepts a shared ``ax``/``fig`` and can suppress its own colorbar
(``add_colorbar=False`` on ``ArrayGlyph``), the Scene can stack any number of layers on one axes and draw a
single colorbar for the layer of interest.
"""
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from cleopatra.styles import colorbar_legend, disjoint_legend
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Scene:
    """A shared-axes host for composing cleopatra glyph layers into one figure.

    Args:
        ax: An existing axes to draw on. When ``None`` a new figure/axes is created.
        fig: The figure owning ``ax``. Ignored unless ``ax`` is also given.
        figsize: Size of the new figure when one is created (``(width, height)`` in inches).

    Attributes:
        fig: The matplotlib figure.
        ax: The matplotlib axes all layers render onto.
        layers: Registered ``(glyph, mappable)`` pairs, in draw order, used for legends/colorbars.
    """

    def __init__(
        self,
        ax: Optional[Axes] = None,
        fig: Optional[Figure] = None,
        figsize: Tuple[float, float] = (8, 8),
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        self.fig: Figure = fig
        self.ax: Axes = ax
        self.layers: List[Tuple[Any, Any]] = []

    def _add_layer(self, glyph: Any, mappable: Any) -> Any:
        """Register a rendered glyph and its mappable, returning the mappable.

        Args:
            glyph: The cleopatra glyph instance that was drawn on :attr:`ax`.
            mappable: The matplotlib mappable/artist the glyph produced (e.g. ``glyph.im``).

        Returns:
            The ``mappable`` (so callers can chain or attach a colorbar).
        """
        self.layers.append((glyph, mappable))
        return mappable

    def colorbar(self, layer: int = -1, **kwargs) -> Any:
        """Draw one colorbar for a registered layer (delegates to ``cleopatra.styles.colorbar_legend``).

        Args:
            layer: Index into :attr:`layers` (default ``-1``, the most recent layer).
            **kwargs: Forwarded to ``colorbar_legend`` / ``matplotlib`` colorbar.

        Returns:
            The created ``matplotlib.colorbar.Colorbar``.

        Raises:
            ValueError: if no layers have been registered.
        """
        if not self.layers:
            raise ValueError("no layers to draw a colorbar for; add a glyph first")
        return colorbar_legend(self.layers[layer][1], ax=self.ax, **kwargs)

    def legend(self, colors: Sequence, labels: Sequence[str], **kwargs) -> Any:
        """Attach a categorical (disjoint) swatch legend (delegates to ``cleopatra.styles.disjoint_legend``).

        Args:
            colors: One color per category.
            labels: One label per category (same length/order as ``colors``).
            **kwargs: Forwarded to ``disjoint_legend`` / ``ax.legend``.

        Returns:
            The created ``matplotlib.legend.Legend``.
        """
        return disjoint_legend(self.ax, colors, labels, **kwargs)

    def set_title(self, title: str, **kwargs) -> None:
        """Set the axes title."""
        self.ax.set_title(title, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """Save the figure to ``path`` (``bbox_inches="tight"`` by default)."""
        self.fig.savefig(path, **{"bbox_inches": "tight", **kwargs})

    def show(self) -> None:
        """Show the figure via ``matplotlib.pyplot.show``."""
        plt.show()
