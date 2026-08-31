"""Scene — a shared matplotlib axes that cleopatra glyphs render onto so layers compose.

A ``Scene`` owns one ``fig``/``ax``. Each plot method (added by subclasses / mixins in later tasks) builds a
cleopatra glyph with ``ax=self.ax``/``fig=self.fig``, calls its ``plot(...)``, and registers the returned
mappable via :meth:`_add_layer`. The Scene then owns figure-level decoration: one aggregated colorbar
(:meth:`colorbar`) or a categorical legend (:meth:`legend`), delegating to cleopatra's builders.

Because every cleopatra 0.10.0 glyph accepts a shared ``ax``/``fig`` and can suppress its own colorbar
(``add_colorbar=False`` on ``ArrayGlyph``), the Scene can stack any number of layers on one axes and draw a
single colorbar for the layer of interest.
"""
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from cleopatra.styling.styles import colorbar_legend, disjoint_legend
from cleopatra.styling.watermark import stamp_mark
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from digitalearth._render_compat import prepare_plot_kwargs


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

    Examples:
        - Create a scene; it owns exactly one axes until a colorbar is added:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.scene import Scene
            >>> scene = Scene()
            >>> len(scene.fig.axes)
            1
            >>> scene.layers
            []

            ```
        - Wrap a caller-supplied figure/axes instead of creating one:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from digitalearth.scene import Scene
            >>> fig, ax = plt.subplots()
            >>> scene = Scene(ax=ax, fig=fig)
            >>> scene.ax is ax
            True

            ```
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

    def _render_glyph(self, glyph: Any, *plot_args: Any, artist: str = "im", **plot_kwargs: Any) -> Any:
        """Plot ``glyph`` on the shared axes, register the produced mappable, and return it.

        Consolidates the recipe every plot method shared — call ``glyph.plot(...)``, find the mappable it
        produced, then :meth:`_add_layer` — so it lives in one place. cleopatra glyphs expose their mappable in
        one of two ways, selected by ``artist``:

        - ``"im"`` (default): the mappable is ``glyph.im`` (``ArrayGlyph`` / ``MeshGlyph``).
        - ``"plot"``: ``glyph.plot()`` returns ``(fig, ax, artist)`` and the mappable is that third element
          (``Scatter`` / ``Polygon`` / ``Vector`` / ``KDE`` / ``Flow`` glyphs).

        Args:
            glyph: An already-constructed cleopatra glyph bound to this Scene's ``ax``/``fig``.
            *plot_args: Positional arguments forwarded to ``glyph.plot`` (e.g. the data array for a mesh).
            artist: Which return convention to read the mappable from (``"im"`` or ``"plot"``).
            **plot_kwargs: Keyword arguments forwarded to ``glyph.plot`` (e.g. ``kind``, ``outline_only``).

        Returns:
            The registered mappable/artist (so callers can chain a colorbar or keep a reference).
        """
        plot_kwargs, deferred_alpha = prepare_plot_kwargs(glyph, plot_kwargs)
        result = glyph.plot(*plot_args, **plot_kwargs)
        mappable = glyph.im if artist == "im" else result[2]
        if deferred_alpha is not None and mappable is not None:
            mappable.set_alpha(deferred_alpha)
        return self._add_layer(glyph, mappable)

    @contextmanager
    def _preserve_view(self) -> Iterator[None]:
        """Hold the current axes limits across the block, but only when data is already drawn.

        A global backdrop or decoration (basemap, coastlines, ocean fill, a Natural-Earth layer) would
        otherwise autoscale a regional view back out to the whole world. When the axes already holds a data
        layer (a registered layer, an image, or a collection), the pre-block x/y limits are captured and
        restored on exit; on an otherwise-empty axes the block is free to set the initial extent.

        Yields:
            None — run the drawing code inside the ``with`` block.
        """
        has_data = bool(self.layers) or bool(self.ax.images) or bool(self.ax.collections)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        yield
        if has_data:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

    def colorbar(self, layer: int = -1, label: Optional[str] = None, **kwargs) -> Any:
        """Draw one colorbar for a registered layer (delegates to ``cleopatra.styling.styles.colorbar_legend``).

        Args:
            layer: Index into :attr:`layers` (default ``-1``, the most recent layer).
            label: Optional text label drawn alongside the colorbar.
            **kwargs: Forwarded to ``colorbar_legend`` / ``matplotlib`` colorbar.

        Returns:
            The created ``matplotlib.colorbar.Colorbar``.

        Raises:
            ValueError: if no layers have been registered.
        """
        if not self.layers:
            raise ValueError("no layers to draw a colorbar for; add a glyph first")
        cbar = colorbar_legend(self.layers[layer][1], ax=self.ax, **kwargs)
        if label is not None:
            cbar.set_label(label)
        return cbar

    def colorbars(self, **kwargs) -> List[Any]:
        """Draw one colorbar per registered layer (aggregation across all layers).

        Args:
            **kwargs: Forwarded to :meth:`colorbar` for every layer.

        Returns:
            The list of created colorbars, one per layer (empty when there are no layers).
        """
        return [self.colorbar(layer=i, **kwargs) for i in range(len(self.layers))]

    def legend(self, colors: Sequence, labels: Sequence[str], **kwargs) -> Any:
        """Attach a categorical (disjoint) swatch legend (delegates to ``cleopatra.styling.styles.disjoint_legend``).

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

    def stamp(self, mark: Any, **kwargs: Any) -> Any:
        """Stamp a logo / watermark onto the figure (delegates to ``cleopatra.styling.watermark.stamp_mark``).

        The mark is placed in one corner of :attr:`fig` on a frameless inset axes in figure-fraction
        coordinates, so it keeps its proportion and corner offset at whatever dpi the figure is later saved
        at. Because it is figure-level rather than axes-level it sits above every layer, and works on any
        scene — a :class:`~digitalearth.scene.map.Map`, a chart, or a bare :class:`Scene`.

        Args:
            mark: The mark image — a file path (any format Pillow can open) or an in-memory ``(H, W, 3)`` /
                ``(H, W, 4)`` array, either ``uint8`` ``0-255`` or float ``0-1``.
            **kwargs: Forwarded to ``stamp_mark`` — ``frac`` (the mark's longer side as a fraction of the
                figure, default ``0.11``), ``corner`` (``"lower right"`` / ``"lower left"`` /
                ``"upper right"`` / ``"upper left"``), ``margin``, ``shadow`` and ``blur``.

        Returns:
            The frameless inset ``Axes`` the mark was drawn on.

        Raises:
            ValueError: if ``corner``, ``frac``, ``margin`` or ``blur`` is out of contract, if the mark plus
                its margin would not fit the figure, or if an array mark has a bad shape/dtype/range.
            FileNotFoundError: if ``mark`` is a path that does not exist.

        Warning:
            Stamp **last** — the mark is baked from the figure's current size, so call this after any
            ``tight_layout()`` and after the final ``set_size_inches``. Note also that :meth:`save` defaults
            to ``bbox_inches="tight"``, which crops surrounding whitespace and so shifts the mark's relative
            margin; pass ``bbox_inches=None`` to :meth:`save` to preserve the placement exactly.

        Examples:
            - Stamp a small opaque mark into the lower-right corner:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import Scene
                >>> scene = Scene(figsize=(8, 6))
                >>> logo = np.zeros((40, 80, 4), dtype=np.uint8)
                >>> logo[..., :3] = 255
                >>> logo[..., 3] = 255
                >>> mark_ax = scene.stamp(logo, frac=0.2, shadow=False)
                >>> [round(float(v), 3) for v in mark_ax.get_position().bounds]
                [0.775, 0.025, 0.2, 0.133]

                ```
        """
        return stamp_mark(self.fig, mark, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """Save the figure to ``path`` (``bbox_inches="tight"`` by default)."""
        self.fig.savefig(path, **{"bbox_inches": "tight", **kwargs})

    def show(self) -> None:
        """Show the figure via ``matplotlib.pyplot.show``."""
        plt.show()

    def __enter__(self) -> "Scene":
        """Enter the runtime context, returning the scene so ``with Scene(...) as s:`` binds it.

        Returns:
            This scene.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the figure on exit so a long run of scenes stays memory-bounded.

        The figure is closed whether or not the body raised; any exception propagates (``__exit__`` returns
        ``False``), so ``with`` never silences errors.

        .. warning::
            This closes the **entire** ``self.fig``. The panels returned by
            :func:`~digitalearth.scene.figure.grid` share **one** figure, so using ``with`` on a single panel
            would close the figure for *all* panels — don't context-manage an individual ``grid`` panel; wrap
            the whole workflow or call :meth:`save` then close the figure yourself instead.

        Returns:
            ``False`` — exceptions are not suppressed.
        """
        plt.close(self.fig)
        return False
