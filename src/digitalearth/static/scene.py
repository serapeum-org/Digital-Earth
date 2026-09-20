"""Scene — a shared matplotlib axes that cleopatra glyphs render onto so layers compose.

A ``Scene`` owns one ``fig``/``ax``. Each plot method (added by subclasses / mixins in later tasks) builds a
cleopatra glyph with ``ax=self.ax``/``fig=self.fig``, calls its ``plot(...)``, and registers the returned
mappable via :meth:`_add_layer`. The Scene then owns figure-level decoration: one aggregated colorbar
(:meth:`colorbar`) or a categorical legend (:meth:`legend`), delegating to cleopatra's builders.

Because every cleopatra 0.10.0 glyph accepts a shared ``ax``/``fig`` and can suppress its own colorbar
(``add_colorbar=False`` on ``ArrayGlyph``), the Scene can stack any number of layers on one axes and draw a
single colorbar for the layer of interest.

Beside the drawing the Scene keeps a **description** of it (#303): a
:class:`~digitalearth.base.spec.LayerTree` of what each layer *is*, a table of where each one's data came
from, and the view they are drawn in — read back as :attr:`Scene.figure_spec`. The two are deliberately
separate structures: ``layers`` holds the matplotlib artists a colorbar is keyed to, and the tree holds the
engine-neutral record a renderer could rebuild them from. A layer with no mappable to register (a tile
basemap, a graticule, a Natural-Earth overlay drawn straight onto the axes) is described without appearing
in ``layers``, which is why the description is not simply read off that list.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
from cleopatra.styling.styles import colorbar_legend, disjoint_legend
from cleopatra.styling.watermark import WatermarkMixin
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from digitalearth.base.custom import custom_kind
from digitalearth.base.registry import forget_object, object_namespace
from digitalearth.base.spec import (
    DataRef,
    FigureSpec,
    LayerSpec,
    LayerTree,
    PanelSpec,
    Symbology,
    Viewport,
)
from digitalearth.static.render_compat import prepare_plot_kwargs

#: The id of the one panel this tier draws into. A matplotlib ``Scene`` owns one axes, so it is one panel;
#: the constant is named — and spelled the same as every other tier's — so a reader of :attr:`Scene.figure_spec`
#: sees the same panel id wherever the figure was built.
PANEL_ID: str = "main"

#: The engine whose objects a caller hands this tier, for :func:`~digitalearth.base.custom.custom_kind`. An
#: artist registered without a kind was built by the caller, not by a builder here, and matplotlib is the only
#: engine it can have come from.
ENGINE: str = "matplotlib"


@dataclass(frozen=True)
class LayerRecord:
    """What a builder says about the layer it just drew, so the figure can describe it.

    A builder draws with cleopatra; this is the engine-neutral half of the same call — the registered kind the
    layer *is*, where its data came from, and how it looks as values rather than as the keyword soup the glyph
    was handed.

    It travels as **one value** rather than as loose keywords because
    :meth:`Scene._render_glyph` forwards ``**plot_kwargs`` straight on to ``glyph.plot``, and cleopatra's own
    ``kind=`` (``"imshow"``, ``"quiver"``, ``"tripcolor"``) travels in that soup: a second ``kind=`` parameter
    beside it would be two different words spelled the same, one naming a matplotlib call and one naming a
    layer kind.

    Attributes:
        kind: The registered, engine-neutral kind — ``"raster"``, ``"points"``, ``"choropleth"`` — never the
            matplotlib artist class, which cannot tell a choropleth from a plain polygon fill.
        source: What the layer draws, recorded in the figure's sources under the layer's id: a path, a URL, or
            the pyramids object itself. ``None`` for a layer drawn from no data, such as a graticule. A u/v
            field records the **pair**, because neither component alone draws it.
        name: The caller's own name for the layer, used as its id and its label when it is free. ``None``
            generates one from the kind.
        band: Where the layer is drawn, when its kind's band is not where *this* layer belongs — a backdrop
            raster drawn under the data, or a caller's own artist, since ``custom:matplotlib`` names the engine
            and says nothing about what it draws. ``None`` takes the kind's band.
        visible: Whether the layer was built visible.
        symbology: How it looks, as values. ``None`` records an empty symbology.

    Examples:
        - The record a raster builder writes, beside the drawing it made:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.static.scene import LayerRecord
            >>> record = LayerRecord("raster", source="dem.tif", symbology=Symbology.of(opacity=0.5))
            >>> record.kind, record.source, record.band, record.visible
            ('raster', 'dem.tif', None, True)

            ```
        - A backdrop is the same kind drawn somewhere else, which is what ``band`` is for:
            ```python
            >>> from digitalearth.static.scene import LayerRecord
            >>> LayerRecord("raster", source="relief.tif", band="underlay").band
            'underlay'

            ```
    """

    kind: str
    source: Any = None
    name: Optional[str] = None
    band: Optional[str] = None
    visible: bool = True
    symbology: Optional[Symbology] = None


class Scene(WatermarkMixin):
    """A shared-axes host for composing cleopatra glyph layers into one figure.

    Inherits cleopatra's ``WatermarkMixin``, whose surface comes with it as-is: :meth:`stamp` below is this
    package's documented spelling for the logo stamp, and ``stamp_watermark(text)`` (diagonal brand text
    with an optional credit line) is available unchanged from upstream, documented there rather than here.

    Args:
        ax: An existing axes to draw on. When ``None`` a new figure/axes is created.
        fig: The figure owning ``ax``. Ignored unless ``ax`` is also given.
        figsize: Size of the new figure when one is created (``(width, height)`` in inches).
        strict: How a layer with nothing to draw is handled. ``False`` (default) skips it and logs a
            warning naming the layer and why; ``True`` raises instead, so a pipeline that must not
            silently produce an empty figure fails where the layer was added.

    Attributes:
        fig: The matplotlib figure.
        ax: The matplotlib axes all layers render onto.
        layers: Registered ``(glyph, mappable)`` pairs, in draw order, used for legends/colorbars.
        strict: Whether a layer that draws nothing raises instead of being skipped.

    Examples:
        - Create a scene; it owns exactly one axes until a colorbar is added, and describes nothing yet:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Scene
            >>> scene = Scene()
            >>> len(scene.fig.axes)
            1
            >>> scene.layers
            []
            >>> scene.layer_ids
            []

            ```
        - Wrap a caller-supplied figure/axes instead of creating one:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from digitalearth.static import Scene
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
        strict: bool = False,
    ):
        """Build the shared figure/axes host that layers render onto.

        Args:
            ax: An existing axes to draw on. When given, the scene does **not** own the figure and will not
                close it on exit — pass one to compose a Digital-Earth layer into a figure you are laying out
                yourself.
            fig: The figure `ax` belongs to; taken from `ax` when omitted.
            figsize: Size of the figure created when `ax` is None, in inches.
            strict: What to do with a layer that has nothing to draw — data entirely outside the view, or a
                band with no finite values. `False` (the default) skips it with a warning naming the layer,
                so one bad frame does not abort a batch; `True` raises `OffLimbError` instead, which is what
                a pipeline that must not publish a map with a layer missing should pass.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        self.fig: Figure = fig
        self.ax: Axes = ax
        self.strict: bool = bool(strict)
        self.layers: List[Tuple[Any, Any]] = []
        #: One entry per registered layer (same order as :attr:`layers`): the label :meth:`colorbar` falls
        #: back to when the caller passes none — the layer's resolved ``units``, or ``None``.
        self._layer_labels: List[Optional[str]] = []
        # What the scene draws, as data. `self.layers` holds the matplotlib artists — the drawing — and this
        # holds the description each was built from, which is what a figure can be written to and read back
        # from (#303). They are not the same list: a basemap, a graticule or a Natural-Earth overlay draws
        # straight onto the axes with no mappable to register, and is described all the same.
        self._layer_tree: LayerTree = LayerTree()
        self._sources: Dict[str, DataRef] = {}
        # Namespaced per scene: every figure restarts its layer numbering, so two maps in one session both
        # mint `raster-1`, and one process-global object table would have the second silently re-point the
        # first figure's captured source at its own data.
        self._objects_ns: str = object_namespace()
        self._id_counter: int = 0
        self._issued_ids: Set[str] = set()

    def _layer_id(self, prefix: str, name: Optional[str] = None) -> str:
        """Return a unique layer id: the caller's name when they gave one, else a generated one.

        Args:
            prefix: What a generated id counts — the kind, usually.
            name: The caller's own name for the layer, used as its id when it is free.

        Returns:
            The id. A caller's name that collides with one already issued is suffixed, because two layers
            sharing an id makes the second unaddressable.
        """
        candidate = name or ""
        while not candidate or candidate in self._issued_ids:
            self._id_counter += 1
            candidate = (
                f"{name}-{self._id_counter}" if name else f"{prefix}-{self._id_counter}"
            )
        self._issued_ids.add(candidate)
        return candidate

    def _index_layer(
        self,
        layer_id: str,
        label: Optional[str],
        *,
        kind: str,
        visible: bool = True,
        band: Optional[str] = None,
        source: Any = None,
        symbology: Any = None,
    ) -> None:
        """Record a layer so the figure describes it rather than only holding its artist.

        Args:
            layer_id: The layer's id, as :attr:`layer_ids` reports it.
            label: What a layer switcher should call it; `None` falls back to the id.
            kind: The registered, engine-neutral kind — `"raster"`, `"points"`, `"choropleth"` — not the
                matplotlib artist class, which cannot tell a choropleth from a plain polygon fill.
            visible: Whether the layer was built visible.
            band: Where it is drawn, when its kind does not say — what a backdrop and a caller's own artist
                need, since `custom:matplotlib` names the engine rather than what it draws.
            source: What the layer draws, recorded under its id. `None` for a layer drawn from no data.
            symbology: How it looks, as values rather than as the flat kwargs the glyph was handed.
        """
        if source is not None:
            self._sources[layer_id] = DataRef.of(
                source, name=f"{self._objects_ns}:{layer_id}"
            )
        self._layer_tree = self._layer_tree.add(
            LayerSpec(
                layer_id,
                kind,
                source_id=layer_id if source is not None else None,
                symbology=Symbology() if symbology is None else symbology,
                label=label or layer_id,
                # By truthiness, because a builder decides visibility the same loose way it decides every
                # other flag, and `LayerSpec` takes only a real boolean.
                visible=bool(visible),
                band=band,
            )
        )

    def _describe_layer(self, record: LayerRecord) -> str:
        """Record one layer in the figure and return the id it was given.

        The single registration funnel: every builder in this tier ends here, whether or not it also had a
        mappable to register in :attr:`layers`. Routing them all through one place is what keeps the kind a
        layer is recorded under an engine-neutral noun rather than whichever cleopatra glyph happened to draw
        it.

        Args:
            record: What the builder drew, as :class:`LayerRecord` describes it.

        Returns:
            The layer's id, which a caller-supplied name becomes when it is free.
        """
        # A generated id counts the kind, so `custom:matplotlib` numbers `custom-1`: a colon cannot appear in
        # the middle of an id, and the engine name is not what a reader is counting anyway.
        layer_id = self._layer_id(record.kind.split(":")[0], record.name)
        self._index_layer(
            layer_id,
            record.name,
            kind=record.kind,
            visible=record.visible,
            band=record.band,
            source=record.source,
            symbology=record.symbology,
        )
        return layer_id

    @property
    def layer_ids(self) -> List[str]:
        """The ids of the layers this scene draws, in draw order, bottom first.

        Returns:
            One id per described layer. A layer the caller named carries that name; an unnamed one gets a
            generated id counting the layers of its kind.
        """
        return list(self._layer_tree.ids)

    @property
    def figure_spec(self) -> FigureSpec:
        """What the scene draws, as data: its layers, their sources and the view.

        Returns:
            A :class:`~digitalearth.base.spec.FigureSpec` with one panel, :data:`PANEL_ID`, whose layers are
            the tree in draw order and whose sources are what each builder was given.
        """
        tree = self._layer_tree
        panel = PanelSpec(PANEL_ID, self.viewport, layers=tuple(tree.ids))
        return FigureSpec(
            panels=(panel,),
            layers=tree,
            sources={
                key: ref for key, ref in self._sources.items() if key in set(tree.ids)
            },
        )

    @property
    def viewport(self) -> Viewport:
        """Where the scene is looking, as a value.

        A bare :class:`Scene` is a figure host rather than a map — it has no display CRS to place anything in
        — so the view is the default one. :class:`~digitalearth.static.maps.base.GeoLayerBase` overrides this
        with the CRS, domain and globe flag it was built with.

        Returns:
            A :class:`~digitalearth.base.spec.Viewport`.
        """
        return Viewport()

    def _add_layer(
        self,
        glyph: Any,
        mappable: Any,
        label: Optional[str] = None,
        *,
        describe: Optional[LayerRecord] = None,
    ) -> Any:
        """Register a rendered glyph and its mappable, returning the mappable.

        Args:
            glyph: The cleopatra glyph instance that was drawn on :attr:`ax`.
            mappable: The matplotlib mappable/artist the glyph produced (e.g. ``glyph.im``).
            label: Optional default colorbar label for this layer (e.g. the units resolved by
                :func:`~digitalearth.base.autostyle.auto_style`); used only when :meth:`colorbar` is
                called without one.
            describe: What the builder drew, recorded in the figure. ``None`` means nobody said — the artist
                is the caller's own, so it is recorded as ``custom:matplotlib`` and described by id, kind and
                label alone: an artist somebody built by hand has no source to reference and no symbology to
                read back, so a figure can name it, hide it and reorder it, but never rebuild it.

        Returns:
            The ``mappable`` (so callers can chain or attach a colorbar).
        """
        self.layers.append((glyph, mappable))
        self._layer_labels.append(label)
        self._describe_layer(
            describe if describe is not None else LayerRecord(custom_kind(ENGINE))
        )
        return mappable

    def _reset_layers(self) -> None:
        """Forget every registered layer — its default label and its description — e.g. between frames.

        The description is reset with the drawing, because an animation redraws every layer on a cleared
        axes: leaving the tree alone would have frame 50 describe 50 copies of one raster. The id counter
        goes back with it, so each frame mints the same ids and the in-memory source each one registered is
        replaced rather than accumulated. Whatever the dropped layers referenced is forgotten too, so a long
        run does not hold every frame's dataset alive through the process-wide object table.
        """
        for ref in self._sources.values():
            forget_object(ref.uri)
        self.layers = []
        self._layer_labels = []
        self._layer_tree = LayerTree()
        self._sources = {}
        self._id_counter = 0
        self._issued_ids = set()

    def _default_label(self, layer: int) -> Optional[str]:
        """Return the default colorbar label recorded for ``layer``, or ``None`` when there is none.

        Args:
            layer: Index into :attr:`layers` (negative indices count from the most recent layer).

        Returns:
            The label recorded by :meth:`_add_layer`, or ``None`` — also when the index is out of range,
            so an unlabelled colorbar is never turned into an ``IndexError``.
        """
        # pragma-guarded: the labels track the layers one-for-one, so the index is always in range.
        try:
            return self._layer_labels[layer]
        except IndexError:  # pragma: no cover - defensive
            return None

    def _render_glyph(
        self,
        glyph: Any,
        *plot_args: Any,
        artist: str = "im",
        label: Optional[str] = None,
        describe: Optional[LayerRecord] = None,
        **plot_kwargs: Any,
    ) -> Any:
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
            label: Optional default colorbar label recorded with the layer (see :meth:`_add_layer`); it is
                *not* forwarded to ``glyph.plot``.
            describe: What this layer is, for the figure — see :class:`LayerRecord`. It is not forwarded to
                ``glyph.plot`` either; it travels as one value precisely so that its ``kind`` (the layer's)
                cannot be confused with the ``kind`` in ``plot_kwargs`` (cleopatra's render call).
            **plot_kwargs: Keyword arguments forwarded to ``glyph.plot`` (e.g. ``kind``, ``outline_only``).

        Returns:
            The registered mappable/artist (so callers can chain a colorbar or keep a reference).
        """
        plot_kwargs, deferred_alpha = prepare_plot_kwargs(glyph, plot_kwargs)
        result = glyph.plot(*plot_args, **plot_kwargs)
        mappable = glyph.im if artist == "im" else result[2]
        if deferred_alpha is not None and mappable is not None:
            mappable.set_alpha(deferred_alpha)
        return self._add_layer(glyph, mappable, label, describe=describe)

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
        has_data = (
            bool(self.layers) or bool(self.ax.images) or bool(self.ax.collections)
        )
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        yield
        if has_data:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

    def colorbar(self, layer: int = -1, label: Optional[str] = None, **kwargs) -> Any:
        """Draw one colorbar for a registered layer (delegates to ``cleopatra.styling.styles.colorbar_legend``).

        Args:
            layer: Index into :attr:`layers` (default ``-1``, the most recent layer). Only *drawn* layers
                are registered — a layer whose data lies outside the display CRS draws nothing and takes
                no slot — so count positions from what was actually rendered, not from the calls made.
                That applies to the ``-1`` default too: if the most recent draw was skipped, ``-1`` is the
                one before it, so check the return value rather than assuming the last call registered.
            label: Optional text label drawn alongside the colorbar. When ``None`` the layer's own label is
                used if it has one — a raster field records the ``units``
                :func:`~digitalearth.base.autostyle.auto_style` resolved for its variable — and the bar is
                left unlabelled otherwise. A caller-supplied label always wins; pass ``""`` for none.
            **kwargs: Forwarded to ``colorbar_legend`` / ``matplotlib`` colorbar.

        Returns:
            The created ``matplotlib.colorbar.Colorbar``.

        Raises:
            ValueError: if no layers have been registered.
        """
        if not self.layers:
            raise ValueError("no layers to draw a colorbar for; add a glyph first")
        cbar = colorbar_legend(self.layers[layer][1], ax=self.ax, **kwargs)
        if label is None:
            label = self._default_label(layer)  # the layer's auto-styled units, if any
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
        """Stamp a logo / watermark onto the figure (delegates to cleopatra's ``WatermarkMixin.stamp_mark``).

        The mark is placed in one corner of :attr:`fig` on a frameless inset axes in figure-fraction
        coordinates, so it keeps its proportion and corner offset at whatever dpi the figure is later saved
        at. Because it is figure-level rather than axes-level it sits above every layer, and works on any
        scene — a :class:`~digitalearth.static.map.Map`, a chart, or a bare :class:`Scene`.

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
                >>> from digitalearth.static import Scene
                >>> scene = Scene(figsize=(8, 6))
                >>> logo = np.zeros((40, 80, 4), dtype=np.uint8)
                >>> logo[..., :3] = 255
                >>> logo[..., 3] = 255
                >>> mark_ax = scene.stamp(logo, frac=0.2, shadow=False)
                >>> [round(float(v), 3) for v in mark_ax.get_position().bounds]
                [0.775, 0.025, 0.2, 0.133]

                ```
        """
        return self.stamp_mark(mark, **kwargs)

    def save(self, path: Union[str, "os.PathLike[str]"], **kwargs) -> Path:
        """Save the figure to ``path`` (``bbox_inches="tight"`` by default).

        Args:
            path: Destination file path; the extension picks the format matplotlib writes.
            **kwargs: Forwarded to ``Figure.savefig`` (e.g. ``dpi``, ``bbox_inches``, ``transparent``).

        Returns:
            The path that was written, as a :class:`pathlib.Path` — every backend's ``save`` returns one, so
            a caller can chain onto the result without re-deriving it.

        Examples:
            - Save a scene and read the written file back off the return value:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import tempfile
                >>> from pathlib import Path
                >>> from digitalearth.static import Scene
                >>> out = Scene().save(Path(tempfile.mkdtemp()) / "scene.png")
                >>> out.suffix
                '.png'
                >>> out.exists()
                True

                ```
        """
        self.fig.savefig(path, **{"bbox_inches": "tight", **kwargs})
        return Path(path)

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
            :func:`~digitalearth.static.figure.grid` share **one** figure, so using ``with`` on a single panel
            would close the figure for *all* panels — don't context-manage an individual ``grid`` panel; wrap
            the whole workflow or call :meth:`save` then close the figure yourself instead.

        Returns:
            ``False`` — exceptions are not suppressed.
        """
        plt.close(self.fig)
        return False
