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
engine-neutral record a renderer rebuilds them from. A layer with no mappable to register (a tile
basemap, a graticule, a Natural-Earth overlay drawn straight onto the axes) is described without appearing
in ``layers``, which is why the description is not simply read off that list.

The description is what the drawing is *made from*, not a note beside it. Every builder ends at
:meth:`Scene._draw`, which records the layer and then asks
:class:`~digitalearth.static.renderer.Renderer` to draw it — so the layer is built once, by the drawer that
replays its recipe. A layer whose drawer declined it (data outside the display CRS, a label on the far side
of a globe) is dropped from the description again, so a figure never names something that was not drawn.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as with_fields
from numbers import Real
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
from cleopatra.styling.styles import colorbar_legend, disjoint_legend
from cleopatra.styling.watermark import WatermarkMixin
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from digitalearth.base.custom import custom_kind
from digitalearth.base.registry import (
    forget_namespace,
    forget_object,
    object_namespace,
)
from digitalearth.base.spec import (
    DataRef,
    FigureSpec,
    LayerSpec,
    LayerTree,
    PanelSpec,
    Symbology,
    Viewport,
)
from digitalearth.base.spec._serial import to_json_value
from digitalearth.static.render_compat import plot_takes, prepare_plot_kwargs
from digitalearth.static.renderer import DrawnLayer, Renderer, drawing_opts

#: The id of the one panel this tier draws into. A matplotlib ``Scene`` owns one axes, so it is one panel;
#: the constant is named — and spelled the same as every other tier's — so a reader of :attr:`Scene.figure_spec`
#: sees the same panel id wherever the figure was built.
PANEL_ID: str = "main"

#: The engine whose objects a caller hands this tier, for :func:`~digitalearth.base.custom.custom_kind`. An
#: artist registered without a kind was built by the caller, not by a builder here, and matplotlib is the only
#: engine it can have come from.
ENGINE: str = "matplotlib"

#: The property a layer's plain engine keywords are described under. Nested rather than merged flat into
#: ``props``, so a caller's ``zorder=`` can never overwrite the one a builder recorded about the layer itself.
DRAWING_OPTS_KEY: str = "opts"

#: Where :func:`_writer_takes` says it is asking about, for the message a refusal would carry.
_WRITER_FIELD: str = f"Symbology.props[{DRAWING_OPTS_KEY!r}]"


def _writer_takes(value: Any) -> bool:
    """Whether the figure writer accepts a value as it stands.

    The oracle is :func:`~digitalearth.base.spec._serial.to_json_value`, which ``Symbology.to_dict`` itself
    applies, so this cannot drift from what a figure accepts — it is what refuses ``nan`` and the infinities.

    Args:
        value: A value a layer is about to record.

    Returns:
        ``True`` when the description can carry it.
    """
    try:
        to_json_value(value, _WRITER_FIELD)
    except (TypeError, ValueError):
        return False
    return True


def travels_in_a_figure(value: Any) -> bool:
    """Whether one of the caller's engine keywords belongs in the description rather than beside the layer.

    The rule is **a plain value JSON reads back as the very same value**: a string, a boolean, a finite
    number, or ``None``. That is what the shared vocabulary already models — ``vmin``/``vmax`` are a
    ``Scale``'s bounds, ``color``, ``width``, ``size`` and ``opacity`` are declared channels — and it is
    what a reader on another machine can act on.

    Everything else stays beside the layer, and each exclusion is a defect this tier has already had:

    * A **container** is not read back as itself. JSON has no tuple, so a dash pattern written as
      ``(0, (5, 5))`` comes back ``[0, [5, 5]]``, which matplotlib refuses outright
      (``ValueError: Unrecognized linestyle``) — describing it would trade a layer that redraws with the
      engine's defaults for one that cannot redraw at all (round 1, H3).
    * An **array** is the layer's data, not its description. Writing one costs a conversion per cell and
      freezing one costs a copy per cell, which is what made a ``1000 x 1000`` per-pixel ``alpha`` take ten
      times the render it belonged to (round 1, L6).
    * An **engine object** — a ``Normalize``, a ``FontProperties``, a ``Colormap`` — has no JSON form at
      all, and a figure holding one could not be saved.

    Args:
        value: The caller's value for one keyword.

    Returns:
        ``True`` when the layer's description carries it.
    """
    if value is None or isinstance(value, (bool, str, Real)):
        return _writer_takes(value)
    return False


def described_opts(opts: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the half of a caller's engine keywords that a figure carries.

    Args:
        opts: Everything the caller passed through ``**opts``, or ``None``.

    Returns:
        The keywords :func:`travels_in_a_figure` accepts, in the order they were given. Empty for a layer
        whose keywords are all engine objects, which records no ``opts`` property at all.
    """
    return {
        key: value for key, value in (opts or {}).items() if travels_in_a_figure(value)
    }


def _with_described_opts(
    symbology: Optional[Symbology], opts: Optional[Mapping[str, Any]]
) -> Symbology:
    """Return a layer's symbology with the plain half of the caller's engine keywords written into it.

    Args:
        symbology: What the builder recorded about the layer, or ``None`` for one that recorded nothing.
        opts: Everything the caller passed through ``**opts``, or ``None``.

    Returns:
        The symbology, with a :data:`DRAWING_OPTS_KEY` property when there is anything to put in it and
        unchanged when there is not — so a layer whose caller passed nothing plain describes no such
        property at all rather than an empty one.
    """
    described = described_opts(opts)
    recorded = Symbology() if symbology is None else symbology
    if not described:
        return recorded
    return with_fields(recorded, props={**recorded.props, DRAWING_OPTS_KEY: described})


def drawing_style(scene: Any, layer: LayerSpec) -> Dict[str, Any]:
    """Return every engine keyword a layer is drawn with: the described half under the held half.

    The pair :func:`~digitalearth.static.renderer.drawing_opts` alone used to be. A description carries the
    plain keywords (see :func:`travels_in_a_figure`) so a figure read back on another scene still draws the
    caller's ``vmin``, ``alpha``, ``color`` and ``title`` rather than the engine's defaults; the scene that
    built the layer holds all of them, as the very objects passed, and those win — so on the scene that
    built it a layer is drawn with exactly what it always was, frozen copies included nowhere.

    Args:
        scene: The scene the layer is drawn on, which holds the caller's own objects.
        layer: The layer being drawn.

    Returns:
        A fresh dict a drawer may pop and set keys on, holding the description's keywords overlaid with
        whatever the scene holds.

    Examples:
        - A scene that holds nothing draws a described layer with what its figure carries:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.base.spec import LayerSpec, Symbology
            >>> from digitalearth.static import Scene
            >>> from digitalearth.static.scene import drawing_style
            >>> layer = LayerSpec("a", "raster", symbology=Symbology(props={"opts": {"vmin": 1.0}}))
            >>> drawing_style(Scene(), layer)
            {'vmin': 1.0}

            ```
    """
    described = dict(layer.symbology.props.get(DRAWING_OPTS_KEY) or {})
    return {**described, **drawing_opts(scene, layer)}


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
        held: The caller's **own engine object**, for a ``custom:matplotlib`` layer: the artist they
            built and handed to :meth:`Scene._add_layer`, with the glyph that made it and its
            colorbar label. A description cannot rebuild it — that is what makes the layer custom —
            so the scene keeps it under the layer's id and the drawer puts it back from there, the
            way every other tier holds its own engine's custom layers.
        key: Something the layer's drawer needs that a **description cannot carry** — a clip boundary (a
            shapely geometry, which a figure written to JSON has no spelling for, and which would make two
            symbologies uncomparable) or a basemap credential (which must never be written into a figure at
            all). It is held on the scene under the layer's id instead, and forgotten with the layer.
            ``None`` for every layer that needs none, which is nearly all of them.
        opts: The caller's **engine keywords** — whatever they passed through ``**opts`` to cleopatra or
            matplotlib — held exactly as passed, beside the layer rather than in its description. A dash
            pattern is a tuple matplotlib refuses as a list, and a ``Normalize``, a ``FontProperties`` or a
            per-pixel ``alpha`` array has no JSON spelling at all: a description that froze them handed the
            engine something else back, or could not be saved. The drawer reads them off the scene, so a
            figure drawn on a scene that does not hold them — one read back from JSON — draws the layer with
            the engine's defaults instead. ``None`` or empty for a layer given none.

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
    key: Any = None
    opts: Optional[Mapping[str, Any]] = None
    held: Any = None


class Scene(WatermarkMixin):
    """A shared-axes host for composing cleopatra glyph layers into one figure.

    Inherits cleopatra's ``WatermarkMixin``, whose surface comes with it as-is: :meth:`stamp` below is this
    package's documented spelling for the logo stamp, and ``stamp_watermark(text)`` (diagonal brand text
    with an optional credit line) is available unchanged from upstream, documented there rather than here.

    **One scene per axes.** A scene may be handed an axes to draw on — to place a map inside a figure you
    are laying out yourself — but **two scenes sharing one axes is not supported**. Each keeps its own
    record of what it has drawn and its own "have I drawn here yet" flag, and a cleopatra glyph clears the
    axes on a scene's first render: so the second scene wipes the first one's drawing while the first goes
    on describing it, and the figure it reports names a layer whose artist is gone. Lay several maps out as
    several axes (:func:`~digitalearth.static.figure.grid` does), and give each its own.

    Args:
        ax: An existing axes to draw on. When ``None`` a new figure/axes is created. One axes, one scene —
            see above.
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
                yourself. Give each scene an axes of its own: a second scene on the same axes clears what the
                first drew on its opening render, and the first still describes it (see the class docstring).
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
        # What a drawer needs that the figure cannot carry — a clip boundary, a basemap credential — keyed
        # by layer id. Deliberately not part of `symbology`: a figure is written to JSON and read back, and
        # neither a shapely geometry nor an API key belongs in one (see `LayerRecord.key`).
        self._layer_keys: Dict[str, Any] = {}
        # The caller's engine keywords, exactly as passed, keyed by layer id (see `LayerRecord.opts`). Held
        # here rather than in `symbology` for the same reason as the keys: a description is plain values,
        # and a dash tuple, a `Normalize` or a colormap object is not one.
        self._layer_opts: Dict[str, Dict[str, Any]] = {}
        # The caller's own objects, for the custom layers they handed in (see `LayerRecord.held`).
        self._held_objects: Dict[str, Any] = {}
        # Whether this scene has already drawn a glyph onto `ax` (#313). A cleopatra glyph clears every
        # glyph's artists off its axes unless it is told to compose, so from the *second* layer onwards a
        # render has to compose or it takes the layer below it off again. Only from the second: the first
        # one keeps replacing, so a scene handed an axes another figure was drawn on — including one an
        # earlier scene drew — still supersedes it rather than stacking on top of it silently.
        self._drew_on_axes: bool = False
        #: The renderer that turns this scene's description into artists on :attr:`ax`.
        self._renderer: Renderer = Renderer(self)

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

        The caller's engine keywords go **both** ways, which is deliberate. The plain ones are written into
        the layer's description (see :func:`travels_in_a_figure`), so a figure read back elsewhere draws the
        ``vmin``, ``alpha``, ``color`` or ``title`` the caller asked for instead of the engine's defaults.
        All of them — plain or not — stay held on the scene as the very objects passed, and
        :func:`drawing_style` lets those win, so a layer on the scene that built it is drawn with exactly
        what it always was rather than with a frozen copy.

        Args:
            record: What the builder drew, as :class:`LayerRecord` describes it.

        Returns:
            The layer's id, which a caller-supplied name becomes when it is free.
        """
        # A generated id counts the kind, so `custom:matplotlib` numbers `custom-1`: a colon cannot appear in
        # the middle of an id, and the engine name is not what a reader is counting anyway.
        layer_id = self._layer_id(record.kind.split(":")[0], record.name)
        if record.key is not None:
            self._layer_keys[layer_id] = record.key
        if record.held is not None:
            self._held_objects[layer_id] = record.held
        if record.opts:
            # A shallow copy: the dict is this layer's, so a caller reusing theirs cannot re-style it later,
            # while each value stays the very object they passed.
            self._layer_opts[layer_id] = dict(record.opts)
        self._index_layer(
            layer_id,
            record.name,
            kind=record.kind,
            visible=record.visible,
            band=record.band,
            source=record.source,
            symbology=_with_described_opts(record.symbology, record.opts),
        )
        return layer_id

    def _draw(self, record: LayerRecord) -> Any:
        """Record a layer and draw it, returning whatever its drawer handed back.

        The funnel every builder of a drawn kind ends at. The order is the point: the layer is **described
        first** and its drawer builds the artists from that description, so the two can never disagree about
        what the figure holds. A builder that computed the artists itself and described them afterwards was
        writing the same layer twice, in two vocabularies, with nothing holding them to each other.

        Args:
            record: What the builder wants drawn, as :class:`LayerRecord` describes it.

        Returns:
            The drawer's ``artist`` — the mappable a field render produced, the ``Text`` of a label, the
            polylines a limb-split coastline became — or ``None`` when the drawer declined the layer, which
            is what an off-limb raster or a far-side label is.

        Raises:
            Exception: whatever the drawer raises, after the layer has been dropped from the description
                again — a figure must not name a layer that was not drawn, and that is as true of a refusal
                as it is of a skip.
        """
        layer_id = self._describe_layer(record)
        try:
            drawn = self._renderer.draw_layer(self.figure_spec, layer_id)
        except BaseException:
            self._forget_layer(layer_id)
            raise
        if drawn is None:
            self._forget_layer(layer_id)
            return None
        return drawn.artist

    def _forget_layer(self, layer_id: str) -> None:
        """Drop a layer from the description, and with it whatever it registered.

        What :meth:`_draw` calls for a layer its drawer declined or refused. The id goes back to the pool
        of free names too, so a caller who names a layer, watches it skip, and names it again gets the name
        they asked for rather than a suffixed one.

        Args:
            layer_id: The layer to forget. One already gone is ignored.
        """
        self._layer_tree = self._layer_tree.remove(layer_id)
        self._issued_ids.discard(layer_id)
        self._forget_layer_data(layer_id)

    def _forget_layer_data(self, layer_id: str) -> None:
        """Let go of the in-memory data, the drawer key and the engine keywords one layer held.

        The object table is process-global and holds strong references, so a layer that is removed and
        never forgotten keeps its dataset alive for the life of the process.

        Args:
            layer_id: The layer whose source, key and keywords are dropped. A layer that registered none of
                them — a graticule, a text label — is ignored.
        """
        ref = self._sources.pop(layer_id, None)
        if ref is not None:
            forget_object(ref.uri)
        self._layer_keys.pop(layer_id, None)
        self._layer_opts.pop(layer_id, None)
        self._held_objects.pop(layer_id, None)

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

            It describes the layers, not the caller's styling: the engine keywords passed through ``**opts``
            are held beside the layer (:attr:`LayerRecord.opts`) rather than in it, so a figure drawn on
            another scene draws with the engine's defaults in their place.

            A scene built from data already in memory can be handed straight to a renderer, but not written:
            ``to_dict()`` refuses an ``object:`` source, because a reference into this process's memory would
            be unreadable everywhere else. Give a builder a path or a URL to get a figure that can be stored.
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

    def _register_artist(
        self, glyph: Any, mappable: Any, label: Optional[str] = None
    ) -> Any:
        """Register a rendered glyph and its mappable so a colorbar or legend can be keyed to it.

        The drawing half of what :meth:`_add_layer` used to do in one step. A drawer calls this; the
        description was already written by :meth:`_draw` before the drawer ran, which is what keeps the two
        structures from disagreeing.

        Args:
            glyph: The cleopatra glyph instance that was drawn on :attr:`ax`, or ``None`` for an artist
                added to the axes directly (a globe's land fill, an artist the caller built).
            mappable: The matplotlib mappable/artist the glyph produced (e.g. ``glyph.im``).
            label: Optional default colorbar label for this layer (e.g. the units resolved by
                :func:`~digitalearth.base.autostyle.auto_style`); used only when :meth:`colorbar` is
                called without one.

        Returns:
            The ``mappable`` (so callers can chain or attach a colorbar).
        """
        self.layers.append((glyph, mappable))
        self._layer_labels.append(label)
        return mappable

    def _unregister_artist(self, drawn: DrawnLayer) -> None:
        """Drop a removed layer's ``(glyph, mappable)`` pair and its default label.

        Matched by identity rather than by position: :attr:`layers` holds only the layers that registered a
        mappable, so a layer's index there is not its index in the description and cannot be derived from
        one.

        Args:
            drawn: What the renderer recorded for the layer being removed. One that registered no mappable
                — a graticule, a basemap, a text label — is not in :attr:`layers` and is left alone.
        """
        if drawn.artist is None:
            # A layer that produced no artist registered none, and matching on a pair of `None`s would
            # unregister whichever layer happened to be first.
            return
        for index, (glyph, mappable) in enumerate(self.layers):
            if glyph is drawn.glyph and mappable is drawn.artist:
                del self.layers[index]
                del self._layer_labels[index]
                return

    def _add_layer(self, glyph: Any, mappable: Any, label: Optional[str] = None) -> Any:
        """Register an artist the caller built themselves, and describe it as a custom layer.

        Not the builders' path: an artist somebody built by hand has no source to reference and no
        symbology to read back, so it is recorded as ``custom:matplotlib`` and described by id, kind and
        label alone. A figure can name it, hide it, reorder it and take it off — but never *rebuild* it,
        which is what :mod:`digitalearth.base.custom` says a reader of such a layer is looking at. So the
        artist itself is held on the scene under the layer's id, and the tier's own custom drawer puts it
        back from there; a figure read back somewhere else describes the layer and skips it.

        Drawing straight onto :attr:`ax` is still the escape hatch, and still invisible to the figure.

        Args:
            glyph: The cleopatra glyph instance that was drawn on :attr:`ax`, or ``None``.
            mappable: The matplotlib mappable/artist to register.
            label: Optional default colorbar label for this layer.

        Returns:
            The ``mappable`` (so callers can chain or attach a colorbar).
        """
        return self._draw(
            LayerRecord(
                custom_kind(ENGINE),
                symbology=Symbology(props={"via": "custom"}),
                held=(glyph, mappable, label),
            )
        )

    def _reset_layers(self) -> None:
        """Forget every registered layer — its default label and its description — e.g. between frames.

        The description is reset with the drawing, because an animation redraws every layer on a cleared
        axes: leaving the tree alone would have frame 50 describe 50 copies of one raster. The id counter
        goes back with it, so each frame mints the same ids and the in-memory source each one registered is
        replaced rather than accumulated. Whatever the dropped layers referenced is forgotten too, so a long
        run does not hold every frame's dataset alive through the process-wide object table.
        """
        # Every source goes and both tables are replaced below, so each object is forgotten directly rather
        # than popped one layer at a time — which is what made the loop need a copy of the table it emptied.
        for ref in self._sources.values():
            forget_object(ref.uri)
        self.layers = []
        self._layer_labels = []
        self._layer_tree = LayerTree()
        self._sources = {}
        self._layer_keys = {}
        self._layer_opts = {}
        self._held_objects = {}
        self._id_counter = 0
        self._issued_ids = set()
        # The artists themselves are gone with the cleared axes, so what the renderer holds is stale rather
        # than removable: it is dropped, not removed.
        self._renderer = Renderer(self)
        # The next frame's first layer draws onto an emptied axes, so it is a first render again and clears
        # rather than composes — which is what keeps a frame from composing over the frame before it.
        self._drew_on_axes = False

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
        **plot_kwargs: Any,
    ) -> DrawnLayer:
        """Plot ``glyph`` on the shared axes, register the produced mappable, and report what it drew.

        Consolidates the recipe every drawer shares — call ``glyph.plot(...)``, find the mappable it
        produced, then :meth:`_register_artist` — so it lives in one place. cleopatra glyphs expose their
        mappable in one of two ways, selected by ``artist``:

        - ``"im"`` (default): the mappable is ``glyph.im`` (``ArrayGlyph`` / ``MeshGlyph``).
        - ``"plot"``: ``glyph.plot()`` returns ``(fig, ax, artist)`` and the mappable is that third element
          (``Scatter`` / ``Polygon`` / ``Vector`` / ``KDE`` / ``Flow`` glyphs).

        It is also where a layer is told to draw **over** the ones already on the axes rather than in place
        of them (#313). A cleopatra glyph clears every glyph's artists off its axes by default, so a scene
        that added two field layers kept only the second: the axes held one image while :attr:`figure_spec`
        described two layers. Every render after this scene's first therefore passes ``compose=True``, which
        narrows that clear to the glyph's own artists. Only after the first: the opening render still
        replaces, so a scene handed an axes something else was drawn on supersedes it rather than stacking
        on it. A rebuilt layer composes too and still leaves one copy of itself, because
        :meth:`~digitalearth.static.renderer.Renderer.remove` has already taken the previous artists off.

        Args:
            glyph: An already-constructed cleopatra glyph bound to this Scene's ``ax``/``fig``.
            *plot_args: Positional arguments forwarded to ``glyph.plot`` (e.g. the data array for a mesh).
            artist: Which return convention to read the mappable from (``"im"`` or ``"plot"``).
            label: Optional default colorbar label recorded with the layer (see :meth:`_register_artist`);
                it is *not* forwarded to ``glyph.plot``.
            **plot_kwargs: Keyword arguments forwarded to ``glyph.plot`` (e.g. ``kind``, ``outline_only``).

        Returns:
            A :class:`~digitalearth.static.renderer.DrawnLayer` holding the mappable, the glyph that made it
            and the artists the layer owns — what the renderer records so the layer can later be hidden or
            taken off the axes.
        """
        plot_kwargs, deferred_alpha = prepare_plot_kwargs(glyph, plot_kwargs)
        # Only the glyphs that clear on render take the keyword; the rest (scatter, polygon, KDE, flow) add
        # to the axes without clearing it and declare no such parameter, so forwarding one would slip
        # through their `**kwargs` and reach matplotlib as an unknown artist property.
        if self._drew_on_axes and plot_takes(glyph, "compose"):
            plot_kwargs.setdefault("compose", True)
        result = glyph.plot(*plot_args, **plot_kwargs)
        # After the call, so a glyph that raised part-way does not leave the next layer composing over a
        # render that never happened.
        self._drew_on_axes = True
        mappable = glyph.im if artist == "im" else result[2]
        if deferred_alpha is not None and mappable is not None:
            mappable.set_alpha(deferred_alpha)
        self._register_artist(glyph, mappable, label)
        return DrawnLayer(
            artist=mappable,
            glyph=glyph,
            artists=() if mappable is None else (mappable,),
        )

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
        """Set the axes title.

        Figure-level decoration rather than a layer: it draws straight onto :attr:`ax` and is not described,
        so :attr:`figure_spec` neither carries it nor loses it when a layer is removed.

        Args:
            title: The text to place above the axes.
            **kwargs: Forwarded to ``Axes.set_title`` (``fontsize``, ``loc``, ``pad``, …).
        """
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

    def close(self) -> None:
        """Close the figure and let go of the in-memory data this scene registered.

        Two things are released, because a scene holds two kinds of memory. The **figure** is matplotlib's,
        and a long run of scenes that never closes one keeps every axes alive. The **object table** is this
        package's: it is process-global and holds strong references, so a session that builds maps keeps
        every dataset they drew until something says otherwise. This is the caller saying so, and it is what
        ``with`` calls on the way out.

        Closing twice is harmless, and a scene that drew nothing has nothing to forget.

        An ``object:`` source in a :attr:`figure_spec` captured from this scene cannot be opened afterwards
        — which is what "closed" means for a figure whose data lived only in this process. Save the data and
        reference it by path to keep such a figure readable.

        .. warning::
            This closes the **entire** ``self.fig``. The panels returned by
            :func:`~digitalearth.static.figure.grid` share **one** figure, so closing a single panel would
            close the figure for *all* panels — don't context-manage an individual ``grid`` panel; wrap the
            whole workflow or call :meth:`save` and close the figure yourself instead.

        Examples:
            - A scene that drew nothing closes quietly:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Scene
                >>> Scene().close()

                ```

        See Also:
            __exit__: calls this on the way out of a ``with`` block.
        """
        forget_namespace(self._objects_ns)
        self._layer_keys = {}
        self._layer_opts = {}
        self._held_objects = {}
        plt.close(self.fig)

    def __enter__(self) -> "Scene":
        """Enter the runtime context, returning the scene so ``with Scene(...) as s:`` binds it.

        Returns:
            This scene.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the scene on exit so a long run of them stays memory-bounded.

        The scene is closed whether or not the body raised; any exception propagates (``__exit__`` returns
        ``False``), so ``with`` never silences errors.

        Args:
            exc_type: The exception type, ignored — closing is unconditional, as it is for a file.
            exc: The exception, ignored.
            tb: The traceback, ignored.

        Returns:
            ``False`` — exceptions are not suppressed.
        """
        self.close()
        return False
