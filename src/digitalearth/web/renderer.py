"""Draw a :class:`~digitalearth.base.spec.FigureSpec` onto a MapLibre widget (#296, the web seam).

The tier used to keep two parallel structures. Each builder wrote a *description* into the layer tree with
``_index_layer`` and, separately, queued a *drawing* — a closure holding the compiled MapLibre source and
layer it had just built. ``figure_spec`` read the first, ``_build_map_widget`` replayed the second, and
nothing held them to each other: a layer could be described one way and drawn another, and the only symptom
was a map that disagreed with what it said it showed.

This module is the other half. A builder now records what it drew — kind, source, visibility and
**symbology as values** — and the drawer here rebuilds the MapLibre source and layer from that description.
The description becomes the single source of truth, which is what makes `to_dict`/`from_dict` round-trip a
real figure rather than a label for one.

**Which figures can be written, and which only drawn.** `to_dict` refuses an `object:` source, so a map whose
builders were handed a GeoDataFrame or a dataset in memory describes itself and draws from that description,
but cannot be stored; give a builder a path or a URL for a figure that survives leaving the process. The
layers a description cannot rebuild at all — a basemap, a point cloud, terrain, a glTF model — are drawn
straight onto the widget and are named in `DRAWN_KINDS`' own note below.

**This tier rebuilds rather than mutates.** PyVista hands out a live plotter whose actors are mutated in
place, so :mod:`digitalearth.three_d.renderer` reconciles a diff against it. MapLibre's widget is built
fresh on every ``render()``, from the map's queue, so there is no long-lived engine object to reconcile
against: :meth:`Renderer.apply` reconciles the renderer's own *record* of what is drawn —
:attr:`Renderer.drawn` — and the contract it signs is the one the shared renderer conformance suite states
for all four tiers.

**On this tier, for this wave, `apply` is record-only.** It does not change what `WebMap` queues, which is
what the widget is built from, and it does not change what `figure_spec` reports, which is the map's own
layer tree. A layer it adds is in the record and never reaches the widget; a layer it removes leaves the
record and stays on the widget. Nothing in the tier calls it: every builder draws through
:meth:`Renderer.draw_layer`, one layer at a time. Wiring `apply` into the map's public state is Wave 7
(order 23); until then a caller who applies a figure has moved the record and nothing a viewer sees.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from digitalearth.base.registry import band_of
from digitalearth.base.spec import Encoding, FigureSpec, LayerSpec, Symbology
from digitalearth.base.spec._serial import thawed_value
from digitalearth.base.spec.style import portable_constants
from digitalearth.web.capabilities import CAPABILITIES


@dataclass(frozen=True)
class DrawnLayer:
    """What one drawer produced: a MapLibre source and the layer(s) reading it.

    Every drawer answers in this shape so the widget builder can add whatever came back without knowing
    which drawer made it — the equivalent of the `(mesh, actor)` pair the 3-D tier's drawers return.

    Attributes:
        source_id: The MapLibre source id, or `None` for a layer that adds no source of its own.
        source_spec: The source, in whatever shape the widget's `add_source` takes it: a MapLibre spec dict
            for an image, a text anchor or a graticule; the display-CRS GeoDataFrame itself (a pyramids
            `FeatureCollection` is one) for the vector kinds, which the widget serialises to GeoJSON; or a
            `maplibre` `GeoJSONSource` for a cluster, which needs its clustering options. `None` alongside a
            `None` id.
        layer: The layer that carries the layer's own id — the one a switcher toggles and `remove_layer`
            addresses.
        extra_layers: Further layers drawn from the same source and owned by the same description, such as
            a graticule's degree labels. They follow `layer` and are removed with it.

    Examples:
        - A graticule is one description drawing two MapLibre layers off one source — the lines, and the
          degree labels that follow them:
            ```python
            >>> from digitalearth.web import WebMap
            >>> drawn = WebMap().graticule(spacing=30)._renderer.drawn["Graticule"]
            >>> drawn.source_id, drawn.layer.id
            ('Graticule-src', 'Graticule')
            >>> [extra.id for extra in drawn.extra_layers]
            ['Graticule-label']

            ```
        - An annotation draws one layer and nothing follows it, so `extra_layers` is empty:
            ```python
            >>> from digitalearth.web import WebMap
            >>> m = WebMap().text(4.9, 52.4, "Amsterdam", name="amsterdam")
            >>> m._renderer.drawn["amsterdam"].extra_layers
            ()

            ```
    """

    source_id: Optional[str]
    source_spec: Any
    layer: Any
    extra_layers: Tuple[Any, ...] = field(default=())


#: The layer kinds this tier draws **from its description**. Names only, so what is drawable can be asked —
#: by a caller, and by the tier's own capability test — without importing every builder module behind them.
#: `drawer_for` resolves them to functions and is checked against this tuple, which keeps the two from
#: drifting.
#:
#: Every kind a builder records is here: the seam is closed (#296), and contours — the last kinds still
#: replayed from the queue — joined when they started recording under their own kind (review L3). A kind here
#: is built by its drawer and must not also be queued by its builder. The kinds the tier declares and does
#: not list — a basemap, a point cloud, terrain and a glTF model — are drawn straight onto the widget without
#: a description, because they are the map's style or deck.gl/terrain objects rather than MapLibre layers a
#: description can rebuild.
DRAWN_KINDS: Tuple[str, ...] = (
    "graticule",
    "text",
    "raster",
    "rgb",
    "points",
    "lines",
    "polygons",
    "choropleth",
    "labels",
    "contours",
    "filled_contours",
    "heatmap",
    "clusters",
    "extrusion",
    "custom:maplibre",
)


#: The MapLibre layer ids a kind's drawer adds beside the layer's own, as suffixes of that id — a graticule's
#: degree labels, a cluster's counts and its loose points. MapLibre keeps one layer per id and drops the rest
#: with only a console error, so these ids are as taken as the layer's own: the map reserves them when it
#: allocates the id (review M5), and the drawers read their names from here rather than spelling them again.
DERIVED_SUFFIXES: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "graticule": ("-label",),
        "clusters": ("-count", "-unclustered"),
    }
)


def derived_ids(kind: str, layer_id: str) -> Tuple[str, ...]:
    """Return the ids of the extra MapLibre layers a kind's drawer adds for one layer.

    Args:
        kind: The layer's kind.
        layer_id: The layer's own id.

    Returns:
        One id per extra layer, in the order the drawer adds them; empty for a kind that draws one layer.

    Examples:
        - A graticule labels its lines in a second layer, named after the first:
            ```python
            >>> from digitalearth.web.renderer import derived_ids
            >>> derived_ids("graticule", "grid")
            ('grid-label',)
            >>> derived_ids("points", "wells")
            ()

            ```
    """
    return tuple(f"{layer_id}{suffix}" for suffix in DERIVED_SUFFIXES.get(kind, ()))


#: Which declared visual channel each MapLibre paint property drives.
#:
#: The tier records a layer's style as MapLibre's own ``paint`` dict, because that is what
#: :func:`draw_vector` and its siblings rebuild the layer from and the rebuild must stay exact. That dict is
#: unreadable anywhere else, though: nothing but MapLibre knows that ``circle-radius`` is the marker size a
#: caller wrote as ``size=``. This table is that translation, and :func:`portable_encodings` applies it so a
#: layer says what it draws in :data:`~digitalearth.base.spec.encoding.CHANNELS` as well as in MapLibre's
#: spelling (#328).
#:
#: **What is deliberately absent.** ``fill-outline-color`` and ``text-halo-*`` drive no declared channel —
#: there is no stroke or halo channel — and ``text-size`` is the label size the tier renamed *away* from
#: ``size`` precisely because a glyph's size and a label's are not one thing. ``heatmap-radius`` is a kernel
#: width rather than a marker's size. Each would be over-claiming, so each is left to ``paint``, where it
#: already reads correctly.
PAINT_CHANNELS: Mapping[str, str] = MappingProxyType(
    {
        "circle-color": "color",
        "circle-opacity": "opacity",
        "circle-radius": "size",
        "fill-color": "color",
        "fill-extrusion-color": "color",
        "fill-extrusion-height": "height",
        "fill-extrusion-opacity": "opacity",
        "fill-opacity": "opacity",
        "heatmap-opacity": "opacity",
        "line-color": "color",
        "line-opacity": "opacity",
        "line-width": "width",
        "raster-opacity": "opacity",
        "text-color": "color",
    }
)


def portable_encodings(symbology: Symbology) -> Dict[str, Encoding]:
    """Return the declared channels a web layer's recorded paint says it drives.

    Additive by construction: it reads ``props["paint"]`` and writes nothing back, so every drawer keeps
    rebuilding its layer from the paint dict it was given and the engine-exact redraw is untouched. What it
    produces is the *second* reading of the same style — the one another tier, and `to_backend()`, can act on.

    Args:
        symbology: The layer's recorded style, as the builder wrote it.

    Returns:
        Channel name -> a constant :class:`~digitalearth.base.spec.encoding.Encoding`. A paint property
        MapLibre compiled into a data-driven expression — a choropleth's ``fill-color``, a cluster's stepped
        ``circle-radius`` — carries a list rather than a value and so contributes nothing: the class edges
        that expression encodes are published portably as ``last_breaks`` instead, and inventing a constant
        from it would describe the layer wrongly.

    Examples:
        - A point layer's radius and opacity read back as the channels a caller asked for:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.web.renderer import portable_encodings
            >>> paint = {"circle-radius": 7.0, "circle-opacity": 0.5, "circle-color": "#f00"}
            >>> lifted = portable_encodings(Symbology(props={"paint": paint}))
            >>> sorted(lifted), lifted["size"].resolve()
            (['color', 'opacity', 'size'], 7.0)

            ```
        - A classified fill is an expression, so the colour channel is left unclaimed:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.web.renderer import portable_encodings
            >>> expression = ("step", ("get", "pop"), "#440154", 5.0, "#fde725")
            >>> sorted(portable_encodings(Symbology(props={"paint": {"fill-color": expression}})))
            []

            ```
    """
    paint = symbology.props.get("paint")
    if not isinstance(paint, dict):
        return {}
    return portable_constants(paint, PAINT_CHANNELS)


def _show(drawn: DrawnLayer, visible: bool) -> None:
    """Set `layout.visibility` on every MapLibre layer one drawing holds.

    Args:
        drawn: What a drawer produced — its own layer and the extra layers the same description owns.
        visible: Whether they are drawn.

    Note:
        A caller's own layer may be a plain MapLibre spec dict rather than a `Layer`, so both shapes are
        set; a callable ``apply(widget)`` wires its own layers, has no layout to reach, and is left alone.
        The layout is replaced rather than edited in place, so a recorded mapping is never written through.
    """
    value = "visible" if visible else "none"
    for layer in (drawn.layer, *drawn.extra_layers):
        if isinstance(layer, dict):
            layer["layout"] = {**(layer.get("layout") or {}), "visibility": value}
        elif layer is not None and hasattr(layer, "layout"):
            layer.layout = {**(layer.layout or {}), "visibility": value}


def _layout_of(layer: Any) -> Mapping[str, Any]:
    """Return the MapLibre `layout` one drawn layer carries, in whichever shape it was built.

    Args:
        layer: A `Layer`, a plain MapLibre spec dict, or a callable that wires its own layers.

    Returns:
        The layout, or an empty mapping for a layer that has none to read.
    """
    layout = (
        layer.get("layout")
        if isinstance(layer, dict)
        else getattr(layer, "layout", None)
    )
    return layout or {}


def _shown(drawn: DrawnLayer) -> bool:
    """Whether every MapLibre layer one drawing holds is currently drawn.

    Args:
        drawn: What a drawer produced — its own layer and the extra layers the same description owns.

    Returns:
        `False` as soon as one of them carries ``layout.visibility == "none"``. A layer with no layout to
        read — a callable ``apply(widget)`` wiring its own layers — cannot be hidden, so it is never
        reported hidden, which is the same carve-out :func:`_show` makes when it writes.
    """
    return all(
        _layout_of(layer).get("visibility") != "none"
        for layer in (drawn.layer, *drawn.extra_layers)
        if layer is not None
    )


def _custom_drawer() -> Any:
    """Return the drawer for a caller's own MapLibre layer.

    Resolved through a function so the drawer table stays a table of names: `draw_custom` lives beside the
    map that holds the objects, in :mod:`digitalearth.web.base`, rather than in a builder module.

    Returns:
        The drawer.
    """
    from digitalearth.web.base import draw_custom

    return draw_custom


def drawer_for(kind: str) -> Any:
    """Return the function that draws one kind of layer in this tier.

    Args:
        kind: A registered layer kind.

    Returns:
        A callable ``draw(web_map, data, layer) -> DrawnLayer | None``, where `data` is the layer's
        opened source (or `None` for a layer drawn from no source, such as a graticule) and `layer` is its
        `LayerSpec`. `None` comes back for a layer the drawer declined to draw — the same shape the 3-D
        tier's drawers answer in, so the two renderers stay readable against each other.

    Raises:
        KeyError: when this tier does not draw `kind`, naming the kinds it does and, when the tier
            declared one, the reason it does not draw this one; or when the drawer table and `DRAWN_KINDS`
            disagree, which is a defect in this module rather than in the caller.

    Examples:
        - Kinds that differ only in what they mean share the drawer that draws them, because on this tier
          they are one GeoJSON source and one typed layer:
            ```python
            >>> from digitalearth.web.renderer import drawer_for
            >>> drawer_for("points").__name__, drawer_for("polygons").__name__
            ('draw_vector', 'draw_vector')
            >>> drawer_for("raster").__name__
            'draw_field'

            ```
        - A kind this tier deliberately does not have is refused with the reason it declared, so the
          caller is told what to do instead rather than only that the kind is missing:
            ```python
            >>> from digitalearth.web.renderer import drawer_for
            >>> drawer_for("mesh")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "the web tier does not draw 'mesh' layers — a raster is drawn as an image ...

            ```
        - A kind the tier has simply not reached yet carries no reason, because it declared none:
            ```python
            >>> from digitalearth.web.renderer import drawer_for
            >>> drawer_for("terrain")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "the web tier does not draw 'terrain' layers; it draws [...]"

            ```
    """
    # Before the imports: the point of naming the kinds separately is that what is drawable can be asked
    # without loading every builder behind them.
    if kind not in DRAWN_KINDS:
        # The reason is the tier's own, read from the declaration rather than written again here: a kind
        # this tier decided against — a mesh, a u/v field — says why in `absent`, and a caller who reaches
        # the refusal needs that sentence more than the list of kinds (#294).
        reason = CAPABILITIES.reason(kind)
        raise KeyError(
            f"the web tier does not draw {kind!r} layers"
            + (f" — {reason}" if reason else "")
            + f"; it draws {sorted(DRAWN_KINDS)}"
        )
    # Imported here rather than at module level: every builder module imports the map, so a module-level
    # import would close a cycle, and a map that draws nothing should not pay for loading all of them.
    from digitalearth.web import bigdata, decoration, raster, threed, vector

    drawers = {
        "graticule": decoration.draw_graticule,
        "text": decoration.draw_text,
        "raster": raster.draw_field,
        "rgb": raster.draw_rgb_composite,
        "points": vector.draw_vector,
        "lines": vector.draw_vector,
        "polygons": vector.draw_vector,
        "choropleth": vector.draw_vector,
        "labels": vector.draw_vector,
        # A traced contour is a GeoJSON source and one line or fill layer, recorded with its paint like
        # the vector kinds above, so the same drawer rebuilds it.
        "contours": vector.draw_vector,
        "filled_contours": vector.draw_vector,
        "heatmap": bigdata.draw_heatmap,
        "clusters": bigdata.draw_clusters,
        "extrusion": threed.draw_extruded_polygons,
        "custom:maplibre": _custom_drawer(),
    }
    # The two lists are one list said twice, and drift either way is a defect: a kind in `drawers` and not
    # in `DRAWN_KINDS` would be refused with a message that is false, and one in `DRAWN_KINDS` with no
    # drawer would raise a bare `KeyError` where the message belongs.
    if set(drawers) != set(DRAWN_KINDS):
        raise KeyError(
            f"the web tier's drawer table and DRAWN_KINDS disagree: "
            f"{sorted(set(drawers).symmetric_difference(DRAWN_KINDS))}"
        )
    return drawers[kind]


def required_props(layer: LayerSpec, *names: str) -> dict:
    """Return a layer's symbology props, checking the ones its drawer needs are present.

    A `LayerSpec` can reach a drawer without them: built by hand, or loaded from a figure an older version
    wrote before this tier recorded what it draws. The dict lookup would then raise a bare `KeyError` naming
    a MapLibre key, far from the layer that is actually malformed.

    Args:
        layer: The layer being drawn.
        *names: The props its drawer reads.

    Returns:
        The props, as a plain dict of the shapes MapLibre's JSON takes: every sequence a list and every
        mapping a fresh plain dict, however nested. A description freezes its sequences to tuples so it can
        be compared and hashed; handed to MapLibre as they are, those tuples reached the drawn layers, and
        `WebMap.layers[i].paint["circle-color"]` read `('interpolate', ('linear',), ...)` where it had read
        a list (review L5). Thawed here, once, so no drawer can forget to — and the recorded mappings are
        copied rather than handed out, so a drawer never writes through to the description.

    Raises:
        ValueError: naming the layer, its kind and what is missing.

    Examples:
        - The props a drawer asked for come back as a plain dict it can read:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, Symbology
            >>> from digitalearth.web.renderer import required_props
            >>> wells = LayerSpec(
            ...     "wells",
            ...     "points",
            ...     symbology=Symbology(
            ...         props={"maplibre_type": "circle", "paint": {"circle-color": "#cc4444"}}
            ...     ),
            ... )
            >>> required_props(wells, "maplibre_type", "paint")["paint"]
            {'circle-color': '#cc4444'}

            ```
        - A description carrying none of them is refused by layer, not by MapLibre key:
            ```python
            >>> from digitalearth.base.spec import LayerSpec
            >>> from digitalearth.web.renderer import required_props
            >>> try:
            ...     required_props(LayerSpec("wells", "points"), "paint")
            ... except ValueError as error:
            ...     print(str(error).split(".")[0])
            layer 'wells' (points) cannot be drawn by the web tier: its symbology records none of ['paint']

            ```
    """
    props: dict = thawed_value(dict(layer.symbology.props))
    missing = [name for name in names if name not in props]
    if missing:
        raise ValueError(
            f"layer {layer.id!r} ({layer.kind}) cannot be drawn by the web tier: its symbology records "
            f"none of {missing}. A layer is drawn from what its builder recorded, so a description built "
            f"by hand — or loaded from a figure written before this tier recorded its styling — has to "
            f"carry them."
        )
    return props


class Renderer:
    """Draws a figure's layers onto a MapLibre widget, and reconciles one figure with another.

    Attributes:
        drawn: Layer id to the MapLibre objects drawn for it, in draw order.

    Examples:
        - A map's renderer holds what it drew, keyed by the same ids
          :attr:`~digitalearth.web.base.WebMapBase.layer_ids` lists:
            ```python
            >>> from digitalearth.web import WebMap
            >>> m = WebMap().text(4.9, 52.4, "Amsterdam", name="amsterdam")
            >>> sorted(m._renderer.drawn)
            ['amsterdam']
            >>> m._renderer.drawn["amsterdam"].source_id
            'amsterdam-src'

            ```
        - What it holds is in the order the layers were drawn, which is the order the widget adds them:
            ```python
            >>> from digitalearth.web import WebMap
            >>> m = WebMap().text(4.9, 52.4, "Amsterdam", name="ams").text(2.35, 48.86, "Paris", name="par")
            >>> list(m._renderer.drawn)
            ['ams', 'par']

            ```
    """

    def __init__(self, web_map: Any) -> None:
        """Bind the renderer to the map whose widget it draws on.

        Args:
            web_map: The `WebMap` this renderer serves. Held rather than passed per call, because a drawer
                reads the map's own helpers — the id allocator, the skip log, the lon/lat framing.
        """
        self._map = web_map
        self._drawn: dict = {}

    @property
    def drawn(self) -> Mapping[str, Any]:
        """What has been drawn, by layer id.

        Returns:
            Layer id to the `DrawnLayer` its drawer produced — the source, the layer, and whatever extra
            layers the same description owns — in the order the layers were drawn. A copy, so writing to it
            does not change what the map draws.
        """
        return dict(self._drawn)

    def draw_layer(
        self, figure: FigureSpec, layer_id: str, *, opened: Any = None
    ) -> Any:
        """Draw one of a figure's layers and record what it produced.

        Args:
            figure: The figure holding the layer and its source.
            layer_id: Which layer to draw.
            opened: The layer's source as the builder that recorded it already holds it — placed in the
                display CRS — so the first draw does not read and warp the same data a second time (review
                M8). The figure still records the caller's own data; a drawer places whatever it is given,
                and placing data already in the display CRS costs nothing, so both paths draw one image.
                `None` opens the recorded source, which is how every redraw from a figure reaches it.

        Returns:
            The `DrawnLayer` the drawer produced, or `None` for a layer it declined to draw — an
            unplaceable raster, a custom object this process does not hold. A declined layer is not
            recorded in :attr:`drawn`.

            A layer the figure describes as not drawn is drawn **hidden** rather than skipped, so it is in
            :attr:`drawn` and :meth:`set_visible` can bring it back. "Not drawn" is the tree's answer, not
            the layer's own flag: a layer switched on inside a group that is hidden is hidden too (review
            M4).

        Raises:
            KeyError: when no layer has that id, or the tier has no drawer for its kind.
            ValueError: when the layer's description does not carry the props its drawer reads (see
                :func:`required_props`).
            OffLimbError: when the map is `strict` and the drawer found nothing it could place; a map that
                is not `strict` gets `None` back and a warning instead.
        """
        layer = figure.layers.get(layer_id)
        data = self._source_object(figure, layer) if opened is None else opened
        drawn = drawer_for(layer.kind)(self._map, data, layer)
        if drawn is None:
            return None
        if not figure.layers.is_visible(layer_id):
            # Applied here rather than trusted to each drawer: the text, heatmap, cluster and extrusion
            # drawers built their layers visible whatever the description said, because their builders
            # never ask for a hidden one — but a figure read back, or reconciled by `apply`, can (M4).
            #
            # Asked of the tree rather than of `layer.visible`: the tree's answer is the layer's own flag
            # *and* its group not being hidden, and a layer hidden by its group — a switcher's whole
            # "Observations" row turned off — drew visible here while the static and 3-D tiers drew it
            # hidden. Three tiers, three answers to one question (review M4).
            _show(drawn, False)
        self._drawn[layer_id] = drawn
        return drawn

    @staticmethod
    def _source_object(figure: FigureSpec, layer: LayerSpec) -> Any:
        """Return the object a layer draws, or `None` when it draws from no source.

        Args:
            figure: The figure holding the sources.
            layer: The layer being drawn.

        Returns:
            The opened source — a pyramids dataset, a GeoDataFrame, an array — or `None` for a layer drawn
            from nothing, such as a graticule.
        """
        if layer.source_id is None:
            return None
        return figure.sources[layer.source_id].open()

    def apply(self, before: FigureSpec, after: FigureSpec) -> None:
        """Bring the renderer's record of what is drawn from one figure to another, or leave it as it was.

        **Record-only on this tier, for this wave.** It reconciles :attr:`drawn` and nothing else: the queue
        the widget is built from and the figure `figure_spec` reports are the map's own, and neither follows.
        Nothing in the tier calls it yet; wiring it into the map is Wave 7 (order 23).

        Reconciling is not atomic — the layers are drawn one after another — so anything that stops it
        partway has already drawn the ones before it. Everything drawn in the attempt is therefore rolled
        back before it is re-raised: a figure this declines leaves the record as it found it, which is the
        contract the shared renderer conformance suite states for all four tiers. "Anything" is meant —
        the rollback catches ``BaseException``, so a ``KeyboardInterrupt`` mid-reconcile puts the record
        back exactly as an error does (review N2).

        Args:
            before: The figure the record currently holds.
            after: The figure it should hold.

        Raises:
            KeyError: when a layer names a kind this tier does not draw. The record is rolled back first, as
                it is for the two below.
            ValueError: when a layer's description does not carry what its drawer needs.
            OffLimbError: when the map is `strict` and a layer cannot be placed.

        Note:
            Unlike the 3-D tier, nothing is mutated in place: MapLibre's widget is rebuilt on every render,
            from the queue rather than from this record. The failure modes the record must still avoid are
            the 3-D tier's, which is why the order below matches that tier's — removals first, then data
            changes, then styling, then additions.
        """
        # `apply` is not atomic: it draws layer by layer, so a refusal on the third layer has already
        # drawn the first two. Rolling back only the caller's description would leave this record holding
        # layers no figure owns — the same defect the 3-D tier fixed in its own `_change`, found here by
        # the shared conformance contract (#305).
        held = dict(self._drawn)
        try:
            self._reconcile(before, after)
        except BaseException:
            # `BaseException`, the same class the static tier catches: what the record must survive is a
            # change stopping part-way, and a `KeyboardInterrupt` stops it exactly as an error does. Three
            # tiers signing one contract with two answers to "what is a refusal" is the drift (review N2).
            self._drawn = held
            raise

    def _reconcile(self, before: FigureSpec, after: FigureSpec) -> None:
        """Draw the difference between two figures, layer by layer.

        Args:
            before: The figure the map currently draws.
            after: The figure it should draw.

        Raises:
            KeyError: when a layer names a kind this tier does not draw.
            ValueError: when a layer's description does not carry what its drawer needs.
            OffLimbError: when the map is `strict` and a layer cannot be placed. Nothing is rolled back
                here — :meth:`apply` is what holds the record together across a refusal.
        """
        change = before.diff(after)
        for layer_id in change.removed:
            self.remove(layer_id)
        # A rebuilt layer is one whose *data* changed — its kind, source, slice, or the reference behind its
        # source id — and every one of those means a new MapLibre source. Only a restyle is worth asking
        # about, because `diff` groups a change of `label` with a change of colour and one of those never
        # reaches the engine.
        for layer_id in change.rebuilt:
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.restyled:
            if not self._reaches_maplibre(before, after, layer_id):
                continue
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.added:
            self.draw_layer(after, layer_id)
        for layer_id in change.shown:
            self.set_visible(layer_id, True)
        for layer_id in change.hidden:
            self.set_visible(layer_id, False)

    @staticmethod
    def _reaches_maplibre(before: FigureSpec, after: FigureSpec, layer_id: str) -> bool:
        """Whether a restyle changes anything MapLibre draws.

        `diff` groups a change to `label` — what a layer switcher calls the layer — with a change to its
        colour, because both are "the description changed". Only one of them reaches the engine.

        Args:
            before: The figure drawn now.
            after: The figure to draw.
            layer_id: The layer whose restyle is in question.

        Returns:
            `True` when the layer's symbology, filter or group differs — the parts a drawer reads — and for
            a layer either figure does not hold, which is a question about the layer rather than its style.
            `False` for a change to `label` alone.
        """
        try:
            was = before.layers.get(layer_id)
            now = after.layers.get(layer_id)
        except KeyError:
            # `restyled` only ever names ids both figures hold, so neither lookup raises on the path that
            # calls this. Guarding both rather than the first is what makes that true of a direct call too,
            # and the answer for a layer that is not in both is to draw it.
            return True
        return (was.symbology, was.filter, was.group) != (
            now.symbology,
            now.filter,
            now.group,
        )

    def remove(self, layer_id: str) -> None:
        """Forget what was drawn for a layer.

        Args:
            layer_id: The layer to remove. An id nothing was drawn for is ignored, so a caller can remove
                a layer the tier declined to draw.
        """
        self._drawn.pop(layer_id, None)

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Record a visibility change for a layer already drawn — every MapLibre layer it drew.

        A description can draw more than one MapLibre layer: a graticule's lines and its degree labels, a
        cluster's bubbles, counts and loose points. They are one layer to a viewer, so they are shown and
        hidden together. Setting only the first left a hidden graticule's labels floating over the map
        (review M4).

        Args:
            layer_id: The layer to show or hide. One nothing was drawn for is ignored.
            visible: Whether it should be drawn.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is not None:
            _show(drawn, visible)

    def is_visible(self, layer_id: str) -> bool:
        """Whether every MapLibre layer drawn for a layer is currently drawn.

        The read-back of :meth:`set_visible`. Every tier's renderer answers this, in its own terms, so the
        question "is this layer drawn hidden?" can be asked of any of them — which is what the shared
        renderer conformance suite does (review M4).

        Args:
            layer_id: The layer to ask about.

        Returns:
            `True` when none of the MapLibre layers the description owns is switched off. A graticule's
            lines and its degree labels are one layer to a viewer, so both have to be on for the answer to
            be `True`.

        Raises:
            KeyError: when nothing was drawn for `layer_id`, naming it. A layer the widget does not hold
                has no visibility to report, and :attr:`drawn` is what says which those are.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            raise KeyError(
                f"nothing is drawn for layer {layer_id!r}, so it has no visibility to report; the web "
                f"tier holds {sorted(self._drawn)}"
            )
        return _shown(drawn)

    def band_for(self, layer: LayerSpec) -> str:
        """Return the draw-order band a layer belongs to.

        Args:
            layer: The layer being placed.

        Returns:
            The layer's own band when it declares one — what a caller's `custom:maplibre` object needs,
            since the engine name says nothing about what it draws — else its kind's band.
        """
        return layer.band or band_of(layer.kind)
