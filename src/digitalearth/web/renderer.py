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

**This tier rebuilds rather than mutates.** PyVista hands out a live plotter whose actors are mutated in
place, so :mod:`digitalearth.three_d.renderer` reconciles a diff against it. MapLibre's widget is built
fresh on every ``render()``, so there is no long-lived engine object to reconcile against: :meth:`Renderer.
apply` reconciles the *description*, and the next build draws it. The observable contract is the same — a
refused figure must change neither what the map reports nor what it next draws — which is why the shared
renderer conformance suite fits both tiers without special-casing either.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from digitalearth.base.registry import band_of
from digitalearth.base.spec import FigureSpec, LayerSpec
from digitalearth.web.capabilities import CAPABILITIES


@dataclass(frozen=True)
class DrawnLayer:
    """What one drawer produced: a MapLibre source and the layer(s) reading it.

    Every drawer answers in this shape so the widget builder can add whatever came back without knowing
    which drawer made it — the equivalent of the `(mesh, actor)` pair the 3-D tier's drawers return.

    Attributes:
        source_id: The MapLibre source id, or `None` for a layer that adds no source of its own.
        source_spec: The source definition, or `None` alongside a `None` id.
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
    source_spec: Optional[dict]
    layer: Any
    extra_layers: Tuple[Any, ...] = field(default=())


#: The layer kinds this tier draws **from its description**. Names only, so what is drawable can be asked —
#: by a caller, and by the tier's own capability test — without importing every builder module behind them.
#: `drawer_for` resolves them to functions and is checked against this tuple, which keeps the two from
#: drifting.
#:
#: This is a growing subset while the seam is opened (#296). A kind here is built by its drawer and must not
#: also be queued by its builder; a kind not here is still replayed from the queue. Listing a kind before its
#: builder stops queuing would draw it twice, so the two move together, one kind per step.
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
    "heatmap",
    "clusters",
    "extrusion",
    "custom:maplibre",
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
        The props, as a plain dict.

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
    props = dict(layer.symbology.props)
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
        if drawn is not None:
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
        """Bring what the map draws from one figure to another, or leave it as it was.

        Reconciling is not atomic — the layers are drawn one after another — so a refusal partway has
        already drawn the ones before it. Everything drawn in the attempt is therefore rolled back before
        the refusal is re-raised: a figure this declines changes neither what the map reports nor what it
        next draws, which is the contract the shared renderer conformance suite states for all three tiers.

        Args:
            before: The figure the map currently draws.
            after: The figure it should draw.

        Raises:
            KeyError: when a layer names a kind this tier does not draw.
            ValueError: when a layer's description does not carry what its drawer needs.
            OffLimbError: when the map is `strict` and a layer cannot be placed.

        Note:
            Unlike the 3-D tier, nothing is mutated in place: MapLibre's widget is rebuilt on every render,
            so this reconciles the *record* of what is drawn and the next build reflects it. The failure
            modes it must still avoid are the same, which is why the order below matches that tier's —
            removals first, then data changes, then styling, then additions.
        """
        # `apply` is not atomic: it draws layer by layer, so a refusal on the third layer has already
        # drawn the first two. Rolling back only the caller's description would leave this record holding
        # layers no figure owns — the same defect the 3-D tier fixed in its own `_change`, found here by
        # the shared conformance contract (#305).
        held = dict(self._drawn)
        try:
            self._reconcile(before, after)
        except Exception:
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
        """Record a visibility change for a layer already drawn.

        Args:
            layer_id: The layer to show or hide.
            visible: Whether it should be drawn.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            return
        layer = getattr(drawn, "layer", None)
        if layer is None:
            return
        layout = dict(getattr(layer, "layout", None) or {})
        layout["visibility"] = "visible" if visible else "none"
        layer.layout = layout

    def band_for(self, layer: LayerSpec) -> str:
        """Return the draw-order band a layer belongs to.

        Args:
            layer: The layer being placed.

        Returns:
            The layer's own band when it declares one — what a caller's `custom:maplibre` object needs,
            since the engine name says nothing about what it draws — else its kind's band.
        """
        return layer.band or band_of(layer.kind)
