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
DRAWN_KINDS: Tuple[str, ...] = ("graticule",)


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
        KeyError: when this tier does not draw `kind`, naming the kinds it does; or when the drawer table
            and `DRAWN_KINDS` disagree, which is a defect in this module rather than in the caller.
    """
    # Before the imports: the point of naming the kinds separately is that what is drawable can be asked
    # without loading every builder behind them.
    if kind not in DRAWN_KINDS:
        raise KeyError(
            f"the web tier does not draw {kind!r} layers; it draws {sorted(DRAWN_KINDS)}"
        )
    # Imported here rather than at module level: every builder module imports the map, so a module-level
    # import would close a cycle, and a map that draws nothing should not pay for loading all of them.
    from digitalearth.web import decoration

    drawers = {
        "graticule": decoration.draw_graticule,
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


class Renderer:
    """Draws a figure's layers onto a MapLibre widget, and reconciles one figure with another.

    Attributes:
        drawn: Layer id to the MapLibre objects drawn for it, in draw order.
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
            Layer id to the `(source_id, source_spec, layer)` triple the drawer produced, in the order the
            layers were drawn.
        """
        return dict(self._drawn)

    def draw_layer(self, figure: FigureSpec, layer_id: str) -> Any:
        """Draw one of a figure's layers and record what it produced.

        Args:
            figure: The figure holding the layer and its source.
            layer_id: Which layer to draw.

        Returns:
            Whatever the drawer produced, or `None` for a layer it declined to draw.

        Raises:
            KeyError: when no layer has that id, or the tier has no drawer for its kind.
        """
        layer = figure.layers.get(layer_id)
        data = self._source_object(figure, layer)
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
        """Bring what the map draws from one figure to another.

        Args:
            before: The figure the map currently draws.
            after: The figure it should draw.

        Raises:
            KeyError: when a layer names a kind this tier does not draw.

        Note:
            Unlike the 3-D tier, nothing is mutated in place: MapLibre's widget is rebuilt on every render,
            so this reconciles the *record* of what is drawn and the next build reflects it. The failure
            modes it must still avoid are the same, which is why the order below matches that tier's —
            removals first, then data changes, then styling, then additions.
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
