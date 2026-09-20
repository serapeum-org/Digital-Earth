"""Draw a :class:`~digitalearth.base.spec.FigureSpec` as HoloViews elements (#300, the interactive seam).

The tier held its drawing and nothing else: `self.layers` is a list of built elements and `self._styles` a
dict keyed by `id(element)`. An element is what HoloViews composes, not something a figure can be written
to — and an identity-keyed style table cannot survive a redraw, which is why a dynamic layer's style had to
be filed twice and the table trimmed by hand.

This is the other half of that seam. A builder records what it drew — kind, source, visibility and
symbology as values — and the drawer here rebuilds the element from that description.

**This tier composes rather than mutates.** PyVista hands out a live plotter whose actors are mutated in
place; HoloViews elements are immutable values composed into an overlay on every `render()`. So, as in the
web tier, :meth:`Renderer.apply` reconciles the *record* of what is drawn and the next compose reflects it.
The observable contract is the same one the shared renderer conformance suite states, which is why all
three tiers can sign it.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from digitalearth.base.registry import band_of
from digitalearth.base.spec import FigureSpec, LayerSpec


@dataclass(frozen=True)
class DrawnLayer:
    """What one drawer produced: the HoloViews element a layer composes into.

    Every drawer answers in this shape so the compose step can take whatever came back without knowing
    which drawer made it — the equivalent of the web tier's source-and-layer pair, and of the 3-D tier's
    `(mesh, actor)`.

    Attributes:
        element: The HoloViews/GeoViews element, or a `DynamicMap` for a layer redrawn per frame.
        style: The options applied to it, as values. Kept beside the element because `.opts()` writes into
            HoloViews' global `Store` and returns nothing a caller can read back.
    """

    element: Any
    style: Mapping[str, Any] = None  # type: ignore[assignment]


#: The layer kinds this tier draws **from its description**. Names only, so what is drawable can be asked
#: without importing every builder module behind them; `drawer_for` resolves them and is checked against
#: this tuple, which keeps the two from drifting.
#:
#: This is a growing subset while the seam is opened (#300). A kind here is built by its drawer and must not
#: also be built by its builder; a kind not here is still built the old way and simply described.
DRAWN_KINDS: Tuple[str, ...] = ("graticule",)


def drawer_for(kind: str) -> Any:
    """Return the function that draws one kind of layer in this tier.

    Args:
        kind: A registered layer kind.

    Returns:
        A callable ``draw(interactive_map, data, layer) -> DrawnLayer | None``, where `data` is the layer's
        opened source (or `None` for a layer drawn from no source) and `layer` is its `LayerSpec`.

    Raises:
        KeyError: when this tier does not draw `kind`, naming the kinds it does; or when the drawer table
            and `DRAWN_KINDS` disagree, which is a defect in this module rather than in the caller.
    """
    if kind not in DRAWN_KINDS:
        raise KeyError(
            f"the interactive tier does not draw {kind!r} layers; it draws {sorted(DRAWN_KINDS)}"
        )
    # Imported here rather than at module level: every builder module imports the map, so a
    # module-level import would close a cycle.
    from digitalearth.interactive import projection

    drawers = {"graticule": projection.draw_graticule}
    if set(drawers) != set(DRAWN_KINDS):
        raise KeyError(
            f"the interactive tier's drawer table and DRAWN_KINDS disagree: "
            f"{sorted(set(drawers).symmetric_difference(DRAWN_KINDS))}"
        )
    return drawers[kind]


class Renderer:
    """Draws a figure's layers as HoloViews elements, and reconciles one figure with another.

    Attributes:
        drawn: Layer id to what was drawn for it, in draw order.
    """

    def __init__(self, interactive_map: Any) -> None:
        """Bind the renderer to the map it draws for.

        Args:
            interactive_map: The `InteractiveMap` this renderer serves. Held rather than passed per call,
                because a drawer reads the map's own helpers — the display CRS, the styling recorder, the
                auto-colormap resolution.
        """
        self._map = interactive_map
        self._drawn: dict = {}

    @property
    def drawn(self) -> Mapping[str, DrawnLayer]:
        """What has been drawn, by layer id.

        Returns:
            Layer id to the `DrawnLayer` its drawer produced, in the order the layers were drawn.
        """
        return dict(self._drawn)

    def draw_layer(self, figure: FigureSpec, layer_id: str) -> Optional[DrawnLayer]:
        """Draw one of a figure's layers and record what it produced.

        Args:
            figure: The figure holding the layer and its source.
            layer_id: Which layer to draw.

        Returns:
            What the drawer produced, or `None` for a layer it declined to draw.

        Raises:
            KeyError: when no layer has that id, or the tier has no drawer for its kind.
        """
        layer = figure.layers.get(layer_id)
        data = self._source_object(figure, layer)
        drawn: Optional[DrawnLayer] = drawer_for(layer.kind)(self._map, data, layer)
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
            The opened source, or `None` for a layer drawn from nothing, such as a graticule.
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
        """
        # `apply` is not atomic: it draws layer by layer, so a refusal on the third has already drawn the
        # first two. Rolling back only the caller's description would leave this record holding layers no
        # figure owns — the defect the shared conformance contract states, and which both other tiers hit.
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
        """
        change = before.diff(after)
        for layer_id in change.removed:
            self.remove(layer_id)
        # A rebuilt layer is one whose *data* changed — its kind, source or slice — and every one of those
        # means a new element. Only a restyle is worth asking about, because `diff` groups a change of
        # `label` with a change of colour and one of those never reaches HoloViews.
        for layer_id in change.rebuilt:
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.restyled:
            if not self._reaches_holoviews(before, after, layer_id):
                continue
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.added:
            self.draw_layer(after, layer_id)

    @staticmethod
    def _reaches_holoviews(
        before: FigureSpec, after: FigureSpec, layer_id: str
    ) -> bool:
        """Whether a restyle changes anything HoloViews draws.

        Args:
            before: The figure drawn now.
            after: The figure to draw.
            layer_id: The layer whose restyle is in question.

        Returns:
            `True` when the layer's symbology, filter or group differs — the parts a drawer reads — and for
            a layer either figure does not hold. `False` for a change to `label` alone.
        """
        try:
            was = before.layers.get(layer_id)
            now = after.layers.get(layer_id)
        except KeyError:
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

    def band_for(self, layer: LayerSpec) -> str:
        """Return the draw-order band a layer belongs to.

        Args:
            layer: The layer being placed.

        Returns:
            The layer's own band when it declares one — what a caller's `custom:holoviews` object needs,
            since the engine name says nothing about what it draws — else its kind's band.
        """
        return layer.band or band_of(layer.kind)
