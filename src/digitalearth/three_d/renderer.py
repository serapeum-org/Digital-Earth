"""The renderer — what turns a `FigureSpec` into meshes and actors on one PyVista plotter.

This is the seam. Before it, a 3-D builder computed a mesh and handed it straight to the plotter, and what the
layer *was* survived only as a `(mesh, actor)` pair in a list: no id, no kind, no source, no style. A scene could
be drawn, and then only looked at.

Now a builder describes what it wants drawn — a :class:`~digitalearth.base.spec.LayerSpec` with a kind, a source
and the engine keywords it was given — and this renderer draws it. One consequence is worth the change on its
own: the description can be saved, read back and drawn again, so
`FigureSpec.from_dict(scene.figure_spec.to_dict())` renders the same picture. The others follow from having ids:
a layer can be removed, hidden or re-described without rebuilding the scene around it.

**Changes go through the diff.** When the scene's figure changes, the renderer is handed the old and the new one
and applies :meth:`~digitalearth.base.spec.FigureSpec.diff` (#289): added layers are drawn, removed ones taken
off the plotter, rebuilt ones drawn again, and shown/hidden ones toggled on their actor. A restyle is drawn
again **only when it reaches the engine** — VTK has no cheap restyle, so a colour change is a rebuild of that
one layer, while a change of `label` alone is not (see :meth:`Renderer3D._reaches_pyvista`). Draw order is
recorded but not applied: VTK composites by depth, not by the order actors were added.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from digitalearth.base.custom import MissingObject, held_object
from digitalearth.base.spec import FigureSpec, LayerSpec
from digitalearth.three_d.bigdata import report_unreduced
from digitalearth.three_d.capabilities import CAPABILITIES

__all__ = ["Renderer3D", "drawer_for"]

logger = logging.getLogger(__name__)


#: The layer kinds this tier draws. Names only, so what is drawable can be asked — by a caller, and by the
#: tier's own capability test — without importing every builder module behind them. `drawer_for` resolves
#: them to functions and is checked against this tuple, which is what keeps the two from drifting: the hole
#: this closes was a kind with a drawer and no declaration (review M6).
DRAWN_KINDS: Tuple[str, ...] = (
    "terrain",
    "point_cloud",
    "volume",
    "isosurface",
    "vectors",
    "extrusion",
    "raster",
    "coastlines",
    "custom:pyvista",
)


def drawer_for(kind: str) -> Any:
    """Return the function that draws one kind of layer in this tier.

    Args:
        kind: The layer's kind, as its `LayerSpec` holds it.

    Returns:
        A callable ``draw(scene, data, layer) -> (mesh, actor) | None``, where `data` is the layer's source
        object (or `None` for a layer drawn from no source) and `layer` is its `LayerSpec`. `None` comes back
        for a layer that had nothing to draw and was skipped.

    Raises:
        KeyError: for a kind this tier does not draw, naming the kinds it does and, when the tier declared
            one, the reason it does not draw this one — the check a figure written for another backend runs
            into.

    Examples:
        - Every kind the tier declares has a drawer:
            ```python
            >>> from digitalearth.three_d.capabilities import CAPABILITIES
            >>> from digitalearth.three_d.renderer import drawer_for
            >>> sorted(kind for kind in CAPABILITIES.kinds if drawer_for(kind) is None)
            []

            ```
        - A kind from another tier is refused by name:
            ```python
            >>> from digitalearth.three_d.renderer import drawer_for
            >>> drawer_for("choropleth")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "the 3-D tier does not draw 'choropleth' layers; it draws [...]"

            ```
    """
    # Before the imports: the point of naming the kinds separately is that what is drawable can be asked
    # without loading every builder behind them (review L4).
    if kind not in DRAWN_KINDS:
        # The reason is the tier's own, read from the declaration rather than written again here (#294).
        # This tier declares no layer kind absent today — its `absent` names features, not kinds — so the
        # clause is usually empty; the rule is the same on all four tiers, and a kind it later decides
        # against explains itself for free.
        reason = CAPABILITIES.reason(kind)
        raise KeyError(
            f"the 3-D tier does not draw {kind!r} layers"
            + (f" — {reason}" if reason else "")
            + f"; it draws {sorted(DRAWN_KINDS)}"
        )
    # Imported here rather than at module level: every builder module imports the scene, so a module-level
    # import would close a cycle, and a scene that draws nothing should not pay for loading all of them.
    from digitalearth.three_d import globe, point_cloud, terrain, vector, volume

    drawers = {
        "terrain": terrain.draw_terrain,
        "point_cloud": point_cloud.draw_point_cloud,
        "volume": volume.draw_volume,
        "isosurface": volume.draw_isosurface,
        "vectors": vector.draw_vectors,
        "extrusion": vector.draw_extruded_polygons,
        "raster": globe.draw_globe,
        "coastlines": globe.draw_coastlines,
        "custom:pyvista": draw_custom,
    }
    # The two lists are one list said twice, and drift either way is a defect: a kind in `drawers` and not
    # in `DRAWN_KINDS` would be refused with a message that is false, and one in `DRAWN_KINDS` with no drawer
    # would raise a bare `KeyError` where the message belongs (review L3).
    if set(drawers) != set(DRAWN_KINDS):
        raise KeyError(
            f"the 3-D tier's drawer table and DRAWN_KINDS disagree: "
            f"{sorted(set(drawers).symmetric_difference(DRAWN_KINDS))}"
        )
    return drawers[kind]


def draw_custom(scene: Any, _data: Any, layer: LayerSpec) -> Optional[Tuple[Any, Any]]:
    """Draw an object the caller built themselves and handed to the scene.

    Args:
        scene: The scene holding the object and the plotter.
        _data: The source slot every drawer takes, unread here — a custom layer has no source, and the
            object is held by the scene (#293).
        layer: The layer to draw. Its `volume` property says whether it is ray-cast.

    Returns:
        The ``(object, actor)`` pair, or `None` when the scene no longer holds the object and is not `strict`.

    Raises:
        OffLimbError: when the object is missing and the scene is `strict`.
    """
    from digitalearth.three_d.layer import drawing_props

    props = drawing_props(layer.symbology.props)
    try:
        obj = held_object(
            layer.id,
            layer.kind,
            scene.held_objects,
            engine="pyvista",
            backend="3d",
        )
    except MissingObject as error:
        scene._skip_empty(layer.id, str(error))  # raises under strict
        return None
    volume = bool(props.pop("volume", False))
    add = scene.plotter.add_volume if volume else scene.plotter.add_mesh
    return obj, add(obj, **props)


class Renderer3D:
    """Draws a scene's `FigureSpec` onto its plotter, and keeps what it drew addressable by layer id.

    Attributes:
        scene: The scene whose plotter is drawn on. The renderer reads its display CRS and its `strict` policy
            through it, and never holds a plotter of its own — a scene has exactly one.

    Examples:
        - A scene's renderer holds what it drew, keyed by layer id:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import get_source
            >>> from digitalearth.three_d import Scene3D
            >>> scene = Scene3D(off_screen=True)
            >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
            >>> sorted(scene.renderer.drawn)
            ['terrain-1']
            >>> scene.close()

            ```
    """

    def __init__(self, scene: Any) -> None:
        """Bind a renderer to the scene whose plotter it draws on.

        Args:
            scene: The `Scene3DBase` (or subclass) that owns the plotter and the figure.
        """
        self.scene = scene
        self._drawn: Dict[str, Tuple[Any, Any]] = {}

    @property
    def drawn(self) -> Mapping[str, Tuple[Any, Any]]:
        """The `(mesh, actor)` pair drawn for each layer, keyed by layer id.

        Returns:
            A read-only view. A layer that was skipped — an empty raster, a missing custom object — has no
            entry, which is how a caller tells "drawn" from "described".
        """
        return dict(self._drawn)

    def draw_layer(self, figure: FigureSpec, layer_id: str) -> Any:
        """Draw one layer of a figure and keep what it produced.

        Args:
            figure: The figure the layer belongs to, which also holds its source.
            layer_id: The layer to draw.

        Returns:
            The PyVista actor, or `None` when the layer had nothing to draw and was skipped.

        Raises:
            KeyError: when the tier has no drawer for the layer's kind.
            OffLimbError: when a layer cannot be drawn and the scene is `strict`.
        """
        layer = figure.layers.get(layer_id)
        drawn = drawer_for(layer.kind)(
            self.scene, self._source_object(figure, layer), layer
        )
        if drawn is None:
            return None
        # The kinds that *can* reduce did so in their drawer, where the mesh was built. This is the other half
        # (#207): a kind with no reduction route is drawn whole, and saying so here — once, for every kind —
        # is what stops "too big" being a hung notebook instead of a line in the log.
        report_unreduced(layer.kind, drawn[0], self.scene.big_data_threshold)
        self._drawn[layer_id] = drawn
        if not figure.layers.is_visible(layer_id):
            _set_visible(drawn[1], False)
        return drawn[1]

    @staticmethod
    def _source_object(figure: FigureSpec, layer: LayerSpec) -> Any:
        """Return the object a layer draws, or `None` when it draws from no source.

        Args:
            figure: The figure holding the sources.
            layer: The layer being drawn.

        Returns:
            The opened source object — a pyramids dataset, a `Source`, an array — or `None`.
        """
        if layer.source_id is None:
            return None
        return figure.sources[layer.source_id].open()

    def apply(self, before: FigureSpec, after: FigureSpec) -> None:
        """Bring the plotter from the figure it was drawn from to another one.

        Args:
            before: The figure the plotter currently shows.
            after: The figure it should show.

        Raises:
            KeyError: when the tier has no drawer for a layer's kind.
            OffLimbError: when a layer cannot be drawn and the scene is `strict`.

        Note:
            `order` is read and deliberately not acted on: VTK composites its actors by depth, so the order
            they were added in does not decide what is in front. The figure still records it, because the
            2-D tiers draw in that order and one description serves all four.
        """
        change = before.diff(after)
        for layer_id in change.removed:
            self.remove(layer_id)
        # A rebuilt layer is one whose *data* changed — its kind, source, slice, or the reference behind its
        # source id — and every one of those means a new mesh. Only a restyle is worth asking about, because
        # `diff` groups a change of `label` with a change of colormap and one of those never reaches VTK.
        # The guard ran over both, so a layer that pointed at new data kept its old mesh while `figure_spec`
        # advertised the new one (review H1).
        for layer_id in change.rebuilt:
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.restyled:
            if not self._reaches_pyvista(before, after, layer_id):
                continue
            # VTK has no cheap restyle: a colormap or a class break is baked into the mesh's scalars and the
            # actor's mapper, so the layer is drawn again from its description.
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.added:
            self.draw_layer(after, layer_id)
        for layer_id in change.shown:
            self.set_visible(layer_id, True)
        for layer_id in change.hidden:
            self.set_visible(layer_id, False)

    @staticmethod
    def _reaches_pyvista(before: FigureSpec, after: FigureSpec, layer_id: str) -> bool:
        """Whether a restyle changes anything PyVista draws.

        `diff` groups a change to `label` — what a layer switcher calls the layer — with a change to its
        colormap, because both are "the description changed". Only one of them reaches the engine, and
        answering both with a rebuild meant renaming a layer re-opened its source, re-ran the reprojection
        and built the whole mesh again (review M8).

        Args:
            before: The figure the plotter shows.
            after: The figure it should show.
            layer_id: The layer whose restyle is being judged.

        Returns:
            `True` when the layer's symbology, filter or group differs — the parts a drawer reads — and for
            a layer either figure does not hold, which is a question about the layer rather than about its
            style. `False` for a change to `label` alone.
        """
        try:
            was = before.layers.get(layer_id)
            now = after.layers.get(layer_id)
        except KeyError:
            # `restyled` only ever names ids both figures hold, so neither lookup raises on the path that
            # calls this. Guarding both rather than the first is what makes that true of a direct call too
            # (review L13) — and the answer for a layer that is not in both is to draw it.
            return True
        return (was.symbology, was.filter, was.group) != (
            now.symbology,
            now.filter,
            now.group,
        )

    def remove(self, layer_id: str) -> None:
        """Take a layer off the plotter and forget what was drawn for it.

        Args:
            layer_id: The layer to remove. One that was never drawn — skipped, or described but not yet
                rendered — is ignored, so removing a layer twice is harmless.
        """
        drawn = self._drawn.pop(layer_id, None)
        if drawn is None:
            return
        # Only through the plotter the scene already has: asking for `scene.plotter` here would build one for
        # a scene that never drew anything, which is what #290 made lazy.
        plotter = getattr(self.scene, "_plotter", None)
        if (
            plotter is not None
        ):  # pragma: no branch - a drawn layer means a plotter was built
            plotter.remove_actor(drawn[1], render=False)

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Show or hide what was drawn for a layer.

        Args:
            layer_id: The layer to toggle. An id nothing was drawn for is ignored, which is how the other
                three tiers answer one too — note its neighbour :meth:`is_visible` **raises** for the same
                id, because there is no visibility to report where there is nothing to toggle.
            visible: Whether it is drawn.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is not None:
            _set_visible(drawn[1], visible)

    def is_visible(self, layer_id: str) -> bool:
        """Whether the plotter is currently drawing the actor this renderer holds for a layer.

        The read-back of :meth:`set_visible`. Every tier's renderer answers this, in its own terms, so the
        question "is this layer drawn hidden?" can be asked of any of them — which is what the shared
        renderer conformance suite does (review M4).

        Args:
            layer_id: The layer to ask about.

        Returns:
            `True` when the actor is visible.

        Raises:
            KeyError: when nothing was drawn for `layer_id`, naming it. A layer the plotter does not hold
                has no visibility to report, and :attr:`drawn` is what says which those are.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            raise KeyError(
                f"nothing is drawn for layer {layer_id!r}, so it has no visibility to report; the 3-D "
                f"tier holds {sorted(self._drawn)}"
            )
        return _is_visible(drawn[1])


def _set_visible(actor: Any, visible: bool) -> None:
    """Set an actor's visibility, whichever PyVista version built it.

    Args:
        actor: The actor to toggle.
        visible: Whether it is drawn.
    """
    try:
        actor.visibility = bool(visible)
    # A volume actor, and older PyVista actors, expose only VTK's own setter.
    except AttributeError:  # pragma: no cover - depends on the installed PyVista
        actor.SetVisibility(bool(visible))


def _is_visible(actor: Any) -> bool:
    """Whether an actor is currently drawn, whichever PyVista version built it.

    Args:
        actor: The actor to ask.

    Returns:
        Its visibility.
    """
    try:
        return bool(actor.visibility)
    # The same pair of spellings :func:`_set_visible` writes through.
    except AttributeError:  # pragma: no cover - depends on the installed PyVista
        return bool(actor.GetVisibility())
