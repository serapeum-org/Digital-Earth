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
off the plotter, rebuilt and restyled ones drawn again — VTK has no cheap restyle, so a colour change is a
rebuild of that one layer — and shown/hidden ones toggled on their actor. Draw order is recorded but not
applied: VTK composites by depth, not by the order actors were added.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from digitalearth.base.custom import MissingObject, held_object
from digitalearth.base.spec import FigureSpec, LayerSpec

__all__ = ["Renderer3D", "drawer_for"]

logger = logging.getLogger(__name__)


def drawer_for(kind: str) -> Any:
    """Return the function that draws one kind of layer in this tier.

    Args:
        kind: The layer's kind, as its `LayerSpec` holds it.

    Returns:
        A callable ``draw(scene, data, layer) -> (mesh, actor) | None``, where `data` is the layer's source
        object (or `None` for a layer drawn from no source) and `layer` is its `LayerSpec`. `None` comes back
        for a layer that had nothing to draw and was skipped.

    Raises:
        KeyError: for a kind this tier does not draw, naming the kinds it does — the check a figure written
            for another backend runs into.

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
    try:
        return drawers[kind]
    except KeyError:
        raise KeyError(
            f"the 3-D tier does not draw {kind!r} layers; it draws {sorted(drawers)}"
        ) from None


def draw_custom(scene: Any, data: Any, layer: LayerSpec) -> Optional[Tuple[Any, Any]]:
    """Draw an object the caller built themselves and handed to the scene.

    Args:
        scene: The scene holding the object and the plotter.
        data: Unused — a custom layer has no source; the object is held by the scene (#293).
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
        for layer_id in (*change.rebuilt, *change.restyled):
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
        if plotter is not None:
            plotter.remove_actor(drawn[1], render=False)

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Show or hide what was drawn for a layer.

        Args:
            layer_id: The layer to toggle.
            visible: Whether it is drawn.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is not None:
            _set_visible(drawn[1], visible)


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
