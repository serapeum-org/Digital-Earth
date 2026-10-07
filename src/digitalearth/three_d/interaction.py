"""InteractionMixin — live widgets and picking on a 3-D scene (TD-9, TD-10).

These are the tier's **interactive** surface, and they sit deliberately *below the seam*: a clip plane, a
slice, a threshold or a value slider is a live VTK callback bound to the render window, and a pick is a gesture
answered against what is currently drawn. None of it is part of a :class:`~digitalearth.base.spec.FigureSpec` —
a figure records the *result* a caller reads back (a clip plane's origin and normal, a slider's value, a picked
layer id), never the widget — which is why these are methods that return a **handle** rather than builders that
record a layer. The shared plan is explicit that hover/tap/draw handlers and feature picking stay below the
seam, so nothing here touches the layer tree, the capabilities declaration or the cross-tier contract.

The widgets wrap :mod:`pyvista`'s own (``add_mesh_clip_plane``, ``add_mesh_slice``, ``add_mesh_threshold``,
``add_mesh_isovalue``, ``add_mesh_clip_box``, ``add_slider_widget``); the pickers wrap its ``enable_*_picking``.
A widget added over a layer hides that layer's plain actor first, so the interactive copy is the only thing
drawn rather than a second mesh on top of the first.
"""

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: The picking gestures :meth:`InteractionMixin.enable_picking` dispatches to, by the PyVista enabler each uses.
_PICKERS: dict[str, str] = {
    "point": "enable_point_picking",
    "cell": "enable_cell_picking",
    "mesh": "enable_mesh_picking",
}


class InteractionMixin(_MixinBase):
    """Adds live widgets (clip / slice / threshold / isovalue / slider) and picking to a :class:`Scene3D`.

    A capability mixin of :class:`~digitalearth.three_d.scene3d.Scene3D`: it is only ever composed into that
    scene class, never instantiated on its own, and reaches the wrapped ``pyvista.Plotter`` and the layer
    registry through ``self``. Every method returns the PyVista handle it created, so a caller can tune or
    remove the widget; the scene's figure is unchanged, because a live widget is below the description seam.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the state these methods use.
    """

    def _widget_mesh(self, layer_id: str) -> Any:
        """Return a drawn layer's mesh and hide its plain actor, so a widget's copy is the only draw.

        Args:
            layer_id: The layer whose mesh the widget acts on.

        Returns:
            The PyVista mesh the layer drew.

        Raises:
            KeyError: if no layer has that id (named by the scene).
            ValueError: if the layer was described but drew nothing — there is no mesh to attach a widget to.
        """
        mesh = self.mesh_of(layer_id)
        if mesh is None:
            raise ValueError(
                f"layer {layer_id!r} has nothing drawn, so there is no mesh to add a widget to"
            )
        # The plain layer is already on the plotter; the widget adds its own (clipped/sliced) copy, so hide the
        # original or the two overdraw. The layer keeps its id and description — only its actor is hidden.
        self.set_visible(layer_id, False)
        return mesh

    def clip_plane(
        self, layer_id: str, *, normal: Any = (1.0, 0.0, 0.0), **kwargs: Any
    ) -> Any:
        """Add an interactive clip-plane widget over a layer's mesh and return its actor.

        Args:
            layer_id: The layer to clip.
            normal: The clipping plane's normal.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh_clip_plane`.

        Returns:
            The clipped-mesh actor PyVista produced.
        """
        return self.plotter.add_mesh_clip_plane(
            self._widget_mesh(layer_id), normal=normal, **kwargs
        )

    def slice_planes(self, layer_id: str, *, normal: Any = "x", **kwargs: Any) -> Any:
        """Add an interactive orthogonal-slice widget over a layer's mesh and return its actor.

        Args:
            layer_id: The layer to slice.
            normal: The slice plane's normal (an axis name or a vector).
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh_slice`.

        Returns:
            The slice actor PyVista produced.
        """
        return self.plotter.add_mesh_slice(
            self._widget_mesh(layer_id), normal=normal, **kwargs
        )

    def clip_box(self, layer_id: str, **kwargs: Any) -> Any:
        """Add an interactive clip-box widget over a layer's mesh and return its actor.

        Args:
            layer_id: The layer to clip.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh_clip_box`.

        Returns:
            The clipped-mesh actor PyVista produced.
        """
        return self.plotter.add_mesh_clip_box(self._widget_mesh(layer_id), **kwargs)

    def threshold(self, layer_id: str, **kwargs: Any) -> Any:
        """Add an interactive threshold widget over a layer's mesh and return its actor.

        Args:
            layer_id: The layer to threshold (its active scalar is thresholded).
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh_threshold`.

        Returns:
            The thresholded-mesh actor PyVista produced.
        """
        return self.plotter.add_mesh_threshold(self._widget_mesh(layer_id), **kwargs)

    def isovalue(self, layer_id: str, **kwargs: Any) -> Any:
        """Add an interactive isovalue (contour) widget over a layer's mesh and return its actor.

        Args:
            layer_id: The layer to contour (its active scalar is contoured).
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh_isovalue`.

        Returns:
            The contour actor PyVista produced.
        """
        return self.plotter.add_mesh_isovalue(self._widget_mesh(layer_id), **kwargs)

    def slider(
        self,
        callback: Callable[[float], Any],
        value_range: Any,
        *,
        value: float | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Add a value slider the render window drives, and return its widget.

        The slider carries a *value* — a clip height, an isovalue, a time step — not a layer, so the figure
        records only the value a caller reads back, never the widget.

        Args:
            callback: Called with the slider's current value as it moves.
            value_range: The ``(low, high)`` the slider spans.
            value: The starting value; ``None`` starts at ``low``.
            title: The slider's label.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_slider_widget`.

        Returns:
            The ``vtkSliderWidget`` PyVista produced.
        """
        low, high = (float(value_range[0]), float(value_range[1]))
        return self.plotter.add_slider_widget(
            callback, [low, high], value=value, title=title, **kwargs
        )

    def enable_picking(
        self,
        callback: Callable[..., Any] | None = None,
        *,
        mode: str = "point",
        **kwargs: Any,
    ) -> Any:
        """Turn on picking so a click reports what it hit, and return PyVista's result.

        Args:
            callback: Called with what was picked (a point, a cell, or a mesh, by ``mode``). ``None`` uses
                PyVista's default handler, which labels the pick on the scene.
            mode: ``"point"``, ``"cell"`` or ``"mesh"`` — which gesture to enable.
            **kwargs: Forwarded to the matching ``pyvista.Plotter.enable_*_picking``.

        Returns:
            Whatever the PyVista enabler returns.

        Raises:
            ValueError: if ``mode`` is not one of ``"point"``, ``"cell"`` or ``"mesh"``.
        """
        if mode not in _PICKERS:
            raise ValueError(
                f"enable_picking() got mode={mode!r}; choose one of {sorted(_PICKERS)}"
            )
        enabler = getattr(self.plotter, _PICKERS[mode])
        return enabler(callback, **kwargs)

    def disable_picking(self) -> None:
        """Turn picking off again (so another picking mode can be enabled)."""
        self.plotter.disable_picking()

    def picked_layer(self) -> str | None:
        """Return the id of the layer the last pick hit, or ``None``.

        This is the half only this tier can supply: PyVista reports the actor that was picked, and the
        renderer knows which layer id each actor belongs to, so a "which layer did I click?" is answerable
        where a bare PyVista pick only gives geometry.

        Returns:
            The layer id whose actor PyVista last picked, or ``None`` when nothing was picked or the picked
            actor is not one of the scene's layers (a widget's own, say).
        """
        picked = getattr(self.plotter, "picked_actor", None)
        if picked is None:
            return None
        for layer_id, (_mesh, actor) in self.renderer.drawn.items():
            if actor is picked:
                return layer_id
        return None
