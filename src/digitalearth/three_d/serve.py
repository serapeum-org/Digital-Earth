"""ServeMixin — a live trame view of the scene in a browser/notebook (TD-12).

:meth:`~digitalearth.three_d.base.Scene3DBase.export_html` writes a **frozen** vtk.js page whose own docstring
measures the cost — "~19 bytes per point … a 4700x4700 DEM gives ~430 MB". A served scene is the correct answer
to that: geometry (or rendered images) stream on demand, so there is no per-point page-size floor, and the
scene is usable from a notebook against a remote or headless machine. The whole trame stack ships with the
``3d`` extra (``pyvista[jupyter]`` pulls trame + trame-vtk/-server/-client/-vuetify), so this is wiring, not a
new dependency.

``serve()`` hands the live view back through PyVista's own trame integration; the three render modes trade off
where rendering happens — ``"client"`` streams geometry and needs no server GPU, ``"server"`` renders remotely
and streams images, ``"trame"`` switches at runtime.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class ServeMixin(_MixinBase):
    """Adds :meth:`serve` — a live trame view — to a :class:`Scene3D`.

    A capability mixin of :class:`~digitalearth.three_d.scene3d.Scene3D`, composed into that scene and never
    instantiated on its own; it reaches the wrapped ``pyvista.Plotter`` through ``self``.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the plotter this serves.
        digitalearth.three_d.base.Scene3DBase.export_html: the frozen-page alternative this streams instead of.
    """

    def serve(self, *, mode: str = "client", **kwargs: Any) -> Any:
        """Open the scene as a live, interactive trame view and return its handle.

        Unlike :meth:`~digitalearth.three_d.base.Scene3DBase.export_html`, which bakes the whole scene into one
        self-contained page, a served view streams on demand — so a huge cloud or DEM that makes a frozen page
        hundreds of megabytes serves without that floor.

        Args:
            mode: Where rendering happens — ``"client"`` (geometry streams to the browser, no server GPU
                needed), ``"server"`` (the server renders and streams images) or ``"trame"`` (switchable at
                runtime).
            **kwargs: Forwarded to :func:`pyvista.trame.jupyter.show_trame` (``name``, ``server``,
                ``collapse_menu`` …).

        Returns:
            The trame view handle PyVista returns — a widget to display in a notebook, or a handle whose
            server a standalone script runs.

        Examples:
            - Build a scene and hand back a live view (shown here through a stand-in for the trame launcher,
              so the doctest does not open a browser):
                ```python
                >>> import numpy as np
                >>> import pyvista.trame.jupyter as trame_jupyter
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> original = trame_jupyter.show_trame
                >>> trame_jupyter.show_trame = lambda plotter, **kw: ("view", kw.get("mode"))
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
                >>> scene.serve(mode="client")
                ('view', 'client')
                >>> trame_jupyter.show_trame = original
                >>> scene.close()

                ```
        """
        # A process must use one VTK build for trame and PyVista; the base's guard raises a clear RuntimeError
        # naming the fix rather than failing deep inside trame on a wrapped-type mismatch (see base.py).
        from digitalearth.three_d.base import _require_one_vtk_build

        _require_one_vtk_build()
        # Imported at call time, not module load: the trame launcher is only needed when a scene is actually
        # served, and importing it eagerly would load the trame stack for every scene that only renders a frame.
        from pyvista.trame.jupyter import show_trame

        return show_trame(self.plotter, mode=mode, **kwargs)
