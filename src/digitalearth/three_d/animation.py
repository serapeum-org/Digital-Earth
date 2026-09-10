"""AnimationMixin — orbit fly-throughs, frame-by-frame animations, and Jupyter delivery for a Scene3D.

Three deliverables on top of a built scene:

- :meth:`orbit` — sweep the camera around the scene on a circular path and write each frame to a GIF/MP4.
- :meth:`animate` — drive a frame-by-frame animation from a sequence of states (e.g. a ``DatasetCollection``
  time stack), via a user ``update`` callback, to a GIF/MP4.
- :meth:`jupyter` — switch PyVista to the trame backend so the scene displays interactively in a notebook.

GIF/MP4 writing uses PyVista's ``open_gif``/``open_movie`` (which need ``imageio`` / ``imageio-ffmpeg`` — both in
the ``3d`` extra). No GIS is touched here: animation is pure rendering of already-built meshes; data still comes
from pyramids upstream.
"""

from typing import TYPE_CHECKING, Any, Callable, Iterable

#: File suffixes routed to ``open_movie`` (everything else → ``open_gif``).
_MOVIE_SUFFIXES = (".mp4", ".mov", ".avi", ".m4v")


def _open_writer(plotter: Any, path: str, framerate: int) -> None:
    """Open the right PyVista frame writer for ``path`` (movie for video suffixes, else GIF).

    Raises:
        AttributeError: if PyVista did not attach its frame writer (``mwriter``) after opening — a fail-fast
            guard against a future PyVista renaming the attribute :func:`_finalize_frames` relies on.
    """
    if str(path).lower().endswith(_MOVIE_SUFFIXES):
        plotter.open_movie(path, framerate=framerate)
    else:
        plotter.open_gif(path, fps=framerate)
    if not hasattr(
        plotter, "mwriter"
    ):  # pragma: no cover - defensive against an upstream API change
        raise AttributeError(
            "PyVista did not expose a frame writer ('mwriter') after open_gif/open_movie; "
            "the installed pyvista version may be incompatible with digitalearth's animation helpers."
        )


def _finalize_frames(plotter: Any) -> None:
    """Flush and close the frame writer so the GIF/MP4 is fully written, leaving the plotter usable."""
    writer = getattr(plotter, "mwriter", None)
    if writer is not None:
        writer.close()


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class AnimationMixin(_MixinBase):
    """Adds :meth:`orbit`, :meth:`animate`, and :meth:`jupyter` to a :class:`Scene3D`.

    A capability mixin of :class:`~digitalearth.three_d.scene3d.Scene3D`: it is only ever composed into that scene
    class, never instantiated or subclassed on its own. Its methods reach the wrapped ``pyvista.Plotter``, the layer
    registry and the render/export lifecycle — and the sibling mixins' methods — through ``self``, and only the
    composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.three_d.base.Scene3DBase`, the state ``Scene3D``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``Scene3D`` MRO exactly
    what it was before the annotation.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the state these methods use.
        digitalearth.three_d.base.Scene3DBase: the typing-only base declared above the class.
    """

    def orbit(
        self,
        path: str,
        *,
        n_frames: int = 36,
        framerate: int = 12,
        **orbit_kwargs: Any,
    ) -> str:
        """Sweep the camera around the scene and write the fly-through to a GIF/MP4.

        Args:
            path: Output file. A video suffix (``.mp4``/``.mov``/``.avi``) writes a movie; anything else a GIF.
            n_frames: Number of frames (camera positions) along the orbit.
            framerate: Frames per second of the output.
            **orbit_kwargs: Forwarded to :meth:`pyvista.Plotter.orbit_on_path` (``factor``, ``viewup`` …).

        Returns:
            The ``path`` written.

        Examples:
            - Orbit a terrain scene to a GIF (needs the ``3d`` extra for imageio):
                ```python
                >>> import numpy as np, os, tempfile
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.sources import get_source
                >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(dem), z_exaggeration=3.0)
                >>> out = scene.orbit(os.path.join(tempfile.mkdtemp(), "spin.gif"), n_frames=6)
                >>> os.path.getsize(out) > 0
                True
                >>> scene.close()

                ```
        """
        _open_writer(self.plotter, path, framerate)
        try:
            orbital_path = self.plotter.generate_orbital_path(n_points=n_frames)
            self.plotter.orbit_on_path(orbital_path, write_frames=True, **orbit_kwargs)
        finally:
            _finalize_frames(
                self.plotter
            )  # always flush/close the writer, even if rendering raised
        return path

    def animate(
        self,
        frames: Iterable[Any],
        path: str,
        update: Callable[["AnimationMixin", Any], None],
        *,
        framerate: int = 8,
    ) -> str:
        """Render a frame-by-frame animation driven by ``update`` and write it to a GIF/MP4.

        For each item in ``frames``, ``update(self, frame)`` mutates the scene (e.g. swaps the active scalars or
        re-adds a layer), then one frame is written. Typical ``frames`` is a ``DatasetCollection`` time stack.

        Args:
            frames: Iterable of per-frame states passed one at a time to ``update``.
            path: Output file. A video suffix writes a movie; anything else a GIF.
            update: Callback ``(scene, frame) -> None`` that updates the scene before each frame is captured.
            framerate: Frames per second of the output.

        Returns:
            The ``path`` written.

        Examples:
            - Animate a growing terrain over three frames:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.sources import get_source
                >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(dem))
                >>> def grow(s, factor):
                ...     s.layers[0][0].points[:, 2] *= factor
                >>> out = scene.animate([1.1, 1.1, 1.1], os.path.join(tempfile.mkdtemp(), "grow.gif"), grow)
                >>> os.path.getsize(out) > 0
                True
                >>> scene.close()

                ```
        """
        _open_writer(self.plotter, path, framerate)
        try:
            for frame in frames:
                update(self, frame)
                self.plotter.write_frame()
        finally:
            _finalize_frames(
                self.plotter
            )  # always flush/close the writer, even if a frame raised
        return path

    def jupyter(self, backend: str = "trame") -> None:
        """Switch PyVista's rendering backend so the scene displays interactively in a notebook.

        Args:
            backend: A PyVista Jupyter backend — ``"trame"`` (default, server/remote), ``"client"`` (vtk.js),
                ``"static"`` (screenshot), or ``"html"``.
        """
        import pyvista as pv

        pv.set_jupyter_backend(backend)
