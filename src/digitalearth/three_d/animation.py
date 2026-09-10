"""AnimationMixin — orbit fly-throughs, frame-by-frame animations, and Jupyter delivery for a Scene3D.

Three deliverables on top of a built scene:

- :meth:`orbit` — sweep the camera around the scene on a circular path and write each frame to a GIF/MP4.
  The path is shapeable: ``factor`` sets its radius and ``shift`` lifts it along ``viewup``, which together
  decide whether near-flat terrain is looked down on or seen edge-on.
- :meth:`animate` — drive a frame-by-frame animation from a sequence of states (e.g. a ``DatasetCollection``
  time stack), via a user ``update`` callback, to a GIF/MP4.
- :meth:`jupyter` — switch PyVista to the trame backend so the scene displays interactively in a notebook.
  That backend is process-wide, not per scene.

GIF/MP4 writing uses PyVista's ``open_gif``/``open_movie`` (which need ``imageio`` / ``imageio-ffmpeg`` — both in
the ``3d`` extra). No GIS is touched here: animation is pure rendering of already-built meshes; data still comes
from pyramids upstream.
"""

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, Union

import numpy as np

#: An up vector: three floats, as a sequence or a numpy array. numpy is the natural way to spell one and
#: is not a ``typing.Sequence``, so both are accepted rather than adding to the mypy arg-type baseline.
UpVector = Union[Sequence[float], np.ndarray]

#: File suffixes routed to ``open_movie`` (everything else → ``open_gif``).
_MOVIE_SUFFIXES = (".mp4", ".mov", ".avi", ".m4v")


def _open_writer(plotter: Any, path: str, framerate: int) -> None:
    """Open the right PyVista frame writer for ``path`` (movie for video suffixes, else GIF).

    Args:
        plotter: The :class:`pyvista.Plotter` to attach the writer to.
        path: Destination file. A suffix in :data:`_MOVIE_SUFFIXES` opens a movie writer, anything else a GIF.
        framerate: Frames per second, passed as ``framerate`` to ``open_movie`` and ``fps`` to ``open_gif``.

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
    """Flush and close the frame writer so the GIF/MP4 is fully written, leaving the plotter usable.

    A no-op when no writer was opened. Both callers open theirs before the ``try``, so this only matters to a
    direct caller — it keeps the helper safe to use on a plotter that never wrote frames.

    Args:
        plotter: The :class:`pyvista.Plotter` whose ``mwriter`` should be flushed and closed.
    """
    writer = getattr(plotter, "mwriter", None)
    if writer is not None:
        writer.close()


def _finite_number(value: Any, name: str) -> float:
    """Return ``value`` as a finite float, naming it in the error when it is not one.

    Args:
        value: The argument to check.
        name: How to describe it in the message.

    Returns:
        The value as a float.

    Raises:
        ValueError: If it is not numeric or not finite. numpy's own message for a wrong type
            (``ufunc 'isfinite' not supported for the input types``) names neither the argument nor the
            caller, which is the reason this wraps it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"orbit() needs a numeric {name}, got {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"orbit() needs a finite {name}, got {value!r}")
    return number


def _up_vector(viewup: "UpVector | None") -> "np.ndarray | None":
    """Return ``viewup`` as a validated ``(3,)`` float array, or ``None``.

    The array returned is the one that must be forwarded to pyvista. Validating a converted copy and passing
    the caller's original lets a masked array clear the finiteness check on its fill data while pyvista then
    multiplies the masked value, and lets a list of numeric strings reach ``np.array(viewup) * shift`` and die
    there naming neither ``viewup`` nor ``orbit``.

    Args:
        viewup: The caller's up vector, or ``None`` to leave pyvista's default.

    Returns:
        The validated vector, or ``None``.

    Raises:
        ValueError: If it is not three finite numbers with a direction — a zero vector names none.
    """
    if viewup is None:
        return None
    try:
        vector = np.asarray(viewup, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"orbit() needs a 3-component numeric viewup, got {viewup!r}"
        ) from exc
    if vector.shape != (3,):
        raise ValueError(
            f"orbit() needs a 3-component viewup, got shape {vector.shape} from {viewup!r}"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"orbit() needs a finite viewup, got {viewup!r}")
    if not vector.any():
        raise ValueError(
            "orbit() cannot use a zero viewup: it names no up direction for the orbit to lie against"
        )
    return vector


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

    Examples:
        - The three methods this mixin contributes, and the runtime base that keeps the composition inert:
            ```python
            >>> from digitalearth.three_d.animation import AnimationMixin
            >>> sorted(name for name in vars(AnimationMixin) if not name.startswith("_"))
            ['animate', 'jupyter', 'orbit']
            >>> AnimationMixin.__bases__
            (<class 'object'>,)

            ```
        - They are reached through the composed scene, never on the mixin itself, which owns no state:
            ```python
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.three_d.animation import AnimationMixin
            >>> AnimationMixin in Scene3D.__mro__
            True
            >>> vars(AnimationMixin).get("__init__") is None
            True

            ```

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
        factor: float = 3.0,
        shift: float = 0.0,
        viewup: UpVector | None = None,
        **orbit_kwargs: Any,
    ) -> str:
        """Sweep the camera around the scene and write the fly-through to a GIF/MP4.

        ``factor``, ``shift`` and ``viewup`` shape the path the camera travels; the rest of ``orbit_kwargs``
        tunes how it is flown along that path. They belong to two different pyvista calls, which is why they
        are separate arguments rather than one pass-through bag.

        ``viewup`` shapes **both** — the circle the camera flies and the camera's own up vector. Before these
        arguments existed it reached only the camera, so a caller who passed it got a path built around the
        theme's up vector and a camera holding a different one, which rolls the horizon through the turn. That
        is now a behaviour change for such a caller, and a deliberate one: the two cannot sensibly disagree.

        Args:
            path: Output file. A suffix in :data:`_MOVIE_SUFFIXES` writes a movie; anything else a GIF.
            n_frames: Number of frames (camera positions) along the orbit. At least 3. pyvista silently
                clamps a smaller value to 3, so fewer used to "work" and produce a three-frame clip; this
                rejects it instead, which is a narrowing of what the argument accepted before.
            framerate: Frames per second of the output.
            factor: Orbit radius as a multiple of the scene's bounding size. Smaller closes in on the data.
                Must be positive.
            shift: How far to lift the orbit off the data's mid-plane, as an absolute offset **along**
                ``viewup`` (so along z while ``viewup`` is z-aligned, which is the default). It is in scene
                units, not a fraction, so the useful value depends on the data's own extent along that vector
                and on any ``z_exaggeration`` applied to it — it has to be picked per scene. The default
                ``0.0`` circles level with the middle of the data, which for near-flat terrain is an edge-on
                view of a sheet.
            viewup: Up vector for the path and the camera alike, as three floats. ``None`` leaves pyvista's
                default.
            **orbit_kwargs: Forwarded to :meth:`pyvista.Plotter.orbit_on_path`, whose signature names what it
                accepts — ``step`` and ``focus`` are the useful ones here. It takes no ``**kwargs``, so a
                keyword it does not name raises :class:`TypeError`; ``factor`` and ``shift`` are not among
                them, which is why they are named arguments above. ``progress_bar=True`` additionally needs
                ``tqdm``, which the ``3d`` extra does not install.

        Returns:
            The ``path`` written.

        Raises:
            ValueError: If ``factor`` is not a positive finite number, ``shift`` is not a finite number,
                ``n_frames`` is not a whole number of at least 3, or ``viewup`` is not three finite numbers
                with a direction (a zero vector gives none). Non-numeric values are rejected here too, by
                name — numpy's own message for them names neither the argument nor this method. Also if
                ``threaded=True`` is passed: :meth:`pyvista.Plotter.orbit_on_path` returns
                before its render thread has written a frame, so the writer here would already be closed and
                no file would be produced — drive that method yourself for a background render.
            TypeError: If ``orbit_kwargs`` carries a keyword ``orbit_on_path`` does not name; it takes no
                ``**kwargs`` of its own. The frame writer is still closed when this happens.

        Examples:
            - Orbit a terrain scene to a GIF (needs the ``3d`` extra for imageio):
                ```python
                >>> import numpy as np, os, tempfile
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.sources import get_source
                >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3D(off_screen=True)
                ...     _ = scene.terrain(get_source(dem), z_exaggeration=3.0)
                ...     out = scene.orbit(os.path.join(folder, "spin.gif"), n_frames=6)
                ...     size = os.path.getsize(out)
                ...     scene.close()
                >>> size > 0
                True

                ```
            - Close the orbit in and lift it clear of the terrain, so a near-flat sheet is looked down on
              rather than seen edge-on. The orbit sits at the middle of the data, so a shift greater than half
              the vertical extent lifts the camera above it — 3.0 for this scene, whose z runs 0..6. Pick it
              from your own data's extent, not from this example:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.sources import get_source
                >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3D(off_screen=True)
                ...     _ = scene.terrain(get_source(dem), z_exaggeration=3.0)
                ...     out = scene.orbit(
                ...         os.path.join(folder, "spin.gif"), n_frames=4, factor=0.9, shift=8.0
                ...     )
                ...     size = os.path.getsize(out)
                ...     scene.close()
                >>> size > 0
                True

                ```
        """
        factor = _finite_number(factor, "factor (orbit radius)")
        if factor <= 0:
            raise ValueError(
                f"orbit() needs a positive, finite factor (orbit radius), got {factor!r}"
            )
        shift = _finite_number(shift, "shift")
        if not isinstance(n_frames, (int, np.integer)) or n_frames < 3:
            raise ValueError(
                f"orbit() needs a whole number of frames, at least 3, got {n_frames!r}"
            )
        viewup = _up_vector(viewup)
        if orbit_kwargs.get("threaded"):
            # orbit_on_path(threaded=True) returns before the render thread has written a frame, so the
            # finally below closes the writer first and the file is never created — silently, with a path
            # returned as if it had been. Refuse rather than hand back a filename for nothing.
            raise ValueError(
                "orbit() writes frames synchronously and closes the writer as it returns, so threaded=True "
                "would finish after the file was closed and produce nothing. Drive plotter.orbit_on_path "
                "directly if you need a background render."
            )
        _open_writer(self.plotter, path, framerate)
        try:
            orbital_path = self.plotter.generate_orbital_path(
                factor=factor, n_points=n_frames, viewup=viewup, shift=shift
            )
            # The path and the camera take the same up vector: generating the path around one while the camera
            # holds another rolls the horizon through the turn.
            self.plotter.orbit_on_path(
                orbital_path, write_frames=True, viewup=viewup, **orbit_kwargs
            )
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
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     out = scene.animate([1.1, 1.1, 1.1], os.path.join(folder, "grow.gif"), grow)
                ...     size = os.path.getsize(out)
                ...     scene.close()
                >>> size > 0
                True

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

        The backend is PyVista's, set process-wide rather than per scene, so it stays in force for every
        plotter until something changes it back.

        Args:
            backend: A PyVista Jupyter backend — ``"trame"`` (default, server/remote), ``"client"`` (vtk.js),
                ``"static"`` (screenshot), or ``"html"``.

        Examples:
            - Switch to the static (screenshot) backend and read the change back. The restore is in a
              ``finally`` because the backend is process-wide: doctests share one interpreter, and a failure
              between the switch and the restore would leave every later one on a renderer nobody chose:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> previous = pv.global_theme.jupyter_backend
                >>> try:
                ...     scene.jupyter("static")
                ...     pv.global_theme.jupyter_backend
                ... finally:
                ...     scene.close()
                ...     pv.set_jupyter_backend(previous)
                'static'

                ```

        See Also:
            orbit: writes a fly-through to a file instead, for when a notebook is not the destination.
        """
        import pyvista as pv

        pv.set_jupyter_backend(backend)
