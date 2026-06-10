"""Scene3D — a PyVista plotter that 3D geospatial layers render onto so a scene composes.

``Scene3D`` is the true-3D counterpart of :class:`digitalearth.scene.scene.Scene`. Where ``Scene`` owns one
matplotlib ``fig``/``ax`` and stacks cleopatra glyphs, ``Scene3D`` owns one :class:`pyvista.Plotter` and stacks
PyVista meshes (terrain, point clouds, volumes, vectors, a textured globe — added by mixins in later tasks). Each
``add_*`` method builds a mesh from a pyramids-sourced numpy array (never xarray/rasterio — see the tier's HARD
RULE) and registers the returned actor via :meth:`_add_actor`.

PyVista is a renderer, not a GIS engine: it receives already-prepared numpy + geometry from pyramids and turns it
into pixels. All CRS/reprojection stays in pyramids.

The default ``off_screen`` follows :data:`pyvista.OFF_SCREEN`, so the same code renders interactively on a desktop
and headless in CI (no ``DISPLAY`` needed on VTK 9.5+ wheels).
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import pyvista as pv


def _house_theme() -> pv.themes.Theme:
    """Return Digital-Earth's default PyVista theme (document-style, anti-aliased, terrain colormap).

    Returns:
        pyvista.themes.Theme: a tuned copy of :class:`pyvista.themes.DocumentTheme` — white background, a
        perceptually reasonable default colormap, and SSAA anti-aliasing for clean publication-grade frames.
    """
    theme = pv.themes.DocumentTheme()
    theme.background = "white"
    theme.cmap = "viridis"
    theme.anti_aliasing = "ssaa"
    theme.font.color = "black"
    return theme


class Scene3D:
    """A single-:class:`pyvista.Plotter` host for composing 3D geospatial layers into one scene.

    Args:
        off_screen: Render without opening a window. ``None`` (default) follows :data:`pyvista.OFF_SCREEN`
            (``True`` in CI/headless, ``False`` on a desktop). Pass ``True`` explicitly for batch/screenshot use.
        window_size: Render window size in pixels (``(width, height)``).
        theme: A PyVista theme to apply. ``None`` uses Digital-Earth's house theme (:func:`_house_theme`).
        **plotter_kwargs: Forwarded to :class:`pyvista.Plotter` (e.g. ``shape`` for sub-plots, ``lighting``).

    Attributes:
        plotter: The wrapped :class:`pyvista.Plotter`.
        layers: Registered ``(mesh, actor)`` pairs, in add order.

    Examples:
        - Create a headless scene; it owns one empty plotter:
            ```python
            >>> from digitalearth.three_d import Scene3D
            >>> scene = Scene3D(off_screen=True)
            >>> scene.layers
            []
            >>> scene.close()

            ```
        - Add a mesh and screenshot it off-screen (a real, non-empty frame):
            ```python
            >>> import numpy as np, pyvista as pv
            >>> from digitalearth.three_d import Scene3D
            >>> scene = Scene3D(off_screen=True)
            >>> grid = pv.ImageData(dimensions=(8, 8, 1))
            >>> grid.point_data["z"] = np.arange(64.0)
            >>> _ = scene.add_mesh(grid.warp_by_scalar("z"), scalars="z")
            >>> img = scene.screenshot()
            >>> img.shape[-1]
            3
            >>> bool(img.any())
            True
            >>> scene.close()

            ```
    """

    def __init__(
        self,
        off_screen: Optional[bool] = None,
        window_size: Tuple[int, int] = (1024, 768),
        theme: Optional[pv.themes.Theme] = None,
        **plotter_kwargs: Any,
    ):
        self.plotter: pv.Plotter = pv.Plotter(
            off_screen=off_screen,
            window_size=list(window_size),
            theme=theme or _house_theme(),
            **plotter_kwargs,
        )
        self.layers: List[Tuple[Any, Any]] = []

    def _add_actor(self, mesh: Any, actor: Any) -> Any:
        """Register a rendered ``mesh`` and its ``actor``, returning the actor.

        Args:
            mesh: The PyVista mesh that was added to the plotter.
            actor: The :class:`pyvista.Actor` the plotter produced.

        Returns:
            The ``actor`` (so callers can chain or tweak its properties).
        """
        self.layers.append((mesh, actor))
        return actor

    def add_mesh(self, mesh: Any, **kwargs: Any) -> Any:
        """Add a PyVista ``mesh`` to the scene and register it as a layer.

        This is the low-level entry point the capability mixins build on (terrain, point clouds, volumes, …).
        ``kwargs`` are passed straight to :meth:`pyvista.Plotter.add_mesh` (``scalars``, ``cmap``, ``opacity``,
        ``show_edges``, ``pbr``, …) — styling is the renderer's job, not handled here.

        Args:
            mesh: Any PyVista dataset (``ImageData``/``StructuredGrid``/``PolyData``/``UnstructuredGrid``).
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered :class:`pyvista.Actor`.
        """
        actor = self.plotter.add_mesh(mesh, **kwargs)
        return self._add_actor(mesh, actor)

    def add_volume(self, volume: Any, **kwargs: Any) -> Any:
        """Add a volumetric ``volume`` (ray-cast rendering) and register it as a layer.

        Args:
            volume: An ``ImageData``/``UnstructuredGrid`` carrying a scalar field to ray-cast.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_volume` (``opacity``, ``cmap``, ``shade``, …).

        Returns:
            The registered volume actor.
        """
        actor = self.plotter.add_volume(volume, **kwargs)
        return self._add_actor(volume, actor)

    def screenshot(self, path: Optional[str] = None, **kwargs: Any) -> np.ndarray:
        """Render the scene off-screen and return the RGB image (optionally writing it to ``path``).

        Args:
            path: Optional file path to save the PNG. When ``None`` the image is only returned.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.screenshot`.

        Returns:
            numpy.ndarray: the ``(height, width, 3)`` RGB frame.
        """
        return self.plotter.screenshot(filename=path, return_img=True, **kwargs)

    def export_html(self, path: str) -> str:
        """Export the scene to a self-contained interactive HTML page (via trame/vtk.js).

        Args:
            path: Destination ``.html`` file.

        Returns:
            The ``path`` written (so callers can chain).
        """
        self.plotter.export_html(path)
        return path

    def save(self, path: str, **kwargs: Any) -> np.ndarray:
        """Save the scene — a PNG screenshot, or interactive HTML when ``path`` ends in ``.html``.

        Args:
            path: Output file. ``*.html`` exports an interactive page; anything else saves a PNG screenshot.
            **kwargs: Forwarded to :meth:`screenshot` for the PNG path (ignored for HTML).

        Returns:
            numpy.ndarray: the rendered frame for the PNG path; an empty array for the HTML path.
        """
        if str(path).lower().endswith(".html"):
            self.export_html(path)
            return np.empty((0,))
        return self.screenshot(path=path, **kwargs)

    def show(self, **kwargs: Any) -> Any:
        """Display the scene interactively (or render a frame off-screen).

        Args:
            **kwargs: Forwarded to :meth:`pyvista.Plotter.show`.

        Returns:
            Whatever :meth:`pyvista.Plotter.show` returns (camera position / image, depending on mode).
        """
        return self.plotter.show(**kwargs)

    def close(self) -> None:
        """Close the wrapped plotter and free its render window."""
        self.plotter.close()

    def __enter__(self) -> "Scene3D":
        """Enter the runtime context, returning the scene so ``with Scene3D(...) as s:`` binds it.

        Returns:
            This scene.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the plotter on exit so a long run of scenes stays resource-bounded.

        The plotter is closed whether or not the body raised; any exception propagates (returns ``False``).

        Returns:
            ``False`` — exceptions are not suppressed.
        """
        self.close()
        return False
