"""Scene3DBase — the core PyVista-plotter plumbing the 3D capability mixins build on.

``Scene3DBase`` owns the :class:`pyvista.Plotter`, the layer registry, and the render/export/context-manager
lifecycle. Capability mixins (terrain, point clouds, volumes, vectors, globe) live in sibling modules and add
``terrain()`` / ``point_cloud()`` / … methods that call ``self.add_mesh(...)``; the public
:class:`digitalearth.three_d.scene3d.Scene3D` composes the base with those mixins — exactly mirroring the 2-D
``Map(GeoLayerBase, RasterMixin, …)`` pattern.

PyVista is a renderer, not a GIS engine: meshes are built from pyramids-sourced numpy (never xarray/rasterio — see
the tier's HARD RULE); all CRS/reproject work stays in pyramids. The default ``off_screen`` follows
:data:`pyvista.OFF_SCREEN`, so the same code renders interactively on a desktop and headless in CI.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import pyvista as pv


def house_theme() -> pv.themes.Theme:
    """Return Digital-Earth's default PyVista theme (document-style, anti-aliased).

    Returns:
        pyvista.themes.Theme: a tuned :class:`pyvista.themes.DocumentTheme` — white background, ``viridis``
        default colormap, SSAA anti-aliasing — for clean publication-grade frames.
    """
    theme = pv.themes.DocumentTheme()
    theme.background = "white"
    theme.cmap = "viridis"
    theme.anti_aliasing = "ssaa"
    theme.font.color = "black"
    return theme


class Scene3DBase:
    """Core single-:class:`pyvista.Plotter` host: layer registry + render/export lifecycle.

    Args:
        off_screen: Render without opening a window. ``None`` (default) follows :data:`pyvista.OFF_SCREEN`.
        window_size: Render window size in pixels (``(width, height)``).
        theme: A PyVista theme to apply. ``None`` uses :func:`house_theme`.
        **plotter_kwargs: Forwarded to :class:`pyvista.Plotter`.

    Attributes:
        plotter: The wrapped :class:`pyvista.Plotter`.
        layers: Registered ``(mesh, actor)`` pairs, in add order.
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
            theme=theme or house_theme(),
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

        The low-level entry point the capability mixins build on. ``kwargs`` pass straight to
        :meth:`pyvista.Plotter.add_mesh` (``scalars``, ``cmap``, ``opacity``, ``show_edges``, ``pbr`` …).

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
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_volume`.

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
            The ``path`` written.
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
            Whatever :meth:`pyvista.Plotter.show` returns.
        """
        return self.plotter.show(**kwargs)

    def close(self) -> None:
        """Close the wrapped plotter and free its render window."""
        self.plotter.close()

    def __enter__(self) -> "Scene3DBase":
        """Enter the runtime context, returning the scene.

        Returns:
            This scene.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the plotter on exit (whether or not the body raised); exceptions propagate.

        Returns:
            ``False`` — exceptions are not suppressed.
        """
        self.close()
        return False
