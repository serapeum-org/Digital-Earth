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
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pyvista as pv

#: Anything acceptable as an output destination.
PathLike = Union[str, "os.PathLike[str]"]


def house_theme() -> pv.themes.Theme:
    """Return Digital-Earth's default PyVista theme (document-style, anti-aliased).

    Returns:
        pyvista.themes.Theme: a tuned :class:`pyvista.themes.DocumentTheme` — white background, ``viridis``
        default colormap, SSAA anti-aliasing — for clean publication-grade frames.

    Examples:
        - Read back the settings a scene renders with by default:
            ```python
            >>> from digitalearth.three_d.base import house_theme
            >>> theme = house_theme()
            >>> theme.cmap
            'viridis'
            >>> theme.background.hex_rgb
            '#ffffff'
            >>> theme.anti_aliasing
            'ssaa'

            ```
        - Every call hands back a fresh theme, so tweaking one scene's colours leaves the next untouched:
            ```python
            >>> from digitalearth.three_d.base import house_theme
            >>> mine = house_theme()
            >>> mine.cmap = "magma"
            >>> house_theme().cmap
            'viridis'

            ```

    See Also:
        Scene3DBase: applies this theme whenever its ``theme`` argument is left as ``None``.
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

    Examples:
        - Build a scene, stack two meshes on its single plotter, and read the layer registry back:
            ```python
            >>> import pyvista as pv
            >>> from digitalearth.three_d.base import Scene3DBase
            >>> scene = Scene3DBase(off_screen=True)
            >>> _ = scene.add_mesh(pv.Sphere())
            >>> _ = scene.add_mesh(pv.Cube(center=(3, 0, 0)))
            >>> len(scene.layers)
            2
            >>> scene.close()

            ```
        - Size the render window, then confirm the frame it produces matches:
            ```python
            >>> from digitalearth.three_d.base import Scene3DBase
            >>> scene = Scene3DBase(off_screen=True, window_size=(320, 240))
            >>> scene.screenshot().shape
            (240, 320, 3)
            >>> scene.close()

            ```
        - Used as a context manager the plotter is closed on the way out, even if the body raises:
            ```python
            >>> import pyvista as pv
            >>> from digitalearth.three_d.base import Scene3DBase
            >>> with Scene3DBase(off_screen=True) as scene:
            ...     _ = scene.add_mesh(pv.Sphere())
            ...     len(scene.layers)
            1

            ```

    See Also:
        digitalearth.three_d.scene3d.Scene3D: composes this base with the terrain/point-cloud/volume/vector
            capability mixins, and is the class to use directly.
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

        Examples:
            - Add one mesh and find it, paired with its actor, in the layer registry:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> actor = scene.add_mesh(pv.Sphere())
                >>> mesh, registered = scene.layers[0]
                >>> registered is actor
                True
                >>> scene.close()

                ```
            - Style the mesh through ``kwargs``, then tune the returned actor further:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> grid = pv.ImageData(dimensions=(8, 8, 1))
                >>> grid.point_data["z"] = np.linspace(0.0, 1.0, 64)
                >>> scene = Scene3DBase(off_screen=True)
                >>> actor = scene.add_mesh(grid, scalars="z", cmap="terrain", show_edges=True)
                >>> actor.prop.opacity = 0.5
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```

        See Also:
            add_volume: the ray-cast counterpart, for scalar fields rather than surfaces.
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

        Examples:
            - Ray-cast a 3-D scalar field and see it registered as one layer:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> grid = pv.ImageData(dimensions=(6, 6, 6))
                >>> grid.cell_data["v"] = np.linspace(0.0, 1.0, 125)
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_volume(grid, cmap="viridis")
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```
            - A volume and a surface share the one plotter, stacking in add order:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> grid = pv.ImageData(dimensions=(6, 6, 6))
                >>> grid.cell_data["v"] = np.linspace(0.0, 1.0, 125)
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_volume(grid)
                >>> _ = scene.add_mesh(pv.Sphere(radius=1.0, center=(2, 2, 2)))
                >>> len(scene.layers)
                2
                >>> scene.close()

                ```

        See Also:
            add_mesh: the surface counterpart, and the method the capability mixins call.
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

        Examples:
            - Render to memory and work with the frame as an array:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True, window_size=(200, 150))
                >>> _ = scene.add_mesh(pv.Sphere())
                >>> frame = scene.screenshot()
                >>> frame.shape
                (150, 200, 3)
                >>> bool(frame.any())
                True
                >>> scene.close()

                ```
            - Write a PNG to disk; the frame is still returned:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_mesh(pv.Cube())
                >>> out = os.path.join(tempfile.mkdtemp(), "frame.png")
                >>> frame = scene.screenshot(path=out)
                >>> os.path.getsize(out) > 0
                True
                >>> scene.close()

                ```

        See Also:
            save: dispatches here for any path that is not ``*.html``.
        """
        return self.plotter.screenshot(filename=path, return_img=True, **kwargs)

    def export_html(self, path: PathLike) -> str:
        """Export the scene to a self-contained interactive HTML page (via trame/vtk.js).

        The page embeds the whole mesh, so it grows with the geometry rather than with the rendered image: about
        a 1 MB vtk.js floor plus ~19 bytes per point (a 12x12 grid gives 1.1 MB, 512x512 gives 6 MB, and a
        4700x4700 DEM gives ~430 MB). Past a modest tile the result is technically interactive but too heavy to
        sit beside a notebook or in docs — prefer :meth:`digitalearth.three_d.Scene3D.orbit`, which writes a
        compact GIF/MP4 fly-through of the same scene.

        The destination is normalised to a ``.html`` suffix before writing, and that normalised path is what
        comes back. pyvista's own behaviour here differs by version — the ``trame-pyvista`` component used from
        0.49 rewrites a non-``.html`` suffix, while 0.48's native export honours the name it was given — so
        normalising up front is what makes the two agree and keeps the returned path the file that exists.

        Args:
            path: Destination file. A suffix other than ``.html`` (including ``.HTML``) is replaced with
                ``.html``; a name with no suffix gains one.

        Returns:
            The ``.html`` path written, as a string.

        Raises:
            ImportError: If the trame/vtk.js export stack is missing. pyvista raises this itself and its message
                names the package to install; the ``3d`` extra pulls the stack via ``pyvista[jupyter]``.

        Examples:
            - Export a small scene; the returned path is the page that was written, so it can be passed straight
              on:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3DBase(off_screen=True)
                ...     _ = scene.add_mesh(pv.Sphere())
                ...     out = scene.export_html(os.path.join(folder, "scene.html"))
                ...     scene.close()
                ...     os.path.basename(out), os.path.getsize(out) > 0
                ('scene.html', True)

                ```
            - Any other suffix is normalised to ``.html``, and the normalised name is what comes back — so the
              returned path always names the file that exists, on either pyvista:
                ```python
                >>> from pathlib import Path
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> Path("map.HTML").with_suffix(".html").name
                'map.html'

                ```

        See Also:
            digitalearth.three_d.Scene3D.orbit: a compact GIF/MP4 fly-through, the better choice for a heavy
                scene.
            save: routes here automatically for a ``*.html`` destination.
        """
        # pyvista >=0.49 moved trame support out into the separate `trame-pyvista` package: the export now lives
        # on a registered `trame` plotter component and `Plotter.export_html` is deprecated. pyvista 0.48 has no
        # such attribute and implements the export natively, so the attribute doubles as the version switch.
        # Falling back (rather than raising here) keeps pyvista's own actionable ImportError when >=0.49 is
        # installed without trame-pyvista.
        destination = str(Path(path).with_suffix(".html"))
        component = getattr(self.plotter, "trame", None)
        if component is None:
            self.plotter.export_html(destination)
        else:
            component.export_html(destination)
        return destination

    def save(self, path: PathLike, **kwargs: Any) -> Optional[np.ndarray]:
        """Save the scene — a PNG screenshot, or interactive HTML when ``path`` ends in ``.html``.

        The HTML branch delegates to :meth:`export_html` — see there for why a heavy scene is better served by
        :meth:`digitalearth.three_d.Scene3D.orbit`, and for the ``.html`` suffix normalisation it applies.

        Args:
            path: Output file. ``*.html`` exports an interactive page; anything else saves a PNG screenshot.
            **kwargs: Forwarded to :meth:`screenshot` for the PNG path (ignored for HTML).

        Returns:
            Optional[numpy.ndarray]: the rendered ``(H, W, 3)`` RGB frame for the PNG path, or ``None`` for the
            HTML path (which writes an interactive page rather than a raster frame).

        Examples:
            - A raster suffix screenshots, returning the frame that was written:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True, window_size=(200, 150))
                >>> _ = scene.add_mesh(pv.Sphere())
                >>> out = os.path.join(tempfile.mkdtemp(), "scene.png")
                >>> frame = scene.save(out)
                >>> frame.shape
                (150, 200, 3)
                >>> os.path.getsize(out) > 0
                True
                >>> scene.close()

                ```
            - An ``.html`` suffix exports an interactive page instead, and returns ``None`` rather than a frame.
              The match is case-insensitive, and :meth:`export_html` normalises the suffix it writes, so
              ``SCENE.HTML`` lands as ``SCENE.html``:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3DBase(off_screen=True)
                ...     _ = scene.add_mesh(pv.Sphere())
                ...     returned = scene.save(os.path.join(folder, "SCENE.HTML"))
                ...     scene.close()
                ...     returned is None, sorted(os.listdir(folder))
                (True, ['SCENE.html'])

                ```

        See Also:
            screenshot: the raster branch, and where ``**kwargs`` end up.
            export_html: the interactive branch.
        """
        if str(path).lower().endswith(".html"):
            self.export_html(path)
            return None
        return self.screenshot(path=path, **kwargs)

    def show(self, **kwargs: Any) -> Any:
        """Display the scene interactively (or render a frame off-screen).

        Args:
            **kwargs: Forwarded to :meth:`pyvista.Plotter.show`.

        Returns:
            Whatever :meth:`pyvista.Plotter.show` returns.

        Examples:
            - Off-screen (as in CI, or under :data:`pyvista.OFF_SCREEN`) it renders a frame and returns without
              opening a window:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_mesh(pv.Sphere())
                >>> scene.show() is None
                True
                >>> scene.close()

                ```
            - Keyword arguments reach :meth:`pyvista.Plotter.show`, so the camera can be framed on the way in:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_mesh(pv.Cube())
                >>> _ = scene.show(cpos="xy")
                >>> scene.close()

                ```

        See Also:
            screenshot: returns the rendered frame as an array instead of displaying it.
        """
        return self.plotter.show(**kwargs)

    def close(self) -> None:
        """Close the wrapped plotter and free its render window.

        Closing twice is harmless, so a scene can be closed explicitly inside a ``with`` block that will close
        it again on exit.

        Examples:
            - Free the render window when the scene is finished with:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_mesh(pv.Sphere())
                >>> scene.close()
                >>> len(scene.layers)
                1

                ```
            - A second close is a no-op, not an error:
                ```python
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> scene.close()
                >>> scene.close()

                ```

        See Also:
            __exit__: calls this on the way out of a ``with`` block.
        """
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
