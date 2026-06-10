"""VolumeMixin — ray-cast a 3-D scalar cube (NetCDF / datacube) and extract isosurfaces.

Turns a 3-D numpy field into a PyVista ``ImageData`` for either **volume rendering** (``add_volume``, a depth-cued
ray-cast of the whole field) or **isosurfaces** (``contour``, shells of constant value). The cube comes from
pyramids — a ``DatasetCollection`` (its ``.values`` stack) or a 3-D array read via the pyramids ``NetCDF`` reader —
**never** xarray (the HARD RULE / ``test_no_competitor_imports`` guard); a ``DatasetCollection`` is duck-typed via
``hasattr(data, "values")`` so nothing GIS is imported.

VTK fills a uniform grid in **Fortran order**, so the scalar attaches with ``ravel(order="F")``. Volume rendering
uses ``cell_data`` on a grid sized ``shape + 1`` (one more point than cells per axis); isosurfacing uses
``point_data`` on a grid sized ``shape``.
"""
from typing import Any, Optional, Sequence

import numpy as np
import pyvista as pv

#: Attribute name the field scalar is stored under on the generated grids.
FIELD = "field"


def _cube(data: Any) -> np.ndarray:
    """Coerce ``data`` to a 3-D float numpy cube (reading ``DatasetCollection.values`` when present).

    Args:
        data: A 3-D array-like, or a pyramids ``DatasetCollection`` exposing a 3-D ``.values`` stack.

    Returns:
        numpy.ndarray: the ``(nz, ny, nx)`` float cube.

    Raises:
        ValueError: if the resolved array is not 3-dimensional.
    """
    arr = data.values if hasattr(data, "values") else data
    arr = np.asarray(arr, dtype="float64")
    if arr.ndim != 3:
        raise ValueError(f"volume expects a 3-D cube, got shape {arr.shape}")
    return arr


def _volume_grid(cube: np.ndarray) -> pv.ImageData:
    """Build a cell-data ``ImageData`` (dimensions ``shape + 1``) for ray-cast volume rendering."""
    grid = pv.ImageData(dimensions=np.array(cube.shape) + 1)
    grid.cell_data[FIELD] = cube.ravel(order="F")
    return grid


def _point_grid(cube: np.ndarray) -> pv.ImageData:
    """Build a point-data ``ImageData`` (dimensions ``shape``) for isosurface extraction."""
    grid = pv.ImageData(dimensions=cube.shape)
    grid.point_data[FIELD] = cube.ravel(order="F")
    return grid


class VolumeMixin:
    """Adds :meth:`volume` and :meth:`isosurface` — render a 3-D scalar field — to a :class:`Scene3D`."""

    def volume(
        self,
        data: Any,
        *,
        cmap: str = "viridis",
        opacity: Any = "sigmoid",
        **kwargs: Any,
    ) -> Any:
        """Ray-cast a 3-D scalar field as a volume and register it as a layer.

        Args:
            data: A 3-D numpy cube, or a pyramids ``DatasetCollection`` whose ``.values`` is a 3-D stack.
            cmap: Colormap for the field.
            opacity: Opacity transfer function — a named ramp (``"sigmoid"``, ``"linear"``, ``"geom"`` …), a
                scalar, or an array. Controls how much of the field is see-through.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_volume`.

        Returns:
            The registered volume actor.

        Examples:
            - Volume-render a synthetic 3-D Gaussian blob:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> ax = np.linspace(-2, 2, 12)
                >>> xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
                >>> cube = np.exp(-(xx**2 + yy**2 + zz**2))
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.volume(cube)
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```
        """
        actor = self.add_volume(_volume_grid(_cube(data)), cmap=cmap, opacity=opacity, **kwargs)
        return actor

    def isosurface(
        self,
        data: Any,
        *,
        isosurfaces: Optional[Sequence[float]] = None,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Any:
        """Extract and render isosurfaces (shells of constant value) from a 3-D field as a layer.

        Args:
            data: A 3-D numpy cube, or a pyramids ``DatasetCollection`` whose ``.values`` is a 3-D stack.
            isosurfaces: Iso-values to extract; ``None`` lets PyVista pick 10 levels across the data range.
            cmap: Colormap for the extracted surfaces.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered :class:`pyvista.Actor` for the isosurface mesh.

        Examples:
            - Two iso-shells of a Gaussian blob:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> ax = np.linspace(-2, 2, 16)
                >>> xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
                >>> cube = np.exp(-(xx**2 + yy**2 + zz**2))
                >>> scene = Scene3D(off_screen=True)
                >>> actor = scene.isosurface(cube, isosurfaces=[0.3, 0.6])
                >>> scene.layers[0][0].n_points > 0
                True
                >>> scene.close()

                ```
        """
        grid = _point_grid(_cube(data))
        contour_kwargs = {} if isosurfaces is None else {"isosurfaces": list(isosurfaces)}
        mesh = grid.contour(scalars=FIELD, **contour_kwargs)
        return self.add_mesh(mesh, scalars=FIELD, cmap=cmap, **kwargs)
