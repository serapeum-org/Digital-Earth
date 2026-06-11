"""TerrainMixin — render a raster/DEM as 3-D relief on a :class:`Scene3D`.

Turns a pyramids raster into a PyVista ``StructuredGrid`` whose z is the (exaggerated) elevation, so a DEM
becomes a true 3-D surface. Built from the uniform :class:`~digitalearth.sources.Source` (numpy z + 1-D x/y
coordinate vectors + CRS) — **no xarray/rasterio**; CRS/reproject stays in pyramids.

Using a ``StructuredGrid`` from pyramids' real x/y cell-centre coordinates (rather than an axis-aligned
``ImageData``) keeps the surface correctly oriented for north→south-descending rasters and naturally supports
non-uniform spacing. The one subtlety VTK imposes: scalars/elevation attach in **Fortran order**
(``ravel(order="F")``) to line up with the structured point ordering — C-order silently mirrors the terrain.
"""
from typing import Any, Optional

import numpy as np
import pyvista as pv

from digitalearth.sources import Source, get_source

#: Attribute name the elevation scalar is stored under on the generated mesh.
ELEVATION = "elevation"

#: Mean metres per degree of latitude (WGS84) — used to bring a geographic DEM's metre-valued
#: elevation into the same (degree) units as its lon/lat coordinates, so the surface isn't a needle.
_METRES_PER_DEGREE = 111_320.0


def _vertical_unit_scale(crs: Any) -> float:
    """Return the factor that converts metre-valued elevation into the horizontal coordinate units.

    For a **projected** CRS the horizontal coordinates are already metres, so elevation needs no rescaling
    (``1.0``). For a **geographic** CRS the coordinates are degrees while elevation is metres — left unscaled the
    surface is ~100 000× taller than it is wide (an invisible vertical needle), so elevation is divided by the
    mean metres-per-degree (:data:`_METRES_PER_DEGREE`). CRS interpretation goes through pyramids (the GIS engine);
    an unknown/unparseable CRS falls back to ``1.0`` (treat as already-consistent units).

    Args:
        crs: The :class:`~digitalearth.sources.Source` CRS (an EPSG int, or anything pyramids can resolve).

    Returns:
        float: ``1 / _METRES_PER_DEGREE`` for a geographic CRS, else ``1.0``.
    """
    try:
        from pyramids.base.crs import sr_from_epsg

        if sr_from_epsg(int(crs)).IsGeographic():
            return 1.0 / _METRES_PER_DEGREE
    except Exception:  # noqa: BLE001 — any CRS-resolution failure: fall back to no rescaling.
        pass
    return 1.0


def _terrain_mesh(z: np.ndarray, x: np.ndarray, y: np.ndarray, z_exaggeration: float) -> pv.StructuredGrid:
    """Build a ``StructuredGrid`` surface from a 2-D elevation array and 1-D coordinate vectors.

    Args:
        z: 2-D elevation array of shape ``(ny, nx)`` (may contain ``NaN`` for masked nodata).
        x: 1-D x/longitude cell-centre coordinates, length ``nx``.
        y: 1-D y/latitude cell-centre coordinates, length ``ny`` (descending for north→south rasters is fine).
        z_exaggeration: Vertical scale factor applied to the elevation (``1.0`` = true scale).

    Returns:
        pyvista.StructuredGrid: a surface mesh carrying the elevation under the :data:`ELEVATION` scalar.
    """
    z = np.asarray(z, dtype="float64")
    xx, yy = np.meshgrid(np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"))
    zz = np.nan_to_num(z, nan=float(np.nanmin(z)) if np.isfinite(z).any() else 0.0) * z_exaggeration
    grid = pv.StructuredGrid(xx, yy, zz)
    # VTK structured points are Fortran-ordered: ravel(order="F") keeps the terrain right-side up (see module docs).
    grid.point_data[ELEVATION] = z.ravel(order="F")
    return grid


class TerrainMixin:
    """Adds :meth:`terrain` — render a DEM/raster as 3-D relief — to a :class:`Scene3D`."""

    def terrain(
        self,
        data: Any,
        *,
        band: int = 1,
        z_exaggeration: float = 1.0,
        cmap: str = "terrain",
        scalars: Optional[str] = ELEVATION,
        **kwargs: Any,
    ) -> Any:
        """Render a raster/DEM as a 3-D relief surface and register it as a layer.

        The raster is read through pyramids (via :func:`~digitalearth.sources.get_source` — numpy + coords +
        CRS, no xarray), turned into a warped ``StructuredGrid``, and added to the plotter. Colour by elevation
        (default) or pass ``scalars=None`` to colour by something else / a uniform colour.

        Args:
            data: A pyramids ``Dataset`` (or anything :func:`get_source` accepts), or an already-built
                :class:`~digitalearth.sources.Source`, holding the elevation raster.
            band: 1-based band index to read.
            z_exaggeration: Vertical exaggeration of the relief (``1.0`` = true scale; ``>1`` accentuates terrain).
            cmap: Matplotlib/colorcet colormap name for the elevation surface.
            scalars: Scalar array to colour by (default the elevation); ``None`` for a flat colour.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh` (``opacity``, ``show_edges``, ``pbr`` …).

        Returns:
            The registered :class:`pyvista.Actor` for the terrain surface.

        Examples:
            - A ramped DEM renders as a non-flat, correctly-oriented surface:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.sources import get_source
                >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
                >>> scene = Scene3D(off_screen=True)
                >>> actor = scene.terrain(get_source(dem), z_exaggeration=3.0)
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```
        """
        src = data if isinstance(data, Source) else get_source(data, band=band)
        # Geographic DEMs carry lon/lat (degrees) horizontally but metre elevation vertically; rescale the
        # vertical so true-scale (z_exaggeration=1.0) is a faithful, *visible* surface rather than a needle.
        vertical_scale = z_exaggeration * _vertical_unit_scale(src.crs)
        mesh = _terrain_mesh(src.z.values, src.x.values, src.y.values, vertical_scale)
        return self.add_mesh(mesh, scalars=scalars, cmap=cmap, **kwargs)
