"""TerrainMixin — render a raster/DEM as 3-D relief on a :class:`Scene3D`.

Turns a pyramids raster into a PyVista ``StructuredGrid`` whose z is the elevation, so a DEM becomes a true 3-D
surface. Built from the uniform :class:`~digitalearth.base.sources.Source` (numpy z + 1-D x/y coordinate vectors
+ CRS) — **no xarray/rasterio**; CRS/reproject stays in pyramids. Vertical *exaggeration* is deliberately not
baked into those points: it is a scene-wide view scale
(:attr:`~digitalearth.three_d.base.Scene3DBase.vertical_exaggeration`), so layers cannot disagree about it and
it can be changed without rebuilding a mesh. The only vertical factor the geometry carries is the CRS unit
conversion (:func:`_vertical_unit_scale`).

Using a ``StructuredGrid`` from pyramids' real x/y cell-centre coordinates (rather than an axis-aligned
``ImageData``) keeps the surface correctly oriented for north→south-descending rasters and naturally supports
non-uniform spacing. The one subtlety VTK imposes: scalars/elevation attach in **Fortran order**
(``ravel(order="F")``) to line up with the structured point ordering — C-order silently mirrors the terrain.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from digitalearth.base.crs import is_geographic
from digitalearth.base.sources import Source, get_source

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
    mean metres-per-degree (:data:`_METRES_PER_DEGREE`). CRS interpretation goes through pyramids (the GIS engine,
    via :func:`~digitalearth.base.crs.is_geographic`, which reads every spelling the ``Source``
    contract allows); an unknown/unparseable CRS falls back to ``1.0``
    (treat as already-consistent units).

    This is a **unit conversion**, not vertical exaggeration: it is baked into the mesh because it is a property
    of that layer's CRS, whereas exaggeration is a property of the view
    (:attr:`~digitalearth.three_d.base.Scene3DBase.vertical_exaggeration`).

    Args:
        crs: The :class:`~digitalearth.base.sources.Source` CRS — an EPSG int, an ``"EPSG:<code>"``
            string, or a proj4/WKT definition.

    Returns:
        float: ``1 / _METRES_PER_DEGREE`` for a geographic CRS, else ``1.0``.
    """
    return 1.0 / _METRES_PER_DEGREE if is_geographic(crs) else 1.0


def _terrain_mesh(
    z: np.ndarray, x: np.ndarray, y: np.ndarray, vertical_scale: float
) -> "pv.StructuredGrid":
    """Build a ``StructuredGrid`` surface from a 2-D elevation array and 1-D coordinate vectors.

    Args:
        z: 2-D elevation array of shape ``(ny, nx)`` (may contain ``NaN`` for masked nodata).
        x: 1-D x/longitude cell-centre coordinates, length ``nx``.
        y: 1-D y/latitude cell-centre coordinates, length ``ny`` (descending for north→south rasters is fine).
        vertical_scale: Unit conversion applied to the elevation before it becomes geometry — see
            :func:`_vertical_unit_scale`. Vertical *exaggeration* is not applied here: it is a view scale
            (:attr:`~digitalearth.three_d.base.Scene3DBase.vertical_exaggeration`), so the points keep the
            elevation the data actually carries.

    Returns:
        pyvista.StructuredGrid: a surface mesh carrying the elevation under the :data:`ELEVATION` scalar.
    """
    z = np.asarray(z, dtype="float64")
    xx, yy = np.meshgrid(np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"))
    zz = (
        np.nan_to_num(z, nan=float(np.nanmin(z)) if np.isfinite(z).any() else 0.0)
        * vertical_scale
    )
    import pyvista as pv

    grid = pv.StructuredGrid(xx, yy, zz)
    # VTK structured points are Fortran-ordered: ravel(order="F") keeps the terrain right-side up (see module docs).
    grid.point_data[ELEVATION] = z.ravel(order="F")
    return grid


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    import pyvista as pv

    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class TerrainMixin(_MixinBase):
    """Adds :meth:`terrain` — render a DEM/raster as 3-D relief — to a :class:`Scene3D`.

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

    def terrain(
        self,
        data: Any,
        *,
        band: int = 1,
        z_exaggeration: float | None = None,
        cmap: str | None = None,
        scalars: str | None = ELEVATION,
        **kwargs: Any,
    ) -> Any:
        """Render a raster/DEM as a 3-D relief surface and register it as a layer.

        The raster is read through pyramids (via :func:`~digitalearth.base.sources.get_source` — numpy + coords +
        CRS, no xarray), turned into a ``StructuredGrid``, and added to the plotter. Colour by elevation
        (default) or pass ``scalars=None`` to colour by something else / a uniform colour.

        The raster is placed in the scene's display CRS: it sets that CRS when the scene has none, and is
        reprojected through pyramids when it is in another. Metre elevations over a geographic scene are
        converted to degrees by the scene's CRS, not the raster's.

        **Vertical exaggeration is a property of the scene, not of this layer.** ``z_exaggeration`` sets
        :attr:`~digitalearth.three_d.base.Scene3DBase.vertical_exaggeration`, a PyVista view scale that applies
        to every actor — so two terrains in one scene cannot disagree about it, the value can be read back off
        the scene, and changing it needs no mesh rebuild. Leaving it ``None`` keeps whatever the scene is
        already set to (``1.0``, true scale, on a fresh scene). The ``exaggeration`` property of the web tier's
        MapLibre terrain behaves the same way.

        Args:
            data: A pyramids ``Dataset`` (or anything :func:`get_source` accepts), or an already-built
                :class:`~digitalearth.base.sources.Source`, holding the elevation raster.
            band: 1-based band index to read.
            z_exaggeration: Vertical exaggeration to render the **scene** at (``1.0`` = true scale; ``>1``
                accentuates relief). ``None`` leaves the scene's current exaggeration untouched.
            cmap: Matplotlib/colorcet colormap name for the elevation surface. ``None`` (the default) resolves
                it from the variable through :func:`~digitalearth.base.autostyle.auto_style` — the same lookup
                the static, interactive and web tiers use, so a DEM/field is drawn in the same colours on all
                four — falling back to ``"terrain"`` only when the lookup yields nothing.
            scalars: Scalar array to colour by (default the elevation); ``None`` for a flat colour.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh` (``opacity``, ``show_edges``, ``pbr`` …).

        Returns:
            The registered :class:`pyvista.Actor` for the terrain surface, or ``None`` when the raster held
            no elevation to build a surface from (see ``strict`` on
            :class:`~digitalearth.three_d.base.Scene3DBase`).

        Raises:
            OffLimbError: only when the scene was built with ``strict=True`` and the raster is empty or
                entirely nodata; by default that layer is skipped with a warning instead, so one blank tile
                does not cost a composed scene its other layers.
            ValueError: if `data` is a `Source` in a CRS other than the scene's — extracted coordinates cannot be
                reprojected.

        Examples:
            - A ramped DEM renders as a non-flat, correctly-oriented surface, and the exaggeration asked for is
              readable back off the scene:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.sources import get_source
                >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
                >>> scene = Scene3D(off_screen=True)
                >>> actor = scene.terrain(get_source(dem), z_exaggeration=3.0)
                >>> len(scene.layers)
                1
                >>> scene.vertical_exaggeration
                3.0
                >>> scene.close()

                ```
            - An all-nodata tile is skipped with a warning rather than rendered as a flat blank sheet; under
              ``strict=True`` it raises instead:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.crs import OffLimbError
                >>> from digitalearth.base.sources import get_source
                >>> blank = get_source(np.full((4, 4), np.nan))
                >>> scene = Scene3D(off_screen=True)
                >>> scene.terrain(blank) is None, len(scene.layers)
                (True, 0)
                >>> scene.close()
                >>> strict = Scene3D(off_screen=True, strict=True)
                >>> try:
                ...     strict.terrain(blank)
                ... except OffLimbError as error:
                ...     print(error)
                ... finally:
                ...     strict.close()
                terrain: the raster holds no finite elevation

                ```
        """
        data = self._place(data, layer="terrain")
        src = data if isinstance(data, Source) else get_source(data, band=band)
        elevation = np.asarray(src.z.values, dtype="float64")
        if elevation.size == 0 or not np.isfinite(elevation).any():
            # A grid with no finite cell has no surface: PyVista would still build a mesh from it and render
            # a blank sheet at z=0, which reads as "the terrain is flat here" rather than "there is no data".
            self._skip_empty("terrain", "the raster holds no finite elevation")
            return None
        if z_exaggeration is not None:
            self.vertical_exaggeration = z_exaggeration
        # Geographic DEMs carry lon/lat (degrees) horizontally but metre elevation vertically; rescale the
        # vertical so true scale is a faithful, *visible* surface rather than a needle. That is a unit
        # conversion the scene's CRS dictates — the horizontal units every layer is drawn in — and the
        # layer's own only when the scene has none; exaggeration is the view scale set just above.
        units_crs = src.crs if self.display_crs is None else self.display_crs
        mesh = _terrain_mesh(
            src.z.values, src.x.values, src.y.values, _vertical_unit_scale(units_crs)
        )
        return self.add_mesh(
            mesh,
            scalars=scalars,
            cmap=self._auto_cmap(src, cmap, fallback="terrain"),
            **kwargs,
        )
