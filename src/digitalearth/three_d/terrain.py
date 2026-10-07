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
from digitalearth.base.spec import DEFAULT_BAND, Encoding, LayerSpec, Selection
from digitalearth.three_d.bigdata import DEFAULT_CELL_BUDGET, reduce_surface
from digitalearth.three_d.layer import drawing_props

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
    nodata = ~np.isfinite(z)
    # A nodata node still needs a finite coordinate or VTK cannot build the structured grid at all; it is
    # given the surface floor so the geometry is well-formed, then **blanked** below so it is never drawn.
    # The floor is only read when at least one cell is finite — `draw_terrain` skips an all-nodata raster
    # before it reaches here — but the guard is kept so the helper is safe to call directly.
    floor = float(np.nanmin(z)) if np.isfinite(z).any() else 0.0
    zz = np.where(nodata, floor, z) * vertical_scale
    import pyvista as pv

    grid = pv.StructuredGrid(xx, yy, zz)
    # VTK structured points are Fortran-ordered: ravel(order="F") keeps the terrain right-side up (see module docs).
    grid.point_data[ELEVATION] = z.ravel(order="F")
    if nodata.any():
        # #200: nodata is a **gap**, not fabricated ground at the floor. Blanking the nodata nodes hides every
        # cell that touches one — the VTK equivalent of geovista's `extract_points(adjacent_cells=False)` — so
        # the surface has a hole there rather than a flat sheet at `floor`, while the mesh stays a
        # `StructuredGrid` (the shape `bigdata.reduce_surface` counts without triangulating). The mask is
        # ravelled in the same Fortran order as the points and the elevation scalar, so it lines up with both.
        grid.hide_points(nodata.ravel(order="F"))
    return grid


def _color_encoding(scalars: Any, kwargs: dict[str, Any]) -> dict[str, Encoding] | None:
    """Return the colour encoding a terrain publishes, or `None` for a genuinely flat surface.

    Args:
        scalars: The `scalars` the builder was given.
        kwargs: The builder's remaining keywords, read for the one that turns scalar mapping off.

    Returns:
        `{"color": Encoding}` naming the array the surface is coloured by, or `None` when it is coloured
        flat — in which case a key over it is refused rather than drawn over nothing.

        **`scalars=None` is not a flat surface on its own.** :meth:`pyvista.Plotter.add_mesh` falls back to
        the mesh's active scalars, which is always the elevation :func:`_terrain_mesh` binds, and draws its own
        bar titled `elevation` for it — measured on PyVista 0.48.4. The surface is flat only when `color=` is
        given as well, which is the keyword that switches scalar mapping off; `color=` on its own is ignored
        while scalars are active. A `scalars` that is neither a name nor `None` publishes nothing either:
        there is no name for a key to be titled after.
    """
    if isinstance(scalars, str) and scalars:
        return {"color": Encoding.by_field("color", scalars)}
    if scalars is not None or "color" in kwargs:
        return None
    return {"color": Encoding.by_field("color", ELEVATION)}


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
        name: Any = None,
        band: int = DEFAULT_BAND,
        z_exaggeration: float | None = None,
        cmap: str | None = None,
        scalars: str | None = ELEVATION,
        big_data_threshold: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a raster/DEM as a 3-D relief surface and register it as a layer.

        The raster is read through pyramids (via :func:`~digitalearth.base.sources.get_source` — numpy + coords +
        CRS, no xarray), turned into a ``StructuredGrid``, and added to the plotter. Colour by elevation
        (default) or pass ``scalars=None`` to colour by something else / a uniform colour.

        The raster is placed in the scene's display CRS: it sets that CRS when the scene has none, and is
        reprojected through pyramids when it is in another.

        **The surface spans the cell centres, not the cell edges.** One height is set per node, at the middle
        of each cell, which is what a surface through sampled elevations means — so the mesh is half a cell
        narrower than the raster's own extent on each side. The 2-D tiers draw the cells themselves and cover
        that extent (#301); whether a draped surface should too is decided with draping (#202).

        Metre elevations over a geographic scene are converted to degrees by the scene's CRS, not the
        raster's.

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
            big_data_threshold: Cells above which the surface is simplified with ``decimate_pro`` before it is
                drawn (#207). ``None`` (the default) uses the scene's
                :attr:`~digitalearth.three_d.base.Scene3DBase.big_data_threshold`; a DEM at or under the budget
                is drawn exactly as it was built. Measured: a 708x708 DEM is drawn whole, a 709x709 one is
                reduced, and a 1000x1000 one goes from a 39.0 MB ``export_html`` page to a 12.3 MB one — see
                :mod:`digitalearth.three_d.bigdata` for the table the budget comes from.
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
                reprojected; or if ``big_data_threshold`` is negative or not a whole number of cells.

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
        if z_exaggeration is not None:
            self.vertical_exaggeration = z_exaggeration
        return self._add_described_layer(
            kind="terrain",
            data=data,
            name=name,
            selection=Selection(band=(band,)),
            # The array the surface's colour comes from, so a key over this layer explains that array and is
            # titled after it (order 24). No scale: VTK holds the mapping from the array's own range through
            # its lookup table, which is what `scale=None` means.
            encodings=_color_encoding(scalars, kwargs),
            cmap=cmap,
            scalars=scalars,
            big_data_threshold=self._resolve_big_data_threshold(
                big_data_threshold, caller="Scene3D.terrain()"
            ),
            **kwargs,
        )


def draw_terrain(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the surface a `terrain` layer describes and add it to the scene's plotter.

    Args:
        scene: The scene being drawn into — its display CRS places the raster and its `strict` policy decides
            what an empty one does.
        data: The layer's source object: a pyramids raster, a `Source`, or a 2-D array.
        layer: The layer's description. Its selection names the band; its symbology's props carry `cmap`,
            `scalars` and whatever else was passed to PyVista.

    Returns:
        The `(mesh, actor)` pair, or `None` when the raster held no finite elevation and the scene is not
        `strict`.

    Raises:
        OffLimbError: when the raster holds nothing to draw and the scene is `strict`.
        ValueError: if `data` is a `Source` in a CRS other than the scene's.
    """
    props = drawing_props(layer.symbology.props)
    cmap = props.pop("cmap", None)
    budget = int(props.pop("big_data_threshold", DEFAULT_CELL_BUDGET))
    band = layer.selection.band[0] if layer.selection.band else 1
    placed = scene._place(data, layer="terrain")
    src = placed if isinstance(placed, Source) else get_source(placed, band=band)
    elevation = np.asarray(src.z.values, dtype="float64")
    if elevation.size == 0 or not np.isfinite(elevation).any():
        # A grid with no finite cell has no surface: PyVista would still build a mesh from it and render
        # a blank sheet at z=0, which reads as "the terrain is flat here" rather than "there is no data".
        scene._skip_empty("terrain", "the raster holds no finite elevation")
        return None
    # Geographic DEMs carry lon/lat (degrees) horizontally but metre elevation vertically; rescale the
    # vertical so true scale is a faithful, *visible* surface rather than a needle. That is a unit
    # conversion the scene's CRS dictates — the horizontal units every layer is drawn in — and the
    # layer's own only when the scene has none; exaggeration is the view scale, kept on the camera.
    units_crs = src.crs if scene.display_crs is None else scene.display_crs
    mesh = reduce_surface(
        _terrain_mesh(
            src.z.values, src.x.values, src.y.values, _vertical_unit_scale(units_crs)
        ),
        budget,
        kind="terrain",
    )
    return mesh, scene.plotter.add_mesh(
        mesh, cmap=scene._auto_cmap(src, cmap, fallback="terrain"), **props
    )
