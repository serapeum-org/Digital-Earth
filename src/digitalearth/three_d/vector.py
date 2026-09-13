"""VectorMixin — 3-D vector-field glyphs and extruded-polygon footprints.

Two vector visualizations:

- :meth:`vectors` — arrow **glyphs** oriented and scaled by a ``(u, v, w)`` field at a set of points
  (``mesh.glyph``); the classic flow/wind arrows in 3-D.
- :meth:`extruded_polygons` — turn polygon footprints into **3-D prisms** (``triangulate().extrude(...)``), the
  building-block of extruded-building / choropleth-relief views.

Polygons come from the GeoDataFrame pyramids returns (a ``FeatureCollection``, which *is* a GeoDataFrame, or
``Dataset.get_cell_polygons()``); only their coordinates are read (``geom.exterior.coords``, ``geom.geoms``), so
this module imports neither shapely nor geopandas (the HARD RULE / ``test_no_competitor_imports`` guard). CRS work
stays in pyramids.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv

from digitalearth.three_d.base import classified_scalars

#: Attribute name the vector array is stored under for glyph orientation/scaling.
VECTORS = "vectors"
#: Attribute name the magnitude / per-prism value scalar is stored under.
MAGNITUDE = "magnitude"
VALUE = "value"


def _exterior_rings(geom: Any) -> Iterator[np.ndarray]:
    """Yield each polygon's exterior ring as an ``(M, 2)`` coordinate array (duck-typed; no shapely import).

    Handles ``Polygon`` (one ring) and ``MultiPolygon`` (one ring per part). Holes are ignored — exterior
    footprints only.

    Args:
        geom: A shapely ``Polygon``/``MultiPolygon`` (read via its public attributes, never imported).

    Yields:
        numpy.ndarray: the ``(M, 2)`` exterior-ring coordinates of each polygon part.

    Raises:
        TypeError: if ``geom`` is not a ``Polygon`` or ``MultiPolygon`` (e.g. a ``Point``/``LineString``).
    """
    geom_type = geom.geom_type
    if geom_type not in ("Polygon", "MultiPolygon"):
        raise TypeError(
            f"extruded_polygons expects Polygon/MultiPolygon geometries, got {geom_type}"
        )
    parts = geom.geoms if geom_type == "MultiPolygon" else [geom]
    for part in parts:
        yield np.asarray(part.exterior.coords, dtype="float64")


def _extrude_ring(ring: np.ndarray, height: float) -> pv.PolyData:
    """Extrude a flat ``(M, 2)`` polygon ring into a capped 3-D prism of the given height.

    Args:
        ring: ``(M, 2)`` exterior-ring coordinates (the first/last point may repeat).
        height: Extrusion height along +z.

    Returns:
        pyvista.PolyData: the solid prism (triangulated cap + walls).
    """
    z = np.zeros((len(ring), 1))
    face = pv.PolyData(np.hstack([ring, z]), faces=np.r_[len(ring), range(len(ring))])
    return face.triangulate().extrude((0.0, 0.0, float(height)), capping=True)


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class VectorMixin(_MixinBase):
    """Adds :meth:`vectors` and :meth:`extruded_polygons` to a :class:`Scene3D`.

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

    def vectors(
        self,
        points: np.ndarray,
        vectors: np.ndarray,
        *,
        factor: float = 1.0,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Any:
        """Render a 3-D vector field as arrow glyphs (oriented + scaled by magnitude) and register it.

        Args:
            points: ``(N, 3)`` glyph anchor positions.
            vectors: ``(N, 3)`` ``(u, v, w)`` vectors at each point.
            factor: Overall arrow length scale.
            cmap: Colormap applied to the arrow magnitudes.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered :class:`pyvista.Actor` for the glyph mesh, or ``None`` when there were no points
            to place a glyph at (see ``strict`` on :class:`~digitalearth.three_d.base.Scene3DBase`).

        Raises:
            ValueError: if ``points`` and ``vectors`` do not have the same shape. Also — only when the scene
                was built with ``strict=True`` — :class:`~digitalearth.base.crs.OffLimbError` for an empty
                field, which is otherwise skipped with a warning.

        Examples:
            - A uniform eastward field renders as same-length arrows coloured by magnitude:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> ax = np.linspace(0, 1, 6)
                >>> xx, yy = np.meshgrid(ax, ax)
                >>> pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
                >>> vec = np.column_stack([np.ones(pts.shape[0]), np.zeros(pts.shape[0]), np.zeros(pts.shape[0])])
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.vectors(pts, vec, factor=0.1)
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```
        """
        pts = np.asarray(points, dtype="float64")
        vec = np.asarray(vectors, dtype="float64")
        if pts.shape != vec.shape:
            raise ValueError(
                f"points and vectors must have the same shape, got {pts.shape} and {vec.shape}"
            )
        if pts.size == 0:
            self._skip_empty("vectors", "the field has no points")
            return None
        cloud = pv.PolyData(pts)
        cloud[VECTORS] = vec
        cloud[MAGNITUDE] = np.linalg.norm(vec, axis=1)
        glyph = cloud.glyph(orient=VECTORS, scale=MAGNITUDE, factor=factor)
        return self.add_mesh(glyph, scalars=MAGNITUDE, cmap=cmap, **kwargs)

    def extruded_polygons(
        self,
        gdf: Any,
        *,
        height: float | str = 1.0,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Any:
        """Extrude polygon footprints into 3-D prisms and register them as a single layer.

        Args:
            gdf: A GeoDataFrame of ``Polygon``/``MultiPolygon`` geometries (e.g. a pyramids
                ``FeatureCollection`` / ``get_cell_polygons()``); coordinates are read by duck-typing (no
                geopandas/shapely import).
            height: Uniform extrusion height (``float``), or the name of a column (``str``) to read a
                per-feature height from.
            column: Optional attribute column to colour the prisms by; ``None`` for a flat colour.
            scheme: How ``column`` is classified. ``None`` (the default) is a continuous ramp over the raw
                values; ``"categorical"`` gives every distinct value its own colour; any other
                ``cleopatra.styling.styles.classify`` scheme name (``"quantiles"``, ``"equal_interval"``,
                ``"fisher_jenks"``, …) colours ``k`` graduated classes — computed exactly as the 2-D tiers
                compute them, so one ``scheme``/``k`` pair means one set of classes on every tier.
            k: Number of classes for a graduated ``scheme``; ignored otherwise.
            cmap: Colormap used when colouring by ``column``.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered :class:`pyvista.Actor` for the merged prism mesh, or ``None`` when there was
            nothing to extrude (see ``strict`` on :class:`~digitalearth.three_d.base.Scene3DBase`).

        Raises:
            ValueError: if ``scheme`` cannot classify ``column``. Also — only when the scene was built with
                ``strict=True`` — :class:`~digitalearth.base.crs.OffLimbError` when no polygon rings were
                found, which is otherwise skipped with a warning.

        Examples:
            - Extrude two squares to different heights and colour by an attribute (needs geopandas at the
              call site — only ever as the type pyramids hands back):
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Polygon
                >>> from digitalearth.three_d import Scene3D
                >>> gdf = gpd.GeoDataFrame(
                ...     {"pop": [10.0, 20.0]},
                ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                ...               Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])],
                ... )
                >>> scene = Scene3D(off_screen=True)
                >>> actor = scene.extruded_polygons(gdf, height="pop", column="pop")
                >>> scene.layers[0][0].n_cells > 0
                True
                >>> scene.close()

                ```
            - Nothing to extrude is a skipped layer with a warning, not a dead scene — unless the scene was
              built ``strict=True``:
                ```python
                >>> import geopandas as gpd
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.crs import OffLimbError
                >>> empty = gpd.GeoDataFrame({"pop": []}, geometry=[])
                >>> scene = Scene3D(off_screen=True)
                >>> scene.extruded_polygons(empty) is None, len(scene.layers)
                (True, 0)
                >>> scene.close()
                >>> strict = Scene3D(off_screen=True, strict=True)
                >>> try:
                ...     strict.extruded_polygons(empty)
                ... except OffLimbError as error:
                ...     print(error)
                ... finally:
                ...     strict.close()
                extruded_polygons: no polygon geometries to extrude

                ```
        """
        geoms = gdf.geometry
        heights = gdf[height].to_numpy() if isinstance(height, str) else None
        colours = gdf[column].to_numpy() if column else None

        prisms: list[pv.PolyData] = []
        for i, geom in enumerate(geoms):
            h = float(heights[i]) if heights is not None else float(height)
            for ring in _exterior_rings(geom):
                prism = _extrude_ring(ring, h)
                if colours is not None:
                    prism.cell_data[VALUE] = np.full(prism.n_cells, float(colours[i]))
                prisms.append(prism)

        if not prisms:
            self._skip_empty("extruded_polygons", "no polygon geometries to extrude")
            return None
        merged = pv.MultiBlock(prisms).combine()
        if colours is None:
            return self.add_mesh(merged, scalars=None, cmap=cmap, **kwargs)
        # Classify against the per-feature column, then push each prism's class code onto its cells — the
        # per-cell VALUE array was filled with the raw value above, so it is re-keyed here rather than
        # classified cell-by-cell (which would re-derive the same breaks from a longer, duplicated array).
        style = classified_scalars(colours, scheme=scheme, k=k, cmap=cmap)
        codes = style.pop("scalars")
        raw_to_code = {float(raw): code for raw, code in zip(colours, codes)}
        merged.cell_data[VALUE] = np.array(
            [raw_to_code[float(value)] for value in merged.cell_data[VALUE]]
        )
        return self.add_mesh(merged, scalars=VALUE, **style, **kwargs)
