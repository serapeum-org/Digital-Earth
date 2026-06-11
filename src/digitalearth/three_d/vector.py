"""VectorMixin — 3-D vector-field glyphs and extruded-polygon footprints.

Two vector visualizations:

- :meth:`vectors` — arrow **glyphs** oriented and scaled by a ``(u, v, w)`` field at a set of points
  (``mesh.glyph``); the classic flow/wind arrows in 3-D.
- :meth:`extruded_polygons` — turn polygon footprints into **3-D prisms** (``triangulate().extrude(...)``), the
  building-block of extruded-building / choropleth-relief views.

Polygons come from the GeoDataFrame pyramids returns (``FeatureCollection.to_geodataframe()`` /
``Dataset.get_cell_polygons()``); only their coordinates are read (``geom.exterior.coords``, ``geom.geoms``), so
this module imports neither shapely nor geopandas (the HARD RULE / ``test_no_competitor_imports`` guard). CRS work
stays in pyramids.
"""
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import pyvista as pv

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
        raise TypeError(f"extruded_polygons expects Polygon/MultiPolygon geometries, got {geom_type}")
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


class VectorMixin:
    """Adds :meth:`vectors` and :meth:`extruded_polygons` to a :class:`Scene3D`."""

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
            The registered :class:`pyvista.Actor` for the glyph mesh.

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
            raise ValueError(f"points and vectors must have the same shape, got {pts.shape} and {vec.shape}")
        cloud = pv.PolyData(pts)
        cloud[VECTORS] = vec
        cloud[MAGNITUDE] = np.linalg.norm(vec, axis=1)
        glyph = cloud.glyph(orient=VECTORS, scale=MAGNITUDE, factor=factor)
        return self.add_mesh(glyph, scalars=MAGNITUDE, cmap=cmap, **kwargs)

    def extruded_polygons(
        self,
        gdf: Any,
        *,
        height: Union[float, str] = 1.0,
        column: Optional[str] = None,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Any:
        """Extrude polygon footprints into 3-D prisms and register them as a single layer.

        Args:
            gdf: A GeoDataFrame of ``Polygon``/``MultiPolygon`` geometries (e.g. ``to_geodataframe()`` /
                ``get_cell_polygons()``); coordinates are read by duck-typing (no geopandas/shapely import).
            height: Uniform extrusion height (``float``), or the name of a column (``str``) to read a
                per-feature height from.
            column: Optional attribute column to colour the prisms by; ``None`` for a flat colour.
            cmap: Colormap used when colouring by ``column``.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered :class:`pyvista.Actor` for the merged prism mesh.

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
        """
        geoms = gdf.geometry
        heights = gdf[height].to_numpy() if isinstance(height, str) else None
        colours = gdf[column].to_numpy() if column else None

        prisms: List[pv.PolyData] = []
        for i, geom in enumerate(geoms):
            h = float(heights[i]) if heights is not None else float(height)
            for ring in _exterior_rings(geom):
                prism = _extrude_ring(ring, h)
                if colours is not None:
                    prism.cell_data[VALUE] = np.full(prism.n_cells, float(colours[i]))
                prisms.append(prism)

        if not prisms:
            raise ValueError("extruded_polygons received no polygon geometries to extrude")
        merged = pv.MultiBlock(prisms).combine()
        scalars = VALUE if colours is not None else None
        return self.add_mesh(merged, scalars=scalars, cmap=cmap, **kwargs)
