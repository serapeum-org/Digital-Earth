"""VectorMixin — 3-D vector-field glyphs and extruded-polygon footprints.

Two vector visualizations:

- :meth:`vectors` — arrow **glyphs** oriented and scaled by a ``(u, v, w)`` field at a set of points
  (``mesh.glyph``); the classic flow/wind arrows in 3-D.
- :meth:`extruded_polygons` — turn polygon footprints into **3-D prisms** (``triangulate().extrude(...)``), the
  building-block of extruded-building / choropleth-relief views. A footprint with holes (interior rings) is
  carved all the way through (#199), so a ring courtyard or a lake in a landmass is a shaft, not a filled column.

Polygons come from the GeoDataFrame pyramids returns (a ``FeatureCollection``, which *is* a GeoDataFrame, or
``Dataset.get_cell_polygons()``); only their coordinates are read (``geom.exterior.coords``,
``geom.interiors``, ``geom.geoms``), so this module imports neither shapely nor geopandas (the HARD RULE /
``test_no_competitor_imports`` guard). CRS work stays in pyramids.
"""

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from digitalearth.base.spec import Encoding, LayerSpec, Scale
from digitalearth.three_d.base import classified_scalars
from digitalearth.three_d.layer import drawing_props

#: Attribute name the vector array is stored under for glyph orientation/scaling.
VECTORS = "vectors"
#: Attribute name the magnitude / per-prism value scalar is stored under.
MAGNITUDE = "magnitude"
VALUE = "value"


def _polygon_parts(geom: Any) -> Iterator[tuple[np.ndarray, list[np.ndarray]]]:
    """Yield each polygon part as its exterior ring and its interior (hole) rings.

    Handles ``Polygon`` (one part) and ``MultiPolygon`` (one part per member), duck-typed on the public
    ``.exterior`` / ``.interiors`` / ``.geoms`` attributes so neither shapely nor geopandas is imported (the
    HARD RULE). This is the hole-aware reader the extrusion draws from; :func:`_exterior_rings` is the
    exterior-only view kept for the callers (and tests) that never cared about holes.

    Args:
        geom: A shapely ``Polygon``/``MultiPolygon`` (read via its public attributes, never imported).

    Yields:
        ``(exterior, interiors)`` per part — the ``(M, 2)`` exterior ring and a (possibly empty) list of
        ``(K, 2)`` interior-ring arrays, one per hole.

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
        exterior = np.asarray(part.exterior.coords, dtype="float64")
        interiors = [
            np.asarray(ring.coords, dtype="float64") for ring in part.interiors
        ]
        yield exterior, interiors


def _exterior_rings(geom: Any) -> Iterator[np.ndarray]:
    """Yield each polygon part's exterior ring as an ``(M, 2)`` coordinate array (duck-typed; no shapely).

    The exterior-only view over :func:`_polygon_parts`; holes are read and carved by the extrusion itself
    (:func:`_extrude_polygon`), not here.

    Args:
        geom: A shapely ``Polygon``/``MultiPolygon`` (read via its public attributes, never imported).

    Yields:
        numpy.ndarray: the ``(M, 2)`` exterior-ring coordinates of each polygon part.

    Raises:
        TypeError: if ``geom`` is not a ``Polygon`` or ``MultiPolygon`` (e.g. a ``Point``/``LineString``).
    """
    for exterior, _interiors in _polygon_parts(geom):
        yield exterior


def _extrude_ring(ring: np.ndarray, height: float) -> "pv.PolyData":
    """Extrude a flat ``(M, 2)`` polygon ring into a capped 3-D prism of the given height.

    The solid-footprint path: a part with no holes is extruded straight from its one ring, so the common case
    keeps the exact geometry it always had. A part *with* holes goes through :func:`_extrude_polygon` instead.

    Args:
        ring: ``(M, 2)`` exterior-ring coordinates (the first/last point may repeat).
        height: Extrusion height along +z.

    Returns:
        pyvista.PolyData: the solid prism (triangulated cap + walls).
    """
    import pyvista as pv

    z = np.zeros((len(ring), 1))
    face = pv.PolyData(np.hstack([ring, z]), faces=np.r_[len(ring), range(len(ring))])
    return face.triangulate().extrude((0.0, 0.0, float(height)), capping=True)


def _ring_loops(rings: list[np.ndarray]) -> "pv.PolyData":
    """Build one ``PolyData`` of closed polyline loops from a list of ``(M, 2)`` rings.

    The input :func:`_cap_with_holes` hands to the contour triangulator: every ring becomes a closed polyline
    cell, so the triangulator sees the exterior and each hole as the nested contours it fills by the even-odd
    rule.

    Args:
        rings: The rings to turn into loops — the exterior first, then each interior.

    Returns:
        pyvista.PolyData: the loops, all at ``z=0``.
    """
    import pyvista as pv

    points: list[np.ndarray] = []
    lines: list[int] = []
    offset = 0
    for ring in rings:
        # Drop a repeated closing vertex; the loop is closed by pointing the last line segment back at the
        # first index, so a duplicated first/last point would otherwise be a zero-length edge.
        coords = ring[:-1] if len(ring) > 1 and np.allclose(ring[0], ring[-1]) else ring
        count = len(coords)
        points.append(np.column_stack([coords, np.zeros(count)]))
        lines.extend([count + 1, *range(offset, offset + count), offset])
        offset += count
    return pv.PolyData(np.vstack(points), lines=np.array(lines))


def _cap_with_holes(exterior: np.ndarray, interiors: list[np.ndarray]) -> "pv.PolyData":
    """Triangulate a polygon-with-holes into a flat cap at ``z=0``, with each interior ring carved out.

    Uses VTK's ``vtkContourTriangulator`` — the one filter that triangulates a set of nested closed contours
    by the even-odd rule, so a hole inside the exterior is removed. ``PolyData.triangulate()`` and
    ``delaunay_2d`` both fill the holes instead (measured), which is the #199 bug. ``vtkmodules`` is the render
    engine PyVista is built on, not a GIS dependency, so importing this one filter is within the tier's rule.

    Args:
        exterior: The ``(M, 2)`` exterior ring.
        interiors: The ``(K, 2)`` interior (hole) rings; never empty when this is called.

    Returns:
        pyvista.PolyData: the triangulated planar cap with the holes carved out.
    """
    import pyvista as pv
    from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator

    triangulator = vtkContourTriangulator()
    triangulator.SetInputData(_ring_loops([exterior, *interiors]))
    triangulator.Update()
    return pv.wrap(triangulator.GetOutput())


def _extrude_polygon(
    exterior: np.ndarray, interiors: list[np.ndarray], height: float
) -> "pv.PolyData":
    """Extrude a polygon **with holes** into a capped prism whose holes pass all the way through.

    The cap is triangulated with its holes carved (:func:`_cap_with_holes`); extruding that surface with
    ``capping=True`` sweeps both the exterior boundary and every hole boundary into walls, so each hole is a
    shaft through the solid rather than a filled column (#199).

    Args:
        exterior: The ``(M, 2)`` exterior ring.
        interiors: The ``(K, 2)`` interior (hole) rings.
        height: Extrusion height along +z.

    Returns:
        pyvista.PolyData: the solid prism with the holes carved through it.
    """
    return _cap_with_holes(exterior, interiors).extrude(
        (0.0, 0.0, float(height)), capping=True
    )


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    import pyvista as pv

    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _classify_or_refuse(
    colours: np.ndarray | None,
    *,
    scheme: str | None,
    k: int,
    cmap: Any,
    pinned: dict[str, Any],
    scale: Scale | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Cut `colours` into classes once, per feature, and refuse a colour keyword the scheme owns.

    Classifying before the prisms are built lets the per-cell array be filled straight from the result, so a
    label column (``scheme="categorical"``) and a column carrying NaN both work — neither survives the
    round-trip through `float()` that re-keying the raw values required.

    Args:
        colours: The value column to classify, or `None` when the caller asked for no classification.
        scheme: Classification scheme passed through to `classified_scalars`.
        k: Number of classes.
        cmap: Colormap the classes are drawn from.
        pinned: The caller's remaining keyword arguments, checked for a collision with the scheme's own.
        scale: The classification the layer's colour encoding already carries, when it has one.
            `_color_encoding` cut this very column with this very `scheme`/`k` to describe the layer, so
            cutting it again here is a second full pass for a result that is by construction identical.
            :func:`~digitalearth.three_d.base.classified_scalars` reads it only as far as it answers this
            request — a `scheme`/`k`/`cmap` the description was not cut with is cut here — so the saving
            cannot become a veto over a restyle (review R2-L11, R2-H3).

    Returns:
        A ``(style, scalars)`` pair: the colour keywords to forward to `add_mesh`, and the per-feature class
        indices — both empty/`None` when `colours` is `None`.

    Raises:
        TypeError: when a keyword the scheme *derives* (`clim` / `n_colors`) was also passed by the caller.
            `nan_color` is not one of them: it is the colour missing data is drawn in, which the classifier
            fills from a shared default rather than computing from the classes, so a caller's own is honoured
            — the same rule :func:`~digitalearth.three_d.point_cloud._refuse_derived_colours` follows, and one
            builder refusing what its neighbour honours is the asymmetry (review R2-L2).
    """
    if colours is None:
        return {}, None
    style = classified_scalars(colours, scheme=scheme, k=k, cmap=cmap, scale=scale)
    scalars = np.asarray(style.pop("scalars"), dtype=float)
    chosen_nan_color = pinned.pop("nan_color", None)
    clashing = sorted(set(style) & set(pinned))
    if clashing:
        # `**style, **kwargs` into one call is a "got multiple values" TypeError about Python's internals;
        # the caller's real mistake is pinning a colour setting the scheme owns, so say that instead
        # (review L1). The scheme's values cannot lose: `clim`/`n_colors` are the class indices the scalars
        # were rewritten to, and overriding them re-colours the wrong classes.
        names = ", ".join(f"{name}=" for name in clashing)
        raise TypeError(
            f"extruded_polygons() got {names} together with scheme={scheme!r}, which sets "
            f"{names} from the classes it cut; drop it, or drop scheme="
        )
    if chosen_nan_color is not None:
        style["nan_color"] = chosen_nan_color
    return style, scalars


def _finite_height(heights: np.ndarray | None, height: Any, index: int) -> float:
    """The extrusion height for one feature, refusing a non-finite one by name.

    Args:
        heights: The height column as an array, or `None` when every feature shares one height.
        height: The caller's `height=` argument — a column name when `heights` is given, else a number.
        index: Position of the feature in the collection, used to name the offending row.

    Returns:
        The finite height to extrude this feature to.

    Raises:
        ValueError: when the height is NaN or infinite. Such a prism has all-NaN z coordinates: it vanishes
            from the render with no error while the actor still reports cells (review L2).
    """
    resolved = float(heights[index]) if heights is not None else float(height)
    if math.isfinite(resolved):
        return resolved
    where = (
        f"row {index} of column {height!r}"
        if heights is not None
        else f"height={height!r}"
    )
    raise ValueError(
        f"extruded_polygons() needs a finite extrusion height; {where} is {resolved}"
    )


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
        name: Any = None,
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
        return self._add_described_layer(
            kind="vectors",
            data=points,
            name=name,
            vectors=vectors,
            factor=factor,
            cmap=cmap,
            **kwargs,
        )

    def extruded_polygons(
        self,
        gdf: Any,
        *,
        name: Any = None,
        height: float | str = 1.0,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Any:
        """Extrude polygon footprints into 3-D prisms and register them as a single layer.

        The footprints are placed in the scene's display CRS — setting it when the scene has none, reprojected
        through pyramids when they are in another.

        Args:
            gdf: A GeoDataFrame of ``Polygon``/``MultiPolygon`` geometries (e.g. a pyramids
                ``FeatureCollection`` / ``get_cell_polygons()``); coordinates are read by duck-typing (no
                geopandas/shapely import).
            height: Uniform extrusion height (``float``), or the name of a column (``str``) to read a
                per-feature height from. Every height must be finite: a ``NaN`` one builds a prism whose
                coordinates are all ``NaN``, which draws nothing while still counting as cells on the
                actor, so it is refused by row rather than rendered as an invisible building.
            column: Optional attribute column to colour the prisms by; ``None`` for a flat colour.
            scheme: How ``column`` is classified. ``None`` (the default) is a continuous ramp over the raw
                values; ``"categorical"`` gives every distinct value its own colour; any other
                ``cleopatra.styling.styles.classify`` scheme name (``"quantiles"``, ``"equal_interval"``,
                ``"fisher_jenks"``, …) colours ``k`` graduated classes — computed exactly as the 2-D tiers
                compute them, so one ``scheme``/``k`` pair means one set of classes on every tier.
            k: Number of classes for a graduated ``scheme``; ignored otherwise.
            cmap: Colormap used when colouring by ``column``.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`. A classified layer derives ``clim`` and
                ``n_colors`` from the classes it cut, so passing one of those *and* a ``scheme`` is a
                ``TypeError`` naming the keyword rather than a "multiple values" error out of Python's call
                machinery. ``nan_color`` is the exception and is honoured: the colour missing data is drawn in
                is a choice the classifier only fills a default for, so a caller's own wins, as ``cmap`` does.

        Returns:
            The registered :class:`pyvista.Actor` for the merged prism mesh, or ``None`` when there was
            nothing to extrude (see ``strict`` on :class:`~digitalearth.three_d.base.Scene3DBase`).

        Raises:
            ValueError: if ``scheme`` cannot classify ``column``, or if an extrusion height is not finite.
                Also — only when the scene was built with ``strict=True`` —
                :class:`~digitalearth.base.crs.OffLimbError` when no polygon rings were found, which is
                otherwise skipped with a warning.
            TypeError: if ``scheme`` is combined with a colour keyword the scheme itself sets
                (``clim`` / ``n_colors`` / ``nan_color``).

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
        return self._add_described_layer(
            kind="extrusion",
            data=gdf,
            name=name,
            # The prisms' colour comes from `column`, so a key over this layer explains that column and is
            # titled after it (order 24). No column is a flat fill, which publishes nothing: there is nothing
            # for a key to label. The scale carries the classes the drawer cuts, from the same classifier, so
            # a legend derived from it cannot disagree with the fill.
            encodings=_color_encoding(gdf, column, scheme, k, cmap),
            height=height,
            column=column,
            scheme=scheme,
            k=k,
            cmap=cmap,
            **kwargs,
        )


def _color_encoding(
    gdf: Any, column: str | None, scheme: Any, k: int, cmap: Any
) -> dict[str, Encoding] | None:
    """Return the colour encoding an extrusion publishes, or `None` for a flat-filled one.

    Args:
        gdf: The polygons, read for `column` when one was named.
        column: The attribute column the fill is coloured by, or `None`.
        scheme: How the values are classified, or `None` for a continuous ramp.
        k: How many classes a graduated scheme cuts.
        cmap: The colormap the classes come from.

    Returns:
        `{"color": Encoding}` naming `column`, with the scale its values are cut into, or `None` when the
        layer colours by nothing. The scale is `None` where the column cannot be reached or classified: the
        drawer reads the same column moments later and raises the tier's own message for it (see
        :func:`~digitalearth.three_d.guides.color_scale`).
    """
    from digitalearth.three_d.guides import color_scale

    if not column:
        return None
    try:
        values = np.asarray(gdf[column])
    # A column these polygons do not carry, or an input that cannot be subscripted. The drawer reads the same
    # column through the placed features and raises there.
    except Exception:  # noqa: BLE001
        return None
    return {
        "color": Encoding.by_field(
            "color", column, scale=color_scale(values, scheme=scheme, k=k, cmap=cmap)
        )
    }


def draw_vectors(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the arrow glyphs a `vectors` layer describes.

    Args:
        scene: The scene being drawn into.
        data: The layer's source object: the `(N, 3)` anchor points.
        layer: The layer's description, whose props carry the vectors, the arrow scale and the colormap.

    Returns:
        The `(glyph, actor)` pair, or `None` when the field had no points and the scene is not `strict`.

    Raises:
        ValueError: if the points and the vectors do not have the same shape.
        OffLimbError: when the field is empty and the scene is `strict`.
    """
    props = drawing_props(layer.symbology.props)
    factor = props.pop("factor", 1.0)
    pts = np.asarray(data, dtype="float64")
    vec = np.asarray(props.pop("vectors"), dtype="float64")
    if pts.shape != vec.shape:
        raise ValueError(
            f"points and vectors must have the same shape, got {pts.shape} and {vec.shape}"
        )
    if pts.size == 0:
        scene._skip_empty("vectors", "the field has no points")
        return None
    import pyvista as pv

    cloud = pv.PolyData(pts)
    cloud[VECTORS] = vec
    cloud[MAGNITUDE] = np.linalg.norm(vec, axis=1)
    glyph = cloud.glyph(orient=VECTORS, scale=MAGNITUDE, factor=factor)
    return glyph, scene.plotter.add_mesh(glyph, scalars=MAGNITUDE, **props)


def draw_extruded_polygons(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the prisms an `extrusion` layer describes.

    Args:
        scene: The scene being drawn into.
        data: The layer's source object: the polygons to extrude.
        layer: The layer's description, whose props carry the height, the colour column and the classification.

    Returns:
        The `(mesh, actor)` pair, or `None` when there was nothing to extrude and the scene is not `strict`.

    Raises:
        ValueError: if the scheme cannot classify the column, or a pinned colour keyword clashes with it.
        OffLimbError: when there are no polygons and the scene is `strict`.
    """
    props = drawing_props(layer.symbology.props)
    height = props.pop("height", 1.0)
    column = props.pop("column", None)
    scheme = props.pop("scheme", None)
    k = props.pop("k", 5)
    cmap = props.pop("cmap", "viridis")
    gdf = scene._place(data, layer="extruded_polygons")
    geoms = gdf.geometry
    heights = gdf[height].to_numpy() if isinstance(height, str) else None
    colours = gdf[column].to_numpy() if column else None

    # Classify once, per feature, *before* building the prisms. The per-cell array is then filled
    # straight from the result, so a label column ("categorical") and a column carrying NaN both work:
    # neither survives a round-trip through float(), which is what re-keying the raw values required.
    # "Once" counts the builder's cut too: the layer's colour encoding already carries the classification
    # `_color_encoding` made of this column, so handing it over is what makes the tier's one-computation
    # claim true of the extrusion and not only of the point cloud (review R2-L11).
    encoding = layer.symbology.encoding("color")
    style, scalars = _classify_or_refuse(
        colours,
        scheme=scheme,
        k=k,
        cmap=cmap,
        pinned=props,
        scale=None if encoding is None else encoding.scale,
    )

    import pyvista as pv

    prisms: list[pv.PolyData] = []
    for i, geom in enumerate(geoms):
        h = _finite_height(heights, height, i)
        for exterior, interiors in _polygon_parts(geom):
            # A solid footprint keeps the exact geometry it always had; a footprint with holes is carved
            # through (#199) rather than filled.
            prism = (
                _extrude_ring(exterior, h)
                if not interiors
                else _extrude_polygon(exterior, interiors, h)
            )
            if scalars is not None:
                prism.cell_data[VALUE] = np.full(prism.n_cells, scalars[i])
            prisms.append(prism)

    if not prisms:
        scene._skip_empty("extruded_polygons", "no polygon geometries to extrude")
        return None

    merged = pv.MultiBlock(prisms).combine()
    if colours is None:
        return merged, scene.plotter.add_mesh(merged, scalars=None, cmap=cmap, **props)
    return merged, scene.plotter.add_mesh(merged, scalars=VALUE, **style, **props)
