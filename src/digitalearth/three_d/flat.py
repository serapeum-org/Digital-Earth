"""FlatMixin — the flat (2-D-on-3-D) Core builders: points, polygons, choropleth (TD-24/25/26).

The tier draws *positioned* geometry — a point with a z, a polygon extruded into a prism — but it also has to
answer the Core verbs for **flat** features drawn on the ground plane: ``points`` (markers, distinct from
``point_cloud``'s z-per-point), ``polygons`` (a filled footprint, distinct from ``extruded_polygons``' prism)
and ``choropleth`` (a polygon fill coloured by a column). All three lie at ``z=0`` so they read as a map laid
under the 3-D scene, and all three reuse the tier's existing machinery: the point cloud's coordinate readers
and classification for ``points``, and the vector module's hole-aware cap triangulation for the fills.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from digitalearth.base.spec import Encoding, LayerSpec
from digitalearth.three_d.base import classified_scalars
from digitalearth.three_d.bigdata import reduce_points
from digitalearth.three_d.layer import drawing_props
from digitalearth.three_d.point_cloud import (
    SCALAR,
    _coords_from_array,
    _coords_from_geodataframe,
)
from digitalearth.three_d.point_cloud import (
    _color_encoding as _point_encoding,
)
from digitalearth.three_d.vector import (
    VALUE,
    _cap_with_holes,
    _classify_or_refuse,
    _polygon_parts,
)
from digitalearth.three_d.vector import (
    _color_encoding as _polygon_encoding,
)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _only_when_given(**candidates: Any) -> dict[str, Any]:
    """Return the keyword arguments whose value is not ``None``.

    ``add_mesh``/``add_points`` read ``None`` for some keywords (``opacity``, ``color``) as "no value" rather
    than "use the default", so an unset optional is dropped rather than passed as ``None``.

    Args:
        **candidates: The optional keywords and their values.

    Returns:
        Only the ones that were actually given.
    """
    return {name: value for name, value in candidates.items() if value is not None}


class FlatMixin(_MixinBase):
    """Adds the flat Core builders :meth:`points`, :meth:`polygons` and :meth:`choropleth` to a :class:`Scene3D`.

    A capability mixin of :class:`~digitalearth.three_d.scene3d.Scene3D`, composed into that scene and never
    instantiated on its own; it reaches the plotter and layer registry through ``self``.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the state these methods use.
    """

    def points(
        self,
        features: Any,
        *,
        name: Any = None,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        size: float = 5.0,
        opacity: float | None = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Draw flat point features (markers on the ground plane, ``z=0``) and register them as one layer.

        The Core ``points`` verb, distinct from :meth:`~digitalearth.three_d.point_cloud.PointCloudMixin.point_cloud`,
        which takes a z per point: ``points`` lays markers flat, as a 2-D map would, under the 3-D scene.

        Args:
            features: A numpy ``(N, 2)``/``(N, 3)`` table (z is ignored — the markers are flattened to the
                ground) or a GeoDataFrame of ``Point`` geometries.
            column: Attribute column to colour by; ``None`` for a flat colour.
            scheme: How ``column`` is classified (``None`` ramp, ``"categorical"``, or a ``classify`` name).
            k: Number of classes for a graduated ``scheme``.
            cmap: Colormap used when colouring by ``column``.
            size: Marker size in pixels.
            opacity: Marker opacity in ``[0, 1]``; ``None`` leaves the engine default.
            visible: Whether the layer is drawn when added.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_points`.

        Returns:
            The registered actor, or ``None`` when there were no points (see ``strict``).
        """
        return self._add_described_layer(
            kind="points",
            data=features,
            name=name,
            visible=visible,
            encodings=_point_encoding(features, None, column, scheme, k, cmap),
            column=column,
            scheme=scheme,
            k=k,
            cmap=cmap,
            size=size,
            opacity=opacity,
            **kwargs,
        )

    def polygons(
        self,
        features: Any,
        *,
        name: Any = None,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        opacity: float | None = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Draw flat filled polygons (footprints on the ground plane, ``z=0``) and register them as one layer.

        The Core ``polygons`` verb, distinct from
        :meth:`~digitalearth.three_d.vector.VectorMixin.extruded_polygons`, which raises each footprint into a
        prism: ``polygons`` fills them flat. Holes are carved (the same hole-aware triangulation the extrusion
        uses).

        Args:
            features: A GeoDataFrame of ``Polygon``/``MultiPolygon`` geometries.
            column: Attribute column to colour by; ``None`` for a flat colour.
            scheme: How ``column`` is classified.
            k: Number of classes for a graduated ``scheme``.
            cmap: Colormap used when colouring by ``column``.
            opacity: Fill opacity; ``None`` leaves the engine default.
            visible: Whether the layer is drawn when added.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered actor, or ``None`` when there were no polygons (see ``strict``).
        """
        return self._add_described_layer(
            kind="polygons",
            data=features,
            name=name,
            visible=visible,
            encodings=_polygon_encoding(features, column, scheme, k, cmap),
            column=column,
            scheme=scheme,
            k=k,
            cmap=cmap,
            opacity=opacity,
            **kwargs,
        )

    def choropleth(
        self,
        features: Any,
        *,
        name: Any = None,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        opacity: float | None = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Draw a flat polygon fill coloured by a column (a map choropleth on the ground plane).

        The Core ``choropleth`` verb: :meth:`polygons` that always colours by a column. ``scheme`` defaults to
        ``None`` — a continuous ramp over the raw values — exactly as the static, web and interactive tiers'
        ``choropleth`` default, so one call classifies the same way whichever tier draws it; pass a scheme name
        (``"quantiles"``, …) to cut classes.

        Args:
            features: A GeoDataFrame of ``Polygon``/``MultiPolygon`` geometries.
            column: The attribute column to colour by — **required**; for an unclassified fill use
                :meth:`polygons`.
            scheme: How ``column`` is classified; ``None`` (the default) is a continuous ramp, as on every
                other tier.
            k: Number of classes for a graduated ``scheme``.
            cmap: Colormap for the classes.
            opacity: Fill opacity; ``None`` leaves the engine default.
            visible: Whether the layer is drawn when added.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered actor, or ``None`` when there were no polygons (see ``strict``).

        Raises:
            ValueError: if ``column`` is not given — a choropleth with no column has nothing to colour by.
        """
        if not column:
            raise ValueError(
                "choropleth() needs a column= to colour by; for an unclassified fill use polygons()"
            )
        return self._add_described_layer(
            kind="choropleth",
            data=features,
            name=name,
            visible=visible,
            encodings=_polygon_encoding(features, column, scheme, k, cmap),
            column=column,
            scheme=scheme,
            k=k,
            cmap=cmap,
            opacity=opacity,
            **kwargs,
        )


def draw_points(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the flat markers a `points` layer describes.

    Args:
        scene: The scene being drawn into.
        data: The layer's source: a point table or a GeoDataFrame of points.
        layer: The layer's description.

    Returns:
        The `(cloud, actor)` pair, or `None` when the table was empty and the scene is not `strict`.
    """
    props = drawing_props(layer.symbology.props)
    column = props.pop("column", None)
    scheme = props.pop("scheme", None)
    k = props.pop("k", 5)
    cmap = props.pop("cmap", "viridis")
    size = props.pop("size", 5.0)
    opacity = props.pop("opacity", None)
    placed = scene._place(data, layer="points")
    if hasattr(placed, "geometry"):
        points, values = _coords_from_geodataframe(placed, column)
    else:
        points, values = _coords_from_array(placed), None
    if len(points) == 0:
        scene._skip_empty("points", "the point table is empty")
        return None
    # Flat: markers sit on the ground plane whatever z the input carried, which is what distinguishes this
    # from point_cloud. Copy first so a caller's array is not mutated.
    points = points.copy()
    points[:, 2] = 0.0

    import pyvista as pv

    cloud = pv.PolyData(points)
    extra = _only_when_given(opacity=opacity)
    extra["point_size"] = float(size)
    if values is not None:
        encoding = layer.symbology.encoding("color")
        style = classified_scalars(
            values,
            scheme=scheme,
            k=k,
            cmap=cmap,
            scale=None if encoding is None else encoding.scale,
        )
        cloud[SCALAR] = style.pop("scalars")
        cloud = reduce_points(cloud, scene.big_data_threshold, kind="points")
        return cloud, scene.plotter.add_points(
            cloud, scalars=SCALAR, **style, **extra, **props
        )
    cloud = reduce_points(cloud, scene.big_data_threshold, kind="points")
    return cloud, scene.plotter.add_points(cloud, cmap=cmap, **extra, **props)


def draw_polygons(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the flat filled polygons a `polygons` or `choropleth` layer describes.

    One drawer serves both kinds: a `choropleth` is a `polygons` layer that always carries a colour column.

    Args:
        scene: The scene being drawn into.
        data: The layer's source: the polygons to fill.
        layer: The layer's description, whose props carry the colour column and the classification.

    Returns:
        The `(mesh, actor)` pair, or `None` when there were no polygons and the scene is not `strict`.
    """
    props = drawing_props(layer.symbology.props)
    column = props.pop("column", None)
    scheme = props.pop("scheme", None)
    k = props.pop("k", 5)
    cmap = props.pop("cmap", "viridis")
    opacity = props.pop("opacity", None)
    gdf = scene._place(data, layer=layer.kind)
    geoms = gdf.geometry
    colours = gdf[column].to_numpy() if column else None

    encoding = layer.symbology.encoding("color")
    style, scalars = _classify_or_refuse(
        colours,
        scheme=scheme,
        k=k,
        cmap=cmap,
        pinned=props,
        scale=None if encoding is None else encoding.scale,
        caller=layer.kind,
    )

    import pyvista as pv

    caps: list[pv.PolyData] = []
    for i, geom in enumerate(geoms):
        for exterior, interiors in _polygon_parts(geom):
            cap = _cap_with_holes(exterior, interiors)
            if scalars is not None:
                cap.cell_data[VALUE] = np.full(cap.n_cells, scalars[i])
            caps.append(cap)

    if not caps:
        scene._skip_empty(layer.kind, "no polygon geometries to draw")
        return None

    merged = pv.MultiBlock(caps).combine()
    extra = _only_when_given(opacity=opacity)
    if colours is None:
        return merged, scene.plotter.add_mesh(
            merged, scalars=None, cmap=cmap, **extra, **props
        )
    return merged, scene.plotter.add_mesh(
        merged, scalars=VALUE, **style, **extra, **props
    )
