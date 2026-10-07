"""PointCloudMixin — render scattered points (LiDAR / observations / raster cells) in 3-D.

Turns a point table into a PyVista ``PolyData`` and renders it with optional per-point colour, sphere markers,
and **eye-dome lighting** (the depth-cueing shading that makes dense point clouds legible). Inputs come from
pyramids — a numpy ``(N, 3)``/``(N, 2)`` table (e.g. ``Dataset.to_xyz()`` / LiDAR), or a GeoDataFrame of points
(e.g. ``Dataset.get_cell_points()`` or a pyramids ``FeatureCollection``, which *is* a GeoDataFrame).

The GeoDataFrame is **duck-typed** (``hasattr(data, "geometry")``) and only its coordinates are read — so this
module imports neither geopandas nor shapely (the HARD RULE / ``test_no_competitor_imports`` guard); pyramids does
all CRS work upstream.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from digitalearth.base.points import PointArrays
from digitalearth.three_d.base import classified_scalars
from digitalearth.three_d.bigdata import DEFAULT_CELL_BUDGET, reduce_points

#: Attribute name the per-point colour scalar is stored under on the generated cloud.
SCALAR = "scalar"


def _coords_from_geodataframe(
    data: Any, value_column: str | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read ``(N, 3)`` coordinates (and an optional value column) off a points GeoDataFrame.

    Args:
        data: A GeoDataFrame of ``Point`` geometries (duck-typed; never imported as a GIS engine).
        value_column: Optional attribute column to colour by; ``None`` for an uncoloured cloud.

    Returns:
        tuple: ``(points, values)`` where ``points`` is an ``(N, 3)`` float array (z filled with 0 when the
        geometries are 2-D) and ``values`` is the column array or ``None``.
    """
    # The x/y/z read, the float64 coercion and the "zeros when the geometry is 2-D" rule are all
    # PointArrays' now — this tier wrote out its own copy of each.
    # centroids=False keeps the error this site has always raised: a point cloud of polygon centroids is
    # a plausible-looking picture of data the caller never asked to reduce.
    points = PointArrays.from_features(data, centroids=False).as_columns()
    values = (
        np.asarray(data[value_column].to_numpy(), dtype="float64")
        if value_column
        else None
    )
    return points, values


def _coords_from_array(data: Any) -> np.ndarray:
    """Coerce a numpy ``(N, 2)``/``(N, 3)`` point table to an ``(N, 3)`` float array (z=0 when 2-D).

    Args:
        data: An ``(N, 3)`` (x, y, z) or ``(N, 2)`` (x, y) array-like.

    Returns:
        numpy.ndarray: the ``(N, 3)`` float coordinate array.

    Raises:
        ValueError: if ``data`` is not 2-D with 2 or 3 columns.
    """
    arr = np.asarray(data, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"point_cloud expects an (N, 3) or (N, 2) table, got shape {arr.shape}"
        )
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr))])
    return arr


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


from digitalearth.base.spec import Encoding, LayerSpec
from digitalearth.three_d.layer import drawing_props


class PointCloudMixin(_MixinBase):
    """Adds :meth:`point_cloud` — render scattered 3-D points — to a :class:`Scene3D`.

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

    def point_cloud(
        self,
        data: Any,
        *,
        name: Any = None,
        values: np.ndarray | None = None,
        value_column: str | None = None,
        size: float = 5.0,
        scheme: Any | None = None,
        k: int = 5,
        render_points_as_spheres: bool = True,
        eye_dome_lighting: bool = True,
        cmap: str = "viridis",
        big_data_threshold: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a point cloud (LiDAR / observations / raster cells) and register it as a layer.

        A GeoDataFrame is placed in the scene's display CRS — setting it when the scene has none, reprojected
        through pyramids when it is in another. A numpy point table declares no CRS and is placed as given.

        Args:
            data: A numpy ``(N, 3)``/``(N, 2)`` point table (e.g. ``Dataset.to_xyz()`` or LiDAR xyz), or a
                GeoDataFrame of ``Point`` geometries (e.g. ``Dataset.get_cell_points()``).
            values: Explicit per-point scalar array to colour by (overrides ``value_column``).
            value_column: Column name to colour by when ``data`` is a GeoDataFrame; ``None`` for no colour.
            size: Marker size in pixels (default ``5.0``). Named ``size`` on every tier, so one number means
                one visual marker size whichever backend drew it.
            scheme: How the colour values are classified, when the cloud is coloured at all. ``None`` (the
                default) is a continuous ramp over the raw values; ``"categorical"`` gives every distinct
                value its own colour; any other ``cleopatra.styling.styles.classify`` scheme name
                (``"quantiles"``, ``"equal_interval"``, ``"fisher_jenks"``, …) colours ``k`` graduated
                classes. The classes are computed exactly as the 2-D tiers compute them.
            k: Number of classes for a graduated ``scheme``; ignored otherwise.
            render_points_as_spheres: Draw points as shaded spheres (cleaner than flat dots).
            eye_dome_lighting: Enable depth-cueing eye-dome lighting (recommended for dense clouds).
            cmap: Colormap used when the cloud is coloured by a scalar.
            big_data_threshold: Points above which the cloud is subsampled (an even stride over the points)
                before it is drawn (#207). ``None`` (the default) uses the scene's
                :attr:`~digitalearth.three_d.base.Scene3DBase.big_data_threshold`; a cloud at or under the
                budget keeps every point. The subsample is deterministic, so the render repeats, and the
                colour scalar is carried through.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_points`. A coloured cloud derives ``scalars``,
                ``clim`` and ``n_colors``, so pinning one of those beside ``values=``/``value_column=`` is a
                ``TypeError`` naming the keyword. ``nan_color`` is honoured instead of refused: the colour
                missing data is drawn in is a choice the colouring only fills a default for.

        Returns:
            The registered :class:`pyvista.Actor` for the point cloud, or ``None`` when the cloud held no
            points (see ``strict`` on :class:`~digitalearth.three_d.base.Scene3DBase`).

        Raises:
            ValueError: if ``values`` does not have one entry per point, if ``scheme`` cannot classify
                the values, or if `data` declares a CRS other than the scene's and cannot be reprojected (a
                `PointArrays`). Also — only when the scene was built with ``strict=True`` —
                :class:`~digitalearth.base.crs.OffLimbError` for an empty cloud, which is otherwise skipped
                with a warning.

        Examples:
            - Render a coloured LiDAR-style xyz table:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> pts = np.column_stack([np.arange(50.0), np.arange(50.0), np.linspace(0, 9, 50)])
                >>> scene = Scene3D(off_screen=True)
                >>> actor = scene.point_cloud(pts, values=pts[:, 2])
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```
            - A 2-D table is lifted to z=0 and rendered uncoloured:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.point_cloud(np.random.default_rng(0).random((30, 2)))
                >>> scene.layers[0][0].n_points
                30
                >>> scene.close()

                ```
            - Classify the colours into quantile classes instead of a continuous ramp:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> pts = np.column_stack([np.arange(20.0), np.arange(20.0), np.zeros(20)])
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.point_cloud(pts, values=pts[:, 0], scheme="quantiles", k=4)
                >>> sorted(set(int(v) for v in scene.layers[0][0]["scalar"]))
                [0, 1, 2, 3]
                >>> scene.close()

                ```
        """
        return self._add_described_layer(
            kind="point_cloud",
            data=data,
            name=name,
            encodings=_color_encoding(data, values, value_column, scheme, k, cmap),
            values=values,
            value_column=value_column,
            size=size,
            scheme=scheme,
            k=k,
            render_points_as_spheres=render_points_as_spheres,
            eye_dome_lighting=eye_dome_lighting,
            cmap=cmap,
            big_data_threshold=self._resolve_big_data_threshold(
                big_data_threshold, caller="Scene3D.point_cloud()"
            ),
            **kwargs,
        )


def _color_encoding(
    data: Any,
    values: Any,
    value_column: Any,
    scheme: Any,
    k: int,
    cmap: Any,
) -> dict[str, Encoding] | None:
    """Return the colour encoding a coloured cloud publishes, or `None` for an uncoloured one.

    The cloud's colour comes from a value per point, so a key over it explains that column — which is what
    order 24 needs recorded on the layer rather than rebuilt from whatever was drawn last.

    Args:
        data: The builder's input, read for `value_column` when one was named.
        values: The explicit per-point value array, or `None`.
        value_column: The attribute column to colour by, or `None`.
        scheme: How the values are classified, or `None` for a continuous ramp.
        k: How many classes a graduated scheme cuts.
        cmap: The colormap the classes come from.

    Returns:
        `{"color": Encoding}` naming :data:`SCALAR` — the array the drawer binds on the cloud — with the scale
        the values are cut into, or `None` when the cloud colours by nothing at all. The **field** is the
        column's name when one was given, since that is what a reader wants a key titled after; the array
        name is what PyVista titles its own bar with, and the two agree for the unnamed case.

        The scale is `None` when the values cannot be reached or cannot be classified: the drawer classifies
        the same column moments later and raises the tier's own message for it (see
        :func:`~digitalearth.three_d.guides.color_scale`).
    """
    from digitalearth.three_d.guides import color_scale

    column = np.asarray(values) if values is not None else None
    if column is None and value_column:
        try:
            column = np.asarray(data[value_column])
        # A column this input does not carry, or an input that cannot be subscripted at all. The drawer
        # reaches the same column through the placed data and raises there; publishing no encoding leaves
        # the layer under-described rather than moving a caller's error to another call.
        except Exception:  # noqa: BLE001
            return None
    if column is None:
        return None
    return {
        "color": Encoding.by_field(
            "color",
            value_column or SCALAR,
            scale=color_scale(column, scheme=scheme, k=k, cmap=cmap),
        )
    }


def _refuse_folded_marker_size(props: dict[str, Any]) -> None:
    """Refuse ``point_size=``, the deleted spelling this drawer folds ``size=`` onto.

    ``point_size`` is a real :meth:`pyvista.Plotter.add_points` keyword, so PyVista's own unknown-keyword
    guard never sees it: it arrived through the builder's ``**kwargs``, and the size fold on the next line
    then overwrote it. ``point_cloud(point_size=40)`` therefore drew at the ``size`` default of 5.0 — it
    neither worked nor refused, while every other tier refuses the spelling outright because the parameter is
    gone from the signature (review H4).

    Args:
        props: The layer's drawing properties, before the ``size`` → ``point_size`` fold.

    Raises:
        TypeError: when ``point_size`` is among them.
    """
    if "point_size" in props:
        raise TypeError(
            "point_cloud() got point_size=, which is the deleted spelling of size=: the marker size is "
            "size= on every tier, and point_size= is the PyVista keyword size= is folded onto here, so "
            "passing it is overwritten — pass size= instead"
        )


def _refuse_derived_colours(
    props: Mapping[str, Any], style: Mapping[str, Any], *, scheme: Any
) -> None:
    """Refuse a colour keyword the cloud's own colouring derives, naming it.

    The style :func:`~digitalearth.three_d.base.classified_scalars` builds is splatted over ``props``, so a
    caller pinning ``clim=`` on a coloured cloud silently lost it to the class-index range — the same clash
    :func:`~digitalearth.three_d.vector._classify_or_refuse` already refuses for an extruded polygon, which
    this drawer did not (review H4). ``scalars`` is in the same position: the cloud binds its own array name.

    What is refused is a *consequence* of the classification, not a choice inside it. ``clim`` and ``n_colors``
    are the class-index range and the class count, and overriding either re-colours the wrong classes;
    ``scalars`` is the array name the cloud binds. ``nan_color`` is neither — it is the colour missing data is
    drawn in, which the classifier fills from
    :data:`~digitalearth.base.symbology.MISSING_COLOR` as a **default** — so it is honoured where the caller
    gave one, the way ``cmap`` already is. Refusing it left no way to choose a missing-data colour on a coloured
    cloud at all, since the advice "drop it, or drop values=" means giving up the colouring (review R2-L2).

    Args:
        props: The caller's remaining keywords, with any ``nan_color`` already taken out by the drawer.
        style: The colour keywords the classification derived, ``scalars`` already taken out.
        scheme: How the values were classified, named in the message so the caller can drop one side.

    Raises:
        TypeError: when a keyword the colouring derives was also passed by the caller.
    """
    clashing = sorted((set(style) | {"scalars"}) & set(props))
    if not clashing:
        return
    names = ", ".join(f"{name}=" for name in clashing)
    # "(scheme=None)" read as a scheme named None on the continuous path, where there is no classification to
    # speak of — the values reach the engine as they are and the ramp is what owns the range (review R2-L2).
    colouring = (
        "the continuous ramp it colours the values by"
        if scheme is None
        else f"the values it colours by (scheme={scheme!r})"
    )
    raise TypeError(
        f"point_cloud() got {names} together with {colouring}, which sets {names} itself; drop it, or drop "
        "values=/value_column="
    )


def draw_point_cloud(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the cloud a `point_cloud` layer describes and add it to the scene's plotter.

    Args:
        scene: The scene being drawn into.
        data: The layer's source object: a point table, or a GeoDataFrame of points.
        layer: The layer's description, whose props carry the values, the classification and the point style.

    Returns:
        The `(cloud, actor)` pair, or `None` when the table held no points and the scene is not `strict`.

    Raises:
        ValueError: if the values do not have one entry per point, or the scheme cannot classify them.
        OffLimbError: when the table is empty and the scene is `strict`.
    """
    props = drawing_props(layer.symbology.props)
    values = props.pop("values", None)
    value_column = props.pop("value_column", None)
    scheme = props.pop("scheme", None)
    k = props.pop("k", 5)
    cmap = props.pop("cmap", "viridis")
    eye_dome_lighting = props.pop("eye_dome_lighting", True)
    budget = int(props.pop("big_data_threshold", DEFAULT_CELL_BUDGET))
    _refuse_folded_marker_size(props)
    props["point_size"] = props.pop("size", None)
    placed = scene._place(data, layer="point_cloud")
    if hasattr(placed, "geometry"):
        points, gdf_values = _coords_from_geodataframe(placed, value_column)
    else:
        points, gdf_values = _coords_from_array(placed), None

    scalar = np.asarray(values) if values is not None else gdf_values
    if len(points) == 0:
        # Validate the scalar length first so an empty cloud paired with values is still a caller error,
        # not a silently skipped layer.
        if scalar is not None and len(scalar) != 0:
            raise ValueError(
                f"values length {len(scalar)} does not match {len(points)} points"
            )
        scene._skip_empty("point_cloud", "the point table is empty")
        return None
    import pyvista as pv

    cloud = pv.PolyData(points)
    if scalar is not None:
        if len(scalar) != len(points):
            raise ValueError(
                f"values length {len(scalar)} does not match {len(points)} points"
            )
        # The layer's own colour encoding carries the classification `_color_encoding` cut to describe this
        # cloud, from the same column with the same scheme and k. Handing it over is what makes that one
        # computation rather than two: a 50,000-point cloud scanned its values twice to draw once, whether
        # or not a key was ever asked for (review N4). A layer whose column could not be cut publishes no
        # scale, and the refusal a caller should see is still raised from here.
        encoding = layer.symbology.encoding("color")
        style = classified_scalars(
            scalar,
            scheme=scheme,
            k=k,
            cmap=cmap,
            scale=None if encoding is None else encoding.scale,
        )
        cloud[SCALAR] = style.pop("scalars")
        # Taken out before the clash check, as `cmap` is above: the classifier fills `nan_color` from the
        # shared missing-data colour as a default, so a caller who named one is choosing, not colliding
        # (review R2-L2).
        chosen_nan_color = props.pop("nan_color", None)
        _refuse_derived_colours(props, style, scheme=scheme)
        if chosen_nan_color is not None:
            style["nan_color"] = chosen_nan_color
        props.update(scalars=SCALAR, **style)

    # Thin the cloud to the budget last, once its colour scalar is attached, so `extract_points` carries the
    # scalar through and the subsample is of the finished cloud rather than of bare coordinates (#207).
    cloud = reduce_points(cloud, budget, kind="point_cloud")
    actor = scene.plotter.add_points(cloud, **props)
    if eye_dome_lighting:
        scene.plotter.enable_eye_dome_lighting()
    return cloud, actor
