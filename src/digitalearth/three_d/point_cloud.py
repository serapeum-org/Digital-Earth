"""PointCloudMixin — render scattered points (LiDAR / observations / raster cells) in 3-D.

Turns a point table into a PyVista ``PolyData`` and renders it with optional per-point colour, sphere markers,
and **eye-dome lighting** (the depth-cueing shading that makes dense point clouds legible). Inputs come from
pyramids — a numpy ``(N, 3)``/``(N, 2)`` table (e.g. ``Dataset.to_xyz()`` / LiDAR), or a GeoDataFrame of points
(e.g. ``Dataset.get_cell_points()`` or a pyramids ``FeatureCollection``, which *is* a GeoDataFrame).

The GeoDataFrame is **duck-typed** (``hasattr(data, "geometry")``) and only its coordinates are read — so this
module imports neither geopandas nor shapely (the HARD RULE / ``test_no_competitor_imports`` guard); pyramids does
all CRS work upstream.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from digitalearth.base.deprecation import renamed_parameter
from digitalearth.base.points import PointArrays
from digitalearth.three_d.base import classified_scalars

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
        values: np.ndarray | None = None,
        value_column: str | None = None,
        size: float | None = None,
        scheme: Any | None = None,
        k: int = 5,
        render_points_as_spheres: bool = True,
        eye_dome_lighting: bool = True,
        cmap: str = "viridis",
        point_size: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a point cloud (LiDAR / observations / raster cells) and register it as a layer.

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
            point_size: **Deprecated** alias of ``size``; passing it warns that ``point_size=`` will be
                removed in a future release and forwards the value unchanged. Passing both is a
                ``TypeError``.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_points`.

        Returns:
            The registered :class:`pyvista.Actor` for the point cloud, or ``None`` when the cloud held no
            points (see ``strict`` on :class:`~digitalearth.three_d.base.Scene3DBase`).

        Raises:
            TypeError: if both ``size`` and the deprecated ``point_size`` are given — they name one
                parameter, so neither can be silently preferred.
            ValueError: if ``values`` does not have one entry per point, or if ``scheme`` cannot classify
                the values. Also — only when the scene was built with ``strict=True`` —
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
        size = renamed_parameter(
            new="size",
            value=size,
            old="point_size",
            alias=point_size,
            caller="Scene3D.point_cloud()",
            default=5.0,
        )
        if hasattr(data, "geometry"):
            points, gdf_values = _coords_from_geodataframe(data, value_column)
        else:
            points, gdf_values = _coords_from_array(data), None

        scalar = values if values is not None else gdf_values
        if len(points) == 0:
            # Validate the scalar length first so an empty cloud paired with values is still a caller error,
            # not a silently skipped layer.
            if scalar is not None and len(scalar) != 0:
                raise ValueError(
                    f"values length {len(scalar)} does not match {len(points)} points"
                )
            self._skip_empty("point_cloud", "the point table is empty")
            return None
        import pyvista as pv

        cloud = pv.PolyData(points)

        add_kwargs = dict(
            point_size=size,
            render_points_as_spheres=render_points_as_spheres,
            **kwargs,
        )
        if scalar is not None:
            if len(scalar) != len(points):
                raise ValueError(
                    f"values length {len(scalar)} does not match {len(points)} points"
                )
            style = classified_scalars(scalar, scheme=scheme, k=k, cmap=cmap)
            cloud[SCALAR] = style.pop("scalars")
            add_kwargs.update(scalars=SCALAR, **style)

        actor = self.plotter.add_points(cloud, **add_kwargs)
        if eye_dome_lighting:
            self.plotter.enable_eye_dome_lighting()
        return self._add_actor(cloud, actor)
