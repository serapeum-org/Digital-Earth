"""PointCloudMixin — render scattered points (LiDAR / observations / raster cells) in 3-D.

Turns a point table into a PyVista ``PolyData`` and renders it with optional per-point colour, sphere markers,
and **eye-dome lighting** (the depth-cueing shading that makes dense point clouds legible). Inputs come from
pyramids — a numpy ``(N, 3)``/``(N, 2)`` table (e.g. ``Dataset.to_xyz()`` / LiDAR), or a GeoDataFrame of points
(e.g. ``Dataset.get_cell_points()`` / ``FeatureCollection.to_geodataframe()``).

The GeoDataFrame is **duck-typed** (``hasattr(data, "geometry")``) and only its coordinates are read — so this
module imports neither geopandas nor shapely (the HARD RULE / ``test_no_competitor_imports`` guard); pyramids does
all CRS work upstream.
"""
from typing import Any, Optional, Tuple

import numpy as np
import pyvista as pv

#: Attribute name the per-point colour scalar is stored under on the generated cloud.
SCALAR = "scalar"


def _coords_from_geodataframe(data: Any, value_column: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read ``(N, 3)`` coordinates (and an optional value column) off a points GeoDataFrame.

    Args:
        data: A GeoDataFrame of ``Point`` geometries (duck-typed; never imported as a GIS engine).
        value_column: Optional attribute column to colour by; ``None`` for an uncoloured cloud.

    Returns:
        tuple: ``(points, values)`` where ``points`` is an ``(N, 3)`` float array (z filled with 0 when the
        geometries are 2-D) and ``values`` is the column array or ``None``.
    """
    geom = data.geometry
    x = np.asarray(geom.x.to_numpy(), dtype="float64")
    y = np.asarray(geom.y.to_numpy(), dtype="float64")
    z = np.asarray(geom.z.to_numpy(), dtype="float64") if bool(geom.has_z.all()) else np.zeros_like(x)
    points = np.column_stack([x, y, z])
    values = np.asarray(data[value_column].to_numpy(), dtype="float64") if value_column else None
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
        raise ValueError(f"point_cloud expects an (N, 3) or (N, 2) table, got shape {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr))])
    return arr


class PointCloudMixin:
    """Adds :meth:`point_cloud` — render scattered 3-D points — to a :class:`Scene3D`."""

    def point_cloud(
        self,
        data: Any,
        *,
        values: Optional[np.ndarray] = None,
        value_column: Optional[str] = None,
        point_size: float = 5.0,
        render_points_as_spheres: bool = True,
        eye_dome_lighting: bool = True,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Any:
        """Render a point cloud (LiDAR / observations / raster cells) and register it as a layer.

        Args:
            data: A numpy ``(N, 3)``/``(N, 2)`` point table (e.g. ``Dataset.to_xyz()`` or LiDAR xyz), or a
                GeoDataFrame of ``Point`` geometries (e.g. ``Dataset.get_cell_points()``).
            values: Explicit per-point scalar array to colour by (overrides ``value_column``).
            value_column: Column name to colour by when ``data`` is a GeoDataFrame; ``None`` for no colour.
            point_size: Marker size in pixels.
            render_points_as_spheres: Draw points as shaded spheres (cleaner than flat dots).
            eye_dome_lighting: Enable depth-cueing eye-dome lighting (recommended for dense clouds).
            cmap: Colormap used when the cloud is coloured by a scalar.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_points`.

        Returns:
            The registered :class:`pyvista.Actor` for the point cloud.

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
        """
        if hasattr(data, "geometry"):
            points, gdf_values = _coords_from_geodataframe(data, value_column)
        else:
            points, gdf_values = _coords_from_array(data), None

        scalar = values if values is not None else gdf_values
        cloud = pv.PolyData(points)

        add_kwargs = dict(point_size=point_size, render_points_as_spheres=render_points_as_spheres, **kwargs)
        if scalar is not None:
            cloud[SCALAR] = np.asarray(scalar, dtype="float64")
            add_kwargs.update(scalars=SCALAR, cmap=cmap)

        actor = self.plotter.add_points(cloud, **add_kwargs)
        if eye_dome_lighting:
            self.plotter.enable_eye_dome_lighting()
        return self._add_actor(cloud, actor)
