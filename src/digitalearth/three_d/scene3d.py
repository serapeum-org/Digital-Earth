"""Scene3D — the public 3-D scene: :class:`Scene3DBase` composed with the capability mixins.

``Scene3D`` is the true-3D counterpart of :class:`digitalearth.scene.map.Map`. It owns one
:class:`pyvista.Plotter` (via :class:`~digitalearth.three_d.base.Scene3DBase`) and gains its geospatial plot
verbs from mixins — exactly mirroring the 2-D ``Map(GeoLayerBase, RasterMixin, …)`` composition:

- :class:`~digitalearth.three_d.terrain.TerrainMixin` → :meth:`terrain` (DEM/raster → 3-D relief).
- :class:`~digitalearth.three_d.point_cloud.PointCloudMixin` → :meth:`point_cloud` (scattered points / LiDAR).
- :class:`~digitalearth.three_d.volume.VolumeMixin` → :meth:`volume` / :meth:`isosurface` (3-D scalar fields).
- :class:`~digitalearth.three_d.vector.VectorMixin` → :meth:`vectors` / :meth:`extruded_polygons`.

The textured-globe mixin is added as the tier is built.

Every layer is built from pyramids-sourced numpy + geometry — **never** xarray/rasterio/pyvista-xarray (enforced
by ``tests/test_no_competitor_imports.py``); all CRS/reproject work stays in pyramids.
"""
from digitalearth.three_d.base import Scene3DBase, house_theme
from digitalearth.three_d.point_cloud import PointCloudMixin
from digitalearth.three_d.terrain import TerrainMixin
from digitalearth.three_d.vector import VectorMixin
from digitalearth.three_d.volume import VolumeMixin

__all__ = ["Scene3D", "house_theme"]


class Scene3D(Scene3DBase, TerrainMixin, PointCloudMixin, VolumeMixin, VectorMixin):
    """A single-:class:`pyvista.Plotter` 3-D scene with geospatial plot verbs.

    Inherits the plotter/layer/render lifecycle from :class:`~digitalearth.three_d.base.Scene3DBase` and the
    plot methods from the capability mixins (:meth:`terrain`, :meth:`point_cloud`, :meth:`volume`,
    :meth:`isosurface`, :meth:`vectors`, :meth:`extruded_polygons`). See those classes for the full surface.

    Examples:
        - Create a headless scene, render a DEM as 3-D relief, screenshot it:
            ```python
            >>> import numpy as np
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.sources import get_source
            >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
            >>> scene = Scene3D(off_screen=True)
            >>> _ = scene.terrain(get_source(dem), z_exaggeration=3.0)
            >>> img = scene.screenshot()
            >>> img.shape[-1], bool(img.any())
            (3, True)
            >>> scene.close()

            ```
    """
