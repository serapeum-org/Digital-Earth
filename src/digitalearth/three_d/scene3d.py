"""Scene3D — the public 3-D scene: :class:`Scene3DBase` composed with the capability mixins.

``Scene3D`` is the true-3D counterpart of :class:`digitalearth.static.map.Map`. It owns one
:class:`pyvista.Plotter` (via :class:`~digitalearth.three_d.base.Scene3DBase`) and gains its geospatial plot
verbs from mixins — exactly mirroring the 2-D ``Map(GeoLayerBase, RasterMixin, …)`` composition:

- :class:`~digitalearth.three_d.terrain.TerrainMixin` → :meth:`terrain` (DEM/raster → 3-D relief).
- :class:`~digitalearth.three_d.point_cloud.PointCloudMixin` → :meth:`point_cloud` (scattered points / LiDAR).
- :class:`~digitalearth.three_d.flat.FlatMixin` → :meth:`points` / :meth:`polygons` / :meth:`choropleth` (the
  flat Core features, drawn on the ground plane under the 3-D scene).
- :class:`~digitalearth.three_d.volume.VolumeMixin` → :meth:`volume` / :meth:`isosurface` (3-D scalar fields).
- :class:`~digitalearth.three_d.vector.VectorMixin` → :meth:`vectors` / :meth:`extruded_polygons`.
- :class:`~digitalearth.three_d.globe.GlobeMixin` → :meth:`globe` (global field on a textured sphere, via the
  optional, lazily-imported geovista).
- :class:`~digitalearth.three_d.decoration.DecorationMixin` → :meth:`set_title` / :meth:`text` / :meth:`axes` /
  :meth:`orientation_axes` (what the scene says about itself, rather than what it draws).
- :class:`~digitalearth.three_d.guides.GuideMixin` → :meth:`colorbar` / :meth:`legend` (the colour key, as a
  guide on the layer's own encoding rather than figure decoration a tier draws and forgets).
- :class:`~digitalearth.three_d.interaction.InteractionMixin` → :meth:`clip_plane` / :meth:`slice_planes` /
  :meth:`threshold` / :meth:`isovalue` / :meth:`slider` / :meth:`enable_picking` (live widgets and picking —
  below the description seam, so they return a handle rather than recording a layer).
- :class:`~digitalearth.three_d.serve.ServeMixin` → :meth:`serve` (a live, streaming trame view — the
  alternative to the frozen page :meth:`export_html` writes).
- :class:`~digitalearth.three_d.animation.AnimationMixin` → :meth:`orbit` / :meth:`record` / :meth:`jupyter`.

Every layer is built from pyramids-sourced numpy + geometry — **never** xarray/rasterio/pyvista-xarray (enforced
by ``tests/test_no_competitor_imports.py``); all CRS/reproject work stays in pyramids.
"""

from digitalearth.three_d.animation import AnimationMixin
from digitalearth.three_d.base import Scene3DBase, house_theme
from digitalearth.three_d.decoration import DecorationMixin
from digitalearth.three_d.flat import FlatMixin
from digitalearth.three_d.globe import GlobeMixin
from digitalearth.three_d.guides import GuideMixin
from digitalearth.three_d.interaction import InteractionMixin
from digitalearth.three_d.point_cloud import PointCloudMixin
from digitalearth.three_d.serve import ServeMixin
from digitalearth.three_d.terrain import TerrainMixin
from digitalearth.three_d.vector import VectorMixin
from digitalearth.three_d.volume import VolumeMixin

__all__ = ["Scene3D", "house_theme"]


class Scene3D(
    TerrainMixin,
    PointCloudMixin,
    FlatMixin,
    VolumeMixin,
    VectorMixin,
    GlobeMixin,
    DecorationMixin,
    GuideMixin,
    InteractionMixin,
    ServeMixin,
    AnimationMixin,
    Scene3DBase,
):
    """A single-:class:`pyvista.Plotter` 3-D scene with geospatial plot verbs.

    Inherits the plotter/layer/render lifecycle from :class:`~digitalearth.three_d.base.Scene3DBase` and the
    plot methods from the capability mixins (:meth:`terrain`, :meth:`point_cloud`, :meth:`volume`,
    :meth:`isosurface`, :meth:`vectors`, :meth:`extruded_polygons`, :meth:`globe`, :meth:`set_title`,
    :meth:`text`, :meth:`axes`, :meth:`orientation_axes`, :meth:`colorbar`, :meth:`legend`, :meth:`orbit`,
    :meth:`record`). See those classes for the full surface.

    Examples:
        - Create a headless scene, render a DEM as 3-D relief, screenshot it:
            ```python
            >>> import numpy as np
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.base.sources import get_source
            >>> dem = np.add.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
            >>> scene = Scene3D(off_screen=True)
            >>> _ = scene.terrain(get_source(dem), z_exaggeration=3.0)
            >>> img = scene.screenshot()
            >>> img.shape[-1], bool(img.any())
            (3, True)
            >>> scene.close()

            ```
    """
