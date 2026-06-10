"""Map — a Scene with a display CRS, pyramids reprojection, and basemap/coastline decoration (no Cartopy).

``Map`` reprojects each input to a chosen display CRS via **pyramids** (``Dataset.to_crs``), renders on a plain
matplotlib axes in that projected space, and decorates it with an XYZ-tile basemap (``cleopatra.tiles``) and
Natural-Earth vector features (``pyramids.basemap.natural_earth``). There is deliberately **no Cartopy**: the
projection is applied to the *data* upstream, not to the axes (see plan §2.4).

The field methods here (``imshow`` and the private ``_field`` recipe) are the foundation T1.1 extends with
``contourf``/``contour``/``pcolormesh``/``block``.
"""
from digitalearth.scene.maps.animation import AnimationMixin
from digitalearth.scene.maps.base import GeoLayerBase
from digitalearth.scene.maps.decoration import DecorationMixin
from digitalearth.scene.maps.projection import ProjectionMixin
from digitalearth.scene.maps.raster import RasterMixin
from digitalearth.scene.maps.vector import VectorMixin

__all__ = ["Map"]


class Map(RasterMixin, VectorMixin, DecorationMixin, ProjectionMixin, AnimationMixin, GeoLayerBase):
    """A geospatial :class:`~digitalearth.scene.scene.Scene` that reprojects to a display CRS.

    Args:
        crs: Display CRS as an EPSG int / string / CRS (anything ``Dataset.to_crs`` accepts). Default 3857.
        domain: Optional named region / bbox used to set the extent (resolved in T5.1); ``None`` uses data bounds.
        ax: Existing axes to draw on (a new figure/axes is created when ``None``).
        fig: Figure owning ``ax``.
        figsize: New-figure size when one is created.

    Attributes:
        crs: The display CRS every layer is reprojected to.
        domain: The configured domain (or ``None``).

    Examples:
        - Create a map in Web Mercator and read its display CRS:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.scene import Map
            >>> m = Map(crs=3857)
            >>> m.crs
            3857
            >>> m.layers
            []

            ```
        - Render a reprojected raster, then iterate the registered layers:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.scene import Map
            >>> ds = Dataset.read_file("examples/data/acc4000.tif")
            >>> m = Map(crs=ds.epsg)          # same CRS -> no reprojection
            >>> _ = m.imshow(ds)
            >>> len(m.layers)
            1

            ```
    """

