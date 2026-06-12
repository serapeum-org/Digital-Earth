"""InteractiveMap — the public interactive-2D scene, composed from the base + capability mixins.

A thin composition, mirroring the 2-D ``Map(GeoLayerBase, RasterMixin, …)`` and 3-D
``Scene3D(Scene3DBase, TerrainMixin, …)`` patterns: all behaviour lives in
:class:`~digitalearth.interactive.base.InteractiveMapBase` and the capability mixins; this module only
assembles them. Builder methods return ``self`` so calls chain::

    from digitalearth.interactive import InteractiveMap

    InteractiveMap().image(dem).choropleth(gdf, column="pop").tiles().save("map.html")
"""

from digitalearth.interactive.animation import AnimationMixin
from digitalearth.interactive.base import InteractiveMapBase
from digitalearth.interactive.bigdata import BigDataMixin
from digitalearth.interactive.dashboard import DashboardMixin
from digitalearth.interactive.decoration import DecorationMixin
from digitalearth.interactive.interaction import InteractionMixin
from digitalearth.interactive.projection import ProjectionMixin
from digitalearth.interactive.raster import RasterMixin
from digitalearth.interactive.temporal import TemporalMixin
from digitalearth.interactive.vector import VectorMixin


class InteractiveMap(
    InteractiveMapBase,
    RasterMixin,
    VectorMixin,
    BigDataMixin,
    TemporalMixin,
    DecorationMixin,
    InteractionMixin,
    ProjectionMixin,
    AnimationMixin,
    DashboardMixin,
):
    """Interactive 2-D web map: pan/zoom/hover Bokeh layers built from pyramids data.

    The interactive sibling of the static :class:`~digitalearth.scene.map.Map`: raster/vector builders
    turn pyramids objects into GeoViews elements (reprojected to the display CRS through pyramids first),
    register them as layers, and ``render()``/``save()`` compose them into one overlay. Needs the optional
    ``interactive`` extra (``pip install 'digitalearth[interactive]'``) only when a builder/render method
    is actually called — importing this module works without it.

    Examples:
        - Importing and constructing needs no HoloViz engine:
            ```python
            >>> from digitalearth.interactive import InteractiveMap
            >>> m = InteractiveMap(crs=3857, width=800, height=600)
            >>> m.layers
            []

            ```
    """
