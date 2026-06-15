"""WebMap — the public web scene, composed from the base + capability mixins.

A thin composition, mirroring the 2-D ``Map(GeoLayerBase, RasterMixin, …)``, the 3-D
``Scene3D(Scene3DBase, …)`` and the interactive ``InteractiveMap(InteractiveMapBase, …)`` patterns: all
behaviour lives in :class:`~digitalearth.web.base.WebMapBase` and the capability mixins; this module only
assembles them. The mixins are empty skeletons at DW.0 — later tasks (DW.1+) land their builder methods
without touching this composition. Builder methods return ``self`` so calls chain::

    from digitalearth.web import WebMap

    WebMap().basemap().choropleth(gdf, column="pop").save("map.html")
"""

from digitalearth.web.base import WebMapBase
from digitalearth.web.bigdata import BigDataMixin
from digitalearth.web.decoration import DecorationMixin
from digitalearth.web.export import ExportMixin
from digitalearth.web.raster import RasterMixin
from digitalearth.web.temporal import TemporalMixin
from digitalearth.web.threed import ThreeDMixin
from digitalearth.web.vector import VectorMixin


class WebMap(
    WebMapBase,
    RasterMixin,
    VectorMixin,
    BigDataMixin,
    ThreeDMixin,
    TemporalMixin,
    DecorationMixin,
    ExportMixin,
):
    """Web map: pan/zoom (and 3-D) MapLibre + deck.gl layers built from pyramids data, shareable as HTML.

    The web sibling of the static :class:`~digitalearth.scene.map.Map`, the interactive
    :class:`~digitalearth.interactive.map.InteractiveMap` and the true-3D
    :class:`~digitalearth.three_d.scene3d.Scene3D`: raster/vector builders turn pyramids objects into
    MapLibre/deck.gl layers (reprojected to the display CRS through pyramids first), register them as layers,
    and ``render()``/``save()`` compose them into one ``maplibre`` widget / standalone HTML page. Needs the
    optional ``web`` extra (``pip install 'digitalearth[web]'``) only when a builder/render method is
    actually called — importing this module works without it.

    Examples:
        - Importing and constructing needs no MapLibre engine:
            ```python
            >>> from digitalearth.web import WebMap
            >>> m = WebMap(zoom=4, style="voyager")
            >>> m.layers
            []
            >>> (m.zoom, m.style, m.crs)
            (4, 'voyager', 3857)

            ```
        - Layers register in draw order and the builders chain:
            ```python
            >>> from digitalearth.web import WebMap
            >>> m = WebMap()
            >>> m.add_layer("dem").add_layer("basemap") is m
            True
            >>> m.layers
            ['dem', 'basemap']

            ```
        - With the ``web`` extra installed, the map renders/saves:
            ```python
            >>> from pyramids.dataset import Dataset                  # doctest: +SKIP
            >>> dem = Dataset.read_file("examples/data/acc4000.tif")  # doctest: +SKIP
            >>> WebMap().add_raster(dem).basemap().save("map.html")   # doctest: +SKIP
            'map.html'

            ```

    See Also:
        digitalearth.scene.map.Map: the static matplotlib sibling of this scene.
        digitalearth.interactive.map.InteractiveMap: the interactive-2D (HoloViz) sibling.
        digitalearth.three_d.scene3d.Scene3D: the true-3D (PyVista) sibling.
    """
