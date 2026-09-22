"""ThreeDMixin — web-tier 3-D builders (DW.4, recipe W5).

The GeoLibre 3-D surface without VTK, all on the GPU:

* ``extrusion`` — a MapLibre ``fill-extrusion`` layer (3-D choropleth: height and colour from a column);
* ``point_cloud`` / ``tiles_3d`` / ``gltf`` — deck.gl ``PointCloudLayer`` / ``Tile3DLayer`` / ``ScenegraphLayer``
  driven through the maplibre widget's ``add_deck_layers`` (deck.gl JSON);
* ``terrain`` — draped 3-D terrain from a served ``raster-dem`` tile source (defaults to the public AWS
  *terrarium* terrain-RGB tiles; MapLibre terrain reads tiles over the network, so a self-contained map points
  at hosted tiles rather than embedding a DEM — encode your own with pyramids ``Dataset.to_terrain_rgb``);
* ``globe`` — switch the map to the spherical (globe) projection.

extrusion reuses the base ``_color_expr`` for graduated/continuous colouring; deck builders reuse the
``_add_deck_layer`` accumulator. maplibre/numpy are imported lazily.
"""

import warnings
from typing import TYPE_CHECKING, Any, Optional, Self, Sequence

from digitalearth.base.deprecation import renamed_method, renamed_parameter
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.web.base import _require_layer_api, placed_features
from digitalearth.web.bigdata import DECK_TYPE_KEY

#: Default DEM for ``terrain`` — AWS Terrain Tiles (open data), terrarium-encoded terrain-RGB. MapLibre terrain
#: needs a served ``raster-dem`` tile source, so the default is hosted tiles; to use your own DEM, encode it to
#: terrain-RGB tiles with pyramids ``Dataset.to_terrain_rgb``, serve them, and pass the tile-URL template as ``dem``.
_DEFAULT_TERRAIN_TILES = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def draw_extruded_polygons(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre extruded-fill layer for a polygon collection.

    Args:
        web_map: The map being drawn, whose display CRS the polygons are placed in.
        data: The layer's source — the polygon frame the builder already placed, or whatever the figure's
            reference opened to when the layer is drawn back from a description.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`.

    Raises:
        ValueError: when the description records no `paint` for the extrusion, naming the layer, its kind
            and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, layer_types = _require_layer_api()
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec=placed_features(web_map, data, layer),
        layer=layer_cls(
            id=layer.id,
            type=layer_types.FILL_EXTRUSION,
            source=source_id,
            paint=dict(required_props(layer, "paint")["paint"]),
        ),
    )


class ThreeDMixin(_MixinBase):
    """3-D builders for :class:`~digitalearth.web.map.WebMap` (fill-extrusion + deck.gl + terrain/globe)."""

    def extrusion(
        self,
        features: Any,
        *,
        height: Any,
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        color: str = "#3388ff",
        opacity: float = 0.9,
    ) -> Self:
        """Draw a 3-D choropleth: polygons extruded by ``height`` and coloured by ``column`` (recipe W5).

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
            height: Extrusion height — a column name (read with ``["get", name]``) or a constant in metres.
            column: Optional value column colouring the extrusions (graduated if ``scheme`` is set, else a
                continuous ramp). ``None`` uses the flat ``color``.
            scheme: A cleopatra classification scheme for graduated colouring.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            color: Flat fill colour used when ``column`` is ``None``.
            opacity: Extrusion opacity in ``[0, 1]``.

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        gdf = self._display_gdf(features, method="extrusion")
        paint: dict = {
            "fill-extrusion-opacity": float(opacity),
            "fill-extrusion-height": ["get", height]
            if isinstance(height, str)
            else float(height),
        }
        if column is not None:
            paint["fill-extrusion-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["fill-extrusion-color"] = color

        layer_id = self._uid("extrusion")
        self._index_layer(
            layer_id,
            None,
            kind="extrusion",
            # The caller's own reference is what a figure can be written down with; the warped frame is
            # handed to the first draw so nothing is warped twice (review H1).
            source=features,
            placed=gdf,
            symbology=Symbology(props={"paint": dict(paint)}),
        )
        self._last_layer_id = layer_id
        return self

    def terrain_tiles(
        self,
        dem: Optional[str] = None,
        *,
        exaggeration: float = 1.0,
        encoding: str = "terrarium",
    ) -> Self:
        """Drape the map over 3-D terrain from a raster-DEM source (recipe W5).

        Args:
            dem: An XYZ terrain-RGB tile URL template (``{z}/{x}/{y}``). ``None`` uses the public AWS
                terrarium terrain tiles. To use your own DEM, encode it to terrain-RGB tiles with pyramids
                ``Dataset.to_terrain_rgb`` and serve them, then pass the tile-URL template here.
            exaggeration: Vertical exaggeration factor.
            encoding: Terrain-RGB encoding of the DEM tiles (``"terrarium"`` or ``"mapbox"``).

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        from maplibre.sources import RasterDEMSource

        src_id = self._uid("dem")
        source = RasterDEMSource(
            tiles=[dem or _DEFAULT_TERRAIN_TILES], encoding=encoding, tile_size=256
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, source)
            widget.set_terrain(src_id, exaggeration)

        return self._queue(apply)

    #: Deprecated spelling of :meth:`terrain_tiles` (#299). The 3-D tier's `terrain` takes a pyramids DEM
    #: and this one a terrain-RGB tile URL; one name cannot mean both, so the tile reader says so.
    terrain = renamed_method(new="terrain_tiles", old="terrain", owner="WebMap")

    def projection(self, name: str = "globe") -> Self:
        """Draw the map in another projection, by name.

        Args:
            name: `"globe"` for the sphere, `"mercator"` for the flat Web-Mercator map. MapLibre's other
                projection names are passed through as they are.

        Returns:
            This map (chainable).

        Examples:
            - The globe is a projection, which is what the interactive tier already calls it:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().projection("globe").viewport.globe
                True

                ```
            - And back again:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().projection("globe").projection("mercator").viewport.globe
                False

                ```
        """
        return self._set_projection(str(name))

    def globe(self, enabled: bool = True) -> Self:
        """Switch the map to the spherical globe projection (or back to Web Mercator).

        Args:
            enabled: ``True`` for the globe projection; ``False`` restores Web Mercator.

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        warnings.warn(
            "WebMap.globe() is deprecated and will be removed in a future release; use "
            f"WebMap.projection({'globe' if enabled else 'mercator'!r}) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._set_projection("globe" if enabled else "mercator")

    def _set_projection(self, projection: str) -> Self:
        """Switch the map's projection and record it on the view.

        Args:
            projection: The MapLibre projection name.

        Returns:
            This map (chainable).
        """
        self._projection = projection

        def apply(widget: Any) -> None:
            widget.set_projection(projection)

        return self._queue(apply)

    @staticmethod
    def _point_cloud_data(points: Any, z_column: Optional[str]) -> list:
        """Build deck.gl ``PointCloudLayer`` rows (``[{"position": [x, y, z]}, …]``) from ``points``.

        Args:
            points: A point GeoDataFrame (z from ``z_column`` or 0) or an ``(N, 2|3)`` array-like of
                coordinates.
            z_column: Elevation column for a GeoDataFrame input; ignored for array input.

        Returns:
            A list of ``{"position": [lng, lat, z]}`` dicts for the deck layer's ``data``.
        """
        import numpy as np

        if hasattr(points, "geometry"):
            xs = points.geometry.x.to_numpy()
            ys = points.geometry.y.to_numpy()
            zs = (
                np.asarray(points[z_column], dtype=float)
                if z_column is not None
                else np.zeros(len(points))
            )
            return [
                {"position": [float(x), float(y), float(z)]}
                for x, y, z in zip(xs, ys, zs)
            ]
        rows = np.asarray(points, dtype=float)
        return [
            {
                "position": [
                    float(r[0]),
                    float(r[1]),
                    float(r[2]) if r.shape[0] > 2 else 0.0,
                ]
            }
            for r in rows
        ]

    def point_cloud(
        self,
        points: Any,
        *,
        z_column: Optional[str] = None,
        color: Sequence[int] = (255, 140, 0),
        size: Optional[float] = None,
        point_size: Optional[float] = None,
    ) -> Self:
        """Render a 3-D point cloud as a deck.gl ``PointCloudLayer`` (recipe W5).

        Args:
            points: A point ``FeatureCollection`` / GeoDataFrame (reprojected to lon/lat) or an
                ``(N, 2|3)`` coordinate array.
            z_column: Elevation column for a GeoDataFrame input (0 when omitted).
            color: RGB point colour (0-255 per channel).
            size: Point size in pixels (``2.0`` when omitted — the signature's ``None`` is the "not
                passed" sentinel the deprecated spelling is resolved against). The same ``size`` that
                means marker size on every tier.
            point_size: **Deprecated** spelling of ``size``; forwarded unchanged, after a
                ``DeprecationWarning`` that ``point_size=`` will be removed in a future release.

        Returns:
            This map (chainable).

        Raises:
            TypeError: when ``points`` is a raster. The full vector guard would be too strict
                here — a bare ``(N, 2|3)`` coordinate array is a valid input — so only a raster
                is refused. Also when both ``size`` and the deprecated ``point_size`` are passed,
                since they name one parameter.

        Examples:
            - An ``(N, 3)`` xyz table becomes deck.gl positions verbatim, in lon/lat/height order
              (needs the ``web`` extra, so the block is skipped without it):
                ```python
                >>> import numpy as np                               # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> xyz = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 9.0]])  # doctest: +SKIP
                >>> m = WebMap().point_cloud(xyz, size=3.0)          # doctest: +SKIP
                >>> m._deck_layers[0]["pointSize"]                   # doctest: +SKIP
                3.0
                >>> m._deck_layers[0]["data"]                        # doctest: +SKIP
                [{'position': [0.0, 0.0, 5.0]}, {'position': [1.0, 1.0, 9.0]}]

                ```
            - A GeoDataFrame is reprojected to lon/lat first, and ``z_column`` supplies the
              height — omit it and every point sits flat at ``z = 0``:
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import Point               # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     {"h": [12.0, 30.0]},
                ...     geometry=[Point(0, 0), Point(1, 1)], crs=4326,
                ... )
                >>> m = WebMap().point_cloud(gdf, z_column="h")      # doctest: +SKIP
                >>> m._deck_layers[0]["data"]                        # doctest: +SKIP
                [{'position': [0.0, 0.0, 12.0]}, {'position': [1.0, 1.0, 30.0]}]

                ```
            - The old ``point_size=`` spelling still lands on ``size``, after saying it is going
              away:
                ```python
                >>> import warnings                                  # doctest: +SKIP
                >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
                ...     warnings.simplefilter("always")
                ...     m = WebMap().point_cloud(xyz, point_size=6.0)
                >>> m._deck_layers[0]["pointSize"]                   # doctest: +SKIP
                6.0
                >>> caught[0].category.__name__                      # doctest: +SKIP
                'DeprecationWarning'

                ```

        See Also:
            digitalearth.web.threed.ThreeDMixin.tiles_3d: streams a prebuilt 3-D tileset instead.
            digitalearth.three_d.Scene3D.point_cloud: the PyVista tier's desktop counterpart.
        """
        _require_layer_api()
        size = renamed_parameter(
            new="size",
            value=size,
            old="point_size",
            alias=point_size,
            caller="WebMap.point_cloud()",
            default=2.0,
        )
        # point_cloud also accepts a raw sequence of xyz triples, so the full vector guard would be too
        # strict here; reject only a raster, which would otherwise die inside `_point_cloud_data`.
        self._reject_raster(points, "point_cloud")
        data = self._point_cloud_data(
            self._display_gdf(points, method="point_cloud")
            if hasattr(points, "geometry")
            else points,
            z_column,
        )
        layer = {
            DECK_TYPE_KEY: "PointCloudLayer",
            "id": self._uid("deck-pointcloud"),
            "data": data,
            "getPosition": "@@=position",
            "getColor": list(color),
            "pointSize": float(size),
        }
        return self._add_deck_layer(layer)

    def tiles_3d(self, url: str, *, opacity: float = 1.0) -> Self:
        """Render an OGC 3D Tiles / Cesium tileset as a deck.gl ``Tile3DLayer`` (recipe W5).

        **Not browser-verified:** the layer spec is built and serialised, but its in-browser render is not
        covered by the headless smoke test because it streams a **remote** ``tileset.json`` (the hermetic CI
        job does not fetch network assets).

        Args:
            url: URL of the tileset's ``tileset.json``.
            opacity: Layer opacity in ``[0, 1]``.

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        layer = {
            DECK_TYPE_KEY: "Tile3DLayer",
            "id": self._uid("deck-tiles3d"),
            "data": url,
            "opacity": float(opacity),
        }
        return self._add_deck_layer(layer)

    def gltf(
        self,
        url: str,
        lng: float,
        lat: float,
        *,
        size: float = 1.0,
    ) -> Self:
        """Place a glTF/GLB 3-D model at ``(lng, lat)`` as a deck.gl ``ScenegraphLayer`` (recipe W5).

        **Not browser-verified:** the layer spec is built and serialised, but its in-browser render is not
        covered by the headless smoke test because it fetches a **remote** ``.gltf`` / ``.glb`` asset (the
        hermetic CI job does not fetch network assets).

        Args:
            url: URL of the ``.gltf`` / ``.glb`` model.
            lng: Longitude to place the model at.
            lat: Latitude to place the model at.
            size: Model size scale factor.

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        layer = {
            DECK_TYPE_KEY: "ScenegraphLayer",
            "id": self._uid("deck-gltf"),
            "data": [{"position": [float(lng), float(lat)]}],
            "scenegraph": url,
            "getPosition": "@@=position",
            "sizeScale": float(size),
            "_lighting": "pbr",
        }
        return self._add_deck_layer(layer)
