"""ThreeDMixin — web-tier 3-D builders (DW.4, recipe W5).

The GeoLibre 3-D surface without VTK, all on the GPU:

* ``extrusion`` — a MapLibre ``fill-extrusion`` layer (3-D choropleth: height and colour from a column);
* ``point_cloud`` / ``tiles_3d`` / ``gltf`` — deck.gl ``PointCloudLayer`` / ``Tile3DLayer`` / ``ScenegraphLayer``
  driven through the maplibre widget's ``add_deck_layers`` (deck.gl JSON);
* ``terrain`` — draped 3-D terrain from a raster-DEM source (defaults to the public AWS *terrarium* terrain-RGB
  tiles, since a pyramids ``to_terrain_rgb`` writer is a confirmed upstream gap);
* ``globe`` — switch the map to the spherical (globe) projection.

extrusion reuses the base ``_color_expr`` for graduated/continuous colouring; deck builders reuse the
``_add_deck_layer`` accumulator. maplibre/numpy are imported lazily.
"""

from typing import Any, Optional, Sequence

from digitalearth.web.base import _require_layer_api

#: Default DEM for ``terrain`` — AWS Terrain Tiles (open data), terrarium-encoded terrain-RGB. Used until a
#: pyramids ``to_terrain_rgb`` writer ships (see planning/maplibre-deckgl/upstream-maplibre-issues.md / PY-IO.9).
_DEFAULT_TERRAIN_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


class ThreeDMixin:
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
    ) -> "ThreeDMixin":
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
        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features)
        paint: dict = {
            "fill-extrusion-opacity": float(opacity),
            "fill-extrusion-height": ["get", height] if isinstance(height, str) else float(height),
        }
        if column is not None:
            paint["fill-extrusion-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["fill-extrusion-color"] = color

        src_id, layer_id = self._uid("ext-src"), self._uid("extrusion")
        layer = Layer(id=layer_id, type=LayerType.FILL_EXTRUSION, source=src_id, paint=paint)

        def apply(widget: Any) -> None:
            widget.add_source(src_id, gdf)
            widget.add_layer(layer)

        self._last_layer_id = layer_id
        return self.add_layer(layer=apply)

    def terrain(
        self,
        dem: Optional[str] = None,
        *,
        exaggeration: float = 1.0,
        encoding: str = "terrarium",
    ) -> "ThreeDMixin":
        """Drape the map over 3-D terrain from a raster-DEM source (recipe W5).

        Args:
            dem: An XYZ terrain-RGB tile URL template (``{z}/{x}/{y}``). ``None`` uses the public AWS
                terrarium terrain tiles (the default until pyramids gains a ``to_terrain_rgb`` writer).
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

        return self.add_layer(layer=apply)

    def globe(self, enabled: bool = True) -> "ThreeDMixin":
        """Switch the map to the spherical globe projection (or back to Web Mercator).

        Args:
            enabled: ``True`` for the globe projection; ``False`` restores Web Mercator.

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        projection = "globe" if enabled else "mercator"

        def apply(widget: Any) -> None:
            widget.set_projection(projection)

        return self.add_layer(layer=apply)

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
            return [{"position": [float(x), float(y), float(z)]} for x, y, z in zip(xs, ys, zs)]
        rows = np.asarray(points, dtype=float)
        return [
            {"position": [float(r[0]), float(r[1]), float(r[2]) if r.shape[0] > 2 else 0.0]}
            for r in rows
        ]

    def point_cloud(
        self,
        points: Any,
        *,
        z_column: Optional[str] = None,
        color: Sequence[int] = (255, 140, 0),
        point_size: float = 2.0,
    ) -> "ThreeDMixin":
        """Render a 3-D point cloud as a deck.gl ``PointCloudLayer`` (recipe W5).

        **Experimental:** the deck.gl layer spec is built and serialised, but its in-browser rendering is not
        verified by the headless test suite (review H1).

        Args:
            points: A point ``FeatureCollection`` / GeoDataFrame (reprojected to lon/lat) or an
                ``(N, 2|3)`` coordinate array.
            z_column: Elevation column for a GeoDataFrame input (0 when omitted).
            color: RGB point colour (0-255 per channel).
            point_size: Point size in pixels.

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        data = self._point_cloud_data(
            self._display_gdf(points) if hasattr(points, "geometry") else points, z_column
        )
        layer = {
            "@@type": "PointCloudLayer",
            "id": self._uid("deck-pointcloud"),
            "data": data,
            "getPosition": "@@=position",
            "getColor": list(color),
            "pointSize": float(point_size),
        }
        return self._add_deck_layer(layer)

    def tiles_3d(self, url: str, *, opacity: float = 1.0) -> "ThreeDMixin":
        """Render an OGC 3D Tiles / Cesium tileset as a deck.gl ``Tile3DLayer`` (recipe W5).

        **Experimental:** the deck.gl layer spec is built and serialised, but its in-browser rendering is not
        verified by the headless test suite (review H1).

        Args:
            url: URL of the tileset's ``tileset.json``.
            opacity: Layer opacity in ``[0, 1]``.

        Returns:
            This map (chainable).
        """
        _require_layer_api()
        layer = {
            "@@type": "Tile3DLayer",
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
    ) -> "ThreeDMixin":
        """Place a glTF/GLB 3-D model at ``(lng, lat)`` as a deck.gl ``ScenegraphLayer`` (recipe W5).

        **Experimental:** the deck.gl layer spec is built and serialised, but its in-browser rendering is not
        verified by the headless test suite (review H1).

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
            "@@type": "ScenegraphLayer",
            "id": self._uid("deck-gltf"),
            "data": [{"position": [float(lng), float(lat)]}],
            "scenegraph": url,
            "getPosition": "@@=position",
            "sizeScale": float(size),
            "_lighting": "pbr",
        }
        return self._add_deck_layer(layer)
