"""ThreeDMixin — web-tier 3-D builders (DW.4, recipe W5).

The GeoLibre 3-D surface without VTK, all on the GPU:

* ``extrusion`` — a MapLibre ``fill-extrusion`` layer (3-D choropleth: height and colour from a column);
* ``point_cloud`` / ``tiles_3d`` / ``gltf`` — deck.gl ``PointCloudLayer`` / ``Tile3DLayer`` / ``ScenegraphLayer``
  driven through the maplibre widget's ``add_deck_layers`` (deck.gl JSON);
* ``terrain_tiles`` — draped 3-D terrain from a ``raster-dem`` tile source: **a pyramids DEM, which is encoded
  here** (:meth:`~pyramids.dataset.Dataset.to_terrain_rgb`), a tile-URL template for a pyramid already served,
  or nothing at all for the public AWS *terrarium* tiles (DE-28);
* ``globe`` — switch the map to the spherical (globe) projection.

extrusion reuses the base ``_color_expr`` for graduated/continuous colouring; deck builders reuse the
``_add_deck_layer`` accumulator. maplibre/numpy are imported lazily.

MapLibre terrain reads its heights from *tiles*, which is why this tier's terrain writes a pyramid rather than
embedding a DEM the way :mod:`digitalearth.web.raster` can embed a band. The tile writing is pyramids'
(``to_terrain_rgb`` packs heights into R/G/B and warps to Web Mercator); what is here is the choice of where it
goes and the promise that the page decodes with the scheme it was written with.
"""

import warnings
from typing import TYPE_CHECKING, Any, Optional, Self, Sequence

from digitalearth.base.deprecation import renamed_method, renamed_parameter
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.web.base import _require_layer_api, as_finite, placed_features
from digitalearth.web.bigdata import DECK_TYPE_KEY

# The tier writes two kinds of tile pyramid — a coloured raster and a terrain-RGB DEM — and they agree on where
# the tiles go, how the zoom range is resolved and how the page addresses them. Shared rather than re-derived,
# so a terrain pyramid cannot end up addressed one way and a raster pyramid another.
from digitalearth.web.raster import (  # noqa: E402 - see the comment above
    _TILE_SIZE,
    _destination,
    _lonlat_bounds,
    _zoom_range,
)

#: Default DEM for ``terrain_tiles`` — AWS Terrain Tiles (open data), terrarium-encoded terrain-RGB. It is the
#: fallback for a map that names no DEM of its own; a pyramids ``Dataset`` passed as ``dem`` is encoded to a
#: pyramid beside the page instead, so a caller with their own elevation data serves nothing.
_DEFAULT_TERRAIN_TILES = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)

#: What marks a string as a tile-URL template rather than a path to a DEM — MapLibre's own zoom placeholder.
_TILE_TEMPLATE_MARK = "{z}"


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
        TypeError: when the figure's source is not a vector layer, and OffLimbError when the warp
            places none of its geometry and the map is `strict` — both from :func:`placed_features`,
            which places the data when the drawer is handed nothing already placed.
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


def draw_terrain(_web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre DEM source that drapes the map over terrain.

    Terrain draws from no source in the figure. MapLibre reads heights from a **tile pyramid** and nothing
    else, so :meth:`ThreeDMixin.terrain_tiles` encodes a DEM to tiles when it is given one and records the
    tile-URL template it then reads — a plain string, which is why `_data` is unused and why a terrain layer
    survives being written down and read back.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        _data: Unused — terrain has no source in the figure to open (see
            :data:`~digitalearth.web.renderer.DESCRIPTION_ONLY_KINDS`).
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` on the terrain route: its source is the
        ``raster-dem`` source, and its "layer" is the ``set_terrain`` call's own arguments, which is what
        MapLibre takes in place of a style layer. Its visibility is read there, so hiding a terrain layer
        leaves the map flat instead of half-draped.

    Raises:
        ValueError: when the description carries neither the tiles the heights are read from, the scheme
            they are encoded with, nor the exaggeration they are drawn at — naming the layer, its kind and
            what is missing.
    """
    from maplibre.sources import RasterDEMSource

    from digitalearth.web.renderer import TERRAIN_ROUTE, DrawnLayer, required_props

    _require_layer_api()
    props = required_props(layer, "tiles", "encoding", "exaggeration")
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec=RasterDEMSource(
            tiles=[props["tiles"]],
            encoding=props["encoding"],
            tile_size=_TILE_SIZE,
        ),
        layer={
            "id": layer.id,
            "source": source_id,
            "exaggeration": float(props["exaggeration"]),
        },
        route=TERRAIN_ROUTE,
    )


def draw_point_cloud(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the deck.gl ``PointCloudLayer`` for a described point cloud.

    Args:
        web_map: The map being drawn, whose display CRS a GeoDataFrame source is placed in.
        data: The layer's source — the frame or coordinate array the builder already placed, or whatever
            the figure's reference opened to when the layer is drawn back from a description.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` on the deck route, carrying the deck.gl JSON layer
        the page composes into its one overlay.

    Raises:
        ValueError: when the description records none of the values the cloud is built from — its colour,
            its point size or the column its heights come from — naming the layer, its kind and what is
            missing.
    """
    from digitalearth.web.renderer import DECK_ROUTE, DrawnLayer, required_props

    _require_layer_api()
    props = required_props(layer, "color", "size", "z_column")
    placed = (
        placed_features(web_map, data, layer) if hasattr(data, "geometry") else data
    )
    return DrawnLayer(
        source_id=None,
        source_spec=None,
        layer={
            DECK_TYPE_KEY: "PointCloudLayer",
            "id": layer.id,
            "data": ThreeDMixin._point_cloud_data(placed, props["z_column"]),
            "getPosition": "@@=position",
            "getColor": list(props["color"]),
            "pointSize": float(props["size"]),
        },
        route=DECK_ROUTE,
    )


def draw_model(_web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the deck.gl ``ScenegraphLayer`` for a described glTF/GLB model.

    A model draws from no source in the figure: the asset is fetched by the page from the URL its
    description carries, and the one position it stands at is what the caller passed — which is why `_data`
    is unused and why a model layer survives being written down and read back.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        _data: Unused — a model has no source in the figure to open.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` on the deck route, carrying the deck.gl JSON layer
        the page composes into its one overlay.

    Raises:
        ValueError: when the description records none of the values the model is placed from — its URL, its
            position or its size scale — naming the layer, its kind and what is missing.
    """
    from digitalearth.web.renderer import DECK_ROUTE, DrawnLayer, required_props

    _require_layer_api()
    props = required_props(layer, "url", "lng", "lat", "size")
    return DrawnLayer(
        source_id=None,
        source_spec=None,
        layer={
            DECK_TYPE_KEY: "ScenegraphLayer",
            "id": layer.id,
            "data": [{"position": [float(props["lng"]), float(props["lat"])]}],
            "scenegraph": props["url"],
            "getPosition": "@@=position",
            "sizeScale": float(props["size"]),
            "_lighting": "pbr",
        },
        route=DECK_ROUTE,
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
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
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

        Raises:
            ValueError: when ``opacity`` is not a finite number, or when ``height`` is a number that is
                not finite — refused at this call, because a figure holding NaN or infinity could not be
                written down.
            TypeError: when ``features`` is not a polygon layer.
            KeyError: when ``column`` names no feature attribute, or when ``features`` is a URL with no
                resolver registered for its scheme.
            FileNotFoundError: when ``features`` is a path that names nothing.
        """
        _require_layer_api()
        paint: dict = {
            "fill-extrusion-opacity": as_finite(
                opacity, "opacity", "WebMap.extrusion()"
            ),
            # A string names the column to read the height from and is carried as it is; a number is the
            # height itself, and the figure has to be able to carry that too (review L1).
            "fill-extrusion-height": ["get", height]
            if isinstance(height, str)
            else as_finite(height, "height", "WebMap.extrusion()"),
        }
        gdf = self._display_gdf(features, method="extrusion")
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
        dem: Any = None,
        *,
        exaggeration: float = 1.0,
        encoding: str = "terrarium",
        tiles_path: Any = None,
        zooms: Any = None,
        name: Optional[str] = None,
    ) -> Self:
        """Drape the map over 3-D terrain, from a pyramids DEM or from tiles already served (recipe W5).

        This took a tile-URL template and nothing else, and its documentation told the caller to encode their
        DEM to terrain-RGB tiles and serve them — work pyramids does in one call
        (:meth:`~pyramids.dataset.Dataset.to_terrain_rgb`), so the advice was pointing at a route the tier
        could have wired itself (DE-28). It now takes the DEM, exactly as the raster builders take a dataset.

        Args:
            dem: What supplies the heights. A pyramids ``Dataset`` (or a path to one) is **encoded here** to a
                ``{z}/{x}/{y}.png`` terrain-RGB pyramid under ``tiles_path``, and the layer reads that. A
                string holding ``{z}`` is taken as a tile-URL template for a pyramid already served, and used
                unchanged. ``None`` uses the public AWS terrarium tiles.
            exaggeration: Vertical exaggeration factor.
            encoding: Terrain-RGB scheme — ``"terrarium"`` or ``"mapbox"``. It is what the page decodes with
                **and** what an encoded DEM is written with, so the two cannot disagree; the schemes are exact
                inverses of different formulae, so a mismatch misreads every height on the map.
            tiles_path: The pyramid's root directory, required when ``dem`` is a DEM to encode and ignored
                otherwise. As with a tiled raster, the page addresses what sits beside it, so the output is a
                **folder** — save the page into the same directory.
            zooms: ``(lowest, highest)`` zoom levels to encode. ``None`` derives ``(0, native)``, the same
                range a tiled raster derives, so terrain and imagery over one extent stop at the same level.
            name: What the terrain is addressed by — ``set_visible``, ``move_layer``, ``replace_layer`` and
                ``remove_layer`` all take it, and hiding it leaves the map flat. ``None`` generates
                ``dem-1``, ``dem-2``, … as every other unnamed layer is numbered.

        Returns:
            This map (chainable).

        Raises:
            TypeError: when ``dem`` is neither ``None``, a string, a path, nor a pyramids ``Dataset`` — an
                ``xyzservices.TileProvider`` in particular, which is a ``dict`` and would otherwise be written
                into the saved page whole, API key included.
            ValueError: when a DEM is given with no ``tiles_path``, ``zooms`` is not an ordered pair of
                levels, or ``exaggeration`` is not a finite number — refused at this call, because a figure
                holding NaN or infinity could not be written down.
            FileNotFoundError: when ``dem`` is a path that names nothing.

        Examples:
            - Encode a DEM and drape the map over it (needs the ``web`` extra, so the block is skipped):
                ```python
                >>> from pyramids.dataset import Dataset                          # doctest: +SKIP
                >>> from digitalearth.web import WebMap                           # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/LisbonElevation.tif")  # doctest: +SKIP
                >>> m = WebMap().basemap().terrain_tiles(dem, tiles_path="out/dem")  # doctest: +SKIP
                >>> m.save("out/index.html")                                      # doctest: +SKIP

                ```
        """
        _require_layer_api()
        template = self._terrain_template(
            dem, encoding=encoding, tiles_path=tiles_path, zooms=zooms
        )
        layer_id = self._layer_id("dem", name)
        # The pyramid, as values: :func:`draw_terrain` builds the DEM source and the `set_terrain` call from
        # exactly this, so the figure describes the terrain rather than naming a closure that drew it. The
        # template is what is recorded rather than the DEM — a DEM is encoded to tiles here, and the tiles
        # are what the page reads — so a terrain layer round-trips through `to_dict` whatever it was given.
        self._index_layer(
            layer_id,
            name,
            kind="terrain",
            symbology=Symbology(
                props={
                    "tiles": template,
                    "encoding": encoding,
                    "exaggeration": as_finite(
                        exaggeration, "exaggeration", "WebMap.terrain_tiles()"
                    ),
                }
            ),
        )
        return self

    def _terrain_template(
        self, dem: Any, *, encoding: str, tiles_path: Any, zooms: Any
    ) -> str:
        """Return the tile-URL template the terrain source reads, encoding a DEM first when given one.

        Args:
            dem: The caller's ``dem=``.
            encoding: The terrain-RGB scheme to write with, which is the one the page decodes with.
            tiles_path: Where an encoded pyramid goes.
            zooms: The zoom range to encode, or ``None`` to derive it.

        Returns:
            A served template unchanged, the tier's default when nothing was named, or — for a DEM — the
            relative reference to the pyramid just written. Relative, because an absolute path names the
            machine that made the page and travels with it when the page is shared.

        Raises:
            TypeError: for anything that is neither a template, a path, nor a raster pyramids can encode.
            ValueError: from :func:`~digitalearth.web.raster._destination` when a DEM names no destination,
                or from :func:`~digitalearth.web.raster._zoom_range` for an unusable range.
            FileNotFoundError: when ``dem`` is a path that names nothing.
        """
        caller = "WebMap.terrain_tiles()"
        if dem is None:
            return _DEFAULT_TERRAIN_TILES
        if isinstance(dem, str) and _TILE_TEMPLATE_MARK in dem:
            return dem
        # A mapping is refused before `_opened` sees it: an `xyzservices.TileProvider` *is* a `dict`, and one
        # handed to the source's `tiles` list lands in the saved page with every field it holds — this repo
        # has leaked a key that way once already. Pull the URL out of it and pass that instead.
        if isinstance(dem, dict):
            raise TypeError(
                f"{caller} dem must be a tile-URL template (a string holding "
                f"'{_TILE_TEMPLATE_MARK}'), a pyramids Dataset, or a path to one; got a "
                f"{type(dem).__name__}, whose every field — an API key among them — would be written into "
                "the saved page. Pass its URL template instead"
            )
        opened = self._opened(dem)
        if not hasattr(opened, "to_terrain_rgb"):
            raise TypeError(
                f"{caller} dem must be a tile-URL template (a string holding "
                f"'{_TILE_TEMPLATE_MARK}'), a pyramids Dataset, or a path to one; got a "
                f"{type(dem).__name__}"
            )
        destination = _destination(tiles_path, "xyz", caller)
        lowest, highest = _zoom_range(_lonlat_bounds(opened), opened, zooms, caller)
        opened.to_terrain_rgb(
            destination,
            encoding=encoding,
            tiles=True,
            min_zoom=lowest,
            max_zoom=highest,
            tile_size=_TILE_SIZE,
        )
        return f"{destination.name}/{{z}}/{{x}}/{{y}}.png"

    #: Deprecated spelling of :meth:`terrain_tiles` (#299). Both tiers' terrain now takes a pyramids DEM, so
    #: the names no longer disagree about the *input* — they still disagree about the output, and that is what
    #: this one says: the 3-D tier renders a surface from the heights in memory, while this one writes (or
    #: reads) a served tile pyramid, because MapLibre takes its terrain from tiles and nothing else.
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
        name: Optional[str] = None,
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
            name: What the cloud is addressed by — ``set_visible``, ``move_layer``, ``replace_layer`` and
                ``remove_layer`` all take it. ``None`` generates ``deck-pointcloud-1``,
                ``deck-pointcloud-2``, … as every other unnamed layer is numbered.

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
                >>> m.layers[0]["pointSize"]                         # doctest: +SKIP
                3.0
                >>> m.layers[0]["data"]                              # doctest: +SKIP
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
                >>> m.layers[0]["data"]                              # doctest: +SKIP
                [{'position': [0.0, 0.0, 12.0]}, {'position': [1.0, 1.0, 30.0]}]

                ```
            - The old ``point_size=`` spelling still lands on ``size``, after saying it is going
              away:
                ```python
                >>> import warnings                                  # doctest: +SKIP
                >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
                ...     warnings.simplefilter("always")
                ...     m = WebMap().point_cloud(xyz, point_size=6.0)
                >>> m.layers[0]["pointSize"]                         # doctest: +SKIP
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
        placed = (
            self._display_gdf(points, method="point_cloud")
            if hasattr(points, "geometry")
            else points
        )
        layer_id = self._layer_id("deck-pointcloud", name)
        # The colour, the point size and the height column, as values: :func:`draw_point_cloud` builds the
        # deck.gl layer from exactly this plus the source. The caller's own reference is what a figure can be
        # written down with; the placed frame is handed to the first draw so nothing is warped twice.
        self._index_layer(
            layer_id,
            name,
            kind="point_cloud",
            source=points,
            placed=placed,
            symbology=Symbology(
                props={
                    "color": list(color),
                    "size": float(size),
                    "z_column": z_column,
                }
            ),
        )
        return self

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
        name: Optional[str] = None,
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
            name: What the model is addressed by — ``set_visible``, ``move_layer``, ``replace_layer`` and
                ``remove_layer`` all take it. ``None`` generates ``deck-gltf-1``, ``deck-gltf-2``, … as
                every other unnamed layer is numbered.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when ``lng``, ``lat`` or ``size`` is not a finite number — refused at this call,
                because a figure holding NaN or infinity could not be written down.
        """
        _require_layer_api()
        caller = "WebMap.gltf()"
        layer_id = self._layer_id("deck-gltf", name)
        # The asset and where it stands, as values: :func:`draw_model` builds the deck.gl layer from exactly
        # this. A URL is the whole source, so a model layer round-trips through `to_dict` unchanged.
        self._index_layer(
            layer_id,
            name,
            kind="model",
            symbology=Symbology(
                props={
                    "url": url,
                    "lng": as_finite(lng, "lng", caller),
                    "lat": as_finite(lat, "lat", caller),
                    "size": as_finite(size, "size", caller),
                }
            ),
        )
        return self
