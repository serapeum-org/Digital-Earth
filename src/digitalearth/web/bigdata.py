"""BigDataMixin — web-tier big-data builders (DW.3, recipes W3/W4).

GPU-friendly renderers for large feature sets:

* ``heatmap`` — a MapLibre ``heatmap`` layer (optionally weighted by a column);
* ``cluster`` — a MapLibre clustered GeoJSON source with cluster bubbles, count labels and unclustered points;
* ``deck_scatter`` / ``deck_polygons`` — deck.gl ``GeoJsonLayer`` overlays driven through the maplibre widget's
  ``add_deck_layers`` (deck.gl JSON), so millions of features render on the GPU instead of one DOM glyph each.

Recipe W3 named ``lonboard`` for its zero-copy GeoArrow path, but py-maplibregl's deck integration consumes
deck.gl **JSON / pydeck** layers, not lonboard widget objects — they cannot be composited into the same
``maplibre`` widget. We therefore drive deck.gl through ``add_deck_layers`` here, and lonboard is no longer a
dependency of the ``[web]`` extra: it was installed and never imported. A lonboard-native renderer would be a
separate widget, and would re-declare it.

Builders that colour by value reuse the base ``_color_expr`` helpers; numpy/maplibre are imported lazily.
"""

from typing import TYPE_CHECKING, Any, Optional, Self, Sequence

from loguru import logger

from digitalearth.base.bigdata import validate_big_data_threshold
from digitalearth.base.deprecation import renamed_parameter
from digitalearth.base.spec import LayerSpec, Scale, Symbology
from digitalearth.web.base import _require_layer_api

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


#: deck.gl's JSON layer protocol names the layer class under this key. The ``@@`` prefix is the
#: converter's marker for "interpret this value, do not pass it through", so it is protocol rather
#: than a label — worth naming once so a typo cannot silently produce a layer deck.gl ignores.
DECK_TYPE_KEY = "@@type"


def draw_heatmap(_web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre heatmap layer over a point collection.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        data: The layer's source — the display-CRS point GeoDataFrame.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`.

    Raises:
        ValueError: when the description records no `paint` for the heatmap, naming the layer, its kind and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, layer_types = _require_layer_api()
    paint = required_props(layer, "paint")["paint"]
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec=data,
        layer=layer_cls(
            id=layer.id,
            type=layer_types.HEATMAP,
            source=source_id,
            paint=dict(paint),
        ),
    )


def draw_clusters(_web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the clustered source and its three layers: bubbles, counts and loose points.

    One description carries all three because they are one thing to a viewer — removing the layer takes
    the counts and the loose points with it.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        data: The layer's source — the display-CRS point GeoDataFrame.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` whose `extra_layers` hold the counts and the
        unclustered points, drawn under the layer's own id suffixed ``-count`` and ``-unclustered``.

    Raises:
        ValueError: when the description records none of the values the clustering is built from —
            its colours, its radius or the zoom it stops clustering at — naming the layer, its kind and what is missing.
    """
    from maplibre.sources import GeoJSONSource, geopandas_to_geojson

    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, layer_types = _require_layer_api()
    props = required_props(layer, "color", "text_color", "radius", "max_zoom")
    source_id = f"{layer.id}-src"
    color, text_color = props["color"], props["text_color"]
    return DrawnLayer(
        source_id=source_id,
        source_spec=GeoJSONSource(
            data=geopandas_to_geojson(data),
            cluster=True,
            cluster_radius=int(props["radius"]),
            cluster_max_zoom=int(props["max_zoom"]),
        ),
        layer=layer_cls(
            id=layer.id,
            type=layer_types.CIRCLE,
            source=source_id,
            filter=["has", "point_count"],
            paint={
                "circle-color": color,
                "circle-radius": ["step", ["get", "point_count"], 15, 50, 20, 200, 25],
            },
        ),
        extra_layers=(
            layer_cls(
                id=f"{layer.id}-count",
                type=layer_types.SYMBOL,
                source=source_id,
                filter=["has", "point_count"],
                layout={
                    "text-field": ["get", "point_count_abbreviated"],
                    "text-size": 12,
                },
                paint={"text-color": text_color},
            ),
            layer_cls(
                id=f"{layer.id}-unclustered",
                type=layer_types.CIRCLE,
                source=source_id,
                filter=["!", ["has", "point_count"]],
                paint={"circle-color": color, "circle-radius": 5},
            ),
        ),
    )


class BigDataMixin(_MixinBase):
    """Heatmap / cluster / deck.gl builders for :class:`~digitalearth.web.map.WebMap`.

    A capability mixin of :class:`~digitalearth.web.map.WebMap`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the layer registry, the display CRS and the render/save
    lifecycle — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.web.base.WebMapBase`, the state ``WebMap`` inherits.
    At runtime that base is plain ``object``, so composing this mixin leaves the ``WebMap`` MRO exactly what it was
    before the annotation.

    See Also:
        digitalearth.web.map.WebMap: the composition that supplies the state these methods use.
        digitalearth.web.base.WebMapBase: the typing-only base declared above the class.
    """

    @staticmethod
    def _require_points(gdf: Any, method: str) -> None:
        """Raise ``TypeError`` unless every geometry in ``gdf`` is a point (the only kind these layers map).

        Args:
            gdf: The display-CRS GeoDataFrame about to be drawn.
            method: The calling builder name (quoted in the error).

        Raises:
            TypeError: when ``gdf`` holds any non-point geometry (MapLibre heatmap/clustering are point-only,
                so polygons/lines would render nothing).
        """
        kinds = set(gdf.geometry.geom_type.unique())
        if not kinds <= {"Point", "MultiPoint"}:
            raise TypeError(
                f"{method}() needs point geometries; got {sorted(kinds)}. Use choropleth/polygons for areas."
            )

    def heatmap(
        self,
        features: Any,
        *,
        weight: Optional[str] = None,
        radius: float = 30.0,
        intensity: float = 1.0,
        opacity: float = 0.8,
    ) -> Self:
        """Render a point ``FeatureCollection`` as a MapLibre heatmap (recipe W4).

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            weight: Optional value column; cells are weighted by it (normalised to ``[0, 1]``). ``None``
                weights every point equally.
            radius: Heat kernel radius in pixels.
            intensity: Global intensity multiplier.
            opacity: Heatmap layer opacity in ``[0, 1]``.

        Returns:
            The same map instance, so builder calls chain.
        """
        import numpy as np

        _require_layer_api()
        gdf = self._display_gdf(features, method="heatmap")
        self._require_points(gdf, "heatmap")
        paint: dict = {
            "heatmap-radius": float(radius),
            "heatmap-intensity": float(intensity),
            "heatmap-opacity": float(opacity),
        }
        if weight is not None:
            values = np.asarray(self._require_column(gdf, weight), dtype=float)
            finite = values[np.isfinite(values)]
            lo, hi = Scale.from_finite(finite).as_limits()

            paint["heatmap-weight"] = [
                "interpolate",
                ["linear"],
                ["get", weight],
                lo,
                0.0,
                hi,
                1.0,
            ]

        layer_id = self._uid("heatmap")
        self._index_layer(
            layer_id,
            None,
            kind="heatmap",
            source=gdf,
            symbology=Symbology(props={"paint": dict(paint)}),
        )
        self._last_layer_id = layer_id
        return self

    def cluster(
        self,
        features: Any,
        *,
        radius: int = 50,
        max_zoom: int = 14,
        color: str = "#51bbd6",
        text_color: str = "#ffffff",
    ) -> Self:
        """Render a point ``FeatureCollection`` as MapLibre clustered circles + count labels (recipe W4).

        Builds a clustered GeoJSON source and three layers: cluster bubbles (sized by point count), the count
        label, and the unclustered points. All three are **one description** under one id — the bubbles'
        — so a layer switcher lists them once and :meth:`~digitalearth.web.base.WebMapBase.remove_layer`
        takes all three off together. The count and the loose points are drawn beside it, under that id
        suffixed ``-count`` and ``-unclustered``.

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            radius: Cluster radius in pixels (MapLibre ``clusterRadius``).
            max_zoom: Zoom at/after which points stop clustering (``clusterMaxZoom``).
            color: Fill colour for cluster bubbles and unclustered points.
            text_color: Colour of the cluster count label.

        Returns:
            The same map instance, so builder calls chain.
        """

        _require_layer_api()
        gdf = self._display_gdf(features, method="cluster")
        self._require_points(gdf, "cluster")
        layer_id = self._uid("clusters")
        # One description covers the bubbles, their counts and the loose points, so removing it removes
        # all three together — they are one thing to a viewer.
        registered = self._index_layer(
            layer_id,
            None,
            kind="clusters",
            source=gdf,
            symbology=Symbology(
                props={
                    "color": color,
                    "text_color": text_color,
                    "radius": int(radius),
                    "max_zoom": int(max_zoom),
                }
            ),
        )
        if registered:
            # The loose points, which are the layer carrying the caller's own columns: the bubbles are
            # aggregates whose only properties are `cluster`, `cluster_id` and the point counts, so a popup
            # bound to them shows none of what was asked for (review H8). It is not the indexed id, so the
            # description is skipped — which is what the guard in `_record_tooltip` is for.
            self._last_layer_id = f"{layer_id}-unclustered"
        return self

    def _add_deck_layer(self, layer: dict) -> Self:
        """Accumulate a deck.gl JSON ``layer`` and ensure a single ``add_deck_layers`` application.

        All deck layers are applied together (deck.gl owns one overlay), so the first deck builder registers
        one applier bound to the shared list and later builders just append to it.

        Args:
            layer: A deck.gl JSON layer dict (``{"@@type": ..., ...}``).

        Returns:
            The same map instance, so builder calls chain.
        """
        if self._deck_layers is None:
            self._deck_layers = []
            deck_layers = self._deck_layers

            def apply(widget: Any) -> None:
                widget.add_deck_layers(deck_layers)

            self._queue(apply)
        self._deck_layers.append(layer)
        return self

    def deck_scatter(
        self,
        features: Any,
        *,
        fill_color: Sequence[int] = (51, 136, 255, 200),
        size: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> Self:
        """Render points as a GPU deck.gl ``GeoJsonLayer`` (recipe W3).

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            fill_color: RGBA fill colour (0-255 per channel).
            size: Point radius in pixels (``5.0`` when omitted — the signature's ``None`` is the "not
                passed" sentinel the deprecated spelling is resolved against). The same ``size`` that
                means marker size on every tier, and the same number
                :meth:`~digitalearth.web.vector.VectorMixin.points` hands down when it routes a large
                layer here.
            radius: **Deprecated** spelling of ``size``; forwarded unchanged, after a
                ``DeprecationWarning`` that ``radius=`` will be removed in a future release.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when ``features`` is a raster rather than a vector layer, or when both ``size``
                and the deprecated ``radius`` are passed — they name one parameter.

        Examples:
            - Build the overlay and read back the spec that will be handed to deck.gl (needs the
              ``web`` extra, so the block is skipped without it):
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import Point               # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     geometry=[Point(0, 0), Point(1, 1)], crs=4326,
                ... )
                >>> m = WebMap().deck_scatter(gdf, size=7.0)         # doctest: +SKIP
                >>> layer = m._deck_layers[0]                        # doctest: +SKIP
                >>> layer["@@type"], layer["getPointRadius"]         # doctest: +SKIP
                ('GeoJsonLayer', 7.0)

                ```
            - A deck overlay is drawn but is **not** a MapLibre style layer, so it never enters the
              registry a switcher toggles — which is why
              :meth:`~digitalearth.web.vector.VectorMixin.points` refuses ``name``/``visible`` when
              it routes a large table here:
                ```python
                >>> m.layer_ids                                      # doctest: +SKIP
                []

                ```
            - The old ``radius=`` spelling still lands on ``size``, after saying it is going away:
                ```python
                >>> import warnings                                  # doctest: +SKIP
                >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
                ...     warnings.simplefilter("always")
                ...     m = WebMap().deck_scatter(gdf, radius=4.0)
                >>> m._deck_layers[0]["getPointRadius"]              # doctest: +SKIP
                4.0
                >>> caught[0].category.__name__                      # doctest: +SKIP
                'DeprecationWarning'

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.points: the per-feature path, which routes
                here above ``big_data_threshold``.
            digitalearth.web.bigdata.BigDataMixin.deck_polygons: the polygon counterpart.
        """
        from maplibre.sources import geopandas_to_geojson

        _require_layer_api()
        size = renamed_parameter(
            new="size",
            value=size,
            old="radius",
            alias=radius,
            caller="WebMap.deck_scatter()",
            default=5.0,
        )
        gdf = self._display_gdf(features, method="deck_scatter")
        layer = {
            DECK_TYPE_KEY: "GeoJsonLayer",
            "id": self._uid("deck-scatter"),
            "data": geopandas_to_geojson(gdf),
            "pointType": "circle",
            "filled": True,
            "getFillColor": list(fill_color),
            "getPointRadius": float(size),
            "pointRadiusUnits": "pixels",
            "pointRadiusMinPixels": float(size),
        }
        return self._add_deck_layer(layer)

    def deck_polygons(
        self,
        features: Any,
        *,
        fill_color: Sequence[int] = (51, 136, 255, 180),
        line_color: Sequence[int] = (255, 255, 255, 255),
    ) -> Self:
        """Render polygons as a GPU deck.gl ``GeoJsonLayer`` (recipe W3).

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
            fill_color: RGBA fill colour (0-255 per channel).
            line_color: RGBA outline colour (0-255 per channel).

        Returns:
            The same map instance, so builder calls chain.
        """
        from maplibre.sources import geopandas_to_geojson

        _require_layer_api()
        gdf = self._display_gdf(features, method="deck_polygons")
        layer = {
            DECK_TYPE_KEY: "GeoJsonLayer",
            "id": self._uid("deck-polygons"),
            "data": geopandas_to_geojson(gdf),
            "filled": True,
            "stroked": True,
            "getFillColor": list(fill_color),
            "getLineColor": list(line_color),
            "lineWidthMinPixels": 1,
        }
        return self._add_deck_layer(layer)

    def _threshold(
        self, big_data_threshold: Optional[int] = None, *, caller: str = "WebMap"
    ) -> int:
        """Resolve the feature count that routes a layer to the GPU: the call's, else the map's.

        The two ways of setting it are deliberately the same name: assigning
        :attr:`~digitalearth.web.base.WebMapBase.big_data_threshold` on the map moves the cutoff for
        every layer that follows, while passing ``big_data_threshold=`` to one builder moves it for that
        call alone and leaves the map's setting untouched.

        A per-call cutoff is checked by :func:`~digitalearth.base.bigdata.validate_big_data_threshold`, the
        shared guard the interactive tier applies too — so a negative cutoff is refused identically on both
        rather than raising here and routing every layer, empty ones included, there.

        Args:
            big_data_threshold: The per-call override, or ``None`` to use the map's attribute.
            caller: The builder the keyword was written on, named in the error so the message points at the
                call rather than at this helper.

        Returns:
            The feature count above which a builder auto-routes to a deck.gl layer.

        Raises:
            ValueError: when the override is negative — a cutoff below zero routes everything, including
                an empty layer, which is never what the caller meant.
        """
        if big_data_threshold is None:
            return int(self.big_data_threshold)
        return validate_big_data_threshold(big_data_threshold, caller=caller)

    def _route_big(
        self, gdf: Any, kind: str, *, threshold: Optional[int] = None
    ) -> bool:
        """Whether ``gdf`` exceeds the big-data threshold — and log the routing decision when it does.

        Args:
            gdf: The display-CRS GeoDataFrame about to be drawn.
            kind: The builder name (for the log message).
            threshold: The caller's per-call ``big_data_threshold``; ``None`` uses the map's attribute.

        Returns:
            ``True`` when the feature count exceeds the resolved ``big_data_threshold`` (the caller should
            route to a GPU layer); ``False`` otherwise. The decision is logged when it fires (the M2
            "never silent" rule).
        """
        limit = self._threshold(threshold, caller=f"WebMap.{kind}()")
        n = len(gdf)
        if n > limit:
            logger.info(
                f"{kind}: {n} features exceed big_data_threshold={limit}; "
                "routing to a GPU deck.gl layer (pass big=False to keep per-feature rendering)"
            )
            return True
        return False
