"""BigDataMixin — web-tier big-data builders (DW.3, recipes W3/W4).

GPU-friendly renderers for large feature sets:

* ``heatmap`` — a MapLibre ``heatmap`` layer (optionally weighted by a column);
* ``cluster`` — a MapLibre clustered GeoJSON source with cluster bubbles, count labels and unclustered points;
* ``deck_scatter`` / ``deck_polygons`` — deck.gl ``GeoJsonLayer`` overlays driven through the maplibre widget's
  ``add_deck_layers`` (deck.gl JSON), so millions of features render on the GPU instead of one DOM glyph each.

The ``[web]`` extra ships ``lonboard`` for its GeoArrow path, but py-maplibregl's deck integration consumes
deck.gl **JSON / pydeck** layers, not lonboard widget objects — they cannot be composited into the same
``maplibre`` widget. We therefore drive deck.gl through ``add_deck_layers`` here; a lonboard-native (zero-copy
GeoArrow) renderer would be a separate widget and is left as a future enhancement.

Builders that colour by value reuse the base ``_color_expr`` helpers; numpy/maplibre are imported lazily.
"""

from typing import Any, Optional, Sequence

from loguru import logger

from digitalearth.web.base import _require_layer_api


class BigDataMixin:
    """Heatmap / cluster / deck.gl builders for :class:`~digitalearth.web.map.WebMap`."""

    def heatmap(
        self,
        features: Any,
        *,
        weight: Optional[str] = None,
        radius: float = 30.0,
        intensity: float = 1.0,
        opacity: float = 0.8,
    ) -> "BigDataMixin":
        """Render a point ``FeatureCollection`` as a MapLibre heatmap (recipe W4).

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            weight: Optional value column; cells are weighted by it (normalised to ``[0, 1]``). ``None``
                weights every point equally.
            radius: Heat kernel radius in pixels.
            intensity: Global intensity multiplier.
            opacity: Heatmap layer opacity in ``[0, 1]``.

        Returns:
            This map (chainable).
        """
        import numpy as np

        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features)
        paint: dict = {
            "heatmap-radius": float(radius),
            "heatmap-intensity": float(intensity),
            "heatmap-opacity": float(opacity),
        }
        if weight is not None:
            values = np.asarray(self._require_column(gdf, weight), dtype=float)
            finite = values[np.isfinite(values)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            paint["heatmap-weight"] = ["interpolate", ["linear"], ["get", weight], lo, 0.0, hi, 1.0]

        src_id, layer_id = self._uid("heat-src"), self._uid("heatmap")
        layer = Layer(id=layer_id, type=LayerType.HEATMAP, source=src_id, paint=paint)

        def apply(widget: Any) -> None:
            widget.add_source(src_id, gdf)
            widget.add_layer(layer)

        self._last_layer_id = layer_id
        return self.add_layer(layer=apply)

    def cluster(
        self,
        features: Any,
        *,
        radius: int = 50,
        max_zoom: int = 14,
        color: str = "#51bbd6",
        text_color: str = "#ffffff",
    ) -> "BigDataMixin":
        """Render a point ``FeatureCollection`` as MapLibre clustered circles + count labels (recipe W4).

        Builds a clustered GeoJSON source and three layers: cluster bubbles (sized by point count), the count
        label, and the unclustered points.

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            radius: Cluster radius in pixels (MapLibre ``clusterRadius``).
            max_zoom: Zoom at/after which points stop clustering (``clusterMaxZoom``).
            color: Fill colour for cluster bubbles and unclustered points.
            text_color: Colour of the cluster count label.

        Returns:
            This map (chainable).
        """
        from maplibre.sources import GeoJSONSource, geopandas_to_geojson

        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features)
        src_id = self._uid("cluster-src")
        source = GeoJSONSource(
            data=geopandas_to_geojson(gdf),
            cluster=True,
            cluster_radius=int(radius),
            cluster_max_zoom=int(max_zoom),
        )
        clusters = Layer(
            id=self._uid("clusters"),
            type=LayerType.CIRCLE,
            source=src_id,
            filter=["has", "point_count"],
            paint={
                "circle-color": color,
                "circle-radius": ["step", ["get", "point_count"], 15, 50, 20, 200, 25],
            },
        )
        count = Layer(
            id=self._uid("cluster-count"),
            type=LayerType.SYMBOL,
            source=src_id,
            filter=["has", "point_count"],
            layout={"text-field": ["get", "point_count_abbreviated"], "text-size": 12},
            paint={"text-color": text_color},
        )
        unclustered = Layer(
            id=self._uid("unclustered"),
            type=LayerType.CIRCLE,
            source=src_id,
            filter=["!", ["has", "point_count"]],
            paint={"circle-color": color, "circle-radius": 5},
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, source)
            widget.add_layer(clusters)
            widget.add_layer(count)
            widget.add_layer(unclustered)

        self._last_layer_id = unclustered.id
        return self.add_layer(layer=apply)

    def _add_deck_layer(self, layer: dict) -> "BigDataMixin":
        """Accumulate a deck.gl JSON ``layer`` and ensure a single ``add_deck_layers`` application.

        All deck layers are applied together (deck.gl owns one overlay), so the first deck builder registers
        one applier bound to the shared list and later builders just append to it.

        Args:
            layer: A deck.gl JSON layer dict (``{"@@type": ..., ...}``).

        Returns:
            This map (chainable).
        """
        if self._deck_layers is None:
            self._deck_layers = []
            deck_layers = self._deck_layers

            def apply(widget: Any) -> None:
                widget.add_deck_layers(deck_layers)

            self.add_layer(layer=apply)
        self._deck_layers.append(layer)
        return self

    def deck_scatter(
        self,
        features: Any,
        *,
        fill_color: Sequence[int] = (51, 136, 255, 200),
        radius: float = 5.0,
    ) -> "BigDataMixin":
        """Render points as a GPU deck.gl ``GeoJsonLayer`` (recipe W3).

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            fill_color: RGBA fill colour (0-255 per channel).
            radius: Point radius in pixels.

        Returns:
            This map (chainable).
        """
        from maplibre.sources import geopandas_to_geojson

        _require_layer_api()
        gdf = self._display_gdf(features)
        layer = {
            "@@type": "GeoJsonLayer",
            "id": self._uid("deck-scatter"),
            "data": geopandas_to_geojson(gdf),
            "pointType": "circle",
            "filled": True,
            "getFillColor": list(fill_color),
            "getPointRadius": float(radius),
            "pointRadiusUnits": "pixels",
            "pointRadiusMinPixels": float(radius),
        }
        return self._add_deck_layer(layer)

    def deck_polygons(
        self,
        features: Any,
        *,
        fill_color: Sequence[int] = (51, 136, 255, 180),
        line_color: Sequence[int] = (255, 255, 255, 255),
    ) -> "BigDataMixin":
        """Render polygons as a GPU deck.gl ``GeoJsonLayer`` (recipe W3).

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
            fill_color: RGBA fill colour (0-255 per channel).
            line_color: RGBA outline colour (0-255 per channel).

        Returns:
            This map (chainable).
        """
        from maplibre.sources import geopandas_to_geojson

        _require_layer_api()
        gdf = self._display_gdf(features)
        layer = {
            "@@type": "GeoJsonLayer",
            "id": self._uid("deck-polygons"),
            "data": geopandas_to_geojson(gdf),
            "filled": True,
            "stroked": True,
            "getFillColor": list(fill_color),
            "getLineColor": list(line_color),
            "lineWidthMinPixels": 1,
        }
        return self._add_deck_layer(layer)

    def _route_big(self, gdf: Any, kind: str) -> bool:
        """Whether ``gdf`` exceeds the big-data threshold — and log the routing decision when it does.

        Args:
            gdf: The display-CRS GeoDataFrame about to be drawn.
            kind: The builder name (for the log message).

        Returns:
            ``True`` when the feature count exceeds ``big_data_threshold`` (the caller should route to a GPU
            layer); ``False`` otherwise. The decision is logged when it fires (the M2 "never silent" rule).
        """
        n = len(gdf)
        if n > self.big_data_threshold:
            logger.info(
                f"{kind}: {n} features exceed big_data_threshold={self.big_data_threshold}; "
                "routing to a GPU deck.gl layer (pass big=False to keep per-feature rendering)"
            )
            return True
        return False
