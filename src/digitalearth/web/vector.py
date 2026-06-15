"""VectorMixin — web-tier vector builders (DW.2, recipe W2).

``points`` / ``lines`` / ``polygons`` turn a pyramids ``FeatureCollection`` (reprojected to lon/lat through
pyramids) into MapLibre circle / line / fill layers over a GeoJSON source; ``choropleth`` is the thematic
polygon map. Colour-by-value compiles into a MapLibre **data-driven paint expression**:

* graduated (``scheme`` set) → a ``["step", ["get", col], …]`` expression whose breaks come from
  **cleopatra.styles.classify** (pure-numpy quantiles / equal-interval / Fisher-Jenks — no mapclassify), the
  same classifier the static tier uses, so the classes match across tiers;
* continuous (``scheme=None``) → an ``["interpolate", ["linear"], ["get", col], …]`` colour ramp.

cleopatra / matplotlib / numpy are imported lazily inside the methods; importing the tier needs none of them.
"""

from typing import Any, List, Optional

from loguru import logger

from digitalearth.web.base import _require_layer_api


class VectorMixin:
    """Point / line / polygon / choropleth builders for :class:`~digitalearth.web.map.WebMap`."""

    def _color_expr(
        self,
        values: Any,
        column: str,
        scheme: Optional[Any],
        k: int,
        cmap: str,
    ) -> list:
        """Compile a MapLibre data-driven colour expression for ``column`` and record the breaks.

        Args:
            values: The 1-D value array driving the colour (used to compute breaks / limits).
            column: The GeoJSON property name the expression reads with ``["get", column]``.
            scheme: A cleopatra classification scheme (``"quantiles"``/``"fisher_jenks"``/… or an explicit
                edge sequence) for a graduated ``step`` expression; ``None`` for a continuous ramp.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap name sampled for the class / ramp colours.

        Returns:
            A MapLibre expression list (``["step", …]`` or ``["interpolate", …]``). Also sets
            ``self.last_breaks`` to the breaks (graduated edges) or ramp stops.

        Raises:
            ValueError: propagated from ``cleopatra.styles.classify`` (unknown scheme, no spread, …).
        """
        import numpy as np

        if scheme is not None:
            from cleopatra.styles import classify

            edges, _ = classify(values, scheme, k)
            colors = self._cmap_hex(cmap, len(edges) - 1)
            expr: list = ["step", ["get", column], colors[0]]
            for edge, color in zip(edges[1:-1], colors[1:]):
                expr.extend([float(edge), color])
            self.last_breaks = [float(e) for e in edges]
            return expr

        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"column {column!r} has no finite values to colour")
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        stops = np.linspace(lo, hi, 5)
        colors = self._cmap_hex(cmap, len(stops))
        expr = ["interpolate", ["linear"], ["get", column]]
        for stop, color in zip(stops, colors):
            expr.extend([float(stop), color])
        self.last_breaks = [float(s) for s in stops]
        return expr

    def _vector_layer(
        self,
        features: Any,
        prefix: str,
        layer_type: Any,
        paint: dict,
    ) -> "VectorMixin":
        """Register a GeoJSON source + a typed layer with ``paint`` and record it as the last data layer.

        Args:
            features: The display-CRS GeoDataFrame to serve as the GeoJSON source.
            prefix: The id prefix / kind tag (``"circle"``/``"line"``/``"fill"``).
            layer_type: The ``maplibre`` ``LayerType`` member for the layer.
            paint: The MapLibre paint dict for the layer.

        Returns:
            This map (chainable).
        """
        Layer, _ = _require_layer_api()
        src_id, layer_id = self._uid(f"{prefix}-src"), self._uid(prefix)
        layer = Layer(id=layer_id, type=layer_type, source=src_id, paint=paint)

        def apply(widget: Any) -> None:
            widget.add_source(src_id, features)
            widget.add_layer(layer)

        self._last_layer_id = layer_id
        return self.add_layer(apply)

    @staticmethod
    def _require_column(gdf: Any, column: str) -> Any:
        """Return ``gdf[column]`` as a numpy array, raising a clear error when the column is absent."""
        if column not in getattr(gdf, "columns", []):
            raise KeyError(f"column {column!r} not found in the feature attributes")
        return gdf[column].to_numpy()

    def points(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        radius: float = 5.0,
        color: str = "#3388ff",
        opacity: float = 0.9,
        big: Optional[bool] = None,
    ) -> "VectorMixin":
        """Draw a point ``FeatureCollection`` as a MapLibre circle layer (recipe W2).

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            column: Optional value column; when given, circles are coloured by it (graduated if ``scheme``
                is set, else a continuous ramp).
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            radius: Circle radius in pixels.
            color: Fixed circle colour used when ``column`` is ``None``.
            opacity: Circle fill opacity in ``[0, 1]``.
            big: Big-data routing — ``None`` (default) auto-routes to a GPU deck.gl layer above
                ``big_data_threshold`` (logged); ``False`` forces per-feature circles; ``True`` forces deck.gl.

        Returns:
            This map (chainable).
        """
        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features)
        # Auto-route to a GPU deck.gl layer only when there is no per-feature symbology to preserve; a forced
        # big=True with a column still routes but warns that the deck path drops the colouring (M1).
        if big or (big is None and column is None and self._route_big(gdf, "points")):
            if column is not None:
                logger.warning(
                    "points: big=True routes {} features to a flat deck.gl layer; column={!r} styling is dropped",
                    len(gdf),
                    column,
                )
            return self.deck_scatter(gdf, radius=radius)
        paint: dict = {"circle-radius": float(radius), "circle-opacity": float(opacity)}
        if column is not None:
            paint["circle-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["circle-color"] = color
        return self._vector_layer(gdf, "circle", LayerType.CIRCLE, paint)

    def lines(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        width: float = 2.0,
        color: str = "#3388ff",
        opacity: float = 1.0,
    ) -> "VectorMixin":
        """Draw a line ``FeatureCollection`` as a MapLibre line layer (recipe W2).

        Args:
            features: A pyramids line ``FeatureCollection`` / GeoDataFrame.
            column: Optional value column to colour the lines by.
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            width: Line width in pixels.
            color: Fixed line colour used when ``column`` is ``None``.
            opacity: Line opacity in ``[0, 1]``.

        Returns:
            This map (chainable).
        """
        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features)
        paint: dict = {"line-width": float(width), "line-opacity": float(opacity)}
        if column is not None:
            paint["line-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["line-color"] = color
        return self._vector_layer(gdf, "line", LayerType.LINE, paint)

    def polygons(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        color: str = "#3388ff",
        opacity: float = 0.6,
        outline_color: str = "#ffffff",
        big: Optional[bool] = None,
    ) -> "VectorMixin":
        """Draw a polygon ``FeatureCollection`` as a MapLibre fill layer (recipe W2).

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
            column: Optional value column to colour the polygons by (graduated if ``scheme`` is set, else a
                continuous ramp). For full thematic symbology prefer :meth:`choropleth`.
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            color: Fixed fill colour used when ``column`` is ``None``.
            opacity: Fill opacity in ``[0, 1]``.
            outline_color: Polygon outline colour.
            big: Big-data routing — ``None`` (default) auto-routes to a GPU deck.gl layer above
                ``big_data_threshold`` (logged); ``False`` forces per-feature fills; ``True`` forces deck.gl.

        Returns:
            This map (chainable).
        """
        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features)
        # Auto-route to deck.gl only when no column styling would be lost; a forced big=True with a column
        # still routes but warns that the deck path drops the colouring (M1).
        if big or (big is None and column is None and self._route_big(gdf, "polygons")):
            if column is not None:
                logger.warning(
                    "polygons: big=True routes {} features to a flat deck.gl layer; column={!r} styling is dropped",
                    len(gdf),
                    column,
                )
            return self.deck_polygons(gdf)
        paint: dict = {"fill-opacity": float(opacity), "fill-outline-color": outline_color}
        if column is not None:
            paint["fill-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["fill-color"] = color
        return self._vector_layer(gdf, "fill", LayerType.FILL, paint)

    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        scheme: Optional[Any] = "quantiles",
        k: int = 5,
        cmap: str = "viridis",
        opacity: float = 0.85,
        outline_color: str = "#ffffff",
    ) -> "VectorMixin":
        """Draw a thematic polygon choropleth coloured by ``column`` (recipe W2).

        Graduated by default (``scheme="quantiles"``): the class breaks come from
        ``cleopatra.styles.classify`` and are compiled into a MapLibre ``step`` paint expression, so the
        classes match the static tier's ``choropleth``. Pass ``scheme=None`` for a continuous ramp. The
        breaks are exposed on ``WebMap.last_breaks`` for building a legend.

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
            column: The numeric attribute that colours the polygons (required).
            scheme: A cleopatra classification scheme (``"quantiles"``, ``"equal_interval"``,
                ``"fisher_jenks"``, …) or an explicit edge sequence; ``None`` for a continuous ramp.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap name.
            opacity: Fill opacity in ``[0, 1]``.
            outline_color: Polygon outline colour.

        Returns:
            This map (chainable).

        Raises:
            KeyError: when ``column`` is not a feature attribute.
            ValueError: propagated from the classifier (unknown scheme, constant data, …).
        """
        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features)
        values = self._require_column(gdf, column)
        paint = {
            "fill-color": self._color_expr(values, column, scheme, k, cmap),
            "fill-opacity": float(opacity),
            "fill-outline-color": outline_color,
        }
        return self._vector_layer(gdf, "fill", LayerType.FILL, paint)
