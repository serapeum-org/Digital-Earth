"""VectorMixin — vector builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``points`` / ``path`` / ``polygons`` / ``choropleth`` (DI.1b); vector fields (DI.5), meshes/density
(DI.6) and graphs (DI.15) land later.

Vector layers come straight from the GeoDataFrame pyramids hands over (``FeatureCollection`` *is a*
GeoDataFrame) — GeoViews reads the geometry column natively, so **no shapely/geopandas import** is ever
needed here. The data CRS is declared via ``gv.util.process_crs(self.crs)``: GeoViews builds the cartopy
CRS object *internally* (the hvPlot pattern), keeping cartopy out of this package (DX.3). Reprojection to
the display CRS happens upstream in pyramids (``FeatureCollection.to_crs``) before the element is built.
"""

from typing import Any, Optional, Tuple

from digitalearth.interactive.base import _require_holoviz


class VectorMixin:
    """Vector builders (DI.1b): point, line and polygon layers with hover tooltips."""

    def _display_gdf(self, features: Any) -> Any:
        """Reproject ``features`` to the display CRS through pyramids and return the GeoDataFrame.

        Args:
            features: A pyramids ``FeatureCollection`` (or any GeoDataFrame-like with
                ``epsg``/``to_crs``); plain GeoDataFrames already in the display CRS pass through.

        Returns:
            The display-CRS GeoDataFrame (read-only from here on — geometry is never rebuilt).
        """
        if (
            hasattr(features, "epsg")
            and hasattr(features, "to_crs")
            and self._needs_reproject(features)
        ):
            features = features.to_crs(self.crs)
        return features

    def _vector_element(self, kind: str, gdf: Any, vdims: Optional[list] = None) -> Any:
        """Build a GeoViews element of ``kind`` from a display-CRS GeoDataFrame.

        The element declares its (already display) CRS via ``gv.util.process_crs`` — GeoViews builds
        the cartopy CRS internally, so this package never imports cartopy (DX.3). The ``datatype`` is
        pinned to the geodataframe interface: pyramids' ``FeatureCollection`` is a GeoDataFrame
        *subclass*, which HoloViews' narwhals interface otherwise claims first and mis-reads the
        attribute columns as key dimensions.

        Args:
            kind: One of ``"Points"`` / ``"Path"`` / ``"Polygons"``.
            gdf: The display-CRS GeoDataFrame.
            vdims: Optional value dimensions (attribute columns) carried for colour/hover; ``None``
                lets GeoViews infer them from the non-geometry columns.

        Returns:
            The GeoViews element.
        """
        gv, hv = _require_holoviz()
        crs = gv.util.process_crs(self.crs)
        factory = getattr(gv, kind)
        # geodataframe first so the GeoPandas interface wins at construction; the rest keep
        # render-time clones (projection produces dict/array data) dispatchable.
        datatype = ["geodataframe", "multitabular", "dictionary", "dataframe", "array"]
        kwargs: dict = {"crs": crs, "datatype": datatype}
        if vdims:
            kwargs["vdims"] = vdims
        return factory(gdf, **kwargs)

    def points(
        self,
        features: Any,
        *,
        value_column: Optional[str] = None,
        size: float = 6.0,
        cmap: str = "viridis",
        rasterize: Any = "auto",
        rasterize_threshold: int = 50_000,
        **opts: Any,
    ) -> "VectorMixin":
        """Add a point layer, optionally coloured by an attribute column.

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries; reprojected to the
                display CRS through pyramids when needed.
            value_column: Optional numeric column colouring the points (also shown on hover).
            size: Marker size in screen pixels.
            cmap: Colormap used when ``value_column`` is given.
            rasterize: ``"auto"`` (default) routes through Datashader above
                ``rasterize_threshold`` rows — logged, never silent; ``True``/``False`` force it.
            rasterize_threshold: Row count above which ``"auto"`` switches to Datashader.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Colour gauging stations by an attribute column:
                ```python
                >>> from pyramids.feature import FeatureCollection              # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")  # doctest: +SKIP
                >>> m = InteractiveMap().points(fc, value_column="fid")         # doctest: +SKIP
                >>> [d.name for d in m.layers[0].vdims]                         # doctest: +SKIP
                ['fid']

                ```

        Returns:
            This map (chainable).
        """
        from digitalearth.interactive.bigdata import _route_through_rasterize

        gdf = self._display_gdf(features)
        if rasterize is True or (
            rasterize == "auto"
            and _route_through_rasterize(self, "points", len(gdf), rasterize_threshold)
        ):
            aggregator = "mean" if value_column else "count"
            return self.rasterize(
                gdf, aggregator=aggregator, column=value_column, cmap=cmap, **opts
            )
        element = self._vector_element(
            "Points", gdf, vdims=[value_column] if value_column else None
        )
        common: dict = {"size": size, **opts}
        if value_column:
            common.update({"color": value_column, "cmap": cmap, "colorbar": True})
        element = self._styled(element, common=common, bokeh={"tools": ["hover"]})
        return self.add_element(element)

    def path(self, features: Any, **opts: Any) -> "VectorMixin":
        """Add a line layer (LineString / MultiLineString features).

        Args:
            features: A pyramids ``FeatureCollection`` of line geometries; reprojected through
                pyramids when needed.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Draw river reaches as an interactive line layer:
                ```python
                >>> from pyramids.feature import FeatureCollection              # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> reaches = FeatureCollection.read_file("reaches.geojson")    # doctest: +SKIP
                >>> InteractiveMap().path(reaches).save("reaches.html")         # doctest: +SKIP
                'reaches.html'

                ```

        Returns:
            This map (chainable).
        """
        gdf = self._display_gdf(features)
        element = self._vector_element("Path", gdf)
        element = self._styled(element, common=opts or None, bokeh={"tools": ["hover"]})
        return self.add_element(element)

    def polygons(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        cmap: str = "viridis",
        rasterize: Any = "auto",
        rasterize_threshold: int = 50_000,
        **opts: Any,
    ) -> "VectorMixin":
        """Add a polygon layer — outlines only, or filled by an attribute column.

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries; reprojected through
                pyramids when needed.
            column: Optional numeric column filling the polygons (also shown on hover); ``None``
                draws unfilled outlines.
            cmap: Colormap used when ``column`` is given.
            rasterize: ``"auto"`` (default) routes through Datashader above
                ``rasterize_threshold`` rows — logged, never silent; ``True``/``False`` force it.
                Polygon datashading needs the optional ``spatialpandas`` package.
            rasterize_threshold: Row count above which ``"auto"`` switches to Datashader.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Outline catchment polygons over a basemap:
                ```python
                >>> from pyramids.feature import FeatureCollection              # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> basins = FeatureCollection.read_file("basins.geojson")      # doctest: +SKIP
                >>> InteractiveMap().polygons(basins).tiles().save("m.html")    # doctest: +SKIP
                'm.html'

                ```

        Returns:
            This map (chainable).
        """
        from digitalearth.interactive.bigdata import _route_through_rasterize

        gdf = self._display_gdf(features)
        if rasterize is True or (
            rasterize == "auto"
            and _route_through_rasterize(
                self, "polygons", len(gdf), rasterize_threshold
            )
        ):
            from importlib.util import find_spec

            if (
                find_spec("spatialpandas") is None
            ):  # datashader's polygon backend (its dep, not ours)
                raise ImportError(
                    "polygon datashading needs the optional spatialpandas package "
                    "(pip install spatialpandas) — or pass rasterize=False to draw raw glyphs"
                )
            element = self._vector_element(  # pragma: no cover - needs optional spatialpandas
                "Polygons", gdf, vdims=[column] if column else None
            )
            return self.rasterize(  # pragma: no cover - needs optional spatialpandas
                element,
                aggregator="mean" if column else "count",
                column=column,
                cmap=cmap,
                **opts,
            )
        element = self._vector_element(
            "Polygons", gdf, vdims=[column] if column else None
        )
        common: dict = dict(opts)
        if column:
            common.update({"color": column, "cmap": cmap, "colorbar": True})
        else:
            common.setdefault("fill_alpha", 0.0)
        element = self._styled(element, common=common, bokeh={"tools": ["hover"]})
        return self.add_element(element)

    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        cmap: str = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        **opts: Any,
    ) -> "VectorMixin":
        """Add a choropleth — polygons filled and coloured by ``column`` (hover shows the value).

        A thin colour-by-attribute :meth:`polygons`, mirroring the static ``Map.choropleth``.

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries; reprojected through
                pyramids when needed.
            column: The numeric column driving the fill colour (required).
            cmap: Colormap name.
            clim: Optional ``(vmin, vmax)`` colour limits; ``None`` auto-scales.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Examples:
            - Fill polygons by a population column with fixed colour limits:
                ```python
                >>> from pyramids.feature import FeatureCollection              # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> admin = FeatureCollection.read_file("admin.geojson")        # doctest: +SKIP
                >>> m = InteractiveMap().choropleth(admin, "pop", clim=(0, 1e6))  # doctest: +SKIP
                >>> [d.name for d in m.layers[0].vdims]                         # doctest: +SKIP
                ['pop']

                ```

        Raises:
            KeyError: when ``column`` is not a column of ``features``.
        """
        if column not in getattr(features, "columns", [column]):
            raise KeyError(
                f"choropleth column {column!r} not found in the feature attributes"
            )
        if clim is not None:
            opts = {"clim": clim, **opts}
        return self.polygons(features, column=column, cmap=cmap, **opts)
