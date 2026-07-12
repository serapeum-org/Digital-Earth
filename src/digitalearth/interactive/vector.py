"""VectorMixin — vector builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``points`` / ``path`` / ``polygons`` / ``choropleth`` (DI.1b) and the u/v vector fields
``vectorfield`` / ``streamlines`` / ``barbs`` (DI.5, recipe I6); meshes/density (DI.6) and graphs
(DI.15) land later.

Vector layers come straight from the GeoDataFrame pyramids hands over (``FeatureCollection`` *is a*
GeoDataFrame) — GeoViews reads the geometry column natively, so **no shapely/geopandas import** is ever
needed here. The data CRS is declared via ``gv.util.process_crs(self.crs)``: GeoViews builds the cartopy
CRS object *internally* (the hvPlot pattern), keeping cartopy out of this package (DX.3). Reprojection to
the display CRS happens upstream in pyramids (``FeatureCollection.to_crs``) before the element is built.

**Barbs are matplotlib-only** — Bokeh has no wind-barb glyph, so ``barbs`` renders through HoloViews'
matplotlib backend (a static PNG via ``save``); it logs that it is not interactive rather than silently
producing an empty Bokeh layer.
"""

from typing import Any, Optional, Tuple

from digitalearth.interactive.base import _masked_to_nan, _require_holoviz


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
            and _route_through_rasterize("points", len(gdf), rasterize_threshold)
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
            and _route_through_rasterize("polygons", len(gdf), rasterize_threshold)
        ):
            from importlib.util import find_spec

            if (
                find_spec("spatialpandas") is None
            ):  # datashader's polygon backend (its dep, not ours)
                raise ImportError(
                    "polygon datashading needs the optional spatialpandas package "
                    "(pip install spatialpandas) — or pass rasterize=False to draw raw glyphs"
                )
            element = (
                self._vector_element(  # pragma: no cover - needs optional spatialpandas
                    "Polygons", gdf, vdims=[column] if column else None
                )
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

    def _categorical_polygons(self, features: Any, column: str, *, cmap: str = "viridis",
                              **opts: Any) -> "VectorMixin":
        """Fill polygons by a distinct-value attribute, one colour per category (DC.8).

        The categorical counterpart of the continuous :meth:`polygons` path: each distinct value of ``column``
        gets a colour from :func:`digitalearth._symbology.categorical_colors`. The colour column is cast to
        **string** and the colours are handed to GeoViews as a ``{label: colour}`` dict ``cmap`` — a numeric
        column would otherwise be treated as a continuous dimension and the palette interpolated, so this is
        what guarantees one discrete colour per distinct value (no continuous colorbar). Missing values
        (``NaN``/``None``) are drawn with a neutral ``"#cccccc"`` fallback, matching the web tier's default.
        The categories (original values, in classifier order) are recorded on ``last_breaks`` for legend parity
        with the web tier.

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected through pyramids).
            column: The attribute to colour by (any hashable value).
            cmap: A qualitative colormap name (defaults to ``"tab10"`` when left at the continuous default).
            **opts: Extra HoloViews style options.

        Returns:
            This map (chainable).
        """
        from digitalearth._symbology import categorical_colors, resolve_categorical_cmap

        gdf = self._display_gdf(features)
        categories, colors = categorical_colors(gdf[column], resolve_categorical_cmap(cmap))
        # Render the column as discrete labels and map each label to its colour, so Bokeh colours it
        # categorically (a numeric column would map continuously and interpolate the palette).
        gdf = gdf.copy()
        missing = gdf[column].isna()
        gdf[column] = gdf[column].astype(str).mask(missing, "n/a")
        cmap_by_label = {str(category): color for category, color in zip(categories, colors)}
        if missing.any():
            # Missing values get an explicit neutral fallback, matching the web tier's "#cccccc" default.
            cmap_by_label["n/a"] = "#cccccc"
        element = self._vector_element("Polygons", gdf, vdims=[column])
        common = {"color": column, "cmap": cmap_by_label, "colorbar": False, **opts}
        element = self._styled(element, common=common, bokeh={"tools": ["hover"]})
        self.last_breaks = list(categories)
        return self.add_element(element)

    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        scheme: Optional[str] = None,
        cmap: str = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        **opts: Any,
    ) -> "VectorMixin":
        """Add a choropleth — polygons filled and coloured by ``column`` (hover shows the value).

        A thin colour-by-attribute :meth:`polygons`, mirroring the static ``Map.choropleth``. Pass
        ``scheme="categorical"`` to colour an unordered attribute by distinct value instead of a continuous
        ramp (DC.8); the categories are recorded on ``last_breaks`` for legend parity with the web tier.

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries; reprojected through
                pyramids when needed.
            column: The column driving the fill colour (required); numeric for the continuous ramp, or any
                hashable value for ``scheme="categorical"``.
            scheme: ``"categorical"`` for distinct-value colouring; ``None`` (default) for a continuous ramp.
                Graduated schemes (``"quantiles"``/``"fisher_jenks"``/…) are **not** supported in the
                interactive tier and raise ``NotImplementedError`` rather than silently degrading.
            cmap: Colormap name (a qualitative map such as ``"tab10"`` is used for the categorical scheme when
                left at the default).
            clim: Optional ``(vmin, vmax)`` colour limits for the continuous ramp; ``None`` auto-scales.
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
            NotImplementedError: when ``scheme`` is a graduated scheme (only ``"categorical"``/``None`` are
                supported in the interactive tier).
        """
        if column not in getattr(features, "columns", [column]):
            raise KeyError(
                f"choropleth column {column!r} not found in the feature attributes"
            )
        if isinstance(scheme, str) and scheme.lower() == "categorical":
            return self._categorical_polygons(features, column, cmap=cmap, **opts)
        if scheme is not None:
            raise NotImplementedError(
                f"interactive choropleth supports scheme='categorical' or None (continuous ramp); the "
                f"graduated scheme {scheme!r} is not implemented in the interactive tier — use the web tier "
                f"for graduated classification, or pass scheme=None for a continuous ramp"
            )
        # Continuous ramp: no discrete breaks — clear any recorded from a prior categorical call.
        self.last_breaks = None
        if clim is not None:
            opts = {"clim": clim, **opts}
        return self.polygons(features, column=column, cmap=cmap, **opts)

    def _uv_arrays(self, u: Any, v: Any, *, band: int, density: float) -> tuple:
        """Extract subsampled ``(x, y, u, v)`` display-CRS arrays from two pyramids bands.

        Args:
            u: The eastward-component ``Dataset`` / ``Source`` (reprojected through pyramids).
            v: The northward-component ``Dataset`` / ``Source`` (same grid as ``u``).
            band: 1-based band read from each.
            density: Keep-fraction in ``(0, 1]`` — the grid is strided by ``round(1/density)`` so the
                field stays legible at web resolution (``1.0`` keeps every cell).

        Returns:
            ``(x, y, u, v)`` — 1-D ``x``/``y`` cell-centre coords and 2-D ``u``/``v`` arrays, strided.

        Raises:
            ValueError: when ``density`` is not in ``(0, 1]``.
        """
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {density!r}")
        su = self._to_display_source(u, band=band)
        sv = self._to_display_source(v, band=band)
        step = max(1, round(1.0 / density))
        x = su.x.values[::step]
        y = su.y.values[::step]
        u_arr = _masked_to_nan(su.z.values)[::step, ::step]
        v_arr = _masked_to_nan(sv.z.values)[::step, ::step]
        return x, y, u_arr, v_arr

    def vectorfield(
        self,
        u: Any,
        v: Any,
        *,
        band: int = 1,
        density: float = 1.0,
        color_by: Optional[str] = "magnitude",
        cmap: str = "viridis",
        **opts: Any,
    ) -> "VectorMixin":
        """Add a u/v vector field as interactive arrows (parity with ``Map.quiver``, recipe I6).

        Args:
            u: Eastward-component ``Dataset`` / ``Source``; reprojected through pyramids.
            v: Northward-component ``Dataset`` / ``Source`` on the same grid.
            band: 1-based band read from each component.
            density: Keep-fraction in ``(0, 1]`` controlling arrow density (``1.0`` = every cell).
            color_by: ``"magnitude"`` colours arrows by speed; ``None`` draws uniform arrows.
            cmap: Colormap used when ``color_by="magnitude"``.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        x, y, u_arr, v_arr = self._uv_arrays(u, v, band=band, density=density)
        element = gv.VectorField.from_uv(
            (x, y, u_arr, v_arr), crs=gv.util.process_crs(self.crs)
        )
        common: dict = dict(opts)
        if color_by == "magnitude":
            common.update({"color": "Magnitude", "cmap": cmap, "colorbar": True})
        element = self._styled(element, common=common, bokeh={"tools": ["hover"]})
        return self.add_element(element)

    def streamlines(
        self, u: Any, v: Any, *, band: int = 1, density: float = 1.0, **opts: Any
    ) -> "VectorMixin":
        """Add streamlines of a u/v field via the matplotlib backend (parity with ``Map.streamplot``).

        Bokeh has no streamline integrator, so streamlines render through HoloViews' matplotlib
        backend (a static layer in the saved PNG). The element is built so ``save("x.png")`` works;
        it is logged as non-interactive rather than emitting an empty Bokeh layer.

        Args:
            u: Eastward-component ``Dataset`` / ``Source``; reprojected through pyramids.
            v: Northward-component ``Dataset`` / ``Source`` on the same grid.
            band: 1-based band read from each component.
            density: Keep-fraction in ``(0, 1]`` subsampling the field before streamline
                integration (same meaning as :meth:`vectorfield`'s ``density``).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).
        """
        from loguru import logger

        gv, hv = _require_holoviz()
        x, y, u_arr, v_arr = self._uv_arrays(u, v, band=band, density=density)
        # VectorField carries the data; the matplotlib backend renders it as streamlines.
        element = gv.VectorField.from_uv(
            (x, y, u_arr, v_arr), crs=gv.util.process_crs(self.crs)
        )
        element = element.opts(backend="matplotlib", **opts) if opts else element
        logger.info(
            "streamlines render through the matplotlib backend (Bokeh has no streamline glyph); "
            "save to a .png/.svg, not interactive .html"
        )
        return self.add_element(element)

    def barbs(
        self, u: Any, v: Any, *, band: int = 1, density: float = 1.0, **opts: Any
    ) -> "VectorMixin":
        """Add wind barbs of a u/v field — **matplotlib backend only** (parity with ``Map.barbs``).

        ``gv.WindBarbs`` has no Bokeh renderer, so barbs are a static matplotlib layer; this logs
        that they are non-interactive rather than silently producing an empty Bokeh layer.

        Args:
            u: Eastward-component ``Dataset`` / ``Source``; reprojected through pyramids.
            v: Northward-component ``Dataset`` / ``Source`` on the same grid.
            band: 1-based band read from each component.
            density: Keep-fraction in ``(0, 1]`` subsampling the field before drawing barbs (same
                meaning as :meth:`vectorfield`'s ``density``; ``1.0`` keeps every cell).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Raises:
            ImportError: when the installed GeoViews has no ``WindBarbs`` element.
            ValueError: when ``density`` is not in ``(0, 1]``.
        """
        from loguru import logger

        gv, hv = _require_holoviz()
        if not hasattr(gv, "WindBarbs"):
            raise ImportError(
                "barbs need gv.WindBarbs (GeoViews ≥1.11 with the matplotlib backend); it is "
                "absent in this GeoViews build"
            )
        x, y, u_arr, v_arr = self._uv_arrays(u, v, band=band, density=density)
        element = gv.WindBarbs.from_uv(
            (x, y, u_arr, v_arr), crs=gv.util.process_crs(self.crs)
        )
        element = element.opts(backend="matplotlib", **opts) if opts else element
        logger.info(
            "barbs render through the matplotlib backend only (Bokeh has no wind-barb glyph); "
            "save to a .png/.svg, not interactive .html"
        )
        return self.add_element(element)

    def trimesh(
        self,
        data: Any,
        *,
        value_column: Optional[str] = None,
        rasterize: Any = "auto",
        rasterize_threshold: int = 50_000,
        cmap: str = "viridis",
        **opts: Any,
    ) -> "VectorMixin":
        """Add an unstructured triangular mesh (parity with ``Map.tricontour``/``tripcolor``, recipe I7).

        Connectivity comes from one of two sources, both pyramids-fed:

        - a **true UGRID mesh** (``pyramids.netcdf.ugrid.Mesh2d`` — anything exposing ``node_x`` /
          ``node_y`` / ``fan_triangles``): nodes + triangles are taken straight off it, no
          triangulation needed;
        - a **point** ``FeatureCollection``: Delaunay-triangulated locally with ``matplotlib.tri``
          (matplotlib is not a forbidden GIS engine), mirroring the static ``tri*`` path.

        Above ``rasterize_threshold`` faces the mesh auto-``rasterize``s to a density image (logged).

        Args:
            data: A UGRID-mesh object or a point ``FeatureCollection`` (reprojected via pyramids).
            value_column: Node-value column (for the FeatureCollection path) driving the colour.
            rasterize: ``"auto"`` (default) rasterizes above the threshold; ``True``/``False`` force it.
            rasterize_threshold: Face count above which ``"auto"`` rasterizes.
            cmap: Colormap name.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        nodes, simplices, vdims = self._mesh_inputs(data, value_column)
        trimesh = gv.TriMesh((simplices, nodes), crs=gv.util.process_crs(self.crs))
        n_faces = len(simplices)
        if rasterize is True or (rasterize == "auto" and n_faces > rasterize_threshold):
            from loguru import logger

            if rasterize == "auto":
                logger.info(
                    f"trimesh: {n_faces:,} faces exceed rasterize_threshold={rasterize_threshold:,}"
                    " — rasterizing the mesh to a density image"
                )
            return self.rasterize(trimesh, dynamic=True, cmap=cmap, **opts)
        common = {"cmap": cmap, **opts} if vdims else dict(opts)
        element = self._styled(
            trimesh, common=common or None, bokeh={"tools": ["hover"]}
        )
        return self.add_element(element)

    def _mesh_inputs(self, data: Any, value_column: Optional[str]) -> tuple:
        """Return ``(nodes_points, simplices, vdims)`` for :meth:`trimesh`.

        Args:
            data: A UGRID-mesh object (``node_x``/``node_y``/``fan_triangles``) or a point
                ``FeatureCollection``.
            value_column: Node-value column for the FeatureCollection path.

        Returns:
            ``(gv.Points nodes, (n_faces, 3) simplices, vdims list)``.
        """
        import numpy as np

        gv, hv = _require_holoviz()
        crs = gv.util.process_crs(self.crs)
        if all(hasattr(data, attr) for attr in ("node_x", "node_y", "fan_triangles")):
            x = np.asarray(data.node_x)
            y = np.asarray(data.node_y)
            simplices = np.asarray(data.fan_triangles)
            nodes = gv.Points((x, y), crs=crs)
            return nodes, simplices, []
        from matplotlib.tri import (
            Triangulation,
        )  # matplotlib, not a forbidden GIS engine

        gdf = self._display_gdf(data)
        x = gdf.geometry.x.to_numpy()
        y = gdf.geometry.y.to_numpy()
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        simplices = Triangulation(x, y).triangles
        if value_column:
            z = gdf[value_column].to_numpy()[finite]
            nodes = gv.Points((x, y, z), vdims=[value_column], crs=crs)
            return nodes, simplices, [value_column]
        return gv.Points((x, y), crs=crs), simplices, []

    def hexbin(
        self,
        features: Any,
        *,
        gridsize: int = 30,
        aggregator: str = "mean",
        column: Optional[str] = None,
        cmap: str = "viridis",
        **opts: Any,
    ) -> "VectorMixin":
        """Add an equal-area hex-bin density layer (honest no-overplot density, recipe I7).

        Args:
            features: A point ``FeatureCollection``; reprojected through pyramids.
            gridsize: Number of hexagons across — higher is finer.
            aggregator: Per-bin reducer (``"count"``/``"mean"``/``"sum"``/…); ``"count"`` ignores
                ``column``.
            column: Value column aggregated per bin (required for non-``count`` aggregators).
            cmap: Colormap name.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).
        """
        import numpy as np

        gv, hv = _require_holoviz()
        gdf = self._display_gdf(features)
        # Build from explicit display-CRS x/y(/value) arrays, not the geometry GeoDataFrame: GeoViews
        # mis-projects a GeoDataFrame's point geometry at bokeh render time. HoloViews' hex aggregation
        # also needs the reducer as a numpy callable (np.size = count), not a string.
        x = gdf.geometry.x.to_numpy()
        y = gdf.geometry.y.to_numpy()
        reducers = {
            "count": np.size,
            "mean": np.mean,
            "sum": np.sum,
            "min": np.min,
            "max": np.max,
            "std": np.std,
        }
        crs = gv.util.process_crs(self.crs)
        if column:
            reducer = reducers.get(aggregator, np.mean)
            element = gv.HexTiles((x, y, gdf[column].to_numpy()), kdims=["x", "y"], vdims=[column], crs=crs)
        else:  # no value column -> count points per hex (np.size), no value dimension
            reducer = np.size
            element = gv.HexTiles((x, y), kdims=["x", "y"], crs=crs)
        element = self._styled(
            element,
            common={"cmap": cmap, "colorbar": True, **opts},
            bokeh={"gridsize": gridsize, "aggregator": reducer, "tools": ["hover"]},
        )
        return self.add_element(element)

    def kde(
        self,
        features: Any,
        *,
        filled: bool = True,
        cmap: str = "viridis",
        **opts: Any,
    ) -> "VectorMixin":
        """Add a 2-D kernel-density layer of point positions (parity with ``Map.kde``, recipe I7).

        Args:
            features: A point ``FeatureCollection``; reprojected through pyramids.
            filled: Fill the density bands (``True``) or draw contour lines (``False``).
            cmap: Colormap name.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        gdf = self._display_gdf(features)
        x = gdf.geometry.x.to_numpy()
        y = gdf.geometry.y.to_numpy()
        element = hv.Bivariate((x, y))
        element = self._styled(
            element,
            common={"cmap": cmap, **opts},
            bokeh={"filled": filled, "colorbar": True},
        )
        return self.add_element(element)

    def graph(
        self,
        nodes: Any,
        edges: Any,
        *,
        weight: Optional[str] = None,
        bundle: bool = False,
        node_id: str = "id",
        cmap: str = "viridis",
        **opts: Any,
    ) -> "VectorMixin":
        """Add a network / origin-destination flow map (parity-plus for ``Map.sankey``, recipe I9).

        Args:
            nodes: A point ``FeatureCollection``/GeoDataFrame of network nodes; reprojected through
                pyramids. Each node's ``node_id`` column is the index the edges reference.
            edges: An iterable of ``(src_id, dst_id[, weight])`` tuples, or a DataFrame with those
                columns.
            weight: Optional edge-weight column driving line width/colour.
            bundle: Bundle edges (``holoviews.operation.connect_edges``) then datashade them — for
                dense networks. Note ``hammer_bundle`` (the heavy variant) is not used; this is the
                cheap ``connect_edges``.
            node_id: The node-id column name on ``nodes``.
            cmap: Colormap used when ``weight`` is given.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).
        """
        import numpy as np
        import pandas as pd

        gv, hv = _require_holoviz()
        gdf = self._display_gdf(nodes)
        crs = gv.util.process_crs(self.crs)
        ids = (
            gdf[node_id].to_numpy()
            if node_id in getattr(gdf, "columns", [])
            else np.arange(len(gdf))
        )
        gv_nodes = gv.Nodes(
            (gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), ids), crs=crs
        )
        # Normalise edges to a DataFrame with named columns — a 3-tuple (src, dst, weight) list with
        # vdims trips gv.Graph's source/target merge, so build the frame explicitly.
        edge_df = (
            edges if isinstance(edges, pd.DataFrame) else pd.DataFrame(list(edges))
        )
        ncols = edge_df.shape[1]
        names = ["source", "target"] + ([weight] if weight and ncols > 2 else [])
        edge_df = edge_df.iloc[:, : len(names)]
        edge_df.columns = names
        vdims = [weight] if (weight and weight in edge_df.columns) else []
        if (
            weight and not vdims
        ):  # weight requested but the edges carry no weight column — say so
            from loguru import logger

            logger.info(
                f"graph: weight={weight!r} requested but the edges have no weight column "
                "(2-tuple edges) — drawing unweighted; pass (src, dst, weight) tuples to weight them"
            )
        graph = gv.Graph((edge_df, gv_nodes), vdims=vdims, crs=crs)
        if bundle:
            # Datashade the straight edge paths into a density image (the cheap path; the heavy
            # hammer_bundle is intentionally avoided — see the plan's DI.15 note).
            from holoviews.operation.datashader import datashade

            return self.add_element(
                self._styled(datashade(graph.edgepaths), common=opts or None)
            )
        common: dict = dict(opts)
        if weight:
            common.update({"edge_color": weight, "edge_cmap": cmap, "colorbar": True})
        element = self._styled(graph, common=common or None, bokeh={"tools": ["hover"]})
        return self.add_element(element)

    def flow(
        self, nodes: Any, edges: Any, *, weight: Optional[str] = None, **opts: Any
    ) -> "VectorMixin":
        """Spatial-flow alias of :meth:`graph` mirroring ``Map.sankey``'s framing (DI.15).

        Args:
            nodes: A point ``FeatureCollection`` of flow endpoints.
            edges: ``(src_id, dst_id[, weight])`` tuples.
            weight: Optional flow-magnitude column.
            **opts: Forwarded to :meth:`graph`.

        Returns:
            This map (chainable).
        """
        return self.graph(nodes, edges, weight=weight, **opts)
