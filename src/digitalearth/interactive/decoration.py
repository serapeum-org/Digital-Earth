"""DecorationMixin — tiles & cartographic features for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``tiles`` / ``coastlines`` / ``features``, the ``legend``/``colorbar`` toggles (DI.1c) and the
``text`` / ``labels`` annotations (DI.5); the full provider catalog + custom WMTS lands in DI.10.

Tile basemaps (``gv.tile_sources``) and Natural-Earth features (``gv.feature``) are Web-Mercator-only in
Bokeh, so every decoration guards on the default ``crs=3857`` display CRS (``_require_web_mercator``) —
on any other CRS they would silently misalign with the pre-reprojected data layers. Constructing these
elements touches no network; tiles/coastline geometry is fetched by the renderer at display time.
"""

from typing import Any

from digitalearth.interactive.base import _require_holoviz


class DecorationMixin:
    """Decoration builders (DI.1c): tile basemaps, Natural-Earth features, legend/colorbar toggles."""

    def tiles(
        self,
        provider: Any = "CartoLight",
        *,
        level: str = "underlay",
        api_key: Any = None,
        **opts: Any,
    ) -> "DecorationMixin":
        """Add a web-tile basemap beneath the data layers (DI.1c + DI.10 catalog / custom WMTS).

        Args:
            provider: A ``geoviews.tile_sources`` provider name (``"CartoLight"``/``"OSM"``/
                ``"EsriImagery"``/…); a raw XYZ/WMTS URL template (``"https://…/{Z}/{X}/{Y}.png"``);
                or an ``xyzservices.TileProvider``.
            level: ``"underlay"`` (default) keeps tiles behind the data layers; ``"overlay"`` puts
                them on top (rare — e.g. a labels overlay).
            api_key: API key for a keyed provider (e.g. Stadia); a keyed provider used without a key
                raises rather than rendering blank tiles.
            **opts: Extra HoloViews style options applied to the tile element.

        Returns:
            This map (chainable) — the tile layer is inserted *beneath* existing layers.

        Examples:
            - Put a light Carto basemap beneath a raster:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().image(dem).tiles("CartoLight")         # doctest: +SKIP
                >>> type(m.layers[0]).__name__                                  # doctest: +SKIP
                'WMTS'

                ```
            - Use a custom XYZ URL template:
                ```python
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> url = "https://a.tile.example/{Z}/{X}/{Y}.png"              # doctest: +SKIP
                >>> InteractiveMap().tiles(url).layers[0]                       # doctest: +SKIP

                ```

        Raises:
            ValueError: when the display CRS is not Web Mercator (Bokeh tiles are 3857-only), when a
                non-Mercator projection is active, or when a provider name is unknown.
            ImportError: when a keyed provider is requested without an ``api_key``.
        """
        gv, hv = _require_holoviz()
        self._require_web_mercator("tiles")
        if getattr(self, "_projection", None) is not None:
            raise ValueError(
                "tiles() cannot compose with a non-Mercator projection() — Bokeh tiles render in "
                "Web-Mercator only. Drop the projection to use a tile basemap."
            )
        element = self._build_tiles(provider, api_key)
        if opts:
            element = element.opts(**opts)
        element = element.opts(level=level)
        # An explicit tiles() call supersedes any provider passed to the constructor, so render()'s
        # one-shot hook does not also prepend a second basemap (L2).
        self._tiles_provider = None
        if level == "overlay":
            self.add_element(element)
        else:
            self.layers.insert(0, element)
        return self

    def _build_tiles(self, provider: Any, api_key: Any) -> Any:
        """Resolve ``provider`` to a ``gv.WMTS``/``gv.Tiles`` element (name, URL, or xyzservices).

        Args:
            provider: A catalog name, a raw ``{Z}/{X}/{Y}`` URL, or an ``xyzservices.TileProvider``.
            api_key: API key for keyed providers.

        Returns:
            The tile element (a fresh clone for catalog names — the shared instance is never mutated).

        Raises:
            ValueError: for an unknown provider name.
            ImportError: when a keyed provider needs an ``api_key`` that was not supplied.
        """
        gv, hv = _require_holoviz()
        if isinstance(provider, str) and "://" in provider:  # raw XYZ/WMTS URL template
            return gv.WMTS(provider)
        if isinstance(provider, str):
            sources = gv.tile_sources.tile_sources
            if provider not in sources:
                known = ", ".join(sorted(sources))
                raise ValueError(
                    f"unknown tile provider {provider!r} — choose one of: {known}"
                )
            if "stadia" in provider.lower() and api_key is None:
                raise ImportError(
                    f"tile provider {provider!r} needs an api_key (Stadia/Stamen require a key for "
                    "non-local use); pass tiles(provider, api_key=...)"
                )
            return sources[provider].clone()
        return gv.WMTS(
            provider
        )  # an xyzservices.TileProvider (or any gv.WMTS-accepted object)

    def list_tile_providers(self) -> list:
        """Return the sorted catalog of named ``geoviews.tile_sources`` providers.

        Returns:
            The provider names accepted by :meth:`tiles`.
        """
        gv, hv = _require_holoviz()
        return sorted(gv.tile_sources.tile_sources)

    def coastlines(self, resolution: str = "110m", **opts: Any) -> "DecorationMixin":
        """Add the Natural-Earth coastline on top of the data layers.

        Args:
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            **opts: Extra HoloViews style options applied to the feature element.

        Returns:
            This map (chainable).

        Examples:
            - Add a medium-resolution coastline on top of the data:
                ```python
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> m = InteractiveMap().coastlines(resolution="50m")           # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                1

                ```

        Raises:
            ValueError: when the display CRS is not Web Mercator.
        """
        gv, hv = _require_holoviz()
        self._require_web_mercator("coastlines")
        # clone: .opts() would otherwise restyle the shared gv.feature.coastline singleton
        element = gv.feature.coastline.clone().opts(scale=resolution)
        if opts:
            element = element.opts(**opts)
        return self.add_element(element)

    def features(
        self,
        *,
        land: bool = False,
        ocean: bool = False,
        borders: bool = False,
        rivers: bool = False,
        lakes: bool = False,
        resolution: str = "110m",
        **opts: Any,
    ) -> "DecorationMixin":
        """Add Natural-Earth context layers (land/ocean beneath the data, borders/rivers on top).

        Args:
            land: Draw the land polygons (underlay).
            ocean: Draw the ocean polygons (underlay).
            borders: Draw country borders (overlay).
            rivers: Draw river centerlines (overlay).
            lakes: Draw lake polygons (overlay).
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            **opts: Extra HoloViews style options applied to every requested feature element.

        Returns:
            This map (chainable).

        Examples:
            - Underlay land and overlay country borders around a raster:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().image(dem).features(land=True, borders=True)  # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                3

                ```

        Raises:
            ValueError: when the display CRS is not Web Mercator.
        """
        gv, hv = _require_holoviz()
        self._require_web_mercator("features")

        def _styled_feature(feature: Any) -> Any:
            element = feature.clone().opts(
                scale=resolution
            )  # clone: keep the gv singletons pristine
            return element.opts(**opts) if opts else element

        for underlay, requested in (("land", land), ("ocean", ocean)):
            if requested:
                self.layers.insert(0, _styled_feature(getattr(gv.feature, underlay)))
        for overlay, requested in (
            ("borders", borders),
            ("rivers", rivers),
            ("lakes", lakes),
        ):
            if requested:
                self.add_element(_styled_feature(getattr(gv.feature, overlay)))
        return self

    def _to_display_xy(self, lon: Any, lat: Any, crs: Any) -> tuple:
        """Reproject ``(lon, lat)`` from ``crs`` to the display CRS via pyramids.

        Uses ``pyramids.feature.geometry.reproject_coordinates`` (pyramids owns the pyproj call —
        this package never imports pyproj/cartopy). A no-op when ``crs`` already equals the display
        CRS.

        Args:
            lon: X / longitude in ``crs`` (scalar or 1-D).
            lat: Y / latitude in ``crs`` (scalar or 1-D).
            crs: The CRS of ``lon``/``lat`` (EPSG int / string).

        Returns:
            ``(x, y)`` lists in the display CRS.
        """
        import numpy as np

        xs = np.atleast_1d(lon).astype(float).tolist()
        ys = np.atleast_1d(lat).astype(float).tolist()
        if crs == self.crs:
            return xs, ys
        from pyramids.feature.geometry import reproject_coordinates

        return reproject_coordinates(xs, ys, from_crs=crs, to_crs=self.crs)

    def text(
        self, lon: Any, lat: Any, s: str, *, crs: Any = 4326, **opts: Any
    ) -> "DecorationMixin":
        """Add a single text annotation at ``(lon, lat)`` (reprojected to the display CRS).

        Args:
            lon: X / longitude of the label anchor, in ``crs``.
            lat: Y / latitude of the label anchor, in ``crs``.
            s: The text to draw.
            crs: CRS of ``lon``/``lat`` (default EPSG:4326); reprojected to the display CRS via
                pyramids.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        (x,), (y,) = self._to_display_xy(lon, lat, crs)
        element = gv.Text(x, y, s, crs=gv.util.process_crs(self.crs))
        if opts:
            element = element.opts(**opts)
        return self.add_element(element)

    def labels(
        self, features: Any, column: str, *, crs: Any = 4326, **opts: Any
    ) -> "DecorationMixin":
        """Add per-feature text labels from a point ``FeatureCollection`` column.

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries; reprojected to the
                display CRS through pyramids when needed.
            column: The attribute column whose values are drawn as labels.
            crs: Unused when ``features`` carries its own CRS (kept for signature symmetry with
                :meth:`text`); reprojection goes through ``FeatureCollection.to_crs``.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Raises:
            KeyError: when ``column`` is not a column of ``features``.
        """
        gv, hv = _require_holoviz()
        if column not in getattr(features, "columns", [column]):
            raise KeyError(
                f"labels column {column!r} not found in the feature attributes"
            )
        gdf = self._display_gdf(features)
        element = gv.Labels(
            gdf,
            vdims=[column],
            crs=gv.util.process_crs(self.crs),
            datatype=[
                "geodataframe",
                "multitabular",
                "dictionary",
                "dataframe",
                "array",
            ],
        )
        if opts:
            element = element.opts(**opts)
        return self.add_element(element)

    def colorbar(self, show: bool = True) -> "DecorationMixin":
        """Toggle the colorbar on the most recently added layer.

        Args:
            show: Whether the last layer draws a colorbar.

        Returns:
            This map (chainable).

        Examples:
            - Drop the colorbar from the last raster layer:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> InteractiveMap().image(dem).colorbar(False).save("m.html")  # doctest: +SKIP
                'm.html'

                ```

        Raises:
            ValueError: when no layer has been added yet.
        """
        if not self.layers:
            raise ValueError(
                "colorbar() needs at least one layer — add a builder call first"
            )
        self.layers[-1] = self.layers[-1].opts(colorbar=show)
        return self

    def legend(self, show: bool = True) -> "DecorationMixin":
        """Toggle the Bokeh legend on the most recently added layer.

        Args:
            show: Whether the last layer contributes to the legend.

        Returns:
            This map (chainable).

        Examples:
            - Hide the legend of a contour layer:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> InteractiveMap().contours(dem).legend(False).save("m.html")  # doctest: +SKIP
                'm.html'

                ```

        Raises:
            ValueError: when no layer has been added yet.
        """
        gv, hv = _require_holoviz()
        if not self.layers:
            raise ValueError(
                "legend() needs at least one layer — add a builder call first"
            )
        self.layers[-1] = self.layers[-1].opts(show_legend=show, backend="bokeh")
        return self
