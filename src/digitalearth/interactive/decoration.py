"""DecorationMixin — tiles & cartographic features for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``tiles`` / ``coastlines`` / ``features`` and the ``legend``/``colorbar`` toggles (DI.1c); the full
provider catalog + custom WMTS (DI.10) and ``text``/``labels`` (DI.5) land later.

Tile basemaps (``gv.tile_sources``) and Natural-Earth features (``gv.feature``) are Web-Mercator-only in
Bokeh, so every decoration guards on the default ``crs=3857`` display CRS (``_require_web_mercator``) —
on any other CRS they would silently misalign with the pre-reprojected data layers. Constructing these
elements touches no network; tiles/coastline geometry is fetched by the renderer at display time.
"""

from typing import Any

from digitalearth.interactive.base import _require_holoviz


class DecorationMixin:
    """Decoration builders (DI.1c): tile basemaps, Natural-Earth features, legend/colorbar toggles."""

    def tiles(self, provider: str = "CartoLight", **opts: Any) -> "DecorationMixin":
        """Add a web-tile basemap beneath the data layers.

        Args:
            provider: A ``geoviews.tile_sources`` provider name (e.g. ``"CartoLight"``, ``"OSM"``,
                ``"EsriImagery"``).
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

        Raises:
            ValueError: when the display CRS is not Web Mercator (Bokeh tiles are 3857-only), or
                when ``provider`` is not a known tile source.
        """
        gv, hv = _require_holoviz()
        self._require_web_mercator("tiles")
        sources = gv.tile_sources.tile_sources
        if provider not in sources:
            known = ", ".join(sorted(sources))
            raise ValueError(
                f"unknown tile provider {provider!r} — choose one of: {known}"
            )
        element = sources[
            provider
        ].clone()  # clone: never mutate the shared module-level instance
        if opts:
            element = element.opts(**opts)
        # An explicit tiles() call supersedes any provider passed to the constructor, so render()'s
        # one-shot hook does not also prepend a second basemap (L2).
        self._tiles_provider = None
        self.layers.insert(0, element)
        return self

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
