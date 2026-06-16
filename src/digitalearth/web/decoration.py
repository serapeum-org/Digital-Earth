"""DecorationMixin — web-tier basemaps/tiles, popups/tooltips, and map controls.

Adds raster XYZ basemaps beneath the data (``basemap``/``tiles``, DW.1a), hover/click attribute readouts
(``tooltip``/``popup``, DW.2), MapLibre UI controls (``navigation``/``scale_bar``/``fullscreen``/``controls``,
ED.13) and a draw-based ``measure`` tool (ED.10). Each builder registers a callable on the map's layer
registry (basemaps as an underlay; popups/tooltips/controls as post-render ``add_*`` calls).

Out of scope here (deferred): ``pmtiles`` (needs the optional ``pmtiles`` reader, intentionally not in the
``[web]`` extra); an on-map ``legend`` control and a **minimap** (py-maplibregl has neither built in — both
need a custom HTML/JS control). ``choropleth`` still exposes its class breaks via ``WebMap.last_breaks`` so a
caller can build a legend out-of-band; the ``measure`` tool exposes the drawn geometry for pyramids to compute
geodesic distance/area (the GIS part).
"""

from typing import Any, List, Optional

from digitalearth.web.base import _require_layer_api, _require_maplibre

#: Named raster XYZ basemaps → ``(url_template, attribution)``. All are token-free public tile services.
_BASEMAP_PROVIDERS = {
    "cartodark": (
        "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors © CARTO",
    ),
    "cartolight": (
        "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors © CARTO",
    ),
    "cartovoyager": (
        "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors © CARTO",
    ),
    "osm": (
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors",
    ),
}


class DecorationMixin:
    """Basemap/tiles and popup/tooltip builders for :class:`~digitalearth.web.map.WebMap`."""

    def tiles(
        self,
        url: str,
        *,
        attribution: str = "",
        tile_size: int = 256,
        opacity: float = 1.0,
    ) -> "DecorationMixin":
        """Add a raster XYZ/WMTS tile layer **beneath** the data (recipe W1).

        Args:
            url: An XYZ tile URL template containing ``{z}/{x}/{y}`` (already Web-Mercator tiles).
            attribution: Attribution text shown in the map's attribution control.
            tile_size: Tile edge length in pixels (256 for standard XYZ; 512 for some retina services).
            opacity: Raster opacity in ``[0, 1]``.

        Returns:
            This map (chainable); the basemap is registered as an underlay so data drawn before or after
            it still renders on top.
        """
        Layer, LayerType = _require_layer_api()
        src_id, layer_id = self._uid("tiles-src"), self._uid("tiles")
        source = {
            "type": "raster",
            "tiles": [url],
            "tileSize": int(tile_size),
        }
        if attribution:
            source["attribution"] = attribution
        layer = Layer(
            id=layer_id,
            type=LayerType.RASTER,
            source=src_id,
            paint={"raster-opacity": float(opacity)},
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, source)
            widget.add_layer(layer)

        return self.add_underlay(apply)

    def basemap(self, provider: str = "CartoDark", *, opacity: float = 1.0) -> "DecorationMixin":
        """Add a named raster basemap beneath the data (recipe W1).

        Args:
            provider: A basemap name — ``"CartoDark"``, ``"CartoLight"``, ``"CartoVoyager"`` or ``"OSM"``
                (case-insensitive). All are token-free public tile services.
            opacity: Basemap opacity in ``[0, 1]``.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when ``provider`` is not a known basemap name.
        """
        key = provider.replace(" ", "").lower()
        if key not in _BASEMAP_PROVIDERS:
            raise ValueError(
                f"unknown basemap provider {provider!r}; choose one of "
                f"{sorted(p.capitalize() for p in _BASEMAP_PROVIDERS)} or pass a tile URL to tiles()"
            )
        url, attribution = _BASEMAP_PROVIDERS[key]
        return self.tiles(url, attribution=attribution, opacity=opacity)

    def navigation(
        self,
        *,
        position: str = "top-right",
        show_compass: bool = True,
        show_zoom: bool = True,
        visualize_pitch: bool = False,
    ) -> "DecorationMixin":
        """Add MapLibre navigation controls — zoom buttons and a compass (ED.13).

        Args:
            position: Corner placement (``"top-right"``/``"top-left"``/``"bottom-right"``/``"bottom-left"``).
            show_compass: Show the compass / bearing-reset button.
            show_zoom: Show the zoom in/out buttons.
            visualize_pitch: Show the map pitch on the compass.

        Returns:
            This map (chainable).
        """
        _require_maplibre()
        from maplibre.controls import NavigationControl

        control = NavigationControl(
            show_compass=show_compass, show_zoom=show_zoom, visualize_pitch=visualize_pitch
        )

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def scale_bar(
        self, *, position: str = "bottom-left", unit: str = "metric", max_width: int = 100
    ) -> "DecorationMixin":
        """Add a MapLibre scale bar (ED.13).

        Args:
            position: Corner placement for the scale bar.
            unit: ``"metric"``, ``"imperial"`` or ``"nautical"``.
            max_width: Maximum scale-bar width in pixels.

        Returns:
            This map (chainable).
        """
        _require_maplibre()
        from maplibre.controls import ScaleControl

        control = ScaleControl(unit=unit, max_width=int(max_width))

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def fullscreen(self, *, position: str = "top-right") -> "DecorationMixin":
        """Add a MapLibre fullscreen toggle control (ED.13).

        Args:
            position: Corner placement for the fullscreen button.

        Returns:
            This map (chainable).
        """
        _require_maplibre()
        from maplibre.controls import FullscreenControl

        control = FullscreenControl()

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def controls(
        self, *, navigation: bool = True, scale: bool = True, fullscreen: bool = False
    ) -> "DecorationMixin":
        """Add the common navigation / scale / fullscreen controls in one call (ED.13).

        A convenience over :meth:`navigation`, :meth:`scale_bar` and :meth:`fullscreen`. Note: py-maplibregl
        has no built-in **minimap** control, so a minimap is not offered here (it would need a custom JS
        control — tracked as a follow-up).

        Args:
            navigation: Add zoom + compass controls.
            scale: Add a scale bar.
            fullscreen: Add a fullscreen toggle.

        Returns:
            This map (chainable).
        """
        if navigation:
            self.navigation()
        if scale:
            self.scale_bar()
        if fullscreen:
            self.fullscreen()
        return self

    def measure(
        self, *, distance: bool = True, area: bool = True, position: str = "top-left"
    ) -> "DecorationMixin":
        """Add a draw-based measure tool — draw a line (distance) or polygon (area) to measure (ED.10).

        Note this adds a **drawing** control, not a live on-map readout: it does not display the distance/area
        number on the map (that GIS computation is left to pyramids, below). It enables MapLibre's draw control
        scoped to line and/or polygon geometries, so the user draws the shape to measure. The drawn GeoJSON is
        available on the rendered widget
        (``draw_feature_collection_all`` and the ``draw_features_created``/``…_updated`` events). Computing the
        numeric distance/area from that geometry is a **GIS** operation — do it in pyramids (geodesic length /
        area), keeping this tier to the (visualization) drawing control.

        Args:
            distance: Offer the line tool (measure distance along a path).
            area: Offer the polygon tool (measure enclosed area).
            position: Corner placement for the draw toolbar.

        Returns:
            This map (chainable).

        Raises:
            ValueError: if neither ``distance`` nor ``area`` is enabled.
        """
        _require_maplibre()
        from maplibre.plugins import MapboxDrawControls, MapboxDrawOptions

        if not (distance or area):
            raise ValueError("measure() needs distance and/or area enabled")
        options = MapboxDrawOptions(
            display_controls_default=False,
            controls=MapboxDrawControls(line_string=distance, polygon=area, trash=True),
        )

        def apply(widget: Any) -> None:
            widget.add_mapbox_draw(options, position)

        return self.add_layer(layer=apply)

    @staticmethod
    def _attribute_template(fields: Optional[List[str]]) -> dict:
        """Build the ``add_popup``/``add_tooltip`` kwargs for ``fields``.

        Args:
            fields: Attribute column names to show. ``None``/empty shows the feature's raw properties; a
                single field uses ``prop=``; several use an HTML ``template=`` with ``{field}`` placeholders.

        Returns:
            A kwargs dict for ``add_popup``/``add_tooltip`` (``{}``, ``{"prop": ...}`` or ``{"template": ...}``).
        """
        if not fields:
            return {}
        if len(fields) == 1:
            return {"prop": fields[0]}
        template = "".join(f"<b>{f}</b>: {{{f}}}<br>" for f in fields)
        return {"template": template}

    def popup(
        self, fields: Optional[List[str]] = None, *, layer: Optional[str] = None
    ) -> "DecorationMixin":
        """Show an attribute popup on **click** for a layer's features (recipe W2).

        Args:
            fields: Attribute columns to display (one → that property; several → an HTML table). ``None``
                shows the feature's raw properties.
            layer: Target layer id; defaults to the most recently added data layer.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when there is no layer to attach to (no ``layer`` and nothing drawn yet).
        """
        _require_layer_api()
        layer_id = layer or self._last_layer_id
        if layer_id is None:
            raise ValueError("popup() needs a layer — draw a data layer first or pass layer=...")
        kwargs = self._attribute_template(fields)

        def apply(widget: Any) -> None:
            widget.add_popup(layer_id, **kwargs)

        return self.add_layer(apply)

    def tooltip(
        self, fields: Optional[List[str]] = None, *, layer: Optional[str] = None
    ) -> "DecorationMixin":
        """Show an attribute tooltip on **hover** for a layer's features (recipe W2).

        Args:
            fields: Attribute columns to display (one → that property; several → an HTML table). ``None``
                shows the feature's raw properties.
            layer: Target layer id; defaults to the most recently added data layer.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when there is no layer to attach to (no ``layer`` and nothing drawn yet).
        """
        _require_layer_api()
        layer_id = layer or self._last_layer_id
        if layer_id is None:
            raise ValueError("tooltip() needs a layer — draw a data layer first or pass layer=...")
        kwargs = self._attribute_template(fields)

        def apply(widget: Any) -> None:
            widget.add_tooltip(layer_id, **kwargs)

        return self.add_layer(apply)
