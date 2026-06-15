"""DecorationMixin — web-tier basemaps/tiles (DW.1a) and popups/tooltips (DW.2).

Adds raster XYZ basemaps beneath the data (``basemap``/``tiles``) and hover/click attribute readouts
(``tooltip``/``popup``) on top of a data layer. Each builder registers a callable on the map's layer
registry (basemaps as an underlay; popups/tooltips as post-layer ``add_popup``/``add_tooltip`` calls).

Out of scope here (deferred): ``pmtiles`` (needs the optional ``pmtiles`` reader, intentionally not in the
``[web]`` extra) and an on-map ``legend`` control (py-maplibregl has no built-in legend widget — it needs a
custom HTML control, a DW.6 export concern). ``choropleth`` still exposes its class breaks via
``WebMap.last_breaks`` so a caller can build a legend out-of-band.
"""

from typing import Any, List, Optional

from digitalearth.web.base import _require_layer_api

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
