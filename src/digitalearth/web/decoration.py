"""DecorationMixin — web-tier basemaps/tiles, popups/tooltips, and map controls.

Adds raster XYZ basemaps beneath the data (``basemap``/``tiles``, DW.1a), hover/click attribute readouts
(``tooltip``/``popup``, DW.2), MapLibre UI controls (``navigation``/``scale_bar``/``fullscreen``/``controls``,
ED.13) and a draw-based ``measure`` tool (ED.10). Each builder registers a callable on the map's layer
registry (basemaps as an underlay; popups/tooltips/controls as post-render ``add_*`` calls).

``legend`` draws the key for the most recent classified layer, built from the colours that classification
actually rendered (``WebMap.last_legend``); ``WebMap.last_breaks`` remains for a caller who would rather build
their own out of band.

Out of scope here (deferred): ``pmtiles`` (needs the optional ``pmtiles`` reader, intentionally not in the
``[web]`` extra); a **minimap** (py-maplibregl ships no such control, and it would need a custom HTML/JS one).
The ``measure`` tool exposes the drawn geometry for pyramids to compute geodesic distance/area (the GIS
part).
"""

from typing import TYPE_CHECKING, Any, List, Optional, Self

from digitalearth.base.basemaps import (
    KEYED_BASEMAP_NAMES,
    get_keyed_basemap,
    is_keyed_basemap,
)
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

#: Canonical (correctly-cased) display names for the basemap keys, used in the not-found error message.
_BASEMAP_DISPLAY_NAMES = {
    "cartodark": "CartoDark",
    "cartolight": "CartoLight",
    "cartovoyager": "CartoVoyager",
    "osm": "OSM",
}

#: The four legal MapLibre control corners.
_CONTROL_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


#: The legend's own styling, kept with the markup that uses it rather than left to the host page.
_LEGEND_CSS = (
    "background: rgba(255, 255, 255, 0.92); color: #222; padding: 8px 10px; border-radius: 4px; "
    "font: 12px/1.4 system-ui, sans-serif; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); max-width: 220px;"
)


def _swatch(color: str) -> str:
    """Return the markup for one colour chip.

    Args:
        color: A CSS colour, as rendered on the map.

    Returns:
        An inline-block span, sized to line up with a row of text.
    """
    return (
        f'<span style="display:inline-block;width:14px;height:14px;margin-right:6px;'
        f'vertical-align:-2px;background:{color};border:1px solid rgba(0,0,0,.25)"></span>'
    )


def _format_number(value: Any) -> str:
    """Render a class edge compactly, without the float noise a raw repr would show.

    Args:
        value: A break edge, a category, or a ramp stop.

    Returns:
        A short string: integers keep no decimal point, floats get three significant decimals, and a
        non-numeric category is passed through as text.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _legend_rows(kind: str, values: list, colors: list, labels: Optional[list]) -> str:
    """Build the body of a legend for one classification.

    Args:
        kind: ``"categorical"``, ``"graduated"`` or ``"continuous"``.
        values: The categories, class edges, or ramp stops.
        colors: The colours those values were drawn with.
        labels: Explicit row labels overriding the derived ones, or ``None``.

    Returns:
        The rows as HTML — swatch-and-label lines for a discrete classification, a gradient bar with end
        labels for a continuous one.
    """
    if kind == "continuous":
        ramp = ", ".join(colors)
        return (
            f'<div style="height:10px;border-radius:2px;background:linear-gradient(to right,{ramp})"></div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:2px">'
            f"<span>{_format_number(values[0])}</span>"
            f"<span>{_format_number(values[-1])}</span></div>"
        )
    if kind == "graduated":
        derived = [
            f"{_format_number(values[i])} – {_format_number(values[i + 1])}"
            for i in range(len(values) - 1)
        ]
    else:
        derived = [_format_number(v) for v in values]
    text = labels if labels is not None else derived
    return "".join(
        f'<div style="white-space:nowrap">{_swatch(color)}{label}</div>'
        for color, label in zip(colors, text)
    )


def _check_position(position: str) -> None:
    """Validate a control corner, raising ``ValueError`` for anything but the four legal MapLibre corners.

    Args:
        position: The requested corner placement.

    Raises:
        ValueError: when ``position`` is not one of ``top-left``/``top-right``/``bottom-left``/``bottom-right``.
    """
    if position not in _CONTROL_POSITIONS:
        raise ValueError(
            f"unknown control position {position!r}; choose one of {list(_CONTROL_POSITIONS)}"
        )


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class DecorationMixin(_MixinBase):
    """Basemap/tiles and popup/tooltip builders for :class:`~digitalearth.web.map.WebMap`.

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

    def tiles(
        self,
        url: str,
        *,
        attribution: str = "",
        tile_size: int = 256,
        opacity: float = 1.0,
        max_zoom: Optional[int] = None,
    ) -> Self:
        """Add a raster XYZ/WMTS tile layer **beneath** the data (recipe W1).

        Args:
            url: An XYZ tile URL template containing ``{z}/{x}/{y}`` (already Web-Mercator tiles).
            attribution: Attribution text shown in the map's attribution control.
            tile_size: Tile edge length in pixels (256 for standard XYZ; 512 for some retina services).
            opacity: Raster opacity in ``[0, 1]``.
            max_zoom: The deepest zoom the service serves. Past it MapLibre over-zooms the last real
                tiles instead of requesting levels that do not exist; ``None`` leaves the source
                unbounded.

        Returns:
            The same map instance, so builder calls chain; the basemap is registered as an underlay so
            data drawn before or after it still renders on top.
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
        if max_zoom is not None:
            source["maxzoom"] = int(max_zoom)
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

    def basemap(
        self,
        provider: str = "CartoDark",
        *,
        opacity: float = 1.0,
        api_key: Optional[str] = None,
        preset: Optional[dict] = None,
    ) -> Self:
        """Add a named raster basemap beneath the data (recipe W1).

        A keyed preset's coverage is **not** checked here, unlike the static tier. A web map is pannable
        and zoomable, so it has no one extent to check against: the initial ``center``/``zoom`` is where
        the viewer starts, not where they stay, and refusing a NICFI basemap because the first view sits
        outside the tropics would block a map the viewer can simply pan into. The static tier renders one
        fixed extent, where an out-of-coverage basemap is a dead end rather than a scroll away, which is
        why the guard lives there.

        Args:
            provider: A token-free basemap name — ``"CartoDark"``, ``"CartoLight"``, ``"CartoVoyager"`` or
                ``"OSM"`` (case-insensitive) — or a **keyed** preset name such as ``"Planet.NICFI"`` (see
                :mod:`digitalearth.base.basemaps`), whose credential is read from the environment.
            opacity: Basemap opacity in ``[0, 1]``.
            api_key: Credential for a keyed preset; ``None`` reads the preset's environment variable.
            preset: The keyed preset's own keywords, as a dict (for NICFI: ``date``, ``flavour``,
                ``mosaic``). A dict rather than loose keywords so that all three tiers take a preset the
                same way, and so a mistyped style argument is an unexpected keyword rather than something
                ``**preset`` silently swallows.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``provider`` is neither a known basemap name nor a keyed preset, when a
                ``preset`` or an ``api_key`` is passed to a provider that takes neither, or when a keyed
                preset's credential is unavailable.
            TypeError: for any other keyword, which Python reports as the unexpected argument it is.

        Examples:
            - A keyed preset resolves its own tile URL and attribution:
                ```python
                >>> from digitalearth.web import WebMap                 # doctest: +SKIP
                >>> WebMap().basemap("Planet.NICFI", preset={"date": "2024-01"})   # doctest: +SKIP

                ```

        See Also:
            digitalearth.base.basemaps: the keyed-preset definitions this resolves.
        """
        if is_keyed_basemap(provider):
            keyed = get_keyed_basemap(provider, **(preset or {}))
            return self.tiles(
                keyed.tile_url(api_key),
                attribution=keyed.attribution,
                opacity=opacity,
                max_zoom=keyed.max_zoom,
            )
        if api_key is not None:
            # Dropping it silently would leave a caller believing they had authenticated.
            raise ValueError(
                f"basemap({provider!r}) takes no api_key; it is a token-free source. Credentials apply "
                f"only to a keyed preset such as 'Planet.NICFI'"
            )
        if preset:
            raise ValueError(
                f"basemap({provider!r}) takes no preset keywords; {sorted(preset)} apply only to a keyed "
                f"preset such as 'Planet.NICFI'"
            )
        key = provider.replace(" ", "").lower()
        if key not in _BASEMAP_PROVIDERS:
            raise ValueError(
                f"unknown basemap provider {provider!r}; choose one of "
                f"{sorted(_BASEMAP_DISPLAY_NAMES.values())}, a keyed preset "
                f"({', '.join(sorted(KEYED_BASEMAP_NAMES.values()))}), or pass a tile URL to tiles()"
            )
        url, attribution = _BASEMAP_PROVIDERS[key]
        return self.tiles(url, attribution=attribution, opacity=opacity)

    def legend(
        self,
        *,
        title: Optional[str] = None,
        position: str = "bottom-right",
        labels: Optional[list] = None,
    ) -> Self:
        """Add a key for the most recent classified layer (recipe W2).

        A thematic map is unreadable without one, and until this the tier drew the classes and left the
        caller to build a key out of band from ``last_breaks``. The classification records what it actually
        coloured with (:attr:`~digitalearth.web.base.WebMapBase.last_legend`), so the key shows the rendered
        colours rather than a second guess at them.

        The three shapes a classification can take are all handled: a **categorical** one gets a swatch per
        category, a **graduated** one a swatch per class with its range, and a **continuous** ramp a
        gradient bar with its end values.

        Args:
            title: Heading above the key. ``None`` uses the classified column's name.
            position: One of the four MapLibre corners.
            labels: Explicit row labels, replacing the derived ones — for units, or for renaming
                categories. Ignored for a continuous ramp, which has no rows.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners, or when no
                classified layer has been added — there is nothing to build a key from, and an empty box
                would be worse than an error.

        Examples:
            - A choropleth and its key:
                ```python
                >>> from digitalearth.web import WebMap                        # doctest: +SKIP
                >>> (                                                          # doctest: +SKIP
                ...     WebMap().basemap().choropleth(gdf, column="pop").legend()
                ... )

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.choropleth: the builder whose classes this describes.
        """
        _require_maplibre()
        _check_position(position)
        spec = self.last_legend
        if not spec:
            raise ValueError(
                "legend() has nothing to describe: no classified layer has been added yet. Add a "
                "choropleth (or any builder given column=...) first."
            )
        from maplibre.controls import InfoBoxControl

        heading = title if title is not None else spec.get("column") or ""
        head = (
            f'<div style="font-weight:600;margin-bottom:4px">{heading}</div>'
            if heading
            else ""
        )
        rows = _legend_rows(
            spec["kind"], list(spec["values"]), list(spec["colors"]), labels
        )
        control = InfoBoxControl(
            content=f"<div>{head}{rows}</div>",
            css_text=_LEGEND_CSS,
            position=position,
        )

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def layer_control(
        self,
        *,
        position: str = "top-right",
        layer_ids: Optional[list] = None,
        theme: str = "default",
    ) -> Self:
        """Add a switcher so a viewer can turn the data layers on and off.

        A map with a basemap, a choropleth and a point overlay had no way to look underneath — which is the
        single most common thing anyone does with a web map. The switch lists the data layers only:
        basemaps are the ground, not something a viewer toggles.

        Args:
            position: One of the four MapLibre corners.
            layer_ids: The layers to offer, defaulting to every data layer added so far
                (:attr:`~digitalearth.web.base.WebMapBase.layer_ids`). Pass a subset to hide the rest from
                the switch without hiding them from the map.
            theme: ``"default"`` or ``"simple"`` — py-maplibregl's two switcher styles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners, when no data layer
                has been added yet, or when an id was given that is not on this map.

        Examples:
            - Two layers and a switch between them:
                ```python
                >>> from digitalearth.web import WebMap                            # doctest: +SKIP
                >>> (                                                              # doctest: +SKIP
                ...     WebMap()
                ...     .basemap()
                ...     .choropleth(gdf, column="pop", name="Population")
                ...     .layer_control()
                ... )

                ```

        See Also:
            digitalearth.web.base.WebMapBase.layer_ids: the ids this offers by default.
        """
        _require_maplibre()
        _check_position(position)
        available = self.layer_ids
        if not available:
            raise ValueError(
                "layer_control() has nothing to switch: no data layer has been added yet. A basemap is "
                "the ground rather than a layer a viewer toggles."
            )
        wanted = list(layer_ids) if layer_ids is not None else available
        unknown = [layer_id for layer_id in wanted if layer_id not in available]
        if unknown:
            raise ValueError(
                f"layer_control() was given {unknown}, which are not on this map; its layers are "
                f"{available}"
            )
        from maplibre.controls import LayerSwitcherControl

        control = LayerSwitcherControl(layer_ids=wanted, theme=theme)

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def text(
        self,
        lon: float,
        lat: float,
        string: str,
        *,
        size: float = 14.0,
        color: str = "#ffffff",
        halo_color: str = "#000000",
        halo_width: float = 1.0,
        name: Optional[str] = None,
    ) -> Self:
        """Place a single line of text at a coordinate.

        The "label this spot" case, which needed a whole GeoDataFrame before —
        :meth:`~digitalearth.web.vector.VectorMixin.labels` is the data-driven counterpart.

        Args:
            lon: Longitude in the display CRS' lon/lat.
            lat: Latitude.
            string: The text to draw.
            size: Text size in pixels.
            color: Text colour.
            halo_color: Colour of the outline behind the glyphs, which keeps it legible over imagery.
            halo_width: Halo width in pixels; ``0`` disables it.
            name: What a layer switcher calls this annotation; ``None`` uses its generated id.

        Returns:
            The same map instance, so builder calls chain.

        Examples:
            - Mark a place:
                ```python
                >>> from digitalearth.web import WebMap                     # doctest: +SKIP
                >>> WebMap().basemap().text(4.9, 52.4, "Amsterdam")         # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.labels: label many features from a column.
        """
        Layer, LayerType = _require_layer_api()
        src_id, layer_id = self._uid("text-src"), self._uid("text")
        source = {
            "type": "geojson",
            "data": {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {"text": string},
            },
        }
        layer = Layer(
            id=layer_id,
            type=LayerType.SYMBOL,
            source=src_id,
            layout={
                "text-field": ["get", "text"],
                "text-size": float(size),
                "text-allow-overlap": True,
            },
            paint={
                "text-color": color,
                "text-halo-color": halo_color,
                "text-halo-width": float(halo_width),
            },
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, source)
            widget.add_layer(layer)

        apply._digitalearth_layer_id = layer_id  # type: ignore[attr-defined]
        self._note_bounds((float(lon), float(lat), float(lon), float(lat)))
        self._index_layer(layer_id, name)
        return self.add_layer(layer=apply)

    def title(
        self,
        heading: str,
        *,
        position: str = "top-left",
        subtitle: Optional[str] = None,
    ) -> Self:
        """Put a title on the map itself, so a saved page carries its own heading.

        An exported page travels alone: whatever context the notebook around it had is gone. The title is
        a control rather than a layer, so it sits above the map and moves with the corner it is pinned to.

        Args:
            heading: The title text.
            position: One of the four MapLibre corners.
            subtitle: A smaller second line — a date, a source, a unit.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners.

        Examples:
            - Title a saved map:
                ```python
                >>> from digitalearth.web import WebMap                            # doctest: +SKIP
                >>> WebMap().basemap().title("Population, 2024")                   # doctest: +SKIP

                ```
        """
        _require_maplibre()
        _check_position(position)
        from maplibre.controls import InfoBoxControl

        body = f'<div style="font-weight:600;font-size:15px">{heading}</div>'
        if subtitle:
            body += f'<div style="opacity:.75;margin-top:2px">{subtitle}</div>'
        control = InfoBoxControl(
            content=f"<div>{body}</div>", css_text=_LEGEND_CSS, position=position
        )

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def navigation(
        self,
        *,
        position: str = "top-right",
        show_compass: bool = True,
        show_zoom: bool = True,
        visualize_pitch: bool = False,
    ) -> Self:
        """Add MapLibre navigation controls — zoom buttons and a compass (ED.13).

        Args:
            position: Corner placement (``"top-right"``/``"top-left"``/``"bottom-right"``/``"bottom-left"``).
            show_compass: Show the compass / bearing-reset button.
            show_zoom: Show the zoom in/out buttons.
            visualize_pitch: Show the map pitch on the compass.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners.
        """
        _require_maplibre()
        _check_position(position)
        from maplibre.controls import NavigationControl

        control = NavigationControl(
            show_compass=show_compass,
            show_zoom=show_zoom,
            visualize_pitch=visualize_pitch,
        )

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def scale_bar(
        self,
        *,
        position: str = "bottom-left",
        unit: str = "metric",
        max_width: int = 100,
    ) -> Self:
        """Add a MapLibre scale bar (ED.13).

        Args:
            position: Corner placement for the scale bar.
            unit: ``"metric"``, ``"imperial"`` or ``"nautical"``.
            max_width: Maximum scale-bar width in pixels.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners.
        """
        _require_maplibre()
        _check_position(position)
        from maplibre.controls import ScaleControl

        control = ScaleControl(unit=unit, max_width=int(max_width))

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def fullscreen(self, *, position: str = "top-right") -> Self:
        """Add a MapLibre fullscreen toggle control (ED.13).

        Args:
            position: Corner placement for the fullscreen button.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners.
        """
        _require_maplibre()
        _check_position(position)
        from maplibre.controls import FullscreenControl

        control = FullscreenControl()

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        return self.add_layer(layer=apply)

    def controls(
        self, *, navigation: bool = True, scale: bool = True, fullscreen: bool = False
    ) -> Self:
        """Add the common navigation / scale / fullscreen controls in one call (ED.13).

        A convenience over :meth:`navigation`, :meth:`scale_bar` and :meth:`fullscreen`. Note: py-maplibregl
        has no built-in **minimap** control, so a minimap is not offered here (it would need a custom JS
        control — tracked as a follow-up).

        Args:
            navigation: Add zoom + compass controls.
            scale: Add a scale bar.
            fullscreen: Add a fullscreen toggle.

        Returns:
            The same map instance, so builder calls chain.
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
    ) -> Self:
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
            The same map instance, so builder calls chain.

        Raises:
            ValueError: if neither ``distance`` nor ``area`` is enabled, or if ``position`` is not one of the
                four legal MapLibre corners.
        """
        _require_maplibre()
        if not (distance or area):
            raise ValueError("measure() needs distance and/or area enabled")
        _check_position(position)
        from maplibre.plugins import MapboxDrawControls, MapboxDrawOptions

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
    ) -> Self:
        """Show an attribute popup on **click** for a layer's features (recipe W2).

        Args:
            fields: Attribute columns to display (one → that property; several → an HTML table). ``None``
                shows the feature's raw properties.
            layer: Target layer id; defaults to the most recently added data layer.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when there is no layer to attach to (no ``layer`` and nothing drawn yet).
        """
        _require_layer_api()
        layer_id = layer or self._last_layer_id
        if layer_id is None:
            raise ValueError(
                "popup() needs a layer — draw a data layer first or pass layer=..."
            )
        kwargs = self._attribute_template(fields)

        def apply(widget: Any) -> None:
            widget.add_popup(layer_id, **kwargs)

        return self.add_layer(apply)

    def tooltip(
        self, fields: Optional[List[str]] = None, *, layer: Optional[str] = None
    ) -> Self:
        """Show an attribute tooltip on **hover** for a layer's features (recipe W2).

        Args:
            fields: Attribute columns to display (one → that property; several → an HTML table). ``None``
                shows the feature's raw properties.
            layer: Target layer id; defaults to the most recently added data layer.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when there is no layer to attach to (no ``layer`` and nothing drawn yet).
        """
        _require_layer_api()
        layer_id = layer or self._last_layer_id
        if layer_id is None:
            raise ValueError(
                "tooltip() needs a layer — draw a data layer first or pass layer=..."
            )
        kwargs = self._attribute_template(fields)

        def apply(widget: Any) -> None:
            widget.add_tooltip(layer_id, **kwargs)

        return self.add_layer(apply)
