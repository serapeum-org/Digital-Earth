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

import html
import re
from typing import TYPE_CHECKING, Any, List, Optional, Self

from digitalearth.base.basemaps import (
    DEFAULT_BASEMAP_PROVIDER,
    KEYED_BASEMAP_NAMES,
    get_keyed_basemap,
    is_keyed_basemap,
)
from digitalearth.base.deprecation import renamed_method, renamed_parameter
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.web.base import _require_layer_api, _require_maplibre

# Note (#247): `tiles()` takes a URL while `basemap()` takes a provider name; the rename that settles that
# collision belongs to the Core contract, not here.

#: The attribution every OpenStreetMap-derived tile set must carry, and the CARTO variant three of them add
#: to it. Written once because they are a legal requirement rather than a label: a typo in one copy is a
#: provider credited wrongly on one basemap and correctly on the others.
_OSM_ATTRIBUTION = "© OpenStreetMap contributors"
_CARTO_ATTRIBUTION = f"{_OSM_ATTRIBUTION} © CARTO"

#: Named raster XYZ basemaps → ``(url_template, attribution)``. All are token-free public tile services.
_BASEMAP_PROVIDERS = {
    "cartodark": (
        "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        _CARTO_ATTRIBUTION,
    ),
    "cartolight": (
        "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        _CARTO_ATTRIBUTION,
    ),
    "cartovoyager": (
        "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        _CARTO_ATTRIBUTION,
    ),
    "osm": (
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        _OSM_ATTRIBUTION,
    ),
}

#: Canonical (correctly-cased) display names for the basemap keys, used in the not-found error message.
_BASEMAP_DISPLAY_NAMES = {
    "cartodark": "CartoDark",
    "cartolight": "CartoLight",
    "cartovoyager": "CartoVoyager",
    "osm": "OSM",
}

#: py-maplibregl's two layer-switcher styles, validated here so a typo is not a pydantic traceback.
_SWITCHER_THEMES = frozenset({"default", "simple"})

#: The four legal MapLibre control corners.
_CONTROL_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


#: Styling for the small floating panels this tier builds — the legend and the title — kept with the
#: markup that uses it rather than left to the host page.
_PANEL_CSS = (
    "background: rgba(255, 255, 255, 0.92); color: #222; padding: 8px 10px; border-radius: 4px; "
    "font: 12px/1.4 system-ui, sans-serif; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); max-width: 220px;"
)


def _text(value: Any) -> str:
    """Escape a value for the control markup.

    ``InfoBoxControl`` assigns its ``content`` to ``innerHTML``, so every interpolated value is markup
    until it is escaped — including class values read straight out of a caller's GeoDataFrame column,
    which is exactly the kind of thing that arrives from a downloaded shapefile. The exported page is
    meant to be shared, so an unescaped column value is stored XSS in the artifact this tier exists to
    produce.

    Args:
        value: Anything destined for the legend or title markup.

    Returns:
        Its string form with ``&``, ``<``, ``>``, ``"`` and ``'`` escaped.
    """
    return html.escape(str(value), quote=True)


#: A CSS colour this tier is willing to put inside a ``style=""`` attribute: a hex value, an rgb()/rgba()
#: call, or a bare colour keyword. Escaping alone is not enough there — quotes cannot break out, but ``;``
#: starts a new declaration inside the attribute.
_SAFE_COLOR = re.compile(r"^(#[0-9a-fA-F]{3,8}|rgba?\([\d.,\s%]+\)|[a-zA-Z]{3,20})$")


def _css_color(color: Any) -> str:
    """Return a colour safe to interpolate into a ``style`` attribute.

    Args:
        color: The colour a classification produced, or one a caller passed.

    Returns:
        The colour when it is a plain hex/rgb()/keyword value, else ``"transparent"`` — a swatch that
        renders wrong is better than one that carries a second CSS declaration into the page.
    """
    text = str(color)
    return text if _SAFE_COLOR.match(text) else "transparent"


def _swatch(color: str) -> str:
    """Return the markup for one colour chip.

    Args:
        color: A CSS colour, as rendered on the map.

    Returns:
        An inline-block span, sized to line up with a row of text.
    """
    return (
        f'<span style="display:inline-block;width:14px;height:14px;margin-right:6px;'
        f'vertical-align:-2px;background:{_css_color(color)};border:1px solid rgba(0,0,0,.25)"></span>'
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
        ramp = ", ".join(_css_color(color) for color in colors)
        return (
            f'<div style="height:10px;border-radius:2px;background:linear-gradient(to right,{ramp})"></div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:2px">'
            f"<span>{_text(_format_number(values[0]))}</span>"
            f"<span>{_text(_format_number(values[-1]))}</span></div>"
        )
    if kind == "graduated":
        derived = [
            f"{_format_number(values[i])} – {_format_number(values[i + 1])}"
            for i in range(len(values) - 1)
        ]
    else:
        derived = [_format_number(v) for v in values]
    if labels is not None and len(labels) != len(derived):
        raise ValueError(
            f"legend(labels=...) has {len(labels)} entries but the classification has {len(derived)}; "
            f"zip would drop the difference and leave classes out of the key"
        )
    text = labels if labels is not None else derived
    return "".join(
        f'<div style="white-space:nowrap">{_swatch(color)}{_text(label)}</div>'
        for color, label in zip(colors, text)
    )


#: Degrees between graticule lines when a caller names neither step — the interactive tier's own default, so
#: one `graticule()` call draws the same grid on both (#263).
_DEFAULT_GRID_STEP: float = 30.0


def _graticule_features(lon_step: float, lat_step: float) -> dict:
    """Build the meridians and parallels of a graticule as a GeoJSON FeatureCollection.

    Generated rather than fetched: a lat/lon grid is arithmetic, not data, so it needs no source and works
    offline. Meridians are drawn as straight segments sampled every 5° of latitude, which is dense enough
    that a projection curves them smoothly.

    Args:
        lon_step: Degrees between meridians.
        lat_step: Degrees between parallels.

    Returns:
        A GeoJSON ``FeatureCollection`` whose features each carry a ``label`` property, e.g. ``"10°E"``.
    """

    def steps(start: float, stop: float, step: float) -> list:
        """Inclusive range over floats, avoiding the drift a repeated addition would accumulate."""
        count = int(round((stop - start) / step))
        return [start + index * step for index in range(count + 1)]

    def anchored(limit: float, step: float) -> list:
        """Return ``(index, value)`` for lines at 0, ±step, ±2·step … within ``±limit``.

        Anchored on zero rather than on the edge of the range, so the equator and the prime meridian are
        always drawn — stepping from -80 upwards misses the equator at any spacing that does not divide 80.
        The index comes back with the value because the caller's decisions ("is this the antimeridian?",
        "which hemisphere?") are about *which* line it is, and asking that of a float would be an equality
        test on a computed product.
        """
        count = int(limit // step)
        return [(index, index * step) for index in range(-count, count + 1)]

    features = []
    last_meridian = int(180.0 // lon_step)
    for index, lon in anchored(180.0, lon_step):
        if index == last_meridian and lon >= 180.0:
            continue  # the antimeridian is the same line as -180
        features.append(
            _graticule_line(
                lon,
                _hemisphere(index, "E", "W"),
                [[lon, lat] for lat in steps(-85.0, 85.0, 5.0)],
            )
        )
    for index, lat in anchored(80.0, lat_step):
        features.append(
            _graticule_line(
                lat,
                _hemisphere(index, "N", "S"),
                [[lon, lat] for lon in steps(-180.0, 180.0, 5.0)],
            )
        )
    return {"type": "FeatureCollection", "features": features}


def _hemisphere(index: int, positive: str, negative: str) -> str:
    """Return the hemisphere letter for the line at ``index`` steps from zero.

    Args:
        index: Step count from the equator / prime meridian; the sign is the hemisphere.
        positive: Letter for the positive side (``"E"`` or ``"N"``).
        negative: Letter for the negative side (``"W"`` or ``"S"``).

    Returns:
        The letter, or ``""`` for the zero line, which belongs to neither hemisphere. Decided from the
        step index rather than the coordinate, so no float is compared for equality.
    """
    if index == 0:
        return ""
    return positive if index > 0 else negative


def _graticule_line(value: float, suffix: str, coordinates: list) -> dict:
    """Build one graticule line as a GeoJSON feature.

    Args:
        value: The line's degree value, used for its label.
        suffix: The hemisphere letter, or ``""`` for the zero line.
        coordinates: The line's sampled vertices.

    Returns:
        A GeoJSON ``Feature`` carrying a ``label`` property such as ``"10°E"``.
    """
    return {
        "type": "Feature",
        "properties": {"label": f"{abs(value):g}°{suffix}"},
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


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


def draw_text(_web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre symbol layer for a single text annotation.

    An annotation draws from no data in the figure: its anchor and its string are what the caller passed,
    recorded as values on its symbology, which is why `_data` is unused. The one-point GeoJSON source the
    symbol reads is built here from those values rather than opened from anywhere.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        _data: Unused — an annotation has no source in the figure to open.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` holding that point source and the symbol layer.

    Raises:
        ValueError: when the description carries none of the values the annotation is built from —
            its anchor, its string, its text size or its colours — naming the layer, its kind and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, layer_types = _require_layer_api()
    props = required_props(
        layer, "lon", "lat", "s", "text_size", "color", "halo_color", "halo_width"
    )
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec={
            "type": "geojson",
            "data": {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(props["lon"]), float(props["lat"])],
                },
                "properties": {"text": props["s"]},
            },
        },
        layer=layer_cls(
            id=layer.id,
            type=layer_types.SYMBOL,
            source=source_id,
            layout={
                "text-field": ["get", "text"],
                "text-size": float(props["text_size"]),
                "text-allow-overlap": True,
            },
            paint={
                "text-color": props["color"],
                "text-halo-color": props["halo_color"],
                "text-halo-width": float(props["halo_width"]),
            },
        ),
    )


def draw_graticule(_web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre lines (and degree labels) for a graticule layer.

    A graticule draws from no data in the figure: its geometry is generated here from the two steps the
    caller asked for, which is why `_data` is unused. Everything it needs is in its symbology, and the
    GeoJSON source the lines read is built from that rather than opened from anywhere.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        _data: Unused — a graticule has no source in the figure to open.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` holding that GeoJSON source, the line layer, and —
        when the description asked for labels — the degree labels beside it as an extra layer. A hidden
        graticule hides its labels too, so the numbers never float with nothing to annotate.

    Raises:
        ValueError: when the description carries none of the values the grid is generated from — its
            two steps, its colour, width and opacity, or whether it is labelled — naming the layer, its kind and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, derived_ids, required_props

    layer_cls, layer_types = _require_layer_api()
    props = required_props(
        layer, "lon_step", "lat_step", "color", "width", "opacity", "labels"
    )
    (label_id,) = derived_ids("graticule", layer.id)
    features = _graticule_features(float(props["lon_step"]), float(props["lat_step"]))
    source_id = f"{layer.id}-src"
    color = props["color"]
    visible = layer.visible
    line = layer_cls(
        id=layer.id,
        type=layer_types.LINE,
        source=source_id,
        paint={
            "line-color": color,
            "line-width": float(props["width"]),
            "line-opacity": float(props["opacity"]),
        },
        layout=None if visible else {"visibility": "none"},
    )
    extra = []
    if props["labels"]:
        label_layout: dict = {
            "text-field": ["get", "label"],
            "text-size": 10.0,
            "symbol-placement": "line",
        }
        if not visible:
            # Otherwise a hidden graticule leaves its degree numbers floating with nothing to annotate.
            label_layout["visibility"] = "none"
        extra.append(
            layer_cls(
                id=label_id,
                type=layer_types.SYMBOL,
                source=source_id,
                layout=label_layout,
                paint={
                    "text-color": color,
                    "text-halo-color": "#000000",
                    "text-halo-width": 1.0,
                },
            )
        )
    return DrawnLayer(
        source_id=source_id,
        source_spec={"type": "geojson", "data": features},
        layer=line,
        extra_layers=tuple(extra),
    )


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
        bounds: Optional[Any] = None,
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
            bounds: ``(west, south, east, north)`` in lon/lat that the service actually covers, passed
                to the MapLibre source's ``bounds``. It is a **declaration**, not a check: the map stays
                pannable everywhere, and MapLibre simply stops requesting tiles outside the box instead
                of collecting 404s across the rest of the world. ``None`` means global.

        Returns:
            The same map instance, so builder calls chain; the basemap is registered as an underlay so
            data drawn before or after it still renders on top.

        Raises:
            ValueError: when ``bounds`` is not four numbers — MapLibre silently ignores a malformed
                ``bounds``, so the coverage would quietly go undeclared.

        Examples:
            - Put an XYZ service on the ground. It draws, but it is not a *data* layer, so it
              never shows up in the registry a layer switcher lists (needs the ``web`` extra, so
              the block is skipped without it):
                ```python
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> m = WebMap().tiles(                              # doctest: +SKIP
                ...     "https://tile.example.org/{z}/{x}/{y}.png",
                ...     attribution="© Example", max_zoom=12,
                ... )
                >>> len(m.layers), m.layer_ids                       # doctest: +SKIP
                (1, [])

                ```
            - Call order does not decide draw order: tiles added *after* the data still land
              beneath it, because this registers an underlay instead of appending on top:
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import Point               # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     geometry=[Point(0, 0), Point(1, 1)], crs=4326,
                ... )
                >>> m = WebMap().points(gdf, name="sites")           # doctest: +SKIP
                >>> data_layer = m.layers[0]                         # doctest: +SKIP
                >>> m = m.tiles("https://t.example.org/{z}/{x}/{y}.png")  # doctest: +SKIP
                >>> m.layers[0] is data_layer, m.layers[-1] is data_layer  # doctest: +SKIP
                (False, True)

                ```
            - A malformed coverage box is refused rather than dropped: MapLibre would ignore it
              without a word, and the service would go on collecting 404s across the rest of the
              world:
                ```python
                >>> try:                                             # doctest: +SKIP
                ...     WebMap().tiles("https://t.example.org/{z}/{x}/{y}.png", bounds=(0, 0, 1))
                ... except ValueError as error:
                ...     print(str(error).split(";")[0])
                tiles(bounds=...) takes (west, south, east, north) in lon/lat

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.basemap: the named-provider wrapper.
            digitalearth.web.base.WebMapBase.add_underlay: the registration that keeps it at the
                bottom of the stack.
        """
        layer_cls, layer_types = _require_layer_api()
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
        if bounds is not None:
            box = [float(value) for value in bounds]
            if len(box) != 4:
                raise ValueError(
                    f"tiles(bounds=...) takes (west, south, east, north) in lon/lat; got {bounds!r}"
                )
            source["bounds"] = box
        layer = layer_cls(
            id=layer_id,
            type=layer_types.RASTER,
            source=src_id,
            paint={"raster-opacity": float(opacity)},
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, source)
            widget.add_layer(layer)

        return self.add_underlay(apply)

    def basemap(
        self,
        provider: str = DEFAULT_BASEMAP_PROVIDER,
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
        why the guard lives there. What a keyed preset's coverage *does* do here is reach the MapLibre
        source as its ``bounds``, so the engine stops asking for tiles the service does not have.

        Args:
            provider: A token-free basemap name — ``"CartoLight"``, ``"CartoDark"``, ``"CartoVoyager"`` or
                ``"OSM"`` (case-insensitive) — or a **keyed** preset name such as ``"Planet.NICFI"`` (see
                :mod:`digitalearth.base.basemaps`), whose credential is read from the environment.
                Defaults to ``digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER``, the one constant the
                interactive tier reads too, so an unqualified basemap looks the same on both.
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
                bounds=keyed.bounds,
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
        layer_id: Optional[str] = None,
        title: Optional[str] = None,
        position: str = "bottom-right",
        labels: Optional[list] = None,
        visible: bool = True,
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
            layer_id: Which layer's classes to describe. `None` takes the most recently classified layer,
                which is what the tier recorded before layers had ids.
            title: Heading above the key. ``None`` uses the classified column's name, followed by the
                units in parentheses when :func:`~digitalearth.base.autostyle.auto_style` supplied them
                for the raster the classification came from. A title given here always wins, and a unit
                is never guessed: without one the heading is the bare column name, as it always was.
            position: One of the four MapLibre corners.
            labels: Explicit row labels, replacing the derived ones — for units, or for renaming
                categories. Ignored for a continuous ramp, which has no rows.
            visible: `False` draws no key, so a caller passing a flag through does not have to branch.
                The corner is still checked, so one spelling of it is not valid only half the time.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners, when `layer_id`
                names a layer that carries no classification, or when no classified layer has been added at
                all — there is nothing to build a key from, and an empty box would be worse than an error.
            KeyError: when `layer_id` names no layer on this map.

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
        # Validated before the flag is read: `legend(position="middle", visible=False)` was accepted while
        # `legend(position="middle")` raised, so a caller passing a flag through was checked half the time
        # (review L7).
        _check_position(position)
        if not visible:
            return self
        _require_maplibre()
        spec = self.last_legend if layer_id is None else self._legend_of(layer_id)
        if not spec:
            raise ValueError(
                "legend() has nothing to describe: no classified layer has been added yet. Add a "
                "choropleth (or any builder given column=...) first."
            )
        if title is not None:
            heading = title
        else:
            heading = spec.get("column") or ""
            units = spec.get("units")
            if heading and units:
                heading = f"{heading} ({units})"
        head = (
            f'<div style="font-weight:600;margin-bottom:4px">{_text(heading)}</div>'
            if heading
            else ""
        )
        rows = _legend_rows(
            spec["kind"], list(spec["values"]), list(spec["colors"]), labels
        )
        # One panel per kind: a second legend replaces the first rather than stacking an identical box
        # in the same corner, and it is built into the widget rather than appended as a layer.
        self._panels["legend"] = (f"<div>{head}{rows}</div>", position)
        return self

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

        Note:
            A row toggles exactly one MapLibre layer, because that is what py-maplibregl's control does.
            Builders that draw more than one layer — ``cluster`` (bubbles, counts, loose points) and
            ``graticule`` (lines, degree labels) — are listed once, under their main layer, so toggling a
            cluster hides its bubbles while the count labels remain. ``remove_layer`` does take the whole
            group. Hiding the parts together needs a control py-maplibregl does not ship.

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
        if theme not in _SWITCHER_THEMES:
            raise ValueError(
                f"layer_control(theme={theme!r}) must be one of {sorted(_SWITCHER_THEMES)}"
            )
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
        # Held as a request, not appended as a layer: the live layers are resolved when the widget is
        # built, so removing a layer afterwards cannot leave a dead row in the saved page, and calling
        # this twice replaces the request rather than stacking a second identical panel.
        self._switcher = {
            "layer_ids": wanted,
            "theme": theme,
            "position": position,
        }
        self._record_furniture(
            "layer_switcher", anchor=position, layers=tuple(wanted), theme=theme
        )
        return self

    def text(
        self,
        lon: float,
        lat: float,
        s: Optional[str] = None,
        *,
        crs: Any = 4326,
        text_size: Optional[float] = None,
        color: str = "#ffffff",
        halo_color: str = "#000000",
        halo_width: float = 1.0,
        name: Optional[str] = None,
        size: Optional[float] = None,
        string: Optional[str] = None,
    ) -> Self:
        """Place a single line of text at a coordinate.

        The "label this spot" case, which needed a whole GeoDataFrame before —
        :meth:`~digitalearth.web.vector.VectorMixin.labels` is the data-driven counterpart.

        Args:
            lon: X of the anchor point, in `crs` — a longitude by default.
            lat: Y of the anchor point, in `crs` — a latitude by default.
            s: The string to draw. Named `s` as it is on the static and interactive tiers (#260), which
                follow matplotlib's own spelling.
            crs: What `lon`/`lat` are measured in. EPSG:4326 by default; a point in another CRS is
                reprojected into the CRS this tier places data in, exactly as a feature is (#260).
            text_size: Text size in pixels (``14.0`` when omitted — the signature's ``None`` is the
                "not passed" sentinel the deprecated spelling is resolved against). Named for the text
                rather than ``size``, which means the visual size of a marker everywhere else.
            color: Text colour.
            halo_color: Colour of the outline behind the glyphs, which keeps it legible over imagery.
            halo_width: Halo width in pixels; ``0`` disables it.
            string: **Deprecated** spelling of `s`; forwarded unchanged, after a
                ``DeprecationWarning`` that ``string=`` will be removed in a future release.
            name: What a layer switcher calls this annotation; ``None`` uses its generated id.
            size: **Deprecated** spelling of ``text_size``; forwarded unchanged, after a
                ``DeprecationWarning`` that ``size=`` will be removed in a future release.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when both ``text_size`` and the deprecated ``size`` are passed — they name one
                parameter, so preferring either would silently drop the other.

        Examples:
            - Mark a place:
                ```python
                >>> from digitalearth.web import WebMap                     # doctest: +SKIP
                >>> WebMap().basemap().text(4.9, 52.4, "Amsterdam")         # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.labels: label many features from a column.
        """
        s = renamed_parameter(
            new="s",
            value=s,
            old="string",
            alias=string,
            caller="WebMap.text()",
        )
        if s is None:
            raise TypeError(
                "text() needs the string to draw; pass it as the third argument"
            )
        lon, lat = self._as_display_point(float(lon), float(lat), crs)
        _require_layer_api()
        text_size = renamed_parameter(
            new="text_size",
            value=text_size,
            old="size",
            alias=size,
            caller="WebMap.text()",
            default=14.0,
        )
        layer_id = self._layer_id("text", name)
        # An annotation is decoration, not data: it must not decide where the map looks. On its own it is
        # a zero-area extent (maximum zoom on a point); beside data it drags the extent to reach it.
        self._index_layer(
            layer_id,
            name,
            kind="text",
            symbology=Symbology(
                props={
                    "lon": float(lon),
                    "lat": float(lat),
                    "s": s,
                    "text_size": float(text_size),
                    "color": color,
                    "halo_color": halo_color,
                    "halo_width": float(halo_width),
                }
            ),
        )
        return self

    def set_title(
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
        body = f'<div style="font-weight:600;font-size:15px">{_text(heading)}</div>'
        if subtitle:
            body += f'<div style="opacity:.75;margin-top:2px">{_text(subtitle)}</div>'
        self._panels["title"] = (f"<div>{body}</div>", position)
        self._title = heading
        return self

    def graticule(
        self,
        *,
        lon_step: Optional[float] = None,
        lat_step: Optional[float] = None,
        spacing: Optional[float] = None,
        color: str = "#888888",
        width: float = 0.5,
        opacity: float = 0.6,
        labels: bool = True,
        name: Optional[str] = None,
        visible: bool = True,
    ) -> Self:
        """Draw a lat/lon grid over the map.

        A graticule is a property of the map, not of a tile service — no basemap provides one, so this tier
        had no way to show one at all. The lines are ordinary GeoJSON, so they embed in the page and a map
        saved with ``offline=True`` keeps its grid with no network.

        Args:
            lon_step: Degrees between meridians; `30` by default, as on the interactive tier (#263).
            lat_step: Degrees between parallels; `30` by default.
            spacing: One step for both, for a caller who wants a square grid; it overrides the two above.
                ``30`` gives a readable global grid. The grid is always
                global — it is not clipped to the view, because a web map is pannable and a grid that
                stopped at the opening extent would end mid-pan — so a tight spacing is expensive:
                ``10`` embeds ~43 KiB of GeoJSON, ``1`` embeds ~420 KiB and 521 labels. Prefer the
                coarsest spacing that reads.
            color: Line colour.
            width: Line width in pixels.
            opacity: Line opacity in ``[0, 1]``; a graticule is reference, so it should sit under the data
                visually as well as in the stack.
            labels: Whether to label each line with its degree value at the map's edge.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when a step is not positive, or is wider than the 180° of latitude there is
                to divide — either produces a grid with no lines and no hint as to why.

        Examples:
            - A 10° grid under the data:
                ```python
                >>> from digitalearth.web import WebMap        # doctest: +SKIP
                >>> WebMap().basemap().graticule()             # doctest: +SKIP

                ```
        """
        _require_layer_api()
        # One step for both is what `spacing=` meant, and it is still the shortest way to ask for a square
        # grid; the two steps are what every other tier takes, and what a reader of a world map usually wants
        # (#263). A tier default of 30 degrees matches the interactive tier's.
        lon_step = _DEFAULT_GRID_STEP if lon_step is None else lon_step
        lat_step = _DEFAULT_GRID_STEP if lat_step is None else lat_step
        if spacing is not None:
            lon_step = lat_step = spacing
        for name_of, step in (("lon_step", lon_step), ("lat_step", lat_step)):
            if step <= 0 or step > 180:
                # 180 is the widest meaningful step: it still yields the prime meridian and the antimeridian,
                # while anything wider leaves a grid with a single line in it.
                raise ValueError(
                    f"graticule({name_of}={step!r}) must be greater than 0 and at most 180 degrees"
                )
        # The degree labels are drawn under an id derived from this one, which the allocation reserves too,
        # so a caller's `name=` can no longer take it (review M5).
        layer_id = self._layer_id("graticule", name or "Graticule", kind="graticule")
        # What was asked for, as values. The lines themselves are built by `draw_graticule` from exactly
        # this, so the figure describes the grid rather than naming one that a closure drew elsewhere.
        self._index_layer(
            layer_id,
            layer_id,
            kind="graticule",
            visible=visible,
            symbology=Symbology(
                props={
                    "lon_step": float(lon_step),
                    "lat_step": float(lat_step),
                    "color": color,
                    "width": float(width),
                    "opacity": float(opacity),
                    "labels": bool(labels),
                }
            ),
        )
        # Reference geography says nothing about where to look, so it does not frame the map. Its band —
        # over the basemap, under the data — comes from the kind's registration, not from here.
        return self

    #: Deprecated spelling of :meth:`set_title`, the contract's name for a figure's heading (#299). The
    #: web tier's other `title=` -- the HTML document's -- is untouched: it names a different thing.
    title = renamed_method(new="set_title", old="title", owner="WebMap")

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

        self._record_furniture(
            "navigation",
            anchor=position,
            show_compass=show_compass,
            show_zoom=show_zoom,
            visualize_pitch=visualize_pitch,
        )
        return self._queue(apply)

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

        self._record_furniture(
            "scale_bar", anchor=position, unit=unit, max_width=int(max_width)
        )
        return self._queue(apply)

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

        self._record_furniture("fullscreen", anchor=position)
        return self._queue(apply)

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

        self._record_furniture("measure", anchor=position, distance=distance, area=area)
        return self._queue(apply)

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

        # Tagged with the layer it belongs to, so `remove_layer` takes the popup off with it rather than
        # leaving a page that pops up over a layer nobody can see.
        apply._digitalearth_layer_id = layer_id  # type: ignore[attr-defined]
        self._record_tooltip(
            layer_id, fields, trigger="click", explicit=layer is not None
        )
        return self._queue(apply)

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

        apply._digitalearth_layer_id = layer_id  # type: ignore[attr-defined]
        self._record_tooltip(
            layer_id, fields, trigger="hover", explicit=layer is not None
        )
        return self._queue(apply)
