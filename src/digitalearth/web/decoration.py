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
The ``measure`` tool's drawn geometry comes back through ``drawn_features`` as a pyramids ``FeatureCollection``,
for pyramids to compute geodesic distance/area (the GIS part).
"""

import html
import re
from dataclasses import replace as _with_fields
from typing import TYPE_CHECKING, Any, Self

from digitalearth.base.ask import UNSET, Ask, Maybe
from digitalearth.base.basemaps import (
    DEFAULT_BASEMAP_PROVIDER,
    KEYED_BASEMAP_NAMES,
    get_keyed_basemap,
    is_keyed_basemap,
)
from digitalearth.base.controls import check_control_position, resolved_controls
from digitalearth.base.spec import Guide, LayerSpec, Symbology
from digitalearth.web.base import _require_layer_api, _require_maplibre, as_finite

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

#: What this tier's ``layer_control`` can actually build: one visibility row per layer, which is what
#: py-maplibregl's switcher control is. An opacity slider or a basemap picker would each need a control the
#: library does not ship, so naming either is refused rather than accepted and dropped.
_OFFERED_CONTROLS = ("visibility",)

#: The CRS MapboxDraw reports drawn geometry in. GeoJSON coordinates are lon/lat by definition, whatever the
#: map shows, so :meth:`DecorationMixin.drawn_features` stamps this rather than reading it off the map.
_DRAWN_CRS = 4326


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


def _legend_rows(kind: str, values: list, colors: list, labels: list | None) -> str:
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


#: Where a colour key sits when its :class:`~digitalearth.base.spec.encoding.Guide` names no corner. A guide
#: written by :meth:`DecorationMixin.legend` always names one — ``position`` defaults to it — so this is for a
#: guide that arrived in a figure read back from disk, where the anchor is optional by design.
_DEFAULT_LEGEND_ANCHOR = "bottom-right"

#: Where a key sits when the caller names no corner. `legend()`'s own default, and — through the shared
#: `_record_key` — `colorbar()`'s as well, so two public spellings of one mechanism cannot drift to two
#: corners. Equal to :data:`_DEFAULT_LEGEND_ANCHOR` and separate from it on purpose: that one answers
#: "a guide arrived carrying no anchor", this one answers "a caller named no corner".
DEFAULT_KEY_ANCHOR = "bottom-right"

#: The symbology property a caller's explicit legend row labels are recorded under, on the layer whose key
#: they label. Held on the layer for the same reason the guide is: the panel is **derived** from the live
#: layers every time one is added, removed, hidden or restyled, so anything the key is drawn from has to
#: survive on the layer rather than in the panel that was built once.
LEGEND_LABELS = "legend_labels"


def keyable_layer_ids(web_map: Any) -> list[str]:
    """Return the live layers a colour key could describe, bottom-first.

    A layer qualifies when two things hold: its ``color`` channel is driven by a **field** rather than by a
    constant — a flat colour has no values to label, which is what makes a key on an unclassified layer
    refusable — and the builder filed the classification it drew with, which is where the swatches and their
    colours come from.

    Args:
        web_map: The map to read.

    Returns:
        The qualifying ids in draw order, bottom first, so the caller reads the topmost off the end.
    """
    keyed: list[str] = []
    for layer_id in web_map.layer_ids:
        encoding = web_map._layer_tree.get(layer_id).symbology.encoding("color")
        if encoding is None or encoding.is_constant:
            continue
        if layer_id in web_map._legends:
            keyed.append(layer_id)
    return keyed


def guided_layer_ids(web_map: Any) -> list[str]:
    """Return the live layers asking to have their colour explained, bottom-first.

    The keyable layers (:func:`keyable_layer_ids`) narrowed to those carrying a
    :class:`~digitalearth.base.spec.encoding.Guide` that says ``show``. Visibility is **not** consulted
    here: a guide on a hidden layer is a key the caller asked for and will see again the moment the layer
    is shown, so it is still one of the map's keys — it is simply not the one on screen, which is
    :func:`drawn_guide_layer`'s question.

    Args:
        web_map: The map to read.

    Returns:
        The ids in draw order, bottom first, so the caller reads the topmost off the end.
    """
    asked: list[str] = []
    for layer_id in keyable_layer_ids(web_map):
        guide = web_map._layer_tree.get(layer_id).symbology.guide()
        if guide is not None and guide.show:
            asked.append(layer_id)
    return asked


def drawn_guide_layer(web_map: Any) -> str | None:
    """Return the layer whose colour key is on screen, or `None` when none is.

    **This tier draws one key**, and this is the one function that decides whose — the topmost *visible*
    layer asking for one. Both callers need the same answer and for the same reason: the panel is derived
    from it (:func:`refresh_legend_panel`), and `colorbar(visible=False)` with no id has to take off the key
    the caller can see rather than whichever classified layer happens to sit on top (review M3). Two
    readings of "whose key" is exactly how those two came to disagree.

    Args:
        web_map: The map to read.

    Returns:
        The layer id, or `None` when nothing on the map asks to be explained or everything that does is
        hidden.
    """
    for layer_id in reversed(guided_layer_ids(web_map)):
        # `is_visible` rather than `layer.visible`, so a layer hidden with its group counts as hidden.
        if web_map._layer_tree.is_visible(layer_id):
            return layer_id
    return None


def _legend_panel(spec: dict, guide: Guide, labels: Any | None) -> tuple:
    """Build one colour key as a floating panel.

    Args:
        spec: The classification the layer was drawn with — its ``kind``, ``column``, ``values`` and the
            ``colors`` actually rendered.
        guide: What the layer says about explaining that channel: the heading, and the corner.
        labels: Explicit row labels replacing the derived ones, or ``None``.

    Returns:
        The ``(content, position)`` pair :attr:`~digitalearth.web.base.WebMapBase._panels` holds.

    Raises:
        ValueError: when `labels` has a different number of entries from the classification, as
            :func:`_legend_rows` refuses it.
    """
    if guide.title is not None:
        heading = guide.title
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
        spec["kind"],
        list(spec["values"]),
        list(spec["colors"]),
        None if labels is None else list(labels),
    )
    return (f"<div>{head}{rows}</div>", guide.anchor or _DEFAULT_LEGEND_ANCHOR)


def refresh_legend_panel(web_map: Any) -> None:
    """Rebuild the map's colour-key panel from the guides its live layers carry.

    The whole of what order 24 buys this tier. The panel used to be written once, by the `legend()` call that
    asked for it, and then maintained by nothing: removing the keyed layer left its key on screen, and hiding
    it did too — a wrong map that looks right. Now the key is **derived**, every time the layers change, from
    the :class:`~digitalearth.base.spec.encoding.Guide` each layer carries on its ``color`` encoding. A guide
    that goes with its layer takes the key with it for free.

    This tier draws **one** key, so when several layers carry a guide the topmost visible one wins; see
    :meth:`DecorationMixin.legend` for what that means to a caller. Which one that is is
    :func:`drawn_guide_layer`'s question, asked here and by `legend(visible=False)` alike so the two cannot
    disagree about whose key is on screen.

    Args:
        web_map: The map whose panel to rebuild. Its ``_panels["legend"]`` is written, replaced, or removed;
            no other panel is touched.
    """
    layer_id = drawn_guide_layer(web_map)
    if layer_id is None:
        # Nothing on the map asks to be explained any more, so neither does the panel. Popped rather than
        # left empty: `_panels` is what the widget builds its `InfoBoxControl`s from, and an entry here is
        # a box.
        web_map._panels.pop("legend", None)
        return
    layer = web_map._layer_tree.get(layer_id)
    web_map._panels["legend"] = _legend_panel(
        web_map._legends[layer_id],
        layer.symbology.guide(),
        layer.symbology.props.get(LEGEND_LABELS),
    )


#: The Natural-Earth resolutions `cleopatra.basemap.reference` publishes, for `coastlines`/`borders` to refuse
#: anything else by name rather than letting the read fail deeper down.
_NATURAL_EARTH_RESOLUTIONS = frozenset({"110m", "50m", "10m"})

#: Degrees between graticule lines when a caller names neither step — the interactive tier's own default, so
#: one `graticule()` call draws the same grid on both (#263).
_DEFAULT_GRID_STEP: float = 30.0

#: What an unstyled basemap and an unstyled graticule are drawn at. The values left the signatures with #334;
#: these are where they live now, cited by name from the docstrings that used to show them.
BASEMAP_OPACITY = 1.0
GRATICULE_OPACITY = 0.6


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
    """Validate a control corner, raising ``ValueError`` for anything but the four legal corners.

    The corners themselves are MapLibre's, and they are now
    :data:`~digitalearth.base.controls.CONTROL_POSITIONS` — the same four the interactive tier anchors its
    layer control to, so ``position=`` means one thing on both (#264). This stays as the module's own name
    because seven builders here call it.

    Args:
        position: The requested corner placement.

    Raises:
        ValueError: when ``position`` is not one of :data:`~digitalearth.base.controls.CONTROL_POSITIONS`.
    """
    check_control_position(position)


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


def draw_tiles(_web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre raster source and layer for a tile basemap.

    A basemap draws from no data in the figure: the XYZ service, its attribution, its tile size and whatever
    coverage was declared for it are values the caller passed, recorded on its symbology, which is why
    `_data` is unused. They are also plain JSON, so a map whose only layer is a basemap describes itself in a
    figure that can be written down and read back.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        _data: Unused — a basemap has no source in the figure to open.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` holding the raster source and the raster layer
        reading it. It is an ordinary style layer, so it takes the ordinary route.

    Raises:
        ValueError: when the description carries neither the MapLibre source the tiles are served from nor
            the opacity they are drawn at, naming the layer, its kind and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, layer_types = _require_layer_api()
    props = required_props(layer, "source", "opacity")
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec=dict(props["source"]),
        layer=layer_cls(
            id=layer.id,
            type=layer_types.RASTER,
            source=source_id,
            paint={"raster-opacity": float(props["opacity"])},
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
            two steps, its colour, width and opacity, or whether it is labelled — naming the layer, its
            kind and what is missing.
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
        opacity: Maybe[float] = UNSET,
        max_zoom: int | None = None,
        bounds: Any | None = None,
        name: str | None = None,
    ) -> Self:
        """Add a raster XYZ/WMTS tile layer **beneath** the data (recipe W1).

        Args:
            url: An XYZ tile URL template containing ``{z}/{x}/{y}`` (already Web-Mercator tiles).
            attribution: Attribution text shown in the map's attribution control.
            tile_size: Tile edge length in pixels (256 for standard XYZ; 512 for some retina services).
            opacity: Raster opacity in ``[0, 1]``; not passed leaves :data:`BASEMAP_OPACITY`.
            max_zoom: The deepest zoom the service serves. Past it MapLibre over-zooms the last real
                tiles instead of requesting levels that do not exist; ``None`` leaves the source
                unbounded.
            bounds: ``(west, south, east, north)`` in lon/lat that the service actually covers, passed
                to the MapLibre source's ``bounds``. It is a **declaration**, not a check: the map stays
                pannable everywhere, and MapLibre simply stops requesting tiles outside the box instead
                of collecting 404s across the rest of the world. ``None`` means global.
            name: What the basemap is addressed by — ``set_visible``, ``move_layer``, ``replace_layer`` and
                ``remove_layer`` all take it. ``None`` generates ``tiles-1``, ``tiles-2``, … as every other
                unnamed layer is numbered.

        Returns:
            The same map instance, so builder calls chain; the basemap is registered as an underlay so
            data drawn before or after it still renders on top.

        Raises:
            ValueError: when ``bounds`` is not four numbers — MapLibre silently ignores a malformed
                ``bounds``, so the coverage would quietly go undeclared.

        Examples:
            - Put an XYZ service on the ground. It is described and addressable like any other layer —
              what keeps it out of a layer switcher is that
              :meth:`~digitalearth.web.decoration.DecorationMixin.layer_control` offers the data layers
              rather than every id (needs the ``web`` extra, so the block is skipped without it):
                ```python
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> m = WebMap().tiles(                              # doctest: +SKIP
                ...     "https://tile.example.org/{z}/{x}/{y}.png",
                ...     attribution="© Example", max_zoom=12,
                ... )
                >>> len(m.layers), m.layer_ids                       # doctest: +SKIP
                (1, ['tiles-1'])

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
        _require_layer_api()
        ask = Ask()
        source: dict = {
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
        layer_id = self._layer_id("tiles", name)
        # The service, as values: :func:`draw_tiles` builds the MapLibre source and the raster layer from
        # exactly this, so the figure describes the basemap rather than naming one a closure drew elsewhere.
        # The MapLibre source id is derived from the layer's, as it is for every other described kind, so
        # the basemap no longer takes a second number beside itself (#296).
        self._index_layer(
            layer_id,
            name,
            kind="basemap",
            symbology=Symbology(
                props={
                    "source": source,
                    "opacity": float(ask("opacity", opacity, BASEMAP_OPACITY)),
                    **ask.record,
                }
            ),
        )
        return self

    def basemap(
        self,
        provider: str = DEFAULT_BASEMAP_PROVIDER,
        *,
        opacity: Maybe[float] = UNSET,
        api_key: str | None = None,
        preset: dict | None = None,
        name: str | None = None,
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
            opacity: Basemap opacity in ``[0, 1]``; not passed leaves :data:`BASEMAP_OPACITY`, and is
                forwarded to :meth:`tiles` as the sentinel so the ask is recorded once, there.
            api_key: Credential for a keyed preset; ``None`` reads the preset's environment variable.
            preset: The keyed preset's own keywords, as a dict (for NICFI: ``date``, ``flavour``,
                ``mosaic``). A dict rather than loose keywords so that all three tiers take a preset the
                same way, and so a mistyped style argument is an unexpected keyword rather than something
                ``**preset`` silently swallows.
            name: What the basemap is addressed by, forwarded to
                :meth:`~digitalearth.web.decoration.DecorationMixin.tiles`.

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
                name=name,
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
        return self.tiles(url, attribution=attribution, opacity=opacity, name=name)

    def legend(
        self,
        *,
        layer_id: str | None = None,
        title: str | None = None,
        position: str = DEFAULT_KEY_ANCHOR,
        labels: list | None = None,
        visible: bool = True,
    ) -> Self:
        """Add a key for a classified layer (recipe W2).

        A thematic map is unreadable without one, and until this the tier drew the classes and left the
        caller to build a key out of band from ``last_breaks``. The classification records what it actually
        coloured with (:attr:`~digitalearth.web.base.WebMapBase.last_legend`), so the key shows the rendered
        colours rather than a second guess at them.

        The three shapes a classification can take are all handled: a **categorical** one gets a swatch per
        category, a **graduated** one a swatch per class with its range, and a **continuous** ramp a
        gradient bar with its end values.

        **The key belongs to the layer, not to the map** (DE-48, order 24). What this call records is a
        :class:`~digitalearth.base.spec.encoding.Guide` on the layer's ``color``
        :class:`~digitalearth.base.spec.encoding.Encoding` — ``show``, the heading, and the corner — and the
        panel is **derived** from the guides the live layers carry, every time the layers change. So the key
        moves, hides and disappears with the layer it describes: remove the layer and the key goes, hide it
        and the key goes, show it again and the key comes back. It also travels: a guide survives
        ``FigureSpec`` being written and read back, which a panel built once never did.

        Args:
            layer_id: Which layer's classes to describe. `None` takes the topmost **visible** layer that
                published a colour scale — a classified ``choropleth``/``points``, or a raster band's ramp,
                since a band carries a colour encoding of its own — and so the most recent one a viewer can
                see, since a key drawn for nothing on screen is the thing this call exists to avoid. It
                falls back to the topmost such layer of any visibility when none is visible. Under
                ``visible=False`` it instead takes the layer whose key is drawn, so "no key" takes off the
                key there is.
            title: Heading above the key. ``None`` uses the classified column's name, followed by the
                units in parentheses when :func:`~digitalearth.base.autostyle.auto_style` supplied them
                for the raster the classification came from. A title given here always wins, and a unit
                is never guessed: without one the heading is the bare column name, as it always was.
            position: One of the four MapLibre corners. Recorded as the guide's ``anchor``, which takes the
                same four spellings (:data:`~digitalearth.base.registry.FURNITURE_ANCHORS` and
                :data:`~digitalearth.base.controls.CONTROL_POSITIONS` are **equal** tuples — two objects,
                one vocabulary), so nothing is translated between them.
            labels: Explicit row labels, replacing the derived ones — for units, or for renaming
                categories. Ignored for a continuous ramp, which has no rows. Recorded on the layer beside
                its guide, so a key rebuilt after another layer is removed still carries them.
            visible: `False` records the guide with ``show=False`` — the layer says outright that its colour
                is explained by nothing — and draws no key, so a caller passing a flag through does not have
                to branch. **Every other argument is checked before this one is read**, whether or not any
                layer has been keyed yet, so no argument is valid only half the time and none is valid only
                in a particular call order. The one refusal the flag does decide is "this map has nothing to
                describe", which is a fact about the map rather than about the call.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners, when `title` is not
                a non-empty string, when `layer_id` names a layer that carries no classification, when no
                classified layer has been added at all — there is nothing to build a key from, and an empty
                box would be worse than an error — or when `labels` has a different number of entries from
                the classification. Only the "nothing classified" one waits on ``visible``.
            KeyError: when `layer_id` names no layer on this map.
            TypeError: when `labels` is not a sequence.

        Note:
            **This tier draws one key, and the four tiers differ here.** ``_panels`` holds one entry per
            kind, so a second legend replaces the first rather than stacking an identical box in the same
            corner. Layers may each carry a guide all the same, and when several do **the topmost visible
            one wins** — draw order decides, not call order. So keying a lower layer after a higher one
            leaves the higher one's key drawn; take the higher one's off with
            ``legend(layer_id=<higher>, visible=False)`` first. The static tier has no such limit
            (matplotlib takes any number of colorbars) and neither does the interactive tier (its guides are
            per-layer options); the 3-D tier has one scalar-bar slot per title.

            A guide on a **hidden** layer is recorded and draws nothing until the layer is shown, which is
            the same rule stated from the other side: the key is drawn while the layer is.

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
            digitalearth.web.decoration.refresh_legend_panel: derives the panel from the recorded guides.
        """
        return self._record_key(
            layer_id,
            title=title,
            position=position,
            labels=labels,
            visible=visible,
            caller="legend()",
        )

    def _record_key(
        self,
        layer_id: str | None,
        *,
        title: str | None,
        position: str = DEFAULT_KEY_ANCHOR,
        labels: list | None,
        visible: bool,
        caller: str,
    ) -> Self:
        """Record a colour key on a layer, told which public spelling the caller reached it through.

        :meth:`legend` and :meth:`~digitalearth.web.base.WebMapBase.colorbar` are one mechanism under two
        names, so the body lives here and each name passes its own. Without that, the one refusal a
        `colorbar()` caller can actually hit — "nothing to describe" — named `legend()` and told them to add
        a `column=`, about a method they never called; this tier names the caller in every other refusal it
        gives (``WebMap.field()``, ``WebMap.extrusion()``).

        **One mechanism means one validation site.** Both public names being pure forwards onto this body is
        what lets `_check_key_arguments` be the single place a caller argument is refused from, so the two
        spellings cannot drift to two answers for the same malformed call (review H2).

        Args:
            layer_id: The caller's chosen layer, or `None` to resolve one.
            title: The heading, or `None` for the classified column's name.
            position: The corner, one of :data:`~digitalearth.base.controls.CONTROL_POSITIONS`.
            labels: Explicit row labels, or `None`.
            visible: Whether a key is drawn; `False` records that the colour is explained by nothing.
            caller: How to spell the public method in a refusal — ``"legend()"`` or ``"colorbar()"``.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: as :meth:`legend` documents.
            KeyError: as :meth:`legend` documents.
        """
        # EVERY caller argument is checked here, in one call, above the first read of `visible` — and that
        # ordering is pinned structurally, not case by case. It came back three times otherwise: the corner
        # (review L7), then the layer (review M1/M2), then `title` and `labels` (review H2), each fix
        # hoisting the one keyword that had just been caught and leaving the next one under the flag. The
        # *class* of defect is "a caller argument read below the flag", so there is now exactly one place a
        # caller argument may be read from — see `_check_key_arguments` — and
        # `tests/web/test_web_legend.py::TestNoArgumentCanBeAddedBelowTheFlag` reads this body's source to
        # refuse a fourth recurrence: a keyword added to either public spelling and not handed to the check
        # above this line reddens with no case written for it.
        self._check_key_arguments(
            layer_id, title=title, position=position, labels=labels
        )
        target = self._guide_target(layer_id, visible=bool(visible), caller=caller)
        if target is None:
            return self
        _require_maplibre()
        guide = Guide(show=bool(visible), title=title, anchor=position)
        # Checked a second time, now against the layer the record actually lands on: `visible=False`
        # resolves to the layer whose key is *drawn*, which need not be the one the check above could see,
        # and `labels` has to fit the classification it will be drawn beside. Built and thrown away, so a
        # refusal leaves the map exactly as it was rather than leaving behind a guide whose key every later
        # rebuild would fail to draw. Pinned by
        # `tests/web/test_web_legend.py::…::test_a_refusal_leaves_the_map_exactly_as_it_was`, so moving the
        # record above this line reddens rather than passing.
        _legend_panel(self._legend_of(target), guide, labels)
        self._attach_guide(target, guide, labels)
        # Derived, not written: the rule for which of several guides is drawn lives in one place, so this
        # call and a later removal cannot disagree about whose key is on screen.
        refresh_legend_panel(self)
        return self

    def _check_key_arguments(
        self,
        layer_id: str | None,
        *,
        title: str | None,
        position: str,
        labels: list | None,
    ) -> None:
        """Refuse a malformed key description, before the `visible` flag has been read at all.

        **The one place a caller argument to :meth:`legend` /
        :meth:`~digitalearth.web.base.WebMapBase.colorbar` may be checked from**, which is what makes the
        rule "no argument is valid only half the time" a property of the method rather than a property of
        the three keywords somebody remembered. The flag is deliberately not a parameter here: it cannot be
        consulted, so it cannot skip anything.

        The split is between two different questions. *Is this a well-formed description?* — the corner, the
        named layer, the heading, the row labels — is a fact about the call and is answered here, whatever
        the flag says. *Does this map have anything to describe?* is a fact about the map, and stays
        :meth:`_guide_target`'s ``visible=True``-only refusal, because the documented point of the flag is
        that ``WebMap().colorbar(visible=False)`` is an answer and not a crash.

        Args:
            layer_id: The caller's chosen layer, or `None`. When given it is looked up and required to
                carry a classification. When `None` the topmost keyable layer stands in, purely so
                `labels` can be counted against a real classification on a map nobody has keyed yet —
                the state the three previous fixes all left open.
            title: The heading, or `None`. Checked by building the
                :class:`~digitalearth.base.spec.encoding.Guide` that will hold it.
            position: The corner, checked against :data:`~digitalearth.base.controls.CONTROL_POSITIONS`.
            labels: Explicit row labels, or `None`. Its *shape* is checked unconditionally — a
                non-sequence is a ``TypeError`` here as it was where the panel used to build it — and its
                *length* whenever there is a classification to count against.

        Raises:
            ValueError: when `position` is not one of the four legal corners, when `layer_id` names a layer
                that carries no classification, when `title` is not a non-empty string, or when `labels`
                has a different number of entries from the classification.
            KeyError: when `layer_id` names no layer on this map.
            TypeError: when `labels` is not a sequence.
        """
        _check_position(position)
        if layer_id is not None:
            # KeyError by name, then ValueError for no classification — the same two refusals
            # `_guide_target` gives a named layer, reached here so neither waits on the flag.
            self._legend_of(layer_id)
            described: str | None = layer_id
        else:
            keyable = keyable_layer_ids(self)
            described = keyable[-1] if keyable else None
        guide = Guide(show=True, title=title, anchor=position)
        rows = None if labels is None else list(labels)
        if described is not None:
            _legend_panel(self._legend_of(described), guide, rows)

    def _guide_target(
        self, layer_id: str | None, *, visible: bool, caller: str = "legend()"
    ) -> str | None:
        """Return the layer a colour key should describe.

        Args:
            layer_id: The caller's choice, or `None` to resolve one from the map.
            visible: Which question a `None` `layer_id` asks. `True` — asking for a key — looks for a
                classified layer to put one on, and a map with none is an error. `False` — asking for no
                key — looks for the key the map already has, and a map with none answers `None` rather than
                raising, which is what keeps ``WebMap().colorbar(visible=False)`` from being a crash on an
                empty map.
            caller: How to spell the public method in the refusal — ``"legend()"`` or ``"colorbar()"``.
                This is the one refusal both spellings can reach, so naming the method that wrote it sent a
                `colorbar()` caller to fix a `legend()` call.

        Returns:
            The layer id, or `None` when nothing qualifies and nothing was required.

        Raises:
            KeyError: when `layer_id` names no layer on this map.
            ValueError: when `layer_id` names a layer that was not drawn with a classification, or when a
                key was asked for and no layer on the map is classified.

        Note:
            **Nothing a caller passed is validated here.** Every caller argument — the corner, the named
            layer, the heading and the row labels — is refused by `_check_key_arguments` before this is
            reached, so no check can be reached through a flag value (review H2). A named layer is looked
            up again below all the same, so this stays answerable on its own; what it must never become is
            the *only* place a caller argument is looked at, which is how `legend(layer_id="nope",
            visible=False)` came to be accepted while `legend(layer_id="nope")` raised. The "nothing on
            this map is classified" question is not a caller argument, so it stays a ``visible=True``-only
            refusal — the documented point of the flag is that it can be passed through without branching.

            **The two flag values resolve differently on purpose, and the flag is not a third refusal
            rule.** Asking for a key and taking one off are different questions of the map: one wants a
            layer to explain, the other wants the key that is already there. One resolution served both,
            `keyable[-1]`, so `colorbar(visible=False)` on a map where a lower layer was keyed wrote
            ``show=False`` onto the topmost *classified* layer — one nobody had keyed — and left the drawn
            key exactly where it was (review M3).

            **Visibility is a preference on both branches, never a refusal.** Asking for a key prefers the
            topmost **visible** keyable layer — one that published a colour scale, which since order 24
            includes a raster band's ramp and not only a classified column (review L3) — because an
            unnamed `colorbar()` means "key what is on the
            map": visibility was consulted nowhere, so a hidden topmost layer took the guide and the call
            drew nothing at all (review L6). Taking one off prefers the key that is drawn for the matching
            reason. Neither *requires* it — a key recorded on a hidden layer draws nothing but is still a
            key the caller asked for, so each branch falls back to the topmost candidate of any visibility:
            otherwise keying a map built hidden would raise, and taking a key off a hidden layer would be a
            no-op the next `set_visible` undid. A hidden layer named outright is keyed as it always was.
        """
        if layer_id is not None:
            self._legend_of(
                layer_id
            )  # KeyError by name, then ValueError for no classification
            return layer_id
        if not visible:
            drawn = drawn_guide_layer(self)
            if drawn is not None:
                return drawn
            asked = guided_layer_ids(self)
            return asked[-1] if asked else None
        keyed = keyable_layer_ids(self)
        shown = [
            candidate for candidate in keyed if self._layer_tree.is_visible(candidate)
        ]
        if shown:
            return shown[-1]
        if keyed:
            return keyed[-1]
        raise ValueError(
            f"{caller} has nothing to describe: no classified layer has been added yet. Add a "
            "choropleth (or any builder given column=...) first."
        )

    def _attach_guide(self, layer_id: str, guide: Guide, labels: list | None) -> None:
        """Record on one layer that its colour is explained, and how.

        Args:
            layer_id: The layer to write on. It carries a field-driven ``color`` encoding — `_guide_target`
                has already established that — which is what a guide can be attached to.
            guide: What to say about the channel.
            labels: Explicit row labels, or `None` to draw the derived ones. `None` **removes** a previous
                override rather than keeping it, so two `legend()` calls do not silently compound.

        Note:
            The tree is written directly rather than through :meth:`~digitalearth.web.base.WebMapBase._change`.
            A guide changes no paint, so there is nothing for the renderer to draw again; routing it through
            the reconcile would rebuild the MapLibre layer of every classified layer a caller keys.
            :meth:`~digitalearth.web.base.WebMapBase._index_layer` writes the tree the same way, and
            :meth:`~digitalearth.web.base.WebMapBase._figure_with` derives the figure from it, so the guide
            reaches `figure_spec` with no further step.
        """
        layer = self._layer_tree.get(layer_id)
        symbology = layer.symbology.with_guide(guide)
        props = {
            key: value for key, value in symbology.props.items() if key != LEGEND_LABELS
        }
        if labels is not None:
            props[LEGEND_LABELS] = tuple(labels)
        self._layer_tree = self._layer_tree.replace(
            _with_fields(
                layer,
                symbology=Symbology(encodings=dict(symbology.encodings), props=props),
            )
        )

    def layer_control(
        self,
        *,
        layers: list | None = None,
        position: str = "top-right",
        controls: list | None = None,
        theme: str = "default",
        opacity: dict | None = None,
        order: list | None = None,
    ) -> Self:
        """Add a switcher so a viewer can turn the data layers on and off, dim them, and reorder them.

        A map with a basemap, a choropleth and a point overlay had no way to look underneath — which is the
        single most common thing anyone does with a web map. The switch lists the data layers only:
        basemaps are the ground, not something a viewer toggles.

        The three keywords are the ones the Tier-2 contract declares and the interactive tier now answers to
        as well — the layers to include, the position, the controls to expose — so the same call adds a layer
        control on either tier (#264). What was ``layer_ids=`` here is ``layers=``, the name that tier uses
        for the same thing.

        **The manager widens past visibility (WB-8), and the widening is declarative.** ``opacity=`` dims a
        layer and ``order=`` reorders the stack, each applied **now** — ``opacity`` through
        :meth:`~digitalearth.web.base.WebMapBase.replace_layer`, which rewrites the layer's own paint, and
        ``order`` through :meth:`~digitalearth.web.base.WebMapBase.move_layer` — so the saved page draws the
        layers dimmed and in that order. The visibility switcher stays a **live page-side control**
        (py-maplibregl's ``LayerSwitcherControl``); a live opacity slider or a drag-to-reorder control would
        each need a browser-JS layer this tier does not ship (the X-2 limitation), so ``controls=`` still
        refuses ``"opacity"`` rather than accept a control it cannot build — the inertness #242/#244 removed.
        The two are therefore different questions: ``controls=`` selects the *live* switcher controls, while
        ``opacity=``/``order=`` are a one-time transform baked into the page and recorded on the switcher's
        furniture.

        Args:
            layers: The layers to offer, by id, defaulting to every data layer added so far — that is,
                :attr:`~digitalearth.web.base.WebMapBase.layer_ids` without the basemaps, which are
                addressable but are the ground rather than something a viewer toggles. Pass a subset to
                hide the rest from the switch without hiding them from the map, or name a basemap to offer
                it after all.
            position: One of the four corners in
                :data:`~digitalearth.base.controls.CONTROL_POSITIONS` — MapLibre's, and now both tiers'.
            controls: Which controls to expose, from
                :data:`~digitalearth.base.controls.LAYER_CONTROLS`. ``None`` (the default) offers everything
                this tier can build, which is ``("visibility",)``: a py-maplibregl switcher is visibility
                rows and nothing else. Naming ``"opacity"`` or ``"basemap"`` is therefore **refused** rather
                than accepted and dropped — the interactive tier builds both, and a caller moving a call
                here should hear that this tier cannot. What the shared resolver answers is recorded on the
                switcher's furniture, so the panel says which controls it draws rather than leaving a reader
                to know that this tier draws one.
            theme: ``"default"`` or ``"simple"`` — py-maplibregl's two switcher styles. This tier's own
                keyword: it styles the switcher rather than choosing what the switcher contains.
            opacity: A mapping of layer id to opacity in ``[0, 1]``, applied **now** by rewriting each named
                layer's paint through :meth:`~digitalearth.web.base.WebMapBase.replace_layer`. A layer that
                records no opacity to change — a text annotation has none — is refused rather than silently
                left opaque. ``None`` dims nothing.
            order: The bottom-to-top draw order for the named layers, applied **now** through
                :meth:`~digitalearth.web.base.WebMapBase.move_layer`. The ids must share a draw-order band —
                a layer cannot be dragged under the basemap, which is the one move the bands forbid — and an
                order that crosses a band is refused with the ``IndexError`` ``move_layer`` raises. ``None``
                leaves the order untouched.

        Note:
            A row toggles exactly one MapLibre layer, because that is what py-maplibregl's control does.
            Builders that draw more than one layer — ``cluster`` (bubbles, counts, loose points) and
            ``graticule`` (lines, degree labels) — are listed once, under their main layer, so toggling a
            cluster hides its bubbles while the count labels remain. ``remove_layer`` does take the whole
            group. Hiding the parts together needs a control py-maplibregl does not ship.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal corners, when ``controls`` names
                something outside the shared vocabulary or something this tier cannot build, when no data
                layer has been added yet, when an id was given (in ``layers``, ``opacity`` or ``order``)
                that is not on this map, when an ``opacity`` value is not a finite fraction in ``[0, 1]``,
                or when an ``opacity`` names a layer that records no opacity to change.
            IndexError: when ``order`` would move a layer out of its draw-order band, as
                :meth:`~digitalearth.web.base.WebMapBase.move_layer` refuses it.

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
            digitalearth.base.controls.resolved_controls: the shared control vocabulary.
        """
        _require_maplibre()
        _check_position(position)
        # The answer is kept, not only the refusal: `resolved_controls` is the one place that knows what this
        # tier can build, and it de-duplicates and orders the names. Recording what it returned is what makes
        # the panel's description and the refusal read one list (review R2-L13). The default goes through it
        # too, so "everything this tier can build" is resolved rather than re-spelled here.
        exposed = resolved_controls(
            _OFFERED_CONTROLS if controls is None else controls,
            offered=_OFFERED_CONTROLS,
            caller="WebMap.layer_control()",
        )
        if theme not in _SWITCHER_THEMES:
            raise ValueError(
                f"layer_control(theme={theme!r}) must be one of {sorted(_SWITCHER_THEMES)}"
            )
        available = self.layer_ids
        # The default offers the *data* layers, not every id: a basemap is described and addressable like
        # any other layer since it started recording one, and offering it here would put "turn the ground
        # off" at the top of every switcher. A caller who means to offer it still can, by naming it in
        # `layers=` — which is why the filter is on the default rather than on what is accepted.
        offered = [
            layer_id
            for layer_id in available
            if self._layer_tree.get(layer_id).kind != "basemap"
        ]
        if not offered:
            raise ValueError(
                "layer_control() has nothing to switch: no data layer has been added yet. A basemap is "
                "the ground rather than a layer a viewer toggles."
            )
        wanted = list(layers) if layers is not None else offered
        named = [*wanted, *(opacity or {}), *(order or [])]
        unknown = [layer_id for layer_id in named if layer_id not in available]
        if unknown:
            raise ValueError(
                f"layer_control() was given {unknown}, which are not on this map; its layers are "
                f"{available}"
            )
        # The declarative half (WB-8): dim, then reorder, through the map's own `replace_layer`/`move_layer`
        # rather than a new mechanism, so the saved page draws what the layers now carry. The WHOLE request is
        # validated before any layer is touched — mirroring `_bind_attribute`'s whole-list pre-check (M2):
        # `_resolve_layer_opacity` rewrites every target's paint (raising on a bad value or a layer with no
        # opacity to dim) and `_check_layer_order` proves the full order a legal in-band permutation on a trial
        # tree, both without mutating. Only then are they applied — opacity before order because `replace_layer`
        # keeps a layer's place, so the two do not fight — and both before the switcher is recorded. So a refusal
        # leaves no half-built control behind: no opacity applied, no move done, no furniture recorded (M1).
        resolved_opacity = self._resolve_layer_opacity(opacity)
        self._check_layer_order(order)
        applied_opacity = self._apply_layer_opacity(resolved_opacity)
        applied_order = self._apply_layer_order(order)
        # Held as a request, not appended as a layer: the live layers are resolved when the widget is
        # built, so removing a layer afterwards cannot leave a dead row in the saved page, and calling
        # this twice replaces the request rather than stacking a second identical panel.
        self._switcher = {
            "layer_ids": wanted,
            "theme": theme,
            "position": position,
        }
        self._record_furniture(
            "layer_switcher",
            anchor=position,
            layers=tuple(wanted),
            theme=theme,
            controls=exposed,
            opacity=applied_opacity,
            order=applied_order,
        )
        return self

    def _resolve_layer_opacity(self, opacity: dict | None) -> tuple:
        """Validate the whole opacity request and build each layer's rewritten spec, without mutating the map.

        The web tier builds no live opacity slider (py-maplibregl's switcher is visibility rows), so opacity
        is applied declaratively: the value is written into the layer's own recorded paint — a vector layer's
        ``*-opacity`` paint key, or a raster's flat ``opacity`` prop. This resolves every target up front — so
        a bad value or a layer with no opacity to change is refused before :meth:`_apply_layer_opacity` touches
        a single layer, which is what keeps :meth:`layer_control` atomic (M1).

        Args:
            opacity: A mapping of layer id to opacity in ``[0, 1]``, or ``None`` to dim nothing. Every id has
                already been checked to be on the map.

        Returns:
            One ``(layer_id, value, spec)`` triple per target, in the order given — ``spec`` being the rewritten
            :class:`~digitalearth.base.spec.LayerSpec` for :meth:`_apply_layer_opacity` to install. Empty when
            nothing was asked to be dimmed.

        Raises:
            ValueError: when a value is not a finite fraction in ``[0, 1]``, or when a named layer records no
                opacity to change (a text annotation has none).
        """
        if not opacity:
            return ()
        resolved = []
        for layer_id, value in opacity.items():
            opacity_value = as_finite(
                value, f"opacity[{layer_id!r}]", "WebMap.layer_control()"
            )
            if not 0.0 <= opacity_value <= 1.0:
                raise ValueError(
                    f"WebMap.layer_control() opacity[{layer_id!r}]={opacity_value} must be a fraction "
                    "in [0, 1]"
                )
            spec = self._layer_with_opacity(layer_id, opacity_value)
            resolved.append((layer_id, opacity_value, spec))
        return tuple(resolved)

    def _apply_layer_opacity(self, resolved: tuple) -> tuple:
        """Install the rewritten specs from :meth:`_resolve_layer_opacity` through ``replace_layer``.

        The request was fully validated by :meth:`_resolve_layer_opacity`, so nothing here raises — each spec
        is handed to :meth:`~digitalearth.web.base.WebMapBase.replace_layer`, which rebuilds the MapLibre layer
        from it so the saved page draws the layer dimmed.

        Args:
            resolved: The ``(layer_id, value, spec)`` triples :meth:`_resolve_layer_opacity` returned.

        Returns:
            The applied pairs as a tuple of ``(layer_id, value)``, sorted by id, for the switcher furniture
            to record what the control did. Empty when nothing was dimmed.
        """
        for _, _, spec in resolved:
            self.replace_layer(spec)
        return tuple(sorted((layer_id, value) for layer_id, value, _ in resolved))

    def _layer_with_opacity(self, layer_id: str, value: float) -> LayerSpec:
        """Return a copy of a layer's description with its opacity set to ``value``.

        Args:
            layer_id: The layer to dim.
            value: The opacity in ``[0, 1]``.

        Returns:
            A new :class:`~digitalearth.base.spec.LayerSpec` with the same id, kind, source and encodings,
            and the opacity rewritten — in the ``*-opacity`` key of its paint dict (a vector layer) or in
            its flat ``opacity`` prop (a raster / field / graticule).

        Raises:
            ValueError: when the layer records no opacity to change — a text annotation carries none, so
                dimming it is refused rather than accepted and ignored.
        """
        layer = self.get_layer(layer_id)
        props = dict(layer.symbology.props)
        paint = props.get("paint")
        if isinstance(paint, dict) and any(key.endswith("-opacity") for key in paint):
            props["paint"] = {
                key: (value if key.endswith("-opacity") else held)
                for key, held in paint.items()
            }
        elif "opacity" in props:
            props["opacity"] = value
        else:
            raise ValueError(
                f"WebMap.layer_control(opacity=...) cannot dim layer {layer_id!r} (kind {layer.kind!r}): "
                "it records no opacity to change. Only the data layers that carry a fill/line/circle/raster "
                "opacity can be dimmed."
            )
        symbology = Symbology(encodings=dict(layer.symbology.encodings), props=props)
        return _with_fields(layer, symbology=symbology)

    def _check_layer_order(self, order: list | None) -> None:
        """Prove the full order is a legal in-band permutation, replaying the moves on a trial tree only.

        :meth:`_apply_layer_order` moves each listed layer through the map's own ``move_layer``, and a move
        that crosses a draw-order band raises an ``IndexError`` partway — leaving the stack half-reordered. So
        the moves are first replayed against a throwaway copy of the layer tree, in the exact sequence (and
        from the same anchor) :meth:`_apply_layer_order` uses, so a band-crossing order is refused before any
        layer actually moves and :meth:`layer_control` stays atomic (M1).

        Args:
            order: The bottom-to-top order for the named layers, or ``None`` to leave the order untouched.
                Every id has already been checked to be on the map.

        Raises:
            IndexError: when a move would take a layer out of its draw-order band, as
                :class:`~digitalearth.base.spec.LayerTree` refuses it — the same error
                :meth:`_apply_layer_order` would raise, surfaced before anything moves.
        """
        if not order:
            return
        tree = self._layer_tree
        anchor = min(self.layer_ids.index(layer_id) for layer_id in order)
        for offset, layer_id in enumerate(order):
            tree = tree.move(layer_id, anchor + offset)

    def _apply_layer_order(self, order: list | None) -> tuple:
        """Reorder the named layers bottom-to-top, through the map's own ``move_layer``.

        The web tier builds no drag-to-reorder control, so ordering is applied declaratively: each listed
        layer is moved to its place in draw order with
        :meth:`~digitalearth.web.base.WebMapBase.move_layer`. Anchored at the lowest position the listed
        layers currently occupy and filled upward, so the listed set ends up contiguous in the given order.
        The order was already proven a legal in-band permutation by :meth:`_check_layer_order`, so no move
        here crosses a band — the atomicity pre-check is what makes that true.

        Args:
            order: The bottom-to-top order for the named layers, or ``None`` to leave the order untouched.
                Every id has already been checked to be on the map, and the permutation proven legal.

        Returns:
            The applied order as a tuple of ids, for the switcher furniture to record. Empty when the order
            was untouched.
        """
        if not order:
            return ()
        anchor = min(self.layer_ids.index(layer_id) for layer_id in order)
        for offset, layer_id in enumerate(order):
            self.move_layer(layer_id, anchor + offset)
        return tuple(order)

    def text(
        self,
        lon: float,
        lat: float,
        s: str | None = None,
        *,
        crs: Any = 4326,
        text_size: float = 14.0,
        color: str = "#ffffff",
        halo_color: str = "#000000",
        halo_width: float = 1.0,
        name: str | None = None,
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
            text_size: Text size in pixels. Named for the text rather than ``size``, which means the
                visual size of a marker everywhere else.
            color: Text colour.
            halo_color: Colour of the outline behind the glyphs, which keeps it legible over imagery.
            halo_width: Halo width in pixels; ``0`` disables it.
            name: What a layer switcher calls this annotation; ``None`` uses its generated id.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when the string to draw is not given.
            ValueError: when ``lon``, ``lat``, ``text_size`` or ``halo_width`` is not a finite number —
                refused at this call, because a figure holding NaN or infinity could not be written
                down.

        Examples:
            - Mark a place:
                ```python
                >>> from digitalearth.web import WebMap                     # doctest: +SKIP
                >>> WebMap().basemap().text(4.9, 52.4, "Amsterdam")         # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.labels: label many features from a column.
        """
        #: This builder's own name, for the refusals below to quote back at the caller.
        call = "WebMap.text()"
        if s is None:
            raise TypeError(
                "text() needs the string to draw; pass it as the third argument"
            )
        lon, lat = self._as_display_point(
            as_finite(lon, "lon", call),
            as_finite(lat, "lat", call),
            crs,
        )
        _require_layer_api()
        text_size = as_finite(text_size, "text_size", call)
        halo_width = as_finite(halo_width, "halo_width", call)
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
        subtitle: str | None = None,
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
                >>> WebMap().basemap().set_title("Population, 2024")                   # doctest: +SKIP

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
        lon_step: float | None = None,
        lat_step: float | None = None,
        spacing: float | None = None,
        color: str = "#888888",
        width: float = 0.5,
        opacity: Maybe[float] = UNSET,
        labels: bool = True,
        name: str | None = None,
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
            labels: Whether to label each line with its degree value at the map's edge. It also decides
                whether the ``"<id>-label"`` id its drawer would derive is reserved: a graticule without
                labels derives none, so a later layer may take that name (review L2).
            name: What a layer switcher calls this layer; ``None`` names it ``"Graticule"``, suffixed if
                that is taken.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``lon_step``, ``lat_step``, ``width`` or ``opacity`` is not a finite number —
                checked first, because a figure holding NaN or infinity could not be written down — or
                when a step is not positive, or is wider than the 180° of latitude there is to divide,
                either of which produces a grid with no lines and no hint as to why.

        Examples:
            - A 10° grid under the data:
                ```python
                >>> from digitalearth.web import WebMap        # doctest: +SKIP
                >>> WebMap().basemap().graticule()             # doctest: +SKIP

                ```
        """
        #: This builder's own name, for the refusals below to quote back at the caller.
        call = "WebMap.graticule()"
        _require_layer_api()
        # One step for both is what `spacing=` meant, and it is still the shortest way to ask for a square
        # grid; the two steps are what every other tier takes, and what a reader of a world map usually wants
        # (#263). A tier default of 30 degrees matches the interactive tier's.
        lon_step = _DEFAULT_GRID_STEP if lon_step is None else lon_step
        lat_step = _DEFAULT_GRID_STEP if lat_step is None else lat_step
        if spacing is not None:
            lon_step = lat_step = spacing
        # Finite first: a NaN step passes the range check below (every comparison against NaN is false) and
        # was written into the description, where it made the whole figure unwritable (review L1).
        lon_step = as_finite(lon_step, "lon_step", call)
        lat_step = as_finite(lat_step, "lat_step", call)
        width = as_finite(width, "width", call)
        ask = Ask()
        opacity = as_finite(ask("opacity", opacity, GRATICULE_OPACITY), "opacity", call)
        for name_of, step in (("lon_step", lon_step), ("lat_step", lat_step)):
            if step <= 0 or step > 180:
                # 180 is the widest meaningful step: it still yields the prime meridian and the antimeridian,
                # while anything wider leaves a grid with a single line in it.
                raise ValueError(
                    f"graticule({name_of}={step!r}) must be greater than 0 and at most 180 degrees"
                )
        # The degree labels are drawn under an id derived from this one, which the allocation reserves too,
        # so a caller's `name=` can no longer take it (review M5). Only when they are drawn, though: the
        # reservation table is keyed by kind, and whether a graticule labels itself is in its description.
        # Reserving unconditionally held an id nothing drew, and the caller who then asked for that name got
        # it silently suffixed — which is the string a layer switcher captions the row with (review L2).
        layer_id = self._layer_id(
            "graticule", name or "Graticule", kind="graticule" if labels else None
        )
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
                    **ask.record,
                }
            ),
        )
        # Reference geography says nothing about where to look, so it does not frame the map. Its band —
        # over the basemap, under the data — comes from the kind's registration, not from here.
        return self

    def coastlines(
        self,
        resolution: str = "110m",
        *,
        color: str = "#000000",
        width: float = 0.8,
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Self:
        """Overlay Natural-Earth coastlines as reference lines (WB-16).

        Coastlines came only with whatever basemap style a caller picked; a map built from the caller's own
        sources, or one with ``basemap(opacity=...)`` turned down, had none. This draws them as a line
        overlay in the reference band — over the basemap, under the data — from the same
        ``cleopatra.basemap.reference`` Natural-Earth coordinates the static tier reads, built here into
        ordinary GeoJSON so the lines embed in a saved page and need no network. No GIS is reimplemented:
        the coordinates are read from the shared source and packaged, nothing more.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``, matching
                the static tier's spelling.
            color: Line colour.
            width: Line width in pixels.
            opacity: Line opacity in ``[0, 1]``; reference geography sits visually under the data.
            visible: Whether the overlay starts drawn.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``resolution`` is not one of ``"110m"``, ``"50m"`` or ``"10m"``, or when
                ``width`` or ``opacity`` is not a finite number.

        Examples:
            - Coastlines over a basemap, under the data (needs the ``web`` extra, so the block is skipped
              without it):
                ```python
                >>> from digitalearth.web import WebMap        # doctest: +SKIP
                >>> WebMap().basemap().coastlines()            # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.borders: the same line path for country boundaries.
            digitalearth.web.base.WebMapBase.add_reference: the band that keeps it under the data.
        """
        return self._reference_lines(
            "coastline",
            "coastlines",
            resolution,
            color=color,
            width=width,
            opacity=opacity,
            visible=visible,
        )

    def borders(
        self,
        resolution: str = "110m",
        *,
        color: str = "#777777",
        width: float = 0.6,
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Self:
        """Overlay Natural-Earth country borders as reference lines (WB-16).

        The same line path as :meth:`coastlines`, for country boundaries rather than the shoreline.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            color: Line colour.
            width: Line width in pixels.
            opacity: Line opacity in ``[0, 1]``.
            visible: Whether the overlay starts drawn.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``resolution`` is not one of ``"110m"``, ``"50m"`` or ``"10m"``, or when
                ``width`` or ``opacity`` is not a finite number.

        See Also:
            digitalearth.web.decoration.DecorationMixin.coastlines: the shoreline counterpart.
        """
        return self._reference_lines(
            "borders",
            "borders",
            resolution,
            color=color,
            width=width,
            opacity=opacity,
            visible=visible,
        )

    def _reference_lines(
        self,
        source_layer: str,
        prefix: str,
        resolution: str,
        *,
        color: str,
        width: float,
        opacity: float,
        visible: bool,
    ) -> Self:
        """Add one Natural-Earth line layer — a coastline or a border — to the reference band.

        :meth:`coastlines` and :meth:`borders` are one mechanism over two Natural-Earth datasets, so the
        body lives here. The geometry is read from ``cleopatra.basemap.reference`` as lon/lat coordinate
        arrays and packaged into GeoJSON; the MapLibre source and line layer are added through a queued
        closure (the shape :meth:`~digitalearth.web.base.WebMapBase.add_reference` takes), so a figure
        written down keeps the embedded lines even offline.

        Args:
            source_layer: The Natural-Earth dataset name cleopatra takes — ``"coastline"`` or ``"borders"``.
            prefix: The id prefix the overlay is numbered under — ``"coastlines"`` or ``"borders"``.
            resolution: One of ``"110m"``, ``"50m"`` or ``"10m"``.
            color: Line colour.
            width: Line width in pixels.
            opacity: Line opacity in ``[0, 1]``.
            visible: Whether the overlay starts drawn.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: for an unknown resolution, or a non-finite width/opacity.
        """
        call = f"WebMap.{prefix}()"
        if resolution not in _NATURAL_EARTH_RESOLUTIONS:
            raise ValueError(
                f"{call} resolution={resolution!r} must be one of "
                f"{sorted(_NATURAL_EARTH_RESOLUTIONS)} — the resolutions Natural Earth publishes"
            )
        width = as_finite(width, "width", call)
        opacity = as_finite(opacity, "opacity", call)
        layer_cls, layer_types = _require_layer_api()
        from cleopatra.basemap.reference import natural_earth

        # Read the reference geometry as lon/lat arrays and package it as GeoJSON. Parts with fewer than two
        # vertices cannot be a line, so they are dropped rather than drawn as a degenerate segment.
        parts = natural_earth(source_layer, resolution)
        features = [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(x), float(y)] for x, y in part],
                },
            }
            for part in parts
            if len(part) >= 2
        ]
        collection = {"type": "FeatureCollection", "features": features}
        layer_id = self._layer_id(prefix, None)
        source_id = f"{layer_id}-src"
        layout = None if visible else {"visibility": "none"}

        def apply(widget: Any) -> None:
            widget.add_source(source_id, {"type": "geojson", "data": collection})
            widget.add_layer(
                layer_cls(
                    id=layer_id,
                    type=layer_types.LINE,
                    source=source_id,
                    paint={
                        "line-color": color,
                        "line-width": float(width),
                        "line-opacity": float(opacity),
                    },
                    layout=layout,
                )
            )

        apply._digitalearth_layer_id = layer_id  # type: ignore[attr-defined]
        return self.add_reference(apply)

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

    def attribution(
        self,
        *,
        position: str = "bottom-right",
        compact: bool = False,
        custom: str | None = None,
    ) -> Self:
        """Add a MapLibre attribution control — the data and tile credits (WB-14).

        A basemap carries its own attribution, but a map built from the caller's own sources has none until
        one is added. This is the control that shows it, and it sits bottom-right as every MapLibre map's
        does.

        Args:
            position: One of the four MapLibre corners.
            compact: Collapse the credits behind an ``ⓘ`` button rather than showing them inline, for a
                small map where the full line would crowd the frame.
            custom: Extra attribution text to show beside whatever the sources declare; ``None`` shows only
                the sources' own.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners.
        """
        _require_maplibre()
        _check_position(position)
        from maplibre.controls import AttributionControl

        settings: dict = {"compact": bool(compact)}
        if custom is not None:
            settings["custom_attribution"] = custom
        control = AttributionControl(**settings)

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        recorded: dict = {"compact": bool(compact)}
        if custom is not None:
            recorded["custom"] = custom
        self._record_furniture("attribution", anchor=position, **recorded)
        return self._queue(apply)

    def geolocate(
        self,
        *,
        position: str = "top-right",
        track: bool = False,
        show_accuracy_circle: bool = True,
        show_heading: bool = False,
    ) -> Self:
        """Add a MapLibre geolocate control — a button that centres the map on the viewer (WB-14).

        Args:
            position: One of the four MapLibre corners.
            track: Keep re-centring on the viewer as they move, rather than locating them once.
            show_accuracy_circle: Draw the circle of positional uncertainty around the located point.
            show_heading: Show which way the viewer's device is facing.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners.
        """
        _require_maplibre()
        _check_position(position)
        from maplibre.controls import GeolocateControl

        control = GeolocateControl(
            track_user_location=bool(track),
            show_accuracy_circle=bool(show_accuracy_circle),
            show_user_heading=bool(show_heading),
        )

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        self._record_furniture(
            "geolocate",
            anchor=position,
            track=bool(track),
            show_accuracy_circle=bool(show_accuracy_circle),
            show_heading=bool(show_heading),
        )
        return self._queue(apply)

    def globe_control(self, *, position: str = "top-right") -> Self:
        """Add a MapLibre globe control — a toggle between the flat map and a 3-D globe (WB-14).

        Args:
            position: One of the four MapLibre corners.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners.
        """
        _require_maplibre()
        _check_position(position)
        from maplibre.controls import GlobeControl

        control = GlobeControl()

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        self._record_furniture("globe_control", anchor=position)
        return self._queue(apply)

    def terrain_control(
        self,
        source: str,
        *,
        exaggeration: float = 1.0,
        position: str = "top-right",
    ) -> Self:
        """Add a MapLibre terrain control — a toggle for 3-D terrain from a DEM source (WB-14).

        The control drives a raster-DEM source that is already on the map (for example the one
        :meth:`~digitalearth.web.threed.ThreeDMixin.terrain_tiles` adds); it is named here by its MapLibre
        source id, the way the control itself references it, so whether the source exists is the caller's
        own responsibility — exactly as it is in MapLibre.

        Args:
            source: The MapLibre id of the raster-DEM source the terrain is built from.
            exaggeration: The vertical exaggeration applied when terrain is switched on.
            position: One of the four MapLibre corners.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``source`` is not a non-empty string — the control would drive nothing — when
                ``exaggeration`` is not a finite number (NaN/inf, refused at the call because a figure holding
                it could not be written down), or when ``position`` is not one of the four legal MapLibre
                corners.
        """
        _require_maplibre()
        _check_position(position)
        if not isinstance(source, str) or not source:
            raise ValueError(
                "terrain_control() needs source= as the MapLibre id of a raster-DEM source to drive; the "
                "control toggles that source's terrain and has nothing to toggle without one"
            )
        # Run `exaggeration` through `as_finite`, as every sibling builder does its numeric kwargs: a NaN/inf
        # value is refused here, named for what the caller wrote, rather than recorded and left to fail late
        # as a figure-serializer `TypeError` when the furniture is written down (L2).
        exaggeration = as_finite(
            exaggeration, "exaggeration", "WebMap.terrain_control()"
        )
        from maplibre.controls import TerrainControl

        control = TerrainControl(source=source, exaggeration=exaggeration)

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        self._record_furniture(
            "terrain_control",
            anchor=position,
            source=source,
            exaggeration=exaggeration,
        )
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
        scoped to line and/or polygon geometries, so the user draws the shape to measure. Once the map is shown,
        :meth:`drawn_features` returns what was drawn as a pyramids ``FeatureCollection`` (the widget's own
        ``draw_features_created``/``…_updated`` events stay available for a caller who wants to observe them).
        Computing the numeric distance/area from that geometry is a **GIS** operation — do it in pyramids
        (geodesic length / area), keeping this tier to the (visualization) drawing control.

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

    def geocoder(
        self,
        api_key: str,
        *,
        position: str = "top-left",
        placeholder: str | None = None,
        language: str | None = None,
        country: str | None = None,
        limit: int | None = None,
        fly_to: bool = True,
    ) -> Self:
        """Add a place-search box backed by MapTiler's geocoding service (WB-15).

        py-maplibregl ships ``MapTilerGeocodingControl`` but the tier exposed no way to add it. The service
        is keyed, and **Digital-Earth ships no key**: the caller supplies their own MapTiler key here, which
        is held only on the live control and is deliberately **not** recorded in the figure's description —
        the same rule ST-24 follows for a basemap credential, so a saved page cannot leak the key. A figure
        read back carries the geocoder as furniture without a key, and so draws its map without the search
        box, rather than embedding the secret.

        Args:
            api_key: The caller's MapTiler API key. Pass your own, for example from the environment
                (``geocoder(os.environ["MAPTILER_KEY"])``); it is never hardcoded or defaulted here.
            position: One of the four MapLibre corners for the search box.
            placeholder: The box's placeholder text; ``None`` leaves py-maplibregl's own.
            language: A language code for the result labels; ``None`` leaves the service default.
            country: An ISO country code to bias results toward; ``None`` searches everywhere.
            limit: The maximum number of suggestions to list; ``None`` leaves the service default.
            fly_to: Whether picking a result flies the map to it.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``api_key`` is not a non-empty string — the service could not authenticate
                without one, so an empty or non-string key is refused at the call rather than failing in a
                browser — or when ``position`` is not one of the four legal MapLibre corners.

        Examples:
            - Add a search box in the top-left corner (needs the ``web`` extra, so the block is skipped
              without it):
                ```python
                >>> from digitalearth.web import WebMap                     # doctest: +SKIP
                >>> WebMap().basemap().geocoder("YOUR-MAPTILER-KEY")        # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.navigation: the control-wiring pattern this follows.
        """
        _require_maplibre()
        _check_position(position)
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(
                "geocoder() needs api_key as a non-empty MapTiler API key; the service is keyed and "
                "Digital-Earth ships none — pass your own, e.g. geocoder(os.environ['MAPTILER_KEY'])"
            )
        from maplibre.controls import MapTilerGeocodingControl

        # The description half: non-secret layout options only. The key is left out on purpose so that
        # `figure_spec` — which a saved page serialises — never carries it (ST-24's rule for a credential).
        recorded: dict = {"fly_to": bool(fly_to)}
        if placeholder is not None:
            recorded["placeholder"] = placeholder
        if language is not None:
            recorded["language"] = language
        if country is not None:
            recorded["country"] = country
        if limit is not None:
            recorded["limit"] = int(limit)
        # The live half: the real control, carrying the key, added through a queued closure so the secret
        # stays out of the serialisable figure and only ever reaches the widget the caller renders.
        control = MapTilerGeocodingControl(api_key=api_key, **recorded)

        def apply(widget: Any) -> None:
            widget.add_control(control, position)

        self._record_furniture("geocoder", anchor=position, **recorded)
        return self._queue(apply)

    def drawn_features(self, widget: Any = None) -> Any:
        """Return the shapes drawn on the map, as a pyramids ``FeatureCollection`` in EPSG:4326.

        The draw control :meth:`measure` adds runs in the browser, and py-maplibregl syncs everything it holds
        back to the widget's ``draw_feature_collection_all`` trait on every create, update and delete. This
        reads that trait — by default from the widget :meth:`render` built last, which is the one ``show()``
        or a notebook cell put on screen — and hands it back as the tier's vector type, ready for a pyramids
        clip, mask or zonal statistic. MapboxDraw always reports lon/lat, so the result is in EPSG:4326.

        Each row keeps MapboxDraw's feature ``id`` in an ``id`` column, which is what the widget's
        ``draw_features_created``/``…_updated``/``…_deleted`` events name, so a row can be matched to them.
        A feature carrying an ``id`` **property** of its own — one set through ``draw.setFeatureProperty``,
        or imported with the shape — does not displace it: the draw id wins wherever there is one. Where
        there is none (``id`` absent or ``None`` beside the geometry, which MapboxDraw never leaves but a
        caller-supplied widget or a trait set from Python can) the feature's own property stands, and a
        feature with no ``id`` of either kind reads ``None``. The column is there on the empty collection
        too, so ``drawn["id"]`` reads the same whether anything was drawn.

        Args:
            widget: The ``MapWidget`` to read, for a caller holding one from an earlier :meth:`render`.
                ``None`` reads the last widget this map rendered.

        Returns:
            A ``FeatureCollection`` with one row per drawn shape — empty (no rows, still EPSG:4326, still
            carrying ``id``) when nothing is drawn or everything drawn was deleted.

        Raises:
            RuntimeError: when ``widget`` is ``None`` and the map has never been rendered, so no widget
                exists for anyone to have drawn on.
            TypeError: when ``widget`` has no ``draw_feature_collection_all`` trait (for example the
                time-slider composite :meth:`render` returns for a temporal map — pass nothing instead).

        Examples:
            - Stand in for the browser by setting the synced trait the way the front-end does after a user
              draws one triangle:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().measure()
                >>> widget = m.render()
                >>> widget.draw_feature_collection_all = {"type": "FeatureCollection", "features": [{
                ...     "id": "a1", "type": "Feature", "properties": {},
                ...     "geometry": {"type": "Polygon",
                ...                  "coordinates": [[[10, 50], [12, 50], [11, 52], [10, 50]]]}}]}
                >>> drawn = m.drawn_features()
                >>> len(drawn), drawn.epsg, list(drawn["id"])
                (1, 4326, ['a1'])
                >>> tuple(float(v) for v in drawn.total_bounds)
                (10.0, 50.0, 12.0, 52.0)

                ```
            - A map nobody drew on gives an empty collection, not an error — and one carrying the same
              ``id`` column, so the matching code above needs no ``len()`` branch in front of it:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().measure()
                >>> _ = m.render()
                >>> empty = m.drawn_features()
                >>> len(empty), list(empty["id"])
                (0, [])

                ```
            - A shape that carries an ``id`` property of its own keeps its other properties and still
              reports the draw id, because the draw id wins wherever there is one:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().measure()
                >>> widget = m.render()
                >>> widget.draw_feature_collection_all = {"type": "FeatureCollection", "features": [{
                ...     "id": "draw-1", "type": "Feature",
                ...     "properties": {"id": "mine", "label": "plot A"},
                ...     "geometry": {"type": "Point", "coordinates": [8.0, 51.0]}}]}
                >>> drawn = m.drawn_features()
                >>> list(drawn["id"]), list(drawn["label"])
                (['draw-1'], ['plot A'])

                ```
            - The same shape with **no** draw id keeps its own ``id`` instead of losing it to ``None``:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().measure()
                >>> widget = m.render()
                >>> widget.draw_feature_collection_all = {"type": "FeatureCollection", "features": [{
                ...     "type": "Feature",
                ...     "properties": {"id": "mine", "label": "plot A"},
                ...     "geometry": {"type": "Point", "coordinates": [8.0, 51.0]}}]}
                >>> drawn = m.drawn_features()
                >>> list(drawn["id"]), list(drawn["label"])
                (['mine'], ['plot A'])

                ```

        See Also:
            measure: adds the draw control this reads from.
            digitalearth.web.base.WebMapBase.render: builds the widget that is read by default.
        """
        if widget is None:
            widget = self._widget
            if widget is None:
                raise RuntimeError(
                    "drawn_features() reads the widget the map was drawn on, and this map has not been "
                    "rendered yet; call render() or show() (or display the map in a notebook) first"
                )
        if not hasattr(widget, "draw_feature_collection_all"):
            raise TypeError(
                f"drawn_features() needs a maplibre MapWidget carrying the draw_feature_collection_all trait; "
                f"got {type(widget).__name__}"
            )
        from pyramids.feature import FeatureCollection

        collection = widget.draw_feature_collection_all or {}
        features = [
            {**feature, "properties": self._drawn_properties(feature)}
            for feature in collection.get("features") or []
        ]
        if not features:
            # Declared with the `id` column rather than built as a bare geometry list: the documented
            # `drawn["id"]` otherwise raised `KeyError` precisely on the "nobody drew anything" path this
            # presents as the safe one, so every caller had to branch on `len()` first (review M5).
            return FeatureCollection({"id": []}, geometry=[], crs=_DRAWN_CRS)
        return FeatureCollection.from_features(features, crs=_DRAWN_CRS)

    @staticmethod
    def _drawn_properties(feature: dict) -> dict:
        """Return one drawn feature's properties with its ``id`` column settled.

        Three readings have to hold at once, and the first two pulled against each other:

        - A feature carrying an ``id`` **property** of its own — set through ``draw.setFeatureProperty``,
          or imported with the shape — must not displace MapboxDraw's draw id, because the draw id is what
          the create/update/delete events name (review M4). So the draw id is written over it.
        - It must not be displaced *by nothing*. The write was unconditional, so a feature with no draw id
          had its own property replaced with ``None`` — measured ``id column: [None]`` for properties
          ``{"id": "mine", "label": "plot A"}`` — which the property survived before the M4 fix. That is a
          regression, so the draw id is written only when there is one (review L4).
        - The column must exist whatever the feature holds, so ``drawn["id"]`` never raises on a path
          presented as the safe one (review M5). Dropping the key altogether would have taken that back,
          so a feature with no id of either kind gets an explicit ``None``.

        Args:
            feature: One GeoJSON feature as the widget's ``draw_feature_collection_all`` holds it.

        Returns:
            Its ``properties``, copied, carrying an ``id`` key: the draw id when there is one, else the
            feature's own ``id`` property, else ``None``.

        Examples:
            - The draw id wins over a property of the same name, and the other properties come through
              untouched:
                ```python
                >>> from digitalearth.web.decoration import DecorationMixin
                >>> DecorationMixin._drawn_properties({
                ...     "id": "draw-1",
                ...     "properties": {"id": "mine", "label": "plot A"},
                ...     "geometry": {"type": "Point", "coordinates": [8.0, 51.0]},
                ... })
                {'id': 'draw-1', 'label': 'plot A'}

                ```
            - With no draw id — ``id`` absent, or present as ``None`` — the feature's own property stands
              rather than being replaced by nothing:
                ```python
                >>> from digitalearth.web.decoration import DecorationMixin
                >>> own = {"properties": {"id": "mine", "label": "plot A"}}
                >>> DecorationMixin._drawn_properties(own)
                {'id': 'mine', 'label': 'plot A'}
                >>> DecorationMixin._drawn_properties({**own, "id": None})
                {'id': 'mine', 'label': 'plot A'}

                ```
            - A feature with no id of either kind still gets the column, so ``drawn["id"]`` reads the same
              whatever was drawn:
                ```python
                >>> from digitalearth.web.decoration import DecorationMixin
                >>> DecorationMixin._drawn_properties({"properties": {"label": "plot B"}})
                {'label': 'plot B', 'id': None}
                >>> DecorationMixin._drawn_properties({})
                {'id': None}

                ```
        """
        properties = dict(feature.get("properties") or {})
        draw_id = feature.get("id")
        if draw_id is not None:
            properties["id"] = draw_id
        properties.setdefault("id", None)
        return properties

    @staticmethod
    def _attribute_template(fields: list[str] | None) -> dict:
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
        self,
        fields: list[str] | None = None,
        *,
        layer: str | list[str] | None = None,
    ) -> Self:
        """Show an attribute popup on **click** for one or several layers' features (recipe W2, WB-11).

        Args:
            fields: Attribute columns to display (one → that property; several → an HTML table). ``None``
                shows the feature's raw properties.
            layer: Which layer(s) to bind. A single id binds that layer; a **list of ids** binds each one,
                so a map with several queryable layers is inspected in one call (``layer=map.layer_ids``
                binds them all). ``None`` defaults to the most recently added data layer, as before.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when there is no layer to attach to (no ``layer`` and nothing drawn yet), or when
                ``layer`` is an empty list, which names nothing to inspect.
            KeyError: when a named id — alone or in the list — is not a layer on this map.
        """
        return self._bind_attribute(fields, layer, trigger="click", add="add_popup")

    def tooltip(
        self,
        fields: list[str] | None = None,
        *,
        layer: str | list[str] | None = None,
    ) -> Self:
        """Show an attribute tooltip on **hover** for one or several layers' features (recipe W2, WB-11).

        Args:
            fields: Attribute columns to display (one → that property; several → an HTML table). ``None``
                shows the feature's raw properties.
            layer: Which layer(s) to bind. A single id binds that layer; a **list of ids** binds each one
                (``layer=map.layer_ids`` binds them all). ``None`` defaults to the most recently added data
                layer, as before.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when there is no layer to attach to (no ``layer`` and nothing drawn yet), or when
                ``layer`` is an empty list, which names nothing to inspect.
            KeyError: when a named id — alone or in the list — is not a layer on this map.
        """
        return self._bind_attribute(fields, layer, trigger="hover", add="add_tooltip")

    def _bind_attribute(
        self,
        fields: list[str] | None,
        layer: str | list[str] | None,
        *,
        trigger: str,
        add: str,
    ) -> Self:
        """Bind a click popup or a hover tooltip across the resolved target layers.

        :meth:`popup` and :meth:`tooltip` are one mechanism under two names — click versus hover — so the
        body lives here and each name passes its trigger and the ``add_*`` method the widget wires it with.
        A target list is bound layer by layer: each gets its own recorded interaction and its own queued
        closure, tagged with its id so :meth:`~digitalearth.web.base.WebMapBase.remove_layer` takes the
        right popup off.

        Args:
            fields: The attribute columns to show, as :meth:`popup` documents.
            layer: The caller's target — ``None`` for the last layer, one id, or a list of ids.
            trigger: ``"click"`` for a popup, ``"hover"`` for a tooltip.
            add: The widget method each closure calls — ``"add_popup"`` or ``"add_tooltip"``.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: as :meth:`popup` documents.
            KeyError: as :meth:`popup` documents.
        """
        _require_layer_api()
        targets = self._inspector_targets(layer)
        explicit = layer is not None
        if explicit:
            # Validate the whole caller-named list up front — mirror `layer_control`'s `named`/`unknown`
            # pre-check (see `layer_control`) — so a bad id in the list leaves the map untouched rather than
            # binding and queuing every valid id before it, which a catch-and-retry would then double-bind
            # (M2). An id the tier chose — a cluster's loose points — is not caller-named, so it is not
            # pre-checked here; it falls through to `_record_tooltip`, which records it undescribed.
            unknown = [
                layer_id for layer_id in targets if layer_id not in self._layer_tree.ids
            ]
            if unknown:
                raise KeyError(
                    f"no layer {unknown[0]!r} on this map; its layers are {self.layer_ids}"
                )
        kwargs = self._attribute_template(fields)
        for layer_id in targets:
            # Nothing below refuses an id now: a caller-named list was validated above, and a tier-chosen id
            # is recorded as the undescribed layer it is. The record then the queue stay paired per layer.
            self._record_tooltip(layer_id, fields, trigger=trigger, explicit=explicit)
            self._queue(self._attribute_closure(layer_id, add, kwargs))
        return self

    def _inspector_targets(self, layer: str | list[str] | None) -> list[str]:
        """Resolve a ``popup``/``tooltip`` ``layer=`` into the list of layer ids to bind.

        Args:
            layer: ``None`` for the most recent data layer, one id, or a list of ids.

        Returns:
            The ids to bind, in the order the caller gave them.

        Raises:
            ValueError: when ``layer`` is ``None`` and nothing has been drawn, or when it is an empty list.
        """
        if layer is None:
            last = self._last_layer_id
            if last is None:
                raise ValueError(
                    "popup()/tooltip() needs a layer — draw a data layer first or pass layer=..."
                )
            return [last]
        if isinstance(layer, str):
            return [layer]
        targets = list(layer)
        if not targets:
            raise ValueError(
                "popup()/tooltip() was given layer=[], which names no layer to inspect; pass one id, a "
                "list of ids, or leave it to bind the most recent layer"
            )
        return targets

    @staticmethod
    def _attribute_closure(layer_id: str, add: str, kwargs: dict) -> Any:
        """Build the queued ``apply(widget)`` closure that wires one layer's popup or tooltip.

        Built in its own frame rather than in :meth:`_bind_attribute`'s loop so that ``layer_id`` is a
        fresh closure cell per target — a loop-local would late-bind every closure to the last id — and so
        it reads back as a nonlocal, which is how ``remove_layer`` finds a closure's layer.

        Args:
            layer_id: The layer this closure binds.
            add: The widget method to call — ``"add_popup"`` or ``"add_tooltip"``.
            kwargs: The ``prop``/``template`` kwargs that method takes.

        Returns:
            The ``apply(widget)`` callable, tagged with its layer id for removal.
        """

        def apply(widget: Any) -> None:
            getattr(widget, add)(layer_id, **kwargs)

        # Tagged with the layer it belongs to, so `remove_layer` takes the popup off with it rather than
        # leaving a page that pops up over a layer nobody can see.
        apply._digitalearth_layer_id = layer_id  # type: ignore[attr-defined]
        return apply
