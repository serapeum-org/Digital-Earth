"""HtmlDocument — Digital-Earth's own standalone-HTML export for the web tier (WB-13/17/20).

py-maplibregl's own ``to_html`` renders through a sealed bundle that hardcodes one container id,
never exposes the map object, and keeps maplibre-gl internal — so a split/swipe export, a synced
minimap, and a live measure readout are all impossible through it. This module emits our **own**
page instead: it loads maplibre-gl from a CDN, ships our small runtime (:mod:`srcjs/de_maplibre.js`)
that rebuilds each map from the very ``{mapOptions, calls}`` state ``to_dict()`` records — against a
container we own and exposed on ``window.DE.maps`` — and wires a cross-map feature on top
(:mod:`srcjs/de_features.js`).

The document is a value object: a tuple of :class:`MapPanel` (a container id and that map's serialized
state) plus the feature markup and the bootstrap that builds the maps and starts the feature. The
builders that produce one — :meth:`HtmlDocument.swipe` today — live here so the feature's HTML, its
bootstrap and its panels are described in one place.

The maplibre-gl version is pinned to the one py-maplibregl's own template references, so a page
exported this way loads the same library the widget path does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

#: The maplibre-gl version py-maplibregl's template references; pinned so both export paths agree.
MAPLIBRE_VERSION = "5.3.0"
_MAPLIBRE_JS = f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.js"
_MAPLIBRE_CSS = f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.css"

#: mapbox-gl-draw, loaded only when a page needs a draw control (the measure readout, or a map that
#: recorded ``addMapboxDraw``). Pinned, and the version py-maplibregl uses for its own draw control.
MAPBOX_DRAW_VERSION = "1.4.3"
_DRAW_JS = f"https://unpkg.com/@mapbox/mapbox-gl-draw@{MAPBOX_DRAW_VERSION}/dist/mapbox-gl-draw.js"
_DRAW_CSS = f"https://unpkg.com/@mapbox/mapbox-gl-draw@{MAPBOX_DRAW_VERSION}/dist/mapbox-gl-draw.css"

#: deck.gl, loaded only when a map recorded a deck.gl overlay (``addDeckOverlay``) — the big-data GPU
#: layers. The standalone bundle carries the layer classes and ``MapboxOverlay`` (to draw them over
#: the map) but **not** the separate ``@deck.gl/json`` converter, so the runtime builds each ``@@type``
#: layer directly from its class rather than through a ``JSONConverter`` (see ``de_maplibre.js``).
DECK_VERSION = "9.0.38"
_DECK_JS = f"https://unpkg.com/deck.gl@{DECK_VERSION}/dist.min.js"

_SRCJS = Path(__file__).parent / "srcjs"


# Assets are cached for the life of the process. They are immutable in a built wheel; a developer
# editing srcjs/*.js in a long-lived process (a running notebook kernel) will not see the change
# without a restart — acceptable for shipped runtime assets, noted so it is not mistaken for a bug.
@cache
def _asset(name: str) -> str:
    """Return the text of a shipped ``srcjs`` asset (cached for the process).

    Args:
        name: The file name under ``web/srcjs`` (e.g. ``"de_maplibre.js"``).

    Returns:
        The file's text.
    """
    return (_SRCJS / name).read_text(encoding="utf-8")


def _embed_json(data: dict) -> str:
    """Serialize ``data`` to JSON safe to embed inline inside a ``<script>`` block.

    ``json.dumps`` does not escape ``<`` / ``>`` / ``&``, so a serialized string containing
    ``</script>`` (an attribution, a basemap provider, a GeoJSON feature property, an ``InfoBox``
    legend) would terminate the ``<script>`` element early and turn the rest of the page into live
    markup — a page break on benign data, a stored-XSS vector when the data is third-party. The three
    characters are escaped to their ``\\uXXXX`` forms, which a JS string/object literal reads back as
    the original character, so the data is preserved while the raw ``</script>`` never appears in the
    HTML source. (U+2028 / U+2029 are already safe: ``json.dumps`` defaults to ``ensure_ascii=True``.)

    Args:
        data: The ``{mapOptions, calls}`` dict (or any JSON-serialisable value) to embed.

    Returns:
        A JSON string with ``<`` / ``>`` / ``&`` unicode-escaped.
    """
    return (
        json.dumps(data)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


@dataclass(frozen=True)
class MapPanel:
    """One map in an :class:`HtmlDocument`: the container it draws into and its serialized state.

    Args:
        container_id: The ``id`` of the ``<div>`` the map is built into — unique within the page.
        data: The map's ``{mapOptions, calls}`` dict, as ``MapWidget.to_dict()`` returns it, which
            the runtime replays to rebuild the map.
    """

    container_id: str
    data: dict


@dataclass(frozen=True)
class HtmlDocument:
    """A standalone HTML page hosting one or more maps plus a cross-map feature.

    Args:
        panels: The maps to build, each a :class:`MapPanel`.
        body_html: The container markup placed in ``<body>`` (the feature's wrapper and map divs).
        bootstrap_js: The script that builds every panel (``DE.buildMap``) and starts the feature
            once the maps are ready.
        feature_js: ``True`` when the page needs ``de_features.js`` (a cross-map feature); ``False``
            for a plain single-map page that needs only the runtime.
        needs_draw: ``True`` when the page needs mapbox-gl-draw on it (the measure document, WB-17);
            the draw library is loaded from the CDN only then, not on every page.
    """

    panels: tuple[MapPanel, ...]
    body_html: str
    bootstrap_js: str
    feature_js: bool
    needs_draw: bool = False

    @classmethod
    def swipe(
        cls,
        before: dict,
        after: dict,
        *,
        height: int = 600,
    ) -> HtmlDocument:
        """Build the document for a WB-13 swipe between two maps.

        The two maps are overlaid in one wrapper; the runtime keeps their cameras in sync and the
        feature clips the "after" map to the right of a draggable divider.

        Args:
            before: The left/under map's ``to_dict()`` state (shows on the left of the divider).
            after: The right/over map's ``to_dict()`` state (clipped to the right of the divider).
            height: The map height in CSS pixels.

        Returns:
            The :class:`HtmlDocument` for the swipe page.
        """
        wrap, left, right = "de-swipe", "de-map-before", "de-map-after"
        body = (
            f'<div id="{wrap}" class="de-swipe-wrap" style="height:{int(height)}px">'
            f'<div id="{left}" class="de-swipe-map"></div>'
            f'<div id="{right}" class="de-swipe-map"></div>'
            f"</div>"
        )
        bootstrap = (
            "(function(){var ready=0;function go(){if(++ready===2){"
            f'window.DE.swipe("{wrap}","{left}","{right}");}}}}'
            f'window.DE.buildMap("{left}",{_embed_json(before)},go);'
            f'window.DE.buildMap("{right}",{_embed_json(after)},go);'
            "})();"
        )
        return cls(
            panels=(MapPanel(left, before), MapPanel(right, after)),
            body_html=body,
            bootstrap_js=bootstrap,
            feature_js=True,
        )

    @classmethod
    def minimap(
        cls,
        main: dict,
        overview: dict,
        *,
        height: int = 600,
        mini_size: tuple[int, int] = (200, 150),
    ) -> HtmlDocument:
        """Build the document for a WB-20 minimap: a main map with a synced overview inset.

        The overview map follows the main map at a lower zoom and draws a rectangle of the main
        map's current view; clicking it recentres the main map.

        Args:
            main: The main map's ``to_dict()`` state (fills the page).
            overview: The overview map's ``to_dict()`` state (the small inset — usually a basemap).
            height: The main map height in CSS pixels.
            mini_size: ``(width, height)`` of the overview inset, in CSS pixels.

        Returns:
            The :class:`HtmlDocument` for the minimap page.
        """
        wrap, main_id, mini_id = "de-minimap-wrap", "de-map-main", "de-minimap"
        mini_w, mini_h = mini_size
        body = (
            f'<div id="{wrap}" class="de-minimap-wrap" style="height:{int(height)}px">'
            f'<div id="{main_id}" class="de-minimap-main"></div>'
            f'<div id="{mini_id}" class="de-minimap" '
            f'style="width:{int(mini_w)}px;height:{int(mini_h)}px"></div>'
            f"</div>"
        )
        bootstrap = (
            "(function(){var ready=0;function go(){if(++ready===2){"
            f'window.DE.minimap("{main_id}","{mini_id}");}}}}'
            f'window.DE.buildMap("{main_id}",{_embed_json(main)},go);'
            f'window.DE.buildMap("{mini_id}",{_embed_json(overview)},go);'
            "})();"
        )
        return cls(
            panels=(MapPanel(main_id, main), MapPanel(mini_id, overview)),
            body_html=body,
            bootstrap_js=bootstrap,
            feature_js=True,
        )

    @classmethod
    def measure(
        cls,
        main: dict,
        *,
        height: int = 600,
        distance: bool = True,
        area: bool = True,
    ) -> HtmlDocument:
        """Build the document for a WB-17 live measure readout.

        The page adds a mapbox-gl-draw control to the map; on every draw change the feature computes
        the drawn line's length and/or polygon's area and writes it into an on-map readout. This is a
        convenience readout — the authoritative geodesic measure is still ``WebMap.drawn_features()``
        handing the geometry to pyramids.

        Args:
            main: The map's ``to_dict()`` state (the map measured on).
            height: The map height in CSS pixels.
            distance: Offer the line tool and show the drawn length.
            area: Offer the polygon tool and show the drawn area.

        Returns:
            The :class:`HtmlDocument` for the measure page.

        Raises:
            ValueError: If ``main`` already carries a draw control (``measure()`` was called on the
                map before exporting). The live readout adds its own draw control, so a map that
                already has one would end up with two; pass the map plain instead.
        """
        if any(call and call[0] == "addMapboxDraw" for call in main.get("calls", [])):
            raise ValueError(
                "measure(): the map already has a draw control (measure() was called on it) — pass "
                "the map without calling measure(); the live readout adds its own draw control"
            )
        wrap, main_id, readout = (
            "de-measure-wrap",
            "de-map-measure",
            "de-measure-readout",
        )
        body = (
            f'<div id="{wrap}" class="de-measure-wrap" style="height:{int(height)}px">'
            f'<div id="{main_id}" class="de-measure-map"></div>'
            f'<div id="{readout}" class="de-measure-readout"></div>'
            f"</div>"
        )
        opts = json.dumps({"distance": bool(distance), "area": bool(area)})
        bootstrap = (
            f'window.DE.buildMap("{main_id}",{_embed_json(main)},function(){{'
            f'window.DE.measure("{main_id}","{readout}",{opts});}});'
        )
        return cls(
            panels=(MapPanel(main_id, main),),
            body_html=body,
            bootstrap_js=bootstrap,
            feature_js=True,
            needs_draw=True,
        )

    def _panels_record(self, *names: str) -> bool:
        """Return ``True`` if any panel's serialized ``calls`` records one of ``names``.

        The CDN libraries are loaded on demand: a page gets deck.gl only when a map actually draws a
        deck.gl overlay, and mapbox-gl-draw only when a map has a draw control — read off the calls
        the maps recorded, so the libraries track the content rather than being loaded on every page.

        Args:
            *names: Call method names to look for (e.g. ``"addDeckOverlay"``).

        Returns:
            Whether any panel records a matching call.
        """
        wanted = set(names)
        return any(
            call and call[0] in wanted
            for panel in self.panels
            for call in panel.data.get("calls", [])
        )

    def render(self, *, title: str = "Digital-Earth map") -> str:
        """Return the standalone HTML string.

        Loads maplibre-gl from the CDN always, and deck.gl / mapbox-gl-draw only when the maps need
        them — deck.gl when a map records a deck.gl overlay, mapbox-gl-draw when a map has a draw
        control or the page is a measure page.

        Args:
            title: The document ``<title>``.

        Returns:
            A complete HTML document: the needed CDN libraries + our CSS/runtime/feature scripts, the
            feature markup, and the bootstrap that builds the maps and starts the feature.
        """
        scripts = [_asset("de_maplibre.js")]
        if self.feature_js:
            scripts.append(_asset("de_features.js"))
        scripts.append(self.bootstrap_js)
        script_tags = "\n".join(f"<script>\n{js}\n</script>" for js in scripts)
        needs_deck = self._panels_record("addDeckOverlay", "setDeckLayers")
        needs_draw = self.needs_draw or self._panels_record("addMapboxDraw")
        draw_css = (
            f'<link rel="stylesheet" href="{_DRAW_CSS}"/>\n' if needs_draw else ""
        )
        draw_js = f'<script src="{_DRAW_JS}"></script>\n' if needs_draw else ""
        deck_js = f'<script src="{_DECK_JS}"></script>\n' if needs_deck else ""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f"<title>{title}</title>\n"
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<link rel="stylesheet" href="{_MAPLIBRE_CSS}"/>\n'
            f"{draw_css}"
            f"<style>\n{_asset('de_maplibre.css')}\n</style>\n"
            "</head>\n<body>\n"
            f"{self.body_html}\n"
            f'<script src="{_MAPLIBRE_JS}"></script>\n'
            f"{deck_js}"
            f"{draw_js}"
            f"{script_tags}\n"
            "</body>\n</html>\n"
        )
