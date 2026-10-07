"""HtmlDocument — Digital-Earth's own standalone-HTML export for the web tier (WB-13/17/20).

py-maplibregl's ``Map.to_html`` renders through a sealed bundle that hardcodes one container id,
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

_SRCJS = Path(__file__).parent / "srcjs"


@cache
def _asset(name: str) -> str:
    """Return the text of a shipped ``srcjs`` asset (cached).

    Args:
        name: The file name under ``web/srcjs`` (e.g. ``"de_maplibre.js"``).

    Returns:
        The file's text.
    """
    return (_SRCJS / name).read_text(encoding="utf-8")


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
    """

    panels: tuple[MapPanel, ...]
    body_html: str
    bootstrap_js: str
    feature_js: bool

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
            f'window.DE.buildMap("{left}",{json.dumps(before)},go);'
            f'window.DE.buildMap("{right}",{json.dumps(after)},go);'
            "})();"
        )
        return cls(
            panels=(MapPanel(left, before), MapPanel(right, after)),
            body_html=body,
            bootstrap_js=bootstrap,
            feature_js=True,
        )

    def render(self, *, title: str = "Digital-Earth map") -> str:
        """Return the standalone HTML string.

        Args:
            title: The document ``<title>``.

        Returns:
            A complete HTML document: maplibre-gl (CDN) + our CSS/runtime/feature scripts, the
            feature markup, and the bootstrap that builds the maps and starts the feature.
        """
        scripts = [_asset("de_maplibre.js")]
        if self.feature_js:
            scripts.append(_asset("de_features.js"))
        scripts.append(self.bootstrap_js)
        script_tags = "\n".join(f"<script>\n{js}\n</script>" for js in scripts)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f"<title>{title}</title>\n"
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<link rel="stylesheet" href="{_MAPLIBRE_CSS}"/>\n'
            f"<style>\n{_asset('de_maplibre.css')}\n</style>\n"
            "</head>\n<body>\n"
            f"{self.body_html}\n"
            f'<script src="{_MAPLIBRE_JS}"></script>\n'
            f"{script_tags}\n"
            "</body>\n</html>\n"
        )
