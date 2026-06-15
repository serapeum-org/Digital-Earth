"""WebMapBase — the core MapLibre + deck.gl plumbing the web capability mixins build on.

``WebMapBase`` owns the layer registry (in add order), the display-CRS reproject-through-pyramids
plumbing, and the render/save/show lifecycle around the ``maplibre`` (py-maplibregl) anywidget. Capability
mixins (raster, vector, big-data, 3-D, temporal, decoration, export) live in sibling modules and add
``add_raster()`` / ``choropleth()`` / … builder methods that call ``self.add_layer(...)``; the public
:class:`digitalearth.web.map.WebMap` composes the base with those mixins — mirroring the 2-D
``Map(GeoLayerBase, RasterMixin, …)``, the 3-D ``Scene3D(Scene3DBase, …)`` and the interactive
``InteractiveMap(InteractiveMapBase, …)`` patterns exactly.

MapLibre + deck.gl are a **renderer, not a GIS engine**: every layer is built from pyramids-sourced numpy /
GeoDataFrames (never xarray/rasterio/cartopy — see the tier's HARD RULE, enforced by
``tests/test_no_competitor_imports.py``). All CRS/reproject work happens upstream in pyramids
(``Dataset.to_crs``) *before* a layer is built, so layers are already in the display CRS — MapLibre and tile
basemaps render in EPSG:3857 / lon-lat (EPSG:4326) only.

The engine import is **lazy**: ``import digitalearth.web`` works without the ``web`` extra installed; only
calling a builder/render method raises an actionable ``ImportError`` (``pip install 'digitalearth[web]'``).
"""

import pathlib
from typing import Any, List, Optional

from digitalearth.sources import get_source
from digitalearth.sources.source import Source

#: The pip extra / pixi env that provides the MapLibre + deck.gl engine, quoted in the lazy-import error.
_INSTALL_HINT = (
    "the web tier needs MapLibre GL JS + deck.gl (maplibre/lonboard). "
    "Install it with `pip install 'digitalearth[web]'` "
    "(or, in this repo, `pixi install -e web`)."
)

#: Short style aliases → CartoCDN basemap slugs. A bare name resolves to the CartoCDN style URL; a full URL
#: (or a MapLibre style ``dict``) is passed through untouched. CartoCDN basemaps need no API token.
_STYLE_ALIASES = {
    "dark": "dark-matter",
    "light": "positron",
    "positron": "positron",
    "voyager": "voyager",
}


def _patch_maplibre_html_encoding() -> None:
    """Shim ``maplibre``'s internal-file reader to UTF-8 (works around a Windows cp1252 crash).

    Upstream bug (py-maplibregl 0.3.6): ``maplibre._utils.read_internal_file`` opens its bundled JS with a
    bare ``open()``, so on Windows (cp1252 default) ``to_html``/``save`` crash with a ``UnicodeDecodeError``
    reading the widget JS. We rebind the reader — in both ``maplibre._utils`` and the ``maplibre.map`` copy
    that imported it by name — to read UTF-8. This lives entirely in *our* code (maplibre is never edited);
    the matching *write* bug is sidestepped by :meth:`WebMapBase.save` writing the HTML itself. Idempotent.

    See ``planning/maplibre-deckgl/web-maplibre-deckgl-plan.md`` (DW.0 notes) — report upstream, remove the
    shim once a fixed ``maplibre`` ships.
    """
    import maplibre._utils as _utils
    import maplibre.map as _map
    from maplibre._utils import get_internal_file_path

    if getattr(_utils.read_internal_file, "_digitalearth_utf8", False):
        return

    def read_internal_file(*args: Any) -> str:
        with open(get_internal_file_path(*args), encoding="utf-8") as handle:
            return handle.read()

    read_internal_file._digitalearth_utf8 = True  # type: ignore[attr-defined]
    _utils.read_internal_file = read_internal_file
    _map.read_internal_file = read_internal_file


def _require_maplibre() -> tuple:
    """Import and return ``(MapOptions, MapWidget)``, raising an actionable error when absent.

    The single lazy-import choke point every engine-touching method calls first, so that
    ``import digitalearth.web`` itself never needs the optional ``web`` extra. Also silences ``maplibre``'s
    optional-``shiny`` warning and applies the UTF-8 HTML shim (:func:`_patch_maplibre_html_encoding`).

    Returns:
        tuple: ``(maplibre.MapOptions, maplibre.ipywidget.MapWidget)``.

    Raises:
        ImportError: when the ``web`` extra is not installed, with the install command in the message.
    """
    try:
        import logging

        # maplibre logs a warning at import when the optional `shiny` binding is absent; quiet it.
        logging.getLogger("maplibre").setLevel(logging.ERROR)
        from maplibre import MapOptions
        from maplibre.ipywidget import MapWidget
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err
    _patch_maplibre_html_encoding()
    return MapOptions, MapWidget


def _resolve_style(style: Any) -> Any:
    """Resolve a basemap ``style`` to a MapLibre style URL/spec.

    A short alias (``"dark"``/``"light"``/``"positron"``/``"voyager"``) becomes its CartoCDN style URL; a
    full ``http(s)`` URL, a ``*.json`` path, or a non-string MapLibre style ``dict`` passes through.

    Args:
        style: A short alias, a style URL/path, or a MapLibre style ``dict``.

    Returns:
        The resolved style URL (``str``) or the original non-string spec.
    """
    if not isinstance(style, str):
        return style
    if style.startswith("http") or style.endswith(".json"):
        return style
    slug = _STYLE_ALIASES.get(style.lower(), style)
    return f"https://basemaps.cartocdn.com/gl/{slug}-gl-style/style.json"


class WebMapBase:
    """Core host: ordered layer registry + display CRS + render/save lifecycle over the MapLibre widget.

    Args:
        center: Optional ``(lon, lat)`` map centre, in the display CRS' lon/lat. ``None`` lets MapLibre pick.
        zoom: Initial zoom level.
        style: Basemap style — a short alias (``"dark"``/``"light"``/``"voyager"``), a style URL, or a
            MapLibre style ``dict`` (see :func:`_resolve_style`).
        crs: Display CRS as an EPSG integer. Default ``3857`` (Web Mercator — what MapLibre and tile
            basemaps render); ``4326`` for a lon/lat globe/Plate-Carrée.
        height: Widget height in pixels (``None`` keeps the MapLibre default).

    Attributes:
        layers: Registered layers, in add (= draw) order. A layer is either a ``maplibre`` ``Layer``/spec
            passed to the widget's ``add_layer``, or a callable ``apply(widget)`` a mixin registers to wire
            up its own source(s) + layer(s).

    Examples:
        - Construct and inspect the display configuration (needs no engine):
            ```python
            >>> from digitalearth.web.base import WebMapBase
            >>> m = WebMapBase(center=(8.0, 47.0), zoom=5, style="light", crs=4326, height=400)
            >>> (m.center, m.zoom, m.style, m.crs, m.height)
            ((8.0, 47.0), 5, 'light', 4326, 400)
            >>> m.layers
            []

            ```
        - Defaults target the Web-Mercator tile contract:
            ```python
            >>> from digitalearth.web.base import WebMapBase
            >>> m = WebMapBase()
            >>> (m.crs, m.zoom)
            (3857, 2)

            ```

    See Also:
        digitalearth.web.map.WebMap: the public composition of this base with the capability mixins.
    """

    def __init__(
        self,
        *,
        center: Optional[Any] = None,
        zoom: int = 2,
        style: Any = "dark",
        crs: int = 3857,
        height: Optional[int] = 500,
    ):
        self.center = center
        self.zoom = zoom
        self.style = style
        self.crs = crs
        self.height = height
        self.layers: List[Any] = []

    def add_layer(self, layer: Any) -> "WebMapBase":
        """Register ``layer`` and return ``self`` (chainable).

        The low-level entry point the capability mixins build on — every builder method ends here. A layer
        is replayed onto the MapLibre widget at :meth:`render` time (see :meth:`_apply_layer`).

        Args:
            layer: A ``maplibre`` ``Layer``/spec, or a callable ``apply(widget)``.

        Returns:
            This map, so builder calls chain: ``m.add_raster(dem).basemap()``.

        Examples:
            - Registration appends in order and returns the map for chaining (any object stands in for a
              layer here — the registry does not inspect it):
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap()
                >>> m.add_layer("raster-layer").add_layer("vector-layer") is m
                True
                >>> m.layers
                ['raster-layer', 'vector-layer']

                ```
        """
        self.layers.append(layer)
        return self

    def _needs_reproject(self, data: Any) -> bool:
        """Whether ``data`` must be reprojected (via pyramids) to the display CRS.

        Args:
            data: A pyramids object exposing ``.epsg`` (``Dataset``/``FeatureCollection``).

        Returns:
            ``False`` only when the display CRS is an ``int`` equal to ``data.epsg``; ``True`` otherwise.
        """
        return not (
            isinstance(self.crs, int) and getattr(data, "epsg", None) == self.crs
        )

    def _to_display_source(self, data: Any, *, band: int = 1) -> Source:
        """Reproject ``data`` to the display CRS through pyramids and wrap it as a :class:`Source`.

        The single display-CRS choke point every raster/vector builder calls (settling the tier's
        projection decision, option A: **pre-reproject in pyramids** — no cartopy anywhere). Inputs already
        in the display CRS pass through untouched; everything else goes through ``data.to_crs(self.crs)``
        (pyramids' warp for rasters, ``FeatureCollection`` reprojection for vectors). MapLibre renders only
        EPSG:3857 / 4326, so ``crs`` should be one of those.

        Args:
            data: A pyramids ``Dataset`` / ``FeatureCollection`` (anything ``get_source`` accepts). Bare
                numpy arrays / :class:`Source` objects pass straight through to extraction.
            band: 1-based band to extract for raster inputs.

        Returns:
            Source: the display-CRS view (``z``/``x``/``y``/``crs``/``metadata``).
        """
        if isinstance(data, Source):
            return data
        if (
            hasattr(data, "epsg")
            and hasattr(data, "to_crs")
            and self._needs_reproject(data)
        ):
            data = data.to_crs(self.crs)
        return get_source(data, band=band)

    def _map_options(self) -> dict:
        """Build the ``MapOptions`` kwargs from the display config (drops an unset ``center``)."""
        options: dict = {"zoom": self.zoom, "style": _resolve_style(self.style)}
        if self.center is not None:
            options["center"] = self.center
        return options

    @staticmethod
    def _apply_layer(widget: Any, layer: Any) -> None:
        """Apply one registered ``layer`` onto the MapLibre ``widget``.

        A callable layer is a mixin-supplied ``apply(widget)`` (it wires its own source(s) + layer(s));
        anything else is handed to the widget's ``add_layer``.

        Args:
            widget: The MapLibre ``MapWidget`` being assembled.
            layer: A callable ``apply(widget)`` or a ``maplibre`` ``Layer``/spec.
        """
        if callable(layer):
            layer(widget)
        else:
            widget.add_layer(layer)

    def render(self) -> Any:
        """Build and return the configured MapLibre ``MapWidget`` (an empty map if no layers are registered).

        Returns:
            The ``maplibre.ipywidget.MapWidget`` — an anywidget that renders inline in notebooks.

        Raises:
            ImportError: when the ``web`` extra is not installed.
        """
        MapOptions, MapWidget = _require_maplibre()
        kwargs: dict = {"map_options": MapOptions(**self._map_options())}
        if self.height is not None:
            kwargs["height"] = int(self.height)
        widget = MapWidget(**kwargs)
        for layer in self.layers:
            self._apply_layer(widget, layer)
        return widget

    def save(self, path: str, *, title: str = "Digital-Earth map", **kwargs: Any) -> str:
        """Save the map as a standalone HTML page and return ``path``.

        Renders the widget, serialises it to HTML via ``MapWidget.to_html``, and writes the bytes ourselves
        as UTF-8 — sidestepping maplibre's cp1252-on-Windows writer bug (see
        :func:`_patch_maplibre_html_encoding`). The page embeds the map state and widget JS; it still
        references ``maplibre-gl`` from a CDN, so it is *self-contained for the data* but needs a network for
        the engine JS (true offline bundling is a DW.6 concern).

        Args:
            path: Output ``*.html`` file.
            title: HTML document title.
            **kwargs: Forwarded to ``MapWidget.to_html``.

        Returns:
            The ``path`` written.

        Raises:
            ImportError: when the ``web`` extra is not installed.
        """
        html = self.render().to_html(title=title, **kwargs)
        pathlib.Path(path).write_text(html, encoding="utf-8")
        return str(path)

    def show(self) -> Any:
        """Render the map and display it inline when IPython is available.

        Returns:
            The MapLibre widget (which a notebook front-end renders richly).

        Raises:
            ImportError: when the ``web`` extra is not installed.
        """
        widget = self.render()
        try:
            from IPython.display import display

            display(widget)
        except (
            ImportError
        ):  # plain-script use: returning the widget is all there is to show
            pass
        return widget

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Render the map inline in notebooks by delegating to the MapLibre widget.

        Returns:
            The widget's mimebundle, or an empty dict when the engine is missing (so a bare repr in a
            notebook degrades gracefully instead of raising).
        """
        try:
            widget = self.render()
        except ImportError:
            return {}
        hook = getattr(widget, "_repr_mimebundle_", None)
        # The Jupyter protocol passes include/exclude as keywords; the maplibre widget (ipywidgets) only
        # accepts them that way, so never call this hook positionally.
        return hook(include=include, exclude=exclude) if hook is not None else {}
