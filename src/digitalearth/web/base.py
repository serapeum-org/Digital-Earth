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

from digitalearth.base.sources import get_source
from digitalearth.base.sources.source import Source

#: The pip extra / pixi env that provides the MapLibre + deck.gl engine, quoted in the lazy-import error.
_INSTALL_HINT = (
    "the web tier needs MapLibre GL JS + deck.gl (maplibre/lonboard). "
    "Install it with `pip install 'digitalearth[web]'` "
    "(or, in this repo, `pixi install -e web`)."
)

#: How many column names an "active geometry missing" error quotes before it elides the rest — a wide
#: table would otherwise print hundreds of names into a traceback.
_MAX_COLUMNS_IN_ERROR = 12

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
    reading the widget JS. We rebind the reader to read UTF-8 across **every** already-imported ``maplibre``
    submodule that still references the original (not just ``_utils``/``map``) — some import it by name — so no
    stale bare-``open`` reader survives; modules imported *after* the patch pick up the shim from the patched
    ``_utils`` automatically. This lives entirely in *our* code (maplibre is never edited); the matching *write*
    bug is sidestepped by :meth:`WebMapBase.save` writing the HTML itself.

    This is a **permanent, self-gating** shim (kept in-house by design, not pending an upstream fix): it only
    patches on a platform whose default encoding actually fails to read the bundled JS, is applied once
    (idempotent), and is a no-op everywhere else.
    """
    import sys

    import maplibre._utils as _utils
    from maplibre._utils import get_internal_file_path

    original = _utils.read_internal_file
    if getattr(original, "_digitalearth_utf8", False):
        return

    # Self-gating: if the platform default encoding already reads the bundled JS (non-Windows, or once
    # upstream specifies utf-8), the shim is unnecessary — don't monkeypatch the dependency at all.
    try:
        with open(get_internal_file_path("srcjs", "pywidget.js")) as handle:
            handle.read()
        return
    except UnicodeDecodeError:
        pass  # the cp1252-on-Windows bug — install the shim below
    except OSError:
        pass  # could not probe (file moved/renamed) — patch defensively

    def read_internal_file(*args: Any, **kwargs: Any) -> str:
        with open(get_internal_file_path(*args, **kwargs), encoding="utf-8") as handle:
            return handle.read()

    read_internal_file._digitalearth_utf8 = True  # type: ignore[attr-defined]
    # Patch the canonical module first, then rebind every maplibre submodule that imported the reader by
    # name and still holds the original (e.g. `maplibre.map`) so no un-shimmed reference is left behind.
    _utils.read_internal_file = read_internal_file
    for name, module in list(sys.modules.items()):
        if name.startswith("maplibre") and getattr(module, "read_internal_file", None) is original:
            module.read_internal_file = read_internal_file


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


def _require_layer_api() -> tuple:
    """Import and return ``(Layer, LayerType)``, raising the actionable error when the engine is absent.

    The builder choke point: every raster/vector/decoration builder calls this first, so a builder invoked
    without the ``web`` extra raises the same actionable ``ImportError`` as :func:`_require_maplibre`.

    Returns:
        tuple: ``(maplibre.Layer, maplibre.LayerType)``.

    Raises:
        ImportError: when the ``web`` extra is not installed.
    """
    _require_maplibre()  # friendly error + UTF-8 shim
    from maplibre import Layer, LayerType

    return Layer, LayerType


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
        crs: The EPSG the data is normalised to before it is handed to MapLibre. Default ``4326`` —
            MapLibre ingests GeoJSON/image-source coordinates as lon/lat (WGS84) and renders Web
            Mercator itself, so ``4326`` is the only value that places inline data correctly.
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
        - Data defaults to lon/lat (what MapLibre ingests):
            ```python
            >>> from digitalearth.web.base import WebMapBase
            >>> m = WebMapBase()
            >>> (m.crs, m.zoom)
            (4326, 2)

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
        crs: int = 4326,
        height: Optional[int] = 500,
    ):
        self.center = center
        self.zoom = zoom
        self.style = style
        self.crs = crs
        self.height = height
        self.layers: List[Any] = []
        #: Monotonic counter handing out unique source/layer ids (see :meth:`_uid`).
        self._id_counter = 0
        #: Id of the most recently added data layer — the default target for ``popup``/``tooltip``.
        self._last_layer_id: Optional[str] = None
        #: Class breaks from the most recent classified ``choropleth``/``points`` (for an out-of-band legend).
        self.last_breaks: Optional[List[float]] = None
        #: Accumulated deck.gl JSON layers, applied in one ``add_deck_layers`` call at render (DW.3).
        self._deck_layers: Optional[List[dict]] = None
        #: Feature count above which ``points``/``polygons`` auto-route to a GPU layer (logged, never silent).
        self.big_data_threshold = 50_000
        #: Time-slider config set by ``timeslider`` (``None`` = no temporal control); read by ``render``.
        #: ``mode`` selects the wiring: ``"vector"`` carries ``layer_id`` and filters one layer by
        #: ``kdim``; ``"raster"`` carries ``layer_ids`` and swaps their visibility. Both carry ``times``.
        self._temporal: Optional[dict] = None

    def _uid(self, prefix: str) -> str:
        """Return a per-map-unique id like ``"fill-3"`` for a MapLibre source/layer.

        Args:
            prefix: A short kind tag (``"fill"``, ``"circle"``, ``"raster"``, …).

        Returns:
            ``f"{prefix}-{n}"`` with a counter that increments on every call, so two builders never
            collide on a MapLibre source/layer id.
        """
        self._id_counter += 1
        return f"{prefix}-{self._id_counter}"

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

    @staticmethod
    def _reject_raster(data: Any, method: str) -> None:
        """Raise ``TypeError`` when ``data`` is a pyramids raster, leaving anything else alone.

        For the builders that legitimately accept something other than a vector layer — ``point_cloud``
        takes a raw sequence of ``xyz`` triples — where :meth:`_require_vector` would be too strict. A
        raster is recognised by reporting ``columns`` as an ``int`` (the grid width in cells); a table
        reports an ``Index`` of names and a bare sequence has no ``columns`` at all.

        Args:
            data: The caller's input.
            method: The calling builder name (quoted in the error).

        Raises:
            TypeError: when ``data`` is a pyramids raster.
        """
        if isinstance(getattr(data, "columns", None), int):
            raise TypeError(
                f"{method}() does not take a raster; got {type(data).__name__}. For a single raster use "
                f"add_raster(); for a raster time stack pass a DatasetCollection to timeslider()."
            )

    @staticmethod
    def _require_vector(features: Any, method: str) -> None:
        """Raise ``TypeError`` unless ``features`` is a vector layer.

        The tier's single input guard, called from :meth:`_display_gdf` so every vector builder rejects a
        raster the same way. Without it a pyramids raster reaches whichever expression the builder happens
        to run first and fails obscurely there — a ``Dataset`` reports ``columns`` as an ``int`` (the grid
        width in cells), so an attribute-membership test raises ``TypeError: argument of type 'int' is not
        iterable``, while other builders raise ``has no len()``, an ``AttributeError``, or nothing at all.

        Args:
            features: The caller's input, expected to be a pyramids ``FeatureCollection`` / GeoDataFrame.
            method: The calling builder name (quoted in the error).

        Raises:
            TypeError: when ``features`` exposes no ``geometry``. A table that has columns but no *active*
                geometry (``set_geometry`` never called) gets its own message naming ``set_geometry``,
                rather than being misreported as a raster.

        Examples:
            - A vector-like input passes the guard silently (the check returns nothing):
                ```python
                >>> from digitalearth.web import WebMap
                >>> layer = type("Layer", (), {"geometry": ()})()
                >>> print(WebMap._require_vector(layer, "points"))
                None

                ```
            - A raster is turned away, and the message names the type that arrived:
                ```python
                >>> from digitalearth.web import WebMap
                >>> try:
                ...     WebMap._require_vector(42, "points")
                ... except TypeError as err:
                ...     print(str(err).split("(")[0])
                points

                ```
            - The message routes the caller to the raster builders in this same tier:
                ```python
                >>> from digitalearth.web import WebMap
                >>> try:
                ...     WebMap._require_vector(42, "points")
                ... except TypeError as err:
                ...     print("add_raster" in str(err), "DatasetCollection" in str(err))
                True True

                ```

        See Also:
            digitalearth.web.bigdata.BigDataMixin._require_points: the narrower geometry-kind guard that
                runs after this one for the heatmap/cluster builders.
        """
        if hasattr(features, "geometry"):
            return
        # A raster reports `columns` as an int (the grid width); a table reports an Index of names. Telling
        # the two apart keeps a GeoDataFrame whose geometry was never activated from being called a raster.
        WebMapBase._reject_raster(features, method)
        columns = getattr(features, "columns", None)
        if columns is not None:
            names = [str(name) for name in columns]
            shown = names[:_MAX_COLUMNS_IN_ERROR]
            if len(names) > _MAX_COLUMNS_IN_ERROR:
                shown.append(f"... (+{len(names) - _MAX_COLUMNS_IN_ERROR} more)")
            raise TypeError(
                f"{method}() needs a layer with an active geometry column; got a "
                f"{type(features).__name__} whose columns are {shown}. Call set_geometry(...) on it first."
            )
        raise TypeError(
            f"{method}() needs a vector layer (a pyramids FeatureCollection / GeoDataFrame); got "
            f"{type(features).__name__}. For a single raster use add_raster(); for a raster time stack "
            f"use timeslider() with a DatasetCollection."
        )

    @staticmethod
    def _json_safe(gdf: Any) -> Any:
        """Return ``gdf`` with date-like columns encoded as ISO-8601 text.

        MapLibre ingests a GeoDataFrame by serialising it to GeoJSON with ``json.dumps``, and GeoJSON has
        no date type — so a ``Timestamp`` column raises ``TypeError: Object of type Timestamp is not JSON
        serializable`` and takes the whole map with it. A timestamp is the norm for the data this tier
        renders (alerts, detections, observations), and ``timeslider`` is hit hardest since the ``kdim`` it
        scrubs is usually the column that breaks the serialisation — so the coercion happens here rather
        than being left to the caller.

        ISO-8601 is the encoding because it is what GeoJSON consumers expect, it still sorts
        lexicographically — which is what keeps :meth:`~digitalearth.web.temporal.TemporalMixin.timeslider`
        ordering its steps correctly — and MapLibre expressions can compare the strings directly. Missing
        values become ``None`` (GeoJSON ``null``) rather than the string ``"NaT"``, which would otherwise
        surface in a popup as though it were a real value.

        Everything GeoJSON cannot represent is covered, because each of these reaches ``json.dumps`` and
        fails there: ``datetime64`` (tz-naive or tz-aware), ``timedelta64`` (an ISO-8601 duration,
        ``P1DT0H0M0S``), ``period`` (its own string form, ``2026-01``), and an ``object`` column holding
        any of those, plus ``datetime.time`` and ``numpy.datetime64``, which the dtype checks miss
        entirely. An object column is encoded when **any** of its values is date-like, not just its first
        — the two orderings of ``["n/a", date(...)]`` are equally unserialisable.

        A mixed column keeps its non-date values as they are, so one GeoJSON property can hold both text
        and an ISO string. That is deliberate: stringifying the rest would destroy the types of numbers
        that serialise perfectly well, and a column mixing dates with other values is already outside what
        a MapLibre expression can compare meaningfully.

        One caveat on ordering: ISO text sorts chronologically only while the UTC offset is constant. A
        tz-aware column spanning a DST change encodes offsets that differ, and lexicographic order then
        diverges from chronological order — convert such a column to UTC before handing it over if
        ``timeslider`` must step through it in true time order.

        Args:
            gdf: The display-CRS GeoDataFrame about to be handed to ``add_source``.

        Returns:
            The same object when it holds no date-like column, else a shallow copy with those columns
            converted. The input is never mutated — it is the caller's frame.

        Examples:
            - A datetime column becomes ISO-8601 text, which ``json`` can serialise:
                ```python
                >>> import geopandas as gpd, pandas as pd
                >>> from shapely.geometry import Point
                >>> from digitalearth.web import WebMap
                >>> gdf = gpd.GeoDataFrame(
                ...     {"when": pd.to_datetime(["2026-01-01 06:00", "2026-02-01 18:30"])},
                ...     geometry=[Point(0, 0), Point(1, 1)], crs=4326,
                ... )
                >>> list(WebMap._json_safe(gdf)["when"])
                ['2026-01-01T06:00:00', '2026-02-01T18:30:00']

                ```
            - A missing timestamp becomes ``None``, so it reaches the map as JSON ``null`` rather than
              the text ``"NaT"``:
                ```python
                >>> import json
                >>> import geopandas as gpd, pandas as pd
                >>> from shapely.geometry import Point
                >>> from digitalearth.web import WebMap
                >>> gdf = gpd.GeoDataFrame(
                ...     {"when": pd.to_datetime(["2026-01-01", "2026-02-01"])},
                ...     geometry=[Point(0, 0), Point(1, 1)], crs=4326,
                ... )
                >>> gdf.loc[1, "when"] = pd.NaT
                >>> json.dumps(WebMap._json_safe(gdf).drop(columns="geometry").to_dict(orient="records"))
                '[{"when": "2026-01-01T00:00:00"}, {"when": null}]'

                ```
            - A frame with nothing date-like is handed back untouched, with no copy taken:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from digitalearth.web import WebMap
                >>> gdf = gpd.GeoDataFrame(
                ...     {"n": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs=4326
                ... )
                >>> WebMap._json_safe(gdf) is gdf
                True

                ```

        See Also:
            digitalearth.web.base.WebMapBase._display_gdf: the choke point that applies this to every
                vector builder's input.
        """
        import datetime as dt

        import numpy as np
        import pandas as pd
        from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

        def missing(value: Any) -> bool:
            """Whether ``value`` is a null marker, tolerating values ``pd.isna`` refuses to judge."""
            try:
                return bool(pd.isna(value))
            except (TypeError, ValueError):  # e.g. a list/ndarray element — not a null marker
                return False

        def date_like(value: Any) -> bool:
            """Whether ``value`` is one of the types GeoJSON cannot represent and we therefore encode."""
            # datetime and Timestamp subclass date; Timedelta subclasses timedelta; time is separate.
            return isinstance(value, (dt.date, dt.time, dt.timedelta, np.datetime64, pd.Period))

        def iso(value: Any) -> Any:
            """ISO-encode one date-like value; anything else is returned untouched."""
            if missing(value):
                return None
            if isinstance(value, dt.timedelta):
                # A plain datetime.timedelta has no isoformat(); only pandas' subclass does.
                return pd.Timedelta(value).isoformat()
            if isinstance(value, np.datetime64):
                return pd.Timestamp(value).isoformat()
            if isinstance(value, (dt.date, dt.time)):
                return value.isoformat()
            if isinstance(value, pd.Period):
                return str(value)
            return value

        def iso_series(series: Any) -> Any:
            """ISO-encode a date-like series as ``object`` dtype, with missing values as real ``None``.

            Built element-wise into an explicit ``object`` Series rather than via ``.map`` / ``.where``:
            pandas 3 infers a ``str`` dtype for those, whose missing marker is a float ``nan``, and
            ``json.dumps`` writes that as the invalid JSON literal ``NaN`` instead of ``null``.
            """
            return pd.Series([iso(value) for value in series], index=series.index, dtype=object)

        def converted(series: Any) -> Any:
            """Return the ISO-encoded form of ``series``, or ``None`` when it needs no change."""
            if (
                is_datetime64_any_dtype(series)
                or is_timedelta64_dtype(series)
                or isinstance(series.dtype, pd.PeriodDtype)
            ):
                return iso_series(series)
            if series.dtype == object and any(date_like(value) for value in series):
                # Scans for *any* date-like value rather than sampling the first non-null one: a column
                # of ["n/a", date(...)] is just as unserialisable as one in the other order, and sampling
                # made the outcome depend on row order. The scan stops at the first hit.
                return iso_series(series)
            return None

        columns = getattr(gdf, "columns", None)
        if columns is None:  # a GeoSeries satisfies the vector guard but has no columns to encode
            return gdf
        geometry_name = getattr(getattr(gdf, "geometry", None), "name", None)
        changed = {}
        for name in columns:
            if name == geometry_name:
                continue
            series = gdf[name]
            if not isinstance(series, pd.Series):
                # A duplicate column label selects a DataFrame; leave it for geopandas to complain about
                # rather than failing here on an attribute the caller never sees.
                continue
            encoded = converted(series)
            if encoded is not None:
                changed[name] = encoded
        if not changed:
            return gdf
        out = gdf.copy()
        for name, series in changed.items():
            out[name] = series
        return out

    def _display_gdf(self, features: Any, *, method: str) -> Any:
        """Reproject a vector input to the display CRS (lon/lat) and return a GeoDataFrame.

        The single vector choke point the point/line/polygon builders call, and so the place the tier
        rejects non-vector input (:meth:`_require_vector`) before any reprojection is paid for, and the
        place date-like columns are made JSON-safe (:meth:`_json_safe`) before the frame reaches
        ``add_source``. A pyramids
        ``FeatureCollection`` *is* a ``geopandas`` GeoDataFrame (the form ``maplibre.Map.add_source`` accepts
        for vector data), so it is reprojected through pyramids (``to_crs``) when needed and returned as-is; a
        bare GeoDataFrame is reprojected via its own ``to_crs``. No shapely/geopandas-as-engine import —
        pyramids owns the reprojection and the GeoDataFrame type.

        Args:
            features: A pyramids ``FeatureCollection`` or a GeoDataFrame.
            method: The calling builder's name, quoted in the guard's error message. Required, so a
                new builder cannot silently inherit a generic label.

        Returns:
            A GeoDataFrame in the display CRS (EPSG:4326 by default), with date-like columns ISO-encoded,
            ready for ``add_source``.

        Raises:
            TypeError: when ``features`` is not a vector layer.

        See Also:
            digitalearth.web.base.WebMapBase._require_vector: the input guard applied first.
            digitalearth.web.base.WebMapBase._json_safe: the date encoding applied to the result.
        """
        self._require_vector(features, method)
        if hasattr(features, "epsg") and hasattr(features, "to_crs"):  # pyramids FeatureCollection (a GeoDataFrame)
            if self._needs_reproject(features):
                features = features.to_crs(self.crs)
            return self._json_safe(features)
        crs_epsg = getattr(getattr(features, "crs", None), "to_epsg", lambda: None)()
        if crs_epsg is not None and crs_epsg != self.crs:  # a bare GeoDataFrame in another CRS
            return self._json_safe(features.to_crs(self.crs))
        return self._json_safe(features)

    def add_underlay(self, layer: Any) -> "WebMapBase":
        """Register ``layer`` at the **bottom** of the stack (drawn first) and return ``self``.

        Basemaps/tiles call this so they sit beneath the data layers regardless of when they are added —
        the mirror of :meth:`add_layer`, which appends on top.

        Args:
            layer: A callable ``apply(widget)`` or a ``maplibre`` ``Layer``/spec.

        Returns:
            This map (chainable).
        """
        self.layers.insert(0, layer)
        return self

    def _auto_cmap(self, source: Any, cmap: Optional[str]) -> str:
        """Resolve a colormap name: the caller's ``cmap`` if given, else the autostyle default.

        Mirrors the interactive tier's ``_auto_cmap`` so a variable looks the same across tiers (the same
        ``digitalearth.base.autostyle`` variable→style lookup, incl. the ECMWF-Magics match); falls back to
        ``"viridis"`` for an unrecognised field.

        Args:
            source: The display-CRS :class:`Source` whose variable drives the lookup.
            cmap: The caller-supplied colormap, or ``None`` to auto-resolve.

        Returns:
            The colormap name to use.
        """
        if cmap is not None:
            return cmap
        from digitalearth.base.autostyle import auto_style

        return auto_style(source).get("cmap", "viridis")

    @staticmethod
    def _cmap_hex(cmap: str, n: int) -> List[str]:
        """Sample ``cmap`` at ``n`` evenly spaced stops and return hex colour strings.

        The colour side of symbology — turning a matplotlib colormap name into the concrete ``#rrggbb``
        strings a MapLibre paint expression needs. matplotlib is imported lazily (only when a builder
        actually colours something), so importing the tier stays engine-free.

        Args:
            cmap: A matplotlib colormap name.
            n: Number of colours to sample (>= 1).

        Returns:
            A list of ``n`` ``#rrggbb`` hex strings spanning the colormap.
        """
        import numpy as np
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        colormap = colormaps[cmap]
        stops = [0.5] if n == 1 else list(np.linspace(0.0, 1.0, n))
        return [to_hex(colormap(s)) for s in stops]

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

    def _build_map_widget(self) -> Any:
        """Build the bare MapLibre ``MapWidget`` with every registered layer applied (no temporal wrap).

        The single map-construction path shared by :meth:`render` (which may wrap it in a time-slider) and
        :meth:`save` (which serialises just the map). An empty map is returned when no layers are registered.

        Returns:
            The configured ``maplibre.ipywidget.MapWidget``.

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

    def render(self) -> Any:
        """Build and return the configured map widget (an empty map if no layers are registered).

        When a time-slider has been added (:meth:`~digitalearth.web.temporal.TemporalMixin.timeslider`), the
        map is wrapped in a slider composite so a notebook front-end renders both together; otherwise the bare
        MapLibre ``MapWidget`` is returned.

        Returns:
            The ``maplibre.ipywidget.MapWidget``, or an ``ipywidgets`` container (slider + map) when temporal.

        Raises:
            ImportError: when the ``web`` extra is not installed.
        """
        widget = self._build_map_widget()
        wrap = getattr(self, "_wrap_temporal", None)
        if wrap is not None and self._temporal is not None:
            return wrap(widget)
        return widget

    def save(
        self,
        path: str,
        *,
        fmt: Optional[str] = None,
        title: str = "Digital-Earth map",
        offline: bool = False,
        **kwargs: Any,
    ) -> str:
        """Save the map — a standalone HTML page or a PNG snapshot — and return ``path`` (DW.6).

        The output kind is ``fmt`` if given, else inferred from the suffix (``.png`` → PNG, otherwise HTML).
        HTML is serialised via ``MapWidget.to_html`` and written as UTF-8 ourselves — sidestepping maplibre's
        cp1252-on-Windows writer bug (see :func:`_patch_maplibre_html_encoding`). By default the page embeds
        the map state and widget JS but references ``maplibre-gl`` from a CDN; ``offline=True`` inlines the
        engine JS/CSS for a fully self-contained page (best-effort, needs network at save time). PNG export is
        delegated to the export mixin and needs a headless browser.

        Args:
            path: Output file (``*.html`` or ``*.png``).
            fmt: Force the format (``"html"`` / ``"png"``); ``None`` infers it from ``path``.
            title: HTML document title.
            offline: When True (HTML only), inline the ``maplibre-gl`` JS/CSS so the page opens offline.
            **kwargs: Forwarded to ``MapWidget.to_html`` (HTML) or the PNG renderer.

        Returns:
            The ``path`` written.

        Raises:
            ImportError: when the ``web`` extra is not installed (or, for PNG, no headless browser is present).
        """
        kind = (fmt or ("png" if str(path).lower().endswith(".png") else "html")).lower()
        if kind == "png":
            return self._render_png(path, title=title, **kwargs)
        html = self._build_map_widget().to_html(title=title, **kwargs)
        if offline:
            html = self._inline_offline_assets(html)
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
