"""WebMapBase — the core MapLibre + deck.gl plumbing the web capability mixins build on.

``WebMapBase`` owns the layer registry (in add order), the display-CRS reproject-through-pyramids
plumbing, and the render/save/show lifecycle around the ``maplibre`` (py-maplibregl) anywidget. Capability
mixins (raster, vector, big-data, 3-D, temporal, decoration, export) live in sibling modules and add
``field()`` / ``choropleth()`` / … builder methods that call ``self.add_layer(...)``; the public
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

import math
import os
import pathlib
from dataclasses import dataclass
from dataclasses import replace as replace_fields
from typing import Any, Dict, List, Optional, Self

from loguru import logger
from pyramids.base.crs import reproject_coordinates

from digitalearth.base.bigdata import (
    DEFAULT_BIG_DATA_THRESHOLD as _SHARED_BIG_DATA_THRESHOLD,
)
from digitalearth.base.crs import OffLimbError, reproject
from digitalearth.base.custom import custom_kind
from digitalearth.base.deprecation import renamed_method
from digitalearth.base.display import (
    auto_cmap,
    needs_reproject,
    to_display_source,
)
from digitalearth.base.registry import (
    KIND_BANDS,
    band_of,
    forget_namespace,
    forget_object,
    kind_info,
    object_namespace,
)
from digitalearth.base.sources.source import Source
from digitalearth.base.spec import (
    Bounds,
    DataRef,
    Encoding,
    FigureSpec,
    Furniture,
    PanelSpec,
    Symbology,
    Viewport,
)
from digitalearth.base.spec.bounds import same_crs
from digitalearth.base.spec.layer import LayerSpec, LayerTree
from digitalearth.base.symbology import sample_cmap

#: The document title an exported page gets when the caller names none. Shared by every export entry
#: point, so a page, a PNG snapshot and an animation frame are titled alike.
DEFAULT_TITLE = "Digital-Earth map"

#: The constructor's zoom when the caller expresses no preference, and the sentinel that says so. A value
#: comparison cannot distinguish an explicit ``zoom=2`` from the default, and passing it is a choice.
_DEFAULT_ZOOM = 2
_UNSET = object()

#: The pip extra / pixi env that provides the MapLibre + deck.gl engine, quoted in the lazy-import error.
_INSTALL_HINT = (
    "the web tier needs MapLibre GL JS + deck.gl (maplibre). "
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

#: The only display CRS that places inline data correctly (see :meth:`WebMapBase._validate_display_crs`),
#: with the spellings a caller may write it as.
DISPLAY_CRS = 4326
_DISPLAY_CRS_SPELLINGS = frozenset(
    {4326, "4326", "EPSG:4326", "epsg:4326", "Epsg:4326"}
)

#: The feature count above which a vector builder auto-routes to a GPU deck.gl layer, unless the map or
#: the call overrides it (see :attr:`WebMapBase.big_data_threshold`).
#: Re-exported from :mod:`digitalearth.base.bigdata` so the tiers cannot drift apart on the number.
DEFAULT_BIG_DATA_THRESHOLD = _SHARED_BIG_DATA_THRESHOLD


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
        if (
            name.startswith("maplibre")
            and getattr(module, "read_internal_file", None) is original
        ):
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


def _is_missing(value: Any) -> bool:
    """Whether ``value`` is a null marker, tolerating values ``pandas.isna`` refuses to judge.

    Args:
        value: One cell of a GeoDataFrame column.

    Returns:
        ``True`` only for a scalar null. A container is a value in its own right — ``pandas.isna``
        judges it element-wise, so a one-element array holding ``NaN`` would otherwise be replaced
        wholesale by ``None``.
    """
    import numpy as np
    import pandas as pd

    if isinstance(value, (np.ndarray, list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_date_like(value: Any) -> bool:
    """Whether ``value`` is one of the types GeoJSON cannot represent and :func:`_iso` therefore encodes.

    Args:
        value: One cell of a GeoDataFrame column.

    Returns:
        ``True`` for a date, time, duration or period in any of its stdlib, numpy or pandas spellings.
    """
    import datetime as dt

    import numpy as np
    import pandas as pd

    # datetime and Timestamp subclass date; Timedelta subclasses timedelta; time is separate.
    return isinstance(
        value,
        (dt.date, dt.time, dt.timedelta, np.datetime64, np.timedelta64, pd.Period),
    )


def _iso(value: Any) -> Any:
    """ISO-encode one date-like value; anything else is returned untouched.

    Args:
        value: One cell of a GeoDataFrame column.

    Returns:
        ``None`` for a missing value, ISO-8601 text for a date-like one, and ``value`` itself for
        anything else — which is what lets a mixed column keep its non-date entries.
    """
    import datetime as dt

    import numpy as np
    import pandas as pd

    if _is_missing(value):
        return None
    if isinstance(value, (dt.timedelta, np.timedelta64)):
        # Neither a plain datetime.timedelta nor a numpy scalar has isoformat(); pandas' wrapper has.
        return pd.Timedelta(value).isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    return value


def _iso_series(series: Any) -> Any:
    """ISO-encode a date-like series as ``object`` dtype, with missing values as real ``None``.

    Built element-wise into an explicit ``object`` Series rather than via ``.map`` / ``.where``: pandas 3
    infers a ``str`` dtype for those, whose missing marker is a float ``nan``, and ``json.dumps`` writes
    that as the invalid JSON literal ``NaN`` instead of ``null``.

    Args:
        series: The column to encode.

    Returns:
        An ``object``-dtype Series carrying the same index.
    """
    import pandas as pd

    return pd.Series(
        [_iso(value) for value in series], index=series.index, dtype=object
    )


def _date_encoded(series: Any) -> Any:
    """Return the ISO-encoded form of ``series``, or ``None`` when it needs no change.

    Args:
        series: One column of the frame about to be serialised.

    Returns:
        The encoded Series, or ``None`` to leave the column exactly as it is.
    """
    import pandas as pd
    from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        # A categorical wraps its real dtype in `.categories`; a repeated timestamp column is routinely
        # stored this way, and it matched none of the checks below.
        dtype = dtype.categories.dtype
    if (
        is_datetime64_any_dtype(dtype)
        or is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    ):
        return _iso_series(series)
    if series.dtype == object and any(_is_date_like(value) for value in series):
        # Scans for *any* date-like value rather than sampling the first non-null one: a column of
        # ["n/a", date(...)] is just as unserialisable as one in the other order, and sampling made the
        # outcome depend on row order. The scan stops at the first hit.
        return _iso_series(series)
    return None


@dataclass(frozen=True)
class _Described:
    """A queue entry standing for a layer the renderer draws from its description.

    The queue is what orders layers into their bands — `_queue_in_band` inserts by the kind's band, and
    `remove_layer` gives the slot back. A converted layer keeps a place in it so that ordering keeps
    working unchanged; what the place holds is a *reference* to the description rather than a closure over
    a compiled MapLibre layer, which is the whole point of the seam.

    Attributes:
        layer_id: The layer this stands for.
    """

    layer_id: str

    @property
    def _digitalearth_layer_id(self) -> str:
        """The id `remove_layer` matches queue entries on.

        Returns:
            The layer's id, so a described layer is removed by the same path as a queued closure.
        """
        return self.layer_id


def draw_custom(web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Hand back the caller's own MapLibre layer, or the callable that adds it.

    A custom layer has no description to rebuild from — that is what `custom:<engine>` means — so the
    object the caller handed in is held by the map and looked up here. A figure loaded from a dict names
    the layer but holds no object, and `held_object` refuses it by name rather than drawing something else.

    Args:
        web_map: The map being drawn.
        _data: Unused — a custom layer's data is inside the caller's own object.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` carrying the object, or `None` when this process
        does not hold it. It carries no source of its own: a caller's layer names sources the caller wired
        up, so there is nothing here to add beside it.

    Raises:
        OffLimbError: when the map is `strict` and this process does not hold the named object — the same
            answer every other unplaceable layer on this tier gets, so one ``except`` clause covers them.
    """
    from digitalearth.base.custom import MissingObject, held_object
    from digitalearth.web.renderer import DrawnLayer

    try:
        obj = held_object(
            layer.id,
            layer.kind,
            web_map._custom,
            engine="maplibre",
            backend="web",
        )
    except MissingObject as error:
        # Contract C7: a figure that names an object this process does not hold is skipped with a warning
        # naming the layer and its engine, and raised under `strict`. The same answer the 3-D tier gives.
        web_map._skipped(layer.id, str(error))
        return None
    return DrawnLayer(source_id=None, source_spec=None, layer=obj)


def _new_renderer(web_map: Any) -> Any:
    """Return the renderer a map draws through.

    Imported here rather than at module scope: :mod:`digitalearth.web.renderer` resolves its drawers from
    the builder modules, and every one of those imports this module, so a top-level import would close a
    cycle.

    Args:
        web_map: The map the renderer serves.

    Returns:
        A :class:`~digitalearth.web.renderer.Renderer` bound to it.
    """
    from digitalearth.web.renderer import Renderer

    return Renderer(web_map)


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


def _layer_is_visible(layer: Any) -> bool:
    """Return whether a caller's own MapLibre layer was built visible.

    MapLibre decides a layer's visibility from ``layout.visibility``, and an absent property means visible —
    which is why no builder writes one for a layer it shows. :meth:`WebMapBase.add_layer` recorded every
    caller's layer as visible whatever the object said, so a layer built hidden was described visible and a
    later ``set_visible(..., True)`` "restored" it to a state it had never been in (review L10).

    Args:
        layer: Whatever was handed to `add_layer` — a ``maplibre`` ``Layer``, a spec dict, or a callable
            ``apply(widget)``, which has no visibility of its own.

    Returns:
        `False` only when the object says ``visibility`` is ``"none"``; `True` otherwise, which is
        MapLibre's own default and the only honest answer for an object that says nothing.
    """
    layout = getattr(layer, "layout", None)
    if layout is None and isinstance(layer, dict):
        layout = layer.get("layout")
    if not isinstance(layout, dict):
        return True
    return layout.get("visibility") != "none"


def as_finite(value: Any, argument: str, caller: str) -> float:
    """Return a builder's numeric keyword as a plain float, refusing one a figure cannot be written with.

    JSON has no spelling for NaN or infinity, so a description holding one cannot be written at all:
    `to_dict` refuses the whole figure. Left to be caught there, the refusal named a MapLibre paint key
    (`Symbology.props['paint']['circle-radius']`) long after the call, for an argument the caller wrote as
    `size=`. `rgb_composite(limits=)` was normalised for exactly this reason; every other numeric keyword
    this tier records was not (review L1). Refusing at the call names what the caller wrote.

    Args:
        value: The keyword's value, as given.
        argument: The keyword's name, as the caller spells it.
        caller: The builder's name, for the message.

    Returns:
        The value as a `float`.

    Raises:
        ValueError: when `value` is NaN or infinite, or is not a number at all — the conversion's own
            failure is re-raised in the builder's words, since an argument named in the signature should
            not surface as a bare `float()` error several frames down.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{caller} needs {argument}= as a finite number; got {value!r}"
        ) from None
    if not math.isfinite(number):
        raise ValueError(
            f"{caller} needs {argument}= as a finite number; got {value!r}. A figure holding it could not "
            f"be written down: JSON has no spelling for NaN or infinity"
        )
    return number


def placed_features(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Return a vector drawer's source placed in the display CRS, ready to serve as GeoJSON.

    Every vector drawer hands its source straight to MapLibre, which reads coordinates as lon/lat and
    reprojects nothing. While a layer recorded the warped frame the builder already held, that was safe by
    construction. Since a layer records the caller's own **path** instead — the one reference a figure can
    be written down with (review H1) — a layer drawn back from its description arrives in whatever CRS the
    file is stored in, and placing it is the drawer's job.

    Args:
        web_map: The map being drawn, whose `crs` the geometry is placed in. `None` leaves `data` alone,
            for a drawer exercised without a map.
        data: The layer's source as the renderer handed it over — already placed on the builder's own
            first draw, freshly opened from the figure's reference on any later one.
        layer: The layer's description, whose kind names the caller in a refusal or a skip warning.

    Returns:
        The frame in the display CRS. One already in it is handed back as it is, so the builder's own
        draw neither warps nor copies anything twice.

    Raises:
        OffLimbError: when the warp places none of the geometry and the map is `strict`.
        TypeError: when the reference opened to something that is not a vector layer — a raster path
            recorded as a vector layer's source.
    """
    if web_map is None:
        return data
    return web_map._display_gdf(data, method=layer.kind)


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


#: The id of the one panel a web map draws into. A map is a single view of a single set of layers, so the
#: figure it writes has exactly one panel; the name is fixed so a stored figure reads the same way every
#: time.
PANEL_ID: str = "main"


class WebMapBase:
    """Core host: ordered layer registry + display CRS + render/save lifecycle over the MapLibre widget.

    Args:
        center: Optional ``(lon, lat)`` map centre, in the display CRS' lon/lat. ``None`` lets MapLibre pick.
        zoom: Initial zoom level.
        style: Basemap style — a short alias (``"dark"``/``"light"``/``"voyager"``), a style URL, or a
            MapLibre style ``dict`` (see :func:`_resolve_style`).
        crs: The EPSG the data is normalised to before it is handed to MapLibre. **EPSG:4326 is the only
            accepted value** and anything else raises: MapLibre ingests GeoJSON and image-source
            coordinates as lon/lat (WGS84) degrees and renders Web Mercator itself, so data normalised to
            any other CRS is handed to the engine as if its metres were degrees and lands somewhere near
            null island. ``"EPSG:4326"`` and ``"4326"`` are accepted spellings and normalise to the int.
        height: Widget height in pixels (``None`` keeps the MapLibre default).
        strict: How a layer that cannot be placed is reported. ``False`` (the default) skips it with a
            warning naming the layer and why, so one unplaceable layer does not lose the rest of the map;
            ``True`` raises instead — always an :class:`~digitalearth.base.crs.OffLimbError`, the same type
            the static, interactive and 3-D tiers raise, so one ``except`` clause covers a pipeline that
            must not ship a silently incomplete map whichever backend drew it.

    Raises:
        ValueError: when ``crs`` is not EPSG:4326 (see :meth:`_validate_display_crs`).

    Attributes:
        layers: What the map draws, in **draw order** — a **derived view**, not the map's state. Draw
            order is the layer tree's, not the order the builders were called in: a graticule sits in the
            reference band and is listed before the data whenever it was added. A layer drawn from its
            description resolves to the ``maplibre`` ``Layer`` the renderer built for it; one still drawn
            the old way is the callable ``apply(widget)`` or ``Layer``/spec that was queued. The map
            itself is described by :attr:`figure_spec` and addressed by id through :attr:`layer_ids`.

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
        - Any other display CRS is refused rather than mis-placing the data:
            ```python
            >>> from digitalearth.web.base import WebMapBase
            >>> try:
            ...     WebMapBase(crs=3857)
            ... except ValueError as err:
            ...     print(str(err).split(";")[0])
            the web tier renders in EPSG:4326 only, but crs=3857 was given

            ```

    See Also:
        digitalearth.web.map.WebMap: the public composition of this base with the capability mixins.
    """

    def __init__(
        self,
        *,
        center: Optional[Any] = None,
        zoom: Any = _UNSET,
        style: Any = "dark",
        crs: Any = DISPLAY_CRS,
        height: Optional[int] = 500,
        strict: bool = False,
    ):
        """Record the display configuration and open the empty layer / decoration registries.

        Every argument is described on the class itself, which is where a reader of the public
        :class:`~digitalearth.web.map.WebMap` meets them; this only validates the display CRS
        and opens the state the builders fill in. Nothing is drawn and nothing is imported from
        MapLibre here — the widget is built on demand by :meth:`render` / :meth:`save` — so a
        map can be composed, and its configuration read back, in an environment with no engine
        installed. It is also why the view is tracked as *whether* one was asked for rather than
        by its value alone: an explicit ``zoom=2`` equals the default, and only the sentinel
        tells the two apart when :meth:`_map_view` decides whether to frame on the data instead.

        Raises:
            ValueError: when ``crs`` is not EPSG:4326 — see :meth:`_validate_display_crs` for
                why a display CRS MapLibre would read as degrees is refused, not mis-placed.
        """
        self.center = center
        #: Whether the caller named a view of their own. Tracked from the sentinel rather than the value,
        #: because an explicit ``zoom=2`` is a choice and equals the default.
        self._view_chosen = center is not None or zoom is not _UNSET
        self.zoom = _DEFAULT_ZOOM if zoom is _UNSET else zoom
        self.style = style
        self.crs = self._validate_display_crs(crs)
        self.height = height
        #: Whether an unplaceable layer raises (``True``) or is skipped with a warning (``False``).
        self.strict = bool(strict)
        self._queued: List[Any] = []
        # Created once and kept: what it holds is what `layers` reports and what `_build_map_widget`
        # adds, so a layer drawn when its builder ran is still there at render time.
        self._renderer = _new_renderer(self)
        #: Monotonic counter handing out unique source/layer ids (see :meth:`_uid`).
        self._id_counter = 0
        #: Every id this map has minted, named or generated. Both allocators reserve from it, so a caller
        #: name shaped like a generated id (``"circle-5"``) cannot later be handed out a second time.
        self._issued_ids: set = set()
        #: Id of the most recently added data layer — the default target for ``popup``/``tooltip``.
        self._last_layer_id: Optional[str] = None
        #: Every data layer added, in draw order — bottom first, as a :class:`~digitalearth.base.spec.layer.LayerTree`
        #: lists them — so a graticule, which joins the reference band beneath the data, sits beneath data added
        #: before it here too. The id addresses the layer in MapLibre; the label is what a layer switcher shows a
        #: viewer. Controls and basemaps are not in here — they are not things a viewer turns on and off.
        self._layer_tree: LayerTree = LayerTree()
        #: Class breaks from the most recent classified ``choropleth``/``points`` (for an out-of-band legend).
        self.last_breaks: Optional[List[float]] = None
        #: Everything :meth:`~digitalearth.web.decoration.DecorationMixin.legend` needs to draw a key for
        #: the most recent classification: its ``kind`` (``"categorical"``/``"graduated"``/``"continuous"``),
        #: the ``column`` it read, the class ``values`` and the ``colors`` actually rendered. Set alongside
        #: :attr:`last_breaks`, which stays the raw-numbers accessor it has always been.
        self.last_legend: Optional[dict] = None
        #: The classification each layer was drawn with, keyed by layer id, so a colour key asked for by id
        #: describes *that* layer. `last_legend` alone answers only "the most recent one", which made
        #: `colorbar("A")` draw layer B's ramp under A's label (review H4).
        self._legends: Dict[str, dict] = {}
        #: The legend dict most recently filed above. A classifying builder always writes a *fresh* dict, so
        #: identity is what tells a new classification from the one still sitting in `last_legend` when an
        #: unclassified layer is indexed after it.
        self._filed_legend: Optional[dict] = None
        #: Accumulated deck.gl JSON layers, applied in one ``add_deck_layers`` call at render (DW.3).
        self._deck_layers: Optional[List[dict]] = None
        #: Feature count above which ``points``/``polygons`` auto-route to a GPU layer (logged, never
        #: silent). Set it once and every subsequent layer on this map honours it; a single builder call
        #: can override it with its own ``big_data_threshold=`` without changing the map's setting.
        self.big_data_threshold = DEFAULT_BIG_DATA_THRESHOLD
        #: What the most recently drawn raster's values are measured in, or ``None``: the caller's
        #: ``field(units=)`` / ``contours(units=)`` when they named one, else the
        #: :func:`~digitalearth.base.autostyle.auto_style` hint. Carried so a key built from that raster's
        #: values can say what they are measured in (see :meth:`_auto_units`).
        self.last_units: Optional[str] = None
        #: Lon/lat extent of everything added so far, unioned as layers arrive (see :meth:`_note_bounds`).
        #: Used to frame the map when the caller gave neither ``center`` nor ``zoom``.
        self._data_bounds: Optional[List[float]] = None
        #: An explicit :meth:`fit_bounds` request, which always wins over the accumulated extent.
        self._fit: Optional[dict] = None
        #: The projection MapLibre draws in, as `globe()` sets it; recorded so the view can say so.
        self._projection: str = "mercator"
        #: Where each layer's data came from, keyed by layer id — the figure's sources (#296).
        self._sources: Dict[str, DataRef] = {}
        #: This map's prefix in the process-global object registry. Every map restarts its layer numbering,
        #: so without one two maps in a session both wrote `circle-2` and the second won (review H2).
        self._objects_ns: str = object_namespace()
        #: The controls the map draws, as furniture on its panel (#292).
        self._furniture: List[Furniture] = []
        #: The panel's title, as `title()` set it.
        self._title: Optional[str] = None
        #: How many layers sit in the basemap band, so :meth:`add_reference` can insert just above them.
        self._underlay_count = 0
        #: How many sit in the reference band, so two of them keep the order they were added in.
        self._reference_count = 0
        #: The engine objects a caller handed to :meth:`add_layer`, keyed by layer id. Held here rather than
        #: in the tree, which describes layers and stores no objects (see :mod:`digitalearth.base.custom`).
        self._custom: Dict[str, Any] = {}
        #: How many sit in the overlay band — text and labels, drawn above the data — so :meth:`add_layer` can
        #: insert beneath them instead of appending on top.
        self._overlay_count = 0
        #: The floating panels — legend and title — as ``{kind: (content, position)}``. Held as state so
        #: calling either twice replaces it rather than stacking two identical boxes in one corner.
        self._panels: dict = {}
        #: The caller's ``layer_control`` request, resolved against the live layers when the widget is
        #: built. Held as state rather than appended to :attr:`layers` so that removing a layer cannot
        #: strand a dead row in a saved page, and so an export can leave the control out entirely.
        self._switcher: Optional[dict] = None
        #: Time-slider config set by ``timeslider`` (``None`` = no temporal control); read by ``render``.
        #: ``mode`` selects the wiring: ``"vector"`` carries ``layer_id`` and filters one layer by
        #: ``kdim``; ``"raster"`` carries ``layer_ids`` and swaps their visibility. Both carry ``times``.
        self._temporal: Optional[dict] = None

    @staticmethod
    def _validate_display_crs(crs: Any) -> int:
        """Return ``crs`` normalised to ``4326``, raising for any other display CRS.

        The tier's display CRS is *declared*, not assumed. MapLibre places a GeoJSON source and an image
        source by lon/lat degrees, so data normalised to anything else is read as degrees anyway: a
        Web-Mercator frame whose easting is 500 000 m is drawn 500 000 degrees east, which MapLibre wraps
        to somewhere meaningless rather than reporting. Refusing the value is the only way a caller finds
        out, so a non-4326 ``crs`` is an error here instead of a wrong map later.

        Args:
            crs: The display CRS as the caller wrote it — ``4326``, ``"EPSG:4326"`` or ``"4326"``.

        Returns:
            ``4326``, whatever spelling arrived, so the reproject helpers can compare it as an int.

        Raises:
            ValueError: for any other CRS, naming what was passed and what to do instead.
        """
        if crs in _DISPLAY_CRS_SPELLINGS:
            return DISPLAY_CRS
        raise ValueError(
            f"the web tier renders in EPSG:{DISPLAY_CRS} only, but crs={crs!r} was given; MapLibre "
            f"places GeoJSON and image sources by lon/lat degrees, so data in another CRS would be "
            f"drawn in the wrong place. Leave crs at its default and let the tier reproject through "
            f"pyramids, or use the static tier, which renders any projection."
        )

    def _skipped(
        self, layer: str, reason: str, error: Optional[Exception] = None
    ) -> None:
        """Skip a layer that cannot be placed — warn and carry on, or re-raise under ``strict``.

        The tier's one answer to "there is nothing renderable here": data behind the limb of the display
        CRS, corners that will not express as lon/lat, a contour trace that found no level. One
        unplaceable layer used to lose the whole map, which is the wrong default for a builder chain that
        may have a dozen layers in it — but a pipeline that must not publish a half-drawn map sets
        ``strict=True`` on the map and gets the exception back.

        Args:
            layer: The public builder that drew nothing, named in the log line.
            reason: Why it drew nothing, in the same line.
            error: The exception behind it, re-raised verbatim under ``strict``; ``None`` raises an
                :class:`~digitalearth.base.crs.OffLimbError` carrying ``reason``.

        Raises:
            Exception: ``error`` when the map is ``strict`` and one was given, else an
                :class:`~digitalearth.base.crs.OffLimbError` built from ``reason``. The reasons this tier
                skips for without an exception behind them — corners that will not express as lon/lat, a
                contour trace with no level in range, a time step that cannot be placed — are the same
                "there is nothing renderable here" the other three tiers signal with ``OffLimbError``, so
                they raise it too: ``except OffLimbError`` around a strict map catches every tier's answer,
                which is the whole point of a shared policy. Note it derives from ``RuntimeError``, so an
                ``except ValueError`` written against the old behaviour no longer catches these.
        """
        if self.strict:
            raise error if error is not None else OffLimbError(f"{layer}: {reason}")
        logger.warning("{}: {} — the layer was skipped", layer, reason)

    def _display_source_or_skip(
        self, data: Any, *, layer: str, band: int = 1
    ) -> Optional[Source]:
        """Return :meth:`_to_display_source`'s result, or ``None`` when the data cannot be placed.

        Args:
            data: A pyramids ``Dataset`` / ``Source`` / array, as :meth:`_to_display_source` takes it.
            layer: The calling builder's name, quoted in the warning.
            band: 1-based band to extract.

        Returns:
            The display-CRS :class:`Source`, or ``None`` when the warp placed none of the data and the
            map is not ``strict``.

        Raises:
            OffLimbError: when the data lies outside the display CRS and the map is ``strict``.
        """
        try:
            return self._to_display_source(data, band=band)
        except OffLimbError as error:
            self._skipped(layer, str(error), error)
            return None

    def _display_raster_or_skip(self, dataset: Any, *, layer: str) -> Optional[Any]:
        """Return :meth:`_to_display_raster`'s result, or ``None`` when the data cannot be placed.

        Args:
            dataset: A pyramids ``Dataset``.
            layer: The calling builder's name, quoted in the warning.

        Returns:
            The dataset in the display CRS, or ``None`` when it is off-limb and the map is not ``strict``.

        Raises:
            OffLimbError: when the data lies outside the display CRS and the map is ``strict``.
        """
        try:
            return self._to_display_raster(dataset)
        except OffLimbError as error:
            self._skipped(layer, str(error), error)
            return None

    def _noted(self, gdf: Any) -> Any:
        """Record a display-CRS frame's extent, then return the frame unchanged.

        Wraps the returns of :meth:`_display_gdf` so every vector builder contributes to the map's framing
        without knowing that it does.

        Args:
            gdf: A GeoDataFrame already in the display CRS.

        Returns:
            The same object, so this can wrap a return expression.
        """
        self._note_bounds(getattr(gdf, "total_bounds", None))
        return gdf

    def _note_lonlat_bounds(self, bounds: Any) -> None:
        """Union an extent that is already lon/lat into the running data extent.

        Args:
            bounds: ``(west, south, east, north)`` in degrees.
        """
        self._note_bounds(bounds, already_lonlat=True)

    def _note_bounds(self, bounds: Any, *, already_lonlat: bool = False) -> None:
        """Union a layer's lon/lat extent into the running data extent.

        Called by the builders' shared choke points — :meth:`_display_gdf` for vector data and
        ``field`` for rasters — so a new builder inherits framing without doing anything.

        Args:
            bounds: ``(west, south, east, north)`` in the **display CRS**, which is converted to lon/lat
                here when that is not already what it is — ``fitBounds`` takes degrees, and ``WebMap(crs=)``
                is a supported public setting. ``None`` is ignored, as is a non-finite value: an empty
                frame reports ``inf`` bounds, and one unusable layer must not decide where the map looks.
            already_lonlat: Set by callers that hold degrees already — the raster builders, whose corners
                have to be lon/lat anyway because that is how MapLibre places an image source.
        """
        if bounds is None:
            return
        try:
            west, south, east, north = (float(value) for value in bounds)
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(value) for value in (west, south, east, north)):
            return
        if not already_lonlat:
            converted = self._as_lonlat(west, south, east, north)
            if converted is None:
                return
            west, south, east, north = converted
        if self._data_bounds is None:
            self._data_bounds = [west, south, east, north]
            return
        current = self._data_bounds
        self._data_bounds = [
            min(current[0], west),
            min(current[1], south),
            max(current[2], east),
            max(current[3], north),
        ]

    def _as_lonlat(
        self, west: float, south: float, east: float, north: float
    ) -> Optional[tuple]:
        """Convert a display-CRS extent to the lon/lat degrees ``fitBounds`` expects.

        Args:
            west: Western edge in the display CRS.
            south: Southern edge.
            east: Eastern edge.
            north: Northern edge.

        Returns:
            The extent in lon/lat, or ``None`` when it cannot be converted — an unknown extent must not
            frame the map on the wrong place, and silently feeding metres to ``fitBounds`` did exactly
            that.
        """
        if self.crs is None or same_crs(self.crs, 4326):
            return west, south, east, north
        try:
            xs, ys = reproject_coordinates(
                [west, east], [south, north], from_crs=self.crs, to_crs=4326
            )
        except (ValueError, RuntimeError):
            return None
        values = (float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1]))
        return values if all(math.isfinite(v) for v in values) else None

    def set_bounds(
        self,
        bounds: Optional[Any] = None,
        *,
        padding: Any = 20,
        animate: bool = False,
    ) -> Self:
        """Frame the map on ``bounds``, or on everything added so far.

        Without this the view comes only from the ``center``/``zoom`` given to the constructor, so a map of
        one country opened on the whole world unless the caller worked out a centre themselves.

        Args:
            bounds: ``(west, south, east, north)`` in lon/lat. ``None`` uses the extent of the data added
                so far, which is what most callers want and is applied automatically anyway (see
                :meth:`_map_view`).
            padding: Breathing room left around the extent — a number of pixels, or MapLibre's
                per-edge mapping (``{"top": 40, "bottom": 10, ...}``).
            animate: Whether the browser eases into the new view rather than jumping.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``bounds`` is ``None`` and nothing with an extent has been added yet — there
                is nothing to frame on, and silently doing nothing would look like the call was ignored.

        Examples:
            - Frame on an explicit box:
                ```python
                >>> from digitalearth.web import WebMap                     # doctest: +SKIP
                >>> WebMap().basemap().set_bounds((4.0, 51.0, 7.0, 54.0))  # doctest: +SKIP

                ```
        """
        if bounds is None:
            if self._data_bounds is None:
                raise ValueError(
                    "set_bounds() has nothing to frame on: no layer with an extent has been added yet. "
                    "Add the data first, or pass bounds=(west, south, east, north)."
                )
            bounds = list(self._data_bounds)
        values = [float(value) for value in bounds]
        if len(values) != 4:
            raise ValueError(
                f"set_bounds(bounds=...) takes (west, south, east, north); got {len(values)} values"
            )
        west, south, east, north = values
        self._fit = {
            "bounds": [west, south, east, north],
            # Not int(): MapLibre also accepts {top, bottom, left, right}, and casting turns that into a
            # TypeError about int() rather than about the argument.
            "padding": padding if isinstance(padding, dict) else int(padding),
            "animate": bool(animate),
        }
        return self

    #: Deprecated spelling of :meth:`set_bounds`, the contract's name for framing a figure on a region (#299).
    fit_bounds = renamed_method(new="set_bounds", old="fit_bounds", owner="WebMap")

    def _switcher_request(self) -> Optional[dict]:
        """Return the layer switcher to add to the widget being built, or ``None`` for no switcher.

        Resolved here rather than when ``layer_control()`` was called, because the set of layers can change
        afterwards: a row naming a removed layer survives into the saved page, where clicking it logs an
        error and toggles nothing.

        Returns:
            The control's ``layer_ids``, ``theme`` and ``position``, or ``None``.
        """
        request = self._switcher
        if request is None:
            request = self._temporal_switcher()
        if request is None:
            return None
        live = [i for i in request["layer_ids"] if i in set(self.layer_ids)]
        if len(live) < request.get("minimum", 1):
            return None
        return {**request, "layer_ids": live}

    def _temporal_switcher(self) -> Optional[dict]:
        """Return the automatic step picker for a temporal map, if this map is one.

        Overridden in :mod:`digitalearth.web.temporal`; the base map has no time dimension.

        Returns:
            ``None``.
        """
        return None

    def _map_view(self) -> Optional[dict]:
        """Return the framing to apply to the built widget, or ``None`` to leave the view alone.

        An explicit :meth:`set_bounds` always wins. Otherwise the accumulated data extent is used, but only
        when the caller expressed no view of their own — passing ``center`` or a non-default ``zoom`` is
        taken as "I have chosen the view", and it is not overridden.

        Returns:
            The ``fit_bounds`` keyword arguments MapLibre takes, or ``None``.
        """
        if self._fit is not None:
            return self._fit
        bounds = self._data_bounds
        if bounds is not None and bounds[0] == bounds[2] and bounds[1] == bounds[3]:
            return None  # a single point is not an extent; framing on it means maximum zoom
        if self._view_chosen or self._data_bounds is None:
            return None
        return {"bounds": list(self._data_bounds), "padding": 20, "animate": False}

    @property
    def layer_ids(self) -> List[str]:
        """The MapLibre ids of the data layers added so far, in draw order.

        The registry used to be write-only: builders minted ids internally and nothing surfaced them, so a
        caller could not address a layer afterwards to hide, remove or switch it.

        Returns:
            The layer ids, bottom first, as a new list: the order they were added in, except that a graticule,
            which is drawn in the reference band beneath the data, is listed before every layer that is not a
            graticule, whenever it was added. A layer the caller named carries that name as its id; an unnamed one
            gets a generated id. A generated id counts the layers of its kind: a layer no longer takes a
            second number for the MapLibre source beside it, because that source is named after the layer
            (#296) rather than allocated next to it.

        Examples:
            - Named and unnamed layers, in the order they were added:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().text(4.9, 52.4, "Amsterdam", name="amsterdam").text(2.35, 48.86, "Paris")
                >>> m.layer_ids
                ['amsterdam', 'text-1']

                ```
            - A graticule added last is listed first, because it is drawn beneath the data:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().text(4.9, 52.4, "Amsterdam", name="amsterdam").graticule().layer_ids
                ['Graticule', 'amsterdam']

                ```
            - A map with no data layers has no ids:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().layer_ids
                []

                ```
        """
        return list(self._layer_tree.ids)

    @property
    def layers(self) -> List[Any]:
        """What the map draws, in draw order — the view this tier has always exposed.

        Returns:
            One entry per drawn layer: the MapLibre ``Layer`` for a layer the renderer built from its
            description, and the queued ``apply(widget)`` callable or spec for one still drawn the old way.

        Note:
            This is a **compatibility view** of what is drawn, not the map's state. The map is described by
            :attr:`figure_spec`, and layers are addressed by id through :attr:`layer_ids`,
            :meth:`get_layer` and :meth:`remove_layer`; a viewer shows and hides them with the switcher
            :meth:`~digitalearth.web.decoration.DecorationMixin.layer_control` adds. It is kept because it is
            the shape callers and this tier's own tests have always read — the same reason the 3-D tier kept
            its `(mesh, actor)` pairs after its seam.

        Examples:
            - A layer the renderer drew is reported as the MapLibre layer itself, under the id it was
              given:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().text(4.9, 52.4, "Amsterdam", name="amsterdam")
                >>> [drawn.id for drawn in m.layers]
                ['amsterdam']

                ```
            - Draw order, not call order: a graticule added last is still drawn beneath the data it
              references, and this reports it first:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().text(2.35, 48.86, "Paris", name="paris").graticule(spacing=30)
                >>> [drawn.id for drawn in m.layers]
                ['Graticule', 'paris']

                ```
        """
        drawn = self._renderer.drawn
        resolved = []
        for entry in self._queued:
            if isinstance(entry, _Described):
                built = drawn.get(entry.layer_id)
                if built is not None:
                    resolved.append(built.layer)
                continue
            resolved.append(entry)
        return resolved

    @property
    def figure_spec(self) -> FigureSpec:
        """What the map draws, as data: its layers, their sources, the view, and the panel's furniture.

        Returns:
            A :class:`~digitalearth.base.spec.FigureSpec` with one panel, `"main"`. Its layers are the tree in
            draw order, its sources are what each builder was given, and its view is a
            :class:`~digitalearth.base.spec.Viewport` in EPSG:4326 — the CRS the tier places data in — carrying
            the centre, zoom and fitted bounds the map was asked for. The panel also holds the title and the
            furniture: the navigation, scale-bar, fullscreen, measure, layer-switcher and time-slider controls,
            each under the name #292 registers it with.

            A map built from a GeoDataFrame or a dataset already in memory can be handed straight to
            :class:`~digitalearth.web.renderer.Renderer`, but not written: `to_dict()` refuses an `object:`
            source, since a reference into this process's memory would be unreadable everywhere else. Give a
            builder a path or a URL to get a figure that can be stored.

        Examples:
            - A map describes what it drew, and where it is looking:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from digitalearth.web import WebMap
                >>> points = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326)
                >>> figure = WebMap(center=(4.9, 52.4), zoom=7).points(points, name="obs").figure_spec
                >>> figure.layers.get("obs").kind, figure.panels[0].view.zoom
                ('points', 7.0)

                ```
            - Controls are the panel's furniture, not layers:
                ```python
                >>> from digitalearth.web import WebMap
                >>> panel = WebMap().navigation().scale_bar().figure_spec.panels[0]
                >>> sorted(item.kind for item in panel.furniture)
                ['navigation', 'scale_bar']

                ```
        """
        tree = self._layer_tree
        panel = PanelSpec(
            PANEL_ID,
            self.viewport,
            layers=tuple(tree.ids),
            title=self._title,
            furniture=tuple(self._furniture),
        )
        return FigureSpec(
            panels=(panel,),
            layers=tree,
            sources={
                key: ref for key, ref in self._sources.items() if key in set(tree.ids)
            },
        )

    @property
    def viewport(self) -> Viewport:
        """Where the map is looking, as a value.

        Returns:
            A :class:`~digitalearth.base.spec.Viewport` in the CRS the tier places data in (EPSG:4326, which
            is what `WebMap(crs=)` accepts), carrying the `center` and `zoom` the map was built with and the
            bounds an explicit :meth:`set_bounds` asked for. The projection MapLibre draws in is a property of
            the style rather than of the data, which is why it is not this CRS.

        Examples:
            - What a map was built to look at is readable as a value:
                ```python
                >>> from digitalearth.web import WebMap
                >>> view = WebMap(center=(4.9, 52.4), zoom=7).viewport
                >>> view.center, view.zoom, view.crs
                ((4.9, 52.4), 7.0, 4326)

                ```
            - A region asked for with `set_bounds` frames the view:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().set_bounds([3.0, 50.0, 7.0, 54.0]).viewport.bounds.as_bbox()
                [3.0, 50.0, 7.0, 54.0]

                ```
        """
        fitted = self._fit if isinstance(self._fit, dict) else {}
        box = fitted.get("bounds")
        bounds = (
            Bounds(box[0], box[1], box[2], box[3], crs=self.crs)
            if box is not None
            else None
        )
        return Viewport(
            self.crs,
            bounds=bounds,
            globe=self._projection == "globe",
            center=tuple(self.center) if self.center is not None else None,
            zoom=float(self.zoom) if self.zoom is not None else None,
        )

    def _as_display_point(self, x: float, y: float, crs: Any) -> tuple:
        """Return one coordinate pair in the CRS this tier places data in.

        Args:
            x: The x coordinate, in `crs`.
            y: The y coordinate, in `crs`.
            crs: What they are measured in.

        Returns:
            `(lon, lat)` in the display CRS. A point already in it is returned unchanged, so the common call
            costs nothing; anything else goes through pyramids, as every other placement in this tier does.

        Raises:
            ValueError: when the point cannot be expressed in the display CRS.
        """
        if crs is None or same_crs(crs, self.crs):
            return x, y
        # A point is a rectangle with no width: `Bounds.to_crs` is the tier's own reprojection, through
        # pyramids, and using it keeps geopandas out of this module (the import guard, and the rule that
        # pyramids owns every CRS operation).
        moved = Bounds(x, y, x, y, crs=crs).to_crs(self.crs)
        return float(moved.xmin), float(moved.ymin)

    def _record_furniture(
        self, kind: str, *, anchor: Any = None, **options: Any
    ) -> None:
        """Record a control the map draws, as furniture on its panel (#292).

        Calling a control builder twice records one item: MapLibre would add a second control in the same
        corner, which is a caller repeating themselves rather than asking for two.

        Args:
            kind: The registered furniture name — `"navigation"`, `"scale_bar"`, ….
            anchor: The corner it sits in; `None` takes the registered default.
            **options: What the control was built with, for a renderer that draws it from the description.
        """
        item = Furniture(kind, anchor=anchor, options=options)
        self._furniture = [held for held in self._furniture if held.kind != kind]
        self._furniture.append(item)

    def _forget_furniture(self, kind: str) -> None:
        """Forget a control the map no longer draws.

        Args:
            kind: The registered furniture name.
        """
        self._furniture = [held for held in self._furniture if held.kind != kind]

    def _record_tooltip(
        self, layer_id: str, fields: Any, *, trigger: str, explicit: bool = False
    ) -> None:
        """Record what a popup or tooltip shows, as a channel on the layer it is bound to (#292).

        Per-layer interaction is an encoding, not a separate object: put it on the layer and it travels with
        it, so removing the layer removes what pops up over it — which is what left a page emitting
        `addPopup` for a layer that was no longer there.

        A MapLibre layer is not always a described one. A cluster draws three — the bubbles, their counts
        and the loose points — under one tree entry; a graticule draws its labels beside its lines; a basemap
        is a layer the tier never indexes. A popup over any of those is a real popup, so it is drawn and
        simply not described: the alternative was a `KeyError` that broke `cluster(...).popup(...)`
        outright (review H3).

        A *named* layer is different. `popup(fields, layer="typoed")` is a caller naming something, and
        skipping the description for it accepted the typo in silence while still queueing the closure — so
        the saved page called `map.on('click', 'typoed', ...)` and failed in a browser console instead
        (review H7). An id the caller wrote is checked; an id the tier chose is not.

        Args:
            layer_id: The layer the popup or tooltip is bound to.
            fields: The field names shown; `None`, like an empty list, means every field, which is what the
                builders already mean by it.
            trigger: `"click"` for a popup, `"hover"` for a tooltip.
            explicit: Whether the caller named the layer, rather than it defaulting to the last one drawn.

        Raises:
            KeyError: when the caller named a layer this map has not drawn.
        """
        if layer_id not in self._layer_tree.ids:
            if explicit:
                raise KeyError(
                    f"no layer {layer_id!r} on this map; its layers are {self.layer_ids}"
                )
            return
        held = self._layer_tree.get(layer_id)
        shown = () if fields is None else tuple(fields)
        # A click popup and a hover tooltip are two interactions, and a layer may carry both. Writing one
        # `tooltip` encoding meant the second call replaced the first: `popup(["v"]).tooltip(["w"])` drew
        # both and described only the hover, losing the popup's own fields (review M10). What each trigger
        # shows is kept beside the channel; the channel itself carries the hover's fields where there is
        # one, since that is the reading a renderer with a single hover slot needs.
        bound = {
            **dict(held.symbology.props.get("interactions", {})),
            trigger: list(shown),
        }
        primary = "hover" if "hover" in bound else trigger
        symbology = Symbology(
            encodings={
                **dict(held.symbology.encodings),
                "tooltip": Encoding.constant("tooltip", tuple(bound[primary])),
            },
            props={**dict(held.symbology.props), "interactions": bound},
        ).with_props(tooltip_trigger=primary)
        self._layer_tree = self._layer_tree.replace(
            replace_fields(held, symbology=symbology)
        )

    def _forget_slider_frame(self, layer_id: str) -> None:
        """Drop a removed layer from the time slider, and the slider itself when nothing is left to step.

        Args:
            layer_id: The layer being removed.

        Note:
            A slider that still named a removed layer filtered a layer that was not there — the page kept a
            control that did nothing, which is the dangling reference this seam is meant to end.
        """
        slider = next(
            (item for item in self._furniture if item.kind == "time_slider"), None
        )
        if slider is None:
            return
        named = slider.options.get("layers", ())
        if layer_id == slider.options.get("layer") or tuple(named) == (layer_id,):
            self._forget_furniture("time_slider")
            return
        if layer_id in named:
            # The frame at the same position goes with it. Dropping the layer alone left a slider with more
            # stops than layers — three frames for two layers — so a renderer drawing the control from the
            # description stepped onto a frame that names nothing (review M9).
            kept = [index for index, held in enumerate(named) if held != layer_id]
            frames = tuple(slider.options.get("frames", ()))
            self._record_furniture(
                "time_slider",
                anchor=slider.anchor,
                **{
                    **dict(slider.options),
                    "layers": tuple(named[index] for index in kept),
                    **(
                        {"frames": tuple(frames[index] for index in kept)}
                        if len(frames) == len(named)
                        else {}
                    ),
                },
            )

    def _index_layer(
        self,
        layer_id: str,
        label: Optional[str],
        *,
        kind: str,
        visible: bool = True,
        band: Optional[str] = None,
        source: Any = None,
        symbology: Any = None,
        placed: Any = None,
    ) -> bool:
        """Record a data layer so it can be addressed later.

        Args:
            layer_id: The MapLibre layer id.
            label: What a layer switcher should call it; `None` falls back to the id.
            kind: What sort of layer it is — a registered, engine-neutral kind such as `"raster"`,
                `"choropleth"` or `"points"` (:func:`~digitalearth.base.registry.kinds`), not the MapLibre layer
                type — so the tree describes the layer rather than only naming it.
            visible: Whether the layer was built visible. A builder that takes `visible=` passes it on, so the
                tree says what the MapLibre layout says; the builders without one always build visible. It is
                recorded by truthiness, as the builders decide the layout by it: `visible=0` draws a hidden layer
                and records one.
            band: Where the layer is drawn, when its kind does not say — what a caller's own object needs,
                since `custom:maplibre` names the engine rather than what it draws. `None` takes the kind's band.
            source: What the layer draws, recorded in the figure's sources under the layer's id — a path, a
                URL, or the object itself. `None` for a layer drawn from no data, such as a graticule.
            symbology: How the layer looks, as values rather than as the compiled engine expression. `None`
                records an empty symbology.
            placed: `source` as the builder already holds it in the display CRS, handed to the first draw so
                the drawer does not warp the same data again (review M8). What the figure records is still
                `source`, the caller's own data, so a redraw from the figure places it for itself. `None` has
                the first draw open `source` too.

        Returns:
            `True` when the layer is registered. `False` when its drawer declined to draw it — an
            unplaceable raster, an empty collection — in which case nothing is left behind and the caller
            must not treat it as the map's last layer.

        Note:
            The tree places the layer in the band its kind declares, so nothing here computes a position.
            For a kind in :data:`~digitalearth.web.renderer.DRAWN_KINDS` this then **draws it too**, from
            the description just recorded, and queues a reference to it in that same band — the builder has
            nothing left to queue. A kind the renderer does not own yet is still drawn by its builder, which
            queues the MapLibre closure itself — with :meth:`_queue` or one of the band registrations — into
            the same band. Either way the tree's order stays the order the map draws in.

        Raises:
            KeyError: when `kind` is not a registered layer kind — a builder passing a name the registry cannot
                look up, which no caller input reaches.
            ValueError: when `LayerSpec` refuses the id (an empty string) or the kind, or when the id is already in
                the tree. A builder reaches none of these: its `name=` becomes the id as given, surrounding
                whitespace included, and `_layer_id` generates an id for an empty name and suffixes a repeated one.
                Also when a drawn kind's `symbology` carries none of the props its drawer reads, which a
                description built by hand can miss.
            OffLimbError: when the map is `strict` and the drawer found nothing it could place. Without
                `strict` the layer is skipped with a warning and `False` comes back instead.
            Exception: whatever else a drawer raises, after the layer has been forgotten again (see
                :meth:`_forget_layer`) — a figure must not name a layer that was not drawn, and that is as
                true of a refusal as it is of a skip.
        """
        kind_info(kind)
        # Whatever this builder classified belongs to the layer it is indexing. Captured here because this
        # is the one place every builder passes its own id, right after drawing. By identity, so the
        # classification of an earlier layer is not filed again under a later unclassified one.
        if self.last_legend is not None and self.last_legend is not self._filed_legend:
            self._legends[layer_id] = self.last_legend
            self._filed_legend = self.last_legend
        if source is not None:
            # Namespaced per map: every `WebMap` restarts its layer numbering, so two maps in one session
            # both minted `circle-2` and the second registration replaced the first — map A's stored figure
            # then described map B's data (review H2).
            self._sources[layer_id] = DataRef.of(
                source, name=f"{self._objects_ns}:{layer_id}"
            )
        self._layer_tree = self._layer_tree.add(
            # By truthiness, as every builder decides the MapLibre layout: `LayerSpec` takes only a real boolean,
            # and handing it `visible=0` turned a call that built a hidden layer into a ValueError.
            LayerSpec(
                layer_id,
                kind,
                source_id=layer_id if source is not None else None,
                symbology=Symbology() if symbology is None else symbology,
                label=label or layer_id,
                visible=bool(visible),
                band=band,
            )
        )
        # Drawn as it is recorded, for the kinds the renderer owns: what it holds is what `layers` reports
        # and what the widget is built from, so a description and its drawing cannot drift apart.
        from digitalearth.web.renderer import DRAWN_KINDS

        if kind in DRAWN_KINDS:
            try:
                drawn = self._renderer.draw_layer(
                    self.figure_spec, layer_id, opened=placed
                )
            except BaseException:
                # A refusal — `strict`, a description its drawer cannot read — is the same drift as a
                # decline, and takes the record back the same way before it reaches the caller.
                self._forget_layer(layer_id)
                raise
            if drawn is None:
                # The drawer declined — an unplaceable raster, an empty collection — and has said why.
                # A described layer nothing draws is exactly the drift this seam removes, so the record
                # goes with the drawing: the caller sees a map that never registered it.
                self._forget_layer(layer_id)
                return False
            # In the layer's *own* band when it declares one — what a caller's `custom:maplibre` object
            # needs, since the engine name says nothing about what it draws — else the kind's.
            self._queue_in_band(_Described(layer_id), band or band_of(kind))
        return True

    def _forget_layer(self, layer_id: str) -> None:
        """Take back everything :meth:`_index_layer` recorded for a layer its drawer did not draw.

        What `_index_layer` calls when a drawer declines or raises. Dropping the tree entry and the source
        reference alone left the rest behind (review M6): the object stayed in the process-global registry
        for the life of the process, and the id stayed issued, so a caller who retried the same `name=` got
        `dem-2` — which a layer switcher shows the viewer. The classification the builder filed under the
        id goes too, since a later layer may now be given the same id. This is what the static tier's
        `Scene._forget_layer` does for the same two paths.

        Unlike :meth:`remove_layer`, the registered object *is* forgotten here: a layer that was never
        drawn was never in a figure this map reported, so no captured `figure_spec` can name it.

        Args:
            layer_id: The layer to forget. Parts it never recorded are skipped, so this is safe to call
                however far `_index_layer` got.
        """
        from digitalearth.web.renderer import derived_ids

        derived: tuple = ()
        if layer_id in self._layer_tree.ids:
            derived = derived_ids(self._layer_tree.get(layer_id).kind, layer_id)
            self._layer_tree = self._layer_tree.remove(layer_id)
        self._sources.pop(layer_id, None)
        # The key `_index_layer` registered under, rather than whatever reference the builder was given:
        # a path, or a `DataRef` the caller registered themselves, is not this map's to forget.
        forget_object(f"{self._objects_ns}:{layer_id}")
        self._custom.pop(layer_id, None)
        # The ids its drawer would have derived were reserved with it (review M5), and go back with it.
        self._issued_ids.difference_update((layer_id, *derived))
        self._forget_legend(layer_id)

    def _forget_legend(self, layer_id: str) -> None:
        """Drop the classification filed under a layer, and stop describing it as the most recent one.

        Args:
            layer_id: The layer whose key goes.

        Note:
            When the dropped key was `last_legend`, the most recent surviving layer's key takes its place,
            or none. `_filed_legend` follows it: it is how `_index_layer` tells a fresh classification from
            the one already filed, and left pointing at the dropped key, the next *unclassified* layer was
            filed under the survivor's key and drew a colour key it was never classified with.
        """
        dropped = self._legends.pop(layer_id, None)
        if dropped is None or dropped is not self.last_legend:
            return
        # The keyed path forgets it; the default path read `last_legend`, which was cleared only when the
        # map went empty — so `legend()` kept drawing the key of a layer that was gone (review M5). The most
        # recent surviving classification takes its place, or none.
        surviving = [
            self._legends[held] for held in self.layer_ids if held in self._legends
        ]
        self.last_legend = surviving[-1] if surviving else None
        self._filed_legend = self.last_legend
        if not surviving:
            self.last_breaks = None

    def get_layer(self, layer_id: str) -> LayerSpec:
        """Return the description of one layer, by id.

        Args:
            layer_id: The layer to look up.

        Returns:
            Its :class:`~digitalearth.base.spec.LayerSpec` — kind, source, symbology, band and visibility.

        Raises:
            KeyError: if no layer has that id, naming the ids that do.

        Examples:
            - What a builder recorded, read back by id:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from digitalearth.web import WebMap
                >>> points = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326)
                >>> WebMap().points(points, name="obs").get_layer("obs").kind
                'points'

                ```
        """
        present = self.layer_ids
        if layer_id not in present:
            raise KeyError(
                f"no layer {layer_id!r} on this map; added layers are {present}"
            )
        return self._layer_tree.get(layer_id)

    def colorbar(
        self,
        layer_id: Optional[str] = None,
        *,
        label: Optional[str] = None,
        visible: bool = True,
    ) -> Self:
        """Show the continuous colour key of a layer.

        The contract's name for a colour key (#299, #261). On this tier the key is drawn by :meth:`legend`,
        which builds a panel from what a layer's classification recorded; a continuous ramp is that panel with
        the ramp's ends labelled, which is why this is a thin call onto it rather than a second mechanism.

        Args:
            layer_id: Which layer's key to show — the classification *that* layer was drawn with. `None`
                takes the most recently classified layer, which is what the tier recorded before layers had
                ids.
            label: What to call the key — the variable and its units, usually.
            visible: `False` draws no key, so a caller passing a flag through does not have to branch.

        Returns:
            This map (chainable).

        Raises:
            KeyError: if `layer_id` names no layer on this map.
            ValueError: when the named layer carries no classification to describe, or — for `None` — when
                no layer on the map does, as :meth:`legend` raises.

        Examples:
            - A classified layer's key, titled:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Polygon
                >>> from digitalearth.web import WebMap
                >>> squares = gpd.GeoDataFrame(
                ...     {"pop": [1, 9]},
                ...     geometry=[
                ...         Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                ...         Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                ...     ],
                ...     crs=4326,
                ... )
                >>> m = WebMap().choropleth(squares, column="pop", name="area")
                >>> sorted(m.colorbar("area", label="People")._panels)
                ['legend']

                ```
            - `visible=False` asks for no key at all, and draws none:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().colorbar(visible=False)._panels
                {}

                ```
        """
        return self.legend(layer_id=layer_id, title=label, visible=visible)

    def _legend_of(self, layer_id: str) -> dict:
        """Return the classification one layer was drawn with.

        Args:
            layer_id: The layer whose classes are wanted.

        Returns:
            What the builder recorded for it. Reading `last_legend` instead described the most recently
            classified layer under the caller's chosen label — a wrong map that looks right (review H4).

        Raises:
            KeyError: when no layer on this map has that id.
            ValueError: when the layer carries no classification to describe.
        """
        self.get_layer(layer_id)  # refuses an id nobody drew, by name
        keyed = self._legends.get(layer_id)
        if keyed is None:
            raise ValueError(
                f"layer {layer_id!r} was not drawn with a classification, so it has no colour key to show; "
                f"the layers that were are {sorted(self._legends)}"
            )
        return keyed

    def remove_layer(self, layer_id: str) -> Self:
        """Drop a previously added layer from the map.

        Building a `WebMap` was one-way: a mistake meant starting over, which in a notebook — where these
        maps are built — is the normal workflow.

        Args:
            layer_id: An id from :attr:`layer_ids`.

        Returns:
            The same map instance, so builder calls chain.

        Note:
            The running data extent, and the classification a legend describes, are only cleared when the
            last layer goes. They are "most recent" accessors rather than a model of what is on the map,
            so after removing one layer of several they still describe the removed one; call
            :meth:`set_bounds` or rebuild the legend if that matters.

        Raises:
            KeyError: when no such layer was added, listing the ids that were — a silent no-op here would
                look exactly like a layer that refused to go away.

        Examples:
            - Remove one layer; the map is returned, so calls chain:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().text(4.9, 52.4, "Amsterdam", name="amsterdam").text(2.35, 48.86, "Paris", name="paris")
                >>> m.remove_layer("amsterdam") is m, m.layer_ids, len(m.layers)
                (True, ['paris'], 1)

                ```
            - An id that was never added is refused, naming the ids that were:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().text(4.9, 52.4, "Amsterdam", name="amsterdam").remove_layer("paris")
                Traceback (most recent call last):
                    ...
                KeyError: "no layer 'paris' on this map; added layers are ['amsterdam']"

                ```
        """
        present = self.layer_ids
        if layer_id not in present:
            raise KeyError(
                f"no layer {layer_id!r} on this map; added layers are {present}"
            )
        self._layer_tree = self._layer_tree.remove(layer_id)
        # A caller's own object is matched by identity: a `maplibre` `Layer` is a model that refuses the marker
        # attribute the builders' closures carry, so there is nothing on it to compare by id.
        held = self._custom.pop(layer_id, None)
        # The reference goes out of *this* map's figure, and the object stays registered: a `FigureSpec`
        # captured before the removal still names it, and forgetting here made every such figure dangle
        # (review M6). `close()` is where a map lets its data go, which is the caller saying they are
        # finished with every figure it produced.
        self._sources.pop(layer_id, None)
        self._forget_legend(layer_id)
        self._forget_slider_frame(layer_id)

        def _is_layer(layer: Any) -> bool:
            """Whether a queued layer is the one being removed.

            Args:
                layer: The queued closure or object.

            Returns:
                `True` for the builder closure that carries this id, or for the caller's own held object.
            """
            if getattr(layer, "_digitalearth_layer_id", None) == layer_id:
                return True
            return held is not None and layer is held

        # The reference band is addressed by a count, so a layer removed from it gives its slot back. Otherwise the
        # next `add_reference` inserts one place too high — above data it belongs beneath.
        self._reference_count -= sum(
            1
            for layer in self._queued[
                self._underlay_count : self._underlay_count + self._reference_count
            ]
            if _is_layer(layer)
        )
        self._underlay_count -= sum(
            1 for layer in self._queued[: self._underlay_count] if _is_layer(layer)
        )
        # The overlay band is counted from the top, for the same reason: remove a label and the next data layer
        # would otherwise be queued one place too low, beneath a label that is no longer there.
        self._overlay_count -= sum(
            1
            for layer in self._queued[len(self._queued) - self._overlay_count :]
            if _is_layer(layer)
        )
        self._queued = [layer for layer in self._queued if not _is_layer(layer)]
        # The renderer keeps its own record of what it drew, and nothing else clears it: `_reconcile` only
        # removes ids that `diff` reports, and a layer removed here is in neither figure it compares. Left
        # alone, the MapLibre source and layer objects of every removed layer are held for the map's life.
        self._renderer.remove(layer_id)
        remaining = self.layer_ids
        if self._last_layer_id == layer_id:
            self._last_layer_id = remaining[-1] if remaining else None
        self._forget_temporal_step(layer_id)
        if not remaining:
            # Nothing is drawn any more, so the classification a legend would describe is gone with it.
            self._data_bounds = None
            self.last_breaks = None
            self.last_legend = None
        return self

    def _forget_temporal_step(self, layer_id: str) -> None:
        """Drop ``layer_id`` from the time-slider config, and the config itself once a series is gone.

        Without this the slider keeps pointing at a layer that is no longer on the map, and the next
        ``render``/``save`` fails inside a helper the caller never invoked.

        Args:
            layer_id: The layer just removed.
        """
        config = self._temporal
        if not config or layer_id not in (config.get("layer_ids") or []):
            return
        index = list(config["layer_ids"]).index(layer_id)
        config["layer_ids"] = [i for i in config["layer_ids"] if i != layer_id]
        times = list(config.get("times") or [])
        if index < len(times):
            config["times"] = times[:index] + times[index + 1 :]
        if len(config["layer_ids"]) < 2:
            self._temporal = None  # one step is not a series

    def _layer_id(
        self, prefix: str, name: Optional[str] = None, *, kind: Optional[str] = None
    ) -> str:
        """Return the id a new layer should take, preferring the caller's own name.

        py-maplibregl's ``LayerSwitcherControl`` has no label channel: its JS captions each row with
        ``textContent = layerId`` and reads that same string back to toggle the layer. The id *is* what a
        viewer reads, so a layer the caller named must carry that name as its id — otherwise the switch in
        a saved page lists ``raster-4`` and ``circle-6``.

        Args:
            prefix: The kind tag used when there is no name (``"circle"``, ``"raster"``, …).
            name: The caller's name for the layer, if any.
            kind: The layer's kind, when its drawer adds MapLibre layers under ids derived from this one
                (:data:`~digitalearth.web.renderer.DERIVED_SUFFIXES`). Those ids are reserved with it, and a
                candidate one of whose derived ids is taken is passed over like a taken id (review M5).

        Returns:
            The name (suffixed ``-2``, ``-3``, … if that name, or an id derived from it, is already on the
            map), or a generated ``"<prefix>-<n>"`` when unnamed.
        """
        if not name:
            return self._uid(prefix, kind=kind)
        candidate, suffix = name, 2
        while not self._reserve(candidate, kind):
            candidate = f"{name}-{suffix}"
            suffix += 1
        return candidate

    def _uid(self, prefix: str, *, kind: Optional[str] = None) -> str:
        """Return a per-map-unique id like ``"fill-3"`` for a MapLibre source/layer.

        Args:
            prefix: A short kind tag (``"fill"``, ``"circle"``, ``"raster"``, …).
            kind: The layer's kind, when its drawer derives further MapLibre ids from this one; they are
                reserved with it, as :meth:`_layer_id` does.

        Returns:
            ``f"{prefix}-{n}"`` with a counter that increments on every call, so two builders never
            collide on a MapLibre source/layer id.
        """
        self._id_counter += 1
        # A caller name can look exactly like a generated id, and two layers sharing a MapLibre id makes
        # addLayer drop the second one with only a console error.
        while not self._reserve(f"{prefix}-{self._id_counter}", kind):
            self._id_counter += 1
        return f"{prefix}-{self._id_counter}"

    def _reserve(self, candidate: str, kind: Optional[str]) -> bool:
        """Reserve an id and every id its drawer derives from it, when none of them is taken.

        Args:
            candidate: The id to reserve.
            kind: The layer's kind, which says which ids are derived from `candidate`; `None` for none.

        Returns:
            `True` when all of them were free and are now issued; `False`, reserving nothing, otherwise.
        """
        from digitalearth.web.renderer import derived_ids

        wanted = (candidate, *(derived_ids(kind, candidate) if kind else ()))
        if any(layer_id in self._issued_ids for layer_id in wanted):
            return False
        self._issued_ids.update(wanted)
        return True

    def add_layer(
        self, layer: Any, *, name: Optional[str] = None, band: str = "data"
    ) -> Self:
        """Register a layer **you built yourself** and return ``self`` (chainable).

        This is where a caller's own MapLibre layer or ``apply(widget)`` callable joins the map. It is recorded
        in the layer tree as a custom layer — kind ``custom:maplibre`` — so it can be addressed like any other:
        `layer_ids` lists it and :meth:`remove_layer` takes it off. What is kept in the figure is the
        *description* (id, kind, label, band, visibility); the object itself is held by this map and is never
        written to a figure, because a MapLibre layer has no description to write. A figure loaded from a dict
        therefore names the layer but cannot rebuild it (see :mod:`digitalearth.base.custom`).

        The package's own builders do not come through here — they describe what they draw and queue it in the
        band their kind declares.

        Args:
            layer: A ``maplibre`` ``Layer``/spec, or a callable ``apply(widget)``.
            name: The id to address it by. Left out, a MapLibre layer's own ``id`` is used, and anything else
                gets a generated one; a name already on the map is suffixed, as every builder's ``name=`` is.
            band: Where it is drawn — ``"underlay"``, ``"reference"``, ``"data"`` (the default) or
                ``"overlay"``. A caller's object can draw anything, so only the caller can say whether it is
                ground cover or a label.

        Returns:
            This map, so builder calls chain: ``m.field(dem).basemap()``.

        Raises:
            ValueError: if `band` is not one of the four bands; if the object carries an id another layer
                on this map already uses, since MapLibre drops the second layer with that id; if `name=`
                names something other than the id the object carries, which the page would never add; or
                if this exact object is already on the map, which is one layer, not two.

        Examples:
            - A layer you built is addressable, and can be taken off again (any object stands in for a layer
              here — the queue does not inspect it):
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap().add_layer("wells-layer", name="wells")
                >>> m.layer_ids
                ['wells']
                >>> m.remove_layer("wells").layer_ids
                []

                ```
            - Registration keeps the drawing order, and returns the map for chaining:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap()
                >>> m.add_layer("raster-layer").add_layer("vector-layer") is m
                True
                >>> m.layers
                ['raster-layer', 'vector-layer']

                ```
        """
        if band not in KIND_BANDS:
            raise ValueError(
                f"add_layer band must be one of {list(KIND_BANDS)}; got {band!r}"
            )
        # By identity, and only where identity means anything: CPython interns equal strings and small
        # integers, so two independent stand-ins compared as one object and the second was refused as a
        # duplicate (review M7). A caller's real layer object is not one of these.
        already = (
            None
            if isinstance(layer, (str, bytes, int, float, bool, tuple, frozenset))
            else next(
                (held for held, obj in self._custom.items() if obj is layer), None
            )
        )
        if already is not None:
            # One object is one layer on the page. Registering it twice gave two tree entries the queue
            # could not tell apart — removing either removed both — and, since the allocated id is written
            # onto the object, the first entry named an id the object no longer carried (review M13).
            raise ValueError(
                f"this layer is already on the map as {already!r}; add a second one to draw it twice, or "
                f"remove_layer({already!r}) first"
            )
        # An object that carries an id is drawn under *that* id: it is what `addLayer` sends, so it is what
        # `layer_ids`, `get_layer`, `layer_control` and `popup(layer=...)` have to name (review H5). The
        # allocator's suffixing would break that, and rewriting the object's id instead reached across to
        # every other map already holding it (review H6) — so a collision is refused, by name, rather than
        # papered over. An object with no id of its own is numbered as before.
        own = getattr(layer, "id", None)
        if own is None:
            layer_id = self._layer_id("custom", name)
        else:
            layer_id = str(own)
            if name is not None and str(name) != layer_id:
                raise ValueError(
                    f"add_layer draws this layer under the id it carries, {layer_id!r}, so name={name!r} "
                    f"would name something the page never adds; drop name=, or build the layer with that id"
                )
            # The ids the drawn layers' own extra layers carry count too — a graticule's labels, a
            # cluster's counts — since the widget adds those beside the layer under their own ids (M5).
            in_use = [
                *self.layer_ids,
                *(
                    extra.id
                    for built in self._renderer.drawn.values()
                    for extra in built.extra_layers
                ),
            ]
            if layer_id in in_use:
                raise ValueError(
                    f"layer id {layer_id!r} is already on this map; MapLibre drops the second layer with "
                    f"that id, so build this one with a free id (the ids in use are {in_use})"
                )
            # Reserved like an allocated id, so a later `name=` — or an id a later drawer derives — steps
            # around it rather than drawing a second layer under it.
            self._issued_ids.add(layer_id)
        # Held before the description is recorded: `draw_custom` reads the object from here, so it has
        # to be in place by the time `_index_layer` draws.
        self._custom[layer_id] = layer
        self._index_layer(
            layer_id,
            name,
            kind=custom_kind("maplibre"),
            band=band,
            visible=_layer_is_visible(layer),
        )
        return self

    def _queue(self, layer: Any) -> Self:
        """Queue a drawn layer among the data, without recording it, and return ``self``.

        What the package's own builders use: they have already described the layer with :meth:`_index_layer`,
        so queueing it again through :meth:`add_layer` would record it twice — once as what it draws and once
        as a caller's own object.

        A layer is queued on top of the data, but beneath the overlay band: text added before a raster is still
        drawn over it, which is what the tree says and what a reader expects of a place name.

        Args:
            layer: A queue entry standing for a drawn description, a callable ``apply(widget)``, or a
                ``maplibre`` ``Layer``/spec.

        Returns:
            This map (chainable).
        """
        self._queued.insert(len(self._queued) - self._overlay_count, layer)
        return self

    def _queue_in_band(self, layer: Any, band: str) -> Self:
        """Queue a drawn layer in one band, and return ``self``.

        Args:
            layer: A queue entry standing for a drawn description, a callable ``apply(widget)``, or a
                ``maplibre`` ``Layer``/spec.
            band: One of :data:`~digitalearth.base.registry.KIND_BANDS`.

        Returns:
            This map (chainable).
        """
        if band == "overlay":
            return self.add_overlay(layer)
        if band == "reference":
            return self.add_reference(layer)
        if band == "underlay":
            return self.add_underlay(layer)
        return self._queue(layer)

    def _needs_reproject(self, data: Any) -> bool:
        """Whether `data` must be reprojected (via pyramids) to the display CRS.

        Args:
            data: A pyramids object exposing `.crs` and/or `.epsg` (`Dataset`/`FeatureCollection`).

        Returns:
            `False` when the data's CRS and the display CRS name the same reference system, however either
            is spelled; `True` otherwise.

        Note:
            This is a thin alias for :func:`digitalearth.base.display.needs_reproject`, kept because tier code
            and tests call it. :meth:`_to_display_source` calls the shared function **directly**, so overriding
            this method no longer changes what *that* method reprojects; the tier's other callers still consult
            it. Nothing in-tree overrides it.
        """
        return needs_reproject(data, self.crs)

    @staticmethod
    def _opened(data: Any) -> Any:
        """Open what a builder was handed, when it names data rather than being the data.

        A path or a URL is the only input a figure can be written down with: a pyramids object does not
        know where it came from, so the only honest reference to one is ``object:``, which resolves in the
        process that made it and nowhere else. The static and interactive tiers take a path because their
        builders never touch the argument — it goes straight into ``DataRef.of`` and the renderer opens it
        at draw time. This tier's builders read the data *before* they record it (to warp it, to resolve a
        colormap from its variable, to frame the map on its extent), so the reference is opened here, at
        the three display choke points, and the builder still records the caller's own path (review H1).

        Args:
            data: Whatever the caller passed — a path, a URL, a :class:`~digitalearth.base.spec.DataRef`,
                or the data itself.

        Returns:
            The opened data for a reference; `data` itself for anything else, which is what a drawer
            re-entering a choke point with an already-opened source hands over.

        Raises:
            FileNotFoundError: when the path names nothing, which is the resolver's message naming it.
            KeyError: when no resolver is registered for the URL's scheme.
        """
        if isinstance(data, (str, os.PathLike, DataRef)):
            return DataRef.of(data).open()
        return data

    def _to_display_source(self, data: Any, *, band: int = 1) -> Source:
        """Reproject ``data`` to the display CRS through pyramids and wrap it as a :class:`Source`.

        The single display-CRS choke point every raster/vector builder calls (settling the tier's
        projection decision, option A: **pre-reproject in pyramids** — no cartopy anywhere). Inputs already
        in the display CRS pass through untouched; everything else goes through ``data.to_crs(self.crs)``
        (pyramids' warp for rasters, ``FeatureCollection`` reprojection for vectors). MapLibre renders only
        EPSG:3857 / 4326, so ``crs`` should be one of those.

        Args:
            data: A pyramids ``Dataset`` / ``FeatureCollection`` (anything ``get_source`` accepts), or a
                path / URL / :class:`~digitalearth.base.spec.DataRef` naming one, which :meth:`_opened`
                reads first. Bare numpy arrays / :class:`Source` objects pass straight through to
                extraction.
            band: 1-based band to extract for raster inputs.

        Returns:
            Source: the display-CRS view (``z``/``x``/``y``/``crs``/``metadata``).
        """
        return to_display_source(self._opened(data), self.crs, band=band)

    def _to_display_raster(self, dataset: Any) -> Any:
        """Return ``dataset`` in the display CRS, reprojected through pyramids when it is not already.

        ``_to_display_source`` gives one band; a composite needs the dataset itself so ``get_stack`` can
        read three from it.

        Args:
            dataset: A pyramids ``Dataset``, or a path / URL naming one (see :meth:`_opened`).

        Returns:
            The dataset in the display CRS.
        """
        dataset = self._opened(dataset)
        if hasattr(dataset, "to_crs") and self._needs_reproject(dataset):
            # Through the tier's own helper, so a dataset that cannot be warped into the display CRS
            # raises the OffLimbError every other builder raises rather than GDAL's raw RuntimeError.
            return reproject(dataset, self.crs)
        return dataset

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
                f"field(); for a raster time stack pass a DatasetCollection to timeslider()."
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
                ...     print("field" in str(err), "DatasetCollection" in str(err))
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
            f"{type(features).__name__}. For a single raster use field(); for a raster time stack "
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
        fails there:

        - ``datetime64``, tz-naive or tz-aware;
        - ``timedelta64``, as an ISO-8601 duration (``P1DT0H0M0S``);
        - ``period``, as its own string form (``2026-01``);
        - ``category``, whose real dtype is unwrapped from ``.categories`` first — a repeated timestamp
          column is routinely stored this way and matches none of the checks above;
        - an ``object`` column holding any of those, plus ``datetime.time``, ``numpy.datetime64`` and
          ``numpy.timedelta64``, none of which the dtype checks can see. An object column is encoded when
          **any** of its values is date-like, not just its first — the two orderings of
          ``["n/a", date(...)]`` are equally unserialisable.

        A mixed column keeps its non-date values as they are, so one GeoJSON property can hold both text
        and an ISO string. That is deliberate: stringifying the rest would destroy the types of numbers
        that serialise perfectly well, and a column mixing dates with other values is already outside what
        a MapLibre expression can compare meaningfully.

        Two caveats on ordering, both mattering only when ``timeslider`` steps through the column:

        - ISO *timestamps* sort chronologically only while the UTC offset is constant. A tz-aware column
          spanning a DST change encodes offsets that differ, and lexicographic order then diverges from
          chronological order — convert such a column to UTC first.
        - ISO *durations* do not sort lexicographically at all: ``P10DT0H0M0S`` precedes ``P1DT0H0M0S``
          as text. A ``timedelta64`` column is fine as a rendered attribute, but is not a usable
          ``kdim``.

        Colouring by an encoded column needs ``scheme="categorical"``: the graduated schemes parse the
        property as a float and cannot read ISO text. (Colouring by a date column never worked — before
        this encoding it failed later, in ``json.dumps``.)

        Args:
            gdf: The display-CRS GeoDataFrame about to be handed to ``add_source``.

        Returns:
            The same object when it holds no date-like column, else a copy with those columns converted.
            The input is never mutated — it is the caller's frame.

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
        import pandas as pd

        columns = getattr(gdf, "columns", None)
        if (
            columns is None
        ):  # a GeoSeries satisfies the vector guard but has no columns to encode
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
            encoded = _date_encoded(series)
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
        ``add_source``. A pyramids ``FeatureCollection`` *is* a ``geopandas`` GeoDataFrame (the form
        ``maplibre.Map.add_source`` accepts
        for vector data), so it is reprojected through pyramids when needed and returned as-is; a bare
        GeoDataFrame in another CRS is reprojected the same way. No shapely/geopandas-as-engine import —
        pyramids owns the reprojection and the GeoDataFrame type. Both reprojections go through
        :meth:`_placed`, so a warp that can place none of the geometry reaches the tier's off-limb policy
        instead of quietly producing a frame of ``inf`` coordinates.

        Args:
            features: A pyramids ``FeatureCollection`` or a GeoDataFrame, or a path / URL naming one,
                which :meth:`_opened` reads first — the only input the layer can then be written down
                with (review H1).
            method: The calling builder's name, quoted in the guard's error message. Required, so a
                new builder cannot silently inherit a generic label.

        Returns:
            A GeoDataFrame in the display CRS (EPSG:4326 by default), with date-like columns ISO-encoded,
            ready for ``add_source``.

        Raises:
            TypeError: when ``features`` is not a vector layer.
            OffLimbError: when the warp places none of the geometry and the map is ``strict``.

        See Also:
            digitalearth.web.base.WebMapBase._require_vector: the input guard applied first.
            digitalearth.web.base.WebMapBase._placed: the reprojection, and the off-limb report.
            digitalearth.web.base.WebMapBase._json_safe: the date encoding applied to the result.
        """
        features = self._opened(features)
        self._require_vector(features, method)
        if hasattr(features, "epsg") and hasattr(
            features, "to_crs"
        ):  # pyramids FeatureCollection (a GeoDataFrame)
            if self._needs_reproject(features):
                features = self._placed(features, method=method)
            return self._noted(self._json_safe(features))
        own_crs = getattr(features, "crs", None)
        # A bare GeoDataFrame in another CRS. Compared by meaning, not by EPSG code: a projection with no
        # authority code has none, and was drawn as if its metres were degrees.
        if own_crs is not None and not same_crs(own_crs, self.crs):
            return self._noted(self._json_safe(self._placed(features, method=method)))
        return self._noted(self._json_safe(features))

    def _placed(self, features: Any, *, method: str) -> Any:
        """Warp ``features`` to the display CRS, reporting a warp that placed none of the geometry.

        The vector half of the tier's off-limb policy. A vector warp does not raise when it can place
        nothing — it hands back ``inf`` coordinates and says nothing — so the reprojection goes through
        the shared :func:`~digitalearth.base.crs.reproject`, which reads the result and reports it, rather
        than through ``to_crs`` directly. Calling ``to_crs`` here is what kept this tier (and the static
        one) outside a contract the module docstring of ``base/crs.py`` says covers every backend.

        Args:
            features: A pyramids ``FeatureCollection`` or a GeoDataFrame, in any CRS.
            method: The calling builder's name, quoted in the warning.

        Returns:
            The display-CRS frame. When the warp placed nothing and the map is not ``strict`` the warning
            is logged and the frame is handed back as the warp produced it — the layer draws nothing
            either way, and the builders read values off the frame the geometry came with.

        Raises:
            OffLimbError: when the warp placed none of the geometry and the map is ``strict``.
        """
        try:
            return reproject(features, self.crs)
        except OffLimbError as error:
            self._skipped(method, str(error), error)  # raises under strict
            return features.to_crs(self.crs)

    def add_reference(self, layer: Any) -> Self:
        """Register ``layer`` above the basemaps but below the data, and return ``self``.

        A graticule or a coastline is reference geography: drawn over the ground so it is visible, under
        the data so it does not obscure it. Neither end of the list expresses that — ``add_underlay``
        puts it beneath the opaque basemap tiles, where it cannot be seen at all.

        Args:
            layer: A queue entry standing for a drawn description, a callable ``apply(widget)``, or a
                ``maplibre`` ``Layer``/spec.

        Returns:
            This map (chainable).
        """
        self._queued.insert(self._underlay_count + self._reference_count, layer)
        self._reference_count += 1
        return self

    def add_overlay(self, layer: Any) -> Self:
        """Register ``layer`` **above the data** (drawn last) and return ``self``.

        Text and labels call this: a place name is drawn over the field it names, and stays over it when a
        raster is added afterwards. The mirror of :meth:`add_underlay`, and the reason :meth:`add_layer` inserts
        beneath this band rather than appending.

        Args:
            layer: A queue entry standing for a drawn description, a callable ``apply(widget)``, or a
                ``maplibre`` ``Layer``/spec.

        Returns:
            This map (chainable).

        Examples:
            - What joins the overlay band stays above what is added afterwards (any object stands in for a
              layer here — the queue does not inspect it):
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().add_overlay("label").add_layer("raster-layer").layers
                ['raster-layer', 'label']

                ```
            - Two overlays keep the order they arrived in:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap().add_overlay("towns").add_overlay("cities").layers
                ['towns', 'cities']

                ```
        """
        self._queued.append(layer)
        self._overlay_count += 1
        return self

    def add_underlay(self, layer: Any) -> Self:
        """Register ``layer`` at the **bottom** of the stack (drawn first) and return ``self``.

        Basemaps/tiles call this so they sit beneath the data layers regardless of when they are added —
        the mirror of :meth:`add_layer`, which appends on top.

        Args:
            layer: A queue entry standing for a drawn description, a callable ``apply(widget)``, or a
                ``maplibre`` ``Layer``/spec.

        Returns:
            This map (chainable).
        """
        self._queued.insert(0, layer)
        self._underlay_count += 1
        return self

    @staticmethod
    def _style_for(source: Any) -> Dict[str, Any]:
        """Return the autostyle parameters for ``source`` (``cmap``, and sometimes ``levels``/``units``).

        The tier's single entry into :func:`digitalearth.base.autostyle.auto_style` — the same
        variable→style lookup the static and interactive tiers use, ECMWF-Magics match included — so one
        variable is drawn the same way whichever tier renders it. The three readers below take one key
        each; going through here means they all see one lookup's answer.

        Args:
            source: The display-CRS :class:`Source` whose variable drives the lookup.

        Returns:
            The style dict; always carries ``cmap``, and carries ``levels``/``units`` for a recognised
            operational field.
        """
        from digitalearth.base.autostyle import auto_style

        return auto_style(source)

    def _auto_cmap(self, source: Any, cmap: Optional[str]) -> str:
        """Resolve a colormap name: the caller's ``cmap`` if given, else the autostyle default.

        Mirrors the interactive tier's ``_auto_cmap`` so a variable looks the same across tiers (the same
        ``digitalearth.base.autostyle`` variable→style lookup, incl. the ECMWF-Magics match); falls back to
        ``"viridis"`` for an unrecognised field — which is also what the autostyle library's ``default``
        group carries, so the `DEFAULT_CMAP` fallback in :func:`~digitalearth.base.display.auto_cmap` only
        covers a lookup that answered with no colormap at all.

        Goes through :meth:`_style_for`, which its docstring names as the tier's single entry into the
        style table — so this, :meth:`_auto_levels` and :meth:`_auto_units` see one lookup's answer.

        Args:
            source: The display-CRS :class:`Source` whose variable drives the lookup.
            cmap: The caller-supplied colormap, or ``None`` to auto-resolve.

        Returns:
            The colormap name to use.
        """
        return auto_cmap(source, cmap, lookup=self._style_for)

    def _auto_levels(self, source: Any, levels: Optional[Any]) -> Optional[Any]:
        """Resolve contour levels: the caller's ``levels`` if given, else the autostyle ones.

        A recognised operational field (mean sea-level pressure, 2-m temperature, …) carries the contour
        levels its community draws it with, and until now the tier read only ``cmap`` from the same
        lookup and made the caller supply them again.

        Args:
            source: The display-CRS :class:`Source` whose variable drives the lookup.
            levels: The caller-supplied levels, or ``None`` to auto-resolve.

        Returns:
            The levels to contour at, or ``None`` when the caller gave none and the variable is not one
            the library knows — a guessed set of levels would be worse than asking for them.
        """
        if levels is not None:
            return levels
        resolved = self._style_for(source).get("levels")
        return list(resolved) if resolved else None

    def _auto_units(self, source: Any, units: Optional[str]) -> Optional[str]:
        """Resolve the units a key labels values with: the caller's if given, else the autostyle hint.

        The same three-way contract as :meth:`_auto_cmap` and :meth:`_auto_levels`, and like theirs the
        caller's half is a real public argument — ``field(units=)`` and ``contours(units=)``. The
        library's hint is canonical rather than measured (it says ``"hPa"`` for mean sea-level pressure),
        so a band that is genuinely in something else needs a way to say so that does not also throw away
        the column name ``legend()`` derives; that is what the argument is for (review L3).

        Args:
            source: The display-CRS :class:`Source` whose variable drives the lookup.
            units: The caller-supplied units, or ``None`` to auto-resolve.

        Returns:
            The units string, or ``None`` when neither the caller nor the library names one — the label
            is then built exactly as it was before, never with a guessed unit.
        """
        if units is not None:
            return units
        resolved = self._style_for(source).get("units")
        return str(resolved) if resolved else None

    @staticmethod
    def _cmap_hex(cmap: str, n: int) -> List[str]:
        """Sample ``cmap`` at ``n`` evenly spaced stops and return hex colour strings.

        Delegates to :func:`~digitalearth.base.symbology.sample_cmap`, the one sampler every tier uses, so a
        graduated web layer and the same layer on another backend cannot land on different colours. Kept as a
        method because the vector builders call it through ``self``.

        Args:
            cmap: A matplotlib colormap name.
            n: Number of colours to sample (>= 1).

        Returns:
            A list of ``n`` ``#rrggbb`` hex strings spanning the colormap.
        """
        return sample_cmap(cmap, n)

    def _map_options(self) -> dict:
        """Build the ``MapOptions`` kwargs from the display config (drops an unset ``center``).

        Returns:
            The ``zoom`` and the resolved ``style``, plus ``center`` only when one was given — MapLibre
            frames an unset centre itself, and passing ``None`` would pin it to null island instead.
        """
        options: dict = {"zoom": self.zoom, "style": _resolve_style(self.style)}
        if self.center is not None:
            options["center"] = self.center
        return options

    def _apply_layer(self, widget: Any, layer: Any) -> None:
        """Apply one queued ``layer`` onto the MapLibre ``widget``.

        Three shapes reach here, in the order the queue holds them. A described layer is looked up in the
        renderer, and what it drew is added source-first, then the layer, then whatever extra layers the
        same description owns (a graticule's degree labels, a cluster's counts and loose points) — a
        description nothing was drawn for adds nothing, which is how a declined layer stays off the page. A
        callable layer is a mixin-supplied ``apply(widget)`` that wires its own source(s) + layer(s), the
        path the kinds outside :data:`~digitalearth.web.renderer.DRAWN_KINDS` still take. Anything else is
        handed straight to the widget's ``add_layer``.

        Args:
            widget: The MapLibre ``MapWidget`` being assembled.
            layer: A queue entry standing for a drawn description, a callable ``apply(widget)``, or a
                ``maplibre`` ``Layer``/spec.
        """
        if isinstance(layer, _Described):
            built = self._renderer.drawn.get(layer.layer_id)
            if built is None:
                return
            if built.source_id is not None:
                widget.add_source(built.source_id, built.source_spec)
            widget.add_layer(built.layer)
            for extra in built.extra_layers:
                widget.add_layer(extra)
        elif callable(layer):
            layer(widget)
        else:
            widget.add_layer(layer)

    def _build_map_widget(self, *, with_controls: bool = True) -> Any:
        """Build the bare MapLibre ``MapWidget`` with every registered layer applied (no temporal wrap).

        The single map-construction path shared by :meth:`render` (which may wrap it in a time-slider) and
        :meth:`save` (which serialises just the map). An empty map is returned when no layers are registered.

        Args:
            with_controls: Whether to add the controls this map adds for itself — today the time-step
                picker. A rendered image wants the map alone: a GIF frame or a PNG snapshot with a
                switcher panel burned into the pixels is not the artifact the caller asked for.

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
        for layer in self._queued:
            self._apply_layer(widget, layer)
        if self._panels:
            from maplibre.controls import InfoBoxControl

            # Imported here, not at module scope: decoration imports base, so the other direction has to
            # wait until it is needed.
            from digitalearth.web.decoration import _PANEL_CSS

            for content, position in self._panels.values():
                widget.add_control(
                    InfoBoxControl(
                        content=content, css_text=_PANEL_CSS, position=position
                    ),
                    position,
                )
        switcher = self._switcher_request() if with_controls else None
        if switcher is not None:
            from maplibre.controls import LayerSwitcherControl

            widget.add_control(
                LayerSwitcherControl(
                    layer_ids=switcher["layer_ids"], theme=switcher["theme"]
                ),
                switcher["position"],
            )
        view = self._map_view()
        if view is not None:
            widget.fit_bounds(
                view["bounds"], padding=view["padding"], animate=view["animate"]
            )
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
        title: str = DEFAULT_TITLE,
        offline: bool = False,
        **kwargs: Any,
    ) -> pathlib.Path:
        """Save the map — a standalone HTML page or a PNG snapshot — and return its path (DW.6).

        The output kind is ``fmt`` if given, else inferred from the suffix: ``.png`` renders a snapshot,
        ``.gif`` runs the animation export (:meth:`~digitalearth.web.export.ExportMixin.animate`, which
        needs a temporal map and a headless browser), anything else writes the HTML page.
        HTML is serialised via ``MapWidget.to_html`` and written as UTF-8 ourselves — sidestepping maplibre's
        cp1252-on-Windows writer bug (see :func:`_patch_maplibre_html_encoding`). By default the page embeds
        the map state and widget JS but references ``maplibre-gl`` from a CDN; ``offline=True`` inlines the
        engine JS/CSS for a fully self-contained page (best-effort, needs network at save time). PNG export is
        delegated to the export mixin and needs a headless browser.

        Args:
            path: Output file (``*.html`` or ``*.png``).
            fmt: Force the format (``"html"`` / ``"png"`` / ``"gif"``); ``None`` infers it from ``path``.
            title: HTML document title.
            offline: When True (HTML only), inline the ``maplibre-gl`` JS/CSS so the page opens offline.
            **kwargs: Forwarded to ``MapWidget.to_html`` (HTML) or the PNG renderer.

        Returns:
            The :class:`pathlib.Path` written — the same type every tier's ``save`` returns, so a caller
            can go straight on to ``.stat()``/``.read_text()`` without re-wrapping a string.

        Raises:
            ImportError: when the ``web`` extra is not installed (or, for PNG, no headless browser is present).
            ValueError: propagated from :meth:`~digitalearth.web.export.ExportMixin.animate` for a
                ``.gif`` destination on a map that carries no renderable time series.

        Examples:
            - Write a standalone page and carry straight on from the path that comes back, instead
              of rebuilding the filename (needs the ``web`` extra, hence skipped here):
                ```python
                >>> import pathlib, tempfile                         # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> out = pathlib.Path(tempfile.mkdtemp()) / "m.html"  # doctest: +SKIP
                >>> written = WebMap(center=(8.0, 47.0)).save(out)   # doctest: +SKIP
                >>> written == out, written.stat().st_size > 0       # doctest: +SKIP
                (True, True)

                ```
            - ``title`` reaches the document itself, so a saved map is recognisable in a browser
              tab rather than opening as one more "Digital-Earth map":
                ```python
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> page = WebMap().save("rain.html", title="Rain")  # doctest: +SKIP
                >>> "<title>Rain</title>" in page.read_text("utf-8")  # doctest: +SKIP
                True

                ```
            - The suffix, not a flag, picks the branch: ``.gif`` hands the call to
              :meth:`~digitalearth.web.export.ExportMixin.animate`, which says so when the map has
              nothing to animate. That dispatch needs no engine, so it runs here:
                ```python
                >>> from digitalearth.web import WebMap
                >>> try:
                ...     WebMap().save("frames.gif")
                ... except ValueError as error:
                ...     print(str(error).split(";")[0])
                save_animation() needs a raster time series with at least two steps

                ```

        See Also:
            digitalearth.web.export.ExportMixin.animate: the ``.gif`` branch.
            render: the in-notebook counterpart — the same widget, without writing a file.
        """
        suffix = pathlib.Path(str(path)).suffix.lower().lstrip(".")
        kind = (fmt or (suffix if suffix in {"png", "gif"} else "html")).lower()
        if kind == "gif":
            # The contract's name: calling the alias told a caller who wrote `save()` to write
            # `save_animation()`, from a line inside the library (review M14).
            return self.save_animation(path, title=title, **kwargs)
        if kind == "png":
            return self._render_png(path, title=title, **kwargs)
        html = self._build_map_widget().to_html(title=title, **kwargs)
        if offline:
            html = self._inline_offline_assets(html)
        out = pathlib.Path(path)
        out.write_text(html, encoding="utf-8")
        return out

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

    def close(self) -> None:
        """Let go of the in-memory data this map registered.

        The object registry is process-global and holds strong references, so a session that builds maps
        keeps every dataset they drew until something says otherwise. This is the caller saying so. Closing
        twice is harmless, and a map that registered nothing has nothing to forget.

        An `object:` source in a `figure_spec` captured from this map cannot be opened afterwards — which is
        what "closed" means for a figure whose data lived only in this process. Save the data and reference
        it by path to keep such a figure readable (review M2).

        Examples:
            - A closed map's in-memory source no longer resolves:
                ```python
                >>> import geopandas as gpd                                    # doctest: +SKIP
                >>> from shapely.geometry import Point                         # doctest: +SKIP
                >>> from digitalearth.web import WebMap                        # doctest: +SKIP
                >>> pts = gpd.GeoDataFrame(                                    # doctest: +SKIP
                ...     {"v": [1]}, geometry=[Point(0, 0)], crs=4326
                ... )
                >>> m = WebMap().points(pts)                                   # doctest: +SKIP
                >>> m.close()                                                  # doctest: +SKIP

                ```

        See Also:
            __exit__: calls this on the way out of a ``with`` block.
        """
        forget_namespace(self._objects_ns)

    def __enter__(self) -> Self:
        """Enter the runtime context, returning the map.

        Returns:
            The same map, so ``with WebMap() as m:`` binds this object and :meth:`close` runs on the way out.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """Let the map's in-memory data go on the way out of a ``with`` block.

        Args:
            *exc: The exception triple, ignored — closing is unconditional, as it is for a file.
        """
        self.close()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Render the map inline in notebooks by delegating to the MapLibre widget.

        Args:
            include: The mime types Jupyter asks for, passed through untouched.
            exclude: The mime types Jupyter asks to be left out, passed through untouched.

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
