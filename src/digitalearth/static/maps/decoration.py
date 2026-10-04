"""DecorationMixin — annotation, basemap tiles and Natural-Earth vector decoration.

Lon/lat text/annotate, a tile basemap, a backdrop ``stock_img``, and the Natural-Earth coastline/border/
land/ocean/lake/river layers (with the globe limb-splitting/closing helpers behind them).

The reference data comes from ``cleopatra.basemap.reference`` (``natural_earth`` raw lon/lat coordinate arrays for
the globe limb-splitting; ``add_features`` for the flat, hole-aware reprojected render) — these helpers
moved out of pyramids into cleopatra in pyramids 0.32 / cleopatra 0.17.

**One deliberate, temporary deviation from the engine boundary.** ``cleopatra.basemap.solar`` owns solar
geometry, and everything here goes through it — except :func:`_globe_nightshade`, which computes the
solar altitude over a display grid with the formula written out locally. It has to: cleopatra exposes the
night side as a *polygon*, which a globe cannot take (half of it is on the far side, where the projection
has no position at all), and exposes no way to ask how high the sun stands at a lon/lat. The missing
upstream helper is requested as **cleopatra#379** (``solar_altitude`` / a night-mask form); until it
lands, the local copy stays, pinned against the upstream it duplicates by
``tests/static/test_nightshade_tissot.py::TestNightshade::test_the_globe_fill_follows_cleopatras_own_subsolar_point``
so the two cannot drift apart unnoticed. It is not to be grown: anything further about solar geometry
belongs in that issue, not in this module (round 3, M7).
"""

import contextlib
import logging
import math
import numbers
import warnings
from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import parse_qs, urlsplit, urlunsplit

import numpy as np
from cleopatra.basemap.ogc import WMSProvider, WMTSProvider
from cleopatra.basemap.reference import add_features, natural_earth
from cleopatra.basemap.solar import (
    DEFAULT_REFRACTION,
    DEFAULT_TERMINATOR_SAMPLES,
    DEFAULT_TISSOT_SAMPLES,
    add_nightshade,
    add_tissot,
    subsolar_point,
    tissot_circles,
)
from cleopatra.basemap.tiles import add_tiles
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.font_manager import font_family_aliases, fontManager
from matplotlib.text import Text
from pyramids.base.crs import CRSError, crs_from_user_input, reproject_coordinates

from digitalearth.base.basemaps import (
    DEFAULT_BASEMAP_PROVIDER,
    PRESET_KEYWORDS,
    KeyedTileSource,
    get_keyed_basemap,
    is_keyed_basemap,
)
from digitalearth.base.domains import resolve_domain
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.base.spec._serial import crs_to_json
from digitalearth.base.spec.bounds import same_crs
from digitalearth.static import projections
from digitalearth.static.renderer import DrawnLayer, artists_added
from digitalearth.static.scene import LayerRecord, drawing_style

logger = logging.getLogger(__name__)

#: How each Natural-Earth layer looks when the caller asks for nothing else — this package's own defaults, in
#: the singular matplotlib keys. The drawer lays whatever the caller passed over them, so a layer drawn on a
#: scene that holds none of the caller's keywords (a figure read back from JSON) still looks like itself.
_NATURAL_EARTH_STYLE: Dict[str, Dict[str, Any]] = {
    "coastline": {"color": "black", "linewidth": 0.5},
    "borders": {"color": "gray", "linewidth": 0.4},
    "land": {"color": "#efefdb", "edgecolor": "none"},
    "ocean": {"color": "#cfe6f5", "edgecolor": "none"},
    "lakes": {"color": "#cfe6f5", "edgecolor": "none"},
    "rivers": {"color": "#5a8fcf", "linewidth": 0.4},
}

#: The canonical :class:`matplotlib.text.Text` properties that set a font family. Every keyword matplotlib
#: accepts for one — ``family``, ``font``, ``font_properties``, ``name`` and the three spellings below — is
#: an alias of one of these, so :func:`_a_font_is_already_chosen` asks
#: :func:`matplotlib.cbook.normalize_kwargs` rather than matching against a list of spellings. The list this
#: replaces was short by one: ``font_properties=`` was not on it, the layer's name went on as ``fontname=``
#: — which ``Text`` applies *after* ``fontproperties`` — and the caller's font was discarded with nothing
#: said (review R2-H5). Naming the three properties instead means a spelling matplotlib adds later is
#: covered by the alias table it ships with, not by an edit here.
_FONT_FAMILY_PROPERTIES = frozenset({"fontfamily", "fontname", "fontproperties"})


def _a_font_is_already_chosen(opts: Dict[str, Any]) -> bool:
    """Say whether a text call's own keywords name a font family, under any spelling matplotlib takes.

    This decides **priority**, and only priority: a layer's name is read as a font solely when the call
    itself named none. It is not, as this once claimed, what keeps matplotlib from being handed the same
    family twice — measured, ``Axes.text(fontname=..., fontfamily=...)`` and
    ``Axes.text(fontname=..., family=...)`` both draw, with the later keyword winning. The only pairs that
    raise ``TypeError: Got both`` are two aliases of *one* property (``family=`` beside ``fontfamily=``,
    ``name=`` beside ``fontname=``), which only the caller can write, and ``name=`` never reaches these
    keywords at all (review R2-L5).

    Args:
        opts: The keywords the call will hand the text artist.

    Returns:
        ``True`` when any of them canonicalises to one of :data:`_FONT_FAMILY_PROPERTIES`, or when
        matplotlib refuses the set outright — both of which mean the layer's name stays out of it.

    Examples:
        - Each spelling of a family is recognised as one, including the one a list of spellings missed:
            ```python
            >>> from digitalearth.static.maps.decoration import _a_font_is_already_chosen
            >>> [
            ...     _a_font_is_already_chosen({key: "DejaVu Sans"})
            ...     for key in ("font", "font_properties", "fontproperties", "family", "fontname")
            ... ]
            [True, True, True, True, True]

            ```
        - A call that styles the label some other way has chosen no font:
            ```python
            >>> from digitalearth.static.maps.decoration import _a_font_is_already_chosen
            >>> _a_font_is_already_chosen({"fontsize": 12, "color": "red"})
            False

            ```
    """
    try:
        chosen = normalize_kwargs(opts, Text)
    except TypeError:
        # Two aliases of one property. `Axes.text` raises on exactly this set, so the call is going to
        # fail whatever is decided here; what matters is not adding a third spelling to the pile first.
        return True
    return bool(_FONT_FAMILY_PROPERTIES.intersection(chosen))


@lru_cache(maxsize=4)
def _families_matplotlib_has(_registered: int) -> FrozenSet[str]:
    """Return the case-folded name of every font family matplotlib has registered.

    Args:
        _registered: ``len(fontManager.ttflist)``. Unread — it is here to be part of the cache key, so a
            family added with ``fontManager.addfont`` after the first call is seen rather than missed for
            the life of the process.

    Returns:
        Every registered family name, case-folded, because matplotlib matches a family case-insensitively.
    """
    return frozenset(entry.name.casefold() for entry in fontManager.ttflist)


def _names_a_font_family(name: str) -> bool:
    """Say whether a string is the name of a font family matplotlib can draw in.

    A **membership test**, deliberately, and not a resolution. The resolution this replaces asked
    ``findfont(FontProperties(family=name), fallback_to_default=False)``, where a lone string is parsed as
    a *fontconfig pattern*: ``-`` separates the size and ``,`` the families. So ``"DejaVu Serif-2"``
    resolved — as 2-point DejaVu Serif — and was then forwarded verbatim, where matplotlib reads it as
    one literal family and finds nothing. That string is exactly what the id allocator mints for a second
    layer with the same name, so the check produced the `findfont: ... not found` it exists to avoid, on
    names this package generates (review R2-H6). What decides and what is forwarded are one thing here:
    both are the literal family name.

    It is also what the resolution cost. Reading the registry is a set lookup, where ``findfont`` scored
    every registered font against the pattern; and because a name that is no font raises rather than
    returns, matplotlib's own ``lru_cache`` never held the answer, so every distinct layer name paid the
    scan again (review R2-L8). The scan could also rebuild the global ``fontManager`` from the filesystem
    — ``findfont``'s ``rebuild_if_missing`` defaults to ``True`` — from inside a layer builder.

    Args:
        name: The string to test, as the caller spelled it.

    Returns:
        ``True`` when ``name`` is a registered family or one of matplotlib's generic aliases
        (``serif``, ``sans-serif``, ``monospace``, ...), matched case-insensitively as matplotlib matches
        them.

    Examples:
        - A family matplotlib ships, whatever case it is written in, and a generic alias:
            ```python
            >>> from digitalearth.static.maps.decoration import _names_a_font_family
            >>> (_names_a_font_family("DejaVu Serif"), _names_a_font_family("dejavu serif"))
            (True, True)
            >>> (_names_a_font_family("sans-serif"), _names_a_font_family("monospace"))
            (True, True)

            ```
        - A layer name, and a string that is only a *pattern* for a family:
            ```python
            >>> from digitalearth.static.maps.decoration import _names_a_font_family
            >>> (_names_a_font_family("wells"), _names_a_font_family("DejaVu Serif-2"))
            (False, False)

            ```
    """
    folded = name.casefold()
    if folded in font_family_aliases:
        return True
    return folded in _families_matplotlib_has(len(fontManager.ttflist))


def _font_family_a_name_still_stands_for(
    name: Optional[str], opts: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the font keyword a text layer's ``name`` also means, or an empty mapping.

    ``matplotlib.axes.Axes.text`` and ``.annotate`` document ``name=`` as an alias of the font family, and
    ``Text``/``Annotation`` are the only two of matplotlib's artists that answer to it — so of the
    thirty-six builders #321 gave a ``name=`` on this tier (counted: the public methods of ``Map`` whose
    signature takes one), these are the only two where the keyword was already taken.
    Giving the layer that name without this would drop the caller's font on the floor: ``text(...,
    name="DejaVu Serif")`` named a layer and drew in sans-serif, with nothing said (review R-M2).

    Both readings are kept instead of one being chosen: the layer takes the name, and the name is drawn in
    as the font as well **when it is the name of a family matplotlib has** — which is
    :func:`_names_a_font_family`, a membership test rather than a resolution, so the string that decides is
    the string that is drawn with (review R2-H6). That condition is what separates the two intents without
    guessing: a name naming no family would have rendered in the default font anyway, so passing it on
    would change no pixel and cost a ``findfont: Font family 'wells' not found.`` on every named label.

    It is the **drawers** that ask this, not the builders, so the answer never reaches the description.
    Whether this machine has a family is a property of the machine: writing the answer into the layer's
    keywords made one call describe itself two ways, carrying ``{"fontname": ...}`` where the family was
    installed and nothing where it was not (review R2-M3). A figure records the name the caller wrote; the
    font is read off that name wherever it is drawn.

    Args:
        name: The name the layer carries — its label, which is what the caller asked for, rather than its
            id, which a duplicate name suffixes ``-2`` into a string that is no family. ``None`` for a
            layer with no name at all.
        opts: The keywords the artist is about to be drawn with, read for a font family they already
            choose — under any spelling matplotlib takes for one, which is what
            :func:`_a_font_is_already_chosen` answers.

    Returns:
        ``{"fontname": name}`` when ``name`` is the name of a font family this machine has and the
        keywords name no other one; an empty mapping otherwise, which draws the label in whatever the
        keywords and matplotlib's defaults say.

    Examples:
        - A family matplotlib ships is read as the font as well as the layer's name:
            ```python
            >>> from digitalearth.static.maps.decoration import _font_family_a_name_still_stands_for
            >>> _font_family_a_name_still_stands_for("DejaVu Serif", {})
            {'fontname': 'DejaVu Serif'}

            ```
        - An ordinary layer name is not a font, and a call that already chose one is left alone:
            ```python
            >>> from digitalearth.static.maps.decoration import _font_family_a_name_still_stands_for
            >>> _font_family_a_name_still_stands_for("wells", {})
            {}
            >>> _font_family_a_name_still_stands_for("DejaVu Serif", {"fontname": "DejaVu Sans"})
            {}
            >>> _font_family_a_name_still_stands_for("DejaVu Serif", {"font_properties": "DejaVu Sans"})
            {}

            ```
        - A name that is only a fontconfig *pattern* for a family is not that family, which matters
          because ``-2`` is what a duplicate name is suffixed with:
            ```python
            >>> from digitalearth.static.maps.decoration import _font_family_a_name_still_stands_for
            >>> _font_family_a_name_still_stands_for("DejaVu Serif-2", {})
            {}

            ```
    """
    if not isinstance(name, str) or _a_font_is_already_chosen(opts):
        return {}
    if not _names_a_font_family(name):
        return {}
    return {"fontname": name}


#: Natural-Earth layers that ``cleopatra.basemap.reference`` renders as filled polygons (vs. line layers); used to
#: translate this package's singular matplotlib style keys to the right collection keys for ``add_features``.
_POLYGON_LAYERS = frozenset({"land", "ocean", "lakes"})

#: The Natural-Earth layer name cleopatra takes -> the registered kind the figure records (#303). Only
#: ``coastline`` differs, and it differs in both directions at once: the cross-tier kind is the plural
#: ``coastlines`` (what :meth:`DecorationMixin.coastlines` is called), while cleopatra's dataset is the
#: singular. Mapping it here keeps that one-word difference out of six builders.
_NATURAL_EARTH_KINDS = {
    "coastline": "coastlines",
    "borders": "borders",
    "land": "land",
    "ocean": "ocean",
    "lakes": "lakes",
    "rivers": "rivers",
}

#: The cross-tier basemap names — :data:`~digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER` among them —
#: keyed by their lower-cased spelling, mapped to the ``xyzservices`` dot-path
#: ``cleopatra.basemap.tiles.get_provider`` resolves. Each tier names these four token-free basemaps the same
#: way and translates to its own engine's spelling (GeoViews' ``tile_sources`` on the interactive tier, a URL
#: template on the web tier); this is the static tier's half of that. Without it a shared name reached
#: ``add_tiles`` as an unknown provider, and ``basemap()`` with no argument fell through to cleopatra's own
#: default (``OpenStreetMap.Mapnik``) — which is how the one backend that needs no extra ended up drawing a
#: different basemap from the other two (#247).
_SHARED_PROVIDERS = {
    "cartolight": "CartoDB.Positron",
    "cartodark": "CartoDB.DarkMatter",
    "cartovoyager": "CartoDB.Voyager",
    "osm": "OpenStreetMap.Mapnik",
}

#: Colormap a raster backdrop falls back to when neither the caller nor ``auto_style`` names one — the
#: hypsometric look a relief/imagery backdrop is usually wanted in. It sits *behind* the lookup rather than
#: in :meth:`DecorationMixin.stock_img`'s signature, so a backdrop whose variable is recognised (a DEM, say)
#: is coloured the same way the same data would be as a data layer.
STOCK_IMG_CMAP = "gist_earth"


def _to_feature_style(kind: str, style: dict) -> dict:
    """Translate singular matplotlib style keys to the plural collection keys ``add_features`` expects.

    ``add_features`` forwards to a ``LineCollection`` (line layers) or ``PathCollection`` (polygon layers),
    whose constructors take plural keys (``colors``/``facecolors``/``edgecolors``/``linewidths``). Map the
    singular ``color``/``facecolor``/``edgecolor``/``linewidth``/``linestyle`` that this package and its
    callers use — ``color`` becomes ``facecolors`` for polygon layers and ``colors`` for line layers — and
    pass anything else (e.g. ``alpha``, ``zorder``) through untouched. Fill/edge colours are dropped for line
    layers (a ``LineCollection`` has no face/edge), mirroring the globe line path which pops them.

    Args:
        kind: ``"polygon"`` or ``"line"`` — selects the destination for a bare ``color``.
        style: The merged default/override style with singular matplotlib keys.

    Returns:
        A style dict keyed for the underlying matplotlib collection.
    """
    mapping = {
        "facecolor": "facecolors",
        "edgecolor": "edgecolors",
        "linewidth": "linewidths",
        "linestyle": "linestyles",
    }
    out: dict = {}
    for key, value in style.items():
        if kind == "line" and key in ("facecolor", "edgecolor"):
            continue  # a LineCollection has no fill/edge; ignore these (as the globe line path does)
        if key == "color":
            out["facecolors" if kind == "polygon" else "colors"] = value
        else:
            out[mapping.get(key, key)] = value
    return out


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


#: cleopatra logs the built tile URL at DEBUG when a fetch fails. For a keyed service that URL carries the
#: credential, so the static backend silences exactly this logger while a keyed basemap is being fetched.
_CLEOPATRA_TILES_LOGGER = "cleopatra.basemap.tiles"


class _DropDebug(logging.Filter):
    """Drops the ``DEBUG`` records that carry the built tile URL, and passes everything else."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Whether a record survives.

        Args:
            record: The record cleopatra is about to emit.

        Returns:
            ``False`` for ``DEBUG``, which is the level the tile URL is logged at, and ``True`` for
            everything at ``INFO`` or above.
        """
        return record.levelno > logging.DEBUG


#: One shared instance, so nesting adds and removes the same filter rather than stacking copies.
_DROP_DEBUG = _DropDebug()


def _edge_samples(
    west: float, south: float, east: float, north: float, count: int = 21
) -> Tuple[list, list]:
    """Return points along all four edges of a box, for reprojecting it as a shape rather than a corner.

    Args:
        west: Western edge, in the box's own CRS.
        south: Southern edge.
        east: Eastern edge.
        north: Northern edge.
        count: Samples per edge. 21 is what cleopatra uses for the same job a moment later, so the
            envelope this produces and the one it tiles against agree.

    Returns:
        ``(xs, ys)``, the sampled coordinates as two parallel lists.
    """
    steps = [index / (count - 1) for index in range(count)]
    xs: list = []
    ys: list = []
    for step in steps:
        x = west + (east - west) * step
        y = south + (north - south) * step
        xs.extend([x, x, west, east])
        ys.extend([south, north, y, y])
    return xs, ys


@contextlib.contextmanager
def _quiet_tile_urls() -> Iterator[None]:
    """Drop the cleopatra tile logger's DEBUG records so a keyed URL cannot reach the logs.

    The credential is part of the tile URL — that is how the service authenticates — and cleopatra logs
    that URL on a failed fetch. Errors still surface: the ``ConnectionError`` cleopatra raises carries the
    tile coordinates, not the URL, so nothing diagnostic is lost.

    A filter is attached rather than the logger's level raised. Raising the level is process-global state
    that two threads rendering keyed basemaps can interleave on — the inner restore puts back the level
    the outer block had already changed — and it silences the logger for unrelated work in the meantime.
    Adding and removing a filter is idempotent per block and leaves the level alone.

    Yields:
        ``None``, with the filter removed on the way out even if the fetch raises.
    """
    logger_obj = logging.getLogger(_CLEOPATRA_TILES_LOGGER)
    logger_obj.addFilter(_DROP_DEBUG)
    try:
        yield
    finally:
        logger_obj.removeFilter(_DROP_DEBUG)


def _keyed_tile_provider(keyed: "KeyedTileSource", api_key: Optional[str]) -> Any:
    """Build the ``xyzservices.TileProvider`` cleopatra renders from a keyed source definition.

    This is the one place a tile-library object is constructed: :mod:`digitalearth.base.basemaps` stays
    engine-neutral and hands out a URL, and only the static backend needs the provider type that
    ``add_tiles`` resolves.

    Args:
        keyed: The resolved keyed basemap.
        api_key: Credential for it, or ``None`` to read the environment.

    Returns:
        An ``xyzservices.TileProvider`` whose URL already carries the credential.
    """
    from xyzservices import TileProvider

    return TileProvider(
        name=keyed.name,
        url=keyed.tile_url(api_key),
        attribution=keyed.attribution,
        max_zoom=keyed.max_zoom,
    )


#: cleopatra's OGC provider classes, under the ``"ogc"`` tag a figure records each one by. The tag is what
#: a figure carries, so it is spelled here and not derived from the class name, which is free to change.
_OGC_PROVIDERS: Dict[str, type] = {"wms": WMSProvider, "wmts": WMTSProvider}

#: The OGC provider field a figure never records. ``extra_params`` is documented as where a service's token
#: goes, so it is held beside the layer with the object itself — the same rule that keeps an ``xyzservices``
#: ``apikey`` out of a figure.
_OGC_SECRET_FIELDS: FrozenSet[str] = frozenset({"extra_params"})

#: The OGC provider field a figure records only the *base* of. A query string is the other ordinary way an
#: OGC service is authenticated (``?token=``, ``?api_key=``, ``?SERVICE_KEY=``), and nothing here can tell
#: such a parameter from a harmless one, so the whole query goes and :func:`_service_base_url` says which
#: parameters it dropped (round 4, H2). Every remaining recorded field names the service.
_OGC_URL_FIELD = "url"


def _service_base_url(url: Any) -> str:
    """Return an OGC service URL with its query string dropped, warning when there was one.

    A query-string credential is normal for an OGC service, and the ``url`` field used to be copied into a
    figure whole — so the same value :func:`_quiet_tile_urls` exists to keep out of the logs was written to
    JSON two functions away (round 4, H2). The parameters are dropped rather than redacted because the
    record has to stay a usable service URL, and dropped **whatever they hold**: a credential is not
    distinguishable from a parameter here, so every query is treated as one. A service that genuinely needs
    a parameter passes it through ``extra_params``, which is held beside the layer and never travels.

    Args:
        url: The provider's ``url`` field.

    Returns:
        Scheme, host and path, with query and fragment removed.

    Warns:
        UserWarning: when a query string was dropped, naming its parameters — the caller has to know to
            pass them through ``extra_params`` instead, since the replayed figure will not carry them.

    Examples:
        - A plain service URL is handed back as it was given, with no warning to ignore:
            ```python
            >>> from digitalearth.static.maps.decoration import _service_base_url
            >>> _service_base_url("https://example.org/geoserver/wms")
            'https://example.org/geoserver/wms'

            ```
        - One carrying a token keeps only its base, and the warning names every parameter it lost:
            ```python
            >>> import warnings
            >>> from digitalearth.static.maps.decoration import _service_base_url
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     _service_base_url("https://example.org/wms?token=s3cret&SERVICE=WMS")
            'https://example.org/wms'
            >>> "['SERVICE', 'token']" in str(caught[0].message)
            True

            ```
    """
    text = str(url)
    parts = urlsplit(text)
    if not parts.query and not parts.fragment:
        return text
    dropped = sorted(parse_qs(parts.query, keep_blank_values=True))
    base = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    warnings.warn(
        f"the figure records this OGC service as {base!r}: its query string is dropped because a "
        f"parameter there may be a credential, and a figure is written to JSON and read back. Dropped "
        f"parameters: {dropped}. Pass what the service needs through extra_params=, which is held beside "
        f"the layer and never recorded.",
        UserWarning,
        stacklevel=3,
    )
    return base


def _describe_ogc_provider(source: Any) -> Optional[Dict[str, Any]]:
    """Describe a cleopatra OGC provider as the plain values a figure can carry, minus every credential path.

    An OGC provider is the one source kind with no ``name`` to record, so recording it by name wrote
    ``None`` — the shared default's spelling — and a figure read back elsewhere silently drew CartoDB
    Positron in place of the service, with nothing marking what was lost (round 3, M10). Both providers are
    dataclasses of plain values, so the description is simply their fields, which is also why it needs no
    per-class spelling: a field added upstream travels without a change here.

    **Exactly what travels**: every field of the dataclass except :data:`_OGC_SECRET_FIELDS`
    (``extra_params``, dropped whole), and ``url`` as its scheme/host/path only — any query string on it is
    dropped and named in a warning (:func:`_service_base_url`). So the two ways an OGC service is normally
    authenticated both stay on the object, and what reaches a figure names the service.

    Args:
        source: A basemap source that is not a name.

    Returns:
        ``{"ogc": <tag>, <field>: <value>, …}``, or ``None`` when ``source`` is not an OGC provider.

    Warns:
        UserWarning: when the provider's ``url`` carried a query string, which is dropped
            (:func:`_service_base_url`).

    Examples:
        - A WMS service is described by its own fields, and the token it carries stays on the object:
            ```python
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> from digitalearth.static.maps.decoration import _describe_ogc_provider
            >>> service = WMSProvider(
            ...     "https://example.org/geoserver/wms", "topp:states", extra_params={"token": "s3cret"}
            ... )
            >>> described = _describe_ogc_provider(service)
            >>> described["ogc"], described["layers"], described["version"]
            ('wms', 'topp:states', '1.3.0')
            >>> "extra_params" in described, dict(service.extra_params)
            (False, {'token': 's3cret'})

            ```
        - A service URL carrying a query string is recorded as its base, and the warning names what went:
            ```python
            >>> import warnings
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> from digitalearth.static.maps.decoration import _describe_ogc_provider
            >>> service = WMSProvider("https://example.org/wms?token=s3cret", "topp:states")
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     described = _describe_ogc_provider(service)
            >>> described["url"], service.url
            ('https://example.org/wms', 'https://example.org/wms?token=s3cret')
            >>> "Dropped parameters: ['token']" in str(caught[0].message)
            True

            ```
        - Anything that is not an OGC provider is not this function's business, and says so:
            ```python
            >>> from digitalearth.static.maps.decoration import _describe_ogc_provider
            >>> print(_describe_ogc_provider("CartoLight"))
            None

            ```
    """
    tag = next(
        (name for name, kind in _OGC_PROVIDERS.items() if isinstance(source, kind)),
        None,
    )
    if tag is None:
        return None
    described: Dict[str, Any] = {"ogc": tag}
    for spec in dataclass_fields(source):
        if spec.name in _OGC_SECRET_FIELDS:
            continue
        value = getattr(source, spec.name)
        described[spec.name] = (
            _service_base_url(value) if spec.name == _OGC_URL_FIELD else value
        )
    return described


def _ogc_provider_from_description(described: Mapping[str, Any]) -> Any:
    """Rebuild a cleopatra OGC provider from what a figure recorded of it.

    The credential the original carried is **not** rebuilt: :data:`_OGC_SECRET_FIELDS` never travels, and
    neither does the ``url``'s query string (:func:`_service_base_url`), so a replayed figure asks the same
    service with ``extra_params`` empty and a bare base URL. A service that needs a token answers with its
    own error, which is the honest outcome — the alternative was drawing a different basemap in silence.

    **A figure this version cannot rebuild is refused the same way, whichever part of it is unfamiliar.**
    :func:`_describe_ogc_provider` records field by field so that "a field added upstream travels without
    a change here" — which was true of writing and false of reading, because the description used to be
    splatted into the constructor: a figure carrying a field this release has never heard of came back as
    ``TypeError: WMSProvider.__init__() got an unexpected keyword argument 'newthing'``, naming a dunder
    rather than the figure, and one short of a required field came back as ``TypeError:
    WMSProvider.__init__() missing 1 required positional argument: 'layers'`` (executed; round 4, M7).
    Both are now the ``ValueError`` the unknown-tag case already gave, naming the fields, what the kind
    does take and which of those it requires. The unknown field is *refused* rather than dropped with a
    warning: a provider field is what the service is asked with, so a figure drawn without one is a
    different request, and silently drawing a different thing is the failure this whole record exists to
    end (round 3, M10).

    Args:
        described: What :func:`_describe_ogc_provider` recorded.

    Returns:
        The rebuilt provider, which ``add_tiles`` tiles from, with no credential on it.

    Raises:
        ValueError: when the recorded ``ogc`` tag is not one this version knows, when the description
            carries a field the provider of that kind does not take, or when it is missing one that kind
            requires. All three are the same scenario — a figure written by another release — and are
            worth naming rather than drawing the default for.

    Examples:
        - What :func:`_describe_ogc_provider` wrote comes back as the same request, with ``extra_params``
          empty because the description never carried it:
            ```python
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> from digitalearth.static.maps.decoration import (
            ...     _describe_ogc_provider, _ogc_provider_from_description,
            ... )
            >>> described = _describe_ogc_provider(
            ...     WMSProvider("https://example.org/wms", "topp:states", extra_params={"token": "s3"})
            ... )
            >>> rebuilt = _ogc_provider_from_description(described)
            >>> rebuilt.url, rebuilt.layers, dict(rebuilt.extra_params)
            ('https://example.org/wms', 'topp:states', {})

            ```
        - A field this release does not take, and one it requires but the figure omits, are both named
          rather than left to the constructor:
            ```python
            >>> from digitalearth.static.maps.decoration import _ogc_provider_from_description
            >>> base = {"ogc": "wms", "url": "https://example.org/wms", "layers": "topp:states"}
            >>> try:
            ...     _ogc_provider_from_description({**base, "newthing": 1})
            ... except ValueError as error:
            ...     print(str(error).split(". It takes")[0])
            this figure records an OGC basemap of kind 'wms' that this version of digitalearth cannot rebuild: field(s) ['newthing'] are not ones it takes and field(s) [] are missing
            >>> try:
            ...     _ogc_provider_from_description({"ogc": "wms", "url": "https://example.org/wms"})
            ... except ValueError as error:
            ...     print(str(error).split("cannot rebuild: ")[1].split(". It takes")[0])
            field(s) [] are not ones it takes and field(s) ['layers'] are missing

            ```
        - A kind this version has never heard of is named too, with the kinds it does know:
            ```python
            >>> from digitalearth.static.maps.decoration import _ogc_provider_from_description
            >>> try:
            ...     _ogc_provider_from_description({"ogc": "wcs", "url": "https://example.org/wcs"})
            ... except ValueError as error:
            ...     print(error)
            this figure records an OGC basemap of kind 'wcs', which this version of digitalearth cannot rebuild; it knows ['wms', 'wmts']

            ```
    """
    tag = str(described.get("ogc", ""))
    provider = _OGC_PROVIDERS.get(tag)
    if provider is None:
        raise ValueError(
            f"this figure records an OGC basemap of kind {tag!r}, which this version of digitalearth "
            f"cannot rebuild; it knows {sorted(_OGC_PROVIDERS)}"
        )
    recorded = {key: value for key, value in described.items() if key != "ogc"}
    specs = dataclass_fields(provider)
    known = {spec.name for spec in specs}
    required = {
        spec.name
        for spec in specs
        if spec.default is MISSING and spec.default_factory is MISSING
    }
    unexpected = sorted(set(recorded) - known)
    absent = sorted(required - set(recorded))
    if unexpected or absent:
        raise ValueError(
            f"this figure records an OGC basemap of kind {tag!r} that this version of digitalearth "
            f"cannot rebuild: field(s) {unexpected} are not ones it takes and field(s) {absent} are "
            f"missing. It takes {sorted(known)}, of which it requires {sorted(required)}"
        )
    return provider(**recorded)


def _described_tile_source(source: Any) -> Tuple[Any, Any]:
    """Split a basemap source into what a description records and the object held beside the layer.

    An ``xyzservices.TileProvider`` is a ``dict`` subclass whose fields include the caller's ``apikey``, so
    recording it wrote that credential into every figure the map produced — and rebuilding it from the
    description handed cleopatra a plain ``dict``, which has no ``build_url`` and cannot be tiled from. Its
    **name** is what a description carries: it names the same tiles wherever the figure is read, and it
    carries no secret.

    An OGC provider has no name, so it is recorded field by field instead
    (:func:`_describe_ogc_provider`) — everything but the ``extra_params`` a token goes in. The object is
    still held beside the layer, because the map drawing now is the one that may legitimately use that
    token; the description is for the figure read somewhere else.

    Args:
        source: The ``source`` a caller passed to :meth:`DecorationMixin.basemap`.

    Returns:
        ``(recorded, held)``. A name or ``None`` passes through and is held nowhere. An OGC provider is
        recorded as a field mapping, any other provider object by its own ``name`` — or as ``None``, the
        shared default, when it has none — and either is held beside the layer, which is what this map
        tiles from.
    """
    if source is None or isinstance(source, str):
        return source, None
    described = _describe_ogc_provider(source)
    if described is not None:
        return described, source
    name = getattr(source, "name", None)
    return (name if isinstance(name, str) else None), source


def _resolve_tile_source(source: Any) -> Any:
    """Resolve an unnamed or cross-tier-named basemap into the provider ``add_tiles`` understands.

    Three translations, the first two of them the static tier's side of the one-basemap-per-name contract
    (#247). ``None`` means "the caller named none", which is
    :data:`~digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER` and not cleopatra's own fallback. A
    shared name such as ``"CartoLight"`` is spelled as the ``xyzservices`` path that resolves to the very
    tiles the other tiers request for it. And a recorded OGC provider — a mapping carrying an ``"ogc"``
    tag, which is what a stored figure holds in place of a name — is rebuilt into the provider object it
    describes (:func:`_ogc_provider_from_description`). Anything else — an ``xyzservices`` path, a
    ``TileProvider``, an OGC provider the caller built, a URL — is handed on untouched.

    Args:
        source: The ``source`` a caller passed to :meth:`DecorationMixin.basemap`, or what a figure
            recorded in its place, already known not to be a keyed preset.

    Returns:
        The provider name or object to hand ``cleopatra.basemap.tiles.add_tiles``.

    Raises:
        ValueError: when a recorded OGC provider names a kind this version cannot rebuild.

    Examples:
        - No source at all resolves to the shared default, in this engine's spelling:
            ```python
            >>> from digitalearth.static.maps.decoration import _resolve_tile_source
            >>> _resolve_tile_source(None)
            'CartoDB.Positron'

            ```
        - A shared name is translated case-insensitively, and anything else passes through:
            ```python
            >>> from digitalearth.static.maps.decoration import _resolve_tile_source
            >>> (_resolve_tile_source("cartodark"), _resolve_tile_source("Esri.WorldImagery"))
            ('CartoDB.DarkMatter', 'Esri.WorldImagery')

            ```
        - A figure's record of a WMS service is rebuilt into the provider that tiles it:
            ```python
            >>> from digitalearth.static.maps.decoration import _resolve_tile_source
            >>> stored = {"ogc": "wms", "url": "https://example.test/wms", "layers": "topp:states"}
            >>> _resolve_tile_source(stored).layers
            'topp:states'

            ```
    """
    if isinstance(source, Mapping) and "ogc" in source:
        return _ogc_provider_from_description(source)
    name = DEFAULT_BASEMAP_PROVIDER if source is None else source
    if isinstance(name, str):
        return _SHARED_PROVIDERS.get(name.lower(), name)
    return name


def draw_text(scene: Any, _data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Place the text label a described layer asks for, at the lon/lat it recorded.

    The label's font is read from its **name** here, which is where that belongs: matplotlib documents
    ``name=`` as an alias of the font family, so a layer named for a family is drawn in it as well as
    called it (review R-M2), but whether a family is installed is a fact about the machine drawing rather
    than about the figure (review R2-M3). See :func:`_font_family_a_name_still_stands_for`.

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — a label draws from a coordinate pair and
            a string, both of which the description carries.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``Text``, or ``None`` when the point
        is on the far side of a globe and reprojects to nothing: a layer that was not drawn rather than an
        empty one.
    """
    props = dict(layer.symbology.props)
    xy = scene._reproject_point(props["lon"], props["lat"], props["crs"])
    if xy is None:
        return None
    style = drawing_style(scene, layer)
    style.update(_font_family_a_name_still_stands_for(layer.label, style))
    drawn = scene.ax.text(xy[0], xy[1], props["s"], **style)
    return DrawnLayer(artist=drawn, artists=(drawn,))


def draw_annotate(scene: Any, _data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Annotate the lon/lat a described layer recorded, optionally with an arrow.

    An ``Annotation`` is a ``Text``, so its name reads as a font family here exactly as in
    :func:`draw_text`, and for the same reason.

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — see :func:`draw_text`.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``Annotation``, or ``None`` when the
        annotated point is on the far side of a globe.
    """
    props = dict(layer.symbology.props)
    xy = scene._reproject_point(props["lon"], props["lat"], props["crs"])
    if xy is None:
        return None
    style = drawing_style(scene, layer)
    style.update(_font_family_a_name_still_stands_for(layer.label, style))
    drawn = scene.ax.annotate(props["s"], xy=xy, xytext=props["xytext"], **style)
    return DrawnLayer(artist=drawn, artists=(drawn,))


def draw_natural_earth(
    scene: Any, _data: Any, layer: LayerSpec
) -> Optional[DrawnLayer]:
    """Draw the Natural-Earth layer a description names, the way this map's projection needs it drawn.

    Three renders behind one recipe, because they are three ways of putting the same reference geography on
    the axes rather than three layers: a globe fills polygons re-closed at the projection limb, a globe
    draws lines split at it, and a flat map hands the whole thing to ``add_features`` — which is hole-aware
    and reprojects for itself.

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — the geography is Natural Earth's, named by
            the layer rather than handed to it.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding whatever that path produced — the
        ``PolyCollection`` of a globe fill, the polylines of a globe line layer, the axes on a flat map — or
        ``None`` when the whole layer is on the far side of the globe, which is a layer that was not drawn.
    """
    props = dict(layer.symbology.props)
    name, zorder = props["via"], props["zorder"]
    style = {**_NATURAL_EARTH_STYLE[name], **drawing_style(scene, layer)}
    if scene.globe:
        if name == "ocean":
            # The disc *is* the ocean: filling the whole projection boundary and letting land overlay it
            # is exact and far cheaper than clipping the global ocean polygon.
            disc: Optional[DrawnLayer] = scene._fill_globe_polygons(
                [np.asarray(scene._frame()[0])],
                zorder=zorder,
                **_to_feature_style("polygon", style),
            )
            return disc
        parts = natural_earth(name, props["resolution"])
        if props["polygon"]:
            filled: Optional[DrawnLayer] = scene._fill_globe_polygons(
                scene._project_polygon_features(parts),
                zorder=zorder,
                **_to_feature_style("polygon", style),
            )
            return filled
        style.pop("edgecolor", None)
        style.pop("facecolor", None)
        drawn = [
            scene.ax.plot(seg[:, 0], seg[:, 1], **style)[0]
            for seg in scene._project_line_features(parts)
        ]
        # One layer, however many polylines the limb split it into — and none at all when the whole layer
        # is on the far side, which is a layer that was not drawn rather than an empty one.
        return DrawnLayer(artist=drawn, artists=tuple(drawn)) if drawn else None
    kind = "polygon" if name in _POLYGON_LAYERS else "line"
    had_data = (
        bool(scene.layers)
        or bool(scene.ax.images)
        or bool(scene.ax.collections)
        or bool(scene.ax.lines)
    )
    with artists_added(scene.ax) as drawn_features:
        add_features(
            scene.ax,
            name,
            props["resolution"],
            crs=scene.crs,
            zorder=zorder,
            **_to_feature_style(kind, style),
        )
    if (
        not had_data
    ):  # add_features pinned the (empty) view; fit it to the layer just drawn
        scene.ax.autoscale()
    # `add_features` hands the axes back rather than what it drew, so the artists the layer owns are the
    # ones that appeared on the axes while it ran. The public return stays the axes, as it always was.
    return DrawnLayer(artist=scene.ax, artists=tuple(drawn_features))


#: The axes limits matplotlib starts an unused axes at. cleopatra reads the same pair as "nothing has been
#: drawn here yet" and refuses to tile it, which is the state a replayed basemap meets (round 2, H2).
_UNFRAMED_LIMITS: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)


def _axes_extent(axes: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return what an axes is looking at as ``(xmin, xmax, ymin, ymax)``, or ``None`` when it is unframed.

    The ordering is matplotlib's own, because the axes are what it is read off and written back to; it is
    **not** what :meth:`~digitalearth.static.maps.projection.ProjectionMixin.set_bounds` takes, which reads
    ``(west, south, east, north)`` like every other tier. The values are in the display CRS, so the frame is
    read and written back without a reprojection.

    Args:
        axes: The axes to read.

    Returns:
        Four plain floats, or ``None`` when the axes still shows matplotlib's default unit square, which
        means nothing has framed it yet rather than that it is looking at one square metre.
    """
    xmin, xmax = (float(value) for value in axes.get_xlim())
    ymin, ymax = (float(value) for value in axes.get_ylim())
    limits = (xmin, xmax, ymin, ymax)
    return None if limits == _UNFRAMED_LIMITS else limits


def _frame_from(scene: Any, extent: Optional[Sequence[float]]) -> None:
    """Frame an unframed axes on the extent a stored layer recorded, so its drawer has a view to work in.

    Args:
        scene: The map being drawn on.
        extent: The recorded ``(xmin, xmax, ymin, ymax)`` in the display CRS, or ``None`` for a layer whose
            description carries none — a figure written before the frame was recorded, or hand-built.

    Note:
        A no-op unless **both** are true: the axes is still at matplotlib's default unit square, and the
        description carries a frame. So it only ever supplies a view nothing else has, and never overrides
        the one a data layer or the caller set.
    """
    if extent is None or _axes_extent(scene.ax) is not None:
        return
    xmin, xmax, ymin, ymax = (float(value) for value in extent)
    scene.ax.set_xlim(xmin, xmax)
    scene.ax.set_ylim(ymin, ymax)


def draw_basemap(scene: Any, _data: Any, layer: LayerSpec) -> DrawnLayer:
    """Fetch and draw the XYZ tiles a described basemap asks for.

    Drawn from the description rather than described after the draw, which is the same guarantee said the
    other way round: tiles come off the network, and :meth:`~digitalearth.static.scene.Scene._draw` drops
    the description again when this raises, so a figure never names a basemap the tier could not fetch.

    **The axes is framed from the description when nothing else has framed it.** A basemap is an underlay,
    so a figure lists it before the data — while the builders run the other way round, data first, because
    ``add_tiles`` tiles whatever the axes is already looking at. Replaying a stored figure therefore reaches
    this drawer with an axes matplotlib has not framed yet, and cleopatra refuses to tile one
    (``ValueError: Axes have no data extent``). The frame the tiles were fetched for is recorded with the
    layer for exactly that reason (``extent``), and is put back here — so the replay asks for the same
    mosaic at the same zoom rather than for whatever a fresh axes would imply (round 2, H2). An axes that
    *is* framed — the caller's own order, and the second pass of a reconcile — is left as it stands.

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — a tile set is named by its provider, which
            is a style property, not data this process holds.
        layer: The layer's description. The credential, and a provider object the caller built (which
            carries one of its own), are held on the scene under the layer's id — never written into a
            figure.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` whose ``artist`` is whatever ``add_tiles``
        returned — the axes, which is this builder's public return — and whose ``artists`` are the tile
        images it actually added, which is what hiding and removing the layer reach.

    Raises:
        ValueError: when a keyed preset's credential is unavailable, when the extent being drawn lies
            entirely outside its coverage, or when neither the axes nor the description carries a frame to
            tile.
        TypeError: when a preset keyword is missing or misspelled.
    """
    props = dict(layer.symbology.props)
    _frame_from(scene, props.get("extent"))
    opts = drawing_style(scene, layer)
    # The provider object, when the caller passed one: held beside the layer, because it is an engine object
    # and carries their credential. A figure read back elsewhere has only the name the description records.
    source = opts.pop("source", props["source"])
    if not is_keyed_basemap(source):
        with artists_added(scene.ax) as drawn_tiles:
            tiles = add_tiles(
                scene.ax, source=_resolve_tile_source(source), crs=scene.crs, **opts
            )
        return DrawnLayer(artist=tiles, artists=tuple(drawn_tiles))
    keyed = get_keyed_basemap(str(source), **dict(props["preset"]))
    keyed.check_bounds(scene._coverage_extent())
    provider = _keyed_tile_provider(keyed, scene._layer_keys.get(layer.id))
    opts.setdefault("attribution", keyed.attribution)
    with _quiet_tile_urls(), artists_added(scene.ax) as drawn_tiles:
        tiles = add_tiles(scene.ax, source=provider, crs=scene.crs, **opts)
    return DrawnLayer(artist=tiles, artists=tuple(drawn_tiles))


#: The latitude a night-shade vertex is pulled back to before it is projected. A night region that holds a
#: pole runs along that pole's map edge; cleopatra drops a non-finite vertex, so a projection with no image
#: of the pole would lose that run and close the shade straight along the terminator, leaving the polar cap
#: unshaded. PROJ clamps rather than returning infinity, but it clamps enormously far out: Web Mercator
#: places latitude 90 at ``y ≈ 2.4e8`` m — twelve times the map's own half-height — and Antarctic Polar
#: Stereographic (EPSG:3031) places the *opposite* pole at ``y ≈ 4.0e23`` m. A tenth of a degree short of the
#: pole is a point every projection places finitely, and a much closer one — the same Web Mercator vertex
#: lands at ``y ≈ 4.5e7`` m — but it is still outside the map, and in EPSG:3031 it is still ``y ≈ 1.4e10`` m
#: out. So this clamp is what keeps the *shade* whole, and not what keeps the *view* usable: that is
#: :func:`_fit_within_crs_extent`'s job (round 3, M2).
_NIGHT_POLE_LATITUDE = 89.9

#: The centres ``tissot()`` draws when it is given none: every 30 degrees of longitude, offset from the
#: antimeridian, and every 30 degrees of latitude from 60 S to 60 N — enough to read the distortion anywhere a
#: world map is read, and clear of the poles, where every projection's indicatrix degenerates.
_TISSOT_LONS: Tuple[float, ...] = tuple(float(lon) for lon in range(-165, 180, 30))
_TISSOT_LATS: Tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0)

#: The ground radius of a Tissot circle when none is given, in metres.
DEFAULT_TISSOT_RADIUS_M = 500_000.0

#: How a night shade looks when the caller asks for nothing else — cleopatra's own ``add_nightshade``
#: defaults, spelled out for the globe path, which fills the limb-clipped rings itself.
_NIGHT_STYLE: Dict[str, Any] = {
    "facecolor": "black",
    "edgecolor": "none",
    "alpha": 0.35,
}

#: The two spellings of a night shade's fill. ``None`` in either means "not given" — matplotlib's own
#: reading of it — so :meth:`DecorationMixin.nightshade` drops it and the shade falls back to
#: :data:`_NIGHT_STYLE`. Passing it through instead resolved to matplotlib's first *cycle* colour on a
#: flat map, drawing a night shade in blue, and raised ``Invalid RGBA argument: None`` on a globe
#: (round 4, L8).
_NIGHT_FILL_KEYWORDS: FrozenSet[str] = frozenset({"color", "facecolor"})


def _utc_moment(when: Any) -> datetime:
    """Read the instant a night shade is drawn for, as an aware UTC ``datetime``.

    Args:
        when: A ``datetime`` or ISO 8601 text. One with no zone is UTC, which is what cleopatra assumes too.

    Returns:
        The same instant, in UTC.

    Raises:
        TypeError: when ``when`` is neither a ``datetime`` nor text.
        ValueError: when the text is not ISO 8601.
    """
    if isinstance(when, str):
        when = datetime.fromisoformat(when)
    if not isinstance(when, datetime):
        raise TypeError(
            f"nightshade(when=) takes a datetime or ISO 8601 text; got {type(when).__name__}"
        )
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def _lonlat_to_display(scene: Any, ring: np.ndarray) -> np.ndarray:
    """Project one lon/lat ring into the display CRS through pyramids.

    Args:
        scene: The map whose display CRS the ring is projected into.
        ring: An ``(m, 2)`` lon/lat array.

    Returns:
        The ``(m, 2)`` projected array; a vertex the projection cannot place is non-finite.
    """
    x, y = reproject_coordinates(
        ring[:, 0].tolist(), ring[:, 1].tolist(), from_crs=4326, to_crs=scene.crs
    )
    return np.column_stack([np.asarray(x, float), np.asarray(y, float)])


def _night_ring_projector(scene: Any) -> Any:
    """Return the ``transform`` cleopatra's ``add_nightshade`` maps each lon/lat night ring through.

    Each ring is densified first, so its straight lon/lat edges — the run along a pole's map edge above all —
    bend with the projection, and its pole vertices are pulled back to :data:`_NIGHT_POLE_LATITUDE` so a
    projection with no finite image of the pole does not throw them orders of magnitude off the map.

    Args:
        scene: The map being drawn on.

    Returns:
        A callable ``(m, 2) lon/lat -> (k, 2) display`` array.
    """

    def project(ring: np.ndarray) -> np.ndarray:
        """Densify, pull back from the poles, and project one ring.

        Args:
            ring: An ``(m, 2)`` lon/lat ring.

        Returns:
            The projected ring.
        """
        dense = projections.densify_lonlat(np.asarray(ring, dtype=float), step_deg=1.0)
        dense[:, 1] = np.clip(dense[:, 1], -_NIGHT_POLE_LATITUDE, _NIGHT_POLE_LATITUDE)
        return _lonlat_to_display(scene, dense)

    return project


def _crs_display_extent(crs: Any) -> Optional[Tuple[float, float, float, float]]:
    """Measure a display CRS's own extent, in its own coordinates, from the area of use it declares.

    The lon/lat area of use is pyproj's, read from the CRS definition; it is projected here rather than
    looked up, because what a view has to be bounded by is a box in *display* coordinates. The edges are
    sampled rather than the four corners, for the reason :meth:`_axes_lonlat_extent` samples them.

    Args:
        crs: The display CRS.

    Returns:
        ``(west, south, east, north)`` in the CRS's own units, or ``None`` when it declares no area of use —
        which a bare PROJ string such as ``+proj=robin`` does not — or when nothing in it projects finitely.
        ``None`` means "unknown", and leaves the view alone.

    Examples:
        - Antarctic Polar Stereographic declares a polar cap, which comes back as the square of metres its
          own map occupies — 1/4223 of the ``1.4e10`` m out where it places latitude
          :data:`_NIGHT_POLE_LATITUDE`, which is what makes this bound worth applying:
            ```python
            >>> from digitalearth.static.maps.decoration import _crs_display_extent
            >>> [round(value) for value in _crs_display_extent(3031)]
            [-3333134, -3333134, 3333134, 3333134]

            ```
        - A bare PROJ string declares no area of use, so there is no bound to apply and the answer is
          ``None`` rather than a guess:
            ```python
            >>> from digitalearth.static.maps.decoration import _crs_display_extent
            >>> print(_crs_display_extent("+proj=robin"))
            None

            ```
    """
    try:
        area = crs_from_user_input(crs).area_of_use
    except (CRSError, TypeError, ValueError):
        return None
    if area is None:
        return None
    west, south, east, north = (float(value) for value in area.bounds)
    xs, ys = _edge_samples(west, south, east, north)
    try:
        x, y = reproject_coordinates(xs, ys, from_crs=4326, to_crs=crs)
    except (ValueError, RuntimeError):
        return None
    finite_x = [value for value in x if math.isfinite(value)]
    finite_y = [value for value in y if math.isfinite(value)]
    if not finite_x or not finite_y:
        return None
    return min(finite_x), min(finite_y), max(finite_x), max(finite_y)


def _fit_within_crs_extent(scene: Any) -> None:
    """Pull a freshly autoscaled view back inside the display CRS's own extent.

    A night region that holds a pole is shaded to :data:`_NIGHT_POLE_LATITUDE`, and a projection with no
    image of the pole places that vertex far outside its own map — ``y ≈ 1.4e10`` m in EPSG:3031 against the
    ``± 3.3e6`` m it declares for itself. Autoscaling a bare axes to it framed a December polar night at
    ``± 1.5e10`` m, roughly 4600 times the projection's extent, with Antarctica a sub-pixel dot (executed;
    round 3, M2). Intersecting the fitted view with the declared extent keeps the fit where the projection
    has something to show. It is a real trade and not a free one: of the ``1346`` vertices a December polar
    night's outline carries on EPSG:3031, ``1215`` lie outside the declared cap, so most of the overlay's
    own outline is off the view (executed). What those vertices describe is the ring's run out toward the
    pole the projection has no image of — out to ``1.4e10`` m from a cap that ends at ``3.3e6`` m — and the
    night over the cap itself is framed and drawn exactly as it was.

    The narrowed view is set with ``auto=True``, so **autoscaling stays on**. The declared area of use is
    not the projection's domain — an EPSG:3031 grid out at ``± 7e6`` m projects perfectly well — and
    matplotlib turns autoscaling off on an axis whose limits are set outright, so without that flag the
    overlay froze the map at its own fit and every later layer was cropped into it: a raster spanning
    ``± 7e6`` m stayed at the ``± 3.333e6`` m this fit had set, over half of it invisible (executed; round
    4, H1). The bound is the shade's frame, not the map's.

    A CRS that declares no area of use is left alone, as is a view that does not meet the extent at all —
    an unknown bound is not a reason to move a view that matplotlib has just fitted to real geometry.

    Args:
        scene: The map whose axes was just autoscaled.

    Examples:
        - A December night shade on a bare EPSG:3031 axes is framed at the cap the projection declares,
          and the axes is still autoscaling afterwards:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth import Map
            >>> with Map(crs=3031) as antarctic:
            ...     _ = antarctic.nightshade("2026-12-21T12:00:00+00:00")
            ...     [round(value) for value in antarctic.ax.get_xlim()]
            ...     antarctic.ax.get_autoscalex_on()
            [-3333134, 3333134]
            True

            ```
        - Because autoscaling stayed on, a later layer out past that cap frames itself rather than being
          cropped into the overlay's fit:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth import Map
            >>> wide = Dataset.from_array(
            ...     arr=np.arange(100.0).reshape(10, 10),
            ...     geo_ref=GeoReference(geo=(-7.0e6, 1.4e6, 0.0, 7.0e6, 0.0, -1.4e6), epsg=3031),
            ...     no_data_value=-9999.0,
            ... )
            >>> with Map(crs=3031) as antarctic:
            ...     _ = antarctic.nightshade("2026-12-21T12:00:00+00:00")
            ...     _ = antarctic.field(wide)
            ...     [round(value) for value in antarctic.ax.get_xlim()]
            [-7000000, 7000000]

            ```
    """
    extent = _crs_display_extent(scene.crs)
    if extent is None:
        return
    west, south, east, north = extent
    xmin, xmax = (float(value) for value in scene.ax.get_xlim())
    ymin, ymax = (float(value) for value in scene.ax.get_ylim())
    left, right = max(xmin, west), min(xmax, east)
    bottom, top = max(ymin, south), min(ymax, north)
    if right > left:
        scene.ax.set_xlim(left, right, auto=True)
    if top > bottom:
        scene.ax.set_ylim(bottom, top, auto=True)


#: Samples per side of the display grid a globe's night shade is evaluated on **at the default ``n``**.
#: Another ``n`` scales it in proportion (:func:`_globe_night_grid`), so the figure's recorded ``n`` means
#: something on a globe too, and the default output is exactly what it was.
_GLOBE_NIGHT_GRID = 400

#: The fewest samples per side a globe's grid is ever evaluated on. Below this the contour stops tracing a
#: terminator and starts tracing the grid, which is worse than being slow.
_GLOBE_NIGHT_GRID_MIN = 16

#: The fewest samples that can describe a ring. cleopatra refuses anything smaller while sampling a flat
#: terminator; this is the same floor, stated here so the two frames answer alike (see :func:`_whole_samples`).
_MIN_RING_SAMPLES = 4

#: The open and closed ends of the solar altitude that can define a terminator, in degrees. ``0`` is the
#: geometric terminator and is **in** range; ``-90`` is the nadir, which the sun cannot be below, so it is
#: not. cleopatra refuses the same interval inside ``add_nightshade``, which only the flat path calls; this
#: is the same interval, stated here so the two frames answer alike (see :func:`_terminator_altitude`).
_MIN_TERMINATOR_ALTITUDE = -90.0

#: The closed end of that interval — see :data:`_MIN_TERMINATOR_ALTITUDE`.
_MAX_TERMINATOR_ALTITUDE = 0.0


def _whole_samples(samples: Any, caller: str) -> int:
    """Return ``n`` as a sample count that can describe a ring, refusing anything that cannot.

    cleopatra refuses a count below four while sampling a flat terminator, and it names the keyword. A globe
    never reaches that sampler — it fills by sun altitude over a display grid — so the same keyword used to be
    refused on one frame and silently accepted on the other, drawing from a grid scaled off a negative ``n``
    and recording it in the figure. One keyword answers one way now, whatever frame it is drawn on.

    Args:
        samples: The caller's ``n``.
        caller: The public method, named in the refusal the way every other refusal in this module names it.

    Returns:
        The count as an ``int``.

    Raises:
        ValueError: when ``n`` is not a whole number of at least :data:`_MIN_RING_SAMPLES`. A bool is refused
            too: ``True`` is an integer to Python, and one sample is not a ring.

    Examples:
        - A usable count comes back as an ``int``; a count too small to form a ring is refused by name:
            ```python
            >>> from digitalearth.static.maps.decoration import _whole_samples
            >>> _whole_samples(180, "nightshade()")
            180
            >>> _whole_samples(3, "nightshade()")
            Traceback (most recent call last):
                ...
            ValueError: nightshade() needs n= as a whole number >= 4, the fewest that describe a ring; got 3

            ```
    """
    if isinstance(samples, bool) or not isinstance(samples, numbers.Integral):
        raise ValueError(
            f"{caller} needs n= as a whole number >= {_MIN_RING_SAMPLES}, the fewest that describe a ring; "
            f"got {samples!r}"
        )
    count = int(samples)
    if count < _MIN_RING_SAMPLES:
        raise ValueError(
            f"{caller} needs n= as a whole number >= {_MIN_RING_SAMPLES}, the fewest that describe a ring; "
            f"got {count}"
        )
    return count


def _terminator_altitude(refraction: Any, caller: str) -> float:
    """Return ``refraction`` as a solar altitude that can define a terminator, refusing anything that cannot.

    The sibling of :func:`_whole_samples`, and it exists for the same reason. cleopatra refuses the range
    inside ``add_nightshade``, which only the flat path calls: a globe fills by sun altitude over a display
    grid and fed the value straight into ``sin(radians(refraction))``, so ``refraction=10`` painted a large
    part of the **day** side as night, and a value past the nadir answered either a wrong shade or ``None``
    — which the public API documents as "a globe whose visible side holds no night", making an out-of-range
    argument indistinguishable from a legitimate empty result (round 4, M2). One keyword answers one way
    now, whatever frame it is drawn on.

    Args:
        refraction: The caller's ``refraction``.
        caller: The public method, named in the refusal the way every other refusal in this module names it.

    Returns:
        The altitude as a ``float``.

    Raises:
        ValueError: when ``refraction`` is not a real number in
            ``(_MIN_TERMINATOR_ALTITUDE, _MAX_TERMINATOR_ALTITUDE]`` — above the horizon there is no night
            to shade, and below the nadir there is no altitude the sun can stand at.

    Examples:
        - The twilight lines pass, and the geometric terminator at the closed end of the range with them;
          an altitude above the horizon is refused by name:
            ```python
            >>> from digitalearth.static.maps.decoration import _terminator_altitude
            >>> (_terminator_altitude(0, "nightshade()"), _terminator_altitude(-18.0, "nightshade()"))
            (0.0, -18.0)
            >>> _terminator_altitude(10.0, "nightshade()")
            Traceback (most recent call last):
                ...
            ValueError: nightshade() needs refraction= in (-90, 0] degrees, where 0 is the geometric terminator and -6/-12/-18 the civil, nautical and astronomical twilight lines; got 10.0

            ```
    """
    if isinstance(refraction, bool) or not isinstance(refraction, numbers.Real):
        altitude = math.nan
    else:
        altitude = float(refraction)
    if not (
        _MIN_TERMINATOR_ALTITUDE < altitude <= _MAX_TERMINATOR_ALTITUDE
    ):  # a nan fails both comparisons, which is the answer for a non-number too
        raise ValueError(
            f"{caller} needs refraction= in ({_MIN_TERMINATOR_ALTITUDE:.0f}, "
            f"{_MAX_TERMINATOR_ALTITUDE:.0f}] degrees, where 0 is the geometric terminator and "
            f"-6/-12/-18 the civil, nautical and astronomical twilight lines; got {refraction!r}"
        )
    return altitude


def _globe_night_grid(samples: int) -> int:
    """Translate a night shade's ``n`` into samples per side of the globe's display grid.

    The flat path spends ``n`` on a one-dimensional curve; the globe spends its budget on a two-dimensional
    field, so the two cannot take the same number. They are tied proportionally instead:
    :data:`_GLOBE_NIGHT_GRID` is the side at cleopatra's default ``n``, and another ``n`` moves it in
    proportion. So the default globe is pixel-for-pixel what it was, a figure that asked for fewer samples
    now gets a cheaper globe, and one that asked for more gets a finer one — where before ``n`` was recorded
    in the figure and then ignored here entirely (round 3, L7).

    Args:
        samples: The layer's recorded ``n``, as the caller gave it and the figure kept it. A count too
            small to describe a ring never reaches here from the public API: :meth:`nightshade` refuses it
            in the builder, on both frames (:func:`_whole_samples`), so ``nightshade(when, n=-5)`` on a
            globe raises rather than drawing. That was not always so — cleopatra's refusal lives in the
            terminator sampler, which only the flat path calls, so a globe used to take any ``n`` and
            record it (round 3, L7) — which is why the floor below is now belt-and-braces rather than the
            only guard. It still earns its place: this function is reached from a *stored* figure too, and
            a figure can be hand-built or written by another tool.

    Returns:
        Samples per side, never below :data:`_GLOBE_NIGHT_GRID_MIN` — which is what a zero, negative or
        otherwise unusable ``n`` resolves to, rather than a grid with nothing in it.

    Examples:
        - The default ``n`` keeps the grid the globe has always used, and a quarter of it quarters the side:
            ```python
            >>> from cleopatra.basemap.solar import DEFAULT_TERMINATOR_SAMPLES
            >>> from digitalearth.static.maps.decoration import _globe_night_grid
            >>> (_globe_night_grid(DEFAULT_TERMINATOR_SAMPLES), _globe_night_grid(180))
            (400, 100)

            ```
        - A count the public API refuses — reachable here only from a stored figure — stops at the floor
          rather than degenerating into the grid itself:
            ```python
            >>> from digitalearth.static.maps.decoration import _globe_night_grid
            >>> (_globe_night_grid(4), _globe_night_grid(0), _globe_night_grid(-5))
            (16, 16, 16)

            ```
    """
    side = int(round(_GLOBE_NIGHT_GRID * int(samples) / DEFAULT_TERMINATOR_SAMPLES))
    return max(side, _GLOBE_NIGHT_GRID_MIN)


def _globe_nightshade(
    scene: Any, when: datetime, refraction: float, samples: int, style: Dict[str, Any]
) -> Optional[DrawnLayer]:
    """Shade a globe's visible night side by filling where the sun stands below the terminator altitude.

    A globe cannot take the night *polygon*: half of it is on the far side, where the projection has no
    position at all, and re-closing what is left along the limb by the shorter arc — the rule the
    Natural-Earth fills use — closes the wrong way whenever the visible night spans more than half the limb,
    which is every view of the night side. So the globe asks the opposite question: each point of a display
    grid is taken back to lon/lat through pyramids, and the solar altitude there is read off cleopatra's
    subsolar point. The fill is the region where it is below ``refraction`` — exact at the terminator (the
    contour interpolates the crossing) and empty off the disc, where the inverse projection has no answer.

    The altitude itself is the module docstring's one deliberate deviation from the engine boundary: the
    formula is written out here because cleopatra exposes no way to ask it, which is **cleopatra#379**.

    Args:
        scene: The globe being drawn on.
        when: The instant, in UTC.
        refraction: The solar altitude, in degrees, that defines the terminator.
        samples: The layer's recorded ``n``, which sets the grid's resolution here
            (:func:`_globe_night_grid`) as it sets the terminator's on a flat map.
        style: The caller's style, in ``PolyCollection`` keywords; ``color``/``facecolor`` becomes the fill
            colour, and the rest (``alpha``, ``zorder``, …) reaches ``contourf``. An ``edgecolor`` is
            resolved through ``to_rgba`` and then dropped, since a filled contour has no edge to draw: an
            invalid one raises here, as it does on a flat map, instead of being swallowed with the keyword.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the filled contour set, or ``None`` when
        no visible point is at night.
    """
    _, (xmin, xmax), (ymin, ymax) = scene._frame()
    side = _globe_night_grid(samples)
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, side),
        np.linspace(ymin, ymax, side),
    )
    lon, lat = reproject_coordinates(
        grid_x.ravel().tolist(),
        grid_y.ravel().tolist(),
        from_crs=scene.crs,
        to_crs=4326,
    )
    lon_r = np.radians(np.asarray(lon, dtype=float)).reshape(grid_x.shape)
    lat_r = np.radians(np.asarray(lat, dtype=float)).reshape(grid_x.shape)
    sun_lon, sun_lat = (math.radians(value) for value in subsolar_point(when))
    with np.errstate(
        invalid="ignore"
    ):  # off the disc the inverse is inf, and its sine and cosine NaN
        sin_altitude = np.sin(lat_r) * math.sin(sun_lat) + np.cos(lat_r) * math.cos(
            sun_lat
        ) * np.cos(lon_r - sun_lon)
    # Below zero at night; non-finite off the disc, which contourf leaves unfilled.
    depth = np.where(
        np.isfinite(sin_altitude),
        sin_altitude - math.sin(math.radians(refraction)),
        np.nan,
    )
    if not np.any(depth < 0):
        return None
    opts = {**_NIGHT_STYLE, **style}
    # `in`, not truthiness: an `or` here read *any* falsy colour as "not given" and fell back to the night
    # default, so `color=""` was refused on a flat map and silently drawn black here (round 4, L8).
    # `None` never reaches this — `nightshade` drops it as the "unset" it means in matplotlib — so what is
    # left is either a colour matplotlib accepts or one it refuses, on both frames alike.
    fill = opts.pop("color") if "color" in opts else opts.pop("facecolor")
    opts.pop("facecolor", None)
    edge = opts.pop("edgecolor", None)
    if edge is not None:
        # A filled contour has no edge to draw, so the keyword is dropped — but dropping it unresolved
        # swallowed an invalid colour that a flat map refused, which is the same disagreement in the edge
        # channel. matplotlib still owns the spelling; this only asks it the question before discarding
        # the answer (round 4, L8).
        to_rgba(edge)
    with scene._preserve_view():
        shade = scene.ax.contourf(
            grid_x, grid_y, depth, levels=[-2.0, 0.0], colors=[fill], **opts
        )
    return DrawnLayer(artist=shade, artists=(shade,))


def draw_nightshade(scene: Any, _data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Shade the night side of the terminator at the instant a described layer recorded.

    cleopatra computes the night region (``cleopatra.basemap.solar.night_polygon``) and leaves its projection
    to the consumer. A flat map hands ``add_nightshade`` a ``transform`` that projects through pyramids. A
    globe is filled from the solar altitude instead (see :func:`_globe_nightshade`), because the night side of
    a globe is half of it and the far half has no projected position at all. The recorded ``n`` reaches both:
    it samples the terminator on a flat map, and sets the display grid's resolution on a globe
    (:func:`_globe_night_grid`).

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — the shade is computed from a clock.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``, or ``None`` on a
        globe whose visible hemisphere holds no night — a layer that was not drawn rather than an empty one.
    """
    props = dict(layer.symbology.props)
    when = _utc_moment(props["when"])
    refraction, samples = float(props["refraction"]), int(props["n"])
    style = drawing_style(scene, layer)
    if scene.globe:
        return _globe_nightshade(scene, when, refraction, samples, style)
    unframed = _axes_extent(scene.ax) is None
    transform = None if same_crs(scene.crs, 4326) else _night_ring_projector(scene)
    shade = add_nightshade(
        scene.ax, when, refraction=refraction, n=samples, transform=transform, **style
    )
    if unframed:
        # cleopatra keeps the view it found, and a bare axes' view is matplotlib's unit square.
        scene.ax.autoscale()
        _fit_within_crs_extent(scene)
    return DrawnLayer(artist=shade, artists=(shade,))


def _continuous_ring(ring: np.ndarray, centre_lon: float) -> np.ndarray:
    """Unwrap a lon/lat ring's longitudes around its centre, so a ring across the antimeridian stays whole.

    This is the lon/lat form, and it is drawable **only** on lon/lat axes. Unwrapped longitudes do not
    survive a reprojection: PROJ wraps lon 184.5 back to lon -175.5, so the vertices of a ring across the
    antimeridian land alternately at the two edges of the world. A projected display CRS therefore unwraps
    in display coordinates instead — see :func:`_continuous_display_ring` (round 3, H2).

    Args:
        ring: An ``(m, 2)`` lon/lat ring whose longitudes are wrapped to ``(-180, 180]``.
        centre_lon: The longitude the ring is drawn around.

    Returns:
        A copy whose longitudes run continuously around ``centre_lon`` — past 180 where the ring crosses it.
    """
    out = np.array(ring, dtype=float)
    out[:, 0] = centre_lon + (out[:, 0] - centre_lon + 180.0) % 360.0 - 180.0
    return out


#: How far either side of the seam, in degrees, the two samples that measure its jump sit. About a
#: millimetre of ground: small enough that the width lost between them is below the rounding of every
#: measurement here (Web Mercator still measures ``40075017`` m), and large enough that PROJ places the two
#: samples on opposite sides of the cut rather than at one point.
_SEAM_PROBE_DEGREES = 1.0e-8

#: How far either side of the *central meridian* the projection's local x-scale is measured, in degrees.
#: A whole degree rather than :data:`_SEAM_PROBE_DEGREES`, because the scale is a ratio and measuring it
#: over a millimetre would read mostly floating-point cancellation.
_SEAM_SCALE_DEGREES = 1.0

#: How much of the projection's own x-scale, in degrees of longitude, the jump at the seam must be worth
#: before it counts as a cut. Half a world: a projection that cuts its seam puts the two sides a *whole*
#: world apart, and one that is merely continuous there puts them :data:`_SEAM_PROBE_DEGREES` apart, so
#: anything between the two is a margin of eight orders of magnitude.
_SEAM_MIN_DEGREES = 180.0

#: The prefix of the pyproj coordinate-operation parameter naming a projection's central meridian. The
#: spelling differs by method — "Longitude of natural origin" for Mercator and Robinson, "Longitude of
#: origin" for polar stereographic, "Longitude of projection centre" for oblique Mercator — and they are
#: all the same quantity, so the prefix is matched rather than the dozen names enumerated.
_CENTRAL_MERIDIAN_PREFIX = "Longitude of "


def _wrapped_longitude(lon: float) -> float:
    """Wrap a longitude into ``(-180, 180]``, the range PROJ and a lon/lat axes both place directly.

    Args:
        lon: A longitude in degrees, of any magnitude.

    Returns:
        The same meridian in ``(-180, 180]`` — so lon 180 stays 180 rather than becoming -180, which is the
        half that matters on lon/lat axes, where no projection normalises the value afterwards.

    Examples:
        - The ends of the range are kept apart, and a longitude past either end comes back inside it:
            ```python
            >>> from digitalearth.static.maps.decoration import _wrapped_longitude
            >>> [_wrapped_longitude(value) for value in (0.0, 180.0, -180.0, 270.0, -190.0)]
            [0.0, 180.0, 180.0, -90.0, 170.0]

            ```
    """
    return 180.0 - (180.0 - lon) % 360.0


def _crs_seam_longitude(crs: Any) -> float:
    """Return the longitude a projection cuts: half a turn from the central meridian it declares.

    A CRS centred on Greenwich cuts the antimeridian, which is why assuming lon 180 worked for Web
    Mercator, Robinson and the polar stereographics. A Pacific-centred one does not: EPSG:3832 is centred
    on lon 150 and cuts near lon -30, and probing lon 180 there measures no seam at all (round 4, H3).

    Args:
        crs: The display CRS, in any form :func:`~pyramids.base.crs.crs_from_user_input` takes.

    Returns:
        The seam longitude in ``(-180, 180]``. A CRS that declares no central meridian — a geographic one,
        or anything PROJ cannot read — is treated as centred on Greenwich, which puts the seam at lon 180.

    Examples:
        - Web Mercator cuts the antimeridian; EPSG:3832, the same projection about lon 150, cuts at -30:
            ```python
            >>> from digitalearth.static.maps.decoration import _crs_seam_longitude
            >>> (_crs_seam_longitude(3857), _crs_seam_longitude(3832))
            (180.0, -30.0)

            ```
        - A geographic CRS declares no projection to read a central meridian off, so the seam is lon 180 —
          which is where a lon/lat axes does wrap:
            ```python
            >>> from digitalearth.static.maps.decoration import _crs_seam_longitude
            >>> _crs_seam_longitude(4326)
            180.0

            ```
    """
    try:
        operation = crs_from_user_input(crs).coordinate_operation
    except (CRSError, TypeError, ValueError):
        operation = None
    params = getattr(operation, "params", ()) or ()
    centre = next(
        (
            float(param.value)
            for param in params
            if str(param.name).startswith(_CENTRAL_MERIDIAN_PREFIX)
        ),
        0.0,
    )
    return _wrapped_longitude(centre + 180.0)


def _antimeridian_periods(scene: Any, lats: Sequence[float]) -> np.ndarray:
    """Measure the display-x width of the world at each latitude, or ``0`` where the CRS has no seam.

    A ring is reconnected across the seam by adding that width to the vertices that fell on the far side of
    it, so the width has to be measured in the display CRS rather than assumed. **Where the seam is has to
    be read off the projection too.** It lies half a turn from the central meridian
    (:func:`_crs_seam_longitude`) — at lon 180 only for a CRS centred on Greenwich. Measuring it as
    ``x(180) - x(-180)`` therefore answered ``0`` — "no seam, leave the ring alone" — for every
    Pacific-centred CRS, because those two longitudes are the *same* point there, while the real cut was 90
    degrees away smearing rings 40 times over (``+proj=merc +lon_0=90``: ``4.007e7`` m against the
    ``1.001e6`` m a 500 km ring should span; round 4, H3).

    The jump is measured between two samples a hair either side of that seam, and it counts as a seam only
    when it is worth at least :data:`_SEAM_MIN_DEGREES` of the projection's own x-scale at that latitude —
    which is why the scale is measured too, over :data:`_SEAM_SCALE_DEGREES` either side of the central
    meridian. That replaces the old two-way "a whole world apart, or the same point" split, which had no
    slot for a projection that is merely *continuous* across the probed longitude: EPSG:3031 answers a jump
    of ``0.0043`` m across the probe rather than an exact ``0`` once the samples are no longer the same
    point (executed), which is four millimetres of a ``4.008e7`` m world. A clipped projection has no image
    of the seam at all (non-finite). Both answer ``0``, which means "leave the projected ring alone".

    Args:
        scene: The map whose display CRS is measured.
        lats: The ring-centre latitudes, measured one per ring because a non-cylindrical projection need not
            cut its seam at the same x everywhere.

    Returns:
        A ``(len(lats),)`` array of display-x widths, ``0`` wherever there is no usable seam.

    Examples:
        - Web Mercator cuts at the same x at every latitude, so every ring is carried the same world width;
          Robinson does not, which is why a width is measured per ring:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth import Map
            >>> from digitalearth.static.maps.decoration import _antimeridian_periods
            >>> with Map(crs=3857) as flat:
            ...     [round(float(width)) for width in _antimeridian_periods(flat, [0.0, 60.0])]
            [40075017, 40075017]
            >>> with Map(crs="+proj=robin") as curved:
            ...     [round(float(width)) for width in _antimeridian_periods(curved, [0.0, 60.0])]
            [34011667, 27161718]

            ```
        - EPSG:3832 is the same projection as Web Mercator about lon 150, so its world is exactly as wide —
          measured at **its** cut, near lon -30, where the old probe at lon 180 found nothing:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth import Map
            >>> from digitalearth.static.maps.decoration import _antimeridian_periods
            >>> with Map(crs=3832) as pacific:
            ...     [round(float(width)) for width in _antimeridian_periods(pacific, [0.0, 60.0])]
            [40075017, 40075017]

            ```
        - A polar stereographic is continuous across its seam longitude, so it reports no seam and the
          rings are left where PROJ put them:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth import Map
            >>> from digitalearth.static.maps.decoration import _antimeridian_periods
            >>> with Map(crs=3031) as polar:
            ...     [round(float(width)) for width in _antimeridian_periods(polar, [-70.0, -80.0])]
            [0, 0]

            ```
    """
    if len(lats) == 0:
        return np.zeros(0, dtype=float)
    seam = _crs_seam_longitude(scene.crs)
    centre = _wrapped_longitude(seam + 180.0)
    probe_lons = [
        _wrapped_longitude(seam - _SEAM_PROBE_DEGREES),
        _wrapped_longitude(seam + _SEAM_PROBE_DEGREES),
        _wrapped_longitude(centre - _SEAM_SCALE_DEGREES),
        _wrapped_longitude(centre + _SEAM_SCALE_DEGREES),
    ]
    edges = np.column_stack(
        [
            np.tile(probe_lons, len(lats)),
            np.repeat(np.asarray(lats, dtype=float), len(probe_lons)),
        ]
    )
    x = _lonlat_to_display(scene, edges)[:, 0].reshape(-1, len(probe_lons))
    with np.errstate(invalid="ignore"):  # inf - inf off a clipped projection's image
        period = x[:, 0] - x[:, 1]
        # Half a world, in the projection's own x per degree at this latitude. A continuous projection's
        # jump across the probe is 2e-8 degrees' worth of this; a cut projection's is a whole world.
        half_world = (
            _SEAM_MIN_DEGREES * np.abs(x[:, 3] - x[:, 2]) / (2.0 * _SEAM_SCALE_DEGREES)
        )
    usable = np.isfinite(period) & np.isfinite(half_world) & (period > half_world)
    return np.where(usable, period, 0.0)


def _continuous_display_ring(
    ring: np.ndarray, centre_x: float, period: float
) -> np.ndarray:
    """Unwrap a projected ring's x around its projected centre, so a ring across the seam stays whole.

    The display-coordinate counterpart of :func:`_continuous_ring`, and the one a projected CRS uses. The
    ring is projected with its longitudes still wrapped — which PROJ places exactly — and the vertices that
    came back from the far side of the seam are carried across it by one world width, the way the lon/lat
    form carries them past 180.

    Args:
        ring: An ``(m, 2)`` projected ring.
        centre_x: The ring centre's own projected x, the value the vertices are gathered around.
        period: The display-x width of the world from :func:`_antimeridian_periods`; ``0`` leaves the ring
            as it is.

    Returns:
        The ring, unchanged when there is no seam to cross or the ring is not wholly placeable, else a copy
        whose x runs continuously around ``centre_x``.

    Examples:
        - A ring centred just inside the eastern edge of a Web Mercator map: the vertex PROJ sent back to
          the *western* edge is carried a world width east, so the ring runs continuously past the seam:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.maps.decoration import _continuous_display_ring
            >>> ring = np.array([[1.9e7, 0.0], [-1.9e7, 1.0], [2.0e7, 2.0]])
            >>> whole = _continuous_display_ring(ring, 1.95e7, 4.0075e7)
            >>> [round(float(x)) for x in whole[:, 0]]
            [19000000, 21075000, 20000000]

            ```
        - A CRS with no usable seam reports a period of ``0``, and the ring is handed back as it was:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.maps.decoration import _continuous_display_ring
            >>> ring = np.array([[1.9e7, 0.0], [-1.9e7, 1.0], [2.0e7, 2.0]])
            >>> _continuous_display_ring(ring, 1.95e7, 0.0) is ring
            True

            ```
    """
    if period <= 0.0 or not math.isfinite(centre_x) or not np.isfinite(ring).all():
        return ring
    out = np.array(ring, dtype=float)
    half = period / 2.0
    out[:, 0] = centre_x + (out[:, 0] - centre_x + half) % period - half
    return out


def _projected_rings(
    scene: Any,
    circles: Sequence[np.ndarray],
    lons: Sequence[float],
    lats: Sequence[float],
) -> List[np.ndarray]:
    """Project lon/lat circles into the display CRS, keeping each one whole across the antimeridian.

    Args:
        scene: The map being drawn on, whose display CRS is not lon/lat.
        circles: The lon/lat rings, longitudes wrapped to ``(-180, 180]`` as ``tissot_circles`` returns them.
        lons: The ring-centre longitudes, one per circle.
        lats: The ring-centre latitudes, one per circle.

    Returns:
        One projected ``(m, 2)`` ring per circle.
    """
    if not circles:
        return []
    periods = _antimeridian_periods(scene, lats)
    centres = _lonlat_to_display(
        scene,
        np.column_stack([np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)]),
    )
    return [
        _continuous_display_ring(
            _lonlat_to_display(scene, np.asarray(circle, dtype=float)),
            float(centre_x),
            float(period),
        )
        for circle, centre_x, period in zip(circles, centres[:, 0], periods)
    ]


def draw_tissot(scene: Any, _data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Draw the Tissot indicatrices a described layer recorded, through the display projection.

    cleopatra builds the geodesic circles (``tissot_circles``) and draws rings it is handed (``add_tissot``);
    what a circle becomes on the map is the projection's doing, so the rings are projected here through
    pyramids. A ring the projection cannot place whole — one on the far side of a globe — is left out rather
    than drawn as a fragment, since half an indicatrix reads as a different distortion.

    A ring across the antimeridian is reconnected in whichever coordinates it is drawn in: in degrees on
    lon/lat axes (:func:`_continuous_ring`), and in display coordinates once a projection is involved
    (:func:`_projected_rings`), because unwrapped longitudes do not survive PROJ.

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — the circles are computed from the centres.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``, or ``None`` when
        no ring could be placed — a layer that was not drawn rather than an empty one.
    """
    props = dict(layer.symbology.props)
    lons, lats = list(props["lons"]), list(props["lats"])
    circles = tissot_circles(lons, lats, float(props["radius_m"]), n=int(props["n"]))
    if same_crs(scene.crs, 4326):
        rings = [
            _continuous_ring(circle, float(lon)) for circle, lon in zip(circles, lons)
        ]
    else:
        rings = _projected_rings(scene, circles, lons, lats)
    rings = [ring for ring in rings if np.isfinite(ring).all()]
    if not rings:
        return None
    unframed = _axes_extent(scene.ax) is None
    indicatrix = add_tissot(scene.ax, rings, **drawing_style(scene, layer))
    if unframed:
        scene.ax.autoscale()
    return DrawnLayer(artist=indicatrix, artists=(indicatrix,))


class DecorationMixin(_MixinBase):
    """Annotation and basemap/Natural-Earth decoration for :class:`~digitalearth.static.map.Map`.

    A capability mixin of :class:`~digitalearth.static.map.Map`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the shared figure/axes, the layer registry and the
    display CRS — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.static.maps.base.GeoLayerBase`, the state ``Map``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``Map`` MRO exactly what
    it was before the annotation.

    See Also:
        digitalearth.static.map.Map: the composition that supplies the state these methods use.
        digitalearth.static.maps.base.GeoLayerBase: the typing-only base declared above the class.
    """

    def _reproject_point(
        self, lon: float, lat: float, crs: Any
    ) -> Optional[Tuple[float, float]]:
        """Reproject one ``(lon, lat)`` in ``crs`` to the display CRS; ``None`` if it lands off the globe.

        A point on the far side of a clipped/globe display CRS reprojects to non-finite coordinates, which
        :meth:`text` / :meth:`annotate` skip rather than drawing garbage.

        Args:
            lon: Longitude (x) in ``crs``.
            lat: Latitude (y) in ``crs``.
            crs: CRS of ``lon``/``lat``.

        Returns:
            The ``(x, y)`` in the display CRS, or ``None`` when the reprojected point is non-finite.
        """
        x, y = reproject_coordinates([lon], [lat], from_crs=crs, to_crs=self.crs)
        if not (np.isfinite(x[0]) and np.isfinite(y[0])):
            return None
        return x[0], y[0]

    def text(
        self,
        lon: float,
        lat: float,
        s: str,
        *,
        crs: Any = 4326,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Place a text label at a ``lon``/``lat`` location (reprojected to the display CRS).

        The point is reprojected from ``crs`` (lon/lat by default) into the display CRS via pyramids, then
        drawn with ``Axes.text``. On a globe, a point on the **far side** reprojects to non-finite coordinates
        and is skipped (no artist, returns ``None``).

        Args:
            lon: Longitude (x) of the label, in ``crs``.
            lat: Latitude (y) of the label, in ``crs``.
            s: The text to draw.
            crs: CRS of ``lon``/``lat`` (default ``4326`` = WGS84 lon/lat).
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321). ``Axes.text`` reads ``name=`` as an alias of the font family,
                and that reading is kept: a name that is the name of a family this machine has is drawn
                in as well, unless the call chooses a font itself (review R-M2). That is applied when the
                label is drawn and is not part of what the figure records, so a figure describes the name
                and not the machine's font list (review R2-M3). Say ``fontname=`` to pick the font
                independently of what the layer is called.
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **kwargs: Forwarded to ``Axes.text`` (e.g. ``ha``, ``va``, ``fontsize``, ``color``). A font
                family given here outranks the layer's name, under every spelling matplotlib takes for
                one — ``font``, ``font_properties``, ``fontproperties``, ``fontfamily``, ``family`` and
                ``fontname``.

        Returns:
            The :class:`matplotlib.text.Text`, or ``None`` if the point is off the visible globe.
        """
        return self._draw(
            LayerRecord(
                "text",
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "text",
                        "lon": float(lon),
                        "lat": float(lat),
                        "s": s,
                        # In the shared CRS spelling: a caller may hand in a CRS object, which a figure
                        # written to JSON has no form for, and this is how every other spec field holds one.
                        "crs": crs_to_json(crs, "Map.text(crs=)"),
                    }
                ),
                opts=kwargs,
            )
        )

    def annotate(
        self,
        lon: float,
        lat: float,
        s: str,
        *,
        xytext: Any = None,
        crs: Any = 4326,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Annotate a ``lon``/``lat`` location (reprojected), optionally with an arrow.

        Like :meth:`text` but via ``Axes.annotate``: the annotated point ``xy`` is the reprojected
        ``lon``/``lat``; pass ``xytext`` (with ``arrowprops``) to draw an arrow from the label to the point.
        A far-side point on a globe is skipped (returns ``None``).

        Args:
            lon: Longitude (x) of the annotated point, in ``crs``.
            lat: Latitude (y) of the annotated point, in ``crs``.
            s: The annotation text.
            xytext: Optional text position (in the coordinate system given by ``textcoords``/``kwargs``); with
                ``arrowprops`` an arrow is drawn from there to the point.
            crs: CRS of ``lon``/``lat`` (default ``4326``).
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321). ``Axes.annotate`` reads ``name=`` as an alias of the font
                family — an ``Annotation`` is a ``Text`` — and that reading is kept: a name that is the
                name of a family this machine has is drawn in as well, unless the call chooses a font
                itself (review R-M2). That is applied when the annotation is drawn and is not part of
                what the figure records (review R2-M3). Say ``fontname=`` to pick the font independently
                of the layer's name.
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **kwargs: Forwarded to ``Axes.annotate`` (e.g. ``arrowprops``, ``textcoords``, ``fontsize``).
                A font family given here outranks the layer's name, under every spelling matplotlib takes
                for one — ``font``, ``font_properties``, ``fontproperties``, ``fontfamily``, ``family``
                and ``fontname``.

        Returns:
            The :class:`matplotlib.text.Annotation`, or ``None`` if the point is off the visible globe.
        """
        return self._draw(
            LayerRecord(
                "text",
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "annotate",
                        "lon": float(lon),
                        "lat": float(lat),
                        "s": s,
                        "xytext": xytext,
                        "crs": crs_to_json(crs, "Map.annotate(crs=)"),
                    }
                ),
                opts=kwargs,
            )
        )

    def stock_img(
        self,
        dataset: Any = None,
        *,
        zorder: float = -3.0,
        cmap: Optional[str] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Draw a background raster (a "stock image" backdrop) beneath all data layers.

        Pass a pyramids ``Dataset`` (e.g. a low-res relief/imagery raster) to draw as the backdrop — it is
        reprojected to the display CRS and drawn at a low ``zorder`` so data layers sit on top, and the current
        view is preserved so a global backdrop never autoscales a regional view out. With ``dataset=None`` a
        best-effort XYZ-tile basemap is used instead (network; skipped offline).

        Note: the no-argument form uses a tile basemap. cleopatra now ships a hypsometric relief backdrop
        (``cleopatra.basemap.reference.add_relief``); wiring it in as the offline no-argument default is tracked
        separately. For a raster backdrop today, supply your own ``Dataset``.

        Args:
            dataset: A pyramids ``Dataset`` to use as the backdrop, or a path or URL to one, or
                ``None`` to try a tile basemap. Only a path-backed layer can be written down.
            zorder: Draw order for the backdrop (default ``-3.0``, below data/coastlines).
            cmap: Colormap for a raster backdrop (ignored for the tile path). ``None`` (default) resolves
                one from the backdrop's own variable via
                :func:`~digitalearth.base.autostyle.auto_style` — so a DEM backdrop is coloured as terrain
                — falling back to :data:`STOCK_IMG_CMAP` when the lookup has no opinion.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`field` (raster) or :meth:`basemap` (tiles).

        Returns:
            The backdrop ``AxesImage`` (raster path), the tile artist, or ``None`` if a tile backdrop is
            unavailable offline or the raster lies outside what the display CRS shows.
        """
        if dataset is None:
            try:
                return self.basemap(name=name, visible=visible, **kwargs)
            except Exception as exc:  # tile servers unavailable — best-effort backdrop
                logger.debug("stock_img tile basemap unavailable: %s", exc)
                return None
        with self._preserve_view():
            # `draw_band="underlay"` is the description's half of `zorder`: one says where the
            # backdrop sits among the figure's layers, the other where it sits on the axes. Both are
            # recorded, so a backdrop drawn again from its description comes back behind the data.
            im = self.field(
                dataset,
                cmap=cmap,
                name=name,
                visible=visible,
                default_cmap=STOCK_IMG_CMAP,
                draw_band="underlay",
                zorder=zorder,
                **kwargs,
            )
        return im

    def _project_line_features(self, parts: List[np.ndarray]) -> List[np.ndarray]:
        """Project lon/lat line parts to the display CRS, split at the projection limb.

        Reprojecting a global line to a clipped projection (e.g. orthographic) sends the far side to
        non-finite coordinates; per-part splitting at those gaps keeps the visible arcs and avoids the
        ``NaN/Inf`` matplotlib would otherwise draw.

        Args:
            parts: ``(N, 2)`` lon/lat arrays (EPSG:4326) as returned by
                ``cleopatra.basemap.reference.natural_earth`` for a line layer.

        Returns:
            A list of finite ``(M, 2)`` projected polyline segments.
        """
        segments: List[np.ndarray] = []
        for part in parts:
            xy = np.asarray(part, dtype=float)
            if xy.size == 0:
                continue
            x, y = reproject_coordinates(
                xy[:, 0].tolist(), xy[:, 1].tolist(), from_crs=4326, to_crs=self.crs
            )
            segments += projections._split_finite(
                np.asarray(x, float), np.asarray(y, float)
            )
        return segments

    def _project_polygon_features(self, parts: List[np.ndarray]) -> List[np.ndarray]:
        """Project lon/lat polygon rings to the display CRS as finite, limb-clipped fill rings.

        The fill analogue of :meth:`_project_line_features`: each exterior ring is densified, reprojected, and
        re-closed against the projection boundary (``self._frame()[0]``) so a ring straddling the limb becomes
        one or more closed, fully-finite rings instead of injecting ``inf``/``nan`` into the fill.
        ``natural_earth`` returns exterior rings only, so interior rings (holes) are not represented on a
        globe — see Digital-Earth#43.

        Args:
            parts: ``(N, 2)`` lon/lat exterior-ring arrays (EPSG:4326) as returned by
                ``cleopatra.basemap.reference.natural_earth`` for a polygon layer.

        Returns:
            A list of closed ``(M, 2)`` projected fill rings (empty when nothing is on the near side).
        """
        boundary = self._frame()[0]
        rings: List[np.ndarray] = []
        for part in parts:
            xy = np.asarray(part, dtype=float)
            if xy.size == 0:
                continue
            xy = projections.densify_lonlat(xy, step_deg=1.0)
            x, y = reproject_coordinates(
                xy[:, 0].tolist(), xy[:, 1].tolist(), from_crs=4326, to_crs=self.crs
            )
            rings += projections.close_visible_runs(
                np.asarray(x, float), np.asarray(y, float), boundary
            )
        return rings

    def _fill_globe_polygons(
        self,
        rings: List[np.ndarray],
        *,
        zorder: float,
        **style: Any,
    ) -> Optional[DrawnLayer]:
        """Fill projected rings with a solid colour on a globe (map-specific overlay; clipped at frame time).

        cleopatra ``PolygonGlyph`` only fills when given per-polygon *values*, so a uniform land/ocean fill is
        drawn as a plain ``PolyCollection`` directly on the axes — exactly like the globe coastline ``ax.plot``
        overlay. ``apply_projection_frame(clip_artists=True)`` clips it to the boundary at render time. The
        view limits are preserved so a global fill never autoscales the axes out.

        Args:
            rings: Closed, finite projected fill rings (from :meth:`_project_polygon_features`).
            zorder: Draw order (ocean below land below data below coastlines).
            **style: What the layer recorded, in the plural keys a collection takes (see
                :func:`_to_feature_style`) — ``facecolors``, and whatever else the caller asked for:
                ``edgecolors``, ``alpha``, ``linewidths``. A globe fill recorded ``alpha`` and
                ``edgecolor`` and then drew neither, because only the fill colour reached here.

        Returns:
            What the fill produced, as a :class:`~digitalearth.static.renderer.DrawnLayer` holding the
            ``PolyCollection``, or ``None`` when ``rings`` is empty — nothing on the near side is a layer
            that was not drawn.
        """
        if not rings:
            return None
        with self._preserve_view():
            # The edge is off unless the layer asks for one: a fill is a fill, and the Natural-Earth
            # defaults spell that out for the layers that have an edge to speak of.
            pc = PolyCollection(rings, zorder=zorder, **{"edgecolors": "none", **style})
            self.ax.add_collection(pc)
        self._register_artist(None, pc)
        return DrawnLayer(artist=pc, artists=(pc,))

    def _natural_earth(
        self,
        layer: str,
        resolution: str,
        *,
        polygon: bool = False,
        zorder: float = 0.5,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Draw a Natural-Earth vector layer reprojected to the display CRS, clipped to the current view.

        On a **globe** map, line layers (coastline/borders/rivers) are projected per-line and split at the
        projection limb (the far side reprojects to non-finite coords) and drawn as plain polylines; polygon
        layers (``polygon=True``: land/lakes) are projected and re-closed at the limb into finite rings and
        filled via :meth:`_fill_globe_polygons`. Both are clipped to the boundary when the frame is applied.
        On a **flat** map, the layer is drawn (and reprojected to the display CRS) by
        ``cleopatra.basemap.reference.add_features`` — hole-aware for polygon layers. ``add_features`` holds the
        current axis limits, so a global layer never autoscales an already-drawn data view out; on an
        otherwise-empty axes the view is autoscaled to the new layer instead.

        Args:
            layer: Natural-Earth layer name (e.g. ``"coastline"``, ``"land"``).
            resolution: Natural-Earth resolution (``"110m"``/``"50m"``/``"10m"``).
            polygon: When True, treat the layer as filled polygons on a globe (else as lines).
            zorder: Draw order (globe polygon fills; also forwarded to ``add_features`` on a flat map).
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **kwargs: Style overrides, laid over the layer's defaults (:data:`_NATURAL_EARTH_STYLE`) when it
                is drawn. Held beside the layer exactly as passed, and the plain ones described as well, so
                a figure read back elsewhere still draws them (see
                :func:`~digitalearth.static.scene.described_opts`).

        Returns:
            Whatever the path that drew it returns — the ``PolyCollection`` of a globe fill, the list of
            polyline artists of a globe line layer, or the axes on a flat map; and ``None`` when nothing
            was on the near side to draw. All three describe one layer of the reference geography's own
            kind, whichever matplotlib objects they happened to leave behind.
        """
        return self._draw(
            LayerRecord(
                _NATURAL_EARTH_KINDS[layer],
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": layer,
                        "resolution": resolution,
                        # How the drawer must draw it, recorded rather than passed: a globe fills polygons
                        # and splits lines at the limb, a flat map hands both to `add_features`.
                        "polygon": polygon,
                        "zorder": zorder,
                    }
                ),
                opts=kwargs,
            )
        )

    def coastlines(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Overlay Natural-Earth coastlines (``cleopatra.basemap.reference`` ``"coastline"`` layer).

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Style overrides laid over this layer's Natural-Earth defaults
                (:data:`_NATURAL_EARTH_STYLE`). A plain one — ``color``, ``linewidth``, ``alpha`` — is
                written into the layer's description as well, so a figure drawn on another scene keeps it;
                a dash tuple or an engine object is held beside the layer alone, and that scene falls back
                to the Natural-Earth default for it.

        Returns:
            The drawn coastline artist (a list of polyline artists on a globe; the reprojected plot artist
            on a flat map).
        """
        return self._natural_earth(
            "coastline", resolution, zorder=2.5, name=name, visible=visible, **kwargs
        )

    def borders(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Overlay Natural-Earth country borders.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Style overrides laid over this layer's Natural-Earth defaults
                (:data:`_NATURAL_EARTH_STYLE`). A plain one — ``color``, ``linewidth``, ``alpha`` — is
                written into the layer's description as well, so a figure drawn on another scene keeps it;
                a dash tuple or an engine object is held beside the layer alone, and that scene falls back
                to the Natural-Earth default for it.

        Returns:
            The drawn border artist (a list of polyline artists on a globe; the reprojected plot artist on a
            flat map).
        """
        return self._natural_earth(
            "borders", resolution, zorder=2.5, name=name, visible=visible, **kwargs
        )

    def land(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Fill Natural-Earth land polygons.

        On a **flat** map the polygons are reprojected and filled directly. On a **globe** map they are
        re-closed at the projection limb into finite rings and filled as a map-specific overlay (drawn below
        data and coastlines, clipped to the boundary). Interior rings (holes) are dropped in v1 — see #43.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Style overrides laid over this layer's Natural-Earth defaults
                (:data:`_NATURAL_EARTH_STYLE`). A plain one — ``color``, ``linewidth``, ``alpha`` — is
                written into the layer's description as well, so a figure drawn on another scene keeps it;
                a dash tuple or an engine object is held beside the layer alone, and that scene falls back
                to the Natural-Earth default for it.

        Returns:
            The land fill layer (a ``PolyCollection`` on a globe, ``None`` when nothing is on the near side;
            the reprojected plot artist on a flat map).
        """
        return self._natural_earth(
            "land",
            resolution,
            polygon=True,
            zorder=-1.5,
            name=name,
            visible=visible,
            **kwargs,
        )

    def ocean(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Fill Natural-Earth ocean polygons.

        On a **globe** map, ``ocean`` fills the whole projection disc (the boundary ring) with the ocean
        colour and lets land overlay it — exact and far cheaper than clipping the global ocean polygon. On a
        **flat** map, the Natural-Earth ocean polygons are reprojected and filled directly.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Style overrides laid over this layer's Natural-Earth defaults
                (:data:`_NATURAL_EARTH_STYLE`). A plain one — ``color``, ``linewidth``, ``alpha`` — is
                written into the layer's description as well, so a figure drawn on another scene keeps it;
                a dash tuple or an engine object is held beside the layer alone, and that scene falls back
                to the Natural-Earth default for it.

        Returns:
            The ocean fill layer (a ``PolyCollection`` disc on a globe; the reprojected plot artist on a flat
            map).
        """
        if self.globe:
            # The disc is the ocean: filling the whole projection boundary and letting land overlay it is
            # exact and far cheaper than clipping the global ocean polygon, and it is still `ocean`. The
            # drawer reads the globe flag off the scene, so what differs here is only the draw order this
            # layer is recorded with.
            return self._natural_earth(
                "ocean",
                resolution,
                polygon=True,
                zorder=-2.0,
                name=name,
                visible=visible,
                **kwargs,
            )
        return self._natural_earth(
            "ocean", resolution, name=name, visible=visible, **kwargs
        )

    def lakes(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Fill Natural-Earth lake polygons.

        Like :meth:`land`, but with a water colour and drawn just above land (so lakes sit on the land) and
        still below data and coastlines. On a globe the polygons are re-closed at the projection limb.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Style overrides laid over this layer's Natural-Earth defaults
                (:data:`_NATURAL_EARTH_STYLE`). A plain one — ``color``, ``linewidth``, ``alpha`` — is
                written into the layer's description as well, so a figure drawn on another scene keeps it;
                a dash tuple or an engine object is held beside the layer alone, and that scene falls back
                to the Natural-Earth default for it.

        Returns:
            The lake fill layer (a ``PolyCollection`` on a globe, ``None`` when nothing is on the near side;
            the reprojected plot artist on a flat map).
        """
        return self._natural_earth(
            "lakes",
            resolution,
            polygon=True,
            zorder=-1.4,
            name=name,
            visible=visible,
            **kwargs,
        )

    def rivers(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Overlay Natural-Earth rivers (line centerlines), split at the projection limb on a globe.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Style overrides laid over this layer's Natural-Earth defaults
                (:data:`_NATURAL_EARTH_STYLE`). A plain one — ``color``, ``linewidth``, ``alpha`` — is
                written into the layer's description as well, so a figure drawn on another scene keeps it;
                a dash tuple or an engine object is held beside the layer alone, and that scene falls back
                to the Natural-Earth default for it.

        Returns:
            The drawn river artist (a list of polyline artists on a globe; the reprojected plot artist on a
            flat map).
        """
        return self._natural_earth(
            "rivers", resolution, zorder=2.4, name=name, visible=visible, **kwargs
        )

    def nightshade(
        self,
        when: Any,
        *,
        refraction: float = DEFAULT_REFRACTION,
        n: int = DEFAULT_TERMINATOR_SAMPLES,
        name: Optional[str] = None,
        visible: bool = True,
        **style: Any,
    ) -> Any:
        """Shade the night side of the day/night terminator at one instant.

        The terminator is computed by cleopatra (``cleopatra.basemap.solar``, a low-precision solar position
        good to well under a degree) and projected into the display CRS through pyramids. A night region that
        holds a pole is shaded to the pole on a flat map, and a globe shades only its visible hemisphere. On
        an axes nothing has framed yet, the map is fitted to the shade and then pulled back inside the
        display CRS's own extent, so a projection that places the pole far outside its map (EPSG:3031 puts
        it ``1.4e10`` m out) still frames the region it covers; a framed view is kept.

        Args:
            when: The instant, as a ``datetime`` or ISO 8601 text. One with no time zone is UTC. The figure
                records it as UTC ISO text, so a figure read back draws the same shade.
            refraction: The solar altitude, in degrees, that defines the terminator — ``-0.83`` (default)
                for sunrise and sunset, ``-6``/``-12``/``-18`` for the civil, nautical and astronomical
                twilight lines. Must lie in ``(-90, 0]``.
            n: Samples along the terminator. On a globe there is no terminator polygon to sample — the
                shade is filled from a display grid — so ``n`` sets that grid's resolution in proportion
                instead: the default draws a 400-sample side, and a quarter of the default draws a
                100-sample side, measured at about 0.01 s against about 0.16 s for the shade itself. Both
                are small beside the globe's own projection frame, which costs about 1.7 s and is computed
                once per globe whatever ``n`` is, so it dominates a first render either way. Either way
                ``n`` is the one knob for the shade's fidelity, and the figure records it — where before it
                was recorded and then ignored on a globe.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **style: Forwarded to the ``PolyCollection`` (``facecolor``/``color``, ``alpha``, ``zorder``,
                …). The default is black at ``alpha=0.35``. ``facecolor`` and ``color`` are the same
                keyword here, and ``None`` in either means "not given", so the default is drawn — it is
                **not** passed through to matplotlib's cycle colour, which drew a night shade in blue
                (round 4, L8). ``alpha=0`` is honoured as the invisible shade it asks for, not read as
                unset. A **globe** fills a ``contourf`` rather than a polygon, so it draws no outline:
                ``edgecolor`` (and ``linewidth``/``linestyle`` with it) is resolved — an invalid colour is
                refused on either frame — and then dropped there, where a flat map draws it.

        Returns:
            The night shade's ``PolyCollection`` — on a globe, the filled ``ContourSet`` it is drawn as there
            — or ``None`` on a globe whose visible side holds no night.

        Raises:
            ValueError: when ``refraction`` is outside ``(-90, 0]``, when ``n`` is not a whole number of at
                least 4 — the fewest that describe a ring — or when ``when`` is text that is not ISO 8601.
                The layer is not described. Both the ``refraction`` and the ``n`` refusal are made here, in
                the builder, so they answer the same way on a flat map and on a globe: cleopatra's own live
                in ``add_nightshade`` and the terminator sampler, which only the flat path calls, and a
                globe used to take either value silently (round 3, L7 for ``n``; round 4, M2 for
                ``refraction``).
            TypeError: when ``when`` is neither a ``datetime`` nor text.

        Examples:
            - At the March equinox at noon UTC the antimeridian is at midnight and Greenwich at noon:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from datetime import datetime, timezone
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> night = m.nightshade(datetime(2026, 3, 20, 12, tzinfo=timezone.utc))
                >>> [any(p.contains_point(xy) for p in night.get_paths()) for xy in ((170, 0), (0, 0))]
                [True, False]
                >>> m.figure_spec.layers.get(m.layer_ids[-1]).kind
                'nightshade'
                >>> m.close()

                ```
            - The twilight lines are the same call with another refraction; the figure records it:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=3857)
                >>> _ = m.nightshade("2026-06-21T12:00:00", refraction=-6.0, alpha=0.2)
                >>> props = m.figure_spec.layers.get(m.layer_ids[-1]).symbology.props
                >>> props["when"], props["refraction"]
                ('2026-06-21T12:00:00+00:00', -6.0)
                >>> m.close()

                ```
            - ``color=None`` is the "not given" matplotlib reads it as, so the default black is drawn
              rather than the first cycle colour; ``alpha=0`` is the invisible shade it asks for:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> with Map(crs=4326) as m:
                ...     shade = m.nightshade("2026-12-21T12:00:00+00:00", color=None)
                ...     [tuple(round(float(c), 2) for c in face) for face in shade.get_facecolor()]
                ...     faint = m.nightshade("2026-12-21T12:00:00+00:00", alpha=0)
                ...     [tuple(round(float(c), 2) for c in face) for face in faint.get_facecolor()]
                [(0.0, 0.0, 0.0, 0.35)]
                [(0.0, 0.0, 0.0, 0.0)]

                ```
            - A refraction above the horizon is refused by name on a globe as on a flat map, so one
              keyword answers one way whatever frame it is drawn on:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> for globe in (False, True):
                ...     with Map(crs=4326, globe=globe) as m:
                ...         try:
                ...             m.nightshade("2026-12-21T12:00:00+00:00", refraction=10.0)
                ...         except ValueError as error:
                ...             print(str(error).split(" degrees")[0])
                nightshade() needs refraction= in (-90, 0]
                nightshade() needs refraction= in (-90, 0]

                ```

        See Also:
            tissot: the other overlay cleopatra's ``solar`` module draws.
        """
        samples = _whole_samples(n, "nightshade()")
        altitude = _terminator_altitude(refraction, "nightshade()")
        moment = _utc_moment(when)
        fill = {
            key: value
            for key, value in style.items()
            if value is not None or key not in _NIGHT_FILL_KEYWORDS
        }
        return self._draw(
            LayerRecord(
                "nightshade",
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "nightshade",
                        "when": moment.isoformat(),
                        "refraction": altitude,
                        "n": samples,
                    }
                ),
                opts=fill,
            )
        )

    def tissot(
        self,
        lons: Optional[Sequence[float]] = None,
        lats: Optional[Sequence[float]] = None,
        *,
        radius_m: float = DEFAULT_TISSOT_RADIUS_M,
        n: int = DEFAULT_TISSOT_SAMPLES,
        name: Optional[str] = None,
        visible: bool = True,
        **style: Any,
    ) -> Any:
        """Draw Tissot's indicatrices: circles of one ground radius, shown as the projection distorts them.

        Each circle is the set of points ``radius_m`` from its centre on the sphere (cleopatra's
        ``tissot_circles``), projected into the display CRS through pyramids — so a Web Mercator map shows
        them swelling toward the poles, and an equal-area one shows them keeping their area. A circle the
        projection cannot place whole, such as one on the far side of a globe, is left out. One straddling
        the antimeridian is *not*: it is reconnected across the seam — in degrees on lon/lat axes, in display
        coordinates once a projection is involved — so it reads as one ring running past the edge of the
        world. Unreconnected it spanned the whole map instead, its vertices landing at both edges at once
        (29 of 64 at one edge and the rest at the other, for a 500 km ring at lon 179.5 on Web Mercator).
        On an axes nothing has framed yet, the map is fitted to the circles; a framed view is kept.

        Args:
            lons: Circle-centre longitudes, in degrees. ``None`` (default, with ``lats`` also ``None``) draws
                a world grid every 30 degrees of longitude and latitude, from 60 S to 60 N.
            lats: Circle-centre latitudes, in degrees, pairing with ``lons``.
            radius_m: The ground radius of every circle, in metres (default 500 km).
            n: Vertices per circle.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **style: Forwarded to the ``PolyCollection`` (``edgecolor``, ``facecolor``, ``linewidth``, …).
                The default is an unfilled black outline.

        Returns:
            The circles' ``PolyCollection``, or ``None`` when no circle could be placed.

        Raises:
            ValueError: when only one of ``lons``/``lats`` is given, when **either** is empty, when they
                differ in length, or when ``n`` is not a whole number of at least 4. All four are refused
                here, in the builder, by name and before any ring is computed, and the layer is not
                described. ``radius_m`` raises the same type but not from here: cleopatra refuses a
                distance that is not positive and sub-antipodal from inside the drawer, because the bound
                is the mean Earth radius' to set and is not restated in this tier. The layer is not
                described in that case either — a drawer that raises takes its own description with it.

        Examples:
            - On Web Mercator a circle at 60 degrees is drawn twice as wide as one at the equator:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth import Map
                >>> m = Map(crs=3857)
                >>> rings = m.tissot([0.0, 0.0], [0.0, 60.0], radius_m=100_000.0)
                >>> equator, sixty = (np.ptp(p.vertices[:, 0]) for p in rings.get_paths())
                >>> round(float(sixty / equator), 1)
                2.0
                >>> m.close()

                ```
            - With no centres, a world grid is drawn and described as one layer:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> len(m.tissot(edgecolor="crimson").get_paths())
                60
                >>> m.figure_spec.layers.get(m.layer_ids[-1]).kind
                'tissot'
                >>> m.close()

                ```
            - A ring at lon 179.5 stays one 500 km circle on Web Mercator: it runs on past the world's own
              eastern edge (``20037508`` m) instead of being torn in two across it:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> with Map(crs=3857) as m:
                ...     xs = m.tissot(lons=[179.5], lats=[0.0], radius_m=500_000.0).get_paths()[0]
                ...     span = xs.vertices[:, 0]
                ...     (round(float(span.min())), round(float(span.max())))
                ...     bool(span.max() > 20037508.0)
                (19481444, 20482253)
                True

                ```
            - An empty centre list is refused by name, because ``None`` already means "every ring was off
              the map" and the two must not read alike. **Either** list empty answers the same way: a
              half-empty pair is what a caller gets when one of two filters came out empty, which is the
              case the refusal is for:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> with Map(crs=4326) as m:
                ...     for pair in (([], []), ([], [0.0]), ([0.0], [])):
                ...         try:
                ...             m.tissot(lons=pair[0], lats=pair[1])
                ...         except ValueError as error:
                ...             print(error)
                tissot() was given no centres to draw; pass lons= and lats=, or neither for the world grid
                tissot() was given no centres to draw; pass lons= and lats=, or neither for the world grid
                tissot() was given no centres to draw; pass lons= and lats=, or neither for the world grid

                ```
            - Two centre lists of different lengths are refused by this method too, with both counts
              named, rather than by cleopatra from inside the drawer:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> with Map(crs=4326) as m:
                ...     try:
                ...         m.tissot(lons=[0.0, 10.0], lats=[0.0])
                ...     except ValueError as error:
                ...         print(error)
                tissot() takes one lat per lon; got 2 in lons= and 1 in lats=

                ```
            - ``n`` is refused here and ``radius_m`` by cleopatra from inside the drawer, so the two
              messages read differently — but neither leaves a layer behind:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> with Map(crs=4326) as m:
                ...     for kwargs in ({"n": 3}, {"radius_m": -1.0}):
                ...         try:
                ...             m.tissot(lons=[0.0], lats=[0.0], **kwargs)
                ...         except ValueError as error:
                ...             print(error)
                ...     m.layer_ids
                tissot() needs n= as a whole number >= 4, the fewest that describe a ring; got 3
                radius_m must be a positive, sub-antipodal ground radius in metres (0, 20015114); got -1.0.
                []

                ```

        See Also:
            nightshade: the other overlay cleopatra's ``solar`` module draws.
        """
        samples = _whole_samples(n, "tissot()")
        if (lons is None) != (lats is None):
            raise ValueError(
                "tissot() takes lons= and lats= together, or neither for the world grid"
            )
        if lons is None or lats is None:
            grid_lon, grid_lat = np.meshgrid(_TISSOT_LONS, _TISSOT_LATS)
            lon_list = [float(v) for v in grid_lon.ravel()]
            lat_list = [float(v) for v in grid_lat.ravel()]
        else:
            lon_list = [float(v) for v in np.atleast_1d(np.asarray(lons, dtype=float))]
            lat_list = [float(v) for v in np.atleast_1d(np.asarray(lats, dtype=float))]
            if not lon_list or not lat_list:
                # `None` is this method's answer for "every ring was off-limb", so an empty centre list
                # must not share it: a caller whose list came out empty would read it as the projection's
                # doing. `facet` refuses an empty stack by name for the same reason (round 3, L5). Gated
                # on **either** list, not both: the half-empty pair is the shape a caller gets when one of
                # two filters came out empty, which is the case this refusal was added for, and it used to
                # fall through to cleopatra's shape message instead (round 4, M6).
                raise ValueError(
                    "tissot() was given no centres to draw; pass lons= and lats=, or neither for the "
                    "world grid"
                )
            if len(lon_list) != len(lat_list):
                # Refused here rather than left to cleopatra's `tissot_circles`, because `Raises` presents
                # it as this method's refusal — so it names this method and the two counts (round 4, M6).
                raise ValueError(
                    f"tissot() takes one lat per lon; got {len(lon_list)} in lons= and "
                    f"{len(lat_list)} in lats="
                )
        return self._draw(
            LayerRecord(
                "tissot",
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "tissot",
                        "lons": lon_list,
                        "lats": lat_list,
                        "radius_m": float(radius_m),
                        "n": samples,
                    }
                ),
                opts=style,
            )
        )

    def basemap(
        self,
        source: Any = None,
        *,
        api_key: Optional[str] = None,
        preset: Optional[dict] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Add an XYZ-tile basemap to the axes via ``cleopatra.basemap.tiles.add_tiles`` in the display CRS.

        ``source`` is passed through to cleopatra unchanged — a provider name or an
        ``xyzservices.TileProvider`` — with two additions. The name of a **keyed** basemap preset (see
        :mod:`digitalearth.base.basemaps`) is resolved here into a configured provider, with its credential
        read from the environment and its coverage checked against the map's domain first. And the four
        cross-tier basemap names (:data:`~digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER` among them)
        are translated into the ``xyzservices`` paths that request the same tiles the interactive and web
        tiers do — including when ``source`` is omitted, which means the shared default here just as it does
        there, rather than cleopatra's own ``OpenStreetMap.Mapnik`` (#247).

        Args:
            source: A cleopatra/``xyzservices`` provider name, an ``xyzservices.TileProvider``, a
                ``cleopatra.basemap.ogc`` ``WMSProvider``/``WMTSProvider``, one of the cross-tier names
                (``"CartoLight"``/``"CartoDark"``/``"CartoVoyager"``/``"OSM"``), a keyed preset name such as
                ``"Planet.NICFI"``, or ``None`` for
                :data:`~digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER`. An OGC provider is recorded in
                the figure field by field, minus its ``extra_params``, so a figure read back through
                ``Map.from_figure`` or ``to_backend(spec, "matplotlib")`` asks the same service —
                unauthenticated, since the credential does not travel. Another *tier* cannot replay a
                static basemap layer at all, OGC or named; see ``docs/reference/basemaps.md``.
            api_key: Credential for a keyed preset; ``None`` reads the preset's environment variable.
            preset: The keyed preset's own keywords, as a dict (for NICFI: ``date``, ``flavour``,
                ``mosaic``). A dict rather than loose keywords because ``**kwargs`` here belongs to
                ``add_tiles``, and the three tiers take presets the same way.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **kwargs: Forwarded to ``add_tiles``.

        Returns:
            The tile artist ``add_tiles`` added to the axes.

        Raises:
            ValueError: when a keyed preset is unknown, its credential is unavailable, when a ``preset``
                or an ``api_key`` is passed to an ordinary source, when a preset keyword is passed loose
                instead of in ``preset``, or when the extent being drawn — the declared ``domain`` if there
                is one, else the current axes limits — lies entirely outside the coverage.
            TypeError: when a preset keyword is missing or misspelled, naming the preset and its keywords.

        Examples:
            - A keyed preset resolves its own provider and credential:
                ```python
                >>> from digitalearth import Map                       # doctest: +SKIP
                >>> Map(domain=(-60, -5, -55, 0)).basemap(             # doctest: +SKIP
                ...     "Planet.NICFI", preset={"date": "2024-01", "flavour": "visual"}
                ... )                                                  # doctest: +SKIP

                ```

        See Also:
            digitalearth.base.basemaps: the keyed-preset definitions this resolves.
        """
        loose = sorted(set(kwargs) & PRESET_KEYWORDS)
        if loose:
            # These would otherwise reach add_tiles and fail there, or worse, be accepted and ignored.
            raise ValueError(
                f"basemap() takes preset keywords in preset={{...}}, not loose; move {loose} into it — "
                f"basemap({source!r}, preset={{{loose[0]!r}: ...}})"
            )
        if not is_keyed_basemap(source):
            if preset:
                raise ValueError(
                    f"basemap({source!r}) takes no preset keywords; {sorted(preset)} apply only to a "
                    f"keyed preset such as 'Planet.NICFI'"
                )
            if api_key is not None:
                # Dropping it silently would leave a caller believing they had authenticated.
                raise ValueError(
                    f"basemap({source!r}) takes no api_key; it is a token-free source. Credentials apply "
                    f"only to a keyed preset such as 'Planet.NICFI'"
                )
        return self._describe_basemap(
            source, preset, api_key, kwargs, name=name, visible=visible
        )

    def _describe_basemap(
        self,
        source: Any,
        preset: Optional[dict],
        api_key: Optional[str],
        options: dict,
        name: Optional[str] = None,
        visible: bool = True,
    ) -> Any:
        """Record the basemap this map should draw and let :func:`draw_basemap` fetch it.

        No data source is recorded — a tile set is named by its provider, which is a style property here,
        not data this process holds and could resolve an ``object:`` reference to.

        The credential is deliberately absent from the *record*. A figure is written to JSON and read back,
        and an API key written into one leaks with it; the keyed presets read theirs from the environment,
        so a reloaded figure asks for the same basemap and authenticates itself. An explicit ``api_key``
        travels with the scene instead, under this layer's id, and is forgotten with the layer. A provider
        **object** travels the same way, and for both reasons at once (see :func:`_described_tile_source`):
        it is an engine object, and an ``xyzservices.TileProvider`` holds the caller's key among its fields.

        What a provider object is recorded *as* is its name, except for an OGC provider, which has none: that
        one is recorded field by field, minus the ``extra_params`` its credential goes in, because recording
        it by a name it does not have made a figure read back draw the shared default in its place (round 3,
        M10).

        The **extent is** recorded, as four plain numbers in the display CRS. It is not styling: a tile
        mosaic *is* the frame it was fetched for, at the zoom that frame implies, so a figure that did not
        carry it could not ask for the same basemap again — and, because a basemap is listed before the
        data it underlays, a replay reaches :func:`draw_basemap` before anything has framed the axes at all
        (round 2, H2). ``None`` when the axes is not framed yet, which is the case cleopatra refuses.

        Args:
            source: The provider as the caller named it, or ``None`` for the shared default.
            preset: The keyed preset's own keywords, or ``None``.
            api_key: The credential for a keyed preset, or ``None`` to read it from the environment.
            options: The keywords forwarded to ``add_tiles``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).

        Returns:
            Whatever ``add_tiles`` returned. Not ``None``: tiles that cannot be fetched raise, and the
            layer is dropped from the description with them — :meth:`stock_img` is the one caller that
            turns that into a quiet ``None``, because a backdrop is best-effort.

        Raises:
            ValueError: from :func:`draw_basemap` — a keyed preset whose credential is unavailable, an
                extent entirely outside the provider's coverage, or neither an axes nor a description
                carrying a frame to tile.
            TypeError: from :func:`draw_basemap`, for a preset keyword that is missing or misspelled.
        """
        recorded, held = _described_tile_source(source)
        if held is not None:
            options = {**options, "source": held}
        return self._draw(
            LayerRecord(
                "basemap",
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "basemap",
                        "source": recorded,
                        "preset": dict(preset or {}),
                        "extent": _axes_extent(self.ax),
                    }
                ),
                key=api_key,
                opts=options,
            )
        )

    def _coverage_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Return the map's domain as lon/lat ``(west, south, east, north)``, or ``None`` when unset.

        A declared ``domain`` is preferred. Without one the axes limits are used instead — reprojected to
        lon/lat when the display CRS is something else — because plotting data and then adding a basemap is
        the common flow, and reading only ``domain`` left the coverage guard inert for it. Matplotlib's
        default unit square is not mistaken for an extent: cleopatra refuses to tile axes with no data, so
        by the time this runs the limits are real.

        Returns:
            The extent in lon/lat, or ``None`` when neither source yields one.
        """
        domain = getattr(self, "domain", None)
        resolved = None
        if domain is not None:
            try:
                resolved = resolve_domain(domain)
            except (KeyError, TypeError, ValueError):
                return None  # an unrecognised domain — leave the coverage question to the service
        if resolved is None:
            return self._axes_lonlat_extent()
        west, south, east, north = (float(value) for value in resolved)
        return west, south, east, north

    def _axes_lonlat_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Return the current axes limits as lon/lat, reprojecting from the display CRS when needed.

        The edges are sampled rather than just the two corners: outside the cylindrical projections a
        projected rectangle's lon/lat envelope is not the envelope of its corners, and taking the corners
        gave a zero-height box at the wrong latitude for a polar stereographic map. The sampling follows
        the edges only, so a pole enclosed *inside* the box is not represented — the envelope is a good
        approximation of what the axes show, not a proof about their interior.

        Returns:
            ``(west, south, east, north)`` in lon/lat, or ``None`` when the limits are matplotlib's
            untouched default, the reprojection fails, or any sampled point lands outside the
            projection's domain — in which case the coverage question is left to the service.
        """
        west, east = (float(v) for v in self.ax.get_xlim())
        south, north = (float(v) for v in self.ax.get_ylim())
        if (west, east, south, north) == (0.0, 1.0, 0.0, 1.0):
            return None  # matplotlib's default unit square — nothing has been drawn yet
        if self.crs is None or same_crs(self.crs, 4326):
            return west, south, east, north
        xs, ys = _edge_samples(west, south, east, north)
        try:
            lons, lats = reproject_coordinates(xs, ys, from_crs=self.crs, to_crs=4326)
        except (ValueError, RuntimeError):
            # A CRS pyproj cannot resolve: leave the coverage question to the service rather than fail
            # the plot. A TypeError here would be a bug in this call, so it is deliberately not caught.
            return None
        if not all(math.isfinite(value) for value in (*lons, *lats)):
            # pyproj answers `inf` for a point outside the projection's domain rather than raising —
            # an orthographic globe's corners are off the visible hemisphere, for instance. That is an
            # unknown extent, which takes the same fail-open path as an unresolvable CRS.
            return None
        return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))
