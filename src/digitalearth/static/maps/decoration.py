"""DecorationMixin — annotation, basemap tiles and Natural-Earth vector decoration.

Lon/lat text/annotate, a tile basemap, a backdrop ``stock_img``, and the Natural-Earth coastline/border/
land/ocean/lake/river layers (with the globe limb-splitting/closing helpers behind them).

The reference data comes from ``cleopatra.basemap.reference`` (``natural_earth`` raw lon/lat coordinate arrays for
the globe limb-splitting; ``add_features`` for the flat, hole-aware reprojected render) — these helpers
moved out of pyramids into cleopatra in pyramids 0.32 / cleopatra 0.17.
"""

import contextlib
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from cleopatra.basemap.reference import add_features, natural_earth
from cleopatra.basemap.tiles import add_tiles
from matplotlib.collections import PolyCollection
from pyramids.base.crs import reproject_coordinates

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


def _described_tile_source(source: Any) -> Tuple[Any, Any]:
    """Split a basemap source into the name a description records and the object held beside the layer.

    An ``xyzservices.TileProvider`` is a ``dict`` subclass whose fields include the caller's ``apikey``, so
    recording it wrote that credential into every figure the map produced — and rebuilding it from the
    description handed cleopatra a plain ``dict``, which has no ``build_url`` and cannot be tiled from. Its
    **name** is what a description carries: it names the same tiles wherever the figure is read, and it
    carries no secret.

    Args:
        source: The ``source`` a caller passed to :meth:`DecorationMixin.basemap`.

    Returns:
        ``(recorded, held)``. A name or ``None`` passes through and is held nowhere. A provider object is
        recorded by its own ``name`` — or as ``None``, the shared default, when it has none — and held beside
        the layer, which is what this map tiles from.
    """
    if source is None or isinstance(source, str):
        return source, None
    name = getattr(source, "name", None)
    return (name if isinstance(name, str) else None), source


def _resolve_tile_source(source: Any) -> Any:
    """Resolve an unnamed or cross-tier-named basemap into the provider ``add_tiles`` understands.

    Two translations, both of them the static tier's side of the one-basemap-per-name contract (#247).
    ``None`` means "the caller named none", which is
    :data:`~digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER` and not cleopatra's own fallback. And a
    shared name such as ``"CartoLight"`` is spelled as the ``xyzservices`` path that resolves to the very
    tiles the other tiers request for it. Anything else — an ``xyzservices`` path, a ``TileProvider``, a
    URL — is handed on untouched.

    Args:
        source: The ``source`` a caller passed to :meth:`DecorationMixin.basemap`, already known not to be a
            keyed preset.

    Returns:
        The provider name or object to hand ``cleopatra.basemap.tiles.add_tiles``.

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
    """
    name = DEFAULT_BASEMAP_PROVIDER if source is None else source
    if isinstance(name, str):
        return _SHARED_PROVIDERS.get(name.lower(), name)
    return name


def draw_text(scene: Any, _data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Place the text label a described layer asks for, at the lon/lat it recorded.

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
    drawn = scene.ax.text(xy[0], xy[1], props["s"], **drawing_style(scene, layer))
    return DrawnLayer(artist=drawn, artists=(drawn,))


def draw_annotate(scene: Any, _data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Annotate the lon/lat a described layer recorded, optionally with an arrow.

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
    drawn = scene.ax.annotate(
        props["s"], xy=xy, xytext=props["xytext"], **drawing_style(scene, layer)
    )
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

    The ordering is matplotlib's own — the one
    :meth:`~digitalearth.static.maps.projection.ProjectionMixin.set_extent` takes — and the values are in
    the display CRS, so the frame is read and written back without a reprojection.

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
        **kwargs,
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
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **kwargs: Forwarded to ``Axes.text`` (e.g. ``ha``, ``va``, ``fontsize``, ``color``).

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
        **kwargs,
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
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **kwargs: Forwarded to ``Axes.annotate`` (e.g. ``arrowprops``, ``textcoords``, ``fontsize``).

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
            **kwargs: Forwarded to :meth:`imshow` (raster) or :meth:`basemap` (tiles).

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
            im = self.imshow(
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
        **kwargs,
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
            source: A cleopatra/``xyzservices`` provider name, an ``xyzservices.TileProvider``, one of the
                cross-tier names (``"CartoLight"``/``"CartoDark"``/``"CartoVoyager"``/``"OSM"``), a keyed
                preset name such as ``"Planet.NICFI"``, or ``None`` for
                :data:`~digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER`.
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
