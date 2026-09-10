"""DecorationMixin — annotation, basemap tiles and Natural-Earth vector decoration.

Lon/lat text/annotate, a tile basemap, a backdrop ``stock_img``, and the Natural-Earth coastline/border/
land/ocean/lake/river layers (with the globe limb-splitting/closing helpers behind them).

The reference data comes from ``cleopatra.basemap.reference`` (``natural_earth`` raw lon/lat coordinate arrays for
the globe limb-splitting; ``add_features`` for the flat, hole-aware reprojected render) — these helpers
moved out of pyramids into cleopatra in pyramids 0.32 / cleopatra 0.17.
"""

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple

import numpy as np
from cleopatra.basemap.reference import add_features, natural_earth
from cleopatra.basemap.tiles import add_tiles
from matplotlib.collections import PolyCollection
from pyramids.base.crs import reproject_coordinates

from digitalearth.base.basemaps import (
    KeyedTileSource,
    get_keyed_basemap,
    is_keyed_basemap,
)
from digitalearth.static import projections
from digitalearth.static.domains import resolve_domain

logger = logging.getLogger(__name__)

#: Natural-Earth layers that ``cleopatra.basemap.reference`` renders as filled polygons (vs. line layers); used to
#: translate this package's singular matplotlib style keys to the right collection keys for ``add_features``.
_POLYGON_LAYERS = frozenset({"land", "ocean", "lakes"})


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

#: Keywords `basemap()` routes to a keyed preset rather than to `add_tiles`. Derived from the presets so a
#: new one cannot be forgotten here.
_PRESET_KEYWORDS = frozenset({"date", "flavour", "mosaic"})


@contextlib.contextmanager
def _quiet_tile_urls() -> Iterator[None]:
    """Raise the cleopatra tile logger above DEBUG so a keyed URL cannot reach the logs.

    The credential is part of the tile URL — that is how the service authenticates — and cleopatra logs
    that URL on a failed fetch. Errors still surface: the ``ConnectionError`` cleopatra raises carries the
    tile coordinates, not the URL, so nothing diagnostic is lost.

    Yields:
        ``None``, with the logger's level restored on the way out even if the fetch raises.
    """
    logger_obj = logging.getLogger(_CLEOPATRA_TILES_LOGGER)
    previous = logger_obj.level
    # NOTSET (0) already compares <= DEBUG, so the one comparison covers both an unset level (which
    # inherits a possibly-DEBUG parent) and an explicitly-DEBUG one.
    if previous <= logging.DEBUG:
        logger_obj.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger_obj.setLevel(previous)


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

    def text(self, lon: float, lat: float, s: str, *, crs: Any = 4326, **kwargs) -> Any:
        """Place a text label at a ``lon``/``lat`` location (reprojected to the display CRS).

        The point is reprojected from ``crs`` (lon/lat by default) into the display CRS via pyramids, then
        drawn with ``Axes.text``. On a globe, a point on the **far side** reprojects to non-finite coordinates
        and is skipped (no artist, returns ``None``).

        Args:
            lon: Longitude (x) of the label, in ``crs``.
            lat: Latitude (y) of the label, in ``crs``.
            s: The text to draw.
            crs: CRS of ``lon``/``lat`` (default ``4326`` = WGS84 lon/lat).
            **kwargs: Forwarded to ``Axes.text`` (e.g. ``ha``, ``va``, ``fontsize``, ``color``).

        Returns:
            The :class:`matplotlib.text.Text`, or ``None`` if the point is off the visible globe.
        """
        xy = self._reproject_point(lon, lat, crs)
        if xy is None:
            return None
        return self.ax.text(xy[0], xy[1], s, **kwargs)

    def annotate(
        self,
        lon: float,
        lat: float,
        s: str,
        *,
        xytext: Any = None,
        crs: Any = 4326,
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
            **kwargs: Forwarded to ``Axes.annotate`` (e.g. ``arrowprops``, ``textcoords``, ``fontsize``).

        Returns:
            The :class:`matplotlib.text.Annotation`, or ``None`` if the point is off the visible globe.
        """
        xy = self._reproject_point(lon, lat, crs)
        if xy is None:
            return None
        return self.ax.annotate(s, xy=xy, xytext=xytext, **kwargs)

    def stock_img(
        self,
        dataset: Any = None,
        *,
        zorder: float = -3.0,
        cmap: str = "gist_earth",
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
            dataset: A pyramids ``Dataset`` to use as the backdrop, or ``None`` to try a tile basemap.
            zorder: Draw order for the backdrop (default ``-3.0``, below data/coastlines).
            cmap: Colormap for a raster backdrop (ignored for the tile path).
            **kwargs: Forwarded to :meth:`imshow` (raster) or :meth:`basemap` (tiles).

        Returns:
            The backdrop ``AxesImage`` (raster path), the tile artist, or ``None`` if a tile backdrop is
            unavailable offline.
        """
        if dataset is None:
            try:
                return self.basemap(**kwargs)
            except Exception as exc:  # tile servers unavailable — best-effort backdrop
                logger.debug("stock_img tile basemap unavailable: %s", exc)
                return None
        with self._preserve_view():
            im = self.imshow(dataset, cmap=cmap, **kwargs)
            im.set_zorder(zorder)
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
        self, rings: List[np.ndarray], *, facecolor: Any, zorder: float
    ) -> Any:
        """Fill projected rings with a solid colour on a globe (map-specific overlay; clipped at frame time).

        cleopatra ``PolygonGlyph`` only fills when given per-polygon *values*, so a uniform land/ocean fill is
        drawn as a plain ``PolyCollection`` directly on the axes — exactly like the globe coastline ``ax.plot``
        overlay. ``apply_projection_frame(clip_artists=True)`` clips it to the boundary at render time. The
        view limits are preserved so a global fill never autoscales the axes out.

        Args:
            rings: Closed, finite projected fill rings (from :meth:`_project_polygon_features`).
            facecolor: Solid fill colour.
            zorder: Draw order (ocean below land below data below coastlines).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer), or ``None`` when ``rings`` is empty.
        """
        if not rings:
            return None
        with self._preserve_view():
            pc = PolyCollection(
                rings, facecolors=facecolor, edgecolors="none", zorder=zorder
            )
            self.ax.add_collection(pc)
        return self._add_layer(None, pc)

    def _natural_earth(
        self,
        layer: str,
        resolution: str,
        defaults: dict,
        *,
        polygon: bool = False,
        zorder: float = 0.5,
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
            defaults: Base style; ``color``/``facecolor`` is the fill colour for polygon layers.
            polygon: When True, treat the layer as filled polygons on a globe (else as lines).
            zorder: Draw order (globe polygon fills; also forwarded to ``add_features`` on a flat map).
            **kwargs: Style overrides merged over ``defaults``.
        """
        if self.globe:
            parts = natural_earth(layer, resolution)
            style = {**defaults, **kwargs}
            if polygon:
                facecolor = style.get("facecolor", style.get("color", "#efefdb"))
                return self._fill_globe_polygons(
                    self._project_polygon_features(parts),
                    facecolor=facecolor,
                    zorder=zorder,
                )
            style.pop("edgecolor", None)
            style.pop("facecolor", None)
            segments = self._project_line_features(parts)
            return [self.ax.plot(seg[:, 0], seg[:, 1], **style)[0] for seg in segments]
        kind = "polygon" if layer in _POLYGON_LAYERS else "line"
        style = _to_feature_style(kind, {**defaults, **kwargs})
        had_data = (
            bool(self.layers)
            or bool(self.ax.images)
            or bool(self.ax.collections)
            or bool(self.ax.lines)
        )
        add_features(self.ax, layer, resolution, crs=self.crs, zorder=zorder, **style)
        if (
            not had_data
        ):  # add_features pinned the (empty) view; fit it to the layer we just drew
            self.ax.autoscale()
        return self.ax

    def coastlines(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth coastlines (``cleopatra.basemap.reference`` ``"coastline"`` layer).

        Returns:
            The drawn coastline artist (a list of polyline artists on a globe; the reprojected plot artist
            on a flat map).
        """
        return self._natural_earth(
            "coastline",
            resolution,
            {"color": "black", "linewidth": 0.5},
            zorder=2.5,
            **kwargs,
        )

    def borders(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth country borders.

        Returns:
            The drawn border artist (a list of polyline artists on a globe; the reprojected plot artist on a
            flat map).
        """
        return self._natural_earth(
            "borders",
            resolution,
            {"color": "gray", "linewidth": 0.4},
            zorder=2.5,
            **kwargs,
        )

    def land(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth land polygons.

        On a **flat** map the polygons are reprojected and filled directly. On a **globe** map they are
        re-closed at the projection limb into finite rings and filled as a map-specific overlay (drawn below
        data and coastlines, clipped to the boundary). Interior rings (holes) are dropped in v1 — see #43.

        Returns:
            The land fill layer (a ``PolyCollection`` on a globe, ``None`` when nothing is on the near side;
            the reprojected plot artist on a flat map).
        """
        return self._natural_earth(
            "land",
            resolution,
            {"color": "#efefdb", "edgecolor": "none"},
            polygon=True,
            zorder=-1.5,
            **kwargs,
        )

    def ocean(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth ocean polygons.

        On a **globe** map, ``ocean`` fills the whole projection disc (the boundary ring) with the ocean
        colour and lets land overlay it — exact and far cheaper than clipping the global ocean polygon. On a
        **flat** map, the Natural-Earth ocean polygons are reprojected and filled directly.

        Returns:
            The ocean fill layer (a ``PolyCollection`` disc on a globe; the reprojected plot artist on a flat
            map).
        """
        color = kwargs.pop("color", "#cfe6f5")
        if self.globe:
            boundary = self._frame()[0]
            return self._fill_globe_polygons(
                [np.asarray(boundary)], facecolor=color, zorder=-2.0
            )
        return self._natural_earth(
            "ocean", resolution, {"color": color, "edgecolor": "none"}, **kwargs
        )

    def lakes(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth lake polygons.

        Like :meth:`land`, but with a water colour and drawn just above land (so lakes sit on the land) and
        still below data and coastlines. On a globe the polygons are re-closed at the projection limb.

        Returns:
            The lake fill layer (a ``PolyCollection`` on a globe, ``None`` when nothing is on the near side;
            the reprojected plot artist on a flat map).
        """
        return self._natural_earth(
            "lakes",
            resolution,
            {"color": "#cfe6f5", "edgecolor": "none"},
            polygon=True,
            zorder=-1.4,
            **kwargs,
        )

    def rivers(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth rivers (line centerlines), split at the projection limb on a globe.

        Returns:
            The drawn river artist (a list of polyline artists on a globe; the reprojected plot artist on a
            flat map).
        """
        return self._natural_earth(
            "rivers",
            resolution,
            {"color": "#5a8fcf", "linewidth": 0.4},
            zorder=2.4,
            **kwargs,
        )

    def basemap(
        self,
        source: Any = None,
        *,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Add an XYZ-tile basemap to the axes via ``cleopatra.basemap.tiles.add_tiles`` in the display CRS.

        ``source`` is passed through to cleopatra unchanged — a provider name, an
        ``xyzservices.TileProvider``, or ``None`` for its default — with one addition: the name of a
        **keyed** basemap preset (see :mod:`digitalearth.base.basemaps`) is resolved here into a configured
        provider, with its credential read from the environment and its coverage checked against the map's
        domain first.

        Args:
            source: A cleopatra provider name, an ``xyzservices.TileProvider``, ``None``, or a keyed preset
                name such as ``"Planet.NICFI"``.
            api_key: Credential for a keyed preset; ``None`` reads the preset's environment variable.
            **kwargs: Forwarded to ``add_tiles`` — plus the preset's own keywords when ``source`` names one
                (for NICFI: ``date``, ``flavour``, ``mosaic``).

        Returns:
            The tile artist ``add_tiles`` added to the axes.

        Raises:
            ValueError: when a keyed preset is unknown, its credential is unavailable, or the map's domain
                lies entirely outside the service's coverage.

        Examples:
            - A keyed preset resolves its own provider and credential:
                ```python
                >>> from digitalearth import Map                       # doctest: +SKIP
                >>> Map(domain=(-60, -5, -55, 0)).basemap(             # doctest: +SKIP
                ...     "Planet.NICFI", date="2024-01", flavour="visual"
                ... )                                                  # doctest: +SKIP

                ```

        See Also:
            digitalearth.base.basemaps: the keyed-preset definitions this resolves.
        """
        if not is_keyed_basemap(source):
            stray = sorted(set(kwargs) & _PRESET_KEYWORDS)
            if stray:
                raise ValueError(
                    f"basemap({source!r}) takes no preset keywords; {stray} apply only to a keyed preset "
                    f"such as 'Planet.NICFI'"
                )
            return add_tiles(self.ax, source=source, crs=self.crs, **kwargs)

        preset_kwargs = {
            key: kwargs.pop(key) for key in list(kwargs) if key in _PRESET_KEYWORDS
        }
        keyed = get_keyed_basemap(str(source), **preset_kwargs)
        keyed.check_bounds(self._lonlat_domain())
        provider = _keyed_tile_provider(keyed, api_key)
        kwargs.setdefault("attribution", keyed.attribution)
        with _quiet_tile_urls():
            return add_tiles(self.ax, source=provider, crs=self.crs, **kwargs)

    def _lonlat_domain(self) -> Optional[Tuple[float, float, float, float]]:
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

        Returns:
            ``(west, south, east, north)`` in lon/lat, or ``None`` when the limits are matplotlib's
            untouched default or the reprojection fails — in which case the coverage question is left to
            the service.
        """
        west, east = (float(v) for v in self.ax.get_xlim())
        south, north = (float(v) for v in self.ax.get_ylim())
        if (west, east, south, north) == (0.0, 1.0, 0.0, 1.0):
            return None  # matplotlib's default unit square — nothing has been drawn yet
        if self.crs in (4326, "EPSG:4326", None):
            return west, south, east, north
        try:
            xs, ys = reproject_coordinates(
                [west, east], [south, north], from_crs=self.crs, to_crs=4326
            )
        except (ValueError, RuntimeError):
            # A CRS pyproj cannot resolve: leave the coverage question to the service rather than fail
            # the plot. A TypeError here would be a bug in this call, so it is deliberately not caught.
            return None
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
