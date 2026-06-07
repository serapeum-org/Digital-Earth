"""DecorationMixin — annotation, basemap tiles and Natural-Earth vector decoration.

Lon/lat text/annotate, a tile basemap, a backdrop ``stock_img``, and the Natural-Earth coastline/border/
land/ocean/lake/river layers (with the globe limb-splitting/closing helpers behind them).
"""
import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib.collections import PolyCollection
from cleopatra.tiles import add_tiles
from pyramids.base.crs import reproject_coordinates
from pyramids.basemap import natural_earth

from digitalearth._crs import source_epsg
from digitalearth.scene import projections

logger = logging.getLogger(__name__)


class DecorationMixin:
    """Annotation and basemap/Natural-Earth decoration for :class:`~digitalearth.scene.map.Map`."""

    def _reproject_point(self, lon: float, lat: float, crs: Any) -> Optional[Tuple[float, float]]:
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

    def annotate(self, lon: float, lat: float, s: str, *, xytext: Any = None, crs: Any = 4326,
                 **kwargs) -> Any:
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

    def stock_img(self, dataset: Any = None, *, zorder: float = -3.0, cmap: str = "gist_earth",
                  **kwargs) -> Any:
        """Draw a background raster (a "stock image" backdrop) beneath all data layers.

        Pass a pyramids ``Dataset`` (e.g. a low-res relief/imagery raster) to draw as the backdrop — it is
        reprojected to the display CRS and drawn at a low ``zorder`` so data layers sit on top, and the current
        view is preserved so a global backdrop never autoscales a regional view out. With ``dataset=None`` a
        best-effort XYZ-tile basemap is used instead (network; skipped offline).

        Note: pyramids/cleopatra ship no bundled relief raster yet, so the no-argument form relies on tiles;
        a bundled low-res relief raster is tracked upstream (see ``stock_img`` task in the remaining-plots
        plan). Until then, supply your own backdrop ``Dataset``.

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


    def _project_line_features(self, fc: Any) -> List[np.ndarray]:
        """Project a line FeatureCollection (lon/lat) to the display CRS, split at the projection limb.

        Reprojecting a global line to a clipped projection (e.g. orthographic) sends the far side to
        non-finite coordinates; per-line splitting at those gaps keeps the visible arcs and avoids the
        ``NaN/Inf`` errors geopandas' ``clip``/``plot`` raise on such geometry.
        """
        from_crs = source_epsg(fc, 4326)
        segments: List[np.ndarray] = []
        for geom in fc.geometry:
            if geom is None:
                continue
            parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for part in parts:
                xy = np.asarray(part.coords, dtype=float)
                if xy.size == 0:
                    continue
                x, y = reproject_coordinates(xy[:, 0].tolist(), xy[:, 1].tolist(),
                                             from_crs=from_crs, to_crs=self.crs)
                segments += projections._split_finite(np.asarray(x, float), np.asarray(y, float))
        return segments

    def _project_polygon_features(self, fc: Any) -> List[np.ndarray]:
        """Project a polygon FeatureCollection (lon/lat) to the display CRS as finite, limb-clipped fill rings.

        The fill analogue of :meth:`_project_line_features`: each exterior ring is densified, reprojected, and
        re-closed against the projection boundary (``self._frame()[0]``) so a polygon straddling the limb
        becomes one or more closed, fully-finite rings instead of injecting ``inf``/``nan`` into the fill.
        Interior rings (holes) are dropped in v1 — see Digital-Earth#43.

        Args:
            fc: A pyramids ``FeatureCollection`` of polygons in lon/lat (or any CRS via ``fc.epsg``).

        Returns:
            A list of closed ``(N, 2)`` projected fill rings (empty when nothing is on the near side).
        """
        from_crs = source_epsg(fc, 4326)
        boundary = self._frame()[0]
        rings: List[np.ndarray] = []
        for geom in fc.geometry:
            if geom is None:
                continue
            parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for part in parts:
                xy = np.asarray(part.exterior.coords, dtype=float)
                if xy.size == 0:
                    continue
                xy = projections.densify_lonlat(xy, step_deg=1.0)
                x, y = reproject_coordinates(xy[:, 0].tolist(), xy[:, 1].tolist(),
                                             from_crs=from_crs, to_crs=self.crs)
                rings += projections.close_visible_runs(np.asarray(x, float), np.asarray(y, float), boundary)
        return rings

    def _fill_globe_polygons(self, rings: List[np.ndarray], *, facecolor: Any, zorder: float) -> Any:
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
            pc = PolyCollection(rings, facecolors=facecolor, edgecolors="none", zorder=zorder)
            self.ax.add_collection(pc)
        return self._add_layer(None, pc)

    def _natural_earth(self, layer: str, resolution: str, defaults: dict, *, polygon: bool = False,
                       zorder: float = 0.5, **kwargs) -> Any:
        """Draw a Natural-Earth vector layer reprojected to the display CRS, clipped to the current view.

        On a **globe** map, line layers (coastline/borders/rivers) are projected per-line and split at the
        projection limb (the far side reprojects to non-finite coords) and drawn as plain polylines; polygon
        layers (``polygon=True``: land/lakes) are projected and re-closed at the limb into finite rings and
        filled via :meth:`_fill_globe_polygons`. Both are clipped to the boundary when the frame is applied.
        On a **flat** map, the layer is reprojected with ``GeoDataFrame.plot``; since Natural Earth is global
        and that autoscales the axes, the data's limits are preserved when a data layer is already present.

        Args:
            layer: Natural-Earth layer name (e.g. ``"coastline"``, ``"land"``).
            resolution: Natural-Earth resolution (``"110m"``/``"50m"``/``"10m"``).
            defaults: Base style; ``color``/``facecolor`` is the globe fill colour for polygon layers.
            polygon: When True, treat the layer as filled polygons on a globe (else as lines).
            zorder: Globe draw order for polygon fills.
            **kwargs: Style overrides merged over ``defaults``.
        """
        fc = natural_earth(layer, resolution)
        if self.globe:
            style = {**defaults, **kwargs}
            if polygon:
                facecolor = style.get("facecolor", style.get("color", "#efefdb"))
                return self._fill_globe_polygons(self._project_polygon_features(fc),
                                                 facecolor=facecolor, zorder=zorder)
            style.pop("edgecolor", None); style.pop("facecolor", None)
            artists = [self.ax.plot(seg[:, 0], seg[:, 1], **style)[0] for seg in self._project_line_features(fc)]
            return artists
        with self._preserve_view():
            artist = fc.to_crs(self.crs).plot(ax=self.ax, **{**defaults, **kwargs})
        return artist

    def coastlines(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth coastlines (``pyramids.basemap.natural_earth("coastline")``)."""
        return self._natural_earth(
            "coastline", resolution, {"color": "black", "linewidth": 0.5}, zorder=2.5, **kwargs
        )

    def borders(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth country borders."""
        return self._natural_earth(
            "borders", resolution, {"color": "gray", "linewidth": 0.4}, zorder=2.5, **kwargs
        )

    def land(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth land polygons.

        On a **flat** map the polygons are reprojected and filled directly. On a **globe** map they are
        re-closed at the projection limb into finite rings and filled as a map-specific overlay (drawn below
        data and coastlines, clipped to the boundary). Interior rings (holes) are dropped in v1 — see #43.
        """
        return self._natural_earth(
            "land", resolution, {"color": "#efefdb", "edgecolor": "none"}, polygon=True, zorder=-1.5, **kwargs
        )

    def ocean(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth ocean polygons.

        On a **globe** map, ``ocean`` fills the whole projection disc (the boundary ring) with the ocean
        colour and lets land overlay it — exact and far cheaper than clipping the global ocean polygon. On a
        **flat** map, the Natural-Earth ocean polygons are reprojected and filled directly.
        """
        color = kwargs.pop("color", "#cfe6f5")
        if self.globe:
            boundary = self._frame()[0]
            return self._fill_globe_polygons([np.asarray(boundary)], facecolor=color, zorder=-2.0)
        return self._natural_earth("ocean", resolution, {"color": color, "edgecolor": "none"}, **kwargs)

    def lakes(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth lake polygons.

        Like :meth:`land`, but with a water colour and drawn just above land (so lakes sit on the land) and
        still below data and coastlines. On a globe the polygons are re-closed at the projection limb.
        """
        return self._natural_earth(
            "lakes", resolution, {"color": "#cfe6f5", "edgecolor": "none"}, polygon=True, zorder=-1.4, **kwargs
        )

    def rivers(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth rivers (line centerlines), split at the projection limb on a globe."""
        return self._natural_earth(
            "rivers", resolution, {"color": "#5a8fcf", "linewidth": 0.4}, zorder=2.4, **kwargs
        )

    def basemap(self, source: Any = None, **kwargs) -> Any:
        """Add an XYZ-tile basemap to the axes via ``cleopatra.tiles.add_tiles`` in the display CRS."""
        return add_tiles(self.ax, source=source, crs=self.crs, **kwargs)

