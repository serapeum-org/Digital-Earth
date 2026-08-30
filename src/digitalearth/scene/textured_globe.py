"""TexturedGlobe — drape a pyramids raster (or a tile basemap) over cleopatra's 3-D textured sphere.

``cleopatra.glyphs.globe.TexturedGlobeGlyph`` paints an equirectangular ``(H, W, 3|4)`` array onto a tilted,
spinnable sphere on a matplotlib ``Axes3D``. It is deliberately geometry-only: it knows nothing about rasters,
CRSes or nodata. This module is the Digital-Earth half of that seam — it turns geospatial inputs into the
texture the glyph wants, and maps lon/lat back onto the rendered sphere:

- :meth:`TexturedGlobe.from_dataset` reprojects a pyramids ``Dataset`` to EPSG:4326 (pyramids does the warp),
  colour-maps the band through cleopatra's colormap resolver, and pastes it into a **global** transparent
  canvas at its true lon/lat position — so a regional raster floats on the globe where it belongs instead of
  being stretched over the whole sphere.
- :meth:`TexturedGlobe.from_provider` pulls a whole-world XYZ basemap via ``cleopatra.basemap.tiles``.
- :meth:`TexturedGlobe.project` and :meth:`TexturedGlobe.points` push lon/lat (or a pyramids
  ``FeatureCollection``) through the glyph's own ``transform``, so overlays land exactly on the drawn surface
  at any ``spin`` — including the axial tilt — rather than being re-derived here.

Like the cleopatra glyph it wraps (and unlike :class:`~digitalearth.scene.map.Map`), this is a standalone
class rather than a :class:`~digitalearth.scene.scene.Scene` subclass: ``Scene`` owns a 2-D axes and the
layer/colorbar lifecycle, none of which applies to a textured sphere.
"""
import warnings
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cleopatra.basemap.tiles import world_texture
from cleopatra.glyphs.globe.textured_globe_glyph import (
    EARTH_TILT_DEG,
    TexturedGlobeGlyph,
)
from cleopatra.styling.colors import resolve_colormap
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from digitalearth._arrays import finite, read_masked_band
from digitalearth._crs import source_epsg
from digitalearth.animation import save_animation

#: Default shape of the global equirectangular canvas built by :meth:`TexturedGlobe.from_dataset`,
#: as ``(rows, columns)`` — a 0.125-degree grid, comfortably finer than the glyph's default mesh.
DEFAULT_TEXTURE_SHAPE = (1440, 2880)

#: EPSG code of the lon/lat grid the glyph's texture is defined on.
_TEXTURE_EPSG = 4326

#: The glyph's own default inter-frame interval, in milliseconds — mirrored here so :meth:`animate` can
#: record the frame rate without reading the animation's private state.
_DEFAULT_INTERVAL_MS = 50


def _texture_axes(n_lat: int, n_lon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ``(lat, lon)`` degree vectors of a texture of shape ``(n_lat, n_lon)``.

    Endpoint-inclusive, matching the glyph's documented layout: row 0 is +90 and the last row -90; column 0
    is -180 and the last column +180.
    """
    return np.linspace(90.0, -90.0, n_lat), np.linspace(-180.0, 180.0, n_lon)


def _nearest_index(targets: np.ndarray, coords: np.ndarray, cell: Optional[float] = None) -> np.ndarray:
    """Nearest-cell index of each ``targets`` value in the uniform 1-D cell-centre vector ``coords``.

    Returns ``-1`` where a target falls outside the grid's **footprint** — the cell centres widened by half a
    cell at each end — which the caller turns into a transparent texture cell. Membership is decided by the
    footprint rather than by whether the target rounds onto a cell, so every canvas sample inside the raster
    is coloured even when the canvas is far coarser than the source; otherwise a small raster on a coarse
    canvas would drop out entirely. Rasters are uniform grids, so the index is arithmetic, not a search.

    Args:
        targets: The coordinates to look up.
        coords: The source grid's cell centres, uniformly spaced (ascending or descending).
        cell: Cell size to assume when ``coords`` holds a single value and the step is therefore unknown.
    """
    if coords.size == 0:
        return np.full(np.shape(targets), -1, dtype=int)
    step = float(coords[1] - coords[0]) if coords.size > 1 else float(cell or 0.0)
    if step == 0.0:
        # A lone cell with no known size spans no area; only an exact hit maps onto it.
        return np.where(np.isclose(targets, float(coords[0])), 0, -1)
    half = abs(step) / 2.0
    lo, hi = min(coords[0], coords[-1]) - half, max(coords[0], coords[-1]) + half
    inside = (targets >= lo) & (targets <= hi)
    idx = np.clip(np.rint((targets - float(coords[0])) / step).astype(int), 0, coords.size - 1)
    return np.where(inside, idx, -1)


def _lonlat_to_body(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convert lon/lat degrees to the glyph's body-frame unit-sphere ``(N, 3)`` coordinates.

    Mirrors the mesh the glyph builds — ``x = cos(lat)cos(lon)``, ``y = cos(lat)sin(lon)``, ``z = sin(lat)``
    — so a point at ``(lon, lat)`` lands on the texture cell showing that longitude and latitude.
    """
    lon_rad, lat_rad = np.deg2rad(np.asarray(lon, dtype=float)), np.deg2rad(np.asarray(lat, dtype=float))
    return np.stack(
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)],
        axis=-1,
    )


def _view_vector(elev: float, azim: float) -> np.ndarray:
    """Unit vector pointing from the sphere's centre toward the camera at ``elev``/``azim`` degrees."""
    e, a = np.deg2rad(float(elev)), np.deg2rad(float(azim))
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


class TexturedGlobe:
    """A 3-D textured globe built from geospatial data.

    Wraps ``cleopatra.glyphs.globe.TexturedGlobeGlyph``: this class owns the data → texture conversion and
    the lon/lat overlay maths, the glyph owns the sphere, the tilt, the lighting and the render.

    Args:
        texture: An equirectangular ``(H, W, 3)`` or ``(H, W, 4)`` array — row 0 at +90 degrees, column 0 at
            -180 degrees. Use :meth:`from_dataset` or :meth:`from_provider` to build one from geodata.
        **kwargs: Forwarded to ``TexturedGlobeGlyph`` — ``tilt_deg`` (default Earth's 23.44), ``n_lon`` /
            ``n_lat`` (mesh resolution, the render-cost driver), ``brightness``, ``sun`` (a world-space
            direction that shades a day/night terminator), ``ambient``, ``fig`` and ``ax``.

    Attributes:
        glyph: The underlying cleopatra ``TexturedGlobeGlyph``.
        fig: The matplotlib figure, once :meth:`draw` or :meth:`animate` has run (``None`` before).
        ax: The matplotlib ``Axes3D``, once :meth:`draw` or :meth:`animate` has run (``None`` before).

    Examples:
        - Build a globe from a two-tone synthetic texture and draw it:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from digitalearth.scene import TexturedGlobe
            >>> texture = np.zeros((90, 180, 3), dtype=np.uint8)
            >>> texture[:45] = (40, 90, 180)
            >>> texture[45:] = (180, 120, 40)
            >>> globe = TexturedGlobe(texture, n_lon=36, n_lat=18)
            >>> fig, ax = globe.draw(spin=45.0)
            >>> ax.name
            '3d'

            ```
        - The north pole sits on the tilted polar axis, not straight up:
            ```python
            >>> import numpy as np
            >>> from digitalearth.scene import TexturedGlobe
            >>> texture = np.zeros((8, 16, 3), dtype=np.uint8)
            >>> globe = TexturedGlobe(texture, tilt_deg=0.0, n_lon=8, n_lat=4)
            >>> np.round(globe.project(0.0, 90.0), 6)
            array([[0., 0., 1.]])

            ```
    """

    def __init__(self, texture: np.ndarray, **kwargs: Any):
        self.glyph = TexturedGlobeGlyph(texture, **kwargs)
        self.fig: Any = None
        self.ax: Any = None

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        *,
        band: int = 1,
        cmap: Any = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        shape: Tuple[int, int] = DEFAULT_TEXTURE_SHAPE,
        **kwargs: Any,
    ) -> "TexturedGlobe":
        """Build a globe from a pyramids ``Dataset``, draped at the raster's true lon/lat position.

        The raster is reprojected to EPSG:4326 by **pyramids** when it is not already there, its band read
        with nodata masked to ``NaN``, colour-mapped through cleopatra's colormap resolver, and pasted into a
        global transparent canvas. Cells the raster does not cover — and its nodata cells — stay fully
        transparent, so a regional dataset shows as a patch on the globe rather than being smeared over it.

        Args:
            dataset: A pyramids ``Dataset``. Must carry a CRS.
            band: 1-based band index to draw.
            cmap: Colormap name or ``Colormap``, resolved by ``cleopatra.styling.colors.resolve_colormap``.
            vmin: Lower bound of the colour scale; defaults to the band's finite minimum.
            vmax: Upper bound of the colour scale; defaults to the band's finite maximum.
            shape: ``(rows, columns)`` of the global canvas. Larger is sharper and slower to build.
            **kwargs: Forwarded to :class:`TexturedGlobe` (and on to the glyph).

        Returns:
            TexturedGlobe: a globe whose texture carries the dataset.

        Raises:
            ValueError: if ``dataset`` has no resolvable CRS (there is then no way to place it on the
                sphere), or if ``shape`` is not two positive integers.
        """
        rows, cols = (int(v) for v in shape)
        if rows < 2 or cols < 2:
            raise ValueError(f"shape must be at least (2, 2), got {shape!r}")

        epsg = getattr(dataset, "epsg", None)
        if epsg is None:
            raise ValueError(
                "from_dataset needs a dataset with a CRS to place it on the globe; this one has none. "
                "Set dataset.crs / dataset.epsg first."
            )
        src = dataset if int(epsg) == _TEXTURE_EPSG else dataset.to_crs(_TEXTURE_EPSG)

        values = read_masked_band(src, band=band)
        rgba = cls._colorize(values, cmap=cmap, vmin=vmin, vmax=vmax)
        texture = cls._paste_global(rgba, np.asarray(src.x, dtype=float), np.asarray(src.y, dtype=float),
                                    rows, cols)
        return cls(texture, **kwargs)

    @classmethod
    def from_provider(cls, provider: Any = "Esri.WorldImagery", **kwargs: Any) -> "TexturedGlobe":
        """Build a globe from a whole-world XYZ tile basemap (delegates to ``cleopatra``'s ``world_texture``).

        Args:
            provider: An ``xyzservices`` provider name or resolved ``TileProvider``. Defaults to
                ``"Esri.WorldImagery"`` — a bulk-permitting imagery provider. Do **not** point this at
                OpenStreetMap's tiles: a whole-world fetch pulls thousands of tiles, which its usage policy
                prohibits.
            **kwargs: Split between ``world_texture`` (``zoom``, ``n_lon``, ``n_lat``, ``cache``,
                ``max_workers``, ``timeout``, ``retries``, ``user_agent``) and :class:`TexturedGlobe`
                (``tilt_deg``, mesh ``n_lon``/``n_lat`` are **not** shared — see below); anything not a
                ``world_texture`` parameter is forwarded to the glyph.

        Returns:
            TexturedGlobe: a globe textured with the provider's imagery.

        Note:
            ``n_lon`` / ``n_lat`` are ambiguous here — they name both the texture grid and the sphere mesh.
            They are read as the **texture** grid (``world_texture``'s meaning); set the mesh resolution with
            ``mesh_n_lon`` / ``mesh_n_lat``.

        Raises:
            ImportError: if cleopatra's ``[tiles]`` extra is unavailable.
        """
        texture_keys = ("zoom", "n_lon", "n_lat", "cache", "max_workers", "timeout", "retries", "user_agent")
        texture_kwargs = {k: kwargs.pop(k) for k in texture_keys if k in kwargs}
        for mesh_key, glyph_key in (("mesh_n_lon", "n_lon"), ("mesh_n_lat", "n_lat")):
            if mesh_key in kwargs:
                kwargs[glyph_key] = kwargs.pop(mesh_key)
        return cls(world_texture(provider, **texture_kwargs), **kwargs)

    # ------------------------------------------------------------------ texture building

    @staticmethod
    def _colorize(values: np.ndarray, *, cmap: Any, vmin: Optional[float],
                  vmax: Optional[float]) -> np.ndarray:
        """Colour-map a NaN-masked 2-D band to an ``(H, W, 4)`` float RGBA array, NaN cells transparent."""
        good = finite(values)
        lo = float(good.min()) if vmin is None and good.size else (0.0 if vmin is None else float(vmin))
        hi = float(good.max()) if vmax is None and good.size else (1.0 if vmax is None else float(vmax))
        if hi <= lo:  # a constant (or empty) band has no range to normalise against
            hi = lo + 1.0
        colormap = resolve_colormap(cmap) or resolve_colormap("viridis")
        rgba = colormap(Normalize(vmin=lo, vmax=hi)(np.asarray(values, dtype=float)))
        rgba = np.asarray(rgba, dtype=float).copy()
        rgba[..., 3] = np.where(np.isfinite(values), rgba[..., 3], 0.0)
        return rgba

    @staticmethod
    def _paste_global(rgba: np.ndarray, src_lon: np.ndarray, src_lat: np.ndarray, rows: int,
                      cols: int) -> np.ndarray:
        """Paste an RGBA lon/lat patch into a transparent global canvas of shape ``(rows, cols, 4)``.

        Samples inversely — every canvas cell asks which source cell covers it — so the result has no holes
        when the source is coarser than the canvas.
        """
        canvas = np.zeros((rows, cols, 4), dtype=float)
        lat_targets, lon_targets = _texture_axes(rows, cols)
        row_idx = _nearest_index(lat_targets, src_lat)
        col_idx = _nearest_index(lon_targets, src_lon)
        valid_rows = np.flatnonzero(row_idx >= 0)
        valid_cols = np.flatnonzero(col_idx >= 0)
        if not (valid_rows.size and valid_cols.size):
            warnings.warn(
                f"the dataset's footprint is smaller than one cell of the {rows}x{cols} globe texture, so "
                "nothing was draped and the globe will render blank. Pass a finer shape= (e.g. "
                "shape=(2880, 5760)) to resolve it.",
                RuntimeWarning,
                stacklevel=3,
            )
            return canvas
        canvas[np.ix_(valid_rows, valid_cols)] = rgba[np.ix_(row_idx[valid_rows], col_idx[valid_cols])]
        return canvas

    # ------------------------------------------------------------------ geometry

    def project(self, lon: Any, lat: Any, *, spin: float = 0.0, altitude: float = 0.0) -> np.ndarray:
        """Map lon/lat degrees onto the drawn sphere, returning world-space ``(N, 3)`` coordinates.

        Pushes the points through the glyph's own ``transform``, so they carry the same spin **and axial
        tilt** as the rendered surface — the reason this is exact rather than an approximation.

        Args:
            lon: Longitude(s) in degrees (scalar or array-like).
            lat: Latitude(s) in degrees, same shape as ``lon``.
            spin: The spin, in degrees, of the frame being drawn — pass the same value given to
                :meth:`draw`.
            altitude: Radial offset from the unit surface. A small positive value (e.g. ``0.01``) lifts an
                overlay clear of the surface so matplotlib does not z-fight it.

        Returns:
            np.ndarray: an ``(N, 3)`` array of world-space coordinates.

        Raises:
            ValueError: if ``lon`` and ``lat`` do not have the same shape.
        """
        lon_arr, lat_arr = np.atleast_1d(np.asarray(lon, dtype=float)), np.atleast_1d(
            np.asarray(lat, dtype=float))
        if lon_arr.shape != lat_arr.shape:
            raise ValueError(f"lon and lat must have the same shape, got {lon_arr.shape} and {lat_arr.shape}")
        body = _lonlat_to_body(lon_arr.ravel(), lat_arr.ravel()) * (1.0 + float(altitude))
        return np.atleast_2d(self.glyph.transform(body, spin=spin))

    def visible(self, world_xyz: np.ndarray) -> np.ndarray:
        """Boolean mask of which world-space points face the camera (i.e. are on the near side).

        A point on a unit sphere is its own outward normal, so it is visible exactly when it points toward
        the camera. Uses the axes' current ``elev``/``azim``, so call it after :meth:`draw`.

        Args:
            world_xyz: An ``(N, 3)`` array of world-space points, as returned by :meth:`project`.

        Returns:
            np.ndarray: an ``(N,)`` boolean mask, ``True`` for the near-side points.

        Raises:
            RuntimeError: if the globe has not been drawn yet (there is no camera to test against).
        """
        if self.ax is None:
            raise RuntimeError("draw() the globe before asking which points are visible")
        view = _view_vector(self.ax.elev, self.ax.azim)
        return np.asarray(np.asarray(world_xyz, dtype=float) @ view > 0.0, dtype=bool)

    # ------------------------------------------------------------------ rendering

    def draw(self, *, spin: float = 0.0, **kwargs: Any) -> Tuple[Any, Any]:
        """Draw the globe, returning the matplotlib ``(fig, ax)`` and recording them on the instance.

        Args:
            spin: Rotation about the polar axis, in degrees.
            **kwargs: Forwarded to the glyph's ``draw`` (``ax``, ``sun``, ``ambient``, ``figsize``,
                ``elev``, ``azim``, ``background``).

        Returns:
            The ``(Figure, Axes3D)`` the globe was drawn on.
        """
        self.fig, self.ax = self.glyph.draw(spin=spin, **kwargs)
        return self.fig, self.ax

    def points(self, data: Any, *, lat: Any = None, spin: float = 0.0, altitude: float = 0.01,
               hide_far_side: bool = True, **kwargs: Any) -> Any:
        """Scatter lon/lat points onto the globe's surface.

        Accepts either a pyramids ``FeatureCollection`` / geopandas ``GeoDataFrame`` of points (its geometry
        is reprojected to lon/lat when needed) or a pair of ``lon``/``lat`` array-likes.

        Args:
            data: A point ``FeatureCollection``/``GeoDataFrame``, or the longitudes when ``lat`` is given.
            lat: Latitudes, when ``data`` holds longitudes.
            spin: The spin of the frame being drawn — pass the same value given to :meth:`draw`.
            altitude: Radial lift above the surface, to avoid z-fighting with it.
            hide_far_side: When True (default), drop the points on the hemisphere facing away from the
                camera, which matplotlib would otherwise draw straight through the sphere.
            **kwargs: Forwarded to ``Axes3D.scatter`` (e.g. ``c``, ``s``, ``marker``, ``color``).

        Returns:
            The ``Path3DCollection`` returned by ``Axes3D.scatter``.

        Raises:
            RuntimeError: if the globe has not been drawn yet.
            ValueError: if ``data`` is a feature collection with no resolvable CRS, or holds no point
                geometry.
        """
        if self.ax is None:
            raise RuntimeError("draw() the globe before adding points to it")
        lon_vals, lat_vals = self._as_lonlat(data, lat)
        world = self.project(lon_vals, lat_vals, spin=spin, altitude=altitude)
        if hide_far_side:
            keep = self.visible(world)
            world = world[keep]
            for key in ("c", "s"):
                value = kwargs.get(key)
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) == keep.size:
                    kwargs[key] = np.asarray(value)[keep]
        return self.ax.scatter(world[:, 0], world[:, 1], world[:, 2], **kwargs)

    @staticmethod
    def _as_lonlat(data: Any, lat: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve ``points``' input into lon/lat degree arrays, reprojecting a feature collection if needed."""
        if lat is not None:
            return np.atleast_1d(np.asarray(data, dtype=float)), np.atleast_1d(np.asarray(lat, dtype=float))
        geometry = getattr(data, "geometry", None)
        if geometry is None:
            raise ValueError("points needs a FeatureCollection/GeoDataFrame, or both lon and lat")
        epsg = source_epsg(data)
        if epsg is None:
            raise ValueError("the feature collection has no CRS, so its points cannot be placed on the globe")
        if int(epsg) != _TEXTURE_EPSG:
            geometry = geometry.to_crs(_TEXTURE_EPSG)
        if not (geometry.geom_type == "Point").all():
            geometry = geometry.centroid
        return geometry.x.to_numpy(), geometry.y.to_numpy()

    def animate(self, ax: Any = None, **kwargs: Any) -> FuncAnimation:
        """Animate a full rotation, returning a matplotlib ``FuncAnimation``.

        When no ``ax`` is given the 3-D axes is created here rather than inside the glyph, so :attr:`fig` and
        :attr:`ax` are known without reaching into the animation's internals — which is what lets
        :meth:`save_gif` and :meth:`stamp` work on an animated globe.

        Args:
            ax: An existing ``Axes3D`` to animate on. A new figure/axes is created when omitted.
            **kwargs: Forwarded to the glyph's ``animate`` (``n_frames``, ``revolutions``, ``start_spin``,
                ``sun``, ``ambient``, ``interval``, plus the render options); ``figsize`` sizes the figure
                created here.

        Returns:
            The ``FuncAnimation`` over the rotation. It is also kept on ``self._animation`` so it is not
            garbage-collected before you save or display it.
        """
        if ax is None:
            figsize = kwargs.pop("figsize", self.glyph.default_options.get("figsize", (6, 6)))
            ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
        anim: FuncAnimation = self.glyph.animate(ax, **kwargs)
        self._animation = anim  # keep a strong reference so it survives until save/display
        self._animation_fps = 1000.0 / float(kwargs.get("interval", _DEFAULT_INTERVAL_MS))
        self.ax, self.fig = ax, ax.get_figure()
        return anim

    def save(self, path: str, **kwargs: Any) -> None:
        """Save the drawn figure to ``path``.

        Args:
            path: Destination file path.
            **kwargs: Forwarded to ``Figure.savefig``.

        Raises:
            RuntimeError: if the globe has not been drawn yet.
        """
        if self.fig is None:
            raise RuntimeError("draw() the globe before saving it")
        self.fig.savefig(path, **kwargs)

    def save_animation(self, path: str, *, fps: Optional[float] = None, gif: Optional[str] = None,
                       **kwargs: Any) -> Any:
        """Save the rotation built by :meth:`animate`, optionally also deriving a GIF from it.

        A textured globe is exactly the case the derive-a-GIF path exists for: every frame is a full 3-D
        surface redraw, so encoding twice off one render is far cheaper than rendering twice.

        Args:
            path: Output path; the extension picks the format.
            fps: Frames per second. Defaults to the animation's own interval.
            gif: Optional second path to derive a GIF at. Requires ``path`` to be a video.
            **kwargs: Forwarded to :func:`digitalearth.animation.save_animation`.

        Returns:
            The written path, or a ``(video, gif)`` pair when ``gif`` was requested.

        Raises:
            RuntimeError: if no animation has been built yet — call :meth:`animate` first.
        """
        anim = getattr(self, "_animation", None)
        if anim is None:
            raise RuntimeError("no animation to save; call animate() first")
        rate = fps if fps is not None else getattr(self, "_animation_fps", None)
        return save_animation(anim, path, fps=rate, gif=gif, **kwargs)

    def stamp(self, mark: Any, **kwargs: Any) -> Any:
        """Stamp a logo / watermark onto the globe's figure (see :meth:`Scene.stamp`).

        Args:
            mark: The mark image — a file path or an ``(H, W, 3)`` / ``(H, W, 4)`` array.
            **kwargs: Forwarded to ``cleopatra.styling.watermark.stamp_mark``.

        Returns:
            The frameless inset ``Axes`` the mark was drawn on.

        Raises:
            RuntimeError: if the globe has not been drawn yet.
        """
        if self.fig is None:
            raise RuntimeError("draw() the globe before stamping it")
        from cleopatra.styling.watermark import stamp_mark

        return stamp_mark(self.fig, mark, **kwargs)


__all__: Sequence[str] = ["TexturedGlobe", "DEFAULT_TEXTURE_SHAPE", "EARTH_TILT_DEG"]
