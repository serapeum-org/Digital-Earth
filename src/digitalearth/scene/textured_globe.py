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
import inspect
import warnings
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cleopatra.basemap.tiles import world_texture
from cleopatra.glyphs.globe.textured_globe_glyph import (
    EARTH_TILT_DEG,
    TexturedGlobeGlyph,
)
from cleopatra.styling.colors import resolve_colormap
from cleopatra.styling.watermark import stamp_mark
from pyramids.dataset import Dataset
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

#: The glyph's own default inter-frame interval, in milliseconds, read from its signature rather than
#: mirrored as a literal: a hand-copied default drifts silently, and the saved frame rate is derived from it.
_DEFAULT_INTERVAL_MS = inspect.signature(TexturedGlobeGlyph.animate).parameters["interval"].default


def _texture_axes(n_lat: int, n_lon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ``(lat, lon)`` degree vectors of a texture of shape ``(n_lat, n_lon)``.

    Endpoint-inclusive, matching the glyph's documented layout: row 0 is +90 and the last row -90; column 0
    is -180 and the last column +180.

    Args:
        n_lat: Number of texture rows.
        n_lon: Number of texture columns.

    Returns:
        A ``(lat, lon)`` pair of 1-D degree vectors, of length ``n_lat`` and ``n_lon`` respectively.
    """
    return np.linspace(90.0, -90.0, n_lat), np.linspace(-180.0, 180.0, n_lon)


def _mesh_sample_indices(texture_shape: Tuple[int, int], n_lon: int,
                        n_lat: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the texture rows and columns the glyph will actually sample for its face colours.

    This mirrors ``TexturedGlobeGlyph``'s own sampling, and mirroring it exactly is the whole point: the glyph
    colours **faces**, not vertices, so it samples the ``(n_lat - 1) x (n_lon - 1)`` grid of face *centres*
    between the mesh edges — not the ``n_lat x n_lon`` edges themselves. Sampling the edges instead lands on
    different texture cells, which is close enough to look right and wrong often enough to matter: measured
    against real renders it misclassified about one small raster in eight, in both directions.

    Args:
        texture_shape: The texture's ``(height, width)`` in cells.
        n_lon: Number of mesh longitude edges.
        n_lat: Number of mesh latitude edges.

    Returns:
        A ``(rows, cols)`` pair of integer index arrays, of length ``n_lat - 1`` and ``n_lon - 1``.
    """
    height, width = texture_shape
    lat_edges = np.linspace(90.0, -90.0, n_lat)
    lon_edges = np.linspace(-180.0, 180.0, n_lon)
    lat_centres = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centres = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    rows = np.clip(np.round((90.0 - lat_centres) / 180.0 * (height - 1)).astype(int), 0, height - 1)
    cols = np.clip(np.round((lon_centres + 180.0) / 360.0 * (width - 1)).astype(int), 0, width - 1)
    return rows, cols


def _as_byte_texture(rgba: np.ndarray) -> np.ndarray:
    """Convert a float RGBA canvas in ``[0, 1]`` to ``uint8``, which is what a texture actually needs.

    A whole-globe canvas at the default 0.125-degree resolution is 1440 x 2880 x 4. Held as ``float64`` that
    is 133 MB, and the glyph then keeps its own normalised copy, so a single globe cost around 265 MB at peak.
    Colour is 8-bit on the way to the screen regardless, so the extra 56 bits per channel buy nothing.

    Args:
        rgba: A float RGBA array with values in ``[0, 1]``.

    Returns:
        The same image as ``uint8`` in ``[0, 255]``, one quarter the size.
    """
    return np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)


def _lonlat_to_body(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convert lon/lat degrees to the glyph's body-frame unit-sphere ``(N, 3)`` coordinates.

    Mirrors the mesh the glyph builds — ``x = cos(lat)cos(lon)``, ``y = cos(lat)sin(lon)``, ``z = sin(lat)``
    — so a point at ``(lon, lat)`` lands on the texture cell showing that longitude and latitude.

    Args:
        lon: Longitudes in degrees.
        lat: Latitudes in degrees, the same shape as ``lon``.

    Returns:
        An ``(N, 3)`` array of unit-sphere coordinates in the glyph's body frame (``+z`` at the north pole).
    """
    lon_rad, lat_rad = np.deg2rad(np.asarray(lon, dtype=float)), np.deg2rad(np.asarray(lat, dtype=float))
    return np.stack(
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)],
        axis=-1,
    )


#: ``Axes3D.scatter`` arguments that may carry one value per point, and so have to be culled alongside the
#: points themselves when the far side is hidden. Anything left whole would silently shift onto the wrong
#: points (``linewidths``) or make matplotlib reject the call outright (``color``, ``c``). ``marker`` and
#: ``hatch`` are deliberately absent: ``scatter`` takes a single value for each.
_PER_POINT_SCATTER_KEYS = (
    "c", "s", "color", "facecolor", "facecolors", "edgecolor", "edgecolors",
    "linewidth", "linewidths", "linestyle", "linestyles", "alpha",
)

#: Colour arguments where a bare 3- or 4-element sequence of numbers is one RGB(A) value rather than one
#: value per point. ``c`` is deliberately excluded: matplotlib gives value-mapping precedence for a sequence
#: whose length matches the point count, so a length-matching ``c`` is per-point data and must be culled.
_RGBA_EXEMPT_KEYS = frozenset({"color", "facecolor", "facecolors", "edgecolor", "edgecolors"})


def _is_rgba_literal(key: str, value: Any) -> bool:
    """Whether ``value`` is a single RGB(A) colour rather than one value per point.

    Applies only to the colour arguments in :data:`_RGBA_EXEMPT_KEYS`, and only to a 3- or 4-element sequence
    of plain numbers. Container type is not the test — ``color=[1, 0, 0, 1]`` is as much a single red as
    ``color=(1, 0, 0, 1)``, and culling it would leave ``[1, 0, 1]``, which renders as magenta with no error.

    Args:
        key: The scatter keyword the value was passed under.
        value: The value the caller supplied.

    Returns:
        ``True`` when the value should be passed through whole rather than culled.
    """
    if key not in _RGBA_EXEMPT_KEYS or len(value) not in (3, 4):
        return False
    return all(isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) for v in value)


def _cull_per_point(kwargs: dict, keep: np.ndarray) -> dict:
    """Drop the hidden points' entries from every per-point scatter argument.

    Only a sequence whose length matches the point count is treated as per-point; a scalar (``color="red"``,
    ``s=30``) applies to every point and is passed through untouched. A bare RGB(A) colour is passed through
    as well — see :func:`_is_rgba_literal` for which arguments that covers and why ``c`` is not one of them.

    Any sized sequence is accepted, including a pandas ``Series``, which is what a caller naturally passes
    when the colours come from a dataframe column.

    Args:
        kwargs: The scatter keyword arguments as the caller supplied them.
        keep: Boolean mask of the points that survive the far-side cull.

    Returns:
        A new mapping with every per-point sequence reduced to the kept points.
    """
    culled = dict(kwargs)
    for key in _PER_POINT_SCATTER_KEYS:
        value = culled.get(key)
        # `np.ndim` is 0 for a scalar and for anything numpy cannot see a shape in (a set, a generator, an
        # arbitrary object), so everything past this point is a real sequence with a length.
        if value is None or isinstance(value, str) or np.ndim(value) == 0:
            continue
        if len(value) != keep.size or _is_rgba_literal(key, value):
            continue
        culled[key] = np.asarray(value)[keep]
    return culled


def _view_vector(elev: float, azim: float) -> np.ndarray:
    """Unit vector pointing from the sphere's centre toward the camera at ``elev``/``azim`` degrees.

    Args:
        elev: The axes' elevation angle, in degrees above the equatorial plane.
        azim: The axes' azimuth angle, in degrees about the polar axis.

    Returns:
        A unit ``(3,)`` array pointing from the origin toward the camera.
    """
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
        """Wrap an equirectangular texture in a cleopatra globe glyph.

        Args:
            texture: An equirectangular ``(H, W, 3)`` or ``(H, W, 4)`` array — row 0 at +90 degrees, column
                0 at -180 degrees.
            **kwargs: Forwarded to ``TexturedGlobeGlyph`` (``tilt_deg``, ``n_lon``, ``n_lat``,
                ``brightness``, ``sun``, ``ambient``, ``fig``, ``ax``).

        Raises:
            ValueError: If ``texture`` is not a 3-channel or 4-channel 2-D image, or if a lighting argument
                is out of contract (a zero-length ``sun``, an ``ambient`` outside ``[0, 1]``).
        """
        self.glyph = TexturedGlobeGlyph(texture, **kwargs)
        self.fig: Any = None
        self.ax: Any = None
        # Declared up front rather than sprung into existence by animate(), so every reader can see the
        # instance's full state and the accessors need no getattr() guards.
        self._animation: Optional[FuncAnimation] = None
        self._animation_fps: Optional[float] = None
        #: The spin the globe was last drawn at, so an overlay defaults to the surface it can see.
        self._spin: float = 0.0
        #: Whether :attr:`fig` is ours to close. A caller-supplied axes belongs to the caller.
        self._owns_fig: bool = False

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
        resampling: str = "nearest",
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
            resampling: How pyramids resamples the raster onto that grid — ``"nearest"`` (the default, which
                preserves categorical values), ``"bilinear"``, ``"cubic"``, ``"average"``, ``"mode"``, and
                the rest of ``Dataset.align``'s methods.
            **kwargs: Forwarded to :class:`TexturedGlobe` (and on to the glyph).

        Returns:
            TexturedGlobe: a globe whose texture carries the dataset.

        Raises:
            ValueError: if ``dataset`` has no resolvable CRS (there is then no way to place it on the
                sphere), if ``shape`` is not two positive integers, or if the colour bounds leave no range.

        Warns:
            RuntimeWarning: If nothing survives the resample onto the texture, or if what does is finer than
                the sphere mesh and so would not be drawn — see :meth:`_warn_if_finer_than_the_mesh`.

        Examples:
            - Drape a whole-globe raster and inspect the texture it produced:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(8, dtype="float32").reshape(2, 4)
                >>> ds = Dataset.create_from_array(arr=arr, geo=(-180.0, 90.0, 0.0, 90.0, 0.0, -90.0),
                ...                                epsg=4326)
                >>> globe = TexturedGlobe.from_dataset(ds, shape=(90, 180))
                >>> globe.glyph.texture.shape
                (90, 180, 4)
                >>> bool((globe.glyph.texture[..., 3] > 0).mean() > 0.98)
                True

                ```
            - A raster covering only part of the world leaves the rest transparent:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> from pyramids.dataset import Dataset
                >>> arr = np.ones((2, 2), dtype="float32")
                >>> ds = Dataset.create_from_array(arr=arr, geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0),
                ...                                epsg=4326)
                >>> globe = TexturedGlobe.from_dataset(ds, shape=(90, 180))
                >>> opaque = globe.glyph.texture[..., 3] > 0
                >>> bool(opaque.any()), bool(opaque.all())
                (True, False)

                ```
        """
        try:
            rows, cols = (int(v) for v in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"shape must be a (rows, columns) pair of integers, got {shape!r}") from exc
        if rows < 2 or cols < 2:
            raise ValueError(f"shape must be at least (2, 2), got {shape!r}")

        # `epsg` is None both for a raster with no CRS and for one whose CRS carries no EPSG authority code
        # (geostationary, Mollweide, a bare ESRI code). Only the first is unplaceable; the second reprojects
        # fine, so the presence of a CRS — not of a code — is what decides.
        epsg = getattr(dataset, "epsg", None)
        if epsg is None and not getattr(dataset, "crs", None):
            raise ValueError(
                "from_dataset needs a dataset with a CRS to place it on the globe; this one has none. "
                "Set dataset.crs / dataset.epsg first."
            )
        # pyramids owns the regrid: align() reprojects to the template's CRS and resamples onto its grid,
        # which also settles the longitude frame (0-360 or antimeridian-crossing) and any non-uniform or
        # single-row source. Doing it here would be re-implementing GIS.
        aligned = dataset.align(cls._global_template(rows, cols), method=resampling)

        values = read_masked_band(aligned, band=band)
        rgba = cls._colorize(values, cmap=cmap, vmin=vmin, vmax=vmax)
        texture = _as_byte_texture(rgba)
        if not (texture[..., 3] > 0).any():
            warnings.warn(
                f"the dataset did not survive the resample onto the {rows}x{cols} globe texture, so nothing "
                "was draped and the globe will render blank. Pass a finer shape= (e.g. shape=(2880, 5760)) "
                "to resolve it.",
                RuntimeWarning,
                stacklevel=2,
            )
        globe = cls(texture, **kwargs)
        globe._warn_if_finer_than_the_mesh(texture)
        return globe

    def _warn_if_finer_than_the_mesh(self, texture: np.ndarray) -> None:
        """Warn when the draped data is too small to survive the sphere mesh.

        A fine texture is not enough to make small data visible. The glyph point-samples the texture down to
        its ``n_lon`` x ``n_lat`` mesh, so detail narrower than one mesh face falls between the samples and
        disappears — however many texture cells it covers. The default mesh is 180x90, roughly 2-degree faces,
        while the default texture is 0.125 degrees, so a raster can occupy hundreds of texture cells and still
        paint nothing.

        Checking the mesh is the only check that predicts what a reader will actually see; the texture-grid
        check alone reports success on a globe that renders blank.

        Args:
            texture: The global RGBA canvas about to be handed to the glyph.

        Warns:
            RuntimeWarning: If the opaque part of ``texture`` covers no mesh sample.
        """
        opaque = texture[..., 3] > 0
        if not opaque.any():
            return  # from_dataset has already warned that nothing was draped at all
        rows, cols = _mesh_sample_indices(texture.shape[:2], self.glyph.n_lon, self.glyph.n_lat)
        if not opaque[np.ix_(rows, cols)].any():
            warnings.warn(
                f"the draped data is finer than the {self.glyph.n_lon}x{self.glyph.n_lat} sphere mesh, so it "
                "falls between the mesh samples and the globe will render without it. Raise the mesh "
                "(n_lon=/n_lat=) until a mesh face fits inside the data's footprint.",
                RuntimeWarning,
                stacklevel=3,
            )

    @classmethod
    def from_provider(cls, provider: Any = "Esri.WorldImagery", **kwargs: Any) -> "TexturedGlobe":
        """Build a globe from a whole-world XYZ tile basemap (delegates to ``cleopatra``'s ``world_texture``).

        Args:
            provider: An ``xyzservices`` provider name or resolved ``TileProvider``. Defaults to
                ``"Esri.WorldImagery"`` — a bulk-permitting imagery provider. Do **not** point this at
                OpenStreetMap's tiles: a whole-world fetch pulls thousands of tiles, which its usage policy
                prohibits.
            **kwargs: The fetch options ``zoom``, ``cache``, ``max_workers``, ``timeout``, ``retries`` and
                ``user_agent`` go to ``world_texture``; ``texture_n_lon`` / ``texture_n_lat`` size the
                fetched texture grid. Everything else — ``n_lon``, ``n_lat``, ``tilt_deg``, ``sun``,
                ``ambient`` — is forwarded to the glyph, exactly as in the other constructors.

        Returns:
            TexturedGlobe: a globe textured with the provider's imagery.

        Note:
            ``n_lon`` / ``n_lat`` always mean the **sphere mesh**, here as everywhere else in this class.
            The texture grid — which ``world_texture`` also calls ``n_lon`` / ``n_lat`` — is set with
            ``texture_n_lon`` / ``texture_n_lat``, so the same keyword never means two different things.

        Raises:
            ValueError: if ``provider`` is not a known ``xyzservices`` provider.

        Examples:
            - Build a photographic Earth and spin it (needs the network on first use; the texture is then
              cached on disk):
                ```python
                >>> from digitalearth.scene import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe.from_provider("Esri.WorldImagery", zoom=3)   # doctest: +SKIP
                >>> globe.glyph.texture.shape                             # doctest: +SKIP
                (1440, 2880, 4)
                >>> fig, ax = globe.draw(spin=30.0, sun=(1.0, 0.3, 0.2))  # doctest: +SKIP

                ```
            - Size the fetched texture and the sphere mesh independently:
                ```python
                >>> from digitalearth.scene import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe.from_provider(                  # doctest: +SKIP
                ...     "Esri.WorldImagery", zoom=2, texture_n_lon=720, texture_n_lat=360,
                ...     n_lon=60, n_lat=30,
                ... )
                >>> globe.glyph.n_lon, globe.glyph.n_lat                  # doctest: +SKIP
                (60, 30)

                ```
        """
        texture_keys = ("zoom", "cache", "max_workers", "timeout", "retries", "user_agent")
        texture_kwargs = {k: kwargs.pop(k) for k in texture_keys if k in kwargs}
        for shape_key, fetch_key in (("texture_n_lon", "n_lon"), ("texture_n_lat", "n_lat")):
            if shape_key in kwargs:
                texture_kwargs[fetch_key] = kwargs.pop(shape_key)
        return cls(world_texture(provider, **texture_kwargs), **kwargs)

    # ------------------------------------------------------------------ texture building

    @staticmethod
    def _validate_colour_bounds(good: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> None:
        """Reject bound combinations that leave nothing to colour.

        Separated from resolving the bounds because it answers a different question: not "what scale do we
        use" but "is what the caller asked for coherent at all". Quietly substituting a workable range
        instead would return a plausible image drawn from a scale nobody chose.

        Args:
            good: The band's finite values, already stripped of nodata. May be empty.
            vmin: Lower bound, or ``None``.
            vmax: Upper bound, or ``None``.

        Raises:
            ValueError: If both bounds are given and ``vmax <= vmin``, or if a single given bound sits
                outside the data on the wrong side.
        """
        if vmin is not None and vmax is not None and float(vmax) <= float(vmin):
            raise ValueError(f"vmax must be greater than vmin, got vmin={vmin!r}, vmax={vmax!r}")
        if not good.size:
            return
        if vmax is None and vmin is not None and float(vmin) >= float(good.max()):
            raise ValueError(
                f"vmin={vmin!r} is at or above the band's maximum ({float(good.max())!r}), so there is no "
                "range to colour. Lower vmin, or pass vmax as well."
            )
        if vmin is None and vmax is not None and float(vmax) <= float(good.min()):
            raise ValueError(
                f"vmax={vmax!r} is at or below the band's minimum ({float(good.min())!r}), so there is no "
                "range to colour. Raise vmax, or pass vmin as well."
            )

    @classmethod
    def _resolve_colour_bounds(cls, good: np.ndarray, vmin: Optional[float],
                               vmax: Optional[float]) -> Tuple[float, float]:
        """Settle the colour scale's ``(lo, hi)`` from the caller's bounds and the band's finite values.

        A bound the caller gives always wins; a bound they leave out comes from the data, or from a unit
        fallback when there is no finite data to take it from.

        Args:
            good: The band's finite values, already stripped of nodata. May be empty.
            vmin: Lower bound, or ``None`` to take the band's minimum.
            vmax: Upper bound, or ``None`` to take the band's maximum.

        Returns:
            The ``(lo, hi)`` pair to normalise against, with ``hi > lo`` guaranteed.

        Raises:
            ValueError: If the bounds leave no range — see :meth:`_validate_colour_bounds`.
        """
        cls._validate_colour_bounds(good, vmin, vmax)
        lo = float(vmin) if vmin is not None else (float(good.min()) if good.size else 0.0)
        hi = float(vmax) if vmax is not None else (float(good.max()) if good.size else 1.0)
        if hi <= lo:  # a constant band has no range to normalise against
            hi = lo + 1.0
        return lo, hi

    @classmethod
    def _colorize(cls, values: np.ndarray, *, cmap: Any, vmin: Optional[float],
                  vmax: Optional[float]) -> np.ndarray:
        """Colour-map a NaN-masked 2-D band to an ``(H, W, 4)`` float RGBA array, NaN cells transparent.

        Args:
            values: The band, with nodata already masked to ``NaN``.
            cmap: Colormap name or ``Colormap``, resolved by cleopatra; ``None`` falls back to viridis.
            vmin: Lower bound of the colour scale, or ``None`` to take the band's finite minimum.
            vmax: Upper bound of the colour scale, or ``None`` to take the band's finite maximum.

        Returns:
            An ``(H, W, 4)`` float RGBA array in ``[0, 1]``, with alpha ``0`` wherever ``values`` is not
            finite.

        Raises:
            ValueError: If the colour bounds leave no range — see :meth:`_resolve_colour_bounds`.

        Warns:
            RuntimeWarning: If no cell of the band is finite, so the result is entirely transparent.
        """
        good = finite(values)
        if not good.size:
            warnings.warn(
                "every cell of this band is nodata or non-finite, so the globe will be fully transparent.",
                RuntimeWarning,
                stacklevel=4,
            )
        lo, hi = cls._resolve_colour_bounds(good, vmin, vmax)
        colormap = resolve_colormap(cmap)
        if colormap is None:  # resolve_colormap returns None only for cmap=None
            colormap = resolve_colormap("viridis")
        rgba = np.asarray(colormap(Normalize(vmin=lo, vmax=hi)(np.asarray(values, dtype=float))),
                          dtype=float).copy()
        rgba[..., 3] = np.where(np.isfinite(values), rgba[..., 3], 0.0)
        return rgba

    @staticmethod
    def _global_template(rows: int, cols: int) -> Dataset:
        """Build the empty global EPSG:4326 raster that a dataset is aligned onto.

        The grid is chosen so its **cell centres** land exactly on the latitudes and longitudes the glyph
        samples — ``linspace(90, -90, rows)`` and ``linspace(-180, 180, cols)`` — rather than on the
        half-cell-offset grid a naive ``(-180, 90)`` corner would give. Without that the whole texture sits
        half a cell off what is drawn.

        Args:
            rows: Number of texture rows.
            cols: Number of texture columns.

        Returns:
            Dataset: an empty raster carrying only the target grid; ``align`` reads its spatial properties.
        """
        d_lat, d_lon = 180.0 / (rows - 1), 360.0 / (cols - 1)
        return Dataset.create_from_array(
            arr=np.zeros((rows, cols), dtype="float32"),
            geo=(-180.0 - d_lon / 2, d_lon, 0.0, 90.0 + d_lat / 2, 0.0, -d_lat),
            epsg=_TEXTURE_EPSG,
        )

    # ------------------------------------------------------------------ geometry

    def project(self, lon: Any, lat: Any, *, spin: Optional[float] = None,
                altitude: float = 0.0) -> np.ndarray:
        """Map lon/lat degrees onto the drawn sphere, returning world-space ``(N, 3)`` coordinates.

        Pushes the points through the glyph's own ``transform``, so they carry the same spin **and axial
        tilt** as the rendered surface — the reason this is exact rather than an approximation.

        Args:
            lon: Longitude(s) in degrees (scalar or array-like).
            lat: Latitude(s) in degrees, same shape as ``lon``.
            spin: The spin, in degrees, to place the points at. Defaults to the spin the globe was last
                drawn at, so an overlay lands on the surface the reader can actually see.
            altitude: Radial offset from the unit surface. A small positive value (e.g. ``0.01``) lifts an
                overlay clear of the surface so matplotlib does not z-fight it.

        Returns:
            np.ndarray: an ``(N, 3)`` array of world-space coordinates.

        Raises:
            ValueError: if ``lon`` and ``lat`` do not have the same shape.

        Examples:
            - With no tilt the body frame is the world frame, so the cardinal points are exact:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> np.round(globe.project(0.0, 0.0), 6)
                array([[1., 0., 0.]])
                >>> np.round(globe.project(90.0, 0.0), 6)
                array([[0., 1., 0.]])
                >>> np.round(globe.project(0.0, 90.0), 6)
                array([[0., 0., 1.]])

                ```
            - Points land on the unit sphere; ``altitude`` lifts an overlay clear of the surface:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> world = globe.project([0.0, 45.0], [10.0, -20.0], spin=30.0)
                >>> world.shape
                (2, 3)
                >>> [round(float(r), 6) for r in np.linalg.norm(world, axis=1)]
                [1.0, 1.0]
                >>> round(float(np.linalg.norm(globe.project(0.0, 0.0, altitude=0.05))), 6)
                1.05

                ```
        """
        spin = self._spin if spin is None else spin
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

        A point sitting exactly on the limb is a tie, and floating point decides it: at ``lon=90`` with the
        camera at ``azim=0``, ``cos(90°)`` evaluates to ``6.1e-17`` rather than ``0``, so the point reads as
        (barely) visible. Do not rely on the classification of points within rounding distance of the limb.

        Args:
            world_xyz: An ``(N, 3)`` array of world-space points, as returned by :meth:`project`.

        Returns:
            np.ndarray: an ``(N,)`` boolean mask, ``True`` for the near-side points.

        Raises:
            RuntimeError: if the globe has not been drawn yet (there is no camera to test against).

        Examples:
            - The hemisphere facing the camera is visible; the one behind it is not:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(elev=0.0, azim=0.0)
                >>> globe.visible(globe.project(0.0, 0.0)).tolist()
                [True]
                >>> globe.visible(globe.project(180.0, 0.0)).tolist()
                [False]

                ```
            - Use it to label only the points a reader can actually see:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(elev=0.0, azim=0.0)
                >>> lon, lat = [0.0, 45.0, 180.0, 225.0], [0.0, 0.0, 0.0, 0.0]
                >>> int(globe.visible(globe.project(lon, lat)).sum())
                2

                ```
        """
        if self.ax is None:
            raise RuntimeError("draw() the globe before asking which points are visible")
        view = _view_vector(self.ax.elev, self.ax.azim)
        points = np.atleast_2d(np.asarray(world_xyz, dtype=float))
        return np.asarray(points @ view > 0.0, dtype=bool).ravel()

    # ------------------------------------------------------------------ rendering

    def draw(self, *, spin: float = 0.0, **kwargs: Any) -> Tuple[Any, Any]:
        """Draw the globe, returning the matplotlib ``(fig, ax)`` and recording them on the instance.

        Args:
            spin: Rotation about the polar axis, in degrees.
            **kwargs: Forwarded to the glyph's ``draw`` (``ax``, ``sun``, ``ambient``, ``figsize``,
                ``elev``, ``azim``, ``background``).

        Returns:
            The ``(Figure, Axes3D)`` the globe was drawn on.

        Examples:
            - Draw the globe and keep the axes for further decoration:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(spin=45.0)
                >>> ax.name
                '3d'
                >>> globe.ax is ax
                True

                ```
            - Draw several spins onto axes you own, to build a contact sheet:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> import matplotlib.pyplot as plt
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
                >>> for spin, sub in zip([0.0, 120.0, 240.0], axes):
                ...     _ = globe.draw(ax=sub, spin=spin)
                >>> len(fig.axes)
                3

                ```
        """
        supplied = kwargs.get("ax") is not None
        if self._owns_fig and self.fig is not None and not supplied:
            plt.close(self.fig)  # do not leak the figure this call is about to replace
        self.fig, self.ax = self.glyph.draw(spin=spin, **kwargs)
        self._owns_fig = not supplied
        self._spin = float(spin)
        return self.fig, self.ax

    def points(self, data: Any, *, lat: Any = None, spin: Optional[float] = None,
               altitude: float = 0.01,
               hide_far_side: bool = True, **kwargs: Any) -> Any:
        """Scatter lon/lat points onto the globe's surface.

        Accepts either a pyramids ``FeatureCollection`` / geopandas ``GeoDataFrame`` of points (its geometry
        is reprojected to lon/lat when needed) or a pair of ``lon``/``lat`` array-likes.

        Args:
            data: A point ``FeatureCollection``/``GeoDataFrame``, or the longitudes when ``lat`` is given.
            lat: Latitudes, when ``data`` holds longitudes.
            spin: The spin, in degrees, to place the points at. Defaults to the spin the globe was last
                drawn at.
            altitude: Radial lift above the surface, to avoid z-fighting with it.
            hide_far_side: When True (default), drop the points on the hemisphere facing away from the
                camera, which matplotlib would otherwise draw straight through the sphere.
            **kwargs: Forwarded to ``Axes3D.scatter`` (e.g. ``c``, ``s``, ``marker``, ``color``).

        Returns:
            The ``Path3DCollection`` returned by ``Axes3D.scatter``.

        Raises:
            RuntimeError: if the globe has not been drawn yet.
            ValueError: if ``data`` is neither a feature collection nor a pair of lon/lat arrays, or is a
                feature collection carrying no CRS.

        Note:
            Non-point geometry (polygons, lines) is reduced to its centroids rather than rejected, so a
            polygon layer can be marked on the globe without converting it first.

        Examples:
            - Scatter lon/lat points; the ones behind the globe are dropped:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(elev=0.0, azim=0.0)
                >>> scatter = globe.points([0.0, 180.0], lat=[0.0, 0.0])
                >>> len(scatter.get_offsets())
                1

                ```
            - Keep the far side when you want the full set drawn:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(elev=0.0, azim=0.0)
                >>> scatter = globe.points([0.0, 180.0], lat=[0.0, 0.0], hide_far_side=False)
                >>> len(scatter.get_offsets())
                2

                ```
        """
        if self.ax is None:
            raise RuntimeError("draw() the globe before adding points to it")
        lon_vals, lat_vals = self._as_lonlat(data, lat)
        world = self.project(lon_vals, lat_vals, spin=spin, altitude=altitude)
        if hide_far_side:
            keep = self.visible(world)
            world = world[keep]
            kwargs = _cull_per_point(kwargs, keep)
        return self.ax.scatter(world[:, 0], world[:, 1], world[:, 2], **kwargs)

    @staticmethod
    def _as_lonlat(data: Any, lat: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve ``points``' input into lon/lat degree arrays, reprojecting a feature collection if needed.

        Args:
            data: A point ``FeatureCollection`` / ``GeoDataFrame``, or the longitudes when ``lat`` is given.
            lat: Latitudes, when ``data`` holds longitudes; ``None`` to read the geometry from ``data``.

        Returns:
            A ``(lon, lat)`` pair of 1-D degree arrays.

        Raises:
            ValueError: If ``data`` has no geometry and no ``lat`` was given, or if a feature collection
                carries no CRS (its coordinates then cannot be placed on the sphere).
        """
        if lat is not None:
            return np.atleast_1d(np.asarray(data, dtype=float)), np.atleast_1d(np.asarray(lat, dtype=float))
        geometry = getattr(data, "geometry", None)
        if geometry is None:
            raise ValueError("points needs a FeatureCollection/GeoDataFrame, or both lon and lat")
        # As on the raster path, a missing EPSG code is not the same as a missing CRS: a projection with no
        # authority code still reprojects, so test the CRS itself.
        epsg = source_epsg(data)
        if epsg is None and getattr(data, "crs", None) is None:
            raise ValueError("the feature collection has no CRS, so its points cannot be placed on the globe")
        # Reduce to centroids *before* reprojecting: a centroid taken on lon/lat degrees is not the centroid
        # of the shape on the ground, and geopandas warns about exactly that.
        if not (geometry.geom_type == "Point").all():
            geometry = geometry.centroid
        if epsg != _TEXTURE_EPSG:
            geometry = geometry.to_crs(_TEXTURE_EPSG)
        return geometry.x.to_numpy(), geometry.y.to_numpy()

    def animate(self, ax: Any = None, **kwargs: Any) -> FuncAnimation:
        """Animate a full rotation, returning a matplotlib ``FuncAnimation``.

        When no ``ax`` is given the 3-D axes is created here rather than inside the glyph, so :attr:`fig` and
        :attr:`ax` are known without reaching into the animation's internals — which is what lets
        :meth:`save_gif` and :meth:`stamp` work on an animated globe.

        Args:
            ax: An existing ``Axes3D`` to animate on. A new figure/axes is created when omitted.
            **kwargs: Forwarded to the glyph's ``animate`` (``n_frames``, ``revolutions``, ``start_spin``,
                ``sun``, ``ambient``, ``interval``, plus the render options). ``figsize`` sizes the figure
                created here, and is ignored when ``ax`` is given — that figure already exists.

        Returns:
            The ``FuncAnimation`` over the rotation. It is also kept on ``self._animation`` so it is not
            garbage-collected before you save or display it.

        Raises:
            ValueError: If ``interval`` is not a positive number of milliseconds.

        Examples:
            - Animate a rotation; the figure and axes are recorded for saving or stamping afterwards:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> anim = globe.animate(n_frames=4, interval=100)
                >>> globe.ax.name
                '3d'
                >>> globe.fig is globe.ax.get_figure()
                True

                ```
            - Animate onto an axes you already own:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> import matplotlib.pyplot as plt
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> ax = plt.figure().add_subplot(projection="3d")
                >>> anim = globe.animate(ax, n_frames=2, interval=200)
                >>> globe.ax is ax
                True

                ```
        """
        interval = float(kwargs.get("interval", _DEFAULT_INTERVAL_MS))
        if interval <= 0:
            raise ValueError(f"interval must be a positive number of milliseconds, got {interval!r}")
        supplied = ax is not None
        if ax is None:
            figsize = kwargs.pop("figsize", self.glyph.default_options.get("figsize", (6, 6)))
            if self._owns_fig and self.fig is not None:
                plt.close(self.fig)
            ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
        anim: FuncAnimation = self.glyph.animate(ax, **kwargs)
        self._owns_fig = not supplied
        self._animation = anim  # keep a strong reference so it survives until save/display
        self._animation_fps = 1000.0 / interval
        self.ax, self.fig = ax, ax.get_figure()
        return anim

    def save(self, path: str, **kwargs: Any) -> None:
        """Save the drawn figure to ``path``.

        Args:
            path: Destination file path.
            **kwargs: Forwarded to ``Figure.savefig``.

        Raises:
            RuntimeError: if the globe has not been drawn yet.

        Examples:
            - Draw, then write the figure to disk:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> import tempfile
                >>> from pathlib import Path
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw()
                >>> out = Path(tempfile.mkdtemp()) / "globe.png"
                >>> globe.save(str(out))
                >>> out.exists() and out.stat().st_size > 0
                True

                ```
            - Saving before drawing is refused, rather than writing an empty figure:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> globe.save("globe.png")
                Traceback (most recent call last):
                    ...
                RuntimeError: draw() the globe before saving it

                ```
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

        Examples:
            - Animate, then write the rotation straight to a GIF:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> import tempfile
                >>> from pathlib import Path
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> anim = globe.animate(n_frames=2, interval=200)
                >>> out = Path(tempfile.mkdtemp()) / "globe.gif"
                >>> written = globe.save_animation(str(out))
                >>> Path(written).exists()
                True

                ```
            - Render once and deliver both a video and a GIF derived from it:
                ```python
                >>> video, gif = globe.save_animation("globe.mp4", gif="globe.gif")  # doctest: +SKIP

                ```
        """
        if self._animation is None:
            raise RuntimeError("no animation to save; call animate() first")
        anim = self._animation
        rate = fps if fps is not None else self._animation_fps
        return save_animation(anim, path, fps=rate, gif=gif, **kwargs)

    def close(self) -> None:
        """Close the globe's figure and drop the animation reference.

        Every :meth:`draw` or :meth:`animate` that is not handed an existing axes creates a pyplot figure,
        and pyplot keeps a reference to it forever. A loop that builds many globes therefore grows without
        bound and eventually trips matplotlib's open-figure warning. Closing releases both the figure and the
        retained animation.

        Only a figure this globe created is closed. An axes handed in by the caller — to ``__init__``,
        ``draw(ax=...)`` or ``animate(ax=...)`` — belongs to the caller, who may well have other subplots on
        it, so its figure is left alone.

        Safe to call more than once, and on a globe that was never drawn.

        Examples:
            - Close a drawn globe and see the figure released:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import matplotlib.pyplot as plt
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> plt.close("all")
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw()
                >>> len(plt.get_fignums())
                1
                >>> globe.close()
                >>> len(plt.get_fignums())
                0

                ```
        """
        if self.fig is not None and self._owns_fig:
            plt.close(self.fig)
        self._owns_fig = False
        self.fig = None
        self.ax = None
        self._animation = None
        self._animation_fps = None

    def __enter__(self) -> "TexturedGlobe":
        """Enter the runtime context, returning the globe so ``with TexturedGlobe(...) as g:`` binds it.

        Returns:
            This globe.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the figure on exit so a long run of globes stays memory-bounded.

        Mirrors :class:`~digitalearth.scene.scene.Scene`: the figure is closed whether or not the body
        raised, and any exception propagates (``__exit__`` returns ``False``), so ``with`` never swallows an
        error.

        Args:
            exc_type: Exception class raised in the body, if any.
            exc: The exception instance, if any.
            tb: The traceback, if any.

        Returns:
            ``False``, so an exception raised inside the block is re-raised.
        """
        self.close()
        return False

    def stamp(self, mark: Any, **kwargs: Any) -> Any:
        """Stamp a logo / watermark onto the globe's figure.

        The same figure-level mark as :meth:`digitalearth.scene.scene.Scene.stamp`, and it carries the same
        two caveats: stamp **last**, because the mark is baked from the figure's current size, and note that
        a ``bbox_inches="tight"`` save crops surrounding whitespace and so shifts the mark's margin.

        Args:
            mark: The mark image — a file path or an ``(H, W, 3)`` / ``(H, W, 4)`` array.
            **kwargs: Forwarded to ``cleopatra.styling.watermark.stamp_mark``.

        Returns:
            The frameless inset ``Axes`` the mark was drawn on.

        Raises:
            RuntimeError: if the globe has not been drawn yet.

        Examples:
            - Stamp a mark onto the drawn globe; it arrives as an extra axes on the figure:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw()
                >>> mark = np.full((8, 16, 4), 255, dtype=np.uint8)
                >>> mark_ax = globe.stamp(mark, frac=0.2, shadow=False)
                >>> len(fig.axes)
                2
                >>> round(float(mark_ax.get_position().bounds[2]), 3)
                0.2

                ```
            - Stamping before drawing is refused:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.scene import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                >>> globe.stamp(np.full((8, 16, 4), 255, dtype=np.uint8))
                Traceback (most recent call last):
                    ...
                RuntimeError: draw() the globe before stamping it

                ```
        """
        if self.fig is None:
            raise RuntimeError("draw() the globe before stamping it")
        return stamp_mark(self.fig, mark, **kwargs)


#: ``EARTH_TILT_DEG`` is cleopatra's constant, re-exported here so a caller setting ``tilt_deg``
#: does not have to import from the glyph module directly.
__all__: List[str] = ["TexturedGlobe", "DEFAULT_TEXTURE_SHAPE", "EARTH_TILT_DEG"]
