"""TexturedGlobe — drape a pyramids raster (or a tile basemap) over cleopatra's 3-D textured sphere.

``cleopatra.glyphs.globe.TexturedGlobeGlyph`` paints an equirectangular ``(H, W, 3|4)`` array onto a tilted,
spinnable sphere on a matplotlib ``Axes3D``. It is deliberately geometry-only: it knows nothing about rasters,
CRSes or nodata. This module is the Digital-Earth half of that seam — it turns geospatial inputs into the
texture the glyph wants, and maps lon/lat back onto the rendered sphere:

- :meth:`TexturedGlobe.from_dataset` aligns a pyramids ``Dataset`` onto a global EPSG:4326 grid (pyramids does
  the reprojection and the resample) and colour-maps the band through cleopatra's colormap resolver — so a
  regional raster sits on the globe where it belongs instead of being stretched over the whole sphere, and the
  rest of the sphere stays transparent.
- :meth:`TexturedGlobe.from_provider` pulls a whole-world XYZ basemap via ``cleopatra.basemap.tiles``.
- :meth:`TexturedGlobe.project` and :meth:`TexturedGlobe.points` push lon/lat (or a pyramids
  ``FeatureCollection``) through the glyph's own ``transform``, so overlays land exactly on the drawn surface
  at any ``spin`` — including the axial tilt — rather than being re-derived here.
- :meth:`TexturedGlobe.coastlines`, :meth:`TexturedGlobe.borders` and :meth:`TexturedGlobe.land` add the
  Natural-Earth reference geography the 2-D globe :class:`~digitalearth.static.map.Map` offers.

Every overlay is one persistent matplotlib collection that places itself on the sphere each time it is drawn:
at the spin of the axes it lives on, and against the camera that axes has *then*. That is what lets it turn
with an animation, survive a mouse-drag of the camera, and hand back an artist that stays valid.

Like the cleopatra glyph it wraps (and unlike :class:`~digitalearth.static.map.Map`), this is a standalone
class rather than a :class:`~digitalearth.static.scene.Scene` subclass: ``Scene`` owns a 2-D axes and the
layer/colorbar lifecycle, none of which applies to a textured sphere.
"""

import inspect
import warnings
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cleopatra.basemap.reference import natural_earth
from cleopatra.basemap.tiles import world_texture
from cleopatra.glyphs.globe.textured_globe_glyph import (
    EARTH_TILT_DEG,
    TexturedGlobeGlyph,
)
from cleopatra.styling.colors import resolve_colormap
from cleopatra.styling.watermark import stamp_mark
from matplotlib import cbook, rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
from pyramids.dataset import Dataset, GeoReference

from digitalearth.base.arrays import finite, read_masked_band, ring_runs
from digitalearth.base.crs import source_epsg
from digitalearth.static.animation import save_animation

#: Default shape of the global equirectangular canvas built by :meth:`TexturedGlobe.from_dataset`,
#: as ``(rows, columns)`` — a 0.125-degree grid, comfortably finer than the glyph's default mesh.
DEFAULT_TEXTURE_SHAPE = (1440, 2880)

#: EPSG code of the lon/lat grid the glyph's texture is defined on.
_TEXTURE_EPSG = 4326

#: How far inside the globe the texture's outermost cell centres sit, as a fraction of one cell. Enough to
#: put them inside a whole-world source rather than on its boundary; far too small to move anything visibly.
_EDGE_INSET = 1e-6

#: The glyph's own default inter-frame interval, in milliseconds, read from its signature rather than
#: mirrored as a literal: a hand-copied default drifts silently, and the saved frame rate is derived from it.
_GLYPH_ANIMATE = inspect.signature(TexturedGlobeGlyph.animate).parameters
_DEFAULT_INTERVAL_MS = _GLYPH_ANIMATE["interval"].default

#: The glyph's own rotation defaults, read the same way: :meth:`TexturedGlobe.animate` sweeps the rotation
#: itself so it knows each frame's spin, and has to sweep exactly the one the glyph's animate would have.
_DEFAULT_N_FRAMES = _GLYPH_ANIMATE["n_frames"].default
_DEFAULT_REVOLUTIONS = _GLYPH_ANIMATE["revolutions"].default


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


def _mesh_sample_indices(
    texture_shape: Tuple[int, int], n_lon: int, n_lat: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    rows = np.clip(
        np.round((90.0 - lat_centres) / 180.0 * (height - 1)).astype(int), 0, height - 1
    )
    cols = np.clip(
        np.round((lon_centres + 180.0) / 360.0 * (width - 1)).astype(int), 0, width - 1
    )
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
    lon_rad, lat_rad = (
        np.deg2rad(np.asarray(lon, dtype=float)),
        np.deg2rad(np.asarray(lat, dtype=float)),
    )
    return np.stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        axis=-1,
    )


#: Where each kind of overlay claims to sit when matplotlib depth-sorts the axes' collections, as a radius
#: along the view axis. All three are nearer the camera than any point of the unit sphere, so a near-side
#: overlay always sorts in front of the surface it lies on; they are ordered so a fill sits under its
#: coastlines and both sit under markers; and anything a caller floats further out than 1.003 still sorts in
#: front of all three. See :func:`_front_depth`.
_FILL_RANK = 1.001
_LINE_RANK = 1.002
_POINT_RANK = 1.003

#: How many points to sample along a limb arc when closing a clipped land ring. The arc is at most a
#: half-circle, so this holds the chord error under a pixel at any figure size these globes are drawn at.
_LIMB_ARC_STEPS = 48


def _limb_point(
    inside: np.ndarray, outside: np.ndarray, view: np.ndarray
) -> np.ndarray:
    """The point on the visible limb between a near-side vertex and its far-side neighbour.

    The camera-facing test is ``p @ view > 0``, which is linear along the straight chord between two
    vertices, so the crossing is found exactly by interpolating that dot product to zero. The result is
    pushed back out to ``inside``'s radius so a clipped ring stays on the same shell as the rest of it.

    Args:
        inside: The ``(3,)`` near-side vertex.
        outside: Its ``(3,)`` far-side neighbour.
        view: The unit ``(3,)`` vector pointing at the camera.

    Returns:
        np.ndarray: the ``(3,)`` crossing point, on the limb and at ``inside``'s radius.

    Examples:
        - A vertex 45 degrees in front of the limb and its neighbour 45 degrees behind it cross on it:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.textured_globe import _limb_point
            >>> view = np.array([1.0, 0.0, 0.0])
            >>> inside = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
            >>> outside = np.array([-np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
            >>> _limb_point(inside, outside, view).round(6).tolist()
            [0.0, 1.0, 0.0]

            ```
        - The crossing stays on the ring's own shell, so a lifted overlay is not pulled onto the sphere:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.textured_globe import _limb_point
            >>> inside = 1.01 * np.array([0.6, 0.8, 0.0])
            >>> outside = 1.01 * np.array([-0.6, 0.8, 0.0])
            >>> crossing = _limb_point(inside, outside, np.array([1.0, 0.0, 0.0]))
            >>> round(float(np.linalg.norm(crossing)), 6)
            1.01

            ```
    """
    depth_in, depth_out = float(inside @ view), float(outside @ view)
    span = depth_in - depth_out
    crossing = (
        inside if span == 0.0 else inside + (depth_in / span) * (outside - inside)
    )
    tangential = crossing - (crossing @ view) * view
    length = float(np.linalg.norm(tangential))
    if length == 0.0:
        return np.asarray(inside, dtype=float)
    return np.asarray(tangential / length * float(np.linalg.norm(inside)), dtype=float)


def _limb_arc(start: np.ndarray, end: np.ndarray, view: np.ndarray) -> np.ndarray:
    """Points along the limb from ``start`` to ``end``, the short way round.

    A land ring that runs off the near side has to be closed along the limb, or the fill cuts a chord
    straight across the disc. The short arc is the right one for any ring spanning less than half the
    limb, which every Natural-Earth land part does at the resolutions these globes use.

    Args:
        start: The ``(3,)`` limb point the ring left the near side at.
        end: The ``(3,)`` limb point it comes back at.
        view: The unit ``(3,)`` vector pointing at the camera.

    Returns:
        np.ndarray: an ``(_LIMB_ARC_STEPS, 3)`` arc, excluding both endpoints; empty when ``start`` lies
        on the view axis, where there is no limb direction to walk along, or when ``end`` coincides with
        it, where there is no distance to cover.

    Examples:
        - A quarter turn round the limb, sampled between its endpoints, every point on the limb:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.textured_globe import _limb_arc
            >>> view = np.array([1.0, 0.0, 0.0])
            >>> arc = _limb_arc(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), view)
            >>> arc.shape
            (48, 3)
            >>> bool(np.allclose(arc @ view, 0.0))
            True

            ```
        - A start facing the camera has no direction to walk, so no arc comes back:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.textured_globe import _limb_arc
            >>> view = np.array([1.0, 0.0, 0.0])
            >>> _limb_arc(view.copy(), np.array([0.0, 1.0, 0.0]), view).shape
            (0, 3)

            ```
    """
    radius = float(np.linalg.norm(start))
    first = start / radius
    second = np.cross(view, first)
    length = float(np.linalg.norm(second))
    if length == 0.0:
        return np.empty((0, 3))
    second = second / length
    angle = float(np.arctan2(end @ second, end @ first))
    # the ring leaves and rejoins the limb at one point, so there is no distance to walk
    if abs(angle) < 1e-9:
        return np.empty((0, 3))
    steps = np.linspace(0.0, angle, _LIMB_ARC_STEPS + 2)[1:-1]
    return radius * (
        np.cos(steps)[:, None] * first[None, :]
        + np.sin(steps)[:, None] * second[None, :]
    )


def _root_figure(axes: Any) -> Any:
    """The top-level figure an axes belongs to, looking through any sub-figure it sits in.

    An animation has to be built on the figure that owns the canvas: a ``SubFigure`` cannot be saved, so a
    globe animated inside one produced an animation that raised on ``save_animation``.

    Args:
        axes: The axes being drawn on.

    Returns:
        The root ``Figure``.
    """
    try:
        return axes.get_figure(root=True)  # matplotlib >= 3.10
    except TypeError:
        figure = axes.get_figure()
        return getattr(figure, "figure", figure) or figure


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


def _front_depth(axes: Any, radius: float) -> float:
    """The projected depth of the point on the view axis at ``radius`` from the sphere's centre.

    An overlay reports this as its depth when matplotlib sorts the axes' collections back to front. The
    sphere's nearest point sits on that same axis at radius 1, so any ``radius`` above 1 is nearer the camera
    than every point of the surface: a near-side overlay sorts in front of the sphere it lies on, while the
    caller's own artists keep matplotlib's depth sort among themselves and against the globe.

    This replaces switching off ``computed_zorder``, which would have been simpler and wrong: it disables
    depth sorting for the whole axes, including artists the caller drew on it.

    Args:
        axes: The ``Axes3D`` being drawn; its current ``elev``/``azim`` and projection matrix are used.
        radius: How far out along the view axis to claim to sit.

    Returns:
        The depth matplotlib's sort key compares: a smaller value is nearer the camera, and matplotlib paints
        the largest first, so the nearest ends up on top.

    Examples:
        - A point just outside the sphere, on the camera's side, sorts nearer than the sphere's centre:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from digitalearth.static.textured_globe import _front_depth
            >>> ax = plt.figure().add_subplot(projection="3d")
            >>> ax.view_init(elev=0.0, azim=0.0)
            >>> ax.M = ax.get_proj()
            >>> bool(_front_depth(ax, 1.003) < _front_depth(ax, 0.0))
            True

            ```
        - Anything floated further out still sorts nearer than an overlay lying on the surface:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from digitalearth.static.textured_globe import _front_depth
            >>> ax = plt.figure().add_subplot(projection="3d")
            >>> ax.view_init(elev=0.0, azim=0.0)
            >>> ax.M = ax.get_proj()
            >>> bool(_front_depth(ax, 2.0) < _front_depth(ax, 1.003))
            True

            ```
    """
    view = _view_vector(axes.elev, axes.azim)
    x, y, z = view * float(radius)
    return float(proj3d.proj_transform(x, y, z, axes.M)[2])


def _clip_ring(world: np.ndarray, view: np.ndarray) -> Optional[np.ndarray]:
    """Clip one closed ring to the hemisphere facing the camera, re-closing it along the limb.

    A closed ring repeats its first vertex, so the repeat is dropped first, and the visible runs are found with
    :func:`~digitalearth.base.arrays.ring_runs`, which keeps a stretch crossing the ring's seam whole. Cut there
    instead, a ring whose data happens to begin on the near side would have each half closed out to the limb
    separately — a spur from the first vertex to the horizon, which shows as soon as the fill has an edge.

    Args:
        world: The ring's ``(N, 3)`` world-space vertices, as placed on the sphere.
        view: The unit ``(3,)`` vector pointing at the camera.

    Returns:
        The clipped ``(M, 3)`` ring, or ``None`` when none of it faces the camera or it has fewer than three
        vertices, which enclose nothing.

    Examples:
        - A ring wholly on the near side comes back whole, without its repeated closing vertex:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.textured_globe import _clip_ring
            >>> ring = np.array([[1.0, 0.0, 0.0], [0.9, 0.3, 0.0], [0.9, 0.0, 0.3], [1.0, 0.0, 0.0]])
            >>> _clip_ring(ring, np.array([1.0, 0.0, 0.0])).shape
            (3, 3)

            ```
        - A ring wholly behind the globe clips to nothing:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.textured_globe import _clip_ring
            >>> ring = np.array([[-1.0, 0.0, 0.0], [-0.9, 0.3, 0.0], [-0.9, 0.0, 0.3]])
            >>> _clip_ring(ring, np.array([1.0, 0.0, 0.0])) is None
            True

            ```
    """
    ring = np.asarray(world, dtype=float)
    if len(ring) > 1 and np.array_equal(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        return None  # fewer than three vertices enclose nothing
    near = ring @ view > 0.0
    if not near.any():
        return None
    if near.all():
        return ring
    runs = ring_runs(near)
    count = len(ring)
    pieces: List[np.ndarray] = []
    for position, run in enumerate(runs):
        # every run is bounded by hidden vertices on both sides, wrapping round the seam if need be
        entry = _limb_point(ring[run[0]], ring[(run[0] - 1) % count], view)
        exit_ = _limb_point(ring[run[-1]], ring[(run[-1] + 1) % count], view)
        following = runs[(position + 1) % len(runs)]
        resume = _limb_point(ring[following[0]], ring[(following[0] - 1) % count], view)
        pieces += [
            entry[None, :],
            ring[run],
            exit_[None, :],
            _limb_arc(exit_, resume, view),
        ]
    # each run contributes its entry, its own vertices and its exit, so a face always has three or more
    return np.vstack(pieces)


class _GlobeOverlay:
    """What every globe overlay shares: lon/lat geometry placed on the sphere afresh at each draw.

    The body-frame coordinates are computed once, when the overlay is made. Each draw pushes them through the
    glyph's own ``transform`` at the overlay's current spin, and decides visibility against the camera the
    axes has at that moment. So turning the globe moves the overlay, turning the camera — a ``view_init`` or
    a mouse drag — re-decides which half of it shows, and neither ever rebuilds the artist. The artist the
    caller got back therefore stays the one on the axes: restyling it, removing it, or keying a colorbar to
    it keep working as the globe turns.

    Mixed into a matplotlib 3-D collection, whose ``do_3d_projection`` each subclass overrides.
    """

    _place: Callable[..., np.ndarray]
    _spin: float

    def _adopt(self, place: Callable[..., np.ndarray], spin: float) -> None:
        """Bind the overlay to the glyph transform that places it, at a starting spin.

        Args:
            place: The glyph's ``transform``, taking body-frame points and a ``spin``.
            spin: The spin, in degrees, to place the overlay at until it is turned.
        """
        self._place = place
        self._spin = float(spin)

    def _world(self, body: np.ndarray) -> np.ndarray:
        """Place body-frame points on the sphere at the overlay's current spin.

        Args:
            body: ``(N, 3)`` body-frame points.

        Returns:
            The ``(N, 3)`` world-space points.
        """
        return np.atleast_2d(
            np.asarray(self._place(body, spin=self._spin), dtype=float)
        )

    def _view(self) -> np.ndarray:
        """The unit vector toward the camera of the axes the overlay is on, as it stands now.

        Returns:
            A unit ``(3,)`` array pointing from the sphere's centre toward the camera.
        """
        axes: Any = getattr(self, "axes")
        return _view_vector(axes.elev, axes.azim)


class _SpherePoints(_GlobeOverlay, Path3DCollection):
    """A globe scatter: follows the spin, and hides its far-side markers at draw time.

    The scatter keeps every point, so every per-point argument (``c``, ``s``, ``linewidths``, colours as a
    list or a ``Series``) stays aligned with its point and the colour scale is fitted once, to all of them —
    rather than to whichever subset happens to face the camera, which would re-colour a value as the globe
    turned. A hidden marker is drawn at size zero with no edge.
    """

    _body: np.ndarray
    _cull: bool
    _sizes_asked: np.ndarray
    _widths_asked: np.ndarray
    _masking: bool = False
    #: Which markers the last render actually painted — the near-side mask it was drawn with.
    _shown: np.ndarray

    def _adopt_points(
        self,
        place: Callable[..., np.ndarray],
        body: np.ndarray,
        spin: float,
        cull: bool,
    ) -> None:
        """Turn a freshly made scatter into a globe overlay.

        Args:
            place: The glyph's ``transform``.
            body: The points' ``(N, 3)`` body-frame coordinates, already lifted to their altitude.
            spin: The starting spin, in degrees.
            cull: Whether to hide the markers on the far hemisphere.
        """
        self._adopt(place, spin)
        self._body = body
        self._cull = bool(cull)
        self._sizes_asked = np.atleast_1d(np.asarray(self.get_sizes(), dtype=float))
        self._widths_asked = np.atleast_1d(
            np.asarray(self.get_linewidths(), dtype=float)
        )
        self._shown = np.ones(len(body), dtype=bool)

    def _restyling(self) -> bool:
        """Whether a size or width change comes from the caller, rather than the far-side mask or a draw."""
        return not (self._masking or getattr(self, "_in_draw", False))

    def set_sizes(self, sizes: Any, dpi: float = 72.0) -> None:
        """Set the marker sizes, remembering a caller's choice so the far-side mask applies over it.

        What matplotlib stored is read back rather than the argument, because ``None`` means "no size of
        my own" and would otherwise be remembered as a NaN and re-applied at every render.

        Args:
            sizes: The marker sizes, in points squared — one for every marker, or one per point.
            dpi: The resolution the sizes are scaled for, as matplotlib's own ``set_sizes`` takes it.
        """
        super().set_sizes(sizes, dpi)
        if self._restyling():
            stored = np.atleast_1d(np.asarray(self.get_sizes(), dtype=float))
            # matplotlib keeps no size at all for None; the mask has to scale from a number
            self._sizes_asked = (
                stored
                if stored.size
                else np.array([float(rcParams["lines.markersize"]) ** 2])
            )

    def set_linewidth(self, lw: Any) -> None:
        """Set the marker edge widths, remembering a caller's choice so the mask applies over it.

        Args:
            lw: The edge width in points — one for every marker, or one per point. ``None`` asks for
                matplotlib's default, and that resolved width is what gets remembered.
        """
        super().set_linewidth(lw)
        if self._restyling():
            self._widths_asked = np.atleast_1d(
                np.asarray(self.get_linewidths(), dtype=float)
            )

    def _apply(self, sizes: np.ndarray, widths: np.ndarray) -> None:
        """Set the sizes and edge widths to draw with, without recording them as a caller's restyle.

        Args:
            sizes: One marker size per point.
            widths: One edge width per point.
        """
        self._masking = True
        try:
            self.set_sizes(sizes)
            self.set_linewidth(widths)
        finally:
            self._masking = False

    def do_3d_projection(self) -> float:
        """Place the points at the current spin, hide the far side, and sort in front of the sphere.

        Returns:
            The depth matplotlib sorts the scatter by: just in front of the sphere when the far side is
            hidden, and the scatter's own nearest point otherwise.
        """
        world = self._world(self._body)
        self._offsets3d = (world[:, 0], world[:, 1], world[:, 2])
        count = len(world)
        if not self._cull:
            self._shown = np.ones(count, dtype=bool)
            depth: float = super().do_3d_projection()
            return depth
        self._shown = world @ self._view() > 0.0
        self._apply(
            np.where(self._shown, np.resize(self._sizes_asked, count), 0.0),
            np.where(self._shown, np.resize(self._widths_asked, count), 0.0),
        )
        super().do_3d_projection()
        return _front_depth(getattr(self, "axes"), _POINT_RANK)

    def draw(self, renderer: Any) -> None:
        """Draw the markers, then hand the caller's sizes and edge widths back.

        The far-side mask belongs to the render and nothing else. matplotlib's legend handler reads the
        sizes and edge widths straight off the artist, so a legend built after a figure had been drawn
        would otherwise take its marker from the mask — half the size asked for and no edge, or nothing
        at all when every point happens to face away.

        Args:
            renderer: The renderer matplotlib is drawing into.
        """
        try:
            super().draw(renderer)
        finally:
            if self._cull:
                self._apply(self._sizes_asked, self._widths_asked)


class _SphereLines(_GlobeOverlay, Line3DCollection):
    """A reference line layer on the globe: one collection holding every near-side arc of every part."""

    _body: np.ndarray
    _part: np.ndarray

    def _adopt_parts(
        self, place: Callable[..., np.ndarray], parts: List[np.ndarray], spin: float
    ) -> None:
        """Hold the layer's parts as one array, tagged by part, so a frame places them in one transform.

        Args:
            place: The glyph's ``transform``.
            parts: One ``(N, 3)`` body-frame array per polyline.
            spin: The starting spin, in degrees.
        """
        self._adopt(place, spin)
        self._body = np.concatenate(parts) if parts else np.empty((0, 3))
        self._part = np.repeat(np.arange(len(parts)), [len(part) for part in parts])

    def _near_side_segments(self) -> List[np.ndarray]:
        """Split every part into the arcs facing the camera, carried out to the limb at each end.

        A run of visible vertices ends one vertex short of the horizon; the crossing point is added at each
        end whose neighbour in the same part is hidden, so the lines meet the limb where a land fill does.

        Returns:
            One ``(M, 3)`` world-space arc per visible run.
        """
        if not len(self._body):
            return []
        world = self._world(self._body)
        view = self._view()
        keep = np.flatnonzero(world @ view > 0.0)
        if keep.size == 0:
            return []
        breaks = (
            np.flatnonzero((np.diff(keep) != 1) | (np.diff(self._part[keep]) != 0)) + 1
        )
        final = len(world) - 1
        segments: List[np.ndarray] = []
        for run in np.split(keep, breaks):
            first, last = int(run[0]), int(run[-1])
            pieces = [world[run]]
            if first > 0 and self._part[first - 1] == self._part[first]:
                pieces.insert(
                    0, _limb_point(world[first], world[first - 1], view)[None, :]
                )
            if last < final and self._part[last + 1] == self._part[last]:
                pieces.append(_limb_point(world[last], world[last + 1], view)[None, :])
            # parts have two or more vertices, so a run always has a hidden neighbour in its own part
            # and gains at least one limb point: every segment has two vertices or more
            segments.append(np.vstack(pieces))
        return segments

    def do_3d_projection(self) -> float:
        """Re-split the layer for the current spin and camera, and sort in front of the sphere.

        Returns:
            The depth matplotlib sorts the layer by — just in front of the sphere, and of any fill.
        """
        self.set_segments(self._near_side_segments())
        super().do_3d_projection()
        return _front_depth(getattr(self, "axes"), _LINE_RANK)


class _SphereFill(_GlobeOverlay, PolyCollection):
    """A reference fill layer on the globe: one 2-D collection holding every near-side face, projected here.

    A plain ``PolyCollection`` on purpose, not a ``Poly3DCollection``. From matplotlib 3.11 the 3-D collection
    pads every face to the length of the longest, so one Natural-Earth ring of ten thousand vertices makes
    every face ten thousand vertices long — hundreds of megabytes a frame at 50m, gigabytes at 10m. Projected
    here, each face stays the size it is. ``Axes3D.draw`` asks a collection for nothing but
    ``do_3d_projection``, so the class it derives from is free.
    """

    _body: np.ndarray
    _starts: np.ndarray

    def _adopt_rings(
        self, place: Callable[..., np.ndarray], rings: List[np.ndarray], spin: float
    ) -> None:
        """Hold the layer's rings as one array plus offsets, so a frame places them in one transform.

        Args:
            place: The glyph's ``transform``.
            rings: One closed ``(N, 3)`` body-frame ring per polygon.
            spin: The starting spin, in degrees.
        """
        self._adopt(place, spin)
        rings = [
            ring[:-1] if len(ring) > 1 and np.array_equal(ring[0], ring[-1]) else ring
            for ring in rings
        ]
        self._body = np.concatenate(rings) if rings else np.empty((0, 3))
        self._starts = np.concatenate(
            [[0], np.cumsum([len(ring) for ring in rings])]
        ).astype(int)

    def _near_side_faces(self) -> List[np.ndarray]:
        """Clip every ring to the hemisphere facing the camera.

        Visibility is decided for every vertex in one pass, so a ring wholly on the far side — about half
        of them at any moment — is skipped without being clipped.

        Returns:
            One ``(M, 3)`` world-space face per ring with any part on the near side.
        """
        if not len(self._body):
            return []
        world = self._world(self._body)
        view = self._view()
        starts, stops = self._starts[:-1], self._starts[1:]
        in_view = np.logical_or.reduceat(world @ view > 0.0, starts)
        faces = []
        for index in np.flatnonzero(in_view):
            face = _clip_ring(world[starts[index] : stops[index]], view)
            if face is not None:
                faces.append(face)
        return faces

    def do_3d_projection(self) -> float:
        """Re-clip the layer for the current spin and camera, project it, and sort in front of the sphere.

        Every face is projected in one pass through the axes' projection matrix, then split back apart.

        Returns:
            The depth matplotlib sorts the fill by — just in front of the sphere, behind any line layer.
        """
        axes: Any = getattr(self, "axes")
        faces = self._near_side_faces()
        if faces:
            stacked = np.concatenate(faces)
            xs, ys, _ = proj3d.proj_transform(
                stacked[:, 0], stacked[:, 1], stacked[:, 2], axes.M
            )
            flat = np.column_stack([xs, ys])
            self.set_verts(
                np.split(flat, np.cumsum([len(face) for face in faces])[:-1])
            )
        else:
            self.set_verts([])
        return _front_depth(axes, _FILL_RANK)


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
            >>> from digitalearth.static import TexturedGlobe
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
            >>> from digitalearth.static import TexturedGlobe
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
        #: A glyph of the same tilt carrying no texture, whose ``transform`` places the overlays. They
        #: cannot hold the drawing glyph's own method: that keeps its texture alive for as long as the
        #: artist lives, and drags it into anything the figure is pickled or copied into — one marker took
        #: a saved figure from about 1 MB to 134 MB. The rotation depends on the tilt and nothing else, so
        #: this places points exactly as the real glyph does.
        self._placer = TexturedGlobeGlyph(
            np.zeros((2, 4, 3), dtype=np.uint8),
            tilt_deg=kwargs.get("tilt_deg", EARTH_TILT_DEG),
            n_lon=4,
            n_lat=2,
        )
        self.fig: Any = None
        self.ax: Any = None
        # Declared up front rather than sprung into existence by animate(), so every reader can see the
        # instance's full state and the accessors need no getattr() guards.
        self._animation: Optional[FuncAnimation] = None
        self._animation_fps: Optional[float] = None
        #: A fig/ax handed to the constructor is the caller's too — the glyph stores it and draws on it even
        #: when no later call passes one, so ownership has to be settled here, not only at draw time.
        self._caller_supplied_axes: bool = (
            kwargs.get("ax") is not None or kwargs.get("fig") is not None
        )
        #: The spin the globe was last drawn at, so an overlay defaults to the surface it can see.
        self._spin: float = 0.0
        #: The overlays that follow the globe, each still on the axes it was drawn on. A draw or animation
        #: frame turns the ones on its own axes (see :meth:`_turn_overlays`).
        self._overlays: List[Any] = []
        #: Whether :attr:`fig` is ours to close. A caller-supplied axes belongs to the caller.
        self._owns_fig: bool = False
        #: The axes the constructor was given, if any. animate() falls back to it so a globe built around a
        #: caller's axes keeps drawing there instead of opening a figure of its own.
        self._ctor_ax: Any = kwargs.get("ax")

    def _is_ours(self, axes: Any) -> bool:
        """Whether ``axes`` sits on the figure this globe made and still owns.

        Args:
            axes: The axes about to be drawn on.

        Returns:
            True when drawing there is the globe drawing on its own figure.
        """
        return (
            self._owns_fig
            and self.fig is not None
            and axes is not None
            and axes.get_figure() is self.fig
        )

    def _reuse_own_axes(self, ax: Any, kwargs: dict) -> Any:
        """Fall back to the axes this globe already owns, resizing its figure if asked.

        A globe that is drawn, decorated and then drawn (or animated) again should carry on where it is.
        Opening a fresh figure each time would leave its overlays behind on a figure about to be closed —
        which is exactly what made a plain ``animate()`` after ``coastlines()`` come out bare.

        Args:
            ax: The axes the caller asked for, or ``None``.
            kwargs: The render options; ``figsize`` is consumed here when the axes is reused.

        Returns:
            The axes to draw on, which may still be ``None`` when this globe owns nothing yet.
        """
        if ax is not None or self._caller_supplied_axes or not self._is_ours(self.ax):
            return ax
        figsize = kwargs.pop("figsize", None)
        if figsize is not None:
            self.fig.set_size_inches(figsize)
        return self.ax

    def _turn_overlays(self, axes: Any, spin: float) -> None:
        """Turn the overlays drawn on ``axes`` to ``spin``, and forget the ones no longer drawn anywhere.

        Scoped to one axes on purpose. A globe drawn into several panels — a contact sheet of spins — keeps
        each panel's overlays at that panel's spin, so turning one panel must not move, strip or duplicate
        another's. An overlay that has left its axes — removed by the caller, or wiped by ``ax.clear()`` — is
        dropped here, so it is never brought back and a hand-written redraw loop cannot pile copies up.

        Args:
            axes: The axes the globe has just been drawn on.
            spin: The spin, in degrees, it was drawn at.
        """
        kept = []
        for overlay in self._overlays:
            if overlay.axes is None:
                continue
            if overlay.axes is axes:
                overlay._spin = float(spin)
                overlay.stale = True
            kept.append(overlay)
        self._overlays = kept

    @staticmethod
    def _to_body(lon: Any, lat: Any, altitude: float) -> np.ndarray:
        """Turn lon/lat degrees into body-frame points lifted ``altitude`` above the unit surface.

        Args:
            lon: Longitude(s) in degrees.
            lat: Latitude(s) in degrees, the same shape as ``lon``.
            altitude: Radial offset from the unit surface.

        Returns:
            An ``(N, 3)`` array of body-frame points.

        Raises:
            ValueError: if ``lon`` and ``lat`` do not have the same shape.
        """
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=float))
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=float))
        if lon_arr.shape != lat_arr.shape:
            raise ValueError(
                f"lon and lat must have the same shape, got {lon_arr.shape} and {lat_arr.shape}"
            )
        return _lonlat_to_body(lon_arr.ravel(), lat_arr.ravel()) * (
            1.0 + float(altitude)
        )

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

        **pyramids** does the geospatial work: the raster is aligned onto a global EPSG:4326 grid, which
        reprojects it and resamples it in one step, and settles the longitude frame along the way. Its band is
        then read with nodata masked to ``NaN`` and colour-mapped through cleopatra's colormap resolver. Cells
        the raster does not cover — and its nodata cells — stay fully transparent, so a regional dataset shows
        as a patch on the globe rather than being smeared over it.

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
                sphere), if ``shape`` is not two positive integers, if ``band`` is out of range, or if the
                colour bounds are non-finite or leave no range.

        Warns:
            RuntimeWarning: If nothing survives the resample onto the texture, or if what does is finer than
                the sphere mesh and so would not be drawn — see :meth:`_warn_if_finer_than_the_mesh`.

        Examples:
            - Drape a whole-globe raster and inspect the texture it produced:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.arange(8, dtype="float32").reshape(2, 4)
                >>> ds = Dataset.from_array(
                ...     arr=arr, geo_ref=GeoReference(geo=(-180.0, 90.0, 0.0, 90.0, 0.0, -90.0), epsg=4326))
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
                >>> from digitalearth.static import TexturedGlobe
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.ones((2, 2), dtype="float32")
                >>> ds = Dataset.from_array(
                ...     arr=arr, geo_ref=GeoReference(geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0), epsg=4326))
                >>> globe = TexturedGlobe.from_dataset(ds, shape=(90, 180))
                >>> opaque = globe.glyph.texture[..., 3] > 0
                >>> bool(opaque.any()), bool(opaque.all())
                (True, False)

                ```
        """
        try:
            rows, cols = (int(v) for v in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"shape must be a (rows, columns) pair of integers, got {shape!r}"
            ) from exc
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
        # Validate here rather than leaning on select_bands, which the single-band shortcut below skips.
        if not 1 <= band <= dataset.band_count:
            raise ValueError(
                f"band index {band} is out of range for a {dataset.band_count}-band dataset "
                f"(valid 1..{dataset.band_count})."
            )
        # Select the wanted band first: align warps every band it is given, and only one is drawn.
        source = dataset.select_bands([band]) if dataset.band_count > 1 else dataset
        aligned = source.align(cls._global_template(rows, cols), method=resampling)

        values = read_masked_band(aligned, band=1)
        rgba = cls._colorize(values, cmap=cmap, vmin=vmin, vmax=vmax)
        texture = _as_byte_texture(rgba)
        if not (texture[..., 3] > 0).any():
            warnings.warn(
                f"nothing was draped onto the {rows}x{cols} globe texture, so it will render blank. Either "
                "the raster is finer than one texture cell — pass a finer shape=, e.g. shape=(2880, 5760) — "
                "or it does not overlap the globe at all, which usually means its CRS or geotransform is "
                "wrong.",
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
        rows, cols = _mesh_sample_indices(
            texture.shape[:2], self.glyph.n_lon, self.glyph.n_lat
        )
        if not opaque[np.ix_(rows, cols)].any():
            warnings.warn(
                f"the draped data is finer than the {self.glyph.n_lon}x{self.glyph.n_lat} sphere mesh, so it "
                "falls between the mesh samples and the globe will render without it. Raise the mesh "
                "(n_lon=/n_lat=) until a mesh face fits inside the data's footprint.",
                RuntimeWarning,
                stacklevel=3,
            )

    @classmethod
    def from_provider(
        cls, provider: Any = "Esri.WorldImagery", **kwargs: Any
    ) -> "TexturedGlobe":
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
            OSError: if the tiles cannot be fetched — the whole-world grid is pulled over the network on
                first use, so a connection failure or an unreachable provider surfaces here.

        Examples:
            - Build a photographic Earth and spin it (needs the network on first use; the texture is then
              cached on disk):
                ```python
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe.from_provider("Esri.WorldImagery", zoom=3)   # doctest: +SKIP
                >>> globe.glyph.texture.shape                             # doctest: +SKIP
                (1440, 2880, 4)
                >>> fig, ax = globe.draw(spin=30.0, sun=(1.0, 0.3, 0.2))  # doctest: +SKIP

                ```
            - Size the fetched texture and the sphere mesh independently:
                ```python
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe.from_provider(                  # doctest: +SKIP
                ...     "Esri.WorldImagery", zoom=2, texture_n_lon=720, texture_n_lat=360,
                ...     n_lon=60, n_lat=30,
                ... )
                >>> globe.glyph.n_lon, globe.glyph.n_lat                  # doctest: +SKIP
                (60, 30)

                ```
        """
        texture_keys = (
            "zoom",
            "cache",
            "max_workers",
            "timeout",
            "retries",
            "user_agent",
        )
        texture_kwargs = {k: kwargs.pop(k) for k in texture_keys if k in kwargs}
        for shape_key, fetch_key in (
            ("texture_n_lon", "n_lon"),
            ("texture_n_lat", "n_lat"),
        ):
            if shape_key in kwargs:
                texture_kwargs[fetch_key] = kwargs.pop(shape_key)
        return cls(world_texture(provider, **texture_kwargs), **kwargs)

    # ------------------------------------------------------------------ texture building

    @staticmethod
    def _validate_colour_bounds(
        good: np.ndarray, vmin: Optional[float], vmax: Optional[float]
    ) -> None:
        """Reject bound combinations that leave nothing to colour.

        Separated from resolving the bounds because it answers a different question: not "what scale do we
        use" but "is what the caller asked for coherent at all". Quietly substituting a workable range
        instead would return a plausible image drawn from a scale nobody chose.

        Args:
            good: The band's finite values, already stripped of nodata. May be empty.
            vmin: Lower bound, or ``None``.
            vmax: Upper bound, or ``None``.

        Raises:
            ValueError: If either bound is not finite, if both are given and ``vmax <= vmin``, or if a
                single given bound sits outside the data on the wrong side.
        """
        for name, bound in (("vmin", vmin), ("vmax", vmax)):
            if bound is not None and not np.isfinite(float(bound)):
                raise ValueError(f"{name} must be a finite number, got {bound!r}")
        if vmin is not None and vmax is not None and float(vmax) <= float(vmin):
            raise ValueError(
                f"vmax must be greater than vmin, got vmin={vmin!r}, vmax={vmax!r}"
            )
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
    def _resolve_colour_bounds(
        cls, good: np.ndarray, vmin: Optional[float], vmax: Optional[float]
    ) -> Tuple[float, float]:
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
        # A unit range is the fallback when the band has no finite value to take a bound from.
        data_lo, data_hi = (
            (float(good.min()), float(good.max())) if good.size else (0.0, 1.0)
        )
        lo = data_lo if vmin is None else float(vmin)
        hi = data_hi if vmax is None else float(vmax)
        if hi <= lo:  # a constant band has no range to normalise against
            hi = lo + 1.0
        return lo, hi

    @classmethod
    def _colorize(
        cls,
        values: np.ndarray,
        *,
        cmap: Any,
        vmin: Optional[float],
        vmax: Optional[float],
    ) -> np.ndarray:
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
                stacklevel=3,
            )
        lo, hi = cls._resolve_colour_bounds(good, vmin, vmax)
        colormap = resolve_colormap(cmap)
        if colormap is None:  # resolve_colormap returns None only for cmap=None
            colormap = resolve_colormap("viridis")
        rgba = np.asarray(
            colormap(Normalize(vmin=lo, vmax=hi)(np.asarray(values, dtype=float))),
            dtype=float,
        ).copy()
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
        # Pull the outermost centres a hair inside the globe. Sitting exactly on +/-90 / +/-180 puts them on
        # the *boundary* of a whole-world source rather than inside it, and the warp returns nothing there —
        # which showed as a hole at the pole and a seam down the antimeridian. The offset is a millionth of a
        # cell, far below anything the texture can express, so nothing moves visibly.
        inset_lat, inset_lon = d_lat * _EDGE_INSET, d_lon * _EDGE_INSET
        return Dataset.from_array(
            arr=np.zeros((rows, cols), dtype="float32"),
            geo_ref=GeoReference(
                geo=(
                    -180.0 + inset_lon - d_lon / 2,
                    d_lon * (1.0 - 2.0 * _EDGE_INSET / (cols - 1)),
                    0.0,
                    90.0 - inset_lat + d_lat / 2,
                    0.0,
                    -d_lat * (1.0 - 2.0 * _EDGE_INSET / (rows - 1)),
                ),
                epsg=_TEXTURE_EPSG,
            ),
        )

    # ------------------------------------------------------------------ geometry

    def project(
        self, lon: Any, lat: Any, *, spin: Optional[float] = None, altitude: float = 0.0
    ) -> np.ndarray:
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
                >>> from digitalearth.static import TexturedGlobe
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
                >>> from digitalearth.static import TexturedGlobe
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
        body = self._to_body(lon, lat, altitude)
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
                >>> from digitalearth.static import TexturedGlobe
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
                >>> from digitalearth.static import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(elev=0.0, azim=0.0)
                >>> lon, lat = [0.0, 45.0, 180.0, 225.0], [0.0, 0.0, 0.0, 0.0]
                >>> int(globe.visible(globe.project(lon, lat)).sum())
                2

                ```
        """
        if self.ax is None:
            raise RuntimeError(
                "draw() the globe before asking which points are visible"
            )
        view = _view_vector(self.ax.elev, self.ax.azim)
        points = np.atleast_2d(np.asarray(world_xyz, dtype=float))
        return np.asarray(points @ view > 0.0, dtype=bool).ravel()

    # ------------------------------------------------------------------ rendering

    def _bind(self, ax: Any, *, owns: bool) -> None:
        """Adopt ``ax`` as the render target, releasing a figure we own and are leaving behind.

        The single rule for figure ownership: we close a figure only if we created it *and* we are no longer
        drawing on it. Rebinding to a different axes releases the old one so nothing is orphaned beyond
        ``close()``'s reach; rebinding to another axes on the *same* figure closes nothing, because that
        figure is still in use.

        Args:
            ax: The axes now being drawn on.
            owns: Whether the figure behind ``ax`` was created by this globe.
        """
        figure = ax.get_figure()
        if self._owns_fig and self.fig is not None and self.fig is not figure:
            closing = self.fig
            plt.close(closing)
            # the overlays on that figure go with it; keeping them registered would only pin a dead figure
            self._overlays = [
                overlay
                for overlay in self._overlays
                if overlay.axes is not None and overlay.axes.get_figure() is not closing
            ]
        self.ax, self.fig = ax, figure
        self._owns_fig = owns

    def draw(
        self, ax: Any = None, *, spin: float = 0.0, **kwargs: Any
    ) -> Tuple[Any, Any]:
        """Draw the globe, returning the matplotlib ``(fig, ax)`` and recording them on the instance.

        A globe keeps drawing where it already is: called without an ``ax``, this reuses the axes it owns, so
                overlays added since the last draw — :meth:`points`, :meth:`coastlines`, :meth:`borders`,
                :meth:`land` without a ``spin=`` — turn to the new ``spin`` and stay. ``figsize`` resizes that
                figure. Pass an ``ax`` to draw somewhere else instead; overlays stay behind with the axes they
                were drawn on, and a figure this globe owned and is leaving is closed.

                Args:
                    ax: An existing ``Axes3D`` to draw on, accepted positionally to match :meth:`animate`. A figure
                        the caller supplies is never closed by :meth:`close`.
                    spin: Rotation about the polar axis, in degrees.
                    **kwargs: Forwarded to the glyph's ``draw`` (``sun``, ``ambient``, ``figsize``, ``elev``,
                        ``azim``, ``background``).

                Returns:
                    The ``(Figure, Axes3D)`` the globe was drawn on.

                Examples:
                    - Draw the globe and keep the axes for further decoration:
                        ```python
                        >>> import matplotlib
                        >>> matplotlib.use("Agg")
                        >>> import numpy as np
                        >>> from digitalearth.static import TexturedGlobe
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
                        >>> from digitalearth.static import TexturedGlobe
                        >>> import matplotlib.pyplot as plt
                        >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                        >>> fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
                        >>> for spin, sub in zip([0.0, 120.0, 240.0], axes):
                        ...     _ = globe.draw(ax=sub, spin=spin)
                        >>> len(fig.axes)
                        3

                        ```
        """
        ax = self._reuse_own_axes(ax, kwargs)
        if ax is not None:
            kwargs["ax"] = ax
        supplied = not (ax is None or self._is_ours(ax)) or self._caller_supplied_axes
        _, drawn_ax = self.glyph.draw(spin=spin, **kwargs)
        self._bind(drawn_ax, owns=not supplied)
        self._spin = float(spin)
        self._turn_overlays(drawn_ax, self._spin)
        return self.fig, self.ax

    def points(
        self,
        data: Any,
        *,
        lat: Any = None,
        spin: Optional[float] = None,
        altitude: float = 0.01,
        hide_far_side: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Scatter lon/lat points onto the globe's surface.

        Accepts either a pyramids ``FeatureCollection`` / geopandas ``GeoDataFrame`` of points (its geometry
        is reprojected to lon/lat when needed) or a pair of ``lon``/``lat`` array-likes.

        Args:
            data: A point ``FeatureCollection``/``GeoDataFrame``, or the longitudes when ``lat`` is given.
            lat: Latitudes, when ``data`` holds longitudes.
            spin: Pin the points to this spin, in degrees. Omitted (the default), they follow the globe:
                they are drawn at the spin it was last drawn at, and redrawn at each animation frame's
                spin so they turn with the surface. Giving a value opts out of that and fixes them there.
            altitude: Radial lift above the surface, to avoid z-fighting with it.
            hide_far_side: When True (default), hide the markers on the hemisphere facing away from the
                camera, which matplotlib would otherwise draw straight through the sphere. Decided at each
                draw, against the camera as it is then, so a camera move re-decides it. With False every
                marker is drawn and the collection is depth-sorted like any other artist, so markers near
                the limb can be painted over by the sphere.
            **kwargs: Forwarded to ``Axes3D.scatter`` (e.g. ``c``, ``s``, ``marker``, ``color``).

        Returns:
            The scatter collection on the axes. It stays there as the globe turns — re-placed at each draw,
            never replaced — so restyling it, removing it, or keying a colorbar to it keeps working. It
            holds every point: a hidden marker is drawn at size zero rather than dropped, which keeps each
            per-point value with its point and fits the colour scale to all of them.

        Raises:
            RuntimeError: if the globe has not been drawn yet.
            ValueError: if ``data`` is neither a feature collection nor a pair of lon/lat arrays, or is a
                feature collection carrying no CRS.

        Note:
            Non-point geometry (polygons, lines) is reduced to its centroids rather than rejected, so a
            polygon layer can be marked on the globe without converting it first.

        Examples:
            - Scatter lon/lat points; the ones behind the globe are drawn at size zero:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(elev=0.0, azim=0.0)
                >>> scatter = globe.points([0.0, 180.0], lat=[0.0, 0.0])
                >>> fig.canvas.draw()
                >>> int((scatter.get_sizes() > 0).sum())
                1

                ```
            - Keep the far side when you want the full set drawn:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> fig, ax = globe.draw(elev=0.0, azim=0.0)
                >>> scatter = globe.points([0.0, 180.0], lat=[0.0, 0.0], hide_far_side=False)
                >>> fig.canvas.draw()
                >>> len(scatter.get_offsets())
                2

                ```
            - Points follow the globe unless pinned; turning it half-way takes the follower out of sight:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import matplotlib.pyplot as plt
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), tilt_deg=0.0,
                ...                       n_lon=8, n_lat=4)
                >>> ax = plt.figure().add_subplot(projection="3d")
                >>> _ = globe.draw(ax, elev=0.0, azim=0.0)
                >>> follower = globe.points([0.0], lat=[0.0])
                >>> pinned = globe.points([0.0], lat=[0.0], spin=0.0)
                >>> _ = globe.draw(ax, elev=0.0, azim=0.0, spin=180.0)
                >>> ax.get_figure().canvas.draw()
                >>> int((follower.get_sizes() > 0).sum()), int((pinned.get_sizes() > 0).sum())
                (0, 1)
                >>> follower in ax.collections
                True

                ```
        """
        if self.ax is None:
            raise RuntimeError("draw() the globe before adding points to it")
        lon_vals, lat_vals = self._as_lonlat(data, lat)
        body = self._to_body(lon_vals, lat_vals, altitude)
        at = self._spin if spin is None else float(spin)
        world = np.atleast_2d(self.glyph.transform(body, spin=at))
        scatter = self.ax.scatter(world[:, 0], world[:, 1], world[:, 2], **kwargs)
        scatter.__class__ = _SpherePoints
        scatter._adopt_points(self._placer.transform, body, at, cull=hide_far_side)
        if spin is None:
            self._overlays.append(scatter)
        return scatter

    # ------------------------------------------------------------------ reference geography

    def _reference_layer(
        self,
        layer: str,
        resolution: str,
        defaults: dict,
        *,
        spin: Optional[float],
        altitude: float,
        fill: bool,
        **kwargs: Any,
    ) -> Any:
        """Add a Natural-Earth layer to the globe as one overlay collection.

        Args:
            layer: The Natural-Earth layer name (``"coastline"``, ``"borders"``, ``"land"``).
            resolution: Natural-Earth resolution (``"110m"`` / ``"50m"`` / ``"10m"``).
            defaults: The layer's base style. The caller's styling overrides it, after matplotlib's short
                aliases (``lw``, ``c``, ``ec``, ...) are spelled out, so both spellings work.
            spin: Pin the layer to this spin, or ``None`` to follow the globe.
            altitude: Radial lift above the surface.
            fill: Whether the parts are closed rings to fill rather than lines to stroke.

        Returns:
            The ``Line3DCollection`` (lines) or ``PolyCollection`` (fill) now on the axes.

        Raises:
            RuntimeError: if the globe has not been drawn yet, so there is no axes to add the layer to.
        """
        if self.ax is None:
            raise RuntimeError(
                f"draw() the globe before adding the {layer!r} layer to it"
            )
        bodies = [
            self._to_body(part[:, 0], part[:, 1], altitude)
            for part in (
                np.asarray(raw, dtype=float) for raw in natural_earth(layer, resolution)
            )
            if part.ndim == 2 and len(part) >= 2
        ]
        style = {
            **defaults,
            **cbook.normalize_kwargs(kwargs, PolyCollection if fill else Line2D),
        }
        at = self._spin if spin is None else float(spin)
        overlay: Any
        if fill:
            overlay = _SphereFill([], **style)
            overlay._adopt_rings(self._placer.transform, bodies, at)
        else:
            overlay = _SphereLines([], **style)
            overlay._adopt_parts(self._placer.transform, bodies, at)
        # add_collection, not add_collection3d: the overlay is already 3-D, starts empty (it is placed at
        # draw time), and add_collection3d's autolim would try to size the view from that empty geometry
        self.ax.add_collection(overlay, autolim=False)
        if spin is None:
            self._overlays.append(overlay)
        return overlay

    def coastlines(
        self,
        resolution: str = "110m",
        *,
        spin: Optional[float] = None,
        altitude: float = 0.002,
        **kwargs: Any,
    ) -> Any:
        """Draw Natural-Earth coastlines on the globe, split at the visible limb.

        Named and styled to match :meth:`~digitalearth.static.Map.coastlines` on the 2-D globe, so the two
        tiers read alike. Each coastline is placed on the sphere and broken into the arcs that face the
        camera, each carried out to the horizon; without that split every line crossing the limb would be
        drawn as a chord through the planet. The split is redone at every draw, for the spin and the camera
        of that moment.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            spin: Pin the coastlines to this spin, in degrees. Omitted, they follow the globe and are
                redrawn at each animation frame's spin.
            altitude: Radial lift above the surface, so the lines are not z-fought by it.
            **kwargs: Line styling forwarded to matplotlib (``color``, ``linewidth``, ``linestyle``,
                ``alpha``), in either the long or the short spelling (``c``, ``lw``, ``ls``).

        Returns:
            The ``Line3DCollection`` holding every near-side arc. It stays on the axes as the globe turns.

        Raises:
            RuntimeError: if the globe has not been drawn yet.

        Examples:
            - Coastlines land on the sphere as a set of near-side arcs. The vectors are fetched (and
              cached) by ``cleopatra``, so this needs a network on first use:
                ```python
                >>> import matplotlib                                      # doctest: +SKIP
                >>> matplotlib.use("Agg")                                  # doctest: +SKIP
                >>> import numpy as np                                     # doctest: +SKIP
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), np.uint8))  # doctest: +SKIP
                >>> _ = globe.draw(elev=0.0, azim=0.0)                     # doctest: +SKIP
                >>> lines = globe.coastlines()                             # doctest: +SKIP
                >>> globe.fig.canvas.draw()                                # doctest: +SKIP
                >>> len(lines.get_segments())                              # doctest: +SKIP
                57

                ```
            - Restyle them like any matplotlib line, and pin them to one spin so they stay put while the
              globe turns:
                ```python
                >>> import matplotlib                                      # doctest: +SKIP
                >>> matplotlib.use("Agg")                                  # doctest: +SKIP
                >>> import numpy as np                                     # doctest: +SKIP
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), np.uint8))  # doctest: +SKIP
                >>> _ = globe.draw(elev=0.0, azim=0.0)                     # doctest: +SKIP
                >>> lines = globe.coastlines(color="white", lw=1.2, spin=0.0)  # doctest: +SKIP
                >>> float(lines.get_linewidth()[0])                            # doctest: +SKIP
                1.2

                ```
            - Asking before the globe is drawn is refused — there is no camera yet to clip against:
                ```python
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
                >>> TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8)).coastlines()
                Traceback (most recent call last):
                    ...
                RuntimeError: draw() the globe before adding the 'coastline' layer to it

                ```
        """
        return self._reference_layer(
            "coastline",
            resolution,
            {"color": "black", "linewidth": 0.5},
            spin=spin,
            altitude=altitude,
            fill=False,
            **kwargs,
        )

    def borders(
        self,
        resolution: str = "110m",
        *,
        spin: Optional[float] = None,
        altitude: float = 0.002,
        **kwargs: Any,
    ) -> Any:
        """Draw Natural-Earth country borders on the globe, split at the visible limb.

        The counterpart of :meth:`~digitalearth.static.Map.borders`, same spelling and same default
        styling, so a globe and a flat map can be given the same reference geography.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            spin: Pin the borders to this spin, in degrees. Omitted, they follow the globe.
            altitude: Radial lift above the surface.
            **kwargs: Line styling forwarded to matplotlib, as for :meth:`coastlines`.

        Returns:
            The ``Line3DCollection`` holding every near-side arc. It stays on the axes as the globe turns.

        Raises:
            RuntimeError: if the globe has not been drawn yet.

        Examples:
            - Borders draw in their own default grey, thinner than the coastlines. The vectors are
              fetched (and cached) by ``cleopatra``, so this needs a network on first use:
                ```python
                >>> import matplotlib                                      # doctest: +SKIP
                >>> matplotlib.use("Agg")                                  # doctest: +SKIP
                >>> import numpy as np                                     # doctest: +SKIP
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), np.uint8))  # doctest: +SKIP
                >>> _ = globe.draw(elev=0.0, azim=0.0)                     # doctest: +SKIP
                >>> float(globe.borders().get_linewidth()[0])              # doctest: +SKIP
                0.4

                ```
            - Ask for the finer cut when the globe is drawn large:
                ```python
                >>> import matplotlib                                      # doctest: +SKIP
                >>> matplotlib.use("Agg")                                  # doctest: +SKIP
                >>> import numpy as np                                     # doctest: +SKIP
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), np.uint8))  # doctest: +SKIP
                >>> _ = globe.draw(elev=0.0, azim=0.0)                     # doctest: +SKIP
                >>> fine, coarse = globe.borders("50m"), globe.borders("110m")  # doctest: +SKIP
                >>> globe.fig.canvas.draw()                                     # doctest: +SKIP
                >>> len(fine.get_segments()) > len(coarse.get_segments())       # doctest: +SKIP
                True

                ```
            - Asking before the globe is drawn is refused — there is no camera yet to clip against:
                ```python
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
                >>> TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8)).borders()
                Traceback (most recent call last):
                    ...
                RuntimeError: draw() the globe before adding the 'borders' layer to it

                ```
        """
        return self._reference_layer(
            "borders",
            resolution,
            {"color": "gray", "linewidth": 0.4},
            spin=spin,
            altitude=altitude,
            fill=False,
            **kwargs,
        )

    def land(
        self,
        resolution: str = "110m",
        *,
        spin: Optional[float] = None,
        altitude: float = 0.001,
        **kwargs: Any,
    ) -> Any:
        """Fill Natural-Earth land polygons on the globe.

        The counterpart of :meth:`~digitalearth.static.Map.land`. Each land ring is clipped to the
        hemisphere facing the camera and re-closed along the limb, so a continent running off the edge is
        filled to the horizon instead of having a chord cut across the disc. This is what turns an ocean
        product — SST, sea ice, chlorophyll — from a ball of colour with white holes into a readable map.

        Interior rings (holes) are dropped, as they are on the 2-D globe. A ring spanning more than half
        the visible limb is closed the short way round, which is the wrong half for such a ring; no
        Natural-Earth land part does that at the resolutions offered here.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            spin: Pin the fill to this spin, in degrees. Omitted, it follows the globe.
            altitude: Radial lift above the surface. The fill sorts under the line layers by the depth
                it reports rather than by its radius, so this only has to clear the sphere itself.
            **kwargs: Patch styling forwarded to matplotlib (``color``, ``alpha``, ``edgecolor``, ...),
                in either the long or the short spelling (``fc``, ``ec``, ``lw``).

        Returns:
            The ``PolyCollection`` holding the fill. It stays on the axes as the globe turns, re-clipped and
            re-projected at each draw; at a camera that shows no land it simply has no faces.

        Raises:
            RuntimeError: if the globe has not been drawn yet.

        Examples:
            - The fill arrives as one collection holding every visible land ring. The vectors are fetched
              (and cached) by ``cleopatra``, so this needs a network on first use:
                ```python
                >>> import matplotlib                                      # doctest: +SKIP
                >>> matplotlib.use("Agg")                                  # doctest: +SKIP
                >>> import numpy as np                                     # doctest: +SKIP
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), np.uint8))  # doctest: +SKIP
                >>> _ = globe.draw(elev=0.0, azim=0.0)                     # doctest: +SKIP
                >>> patch = globe.land()                                   # doctest: +SKIP
                >>> patch in globe.ax.collections                          # doctest: +SKIP
                True

                ```
            - Colour it like any patch — a translucent fill still lets the texture read through:
                ```python
                >>> import matplotlib                                      # doctest: +SKIP
                >>> matplotlib.use("Agg")                                  # doctest: +SKIP
                >>> import numpy as np                                     # doctest: +SKIP
                >>> from digitalearth.static import TexturedGlobe          # doctest: +SKIP
                >>> globe = TexturedGlobe(np.zeros((8, 16, 3), np.uint8))  # doctest: +SKIP
                >>> _ = globe.draw(elev=0.0, azim=0.0)                     # doctest: +SKIP
                >>> patch = globe.land(color="0.4", alpha=0.5)             # doctest: +SKIP
                >>> round(float(patch.get_facecolor()[0][3]), 2)           # doctest: +SKIP
                0.5

                ```
            - Asking before the globe is drawn is refused — there is no camera yet to clip against:
                ```python
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
                >>> TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8)).land()
                Traceback (most recent call last):
                    ...
                RuntimeError: draw() the globe before adding the 'land' layer to it

                ```
        """
        return self._reference_layer(
            "land",
            resolution,
            {"color": "#efefdb", "edgecolor": "none"},
            spin=spin,
            altitude=altitude,
            fill=True,
            **kwargs,
        )

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
            return np.atleast_1d(np.asarray(data, dtype=float)), np.atleast_1d(
                np.asarray(lat, dtype=float)
            )
        geometry = getattr(data, "geometry", None)
        if geometry is None:
            raise ValueError(
                "points needs a FeatureCollection/GeoDataFrame, or both lon and lat"
            )
        # As on the raster path, a missing EPSG code is not the same as a missing CRS: a projection with no
        # authority code still reprojects, so test the CRS itself.
        epsg = source_epsg(data)
        if epsg is None and getattr(data, "crs", None) is None:
            raise ValueError(
                "the feature collection has no CRS, so its points cannot be placed on the globe"
            )
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

        Overlays turn with the sphere. Anything added with :meth:`points`, :meth:`coastlines`,
                :meth:`borders` or :meth:`land` is turned to each frame's spin, so markers and coastlines stay
                attached to the ground; pass ``spin=`` to one of those to pin it in place instead. Called without
                an ``ax``, this animates on the axes the globe already owns, so a globe that was drawn and
                decorated keeps its overlays.

                The rotation is swept here rather than by the glyph's own ``animate``, which is how each frame's
                spin is known; it sweeps the same one, ``start_spin`` plus ``revolutions`` whole turns spread over
                ``n_frames`` frames. The first frame is drawn by this call, so a bad render option is refused here.

                Args:
                    ax: An existing ``Axes3D`` to animate on. When omitted, the constructor's axes is used if one
                        was given, and only otherwise is a new figure created.
                    **kwargs: Forwarded to the glyph's ``animate`` (``n_frames``, ``revolutions``, ``start_spin``,
                        ``sun``, ``ambient``, ``interval``, plus the render options). ``figsize`` sizes the figure
                        created here, and is ignored when ``ax`` is given — that figure already exists.

                Returns:
                    The ``FuncAnimation`` over the rotation. It is also kept on ``self._animation`` so it is not
                    garbage-collected before you save or display it.

                Raises:
                    ValueError: If ``interval`` is not a positive number of milliseconds, ``n_frames`` is below 1,
                        or a render option or lighting argument is out of contract.

                Examples:
                    - Animate a rotation; the figure and axes are recorded for saving or stamping afterwards:
                        ```python
                        >>> import matplotlib
                        >>> matplotlib.use("Agg")
                        >>> import numpy as np
                        >>> from digitalearth.static import TexturedGlobe
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
                        >>> from digitalearth.static import TexturedGlobe
                        >>> import matplotlib.pyplot as plt
                        >>> globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
                        >>> ax = plt.figure().add_subplot(projection="3d")
                        >>> anim = globe.animate(ax, n_frames=2, interval=200)
                        >>> globe.ax is ax
                        True

                        ```
        """
        interval = float(kwargs.pop("interval", _DEFAULT_INTERVAL_MS))
        if interval <= 0:
            raise ValueError(
                f"interval must be a positive number of milliseconds, got {interval!r}"
            )
        n_frames = int(kwargs.pop("n_frames", _DEFAULT_N_FRAMES))
        if n_frames < 1:
            raise ValueError(f"n_frames must be at least 1, got {n_frames}")
        spins = float(kwargs.pop("start_spin", 0.0)) + np.linspace(
            0.0,
            360.0 * float(kwargs.pop("revolutions", _DEFAULT_REVOLUTIONS)),
            n_frames,
            endpoint=False,
        )
        supplied = not (ax is None or self._is_ours(ax)) or self._caller_supplied_axes
        if ax is None:
            ax = (
                self._ctor_ax
            )  # a globe built around the caller's axes keeps animating there
        ax = self._reuse_own_axes(ax, kwargs)
        if ax is None:
            figsize = kwargs.pop(
                "figsize", self.glyph.default_options.get("figsize", (6, 6))
            )
            ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
        # The rotation is swept here rather than by the glyph's own animate, so each frame's spin is known
        # without reaching into the glyph. The first frame is drawn now: a bad render option or lighting
        # argument is refused at this call, as the glyph's animate would, and not from inside the frame loop.
        self.glyph.draw(ax, spin=float(spins[0]), **kwargs)
        self._bind(ax, owns=not supplied)
        self._spin = float(spins[0])
        self._turn_overlays(ax, self._spin)

        def _frame(index: int) -> Tuple[Any, ...]:
            spin = float(spins[index])
            self.glyph.draw(ax, spin=spin, **kwargs)
            self._spin = spin
            self._turn_overlays(ax, spin)
            return (self.glyph.surface,)

        anim = FuncAnimation(
            _root_figure(ax), _frame, frames=n_frames, interval=interval, blit=False
        )
        self._animation = (
            anim  # keep a strong reference so it survives until save/display
        )
        self._animation_fps = 1000.0 / interval
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
                >>> from digitalearth.static import TexturedGlobe
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
                >>> from digitalearth.static import TexturedGlobe
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

    def save_animation(
        self,
        path: str,
        *,
        fps: Optional[float] = None,
        gif: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Save the rotation built by :meth:`animate`, optionally also deriving a GIF from it.

        A textured globe is exactly the case the derive-a-GIF path exists for: every frame is a full 3-D
        surface redraw, so encoding twice off one render is far cheaper than rendering twice.

        Args:
            path: Output path; the extension picks the format.
            fps: Frames per second. Defaults to the animation's own interval.
            gif: Optional second path to derive a GIF at. Requires ``path`` to be a video.
            **kwargs: Forwarded to :func:`digitalearth.static.animation.save_animation`.

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
                >>> from digitalearth.static import TexturedGlobe
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
        """Close the globe's figure, drop the animation reference, and forget the overlays.

        Every :meth:`draw` or :meth:`animate` that is not handed an existing axes creates a pyplot figure,
        and pyplot keeps a reference to it forever. A loop that builds many globes therefore grows without
        bound and eventually trips matplotlib's open-figure warning. Closing releases both the figure and the
        retained animation.

        Only a figure this globe created is closed. An axes handed in by the caller — to ``__init__``,
        ``draw(ax=...)`` or ``animate(ax=...)`` — belongs to the caller, who may well have other subplots on
        it, so its figure is left alone.

        The overlays are forgotten either way, so a later draw — on any axes — starts without them rather
        than turning markers that belong to a figure the caller has finished with.

        Safe to call more than once, and on a globe that was never drawn.

        Examples:
            - Close a drawn globe and see the figure released:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import matplotlib.pyplot as plt
                >>> import numpy as np
                >>> from digitalearth.static import TexturedGlobe
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
        self._overlays = []

    def __enter__(self) -> "TexturedGlobe":
        """Enter the runtime context, returning the globe so ``with TexturedGlobe(...) as g:`` binds it.

        Returns:
            This globe.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the figure on exit so a long run of globes stays memory-bounded.

        Mirrors :class:`~digitalearth.static.scene.Scene`: the figure is closed whether or not the body
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

        The same figure-level mark as :meth:`digitalearth.static.scene.Scene.stamp`, and it carries the same
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
                >>> from digitalearth.static import TexturedGlobe
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
                >>> from digitalearth.static import TexturedGlobe
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


#: ``EARTH_TILT_DEG`` is cleopatra's constant, re-exported from this module (not from
#: ``digitalearth.static``) so a caller adjusting ``tilt_deg`` can reach it without importing from the glyph.
__all__: List[str] = ["TexturedGlobe", "DEFAULT_TEXTURE_SHAPE", "EARTH_TILT_DEG"]
