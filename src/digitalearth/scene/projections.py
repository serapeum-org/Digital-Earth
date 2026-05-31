"""Named map projections + projected-frame geometry for the globe/projection tier.

This module is the digitalearth side of the projected-maps plan (tasks DC.1–DC.3). It provides:

- **named projections** (`orthographic`, `robinson`, `mollweide`, `polar_north`/`polar_south`, `web_mercator`)
  resolving to a CRS spec that pyramids' ``to_crs`` / ``reproject_coordinates`` accept;
- :func:`projection_frame` — the projection's **boundary** polygon + projected limits;
- :func:`graticule` — projected lon/lat **grid polylines**.

The boundary/graticule geometry is assembled here from pyramids' existing coordinate transform
(``pyramids.base.crs.reproject_coordinates``, which accepts arbitrary CRS) plus numpy — no shapely in
digitalearth, no new pyramids code. cleopatra then *draws* this geometry via ``apply_projection_frame``.
"""
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from pyramids.base.crs import reproject_coordinates

__all__ = ["PROJECTIONS", "get", "projection_frame", "graticule",
           "orthographic", "robinson", "mollweide", "polar_north", "polar_south", "web_mercator"]


# ---------------------------------------------------------------- named projections


def web_mercator() -> int:
    """Web Mercator (EPSG:3857) — the slippy-map default."""
    return 3857


def polar_north() -> int:
    """NSIDC Sea Ice Polar Stereographic North (EPSG:3413)."""
    return 3413


def polar_south() -> int:
    """Antarctic Polar Stereographic (EPSG:3031)."""
    return 3031


def orthographic(lon: float = 0.0, lat: float = 0.0) -> str:
    """Orthographic 'globe' centred on ``(lon, lat)`` — a proj4 string (no EPSG code)."""
    return f"+proj=ortho +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"


def robinson(lon: float = 0.0) -> str:
    """Robinson world projection centred on ``lon`` — a proj4 string."""
    return f"+proj=robin +lon_0={lon} +datum=WGS84 +units=m +no_defs"


def mollweide(lon: float = 0.0) -> str:
    """Mollweide equal-area world projection centred on ``lon`` — a proj4 string."""
    return f"+proj=moll +lon_0={lon} +datum=WGS84 +units=m +no_defs"


#: Registry of projection-name → factory. ``get(name, **kw)`` resolves a CRS spec.
PROJECTIONS: Dict[str, Callable[..., Any]] = {
    "web_mercator": web_mercator,
    "polar_north": polar_north,
    "polar_south": polar_south,
    "orthographic": orthographic,
    "robinson": robinson,
    "mollweide": mollweide,
}


def get(name: str, **kwargs) -> Any:
    """Resolve a registered projection name to a CRS spec.

    Args:
        name: A key of :data:`PROJECTIONS` (case-insensitive), e.g. ``"orthographic"``.
        **kwargs: Forwarded to the factory (e.g. ``lon``/``lat`` for ``orthographic``).

    Returns:
        The CRS spec (an EPSG int or a proj4 string) for pyramids ``to_crs``.

    Raises:
        KeyError: if ``name`` is not registered.

    Examples:
        - Resolve an EPSG-coded projection:
            ```python
            >>> from digitalearth.scene import projections
            >>> projections.get("web_mercator")
            3857
            
            ```
        - Resolve a parametrised proj4 projection:
            ```python
            >>> from digitalearth.scene import projections
            >>> projections.get("orthographic", lon=-9, lat=39)
            '+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84 +units=m +no_defs'
            
            ```
    """
    key = name.strip().lower()
    if key not in PROJECTIONS:
        raise KeyError(f"unknown projection {name!r}; known: {sorted(PROJECTIONS)}")
    return PROJECTIONS[key](**kwargs)


# ---------------------------------------------------------------- geometry helpers


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the convex-hull ring (closed) of 2-D ``points`` via Andrew's monotone chain (pure numpy).

    Fewer than three (unique) points cannot form a polygon, so the degenerate input is returned closed as-is:
    one point repeats to a 2-vertex ring, an empty input stays empty.

    Args:
        points: An ``(N, 2)`` array of 2-D points (duplicates are collapsed before hulling).

    Returns:
        The closed hull ring as an ``(M, 2)`` array whose first vertex repeats at the end; an empty
        ``(0, 2)`` array when ``points`` is empty.

    Examples:
        - The hull of a filled square is its four corners, closed back to the start:
            ```python
            >>> import numpy as np
            >>> from digitalearth.scene.projections import _convex_hull
            >>> sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
            >>> ring = _convex_hull(sq)
            >>> bool(np.allclose(ring[0], ring[-1]))
            True
            >>> len(ring)
            5

            ```
        - A single point is returned as a closed degenerate ring (itself, repeated):
            ```python
            >>> import numpy as np
            >>> from digitalearth.scene.projections import _convex_hull
            >>> _convex_hull(np.array([[2.0, 3.0]])).tolist()
            [[2.0, 3.0], [2.0, 3.0]]

            ```
    """
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return np.vstack([pts, pts[:1]]) if len(pts) else pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross2d(o, a, b):
        """2-D cross product (b - o) x (a - o); >0 = left turn (numpy.cross on 2-D is deprecated)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(chain_pts):
        hull: List[np.ndarray] = []
        for p in chain_pts:
            while len(hull) >= 2 and cross2d(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull[:-1]

    lower = half(pts)
    upper = half(pts[::-1])
    ring = np.array(lower + upper)
    return np.vstack([ring, ring[:1]])


def _split_finite(x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    """Split parallel x/y into contiguous finite-run polylines (drops off-domain / antimeridian gaps).

    A line crossing the projection limb (or the antimeridian) reprojects to runs of finite points separated
    by ``inf``/``nan``; this keeps each finite run as its own polyline so nothing wraps across the figure.
    Single-point runs are discarded — a polyline needs at least two vertices to draw.

    Args:
        x: X coordinates of the polyline, in order; non-finite entries mark the breaks.
        y: Y coordinates, parallel to ``x`` (the same index is a break if either ``x`` or ``y`` is
            non-finite).

    Returns:
        A list of ``(M, 2)`` arrays, one per contiguous finite run of length ≥ 2 (empty list if none).

    Examples:
        - A gap (``nan``) splits one line into two separate finite polylines:
            ```python
            >>> import numpy as np
            >>> from digitalearth.scene.projections import _split_finite
            >>> x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
            >>> y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
            >>> segs = _split_finite(x, y)
            >>> [seg.tolist() for seg in segs]
            [[[0.0, 0.0], [1.0, 1.0]], [[3.0, 3.0], [4.0, 4.0]]]

            ```
        - An isolated finite point (length-1 run) is dropped:
            ```python
            >>> import numpy as np
            >>> from digitalearth.scene.projections import _split_finite
            >>> x = np.array([np.inf, 5.0, np.inf])
            >>> y = np.array([0.0, 5.0, 0.0])
            >>> _split_finite(x, y)
            []

            ```
    """
    finite = np.isfinite(x) & np.isfinite(y)
    out: List[np.ndarray] = []
    run: List[Tuple[float, float]] = []
    for xi, yi, ok in zip(x, y, finite):
        if ok:
            run.append((xi, yi))
        elif run:
            out.append(np.asarray(run))
            run = []
    if run:
        out.append(np.asarray(run))
    return [r for r in out if len(r) > 1]


def projection_frame(crs: Any, n: int = 720) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Return the boundary polygon and projected limits of a CRS's valid domain.

    Samples the whole sphere on a lon/lat grid, projects it via pyramids, keeps the finite points, and takes
    their convex hull — a circle for orthographic, a rounded rectangle for Robinson/Mollweide, a rectangle
    for cylindrical CRSs.

    Args:
        crs: Target CRS (EPSG int or proj4 string) — anything pyramids ``reproject_coordinates`` accepts.
        n: Longitude sample count (latitude uses ``n // 2``); higher = smoother boundary.

    Returns:
        ``(boundary_xy, (xmin, xmax), (ymin, ymax))`` — an ``(N, 2)`` closed ring + projected limits.

    Raises:
        ValueError: if every sphere sample projects to a non-finite coordinate, so the CRS has no finite
            projected domain (rather than letting ``min()``/``max()`` fail on an empty array).

    Examples:
        - The Web-Mercator domain is a rectangle whose limits are symmetric about 0:
            ```python
            >>> from digitalearth.scene import projections
            >>> ring, xlim, ylim = projections.projection_frame(3857, n=180)
            >>> ring.shape[1]
            2
            >>> bool(xlim[0] < 0 < xlim[1])
            True
            
            ```
    """
    lon = np.linspace(-180, 180, n)
    lat = np.linspace(-90, 90, max(n // 2, 2))
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    px, py = reproject_coordinates(grid_lon.ravel().tolist(), grid_lat.ravel().tolist(),
                                   from_crs=4326, to_crs=crs)
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    mask = np.isfinite(px) & np.isfinite(py)
    if not mask.any():
        raise ValueError(f"CRS {crs!r} has no finite projected domain over the sphere")
    pts = np.column_stack([px[mask], py[mask]])
    ring = _convex_hull(pts)
    return ring, (float(pts[:, 0].min()), float(pts[:, 0].max())), (float(pts[:, 1].min()), float(pts[:, 1].max()))


def graticule(crs: Any, lon_step: float = 30.0, lat_step: float = 30.0, dens: int = 200) -> List[np.ndarray]:
    """Return projected lon/lat grid polylines (meridians + parallels) for a CRS.

    Each meridian/parallel is densified in lon/lat, projected via pyramids, and split at non-finite points
    (the projection limb / antimeridian) so nothing wraps across the figure.

    Args:
        crs: Target CRS (EPSG int or proj4 string).
        lon_step: Spacing between meridians, degrees.
        lat_step: Spacing between parallels, degrees.
        dens: Points per line before splitting (higher = smoother curves).

    Returns:
        A list of ``(M, 2)`` projected polylines.

    Examples:
        - A Web-Mercator graticule yields several straight grid lines:
            ```python
            >>> from digitalearth.scene import projections
            >>> lines = projections.graticule(3857, lon_step=60, lat_step=30)
            >>> len(lines) > 0
            True
            >>> lines[0].shape[1]
            2
            
            ```
    """
    lines: List[np.ndarray] = []
    for lon in np.arange(-180, 180 + lon_step, lon_step):
        lat = np.linspace(-89.5, 89.5, dens)
        x, y = reproject_coordinates(np.full_like(lat, lon).tolist(), lat.tolist(), from_crs=4326, to_crs=crs)
        lines += _split_finite(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    for lat in np.arange(-90 + lat_step, 90, lat_step):
        lon = np.linspace(-180, 180, dens)
        x, y = reproject_coordinates(lon.tolist(), np.full_like(lon, lat).tolist(), from_crs=4326, to_crs=crs)
        lines += _split_finite(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return lines
