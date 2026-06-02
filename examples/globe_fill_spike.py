"""T1 spike for Digital-Earth#43 — clip a filled polygon to a projection boundary (no network).

Decision recorded here (the spike's deliverable):

- **Approach B (densify -> reproject -> re-close at the boundary), pure numpy.** Matches the DC.2/DC.3
  precedent (``reproject_coordinates`` + numpy, no shapely in digitalearth) and needs no upstream change.
  Option A (clip the spherical cap in lon/lat) is cleaner topologically but the ortho cap-clip is fiddly;
  Option C (geopandas ``.clip``) uses geopandas as a GIS engine, which the two-engine rule discourages.
- **Fill drawing:** cleopatra ``PolygonGlyph`` fills only when given per-polygon *values*
  (``facecolors="none"`` otherwise), so a uniform land/ocean fill is drawn as a map-specific overlay
  directly on the axes with a matplotlib ``PolyCollection(facecolors=color)`` — exactly like the existing
  globe coastline ``ax.plot`` overlay. cleopatra ``apply_projection_frame(clip_artists=True)`` then clips
  the collection to the boundary path, so the fill is visually confined to the disc for free.
- **Closure heuristic (v1):** each visible finite run is closed along the **shorter** boundary arc between
  its endpoints. Correct when a polygon's visible span is < half the limb (true for most continents on a
  hemisphere view); a continent spanning > half the limb may mis-close — documented as a known v1 limit and
  a follow-up. The boundary clip keeps every result inside the disc regardless.
- **Ocean shortcut:** ``ocean()`` on a globe fills the whole boundary disc with the ocean colour (no
  per-polygon clip) and lets land overlay it — exact and cheap.

Run: ``pixi run -e dev --frozen --no-install python examples/globe_fill_spike.py``
"""
import numpy as np

from digitalearth.scene import projections


def _nearest_boundary_index(boundary_open: np.ndarray, pt: np.ndarray) -> int:
    """Index of the boundary vertex nearest ``pt`` (projected coords)."""
    d = np.hypot(boundary_open[:, 0] - pt[0], boundary_open[:, 1] - pt[1])
    return int(np.argmin(d))


def _boundary_arc(boundary_open: np.ndarray, i: int, j: int) -> np.ndarray:
    """Shorter cyclic arc of boundary vertices from index ``i`` to ``j`` (inclusive)."""
    n = len(boundary_open)
    fwd = (j - i) % n
    bwd = (i - j) % n
    if fwd <= bwd:
        idx = [(i + k) % n for k in range(fwd + 1)]
    else:
        idx = [(i - k) % n for k in range(bwd + 1)]
    return boundary_open[idx]


def close_visible_runs(x: np.ndarray, y: np.ndarray, boundary: np.ndarray) -> list:
    """Turn reprojected ring coords into finite, closed, limb-clipped rings (Approach B, v1)."""
    runs = projections._split_finite(x, y)
    fully = bool((np.isfinite(x) & np.isfinite(y)).all())
    boundary_open = boundary[:-1]
    out = []
    for run in runs:
        if fully:
            ring = np.vstack([run, run[:1]])
        else:
            i_end = _nearest_boundary_index(boundary_open, run[-1])
            i_start = _nearest_boundary_index(boundary_open, run[0])
            arc = _boundary_arc(boundary_open, i_end, i_start)
            ring = np.vstack([run, arc, run[:1]])
        out.append(ring)
    return out


def main() -> None:
    """Validate the closure on a synthetic limb-crossing polygon against the real ortho boundary."""
    crs = projections.orthographic(lon=0, lat=0)
    boundary, xlim, ylim = projections.projection_frame(crs)

    # A big lon/lat quad straddling the limb of an ortho centred at (0, 0): spans 60E..120E, so its
    # eastern half is on the far side (reprojects to inf).
    lons = np.linspace(60, 120, 60)
    ring_lonlat = np.vstack([
        np.column_stack([lons, np.full_like(lons, -40.0)]),
        np.column_stack([lons[::-1], np.full_like(lons, 40.0)]),
    ])
    x, y = projections.reproject_coordinates(ring_lonlat[:, 0].tolist(), ring_lonlat[:, 1].tolist(),
                                             from_crs=4326, to_crs=crs)
    x, y = np.asarray(x, float), np.asarray(y, float)

    n_inf = int((~np.isfinite(x)).sum())
    rings = close_visible_runs(x, y, boundary)
    allv = np.vstack(rings) if rings else np.zeros((0, 2))

    print(f"boundary verts: {len(boundary)}  xlim={xlim}  ylim={ylim}")
    print(f"ring vertices: {len(x)}  far-side(inf): {n_inf}")
    print(f"closed rings: {len(rings)}  total verts: {len(allv)}")
    print(f"all finite: {np.isfinite(allv).all()}")
    print(f"rings closed: {all(np.allclose(r[0], r[-1]) for r in rings)}")
    radius = max(xlim[1], ylim[1])
    within = bool((np.hypot(allv[:, 0], allv[:, 1]) <= radius * 1.001).all()) if len(allv) else True
    print(f"within boundary disc: {within}")

    assert n_inf > 0, "test polygon should straddle the limb"
    assert rings, "expected at least one closed visible ring"
    assert np.isfinite(allv).all(), "closure must not emit inf/nan"
    assert all(np.allclose(r[0], r[-1]) for r in rings), "rings must be closed"
    assert within, "closed rings must stay within the projection disc"
    print("\nSPIKE OK — Approach B produces finite, closed, in-disc rings.")


if __name__ == "__main__":
    main()
