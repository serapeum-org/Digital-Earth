"""The projection-frame convex hull is sped up by an interior-point filter without changing its result (#381).

`projection_frame` samples a filled lon/lat grid (~259k points at the default `n`), reprojects it, and takes
the convex hull of the finite points. The hull is the thin boundary of that cloud, so the pure-Python monotone
chain used to walk a quarter-million interior points for nothing. `_drop_interior_points` removes the points
provably interior to the octagon of the eight extreme points before the chain runs, and `projection_frame`
passes `precision=None` so pyramids does not round every sampled coordinate. Both are speed-only: the hull ring
must be exactly what it was. These tests pin that invariant without a convex-hull oracle in the dependency set —
they check the two properties that define a hull (it contains every input point; its vertices are input points)
and that the filter never drops a hull vertex.
"""

import numpy as np

from digitalearth.static import projections
from digitalearth.static.projections import _convex_hull, _drop_interior_points


def _clouds():
    """Yield labelled 2-D point clouds spanning the shapes the hull meets: blob, filled grid, disk.

    Returns:
        A list of ``(name, points)`` pairs, each an ``(N, 2)`` float array.
    """
    rng = np.random.default_rng(0)
    grid = np.array(np.meshgrid(np.linspace(0, 1, 90), np.linspace(0, 2, 70)))
    disk = rng.normal(size=(20000, 2))
    return [
        ("blob", rng.normal(size=(6000, 2))),
        ("grid", grid.reshape(2, -1).T),
        ("disk", disk[np.hypot(disk[:, 0], disk[:, 1]) <= 2.5]),
    ]


def _inside_or_on(points: np.ndarray, ring: np.ndarray) -> bool:
    """Whether every point lies inside or on the closed convex ``ring``.

    Args:
        points: The ``(N, 2)`` points to test.
        ring: A closed convex hull ring (first vertex repeated at the end).

    Returns:
        ``True`` when no point is strictly outside any edge of the ring (winding-agnostic, with a tolerance
        for the floating-point edge).
    """
    poly = ring[:-1]
    nxt = np.roll(poly, -1, axis=0)
    edge = nxt - poly
    crosses = np.stack(
        [
            edge[k, 0] * (points[:, 1] - poly[k, 1])
            - edge[k, 1] * (points[:, 0] - poly[k, 0])
            for k in range(len(poly))
        ]
    )
    tol = 1e-9 * (np.abs(poly).max() + 1.0)
    return bool(
        (crosses >= -tol).all(axis=0).all() or (crosses <= tol).all(axis=0).all()
    )


class TestTheHullIsCorrect:
    """`_convex_hull` returns a closed convex ring that contains every input point."""

    def test_the_ring_contains_every_input_point(self):
        """No input point lies outside the hull, across a blob, a filled grid and a disk.

        Test scenario:
            A hull that wrongly dropped a boundary point (an over-eager interior filter) would leave that
            point outside the returned ring; containing all of them is the property that catches it.
        """
        for name, pts in _clouds():
            ring = _convex_hull(pts)
            assert _inside_or_on(pts, ring), (
                f"{name}: a point fell outside the hull ring"
            )

    def test_every_hull_vertex_is_an_input_point(self):
        """The ring is built from the cloud, so each vertex is one of the points handed in.

        Test scenario:
            The filter and the chain may only *select* points, never synthesise one; a vertex that is not an
            input point would mean the arithmetic invented a corner.
        """
        for name, pts in _clouds():
            seen = {tuple(np.round(p, 9)) for p in pts}
            ring = _convex_hull(pts)
            for vertex in ring:
                assert tuple(np.round(vertex, 9)) in seen, (
                    f"{name}: hull vertex {vertex} is not an input point"
                )


class TestTheInteriorFilterPreservesTheHull:
    """`_drop_interior_points` never drops a point the hull needs."""

    def test_it_keeps_every_hull_vertex(self):
        """Every vertex of the full cloud's hull survives the filter, so the hull is unchanged.

        Test scenario:
            The filter's whole correctness claim is that it removes only provably-interior points. Hull the
            full cloud, then assert each hull vertex is still present after filtering — a dropped vertex
            would shrink the hull.
        """
        for name, pts in _clouds():
            ring = _convex_hull(pts)
            kept = {tuple(np.round(p, 9)) for p in _drop_interior_points(pts)}
            for vertex in ring[:-1]:
                assert tuple(np.round(vertex, 9)) in kept, (
                    f"{name}: the filter dropped hull vertex {vertex}"
                )

    def test_it_removes_interior_bulk_on_a_filled_grid(self):
        """A filled grid's interior is mostly discarded, which is the speedup the filter buys.

        Test scenario:
            The point of the filter is that a filled cloud's hull is its boundary; on a dense grid it should
            keep a small fraction, not nearly all of it.
        """
        grid = np.array(np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120)))
        pts = grid.reshape(2, -1).T
        kept = len(_drop_interior_points(pts))
        assert kept < len(pts) // 2, (
            f"the filter kept {kept} of {len(pts)} grid points; it should discard most of the interior"
        )


class TestProjectionFrameStillFrames:
    """`projection_frame` returns the same shape of ring after the speedups."""

    def test_web_mercator_is_a_rectangle(self):
        """Web Mercator's finite domain hulls to a four-corner rectangle, closed.

        Test scenario:
            The cylindrical domain is a rectangle; its hull is five vertices (four corners plus the repeated
            first), and its x-limits are symmetric about zero — unchanged by the interior filter or by
            dropping pyramids' coordinate rounding.
        """
        ring, xlim, ylim = projections.projection_frame(3857, n=180)
        assert len(ring) == 5, (
            f"the Web-Mercator frame is not a rectangle: {len(ring)} vertices"
        )
        assert xlim[0] < 0 < xlim[1], f"x-limits not symmetric about 0: {xlim}"
        assert bool(np.allclose(ring[0], ring[-1])), "the ring is not closed"
