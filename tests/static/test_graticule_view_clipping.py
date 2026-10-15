"""A graticule's lines are cut to the view the map is framed on (#393).

Before #393 the lines spanned the whole world whatever the map was looking at, so only the degree *labels*
tracked the view — the collection always held every world-phase line the step reached. Two things followed
from that, both wrong for a reference grid:

- a fine step was refused for the whole world's line count even when the map was framed on a small region
  that would only ever show a handful of them — ``Map(crs=4326).set_bounds([4, 50, 6, 52]).graticule(0.1)``
  was told it would draw ~3,600 lines and refused, though the 2x2-degree view holds about forty;
- a grid drawn on the world and then narrowed kept all its world-spanning lines on the axes, relying on
  matplotlib to clip the off-view ones at draw time rather than not cutting them at all.

Since #393 the line generation and the line budget both read the framed view: a step is budgeted against the
degrees the view spans, and the lines are cut to it. An **unframed or global** view still budgets against
the whole world, so a genuinely unbounded step is refused as before. These tests pin both halves — the fine
step a small view now accepts, and the world step every view still refuses — and the re-cut on reframe.
"""

import pytest

from digitalearth.static import Map
from digitalearth.static.maps.projection import MAX_GRATICULE_LINES

#: A small framed region, as ``(west, south, east, north)`` in EPSG:4326: 2 degrees of longitude by 2 of
#: latitude, far too narrow to hold anything near the world's line count at a fine step.
SMALL_VIEW = [4.0, 50.0, 6.0, 52.0]

#: A step whose whole-world grid (``360 / 0.1`` meridians) is well past :data:`MAX_GRATICULE_LINES`, so an
#: unframed view refuses it — but whose count across :data:`SMALL_VIEW` (``~2 / 0.1``) is a handful.
FINE_STEP = 0.1


@pytest.fixture
def small():
    """Yield a flat lon/lat map framed on :data:`SMALL_VIEW`, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` framed on a 2x2-degree region.
    """
    scene = Map(crs=4326)
    scene.set_bounds(SMALL_VIEW)
    yield scene
    scene.close()


class TestAFineStepDrawsOnASmallView:
    """``graticule(0.1)`` on a 2x2-degree view draws the lines that view holds, not the world's."""

    def test_it_is_not_refused(self, small):
        """A step the view can show is accepted, where the same step on the world is refused.

        Args:
            small: A flat lon/lat map framed on :data:`SMALL_VIEW`.

        Test scenario:
            ``0.1`` builds ~3,600 world meridians, past the budget — but the 2-degree view spans ~20 of
            them, so the budget (now the view's) lets it through rather than refusing it outright.
        """
        small.graticule(spacing=FINE_STEP)
        assert small._graticule_id in small.layer_ids, (
            "a fine step the framed view can show should have drawn a graticule, not been refused"
        )

    def test_it_draws_only_the_lines_the_view_holds(self, small):
        """And the collection holds a handful of lines, nowhere near the whole world's count.

        Args:
            small: A flat lon/lat map framed on :data:`SMALL_VIEW`.

        Test scenario:
            A 2x2-degree view at ``0.1`` holds about forty lines; the whole-world grid would be ~5,400. The
            exact count depends on the one-step margin the cut keeps, so this asserts the order of magnitude
            the clip guarantees rather than a brittle exact number.
        """
        small.graticule(spacing=FINE_STEP)
        drawn = len(small.artist(small._graticule_id))
        assert 0 < drawn < MAX_GRATICULE_LINES, (
            f"the fine grid on a small view should hold a handful of lines; got {drawn}"
        )


class TestTheSameStepIsRefusedWithoutAFramedView:
    """An unframed or global view budgets against the whole world, so a fine step is refused there."""

    def test_an_unframed_view_refuses_it(self):
        """A bare ``Map`` has not framed the axes, so the budget is the world's and the step is refused.

        Test scenario:
            The world grid at ``0.1`` is ~3,600 meridians, past :data:`MAX_GRATICULE_LINES`, so the refusal
            names the keyword — the behaviour #393 keeps for the one view that really would draw them all.
        """
        scene = Map(crs=4326)
        try:
            with pytest.raises(ValueError, match=r"graticule\(\) was given spacing="):
                scene.graticule(spacing=FINE_STEP)
        finally:
            scene.close()

    def test_a_global_view_refuses_it(self):
        """``set_global`` frames the whole world, so the budget is the world's and the step is refused.

        Test scenario:
            The complement of the small-view case: the same step, the same map, but framed on everything —
            which is exactly the view whose line count is the world's.
        """
        scene = Map(crs=4326)
        scene.set_global()
        try:
            with pytest.raises(ValueError, match=r"graticule\(\) was given spacing="):
                scene.graticule(spacing=FINE_STEP)
        finally:
            scene.close()


class TestReframingRecutsTheGrid:
    """A grid drawn on one view is re-cut when a later framing call moves the view."""

    def test_narrowing_the_view_drops_lines(self):
        """Framing a world grid onto a small region re-cuts it to the lines that region holds.

        Test scenario:
            ``graticule(30)`` on the world draws 18 lines; ``set_bounds`` onto :data:`SMALL_VIEW` re-cuts
            the collection to the few 30-degree lines the 2-degree view still reaches, fewer than 18.
        """
        scene = Map(crs=4326)
        scene.set_global()
        scene.graticule(spacing=30.0)
        before = len(scene.ax.collections[0].get_segments())
        scene.set_bounds(SMALL_VIEW)
        after = len(scene.ax.collections[0].get_segments())
        scene.close()
        assert after < before, (
            f"narrowing the view should have re-cut the grid below its {before} lines; got {after}"
        )

    def test_a_bare_grid_is_recut_too(self):
        """A ``labels=False`` grid is re-cut on reframe even though it carries no degrees.

        Test scenario:
            Its lines depend on the view whether or not they are labelled, so framing has to regenerate
            them — the half of :meth:`~digitalearth.static.maps.projection.ProjectionMixin._relabel_graticule`
            #393 widened from labelled grids to every grid.
        """
        scene = Map(crs=4326)
        scene.set_global()
        scene.graticule(spacing=30.0, labels=False)
        before = len(scene.ax.collections[0].get_segments())
        scene.set_bounds(SMALL_VIEW)
        after = len(scene.ax.collections[0].get_segments())
        scene.close()
        assert after < before, (
            f"a bare grid should also be re-cut when the view narrows; stayed at {before} -> {after}"
        )
