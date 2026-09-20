"""The two clip helpers `voronoi` and `quadtree` hand their collect loops to.

Both were inline loops inside methods over the cognitive-complexity limit; extracting them made them
testable directly, which is worth doing because their interesting arm is one neither map builder reaches in
the ordinary case: a clip whose intersection is **not** a polygon.

That happens when a cell meets the clip boundary edge-on — the intersection is a line, which has no area and
no exterior ring to draw. Drawn anyway it would raise from `part.exterior`; counted anyway it would put a
value in the array with no polygon beside it, and every later value would be off by one.
"""

import numpy as np
from shapely.geometry import MultiPolygon, box

from digitalearth.static.maps.vector import _clipped_cell_boxes, _clipped_cell_rings


class TestAQuadtreeCellClippedToNothingDrawable:
    """`_clipped_cell_boxes` — the quadtree side."""

    def test_a_cell_meeting_the_boundary_edge_on_is_dropped(self):
        """The intersection is a line: no area, no ring, and no value to record.

        Test scenario:
            `box(0, 0, 1, 1)` against a boundary starting at x=1 intersects in the shared edge only. The
            result is a `LineString`, which is not empty — so an emptiness check alone would let it
            through to `part.exterior` and raise.
        """
        rings, values = _clipped_cell_boxes(
            [(0.0, 0.0, 1.0, 1.0, 5.0)], box(1.0, 0.0, 2.0, 1.0)
        )
        assert rings == [], f"a line has no ring to draw, got {rings}"
        assert values == [], "a dropped cell must not leave its value behind"

    def test_an_overlapping_cell_is_kept_with_its_value(self):
        """The guard must not drop the cells the helper exists to return."""
        rings, values = _clipped_cell_boxes(
            [(0.0, 0.0, 2.0, 2.0, 7.0)], box(1.0, 1.0, 3.0, 3.0)
        )
        assert len(rings) == 1, (
            f"one overlapping cell should give one ring, got {len(rings)}"
        )
        assert values == [7.0], f"the cell's value must follow its ring, got {values}"

    def test_without_a_boundary_a_cell_is_its_own_four_corners(self):
        """The unclipped path is a rectangle, not an intersection."""
        rings, values = _clipped_cell_boxes([(0.0, 0.0, 1.0, 1.0, 3.0)], None)
        assert values == [3.0], values
        assert rings[0].shape == (4, 2), f"expected four corners, got {rings[0].shape}"


class TestAVoronoiCellClippedToNothingDrawable:
    """`_clipped_cell_rings` — the voronoi side, which also carries the values by position."""

    def test_a_cell_meeting_the_boundary_edge_on_is_dropped(self):
        """Same geometry, and the same reason: a line carries no ring and no value."""
        cells = MultiPolygon([box(0.0, 0.0, 1.0, 1.0)])
        rings, values = _clipped_cell_rings(
            cells, box(1.0, 0.0, 2.0, 1.0), np.array([9.0]), "v"
        )
        assert rings == [], f"a line has no ring to draw, got {rings}"
        assert values == [], "a dropped cell must not leave its value behind"

    def test_a_kept_cell_carries_the_value_of_the_point_it_came_from(self):
        """`ordered=True` aligns cell *i* with point *i*; the helper must preserve that.

        Test scenario:
            Two cells with two values: whichever survives the clip has to take *its own* value, not the
            first one, which is what indexing by enumeration position rather than by output position buys.
        """
        cells = MultiPolygon([box(0.0, 0.0, 1.0, 1.0), box(5.0, 5.0, 6.0, 6.0)])
        rings, values = _clipped_cell_rings(
            cells, box(4.5, 4.5, 7.0, 7.0), np.array([10.0, 20.0]), "v"
        )
        assert len(rings) == 1, f"only the second cell overlaps, got {len(rings)} rings"
        assert values == [20.0], (
            f"the surviving cell must keep its own value, got {values}"
        )

    def test_an_outline_only_draw_records_no_values_at_all(self):
        """`column=None` means the caller asked for outlines; there is nothing to colour by."""
        cells = MultiPolygon([box(0.0, 0.0, 1.0, 1.0)])
        rings, values = _clipped_cell_rings(cells, None, None, None)
        assert len(rings) == 1, (
            f"the cell is unclipped and should be kept, got {len(rings)}"
        )
        assert values is None, f"an outline-only draw has no values, got {values}"
