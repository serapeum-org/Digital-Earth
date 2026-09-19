"""Where a raster is placed: on the cells it has, not on the line through their centres (DE-37, #301).

An extent taken from the smallest and largest cell centre stops half a cell short on every side — the image is
one cell narrower and one shorter than the data, and each pixel is drawn at `(n - 1) / n` of its size. On
`examples/data/acc4000.tif`, 14 by 13 cells of 4,000 m, that was 2,000 m missing from each edge, in the static
`imshow` extent and in the web tier's image corners.

These cover the rule that replaces it, and the agreement between the tiers that read it.
"""

import numpy as np
import pytest

from digitalearth.base.sources import get_source
from digitalearth.base.spec import Bounds

RASTER = "examples/data/acc4000.tif"


@pytest.fixture(scope="module")
def grid():
    """Return a 3 x 3 raster of 10,000 m cells, its edges at 0 and 30,000.

    Returns:
        A pyramids `Dataset` whose centres are 5,000 / 15,000 / 25,000.
    """
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset

    return Dataset.from_array(
        np.arange(9, dtype="float32").reshape(3, 3),
        geo_ref=GeoReference(
            top_left_corner=(0.0, 30000.0), cell_size=10000.0, epsg=32618
        ),
    )


class TestTheRule:
    """`Bounds.cell_edges` — half the outermost spacing outward, on each side."""

    def test_the_cells_are_covered_rather_than_their_centres_joined(self, grid):
        """Three 10 km cells centred at 5/15/25 km cover 0 to 30 km."""
        source = get_source(grid)
        edges = Bounds.cell_edges(source.x.values, source.y.values, crs=grid.epsg)
        assert edges.as_bbox() == [0.0, 0.0, 30000.0, 30000.0], edges.as_bbox()

    def test_it_equals_the_raster_s_own_bbox(self):
        """The rule and pyramids agree on where a north-up raster is, to the metre."""
        from pyramids.dataset import Dataset

        dataset = Dataset.read_file(RASTER)
        source = get_source(dataset)
        edges = Bounds.cell_edges(source.x.values, source.y.values, crs=dataset.epsg)
        assert edges.as_bbox() == pytest.approx(list(dataset.bbox)), (
            f"{edges.as_bbox()} is not the raster's {list(dataset.bbox)}"
        )

    def test_it_still_equals_the_bbox_after_a_warp(self):
        """A reprojected raster is placed on its own warped cells, not on the centres of them."""
        from pyramids.dataset import Dataset

        warped = Dataset.read_file(RASTER).to_crs(3857)
        source = get_source(warped)
        edges = Bounds.cell_edges(source.x.values, source.y.values, crs=3857)
        assert edges.as_bbox() == pytest.approx(list(warped.bbox), rel=1e-9), (
            f"{edges.as_bbox()} is not the warped raster's {list(warped.bbox)}"
        )

    def test_a_descending_axis_covers_the_same_rectangle(self):
        """A north-up raster stores its rows top down; the rectangle is the same either way."""
        up = Bounds.cell_edges([5.0, 15.0, 25.0], [5.0, 15.0, 25.0], crs=4326)
        down = Bounds.cell_edges([5.0, 15.0, 25.0], [25.0, 15.0, 5.0], crs=4326)
        assert up.as_bbox() == down.as_bbox(), (up.as_bbox(), down.as_bbox())

    def test_a_one_cell_axis_is_widened_only_when_the_step_is_given(self):
        """A single cell has no spacing to read, so a width is given or nothing is invented."""
        line = Bounds.cell_edges([5.0], [5.0], crs=4326)
        widened = Bounds.cell_edges([5.0], [5.0], crs=4326, step=(10.0, 4.0))
        assert line.as_bbox() == [5.0, 5.0, 5.0, 5.0], line.as_bbox()
        assert widened.as_bbox() == [0.0, 3.0, 10.0, 7.0], widened.as_bbox()

    def test_a_raster_one_row_tall_covers_a_row_of_cells(self):
        """One row is a row of square cells, not a line of zero height.

        Test scenario:
            Both 2-D tiers adopted this rule without a `step=`, so a raster one cell tall was placed on a
            rectangle with no height and drawn invisible (review M4). The other axis says what a cell
            measures, which is the whole reason a raster has two axes.
        """
        edges = Bounds.cell_edges([0.5, 1.5, 2.5, 3.5, 4.5], [9.5], crs=4326)
        assert edges.as_bbox() == [0.0, 9.0, 5.0, 10.0], edges.as_bbox()

    def test_a_raster_one_column_wide_covers_a_column_of_cells(self):
        """The same, the other way round."""
        edges = Bounds.cell_edges([4.25], [52.0, 53.0, 54.0], crs=4326)
        assert edges.as_bbox() == [3.75, 51.5, 4.75, 54.5], edges.as_bbox()

    def test_a_given_step_still_wins_over_the_borrowed_one(self):
        """A caller who knows what a cell measures is not overruled by the other axis."""
        edges = Bounds.cell_edges([0.0, 2.0], [5.0], crs=4326, step=(2.0, 10.0))
        assert edges.as_bbox() == [-1.0, 0.0, 3.0, 10.0], edges.as_bbox()

    def test_an_irregular_grid_is_placed_by_its_own_outermost_cells(self):
        """Half of the first spacing at one end and half of the last at the other, not an average."""
        edges = Bounds.cell_edges([0.0, 2.0, 10.0], [0.0, 1.0, 2.0], crs=4326)
        assert edges.as_bbox() == [-1.0, -0.5, 14.0, 2.5], edges.as_bbox()

    def test_an_empty_axis_is_refused(self):
        """A raster with no cells has no rectangle, and says so rather than raising from numpy."""
        with pytest.raises(ValueError, match="needs coordinates on both axes"):
            Bounds.cell_edges([], [1.0, 2.0], crs=4326)

    def test_coordinates_that_are_not_numbers_name_the_axis(self):
        """numpy's own `TypeError` says what it could not convert, not which argument it came from."""
        with pytest.raises(ValueError, match="needs numbers for x"):
            Bounds.cell_edges("not coordinates", [1.0, 2.0], crs=4326)

    def test_a_step_that_is_not_a_pair_names_the_shape(self):
        """`step=4000.0` is the plausible mistake, and it is answered rather than indexed into."""
        with pytest.raises(ValueError, match=r"step must be a \(dx, dy\) pair"):
            Bounds.cell_edges([1.0], [1.0], crs=4326, step=4000.0)

    def test_the_crs_is_stored_as_given(self):
        """The rule places coordinates; it does not reproject them."""
        assert Bounds.cell_edges([1.0, 2.0], [1.0, 2.0], crs="EPSG:3857").crs == (
            "EPSG:3857"
        ), "the CRS is carried, not interpreted"


class TestTheTiersAgree:
    """One raster, three tiers, one extent."""

    def test_static_and_web_place_the_grid_on_the_same_rectangle(self, grid):
        """The two tiers that took their extent from the centres now cover the cells.

        Args:
            grid: The 3 x 3 raster whose cells span 0 to 30,000.

        Test scenario:
            The interactive tier was already right — HoloViews derives the edges from the centres itself — so
            it is the standard the other two are held to. Its `hv.Image` bounds are `(0, 0, 30000, 30000)`.
        """
        from digitalearth.static import Map

        source = get_source(grid)
        static_extent = Map._extent_of(source.x.values, source.y.values)
        assert static_extent == [0.0, 0.0, 30000.0, 30000.0], static_extent

    def test_the_web_corners_cover_the_cells(self, grid):
        """The image source's corners are the cell edges, so the map frames on the data.

        Args:
            grid: The 3 x 3 raster whose cells span 0 to 30,000.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        source = get_source(grid)
        corners = WebMap()._image_coordinates(source.x.values, source.y.values)
        assert corners == [
            [0.0, 30000.0],
            [30000.0, 30000.0],
            [30000.0, 0.0],
            [0.0, 0.0],
        ], corners

    def test_the_interactive_tier_was_already_placing_the_edges(self, grid):
        """Its bounds are the rule's answer, which is why they are what the others match.

        Args:
            grid: The 3 x 3 raster whose cells span 0 to 30,000.
        """
        pytest.importorskip("holoviews")
        from digitalearth.interactive import InteractiveMap

        m = InteractiveMap(crs=grid.epsg)
        m.image(grid)
        west, south, east, north = m.layers[0].bounds.lbrt()
        assert (west, south, east, north) == (0.0, 0.0, 30000.0, 30000.0), m.layers[
            0
        ].bounds.lbrt()
