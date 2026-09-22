"""`large_image` reads through the data tier, one window at a time (IN-6, #297).

`SourceView.reread` existed and nothing called it: the builder drove its own loop, warping the whole raster up
front, sizing a square canvas whatever the map's shape, labelling each frame with the bbox it asked for rather
than the one it got, and letting a nodata sentinel of -3.4e38 into the colour range. These cover the read that
replaces it — the request, the canvas, the masking, and what a window off the edge of the data draws.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")

RASTER = "examples/data/acc4000.tif"


class _RecordingCOG:
    """A pyramids-Dataset-like raster that records every window it is asked for."""

    epsg = 3857
    bbox = (-1.0e6, -1.0e6, 1.0e6, 1.0e6)
    cell_size = 1.0e4
    #: Origin and cell size, as pyramids reports them, so a window snaps to this raster's own pixel edges.
    geotransform = (-1.0e6, 1.0e4, 0.0, 1.0e6, 0.0, -1.0e4)
    no_data_value = (-9999.0,)
    scale = [1.0]
    offset = [0.0]

    def __init__(self):
        """Start with no reads recorded."""
        self.read_calls = []

    def preview(self, *, max_size=1024, band=1):
        """Return a square overview, and record nothing — nothing should call this any more.

        Args:
            max_size: The side asked for.
            band: The 0-based band.

        Returns:
            A random square.
        """
        raise AssertionError("large_image must not read through preview() any more")

    def read_array(self, *, band=0, bbox=None, masked=False):
        """Return a window at full resolution, masked as pyramids masks one.

        Args:
            band: The 0-based band.
            bbox: The window, in this raster's CRS.
            masked: Whether nodata cells come back masked.

        Returns:
            A masked array of ones with one missing cell.
        """
        west, south, east, north = bbox if bbox is not None else self.bbox
        columns = max(1, int(round((east - west) / self.cell_size)))
        rows = max(1, int(round((north - south) / self.cell_size)))
        self.read_calls.append(
            (tuple(bbox) if bbox else self.bbox, columns, rows, band)
        )
        values = np.ones((rows, columns))
        values[0, 0] = self.no_data_value[0]
        if not masked:
            return values
        return np.ma.masked_equal(values, self.no_data_value[0])

    def read_part(self, *, bbox, dst_width, dst_height, bbox_crs=4326, band=1):
        """Return a window, recording what was asked for.

        Args:
            bbox: The window, in `bbox_crs`.
            dst_width: Cells across the canvas.
            dst_height: Cells down it.
            bbox_crs: The CRS the window is given in.
            band: The 0-based band.

        Returns:
            A canvas of ones, with one nodata cell.
        """
        self.read_calls.append((bbox, dst_width, dst_height, band))
        values = np.ones((dst_height, dst_width))
        values[0, 0] = self.no_data_value[0]
        return values


@pytest.fixture
def cog() -> _RecordingCOG:
    """Return a recording raster.

    Returns:
        The fake COG.
    """
    return _RecordingCOG()


class TestTheReadRequest:
    """Every frame is a request built by `RenderTarget`, answered by `SourceView`."""

    def test_the_first_frame_windows_the_whole_raster(self, cog):
        """A canvas with no region reads the source's own extent.

        Args:
            cog: The recording raster.
        """
        InteractiveMap(crs=3857).large_image(cog, dynamic=False, max_pixels=64 * 64)
        assert cog.read_calls[-1][0] == pytest.approx(cog.bbox), cog.read_calls

    def test_the_canvas_follows_the_map_rather_than_a_square(self, cog):
        """A wide map reads a wide window, which a square canvas could not express.

        Args:
            cog: The recording raster.

        Test scenario:
            The builder sized `side = max(64, sqrt(max_pixels))` and read `side x side`, so a 800x300 map
            read a square and threw half of it away.
        """
        wide = InteractiveMap(crs=3857, width=800, height=300)
        wide.large_image(cog, dynamic=False, max_pixels=200 * 200)
        _, width, height, _ = cog.read_calls[-1]
        assert width > height, (
            f"a wide map must read a wide canvas, got {width}x{height}"
        )
        assert width * height <= 200 * 200, f"{width}x{height} exceeds the budget"

    def test_a_viewport_event_reads_the_window_it_asks_for(self, cog):
        """A `RangeXY` event reads its own window, in the map's CRS.

        Args:
            cog: The recording raster.
        """
        from holoviews.streams import RangeXY

        m = InteractiveMap(crs=3857)
        m.large_image(cog, dynamic=True, max_pixels=100 * 100)
        stream = next(s for s in m.layers[0].streams if isinstance(s, RangeXY))
        stream.event(x_range=(-5.0e5, 5.0e5), y_range=(-4.0e5, 4.0e5))
        m.layers[0][()]
        assert cog.read_calls[-1][0] == pytest.approx((-5.0e5, -4.0e5, 5.0e5, 4.0e5)), (
            cog.read_calls[-1]
        )

    def test_a_window_inset_by_less_than_a_cell_is_read_on_pixel_edges(self, cog):
        """The window is snapped outward before the read, so the frame is labelled where it was read.

        Args:
            cog: The recording raster, whose cells are 10 km.

        Test scenario:
            `read_part` snaps the window to source pixels internally and returns the snapped buffer, so
            labelling the result with the requested bbox misplaced every cell by up to one pixel per edge.
        """
        from holoviews.streams import RangeXY

        m = InteractiveMap(crs=3857)
        m.large_image(cog, dynamic=True, max_pixels=64 * 64)
        stream = next(s for s in m.layers[0].streams if isinstance(s, RangeXY))
        stream.event(x_range=(-1.0e6 + 1000.0, 0.0), y_range=(-1.0e6 + 1000.0, 0.0))
        m.layers[0][()]
        west, south, east, north = cog.read_calls[-1][0]
        assert (west, south) == pytest.approx((-1.0e6, -1.0e6)), (
            f"the window must snap outward to the cell edge, got {(west, south)}"
        )

    def test_a_dynamic_layer_reads_nothing_until_a_frame_is_drawn(self, cog):
        """A `DynamicMap` is lazy, so building one must not pay for a read nobody looks at.

        Args:
            cog: The recording raster.

        Test scenario:
            The drawer called its frame function once before returning the `DynamicMap`, to file the style
            against the layer, and threw the frame away — a full window read per dynamic layer, at build
            time, of a raster this builder exists to avoid reading whole (review M8).
        """
        m = InteractiveMap(crs=3857)
        try:
            m.large_image(cog, dynamic=True, max_pixels=64 * 64)
            assert cog.read_calls == [], cog.read_calls
        finally:
            m.close()


class TestWhatTheFrameShows:
    """The frame is drawn the way `image()` draws one, and says what is missing."""

    def test_the_frame_is_styled_like_any_other_layer(self, cog):
        """`style_of` records the colormap, which it could not before.

        Args:
            cog: The recording raster.
        """
        m = InteractiveMap(crs=3857, width=700, height=400, title="Flow")
        m.large_image(cog, dynamic=False, cmap="magma")
        style = m.style_of(0)
        assert style["common"]["cmap"] == "magma", style

    def test_the_map_s_frame_reaches_the_element(self, cog):
        """Width, height and title are the map's, as they are for `image()`.

        Args:
            cog: The recording raster.
        """
        m = InteractiveMap(crs=3857, width=700, height=400, title="Flow")
        m.large_image(cog, dynamic=False)
        frame = m.style_of(0)["bokeh"]
        assert (frame["width"], frame["height"], frame["title"]) == (
            700,
            400,
            "Flow",
        ), frame

    def test_a_dynamic_layer_records_the_style_its_frames_are_drawn_with(self, cog):
        """The default path is `dynamic=True`, and its style was readable on neither the map nor the frame.

        Args:
            cog: The recording raster.

        Test scenario:
            `_styled` filed the style against the *frame*, while the layer a caller holds is the
            `DynamicMap`. `style_of` therefore answered `{}` for the tier's own big-raster builder, and the
            dashboard's widgets — which read it to merge over — passed the layer by (review M17).
        """
        m = InteractiveMap(crs=cog.epsg)
        m.large_image(cog, cmap="magma")
        m.layers[0][()]  # draw the first frame, as a renderer would
        assert m.style_of(0)["common"]["cmap"] == "magma", m.style_of(0)

    def test_a_dynamic_layer_s_style_is_readable_before_any_frame_is_drawn(self, cog):
        """The style is filed against the layer itself, so no frame has to be drawn to file it.

        Args:
            cog: The recording raster.

        Test scenario:
            Dropping the discarded build-time frame must not bring back review M17, where `style_of` answered
            `{}` for a dynamic layer until something rendered it.
        """
        m = InteractiveMap(crs=cog.epsg)
        try:
            m.large_image(cog, cmap="magma")
            assert m.style_of(0)["common"]["cmap"] == "magma", m.style_of(0)
        finally:
            m.close()

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_the_drawn_record_carries_the_style_that_was_applied(self, cog, dynamic):
        """`DrawnLayer.style` is what the drawer applied, as it is for every other drawer.

        Args:
            cog: The recording raster.
            dynamic: Whether the layer is one frame or a `DynamicMap`.

        Test scenario:
            The record held only the caller's extra options — `{}` for a plain call — although the frames
            were drawn with a colormap and a colorbar (review L8).
        """
        m = InteractiveMap(crs=cog.epsg)
        try:
            m.large_image(cog, cmap="magma", dynamic=dynamic)
            style = m._renderer.drawn[m.layer_ids[0]].style
            assert style["cmap"] == "magma", style
        finally:
            m.close()

    def test_a_later_layer_does_not_overwrite_a_dynamic_layer_s_style(self, dataset):
        """A frame says which layer it belongs to; it used to be guessed from the last one registered.

        Args:
            dataset: A real raster, since the static builder reads one directly.

        Test scenario:
            Every builder styles its element *before* registering it, so while a builder styles, the last
            registered layer is still the previous one. A static layer drawn after a dynamic one therefore
            filed its style against the dynamic layer — and the dashboard merges widget values over
            `style_of`, so layer 0 was redrawn in layer 1's colours: the #300 defect again (review H3).
        """
        m = InteractiveMap(crs=dataset.epsg)
        m.large_image(dataset, cmap="magma")
        m.layers[0][()]  # draw a frame, as a renderer would
        m.image(dataset, cmap="Blues")
        assert m.style_of(0)["common"]["cmap"] == "magma", m.style_of(0)
        assert m.style_of(1)["common"]["cmap"] == "Blues", m.style_of(1)

    def test_two_dynamic_layers_keep_their_own_styles(self, cog):
        """Each frame names its own owner, so one dynamic layer cannot answer for another.

        Args:
            cog: The recording raster.
        """
        m = InteractiveMap(crs=cog.epsg)
        m.large_image(cog, cmap="magma")
        m.large_image(cog, cmap="cividis")
        m.layers[0][()]
        m.layers[1][()]
        assert m.style_of(0)["common"]["cmap"] == "magma", m.style_of(0)
        assert m.style_of(1)["common"]["cmap"] == "cividis", m.style_of(1)

    def test_panning_does_not_grow_the_style_table(self, cog):
        """Every frame is a new object, and the table was keyed by `id()` of the ones already gone.

        Args:
            cog: The recording raster.

        Test scenario:
            One dead entry per pan, keyed by the id of a collected object — which another object may later
            be handed, making a stale style answer for it (review M18).
        """
        m = InteractiveMap(crs=cog.epsg)
        m.large_image(cog, cmap="magma")
        dmap = m.layers[0]
        west, south, east, north = cog.bbox
        for step in range(8):
            dmap[(west + step, east - step), (south + step, north - step)]
        assert len(m._styles) <= len(m.layers) + 1, len(m._styles)

    def test_a_canvas_of_one_cell_is_still_placed_on_what_it_read(self, cog):
        """A budget small enough to read one cell has no spacing for a renderer to derive.

        Args:
            cog: The recording raster.

        Test scenario:
            HoloViews infers an image's placement from the gaps between its coordinates. Given one sample
            on an axis there are no gaps: it returned `nan` bounds, and raised
            `ValueError: cannot convert float NaN to integer` as soon as the other axis had two. The read
            knows the rectangle, so it carries it.
        """
        m = InteractiveMap(crs=cog.epsg)
        m.large_image(cog, dynamic=False, max_pixels=4)
        import math

        drawn = m.layers[0].bounds.lbrt()
        assert not any(math.isnan(edge) for edge in drawn), f"nan bounds: {drawn}"
        assert [round(float(edge), 3) for edge in drawn] == [
            round(float(edge), 3) for edge in cog.bbox
        ], f"{drawn} is not the raster's {cog.bbox}"

    @pytest.mark.parametrize(
        "label, x, y, expected",
        [
            (
                "north-up",
                [0.0, 1.0, 2.0],
                [9.0, 0.0],
                [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
            ),
            (
                "south-up",
                [0.0, 1.0, 2.0],
                [0.0, 9.0],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ),
            (
                "east-left",
                [2.0, 1.0, 0.0],
                [9.0, 0.0],
                [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
            ),
        ],
    )
    def test_the_axes_keep_their_direction(self, label, x, y, expected):
        """A grid's axes carry which way it runs, and the placement must not throw that away.

        Args:
            label: Which orientation is under test.
            x: The x cell centres, ascending or descending.
            y: The y cell centres.
            expected: What the element should hold, row 0 first.

        Test scenario:
            `hv.Image(arr, bounds=...)` assumes row 0 is north and column 0 is west, so placing every
            windowed read by its bounds mirrored a south-up or east-left raster — silently, because a
            flipped raster draws perfectly happily (review H4). Bounds are the fallback for the one case
            the axes cannot answer, which is a single cell.
        """
        import numpy as np

        m = InteractiveMap(crs=4326)
        element = m._raster_element(
            np.array(x),
            np.array(y),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "v",
            bounds=[0.0, 0.0, 2.0, 9.0],
        )
        drawn = element.dimension_values(2, flat=False).tolist()
        assert drawn == expected, f"{label}: {drawn}"

    def test_a_one_cell_canvas_does_not_raise(self, cog):
        """The budget below that one drew nothing and said nothing; it is a picture now.

        Args:
            cog: The recording raster.
        """
        m = InteractiveMap(crs=cog.epsg)
        m.large_image(cog, dynamic=False, max_pixels=1)
        values = m.layers[0].dimension_values(2, flat=False)
        assert values.shape == (1, 1), values.shape

    def test_a_nodata_cell_does_not_reach_the_colour_range(self, cog):
        """The sentinel is `NaN` in the frame, not -9999 at the end of the ramp.

        Args:
            cog: The recording raster, whose first cell is nodata.
        """
        m = InteractiveMap(crs=3857)
        m.large_image(cog, dynamic=False, max_pixels=32 * 32)
        values = np.asarray(
            m.layers[0].dimension_values(2, flat=False), dtype="float64"
        )
        assert np.isnan(values).any(), "the nodata cell must be missing, not a number"
        assert np.nanmin(values) == 1.0, (
            f"the sentinel reached the range: {np.nanmin(values)}"
        )

    def test_a_real_raster_keeps_the_missing_cells_image_keeps(self):
        """A whole-raster frame of `acc4000.tif` has the 93 nodata cells `image()` has.

        Test scenario:
            The defect measured in the issue: the static frame had 0 NaN and a minimum of -3.4e38, because a
            windowed read returns a plain array and nothing masked it.
        """
        from pyramids.dataset import Dataset

        dataset = Dataset.read_file(RASTER)
        m = InteractiveMap(crs=dataset.epsg)
        m.large_image(dataset, dynamic=False, max_pixels=80 * 80)
        values = np.asarray(
            m.layers[0].dimension_values(2, flat=False), dtype="float64"
        )
        assert int(np.isnan(values).sum()) == 93, (
            f"a whole-raster frame must miss 93 cells, got {int(np.isnan(values).sum())}"
        )

    def test_a_packed_band_is_masked_too(self, tmp_path):
        """A band whose nodata unpacks to an ordinary number is still missing in the frame.

        Args:
            tmp_path: Where the packed raster is written.

        Test scenario:
            DE-6's trap: comparing an unpacked value against the stored sentinel misses every nodata cell of
            a CF-packed band, which is why a full-resolution window is masked by pyramids instead.
        """
        from pyramids.base.georeference import GeoReference
        from pyramids.dataset import Dataset

        packed = Dataset.from_array(
            np.array([[100, 200], [300, -9999]], dtype="int16"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            no_data_value=-9999,
        )
        path = tmp_path / "packed.tif"
        packed.to_file(str(path))
        m = InteractiveMap(crs=4326)
        m.large_image(Dataset.read_file(str(path)), dynamic=False, max_pixels=64 * 64)
        values = np.asarray(
            m.layers[0].dimension_values(2, flat=False), dtype="float64"
        )
        assert int(np.isnan(values).sum()) == 1, f"the packed nodata cell: {values}"

    def test_a_window_off_the_raster_draws_an_empty_frame(self):
        """Panning off the data draws nothing, rather than raising inside the callback.

        Test scenario:
            `OutOfBoundsError` reached the `DynamicMap` callback, which is a traceback in a notebook cell
            where the map should simply show empty space.
        """
        from holoviews.streams import RangeXY
        from pyramids.dataset import Dataset

        dataset = Dataset.read_file(RASTER)
        m = InteractiveMap(crs=dataset.epsg)
        m.large_image(dataset, dynamic=True, max_pixels=64 * 64)
        stream = next(s for s in m.layers[0].streams if isinstance(s, RangeXY))
        stream.event(x_range=(9.0e6, 9.1e6), y_range=(9.0e6, 9.1e6))
        values = np.asarray(
            m.layers[0][()].dimension_values(2, flat=False), dtype="float64"
        )
        assert np.isnan(values).all(), "a window off the data must draw an empty frame"


class TestWarpingOnlyTheWindow:
    """A raster in another CRS is warped a window at a time, not once in full."""

    def test_a_raster_in_another_crs_is_read_through_a_warped_view(self, mocker):
        """`warped_view` is lazy; `to_crs` warps the whole raster at full resolution first.

        Args:
            mocker: Records which pyramids call the builder makes.
        """
        from pyramids.dataset import Dataset

        dataset = Dataset.read_file(RASTER)
        warped = mocker.spy(Dataset, "warped_view")
        whole = mocker.patch("digitalearth.interactive.raster.reproject")
        InteractiveMap(crs=3857).large_image(dataset, dynamic=False, max_pixels=64 * 64)
        assert warped.call_count == 1, "the read must go through a lazily warped view"
        assert not whole.called, "the whole raster must not be warped up front"

    def test_a_raster_already_in_the_display_crs_is_not_warped(self, mocker):
        """Nothing is warped when there is nothing to warp.

        Args:
            mocker: Records the warp.
        """
        from pyramids.dataset import Dataset

        dataset = Dataset.read_file(RASTER)
        warped = mocker.spy(Dataset, "warped_view")
        m = InteractiveMap(crs=dataset.epsg)
        m.large_image(dataset, dynamic=False, max_pixels=64 * 64)
        assert warped.call_count == 0, "a raster in the display CRS needs no warp"

    def test_a_pan_reads_only_its_window(self):
        """Each frame reads the window it is looking at, not the whole raster again.

        Test scenario:
            The point of the lazy warp: a pan is a small read. The window here is a quarter of the raster,
            and the cells read are a quarter of what the whole raster would cost at the same resolution.
        """
        from holoviews.streams import RangeXY
        from pyramids.dataset import Dataset

        dataset = Dataset.read_file(RASTER)
        west, south, east, north = dataset.bbox
        m = InteractiveMap(crs=dataset.epsg)
        m.large_image(dataset, dynamic=True, max_pixels=64 * 64)
        stream = next(s for s in m.layers[0].streams if isinstance(s, RangeXY))
        stream.event(
            x_range=(west, (west + east) / 2), y_range=(south, (south + north) / 2)
        )
        values = np.asarray(
            m.layers[0][()].dimension_values(2, flat=False), dtype="float64"
        )
        assert values.size < 13 * 14, f"a quarter window read {values.size} cells"
