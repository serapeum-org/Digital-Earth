"""`SourceView` and `ViewRequest` — a view that can be read again (DE-16, #276).

A `Source` forgot where it came from, so the only way to get different pixels was to start again from the
caller's object. That is what blocks dynamic tiling (#189), 3-D level of detail (#207) and point clouds too
large to hold (#206). These cover the address the view now carries, and the request it answers.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

from digitalearth.base.sources import DimensionInfo, Source
from digitalearth.base.sources.view import SourceView
from digitalearth.base.spec import Bounds, DataRef, Selection, ViewRequest

#: Anchored on this file, so the read works whatever directory pytest was started from.
RASTER = Path(__file__).resolve().parents[2] / "examples" / "data" / "acc4000.tif"


#: The three storage orders an 8x8 unit raster can have, as `(origin_x, step_x, 0, origin_y, 0, step_y)`.
#: `read_part` hands rows and columns back in storage order, so the step signs decide which way the axes run.
NORTH_UP = (0.0, 1.0, 0.0, 8.0, 0.0, -1.0)
SOUTH_UP = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
EAST_LEFT = (8.0, -1.0, 0.0, 8.0, 0.0, -1.0)

STORAGE_ORDERS = [
    ("north_up", NORTH_UP),
    ("south_up", SOUTH_UP),
    ("east_left", EAST_LEFT),
]


def _write_labelled(directory: Path, name: str, geo: tuple) -> Path:
    """Write an 8x8 raster whose every value names the cell it sits in (`column + 10 * row`).

    The cells are one unit square and cover x, y in 0..8 in every storage order, so a value of `34` is the
    cell in stored row 3, stored column 4 wherever that lands in world coordinates. That lets a coordinate
    assertion be made about the *data* rather than the shape.
    """
    path = directory / f"{name}.tif"
    rows, columns = np.mgrid[0:8, 0:8]
    Dataset.from_array(
        arr=(columns + 10 * rows).astype("float64"),
        geo_ref=GeoReference(geo=geo, epsg=3857),
    ).to_file(str(path))
    return path


@pytest.fixture(scope="module")
def labelled_rasters(tmp_path_factory) -> dict:
    """The same labelled 8x8 raster written in each of the three storage orders, keyed by order name."""
    directory = tmp_path_factory.mktemp("labelled")
    return {name: _write_labelled(directory, name, geo) for name, geo in STORAGE_ORDERS}


def _raster_parts(ref: DataRef) -> tuple:
    """Return the `(z, x, y, crs)` of the example raster, for building a view with chosen metadata."""
    view = SourceView.of(ref.open(), ref=ref)
    return view.z, view.x, view.y, view.crs


def _axis(name: str = "x") -> DimensionInfo:
    """Return a one-value axis, for views whose coordinates are not what is under test."""
    return DimensionInfo(np.array([0.0]), name)


class _WindowedReader:
    """A windowable stand-in that declares no CRS and, unless given one, no geotransform.

    A `Dataset` on disk always has both, so the arms of `_windowed` that handle their absence cannot be reached
    through a real raster. Each `read_part` call is recorded, so a test can see the window actually asked for.
    """

    def __init__(self, geotransform=None):
        self.calls = []
        self.bbox = (0.0, 0.0, 8.0, 8.0)
        if geotransform is not None:
            self.geotransform = geotransform

    def read_part(self, bbox, *, dst_width, dst_height, bbox_crs, band):
        """Record the window and CRS asked for, and return a zero array of the requested shape."""
        self.calls.append({"bbox": list(bbox), "bbox_crs": bbox_crs, "band": band})
        return np.zeros((dst_height, dst_width))


class TestTheRequest:
    """What a renderer asks for: a region, a resolution, a budget."""

    def test_pixels_counts_the_canvas_including_the_device_ratio(self):
        """A HiDPI canvas asks for the pixels it will actually draw.

        Test scenario:
            The budget comparison has to be against real device pixels, or a retina canvas silently reads a
            quarter of the detail it renders at.
        """
        assert ViewRequest(width=800, height=600).pixels == 480_000
        assert ViewRequest(width=800, height=600, pixel_ratio=2.0).pixels == 1_920_000

    def test_a_request_with_no_canvas_has_no_pixel_count(self):
        """Width and height are optional; a budget alone is a valid request.

        Test scenario:
            A decimating reader may be told only what it can afford, with no canvas in the question at all.
        """
        assert ViewRequest(budget=1000).pixels is None

    def test_side_is_the_square_a_decimating_reader_aims_for(self):
        """`side()` reproduces the canvas sizing the interactive tier does by hand.

        Test scenario:
            `interactive/raster.py` computes `max(64, int(sqrt(max_pixels)))` inline. Lifting it means the
            next reader that needs it does not write a fourth version.
        """
        assert ViewRequest(budget=1_000_000).side() == 1000
        assert ViewRequest(budget=4).side() == 64, "a tiny budget still floors at 64"

    def test_side_falls_back_to_the_canvas_then_the_floor(self):
        """The smaller of the budget and the canvas, then whichever is set, then the floor.

        Test scenario:
            A request may carry either, both or neither, and every reader needs an answer.
        """
        assert ViewRequest(width=400, height=400).side() == 400, "the canvas sizes it"
        assert ViewRequest().side() == 64, "with neither, the floor"

    @pytest.mark.parametrize(
        "request_, side",
        [
            (ViewRequest(width=400, height=400, budget=4_000_000), 400),
            (ViewRequest(width=4000, height=4000, budget=1_000_000), 1000),
            (
                ViewRequest(width=400, height=400, pixel_ratio=2.0, budget=4_000_000),
                800,
            ),
        ],
        ids=["canvas-under-budget", "canvas-over-budget", "retina-canvas"],
    )
    def test_with_a_canvas_and_a_budget_the_smaller_sizes_the_read(
        self, request_, side
    ):
        """`width`/`height` say what is wanted and `budget` what is affordable, so the side honours both.

        Args:
            request_: A request naming both a canvas and a budget.
            side: The side it must size to.

        Test scenario:
            The budget alone set the side, so a 400x400 canvas under a 4,000,000-cell budget was sized to 2,000 a
            side — reading 25 times what it can draw. The windowed read in `SourceView` already reads the canvas
            when it fits the budget; this is the same rule for the square a decimating reader aims for. The device
            ratio counts, as it does in `pixels`.
        """
        assert request_.side() == side, request_

    def test_within_budget_answers_true_when_there_is_no_budget(self):
        """An unbounded request affords anything.

        Test scenario:
            `None` means unbounded, and a reader that treated it as zero would refuse every read.
        """
        assert ViewRequest().within_budget(10**9) is True
        assert ViewRequest(budget=100).within_budget(101) is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"width": 0},
            {"height": -5},
            {"budget": 0},
            {"width": True},
            {"pixel_ratio": 0.0},
        ],
    )
    def test_a_dimension_that_reads_nothing_is_refused(self, kwargs):
        """Zero, negative and bool dimensions are caught where they are written.

        Args:
            kwargs: The bad field under test.

        Test scenario:
            A zero-width request reads nothing and a negative one is a computed value that went wrong
            upstream. Both would otherwise surface from inside a reader, naming neither.
        """
        with pytest.raises(ValueError):
            ViewRequest(**kwargs)

    def test_the_region_comes_back_in_bbox_order(self):
        """`as_bbox` takes its ordering from `Bounds`, not from a tuple written out here.

        Test scenario:
            The ordering bug `Bounds` exists to remove would reappear the moment this module spelled the
            four corners itself.
        """
        request = ViewRequest(bounds=Bounds(0.0, 1.0, 2.0, 3.0, crs=4326))
        assert request.as_bbox() == (0.0, 1.0, 2.0, 3.0)
        assert request.crs == 4326, "and the CRS travels with it"

    def test_a_request_with_no_region_has_no_bbox(self):
        """Asking for everything is expressed by asking for no region.

        Test scenario:
            A reader distinguishes "this window" from "whatever you have"; returning a degenerate bbox for
            the second would read as the origin.
        """
        assert ViewRequest(budget=10).as_bbox() is None
        assert ViewRequest(budget=10).crs is None


class TestTheViewIsStillASource:
    """The compatibility that made subclassing the right call."""

    def test_a_view_is_a_source(self):
        """Every existing consumer takes one unchanged.

        Test scenario:
            `Source` is read by all four tiers. Replacing it outright would have meant touching every reader
            in the package to gain a capability none of them uses yet.
        """
        assert isinstance(SourceView(None, _axis(), _axis("y")), Source)

    def test_it_keeps_the_source_surface(self):
        """crs, epsg, units and metadata still answer.

        Test scenario:
            A subclass that shadowed any of these would break a consumer in a way the type check above would
            not catch.
        """
        view = SourceView(
            None,
            _axis(),
            _axis("y"),
            crs=3857,
            metadata={"variable": "rain"},
            units="mm",
        )
        assert (view.crs, view.epsg, view.units) == (3857, 3857, "mm")
        assert view.metadata("variable") == "rain"


class TestTheAddress:
    """What the view remembers, and what it can do with it."""

    def test_a_view_carries_its_reference_and_slice(self):
        """The two things a `Source` forgot.

        Test scenario:
            `DataRef` (#273) and `Selection` (#272) shipped in PR #275 with no production caller. This is it.
        """
        view = SourceView(
            None,
            _axis(),
            _axis("y"),
            ref=DataRef("data/dem.tif"),
            selection=Selection.of(3, time="2024-01"),
        )
        assert view.ref.uri == "data/dem.tif"
        assert (view.selection.first_band, view.selection.time) == (3, "2024-01")

    def test_a_view_with_no_reference_defaults_its_selection(self):
        """A caller never has to guard before narrowing.

        Test scenario:
            Returning `None` would make every consumer check before calling `with_band`, which is how the
            scalar `band` parameter spread in the first place.
        """
        assert SourceView(None, _axis(), _axis("y")).selection == Selection()

    def test_a_view_built_from_a_held_object_says_it_cannot_be_reread(self):
        """No reference is a legitimate state, reported rather than crashed on.

        Test scenario:
            A caller may pass a `Dataset` they built in a notebook. That view is perfectly usable — it just
            cannot be re-read from an address it never had.
        """
        view = SourceView(None, _axis(), _axis("y"))
        request = ViewRequest(budget=100)
        assert view.rereadable is False
        with pytest.raises(RuntimeError, match="has no DataRef"):
            view.reread(request)


class TestRereading:
    """The capability the whole task exists for, against a real raster."""

    def test_a_view_can_be_read_again_at_another_resolution(self):
        """`reread` returns the same slice, sized to the new request but never finer than the data.

        Test scenario:
            This is what `interactive/raster.py` drove by hand and nothing else could reuse. The budget sets
            how many cells may be read; the source sets how many there are to read. `acc4000.tif` holds
            13x14 cells, so a budget of 10,000 reads 13x14 — this test pinned 100x100 before #297, which was
            a picture of the resampler rather than of the raster.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref, selection=Selection.of(1))
        again = view.reread(
            ViewRequest(
                bounds=Bounds.from_bbox(list(ref.open().bbox), crs=view.crs),
                budget=10_000,
            )
        )
        assert again.z.values.shape == (13, 14), (
            f"a whole-raster window must read the raster's own 13x14 cells, got {again.z.values.shape}"
        )

    def test_the_address_survives_the_reread(self):
        """The new view can be read again too.

        Test scenario:
            A viewport loop re-reads on every pan and zoom. A view that lost its address after one read would
            support exactly one.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref, selection=Selection.of(1))
        again = view.reread(ViewRequest(budget=4096))
        assert again.ref == ref, "the reference must survive"
        assert again.selection == view.selection, "and so must the slice"
        assert again.rereadable, "so the next pan can read again"

    def test_the_request_is_recorded_on_the_result(self):
        """A caller can tell what it asked for from what it got.

        Test scenario:
            A reader may return fewer cells than the budget allowed — the source may simply not have them.
            Keeping the request is how a caller notices.
        """
        ref = DataRef(str(RASTER))
        request = ViewRequest(budget=4096)
        view = SourceView.of(ref.open(), ref=ref).reread(request)
        assert view.request == request, "the answered request must be kept"

    def test_a_reader_that_cannot_window_still_reads(self):
        """A request against a reader with no windowed read is recorded, not enforced.

        Test scenario:
            Only some sources expose `read_part`. Refusing the read for the rest would make `ViewRequest`
            usable on COGs alone, when its point is to be the one question every reader is asked.
        """
        view = SourceView.of(
            np.arange(12.0).reshape(3, 4), request=ViewRequest(budget=2), crs=4326
        )
        assert view.z.values.shape == (3, 4), "a plain array is read whole"
        assert view.request.budget == 2, "and the unmet request is still recorded"


class TestARereadIsStillGeoreferenced:
    """The class of defect a shape assertion cannot see."""

    def test_the_window_carries_real_coordinates_not_pixel_indices(self):
        """A re-read view's axes are in its CRS, not 0, 1, 2, ...

        Test scenario:
            pyramids' `read_part` is an array reader — its own docstring says "Pixel values only — no
            transform, bounds, or CRS is attached". Handed straight to the extractor it produced `np.arange`
            axes while the view went on claiming EPSG:32618, so anything that plotted, extented or
            reprojected the result drew it at the projected origin. The original `reread` test asserted the
            shape, which is exactly what the defect preserved.
        """
        ref = DataRef(str(RASTER))
        dataset = ref.open()
        view = SourceView.of(dataset, ref=ref, selection=Selection.of(1))
        bbox = list(dataset.bbox)
        again = view.reread(
            ViewRequest(bounds=Bounds.from_bbox(bbox, crs=view.crs), budget=10_000)
        )
        half_cell = (bbox[2] - bbox[0]) / len(again.x.values) / 2
        assert again.x.values[0] == pytest.approx(bbox[0] + half_cell), (
            f"the first cell centre must be half a cell inside the window, got {again.x.values[0]}"
        )
        assert again.x.values[0] > 1000, (
            "a projected easting, not a pixel index — this is the assertion the shape check could not make"
        )

    def test_the_y_axis_runs_north_to_south(self):
        """A raster's rows are top-down, so the y axis descends.

        Test scenario:
            Getting this backwards flips the image vertically — which renders perfectly happily, and is the
            second thing a shape assertion cannot see.
        """
        ref = DataRef(str(RASTER))
        dataset = ref.open()
        view = SourceView.of(dataset, ref=ref, selection=Selection.of(1))
        again = view.reread(
            ViewRequest(
                bounds=Bounds.from_bbox(list(dataset.bbox), crs=view.crs), budget=4096
            )
        )
        assert again.y.values[0] > again.y.values[-1], (
            "the y axis must descend from north to south"
        )

    def test_an_unaligned_window_labels_the_cells_it_actually_read(
        self, labelled_rasters
    ):
        """A sub-window off the source's pixel grid is labelled where its data really sits.

        Test scenario:
            `read_part` snaps a window **outward** to whole source pixels — `floor`/`ceil` through
            `world_to_pixel` — so the buffer it returns spans the snapped rectangle, not the one asked for.
            Labelling it from the requested bbox shifts *and* scales the axes by up to one source cell per
            edge, worst at the decimation factors this class exists to serve. The two tests above cannot see
            it: both window the full extent, which is pixel-aligned by construction, so the snap is a no-op.

            The raster encodes its own coordinates (`value == column + 10 * row`), and the read is at the
            snapped window's native 4x4, so no resampling stands between a cell's label and its identity —
            every assertion below is exact.
        """
        ref = DataRef(str(labelled_rasters["north_up"]))
        view = SourceView.of(ref.open(), ref=ref, selection=Selection.of(1))
        again = view.reread(
            ViewRequest(
                bounds=Bounds(1.5, 1.5, 4.5, 4.5, crs=view.crs),
                width=4,
                height=4,
                budget=10_000,
            )
        )
        for column, x in enumerate(again.x.values):
            assert again.z.values[0][column] % 10 == pytest.approx(x - 0.5), (
                f"the cell labelled x={x} holds source column {again.z.values[0][column] % 10}"
            )
        for row, y in enumerate(again.y.values):
            assert again.z.values[row][0] // 10 == pytest.approx(7.5 - y), (
                f"the cell labelled y={y} holds source row {again.z.values[row][0] // 10}"
            )

    @pytest.mark.parametrize("stored, geo", STORAGE_ORDERS)
    def test_the_axes_follow_the_direction_the_rows_are_stored_in(
        self, labelled_rasters, stored, geo
    ):
        """A raster stored south-up or east-left is labelled the way its cells are actually laid out.

        Test scenario:
            `read_part` returns rows and columns in **storage** order, so the direction of each axis is the
            sign of the matching geotransform step. Labelling a south-up raster north-to-south, or an
            east-left one west-to-east, renders it mirrored — silently, because a flipped raster draws
            perfectly happily, which is why `test_the_y_axis_runs_north_to_south` exists at all.

            Both flips are exercised because the fix touches one expression per axis, and the y arm alone
            was what the review named. The expected value is derived from each cell's own label through the
            geotransform, so a single rule covers all three orders rather than three hand-written tables.
        """
        ref = DataRef(str(labelled_rasters[stored]))
        view = SourceView.of(ref.open(), ref=ref, selection=Selection.of(1))
        again = view.reread(
            ViewRequest(
                bounds=Bounds(1.5, 1.5, 4.5, 4.5, crs=view.crs),
                width=4,
                height=4,
                budget=10_000,
            )
        )
        origin_x, step_x, _, origin_y, _, step_y = geo
        for row, y in enumerate(again.y.values):
            for column, x in enumerate(again.x.values):
                stored_column = (x - origin_x) / step_x - 0.5
                stored_row = (y - origin_y) / step_y - 0.5
                assert again.z.values[row][column] == pytest.approx(
                    stored_column + 10 * stored_row
                ), (
                    f"the cell labelled ({x}, {y}) must hold what is stored at row {stored_row}, "
                    f"column {stored_column}, got {again.z.values[row][column]}"
                )

    def test_a_rotated_raster_is_refused_rather_than_windowed_as_if_unrotated(
        self, tmp_path
    ):
        """A geotransform with rotation terms cannot be labelled by one x and one y axis, so it is refused.

        Test scenario:
            `_grid` drops `geotransform[2]` and `[4]`, so a rotated raster was snapped and labelled as though
            it were north-up. On `(0, 1, 0.5, 8, 0, -1)` a 4x4 read labelled `x = [1.5, 2.5, 3.5, 4.5]` held
            nodata and interpolated values — silently wrong, which is worse than an error that names why.
        """
        path = _write_labelled(tmp_path, "rotated", (0.0, 1.0, 0.5, 8.0, 0.0, -1.0))
        ref = DataRef(str(path))
        view = SourceView.of(ref.open(), ref=ref, selection=Selection.of(1))
        request = ViewRequest(
            bounds=Bounds(1.5, 1.5, 4.5, 4.5, crs=3857),
            width=4,
            height=4,
            budget=10_000,
        )
        with pytest.raises(ValueError, match="cannot window a rotated raster"):
            view.reread(request)

    def test_what_the_data_is_survives_a_reread(self):
        """The variable, the kind and the units describe the data, not the window.

        Test scenario:
            A windowed read returns a bare array with no band name, so re-reading the same slice at another
            resolution silently emptied them. `standard_name` in particular is what autostyle matches on, so
            a re-read view lost the ECMWF-Magics identity this wave's auto_cmap consolidation exists to keep
            consistent across tiers.
        """
        ref = DataRef(str(RASTER))
        dataset = ref.open()
        view = SourceView.of(dataset, ref=ref, selection=Selection.of(1))
        again = view.reread(
            ViewRequest(
                bounds=Bounds.from_bbox(list(dataset.bbox), crs=view.crs), budget=4096
            )
        )
        assert again.metadata("variable") == view.metadata("variable"), (
            "the variable must survive the re-read"
        )
        assert again.metadata("kind") == "raster", "and so must the kind"

    def test_the_first_read_keeps_every_metadata_key(self):
        """`of` passes the extractor's metadata through rather than rebuilding it.

        Test scenario:
            It was rebuilt as `{"variable": ...}`, which dropped `kind` and `standard_name` before any
            re-read was involved.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref)
        assert view.metadata("kind") == "raster", (
            "the extractor's `kind` must not be dropped on the way in"
        )


class TestWhatTheViewRefuses:
    """The guards added in review round 1, each closing a way to report a slice the view does not hold."""

    @pytest.mark.parametrize(
        "axis", [{"time": "2024-01"}, {"level": 850}, {"member": 3}, {"overview": 2}]
    )
    def test_an_axis_the_view_cannot_honour_is_refused(self, axis):
        """A `Selection` carrying more than a band is rejected, not silently trimmed.

        Args:
            axis: The unhonoured axis under test.

        Test scenario:
            Only the band reaches the extractor. Storing the rest let a caller read `.selection` back and
            believe the view held that slice — and `overview` in particular reads as honoured, because
            `read_part` genuinely does choose one, just from the requested size rather than from here.
        """
        grid = np.arange(6.0).reshape(2, 3)
        selection = Selection.of(1, **axis)
        with pytest.raises(ValueError, match="cannot yet honour"):
            SourceView.of(grid, selection=selection)

    def test_a_selection_budget_is_applied_not_just_stored(self):
        """`Selection.of(1, budget=64)` limits the read, and the view records the request it applied.

        Test scenario:
            The guard refused `time`, `level`, `member` and `overview` because a view storing them would
            report a slice it does not hold — and let `budget` through neither refused nor applied, so
            `.selection.budget` read back as a limit that was never respected. The example raster is 182
            cells, so a budget of 64 is one the unfixed read visibly ignored.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref, selection=Selection.of(1, budget=64))
        assert view.z.values.size <= 64, (
            f"the budget must limit the read, got {view.z.values.size} cells"
        )
        assert view.request is not None, "the view must record the request it applied"
        assert view.request.budget == 64, (
            f"and that request must carry the selection's budget, got {view.request.budget}"
        )

    @pytest.mark.parametrize(
        "selection_budget, request_budget", [(400, 100), (100, 400)]
    )
    def test_the_tighter_of_two_budgets_wins(self, selection_budget, request_budget):
        """A selection budget and a request budget are both limits, so the smaller applies.

        Args:
            selection_budget: The budget named on the selection.
            request_budget: The budget named on the request.

        Test scenario:
            Either order — a looser request must not lift the selection's limit, and a looser selection must
            not lift the request's.
        """
        folded = SourceView._budgeted(
            ViewRequest(budget=request_budget), selection_budget
        )
        assert folded.budget == min(selection_budget, request_budget), (
            f"expected {min(selection_budget, request_budget)}, got {folded.budget}"
        )

    def test_a_request_keeps_its_region_when_a_selection_budget_is_folded_in(self):
        """Folding the budget in replaces only the budget.

        Test scenario:
            The request is rebuilt with `dataclasses.replace`; the bounds and canvas it carried must survive.
        """
        request = ViewRequest(
            bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=3857), width=8, height=4
        )
        folded = SourceView._budgeted(request, 16)
        assert (folded.bounds, folded.width, folded.height, folded.budget) == (
            request.bounds,
            8,
            4,
            16,
        ), f"only the budget may change, got {folded}"

    @pytest.mark.parametrize(
        "request_budget, selection_budget",
        [(None, None), (100, None), (100, 100), (50, 100)],
        ids=["neither", "request_only", "equal", "request_tighter"],
    )
    def test_a_request_the_selection_budget_cannot_tighten_comes_back_untouched(
        self, request_budget, selection_budget
    ):
        """No selection budget, or one no tighter than the request's, returns the very request that was passed.

        Args:
            request_budget: The budget named on the request.
            selection_budget: The budget named on the selection.

        Test scenario:
            Only a strictly tighter selection budget rebuilds the request. Equal budgets are the boundary: a `<`
            in place of `<=` would rebuild an identical request, which an equality check cannot see and an
            identity check can.
        """
        request = ViewRequest(budget=request_budget)
        assert SourceView._budgeted(request, selection_budget) is request, (
            f"a request budget of {request_budget} must not be rebuilt for a selection budget of "
            f"{selection_budget}"
        )

    def test_no_request_and_no_selection_budget_leaves_the_read_unbounded(self):
        """With neither a request nor a selection budget there is nothing to fold, so no request is invented.

        Test scenario:
            A request is only built from the selection when it names a budget; inventing an empty one here
            would make every unwindowed view claim to have answered a request nobody made.
        """
        assert SourceView._budgeted(None, None) is None, (
            "no request must stay no request"
        )

    def test_a_composite_selection_is_refused(self):
        """A view holds one band, so a three-band selection says so.

        Test scenario:
            `Selection.of((3, 2, 1))` reads as an RGB composite. Taking `first_band` and dropping the rest
            would draw one channel and report three.
        """
        grid = np.arange(6.0).reshape(2, 3)
        selection = Selection.of((3, 2, 1))
        with pytest.raises(ValueError, match="reads one band"):
            SourceView.of(grid, selection=selection)


class TestTheRequestedShape:
    """`_shape` — what a windowed read is actually asked to produce."""

    def test_a_canvas_within_budget_keeps_its_aspect_ratio(self):
        """800x600 is read as 800x600, not as a square.

        Test scenario:
            The window was always square, so a request naming a canvas had its aspect ratio distorted — the
            one thing `width`/`height` exist to describe.
        """
        assert SourceView._shape(ViewRequest(width=800, height=600, budget=10**6)) == (
            800,
            600,
        )

    def test_a_hidpi_canvas_reads_the_pixels_it_will_draw(self):
        """`pixel_ratio=2.0` reads 1600x1200, not the 800x600 the canvas is laid out at.

        Test scenario:
            `ViewRequest.pixels` exists to fold the device ratio in, and `side()` already does — but the
            windowed path compared and returned raw CSS pixels, so a retina canvas read a quarter of the
            detail it renders at. Silently, and *under* the budget the caller set, so the budget could not
            be blamed for it either.
        """
        assert SourceView._shape(
            ViewRequest(width=800, height=600, pixel_ratio=2.0, budget=10**7)
        ) == (1600, 1200)

    def test_a_hidpi_canvas_is_measured_against_the_budget_in_device_pixels(self):
        """The ratio counts towards the budget as well as towards the detail.

        Test scenario:
            800x600 at ratio 2.0 is 1,920,000 cells, which a budget of 1,000,000 cannot afford — so it
            scales, where the CSS canvas (480,000) would have sailed through untouched.
        """
        columns, rows = SourceView._shape(
            ViewRequest(width=800, height=600, pixel_ratio=2.0, budget=1_000_000)
        )
        assert columns * rows <= 1_000_000, (
            f"the device canvas must be held to the budget, got {columns}x{rows}"
        )
        assert (columns, rows) != (800, 600), (
            "and scaling it must not land back on the CSS canvas by coincidence"
        )

    @pytest.mark.parametrize(
        "width, height, budget",
        [
            (800, 600, 4800),
            (10_000, 1, 100),
            (1_000, 1, 100),
            (4_000, 3, 1_000),
            (1, 10_000, 100),
            (1920, 1080, 10),
            (7, 5, 11),
            (2, 3, 1),
        ],
    )
    def test_an_over_budget_canvas_never_reads_more_cells_than_the_budget(
        self, width, height, budget
    ):
        """Whatever the shape, the product of the two numbers returned fits the budget.

        Test scenario:
            The budget is the memory guard — `ViewRequest.budget` is documented as the limit "a reader that
            cannot serve both must respect". The previous spelling clamped each axis to a minimum of one
            *after* scaling, so on an elongated canvas the short axis truncated to zero, was lifted back to
            one, and left the long axis at its scaled value: `10000x1` under a budget of `100` came back as
            `1000x1`, ten times the limit. The single case the old test sampled (`800x600 / 4800`) divides
            exactly, so it passed while the property it named did not hold.

            The rows below are the shapes that break it — `width >> height` and its transpose — plus the
            dividing case, so the assertion is on the property rather than on a sample.
        """
        columns, rows = SourceView._shape(
            ViewRequest(width=width, height=height, budget=budget)
        )
        assert columns >= 1, f"a read has to return at least one column, got {columns}"
        assert rows >= 1, f"a read has to return at least one row, got {rows}"
        assert columns * rows <= budget, (
            f"{width}x{height} under a budget of {budget} came back as {columns}x{rows} = "
            f"{columns * rows} cells"
        )

    @pytest.mark.parametrize(
        "width, height, budget",
        [(800, 600, 4800), (1920, 1080, 480_000), (1000, 250, 40_000)],
    )
    def test_a_scaled_canvas_keeps_its_shape_to_the_nearest_whole_cell(
        self, width, height, budget
    ):
        """The aspect ratio survives the scaling, as closely as a whole number of cells allows.

        Test scenario:
            The budget is the guarantee and the shape is best effort, so this states the bound rather than
            an equality: each axis is within one cell of the exactly-scaled canvas. `round(w / h, 2)` — what
            the old test asserted — holds only where the scale factor happens to divide.
        """
        columns, rows = SourceView._shape(
            ViewRequest(width=width, height=height, budget=budget)
        )
        scale = (budget / (width * height)) ** 0.5
        assert abs(columns - width * scale) < 1, (
            f"the width must be within a cell of {width * scale:.3f}, got {columns}"
        )
        assert abs(rows - height * scale) < 1, (
            f"the height must be within a cell of {height * scale:.3f}, got {rows}"
        )

    def test_a_budget_alone_wins_over_the_readability_floor(self):
        """A tiny budget gives a tiny window, because the budget is the hard limit.

        Test scenario:
            `side()` floors at 64 for readability, which turned a budget of 16 into 4,096 cells — 256x the
            limit `ViewRequest.budget` documents as the one a reader must respect.
        """
        assert SourceView._shape(ViewRequest(budget=16)) == (4, 4)

    def test_no_canvas_and_no_budget_falls_back_to_the_floor(self):
        """With nothing to go on, the readable default applies.

        Test scenario:
            The remaining arm: a request that names neither still has to produce a size.
        """
        assert SourceView._shape(ViewRequest()) == (64, 64)

    @pytest.mark.parametrize("width, height", [(800, None), (None, 600)])
    def test_a_canvas_missing_one_dimension_is_sized_by_the_budget_alone(
        self, width, height
    ):
        """One named dimension is not a canvas, so the read is the square the budget affords.

        Args:
            width: The canvas width, or ``None``.
            height: The canvas height, or ``None``.

        Test scenario:
            The aspect-ratio arm needs both `width` and `height`; with either missing there is no ratio to keep,
            so a budget of 100 reads 10x10 rather than a strip sized from the one dimension given.
        """
        shape = SourceView._shape(ViewRequest(width=width, height=height, budget=100))
        assert shape == (10, 10), (
            f"{width}x{height} under a budget of 100 must read 10x10, got {shape}"
        )

    @pytest.mark.parametrize(
        "width, height, expected",
        [(10_000, 1, (100, 1)), (5_000, 2, (100, 1)), (1, 10_000, (1, 100))],
    )
    def test_an_elongated_canvas_spends_the_whole_budget_on_its_long_axis(
        self, width, height, expected
    ):
        """When the short axis scales to nothing, the long axis gets every cell the budget allows.

        Args:
            width: The canvas width in cells.
            height: The canvas height in cells.
            expected: The ``(width, height)`` the budget of 100 should buy.

        Test scenario:
            The budget tests above assert only that the product fits, which `(1, 1)` would satisfy too. The
            documented answer for a canvas this elongated is the most detail the limit can buy: the long axis
            capped at the budget and the short axis at one, in whichever orientation the canvas has.
        """
        fitted = SourceView._fitted(width, height, 100)
        assert fitted == expected, (
            f"{width}x{height} under a budget of 100 must fit as {expected}, got {fitted}"
        )


class TestWindowingWhatCannotBeWindowed:
    """The arms of `_windowed` that decline rather than read."""

    def test_an_object_with_no_bbox_and_no_bounds_is_read_whole(self):
        """A budget-only request needs a region to window against.

        Test scenario:
            The budget is windowed against the source's own bbox — but a plain array has none, so there is
            nothing to window and the read proceeds unwindowed rather than failing.
        """
        view = SourceView.of(
            np.arange(12.0).reshape(3, 4), request=ViewRequest(budget=2)
        )
        assert view.z.values.shape == (3, 4), "a plain array has no window to take"

    def test_a_windowable_object_with_no_bounds_honours_the_budget(self):
        """The budget alone is enough to window, against the source's own extent.

        Test scenario:
            This previously returned the data unchanged, ignoring the budget entirely even though the object
            could window — 182 cells returned for a budget of 16.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref).reread(ViewRequest(budget=16))
        cells = view.z.values.size
        assert cells <= 16, f"a budget of 16 must not return {cells} cells"
        assert view.request.within_budget(cells), "and the view must agree it fits"


class TestTheRemainingWindowArms:
    """The branches the round-1 fixes added that the happy paths do not reach."""

    def test_the_units_survive_a_reread_when_the_new_read_supplies_none(self):
        """Units describe the data, not the window.

        Test scenario:
            A windowed read returns a bare array, so it carries no units. The metadata carry-over covers the
            dict; `units` is a separate attribute and needed its own arm.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref)
        stamped = SourceView(
            view.z, view.x, view.y, view.crs, {"variable": "rain"}, "mm", ref=ref
        )
        again = stamped.reread(ViewRequest(budget=64))
        assert again.units == "mm", (
            "the units must survive a re-read that supplies none"
        )

    def test_a_falsy_value_from_the_new_read_is_not_replaced_by_the_old_one(self):
        """`member=0` from the new read stands; it is an answer, not an absence.

        Test scenario:
            The carry-over tested `not fresh.get(key)`, so every falsy value was overwritten by the previous
            read's. The collection extractor writes a **0-based** `member`, so re-reading member 0 after a
            read of member 3 would have reported member 3. `False` and `""` from a caller's own metadata
            meet the same fate.
        """
        merged = SourceView._carried(
            {"member": 3, "variable": "t2m", "flag": True},
            {"member": 0, "variable": "t2m", "flag": False},
        )
        assert merged["member"] == 0, (
            f"the new read's member must stand, got {merged['member']}"
        )
        assert merged["flag"] is False, "and so must any other falsy value it wrote"

    def test_only_the_keys_that_describe_the_data_are_carried(self):
        """A key describing the read itself is never filled in from an earlier read.

        Test scenario:
            The loop copied **every** key the previous read had. Anything a reader attaches about *this*
            read — the window, the overview it chose — would have been inherited by a read that did not
            produce it.
        """
        merged = SourceView._carried(
            {
                "variable": "t2m",
                "kind": "raster",
                "standard_name": "air_temperature",
                "overview": 2,
            },
            {"variable": "", "kind": "raster"},
        )
        assert merged == {
            "variable": "t2m",
            "kind": "raster",
            "standard_name": "air_temperature",
        }, (
            f"only the describing keys are carried, and the empty variable is filled, got {merged}"
        )

    def test_a_reread_does_not_write_into_the_view_it_just_read(self):
        """The carry-over builds a new view, so `again` is never mutated through private attributes.

        Test scenario:
            It assigned `again._meta[key]` and `again._units` directly, although `Source` presents its state
            as read-only properties. The observable guarantee is that the original view's metadata is not
            touched either.
        """
        ref = DataRef(str(RASTER))
        view = SourceView(*_raster_parts(ref), {"variable": "rain"}, "mm", ref=ref)
        before = dict(view._meta)
        again = view.reread(ViewRequest(budget=64))
        assert again.metadata("variable") == "rain", (
            "the variable is carried to the new view"
        )
        assert view._meta == before, (
            "and the view that was re-read is left exactly as it was"
        )
        assert again is not view, "a re-read is a new view"

    def test_a_canvas_with_no_region_windows_the_whole_source_at_that_size(self):
        """`ViewRequest(width=4, height=3)` reads a 3x4 array, not the source's own 13x14 cells.

        Test scenario:
            A budget with no region already windowed against the source's extent, because the object can
            window and skipping blew the budget. A full canvas with no region was recorded and ignored, so
            the caller who asked for 4x3 got every cell — the same gap on the other sizing field.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(
            ref.open(), ref=ref, request=ViewRequest(width=4, height=3)
        )
        assert view.z.values.shape == (3, 4), (
            f"the canvas must size the read, got {view.z.values.shape}"
        )

    def test_half_a_canvas_with_no_region_is_recorded_but_not_applied(self):
        """A lone `width` cannot size a read, so the source is read whole rather than as a 64x64 floor square.

        Test scenario:
            `_shape` cannot use one dimension without the other and falls back to `side()`'s readability
            floor. Windowing on it produced a 64x64 upsample of a 13x14 raster, honouring nothing named.
        """
        ref = DataRef(str(RASTER))
        full = SourceView.of(ref.open(), ref=ref)
        view = SourceView.of(ref.open(), ref=ref, request=ViewRequest(width=5))
        assert view.z.values.shape == full.z.values.shape, (
            f"half a canvas must leave the read whole, got {view.z.values.shape}"
        )

    def test_a_budget_only_request_against_an_object_with_no_budget_declines(self):
        """Both halves of the guard are needed: a bbox *and* a budget.

        Test scenario:
            A windowable object with a bbox but a request carrying no budget has nothing to size the window
            by, so it reads whole rather than inventing a limit.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref)
        again = view.reread(ViewRequest())
        assert again.z.values.shape == view.z.values.shape, (
            "with neither bounds nor budget the read is unwindowed"
        )

    def test_a_window_given_in_another_crs_is_converted_before_reading(self):
        """`read_part` returns data in the dataset's CRS, so the bbox is converted into it first.

        Test scenario:
            Describing the window in the requested CRS while the cells are measured in the dataset's would be
            a second way to produce the coordinates H1 was about — right-looking numbers against the wrong
            axis.
        """
        ref = DataRef(str(RASTER))
        dataset = ref.open()
        view = SourceView.of(dataset, ref=ref)
        in_wgs84 = Bounds.from_bbox(list(dataset.bbox), crs=view.crs).to_crs(4326)
        again = view.reread(ViewRequest(bounds=in_wgs84, budget=4096))
        assert again.crs == view.crs, (
            f"the view must report the CRS its cells are in, got {again.crs}"
        )
        assert again.x.values[0] > 1000, (
            "and hold projected coordinates, not the degrees the request was written in"
        )

    def test_a_window_with_no_crs_is_read_in_the_rasters_own_coordinates(
        self, labelled_rasters
    ):
        """A region naming no CRS is not converted, and its cells are still labelled where they sit.

        Test scenario:
            Conversion needs a CRS to convert from, so it runs only when the request names one; pyramids reads
            `bbox_crs=None` as "already in the raster's coordinates". The view falls back to the CRS `reread`
            passes through, and the labelled raster (`value == column + 10 * row`) shows the window was still
            snapped and labelled correctly: the top-left cell of the 1..5 window is stored row 3, column 1.
        """
        ref = DataRef(str(labelled_rasters["north_up"]))
        view = SourceView.of(ref.open(), ref=ref)
        again = view.reread(
            ViewRequest(bounds=Bounds(1.5, 1.5, 4.5, 4.5, crs=None), width=4, height=4)
        )
        assert again.crs == 3857, (
            f"the view must report the raster's own CRS, got {again.crs}"
        )
        assert again.x.values.tolist() == [1.5, 2.5, 3.5, 4.5], (
            f"got x={again.x.values}"
        )
        assert again.z.values[0][0] == 31.0, (
            f"the top-left cell must be stored row 3, column 1, got {again.z.values}"
        )

    def test_a_reader_declaring_no_crs_keeps_the_window_in_the_requested_one(self):
        """With no CRS to convert into, the bbox reaches `read_part` as written and the view reports its CRS.

        Test scenario:
            The conversion needs both ends. A windowable object with neither `epsg` nor `crs` gives no target,
            so converting would guess; instead the window stays in the CRS the request named, the only one its
            numbers are known to be in. The window is pixel-aligned, so snapping plays no part in the bbox.

            The canvas is half the window's 4x4 cells, so the read is decimated and goes through `read_part`
            — the call this test reads. At the window's own resolution it would be read with `read_array`
            instead, which is the point of `_read_window`'s two paths.
        """
        reader = _WindowedReader(geotransform=(0.0, 1.0, 0.0, 8.0, 0.0, -1.0))
        view = SourceView.of(
            reader,
            request=ViewRequest(
                bounds=Bounds(1.0, 2.0, 5.0, 6.0, crs=4326), width=2, height=2
            ),
        )
        assert reader.calls == [
            {"bbox": [1.0, 2.0, 5.0, 6.0], "bbox_crs": 4326, "band": 0}
        ], f"the window must be read unconverted, got {reader.calls}"
        assert view.crs == 4326, (
            f"and the view must report the request's CRS, got {view.crs}"
        )

    def test_a_reader_with_no_geotransform_reads_the_window_exactly_as_asked(self):
        """With no grid to snap to, the requested rectangle is the one read, labelled north-up and west-left.

        Test scenario:
            `_aligned` needs a geotransform to find pixel edges, so without one the window is not snapped. `_axes`
            then has no step signs to consult and assumes the convention every raster pyramids writes uses: x
            ascending, y descending, each at the centre of its cell.
        """
        reader = _WindowedReader()
        view = SourceView.of(
            reader,
            request=ViewRequest(
                bounds=Bounds(0.25, 0.5, 4.25, 2.5, crs=3857), width=4, height=2
            ),
        )
        assert reader.calls[0]["bbox"] == [0.25, 0.5, 4.25, 2.5], (
            f"an ungridded window must not be snapped, got {reader.calls[0]['bbox']}"
        )
        assert view.x.values.tolist() == [0.75, 1.75, 2.75, 3.75], (
            f"x must ascend, got {view.x.values}"
        )
        assert view.y.values.tolist() == [2.0, 1.0], (
            f"y must descend, got {view.y.values}"
        )

    @pytest.mark.parametrize(
        "geotransform",
        [
            None,
            (),
            (0.0, 1.0, 0.0, 8.0, 0.0),
            (0.0, 0.0, 0.0, 8.0, 0.0, -1.0),
            (0.0, 1.0, 0.0, 8.0, 0.0, 0.0),
        ],
        ids=["none", "empty", "short", "zero_x_step", "zero_y_step"],
    )
    def test_a_geotransform_that_cannot_place_a_cell_is_no_grid(self, geotransform):
        """A missing, short or zero-step geotransform reads as no grid, and the window is left as it came.

        Args:
            geotransform: The unusable geotransform under test.

        Test scenario:
            Fewer than six entries has no step to read, and a zero step would divide by zero in `_snapped`.
            Aligning against either would crash or snap to nonsense, so both answer as a reader exposing no
            geotransform at all does.
        """
        data = SimpleNamespace(geotransform=geotransform)
        window = Bounds(0.25, 0.5, 4.25, 2.5, crs=3857)
        assert SourceView._grid(data) is None, (
            f"{geotransform!r} must not read as a grid"
        )
        assert SourceView._aligned(window, data) is window, (
            "and the window must come back unchanged"
        )

    def test_a_usable_geotransform_is_read_as_floats_without_its_rotation_terms(self):
        """`_grid` keeps each axis's origin and step, as floats, and drops the two rotation terms.

        Test scenario:
            An axis-aligned window cannot describe a rotated raster, so the rotation entries take no part in
            the grid a window snaps to; integer entries come back as floats for the pixel arithmetic.
        """
        grid = SourceView._grid(SimpleNamespace(geotransform=[10, 2, 7, 20, 9, -2]))
        assert grid == (10.0, 2.0, 20.0, -2.0), (
            f"expected origin and step per axis, got {grid}"
        )
        assert all(isinstance(entry, float) for entry in grid), (
            f"every entry must be a float, got {grid}"
        )

    @pytest.mark.parametrize(
        "low, high, origin, step, expected",
        [
            (1.3, 4.6, 0.0, 1.0, (1.0, 5.0)),
            (1.3, 4.6, 8.0, -1.0, (1.0, 5.0)),
            (1.3, 4.4, 0.0, 0.5, (1.0, 4.5)),
            (1.3, 4.6, 8.0, -2.0, (0.0, 6.0)),
            (2.0, 5.0, 8.0, -1.0, (2.0, 5.0)),
        ],
        ids=[
            "positive_step",
            "negative_step",
            "half_step",
            "wide_negative_step",
            "already_aligned",
        ],
    )
    def test_an_axis_is_snapped_outward_to_the_enclosing_cell_edges(
        self, low, high, origin, step, expected
    ):
        """Each edge moves out to the nearest cell edge beyond it, whichever way the axis is stored.

        Args:
            low: The lower world coordinate of the window.
            high: The upper world coordinate.
            origin: The geotransform origin for the axis.
            step: The geotransform step, which may be negative.
            expected: The enclosing ``(low, high)`` cell edges.

        Test scenario:
            A negative step reverses which pixel index belongs to which world edge, so flooring the low world
            coordinate directly snaps *inward* there and silently crops. The rows cover both signs, steps other
            than one, and an axis already on the grid, which must come back as it went in.
        """
        snapped = SourceView._snapped(low, high, origin, step)
        assert snapped == pytest.approx(expected), (
            f"({low}, {high}) on step {step} snapped to {snapped}"
        )
        assert snapped[0] <= low, f"{snapped} must start at or below {low}"
        assert snapped[1] >= high, f"{snapped} must end at or above {high}"

    def test_only_an_empty_variable_is_a_placeholder(self):
        """`variable=""` is filled from the previous read; an empty `kind` or `standard_name` stands.

        Test scenario:
            The extractor writes `variable=""` for a bare array, which is a placeholder rather than an answer.
            The exception is named for that one key; widening it to every describing key would bring back the
            truthiness test that let an old read overwrite values the new one really wrote.
        """
        merged = SourceView._carried(
            {"variable": "t2m", "kind": "raster", "standard_name": "air_temperature"},
            {"variable": "", "kind": "", "standard_name": ""},
        )
        assert merged == {"variable": "t2m", "kind": "", "standard_name": ""}, (
            f"only the empty variable may be filled in, got {merged}"
        )

    def test_a_variable_the_new_read_named_is_not_overwritten(self):
        """Carrying fills gaps; it does not arbitrate between two reads that both answered.

        Test scenario:
            A describing key the new read supplied belongs to the new read, even where the previous read said
            something else — and a describing key neither read supplied is not invented.
        """
        merged = SourceView._carried({"variable": "t2m"}, {"variable": "precip"})
        assert merged == {"variable": "precip"}, (
            f"the new read's variable must stand alone, got {merged}"
        )


class TestAWindowThatMissesTheSource:
    """A viewport panned off the data draws nothing, rather than raising (#297)."""

    class _Missing(Exception):
        """Stands in for pyramids' out-of-bounds report, which this module matches by name."""

    class _OffRaster:
        """A raster whose every windowed read reports the window as out of bounds."""

        epsg = 4326
        bbox = (0.0, 0.0, 4.0, 4.0)
        cell_size = 1.0
        geotransform = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        no_data_value = (-9999.0,)

        def read_part(self, **kwargs):
            """Refuse every window.

            Args:
                **kwargs: The window asked for.

            Raises:
                OutOfBoundsError: always.
            """
            raise _out_of_bounds()

        def read_array(self, **kwargs):
            """Refuse every window.

            Args:
                **kwargs: The window asked for.

            Raises:
                OutOfBoundsError: always.
            """
            raise _out_of_bounds()

    class _Broken:
        """A raster whose reads fail for a reason that has nothing to do with the window."""

        epsg = 4326
        bbox = (0.0, 0.0, 4.0, 4.0)
        cell_size = 1.0
        geotransform = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        no_data_value = (-9999.0,)

        def read_part(self, **kwargs):
            """Fail the way a broken file does.

            Args:
                **kwargs: The window asked for.

            Raises:
                OSError: always.
            """
            raise OSError("the file is truncated")

        def read_array(self, **kwargs):
            """Fail the way a broken file does.

            Args:
                **kwargs: The window asked for.

            Raises:
                OSError: always.
            """
            raise OSError("the file is truncated")

    def test_an_off_source_window_comes_back_empty(self):
        """The canvas is all missing, which is what an empty place looks like."""
        source = self._OffRaster()
        ref = DataRef.to_object(source, name="off-source-view")
        view = SourceView.of(
            source,
            ref=ref,
            selection=Selection.of(1),
            request=ViewRequest(
                bounds=Bounds(10.0, 10.0, 12.0, 12.0, crs=4326), budget=64
            ),
        )
        assert np.isnan(view.z.values).all(), view.z.values

    def test_an_indexing_defect_is_not_a_viewport_off_the_edge(self):
        """ "out of bounds" is what numpy says about a bad index, and it was read as "nothing here".

        Test scenario:
            The guard matched the phrase anywhere in the message, and
            `IndexError: index 3 is out of bounds for axis 0 with size 2` is the most common error text in
            Python. Any indexing defect reached through a windowed read came back as a silent all-`NaN`
            canvas instead of a traceback (review M2).
        """

        class _Broken:
            """A windowable reader whose read raises the commonest indexing error there is."""

            bbox = (0.0, 0.0, 8.0, 8.0)
            cell_size = 1.0
            epsg = 4326

            def read_part(self, **_kwargs):
                """Raise the way numpy does for a bad index.

                Raises:
                    IndexError: always.
                """
                raise IndexError("index 3 is out of bounds for axis 0 with size 2")

        reader = _Broken()
        request = ViewRequest(bounds=Bounds(0.0, 0.0, 4.0, 4.0, crs=4326), budget=4)
        with pytest.raises(IndexError, match="out of bounds"):
            SourceView.of(reader, request=request)

    def test_a_grid_with_tall_cells_is_counted_by_both_its_spacings(self):
        """A cell is two numbers. Counting rows with the x spacing invents rows that are not there.

        Test scenario:
            pyramids documents `cell_size` as the magnitude of the *x* pixel size, and it was used for both
            axes: a window over a 1 x 4 grid reported four times the rows it holds, so `_native`'s cap read
            at the resampler's resolution rather than the raster's (review M3).
        """

        class _Tall:
            """A reader whose cells are four times as tall as they are wide."""

            bbox = (0.0, 0.0, 20.0, 40.0)
            cell_size = 1.0
            epsg = 4326
            geotransform = (0.0, 1.0, 0.0, 40.0, 0.0, -4.0)

        window = Bounds(0.0, 0.0, 20.0, 40.0, crs=4326)
        assert SourceView._native_cells(window, _Tall()) == (20, 10), (
            SourceView._native_cells(window, _Tall())
        )

    def test_a_degenerate_geotransform_falls_back_to_the_cell_size(self):
        """A geotransform that states a zero spacing says nothing usable, so the square size is read."""

        class _Zeroed:
            """A reader whose geotransform carries no pixel size."""

            bbox = (0.0, 0.0, 8.0, 8.0)
            cell_size = 2.0
            epsg = 4326
            geotransform = (0.0, 0.0, 0.0, 8.0, 0.0, 0.0)

        assert SourceView._spacing(_Zeroed()) == (2.0, 2.0), SourceView._spacing(
            _Zeroed()
        )

    def test_a_square_grid_still_counts_by_its_cell_size(self):
        """A reader with no geotransform says one number, and it means both axes."""

        class _Square:
            """A reader that states only a square cell size."""

            bbox = (0.0, 0.0, 8.0, 8.0)
            cell_size = 2.0
            epsg = 4326

        window = Bounds(0.0, 0.0, 8.0, 8.0, crs=4326)
        assert SourceView._native_cells(window, _Square()) == (4, 4), (
            SourceView._native_cells(window, _Square())
        )

    def test_a_read_that_failed_for_another_reason_still_raises(self):
        """A truncated file is not an empty place, and is not hidden as one."""
        source = self._Broken()
        ref = DataRef.to_object(source, name="broken-view")
        selection = Selection.of(1)
        request = ViewRequest(bounds=Bounds(0.0, 0.0, 2.0, 2.0, crs=4326), budget=64)
        with pytest.raises(OSError, match="truncated"):
            SourceView.of(source, ref=ref, selection=selection, request=request)


class TestMaskingAWindowedRead:
    """What a decimated read calls missing, where pyramids cannot mask it (#297)."""

    class _Packed:
        """A raster whose band is packed, so its sentinel unpacks to an ordinary-looking number."""

        epsg = 4326
        bbox = (0.0, 0.0, 8.0, 8.0)
        cell_size = 1.0
        geotransform = (0.0, 1.0, 0.0, 8.0, 0.0, -1.0)
        no_data_value = (-9999,)
        scale = [0.01]
        offset = [5.0]

        def read_part(self, *, bbox, dst_width, dst_height, bbox_crs=None, band=0):
            """Return a decimated window whose first cell is the unpacked sentinel.

            Args:
                bbox: The window.
                dst_width: Cells across.
                dst_height: Cells down.
                bbox_crs: The window's CRS.
                band: The 0-based band.

            Returns:
                The canvas, in physical units.
            """
            values = np.full((dst_height, dst_width), 7.0)
            values[0, 0] = -9999 * 0.01 + 5.0
            return values

    def test_a_packed_sentinel_is_missing_in_the_units_the_read_returns(self):
        """The sentinel is scaled and offset before it is compared, as the read's values are."""
        source = self._Packed()
        ref = DataRef.to_object(source, name="packed-view")
        view = SourceView.of(
            source,
            ref=ref,
            selection=Selection.of(1),
            request=ViewRequest(bounds=Bounds(0.0, 0.0, 8.0, 8.0, crs=4326), budget=16),
        )
        assert int(np.isnan(view.z.values).sum()) == 1, view.z.values

    @pytest.mark.parametrize("declared", [(), (None,)], ids=["no-entry", "no-value"])
    def test_a_band_that_declares_no_sentinel_keeps_every_value(self, declared):
        """Nothing is guessed: a raster with no nodata value has no missing cells.

        Args:
            declared: What the band says about its nodata value — nothing at all, or explicitly none.
        """

        class _Plain(TestMaskingAWindowedRead._Packed):
            """The same raster, without a usable nodata value."""

            no_data_value = declared

        source = _Plain()
        ref = DataRef.to_object(source, name="plain-view")
        view = SourceView.of(
            source,
            ref=ref,
            selection=Selection.of(1),
            request=ViewRequest(bounds=Bounds(0.0, 0.0, 8.0, 8.0, crs=4326), budget=16),
        )
        assert not np.isnan(view.z.values).any(), view.z.values


def _out_of_bounds() -> Exception:
    """Return an exception named the way pyramids names its out-of-bounds report.

    Returns:
        The exception to raise.
    """

    class OutOfBoundsError(Exception):
        """Named, not imported: the module matches by name so the tier does not depend on where it lives."""

    return OutOfBoundsError("the window is outside the raster")
