"""`SourceView` and `ViewRequest` — a view that can be read again (DE-16, #276).

A `Source` forgot where it came from, so the only way to get different pixels was to start again from the
caller's object. That is what blocks dynamic tiling (#189), 3-D level of detail (#207) and point clouds too
large to hold (#206). These cover the address the view now carries, and the request it answers.
"""

from pathlib import Path

import numpy as np
import pytest

from digitalearth.base.sources import DimensionInfo, Source
from digitalearth.base.sources.view import SourceView
from digitalearth.base.spec import Bounds, DataRef, Selection, ViewRequest

#: Anchored on this file, so the read works whatever directory pytest was started from.
RASTER = Path(__file__).resolve().parents[2] / "examples" / "data" / "acc4000.tif"


def _axis(name: str = "x") -> DimensionInfo:
    """Return a one-value axis, for views whose coordinates are not what is under test."""
    return DimensionInfo(np.array([0.0]), name)


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
        """The budget wins, then the canvas, then the floor.

        Test scenario:
            A request may carry either, both or neither, and every reader needs an answer.
        """
        assert ViewRequest(width=400, height=400).side() == 400, "the canvas sizes it"
        assert ViewRequest().side() == 64, "with neither, the floor"

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
        """`reread` returns the same slice, sized to the new request.

        Test scenario:
            This is what `interactive/raster.py` drives by hand and nothing else can reuse. The budget sets
            the square side, exactly as its inline `max(64, sqrt(max_pixels))` does.
        """
        ref = DataRef(str(RASTER))
        view = SourceView.of(ref.open(), ref=ref, selection=Selection.of(1))
        again = view.reread(
            ViewRequest(
                bounds=Bounds.from_bbox(list(ref.open().bbox), crs=view.crs),
                budget=10_000,
            )
        )
        assert again.z.values.shape == (100, 100), (
            f"a budget of 10,000 cells must read a 100x100 window, got {again.z.values.shape}"
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
        with pytest.raises(ValueError, match="cannot yet honour"):
            SourceView.of(
                np.arange(6.0).reshape(2, 3), selection=Selection.of(1, **axis)
            )

    def test_a_composite_selection_is_refused(self):
        """A view holds one band, so a three-band selection says so.

        Test scenario:
            `Selection.of((3, 2, 1))` reads as an RGB composite. Taking `first_band` and dropping the rest
            would draw one channel and report three.
        """
        with pytest.raises(ValueError, match="reads one band"):
            SourceView.of(
                np.arange(6.0).reshape(2, 3), selection=Selection.of((3, 2, 1))
            )


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

    def test_a_canvas_over_budget_is_scaled_down_keeping_its_shape(self):
        """The budget is the limit; the canvas is only what would look best.

        Test scenario:
            Refusing would be unhelpful and reading it whole would blow the budget, so it is scaled — and the
            aspect ratio survives the scaling.
        """
        width, height = SourceView._shape(
            ViewRequest(width=800, height=600, budget=4800)
        )
        assert width * height <= 4800, f"{width}x{height} must fit the budget"
        assert round(width / height, 2) == round(800 / 600, 2), (
            "and keep the aspect ratio"
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
