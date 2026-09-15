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
        assert request.crs() == 4326, "and the CRS travels with it"

    def test_a_request_with_no_region_has_no_bbox(self):
        """Asking for everything is expressed by asking for no region.

        Test scenario:
            A reader distinguishes "this window" from "whatever you have"; returning a degenerate bbox for
            the second would read as the origin.
        """
        assert ViewRequest(budget=10).as_bbox() is None
        assert ViewRequest(budget=10).crs() is None


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
        assert view.rereadable is False
        with pytest.raises(RuntimeError, match="carries no DataRef"):
            view.reread(ViewRequest(budget=100))


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
