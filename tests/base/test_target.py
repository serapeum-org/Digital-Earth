"""`RenderTarget` — the output chooses the read budget, never the layer (DE-35, #284).

The budget belonged to whoever made the call: `large_image` hard-codes `max_pixels=4_000_000`, and an exported page
and a live window got the same one. These cover the target that owns it, and the one path from a view to a
`ViewRequest`.
"""

import json

import pytest

from digitalearth.base.spec import (
    DEFAULT_BUDGETS,
    TARGET_KINDS,
    Bounds,
    Camera,
    RenderTarget,
    Viewport,
    ViewRequest,
)


class TestTheTarget:
    """What a target is, and what it refuses."""

    def test_the_four_kinds(self):
        """Window, page, still and batch — each with a default budget."""
        assert TARGET_KINDS == ("window", "html", "image", "batch")
        assert set(DEFAULT_BUDGETS) == set(TARGET_KINDS), sorted(DEFAULT_BUDGETS)

    def test_a_page_is_lighter_than_a_window(self):
        """A page carries its pixels inline, so its default budget is smaller.

        Test scenario:
            The window default is the budget the package already uses for a live view — `large_image`'s
            `max_pixels`. Checked against that source rather than restated.
        """
        import inspect

        from digitalearth.interactive.raster import RasterMixin

        live_default = (
            inspect.signature(RasterMixin.large_image).parameters["max_pixels"].default
        )
        assert RenderTarget("window").effective_budget == live_default, live_default
        assert (
            RenderTarget("html").effective_budget
            < RenderTarget("window").effective_budget
        )

    def test_an_explicit_budget_wins_over_the_kinds_default(self):
        """A caller who names a budget gets it."""
        assert RenderTarget("html", budget=250_000).effective_budget == 250_000

    def test_an_unknown_kind_names_the_kinds_that_exist(self):
        """A typo in the kind is refused with the list to choose from."""
        with pytest.raises(
            ValueError,
            match=r"one of \['window', 'html', 'image', 'batch'\]; got 'pdf'",
        ):
            RenderTarget("pdf")

    @pytest.mark.parametrize(
        "field, value",
        [("width", 0), ("height", -4), ("budget", 0), ("width", 1.5), ("budget", True)],
    )
    def test_a_canvas_or_budget_that_is_not_a_positive_whole_number_is_refused(
        self, field, value
    ):
        """Zero reads nothing, a fraction is not a pixel count, and `True` is not a budget.

        Args:
            field: The field under test.
            value: The value under test.
        """
        with pytest.raises(ValueError, match=f"RenderTarget {field} must be"):
            RenderTarget("image", **{field: value})

    @pytest.mark.parametrize("ratio", [0, -1.0, float("inf"), float("nan"), True, "2"])
    def test_a_pixel_ratio_that_is_not_a_positive_number_is_refused(self, ratio):
        """The ratio multiplies a canvas; zero, negative, non-finite and non-numeric ratios are each refused.

        Args:
            ratio: The ratio under test.
        """
        with pytest.raises(ValueError, match="pixel_ratio must be a positive number"):
            RenderTarget("window", pixel_ratio=ratio)

    def test_numbers_are_normalised(self):
        """An integer ratio is stored as a float and a numpy integer canvas as an int."""
        import numpy as np

        target = RenderTarget("image", width=np.int64(640), pixel_ratio=2)
        assert (type(target.width), target.width) == (int, 640), target.width
        assert (type(target.pixel_ratio), target.pixel_ratio) == (float, 2.0), (
            target.pixel_ratio
        )


class TestTheRequestAViewMakes:
    """`view_request` — the region from the view, the canvas and budget from the target."""

    def test_the_request_combines_the_view_region_with_the_target(self):
        """Region, canvas, ratio and budget each come from where the rule says.

        Test scenario:
            The two sides are built by different routes: `view_request` on one, the `ViewRequest` constructor with
            the same parts on the other.
        """
        region = Bounds(0.0, 0.0, 10.0, 5.0, crs=4326)
        target = RenderTarget("html", width=800, height=400, pixel_ratio=2.0)
        request = target.view_request(Viewport(4326, bounds=region))
        expected = ViewRequest(
            bounds=region, width=800, height=400, pixel_ratio=2.0, budget=1_000_000
        )
        assert request == expected, request

    def test_explicit_bounds_are_reprojected_into_the_view_crs(self):
        """The reader is asked in the view's CRS, so a degree box on a metre view arrives in metres."""
        request = RenderTarget("window").view_request(
            Viewport(3857), bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
        )
        assert request.crs == 3857, request.crs
        assert request.bounds.xmax == pytest.approx(111319.49, abs=1.0), request.bounds

    def test_an_unframed_view_asks_for_the_whole_source(self):
        """No region means the reader decides, and the budget still applies."""
        request = RenderTarget("window").view_request(Viewport(4326))
        assert request.bounds is None, request.bounds
        assert request.budget == DEFAULT_BUDGETS["window"], request.budget

    @pytest.mark.parametrize(
        "kind, width, height, side",
        [
            ("image", 640, 480, 554),
            ("html", 800, 400, 565),
            ("window", 4000, 4000, 2000),
        ],
        ids=["small-still", "small-page", "canvas-over-budget"],
    )
    def test_a_named_canvas_sizes_the_read_within_the_budget(
        self, kind, width, height, side
    ):
        """A target with a canvas reads what the canvas can show, and never more than its budget.

        Args:
            kind: The target kind, which sets the budget.
            width: The canvas width.
            height: The canvas height.
            side: The square side the request must size a decimated read to.

        Test scenario:
            `view_request` always sets a budget, and `ViewRequest.side()` preferred the budget to the canvas, so a
            640x480 still sized its read to 2,000 cells a side — 13 times the pixels it can show — against the
            `DEFAULT_BUDGETS` note that a still with a canvas "is sized from the canvas anyway, within the budget".
            A canvas bigger than the budget is still held to the budget.
        """
        request = RenderTarget(kind, width=width, height=height).view_request()
        assert request.side() == side, request

    @pytest.mark.parametrize(
        "crs, expected",
        [
            (4326, [-10.0, 35.0, 30.0, 60.0]),
            (3857, [-1113194.9, 4163881.1, 3339584.7, 8399737.9]),
        ],
        ids=["in-degrees", "reprojected"],
    )
    def test_a_box_domain_is_the_region_asked_for(self, crs, expected):
        """A view framed by a `(west, south, east, north)` domain asks for that box, in the view's CRS.

        Args:
            crs: The view's CRS.
            expected: The requested bbox, rounded to a decimetre.

        Test scenario:
            `view_request` read only `view.bounds`, and a view holds bounds or a domain, never both — so every
            domain-framed view sent the reader for the whole source.
        """
        request = RenderTarget().view_request(
            Viewport(crs, domain=(-10.0, 35.0, 30.0, 60.0))
        )
        assert [round(edge, 1) for edge in request.as_bbox()] == expected, (
            request.bounds
        )
        assert request.crs == crs, request.crs

    def test_a_named_domain_is_refused_rather_than_read_as_the_whole_source(self):
        """A region name resolves in the static tier, which `base` cannot import, so the request says so.

        Test scenario:
            ``Viewport(3857, domain="europe")`` made a request with no region at all, reading the whole source for a
            view of Europe without a word.
        """
        view = Viewport(3857, domain="europe")
        with pytest.raises(ValueError, match="named domain 'europe'"):
            RenderTarget().view_request(view)

    def test_a_camera_supplies_no_region(self):
        """A 3-D view has no rectangle to read; the target's budget is what limits it."""
        request = RenderTarget("image", width=1024, height=768).view_request(
            Camera((0.0, -10.0, 5.0))
        )
        assert request.bounds is None, request.bounds
        assert (request.width, request.height, request.budget) == (
            1024,
            768,
            4_000_000,
        ), request

    def test_bounds_the_view_projection_cannot_show_are_refused_as_a_reprojection(self):
        """`view_request` for a region off an orthographic view's globe says the reprojection failed.

        Test scenario:
            The docstring promises a `ValueError` when the bounds "cannot be reprojected into the viewport's CRS",
            but the one that surfaced was the `Bounds` constructor's "needs finite edges; got xmin=inf".
        """
        view = Viewport("+proj=ortho +lat_0=0 +lon_0=0")
        region = Bounds(170.0, -10.0, 180.0, 10.0, crs=4326)
        with pytest.raises(ValueError, match="cannot be reprojected into"):
            RenderTarget().view_request(view, bounds=region)

    def test_with_no_view_explicit_bounds_are_used_as_given(self):
        """Without a view there is no CRS to reproject into, so the bounds pass through."""
        region = Bounds(0.0, 0.0, 1.0, 1.0, crs=32618)
        assert RenderTarget("batch").view_request(bounds=region).bounds == region

    def test_something_that_is_not_a_view_is_refused(self):
        """A bare CRS is not a view."""
        target = RenderTarget("window")
        with pytest.raises(ValueError, match="needs a Viewport, a Camera or None"):
            target.view_request(4326)

    def test_bounds_that_are_not_a_bounds_are_refused(self):
        """A bbox list cannot say which edge is which."""
        target = RenderTarget("window")
        with pytest.raises(ValueError, match="needs bounds as a Bounds"):
            target.view_request(bounds=[0, 0, 1, 1])


class TestSerialisation:
    """A target round-trips, and an unset budget stays unset."""

    @pytest.mark.parametrize(
        "target",
        [
            RenderTarget(),
            RenderTarget("html", budget=500_000),
            RenderTarget("image", width=1200, height=800, pixel_ratio=2.0),
        ],
        ids=["default", "html-budget", "retina-image"],
    )
    def test_a_target_survives_a_json_round_trip(self, target):
        """What goes out through JSON text comes back equal.

        Args:
            target: The target under test.
        """
        rebuilt = RenderTarget.from_dict(json.loads(json.dumps(target.to_dict())))
        assert rebuilt == target, f"the target changed in a round trip: {rebuilt!r}"

    def test_an_unset_budget_is_not_written_as_the_default(self):
        """A stored target keeps following the kind's default rather than freezing today's number."""
        assert "budget" not in RenderTarget("html").to_dict()

    def test_from_dict_defaults_the_kind_to_a_window(self):
        """An empty dict is the default target."""
        assert RenderTarget.from_dict({}) == RenderTarget("window")

    def test_from_dict_refuses_an_unknown_key(self):
        """A key a newer writer added is refused, not dropped."""
        stored = {"kind": "html", "dpi": 300}
        with pytest.raises(ValueError, match=r"unknown keys \['dpi'\]"):
            RenderTarget.from_dict(stored)
