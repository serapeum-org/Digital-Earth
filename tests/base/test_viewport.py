"""`Viewport` and `Camera` — the view as a value that round-trips (DE-21, #282).

On every tier the view was loose state on the facade: three attributes on the static `Map`, mutated in place to
animate a rotation, and no camera at all on the 3-D tier. These cover the values that replace that.
"""

import json
import math
from types import SimpleNamespace

import pytest

from digitalearth.base.spec import DEFAULT_VIEW_ANGLE, Bounds, Camera, Viewport


class TestViewport:
    """What a flat map shows."""

    def test_the_default_view_matches_the_static_map(self):
        """Web Mercator, unframed, flat — what `Map()` does with no arguments."""
        view = Viewport()
        assert view.crs == 3857, view.crs
        assert view.bounds is None, view.bounds
        assert view.globe is False, view.globe

    @pytest.mark.parametrize("crs", [None, True, False])
    def test_a_view_needs_a_display_crs(self, crs):
        """`None` and booleans name no reference system to draw in.

        Args:
            crs: The CRS under test.
        """
        with pytest.raises(ValueError, match="needs a display CRS"):
            Viewport(crs)

    @pytest.mark.parametrize(
        "crs, kind",
        [(4326.0, "float"), ([4326], "list"), (object(), "object")],
        ids=["float", "list", "object"],
    )
    def test_a_crs_a_figure_cannot_store_is_refused_when_the_view_is_built(
        self, crs, kind
    ):
        """A CRS spelling with no stored form is refused by the constructor, not when the figure is saved.

        Args:
            crs: A CRS value pyramids cannot read.
            kind: The type name the message reports.

        Test scenario:
            `Viewport(4326.0)` built, and the error surfaced only from `to_dict` — at save time, far from the line
            that made the view.
        """
        with pytest.raises(
            ValueError, match=f"Viewport.crs holds a {kind} that is not a readable CRS"
        ):
            Viewport(crs)

    @pytest.mark.parametrize("crs", [0, "", "junk"], ids=["zero", "empty", "junk"])
    def test_a_crs_that_names_no_reference_system_is_refused(self, crs):
        """A display CRS pyramids cannot read names nothing a map could be drawn in, so the view refuses it.

        Args:
            crs: A value with a written form that names no CRS.

        Test scenario:
            `Viewport(0)`, `Viewport("")` and `Viewport("junk")` built — `crs_to_json` writes any int or string as
            given — and failed only later, when `framed` asked pyramids to reproject into them.
        """
        with pytest.raises(ValueError, match="names no coordinate reference system"):
            Viewport(crs)

    def test_bounds_with_no_crs_are_named_as_such(self):
        """Stored bounds without a `crs` are refused for having none, not pointed at `framed`, which cannot help.

        Test scenario:
            A missing bounds `crs` reads as `None`, which is not the view's CRS, so the refusal said "use
            Viewport.framed(bounds), which reprojects" — and `framed` cannot reproject from no CRS.
        """
        stored = {"crs": 4326, "bounds": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}}
        with pytest.raises(ValueError, match="bounds carry no CRS"):
            Viewport.from_dict(stored)

    def test_a_crs_given_as_a_dict_builds_a_hashable_view(self):
        """A proj dict pyramids reads is held in its written spelling, so the view hashes."""
        view = Viewport({"proj": "longlat", "datum": "WGS84", "no_defs": True})
        assert isinstance(hash(view), int), view

    @pytest.mark.parametrize(
        "domain",
        [(170.0, -50.0, -170.0, 50.0), (0.0, 50.0, 10.0, 40.0)],
        ids=["crosses-the-antimeridian", "south-above-north"],
    )
    def test_a_domain_box_with_its_corners_the_wrong_way_round_is_refused(self, domain):
        """A `(west, south, east, north)` box with west past east, or south past north, is refused by the view.

        Args:
            domain: An inverted box.

        Test scenario:
            The view held it, and `RenderTarget.view_request` then failed inside `Bounds` with "Bounds needs xmin <=
            xmax; got xmin=170.0, xmax=-170.0", naming neither the domain nor why — the static tier's `set_domain`
            refuses the same box and says it cannot cross the antimeridian.
        """
        with pytest.raises(ValueError, match="has its corners the wrong way round"):
            Viewport(4326, domain=domain)

    def test_bounds_and_a_domain_together_are_refused(self):
        """A view holds one region, so `bounds` and `domain` together are refused rather than ranked silently.

        Test scenario:
            `Viewport(4326, bounds=..., domain="europe")` held two competing regions with no stated precedence, so a
            renderer would have had to guess which one the map shows.
        """
        box = Bounds(0.0, 0.0, 10.0, 10.0, crs=4326)
        with pytest.raises(ValueError, match="bounds or a domain, not both"):
            Viewport(4326, bounds=box, domain="europe")

    def test_framing_a_view_on_bounds_replaces_its_domain(self):
        """`framed` sets the region the map shows, so a named domain the view had gives way to it."""
        framed = Viewport(4326, domain="europe").framed(
            Bounds(0.0, 0.0, 10.0, 5.0, crs=4326)
        )
        assert (framed.domain, framed.bounds.as_bbox()) == (
            None,
            [0.0, 0.0, 10.0, 5.0],
        ), framed

    def test_bounds_in_another_crs_are_refused_by_the_constructor(self):
        """A rectangle in degrees read as metres draws the wrong place, silently.

        Test scenario:
            The constructor refuses and names `framed`, which reprojects; so a view never holds a region its own CRS
            cannot place.
        """
        box = Bounds(0.0, 0.0, 10.0, 10.0, crs=4326)
        with pytest.raises(ValueError, match=r"use Viewport\.framed"):
            Viewport(3857, bounds=box)

    def test_bounds_in_a_differently_spelled_but_equal_crs_are_accepted(self):
        """`4326` and `"EPSG:4326"` are one system, so neither is refused for the other."""
        view = Viewport("EPSG:4326", bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
        assert view.bounds.crs == 4326, view.bounds

    def test_a_crs_object_view_holds_bounds_in_that_crs(self):
        """A view in a pyproj CRS object accepts bounds in the same object, frames, and makes requests.

        Test scenario:
            `GeoDataFrame.crs` is a CRS object. pyramids' `crs_equal` reads only int/str/None and answers False for
            an object compared with itself, so the constructor refused these bounds, `framed` refused the bounds it
            had just reprojected, and `RenderTarget.view_request` failed the same way — each with a message telling
            the caller to use `framed`.
        """
        from pyramids.base.crs import crs_from_user_input

        from digitalearth.base.spec import RenderTarget

        crs = crs_from_user_input(3857)
        held = Viewport(crs, bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=crs))
        framed = Viewport(crs).framed(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
        request = RenderTarget().view_request(
            Viewport(crs), bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
        )
        assert held.bounds.as_bbox() == [0.0, 0.0, 1.0, 1.0], held.bounds
        assert framed.bounds.xmax == pytest.approx(111319.49, abs=1.0), framed.bounds
        assert request.bounds.xmax == pytest.approx(111319.49, abs=1.0), request.bounds

    def test_bounds_must_be_a_bounds(self):
        """A bare list cannot say which edge is which."""
        with pytest.raises(ValueError, match="bounds must be a Bounds"):
            Viewport(4326, bounds=[0, 0, 1, 1])

    def test_a_domain_box_is_stored_as_a_tuple_of_floats(self):
        """A list passed in is frozen, and integers become floats."""
        assert Viewport(domain=[-10, 35, 30, 60]).domain == (-10.0, 35.0, 30.0, 60.0)

    @pytest.mark.parametrize(
        "domain", ["", "   ", [0, 1, 2], 5, [0, 1, 2, float("nan")]]
    )
    def test_a_domain_that_is_neither_a_name_nor_four_numbers_is_refused(self, domain):
        """An empty name, the wrong count, a scalar or a non-finite edge.

        Args:
            domain: The domain under test.
        """
        with pytest.raises(ValueError, match="domain"):
            Viewport(domain=domain)

    def test_globe_must_be_a_boolean(self):
        """`"yes"` is truthy and says nothing."""
        with pytest.raises(ValueError, match="globe must be True or False"):
            Viewport(globe="yes")

    def test_framed_returns_a_new_view_and_leaves_the_original_alone(self):
        """A change of view is a new value, never an edit.

        Test scenario:
            The static tier animates a rotation by assigning `self.crs` per frame and restoring it afterwards. A view
            that is a value makes each frame its own view instead.
        """
        view = Viewport(4326)
        framed = view.framed(Bounds(0.0, 0.0, 10.0, 5.0, crs=4326))
        assert view.bounds is None, "the original view must not change"
        assert framed.bounds == Bounds(0.0, 0.0, 10.0, 5.0, crs=4326), framed.bounds

    def test_framed_reprojects_bounds_given_in_another_crs(self):
        """Framing on a degree box in a metre CRS lands the box in metres.

        Test scenario:
            One degree of longitude at the equator is 111,319.49 m in Web Mercator; the reprojection is pyramids', so
            this asserts the edge to the metre rather than re-deriving it.
        """
        framed = Viewport(3857).framed(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
        assert framed.bounds.crs == 3857, framed.bounds.crs
        assert framed.bounds.xmax == pytest.approx(111319.49, abs=1.0), framed.bounds

    def test_framed_refuses_something_that_is_not_bounds(self):
        """A bbox list is refused; the caller builds a `Bounds` so the ordering is named."""
        view = Viewport(4326)
        with pytest.raises(ValueError, match="needs a Bounds"):
            view.framed([0, 0, 1, 1])

    @pytest.mark.parametrize("epsg, expected", [(4326, False), (32618, True)])
    def test_needs_reproject_follows_the_shared_rule(self, epsg, expected):
        """Data already in the view's EPSG code needs no warp; data in another does.

        Args:
            epsg: The data's EPSG code.
            expected: Whether a warp is needed.
        """
        assert Viewport(4326).needs_reproject(SimpleNamespace(epsg=epsg)) is expected

    @pytest.mark.parametrize(
        "view",
        [
            Viewport(),
            Viewport(
                4326, bounds=Bounds(-10.0, 35.0, 30.0, 60.0, crs=4326), globe=True
            ),
            Viewport("+proj=ortho +lat_0=30 +lon_0=10", domain="europe"),
            Viewport(3857, domain=(-10.0, 35.0, 30.0, 60.0)),
        ],
        ids=["default", "framed-globe", "ortho-named-domain", "domain-box"],
    )
    def test_a_view_survives_a_json_round_trip(self, view):
        """What goes out through JSON text comes back equal.

        Args:
            view: The view under test, built by its constructor.
        """
        rebuilt = Viewport.from_dict(json.loads(json.dumps(view.to_dict())))
        assert rebuilt == view, f"the view changed in a round trip: {rebuilt!r}"

    def test_from_dict_needs_a_crs(self):
        """A stored view with no CRS cannot be drawn; the key is named."""
        stored = {"globe": True}
        with pytest.raises(ValueError, match="needs 'crs'"):
            Viewport.from_dict(stored)

    def test_from_dict_refuses_an_unknown_key(self):
        """A key a newer writer added is refused, not dropped."""
        stored = {"crs": 4326, "zoom": 3}
        with pytest.raises(ValueError, match=r"unknown keys \['zoom'\]"):
            Viewport.from_dict(stored)


class TestCameraConstruction:
    """A camera that could not frame anything is refused."""

    def test_defaults_look_at_the_origin_with_z_up(self):
        """A position alone is enough; the rest has VTK's defaults."""
        camera = Camera((0.0, -10.0, 5.0))
        assert camera.focal_point == (0.0, 0.0, 0.0), camera.focal_point
        assert camera.view_up == (0.0, 0.0, 1.0), camera.view_up
        assert camera.view_angle == DEFAULT_VIEW_ANGLE, camera.view_angle
        assert camera.vertical_exaggeration == 1.0, camera.vertical_exaggeration

    def test_vectors_are_stored_as_tuples_of_floats(self):
        """A list of ints is frozen into floats, so equality does not depend on how it was spelled."""
        camera = Camera([0, -10, 5], focal_point=[1, 2, 3])
        assert camera.position == (0.0, -10.0, 5.0), camera.position
        assert camera.focal_point == (1.0, 2.0, 3.0), camera.focal_point

    @pytest.mark.parametrize(
        "position",
        [
            (0.0, 1.0),
            (0.0, 1.0, 2.0, 3.0),
            "abc",
            5.0,
            (0.0, float("nan"), 1.0),
            (True, 0.0, 1.0),
        ],
    )
    def test_a_position_that_is_not_three_finite_numbers_is_refused(self, position):
        """Two or four components, a string, a scalar, a `nan` and a boolean are each refused.

        Args:
            position: The position under test.
        """
        with pytest.raises(ValueError, match="position"):
            Camera(position)

    def test_a_camera_at_its_own_focal_point_is_refused(self):
        """Zero distance has no line of sight."""
        with pytest.raises(ValueError, match="looks at nothing"):
            Camera((1.0, 1.0, 1.0), focal_point=(1.0, 1.0, 1.0))

    def test_a_zero_view_up_is_refused(self):
        """The zero vector names no direction."""
        with pytest.raises(ValueError, match="zero vector"):
            Camera((0.0, -10.0, 0.0), view_up=(0.0, 0.0, 0.0))

    @pytest.mark.parametrize("view_up", [(0.0, 0.0, 1.0), (0.0, 0.0, -3.0)])
    def test_a_view_up_along_the_line_of_sight_is_refused(self, view_up):
        """Looking straight down with z up leaves "up on the screen" undefined.

        Args:
            view_up: A view-up parallel (or anti-parallel) to the line of sight.

        Test scenario:
            The message names the fix — a horizontal view-up — because straight down is the most common top view.
        """
        with pytest.raises(
            ValueError, match=r"parallel to the line of sight.*\(0, 1, 0\)"
        ):
            Camera((0.0, 0.0, 10.0), view_up=view_up)

    @pytest.mark.parametrize("angle", [0.0, 180.0, -5.0, 200.0])
    def test_a_view_angle_outside_zero_to_180_is_refused(self, angle):
        """A field of view of 0 or 180 degrees frames nothing.

        Args:
            angle: The angle under test.
        """
        with pytest.raises(ValueError, match="strictly between 0 and 180"):
            Camera((0.0, -10.0, 0.0), view_angle=angle)

    @pytest.mark.parametrize("factor", [0.0, -2.0])
    def test_a_non_positive_exaggeration_is_refused(self, factor):
        """Zero flattens the scene and a negative factor turns it upside down.

        Args:
            factor: The exaggeration under test.
        """
        with pytest.raises(ValueError, match="vertical_exaggeration must be positive"):
            Camera((0.0, -10.0, 0.0), vertical_exaggeration=factor)

    def test_two_parallel_views_at_different_scales_are_different_cameras(self):
        """A parallel projection's zoom is its scale, so two scales are two views.

        Test scenario:
            In a parallel projection neither the distance nor the view angle sets the zoom; VTK's
            `parallel_scale` does — half the view's height in world units. Without the field, a camera zoomed to
            a street and one zoomed to a country, placed at the same point, stored identically.
        """
        street = Camera((0.0, -10.0, 5.0), parallel=True, parallel_scale=50.0)
        country = Camera((0.0, -10.0, 5.0), parallel=True, parallel_scale=500_000.0)
        assert street != country, (
            "two parallel scales must describe two different views"
        )
        assert (street.parallel_scale, country.parallel_scale) == (50.0, 500_000.0)

    def test_parallel_scale_defaults_to_none_and_accepts_an_int(self):
        """Unset, the scale is `None` — the renderer fits the scene; given, it is stored as a float."""
        assert Camera((0.0, -10.0, 5.0), parallel=True).parallel_scale is None
        assert Camera((0.0, -10.0, 5.0), parallel_scale=20).parallel_scale == 20.0

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_parallel_scale_must_be_a_positive_finite_number(self, scale):
        """A scale of zero or less, or one that is not finite, frames nothing.

        Args:
            scale: The rejected scale.
        """
        with pytest.raises(ValueError, match="parallel_scale"):
            Camera((0.0, -10.0, 5.0), parallel=True, parallel_scale=scale)

    def test_parallel_must_be_a_boolean(self):
        """`1` is truthy and says nothing."""
        with pytest.raises(ValueError, match="parallel must be True or False"):
            Camera((0.0, -10.0, 0.0), parallel=1)


class TestLookingAtAPoint:
    """`look_at` places a camera in the terms a map reader uses."""

    @pytest.mark.parametrize(
        "azimuth, elevation, direction",
        [
            (0.0, 0.0, (0.0, 1.0, 0.0)),
            (90.0, 0.0, (1.0, 0.0, 0.0)),
            (180.0, 0.0, (0.0, -1.0, 0.0)),
            (270.0, 0.0, (-1.0, 0.0, 0.0)),
            (225.0, 0.0, (-math.sqrt(0.5), -math.sqrt(0.5), 0.0)),
            (0.0, 30.0, (0.0, math.cos(math.radians(30.0)), 0.5)),
        ],
        ids=["north", "east", "south", "west", "south-west", "north-30-up"],
    )
    def test_the_camera_sits_on_the_stated_bearing_and_elevation(
        self, azimuth, elevation, direction
    ):
        """Azimuth is a compass bearing from the focal point to the camera, with north along +y.

        Args:
            azimuth: The bearing under test.
            elevation: The elevation under test.
            direction: The unit vector from focal point to camera that the bearing and elevation mean.

        Test scenario:
            Checked against unit vectors written out by hand rather than by re-running the same trigonometry, so a
            swapped sine and cosine — which puts "north" to the east — fails here.
        """
        focal = (100.0, 200.0, 10.0)
        camera = Camera.look_at(
            focal, azimuth=azimuth, elevation=elevation, distance=50.0
        )
        expected = tuple(f + 50.0 * d for f, d in zip(focal, direction))
        assert camera.position == pytest.approx(expected, abs=1e-9), camera.position

    @pytest.mark.parametrize(
        "azimuth, elevation", [(225.0, 30.0), (10.0, -20.0), (359.0, 89.0), (0.0, 45.0)]
    )
    def test_azimuth_elevation_and_distance_read_back_what_look_at_was_given(
        self, azimuth, elevation
    ):
        """The read-back properties invert `look_at`, which is what a stored view needs.

        Args:
            azimuth: The bearing given.
            elevation: The elevation given.
        """
        camera = Camera.look_at(
            (1.0, 2.0, 3.0), azimuth=azimuth, elevation=elevation, distance=7.5
        )
        assert camera.azimuth == pytest.approx(azimuth, abs=1e-9), camera.azimuth
        assert camera.elevation == pytest.approx(elevation, abs=1e-9), camera.elevation
        assert camera.distance == pytest.approx(7.5, abs=1e-12), camera.distance

    def test_looking_straight_down_needs_a_horizontal_view_up(self):
        """With the default z-up it is refused; with y-up it frames a map-like top view."""
        with pytest.raises(ValueError, match="parallel to the line of sight"):
            Camera.look_at((0.0, 0.0, 0.0), azimuth=0.0, elevation=90.0, distance=10.0)
        top = Camera.look_at(
            (0.0, 0.0, 0.0),
            azimuth=0.0,
            elevation=90.0,
            distance=10.0,
            view_up=(0.0, 1.0, 0.0),
        )
        assert top.position == pytest.approx((0.0, 0.0, 10.0), abs=1e-9), top.position

    @pytest.mark.parametrize(
        "camera",
        [
            Camera.look_at((0.0, 0.0, 0.0), azimuth=360.0, elevation=0.0, distance=1.0),
            Camera((-1e-20, 1.0, 0.0)),
        ],
        ids=["look-at-360", "a-hair-west-of-north"],
    )
    def test_a_bearing_a_hair_west_of_north_is_zero_not_360(self, camera):
        """The bearing is in `[0, 360)`, so a camera a rounding error west of due north reads as `0`.

        Args:
            camera: A camera whose x offset is a tiny negative number.

        Test scenario:
            `degrees(atan2(dx, dy)) % 360.0` rounds a tiny negative angle up to `360.0` — outside the documented
            range, and a bucket-by-bearing caller's ``< 360`` check fails on it.
        """
        assert camera.azimuth == 0.0, camera.azimuth

    def test_a_camera_directly_above_reports_azimuth_zero(self):
        """No horizontal offset has no bearing; the property says 0 rather than raising."""
        assert Camera((0.0, 0.0, 10.0), view_up=(0.0, 1.0, 0.0)).azimuth == 0.0

    @pytest.mark.parametrize("distance", [0.0, -1.0, float("inf")])
    def test_look_at_refuses_a_distance_that_places_no_camera(self, distance):
        """A camera needs to be somewhere finite and away from what it looks at.

        Args:
            distance: The distance under test.
        """
        with pytest.raises(ValueError, match="distance"):
            Camera.look_at(
                (0.0, 0.0, 0.0), azimuth=0.0, elevation=0.0, distance=distance
            )

    @pytest.mark.parametrize("elevation", [120.0, -90.5, 270.0])
    def test_look_at_refuses_an_elevation_past_the_vertical(self, elevation):
        """An elevation outside `[-90, 90]` is refused, naming it, rather than flipping the camera round.

        Args:
            elevation: An elevation past straight up or straight down.

        Test scenario:
            ``elevation=120`` placed the camera on the far side of the vertical — to the south at 60 degrees for an
            azimuth of 0 — so `azimuth` and `elevation` read back as `(180, 60)`, not what was asked for. A caller
            animating the elevation past 90 saw the camera jump to the opposite side with no error.
        """
        with pytest.raises(ValueError, match=r"elevation between -90 and 90"):
            Camera.look_at(
                (0.0, 0.0, 0.0),
                azimuth=0.0,
                elevation=elevation,
                distance=10.0,
                view_up=(0.0, 1.0, 0.0),
            )

    @pytest.mark.parametrize("elevation", [90.0, -90.0])
    def test_look_at_accepts_straight_up_and_straight_down(self, elevation):
        """The ends of the range are real views, and read back as they were given.

        Args:
            elevation: Straight down onto the focal point, or straight up at it.
        """
        camera = Camera.look_at(
            (0.0, 0.0, 0.0),
            azimuth=0.0,
            elevation=elevation,
            distance=10.0,
            view_up=(0.0, 1.0, 0.0),
        )
        assert camera.elevation == pytest.approx(elevation), camera

    def test_look_at_carries_the_other_settings_through(self):
        """Projection, field of view and exaggeration are passed to the camera it builds."""
        camera = Camera.look_at(
            (0, 0, 0),
            azimuth=90,
            elevation=20,
            distance=3,
            view_angle=45,
            parallel=True,
            parallel_scale=12,
            vertical_exaggeration=4,
        )
        assert (
            camera.view_angle,
            camera.parallel,
            camera.parallel_scale,
            camera.vertical_exaggeration,
        ) == (45.0, True, 12.0, 4.0)


class TestCameraSerialisation:
    """A camera round-trips, so a view can be captured and replayed (#204)."""

    @pytest.mark.parametrize(
        "camera",
        [
            Camera((0.0, -10.0, 5.0)),
            Camera.look_at(
                (10.0, 20.0, 0.0), azimuth=225.0, elevation=30.0, distance=500.0
            ),
            Camera(
                (1.0, 2.0, 30.0),
                focal_point=(1.0, 2.0, 0.0),
                view_up=(0.0, 1.0, 0.0),
                view_angle=45.0,
                parallel=True,
                parallel_scale=250.0,
                vertical_exaggeration=3.0,
            ),
        ],
        ids=["position-only", "look-at", "every-field"],
    )
    def test_a_camera_survives_a_json_round_trip(self, camera):
        """What goes out through JSON text comes back equal, float for float.

        Args:
            camera: The camera under test, built by its constructor or by `look_at`.
        """
        rebuilt = Camera.from_dict(json.loads(json.dumps(camera.to_dict())))
        assert rebuilt == camera, f"the camera changed in a round trip: {rebuilt!r}"

    def test_to_dict_writes_every_setting_including_the_defaults(self):
        """A stored camera records the settings in force, so it reproduces the view if a default changes.

        Test scenario:
            Omitting defaults meant comparing floats for equality with the default to decide what to leave out
            (SonarCloud python:S1244); writing every field removes the comparison and makes the record complete.
        """
        assert Camera((0.0, -10.0, 5.0)).to_dict() == {
            "position": [0.0, -10.0, 5.0],
            "focal_point": [0.0, 0.0, 0.0],
            "view_up": [0.0, 0.0, 1.0],
            "view_angle": 30.0,
            "parallel": False,
            "parallel_scale": None,
            "vertical_exaggeration": 1.0,
        }

    def test_from_dict_needs_a_position(self):
        """A stored camera with no position cannot be placed; the key is named."""
        stored = {"focal_point": [0, 0, 0]}
        with pytest.raises(ValueError, match="needs 'position'"):
            Camera.from_dict(stored)

    def test_from_dict_refuses_an_unknown_key(self):
        """A key a newer writer added is refused, not dropped."""
        stored = {"position": [0, -10, 5], "roll": 15}
        with pytest.raises(ValueError, match=r"unknown keys \['roll'\]"):
            Camera.from_dict(stored)
