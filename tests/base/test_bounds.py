"""`Bounds` — one rectangle type, with its ordering named rather than implied (DE-11, #270).

Two orderings were live in the static tier at once — matplotlib's ``[xmin, xmax, ymin, ymax]`` and cleopatra's
``[xmin, ymin, xmax, ymax]`` — plus a third convention that hid an EPSG:4326 assumption inside a bare tuple.
These cover the type that replaces all three.
"""

import pytest

from digitalearth.base.spec import Bounds
from digitalearth.base.spec.bounds import same_crs

#: A projected CRS with no authority code of its own — the kind a `.prj` file gives a `GeoDataFrame`. PROJ's
#: identification guesses EPSG:23031 (ED50 / UTM 31N) for it at 70% confidence, which is a different datum.
_CODELESS_UTM = "+proj=utm +zone=31 +ellps=intl +units=m +no_defs"


@pytest.fixture
def identifications(monkeypatch):
    """Count every PROJ identification (`CRS.to_epsg` / `CRS.to_authority`) made while a test runs.

    Args:
        monkeypatch: pytest's patcher, used to wrap the two pyproj methods.

    Returns:
        A dict whose ``"count"`` the wrapped methods increment.
    """
    import pyproj

    seen = {"count": 0}
    for name in ("to_epsg", "to_authority"):
        original = getattr(pyproj.CRS, name)
        monkeypatch.setattr(pyproj.CRS, name, _counted(original, seen))
    return seen


def _counted(method, seen):
    """Wrap a pyproj method so each call increments ``seen["count"]``."""

    def wrapper(self, *args, **kwargs):
        seen["count"] += 1
        return method(self, *args, **kwargs)

    return wrapper


class TestTheOrderingIsNamed:
    """The two conventions are methods, so a call site states which it wants."""

    def test_the_two_orderings_differ_and_both_are_available(self):
        """One rectangle answers both spellings, and they are genuinely different.

        Test scenario:
            This is the whole point of the type. A bare 4-sequence cannot say which order it is in, so a caller
            that guessed wrong got a silently wrong extent. If these two ever returned the same thing the type
            would be pointless, so the difference is asserted, not assumed.
        """
        box = Bounds(0.0, 10.0, 4.0, 20.0, crs=4326)
        assert box.as_bbox() == [0.0, 10.0, 4.0, 20.0], (
            f"bbox order is [xmin, ymin, xmax, ymax], got {box.as_bbox()}"
        )
        assert box.as_mpl() == [0.0, 4.0, 10.0, 20.0], (
            f"matplotlib order is [xmin, xmax, ymin, ymax], got {box.as_mpl()}"
        )

    def test_each_builder_round_trips_its_own_ordering(self):
        """`from_bbox` and `from_mpl` are inverses of their matching readers.

        Test scenario:
            A builder that transposed its input would produce a rectangle that still looks plausible — the
            failure only shows as a wrong picture. Round-tripping each ordering pins them independently.
        """
        bbox = [1.0, 2.0, 3.0, 4.0]
        assert Bounds.from_bbox(bbox, crs=4326).as_bbox() == bbox, "bbox round trip"
        mpl = [1.0, 3.0, 2.0, 4.0]
        assert Bounds.from_mpl(mpl, crs=4326).as_mpl() == mpl, "matplotlib round trip"

    def test_the_two_builders_disagree_on_the_same_input(self):
        """Feeding one sequence to both builders yields different rectangles.

        Test scenario:
            Direct evidence that the ordering matters: if a caller picks the wrong builder the rectangle is
            wrong, which is exactly the class of bug the bare sequence allowed silently.
        """
        seq = [0.0, 8.0, 2.0, 9.0]
        assert Bounds.from_bbox(seq, crs=4326) != Bounds.from_mpl(seq, crs=4326), (
            "the two orderings must not collapse onto the same rectangle"
        )

    @pytest.mark.parametrize("bad", [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], []])
    def test_a_sequence_of_the_wrong_length_is_refused(self, bad):
        """Anything but four values raises, naming the argument.

        Args:
            bad: A sequence that is not four values.

        Test scenario:
            Silently taking the first four, or unpacking short, would produce a rectangle from garbage.
        """
        with pytest.raises(ValueError, match="exactly 4 values"):
            Bounds.from_bbox(bad, crs=4326)


class TestARectangleMustBeOne:
    """A rectangle that cannot bound anything is refused where it is built."""

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((10.0, 0.0, 1.0, 1.0), "xmin <= xmax"),
            ((0.0, 10.0, 1.0, 1.0), "ymin <= ymax"),
        ],
    )
    def test_an_inverted_rectangle_raises(self, args, match):
        """Corners the wrong way round raise rather than drawing nothing.

        Args:
            args: The inverted edges.
            match: The part of the message that must name which pair is wrong.

        Test scenario:
            matplotlib accepts inverted limits and quietly flips the axis; cleopatra draws an empty image.
            Neither reports the mistake, so it has to be caught at construction.
        """
        with pytest.raises(ValueError, match=match):
            Bounds(*args, crs=4326)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_edge_raises(self, bad):
        """A `NaN` or infinite edge is refused.

        Args:
            bad: The non-finite edge value.

        Test scenario:
            A NaN edge propagates into axes limits and produces a blank figure with no error — the same
            silent-failure shape the off-limb work spent Wave 0 removing.
        """
        with pytest.raises(ValueError, match="finite edges"):
            Bounds(0.0, 0.0, bad, 1.0, crs=4326)

    def test_a_degenerate_rectangle_is_allowed(self):
        """Zero width or height is a point or a line, not an error.

        Test scenario:
            A single-point layer has zero extent, and refusing it would make auto-framing fail on exactly the
            input a user is most likely to be debugging.
        """
        assert Bounds(1.0, 1.0, 1.0, 1.0, crs=4326).as_bbox() == [1.0, 1.0, 1.0, 1.0], (
            "a zero-area rectangle must be constructible"
        )

    def test_no_coordinates_at_all_is_refused(self):
        """A rectangle cannot be measured from nothing.

        Test scenario:
            The empty case reaches here whenever a layer's geometry was entirely filtered out — every point
            off the limb of a globe CRS, say. `min()` over an empty sequence raises `ValueError` with a
            message about `arg is an empty sequence`, which says nothing about bounds; this names the call.
        """
        with pytest.raises(ValueError, match="at least one coordinate pair"):
            Bounds.from_points([], [], crs=4326)


class TestOperations:
    """`union` and `padded` are what auto-framing is built from."""

    def test_union_encloses_both(self):
        """The union covers both rectangles and nothing less.

        Test scenario:
            Auto-framing (ST-13, WB-5, TD-2) is a union over layer bounds, so this is the operation that has
            to be right before any of them can be.
        """
        a = Bounds(0.0, 0.0, 2.0, 2.0, crs=3857)
        b = Bounds(1.0, -5.0, 3.0, 1.0, crs=3857)
        assert a.union(b).as_bbox() == [0.0, -5.0, 3.0, 2.0], (
            f"the union must enclose both, got {a.union(b).as_bbox()}"
        )

    def test_union_is_symmetric(self):
        """Argument order does not change the answer.

        Test scenario:
            A union that depended on which side it was called from would make auto-framing depend on layer
            order, which is the kind of bug that only shows on a map with three layers.
        """
        a = Bounds(0.0, 0.0, 2.0, 2.0, crs=3857)
        b = Bounds(1.0, -5.0, 3.0, 1.0, crs=3857)
        assert a.union(b) == b.union(a), "union must not depend on argument order"

    def test_union_across_crss_is_refused(self):
        """Two rectangles in different CRSs cannot be merged by ignoring that.

        Test scenario:
            The numbers would combine happily and describe a region nobody asked for. Carrying the CRS is what
            lets this be caught; the message says to reproject first.
        """
        with pytest.raises(ValueError, match="one CRS"):
            Bounds(0.0, 0.0, 1.0, 1.0, crs=4326).union(
                Bounds(0.0, 0.0, 1.0, 1.0, crs=3857)
            )

    def test_padded_grows_by_a_fraction_of_the_span(self):
        """A margin is proportional to the data, not an absolute distance.

        Test scenario:
            An absolute pad cannot serve both a 10-metre and a 10-degree extent; a fraction does.
        """
        assert Bounds(0.0, 0.0, 10.0, 10.0, crs=4326).padded(0.1).as_bbox() == [
            -1.0,
            -1.0,
            11.0,
            11.0,
        ], "a 10% pad on a span of 10 adds 1 on each side"

    def test_padding_that_would_invert_the_rectangle_raises(self):
        """Shrinking past the centre is refused, and the error names the padding that did it.

        Test scenario:
            A negative pad is legitimate for tightening a frame, but one large enough to cross the centre
            produces an inverted rectangle, which must not escape as a silently flipped axis. Leaving the
            constructor to catch it reported `xmin=7.5, xmax=2.5` for a call written as `padded(-0.75)` —
            neither the method, nor the fraction, nor the reason appears, so the caller has to
            reverse-engineer where those numbers came from.
        """
        with pytest.raises(ValueError, match=r"padded\(-0.75\) would invert"):
            Bounds(0.0, 0.0, 10.0, 10.0, crs=4326).padded(-0.75)

    def test_two_spellings_of_one_crs_are_the_same_crs(self):
        """`4326` and `"EPSG:4326"` name the same system, so a union across them works.

        Test scenario:
            The type exists to stop a rectangle being measured against the wrong CRS. Comparing the spellings
            with `==` also refuses a great many *right* ones — an EPSG int from one source and an authority
            string from another are the same reference system, and pyramids already owns that normalisation.
        """
        merged = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326).union(
            Bounds(0.0, 0.0, 2.0, 2.0, crs="EPSG:4326")
        )
        assert merged.as_bbox() == [0.0, 0.0, 2.0, 2.0], (
            "two spellings of one CRS must union rather than raise"
        )

    def test_genuinely_different_crss_are_still_refused(self):
        """The guard still catches the case it was written for.

        Test scenario:
            Guards the boundary of the test above — normalising spellings must not normalise away a real
            mismatch, which would silently union degrees with metres.
        """
        with pytest.raises(ValueError, match="one CRS"):
            Bounds(0.0, 0.0, 1.0, 1.0, crs=4326).union(
                Bounds(0.0, 0.0, 2.0, 2.0, crs=3857)
            )

    def test_a_crs_object_is_the_same_crs_as_its_own_code(self):
        """`to_crs` into the CRS object a rectangle already carries returns the rectangle, not a reprojection.

        Test scenario:
            Compared through pyramids' `crs_equal`, which reads only int/str/None, a CRS object was never the same as
            anything — itself included — so `to_crs` reprojected a rectangle into its own CRS.
        """
        from pyramids.base.crs import crs_from_user_input

        crs = crs_from_user_input(3857)
        box = Bounds(0.0, 0.0, 1.0, 1.0, crs=crs)
        assert box.to_crs(crs) is box, (
            "the same CRS object must be recognised as the same CRS"
        )
        assert box.to_crs(3857) is box, "and so must its EPSG code"

    def test_a_crs_object_without_a_code_is_not_the_code_proj_guesses_for_it(self):
        """A code-less CRS object is the same CRS as its own definition, and not the EPSG code PROJ guesses for it.

        Test scenario:
            `same_crs` wrote a CRS object through PROJ's identification at 70% confidence, so this UTM zone on the
            International ellipsoid compared equal to EPSG:23031 (ED50 / UTM 31N, another datum) — while its own WKT,
            which names the same definition, did not. Two spellings of one CRS disagreed, and a `Viewport` in the
            object accepted a rectangle in the guessed code without reprojecting it.
        """
        from pyramids.base.crs import crs_from_user_input

        obj = crs_from_user_input(_CODELESS_UTM)
        assert same_crs(obj, 23031) is False, "a guessed code is not the object's CRS"
        assert same_crs(obj, obj.to_wkt()) is True, (
            "the object's own definition is its CRS"
        )

    def test_comparing_and_reprojecting_crs_objects_runs_no_proj_identification(
        self, identifications
    ):
        """`same_crs`, a no-op `to_crs` and `Viewport.framed` read a CRS object's definition, not the PROJ database.

        Args:
            identifications: Counts PROJ identification calls.

        Test scenario:
            Each comparison identified the object against the PROJ database, uncached — about 50-130 ms per call for a
            code-less CRS — so the static tier's `set_domain` on a `GeoDataFrame.crs` went from 0.008 s to 0.3 s. It
            made 2 identifications for one `same_crs`, 2 for a no-op `to_crs` and 5 for `framed`.
        """
        from pyramids.base.crs import crs_from_user_input

        from digitalearth.base.spec import Viewport

        obj = crs_from_user_input(_CODELESS_UTM)
        same_crs(obj, obj)
        Bounds(0.0, 0.0, 1.0, 1.0, crs=obj).to_crs(obj)
        Viewport(obj).framed(Bounds(2.0, 50.0, 3.0, 51.0, crs=4326))
        assert identifications["count"] == 0, (
            f"{identifications['count']} PROJ identifications"
        )

    def test_an_unreadable_crs_spelling_falls_back_to_plain_inequality(self):
        """A CRS neither pyramids nor equality can match is treated as different, not as an error.

        Test scenario:
            `same_crs` decides whether reprojection is needed, not whether input is valid — so a spelling
            pyramids cannot parse must answer "not the same" rather than raise out of `union`, where the
            caller would get a CRS-parsing traceback for what is really a mismatched-rectangle message.
        """
        with pytest.raises(ValueError, match="one CRS"):
            Bounds(0.0, 0.0, 1.0, 1.0, crs="not-a-crs-at-all").union(
                Bounds(0.0, 0.0, 2.0, 2.0, crs=4326)
            )

    @pytest.mark.parametrize(
        "crs", [4326.5, [4326], True], ids=["float", "list", "bool"]
    )
    def test_a_crs_with_no_written_form_is_refused_when_the_rectangle_is_built(
        self, crs
    ):
        """A CRS a figure could not store is refused by the constructor, naming the field.

        Args:
            crs: A value that names no CRS pyramids can read.

        Test scenario:
            The rectangle held the value and failed only when written — or, before that, when compared.
        """
        with pytest.raises(ValueError, match="Bounds.crs"):
            Bounds(0.0, 0.0, 1.0, 1.0, crs=crs)

    def test_comparing_a_float_crs_does_not_change_how_its_integer_compares(self):
        """After a float CRS is compared, the integer with the same value still matches itself.

        Test scenario:
            A CRS value with no written form was handed to pyramids' `crs_equal`, whose answers are cached by value
            without regard to type: `32634.0 == 32634` and they hash alike, so the float's "not the same" answer was
            reused for the integer. After `same_crs(32634.0, 32634)`, two rectangles in `crs=32634` refused to union
            ("got 32634 and 32634") and `to_crs(32634)` reprojected a rectangle into its own CRS. `32634` is used
            because no other test compares it, so the cache holds no earlier answer for it.
        """
        assert same_crs(32634.0, 32634) is False
        assert same_crs(32634, 32634) is True, "the integer CRS must still match itself"

    @pytest.mark.parametrize("value", [[4326], {4326}], ids=["list", "set"])
    def test_an_unhashable_crs_value_compares_as_different(self, value):
        """A list or a set names no CRS this can write, so it compares as different rather than raising.

        Args:
            value: An unhashable CRS value.

        Test scenario:
            It reached `crs_equal`, which caches by argument, and raised `TypeError: unhashable type: 'list'` —
            from `same_crs`, `union` and `to_crs` — where the docstring says such a value compares as different.
        """
        assert same_crs(value, 4326) is False

    def test_a_value_that_names_no_crs_is_not_equal_to_itself(self):
        """`crs=0` does not match `crs=0`, because neither names a reference system.

        Test scenario:
            pyramids' `crs_equal` carries a guard for exactly this, and an `==` or `is` shortcut in front of
            it re-opened the hole: CPython interns `0`, `''` and `True`, so two rectangles built with `crs=0`
            compared equal and unioned as though they agreed on a CRS neither had.
        """
        with pytest.raises(ValueError, match="one CRS"):
            Bounds(0.0, 0.0, 1.0, 1.0, crs=0).union(Bounds(0.0, 0.0, 2.0, 2.0, crs=0))

    def test_two_unset_crss_still_count_as_the_same(self):
        """Two rectangles that declare no CRS need no reprojection between them.

        Test scenario:
            The one case a shortcut would have been for. `crs_equal` already answers it, which is why no
            shortcut is needed — and `Bounds.from_points` defaults `crs` to None, so this is a real path.
        """
        merged = Bounds(0.0, 0.0, 1.0, 1.0, crs=None).union(
            Bounds(0.0, 0.0, 2.0, 2.0, crs=None)
        )
        assert merged.as_bbox() == [0.0, 0.0, 2.0, 2.0], "two unset CRSs must union"


class TestReprojection:
    """`to_crs` delegates to pyramids; this package does no coordinate maths."""

    def test_a_rectangle_outside_the_target_projection_names_both_crss(self):
        """A rectangle on the far side of an orthographic globe is refused as a failed reprojection.

        Test scenario:
            pyramids returns infinite coordinates for a corner the target projection cannot show, and the enclosing
            rectangle was handed straight to the constructor, which reported "Bounds needs finite edges; got
            xmin=inf" — naming neither CRS nor the reprojection that produced the infinity.
        """
        far_side = Bounds(170.0, -10.0, 180.0, 10.0, crs=4326)
        with pytest.raises(
            ValueError,
            match=r"in 4326 cannot be reprojected into '\+proj=ortho \+lat_0=0 \+lon_0=0'",
        ):
            far_side.to_crs("+proj=ortho +lat_0=0 +lon_0=0")

    def test_the_same_crs_is_returned_unchanged(self):
        """Reprojecting to the CRS it already has does no work.

        Test scenario:
            The common case on a map whose layers already share a display CRS. Doing the warp anyway would
            cost a transform per layer and introduce floating-point drift for nothing.
        """
        box = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
        assert box.to_crs(4326) is box, "an identical CRS must short-circuit"

    def test_a_round_trip_returns_the_original(self):
        """Out to Web Mercator and back recovers the input.

        Test scenario:
            The corners have to be transformed as ``(x, y)`` pairs in pyramids' order. A transposed call still
            returns four plausible numbers, so only a round trip catches it.
        """
        box = Bounds(-10.0, 40.0, 10.0, 60.0, crs=4326)
        back = box.to_crs(3857).to_crs(4326)
        for got, want in zip(back.as_bbox(), box.as_bbox()):
            assert got == pytest.approx(want, abs=1e-6), (
                f"round trip moved the rectangle: {back.as_bbox()} vs {box.as_bbox()}"
            )

    def test_reprojection_carries_the_new_crs(self):
        """The result says which CRS it is in.

        Test scenario:
            A reprojected rectangle still claiming the old CRS is worse than no CRS at all — it would pass the
            `union` guard while describing different ground.
        """
        assert Bounds(0.0, 0.0, 1.0, 1.0, crs=4326).to_crs(3857).crs == 3857, (
            "to_crs must record the CRS it produced"
        )


class TestItStaysAValue:
    """Frozen and comparable, so it can be compared and shared without defensive copying."""

    def test_two_equal_rectangles_compare_equal(self):
        """Equality is by value, which is what makes it testable.

        Test scenario:
            Identity comparison would make every assertion in this file need a manual field-by-field
            check. The two sides are built by different routes on purpose: one expression compared with
            itself would pass even if equality *were* identity.
        """
        assert Bounds(0.0, 1.0, 2.0, 3.0, crs=4326) == Bounds.from_bbox(
            [0.0, 1.0, 2.0, 3.0], crs=4326
        ), "the constructor and from_bbox must produce equal rectangles"

    def test_it_cannot_be_mutated(self):
        """A rectangle handed to a renderer cannot be changed underneath it.

        Test scenario:
            The 3-D tier's vertical exaggeration bug is the same shape — state mutated in place and read later
            by something that assumed it had not moved.
        """
        box = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
        with pytest.raises(AttributeError):
            box.xmin = 5.0
