"""The three colours a scale gives the values it cannot place on its ramp (ST-10).

`Scale` had one of them — `missing`, the colour of a nodata cell or an unseen category — and no way to say
what a value *below* the domain or *above* it should be. Those two were left to each renderer, which meant
they were left to matplotlib's defaults: a colormap paints an out-of-range value with its own end colour, so
a field clipped at `vmax` looks exactly like a field that peaks there.

These pin the declaration: two fields beside `missing`, one reader that answers "which of the three did the
caller state", one builder that states them, and the serialisation that carries all three — the part a
`Scale` shared by four tiers needs, since a figure stored by one is read back by another.
"""

from digitalearth.base.spec import Scale

#: Deliberately not the grey `digitalearth.base.symbology.MISSING_COLOR` names: these tests are about a
#: *stated* colour surviving, so the values have to be ones no default could supply.
MISSING = "#cccccc"
OVER = "#ff0000"
UNDER = "#0000ff"


class TestTheExtremeColoursAreFields:
    """`over` and `under` sit beside `missing`, with the same shape and the same default."""

    def test_a_scale_states_no_over_colour_by_default(self):
        """A scale built without one declares nothing, so a renderer leaves its colormap alone.

        Test scenario:
            The default has to be `None` rather than a colour: a scale that *always* carried an over colour
            would repaint every existing figure's top end the moment the field was added.
        """
        assert Scale.from_limits(0.0, 10.0).over is None, (
            "a scale states no over colour until one is given"
        )

    def test_a_scale_states_no_under_colour_by_default(self):
        """The same for the bottom end."""
        assert Scale.from_limits(0.0, 10.0).under is None, (
            "a scale states no under colour until one is given"
        )

    def test_the_constructor_takes_both(self):
        """They are ordinary fields, so the public constructor sets them.

        Test scenario:
            `missing` was always constructor-settable; the two new ones have to be too, or a caller
            rebuilding a scale by hand can carry one of the three and not the others.
        """
        scale = Scale(0.0, 10.0, missing=MISSING, over=OVER, under=UNDER)
        assert (scale.missing, scale.over, scale.under) == (MISSING, OVER, UNDER), (
            f"all three must be settable; got {scale}"
        )


class TestStatingTheExtremes:
    """`with_extremes` is how a scale that was derived from data is given its extreme colours."""

    def test_with_extremes_states_the_three_colours(self):
        """The builder records exactly what it was handed.

        Test scenario:
            A scale is normally *derived* — `from_values` measures the domain — and the extreme colours are a
            caller's styling, which arrives separately. So stating them has to be a step on an existing
            scale rather than only an argument to every builder.
        """
        scale = Scale.from_values([1.0, 9.0]).with_extremes(
            missing=MISSING, over=OVER, under=UNDER
        )
        assert scale.extremes() == {
            "missing": MISSING,
            "over": OVER,
            "under": UNDER,
        }, f"the three stated colours must come back; got {scale.extremes()}"

    def test_the_domain_is_untouched(self):
        """Stating a colour is not a reason to re-derive the limits."""
        assert Scale.from_values([1.0, 9.0]).with_extremes(over=OVER).as_limits() == (
            1.0,
            9.0,
        ), "with_extremes must carry the domain through unchanged"

    def test_an_unnamed_colour_is_left_as_it_was(self):
        """`None` means "leave this one alone", matching matplotlib's own `with_extremes`.

        Test scenario:
            A tier states the colours it was given and nothing else, so two calls have to compose: one
            stating `over` followed by one stating `missing` must end up with both.
        """
        scale = Scale.from_limits(0.0, 10.0).with_extremes(over=OVER)
        assert scale.with_extremes(missing=MISSING).over == OVER, (
            "a colour stated earlier must survive a later call that does not name it"
        )

    def test_the_original_scale_is_not_changed(self):
        """`Scale` is a frozen value, so the builder returns a new one.

        Test scenario:
            The same reason matplotlib's `with_extremes` copies: a scale derived once and shared across an
            animation's frames must not pick up one frame's styling.
        """
        original = Scale.from_limits(0.0, 10.0)
        original.with_extremes(missing=MISSING)
        assert original.missing is None, (
            "the scale the builder was called on must be unchanged"
        )

    def test_a_scale_that_states_none_reports_none(self):
        """The reader is empty rather than full of `None`s, so "did the caller state any" is one test."""
        assert Scale.from_limits(0.0, 10.0).extremes() == {}, (
            "a scale with no stated extreme colours must report an empty mapping"
        )

    def test_only_the_stated_colours_are_reported(self):
        """A half-stated scale reports its half.

        Test scenario:
            This is what lets a renderer hand the reader straight to matplotlib: the keys absent from it are
            exactly the extremes it must not overwrite.
        """
        reported = Scale.from_limits(0.0, 10.0).with_extremes(under=UNDER).extremes()
        assert reported == {"under": UNDER}, (
            f"only the stated colour must be reported; got {reported}"
        )


class TestTheExtremesAreCarriedByTheSerialisation:
    """A `Scale` crosses tiers as a dict, so a colour that does not survive that is not declared at all."""

    def test_to_dict_writes_the_two_new_colours(self):
        """Both appear under their own names, beside `missing`."""
        stored = Scale.from_limits(0.0, 10.0).with_extremes(over=OVER, under=UNDER)
        assert stored.to_dict() == {
            "vmin": 0.0,
            "vmax": 10.0,
            "over": OVER,
            "under": UNDER,
        }, f"to_dict must carry both; got {stored.to_dict()}"

    def test_to_dict_omits_a_colour_nobody_stated(self):
        """An unstated colour is absent rather than written as `null`.

        Test scenario:
            The same rule `missing` already followed. A figure's stored form stays the short one it was
            before the fields existed, so adding them does not rewrite every committed snapshot.
        """
        assert "over" not in Scale.from_limits(0.0, 10.0).to_dict(), (
            "an unstated extreme colour must not be written"
        )

    def test_a_round_trip_keeps_all_three(self):
        """`from_dict(to_dict())` is the test the description tier is built around."""
        scale = Scale.from_limits(-5.0, 5.0).with_extremes(
            missing=MISSING, over=OVER, under=UNDER
        )
        rebuilt = Scale.from_dict(scale.to_dict())
        assert rebuilt.extremes() == {
            "missing": MISSING,
            "over": OVER,
            "under": UNDER,
        }, f"a round trip must keep the three colours; got {rebuilt.extremes()}"

    def test_a_round_trip_of_a_categorical_scale_keeps_them_too(self):
        """The categorical branch of the serialisation carries them as well.

        Test scenario:
            `to_dict` writes categories and colours on their own branch, so a field added to only the
            numeric path would be dropped for a land-cover scale — the very case `missing` exists for.
        """
        scale = Scale.categorical(
            ["land", "sea"], ["#8b4513", "#1e90ff"]
        ).with_extremes(missing=MISSING, over=OVER)
        rebuilt = Scale.from_dict(scale.to_dict())
        assert rebuilt.extremes() == {"missing": MISSING, "over": OVER}, (
            f"a categorical round trip must keep them; got {rebuilt.extremes()}"
        )
