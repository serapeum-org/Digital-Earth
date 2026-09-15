"""`Encoding` — a visual channel as a key rather than a parameter on every builder (DE-15, #274).

Colour is the only channel this package treats as first class; size, width, height/extrusion (#199) and text
(#191) are each a keyword bolted onto whichever builders needed them. These cover the type that makes a
channel a row in a table instead.
"""

import pytest

from digitalearth.base.spec import CHANNELS, Encoding, Scale


class TestBinding:
    """A channel is driven by a constant or by a field — the same type either way."""

    def test_a_constant_resolves_to_itself(self):
        """The plain `color="#f00"` case.

        Test scenario:
            The overwhelmingly common binding. If it needed a different type from the data-driven one, a
            renderer would be back to checking which keyword it got.
        """
        assert Encoding.constant("color", "#f00").resolve() == "#f00", (
            "a constant must resolve to the value it was given"
        )

    def test_a_constant_broadcasts_when_values_are_supplied(self):
        """One value per datum, so a renderer has a single code path.

        Test scenario:
            A renderer drawing per-feature styling should not have to branch on whether the channel varies.
        """
        assert Encoding.constant("opacity", 0.5).resolve([1, 2, 3]) == [
            0.5,
            0.5,
            0.5,
        ], "a constant must broadcast to one entry per datum"

    def test_naming_both_a_constant_and_a_field_is_refused(self):
        """An encoding says one thing about where its value comes from.

        Test scenario:
            Accepting both leaves the precedence undefined — and whichever wins, the caller meant the other
            half to have an effect.
        """
        with pytest.raises(ValueError, match="exactly one of"):
            Encoding(channel="color", value="#f00", field="elevation")

    def test_naming_neither_is_refused(self):
        """A binding to nothing is not a binding.

        Test scenario:
            It would resolve to `None` for every feature, which reads as "missing data" rather than as the
            construction error it is.
        """
        with pytest.raises(ValueError, match="exactly one of"):
            Encoding(channel="color")

    def test_an_undeclared_channel_is_refused_and_lists_the_declared_ones(self):
        """A channel name is checked against the table, and the error shows the table.

        Test scenario:
            The whole point of declaring channels is that a misspelt one stops being silently ignored — so
            the error has to be the thing that replaces the silence.
        """
        with pytest.raises(ValueError, match=r"not a visual channel.*'color'"):
            Encoding.constant("colour", "#f00")

    def test_a_constant_cannot_carry_a_mapping(self):
        """A scale on a constant is a caller who expected the scale to apply.

        Test scenario:
            Silently ignoring it draws one flat colour over data the caller meant to vary — the failure mode
            that looks like working software.
        """
        with pytest.raises(ValueError, match="cannot carry a scale"):
            Encoding(channel="color", value="#f00", scale=Scale.from_limits(0.0, 1.0))

    def test_a_field_encoding_needs_its_values_to_resolve(self):
        """Resolving a data-driven channel without the data says so.

        Test scenario:
            Returning the field *name* — a string, which a colour channel would happily accept — is the
            plausible wrong answer here.
        """
        with pytest.raises(ValueError, match="needs its values"):
            Encoding.by_field("color", "elevation").resolve()

    def test_a_field_encoding_needs_a_field_name(self):
        """An empty field name is not a binding to the data either.

        Test scenario:
            It comes from a caller threading an unset column through — `color_column or ""`. Left through, it
            would resolve every feature against a column nobody can name, and the error would surface far
            away, inside whatever tried the lookup.
        """
        with pytest.raises(ValueError, match="non-empty field name"):
            Encoding.by_field("color", "")


class TestResolving:
    """What a channel resolves to, and why it is not a rendered value."""

    def test_no_scale_passes_the_values_through(self):
        """A renderer holding its own mapping gets the raw field.

        Test scenario:
            cleopatra's glyphs take a `values=` array and colour it themselves; an encoding that insisted on
            mapping first would duplicate — and disagree with — that.
        """
        assert Encoding.by_field("color", "v").resolve([1, 2, 3]) == [1, 2, 3], (
            "without a scale the values are the answer"
        )

    def test_a_continuous_scale_resolves_to_ramp_positions(self):
        """Colour resolves to `[0, 1]`, not to RGBA.

        Test scenario:
            Turning a position into a colour needs a colormap, and colormaps are the renderer's business —
            each backend already has one. A position is the largest answer that is engine-neutral.
        """
        enc = Encoding.by_field("color", "v", scale=Scale.from_limits(0.0, 10.0))
        assert enc.resolve([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0], (
            "a continuous scale must resolve to positions along the ramp"
        )

    def test_a_value_outside_the_domain_is_clamped(self):
        """Out-of-domain data lands at the end of the ramp rather than off it.

        Test scenario:
            A renderer clamps; an encoding that returned 1.7 would push a colormap lookup out of range or
            leave a hole the legend cannot explain.
        """
        enc = Encoding.by_field("color", "v", scale=Scale.from_limits(0.0, 10.0))
        assert enc.resolve([-5.0, 17.0]) == [0.0, 1.0], (
            "values beyond the domain must clamp to its ends"
        )

    def test_a_classified_scale_steps(self):
        """Every value in a class resolves to that class's position.

        Test scenario:
            This is what makes the drawn colours match the legend's swatches. A classified scale that still
            resolved continuously would draw a smooth ramp under a stepped legend.
        """
        scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)
        assert Encoding.by_field("color", "v", scale=scale).resolve([1.0, 2.0]) == [
            0.0,
            0.0,
        ], "two values in one class must resolve identically"

    def test_a_categorical_scale_resolves_to_its_colours(self):
        """Categories are named, not measured, so they resolve to the colours themselves.

        Test scenario:
            A position along a ramp is meaningless for unordered categories — and an unseen category must
            read as missing rather than as the first class.
        """
        scale = Scale.categorical(["a", "b"], ["#f00", "#0f0"], missing="#ccc")
        assert Encoding.by_field("color", "class", scale=scale).resolve(
            ["b", "zzz"]
        ) == ["#0f0", "#ccc"], (
            "a category resolves to its colour, an unseen one to missing"
        )

    def test_a_non_finite_value_resolves_to_nothing(self):
        """Nodata is `None`, which is the renderer's cue to draw it as missing.

        Test scenario:
            Clamping NaN to either end of the ramp would colour a hole as real data — the silent wrongness
            Wave 0 removed from the 3-D and web tiers.
        """
        enc = Encoding.by_field("color", "v", scale=Scale.from_limits(0.0, 1.0))
        assert enc.resolve([float("nan"), "not a number"]) == [None, None], (
            "an unplaceable value must resolve to None"
        )


class TestChannelsAreTheGrowthAxis:
    """The property the type exists for: a new channel is a row, not a parameter."""

    def test_the_same_shape_drives_a_channel_that_is_not_colour(self):
        """`size` resolves in real units through the identical construction.

        Test scenario:
            This is the DoD's "one channel beyond colour". Nothing about `Encoding` is colour-specific — the
            only difference is the output range, which is a field rather than a new class.
        """
        enc = Encoding.by_field(
            "size",
            "population",
            scale=Scale.from_limits(0.0, 100.0),
            output_range=(4.0, 20.0),
        )
        assert enc.resolve([0.0, 50.0, 100.0]) == [4.0, 12.0, 20.0], (
            "a numeric channel must resolve into its own units"
        )

    def test_the_channels_already_asked_for_are_declared(self):
        """`height` and `text` exist before the features that need them.

        Test scenario:
            #199 (extrusion) and #191 (labels) are the named future capabilities this table is sized for;
            declaring them now is what makes those a fold rule per backend rather than a keyword per builder.
        """
        assert {"color", "opacity", "size", "width", "height", "text"} <= set(
            CHANNELS
        ), (
            f"the declared channels must cover the asked-for ones; got {sorted(CHANNELS)}"
        )

    def test_every_channel_declares_what_a_resolved_value_is(self):
        """A renderer can tell a colour from a magnitude without a lookup table of its own.

        Test scenario:
            `kind` is how a backend decides whether to hand the result to a colormap. A channel added without
            one would be undeclared in exactly the way this whole type removes.
        """
        kinds = {channel.kind for channel in CHANNELS.values()}
        assert kinds <= {"color", "number", "text"}, (
            f"every channel must declare a known kind; got {sorted(kinds)}"
        )


class TestItStaysAValue:
    """Frozen, so a style handed to a renderer cannot move underneath it."""

    def test_it_cannot_be_mutated(self):
        """Assigning to a field raises.

        Test scenario:
            A symbology is meant to be shareable between a layer and its legend; if either could edit it, the
            legend would stop describing the picture.
        """
        with pytest.raises(AttributeError):
            Encoding.constant("color", "#f00").value = "#0f0"

    def test_two_equal_encodings_compare_equal(self):
        """Equality is by value.

        Test scenario:
            What lets a test assert two tiers resolved the same styling rather than comparing field by field.
        """
        assert Encoding.constant("size", 6) == Encoding.constant("size", 6), (
            "equal bindings must compare equal"
        )
