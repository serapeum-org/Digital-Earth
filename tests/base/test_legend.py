"""`LegendSpec` — a legend derived from the scale that was drawn (DE-19, #279).

Every tier built one its own way, and the web tier assembled its stored shape three separate times in one
module. Nothing structurally tied a swatch to the colour actually drawn, which is the disagreement #185
named. These cover the type that makes the agreement a construction rather than a maintenance task.
"""

import pytest

from digitalearth.base.spec import LegendEntry, LegendSpec, Scale


class TestFromACategoricalScale:
    """The case where the guarantee is total, because the scale carries the colours."""

    def test_the_swatches_are_the_colours_the_scale_draws(self):
        """The guarantee #185 asked for, for the kind of scale that can give it outright.

        Test scenario:
            A categorical `Scale` holds the category-to-colour mapping the renderer uses. Reading the
            swatches back out of it means the legend cannot disagree with the picture — there is only one
            mapping, not two that happen to match.
        """
        scale = Scale.categorical(
            ["land", "sea", "ice"], ["#8b4513", "#1e90ff", "#fff"]
        )
        legend = LegendSpec.from_scale(scale, title="cover")
        drawn = [scale.color_for(category) for category in scale.categories]
        assert [entry.color for entry in legend.entries] == drawn, (
            "every swatch must be the colour the scale draws that category"
        )

    def test_each_row_keeps_the_value_it_stands_for(self):
        """A row carries its category, not just a label.

        Test scenario:
            An interactive legend filters or highlights by the underlying value; a label alone would make
            every tier parse its own text back.
        """
        scale = Scale.categorical([1, 2], ["#f00", "#0f0"])
        assert [entry.value for entry in LegendSpec.from_scale(scale).entries] == [1, 2]

    def test_colours_cannot_be_passed_in_to_override_the_scale(self):
        """Passing colours for a categorical scale is ignored, deliberately.

        Test scenario:
            Accepting them would reintroduce the second source of truth the type removes — the legend would
            once again be free to disagree with what was drawn.
        """
        scale = Scale.categorical(["a"], ["#f00"])
        legend = LegendSpec.from_scale(scale, colors=["#0f0"])
        assert legend.entries[0].color == "#f00", "the scale's colour must win"


class TestFromAClassifiedScale:
    """One row per class, labelled by the range it covers."""

    def test_there_is_one_row_per_class(self):
        """`k` classes give `k` rows, not `k + 1` edges.

        Test scenario:
            The off-by-one every tier had to get right on its own when turning edges into swatches.
        """
        scale = Scale.from_values(list(range(10)), scheme="equal_interval", k=3)
        legend = LegendSpec.from_scale(scale, colors=["#eee", "#888", "#111"])
        assert len(legend.entries) == 3, "3 classes must give 3 legend rows"
        assert legend.kind == "graduated"

    def test_a_row_is_labelled_by_its_range(self):
        """The label says what the class covers.

        Test scenario:
            A graduated legend that showed only an edge would leave the reader guessing which side of it
            each swatch belongs to.
        """
        scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)
        legend = LegendSpec.from_scale(scale, colors=["#eee", "#111"], format=".1f")
        assert " – " in legend.entries[0].label, (
            f"expected a range label, got {legend.entries[0].label!r}"
        )
        assert legend.entries[0].label.startswith("0.0"), "and formatted as asked"

    def test_the_wrong_number_of_colours_is_refused(self):
        """A short list leaves classes sharing a colour; a long one leaves colours unused.

        Test scenario:
            Either way the legend stops matching the picture — which is precisely the failure this type
            exists to prevent, so it must not be constructible.
        """
        scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=3)
        with pytest.raises(ValueError, match="one colour per class"):
            LegendSpec.from_scale(scale, colors=["#eee", "#111"])

    def test_omitting_the_colours_says_where_they_come_from(self):
        """A graduated scale does not carry its colours, and the error says so.

        Test scenario:
            Unlike a categorical scale, the colours come from the renderer's colormap. A caller who expected
            the scale to supply them needs to be told which side of the seam they are on.
        """
        scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)
        with pytest.raises(ValueError, match="not from the scale"):
            LegendSpec.from_scale(scale)


class TestFromAContinuousScale:
    """A ramp, described by stops along it."""

    def test_the_ramp_is_described_by_its_stops(self):
        """A continuous scale becomes evenly spaced samples of its domain.

        Test scenario:
            The web tier already did this with five stops; sharing the number is what stops a sixth
            appearing when another tier grows a ramp legend.
        """
        legend = LegendSpec.from_scale(
            Scale.from_limits(0.0, 100.0), colors=["#0", "#1", "#2", "#3", "#4"]
        )
        assert [entry.value for entry in legend.entries] == [
            0.0,
            25.0,
            50.0,
            75.0,
            100.0,
        ]
        assert legend.kind == "continuous"

    def test_the_number_of_stops_can_be_chosen(self):
        """A tier with room for more rows can ask for them.

        Test scenario:
            Five is a default, not a constraint — a tall vertical legend can afford more.
        """
        legend = LegendSpec.from_scale(
            Scale.from_limits(0.0, 1.0), colors=["#a", "#b", "#c"], stops=3
        )
        assert len(legend.entries) == 3

    @pytest.mark.parametrize(
        "scale, colors",
        [
            (Scale.categorical(["a"], ["#f00"]), None),
            (Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2), ["#a", "#b"]),
        ],
        ids=["categorical", "graduated"],
    )
    def test_a_legend_with_no_ramp_ignores_the_stop_count(self, scale, colors):
        """Only a continuous legend reads `stops`, so only it may be refused over one.

        Test scenario:
            The guard ran before the categorical and graduated branches, so both refused a `stops` value
            they go on to ignore — `from_scale(Scale.categorical(...), stops=1)` raised "a continuous legend
            needs at least two stops" about a legend that is not continuous and has no ramp.
        """
        legend = LegendSpec.from_scale(scale, colors=colors, stops=1)
        assert legend.kind != "continuous", "the arm under test is one that reads no stops"
        assert len(legend.entries) >= 1, "and it still produced its rows"

    @pytest.mark.parametrize("stops", [0, 1])
    def test_a_ramp_needs_at_least_two_stops(self, stops):
        """One stop divides by zero; none gives an empty legend.

        Args:
            stops: The unusable stop count under test.

        Test scenario:
            Every other field on this type is validated in `__post_init__`; `stops` reached the spacing
            arithmetic directly, so `stops=1` raised ZeroDivisionError from inside a list comprehension.
        """
        with pytest.raises(ValueError, match="at least two stops"):
            LegendSpec.from_scale(
                Scale.from_limits(0.0, 1.0), colors=["#a"] * max(stops, 1), stops=stops
            )


class TestTheSpecItself:
    """The value, and what it refuses."""

    def test_an_unknown_kind_is_refused(self):
        """`kind` selects a drawing path in every tier.

        Test scenario:
            An unrecognised one is silently ignored at render time and the legend simply does not appear —
            a legend that is absent rather than wrong, which is harder to notice.
        """
        with pytest.raises(ValueError, match="kind must be one of"):
            LegendSpec(kind="rainbow")

    def test_an_unknown_orientation_is_refused(self):
        """Same reason as the kind.

        Test scenario:
            Guards the other field that picks a layout branch.
        """
        with pytest.raises(ValueError, match="orientation must be"):
            LegendSpec(orientation="diagonal")

    def test_it_is_a_value(self):
        """A legend built two ways compares equal to one built by hand.

        Test scenario:
            What lets a test assert two tiers produced the same legend rather than comparing row by row.
            The two sides are constructed differently on purpose: comparing one expression with itself would
            pass even if equality were identity, which is the opposite of what a value object promises.
        """
        built = LegendSpec.from_scale(Scale.categorical(["a"], ["#f00"]))
        assert built == LegendSpec((LegendEntry("a", "#f00", "a"),), "categorical"), (
            "from_scale and the constructor must produce equal legends"
        )

    def test_entries_are_stored_as_a_tuple(self):
        """A list passed in does not stay the caller's.

        Test scenario:
            The same freezing rule the rest of `base/spec` follows — a frozen value that quietly tracks a
            mutable argument is not frozen.
        """
        rows = [LegendEntry("a", "#f00")]
        legend = LegendSpec(rows, "categorical")
        rows.append(LegendEntry("b", "#0f0"))
        assert len(legend.entries) == 1, "the legend must not track the caller's list"

    def test_to_dict_carries_labels_values_and_colours(self):
        """The serialisable shape a tier stores.

        Test scenario:
            The web tier's `last_legend` is derived from this, so the three keys it reads have to be here.
        """
        scale = Scale.categorical(["a", "b"], ["#f00", "#0f0"])
        payload = LegendSpec.from_scale(scale, title="cover").to_dict()
        assert payload["kind"] == "categorical"
        assert payload["colors"] == ["#f00", "#0f0"]
        assert payload["values"] == ["a", "b"]
        assert payload["title"] == "cover"
