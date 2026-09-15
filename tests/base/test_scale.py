"""`Scale` — one rule for turning a value into a colour (DE-12, #271).

Five places wrote out "a constant band has no range, widen it by one", and three tiers each called cleopatra's
classifier with their own error handling. These cover the type that absorbed both.
"""

import pytest

from digitalearth.base.spec import DEFAULT_CLASS_COUNT, Scale


class TestTheDomain:
    """Deriving limits, including the two degenerate cases every tier had to handle."""

    def test_limits_come_from_the_data(self):
        """An ordinary range is measured, not invented.

        Test scenario:
            The base case the other tests are exceptions to.
        """
        assert Scale.from_values([1.0, 5.0, 9.0]).as_limits() == (1.0, 9.0), (
            "the domain must span the data"
        )

    def test_a_constant_domain_is_widened_rather_than_divided_by(self):
        """A band whose values are all equal gets a usable range.

        Test scenario:
            The rule that was written out five times. Normalising against a zero-width range divides by zero;
            every copy chose ``+1``, so that is what is kept here.
        """
        assert Scale.from_values([7.0, 7.0, 7.0]).as_limits() == (7.0, 8.0), (
            "a constant domain must widen to (v, v + 1)"
        )

    def test_nothing_finite_falls_back_to_the_unit_range(self):
        """All-nodata data still yields limits a colormap can take.

        Test scenario:
            A stack whose every frame is off the view has no range to report, and a builder still needs two
            numbers. Raising here would turn an empty map into a crash.
        """
        assert Scale.from_values([float("nan"), float("inf")]).as_limits() == (
            0.0,
            1.0,
        ), "an unmeasurable domain must fall back to (0, 1)"

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [({"vmin": 0.0}, (0.0, 3.0)), ({"vmax": 10.0}, (1.0, 10.0))],
    )
    def test_an_explicit_limit_overrides_the_measured_one(self, kwargs, expected):
        """A caller's `vmin`/`vmax` wins over the data.

        Args:
            kwargs: The explicit limit under test.
            expected: The resulting domain.

        Test scenario:
            Every migrated call site had this override, so losing it would silently ignore a caller's `clim=`.
        """
        assert Scale.from_values([1.0, 2.0, 3.0], **kwargs).as_limits() == expected, (
            f"an explicit {list(kwargs)[0]} must win"
        )

    def test_an_explicit_pair_that_is_degenerate_is_widened_too(self):
        """A caller who passes `vmin == vmax` gets the same widening as measured data.

        Test scenario:
            The web tier's raster path applied the guard *after* the override, so an explicit constant pair
            was widened as well. Applying it only to measured data would change that behaviour silently.
        """
        assert Scale.from_values([1.0, 2.0], vmin=5.0, vmax=5.0).as_limits() == (
            5.0,
            6.0,
        ), "an explicit constant pair must widen too"

    def test_a_degenerate_domain_cannot_be_constructed_directly(self):
        """The invariant holds even when the dataclass is built by hand.

        Test scenario:
            `from_values` widens, but a caller reaching the constructor must not be able to produce a scale
            that divides by zero downstream.
        """
        with pytest.raises(ValueError, match="vmax > vmin"):
            Scale(5.0, 5.0)

    def test_from_limits_applies_the_same_rule(self):
        """Limits derived elsewhere — a frozen stack range — go through the same widening.

        Test scenario:
            `base/clim.py` can return a constant range for a stack of identical frames; it must not reach a
            normaliser unwidened.
        """
        assert Scale.from_limits(4.0, 4.0).as_limits() == (4.0, 5.0), (
            "from_limits must widen a degenerate pair"
        )

    def test_a_non_finite_domain_cannot_be_constructed(self):
        """NaN or infinity for a limit is refused at construction.

        Test scenario:
            `from_values` filters non-finite data out before measuring, so this is the hand-built path — and
            a NaN limit does not raise downstream, it silently normalises every value to NaN and draws a
            blank layer.
        """
        with pytest.raises(ValueError, match="finite domain"):
            Scale(float("nan"), 1.0)

    def test_from_limits_refuses_a_non_finite_limit(self):
        """The same check on the limits-in builder.

        Test scenario:
            `from_limits` takes a caller's `clim=` directly, so it is the likeliest way an infinity gets in —
            a stack whose measured range came back unbounded, for instance.
        """
        with pytest.raises(ValueError, match="finite limits"):
            Scale.from_limits(0.0, float("inf"))

    def test_a_sequence_field_is_stored_as_a_tuple(self):
        """A caller's list does not become the scale's storage.

        Test scenario:
            `scheme` is a documented sequence-of-edges input on three tiers, so a real call path hands a
            mutable list to a type that calls itself frozen. Stored as given, the caller could edit the
            scale's classification afterwards and the scale would not hash.
        """
        edges = [0.0, 1.0, 2.0]
        scale = Scale(0.0, 1.0, scheme=edges, breaks=edges)
        edges.append(99.0)
        assert scale.breaks == (0.0, 1.0, 2.0), (
            "the scale must not track the caller's list"
        )
        assert scale.scheme == (0.0, 1.0, 2.0), "including the scheme spelling"

    def test_a_scale_can_key_a_cache(self):
        """`Scale` hashes, so a resolved scale can go in a set or a dict key.

        Test scenario:
            Freezing a scale once and reusing it across frames is the type's whole purpose; being unhashable
            would stop the cache that makes that worthwhile.
        """
        scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)
        assert (
            len({scale, Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)})
            == 1
        ), "two equal scales must collapse to one entry"

    def test_two_explicit_limits_the_wrong_way_round_are_refused(self):
        """A caller who swapped `vmin` and `vmax` is told, not silently corrected.

        Test scenario:
            The widening rule exists for a *measured* constant domain. Applied to two limits the caller wrote
            down *in the wrong order*, it dropped their `vmax` and drew against `(vmin, vmin + 1)` — so
            `vmin=10, vmax=5` became `(10, 11)`, hiding an obvious typo behind a plausible-looking range.
            An explicit *equal* pair is different and still widens: see the neighbouring test.
        """
        with pytest.raises(ValueError, match="reversed pair"):
            Scale.from_values([1.0, 2.0, 3.0], vmin=10.0, vmax=5.0)

    def test_one_explicit_limit_still_widens_against_the_data(self):
        """Only the both-explicit case is a typo; one limit plus a measurement is not.

        Test scenario:
            Guards the boundary of the rule above — a caller who pins only `vmin` above the data still gets a
            usable range rather than an error.
        """
        assert Scale.from_values([1.0, 2.0, 3.0], vmin=10.0).as_limits() == (
            10.0,
            11.0,
        ), "a single explicit limit must still widen rather than raise"


class TestClassification:
    """The classifier is reached once, from here, rather than three times from three tiers."""

    def test_a_scheme_produces_one_more_edge_than_classes(self):
        """`k` classes give `k + 1` edges, which is what a legend needs to label ranges.

        Test scenario:
            The off-by-one every tier had to get right independently when turning edges into swatches.
        """
        scale = Scale.from_values(list(range(10)), scheme="equal_interval", k=3)
        assert len(scale.breaks) == 4, (
            f"3 classes need 4 edges, got {len(scale.breaks)}"
        )
        assert len(scale.class_ranges()) == 3, "and 3 labelled ranges"

    def test_an_unclassified_scale_has_no_breaks(self):
        """A continuous ramp is the default and cuts nothing.

        Test scenario:
            Classification is opt-in on every tier; a scale that binned by default would silently change a
            continuous map into a stepped one.
        """
        scale = Scale.from_values([1.0, 2.0, 3.0])
        assert not scale.is_classified, "no scheme means no classes"
        assert scale.breaks == (), "and no edges"

    def test_a_bad_scheme_names_the_scheme_and_the_count(self):
        """The error says what the caller wrote, not just what the classifier objected to.

        Test scenario:
            cleopatra's own message names neither the scheme nor `k`, which is why all three tiers wrapped it.
            Wrapping once here is the consolidation.
        """
        with pytest.raises(ValueError, match=r"scheme='nonsense'.*k=5"):
            Scale.from_values([1.0, 2.0, 3.0], scheme="nonsense")

    def test_the_default_class_count_is_shared(self):
        """`k` defaults to the one shared constant rather than a per-tier literal.

        Test scenario:
            Five was written out separately in each tier; a sixth value appearing is the drift this prevents.
        """
        assert DEFAULT_CLASS_COUNT == 5, "the cartographic default is five classes"
        scale = Scale.from_values(list(range(20)), scheme="equal_interval")
        assert len(scale.class_ranges()) == DEFAULT_CLASS_COUNT, (
            "an unspecified k must use the shared default"
        )

    def test_a_value_is_placed_in_a_class(self):
        """`class_of` maps a value to the class a renderer would colour it as.

        Test scenario:
            The three tiers each digitised edges their own way; this is the shared answer.
        """
        scale = Scale.from_values(list(range(10)), scheme="equal_interval", k=2)
        assert scale.class_of(0.0) == 0, "the lowest value is in the first class"
        assert scale.class_of(9.0) == 1, "the highest is in the last"

    def test_a_value_beyond_the_edges_is_clamped_not_dropped(self):
        """Out-of-domain values land in the nearest class.

        Test scenario:
            A renderer clamps a value outside its domain rather than leaving a hole, so the classifier must
            agree or the legend will not match the picture.
        """
        scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)
        assert scale.class_of(-50.0) == 0, "below the domain clamps to the first class"
        assert scale.class_of(50.0) == 1, "above it clamps to the last"

    def test_an_unclassified_scale_places_nothing(self):
        """`class_of` answers `None` when there are no classes.

        Test scenario:
            Returning 0 would let a continuous scale masquerade as a one-class scheme.
        """
        assert Scale.from_values([1.0, 2.0]).class_of(1.5) is None, (
            "a continuous scale has no class to report"
        )

    def test_edges_that_bound_no_class_are_refused(self):
        """A single class edge describes nothing, so it cannot be held.

        Test scenario:
            `k` classes give `k + 1` edges, so one edge means zero classes. Left unchecked, `class_of` walks
            an empty list of upper bounds and returns `-1` — a negative class index that reads as valid and
            colours the value from the wrong end of the ramp. An empty tuple stays legal: that is how a
            continuous scale says it cuts no classes at all.
        """
        with pytest.raises(ValueError, match="at least two class edges"):
            Scale(0.0, 1.0, scheme="equal_interval", breaks=(5.0,))

    def test_breaks_of_cuts_edges_without_deriving_a_domain(self):
        """The classify-only path returns the same edges `from_values` would, and measures nothing.

        Test scenario:
            The three classification sites read `.breaks` and nothing else, so deriving a domain for them —
            a full pass plus a compacted copy of every finite value — produced an answer thrown straight
            away. This must stay identical to the long way round, or the tiers would cut different classes
            depending on which call they happened to use.
        """
        values = list(range(10))
        assert (
            Scale.breaks_of(values, "equal_interval", 3)
            == Scale.from_values(values, scheme="equal_interval", k=3).breaks
        ), "the short path must cut exactly the edges the long one does"

    def test_breaks_of_reports_a_bad_scheme_the_same_way(self):
        """The shorter path keeps the error that names the scheme and `k`.

        Test scenario:
            It is the same classifier and the same wrapper, so a caller cannot get a worse message by taking
            the path that skips the measurement.
        """
        with pytest.raises(ValueError, match=r"scheme='nonsense'.*k=5"):
            Scale.breaks_of([1.0, 2.0, 3.0], "nonsense")

    def test_from_finite_matches_from_values_on_already_filtered_data(self):
        """The pre-filtered builder settles on exactly the domain the filtering one would.

        Test scenario:
            Four call sites had already filtered to finite values and handed the result straight back to
            `from_values`, which filtered it again — a second full pass and, on a global canvas, a second
            ~75 MB copy. The two builders must agree exactly, or which one a tier happened to call would
            change the colours it drew.
        """
        finite_values = [1.0, 5.0, 9.0]
        assert (
            Scale.from_finite(finite_values).as_limits()
            == Scale.from_values(finite_values).as_limits()
        ), "the two builders must settle on the same domain"

    def test_from_finite_keeps_the_empty_and_widening_rules(self):
        """Skipping the filter does not skip the degenerate-case handling.

        Test scenario:
            An empty selection must still yield limits a colormap can take, and a constant one must still
            widen — those rules are the reason the type exists, not part of the filtering it skips.
        """
        assert Scale.from_finite([]).as_limits() == (0.0, 1.0), (
            "empty falls back to the unit range"
        )
        assert Scale.from_finite([7.0, 7.0]).as_limits() == (7.0, 8.0), (
            "a constant domain widens"
        )


class TestCategorical:
    """Unordered categories, where a numeric domain is not the question."""

    def test_each_category_keeps_its_colour(self):
        """A category maps to the colour it was assigned.

        Test scenario:
            The legend must show the colours that were drawn; that only holds if the mapping is one object.
        """
        scale = Scale.categorical(["a", "b", "c"], ["#f00", "#0f0", "#00f"])
        assert scale.color_for("b") == "#0f0", (
            "the second category keeps the second colour"
        )
        assert scale.is_categorical, "and the scale knows it is categorical"

    def test_an_unseen_category_gets_the_missing_colour(self):
        """A value the scale never saw is drawn neutral, not as some other class.

        Test scenario:
            Colouring an unknown category as the first class is the silent-wrongness Wave 0 removed from the
            3-D and web tiers; the shared type must not reintroduce it.
        """
        scale = Scale.categorical(["a"], ["#f00"], missing="#ccc")
        assert scale.color_for("zzz") == "#ccc", "an unseen category reads as missing"

    def test_mismatched_counts_are_refused(self):
        """One colour per category, or the picture does not show the mapping it claims.

        Test scenario:
            A short colour list leaves categories sharing a colour; a long one leaves colours unused. Either
            way the legend lies.
        """
        with pytest.raises(ValueError, match="one colour per category"):
            Scale.categorical(["a", "b"], ["#f00"])

    def test_no_categories_is_refused(self):
        """An empty categorical scale has nothing to colour.

        Test scenario:
            It would otherwise produce an empty legend and colour every feature as missing.
        """
        with pytest.raises(ValueError, match="at least one category"):
            Scale.categorical([], [])

    def test_a_boolean_category_is_not_the_same_label_as_one(self):
        """`True` and `1` are different categories, though they compare equal.

        Test scenario:
            `in`/`.index` compare with `==`, and `True == 1` in Python — so a layer with both as categories
            coloured every `1` with `True`'s colour. Categories are labels arriving straight from a data
            column, which is exactly where a bool column and an int column meet.
        """
        scale = Scale.categorical([True, 1], ["#f00", "#0f0"])
        assert scale.color_for(True) == "#f00", "True keeps the colour assigned to True"
        assert scale.color_for(1) == "#0f0", "and 1 keeps the one assigned to 1"

    def test_a_nan_category_can_still_be_looked_up(self):
        """A category that was stored can be found again, even if it is NaN.

        Test scenario:
            NaN compares unequal to itself, so an `==`-based lookup could never reach it and every NaN read
            as missing — including one the scale was explicitly given a colour for.
        """
        nan = float("nan")
        scale = Scale.categorical([nan, "land"], ["#aaa", "#8b4513"], missing="#ccc")
        assert scale.color_for(nan) == "#aaa", (
            "the stored NaN category must be reachable"
        )


class TestItStaysAValue:
    """Frozen, so a scale handed to a renderer cannot move underneath it."""

    def test_it_cannot_be_mutated(self):
        """Assigning to a field raises.

        Test scenario:
            A frozen scale is the whole mechanism behind a time sequence keeping one colour range; if it could
            be edited in place, "frozen" would be a naming convention rather than a guarantee.
        """
        scale = Scale.from_values([1.0, 2.0])
        with pytest.raises(AttributeError):
            scale.vmin = 99.0

    def test_freeze_returns_an_equal_scale(self):
        """`freeze()` is explicit at the call site and changes nothing.

        Test scenario:
            It reads as intent — "this range is fixed now" — on a type that is already immutable.
        """
        scale = Scale.from_limits(0.0, 1.0)
        assert scale.freeze() == scale, "freezing an immutable scale is a no-op"

    def test_two_equal_scales_compare_equal(self):
        """Equality is by value.

        Test scenario:
            What lets a test assert two tiers resolved the same scale, rather than comparing field by field.
        """
        assert Scale.from_limits(0.0, 1.0) == Scale.from_limits(0.0, 1.0), (
            "equal domains must compare equal"
        )
