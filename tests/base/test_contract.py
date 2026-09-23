"""The contract as data, and the helper that keeps a renamed method working (U-3, #299).

`base/contract.py` is what a tier is held against, and `renamed_method` is how a spelling is retired without
breaking the caller who already wrote it. These cover both in the environment that owns them — no engine, no
facade, just the vocabulary and the shim.
"""

import re
import warnings

import pytest

from digitalearth.base.contract import (
    CORE,
    PENDING,
    ROADMAP_ORDERS,
    TIER2,
    Method,
    alias_table,
    core_method,
    orders_named_in,
    pending_for,
)
from digitalearth.base.deprecation import renamed_method, renamed_parameter
from tests.open_issues import KNOWN_OPEN_ISSUES, issues_named_in

#: A wave reference inside a reason. It is the form that rots: inserting one wave renumbers every wave after
#: it, so the string keeps its number and loses its meaning, while an order keeps both (#317).
WAVE_REFERENCE = re.compile(r"\bwave\s+\d+", re.IGNORECASE)


class TestReadingTheContract:
    """Looking a name up, and being told when there is none."""

    def test_a_core_name_is_found(self):
        """The declaration says what a name means before any tier is consulted."""
        assert core_method("save").returns == "path", core_method("save")

    def test_a_name_outside_the_core_is_refused_with_the_vocabulary(self):
        """A misspelling is answered with the names there are, not with `None`."""
        with pytest.raises(KeyError, match="is not a Core method"):
            core_method("add_raster")

    def test_the_last_name_is_reachable(self):
        """The lookup walks the whole table, not just its head."""
        assert core_method(CORE[-1].name).name == CORE[-1].name, CORE[-1]

    def test_a_tier_that_renamed_nothing_has_an_empty_table(self):
        """`alias_table` answers for every backend, including one nobody renamed."""
        assert alias_table("nobody") == {}, alias_table("nobody")

    def test_a_tier_with_nothing_pending_answers_the_same_way(self):
        """`pending_for` is a question about a tier, not about the table's keys."""
        assert pending_for("nobody") == {}, pending_for("nobody")

    def test_a_method_declares_its_keywords(self):
        """The canonical keywords are part of the promise, not documentation beside it."""
        assert "cmap" in core_method("field").keywords, core_method("field")

    def test_a_method_can_be_built_with_defaults(self):
        """A declaration is a value: the optional halves have defaults a reader can rely on."""
        built = Method("demo", "a demonstration method")
        assert (built.returns, built.keywords, built.builds_in) == (
            "self",
            frozenset(),
            None,
        ), built


class TestRenamingAMethod:
    """The alias keeps the old call working, and says what to write instead."""

    class _Map:
        """A stand-in facade with one renamed method and one renamed parameter."""

        def set_bounds(self, bounds, *, padding=0):
            """Frame on a region.

            Args:
                bounds: The region.
                padding: Pixels around it.

            Returns:
                A description of what was framed.
            """
            return f"{bounds} with {padding}"

        def save_animation(self, path, *, fps=None, framerate=None):
            """Write frames to a file, resolving the deprecated frame-rate spelling.

            Args:
                path: Where to write.
                fps: Frames per second.
                framerate: The deprecated spelling.

            Returns:
                A description of what was written.
            """
            rate = renamed_parameter(
                new="fps",
                value=fps,
                old="framerate",
                alias=framerate,
                caller="Map.save_animation()",
                default=3.0,
            )
            return f"{path} at {rate}"

        fit_bounds = renamed_method(new="set_bounds", old="fit_bounds", owner="Map")
        animate = renamed_method(new="save_animation", old="animate", owner="Map")

    def test_the_old_name_still_works(self):
        """A promise to the caller who already wrote it."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert self._Map().fit_bounds([0, 0, 1, 1]) == "[0, 0, 1, 1] with 0", (
                "forwarded"
            )

    def test_keywords_travel_through_the_alias(self):
        """A whole call keeps working, not only its positional half."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert self._Map().fit_bounds([0, 0, 1, 1], padding=8) == (
                "[0, 0, 1, 1] with 8"
            ), "keywords must travel"

    def test_the_warning_names_both_spellings(self):
        """The other half of the promise: the old name must not linger silently."""
        facade = self._Map()
        with pytest.warns(DeprecationWarning) as caught:
            facade.fit_bounds([0, 0, 1, 1])
        message = str(caught[0].message)
        assert "Map.fit_bounds()" in message, message
        assert "Map.set_bounds()" in message, message

    def test_the_warning_lands_on_the_caller_s_line(self):
        """A warning pointing at the helper is one nobody can act on."""
        facade = self._Map()
        with pytest.warns(DeprecationWarning) as caught:
            facade.fit_bounds([0, 0, 1, 1])
        assert caught[0].filename == __file__, caught[0].filename

    def test_a_parameter_rename_inside_the_new_method_also_points_at_the_caller(self):
        """The alias counts the frame it added, so both warnings name the caller's line.

        Test scenario:
            The case the one hand-written alias handled by hand: calling the old method *and* the old keyword
            raised two warnings, and without the frame count the second landed in `deprecation.py`.
        """
        facade = self._Map()
        with pytest.warns(DeprecationWarning) as caught:
            facade.animate("out.gif", framerate=9.0)
        attributed = {
            str(record.message).split("(")[0]: record.filename for record in caught
        }
        assert len(attributed) == 2, [str(record.message) for record in caught]
        for label, filename in attributed.items():
            assert filename == __file__, f"{label} landed in {filename}"

    def test_the_frame_count_is_put_back_afterwards(self):
        """A later call through the new name is not credited with the alias's frame."""
        facade = self._Map()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            facade.animate("out.gif", framerate=9.0)
        with pytest.warns(DeprecationWarning) as caught:
            facade.save_animation("out.gif", framerate=9.0)
        assert caught[0].filename == __file__, caught[0].filename

    def test_the_alias_carries_the_names_it_was_built_with(self):
        """A traceback and a `help()` should show the method a caller actually wrote."""
        held = self._Map.__dict__["fit_bounds"]
        assert (held.__name__, held.__qualname__) == ("fit_bounds", "Map.fit_bounds"), (
            held
        )

    def test_the_alias_documents_where_it_forwards(self):
        """`help(Map.fit_bounds)` must name its replacement."""
        assert "set_bounds" in (self._Map.__dict__["fit_bounds"].__doc__ or ""), "doc"


class TestTheRenamesNobodyHasAdopted:
    """Review M5 — `PLANNED_RENAMES` is read through its own accessor."""

    def test_a_tier_that_has_agreed_a_rename_names_it(self):
        """The static tier's raster builder is `imshow`, and is to become `field`."""
        from digitalearth.base.contract import planned_renames

        assert planned_renames("matplotlib")["imshow"] == "field", planned_renames(
            "matplotlib"
        )

    def test_a_tier_with_none_answers_empty(self):
        """`planned_renames` is a question about a tier, not about the table's keys."""
        from digitalearth.base.contract import planned_renames

        assert dict(planned_renames("nobody")) == {}, planned_renames("nobody")


def _issues_named_in_reasons():
    """Return every issue a `PENDING` reason names, as `[(backend, name, reason, number)]`.

    Returns:
        One row per reference, so a reason naming two issues is reported twice and both are checked.
    """
    return [
        (backend, name, reason, number)
        for backend, table in PENDING.items()
        for name, reason in table.items()
        for number in issues_named_in(reason)
    ]


def _orders_named_in_reasons():
    """Return every roadmap order a `PENDING` reason names, as `[(backend, name, order)]`.

    Returns:
        One row per reference, so a reason naming two orders is reported twice and both are checked.
    """
    return [
        (backend, name, order)
        for backend, table in PENDING.items()
        for name, reason in table.items()
        for order in orders_named_in(reason)
    ]


class TestAPendingReasonPointsAtLiveWork:
    """A reason is a forward promise, so the thing it points at has to still be ahead (#319).

    `PENDING` is the only place in the package where an issue number is shown to a *user*: everywhere else one
    is provenance in a comment, where "closed" reads correctly as "this is why the code looks like this". Here
    it reads as "follow this to find out when the method arrives", and a closed issue answers that with "it is
    already done" — which is how fifteen reasons came to point at #300 and #303 after both seams landed in
    Wave 6 without adopting the Core names (#319), and how three tiers' reasons pointed at a shipped wave
    before that (#317). Both were found by reading. This is what finds the next one.

    The rule is therefore: a reason may name a roadmap **order**, which needs no bookkeeping because orders
    keep their numbers when the plan moves, or an issue listed in
    :data:`~tests.open_issues.KNOWN_OPEN_ISSUES`. That allowlist moved out of this module when the keyword
    table next door was held to it too (review R-L2), and the check that nothing in it has gone stale moved
    with it, to `tests/test_contract_names.py` — the one place both tables that consume it are in scope.
    A **wave** number
    is refused outright, with no allowlist to escape through, because a wave is the part that renumbers — the
    reasoning `contract.py` already carries from #317, which the first version of this class did not enforce
    and which let five 3-D reasons go on naming a closed Wave 5. Anything else fails, naming the tier, the
    method and the reason so the fix is a decision and not a search.

    A reason that points at neither is fine and is not what this checks: where nothing schedules the work, the
    honest answer is to say so, which is why three 3-D rows read "unscheduled".
    """

    def test_no_reason_names_an_issue_outside_the_allowlist(self):
        """A number nobody has vouched for is the exact shape both instances of this defect had."""
        unvouched = [
            (backend, name, reason, number)
            for backend, name, reason, number in _issues_named_in_reasons()
            if number not in KNOWN_OPEN_ISSUES
        ]
        listed = "; ".join(
            f"{backend}.{name} -> {reason!r}" for backend, name, reason, _ in unvouched
        )
        assert unvouched == [], (
            f"a PENDING reason names an issue that is not in KNOWN_OPEN_ISSUES: {listed}. "
            "Name the roadmap order that builds it, or add the issue with its title if it is still open."
        )

    def test_no_reason_names_a_wave(self):
        """A wave number is a fact with a shelf life, and five of these outlived theirs.

        Test scenario:
            Wave 5 closed having built none of what the 3-D rows said it would; the plan had a wave inserted
            ahead of it and everything after renumbered. Unlike the issue check there is no allowlist here,
            because an open wave becomes a closed one on its own — the form is what is wrong, not the number.
        """
        dated = [
            (backend, name, reason)
            for backend, table in PENDING.items()
            for name, reason in table.items()
            if WAVE_REFERENCE.search(reason)
        ]
        listed = "; ".join(
            f"{backend}.{name} -> {reason!r}" for backend, name, reason in dated
        )
        assert dated == [], (
            f"a PENDING reason names a wave, which renumbers when the plan moves: {listed}. "
            "Name the roadmap order that builds it, the open issue that tracks it, or say it is unscheduled."
        )

    def test_no_declared_build_order_names_a_wave(self):
        """`Method.builds_in` is read out by `core_method(...)`, so the same rule governs it.

        Test scenario:
            This is #317 itself: `builds_in` said "Wave 5, order 23" and was interpolated into nine `PENDING`
            reasons. Both halves are checked, since a `TIER2` name may declare one too.
        """
        dated = [
            (method.name, method.builds_in)
            for method in CORE + TIER2
            if method.builds_in and WAVE_REFERENCE.search(method.builds_in)
        ]
        assert dated == [], (
            f"a declared build order names a wave, which renumbers when the plan moves: {dated}"
        )

    def test_no_reason_names_an_order_the_contract_does_not_declare(self):
        """An order reference was accepted unconditionally, so `order 99` read as a plan (review R-L3).

        Test scenario:
            The issue half of this rule has had an allowlist since #319; the order half had nothing — the
            checks above ask only whether the form is an order rather than a wave, and "order 99" is a
            perfectly well-formed order. Measured before :data:`ROADMAP_ORDERS` existed: a `PENDING` row
            rewritten to name order 99 passed all three of them. The roadmap itself is outside this
            repository, so what is checked against is the vendored list of the orders this contract points
            at — enough to refuse a number nobody wrote down, and enough for a reader to see what each one
            builds without going and finding the plan.
        """
        invented = sorted(
            f"{backend}.{name} -> {order!r}"
            for backend, name, order in _orders_named_in_reasons()
            if order not in ROADMAP_ORDERS
        )
        assert invented == [], (
            f"a PENDING reason names an order the contract does not declare: {invented}. Add it to "
            "ROADMAP_ORDERS with what it builds, or correct the citation."
        )

    def test_no_declared_build_order_is_one_the_contract_does_not_declare(self):
        """`Method.builds_in` is read out by `core_method(...)`, so the same rule governs it.

        Test scenario:
            The same pairing as the wave check above: a `builds_in` is shown to a user asking when a name
            arrives, so it has to point somewhere real. `TIER2` is read too, since a name that must agree
            wherever it appears may declare one.
        """
        invented = [
            (method.name, order)
            for method in CORE + TIER2
            if method.builds_in
            for order in orders_named_in(method.builds_in)
            if order not in ROADMAP_ORDERS
        ]
        assert invented == [], (
            f"a declared build order names an order the contract does not declare: {invented}"
        )

    def test_the_declared_orders_are_all_still_cited(self):
        """The other direction: a vendored order nothing points at is the next thing to go stale.

        Test scenario:
            :data:`ROADMAP_ORDERS` is a copy of four lines of a document that lives elsewhere, which is
            exactly the shape that rots. An order stops being cited when the work lands and its `PENDING`
            rows come off — and the entry describing it should come off in the same change, not survive as
            a description of something already built.
        """
        cited = {order for _, _, order in _orders_named_in_reasons()}
        cited |= {
            order
            for method in CORE + TIER2
            if method.builds_in
            for order in orders_named_in(method.builds_in)
        }
        orphaned = sorted(set(ROADMAP_ORDERS) - cited)
        assert orphaned == [], (
            f"ROADMAP_ORDERS describes {orphaned}, which no PENDING reason and no declared build order "
            "names any more; take each off now that nothing is waiting on it"
        )
