"""The Core contract, held against each facade (U-3, #299).

Four tiers grew four vocabularies: a raster field was `imshow`, `image`, `add_raster` and `terrain`; a colour
key was a builder on two tiers and a toggle on a third; `render` returned the engine's object on two and `None`
on one; and eight names meant two different things depending on which facade you held. `base/contract.py` is
the agreement that ends that, and these are what hold the tiers to it — including the honest half, which is
what a tier has *not* built and which order builds it.

The tiers that adopted the Core names (web, 3-D) must answer to every one they can draw; interactive and
static carry pending lists, each entry naming the roadmap order that empties it.

The **keywords** are held the same way, on every tier and over both tiers of the contract — which they were
not (#324). :class:`TestTheKeywordsAreThePromiseToo` was scoped to the two tiers that already passed it, read
`CORE` alone, and stepped over any member that was not `callable`; what a tier is still short of now lives in
:data:`KEYWORD_SHORTFALLS`, one row per gap, each naming the issue or roadmap order that settles it and each
guarded so it cannot outlive the gap.
"""

import inspect
import re
import warnings
from types import MappingProxyType
from typing import Mapping, Tuple

import pytest

from digitalearth.base.contract import (
    ALIASES,
    CORE,
    PENDING,
    ROADMAP_ORDERS,
    TIER2,
    alias_table,
    orders_named_in,
    pending_for,
    roadmap_order,
)
from tests.open_issues import KNOWN_OPEN_ISSUES, issues_named_in

#: The facades, by the backend name `quickmap` spells them with. Each is imported lazily, so a missing extra
#: skips that tier rather than failing the file.
FACADES = {
    "web": ("digitalearth.web", "WebMap"),
    "3d": ("digitalearth.three_d", "Scene3D"),
    "interactive": ("digitalearth.interactive", "InteractiveMap"),
    "matplotlib": ("digitalearth.static", "Map"),
}

#: The tiers whose renderer seam has landed, and which therefore answer to the Core now.
SEAMED = ("web", "3d")

#: Every tier there is, read off :data:`FACADES` rather than written down a second time.
#:
#: This is the scope of the keyword check below, and it is derived rather than listed **because the listed
#: version went stale silently** (#324): the check was scoped to :data:`SEAMED`, Wave 6 seamed static and
#: interactive, and nobody widened the tuple — so the two tiers with the most divergence were the two the
#: check skipped, and it passed. A scope read off the facades cannot drift that way: a tier is in it the
#: moment it has a facade, and leaving it out means deleting the facade.
EVERY_TIER: Tuple[str, ...] = tuple(sorted(FACADES))

#: What a tier's methods are still short of, as `{(backend, method): (missing keywords, who owns the gap)}`.
#:
#: Wave 7 closed the `name`/`visible` columns of this table on both 2-D tiers (#321, #327) and, with them,
#: `text`, `labels` and `graticule`. What is left is **tracked divergence**: every row names an open issue or
#: the roadmap order that settles it, and the one row nothing schedules says so in that word — the rule
#: :data:`~digitalearth.base.contract.PENDING` already documents for a reason a user reads.
#:
#: The keywords are listed **exactly**, not as a floor, because
#: :meth:`TestTheKeywordsAreThePromiseToo.test_a_listed_shortfall_is_still_exactly_that`
#: compares both ways: a tier that gains one of these, or loses another, fails until the row is corrected.
#: That is what stops the list outliving what it excuses, the same shape as `CANNOT_HIDE_AT_BUILD` and
#: `UNDRAWN_KINDS` elsewhere in the suite.
KEYWORD_SHORTFALLS: Mapping[Tuple[str, str], Tuple[Tuple[str, ...], str]] = (
    MappingProxyType(
        {
            # The five rows the Core renames surfaced. Each of these methods answered to the tier's own
            # spelling until order 27a, so `pending_for` excused it from this check entirely and the
            # divergence was invisible — adopting the Core name is what makes the keywords measurable.
            # Every keyword below was probed against the live builder rather than read off a signature: a
            # name the glyph takes through `**opts` is listed because the *signature* does not name it (the
            # rule #324 set), a name that raises is listed because the tier has no such keyword at all.
            ("matplotlib", "field"): (
                ("band", "cmap", "limits", "opacity"),
                "band and cmap reach the render through **kwargs unnamed; limits is spelled vmin/vmax, "
                "and opacity alpha — #332, order 27a",
            ),
            ("matplotlib", "points"): (
                ("cmap", "column", "k", "opacity", "scheme", "size"),
                "cmap, scheme, k and size reach ScatterGlyph through **opts unnamed; this tier's points "
                "have no colour column at all, and opacity is spelled alpha — #332, order 27a",
            ),
            ("matplotlib", "polygons"): (
                ("cmap", "column", "k", "opacity", "scheme"),
                "cmap, scheme and k reach PolygonGlyph through **opts unnamed; a classified fill is "
                "choropleth() here, so there is no column, and opacity is spelled alpha — #332, order 27a",
            ),
            ("interactive", "field"): (
                ("limits", "opacity"),
                "limits is spelled clim here, and opacity alpha — #332, order 27a",
            ),
            ("interactive", "lines"): (
                ("cmap", "column", "k", "opacity", "scheme", "width"),
                "cmap reaches HoloViews through **opts unnamed; a line layer carries no value here, so "
                "column, scheme and k classify nothing; opacity is spelled alpha, and a bare width= is "
                "HoloViews' own plot width rather than the line's — #332, order 27a",
            ),
            ("matplotlib", "colorbar"): (
                ("layer_id", "visible"),
                "keyed by position (layer=-1) and drawn when called rather than toggled — #261, order 24",
            ),
            ("matplotlib", "legend"): (
                ("layer_id", "title", "visible"),
                "takes the colours and labels themselves rather than a layer id — #261, order 24",
            ),
            ("matplotlib", "basemap"): (
                ("provider",),
                "the provider is spelled source= here — #268",
            ),
            ("matplotlib", "set_title"): (
                ("subtitle",),
                "one axes title, with no second line to carry a subtitle — #265",
            ),
            # `("matplotlib", "set_bounds")` was listed here against `("padding",)`, to keep order 27a's
            # rename from reading as the capability behind it: the name and the chainable return landed at
            # 27a, and `padding` with the `None` that fits the data did not. Both landed at the framing
            # order, on this tier and on the interactive one, so the row is off and `set_bounds` is no
            # longer short of anything the Core declares for it.
            ("interactive", "polygons"): (
                ("k", "scheme"),
                # This said the scheme/k half was "unscheduled", and #331 — open, and titled for exactly
                # this gap — schedules it. A reason may say `unscheduled` only where nothing does. The
                # `opacity` half came off when order 27a took the keyword (#332).
                "classification is choropleth() on this tier, so polygons() takes no scheme/k at all — #331",
            ),
            ("interactive", "colorbar"): (
                ("label", "layer_id", "visible"),
                "a show= toggle on the layer added last, not a builder keyed by id — #261, order 24",
            ),
            ("interactive", "legend"): (
                ("labels", "layer_id", "title", "visible"),
                "the same show= toggle as its colorbar — #261, order 24",
            ),
            ("interactive", "tiles"): (
                ("url",),
                "a name and a URL both arrive through one overloaded provider= — #268",
            ),
            ("web", "labels"): (
                ("crs",),
                "places labels in EPSG:4326 only, so a column in another CRS cannot be labelled — #265",
            ),
        }
    )
)

#: What a shortfall's reason must name: an open issue, a roadmap order, or — where neither exists — the word
#: that says so. Written as one pattern because the three are alternatives to each other, and because a reason
#: that names none of them is a shrug rather than a plan.
OWNER_PATTERN = re.compile(r"#\d+|order \d+|unscheduled")


def _every_user_facing_reason() -> Tuple[str, ...]:
    """Return every reason the suite shows a *user*, from both tables that carry one.

    `PENDING` answers "when does this tier get the method"; :data:`KEYWORD_SHORTFALLS` answers "why does this
    tier spell the keyword differently". They are the two consumers of the shared open-issue allowlist, so a
    check over that allowlist has to read both or it reports every row the other table owns as stale.

    Returns:
        The reason strings, `PENDING`'s first.
    """
    pending = tuple(reason for table in PENDING.values() for reason in table.values())
    shortfalls = tuple(owner for _, owner in KEYWORD_SHORTFALLS.values())
    return pending + shortfalls


def _facade(backend: str):
    """Return a tier's facade class, skipping the test when its extra is not installed.

    Args:
        backend: The tier to load.

    Returns:
        The facade class.
    """
    module, name = FACADES[backend]
    return getattr(pytest.importorskip(module), name)


class TestTheContractIsWellFormed:
    """The declaration itself, before any tier is consulted."""

    def test_every_core_name_is_declared_once(self):
        """One row per name, or a tier could satisfy two different promises under one spelling."""
        names = [method.name for method in CORE]
        assert len(names) == len(set(names)), f"repeated Core names: {names}"

    def test_every_tier_2_name_is_declared_once(self):
        """The same, for the names that must agree wherever they appear."""
        names = [method.name for method in TIER2]
        assert len(names) == len(set(names)), f"repeated Tier 2 names: {names}"

    def test_a_name_is_never_both_core_and_tier_2(self):
        """A name is either answered everywhere or agreed wherever it appears, not both."""
        overlap = {method.name for method in CORE} & {method.name for method in TIER2}
        assert overlap == set(), (
            f"names in both tiers of the contract: {sorted(overlap)}"
        )

    def test_every_alias_points_at_a_name_the_contract_declares(self):
        """An alias to nowhere would deprecate a spelling for a name nothing promises."""
        declared = {method.name for method in CORE} | {method.name for method in TIER2}
        # `record` and `terrain_tiles` are tier-native names a collision freed up, not contract names.
        native = {"record", "terrain_tiles", "grid_points"}
        for backend, table in ALIASES.items():
            unknown = sorted(set(table.values()) - declared - native)
            assert unknown == [], f"{backend} aliases point at {unknown}"

    def test_every_pending_entry_names_a_core_method(self):
        """A pending list is about the Core, so an entry outside it is a stale note."""
        declared = {method.name for method in CORE}
        for backend, table in PENDING.items():
            unknown = sorted(set(table) - declared)
            assert unknown == [], (
                f"{backend} lists {unknown} as pending, which is not Core"
            )

    def test_every_pending_entry_says_why_or_when(self):
        """The honest half: a name absent by design reads differently from one nobody has written."""
        for backend, table in PENDING.items():
            silent = [name for name, reason in table.items() if not reason.strip()]
            assert silent == [], f"{backend} gives no reason for {silent}"


class TestTheSeamedTiersAnswerToTheCore:
    """web and 3-D, whose seams have landed."""

    @pytest.mark.parametrize("backend", SEAMED)
    def test_every_core_name_is_present_or_pending(self, backend):
        """A Core name is either on the facade or declared pending, with a reason.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        pending = pending_for(backend)
        missing = [
            method.name
            for method in CORE
            if not hasattr(facade, method.name) and method.name not in pending
        ]
        assert missing == [], (
            f"{backend} answers to neither the name nor a reason: {missing}"
        )

    @pytest.mark.parametrize("backend", SEAMED)
    def test_nothing_pending_is_quietly_present(self, backend):
        """A name listed as pending that the tier does have is a stale list.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        built = [name for name in pending_for(backend) if hasattr(facade, name)]
        assert built == [], f"{backend} lists {built} as pending, but has them"

    @pytest.mark.parametrize("backend", sorted(FACADES))
    def test_every_old_spelling_still_works(self, backend):
        """An alias is a promise to the caller who already wrote the old name.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        broken = [old for old in alias_table(backend) if not hasattr(facade, old)]
        assert broken == [], f"{backend} dropped {broken} instead of deprecating them"

    @pytest.mark.parametrize("backend", sorted(FACADES))
    def test_every_alias_warns_and_names_its_replacement(self, backend):
        """The other half of the promise: the old name must not linger silently.

        Args:
            backend: The tier under test.

        Test scenario:
            The alias is read off the class rather than called, since calling one draws a map. What is checked
            is that it is a deprecation shim at all — a plain method assigned to two names would pass every
            other test here and warn nobody.
        """
        facade = _facade(backend)
        for old, new in alias_table(backend).items():
            held = inspect.getattr_static(facade, old)
            source = inspect.getsource(held) if inspect.isfunction(held) else ""
            assert "DeprecationWarning" in source, f"{backend}.{old} does not warn"
            assert new in source or new in (held.__doc__ or ""), (
                f"{backend}.{old} does not name {new} as its replacement"
            )


def _declared_methods():
    """Return every method the contract declares, both tiers of it.

    Returns:
        `CORE + TIER2`. The keyword check used to iterate `CORE` alone, so every keyword `TIER2` declares —
        `text(crs=)`, `labels(column=)`, `graticule(spacing=)`, `layer_control(layers=)` — was checked on no
        tier at all, which is precisely where a Tier 2 name is *allowed* to be absent and so most needs its
        arguments held to one spelling where it is present (#324).
    """
    return CORE + TIER2


def _keyword_shortfall(facade, backend: str, method) -> Tuple[str, ...]:
    """Return the keywords one tier's method does not take but the contract declares.

    Args:
        facade: The tier's facade class.
        backend: The tier, for its pending list.
        method: The declared :class:`~digitalearth.base.contract.Method`.

    Returns:
        The missing keywords, sorted, or `()` when there are none. `()` also for a name the tier does not
        have — Tier 2 is "agree wherever it appears", and a Core name it lacks is either pending, which the
        name checks cover, or already failing them — and for a name answered by a **property**, which is
        held to the stricter rule in
        :meth:`TestTheKeywordsAreThePromiseToo.test_a_name_answered_by_a_property_declares_no_keywords`
        instead. A property was silently dropped by a `not callable(...)` guard before, which is why the
        skip is now stated rather than implied: `layer_ids` is a property on all three 2-D tiers, and the
        day the contract declares a keyword against it nothing would have said so (#324).
    """
    held = inspect.getattr_static(facade, method.name, None)
    if held is None or method.name in pending_for(backend):
        return ()
    if isinstance(held, property):
        return ()
    bound = getattr(facade, method.name)
    if not callable(bound):
        return ()
    taken = set(inspect.signature(bound).parameters)
    return tuple(sorted(method.keywords - taken))


def _measured(backend: str) -> dict:
    """Return everything one tier is short of, keyed by method name.

    Args:
        backend: The tier to measure.

    Returns:
        `{method: missing}` for the methods that are short of something, empty for a tier that is short of
        nothing.
    """
    facade = _facade(backend)
    found = {}
    for method in _declared_methods():
        missing = _keyword_shortfall(facade, backend, method)
        if missing:
            found[method.name] = missing
    return found


def _listed(backend: str) -> dict:
    """Return the shortfalls :data:`KEYWORD_SHORTFALLS` records for one tier.

    Args:
        backend: The tier to look up.

    Returns:
        `{method: missing}`, in the same shape :func:`_measured` returns, so the two compare directly.
    """
    return {
        method: missing
        for (tier, method), (missing, _) in KEYWORD_SHORTFALLS.items()
        if tier == backend
    }


class TestTheKeywordsAreThePromiseToo:
    """Review L12 and #324 — the contract declares keywords, and almost nothing checked them.

    Three holes, all of them the same shape: the check was narrower than the thing it checked, and nothing
    said so. It ran on :data:`SEAMED`, the two tiers that passed it, while Wave 6 seamed the other two; it
    iterated `CORE`, so `TIER2`'s keywords were held against no tier; and it skipped a member that was not
    `callable`, which is every property. What replaces it runs on :data:`EVERY_TIER`, over both tiers of the
    contract, and answers for a property rather than stepping over it — and what it still cannot hold a tier
    to is written down in :data:`KEYWORD_SHORTFALLS` with the issue or order that owns it.
    """

    @pytest.mark.parametrize("backend", EVERY_TIER)
    def test_every_declared_keyword_is_taken_or_owned(self, backend):
        """A name that takes different arguments on each tier is not one name.

        Args:
            backend: The tier under test.

        Test scenario:
            The contract test checked that the *name* existed and never looked at `Method.keywords`.
            `WebMap.legend` was `(title, position, labels)` against a declaration of `{layer_id, title,
            labels, visible}`, and `field` had no `limits`. Widened here from the two tiers that already
            passed to all four, and from `CORE` to `CORE + TIER2` — which is what surfaced the rest of
            :data:`KEYWORD_SHORTFALLS`. A tier short of something no row owns fails here, so the only way
            past this check is to fix the tier or to name who will.
        """
        unowned = {
            method: missing
            for method, missing in _measured(backend).items()
            if _listed(backend).get(method) != missing
        }
        assert unowned == {}, (
            f"{backend} is short of declared keywords nothing owns: {unowned}; take the keyword, or add a "
            "row to KEYWORD_SHORTFALLS naming the issue or order that will"
        )

    @pytest.mark.parametrize("backend", EVERY_TIER)
    def test_a_listed_shortfall_is_still_exactly_that(self, backend):
        """The drift guard: a row that no longer describes the tier has to come off.

        Args:
            backend: The tier under test.

        Test scenario:
            The half that keeps the allowlist honest, and the reason the rows list their keywords exactly
            rather than as a floor. A tier that gains `opacity`, or that loses `cmap`, stops matching its
            row and fails here — so the list is corrected in the change that moved the tier, not years
            later. Without it a row would outlive the gap it excuses and go on excusing nothing, which is
            how the `SEAMED` tuple this class replaces went stale in the first place.
        """
        measured = _measured(backend)
        stale = {
            method: (missing, measured.get(method, ()))
            for method, missing in _listed(backend).items()
            if measured.get(method, ()) != missing
        }
        assert stale == {}, (
            f"{backend} no longer matches KEYWORD_SHORTFALLS — {{method: (listed, measured)}}: {stale}; "
            "correct the row, or take it off now that the tier answers"
        )

    def test_every_listed_shortfall_names_who_settles_it(self):
        """A row is a plan, not a shrug — the same rule `PENDING` is held to.

        Test scenario:
            The point of the table is to turn invisible divergence into *tracked* divergence, and a reason
            naming neither an issue nor a roadmap order tracks nothing. Where genuinely nothing schedules
            the work the reason has to say `unscheduled` in that word, which is a statement a reader can act
            on rather than a silence they have to interpret.
        """
        vague = sorted(
            f"{tier}.{method}"
            for (tier, method), (_, owner) in KEYWORD_SHORTFALLS.items()
            if OWNER_PATTERN.search(owner) is None
        )
        assert vague == [], f"no issue, order or 'unscheduled' against {vague}"

    def test_every_issue_a_shortfall_names_is_one_the_suite_vouches_for(self):
        """An issue reference is only a plan while the issue is open, and this table never asked.

        Test scenario:
            The rows above were held to a *shape* — an issue, an order, or the word "unscheduled" — and a
            row naming a closed issue satisfies that shape perfectly (review R-L2). It is the defect #319
            fixed for `PENDING`, left unfixed one table over: a reader follows `#261` expecting to find out
            when `colorbar` gains `layer_id`, and a closed issue answers "it already did". The allowlist is
            shared with `PENDING`'s check rather than copied, so the two tables cannot vouch for different
            sets of issues.
        """
        unvouched = sorted(
            f"{tier}.{method} -> #{number}"
            for (tier, method), (_, owner) in KEYWORD_SHORTFALLS.items()
            for number in issues_named_in(owner)
            if number not in KNOWN_OPEN_ISSUES
        )
        assert unvouched == [], (
            f"a KEYWORD_SHORTFALLS reason names an issue nothing vouches for: {unvouched}. Name the "
            "roadmap order that settles it, or add the issue to tests/open_issues.py with its title if it "
            "is still open."
        )

    def test_every_order_a_shortfall_names_is_one_the_contract_declares(self):
        """An order reference is a pointer into a plan, and nothing checked the other end (review R-L3).

        Test scenario:
            The same hole as the issue half, one table over: `OWNER_PATTERN` accepts any `order \\d+`, so a
            row naming order 99 would read as tracked divergence. `ROADMAP_ORDERS` is the contract's own
            vendored list of the orders it points at, and both tables are now held to it — a shortfall
            cannot promise a step of a plan that has no such step.
        """
        invented = sorted(
            f"{tier}.{method} -> {order!r}"
            for (tier, method), (_, owner) in KEYWORD_SHORTFALLS.items()
            for order in orders_named_in(owner)
            if order not in ROADMAP_ORDERS
        )
        assert invented == [], (
            f"a KEYWORD_SHORTFALLS reason names an order the contract does not declare: {invented}. Add it "
            "to ROADMAP_ORDERS with what it builds, or correct the citation."
        )

    def test_the_allowlist_holds_nothing_either_table_stopped_naming(self):
        """An allowlist outliving its reasons is the next stale pointer, one indirection further away.

        Test scenario:
            The reverse half of the check above, and of `PENDING`'s. It lives here because the allowlist now
            has two consumers and an entry is stale only when *neither* names it — asked of one table alone
            it would fail for every issue the other one owns. An issue that closes is taken off the list,
            which fails the forward checks until each row that named it is corrected; an issue whose row
            goes away is taken off here.
        """
        named = {
            number
            for reason in _every_user_facing_reason()
            for number in issues_named_in(reason)
        }
        unused = sorted(set(KNOWN_OPEN_ISSUES) - named)
        assert unused == [], (
            f"tests/open_issues.py vouches for {unused}, which no PENDING reason and no KEYWORD_SHORTFALLS "
            "row names any more"
        )

    @pytest.mark.parametrize("backend", EVERY_TIER)
    def test_a_name_answered_by_a_property_declares_no_keywords(self, backend):
        """A property takes no arguments, so a keyword declared against one can never be met.

        Args:
            backend: The tier under test.

        Test scenario:
            The skip that used to be silent, stated and enforced. `layer_ids` is a property on all three
            2-D tiers and the check stepped over it as "not callable"; it declares no keywords today, so
            nothing was being missed — but nothing would have said so if it had. Declaring a keyword
            against a name a tier answers with a property is a contradiction in the *contract* rather than
            a gap in the tier, so it is reported as one: either the contract stops declaring it, or the
            tier stops answering with a property.
        """
        facade = _facade(backend)
        impossible = {
            method.name: sorted(method.keywords)
            for method in _declared_methods()
            if isinstance(inspect.getattr_static(facade, method.name, None), property)
            and method.keywords
        }
        assert impossible == {}, (
            f"{backend} answers {sorted(impossible)} with a property, which can take no keywords, while "
            f"the contract declares {impossible}"
        )


class TestWhatARoadmapOrderPromises:
    """An order reference is a pointer into a plan the repository does not hold, so the number has to resolve.

    :data:`ROADMAP_ORDERS` is what a `PENDING` reason and a `KEYWORD_SHORTFALLS` row point at when they say
    "order 26" instead of naming an issue, and `roadmap_order` is the only way to follow one. The guards
    above ask whether a cited order is *in* the table; nothing asked what following one gives back, so a
    lookup that returned the key, or that let a bare `KeyError` out with no message, would still leave both
    guards green while the pointer answered nothing.
    """

    def test_an_order_the_contract_names_says_what_it_builds(self):
        """Following a pointer gives the line that tells a waiting caller what they are waiting for.

        Test scenario:
            `order 26` is what `interactive.set_bounds` is pending on. A lookup answering the key back, or
            the short handle, would read as a plan while saying nothing a reader can act on.
        """
        built = roadmap_order("order 26")
        assert built.startswith("auto-framing and camera round-trip"), built

    def test_an_order_nobody_wrote_down_is_refused_with_the_ones_there_are(self):
        """A number that is not a step of the plan fails loudly rather than reading as one (review R-L3).

        Test scenario:
            The defect this table was added for: every guard accepted any `order \\d+`, so a reason naming
            order 99 passed all three checks in `TestAPendingReasonPointsAtLiveWork`. The refusal has to
            name the bad order *and* list the real ones — a bare `KeyError('order 99')` would be the
            unhelpful answer this replaces, and would still satisfy a check that only asserted it raised.
        """
        with pytest.raises(KeyError) as refused:
            roadmap_order("order 99")
        message = refused.value.args[0]
        assert "'order 99' is not a roadmap order" in message, message
        assert "order 26" in message, message


class TestAPlannedRenameIsNotAnAlias:
    """Review M5 — `ALIASES` promises the old name works; a rename nobody has adopted promises nothing.

    **`PLANNED_RENAMES` is empty as of order 27a**, which adopted all six rows it held. The two checks over
    the live table therefore iterate nothing and cannot fail — and a check that cannot fail is a finding
    rather than a convenience, because the day a seventh rename is agreed it ships under a guard nobody has
    seen work. So each is paired with the same predicate applied to a **synthetic** one-row table, which is
    what keeps the rule itself exercised while the real table is empty. The synthetic rows are deliberately
    built from a live alias (`Map.imshow -> field`), so they describe a shape the package really has: a
    method present under its old name, and one present under both.
    """

    #: A planned rename that does describe the tier, for the predicate checks: `Map` has `imshow`, so a row
    #: naming it as the *old* spelling of something the tier does **not** have is well-formed.
    UNADOPTED = {"imshow": "a_name_no_tier_has"}

    #: A planned rename that no longer describes the tier: `Map` has both `imshow` and `field`, so the row
    #: has been adopted and belongs in `ALIASES`.
    ADOPTED = {"imshow": "field"}

    @staticmethod
    def _missing(facade, table) -> list:
        """Return the old spellings a table names that the facade does not have.

        Args:
            facade: The tier's facade class.
            table: A `{old: new}` mapping.

        Returns:
            The old names, so a row naming nothing is reported.
        """
        return [old for old in table if not hasattr(facade, old)]

    @staticmethod
    def _adopted(facade, table) -> list:
        """Return the rows whose *new* spelling the facade already answers to.

        Args:
            facade: The tier's facade class.
            table: A `{old: new}` mapping.

        Returns:
            The old names, so a row the tier has already adopted is reported.
        """
        return [old for old, new in table.items() if hasattr(facade, new)]

    @pytest.mark.parametrize("backend", ["interactive", "matplotlib"])
    def test_the_old_name_is_what_the_tier_still_calls_it(self, backend):
        """A planned rename names a method that exists today under its old spelling.

        Args:
            backend: The tier under test.

        Test scenario:
            Vacuous while `PLANNED_RENAMES` is empty, and kept for the row that comes next;
            :meth:`test_the_predicate_reports_a_row_naming_a_method_nobody_has` is what shows the predicate
            still works.
        """
        from digitalearth.base.contract import planned_renames

        facade = _facade(backend)
        missing = self._missing(facade, planned_renames(backend))
        assert missing == [], f"{backend} has no {missing}, so the rename names nothing"

    def test_the_predicate_reports_a_row_naming_a_method_nobody_has(self):
        """The check above, run against a table that really breaks the rule.

        Test scenario:
            The rule is "an unadopted rename's *old* name is what the tier calls the method today". A row
            whose old spelling is absent names nothing, and has to be reported — asked of a synthetic table
            because the real one is empty.
        """
        facade = _facade("matplotlib")
        assert self._missing(facade, {"a_name_no_tier_has": "field"}) == [
            "a_name_no_tier_has"
        ]

    @pytest.mark.parametrize("backend", ["interactive", "matplotlib"])
    def test_a_planned_rename_is_not_claimed_as_live(self, backend):
        """The two tables must not overlap: one says "still works", the other says "will be called".

        Args:
            backend: The tier under test.

        Test scenario:
            This is the guard that makes the split self-correcting, and it is the one that fired when order
            27a adopted the six ("matplotlib has adopted ['imshow', 'scatter', 'shapes']; move them to
            ALIASES so their liveness is checked"). Vacuous now that the table is empty, so
            :meth:`test_the_predicate_reports_a_row_the_tier_has_already_adopted` keeps the predicate itself
            under test.
        """
        from digitalearth.base.contract import planned_renames

        facade = _facade(backend)
        adopted = self._adopted(facade, planned_renames(backend))
        assert adopted == [], (
            f"{backend} has adopted {adopted}; move them to ALIASES so their liveness is checked"
        )

    def test_the_predicate_reports_a_row_the_tier_has_already_adopted(self):
        """And the same for the overlap rule, against a table that really breaks it.

        Test scenario:
            `Map` answers to both `imshow` and `field` — the first as a live alias for the second — so a
            `PLANNED_RENAMES` row saying `imshow` is *to become* `field` would be claiming the rename has
            not happened. That is what the predicate has to report.
        """
        facade = _facade("matplotlib")
        assert self._adopted(facade, self.ADOPTED) == ["imshow"]

    def test_the_unadopted_shape_is_reported_as_neither(self):
        """The negative control: a well-formed unadopted row must trip neither predicate.

        Test scenario:
            Two checks that reported every table would pass the two above and be useless. A row whose old
            spelling the tier has and whose new spelling it does not is exactly what `PLANNED_RENAMES` is
            for, and it has to come back clean from both.
        """
        facade = _facade("matplotlib")
        assert self._missing(facade, self.UNADOPTED) == []
        assert self._adopted(facade, self.UNADOPTED) == []


class TestTheUnseamedTiersDeclareTheirGap:
    """interactive and static, which answer to less of the Core than the two tiers that adopted its names.

    Their seams (#300, #303) landed with Wave 6 and left these lists standing, because rendering a figure from
    its description and calling the builders what the Core calls them were never the same job (#319).
    """

    @pytest.mark.parametrize("backend", ["interactive", "matplotlib"])
    def test_what_is_missing_is_listed_rather_than_discovered(self, backend):
        """Every Core name the facade lacks is in its pending list, naming what will build it.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        pending = pending_for(backend)
        missing = [
            method.name
            for method in CORE
            if not hasattr(facade, method.name) and method.name not in pending
        ]
        assert missing == [], f"{backend} is missing {missing} without saying so"

    @pytest.mark.parametrize("backend", ["interactive", "matplotlib"])
    def test_nothing_pending_is_quietly_present(self, backend):
        """A name the tier does have is not pending, whatever the list says.

        Args:
            backend: The tier under test.

        Test scenario:
            The seamed tiers have been held to this since the contract froze; these two were not, and both
            listed `layer_ids` as pending while Wave 6 was giving every tier the property (#319). Nothing in
            the suite noticed, because the check stopped at the tiers that had adopted the Core names.
        """
        facade = _facade(backend)
        built = [name for name in pending_for(backend) if hasattr(facade, name)]
        assert built == [], f"{backend} lists {built} as pending, but has them"

    @pytest.mark.parametrize("backend", ["interactive", "matplotlib"])
    def test_each_pending_name_says_which_seam_or_order_builds_it(self, backend):
        """A pending entry is a plan, not a shrug.

        Args:
            backend: The tier under test.
        """
        vague = [
            name
            for name, reason in pending_for(backend).items()
            if "#" not in reason and "order" not in reason
        ]
        assert vague == [], f"{backend} does not say what builds {vague}"


class TestOneNamePerMeaning:
    """The collisions the contract settles, checked where they were live."""

    def test_the_web_tier_draws_a_field_under_that_name(self):
        """`add_raster` is the alias now; `field` is the name every tier answers to."""
        web = _facade("web")
        assert callable(web.field), "web must draw a field"
        assert inspect.getattr_static(web, "add_raster") is not inspect.getattr_static(
            web, "field"
        ), "the alias must be a shim, not the same function under two names"

    def test_the_web_tier_switches_projection_by_name(self):
        """`globe(True)` was a projection switch spelled as a layer builder."""
        web = _facade("web")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert web().projection("globe").viewport.globe is True, "globe by name"

    def test_the_web_tier_writes_an_animation_under_the_shared_name(self):
        """`save_animation` is what static and interactive already called it."""
        web = _facade("web")
        assert callable(web.save_animation), "web must write an animation"

    def test_the_3d_tier_records_under_its_own_name(self):
        """A callback loop is not what `animate` means on the other tiers, so it is `record`."""
        scene = _facade("3d")
        assert callable(scene.record), "the 3-D callback loop is record()"

    def test_terrain_means_one_thing_per_tier(self):
        """A pyramids DEM in 3-D, a terrain-RGB tile URL on web — under different names."""
        web = _facade("web")
        scene = _facade("3d")
        assert callable(web.terrain_tiles), "the web tier reads terrain tiles"
        assert callable(scene.terrain), "the 3-D tier draws a DEM"


class TestTheAttachedIssues:
    """#260 and #263, whose web halves close with this contract."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip the class when MapLibre is not installed: two of these draw a map."""
        pytest.importorskip("maplibre")

    def test_web_text_takes_the_string_as_s(self):
        """#260: the third argument is `s`, as it is on the static and interactive tiers."""
        web = _facade("web")
        parameters = inspect.signature(web.text).parameters
        assert "s" in parameters, sorted(parameters)

    def test_web_text_takes_a_crs(self):
        """#260: a point in another CRS is placed, not drawn in the wrong ocean."""
        web = _facade("web")
        assert "crs" in inspect.signature(web.text).parameters, "text() must take crs="

    def test_web_text_still_accepts_the_old_keyword(self):
        """The rename is a promise to the caller who already wrote `string=`."""
        built = _facade("web")()
        with pytest.warns(DeprecationWarning, match="string"):
            drawn = built.text(4.9, 52.4, string="Amsterdam")
        assert drawn.layer_ids, drawn.layer_ids

    def test_web_graticule_takes_two_steps(self):
        """#263: meridians and parallels are spaced separately, as on the interactive tier."""
        web = _facade("web")
        parameters = inspect.signature(web.graticule).parameters
        assert {"lon_step", "lat_step"} <= set(parameters), sorted(parameters)

    def test_web_graticule_still_takes_one_spacing(self):
        """`spacing=` is the short way to ask for a square grid, and still works."""
        web = _facade("web")
        assert web().graticule(spacing=20.0).layer_ids, "a square grid still draws"

    def test_web_text_places_a_point_given_in_another_crs(self):
        """#260: a Web-Mercator coordinate lands where it belongs, not in the Atlantic.

        Test scenario:
            The tier places data in EPSG:4326, so a point given in another CRS has to be reprojected — which
            it goes through `Bounds.to_crs`, keeping the CRS work in pyramids.
        """
        web = _facade("web")
        placed = web().text(500000.0, 6800000.0, "Utrecht", crs=3857)
        assert placed.layer_ids, placed.layer_ids

    def test_web_text_needs_a_string(self):
        """A call with no string names what is missing rather than drawing an empty label."""
        drawn = _facade("web")()
        with pytest.raises(TypeError, match="needs the string to draw"):
            drawn.text(4.9, 52.4)

    def test_web_get_layer_returns_the_description(self):
        """The contract's read-back: a layer's own `LayerSpec`, by id."""
        web = _facade("web")
        drawn = web().text(4.9, 52.4, "Amsterdam", name="label")
        assert drawn.get_layer("label").kind == "text", drawn.get_layer("label")

    def test_web_get_layer_refuses_an_unknown_id(self):
        """The message names the layers that are there."""
        drawn = _facade("web")()
        with pytest.raises(KeyError, match="no layer 'nope' on this map"):
            drawn.get_layer("nope")

    def test_web_colorbar_draws_the_colour_key(self):
        """The contract's name for a colour key, which this tier draws through `legend`."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        web = _facade("web")
        squares = gpd.GeoDataFrame(
            {"pop": [1, 9]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs=4326,
        )
        drawn = web().choropleth(squares, column="pop", name="area")
        assert sorted(drawn.colorbar("area", label="People")._panels) == ["legend"], (
            drawn._panels
        )

    def test_web_colorbar_without_an_id_keys_the_last_classified_layer(self):
        """A caller who drew one thing should not have to name it."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        web = _facade("web")
        squares = gpd.GeoDataFrame(
            {"pop": [1, 9]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs=4326,
        )
        drawn = web().choropleth(squares, column="pop", name="area").colorbar()
        assert sorted(drawn._panels) == ["legend"], drawn._panels

    def test_web_colorbar_refuses_an_unknown_layer(self):
        """A key for a layer nobody drew is a caller's mistake, not an empty panel."""
        drawn = _facade("web")()
        with pytest.raises(KeyError, match="no layer 'nope' on this map"):
            drawn.colorbar("nope")

    def test_web_colorbar_draws_nothing_when_it_is_not_wanted(self):
        """`visible=False` is a caller passing a flag through, not an error."""
        web = _facade("web")
        assert web().colorbar(visible=False)._panels == {}, "no key was asked for"


class TestTheDispatcherPassesUnderDeprecationErrors:
    """Nothing the package calls itself may go through a deprecated spelling."""

    @pytest.mark.parametrize("backend", SEAMED)
    def test_quickmap_uses_the_canonical_names(self, backend, tmp_path):
        """`quickmap` on a seamed backend raises no deprecation warning of its own.

        Args:
            backend: The tier under test.
            tmp_path: Unused; keeps the signature uniform with the other cases.
        """
        # The engine, not the tier: a tier imports without its engine (#290), so a missing extra would only
        # surface when the drawing started.
        pytest.importorskip({"web": "maplibre", "3d": "pyvista"}[backend])
        from pyramids.dataset import Dataset

        from digitalearth import quickmap

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            drawn = quickmap(
                Dataset.read_file("examples/data/acc4000.tif"), backend=backend
            )
        # Only this package's own warnings: PyVista and NumPy warn about things neither the caller nor this
        # package can act on, and failing on those would make the gate about the installed versions.
        ours = [
            str(record.message)
            for record in caught
            if issubclass(record.category, DeprecationWarning)
            and "digitalearth" in record.filename
        ]
        assert ours == [], f"quickmap reached a deprecated spelling: {ours}"
        assert drawn is not None, "quickmap must draw"
        if hasattr(drawn, "close"):
            drawn.close()
