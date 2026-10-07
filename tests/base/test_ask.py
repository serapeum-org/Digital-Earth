"""The record that replaces the tables of measured defaults (#334).

`Symbology.encodings` carries what a caller *asked* a builder for, and a keyword defaulted in a signature
destroys that before anything can record it. Both 2-D tiers used to reconstruct the ask by subtracting a
table of measured defaults, which could not tell a caller who asked for exactly their tier's default from one
who asked for nothing, and tied no row to the default it mirrored.

These check the replacement at its own level: the sentinel is distinguishable from every value a caller may
pass, resolving through :class:`~digitalearth.base.ask.Ask` narrows back to the keyword's own type, the record
is what the builder was *given* rather than what it resolved to, and what the record is written as is
something a figure can actually carry. The tier-level questions — that a real `points(size=…)` publishes and a
real `points()` does not — are asked of the builders in `tests/web` and `tests/interactive`.
"""

from typing import get_args

from digitalearth.base.ask import (
    ASKED_PROP,
    UNSET,
    Ask,
    Maybe,
    Unset,
    asked_style,
)
from digitalearth.base.spec import Symbology
from digitalearth.base.spec._serial import travels_in_a_figure

#: The web tier's own marker size, which is the value the subtractive tables could not attribute.
TIER_DEFAULT = 5.0

#: A size no tier on this package defaults to, so an ask for it is unambiguous either way.
ELSEWHERE = 12.0


class TestTheSentinelIsNotAValue:
    """ "Not passed" has to be distinguishable from everything a caller may legitimately pass."""

    def test_the_sentinel_reads_as_itself(self):
        """The spelling a signature, `help()` and a traceback show.

        Test scenario:
            A bare `object()` prints as `<object object at 0x…>`, which tells a reader of a signature
            nothing at all about why the default is not the number the docstring names.
        """
        shown = repr(UNSET)
        assert shown == "<unset>", (
            f"the sentinel prints as {shown}, which a reader of a signature cannot make sense of"
        )

    def test_the_sentinel_is_the_one_instance_of_its_own_type(self):
        """`isinstance` is the check every builder makes, so the singleton has to answer it.

        Test scenario:
            Identity against a module-level singleton is what the sentinel was before it had a type;
            annotating absence means the *type* is what a builder and a checker both narrow on, and a
            singleton that were not an instance of it would narrow one way in mypy and the other at runtime.
        """
        assert isinstance(UNSET, Unset), (
            f"{UNSET!r} is not an instance of Unset, so every isinstance() guard in a builder is dead"
        )

    def test_a_falsey_value_a_caller_may_pass_is_not_read_as_absence(self):
        """`0.0`, `""` and `None` are asks, which is why absence cannot be spelled by a value.

        Test scenario:
            `opacity=0.0` is a caller asking for an invisible layer and `color=None` is a caller declining a
            colour; a sentinel that were falsey, or that were `None`, would read both as "not passed" and
            quietly draw the tier's default instead.
        """
        ask = Ask()
        resolved = [
            ask("circle-opacity", 0.0, 0.9),
            ask("circle-color", None, "#3388ff"),
        ]
        assert resolved == [0.0, None], (
            f"a caller's own falsey values resolved to {resolved} rather than to themselves"
        )

    def test_the_alias_is_the_union_of_the_keyword_and_absence(self):
        """`Maybe[float]` is what two dozen signatures are annotated with, so it has to be that union.

        Test scenario:
            The alias is the whole typing approach: one name, rather than a hand-widened annotation per
            keyword. An alias that had lost the sentinel from its union would type-check every builder
            while its default was a value mypy says cannot be there.
        """
        members = get_args(Maybe[float])
        assert members == (float, Unset), (
            f"Maybe[float] is {members}; it has to be the keyword's own type together with absence"
        )

    def test_the_alias_widens_the_keywords_own_type_rather_than_replacing_it(self):
        """The other half: the union must still admit the value the builder goes on to use.

        Test scenario:
            Written the other way round from the check above — this one names the union it expects instead
            of reading the alias — so an alias that had collapsed to `Unset` alone, or to `Any`, fails here
            rather than passing both.
        """
        assert Maybe[str] == (str | Unset), (
            f"Maybe[str] is {Maybe[str]}, which is not the str-or-absent a keyword needs"
        )


class TestTheRecordIsWhatTheCallerGaveNotWhatTheBuilderResolved:
    """The whole of what #334 changes: the ask is recorded, not reconstructed."""

    def test_a_keyword_the_caller_passed_is_recorded(self):
        """The ask reaches the description.

        Test scenario:
            The value alone was never enough — it is the resolved style either way — so the record is the
            only thing that says a keyword was named.
        """
        ask = Ask()
        resolved = ask("circle-radius", ELSEWHERE, TIER_DEFAULT)
        assert (resolved, ask.named) == (ELSEWHERE, ["circle-radius"]), (
            f"an explicit ask resolved to {resolved!r} and recorded {ask.named}"
        )

    def test_a_keyword_the_caller_left_alone_resolves_to_the_tiers_default_and_is_not_recorded(
        self,
    ):
        """The other direction, which is what stops an unstyled layer publishing anything.

        Test scenario:
            The layer still has to be *drawn* at the tier's default, so resolution and attribution are two
            answers to one call: the builder gets the number, the description gets nothing.
        """
        ask = Ask()
        resolved = ask("circle-radius", UNSET, TIER_DEFAULT)
        assert (resolved, ask.named) == (TIER_DEFAULT, []), (
            f"an unasked keyword resolved to {resolved!r} and recorded {ask.named}"
        )

    def test_asking_for_exactly_the_tiers_own_default_is_still_an_ask(self):
        """The case no subtraction of measured defaults could ever answer (#334).

        Test scenario:
            This is the cost the tables were accepted with: `points(size=5.0)` on the web tier was
            indistinguishable from `points()`, so the channel published nothing and a figure carried
            elsewhere was drawn at *that* tier's default. The record makes the two calls different calls.
        """
        ask = Ask()
        resolved = ask("circle-radius", TIER_DEFAULT, TIER_DEFAULT)
        assert (resolved, ask.named) == (TIER_DEFAULT, ["circle-radius"]), (
            f"a caller asking for the tier's own default resolved to {resolved!r} and recorded "
            f"{ask.named}; that is the subtraction's loss, not the record's"
        )

    def test_a_key_the_builder_cannot_default_is_recorded_without_one(self):
        """A required argument, and a keyword whose absence is already spelled `None`.

        Test scenario:
            `extrusion(height=)` has no default to resolve against and `contours(color=None)` already
            spells absence its own way, so neither goes through the resolving call — and both are the
            caller's whenever they carry anything.
        """
        ask = Ask()
        ask.always("fill-extrusion-height")
        assert ask.named == ["fill-extrusion-height"], (
            f"a key with no default recorded {ask.named}; any value under it is the caller's"
        )

    def test_one_key_resolved_twice_is_one_ask(self):
        """A builder that writes the same channel twice has still been asked once.

        Test scenario:
            A filled contour resolves its opacity for the outline and again for the fill. The record is a
            set of names, so the duplicate collapses instead of reaching a figure as a repeated entry.
        """
        ask = Ask()
        first = ask("fill-opacity", 0.25, 1.0)
        second = ask("fill-opacity", 0.25, 1.0)
        assert [first, second, ask.named] == [0.25, 0.25, ["fill-opacity"]], (
            f"resolving one key twice recorded {ask.named}"
        )

    def test_the_record_does_not_depend_on_the_order_the_keywords_were_resolved_in(
        self,
    ):
        """Two builders resolving the same keywords in different orders record the same thing.

        Test scenario:
            The record reaches a saved figure, so an incidental ordering would make two identical calls
            two different figures — and a figure is compared, hashed and round-tripped.
        """
        forwards, backwards = Ask(), Ask()
        forwards("line-color", "#ff0000", "#3388ff")
        forwards("line-width", 3.0, 2.0)
        backwards("line-width", 3.0, 2.0)
        backwards("line-color", "#ff0000", "#3388ff")
        assert forwards.named == backwards.named, (
            f"one call recorded {forwards.named} and the same call the other way round "
            f"{backwards.named}"
        )

    def test_reading_the_record_does_not_hand_out_the_builders_own_list(self):
        """A description must not be writable through the object that produced it.

        Test scenario:
            `Symbology` freezes what it is given, but the builder hands the list over before that and goes
            on using the same `Ask`; a shared list would let a later keyword appear in a description
            already recorded.
        """
        ask = Ask()
        ask("size", 9.0, 6.0)
        published = ask.named
        published.append("smuggled")
        assert ask.named == ["size"], (
            f"the record answered {ask.named} after a caller mutated what it handed out"
        )


class TestTheRecordIsSomethingAFigureCanCarry:
    """The record reaches a saved figure, so the shared travel rule has to accept it."""

    def test_what_a_builder_records_travels_in_a_figure(self):
        """A list of strings is the one container shape that survives the JSON round trip.

        Test scenario:
            A tuple is refused precisely because JSON reads it back as a list, and a set has no JSON form
            at all — either would be dropped from every saved figure, which is the R2-M13 failure one field
            along.
        """
        ask = Ask()
        ask("circle-radius", ELSEWHERE, TIER_DEFAULT)
        assert travels_in_a_figure(ask.named), (
            f"{ask.named!r} cannot be written into a figure, so no saved layer would carry its asks"
        )

    def test_the_record_survives_the_freeze_a_symbology_applies(self):
        """`Symbology` turns the recorded list into a tuple, and the read side has to answer either way.

        Test scenario:
            The write side hands over a list because that is what travels; the read side sees whatever the
            spec froze it to. Reading it back off a real `Symbology` is what stops the two halves being
            written against different shapes.
        """
        ask = Ask()
        ask("circle-radius", ELSEWHERE, TIER_DEFAULT)
        ask("circle-opacity", 0.25, 0.9)
        recorded = Symbology(props={ASKED_PROP: ask.named})
        assert asked_style(recorded.props) == {"circle-radius", "circle-opacity"}, (
            f"a recorded ask read back as {sorted(asked_style(recorded.props))}"
        )

    def test_a_description_that_records_no_ask_has_asked_for_nothing(self):
        """The default reading, which is what makes the record fail-closed.

        Test scenario:
            A layer nobody styled records nothing, and so does a `Symbology` written by hand. Both must
            read as "no ask" rather than as "lift whatever style is here", which is what the tables'
            missing rows did.
        """
        read = asked_style({"paint": {"circle-radius": TIER_DEFAULT}})
        assert read == frozenset(), (
            f"a description with no record answered {sorted(read)}; a missing record is no ask, not "
            "every ask"
        )
