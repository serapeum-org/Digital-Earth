"""One catchable type for a layer that could not be drawn (#325).

`tests/base/test_renderer_conformance.py` holds each tier to this through its own drawer. This module holds
the two exception classes behind it to the same rule directly, because the property is a fact about the type
hierarchy and a reader looking for *why* ``except OffLimbError`` is enough should find it stated once, near
the classes rather than inside a per-tier adapter.

The defect it pins: a layer with nothing to draw is raised two ways — :class:`MissingObject` when a custom
layer's engine object is not in this process, :class:`OffLimbError` when the warp places none of the data —
and the two used to be unrelated (`LookupError` and `RuntimeError`). ``except OffLimbError`` therefore
covered the off-limb case on all four tiers and the missing-object case on three, missing the static tier,
which is the default one and the only tier that re-raises `MissingObject` rather than translating it.
"""

import pytest

from digitalearth.base.crs import OffLimbError
from digitalearth.base.custom import MissingObject, custom_kind, held_object
from digitalearth.base.registry import kind_info, resolve_uri

#: The layer every refusal here names, so the two messages are read side by side.
_LAYER = "wells"


def _a_custom_object_this_process_does_not_hold() -> None:
    """Ask for a custom layer's object where none is held, the way every tier's drawer does.

    Raises:
        MissingObject: always — an empty table is the case this function exists to provoke.
    """
    held_object(_LAYER, "custom:maplibre", {}, engine="maplibre", backend="web")


def _data_the_display_crs_cannot_place() -> None:
    """Refuse a layer the way a strict static map does when a warp places none of its data.

    Goes through the tier's own guard rather than raising the class here, so the check is over what a
    caller actually meets and not over a hand-built exception.

    Raises:
        OffLimbError: always — the guard is asked on a map built ``strict=True``.
    """
    from digitalearth.static import Map

    Map(crs=3857, strict=True)._skipped_off_limb("imshow")


def _a_url_with_no_resolver() -> None:
    """Ask the registry to open a scheme nobody registered.

    Raises:
        KeyError: always. A `LookupError`, like `MissingObject` — the nearest neighbour there is, and so
            the one most likely to be folded into it by a later tidy-up.
    """
    resolve_uri("nosuch://thing")


def _a_kind_nobody_registered() -> None:
    """Ask for the description of a layer kind that does not exist.

    Raises:
        KeyError: always. The other lookup in the same registry, and a caller error rather than a fact
            about the data — which is the line the shared type is not supposed to cross.
    """
    kind_info("nosuchkind")


def _a_name_that_cannot_be_a_kind() -> None:
    """Ask for a custom kind from a name that is not spellable as one.

    Raises:
        ValueError: always. Raised by the same module that defines `MissingObject`, so it is the refusal
            most easily reached by a change meant for the other one.
    """
    custom_kind("PyVista")


def _swallowed_by_the_shared_clause(refuse) -> bool:
    """Whether ``except OffLimbError`` would silence this refusal.

    The clause is written out and run, rather than asked about with `isinstance`, because that is what a
    caller writes and it is the thing the answer has to be true of.

    Args:
        refuse: A call that refuses something.

    Returns:
        `True` when the shared clause catches it, `False` when the refusal gets past it to a clause of its
        own.

    Raises:
        AssertionError: if the call refuses nothing, in which case the clause was never reached and the
            answer would be meaningless.
    """
    try:
        refuse()
    except OffLimbError:
        return True
    except Exception:
        return False
    raise AssertionError(f"{refuse.__name__} refused nothing, so no clause was reached")


class TestOneCatchableTypeForALayerThatCouldNotBeDrawn:
    """Both ways a layer ends up with nothing in it are caught by one ``except`` clause."""

    @pytest.mark.parametrize(
        "refuse",
        [
            _a_custom_object_this_process_does_not_hold,
            _data_the_display_crs_cannot_place,
        ],
        ids=["a missing custom object", "data the display CRS cannot place"],
    )
    def test_each_refusal_is_caught_by_the_shared_type(self, refuse):
        """`OffLimbError` reaches both, which is what the tiers' docstrings promise a caller.

        Args:
            refuse: The refusal to provoke.
        """
        with pytest.raises(OffLimbError):
            refuse()

    def test_the_missing_object_case_is_the_one_that_used_to_escape(self):
        """The half of the pair that moved: it was a bare `LookupError`, so the clause missed it."""
        assert issubclass(MissingObject, OffLimbError), (
            "a custom layer whose object is missing is a layer that could not be drawn, so it has to be "
            "catchable as one; without this base `except OffLimbError` misses the static tier"
        )

    def test_the_lookup_reading_is_kept(self):
        """Dropping `LookupError` would narrow a public exception for callers this repo cannot see.

        Test scenario:
            Nothing in `src/` or `tests/` catches `MissingObject` as a `LookupError` — measured, not
            assumed — so the base is kept for what it says rather than for a caller here. That makes it
            exactly the kind of base a later tidy-up removes without noticing, which is what this pins.
        """
        assert issubclass(MissingObject, LookupError), (
            "MissingObject is a lookup in the renderer's table of held objects that came back empty; "
            "removing the base silently narrows what a caller outside this repo can catch"
        )

    def test_the_two_bases_linearise_in_the_documented_order(self):
        """Multiple inheritance across two `Exception` branches is legal, and the order is the contract.

        Test scenario:
            `RuntimeError` and `LookupError` are both plain `Exception` subclasses with no conflicting
            C-level layout, so the pair resolves. Pinned rather than assumed because a linearisation
            failure is a `TypeError` at import time and a *reordering* is silent — and the order decides
            which class an `except` chain that names several of them picks.
        """
        order = [cls.__name__ for cls in MissingObject.__mro__]
        assert order == [
            "MissingObject",
            "OffLimbError",
            "RuntimeError",
            "LookupError",
            "Exception",
            "BaseException",
            "object",
        ], f"MissingObject resolves to {order}"

    @pytest.mark.parametrize(
        "refuse",
        [
            _a_url_with_no_resolver,
            _a_kind_nobody_registered,
            _a_name_that_cannot_be_a_kind,
        ],
        ids=[
            "a URL with no resolver",
            "a kind nobody registered",
            "a name that cannot be a kind",
        ],
    )
    def test_the_shared_type_does_not_reach_past_the_two_it_covers(self, refuse):
        """One type to catch must not become a clause that swallows unrelated failures.

        Args:
            refuse: A refusal the package really raises, which does not mean "this layer drew nothing".

        Test scenario:
            This used to hold hand-built `RuntimeError`/`KeyError`/`LookupError`/`ValueError` *instances*
            and ask whether each was an `OffLimbError`. Since `OffLimbError` is a strict subclass of all
            of those, no base-class instance could ever satisfy it: the test could only have failed if
            `OffLimbError` stopped being a `RuntimeError`, which is the opposite of what it is about
            (`R2-N4`). The refusals are now the package's own, provoked rather than constructed, and the
            clause is written out and run rather than asked about — so folding any of these into
            `MissingObject`, the change this exists to catch, fails here.
        """
        assert not _swallowed_by_the_shared_clause(refuse), (
            f"`except OffLimbError` silences {refuse.__name__}, which is a caller error and not a layer "
            "that drew nothing"
        )

    def test_the_message_still_says_which_of_the_two_cases_it_is(self):
        """A shared type is only usable while the messages stay distinguishable."""
        with pytest.raises(OffLimbError) as refusal:
            _a_custom_object_this_process_does_not_hold()
        assert "this figure does not carry" in str(refusal.value), (
            f"the missing-object message no longer says which case it is: {refusal.value}"
        )


def _a_builder_that_meets_a_missing_object(tier):
    """Run the tier's off-limb guard over a builder whose custom object is not here.

    The guard rather than a builder, because no builder that draws a custom layer carries it today: the
    two halves have not met in the package yet, and this is where they do. Declared as a function so the
    decorator wraps something with a name the guard can log.

    Args:
        tier: The map to run it on.

    Returns:
        Whatever the guard answers — the map itself when the layer was skipped.
    """
    from digitalearth.interactive.base import _skips_off_limb

    @_skips_off_limb
    def custom(self):
        """Reach for a custom layer's held object, the way a drawer does.

        Args:
            self: The map the builder is bound to.

        Raises:
            MissingObject: always — nothing is held under this id.
        """
        held_object(
            _LAYER, "custom:holoviews", {}, engine="holoviews", backend="interactive"
        )

    return custom(tier)


class TestATierSkippingAnOffLimbLayerSkipsAMissingObjectToo:
    """What the widened base means where the shared clause is actually written (`R2-L9`).

    Giving `MissingObject` an `OffLimbError` base did more than let a caller catch both: it put the
    missing-object case inside every ``except OffLimbError`` already in the package, and the interactive
    tier has one on its builders. That is a behaviour change nobody asked for in those words, so it is run
    here rather than reasoned about — and it turns out to be the answer the two modules already promise.
    `digitalearth.base.custom` says a missing object is "a layer with nothing to draw — a fact about the
    data, not about the call — which is why leniency is useful here and ``strict=True`` is how a caller
    opts out of it", and skip-unless-strict is exactly what the guard does with it.

    Latent today: no builder that draws a custom layer carries the guard. Pinned all the same, because the
    day one does, this is the behaviour it will have, and nothing else in the suite would say so.
    """

    def test_a_lenient_map_skips_the_layer_and_carries_on(self):
        """The lenient half: the rest of the figure still renders, as it does for off-limb data.

        Test scenario:
            The guard's contract is "skip the layer, warn, return the map so the call still chains". A
            missing object reaches it through the widened base and is answered the same way — measured,
            not assumed: the call returns the map itself and the figure gains no layer.
        """
        from digitalearth.interactive.map import InteractiveMap

        lenient = InteractiveMap()
        answered = _a_builder_that_meets_a_missing_object(lenient)
        assert answered is lenient, (
            f"the skipped call returned {answered!r}, so it no longer chains"
        )
        assert [layer.id for layer in lenient.figure_spec.layers] == [], (
            "a layer that drew nothing must not be described as though it had"
        )

    def test_a_strict_map_still_raises_the_missing_object_itself(self):
        """The strict half, and the reason the widening is not a loss of information.

        Test scenario:
            `strict=True` re-raises the error it was given, so what reaches the caller is still
            `MissingObject` with its own message — not an `OffLimbError` about the display CRS. A caller
            who opted out of leniency can still tell the two cases apart, which is the whole basis for
            giving them one base.
        """
        from digitalearth.interactive.map import InteractiveMap

        strict = InteractiveMap(strict=True)
        with pytest.raises(MissingObject) as refusal:
            _a_builder_that_meets_a_missing_object(strict)
        assert "this figure does not carry" in str(refusal.value), (
            f"the strict tier re-raised something else: {refusal.value}"
        )
