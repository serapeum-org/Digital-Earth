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
from digitalearth.base.custom import MissingObject, held_object

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
            C-level layout, so the pair resolves. Pinned rather than assumed because a C3 failure is a
            `TypeError` at import time and a *reordering* is silent — and the order decides which class an
            `except` chain that names several of them picks.
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

    def test_the_shared_type_does_not_reach_past_the_two_it_covers(self):
        """One type to catch must not become a clause that swallows unrelated failures.

        Test scenario:
            `OffLimbError` sits under `RuntimeError`, so the risk runs the other way: a caller writing
            ``except OffLimbError`` must still see a warp that failed for a real reason, a URL with no
            resolver, or a bad level list. Each of those is raised elsewhere in the package and none of
            them means "this layer drew nothing".
        """
        unrelated = [
            RuntimeError("the warp failed for a reason of its own"),
            KeyError("no resolver is registered for that URL scheme"),
            LookupError("an unrelated lookup"),
            ValueError("no level lies inside the data"),
        ]
        swallowed = [
            type(error).__name__
            for error in unrelated
            if isinstance(error, OffLimbError)
        ]
        assert swallowed == [], f"`except OffLimbError` would also silence {swallowed}"

    def test_the_message_still_says_which_of_the_two_cases_it_is(self):
        """A shared type is only usable while the messages stay distinguishable."""
        with pytest.raises(OffLimbError) as refusal:
            _a_custom_object_this_process_does_not_hold()
        assert "this figure does not carry" in str(refusal.value), (
            f"the missing-object message no longer says which case it is: {refusal.value}"
        )
