"""What a tier may lift out of its own style bucket onto a declared channel (`R-L9`).

`portable_constants` is the second half of `StyleSchema.route`: a backend that has already resolved a
caller's keywords into its engine's spelling calls it to say the same thing again in the vocabulary the
other tiers read. Which values it may claim is the same question `travels_in_a_figure` answers for a
layer's description, and the two used to answer it differently — a `numpy` integer or boolean lifted no
channel at all, while the rule beside it carried both. These pin the two together, and pin the one place
they are deliberately not the same: a container travels in a figure but is never a channel's constant.
"""

import numpy as np
import pytest

from digitalearth.base.spec._serial import travels_in_a_figure
from digitalearth.base.spec.style import portable_constants

#: The one key the probes route, and the channel it drives. `portable_constants` passes over anything the
#: mapping does not name, so a single pair is the whole surface a value has to cross.
_CHANNELS = {"circle-radius": "size"}


class _KeyedProvider(dict):
    """A `dict` subclass carrying a credential, standing in for an `xyzservices.TileProvider`.

    Declared here rather than imported so the rule is checked without the tile stack installed.
    """


def _lifted(value):
    """Return the channels `portable_constants` claims for one style value.

    Args:
        value: What the tier recorded under the routed key.

    Returns:
        The sorted channel names, so a claim and a refusal are both readable in a failure message.
    """
    return sorted(portable_constants({"circle-radius": value}, _CHANNELS))


class TestTheLiftAndTheTravelRuleAgree:
    """One question — "can this value cross to another engine?" — must not have two answers."""

    @pytest.mark.parametrize(
        "value",
        [np.float64(2.5), np.int64(7), np.bool_(True), np.float32(1.5), np.uint8(3)],
        ids=["np-float64", "np-int64", "np-bool", "np-float32", "np-uint8"],
    )
    def test_every_numpy_scalar_the_rule_carries_lifts_a_channel(self, value):
        """The numpy family answers as one here too, rather than per type.

        Args:
            value: The numpy scalar under test.

        Test scenario:
            The gate was `isinstance(value, (bool, int, float, str))`. `np.float64` subclasses `float` so
            it lifted, but `np.int64` and `np.bool_` subclass neither, so a tier that recorded a numpy
            integer or a numpy flag published no channel and the layer crossed unstyled — the same
            per-type accident `travels_in_a_figure` was widened to end (#329), one module over.
        """
        assert _lifted(value) == ["size"], (
            f"{value!r} is a scalar the shared rule carries, so the channel it drives is published"
        )

    @pytest.mark.parametrize(
        "value",
        [np.datetime64("2024-01-01"), np.complex128(1 + 2j), np.timedelta64(5, "D")],
        ids=["np-datetime64", "np-complex128", "np-timedelta64"],
    )
    def test_a_numpy_scalar_the_rule_refuses_lifts_nothing(self, value):
        """Admitting the family must not admit what no figure can be written with.

        Args:
            value: The numpy scalar under test.

        Test scenario:
            The counterpart to the check above, and the reason the widening asks the shared rule rather
            than adding `np.generic` to a membership tuple: a lifted `datetime64` would sit in an
            `Encoding` that `Symbology.to_dict` then refuses, turning a drawable layer into a figure that
            cannot be saved at all.
        """
        assert _lifted(value) == [], (
            f"{value!r} has no JSON form, so claiming a channel for it would make the figure unsavable"
        )

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_a_number_json_cannot_spell_lifts_nothing(self, value):
        """The non-finite refusal survives the widening, because the shared rule makes it too.

        Args:
            value: The non-finite number under test.
        """
        assert _lifted(value) == [], (
            f"{value!r} would make the whole figure unwritable, so no channel is claimed for it"
        )

    def test_a_declined_key_lifts_nothing(self):
        """`None` is a caller declining a key rather than binding one, which is not a constant."""
        assert _lifted(None) == [], (
            "a declined key binds no channel, so nothing is published for it"
        )


class TestAContainerTravelsButIsNeverAConstant:
    """The one place the two rules differ on purpose, so the difference is written down."""

    @pytest.mark.parametrize(
        "value",
        [["#440154", "#fde725"], [0.0, 0.5, 1.0], {"a": 1}],
        ids=["palette", "color-levels", "mapping"],
    )
    def test_a_container_the_rule_carries_still_lifts_no_channel(self, value):
        """A figure carries a palette; a *channel constant* is one value, not a list of them.

        Args:
            value: The container under test.

        Test scenario:
            `travels_in_a_figure` says yes to all three — that is what #330 restored — so the refusal
            here cannot be inherited from it and has to be its own check. A per-class colormap list or a
            classifier's edges describe the layer's scale, not one channel's constant, and publishing one
            as a constant would describe the layer wrongly rather than not at all.
        """
        assert travels_in_a_figure(value), (
            f"{value!r} is carried by a figure, so this is not a refusal inherited from the rule"
        )
        assert _lifted(value) == [], (
            f"but {value!r} is not one channel's constant, so no channel is claimed for it"
        )

    def test_a_mapping_subclass_carrying_a_credential_lifts_nothing(self):
        """An `xyzservices.TileProvider` *is* a mapping, and one of its values is the caller's API key.

        Test scenario:
            The reason the lift is restricted to a scalar at all. A tier records the provider it was
            handed; copying it onto a channel would write the key into the next figure saved. The rule
            refuses it as a `dict` subclass and the lift refuses it as a container, so both ends are shut.
        """
        provider = _KeyedProvider(
            {"name": "Thunderforest", "apikey": "FAKE-KEY-NOT-REAL"}
        )
        assert _lifted(provider) == [], (
            "a mapping is never a channel's constant, credential or not"
        )

    def test_an_engine_expression_lifts_nothing(self):
        """A compiled MapLibre expression is that engine's machinery and means nothing to another.

        Test scenario:
            The case the module's own example documents, kept as a test so the widening cannot quietly
            start claiming a channel for a tuple.
        """
        paint = ("step", ("get", "pop"), "#440154", "#fde725")
        assert (
            portable_constants({"fill-color": paint}, {"fill-color": "color"}) == {}
        ), "a data-driven expression is not a constant, so no channel is claimed for it"
