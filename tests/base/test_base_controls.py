"""The shared layer-control vocabulary, held to one answer for both tiers that have a layer control.

``layer_control`` shared not one parameter between the web and interactive tiers (#264). Two of the three
names the settled signature agreed on are *vocabularies* rather than values — which controls may be exposed,
and which corners a control may sit in — so they live in :mod:`digitalearth.base.controls`, where neither
tier can drift from the other's spelling of them.

What is checked here is the refusals, because that is where a shared vocabulary earns its place: a tier that
cannot build a control says so **by name** rather than accepting the request and building nothing, which is
how two of the interactive tier's own flags came to look implemented (#242, #244).
"""

import pytest

from digitalearth.base.controls import (
    CONTROL_POSITIONS,
    LAYER_CONTROLS,
    REQUIRED_CONTROL,
    check_control_position,
    resolved_controls,
)

#: The caller name every refusal below has to quote back, so a message can be traced to a call.
CALLER = "WebMap.layer_control()"


class TestTheVocabularyItself:
    """Two tables a tier reads rather than re-spells."""

    def test_reorder_is_not_a_control(self):
        """No tier can reorder layers, so advertising it would only create something to refuse."""
        assert "reorder" not in LAYER_CONTROLS

    def test_visibility_is_one_of_the_controls_it_requires(self):
        """The mandatory control has to be a control, or nothing could ever satisfy the rule."""
        assert REQUIRED_CONTROL in LAYER_CONTROLS

    def test_the_corners_are_the_four_maplibre_names(self):
        """Both tiers anchor to these, so the set is the agreement rather than a tier's own list."""
        assert sorted(CONTROL_POSITIONS) == [
            "bottom-left",
            "bottom-right",
            "top-left",
            "top-right",
        ]


class TestACornerIsOneOfFour:
    """A browser ignores a corner it does not know, so a typo has to be refused here."""

    @pytest.mark.parametrize("position", sorted(CONTROL_POSITIONS))
    def test_each_corner_passes(self, position):
        """Every corner in the table is accepted, or the table is not the vocabulary.

        Args:
            position: The corner under test.
        """
        assert check_control_position(position) is None

    def test_anything_else_names_the_four(self):
        """The refusal has to say what *is* legal, since the caller has to pick one."""
        with pytest.raises(ValueError) as refused:
            check_control_position("middle")
        assert "top-left" in str(refused.value), refused.value

    def test_the_refusal_quotes_what_was_asked_for(self):
        """A message that does not name the typo cannot be acted on."""
        with pytest.raises(ValueError) as refused:
            check_control_position("centre")
        assert "'centre'" in str(refused.value), refused.value


class TestWhatATierWillBuild:
    """``controls=`` is honoured to the letter, or refused naming the reason."""

    def test_the_names_come_back_in_the_callers_order(self):
        """The order is the layout order, so it is the caller's to choose."""
        assert resolved_controls(
            ["visibility", "basemap", "opacity"], offered=LAYER_CONTROLS, caller=CALLER
        ) == ("visibility", "basemap", "opacity")

    def test_a_repeated_control_is_collapsed(self):
        """Naming one twice is one widget asked for twice, not two widgets."""
        assert resolved_controls(
            ["visibility", "opacity", "opacity"],
            offered=LAYER_CONTROLS,
            caller=CALLER,
        ) == ("visibility", "opacity")

    def test_a_name_no_tier_has_is_refused_with_the_vocabulary(self):
        """A typo builds nothing and says nothing unless it is refused here."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(
                ["visibility", "opacty"], offered=LAYER_CONTROLS, caller=CALLER
            )
        assert "['opacty'], which name no control" in str(refused.value), refused.value

    def test_reorder_is_refused_like_any_other_unknown_name(self):
        """It reads like a control and is not one, which is exactly why it must not pass."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(
                ["visibility", "reorder"], offered=LAYER_CONTROLS, caller=CALLER
            )
        assert "['reorder'], which name no control" in str(refused.value), refused.value

    def test_dropping_the_required_control_is_refused(self):
        """A layer control with no per-layer toggle has nothing to switch."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(["opacity"], offered=LAYER_CONTROLS, caller=CALLER)
        assert REQUIRED_CONTROL in str(refused.value), refused.value

    def test_an_empty_control_set_is_refused_for_the_same_reason(self):
        """Nothing at all is also a control set without the toggle."""
        with pytest.raises(ValueError) as refused:
            resolved_controls([], offered=LAYER_CONTROLS, caller=CALLER)
        assert REQUIRED_CONTROL in str(refused.value), refused.value

    def test_a_control_the_tier_cannot_build_is_refused_naming_what_it_can(self):
        """The web tier's case: a real control, and no widget for it in this engine."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(
                ["visibility", "opacity"], offered=("visibility",), caller=CALLER
            )
        assert "['visibility']" in str(refused.value), refused.value

    def test_every_refusal_says_where_the_call_was(self):
        """Three messages, one rule: each quotes the builder the keywords were written on."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(["nonsense"], offered=LAYER_CONTROLS, caller=CALLER)
        assert CALLER in str(refused.value), refused.value
