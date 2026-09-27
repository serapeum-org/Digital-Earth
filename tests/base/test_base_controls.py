"""The shared layer-control vocabulary, held to one answer for both tiers that have a layer control.

``layer_control`` shared not one parameter between the web and interactive tiers (#264). Two of the three
names the settled signature agreed on are *vocabularies* rather than values — which controls may be exposed,
and which corners a control may sit in — so they live in :mod:`digitalearth.base.controls`, where neither
tier can drift from the other's spelling of them.

What is checked here is the refusals, because that is where a shared vocabulary earns its place: a tier that
cannot build a control says so **by name** rather than accepting the request and building nothing, which is
how two of the interactive tier's own flags came to look implemented (#242, #244).
"""

import ast
import pathlib

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
        """No tier builds a drag widget, so advertising the control would only create something to refuse."""
        assert "reorder" not in LAYER_CONTROLS, (
            f"reorder names an operation, not a widget any tier draws: {list(LAYER_CONTROLS)}"
        )

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

    def test_a_control_the_tier_cannot_build_is_named_before_the_missing_toggle(self):
        """The refusal a caller can act on is "this tier has no such widget", so it comes first.

        Test scenario:
            The required-control check ran first, so a web-tier caller writing `controls=["opacity"]` — the
            exact mistake the web tier's own docstring says it should hear about, "a caller moving a call here
            should hear that this tier cannot" — was told it needed `visibility` instead. The unbuildable
            message, which `base/controls.py` calls "the case worth refusing loudly", was unreachable for any
            tier whose `offered` omits a control unless the caller *also* spelled `visibility` (review R2-L13).
        """
        with pytest.raises(ValueError) as refused:
            resolved_controls(["opacity"], offered=("visibility",), caller=CALLER)
        assert "cannot build the controls ['opacity']" in str(refused.value), refused.value

    def test_the_missing_toggle_is_still_refused_when_every_name_is_buildable(self):
        """Reordering must not swallow the other refusal, so the tier that builds all three still hears it."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(["opacity"], offered=LAYER_CONTROLS, caller=CALLER)
        assert REQUIRED_CONTROL in str(refused.value), refused.value

    def test_a_name_no_tier_has_is_still_refused_before_either(self):
        """A typo is neither unbuildable nor a missing toggle, and stays the first thing checked."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(["opacty"], offered=("visibility",), caller=CALLER)
        assert "which name no control" in str(refused.value), refused.value

    def test_every_refusal_says_where_the_call_was(self):
        """Three messages, one rule: each quotes the builder the keywords were written on."""
        with pytest.raises(ValueError) as refused:
            resolved_controls(["nonsense"], offered=LAYER_CONTROLS, caller=CALLER)
        assert CALLER in str(refused.value), refused.value


class TestTheReorderRationaleIsStillTrue:
    """M4 — the reason `"reorder"` is omitted must be a reason that still holds.

    The `#:` comment on :data:`LAYER_CONTROLS` said "no tier can reorder layers (the registry has no stable
    per-layer handle to reorder by)". Order 23 gave all four tiers `move_layer(layer_id, index)` — layers are
    addressed by id and draw order is manipulated by it — so the blocker the comment cites was removed on the
    same branch the comment was written on. This is the file that decides the vocabulary every tier shares,
    so a false reason here is worse than ordinary stale prose.
    """

    #: The tier modules that define `move_layer`, read as source so no engine has to be installed.
    MOVERS = (
        "static/scene.py",
        "interactive/base.py",
        "web/base.py",
        "three_d/base.py",
    )

    @staticmethod
    def _rationale() -> str:
        """The `#:` doc comment block that precedes `LAYER_CONTROLS`.

        Returns:
            The comment lines joined into one string, with the `#:` markers stripped. Read from the source
            because `#:` comments do not exist at runtime — the same way
            `tests/base/test_default_basemap.py` reads the basemap constant's.
        """
        from digitalearth.base import controls

        lines = pathlib.Path(controls.__file__).read_text(encoding="utf-8").splitlines()
        index = next(
            number
            for number, line in enumerate(lines)
            if line.startswith("LAYER_CONTROLS")
        )
        block = []
        while index > 0 and lines[index - 1].startswith("#:"):
            index -= 1
            block.insert(0, lines[index].removeprefix("#:").strip())
        return " ".join(block)

    def test_it_does_not_cite_the_handle_order_23_added(self):
        """The omission must not be explained by a missing per-layer handle, which now exists.

        Test scenario:
            A caller reading this comment is told reordering is impossible anywhere. It is not: it is
            `move_layer`, on every tier. What no tier has is the *widget* — the drag control a layer switcher
            would need — which is a different claim and the one that is still true.
        """
        rationale = self._rationale()
        stale = [
            phrase
            for phrase in ("stable per-layer handle", "no tier can reorder")
            if phrase in rationale
        ]
        assert stale == [], (
            f"the reorder rationale still cites {stale}, which order 23 removed: {rationale}"
        )

    def test_it_names_the_reorder_the_tiers_do_have(self):
        """The comment has to point at `move_layer`, so the omission reads as "no widget", not "no reorder"."""
        rationale = self._rationale()
        assert "move_layer" in rationale, (
            f"the reorder rationale must name the programmatic reorder callers do have: {rationale}"
        )

    @pytest.mark.parametrize("module", MOVERS, ids=lambda value: value)
    def test_every_tier_defines_the_reorder_by_id(self, module):
        """The fact that makes the old reason false: all four tiers take a layer id and an index.

        Args:
            module: The tier module, relative to `src/digitalearth`.

        Test scenario:
            Parsed rather than imported, because three of the four need an engine the lean environment has
            no reason to carry — and the claim is about the surface, which the source states.
        """
        root = pathlib.Path(__file__).resolve().parents[2] / "src" / "digitalearth"
        tree = ast.parse((root / module).read_text(encoding="utf-8"))
        movers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "move_layer"
        ]
        assert movers != [], (
            f"{module} defines no move_layer for the rationale to point at"
        )
        taken = [arg.arg for arg in movers[0].args.args]
        assert taken[1:] == ["layer_id", "index"], (
            f"{module}'s move_layer takes {taken[1:]}, not the (layer_id, index) handle"
        )
