"""One set of checks every tier's **public** layer management answers to (order 23).

The renderers grew `set_visible`/`is_visible`/`remove` first, and the 3-D scene was the only tier that
reached them: `grep '\\.apply(' src/` found two call sites, both in `three_d/base.py`. So the other three
tiers described a figure they could not change — `get_layer`, `remove_layer`, `set_visible`, `move_layer` and
`replace_layer` were absent from their facades, and each renderer's `apply` was dead code in production.

This module is the contract those five (six, with `add_layer`) methods sign, written once rather than three
times. A tier supplies an adapter — how to build itself, how to draw two layers in one band, how to read the
order its **engine** will draw in, and a layer kind it cannot draw — and inherits every probe below.

**Why it is not in a `test_*.py` module.** Each tier's engine lives in its own pixi environment, and each
CI job collects one tier's directory. A shared module under `tests/base` is collected by the `dev` matrix
alone, where neither PyVista, HoloViz nor MapLibre is installed — which is how the shared *renderer*
contract skipped every check it declared for its whole life (`tests/base/test_renderer_conformance.py`,
review M6). So the probes live here, unnamed, and each tier subclasses them in its own directory:
`tests/static/test_static_layer_management.py`, `tests/interactive/...`, `tests/web/...`,
`tests/three_d/...`. Every tier's own job collects its own file, and no job has to name this one.

**What the probes read.** Visibility is read off the **renderer**, through the `is_visible` every tier's
renderer answers — never off the description. This package has shipped a recorded value nothing drew more
than once: a `visible=False` that hid the element while the record said shown, a held `Colormap` four
builders never read. A probe that reads `figure_spec` after `set_visible` agrees with the record and says
nothing about the picture, so each visibility probe asks the engine and the description separately.

Draw **order** is read off the engine too, through :meth:`LayerManagementContract.engine_order`: the
artists on a matplotlib axes in painting order, the queue a MapLibre widget is built from, the list a
HoloViews overlay composes. A tier whose engine has no order to read — VTK composites its actors by depth,
so the 3-D tier records draw order and does not apply it — returns `None` and those probes skip, loudly.
"""

from dataclasses import replace as with_fields
from typing import Any, Optional, Tuple

import pytest

from digitalearth.base.registry import band_of
from digitalearth.base.spec import LayerSpec

#: The group a replacement is filed under. A `group` change is what every tier's redraw guard reports as
#: reaching the engine — `(symbology, filter, group)` is the triple all four compare — and, unlike a
#: symbology prop, it needs no knowledge of which props the tier's drawer reads. So it is the portable
#: "restyle that must be drawn again", and the probes use it in place of a per-tier colormap.
REGROUPED = "layer-management-regrouped"

#: The name the second layer is drawn under. Both probe layers are one kind, so they share a band and a
#: move between them stays inside it — `LayerTree.move` refuses a position in another band, which is the
#: rule a layer switcher is held to.
TOP = "upper"

#: The name the first (bottom) layer is drawn under.
BOTTOM = "lower"


class LayerManagementContract:
    """What a tier supplies so the shared probes can run against its facade.

    Attributes:
        backend: The tier's name as its `Capabilities` spells it, used in failure messages.
        undrawable_kind: A registered layer kind this tier has **no drawer for**, whose data class is
            `"none"` so a replacement naming it carries no source. Replacing a layer with it is how the
            probes reach a refusal inside `apply` — the one path a change has to roll back from.
    """

    backend: str = ""
    undrawable_kind: str = ""

    def make(self) -> Any:
        """Return an open tier object with nothing drawn on it.

        Returns:
            The tier object.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def draw_two(self, tier: Any) -> Tuple[str, str]:
        """Draw two layers of one kind, bottom first, named :data:`BOTTOM` and :data:`TOP`.

        One kind, so both sit in one band: a move between them is a move a layer switcher can make, and a
        move across bands is what `LayerTree.move` refuses.

        Args:
            tier: The object :meth:`make` returned.

        Returns:
            The two ids, bottom first.

        Raises:
            NotImplementedError: when the tier has not supplied them.
        """
        raise NotImplementedError

    def engine_order(self, tier: Any) -> Optional[Tuple[str, ...]]:
        """Return the layer ids in the order the tier's **engine** will draw them.

        Read from the engine's own structure — the axes' artist list, the queue the widget is built from,
        the list the overlay composes — rather than from the description, which is what a move must not
        only change.

        Args:
            tier: The object :meth:`make` returned.

        Returns:
            The ids, bottom first, or `None` when the engine has no order to read: VTK composites by depth,
            so the 3-D tier records draw order and does not apply it.
        """
        return None

    def engine_holds(self, tier: Any, layer_id: str) -> Any:
        """Return what the engine holds for one layer, as something comparable by identity.

        Args:
            tier: The object :meth:`make` returned.
            layer_id: The layer to look up.

        Returns:
            What the renderer recorded for it, or `None` when it holds nothing. The renderer's record is
            the one reading every tier keeps, and a redraw replaces the value under the same key — which
            is what tells a redraw from a description that merely changed.
        """
        return tier._renderer.drawn.get(layer_id)

    def engine_objects(self, tier: Any) -> Tuple[Any, ...]:
        """Return everything the engine currently draws, as the objects themselves.

        The counterpart to :meth:`engine_holds`, and what the removal probe asks: a record cleared beside an
        engine that still draws the layer is the defect the shared renderer contract was written from, and
        only a reading that lists the engine's own objects can tell the two apart. Compared by **identity**,
        so a tier is held to having let go of the very object it drew.

        Args:
            tier: The object :meth:`make` returned.

        Returns:
            The objects, in any order — the artists on the axes, the elements the overlay composes, the
            entries the widget is built from, the actors on the plotter.
        """
        return tuple(tier._renderer.drawn.values())

    def drawn_is_visible(self, tier: Any, layer_id: str) -> bool:
        """Return whether the engine is currently drawing what was drawn for a layer.

        Args:
            tier: The object :meth:`make` returned.
            layer_id: The layer to ask about.

        Returns:
            `True` when the engine draws it. Asked of the renderer, in the tier's own terms — a matplotlib
            artist's flag, a VTK actor's visibility, MapLibre's `layout.visibility`, the HoloViews option
            the element carries.
        """
        return bool(tier._renderer.is_visible(layer_id))

    def close(self, tier: Any) -> None:
        """Release the tier and whatever it registered.

        Args:
            tier: The object :meth:`make` returned.
        """
        tier.close()


def _regrouped(tier: Any, layer_id: str) -> LayerSpec:
    """Return a layer's description with its group changed and nothing else.

    Args:
        tier: The tier holding the layer.
        layer_id: The layer to re-group.

    Returns:
        The new description — the portable "restyle that reaches the engine", see :data:`REGROUPED`.
    """
    return with_fields(tier.get_layer(layer_id), group=REGROUPED)


class LayerManagementConformance:
    """The probes themselves. A tier inherits them by setting `contract`."""

    #: The tier's adapter. A subclass sets this.
    contract: LayerManagementContract = LayerManagementContract()

    @pytest.fixture
    def drawn(self):
        """Yield a tier object with two layers of one kind drawn on it.

        Yields:
            The tier object.
        """
        built = self.contract.make()
        self.contract.draw_two(built)
        yield built
        try:
            self.contract.close(built)
        except Exception:  # pragma: no cover - a closed tier may refuse a second close
            pass

    def _engine_order(self, tier) -> Tuple[str, ...]:
        """Return the engine's draw order, skipping when the tier has none.

        Args:
            tier: The tier under test.

        Returns:
            The ids the engine draws, bottom first.
        """
        order = self.contract.engine_order(tier)
        if order is None:
            pytest.skip(
                f"the {self.contract.backend} tier's engine has no draw order to read, so a probe that "
                "read one would pass whatever the change did"
            )
        return order

    def test_get_layer_hands_back_the_description_filed_under_the_id(self, drawn):
        """`get_layer` is the read-back of what a builder recorded, by id.

        Args:
            drawn: The tier under test.
        """
        assert drawn.get_layer(TOP).id == TOP, (
            f"the {self.contract.backend} tier filed its top layer under another id"
        )

    def test_get_layer_refuses_an_id_nobody_drew_and_names_the_ones_that_were(
        self, drawn
    ):
        """A silent `None` for an unknown id looks exactly like a layer that refused to draw.

        Args:
            drawn: The tier under test.
        """
        with pytest.raises(KeyError) as refusal:
            drawn.get_layer("nope")
        assert TOP in str(refusal.value), (
            f"the {self.contract.backend} tier refused an unknown id without naming the ids it holds: "
            f"{refusal.value}"
        )

    def test_remove_layer_takes_the_layer_out_of_the_description(self, drawn):
        """What the tier reports must not name a layer it no longer draws.

        Args:
            drawn: The tier under test.
        """
        drawn.remove_layer(BOTTOM)
        assert drawn.layer_ids == [TOP], (
            f"the {self.contract.backend} tier still describes {BOTTOM!r}: {drawn.layer_ids}"
        )

    def test_remove_layer_takes_the_drawing_off_the_engine(self, drawn):
        """The half a description-only removal left behind: the engine still holding the layer.

        Args:
            drawn: The tier under test.

        Test scenario:
            The defect the shared renderer contract was written from, in its public form — an actor left on
            the engine under an id no layer owned, which `remove_layer` could then never reach.
        """
        held = self.contract.engine_holds(drawn, BOTTOM)
        assert held is not None, (
            f"the {self.contract.backend} tier's adapter reports nothing drawn for {BOTTOM!r}, so the check "
            "below would pass vacuously"
        )
        drawn.remove_layer(BOTTOM)
        left = [obj for obj in self.contract.engine_objects(drawn) if obj is held]
        assert left == [], (
            f"the {self.contract.backend} tier removed {BOTTOM!r} from its description and left what it "
            "drew on the engine"
        )

    def test_remove_layer_refuses_an_id_nobody_drew(self, drawn):
        """A no-op removal is indistinguishable from a layer that would not go away.

        Args:
            drawn: The tier under test.
        """
        with pytest.raises(KeyError):
            drawn.remove_layer("nope")

    def test_remove_layer_gives_the_name_back(self, drawn):
        """An id is reserved exactly while the layer is on the figure (`free_layer_id`).

        Args:
            drawn: The tier under test.

        Test scenario:
            On a tier whose layer switcher captions a layer by its id, a name removed and asked for again
            coming back as `lower-2` is a caption the viewer reads (review R2-M9).
        """
        drawn.remove_layer(BOTTOM)
        self.contract.draw_two(drawn)
        assert drawn.layer_ids.count(BOTTOM) == 1, (
            f"the {self.contract.backend} tier did not hand {BOTTOM!r} back: {drawn.layer_ids}"
        )

    def test_remove_layer_returns_the_tier_so_calls_chain(self, drawn):
        """Every builder on every tier chains; a layer-management call is no different.

        Args:
            drawn: The tier under test.
        """
        assert drawn.remove_layer(BOTTOM) is drawn, "remove_layer must return the tier"

    def test_set_visible_false_stops_the_engine_drawing_the_layer(self, drawn):
        """Read off the renderer, not the record: a hidden record that still draws is the defect.

        Args:
            drawn: The tier under test.
        """
        drawn.set_visible(TOP, False)
        assert self.contract.drawn_is_visible(drawn, TOP) is False, (
            f"the {self.contract.backend} tier recorded {TOP!r} hidden and its engine still draws it"
        )

    def test_set_visible_false_is_described_as_well(self, drawn):
        """A hidden layer is still described, which is what lets a viewer switch it back on.

        Args:
            drawn: The tier under test.
        """
        drawn.set_visible(TOP, False)
        assert drawn.figure_spec.layers.is_visible(TOP) is False, (
            f"the {self.contract.backend} tier hid {TOP!r} without describing it hidden"
        )

    def test_set_visible_brings_the_layer_back_on_the_engine(self, drawn):
        """The other direction, because a tier that hides everything passes the first probe.

        Args:
            drawn: The tier under test.
        """
        drawn.set_visible(TOP, False)
        drawn.set_visible(TOP)
        assert self.contract.drawn_is_visible(drawn, TOP) is True, (
            f"the {self.contract.backend} tier could not draw {TOP!r} again after hiding it"
        )

    def test_set_visible_leaves_the_other_layer_alone(self, drawn):
        """One layer is toggled, not the band it sits in.

        Args:
            drawn: The tier under test.
        """
        drawn.set_visible(TOP, False)
        assert self.contract.drawn_is_visible(drawn, BOTTOM) is True, (
            f"the {self.contract.backend} tier hid {BOTTOM!r} along with {TOP!r}"
        )

    def test_set_visible_refuses_an_id_nobody_drew(self, drawn):
        """An id nothing draws has no visibility to set.

        Args:
            drawn: The tier under test.
        """
        with pytest.raises(KeyError):
            drawn.set_visible("nope", False)

    def test_set_visible_returns_the_tier_so_calls_chain(self, drawn):
        """A toggle chains like a builder does.

        Args:
            drawn: The tier under test.
        """
        assert drawn.set_visible(TOP, False) is drawn, (
            "set_visible must return the tier"
        )

    def test_move_layer_reorders_the_description(self, drawn):
        """`move_layer` addresses a layer by id and a position by index, as a list does.

        Args:
            drawn: The tier under test.
        """
        drawn.move_layer(TOP, 0)
        assert drawn.layer_ids == [TOP, BOTTOM], (
            f"the {self.contract.backend} tier did not reorder its description: {drawn.layer_ids}"
        )

    def test_move_layer_reorders_what_the_engine_draws(self, drawn):
        """Draw order decides what is on top, so a move the engine never hears about is a no-op.

        Args:
            drawn: The tier under test.

        Test scenario:
            `FigureSpec.diff` has reported `order` since the seam landed and **no renderer acted on it**:
            every `_reconcile` handled added/removed/rebuilt/restyled/shown/hidden and skipped it. So a
            `move_layer` wired only into the tree would reorder `layer_ids` and leave the picture exactly
            as it was — the record-without-a-drawing shape this repo has shipped before.
        """
        before = self._engine_order(drawn)
        drawn.move_layer(TOP, 0)
        after = self._engine_order(drawn)
        assert (after, before) == ((TOP, BOTTOM), (BOTTOM, TOP)), (
            f"the {self.contract.backend} tier's engine drew {after} after moving {TOP!r} to the bottom"
        )

    def test_move_layer_refuses_a_position_outside_the_figure(self, drawn):
        """A position no layer can take is a caller's mistake, not a clamp.

        Args:
            drawn: The tier under test.
        """
        with pytest.raises(IndexError):
            drawn.move_layer(TOP, 9)

    def test_move_layer_refuses_an_id_nobody_drew(self, drawn):
        """There is no position for a layer the figure does not hold.

        Args:
            drawn: The tier under test.
        """
        with pytest.raises(KeyError):
            drawn.move_layer("nope", 0)

    def test_move_layer_returns_the_tier_so_calls_chain(self, drawn):
        """A reorder chains like a builder does.

        Args:
            drawn: The tier under test.
        """
        assert drawn.move_layer(TOP, 0) is drawn, "move_layer must return the tier"

    def test_replace_layer_keeps_the_id_and_the_place_in_draw_order(self, drawn):
        """A replacement re-describes one layer; it does not re-add it on top.

        Args:
            drawn: The tier under test.
        """
        drawn.replace_layer(_regrouped(drawn, BOTTOM))
        assert drawn.layer_ids == [BOTTOM, TOP], (
            f"the {self.contract.backend} tier moved {BOTTOM!r} while replacing it: {drawn.layer_ids}"
        )

    def test_replace_layer_files_the_new_description(self, drawn):
        """What the tier reports afterwards is the description it was handed.

        Args:
            drawn: The tier under test.
        """
        drawn.replace_layer(_regrouped(drawn, BOTTOM))
        assert drawn.get_layer(BOTTOM).group == REGROUPED, (
            f"the {self.contract.backend} tier kept the old description of {BOTTOM!r}"
        )

    def test_replace_layer_draws_the_layer_again(self, drawn):
        """The engine has to hear about it: a re-styled layer is drawn from the new description.

        Args:
            drawn: The tier under test.

        Test scenario:
            Identity, not equality: a tier's record of what it drew is a frozen value on three of the four,
            so a layer drawn again from an equal description would compare equal to the one it replaced.
        """
        held = self.contract.engine_holds(drawn, BOTTOM)
        drawn.replace_layer(_regrouped(drawn, BOTTOM))
        # `held` stays referenced while the two are compared, so the old value cannot be freed and its
        # address handed to the one that replaced it.
        assert self.contract.engine_holds(drawn, BOTTOM) is not held, (
            f"the {self.contract.backend} tier re-described {BOTTOM!r} without drawing it again"
        )

    def test_replace_layer_refuses_an_id_nobody_drew(self, drawn):
        """A replacement names the layer it replaces, so an unknown id replaces nothing.

        Args:
            drawn: The tier under test.
        """
        with pytest.raises(KeyError):
            drawn.replace_layer(with_fields(drawn.get_layer(TOP), id="nope"))

    def test_replace_layer_returns_the_tier_so_calls_chain(self, drawn):
        """A replacement chains like a builder does.

        Args:
            drawn: The tier under test.
        """
        replacement = _regrouped(drawn, TOP)
        assert drawn.replace_layer(replacement) is drawn, (
            "replace_layer must return the tier"
        )

    def _refuse_a_replacement(self, tier) -> None:
        """Replace one layer with a kind this tier cannot draw, expecting the refusal.

        Args:
            tier: The tier under test.
        """
        held = tier.get_layer(TOP)
        undrawable = LayerSpec(
            TOP,
            self.contract.undrawable_kind,
            # The band the layer already sits in, spelled out: a replacement that changes the band is
            # re-placed at the top of its new one, which would make this a move as well as a refusal.
            band=held.band or band_of(held.kind),
        )
        with pytest.raises(KeyError):
            tier.replace_layer(undrawable)

    def test_a_refused_change_is_not_the_figure_the_tier_reports(self, drawn):
        """The change path installs the new description only once the engine has drawn it.

        Args:
            drawn: The tier under test.

        Test scenario:
            The 3-D scene installed the figure first and kept it when `apply` raised, so `layer_ids`,
            `figure_spec` and `to_dict` all advertised a layer that was never drawn and could not be
            (review H7). Its `_change` is the fix, and this is the same question asked of the three change
            paths order 23 adds.
        """
        held = drawn.figure_spec
        self._refuse_a_replacement(drawn)
        assert drawn.figure_spec == held, (
            f"the {self.contract.backend} tier kept a figure it could not draw"
        )

    def test_a_refused_change_leaves_the_engine_as_it_found_it(self, drawn):
        """The drawing rolls back with the description, or the two disagree.

        Args:
            drawn: The tier under test.
        """
        before = self.contract.engine_order(drawn)
        self._refuse_a_replacement(drawn)
        assert self.contract.engine_order(drawn) == before, (
            f"a refused change left the {self.contract.backend} tier's engine drawing "
            f"{self.contract.engine_order(drawn)}; it drew {before}"
        )
