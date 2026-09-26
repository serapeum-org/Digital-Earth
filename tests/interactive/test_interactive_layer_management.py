"""The interactive tier's half of order 23: the public toggle, reorder, replace, read-back and removal.

The shared probes are :mod:`tests.base.layer_management`; this module supplies the adapter and the checks
that are this tier's own — chiefly what happens to a **custom** layer, the one kind this tier keeps outside
its renderer and so cannot draw again from a description.

**Why the engine reading is the overlay list.** `render()` composes
:attr:`~digitalearth.interactive.base.InteractiveMapBase.layers` with `*`, bottom first, so that list is what
a viewer sees. Until order 23 only the builders wrote it and `Renderer.apply` reconciled the renderer's own
record beside it — so a change reached a record and never the picture (review M1). The probes read the list.
"""

from dataclasses import replace as with_fields

import pytest

from digitalearth.base.capabilities import CapabilityError
from digitalearth.base.spec import Symbology
from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip(
    "holoviews", reason="the interactive tier needs the interactive environment"
)
pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)
from tests.base.layer_management import (
    BOTTOM,
    TOP,
    LayerManagementConformance,
    LayerManagementContract,
)


class InteractiveContract(LayerManagementContract):
    """The interactive tier's adapter for the shared layer-management probes."""

    backend = "interactive"
    #: A caller's own PyVista object: registered, drawn from no data, and nothing this tier can draw.
    undrawable_kind = "custom:pyvista"

    def make(self):
        """Return an empty map.

        Returns:
            The map. Constructing one touches no engine; the builders lazy-import HoloViz.
        """
        return InteractiveMap()

    def draw_two(self, tier):
        """Draw two text labels, which are one kind and so share one band.

        Args:
            tier: The map.

        Returns:
            The two ids, bottom first.
        """
        tier.text(0.0, 0.0, "lower", name=BOTTOM)
        tier.text(1.0, 1.0, "upper", name=TOP)
        return BOTTOM, TOP

    def engine_order(self, tier):
        """Return the layer ids in the order `render()` overlays their elements.

        Args:
            tier: The map.

        Returns:
            The ids, bottom first, read by matching each overlaid element against what the renderer drew —
            so this answers from the list the overlay is composed from rather than from the record.
        """
        owner = {
            id(drawn.element): layer_id
            for layer_id, drawn in tier._renderer.drawn.items()
        }
        return tuple(
            owner[id(element)] for element in tier.layers if id(element) in owner
        )

    def engine_objects(self, tier):
        """Return every element `render()` would overlay.

        Args:
            tier: The map.

        Returns:
            The elements themselves, so a removal is held to having taken the very element it drew out of
            the overlay rather than only out of the renderer's record — which `Renderer.remove` clears
            whether or not the picture follows.
        """
        return tuple(tier.layers)

    def engine_holds(self, tier, layer_id):
        """Return the element the overlay holds for one layer.

        The overlay rather than the renderer's record: the record is what decides whether the overlay is
        re-arranged, so a probe reading it back would be asking the decision about itself. `layers` is kept
        in the tree's order — each element is placed at its layer's index — so the index is the key.

        Args:
            tier: The map.
            layer_id: The layer to look up.

        Returns:
            The element, or `None` when the map overlays none for it.
        """
        ids = tier.layer_ids
        if layer_id not in ids:
            return None
        return tier.layers[ids.index(layer_id)]


class TestInteractiveLayerManagement(LayerManagementConformance):
    """The interactive tier, signing the contract the 3-D tier already passes."""

    contract = InteractiveContract()


@pytest.fixture
def held():
    """Yield a map holding two of the caller's own objects, engine-free.

    A string stands in for a HoloViews element: the registry does not inspect what it is handed, which is
    what lets these checks run without the `interactive` extra — and a custom layer is exactly the case the
    renderer has no drawer for, so nothing here builds an element. A string also has no visibility option
    to write, which is the other half of what :class:`TestHidingACallersOwnElement` measures.

    Yields:
        The map, with `lower` and `upper` registered.
    """
    built = InteractiveMap()
    built.add_layer("lower-element", name=BOTTOM)
    built.add_layer("upper-element", name=TOP)
    yield built
    built.close()


@pytest.fixture
def kept():
    """Yield a map holding one real HoloViews element of the caller's own.

    The `held` fixture's strings cannot be hidden, so they can only show the warning. This one is a
    `Points`, which Bokeh *does* take `visible` for — the case where "hidden" has to reach the engine.

    Yields:
        The map, with `lower` registered.
    """
    built = InteractiveMap()
    built.add_layer(hv.Points([(0.0, 0.0)]), name=BOTTOM)
    yield built
    built.close()


def _bokeh_visible(element):
    """Return what HoloViews' own Bokeh option store holds for one element's `visible`.

    Args:
        element: The element to ask.

    Returns:
        The applied option, or `None` when none was applied. Read out of the engine's store — the same
        place a Bokeh render reads it — rather than out of this package's record, which is the whole
        question review H8 asks. `hv.renderer` registers the store without making Bokeh current, the
        idiom :func:`~digitalearth.interactive.style_fold.allowed_options` uses for the same read.
    """
    hv.renderer("bokeh")
    return hv.Store.lookup_options("bokeh", element, "style").kwargs.get("visible")


class TestACallersOwnElement:
    """`custom:holoviews` — described, addressable, and the one kind no description can rebuild."""

    def test_removing_it_takes_it_out_of_the_overlay(self, held):
        """The element goes with the description, or `render()` keeps composing it.

        Args:
            held: The map under test.

        Test scenario:
            A custom layer is not in the renderer's record — the tier keeps the caller's object outside it —
            so a removal that only cleared the record would leave the element overlaid under no id at all.
        """
        held.remove_layer(BOTTOM)
        assert held.layers == ["upper-element"], held.layers

    def test_moving_it_reorders_the_overlay(self, held):
        """A move has to reach the list `render()` composes, custom layer or not.

        Args:
            held: The map under test.
        """
        held.move_layer(TOP, 0)
        assert held.layers == ["upper-element", "lower-element"], held.layers

    def test_replacing_its_description_leaves_the_element_where_it_is(self, held):
        """There is nothing to draw again: what a figure keeps of a custom layer is its description.

        Args:
            held: The map under test.

        Test scenario:
            The element is the caller's and the tier holds no recipe for it, so a replacement re-describes
            the layer and must not drop the object — which is what rebuilding the overlay from the
            renderer's record alone would have done.
        """
        held.replace_layer(with_fields(held.get_layer(BOTTOM), label="Lower"))
        assert held.layers == ["lower-element", "upper-element"], held.layers

    def test_regrouping_it_is_refused_rather_than_sent_to_a_drawer(self, held):
        """Review H7: the gate passed on the declaration, then the drawer table refused the same kind.

        Args:
            held: The map under test.

        Test scenario:
            A group change is one the tier classes as reaching HoloViews, so the reconcile rebuilt the layer
            — and there is nothing to rebuild a caller's own element from. The refusal came out of
            `drawer_for` as a bare `KeyError` naming a kind this tier's own `Capabilities` declares, which is
            the refusal `Capabilities.require` was added to remove.
        """
        regrouped = with_fields(held.get_layer(BOTTOM), group="observations")
        with pytest.raises(CapabilityError) as refusal:
            held.replace_layer(regrouped)
        assert "custom:holoviews" in str(refusal.value), refusal.value

    def test_the_refusal_says_the_element_is_the_callers_own(self, held):
        """A refusal a caller can act on names the reason, not only the kind.

        Args:
            held: The map under test.

        Test scenario:
            The reason is the tier's own — the one `INTERACTIVE_UNDRAWN_KINDS` records — so a reader who
            hits the refusal learns what to do instead: hand the new object to `add_layer`.
        """
        restyled = with_fields(
            held.get_layer(BOTTOM), symbology=Symbology(props={"size": 9})
        )
        with pytest.raises(CapabilityError) as refusal:
            held.replace_layer(restyled)
        assert "add_layer" in str(refusal.value), refusal.value

    def test_a_refused_replacement_leaves_the_map_as_it_was(self, held):
        """Refused up front means refused before a drawer runs, so nothing is half-applied.

        Args:
            held: The map under test.
        """
        restyled = with_fields(
            held.get_layer(BOTTOM), symbology=Symbology(props={"size": 9})
        )
        with pytest.raises(CapabilityError):
            held.replace_layer(restyled)
        assert (held.layer_ids, held.layers) == (
            [BOTTOM, TOP],
            ["lower-element", "upper-element"],
        ), (held.layer_ids, held.layers)

    def test_the_removed_layer_stops_being_the_one_the_toggles_act_on(self, held):
        """`colorbar()` and `legend()` act on the layer added last; a removed one is not it.

        Args:
            held: The map under test.

        Test scenario:
            The pointer is resolved by looking the id up in the tree, so leaving it on a removed layer made
            the next `colorbar()` answer `ValueError: tuple.index(x): x not in tuple` — review N8, found on
            the `_forget_layer` path and reachable the same way from here.
        """
        held.remove_layer(TOP)
        assert held._last_layer_index("colorbar") == 0, held._last_layer_id


class TestWhatReplaceLayerRefusesAndSaysItRefuses:
    """`replace_layer`'s refusals, and the docstring a caller writes their `except` clause from.

    The class the static tier settled in round 1 (`tests/static/test_static_layer_management.py`), in this
    tier's terms. Review M9 measured four tiers answering one mistake with four exception classes — this one
    said `KeyError: "no layer None on this map"`, about an id the caller never wrote — and three docstrings
    promising a class the code does not raise.
    """

    def test_the_refusal_class_the_docstring_promises_is_the_one_raised(self, held):
        """A `Raises:` block naming the wrong class is an `except` clause the exception walks straight past.

        Args:
            held: The map under test.

        Test scenario:
            The gate became `Capabilities.require`, which raises `CapabilityError`, while the docstring went
            on promising `KeyError` for a kind the tier cannot draw. The two readings are built differently
            on purpose — one by running the call, one by reading what it says about itself.
        """
        replacement = with_fields(held.get_layer(BOTTOM), kind="custom:pyvista")
        with pytest.raises(CapabilityError) as refusal:
            held.replace_layer(replacement)
        raised = type(refusal.value).__name__
        assert raised in (InteractiveMap.replace_layer.__doc__ or ""), (
            f"replace_layer raises {raised} and its docstring does not name it"
        )

    def test_a_replacement_that_is_not_a_spec_is_refused_by_type(self, held):
        """An id passed where the description belongs is the caller's mistake, named as such.

        Args:
            held: The map under test.

        Test scenario:
            `getattr(layer, "id", None)` made the id lookup fail first, so the message named `None` — an id
            the caller never passed — and said nothing about the argument that was actually wrong.
        """
        with pytest.raises(ValueError, match=r"needs a LayerSpec.*got str"):
            held.replace_layer(BOTTOM)

    def test_the_refusal_names_what_was_passed_rather_than_a_missing_id(self, held):
        """The message a caller reads has to point at the argument, not at the layer list.

        Args:
            held: The map under test.
        """
        with pytest.raises(ValueError) as refusal:
            held.replace_layer(42)
        assert "no layer" not in str(refusal.value), (
            f"the refusal blamed a missing layer: {refusal.value}"
        )


class TestHidingACallersOwnElement:
    """Review H8: `set_visible` reported success on a custom layer, reached no engine, and stayed silent.

    The method promises one of two things and did neither. Its `Warns:` clause says a `UserWarning` is
    emitted "when the element has no keyword to say it with … The layer stays drawn and the warning says
    so" — and this was the one path that took neither branch: the element was never asked, so a caller was
    told the layer was hidden while `render()` went on composing it. It is the "reports success while the
    engine was not reached" mode, and it was the only tier with it (static, web and 3-D all hide theirs).
    """

    def test_hiding_it_writes_the_option_onto_the_element(self, kept):
        """A HoloViews element of the caller's own *can* be hidden, so hiding it must reach it.

        Args:
            kept: A map holding one real `Points` element.

        Test scenario:
            `.opts()` writes into HoloViews' global option `Store` against the object it is called on, and
            the map holds that object — so there is nothing standing between `set_visible` and the engine
            except the renderer's record, which never held a custom layer. Read back out of the store, off
            `layers[0]`, which is the element `render()` composes: a fix that dropped the layer instead of
            hiding it would not get as far as the assertion.
        """
        kept.set_visible(BOTTOM, False)
        applied = _bokeh_visible(kept.layers[0])
        assert applied is False, f"the engine's store holds visible={applied!r}"

    def test_showing_it_again_turns_it_back_on(self, kept):
        """The switch has to work in both directions, or a hidden layer can never come back.

        Args:
            kept: A map holding one real `Points` element.
        """
        kept.set_visible(BOTTOM, False)
        kept.set_visible(BOTTOM, True)
        applied = _bokeh_visible(kept.layers[0])
        assert applied is True, f"the engine's store holds visible={applied!r}"

    def test_an_object_with_no_visibility_keyword_warns_instead(self, held):
        """The promise the `Warns:` clause makes, on the path that could not keep the other one.

        Args:
            held: The map under test, holding strings.

        Test scenario:
            A string has no Bokeh option tree, exactly as a `Tiles` basemap and a frameless `DynamicMap`
            have none, so there is no keyword to write. Saying nothing is what left a caller believing a
            layer was hidden.
        """
        with pytest.warns(UserWarning, match="no way to hide"):
            held.set_visible(BOTTOM, False)

    def test_such_an_object_stays_described_hidden_and_drawn(self, held):
        """The warning's own wording — "it stays drawn" — held against the map.

        Args:
            held: The map under test, holding strings.

        Test scenario:
            The description records the request, so a switcher reading the figure still shows the layer
            off; the overlay keeps the object, because nothing can express the request on it. Pinned rather
            than left implied, because the alternative reading is that the tier drops the layer.
        """
        with pytest.warns(UserWarning):
            held.set_visible(BOTTOM, False)
        assert (held.figure_spec.layers.is_visible(BOTTOM), held.layers) == (
            False,
            ["lower-element", "upper-element"],
        ), held.layers
