"""One set of checks every tier's renderer answers to (#305).

Wave 4 landed the first renderer — `three_d/renderer.py` — and its **second** review round found ten
High-severity defects, all of them in the first round's fix code and four of them in the renderer itself.
Three more renderers are about to be written (#296 web, #300 interactive, #303 static) against that same
implementation as a template. Written by hand three more times, those four defects get found three more
times, separately and late.

So they are written down once, here, as a contract a tier signs rather than a set of tests a tier copies.
A tier supplies a :class:`RendererContract` — how to build itself, how to draw one layer, what a figure it
must refuse looks like, and how to read what its engine currently shows — and inherits every check below.

**Adding a tier is a subclass and an adapter, nothing else** — this one is at the foot of this module:

```python
class TestStaticRendererConformance(RendererConformance):
    contract = StaticContract()
```

Three of the four live here. The web tier's pair, `WebContract` and `TestWebRendererConformance`, lives in
`tests/web/test_web_seam.py` instead, because the `test-web` job collects `tests/web` and nothing else — the
same gap the other two tiers had until their jobs collected this module as well (review M6, N6).

The four defects this exists to prevent, each found the expensive way:

1. **`apply` is not atomic, so it must roll back.** It draws layer by layer; a refusal on the second layer
   has already drawn the first. Rolling back only the *description* left an actor on the engine under an id
   no layer owned — and `remove_layer` refuses an id the scene does not have, so nothing could ever reach
   it again.
2. **A registry of engine objects needs a lifecycle.** `base.registry` holds strong references. A tier that
   registers a caller's data and never forgets it leaks every layer of every figure for the life of the
   process — and namespacing made that *worse*, by stopping the entries from colliding.
3. **The redraw guard has to cover the right collection.** A change to `label` alone must not reach the
   engine; a change to a layer's *data* always must. Shipped over the wrong one, a layer pointing at new
   data kept its old mesh while `figure_spec` advertised the new one.
4. **A renderer object must not be probed with `hasattr`.** PyVista's `camera` property *resets the camera*
   when nothing has set one. Asked from the lazy plotter getter it framed an empty renderer, and every 3-D
   notebook rendered blank — past both review rounds, caught only by CI.

A fifth check comes from the same round: the drawer table and the tier's `Capabilities` are one list said
twice, and drift either way is a defect — bar the kinds a tier names in `UNDRAWN_KINDS`, each with the
reason it declares a kind it has no drawer for.

A sixth comes from round 2 of the web seam, and is the first found *because* four tiers were written against
this contract rather than in spite of it. **A layer a figure describes hidden must be drawn hidden.** Asked
of the four `draw_layer`s, that question had three answers: static and 3-D read `figure.layers.is_visible`,
which is the layer's own flag *and* its group not being hidden; web read `layer.visible`, the flag alone, so
a layer hidden by its group redrew visible; interactive read neither, so any hidden layer redrew visible.
Nothing here asked — `drawn_is_in_view` is about the camera — and only the web tier had a local check, over
the one of the three readings that was wrong. The check is asked in both forms, by the flag and by the
group, and a third check pins that a layer described *visible* is not drawn hidden, so a tier cannot pass
the first two by hiding everything.

**Not every tier's `apply` reaches what the tier draws.** On the web and interactive tiers `Renderer.apply`
updates the renderer's own record and nothing the tier renders from — the widget is built from the queue,
the overlay from `layers` — and on none of the 2-D tiers does it move the figure the tier reports. That is
deliberate for now; wiring it through is later work (review M1). A check that reads what `apply` never
touches passes whatever `apply` did, so each adapter declares what its `apply` reaches, and the checks that
read the engine or the description run only where it does. The rollback and the redraw guard are also held
to the one thing every tier's `apply` does change, the renderer's own record, so no tier goes unchecked.
"""

import importlib.util
from dataclasses import replace as with_fields

import pytest

from digitalearth.base.registry import _OBJECTS
from digitalearth.base.spec import LayerSpec

#: The group the group-hidden check files a layer under. Named after this contract so it cannot collide with
#: a group a tier put a layer in itself.
_HIDDEN_GROUP = "conformance-hidden-group"


def _hidden_by_its_own_flag(figure, layer_id: str):
    """Return `figure` with one layer's own `visible` flag turned off.

    Args:
        figure: The figure to change.
        layer_id: The layer to hide.

    Returns:
        The figure, with that layer described hidden and nothing else changed.
    """
    return with_fields(figure, layers=figure.layers.set_visible(layer_id, False))


def _hidden_by_its_group(figure, layer_id: str):
    """Return `figure` with one layer put in a group, and that group hidden.

    The layer's own flag stays `True`, which is the point: `LayerTree.is_visible` is the only reading that
    answers both, and a tier reading `layer.visible` alone draws this one visible.

    Args:
        figure: The figure to change.
        layer_id: The layer to hide by its group.

    Returns:
        The figure, with that layer in a hidden group.
    """
    grouped = with_fields(figure.layers.get(layer_id), group=_HIDDEN_GROUP)
    tree = figure.layers.replace(grouped).set_group_visible(_HIDDEN_GROUP, False)
    return with_fields(figure, layers=tree)


def _identities(record) -> dict:
    """Return layer id to the identity of what a renderer recorded for it.

    A record's values compare by *value* on three tiers — a frozen dataclass of what was drawn — so a layer
    drawn again from the same description would compare equal to the one it replaced. Identity is what a
    redraw changes.

    Args:
        record: A renderer's record, as `RendererContract.renderer_record` returns it. The caller keeps it
            referenced while comparing, so none of its values can be freed and its id handed on.

    Returns:
        Layer id to `id()` of the recorded value.
    """
    return {layer_id: id(value) for layer_id, value in record.items()}


#: The reason a declared kind is drawn but not from a description: its builder queues the drawing itself.
_QUEUED = "still drawn through the queue"

#: The kinds each tier declares but has no drawer for, each with the reason that is allowed.
#:
#: The drift guard lets `declared - drawn` through only by these names, and only while they are true: an
#: entry that is drawn now, or no longer declared, fails it as surely as an unnamed gap does. So each list is
#: exactly what is deferred today, and a kind converted to its drawer comes off here in the same change. They
#: live here rather than on each adapter so the web adapter, which sits beside the web tier's own tests, is
#: held to the same list.
#:
#: **The web tier's four kinds are named in a second place**, as `DRAWN_BUT_NOT_DESCRIBED` in
#: `tests/web/test_web_capabilities.py`, which holds a builder call per kind where this holds the reason —
#: two payloads over one set, because neither job collects the other's module (review M6). They cannot drift
#: silently: a kind that starts recording a layer fails the guard below until it is taken off here, and fails
#: the capability test there until it is taken off that list, so the two are corrected together or not at all.
#: A reader changing either has to know both exist, which is what this note is for (review N6).
THREE_D_UNDRAWN_KINDS: dict[str, str] = {}
WEB_UNDRAWN_KINDS: dict[str, str] = {
    "basemap": f"{_QUEUED}: `tiles` queues an underlay and records no layer to draw it from",
    "terrain": f"{_QUEUED}: `terrain_tiles` queues `set_terrain` and records no layer",
    "point_cloud": f"{_QUEUED}: a deck.gl layer, added in one queued `add_deck_layers` call, records no layer",
    "model": f"{_QUEUED}: `gltf` adds a deck.gl layer the way `point_cloud` does, and records no layer",
}
INTERACTIVE_UNDRAWN_KINDS: dict[str, str] = {
    "custom:holoviews": (
        "a caller's own element, drawn by being kept: `add_element` holds no description to rebuild it from"
    ),
}
STATIC_UNDRAWN_KINDS: dict[str, str] = {}

#: Each tier's list, by the name its `Capabilities` gives it.
UNDRAWN_KINDS: dict[str, dict[str, str]] = {
    "3d": THREE_D_UNDRAWN_KINDS,
    "web": WEB_UNDRAWN_KINDS,
    "interactive": INTERACTIVE_UNDRAWN_KINDS,
    "matplotlib": STATIC_UNDRAWN_KINDS,
}


class RendererContract:
    """What a tier supplies so the shared checks can run against its renderer.

    A tier implements this once. Every method may raise `NotImplementedError` only if the tier genuinely
    has no such concept — the check that needs it then skips, loudly, rather than passing.

    Attributes:
        backend: The tier's name as its `Capabilities` spells it.
        apply_reaches_engine: Whether `apply_figure` changes what `engine_holds` reads. Off by default, so a
            tier has to *say* its engine follows `apply` before a check may rely on it: where it does not,
            reading the engine after `apply` could not fail, and the checks that do skip rather than pass.
        apply_reaches_description: Whether `apply_figure` moves the figure the tier reports along with what
            it draws. Only an adapter that goes through the tier's own change path can; one that calls the
            renderer directly never touches the description, so comparing it could not fail either.
    """

    backend: str = ""
    apply_reaches_engine: bool = False
    apply_reaches_description: bool = False

    def make(self):
        """Return an open tier object — a scene, a map — with nothing drawn on it yet.

        Returns:
            The tier object.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def draw_one(self, tier) -> str:
        """Draw one layer that the renderer really reaches the engine for.

        Args:
            tier: The object `make` returned.

        Returns:
            The drawn layer's id.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def refused_figure(self, tier):
        """Return a figure the renderer must refuse *after* drawing part of it.

        The point is a partial application: the figure holds something drawable followed by something the
        tier does not draw, so `apply` gets far enough to touch the engine before it fails.

        Args:
            tier: The object `make` returned.

        Returns:
            The figure.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def apply_figure(self, tier, figure) -> None:
        """Ask the tier to move to `figure`, letting any refusal propagate.

        Args:
            tier: The object `make` returned.
            figure: The figure to move to.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def engine_holds(self, tier):
        """Return what the engine currently shows, as something comparable across two readings.

        The engine's own objects, not the renderer's record of them: a rollback that restored the record and
        left an object on the engine reads clean from the record, and that is the first defect above. The
        objects themselves rather than their ids, too. Every engine object here compares by identity, and a
        snapshot holding them stops a freed object's address being reused by the one that replaced it —
        which would make a redraw read as no change.

        Args:
            tier: The object `make` returned.

        Returns:
            A snapshot of the engine's objects, equal to another only when it holds the same ones.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def renderer_record(self, tier) -> dict:
        """Return the renderer's own record of what it drew, by layer id.

        Every tier's renderer keeps one: it is what `apply` reconciles and what a rollback restores. So it
        is the one reading a refused or relabelled `apply` can change on every tier, whether or not the
        engine follows.

        Args:
            tier: The object `make` returned.

        Returns:
            Layer id to what the renderer recorded for it.
        """
        return dict(tier._renderer.drawn)

    def relabel(self, tier, layer_id: str):
        """Return a figure identical to the tier's, with one layer's `label` changed and nothing else.

        Args:
            tier: The object `make` returned.
            layer_id: The layer to relabel.

        Returns:
            The figure.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def draw_from(self, tier, figure, layer_id: str) -> None:
        """Draw one of `figure`'s layers through the tier's renderer, the way a redraw reaches it.

        Every tier's renderer is asked the same way — `draw_layer(figure, layer_id)` — which is the single
        entry point a figure is drawn back through, so no adapter has to supply this.

        Args:
            tier: The object `make` returned.
            figure: The figure to draw from: the tier's own, or one changed from it.
            layer_id: The layer to draw.
        """
        tier._renderer.draw_layer(figure, layer_id)

    def drawn_is_hidden(self, tier, layer_id: str) -> bool:
        """Whether the engine is currently *not* drawing what the renderer holds for a layer.

        Read from the engine rather than from the description, through the reader every renderer has:
        `set_visible` writes it and `is_visible` reads it back, each tier in its own terms — a matplotlib
        artist's flag, a VTK actor's visibility, MapLibre's `layout.visibility`, the HoloViews option the
        element was drawn with. A tier that cannot answer has no way to honour a hidden layer either, so
        this is deliberately not optional.

        Args:
            tier: The object `make` returned.
            layer_id: The layer to ask about.

        Returns:
            `True` when the layer was drawn hidden.

        Raises:
            KeyError: when nothing has been drawn for that layer, which is a defect in the check rather
                than an answer.
        """
        return not tier._renderer.is_visible(layer_id)

    def drawn_is_in_view(self, tier):
        """Whether what has been drawn is actually inside the view the engine will render.

        This is the portable form of "do not probe an engine object with `hasattr`". The defect was not
        that a flag changed; it was that a *question* about the view moved the view, so the data ended up
        outside it. Each tier answers in its own terms — a frame that differs from its background, a
        viewport that contains the layer's bounds.

        Args:
            tier: The object `make` returned.

        Returns:
            `True` / `False`, or `None` when the tier has no view to be outside of.
        """
        return None

    def close(self, tier) -> None:
        """Release the tier and whatever it registered.

        Args:
            tier: The object `make` returned.
        """
        tier.close()

    def declared_kinds(self) -> frozenset:
        """Return the kinds the tier's `Capabilities` declares.

        Returns:
            The declared kinds.

        Raises:
            NotImplementedError: when the tier has not supplied them.
        """
        raise NotImplementedError

    def drawn_kinds(self) -> tuple:
        """Return the kinds the renderer's own drawer table claims.

        Returns:
            The drawable kinds.

        Raises:
            NotImplementedError: when the tier has not supplied them.
        """
        raise NotImplementedError

    def drawer_for(self, kind: str):
        """Return the drawer registered for `kind`, raising when there is none.

        Args:
            kind: The layer kind.

        Returns:
            The drawer.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def undrawn_kinds(self) -> dict[str, str]:
        """Return the kinds this tier declares but has no drawer for, each with the reason.

        Returns:
            Kind to reason, from :data:`UNDRAWN_KINDS` by `backend`. A tier with no entry there gets none,
            so every kind it declares must be drawn.
        """
        return UNDRAWN_KINDS.get(self.backend, {})


class RendererConformance:
    """The checks themselves. A tier inherits these by setting `contract`."""

    #: The tier's adapter. A subclass sets this.
    contract: RendererContract = RendererContract()

    @pytest.fixture
    def tier(self):
        """Yield an open tier object, closed on the way out.

        Yields:
            The tier object.
        """
        built = self.contract.make()
        yield built
        try:
            self.contract.close(built)
        except Exception:  # pragma: no cover - a closed tier may refuse a second close
            pass

    def _skip_unless(self, reaches: bool, what: str) -> None:
        """Skip, saying why, when this tier's `apply` does not reach what a check reads.

        Args:
            reaches: The contract's own declaration.
            what: What the check reads, for the message.
        """
        if not reaches:
            pytest.skip(
                f"`apply` on the {self.contract.backend} tier does not reach {what}, so reading it after "
                "`apply` would pass whatever `apply` did"
            )

    def _skip_unless_apply_reaches_the_engine(self) -> None:
        """Skip a check that reads the engine on a tier whose `apply` does not change it."""
        self._skip_unless(self.contract.apply_reaches_engine, "its engine")

    def test_a_refused_figure_leaves_the_engine_as_it_found_it(self, tier):
        """`apply` draws layer by layer, so a refusal can come after something was already drawn.

        Args:
            tier: The tier under test.

        Test scenario:
            Rolling back only the description left an actor on the engine under an id no layer owned, and
            `remove_layer` refuses an id the tier does not have — so there was no way to reach it again.
            The description and the engine have to roll back together.
        """
        self._skip_unless_apply_reaches_the_engine()
        self.contract.draw_one(tier)
        before_engine = self.contract.engine_holds(tier)
        refused = self.contract.refused_figure(tier)
        with pytest.raises((KeyError, ValueError)):
            self.contract.apply_figure(tier, refused)
        assert self.contract.engine_holds(tier) == before_engine, (
            "a refused change left something behind on the engine: "
            f"{self.contract.engine_holds(tier)} vs {before_engine}"
        )

    def test_a_refused_figure_leaves_the_renderers_record_as_it_found_it(self, tier):
        """The record rolls back with the engine, and it is the one thing every tier's `apply` changes.

        Args:
            tier: The tier under test.

        Test scenario:
            On the web and interactive tiers `apply` reaches nothing but this record (review M1), so this is
            the check a missing rollback fails on all four tiers. Identities rather than keys alone: a
            rollback that re-drew a layer it had no reason to touch keeps the key and changes the value.
        """
        self.contract.draw_one(tier)
        held = self.contract.renderer_record(tier)
        refused = self.contract.refused_figure(tier)
        with pytest.raises((KeyError, ValueError)):
            self.contract.apply_figure(tier, refused)
        kept = self.contract.renderer_record(tier)
        # `held` is still referenced here, so none of its values has been freed and no id reused.
        assert _identities(kept) == _identities(held), (
            f"a refused change left the renderer recording {sorted(kept)}; it held {sorted(held)}"
        )

    def test_a_refused_figure_is_not_the_one_the_tier_reports(self, tier):
        """The description must not advertise a figure the engine never drew.

        Args:
            tier: The tier under test.

        Test scenario:
            The 3-D scene once installed a figure before drawing it and kept it when `apply` raised, so
            `figure_spec` named a layer that was never drawn and could not be. The check runs only where
            `apply_figure` goes through the tier's change path: an adapter that calls the renderer directly
            never touches `figure_spec`, and comparing it passed on three tiers with the rollback deleted.
        """
        self._skip_unless(
            self.contract.apply_reaches_description, "the figure it reports"
        )
        self.contract.draw_one(tier)
        held = tier.figure_spec
        refused_figure = self.contract.refused_figure(tier)
        with pytest.raises((KeyError, ValueError)):
            self.contract.apply_figure(tier, refused_figure)
        assert tier.figure_spec == held, "the tier kept a figure it could not draw"

    def test_closing_the_tier_lets_its_registered_data_go(self, tier):
        """The registry holds strong references, so a tier that never forgets leaks every layer it drew.

        Args:
            tier: The tier under test.

        Test scenario:
            Namespacing the entries stopped them colliding, which made this *worse*: re-running a notebook
            cell left every previous figure's data alive for the life of the process.
        """
        before = set(_OBJECTS)
        self.contract.draw_one(tier)
        registered = set(_OBJECTS) - before
        assert registered, (
            "this tier registered nothing, so the check below would pass vacuously; either the tier does "
            "not use the object registry, or `draw_one` does not reach the path that does"
        )
        self.contract.close(tier)
        left = set(_OBJECTS) & registered
        assert left == set(), (
            f"closing the tier left {sorted(left)} in the object registry"
        )

    def test_a_label_is_not_a_reason_to_redraw(self, tier):
        """`diff` groups a label change with a colormap change; only one of them reaches the engine.

        Args:
            tier: The tier under test.
        """
        self._skip_unless_apply_reaches_the_engine()
        layer_id = self.contract.draw_one(tier)
        before = self.contract.engine_holds(tier)
        self.contract.apply_figure(tier, self.contract.relabel(tier, layer_id))
        assert self.contract.engine_holds(tier) == before, (
            "renaming a layer redrew it; only what the engine draws should trigger a redraw"
        )

    def test_a_label_is_not_a_reason_to_redraw_the_renderers_record(self, tier):
        """The redraw guard sits in the renderer, so the record shows a wrong redraw on every tier.

        Args:
            tier: The tier under test.

        Test scenario:
            A redraw replaces what the renderer recorded for a layer under the same id. Where `apply` does
            not reach the engine, the engine check above skips, and this is what still holds the guard.
        """
        layer_id = self.contract.draw_one(tier)
        before = self.contract.renderer_record(tier)
        self.contract.apply_figure(tier, self.contract.relabel(tier, layer_id))
        after = self.contract.renderer_record(tier)
        assert _identities(after) == _identities(before), (
            f"renaming {layer_id!r} replaced what the renderer recorded for it; only what the engine draws "
            "should trigger a redraw"
        )

    def test_what_was_drawn_is_inside_the_view(self, tier):
        """A renderer must not *ask* an engine object a question that changes the answer.

        Args:
            tier: The tier under test.

        Test scenario:
            PyVista's `camera` property resets the camera when nothing has set one, so probing it with
            `hasattr` from the lazy plotter getter framed a renderer holding no actors — and every layer
            drawn afterwards sat outside the view. Nothing in the tier's own suite noticed, because its
            test scenes are drawn around the origin where an empty-scene camera catches them anyway; only
            the notebooks, which use a projected raster, rendered blank.
        """
        self.contract.draw_one(tier)
        visible = self.contract.drawn_is_in_view(tier)
        if visible is None:
            pytest.skip(
                f"the {self.contract.backend} tier has no view to be outside of"
            )
        assert visible is True, (
            "the drawn layer is outside the view the engine will render"
        )

    def test_the_drawer_table_and_the_declaration_differ_only_by_the_named_undrawn_kinds(
        self,
    ):
        """Drift either way is a defect: a false refusal, or a bare `KeyError` where a message belongs.

        Test scenario:
            A kind drawn but not declared is refused by the declaration before its drawer is asked. A kind
            declared but not drawn reaches `drawer_for` and is refused there. That is right only when the
            tier means it: a kind still drawn through the queue, or a caller's own object with nothing to
            rebuild. Those are named per tier in :data:`UNDRAWN_KINDS`, with the reason, and nothing else
            may be missing. This used to check the first direction only (review L1). A named kind that is
            drawn now, or no longer declared, fails too, so the list cannot outlive what it excuses.
        """
        declared = set(self.contract.declared_kinds())
        drawn = set(self.contract.drawn_kinds())
        named = set(self.contract.undrawn_kinds())
        undeclared = sorted(drawn - declared)
        assert undeclared == [], f"the tier draws {undeclared} without declaring them"
        unexplained = sorted(declared - drawn - named)
        assert unexplained == [], (
            f"the tier declares {unexplained} but has no drawer for them; draw them, or name each in the "
            f"{self.contract.backend!r} entry of UNDRAWN_KINDS with the reason"
        )
        stale = sorted(named - (declared - drawn))
        assert stale == [], (
            f"UNDRAWN_KINDS names {stale} for the {self.contract.backend!r} tier, which it now draws or no "
            "longer declares; take them off the list"
        )

    def _redraw_hidden(self, tier, hide) -> str:
        """Draw one layer, then draw it again from a figure that describes it hidden.

        Args:
            tier: The tier under test.
            hide: What makes the figure describe the layer hidden — its own flag, or its group.

        Returns:
            The layer's id.
        """
        layer_id = self.contract.draw_one(tier)
        self.contract.draw_from(tier, hide(tier.figure_spec, layer_id), layer_id)
        return layer_id

    def test_a_layer_described_hidden_is_drawn_hidden(self, tier):
        """A figure says whether a layer is drawn; drawing it back must not turn it on.

        Args:
            tier: The tier under test.

        Test scenario:
            Reachable the moment a figure is read back or reconciled, which is what this seam is for. The
            tier's builders never ask for a hidden layer, so three of the four drawer sets built their
            layers visible whatever the description said (review M4).
        """
        layer_id = self._redraw_hidden(tier, _hidden_by_its_own_flag)
        assert self.contract.drawn_is_hidden(tier, layer_id) is True, (
            f"{layer_id!r} is described hidden and the {self.contract.backend} tier drew it visible"
        )

    def test_a_layer_hidden_by_its_group_is_drawn_hidden(self, tier):
        """A group hides its layers without touching their own flags, and only one reading sees that.

        Args:
            tier: The tier under test.

        Test scenario:
            `LayerTree.is_visible` is the layer's flag **and** its group's; `layer.visible` is the flag
            alone. A tier asking the second draws a layer hidden by its group — a switcher's whole
            "Observations" group turned off — as if nothing had been hidden at all.
        """
        layer_id = self._redraw_hidden(tier, _hidden_by_its_group)
        assert self.contract.drawn_is_hidden(tier, layer_id) is True, (
            f"{layer_id!r} is in a hidden group and the {self.contract.backend} tier drew it visible"
        )

    def test_a_layer_described_visible_is_not_drawn_hidden(self, tier):
        """The two checks above must not be passable by drawing everything hidden.

        Args:
            tier: The tier under test.
        """
        layer_id = self.contract.draw_one(tier)
        assert self.contract.drawn_is_hidden(tier, layer_id) is False, (
            f"{layer_id!r} is described visible and the {self.contract.backend} tier drew it hidden"
        )

    def test_every_drawable_kind_resolves_to_a_drawer(self):
        """A kind on the list with no drawer raises a bare `KeyError` far from the cause."""
        unresolved = []
        for kind in self.contract.drawn_kinds():
            try:
                self.contract.drawer_for(kind)
            except KeyError:
                unresolved.append(kind)
        assert unresolved == [], f"{unresolved} are declared drawable with no drawer"


class ThreeDContract(RendererContract):
    """The 3-D tier's adapter — the reference implementation every other seam copies.

    The one tier whose `apply` reaches both halves: `apply_figure` goes through the scene's `_change`, which
    moves the plotter and the figure the scene reports together, and rolls both back together.
    """

    backend = "3d"
    apply_reaches_engine = True
    apply_reaches_description = True

    def make(self):
        """Return an off-screen scene.

        Returns:
            The scene.
        """
        from digitalearth.three_d import Scene3D

        return Scene3D(off_screen=True)

    def draw_one(self, tier) -> str:
        """Draw a terrain layer far from the origin, which reaches PyVista.

        The coordinates matter. A grid drawn around zero is inside the default camera either way, which is
        exactly why the tier's own suite missed the blank-render defect; this one is placed where only a
        camera that framed the *data* will catch it.

        Args:
            tier: The scene.

        Returns:
            The layer's id.
        """
        import numpy as np

        from digitalearth.base.sources import get_source

        grid = np.add.outer(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 8))
        source = get_source(
            grid,
            x=np.linspace(400000.0, 430000.0, 8),
            y=np.linspace(5000000.0, 5020000.0, 6),
            crs=3857,
        )
        tier.terrain(source)
        return tier.layer_ids[-1]

    def refused_figure(self, tier):
        """Return a figure whose second layer names a kind the tier does not draw.

        Args:
            tier: The scene.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        figure = tier.figure_spec
        drawable = figure.layers.get(tier.layer_ids[-1])
        # Two layers, drawable first: `apply` has to get far enough to touch the engine before it fails,
        # or there is nothing for a rollback to undo and the check cannot tell the two implementations
        # apart.
        tree = figure.layers.add(
            LayerSpec("second", "terrain", source_id=drawable.source_id)
        ).add(LayerSpec("refused", "choropleth", source_id=drawable.source_id))
        return with_fields(figure, layers=tree)

    def apply_figure(self, tier, figure) -> None:
        """Move the scene to `figure` through the path every change goes through.

        Args:
            tier: The scene.
            figure: The figure.
        """
        tier._change(figure)

    def engine_holds(self, tier):
        """Return every actor on the plotter, by the name PyVista filed it under.

        The plotter, not the renderer's `drawn`: the defect this contract was written from was an actor
        left on the plotter under an id no layer owned, which a record rolled back on its own does not
        show. The actors themselves, not layer ids: a redraw removes a layer and adds it again under the
        same id — which is how a redraw guard over the wrong collection went unnoticed — but the actor it
        adds is a different object.

        Args:
            tier: The scene.

        Returns:
            Actor name to actor, for everything the plotter renders — a layer's mesh and its scalar bar.
        """
        return dict(tier.plotter.renderer.actors)

    def relabel(self, tier, layer_id: str):
        """Return the scene's figure with one layer's label changed.

        Args:
            tier: The scene.
            layer_id: The layer to relabel.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        figure = tier.figure_spec
        layer = figure.layers.get(layer_id)
        renamed = with_fields(layer, label="a different name")
        return with_fields(figure, layers=figure.layers.replace(renamed))

    def drawn_is_in_view(self, tier):
        """Whether the rendered frame shows the terrain rather than an empty background.

        Args:
            tier: The scene.

        Returns:
            `True` when enough of the frame differs from its own background colour.
        """
        import numpy as np

        frame = tier.screenshot()
        background = frame[0, 0].astype(int)
        # The colour bar is furniture and is drawn in the bottom strip whether or not the terrain is in
        # shot, so it is cropped away; what is left is the scene itself.
        scene_area = frame[: int(frame.shape[0] * 0.8)]
        lit = (np.abs(scene_area.astype(int) - background).sum(-1) > 30).mean()
        return bool(lit > 0.01)

    def declared_kinds(self) -> frozenset:
        """Return the tier's declared kinds.

        Returns:
            The declared kinds.
        """
        from digitalearth.three_d.capabilities import CAPABILITIES

        return CAPABILITIES.kinds

    def drawn_kinds(self) -> tuple:
        """Return the renderer's own kind list.

        Returns:
            The drawable kinds.
        """
        from digitalearth.three_d.renderer import DRAWN_KINDS

        return DRAWN_KINDS

    def drawer_for(self, kind: str):
        """Return the drawer registered for `kind`.

        Args:
            kind: The layer kind.

        Returns:
            The drawer.
        """
        from digitalearth.three_d.renderer import drawer_for

        return drawer_for(kind)


#: Skipping is per tier, not per module. A module-level `importorskip` would skip the contract classes
#: themselves, so a web or interactive subclass importing them from here would be skipped too — in an
#: environment where that tier is perfectly installed.
needs_pyvista = pytest.mark.skipif(
    importlib.util.find_spec("pyvista") is None,
    reason="the 3-D tier needs the viz3d environment",
)


@needs_pyvista
class TestThreeDRendererConformance(RendererConformance):
    """The 3-D tier, which is the implementation the contract was written from."""

    contract = ThreeDContract()


class InteractiveContract(RendererContract):
    """The interactive tier's adapter for the shared contract (#300).

    **`apply` does not reach this tier's engine.** What the tier renders is the overlay `render()` composes
    from `InteractiveMap.layers`, and only the builders write that list. `Renderer.apply` builds elements
    into its own record and stops there, so after a remove and an add, `render()` still overlays what the
    builders put down (review M1). That stays so for this wave; wiring `apply` into the overlay is later
    work. Until then `apply_reaches_engine` is `False`, and the engine checks skip here, because they would
    pass whatever `apply` did. The rollback and the redraw guard are held to the renderer's record instead,
    the one thing `apply` does change here, and those checks run on this tier as on every other.

    `apply_reaches_description` stays `False` too: `apply_figure` calls the renderer, and the renderer never
    touches the map's `figure_spec`.
    """

    backend = "interactive"

    def make(self):
        """Return an empty map.

        Returns:
            The map. Constructing one touches no engine; the builders lazy-import HoloViz.
        """
        from digitalearth.interactive import InteractiveMap

        return InteractiveMap()

    def draw_one(self, tier) -> str:
        """Draw one point layer, which reaches HoloViews and registers its data.

        The collection is a pyramids `FeatureCollection` in EPSG:4326 rather than a bare GeoDataFrame, so
        the builder really goes through the reproject-to-display-CRS path — and so the object registry
        really holds it, which is what the lifecycle check below needs.

        Args:
            tier: The map.

        Returns:
            The drawn layer's id.
        """
        import geopandas as gpd
        from pyramids.feature import FeatureCollection
        from shapely.geometry import Point

        features = FeatureCollection(
            gpd.GeoDataFrame(
                {"value": [1.0, 2.0]},
                geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
                crs=4326,
            )
        )
        tier.points(features)
        return tier.layer_ids[-1]

    def refused_figure(self, tier):
        """Return a figure whose second new layer names a kind this tier does not draw.

        Args:
            tier: The map.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        figure = tier.figure_spec
        drawable = figure.layers.get(tier.layer_ids[-1])
        # The first of the two added layers must really draw, or the refusal is reached with nothing
        # drawn and the rollback check passes for the wrong reason. It copies the drawn layer's
        # symbology rather than inventing one, so its recipe is one this tier has.
        tree = figure.layers.add(with_fields(drawable, id="second")).add(
            LayerSpec("refused", "terrain", source_id=drawable.source_id)
        )
        return with_fields(figure, layers=tree)

    def apply_figure(self, tier, figure) -> None:
        """Move the map to `figure` through the renderer.

        Args:
            tier: The map.
            figure: The figure.
        """
        tier._renderer.apply(tier.figure_spec, figure)

    def engine_holds(self, tier):
        """Return the elements `render()` overlays, bottom first.

        No check reads this while `apply_reaches_engine` is `False` — `apply` never changes this list, which
        is the point of the class docstring. It is still the honest answer to "what does the engine hold",
        and it is what the engine checks will read once `apply` is wired into the overlay and the flag is
        turned on. HoloViews elements compare by identity, so a redraw reads as a change.

        Args:
            tier: The map.

        Returns:
            The elements, in the order `render()` overlays them.
        """
        return tuple(tier.layers)

    def relabel(self, tier, layer_id: str):
        """Return the map's figure with one layer's label changed and nothing else.

        Args:
            tier: The map.
            layer_id: The layer to relabel.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        figure = tier.figure_spec
        renamed = with_fields(figure.layers.get(layer_id), label="a different name")
        return with_fields(figure, layers=figure.layers.replace(renamed))

    def drawn_is_in_view(self, tier):
        """The map is framed by what the viewer does, not by a region set before it is drawn.

        Args:
            tier: The map.

        Returns:
            `None` — Bokeh ranges on the data it is handed, and the tier declares `domain` absent for
            exactly this reason, so there is no view a drawn layer can fall outside of.
        """
        return None

    def close(self, tier) -> None:
        """Let the map go of the data it registered.

        Args:
            tier: The map.
        """
        tier.close()

    def declared_kinds(self) -> frozenset:
        """Return the kinds the tier declares.

        Returns:
            The declared kinds.
        """
        from digitalearth.interactive.capabilities import CAPABILITIES

        return CAPABILITIES.kinds

    def drawn_kinds(self) -> tuple:
        """Return the kinds the renderer draws from a description.

        Returns:
            The drawable kinds.
        """
        from digitalearth.interactive.renderer import DRAWN_KINDS

        return DRAWN_KINDS

    def drawer_for(self, kind: str):
        """Return the drawer registered for `kind`.

        Args:
            kind: The layer kind.

        Returns:
            The drawer.
        """
        from digitalearth.interactive.renderer import drawer_for

        return drawer_for(kind)


needs_geoviews = pytest.mark.skipif(
    importlib.util.find_spec("geoviews") is None,
    reason="the interactive tier needs the interactive environment",
)


@needs_geoviews
class TestInteractiveRendererConformance(RendererConformance):
    """The interactive tier, signing the contract the 3-D and web tiers already pass."""

    contract = InteractiveContract()


class StaticContract(RendererContract):
    """The matplotlib tier's adapter for the shared contract (#303).

    The last tier to sign it, and the one closest in shape to the 3-D reference: matplotlib hands out live
    artists on a live axes, and `Renderer.apply` draws onto and takes off that axes, so the engine follows
    `apply` and `engine_holds` reads the axes themselves. The figure the map reports does not follow it:
    `apply_figure` calls the renderer, which never touches the scene's `figure_spec` (review M1), so
    `apply_reaches_description` stays `False`.
    """

    backend = "matplotlib"
    apply_reaches_engine = True

    def make(self):
        """Return an empty map in Web Mercator.

        Returns:
            The map. It owns its own figure, which `close` shuts.
        """
        from digitalearth.static import Map

        return Map(crs=3857)

    def draw_one(self, tier) -> str:
        """Draw one raster layer far from the origin, which reaches matplotlib and registers its data.

        The coordinates matter for the same reason they do in the 3-D adapter: a grid drawn around zero is
        inside a default view either way, and the point of `drawn_is_in_view` is to catch a view that was
        framed before the data arrived.

        Args:
            tier: The map.

        Returns:
            The layer's id.
        """
        import numpy as np
        from pyramids.dataset import Dataset, GeoReference

        grid = np.add.outer(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 8))
        dataset = Dataset.from_array(
            arr=grid.astype("float32"),
            # (x origin, cell width, 0, y origin, 0, -cell height) — a 240 x 120 km patch of Web Mercator
            # a long way from (0, 0), so only a view fitted to the data contains it.
            geo_ref=GeoReference(
                geo=(400000.0, 30000.0, 0.0, 5020000.0, 0.0, -20000.0), epsg=3857
            ),
        )
        tier.imshow(dataset)
        return tier.layer_ids[-1]

    def refused_figure(self, tier):
        """Return a figure whose second new layer names a kind this tier does not draw.

        Args:
            tier: The map.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Symbology

        figure = tier.figure_spec
        # One of the two added layers must really draw, or the refusal is reached with nothing drawn and
        # the rollback check passes for the wrong reason. Two things decide that it does. A text label
        # rather than a second raster, because cleopatra's glyphs replace what is already on the axes
        # unless a render opts into composing — a second raster would take the drawn one's image off. And
        # the undrawable layer is put in the label's own band, because `LayerTree.add` files a layer by
        # band and `added` follows the tree: a `terrain` layer belongs among the data, so it would
        # otherwise come first and be refused before the label is ever drawn.
        label = LayerSpec(
            "also-drawn",
            "text",
            symbology=Symbology(
                props={"via": "text", "lon": 0.0, "lat": 0.0, "s": "here", "crs": 4326}
            ),
        )
        tree = figure.layers.add(label).add(
            LayerSpec("refused", "terrain", band="overlay")
        )
        return with_fields(figure, layers=tree)

    def apply_figure(self, tier, figure) -> None:
        """Move the map to `figure` through the renderer.

        Args:
            tier: The map.
            figure: The figure.
        """
        tier._renderer.apply(tier.figure_spec, figure)

    def engine_holds(self, tier):
        """Return every artist on every axes of the map's figure, in the order matplotlib holds them.

        The axes, not the renderer's `drawn`: a rollback that restored the record and left an artist on the
        axes reads clean from the record, which is the 3-D tier's first defect in this tier's terms. Every
        axes, so a layer that adds one of its own — a colorbar — is seen as well. matplotlib artists
        compare by identity, so a redraw, a new artist for the same layer, reads as a change.

        Args:
            tier: The map.

        Returns:
            The artists.
        """
        return tuple(child for axes in tier.fig.axes for child in axes.get_children())

    def relabel(self, tier, layer_id: str):
        """Return the map's figure with one layer's label changed and nothing else.

        Args:
            tier: The map.
            layer_id: The layer to relabel.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        figure = tier.figure_spec
        renamed = with_fields(figure.layers.get(layer_id), label="a different name")
        return with_fields(figure, layers=figure.layers.replace(renamed))

    def drawn_is_in_view(self, tier):
        """Whether the drawn raster overlaps the rectangle the axes will render.

        Args:
            tier: The map.

        Returns:
            `True` when the image's extent and the axes limits intersect. matplotlib autoscales to the
            data, so the answer is only `False` if something framed the axes without looking at what was
            drawn — which is this tier's form of the blank 3-D render.
        """
        drawn = tier._renderer.drawn[tier.layer_ids[-1]]
        left, right, bottom, top = drawn.artist.get_extent()
        xlim, ylim = tier.ax.get_xlim(), tier.ax.get_ylim()
        across = min(right, max(xlim)) > max(left, min(xlim))
        down = min(top, max(ylim)) > max(bottom, min(ylim))
        return bool(across and down)

    def declared_kinds(self) -> frozenset:
        """Return the kinds the tier declares.

        Returns:
            The declared kinds.
        """
        from digitalearth.static.capabilities import CAPABILITIES

        return CAPABILITIES.kinds

    def drawn_kinds(self) -> tuple:
        """Return the kinds the renderer draws from a description.

        Returns:
            The drawable kinds.
        """
        from digitalearth.static.renderer import DRAWN_KINDS

        return DRAWN_KINDS

    def drawer_for(self, kind: str):
        """Return the drawer registered for `kind`.

        Args:
            kind: The layer kind.

        Returns:
            The drawer.
        """
        from digitalearth.static.renderer import drawer_for

        return drawer_for(kind)


class TestStaticRendererConformance(RendererConformance):
    """The matplotlib tier, signing the contract the other three already pass.

    No skip guard: matplotlib is this package's one non-optional engine, so an environment that can import
    `digitalearth` can run these.
    """

    contract = StaticContract()


#: The tiers whose `Renderer.apply` rolls its own record back, as importable module names. The 3-D tier is
#: deliberately absent: its rollback is the scene's `_change` (`three_d/base.py`), which re-applies the
#: figure the scene still shows, so its `apply` has no handler of its own to agree with.
_RENDERERS_THAT_ROLL_BACK = (
    "digitalearth.static.renderer",
    "digitalearth.web.renderer",
    "digitalearth.interactive.renderer",
)


def _caught_by(method) -> tuple:
    """Return the exception class names one method's `except` clauses name.

    Read from the source rather than provoked, because what separates the two answers is an interruption —
    `KeyboardInterrupt`, `SystemExit` — and raising one of those through a tier to watch it be caught is a
    worse test than reading the clause that decides it.

    Args:
        method: The function or method to read.

    Returns:
        The class names, sorted and de-duplicated.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    return tuple(
        sorted(
            {
                handler.type.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Try)
                for handler in node.handlers
                if isinstance(handler.type, ast.Name)
            }
        )
    )


class TestTheTiersAgreeOnWhatARefusalIs:
    """`apply` rolls its record back on the way out, and the tiers disagreed on what triggers that (N2)."""

    def test_every_rolling_back_apply_catches_the_same_class(self):
        """One contract, one answer: a rollback either covers an interruption everywhere or nowhere.

        Test scenario:
            Static caught `BaseException`, web and interactive `Exception`. So a `KeyboardInterrupt` part-way
            through a change left the static record consistent and the other two holding layers no figure
            owned — three tiers signing one contract with two answers to what a refusal is. Cosmetic while
            `apply` is record-only, and not once it is wired into what a viewer sees.
        """
        import importlib

        caught = {
            name: _caught_by(importlib.import_module(name).Renderer.apply)
            for name in _RENDERERS_THAT_ROLL_BACK
        }
        assert len(set(caught.values())) == 1, (
            f"the tiers' rollbacks catch different things: {caught}"
        )

    def test_the_class_they_agree_on_covers_an_interruption(self):
        """Agreeing on `Exception` would be agreeing to leave the record broken by a Ctrl-C."""
        import importlib

        static = _caught_by(
            importlib.import_module("digitalearth.static.renderer").Renderer.apply
        )
        assert static == ("BaseException",), (
            f"a rollback has to survive an interruption, not only an error; static catches {static}"
        )
