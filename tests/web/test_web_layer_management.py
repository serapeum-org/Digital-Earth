"""The web tier's half of order 23: the public toggle, reorder and replace over its own renderer.

`get_layer` and `remove_layer` were already here — the identity half landed with the seam — so what order 23
adds on this tier is `set_visible`, `move_layer` and `replace_layer`, and the change path all three take.

The shared probes are :mod:`tests.base.layer_management`; this module supplies the adapter and the checks that
are this tier's own: what happens to a queue entry **no layer id addresses**, and what happens to the four
kinds that used to be nothing but such an entry.

**The last gap.** `basemap`, `terrain`, `point_cloud` and `model` queued a closure and recorded no layer, so
the three methods above — which address a layer by its id — could not reach them at all: they were invisible
to the very feature order 23 added. Each records a layer now, and `TestTheKindsThatUsedToRecordNoLayer` below
holds that to the page rather than to the description, because recording a layer nothing draws differently is
the drift this seam exists to remove.

**Why the engine reading is the built page.** `_build_map_widget` replays `WebMap._queued` into the widget,
entry by entry. Until order 23 only the builders and `remove_layer` wrote that list and `Renderer.apply`
reconciled the renderer's record beside it, so a change reached a record and never the page (review M1) — and
the first fix read the *queue* back, which is a proxy and not the page: a deck.gl layer's slot in the queue
does not say where its interleaved overlay draws it, so `move_layer` across a deck layer and a style layer
reordered the queue and left the widget's calls byte-identical (review H6). So `engine_order` and
`_page_layer_ids` read the calls the build actually makes.
"""

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap
from digitalearth.web.base import _Described
from tests.base.layer_management import (
    BOTTOM,
    TOP,
    LayerManagementConformance,
    LayerManagementContract,
)


class WebContract(LayerManagementContract):
    """The web tier's adapter for the shared layer-management probes."""

    backend = "web"
    #: A caller's own PyVista object: registered, drawn from no data, and nothing this tier can draw.
    undrawable_kind = "custom:pyvista"

    def make(self):
        """Return an empty map.

        Returns:
            The map.
        """
        return WebMap()

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
        """Return the layer ids in the order the built page draws them.

        The page, not the queue. `layers` resolves the queue into the objects a build would hand MapLibre,
        which reads as an order and is not one: a deck.gl layer goes into an interleaved overlay whose
        placement the queue does not carry, and it is not even a MapLibre ``Layer``, so reading `.id` off the
        list dropped it silently. That is the proxy review H6 slipped through — the queue moved and the
        widget's calls came out byte-identical.

        Args:
            tier: The map.

        Returns:
            The ids, bottom first, as :func:`_page_layer_ids` reads them off the calls the build makes — so a
            queue entry no call comes out of is absent here, as it is from the page.
        """
        return tuple(_page_layer_ids(tier))

    def engine_objects(self, tier):
        """Return every MapLibre layer the widget would be given.

        Args:
            tier: The map.

        Returns:
            The layer objects themselves, so a removal is held to having taken the very layer it built out of
            the queue rather than only out of the renderer's record.
        """
        return tuple(tier.layers)

    def engine_holds(self, tier, layer_id):
        """Return the MapLibre layer the queue resolves for one id.

        The queue rather than the renderer's record: the record is what decides whether the queue is
        re-arranged, so a probe reading it back would be asking the decision about itself.

        Args:
            tier: The map.
            layer_id: The layer to look up.

        Returns:
            The MapLibre layer, or `None` when the page would add none for it.
        """
        for drawn in tier.layers:
            if getattr(drawn, "id", None) == layer_id:
                return drawn
        return None


class TestWebLayerManagement(LayerManagementConformance):
    """The web tier, signing the contract the 3-D tier already passes."""

    contract = WebContract()


@pytest.fixture
def mixed():
    """Yield a map holding a queued entry no layer id addresses, beneath two described labels.

    A caller's own `apply(widget)` callable, handed straight to `add_underlay`, is the entry: it is the shape
    the tier's own builders queued for `basemap`, `terrain`, `point_cloud` and `model` until each started
    recording a layer, and it is still what a caller who wires their own MapLibre calls gets. The underlay
    band, because that is where `tiles()` files its own.

    Yields:
        The map.
    """

    def draw(widget):
        """Stand in for a builder's own MapLibre call.

        Args:
            widget: The widget the entry would be applied to.
        """

    built = WebMap()
    built.add_underlay(draw)
    built.text(0.0, 0.0, "lower", name=BOTTOM)
    built.text(1.0, 1.0, "upper", name=TOP)
    yield built
    built.close()


class TestAQueueEntryNoLayerIdAddresses:
    """The tier's unconverted builders queue a closure; a reorder must leave it exactly where it is."""

    @staticmethod
    def _shape(tier):
        """Return the queue as a comparable shape: a layer id, or `None` for a closure.

        Args:
            tier: The map.

        Returns:
            One entry per queued item, in order.
        """
        return [
            entry.layer_id if isinstance(entry, _Described) else None
            for entry in tier._queued
        ]

    def test_a_reorder_moves_the_described_layers_and_not_the_closure(self, mixed):
        """A closure is not addressed by an id, so nothing about it is being reordered.

        Args:
            mixed: The map under test.
        """
        mixed.move_layer(TOP, 0)
        assert self._shape(mixed) == [None, TOP, BOTTOM], self._shape(mixed)

    def test_the_band_counts_still_describe_the_queue(self, mixed):
        """The queue is addressed by counts from its two ends, so a reorder has to keep them true.

        Args:
            mixed: The map under test.

        Test scenario:
            `add_underlay` inserts by `_underlay_count` and `add_overlay` from the back. A reorder that left
            a count pointing at the wrong slot would put the *next* basemap above the data it belongs under —
            silently, and only on the next call.
        """
        mixed.move_layer(TOP, 0)
        counts = (
            mixed._underlay_count,
            mixed._reference_count,
            mixed._overlay_count,
        )
        assert counts == (1, 0, 2), counts

    def test_a_refused_change_leaves_the_band_counts_alone(self, mixed):
        """The counts roll back with the queue, or the next layer lands in the wrong band.

        Args:
            mixed: The map under test.
        """
        from digitalearth.base.capabilities import CapabilityError
        from digitalearth.base.spec import LayerSpec

        held = (mixed._underlay_count, mixed._reference_count, mixed._overlay_count)
        described = LayerSpec(TOP, "custom:pyvista", band="overlay")
        with pytest.raises(CapabilityError):
            mixed.replace_layer(described)
        counts = (
            mixed._underlay_count,
            mixed._reference_count,
            mixed._overlay_count,
        )
        assert counts == held, counts

    def test_an_arrangement_that_stops_part_way_puts_the_queue_back(
        self, mixed, monkeypatch
    ):
        """The queue and its counts go back together, or a count addresses a slot that is not there.

        Args:
            mixed: The map under test.
            monkeypatch: Stands a failure into the arrangement itself.

        Test scenario:
            The arrangement writes the list and then the three counts, so an interruption between them leaves
            the map addressing its bands by counts that no longer describe the queue — and the next
            `add_underlay` puts a basemap above the data. Provoked rather than waited for: nothing in the
            reconcile touches the queue, so this restore is unreachable from a refused *figure* and would be
            dead code with only that check behind it.
        """
        held = list(mixed._queued)

        def refuse(_after):
            """Stand in for an arrangement that empties the queue and then fails.

            Args:
                _after: The figure being arranged, unread.

            Raises:
                RuntimeError: always.
            """
            mixed._queued.clear()
            raise RuntimeError("stopped part-way")

        monkeypatch.setattr(mixed._renderer, "_arrange", refuse)
        with pytest.raises(RuntimeError):
            mixed.move_layer(TOP, 0)
        assert mixed._queued == held, self._shape(mixed)

    def test_a_removed_layer_gives_its_slot_up_without_moving_the_closure(self, mixed):
        """`remove_layer` is the tier's own path, and the arrangement has to agree with it.

        Args:
            mixed: The map under test.

        Test scenario:
            The removal and the reconcile are two ways to take a layer out of the queue — `remove_layer`
            filters the list itself — so this pins that a later reorder reads the queue the removal left and
            not a stale count.
        """
        mixed.remove_layer(BOTTOM)
        mixed.move_layer(TOP, 0)
        assert self._shape(mixed) == [None, TOP], self._shape(mixed)


#: The four kinds that queued a closure and recorded no layer until this gap was closed, resolved to a builder
#: call that takes a `name=` — so two of one kind can be built and reordered against each other inside the one
#: band their kind declares, which is the only move `LayerTree.move` allows.
#:
#: They are the same four `WEB_UNDRAWN_KINDS` in `tests/base/test_renderer_conformance.py` named, and the same
#: four `DRAWN_BUT_NOT_DESCRIBED` in `tests/web/test_web_capabilities.py` held a builder call for. Both lists
#: emptied when these were converted; this one is what holds the conversion to the *page*.
CONVERTED = {
    "basemap": lambda tier, name: tier.tiles(
        f"https://tiles.example/{name}/{{z}}/{{x}}/{{y}}.png", name=name
    ),
    "terrain": lambda tier, name: tier.terrain_tiles(name=name),
    "point_cloud": lambda tier, name: tier.point_cloud(
        [(4.9, 52.4, 10.0), (5.0, 52.2, 20.0)], name=name
    ),
    "model": lambda tier, name: tier.gltf(
        "https://example.invalid/model.glb", 4.9, 52.4, name=name
    ),
}


def _page_calls(tier):
    """Return what building the page asks of MapLibre, in order.

    The widget's own record, not a stand-in: `MapWidget.add_call` queues every call it will send, so this is
    the page as the viewer would get it.

    Args:
        tier: The map.

    Returns:
        A list of `(method, args)` pairs, in the order the build makes them.
    """
    return list(tier._build_map_widget()._message_queue)


def _page_layer_ids(tier):
    """Return the ids the built page draws, in the order it draws them.

    Each of the three routes is read in its own terms, because none of them is the others: a style layer is an
    ``addLayer`` whose ``layout.visibility`` is not ``none``; terrain is a ``setTerrain`` naming the source
    derived from the layer's id, and a hidden one is **not called at all**; and a deck.gl layer is an entry of
    the one ``addDeckOverlay`` whose own ``visible`` property is not off.

    The deck overlay is **not** read as "whatever the one call carries, last". It is interleaved, so deck.gl
    places each of its layers by the ``beforeId`` it carries: immediately beneath the style layer it names,
    in the order the overlay lists them, and on top of everything when it names nothing (or names a layer the
    style does not hold). Reading the call list in order instead is the proxy that let review H6 through — it
    agreed with the engine only because every probe drew two layers of one route.

    Args:
        tier: The map.

    Returns:
        The layer ids the viewer would see, bottom first.
    """
    order: list = []
    hidden: set = set()
    anchorable: set = set()
    overlay: list = []
    for method, args in _page_calls(tier):
        if method == "addLayer":
            layer = args[0]
            order.append(layer["id"])
            anchorable.add(layer["id"])
            if (layer.get("layout") or {}).get("visibility") == "none":
                hidden.add(layer["id"])
        elif method == "setTerrain":
            order.append(args[0]["source"].removesuffix("-src"))
        elif method == "addDeckOverlay":
            overlay = [
                (layer.get("beforeId"), layer["id"])
                for layer in args[0]
                if layer.get("visible", True)
            ]
    drawn = []
    for layer_id in order:
        drawn.extend(deck_id for anchor, deck_id in overlay if anchor == layer_id)
        if layer_id not in hidden:
            drawn.append(layer_id)
    drawn.extend(deck_id for anchor, deck_id in overlay if anchor not in anchorable)
    return drawn


@pytest.fixture(params=sorted(CONVERTED))
def converted(request):
    """Yield the kind under test and a map holding two layers of it, named `BOTTOM` and `TOP`.

    Two, so a reorder has somewhere to go inside the band the kind declares — a basemap cannot be dragged over
    the data, which is what `LayerTree.move` refuses and what would make a single layer untestable here.

    Args:
        request: pytest's request, carrying the kind under test.

    Yields:
        The kind, and the map.
    """
    build = CONVERTED[request.param]
    built = WebMap()
    build(built, BOTTOM)
    build(built, TOP)
    yield request.param, built
    built.close()


class TestTheKindsThatUsedToRecordNoLayer:
    """`basemap`, `terrain`, `point_cloud` and `model`, held to the page rather than to the record."""

    def test_the_builder_records_a_layer_under_the_name_it_was_given(self, converted):
        """A layer id is what all three methods address, and these builders minted none.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        kind, tier = converted
        assert tier.layer_ids == [BOTTOM, TOP], tier.layer_ids
        kinds = [layer.kind for layer in tier.figure_spec.layers]
        assert kinds == [kind, kind], kinds

    def test_the_page_draws_both_of_them(self, converted):
        """The description has to be what the page is built from, or hiding one would change nothing.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        assert _page_layer_ids(tier) == [BOTTOM, TOP], _page_layer_ids(tier)

    def test_hiding_one_takes_it_off_the_page(self, converted):
        """`set_visible` reaches the viewer, not only the renderer's record.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.set_visible(BOTTOM, False)
        assert _page_layer_ids(tier) == [TOP], _page_layer_ids(tier)

    def test_showing_it_again_puts_it_back(self, converted):
        """The other direction, so a fix that only ever hid could not pass.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.set_visible(BOTTOM, False)
        tier.set_visible(BOTTOM, True)
        assert _page_layer_ids(tier) == [BOTTOM, TOP], _page_layer_ids(tier)

    def test_the_renderer_reports_the_visibility_it_was_asked_for(self, converted):
        """Every tier's renderer answers `is_visible`, and these routes had no answer to give at all.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.set_visible(BOTTOM, False)
        assert tier._renderer.is_visible(BOTTOM) is False
        assert tier._renderer.is_visible(TOP) is True

    def test_moving_one_changes_what_the_page_draws_last(self, converted):
        """`move_layer` is the reorder, and what MapLibre adds last it draws on top.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.move_layer(BOTTOM, 1)
        assert _page_layer_ids(tier) == [TOP, BOTTOM], _page_layer_ids(tier)

    def test_removing_one_takes_it_off_the_page(self, converted):
        """`remove_layer` was already public; it had nothing to address on these four.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.remove_layer(BOTTOM)
        assert _page_layer_ids(tier) == [TOP], _page_layer_ids(tier)

    def test_replacing_one_keeps_its_id_and_its_place(self, converted):
        """A replacement is drawn again from the new description, in the slot the old one held.

        Args:
            converted: The kind under test and a map holding two of it.

        Test scenario:
            `group` is the portable "restyle that must reach the engine" the shared probes use: every tier's
            redraw guard compares `(symbology, filter, group)`, and it needs no knowledge of which props this
            tier's drawers read.
        """
        from dataclasses import replace as with_fields

        _, tier = converted
        tier.replace_layer(with_fields(tier.get_layer(BOTTOM), group="regrouped"))
        assert tier.get_layer(BOTTOM).group == "regrouped"
        assert _page_layer_ids(tier) == [BOTTOM, TOP], _page_layer_ids(tier)


class TestTheDeckOverlayIsOneCall:
    """deck.gl owns one overlay per page, and two kinds now reach it from their own queue slots."""

    def test_a_described_cloud_and_an_undescribed_scatter_share_one_overlay(self):
        """A second ``addDeckOverlay`` replaces the first, so every deck layer has to go in one call.

        Test scenario:
            `point_cloud` and `model` are described and join the overlay from their queue slots;
            `deck_scatter`, `deck_polygons` and `tiles_3d` record nothing and accumulate into
            `WebMap._deck_layers`. There used to be one accumulator and one closure, so one call was
            automatic. The page composes the overlay from the queue now, and a composition that left either
            source out — or issued a call per source — would drop layers silently.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            geometry=[Point(4.9, 52.4), Point(5.0, 52.2)], crs=4326
        )
        tier = WebMap()
        tier.deck_scatter(frame)
        tier.point_cloud([(4.9, 52.4, 10.0)], name="cloud")
        tier.gltf("https://example.invalid/model.glb", 4.9, 52.4, name="statue")
        overlays = [
            args[0] for method, args in _page_calls(tier) if method == "addDeckOverlay"
        ]
        assert len(overlays) == 1, (
            f"{len(overlays)} deck overlays; all but the last are lost"
        )
        ids = [layer.get("id") for layer in overlays[0]]
        assert ids[1:] == ["cloud", "statue"], ids
        tier.close()

    def test_reordering_a_described_cloud_reorders_the_overlay(self):
        """The overlay is composed from the queue, so `move_layer` reaches deck.gl's draw order too."""
        tier = WebMap()
        tier.point_cloud([(4.9, 52.4, 10.0)], name="lower")
        tier.point_cloud([(5.0, 52.2, 20.0)], name="upper")
        tier.move_layer("lower", 1)
        assert _page_layer_ids(tier) == ["upper", "lower"], _page_layer_ids(tier)
        tier.close()


def _page_overlay(tier):
    """Return the deck.gl layers the built page composes into its one overlay.

    Args:
        tier: The map.

    Returns:
        The deck.gl JSON layers the single ``addDeckOverlay`` call carries, in its own order, or an empty
        list when the page composes no overlay at all.
    """
    for method, args in _page_calls(tier):
        if method == "addDeckOverlay":
            return list(args[0])
    return []


@pytest.fixture
def routed():
    """Yield a map with a deck-routed layer described **beneath** a MapLibre style layer, in one band.

    `point_cloud` and `points` are both `data` layers, so a move between them is one `LayerTree.move`
    allows — and they take different routes, which is what `draw_two` cannot express: the shared probes
    draw two layers of **one kind**, so both always take one route and a reorder across routes is out of
    their reach (review H6).

    Yields:
        The map.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]}, geometry=[Point(4.9, 52.4), Point(5.0, 52.2)], crs=4326
    )
    built = WebMap()
    built.point_cloud([(4.9, 52.4, 10.0), (5.0, 52.2, 20.0)], name=BOTTOM)
    built.points(frame, name=TOP)
    yield built
    built.close()


class TestADeckLayerOrderedAgainstAStyleLayer:
    """A deck.gl layer and a MapLibre style layer in one band, held to the page and not to the queue.

    The page composes **one** deck overlay and adds it after the whole queue, so for a while every deck
    layer drew above every style layer whatever the description said: `move_layer` reordered the queue and
    the widget's calls came out byte-identical (review H6). deck.gl's overlay is interleaved, so each layer
    carries the style layer it draws beneath instead.
    """

    @staticmethod
    def _routes(tier):
        """Return the route each layer was drawn for, in draw order.

        Args:
            tier: The map.

        Returns:
            One route per described layer, so a fixture that quietly drew two style layers cannot pass the
            probes below by measuring the easy case.
        """
        return [tier._renderer.drawn[layer_id].route for layer_id in tier.layer_ids]

    def test_the_two_layers_really_do_take_different_routes(self, routed):
        """The fixture is the whole point, so it is checked rather than assumed.

        Args:
            routed: The map under test.
        """
        routes = self._routes(routed)
        assert routes == ["deck", "style"], f"the fixture drew {routes}"

    def test_the_page_draws_the_deck_layer_beneath_the_style_layer(self, routed):
        """Described at the bottom of the band is drawn at the bottom of the band.

        Args:
            routed: The map under test.

        Test scenario:
            This is the "before" half of H6: the overlay was added after the whole queue, so the layer
            described **under** the style layer was drawn **over** it, and nothing in the suite read the
            page closely enough to say so.
        """
        assert _page_layer_ids(routed) == [BOTTOM, TOP], _page_layer_ids(routed)

    def test_moving_the_deck_layer_over_the_style_layer_reaches_the_page(self, routed):
        """`move_layer` across routes has to change what the page draws, not only what it records.

        Args:
            routed: The map under test.
        """
        routed.move_layer(BOTTOM, 1)
        assert _page_layer_ids(routed) == [TOP, BOTTOM], _page_layer_ids(routed)

    def test_the_overlay_names_the_style_layer_the_deck_layer_draws_beneath(
        self, routed
    ):
        """The page is told the placement, which is the reading the queue could not give.

        Args:
            routed: The map under test.

        Test scenario:
            `beforeId` is deck.gl's own answer for an interleaved overlay — the widget builds it with
            ``interleaved: true`` — and it is what makes the move reach the page rather than the record.
        """
        anchors = [layer.get("beforeId") for layer in _page_overlay(routed)]
        assert anchors == [TOP], anchors

    def test_a_deck_layer_described_on_top_names_no_style_layer(self, routed):
        """Nothing above it means nothing to draw beneath, and deck.gl puts it on top.

        Args:
            routed: The map under test.
        """
        routed.move_layer(BOTTOM, 1)
        anchors = [layer.get("beforeId") for layer in _page_overlay(routed)]
        assert anchors == [None], anchors

    def test_the_placement_is_not_written_into_the_renderer_record(self, routed):
        """Where a layer goes on one page is not part of what was drawn.

        Args:
            routed: The map under test.

        Test scenario:
            The record is rebuilt into a widget on every `render()`, and `figure_spec`/`to_dict` read it —
            so a placement written back into it would leak a page's layer stack into the description and
            travel with a saved figure.
        """
        _page_overlay(routed)
        held = routed._renderer.drawn[BOTTOM].layer
        assert "beforeId" not in held, sorted(held)

    def test_hiding_the_deck_layer_still_takes_it_off_the_page(self, routed):
        """`set_visible` is the sibling call, and the mixed case has to hold for it too.

        Args:
            routed: The map under test.
        """
        routed.set_visible(BOTTOM, False)
        assert _page_layer_ids(routed) == [TOP], _page_layer_ids(routed)

    def test_replacing_the_deck_layer_keeps_it_under_the_style_layer(self, routed):
        """`replace_layer` is the other sibling: a redraw must land back in the described place.

        Args:
            routed: The map under test.
        """
        from dataclasses import replace as with_fields

        routed.replace_layer(with_fields(routed.get_layer(BOTTOM), group="regrouped"))
        assert _page_layer_ids(routed) == [BOTTOM, TOP], _page_layer_ids(routed)


@pytest.fixture
def undescribed():
    """Yield a map with an undescribed `deck_scatter`, a described deck layer and a style layer.

    `deck_scatter` records no `LayerSpec`: it appends to `WebMap._deck_layers` and the queue holds **one**
    `_DeckOverlay` marker for every builder that does, so that slot is shared and says nothing about where
    any one of those layers belongs. `point_cloud` is the described counterpart — it owns its own queue slot
    — and `points` is the style layer both are ordered against, all three in the `data` band.

    Yields:
        The map.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]}, geometry=[Point(4.9, 52.4), Point(5.0, 52.2)], crs=4326
    )
    built = WebMap()
    built.deck_scatter(frame)
    built.point_cloud([(4.9, 52.4, 10.0), (5.0, 52.2, 20.0)], name=BOTTOM)
    built.points(frame, name=TOP)
    yield built
    built.close()


class TestAnUndescribedDeckBuilderIsNotOrderedAtAll:
    """`deck_scatter`/`deck_polygons`/`tiles_3d` share one queue slot, so nothing anchors them.

    Anchoring reads one anchor per **queue entry**, and these builders have no entry of their own: they
    accumulate behind a single `_DeckOverlay` marker whichever order they were called in. Anchoring them off
    that slot gave every one of them the first builder's position — drawn beneath the next style layer the
    queue adds, where they used to draw on top of every style layer — and since they record no `LayerSpec`
    they are not in `layer_ids`, so `move_layer` cannot put them back (review R2-H5). They keep no
    `beforeId`, which is deck.gl's own "on top", and the described deck layers keep theirs.
    """

    def test_the_overlay_anchors_the_described_layer_and_not_the_shared_slot(
        self, undescribed
    ):
        """The page is where the two are told apart, so the page is what is read.

        Args:
            undescribed: The map under test.
        """
        placed = [
            (layer["id"], layer.get("beforeId")) for layer in _page_overlay(undescribed)
        ]
        assert placed == [("deck-scatter-1", None), (BOTTOM, TOP)], placed

    def test_the_page_draws_the_shared_slot_above_every_style_layer(self, undescribed):
        """A layer a caller cannot address must not be moved under one they can.

        Args:
            undescribed: The map under test.
        """
        drawn = _page_layer_ids(undescribed)
        assert drawn == [BOTTOM, TOP, "deck-scatter-1"], drawn

    def test_a_builder_called_after_the_style_layer_is_on_top_too(self, undescribed):
        """The shared slot is queued once, so its position cannot describe a later builder either.

        Args:
            undescribed: The map under test.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        undescribed.deck_polygons(
            gpd.GeoDataFrame(
                geometry=[Point(4.9, 52.4).buffer(0.1)],
                crs=4326,
            )
        )
        drawn = _page_layer_ids(undescribed)
        assert drawn[-2:] == ["deck-scatter-1", "deck-polygons-1"], drawn

    def test_the_described_deck_layer_can_still_be_moved_over_the_style_layer(
        self, undescribed
    ):
        """The half that has a queue slot keeps the ordering the anchors were added for.

        Args:
            undescribed: The map under test.

        Test scenario:
            The described cloud starts beneath the style layer and is moved over it, which is the ordering
            `beforeId` was added for and must survive. It lands above the scatter rather than beneath it
            because both now name no style layer, and deck.gl draws one bucket in the overlay's own array
            order — which is the queue's, and the shared slot was queued first.
        """
        undescribed.move_layer(BOTTOM, 1)
        drawn = _page_layer_ids(undescribed)
        assert drawn == [TOP, "deck-scatter-1", BOTTOM], drawn


@pytest.fixture
def draped():
    """Yield a map whose second label takes its elevation from the first, so the first cannot be removed.

    `LayerTree.remove` refuses a layer another one is draped over — a removal that would leave the second
    reading an elevation nothing provides — and that is the one refusal `WebMap.remove_layer` can meet after
    it has already started taking the layer apart.

    Yields:
        The map.
    """
    from dataclasses import replace as with_fields

    built = WebMap().text(4.9, 52.4, "Amsterdam", name=BOTTOM)
    built.text(2.35, 48.86, "Paris", name=TOP)
    built.replace_layer(with_fields(built.get_layer(TOP), z_source=f"layer:{BOTTOM}"))
    yield built
    built.close()


class TestARefusedRemovalGivesNoIdBack:
    """A removal that cannot go through must leave the id pool exactly as it found it.

    The ids were freed before the tree was asked, so a refused `remove_layer` left the pool one id short of
    the tree: the layer was still on the map under a name the map no longer counted as taken (review R2-M8).
    """

    def test_the_refusal_names_the_layers_draped_over_it(self, draped):
        """The refusal itself, so a fix that stopped refusing could not pass the two probes below.

        Args:
            draped: The map under test.
        """
        with pytest.raises(ValueError, match="take their elevation from it"):
            draped.remove_layer(BOTTOM)

    def test_the_layer_that_did_not_go_is_still_on_the_map(self, draped):
        """Nothing else may have moved either — the tree, the renderer and the page all still hold it.

        Args:
            draped: The map under test.
        """
        with pytest.raises(ValueError, match="take their elevation from it"):
            draped.remove_layer(BOTTOM)
        assert draped.layer_ids == [BOTTOM, TOP], draped.layer_ids
        assert _page_layer_ids(draped) == [BOTTOM, TOP], _page_layer_ids(draped)

    def test_the_name_is_still_taken_so_the_next_builder_suffixes_it(self, draped):
        """The consequence a caller meets: a name still on the map has to be generated around, not reused.

        Args:
            draped: The map under test.

        Test scenario:
            Freeing the id first made the next `text(name=BOTTOM)` collide inside `LayerTree.add` — "a layer
            with id 'lower' is already in the tree", an internal message about a name the map had just told
            itself was free. The suffix is what every other tier gives, and what this tier gives when the
            removal really happens.
        """
        with pytest.raises(ValueError, match="take their elevation from it"):
            draped.remove_layer(BOTTOM)
        draped.text(0.0, 0.0, "again", name=BOTTOM)
        assert draped.layer_ids == [BOTTOM, TOP, f"{BOTTOM}-2"], draped.layer_ids
