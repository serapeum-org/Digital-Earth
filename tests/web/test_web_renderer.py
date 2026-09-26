"""The web renderer's own API: reconciling two figures, and refusing what it cannot draw (#296).

`tests/web/test_web_seam.py` checks the seam through the map — that a description is what gets drawn.
These reach the renderer directly, for the paths a builder does not take: reconciling an arbitrary pair of
figures, removing and re-showing a layer, and the guards that turn a malformed description into a message
rather than a `KeyError` from somewhere far away.
"""

from dataclasses import replace as with_fields

import pytest

from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.web.renderer import (
    DRAWN_KINDS,
    PAINT_CHANNELS,
    Renderer,
    drawer_for,
    portable_encodings,
    required_props,
)

from .test_web_capabilities import DESCRIBED_AND_DRAWN

pytest.importorskip("maplibre", reason="the web tier needs the web environment")


@pytest.fixture
def points_gdf():
    """Return a small point collection in EPSG:4326.

    Returns:
        A two-point GeoDataFrame.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
        crs=4326,
    )


@pytest.fixture
def drawn_map(points_gdf):
    """Return a map with one drawn point layer.

    Args:
        points_gdf: The features to draw.

    Returns:
        The map.
    """
    from digitalearth.web import WebMap

    return WebMap().points(points_gdf, name="obs")


class TestADescriptionItCannotDrawIsRefusedByName:
    """A bare `KeyError` on a MapLibre key names neither the layer nor what is wrong with it."""

    def test_a_layer_with_no_symbology_names_itself_and_what_is_missing(self):
        """A `LayerSpec` can reach a drawer without what its builder would have recorded."""
        # Built outside the block, so the only call inside it is the one under test.
        layer = LayerSpec("x", "points")
        with pytest.raises(ValueError, match="cannot be drawn by the web tier"):
            required_props(layer, "paint")

    def test_the_message_names_the_layer_its_kind_and_the_missing_props(self):
        """Each of the three is something the reader needs to find the malformed description."""
        layer = LayerSpec("wells", "polygons")
        with pytest.raises(ValueError) as caught:
            required_props(layer, "paint", "maplibre_type")
        message = str(caught.value)
        assert "'wells'" in message, message
        assert "polygons" in message, message
        assert "maplibre_type" in message, message

    def test_a_description_that_carries_them_is_returned_as_a_plain_dict(self):
        """The guard must not refuse what it exists to let through."""
        layer = LayerSpec("x", "points", symbology=Symbology(props={"paint": {"a": 1}}))
        assert required_props(layer, "paint") == {"paint": {"a": 1}}


class TestAKindThisTierDoesNotDraw:
    """A figure written for another backend should say so."""

    def test_it_is_refused_by_name_and_told_what_is_drawn(self):
        """The message lists the kinds, so the caller can see what they meant."""
        with pytest.raises(KeyError, match="does not draw 'point_cloud'"):
            drawer_for("point_cloud")

    def test_the_drawer_table_is_held_against_the_declared_kinds(self, monkeypatch):
        """Drift either way is a defect in this module, not in the caller.

        Args:
            monkeypatch: Used to make the two lists disagree.

        Test scenario:
            A kind in the table and not in `DRAWN_KINDS` would be refused with a message that is false; one
            in `DRAWN_KINDS` with no drawer would raise a bare `KeyError` where the message belongs.
        """
        from digitalearth.web import renderer

        monkeypatch.setattr(renderer, "DRAWN_KINDS", ("points", "nonesuch"))
        with pytest.raises(KeyError, match="drawer table and DRAWN_KINDS disagree"):
            renderer.drawer_for("points")


class TestReconcilingTwoFigures:
    """`apply` is how a map moves from one description to another."""

    def test_a_removed_layer_is_no_longer_drawn(self, drawn_map):
        """What the figure drops, the renderer drops.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        # The panel lists the layers it shows, so dropping a layer drops it from both — a figure that
        # named a layer it no longer holds is refused by the spec, which is the point of that check.
        panel = with_fields(figure.panels[0], layers=())
        emptied = with_fields(
            figure, layers=figure.layers.remove("obs"), panels=(panel,)
        )
        drawn_map._renderer.apply(figure, emptied)
        assert drawn_map._renderer.drawn == {}, drawn_map._renderer.drawn

    def test_a_layer_whose_data_changed_is_drawn_again(self, drawn_map, points_gdf):
        """A new source means a new MapLibre source, so the layer is rebuilt rather than restyled.

        Args:
            drawn_map: A map with one drawn layer.
            points_gdf: Used to register a second source to point at.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["obs"].layer
        moved = with_fields(figure.layers.get("obs"), kind="lines")
        rebuilt = with_fields(figure, layers=figure.layers.replace(moved))
        drawn_map._renderer.apply(figure, rebuilt)
        assert drawn_map._renderer.drawn["obs"].layer is not was, (
            "the layer was not rebuilt"
        )

    def test_a_label_change_alone_does_not_reach_maplibre(self, drawn_map):
        """`diff` groups a label with a colour; only one of them is drawn.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["obs"].layer
        renamed = with_fields(figure.layers.get("obs"), label="something else")
        relabelled = with_fields(figure, layers=figure.layers.replace(renamed))
        drawn_map._renderer.apply(figure, relabelled)
        assert drawn_map._renderer.drawn["obs"].layer is was, (
            "a rename redrew the layer"
        )

    def test_a_style_change_does_reach_maplibre(self, drawn_map):
        """The other half of the same guard.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["obs"].layer
        layer = figure.layers.get("obs")
        paint = {**dict(layer.symbology.props["paint"]), "circle-radius": 11.0}
        restyled = with_fields(
            figure,
            layers=figure.layers.replace(
                with_fields(
                    layer,
                    symbology=Symbology(
                        props={**layer.symbology.props, "paint": paint}
                    ),
                )
            ),
        )
        drawn_map._renderer.apply(figure, restyled)
        drawn = drawn_map._renderer.drawn["obs"].layer
        assert drawn is not was, "a restyle must redraw"
        assert drawn.paint["circle-radius"] == 11.0, drawn.paint

    def test_hiding_and_showing_a_layer_moves_its_visibility(self, drawn_map):
        """A layer switcher toggles what is already drawn rather than rebuilding it.

        Args:
            drawn_map: A map with one drawn layer.
        """
        renderer = drawn_map._renderer
        renderer.set_visible("obs", False)
        assert renderer.drawn["obs"].layer.layout["visibility"] == "none"
        renderer.set_visible("obs", True)
        assert renderer.drawn["obs"].layer.layout["visibility"] == "visible"

    def test_a_figure_that_hides_a_layer_hides_what_was_drawn(self, drawn_map):
        """A layer switcher's toggle is a change to the description, reconciled like any other.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        hidden = with_fields(
            figure,
            layers=figure.layers.replace(
                with_fields(figure.layers.get("obs"), visible=False)
            ),
        )
        drawn_map._renderer.apply(figure, hidden)
        assert drawn_map._renderer.drawn["obs"].layer.layout["visibility"] == "none"

        drawn_map._renderer.apply(hidden, figure)
        assert drawn_map._renderer.drawn["obs"].layer.layout["visibility"] == "visible"

    def test_toggling_a_drawing_that_carries_no_layer_is_quiet(self, drawn_map):
        """A drawer may produce a source and no layer; toggling it must not raise.

        Args:
            drawn_map: A map with one drawn layer.
        """
        from digitalearth.web.renderer import DrawnLayer

        drawn_map._renderer._drawn["headless"] = DrawnLayer(
            source_id=None, source_spec=None, layer=None
        )
        assert drawn_map._renderer.set_visible("headless", False) is None

    def test_showing_a_layer_nothing_drew_is_quiet(self, drawn_map):
        """A declined layer has no drawing to toggle, and toggling it is not an error.

        Args:
            drawn_map: A map with one drawn layer.
        """
        assert drawn_map._renderer.set_visible("nobody", True) is None

    def test_removing_a_layer_nothing_drew_is_quiet(self, drawn_map):
        """So a caller can remove a layer the tier declined to draw.

        Args:
            drawn_map: A map with one drawn layer.
        """
        held = dict(drawn_map._renderer.drawn)
        drawn_map._renderer.remove("nobody")
        assert drawn_map._renderer.drawn == held, (
            "removing an unknown id touched something"
        )


class TestApplyReachesTheQueue:
    """Review M1, answered: `apply` moves the record **and** the queue the page is built from."""

    @staticmethod
    def _emptied(drawn_map):
        """Return the map's figure with its one layer taken out.

        Args:
            drawn_map: A map with one drawn layer.

        Returns:
            The figure, and the panel emptied with it so `FigureSpec` accepts it.
        """
        figure = drawn_map.figure_spec
        panel = with_fields(figure.panels[0], layers=())
        return with_fields(figure, layers=figure.layers.remove("obs"), panels=(panel,))

    def test_a_successful_apply_takes_the_layer_out_of_the_queue(self, drawn_map):
        """A removed layer has to leave the page, not only the record beside it.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The module once said "the next build draws it", and it did not: the widget is built from the
            map's queue, which `apply` never touched, so a layer it removed stayed on the page and one it
            added never arrived. This is the same call, asked of the queue.
        """
        drawn_map._renderer.apply(drawn_map.figure_spec, self._emptied(drawn_map))
        assert drawn_map._queued == [], drawn_map._queued

    def test_a_successful_apply_moves_the_record_too(self, drawn_map):
        """So the check above is not passing because `apply` emptied the queue and nothing else.

        Args:
            drawn_map: A map with one drawn layer.
        """
        drawn_map._renderer.apply(drawn_map.figure_spec, self._emptied(drawn_map))
        assert sorted(drawn_map._renderer.drawn) == [], drawn_map._renderer.drawn

    def test_a_successful_apply_leaves_the_figure_the_map_reports_alone(
        self, drawn_map
    ):
        """The description is the map's own, installed by `_change` once `apply` has returned.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The split matters: `apply` called directly — by a caller composing a figure of their own, or by
            the conformance adapter — must not leave the map describing something nobody asked it to. That is
            what makes "a figure this refuses is never one the map reports" true of the public methods, which
            install the description themselves.
        """
        drawn_map._renderer.apply(drawn_map.figure_spec, self._emptied(drawn_map))
        assert list(drawn_map.figure_spec.layers.ids) == ["obs"], drawn_map.layer_ids


def _visibilities(drawn) -> list:
    """Return the `layout.visibility` of every MapLibre layer one drawing holds, its extra layers included.

    Args:
        drawn: A `DrawnLayer`.

    Returns:
        One entry per layer — the drawing's own, then its extra layers — reading `None` where the layout
        says nothing, which MapLibre draws as visible. A caller's own layer may be a plain dict spec, so
        that shape is read too.
    """
    visibilities = []
    for layer in (drawn.layer, *drawn.extra_layers):
        layout = layer.get("layout") if isinstance(layer, dict) else layer.layout
        visibilities.append((layout or {}).get("visibility"))
    return visibilities


def _all_hidden(figure):
    """Return `figure` with every layer described as hidden, and nothing else changed.

    Args:
        figure: The figure to hide.

    Returns:
        The figure.
    """
    tree = figure.layers
    for layer in list(tree):
        tree = tree.replace(with_fields(layer, visible=False))
    return with_fields(figure, layers=tree)


class TestVisibilityReachesEveryLayerADescriptionDraws:
    """Review M4: a description that draws several MapLibre layers is shown and hidden as one."""

    def test_hiding_a_labelled_graticule_hides_its_labels(self):
        """`draw_graticule` promises "a hidden graticule hides its labels too"; that held only at build time.

        Test scenario:
            A layer switcher's toggle is a change to the description, reconciled through `apply`, which
            reaches `set_visible`. That set the primary layer's visibility and left the degree labels
            floating over a map with no grid.
        """
        from digitalearth.web import WebMap

        m = WebMap().graticule(name="grid")
        figure = m.figure_spec
        m._renderer.apply(figure, _all_hidden(figure))
        assert _visibilities(m._renderer.drawn["grid"]) == ["none", "none"]

    def test_showing_it_again_shows_its_labels_too(self):
        """The other direction, so a fix that only ever hid could not pass."""
        from digitalearth.web import WebMap

        m = WebMap().graticule(name="grid")
        figure = m.figure_spec
        hidden = _all_hidden(figure)
        m._renderer.apply(figure, hidden)
        m._renderer.apply(hidden, figure)
        assert _visibilities(m._renderer.drawn["grid"]) == ["visible", "visible"]

    def test_hiding_a_cluster_hides_its_counts_and_loose_points(self, points_gdf):
        """The other drawer with extra layers: its counts and its unclustered points go with the bubbles.

        Args:
            points_gdf: A small point collection.
        """
        from digitalearth.web import WebMap

        m = WebMap().cluster(points_gdf)
        layer_id = m.layer_ids[0]
        m._renderer.set_visible(layer_id, False)
        assert _visibilities(m._renderer.drawn[layer_id]) == ["none"] * 3

    @pytest.mark.parametrize("kind", sorted(DESCRIBED_AND_DRAWN))
    def test_a_hidden_description_is_drawn_hidden(self, kind):
        """Every drawer honours the description's visibility, not only those whose builder takes `visible=`.

        Args:
            kind: The drawn kind under test.

        Test scenario:
            A figure read back from a file, or reconciled by `apply`, can describe any layer as hidden.
            The text, heatmap, cluster and extrusion drawers built their layers visible whatever the
            description said, because their builders never pass `visible=False`.
        """
        from digitalearth.web import WebMap

        m = WebMap()
        DESCRIBED_AND_DRAWN[kind](m)
        hidden = _all_hidden(m.figure_spec)
        drawn = {
            layer_id: _visibilities(m._renderer.draw_layer(hidden, layer_id))
            for layer_id in hidden.layers.ids
        }
        shown = {
            layer_id: visibilities
            for layer_id, visibilities in drawn.items()
            if set(visibilities) != {"none"}
        }
        assert shown == {}, f"{kind}: these are drawn visible though described hidden"


def _tuples_in(value, path="") -> list:
    """Return where a tuple sits anywhere inside `value`.

    Args:
        value: A MapLibre paint or layout value, or any part of one.
        path: Where `value` itself sits, for the report.

    Returns:
        The path of every tuple found, outermost first — empty when there is none.
    """
    if isinstance(value, tuple):
        return [path or "<root>"]
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            for found in _tuples_in(item, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            found
            for index, item in enumerate(value)
            for found in _tuples_in(item, f"{path}[{index}]")
        ]
    return []


#: Builder calls whose paint or layout holds a MapLibre *expression* — a nested sequence — which is where a
#: frozen tuple would show. Every drawn kind's plain call is covered by `DESCRIBED_AND_DRAWN` as well.
_EXPRESSION_BUILDS = {
    "points-by-column": lambda m: m.points(_frame(), column="value"),
    "lines-by-column": lambda m: m.lines(_frame("lines"), column="value"),
    "polygons-by-column": lambda m: m.polygons(_frame("polygons"), column="value"),
    "choropleth-graduated": lambda m: m.choropleth(
        _frame("polygons"), column="value", scheme="quantiles", k=2
    ),
    "choropleth-categorical": lambda m: m.choropleth(
        _frame("polygons"), column="value", scheme="categorical"
    ),
    "labels-with-offset": lambda m: m.labels(_frame(), "value", offset=(0.0, -1.2)),
    "heatmap-weighted": lambda m: m.heatmap(_frame(), weight="value"),
    "extrusion-by-column": lambda m: m.extrusion(
        _frame("polygons"), height="value", column="value"
    ),
}


def _frame(geometry: str = "points"):
    """Return two features of one geometry type in EPSG:4326, with a numeric `value` column.

    Args:
        geometry: `"points"`, `"lines"` or `"polygons"`.

    Returns:
        A GeoDataFrame.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon

    shapes = {
        "points": [Point(4.9, 52.4), Point(5.1, 52.1)],
        "lines": [
            LineString([(4.0, 52.0), (5.0, 53.0)]),
            LineString([(5.0, 52.0), (6.0, 53.0)]),
        ],
        "polygons": [
            Polygon([(4.0, 52.0), (5.0, 52.0), (5.0, 53.0), (4.0, 53.0)]),
            Polygon([(5.0, 52.0), (6.0, 52.0), (6.0, 53.0), (5.0, 53.0)]),
        ],
    }
    return gpd.GeoDataFrame({"value": [1.0, 2.0]}, geometry=shapes[geometry], crs=4326)


class TestTheDrawnLayersCarryLists:
    """Review L5: `WebMap.layers` hands MapLibre's JSON shapes back, not the description's frozen tuples."""

    @pytest.mark.parametrize(
        "build",
        [*DESCRIBED_AND_DRAWN.values(), *_EXPRESSION_BUILDS.values()],
        ids=[*DESCRIBED_AND_DRAWN, *_EXPRESSION_BUILDS],
    )
    def test_no_drawn_paint_or_layout_holds_a_tuple(self, build):
        """A description freezes its sequences to tuples; what a drawer hands MapLibre is thawed back.

        Args:
            build: The builder call under test.

        Test scenario:
            The emitted page was unaffected — a tuple serialises as a JSON array — but the public
            `layers` view changed type: `paint["circle-color"]` read `('interpolate', ('linear',), ...)`
            where it had read a list. Every drawn layer, extra layers included, is walked for a tuple.
        """
        from digitalearth.web import WebMap

        m = WebMap()
        build(m)
        found = {
            layer.id: _tuples_in(
                {"paint": layer.paint or {}, "layout": layer.layout or {}}
            )
            for built in m._renderer.drawn.values()
            for layer in (built.layer, *built.extra_layers)
            if not isinstance(layer, dict)
        }
        assert {key: paths for key, paths in found.items() if paths} == {}, found

    def test_the_colour_expression_reads_back_as_a_list(self):
        """The review's own reading: `layers[i].paint["circle-color"]` of a column-coloured layer."""
        from digitalearth.web import WebMap

        paint = WebMap().points(_frame(), column="value").layers[0].paint
        assert paint["circle-color"][:2] == ["interpolate", ["linear"]], paint


class TestWhatTheRendererReports:
    """The record is read by `WebMap.layers` and by the widget builder."""

    def test_a_layer_not_in_both_figures_is_treated_as_a_redraw(self, drawn_map):
        """`restyled` only names ids both hold, so this is about a direct call.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        added = with_fields(
            figure, layers=figure.layers.add(LayerSpec("later", "points"))
        )
        assert Renderer._reaches_maplibre(figure, added, "later") is True

    def test_the_compatibility_view_skips_a_layer_whose_drawing_is_gone(
        self, drawn_map
    ):
        """`WebMap.layers` resolves each queued marker through the record, which a removal empties.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The marker in the queue and the drawing in the record go together on every path that removes
            a layer, but the view must not assume it: a marker left standing alone resolves to nothing,
            and reading `.layer` off that nothing raises inside a property every caller touches.
            `tests/web/test_web_seam.py` asks the same question of the widget builder.
        """
        drawn_map._renderer.remove("obs")
        assert drawn_map.layers == [], drawn_map.layers
        assert len(drawn_map._queued) == 1, (
            "the marker must still be queued, or an empty queue is what emptied the view"
        )

    def test_the_declared_kinds_are_the_ones_drawn_from_a_description(self):
        """The tuple is the tier's answer to "what do you draw?"."""
        assert "points" in DRAWN_KINDS
        assert "terrain" not in DRAWN_KINDS, (
            "terrain is a deck.gl path, still queued, and must not claim to be described"
        )

    def test_band_for_prefers_the_layer_s_own_band(self, drawn_map):
        """A custom layer's band is the only thing that says where it belongs.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        own = LayerSpec("x", "custom:maplibre", band="underlay")
        assert drawn_map._renderer.band_for(own) == "underlay"

    def test_band_for_falls_back_to_the_kind_s_band(self, drawn_map):
        """Every other kind declares its own band once, at registration.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        from digitalearth.base.registry import band_of

        assert drawn_map._renderer.band_for(LayerSpec("x", "graticule")) == band_of(
            "graticule"
        )


class TestAskingWhetherALayerIsDrawn:
    """Review M4 — `is_visible` is the read-back `set_visible` writes, and every tier answers it."""

    def test_a_drawn_layer_reads_back_as_drawn(self, drawn_map):
        """The reader must answer the engine, not a description, so the built case is the control.

        Args:
            drawn_map: A map with one drawn point layer.

        Test scenario:
            A reader that always said `False` would pass every hiding check on its own; this is the half
            of the contract that fails when it does.
        """
        assert drawn_map._renderer.is_visible("obs") is True, (
            "a layer nothing hid must read back drawn"
        )

    def test_a_hidden_layer_reads_back_hidden(self, drawn_map):
        """MapLibre's `layout.visibility` is what the answer comes from, so hiding must move it.

        Args:
            drawn_map: A map with one drawn point layer.
        """
        drawn_map._renderer.set_visible("obs", False)
        assert drawn_map._renderer.is_visible("obs") is False, (
            "set_visible must be readable back through is_visible"
        )

    def test_an_id_nothing_drew_is_refused_by_name(self, drawn_map):
        """A layer the widget does not hold has no visibility, and guessing one would hide a defect.

        Args:
            drawn_map: A map with one drawn point layer.

        Test scenario:
            The conformance suite asks this question of every tier through `drawn_is_hidden`. Answering
            `True` for an unknown id would let a check pass against a layer that was never drawn at all.
        """
        renderer = drawn_map._renderer
        with pytest.raises(KeyError, match="nothing is drawn for layer 'nope'"):
            renderer.is_visible("nope")

    def test_the_refusal_says_what_the_tier_does_hold(self, drawn_map):
        """Naming the ids that are there is what turns the refusal into a diagnosis.

        Args:
            drawn_map: A map with one drawn point layer.
        """
        renderer = drawn_map._renderer
        with pytest.raises(KeyError) as refused:
            renderer.is_visible("nope")
        assert "obs" in str(refused.value), (
            f"the refusal must list the drawn ids; got {refused.value}"
        )


class TestThePaintIsReadBackAsChannels:
    """What the tier already records, read back in the portable vocabulary (#328).

    Every vector builder funnels through `_vector_layer`, which records MapLibre's own `paint` dict because
    that is what :func:`~digitalearth.web.renderer.draw_vector` rebuilds the layer from. That dict is
    unreadable anywhere else, so a layer this tier drew described its style in a form only this tier could
    use, and `Symbology.encodings` came back empty however the layer was styled.
    :func:`~digitalearth.web.renderer.portable_encodings` is the read-back, and it changes nothing a drawer
    sees: `paint` stays exactly as the builder wrote it.
    """

    def test_every_mapped_paint_key_drives_a_declared_channel(self):
        """A row pointing at a channel the vocabulary does not declare would be a silent no-op."""
        from digitalearth.base.spec import CHANNELS

        unknown = sorted(set(PAINT_CHANNELS.values()) - set(CHANNELS))
        assert unknown == [], f"PAINT_CHANNELS names undeclared channels: {unknown}"

    def test_a_points_paint_reads_back_as_the_channels_a_caller_asked_for(self):
        """The three a point layer always writes: its radius, its opacity and its colour."""
        paint = {"circle-radius": 7.0, "circle-opacity": 0.5, "circle-color": "#cc4444"}
        lifted = portable_encodings(Symbology(props={"paint": paint}))
        resolved = {channel: lifted[channel].resolve() for channel in sorted(lifted)}
        assert resolved == {"color": "#cc4444", "opacity": 0.5, "size": 7.0}, resolved

    def test_a_compiled_expression_claims_no_channel(self):
        """A classified fill is MapLibre's own data-driven form, and means nothing to another engine."""
        expression = ("step", ("get", "pop"), "#440154", 5.0, "#fde725")
        paint = {"fill-color": expression, "fill-opacity": 0.4}
        lifted = portable_encodings(Symbology(props={"paint": paint}))
        assert sorted(lifted) == ["opacity"], sorted(lifted)

    def test_a_paint_key_that_drives_no_channel_stays_in_the_paint_alone(self):
        """There is no stroke or halo channel, so `fill-outline-color` has no portable reading."""
        paint = {"fill-outline-color": "#ffffff", "text-halo-width": 1.0}
        assert sorted(portable_encodings(Symbology(props={"paint": paint}))) == []

    def test_a_layer_with_no_paint_at_all_is_answered_rather_than_refused(self):
        """A graticule and a caller's own MapLibre object record no paint, and must not raise here."""
        assert portable_encodings(Symbology(props={"via": "graticule"})) == {}

    def test_a_container_is_never_lifted_into_the_description(self):
        """The hazard this rule exists for: a container can be carrying a credential.

        Test scenario:
            An `xyzservices.TileProvider` **is** a mapping, and one of its values is the caller's API key,
            so a lift that copied a container into a layer's description would write that key into every
            saved figure. Only a scalar a channel can hold is lifted.
        """
        provider = {
            "url": "https://tiles.example/{z}/{x}/{y}.png",
            "apikey": "NOT-REAL",
        }
        lifted = portable_encodings(
            Symbology(props={"paint": {"circle-color": provider}})
        )
        assert sorted(lifted) == [], lifted
