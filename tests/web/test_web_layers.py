"""Addressing the layers on a web map (#188) — ids, visibility, removal and the viewer's switch.

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env. Assertions
read the exported page's call payload rather than the page as a whole, which also contains the MapLibre
library.
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


@pytest.fixture
def points():
    """Two points with a numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"v": [1, 2]}, geometry=[Point(0.0, 0.0), Point(1.0, 1.0)], crs="EPSG:4326"
    )


@pytest.fixture
def polygons():
    """Two disjoint squares with a numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"pop": [1, 9]},
        geometry=[
            Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
            Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def lines():
    """Two line strings with a numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"v": [1, 2]},
        geometry=[
            LineString([(0.0, 0.0), (1.0, 1.0)]),
            LineString([(1.0, 0.0), (2.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def raster_stack(tmp_path):
    """A 3-member ``DatasetCollection``, one file per step.

    Args:
        tmp_path: pytest's per-test directory.

    Returns:
        The collection.
    """
    pytest.importorskip("pyramids")
    import numpy as np
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset
    from pyramids.dataset.collection import DatasetCollection

    geo_ref = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.02, epsg=4326)
    paths = []
    for step in range(3):
        _, xx = np.mgrid[0:8, 0:9]
        path = tmp_path / f"step{step}.tif"
        Dataset.from_array(
            (10.0 * step + xx / 10.0).astype("float32"),
            geo_ref=geo_ref,
            no_data_value=-9999.0,
        ).to_file(str(path))
        paths.append(str(path))
    return DatasetCollection.from_files(paths)


@pytest.fixture
def raster():
    """A small single-band in-memory raster.

    Returns:
        A pyramids ``Dataset`` in EPSG:4326.
    """
    pytest.importorskip("pyramids")
    import numpy as np
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset

    geo_ref = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.02, epsg=4326)
    _, xx = np.mgrid[0:8, 0:9]
    return Dataset.from_array(
        (xx / 10.0).astype("float32"), geo_ref=geo_ref, no_data_value=-9999.0
    )


def _drawn(web_map):
    """Return the ids of a map's addressable layers in the order they are drawn, bottom first.

    Args:
        web_map: The map.

    Returns:
        Each registered layer's id, in `layers` order, skipping basemaps and controls, which carry none.
    """
    ids = [getattr(layer, "_digitalearth_layer_id", None) for layer in web_map.layers]
    return [layer_id for layer_id in ids if layer_id is not None]


def _payload(html):
    """Return the page's call payload — what this map does, not what the library contains.

    Args:
        html: A page from ``to_html``.

    Returns:
        The substring from ``var data =`` to the end of the page.
    """
    marker = html.rfind("var data = ")
    assert marker != -1, "the exported page carries no call payload"
    return html[marker:]


class TestTheRegistryIsAddressable:
    """A write-only registry cannot be revised, and in a notebook revising is the normal workflow."""

    def test_data_layers_report_their_ids_in_order(self, points, polygons):
        """Builders minted ids internally and surfaced none, so nothing could be addressed afterwards."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(polygons, column="pop").points(points)
        assert len(m.layer_ids) == 2, m.layer_ids
        assert m.layer_ids[0].startswith("fill"), m.layer_ids
        assert m.layer_ids[1].startswith("circle"), m.layer_ids

    def test_each_data_layer_is_described_with_its_kind_and_label(
        self, points, polygons
    ):
        """The index is a `LayerTree`, so a layer is described as well as named (DE-20, #281).

        Test scenario:
            The index held `(id, label)` pairs and the label was never read back. Backed by a `LayerTree`, each
            entry says what sort of layer it is — an engine-neutral kind from the registry (#288), not the
            MapLibre paint type — and carries the caller's name as its label, which is what a later export or a
            layer switcher needs.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .choropleth(polygons, column="pop", name="Population")
            .points(points)
        )
        fill, circle = m.layer_ids
        assert m._layer_tree.ids == (fill, circle), m._layer_tree.ids
        assert m._layer_tree.get(fill).kind == "choropleth", m._layer_tree.get(fill)
        assert m._layer_tree.get(fill).display_label == "Population", m._layer_tree.get(
            fill
        )
        assert m._layer_tree.get(circle).kind == "points", m._layer_tree.get(circle)

    @pytest.mark.parametrize(
        "method, fixture, kwargs, kind",
        [
            ("points", "points", {}, "points"),
            ("lines", "lines", {}, "lines"),
            ("polygons", "polygons", {}, "polygons"),
            ("choropleth", "polygons", {"column": "pop"}, "choropleth"),
            ("contours", "raster", {"levels": [0.3, 0.6]}, "contours"),
            (
                "contours",
                "raster",
                {"levels": [0.3, 0.6], "filled": True},
                "filled_contours",
            ),
        ],
        ids=[
            "points",
            "lines",
            "polygons",
            "choropleth",
            "contours",
            "filled-contours",
        ],
    )
    def test_the_vector_builders_record_engine_neutral_kinds(
        self, request, method, fixture, kwargs, kind
    ):
        """A vector layer's kind names what is drawn, not the MapLibre layer type drawing it (#288).

        Args:
            request: pytest's request, used to fetch the input fixture the builder needs.
            method: The `WebMap` builder under test.
            fixture: The fixture holding the builder's data.
            kwargs: Keyword arguments the builder needs.
            kind: The kind the builder must record.

        Test scenario:
            The tree recorded the MapLibre type, so `polygons`, `choropleth` and filled `contours` were all
            `fill`, `points` was `circle`, and contour lines were `line` — a figure could not tell them apart.
        """
        from digitalearth.web import WebMap

        m = getattr(WebMap().basemap(), method)(
            request.getfixturevalue(fixture), **kwargs
        )
        recorded = m._layer_tree.get(m.layer_ids[0]).kind
        assert recorded == kind, (
            f"{method} recorded kind {recorded!r}, expected {kind!r}"
        )

    def test_the_layer_ids_keep_their_engine_prefixes(self, points, polygons):
        """Changing the recorded kind leaves the public ids as they were: `fill-…` and `circle-…`."""
        from digitalearth.web import WebMap

        m = WebMap().polygons(polygons).points(points)
        prefixes = [layer_id.split("-")[0] for layer_id in m.layer_ids]
        assert prefixes == ["fill", "circle"], m.layer_ids

    def test_every_kind_the_tree_records_is_registered(self, points, polygons, raster):
        """No builder writes a kind the registry cannot look up (#288)."""
        from digitalearth.base.registry import kinds
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .add_raster(raster)
            .contours(raster, levels=[0.3], labels=True)
            .choropleth(polygons, column="pop")
            .points(points)
            .heatmap(points)
            .graticule()
            .text(4.9, 52.4, "Amsterdam")
        )
        unregistered = sorted(
            {m._layer_tree.get(layer).kind for layer in m.layer_ids} - set(kinds())
        )
        assert unregistered == [], unregistered

    @pytest.mark.parametrize(
        "method, fixture, args, kwargs, kind",
        [
            ("add_raster", "raster", (), {}, "raster"),
            ("rgb_composite", "raster", (), {"bands": (1, 1, 1)}, "rgb"),
            ("heatmap", "points", (), {}, "heatmap"),
            ("cluster", "points", (), {}, "clusters"),
            ("extrusion", "polygons", (), {"height": "pop"}, "extrusion"),
            ("text", None, (4.9, 52.4, "Amsterdam"), {}, "text"),
            ("graticule", None, (), {}, "graticule"),
            ("labels", "points", ("v",), {}, "labels"),
        ],
        ids=[
            "add_raster",
            "rgb_composite",
            "heatmap",
            "cluster",
            "extrusion",
            "text",
            "graticule",
            "labels",
        ],
    )
    def test_every_other_builder_records_the_kind_of_layer_it_adds(
        self, request, method, fixture, args, kwargs, kind
    ):
        """Each builder beyond the vector ones names its layer's kind in the tree.

        Args:
            request: pytest's request, used to fetch the input fixture the builder needs.
            method: The `WebMap` builder under test.
            fixture: The fixture holding the builder's data, or ``None`` for a builder that takes none.
            args: Positional arguments after the data.
            kwargs: Keyword arguments the builder needs.
            kind: The kind the builder must record.

        Test scenario:
            The vector builders record their paint type, asserted above. The raster, composite, big-data, 3-D and
            decoration builders each pass their own kind, and a wrong one would describe the layer as something
            it is not in any later export. `labels` builds a MapLibre symbol layer and records `"labels"`.
        """
        from digitalearth.web import WebMap

        inputs = () if fixture is None else (request.getfixturevalue(fixture),)
        m = getattr(WebMap().basemap(), method)(*inputs, *args, **kwargs)
        assert len(m.layer_ids) == 1, (
            f"{method} should index one layer; got {m.layer_ids}"
        )
        recorded = m._layer_tree.get(m.layer_ids[0])
        assert recorded.kind == kind, (
            f"{method} recorded kind {recorded.kind!r}, expected {kind!r}"
        )

    @pytest.mark.parametrize(
        "method, fixture, kwargs",
        [
            ("add_raster", "raster", {}),
            ("rgb_composite", "raster", {"bands": (1, 1, 1)}),
            ("points", "points", {}),
            ("lines", "lines", {}),
            ("polygons", "polygons", {}),
            ("choropleth", "polygons", {"column": "pop"}),
            ("labels", "points", {"column": "v"}),
            ("contours", "raster", {"levels": [0.3, 0.6], "labels": True}),
            ("graticule", None, {}),
        ],
        ids=[
            "add_raster",
            "rgb_composite",
            "points",
            "lines",
            "polygons",
            "choropleth",
            "labels",
            "contours",
            "graticule",
        ],
    )
    def test_a_layer_built_hidden_is_recorded_hidden(
        self, request, method, fixture, kwargs
    ):
        """Every builder that takes `visible=` records the layer's visibility in the tree, not only in MapLibre.

        Args:
            request: pytest's request, used to fetch the input fixture the builder needs.
            method: The `WebMap` builder under test.
            fixture: The fixture holding the builder's data, or ``None`` for a builder that takes none.
            kwargs: Keyword arguments the builder needs besides `visible`.

        Test scenario:
            `_index_layer` took no `visible`, so the tree described every layer as visible while the emitted
            MapLibre layout said ``visibility: none``. A layer switcher, an export or a reconciler reading the tree
            would have shown every hidden layer. `contours` with labels indexes two layers, and both must be hidden.
        """
        from digitalearth.web import WebMap

        inputs = () if fixture is None else (request.getfixturevalue(fixture),)
        m = getattr(WebMap().basemap(), method)(*inputs, visible=False, **kwargs)
        assert m.layer_ids, f"{method} indexed no layer"
        shown = [layer for layer in m.layer_ids if m._layer_tree.is_visible(layer)]
        assert shown == [], f"{method}(visible=False) recorded {shown} as visible"

    def test_a_layer_built_visible_is_recorded_visible(self, points):
        """The default builds a visible layer, and the tree says so.

        Test scenario:
            The other half of the hidden case: threading `visible` through must not turn the default into hidden.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points)
        assert m._layer_tree.is_visible(m.layer_ids[0]), m._layer_tree

    @pytest.mark.parametrize(
        "method, fixture, visible, drawn",
        [
            ("points", "points", 0, False),
            ("points", "points", 1, True),
            ("add_raster", "raster", None, False),
            ("graticule", None, 0, False),
        ],
        ids=["points-0", "points-1", "raster-none", "graticule-0"],
    )
    def test_a_non_boolean_visible_builds_and_is_recorded_as_drawn(
        self, request, method, fixture, visible, drawn
    ):
        """A builder given `visible=0`, `1` or `None` builds as it always did, and the tree records what it drew.

        Args:
            request: pytest's request, used to fetch the input fixture the builder needs.
            method: The `WebMap` builder under test.
            fixture: The fixture holding the builder's data, or ``None`` for a builder that takes none.
            visible: A non-boolean `visible=` value.
            drawn: Whether the builder draws the layer visible for that value — its truthiness.

        Test scenario:
            The builders decide the MapLibre layout by the value's truthiness, so `visible=0` built a hidden layer
            on `main`. Passing the value on to `LayerSpec`, which accepts only a real boolean, turned each of these
            working calls into "ValueError: LayerSpec visible must be True or False", naming a type the caller never
            used.
        """
        from digitalearth.web import WebMap

        inputs = () if fixture is None else (request.getfixturevalue(fixture),)
        m = getattr(WebMap().basemap(), method)(*inputs, visible=visible)
        recorded = [m._layer_tree.get(layer).visible for layer in m.layer_ids]
        assert recorded == [drawn], recorded

    def test_only_the_first_timeslider_frame_is_recorded_visible(self, raster_stack):
        """A raster timeslider builds its first frame visible and the rest hidden, and the tree records that.

        Test scenario:
            Every frame after the first is built with ``visible=False`` so a saved page shows one frame. The tree
            recorded all of them as visible.
        """
        from digitalearth.web import WebMap

        m = WebMap().timeslider(raster_stack)
        recorded = [m._layer_tree.is_visible(layer) for layer in m.layer_ids]
        assert recorded == [True, False, False], recorded

    def test_a_graticule_added_after_data_sits_beneath_it_in_the_tree(self):
        """The tree lists layers bottom first, so a graticule drawn under earlier data comes first in it too.

        Test scenario:
            `graticule` joins the reference band beneath the data, but was appended to the tree, so the tree read
            `('data-1', 'Graticule')` while the map drew `['Graticule', 'data-1']`. A renderer trusting the tree's
            order — its documented meaning — would draw the graticule over the data.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().text(4.9, 52.4, "A", name="data-1").graticule()
        assert list(m._layer_tree.ids) == _drawn(m), (m._layer_tree.ids, _drawn(m))
        assert _drawn(m) == ["Graticule", "data-1"], _drawn(m)

    def test_a_graticule_re_added_after_removing_one_still_sits_beneath_the_data(self):
        """Removing a graticule gives its place in the reference band back, so the next one lands under the data.

        Test scenario:
            `remove_layer` took the graticule off the list but left the reference-band count at one, so the next
            `add_reference` inserted one slot too high — above the data it should sit under — and the tree, which
            follows the band, would have agreed with the wrong order.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().text(4.9, 52.4, "A", name="data-1").graticule()
        m.remove_layer("Graticule").graticule()
        (graticule,) = [layer for layer in m.layer_ids if layer != "data-1"]
        assert _drawn(m) == [graticule, "data-1"], _drawn(m)
        assert list(m._layer_tree.ids) == _drawn(m), (m._layer_tree.ids, _drawn(m))

    @pytest.mark.parametrize("name", [" amsterdam", "amsterdam ", "   "])
    def test_a_padded_or_blank_layer_name_still_builds_a_layer(self, name):
        """A name the web tier accepted before its index became a `LayerTree` is still accepted, verbatim.

        Args:
            name: A padded or blank layer name.

        Test scenario:
            The tier uses the caller's name as the MapLibre id and as the label, unchanged. `LayerSpec` refused ids
            with surrounding whitespace and blank labels, so `text(..., name=" amsterdam")` raised a `ValueError` about
            `LayerSpec` — a regression on a call that had worked, naming a type the caller never used.
        """
        from digitalearth.web import WebMap

        m = WebMap().text(4.9, 52.4, "A", name=name)
        assert m.layer_ids == [name], m.layer_ids
        assert m._layer_tree.get(name).display_label == name, m._layer_tree.get(name)

    def test_a_map_with_a_layer_can_be_deep_copied(self):
        """`copy.deepcopy` of a map with a data layer works and gives an independent map.

        Test scenario:
            The layer index became a `LayerTree`, whose layers hold a `Symbology` with read-only mapping views that
            cannot be pickled; deep-copying any map with a layer then raised `cannot pickle 'mappingproxy' object`,
            where `main`'s list index copied fine.
        """
        import copy

        from digitalearth.web import WebMap

        original = WebMap().text(4.9, 52.4, "A", name="amsterdam")
        clone = copy.deepcopy(original)
        clone.remove_layer("amsterdam")
        assert original.layer_ids == ["amsterdam"], (
            "removing from the copy must not touch the original"
        )
        assert clone.layer_ids == [], clone.layer_ids

    def test_removing_a_layer_removes_it_from_the_tree(self, points):
        """`remove_layer` and the tree agree, so the description never outlives the layer."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points).points(points)
        first, second = m.layer_ids
        m.remove_layer(first)
        assert first not in m._layer_tree, m._layer_tree.ids
        assert m._layer_tree.ids == (second,), m._layer_tree.ids

    def test_a_basemap_is_not_a_data_layer(self, points):
        """The basemap is the ground; listing it among the toggles would invite turning the map off.

        Test scenario:
            Two basemaps and one data layer — the index must hold exactly the data layer, and nothing
            whose id came from the tile builder.
        """
        from digitalearth.web import WebMap

        assert WebMap().basemap().layer_ids == []
        m = WebMap().basemap().tiles("https://a/{z}/{x}/{y}.png").points(points)
        assert len(m.layer_ids) == 1, m.layer_ids
        assert not any("tiles" in layer_id for layer_id in m.layer_ids), m.layer_ids

    def test_removing_a_layer_drops_it_from_the_map_and_the_index(self, points):
        """A mistake used to mean starting over."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points).points(points)
        first, second = m.layer_ids
        before = len(m.layers)
        m.remove_layer(first)
        assert m.layer_ids == [second]
        assert len(m.layers) == before - 1, (
            "the layer is gone from the index but still drawn"
        )

    def test_removing_the_last_layer_moves_the_popup_target(self, points):
        """``popup``/``tooltip`` default to the most recent layer, which must not be a removed one."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points).points(points)
        m.remove_layer(m.layer_ids[-1])
        assert m._last_layer_id == m.layer_ids[-1]

    def test_removing_an_unknown_layer_says_which_exist(self, points):
        """A silent no-op looks exactly like a layer that refuses to go away."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points)
        with pytest.raises(KeyError, match="no layer"):
            m.remove_layer("circle-999")


class TestVisibility:
    """A layer that starts hidden is how a switcher gets something to turn on."""

    def test_a_hidden_layer_carries_the_layout_property(self, points):
        """MapLibre hides a layer through ``layout.visibility``, so that has to reach the spec."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().points(points, visible=False).to_html())
        assert '"visibility": "none"' in payload, payload[-400:]

    def test_a_visible_layer_says_nothing_about_visibility(self, points):
        """The default must not write a property that was never asked for."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().points(points).to_html())
        assert '"visibility": "none"' not in payload


class TestTheSwitchReachesTheViewer:
    """The point is a control in a saved page, not a Python-side toggle."""

    def test_the_switcher_offers_every_data_layer(self, points, polygons):
        """A switch that lists some layers silently strands the rest."""
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .choropleth(polygons, column="pop", name="Population")
            .points(points, name="Stations")
            .layer_control()
        )
        payload = _payload(m.to_html())
        assert "LayerSwitcherControl" in payload
        for layer_id in m.layer_ids:
            assert layer_id in payload, (
                f"{layer_id} is on the map but not in the switch"
            )

    def test_an_explicit_subset_is_honoured(self, points, polygons):
        """Offering a subset hides a layer from the switch without hiding it from the map.

        Test scenario:
            The excluded id is still on the map, so asserting it is absent from the *page* would fail on
            the layer itself. The switch's own ``layerIds`` list is what has to be checked.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(polygons, column="pop").points(points)
        keep, dropped = m.layer_ids[0], m.layer_ids[1]
        payload = _payload(m.layer_control(layer_ids=[keep]).to_html())
        assert f'"layerIds": ["{keep}"]' in payload, payload[-400:]
        assert dropped in payload, "the excluded layer should still be drawn"

    def test_a_switch_with_nothing_to_switch_is_refused(self):
        """An empty control in the corner explains nothing."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="nothing to switch"):
            web_map.layer_control()

    def test_an_unknown_id_is_refused(self, points):
        """A typo would produce a dead entry in the switch rather than an error."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap().points(points)
        with pytest.raises(ValueError, match="not on this map"):
            web_map.layer_control(layer_ids=["circle-999"])

    def test_a_bad_position_is_refused(self, points):
        """The four corners are MapLibre's; a typo is silently ignored by the browser."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap().points(points)
        with pytest.raises(ValueError):
            web_map.layer_control(position="middle")


class TestANamedLayerIsAddressableByItsName:
    """H1: the switcher captions each row with the layer id, so a name has to become the id."""

    def test_the_name_becomes_the_id(self, polygons):
        """Otherwise the caller's name is stored and never shown anywhere.

        Args:
            polygons: The fixture frame.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(polygons, column="pop", name="Population")
        assert m.layer_ids == ["Population"], m.layer_ids

    def test_a_repeated_name_is_uniquified(self, polygons, points):
        """Two layers cannot share a MapLibre id, and the second must still be addressable.

        Args:
            polygons: The fixture frame.
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .choropleth(polygons, column="pop", name="Layer")
            .points(points, name="Layer")
        )
        assert m.layer_ids == ["Layer", "Layer-2"], m.layer_ids

    def test_an_unnamed_layer_keeps_a_generated_id(self, points):
        """The generated ids stay the default, so nothing that worked before changes.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        assert WebMap().basemap().points(points).layer_ids[0].startswith("circle-")

    def test_removing_a_named_layer_works_by_name(self, polygons):
        """The name is the handle a caller would reach for.

        Args:
            polygons: The fixture frame.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(polygons, column="pop", name="Population")
        m.remove_layer("Population")
        assert m.layer_ids == []


class TestTheBigDataPathIsHonestAboutTheRegistry:
    """M7: a deck.gl overlay is not a MapLibre style layer, so the switcher cannot reach it."""

    @pytest.mark.parametrize("kwargs", [{"name": "Cities"}, {"visible": False}])
    def test_name_and_visible_are_refused_on_the_deck_path(self, points, kwargs):
        """Crossing the feature threshold must not silently change what the arguments do.

        Args:
            points: The fixture points.
            kwargs: The argument the deck path cannot honour.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="deck.gl overlay"):
            web_map.points(points, big=True, **kwargs)

    def test_the_maplibre_builders_do_join_the_registry(self, points):
        """A heatmap or a cluster is a style layer, so a viewer can switch it off.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        assert WebMap().basemap().heatmap(points).layer_ids, (
            "heatmap is not addressable"
        )
        assert WebMap().basemap().cluster(points).layer_ids, (
            "cluster is not addressable"
        )


class TestRemovingAStepKeepsTheMapRenderable:
    """H4: the time-slider config outlived the layers it pointed at."""

    def test_a_map_still_renders_after_a_step_is_removed(self, raster_stack):
        """The failure was a permanently unrenderable map, blaming a method never called.

        Args:
            raster_stack: A 3-member collection.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().timeslider(raster_stack)
        m.remove_layer(m.layer_ids[0])
        m.to_html()

    def test_dropping_below_two_steps_ends_the_series(self, raster_stack):
        """One step is not a series, so the slider config should not survive it.

        Args:
            raster_stack: A 3-member collection.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().timeslider(raster_stack)
        for layer_id in list(m.layer_ids)[:2]:
            m.remove_layer(layer_id)
        assert m._temporal is None, m._temporal


def _emitted_layer_ids(html):
    """Return the layer ids the page actually adds, in order.

    Args:
        html: A page from ``to_html``.

    Returns:
        The ids from the emitted ``addLayer`` calls.
    """
    import re

    return re.findall(r'\["addLayer", \[\{"id": "([^"]+)"', _payload(html))


class TestRemovalReachesThePage:
    """A removal that only updates the index leaves the layer drawn and unreachable."""

    @pytest.mark.parametrize("builder", ["points", "heatmap", "cluster"])
    def test_every_indexed_builder_is_really_removable(self, points, builder):
        """`remove_layer` filters `self.layers` on an attribute each builder has to set.

        Args:
            points: The fixture points.
            builder: The builder under test.

        Test scenario:
            heatmap, cluster and extrusion joined the registry without setting it, so removal reported
            success, dropped the entry from `layer_ids`, and left the layer on the map — neither
            removable nor switchable. Asserting `layer_ids` alone passes while that is true, so this
            asserts the emitted addLayer ids.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap()
        getattr(m, builder)(points)
        assert m.layer_ids, f"{builder} is not in the registry"
        before = _emitted_layer_ids(m.to_html())
        m.remove_layer(m.layer_ids[0])
        after = _emitted_layer_ids(m.to_html())
        assert len(after) < len(before), f"{builder} survived removal: {after}"
        assert m.layer_ids == []

    def test_a_cluster_removes_all_three_of_its_layers(self, points):
        """Bubbles, counts and loose points are one entry, so they go together.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().cluster(points)
        m.remove_layer(m.layer_ids[0])
        assert _emitted_layer_ids(m.to_html()) == ["tiles-2"], _emitted_layer_ids(
            m.to_html()
        )


class TestTheSwitcherFollowsTheLiveLayers:
    """The control is resolved when the widget is built, not when it was asked for."""

    def test_a_removed_layer_leaves_no_dead_row(self, points, polygons):
        """A row naming a removed layer logs an error in the browser and toggles nothing.

        Args:
            points: The fixture points.
            polygons: The fixture frame.
        """
        import re

        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .points(points, name="A")
            .polygons(polygons, column="pop", name="B")
            .layer_control()
        )
        m.remove_layer("A")
        rows = re.search(r'"layerIds": (\[[^\]]*\])', _payload(m.to_html()))
        assert rows.group(1) == '["B"]', rows.group(1)

    def test_asking_twice_does_not_stack_two_switchers(self, points):
        """Two identical panels in one corner is a bug, not a feature.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points, name="A").layer_control().layer_control()
        assert _payload(m.to_html()).count('"LayerSwitcherControl"') == 1

    def test_a_caller_switcher_replaces_the_automatic_one(self, raster_stack):
        """A temporal map that also asks for a switcher gets one, not two.

        Args:
            raster_stack: A 3-member collection.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().timeslider(raster_stack).layer_control()
        assert _payload(m.to_html()).count('"LayerSwitcherControl"') == 1

    def test_an_export_after_a_render_carries_no_picker(self, raster_stack):
        """The notebook path — look at the map, then export it — is the likely one.

        Args:
            raster_stack: A 3-member collection.

        Test scenario:
            While the control was materialised into the layer list, `with_controls=False` could only skip
            adding it again; a render had already put it there, so every GIF frame kept it.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().timeslider(raster_stack)
        m._build_map_widget()
        exported = m._build_map_widget(with_controls=False).to_html()
        assert exported[exported.rfind("var data = ") :].count('"addControl"') == 0


class TestIdsAreAllocatedOnce:
    """A caller name and a generated id come out of the same allocator."""

    def test_a_name_shaped_like_a_generated_id_does_not_collide(self, points):
        """Two layers sharing a MapLibre id makes addLayer drop the second with a console error.

        Args:
            points: The fixture points.

        Test scenario:
            `line-1` and `fill-3` are plausible names for reaches or basins, so this is reachable.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .points(points, name="circle-5")
            .points(points)
            .points(points)
        )
        assert len(m.layer_ids) == len(set(m.layer_ids)), m.layer_ids
        emitted = _emitted_layer_ids(m.to_html())
        assert len(emitted) == len(set(emitted)), emitted

    def test_a_third_collision_gets_a_third_suffix(self, polygons, points):
        """Three layers sharing a name must come out as three distinct, addressable ids.

        Args:
            polygons: The fixture frame.
            points: The fixture points.

        Test scenario:
            Two colliding names were covered; three were not, and `Layer-2` already being taken is the
            only state the suffix search exists for — a loop that never advanced would still have passed.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .choropleth(polygons, column="pop", name="Layer")
            .points(points, name="Layer")
            .points(points, name="Layer")
        )
        assert m.layer_ids == ["Layer", "Layer-2", "Layer-3"], m.layer_ids


class TestTheDeckRefusalNamesTheRealCause:
    """M7: the message blamed a `big=True` the caller may never have passed."""

    def test_the_threshold_route_says_so(self, points, monkeypatch):
        """Crossing the threshold is the surprising route, so the error has to name it.

        Args:
            points: The fixture points.
            monkeypatch: pytest's patcher, used to lower the threshold rather than build 50 000 features.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        monkeypatch.setattr(web_map, "big_data_threshold", 1)
        with pytest.raises(ValueError, match="threshold"):
            web_map.points(points, name="Cities")

    def test_the_explicit_route_does_not_mention_a_threshold(self, points):
        """`big=True` is the caller's own choice, so the threshold is irrelevant to them.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="deck.gl overlay") as err:
            web_map.polygons(points, big=True, name="Cities")
        assert "threshold" not in str(err.value), str(err.value)
