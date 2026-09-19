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
        Each registered layer's id, in `layers` order, skipping basemaps and controls, which carry none. A
        caller's own object carries no marker either, so it is looked up in the map's custom table by identity.
    """
    held = {id(obj): layer_id for layer_id, obj in web_map._custom.items()}
    ids = [
        getattr(layer, "_digitalearth_layer_id", None) or held.get(id(layer))
        for layer in web_map.layers
    ]
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

    def test_an_unregistered_kind_is_refused_before_anything_is_indexed(self):
        """A builder passing a kind the registry cannot look up fails, and the tree is left unchanged.

        Test scenario:
            The index would otherwise describe a layer with a name no renderer can resolve.
        """
        from digitalearth.web import WebMap

        m = WebMap()
        with pytest.raises(KeyError, match="no layer kind 'fill' is registered"):
            m._index_layer("fill-0", None, kind="fill")
        assert m._layer_tree.ids == (), m._layer_tree.ids

    def test_rekinding_to_an_unregistered_kind_keeps_the_recorded_kind(self, points):
        """`_rekind_layer` checks the kind first, so a refused name leaves the layer described as it was."""
        from digitalearth.web import WebMap

        m = WebMap().points(points)
        layer_id = m.layer_ids[0]
        with pytest.raises(KeyError, match="no layer kind 'dots' is registered"):
            m._rekind_layer(layer_id, "dots")
        assert m._layer_tree.get(layer_id).kind == "points", m._layer_tree.get(layer_id)

    def test_rekinding_keeps_the_label_visibility_and_place(self, points, polygons):
        """Only the kind changes: the layer keeps its label, visibility and position in draw order."""
        from digitalearth.web import WebMap

        m = WebMap().polygons(polygons, name="Areas", visible=False).points(points)
        first = m.layer_ids[0]
        m._rekind_layer(first, "choropleth")
        held = m._layer_tree.get(first)
        described = (held.kind, held.display_label, held.visible, m.layer_ids[0])
        assert described == ("choropleth", "Areas", False, first), described

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

    def test_data_added_after_a_label_is_drawn_beneath_it(self, points):
        """A place name stays over the data that arrives after it, and the tree says the same (#292).

        Args:
            points: The features drawn under the label.

        Test scenario:
            `text` joins the overlay band, which the tree bands by kind. The queue appended, so the points were
            drawn over the label while the tree listed the label on top — the two disagreed the moment a data
            layer followed a label.
        """
        from digitalearth.web import WebMap

        m = WebMap().text(4.9, 52.4, "A", name="label").points(points, name="obs")
        assert _drawn(m) == ["obs", "label"], _drawn(m)
        assert list(m._layer_tree.ids) == _drawn(m), (m._layer_tree.ids, _drawn(m))

    def test_labels_from_a_column_join_the_same_band(self, points):
        """`labels` is text too, so it is drawn over data added afterwards.

        Args:
            points: The features the labels are taken from.
        """
        from digitalearth.web import WebMap

        m = WebMap().labels(points, column="v", name="names").points(points, name="obs")
        assert _drawn(m) == ["obs", "names"], _drawn(m)

    def test_removing_a_label_gives_its_place_back(self, points):
        """The overlay band is counted, so removing a label does not strand the count.

        Args:
            points: The features drawn under the label.

        Test scenario:
            The mirror of the reference-band regression: with the count left at one, the next data layer would be
            queued one place too low — beneath a label that is no longer there.
        """
        from digitalearth.web import WebMap

        m = WebMap().text(4.9, 52.4, "A", name="label").points(points, name="obs")
        m.remove_layer("label").points(points, name="more")
        assert _drawn(m) == ["obs", "more"], _drawn(m)
        assert list(m._layer_tree.ids) == _drawn(m), (m._layer_tree.ids, _drawn(m))

    def test_add_overlay_queues_above_a_layer_added_later(self):
        """The low-level entry points band the queue the way the builders do.

        Test scenario:
            Any object stands in for a layer here — the queue does not inspect it.
        """
        from digitalearth.web import WebMap

        m = WebMap().add_overlay("label").add_layer("data").add_underlay("tiles")
        assert m.layers == ["tiles", "data", "label"], m.layers

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


class TestTheSliderKeepsOneFramePerLayer:
    """Review M9 — a control drawn from the description must not offer a stop that names nothing."""

    def test_a_removed_layer_takes_its_frame_with_it(self, raster_stack):
        """A slider has one stop per layer, and it has to stay that way.

        Args:
            raster_stack: Three rasters the slider steps through.

        Test scenario:
            The measured defect: `layers` was trimmed and `frames` was not, so a control drawn from the
            description offered three stops for two layers — the third naming nothing (review M9).
        """
        from digitalearth.web import WebMap

        m = WebMap().timeslider(raster_stack, labels=["a", "b", "c"])
        slider = m.figure_spec.panels[0].furniture[0]
        assert len(slider.options["frames"]) == len(slider.options["layers"]), slider
        m.remove_layer(m.layer_ids[0])
        slider = m.figure_spec.panels[0].furniture[0]
        assert len(slider.options["frames"]) == len(slider.options["layers"]), slider


class TestAPopupOverAnUndescribedLayer:
    """Review H3 — not every MapLibre layer is a described one, and a popup over one is still a popup."""

    def test_a_clustered_map_can_be_given_a_popup(self, points):
        """`cluster(...).popup(...)` is a documented recipe, and it raised.

        Args:
            points: The fixture points.

        Test scenario:
            A cluster draws three MapLibre layers under one tree entry, and `cluster` left the *unclustered*
            id as the one a following `popup()` would default to — an id guaranteed absent from the tree.
        """
        from digitalearth.web import WebMap

        m = WebMap().cluster(points)
        assert m.popup(["v"]) is not None, "a clustered map must take a popup"
        assert m.layer_ids == ["clusters-2"], m.layer_ids

    def test_a_popup_naming_a_layer_this_map_has_not_drawn_is_refused(self, points):
        """An id a caller wrote is checked; skipping it accepted a typo in silence.

        Args:
            points: The fixture points.

        Test scenario:
            The guard exists for the ids the *tier* chooses — a cluster's sub-layers, a graticule's labels
            — which are drawn and simply not described. Applying it to an explicit `layer=` meant the
            closure was still queued, so the saved page called `map.on('click', 'typoed', ...)` and failed
            in a browser console with nothing said on the Python side (review H7).
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs")
        with pytest.raises(KeyError, match="no layer 'obs-clusters' on this map"):
            m.popup(["v"], layer="obs-clusters")

    def test_a_layer_with_both_records_both(self, points):
        """A click popup and a hover tooltip are two interactions, and a layer may carry both.

        Args:
            points: The fixture points.

        Test scenario:
            The measured defect: both were drawn — three applies queued — but one `tooltip` encoding held
            them, so the second call replaced the first and the popup's own fields were lost from the
            figure (review M10).
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs").popup(["v"]).tooltip(["v", "geometry"])
        bound = m.figure_spec.layers.get("obs").symbology.props["interactions"]
        assert tuple(bound["click"]) == ("v",), bound
        assert tuple(bound["hover"]) == ("v", "geometry"), bound

    def test_the_channel_carries_the_hover_fields_when_there_are_both(self, points):
        """One hover slot, and the hover's fields are what belongs in it.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs").popup(["v"]).tooltip(["geometry"])
        symbology = m.figure_spec.layers.get("obs").symbology
        assert symbology.encodings["tooltip"].resolve() == ("geometry",), symbology
        assert symbology.props["tooltip_trigger"] == "hover", symbology.props

    def test_a_cluster_s_popup_goes_on_the_layer_carrying_the_columns(self, points):
        """A cluster draws bubbles, their counts, and the loose points; only one has the caller's data.

        Args:
            points: The fixture points.

        Test scenario:
            The bubbles are aggregates whose only properties are `cluster`, `cluster_id` and the point
            counts. Pointing the popup at them to satisfy the tree lookup produced an empty popup for
            every feature a caller clicked (review H8).
        """
        import inspect

        from digitalearth.web import WebMap

        m = WebMap().cluster(points)
        m.popup(["v"])
        bound = [
            held.get("layer_id")
            for held in (
                inspect.getclosurevars(item).nonlocals
                for item in m.layers
                if callable(item)
            )
            if "layer_id" in held
        ]
        assert bound == ["unclustered-4"], bound

    def test_a_described_layer_still_records_what_pops_up(self, points):
        """The guard skips the description, and must not skip it for a layer that has one.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs")
        m.popup(["v"])
        shown = m.figure_spec.layers.get("obs").symbology.encodings["tooltip"]
        assert shown.resolve() == ("v",), shown


class TestAColourKeyDescribesTheLayerItNames:
    """Review H4 — `colorbar(layer_id)` used the id only to validate, then drew someone else's key."""

    @staticmethod
    def _squares(column, values):
        """Return two squares carrying `values` in `column`.

        Args:
            column: The column name.
            values: Two numbers.

        Returns:
            A GeoDataFrame.
        """
        return gpd.GeoDataFrame(
            {column: values},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs=4326,
        )

    def test_each_layer_s_key_shows_its_own_range(self):
        """The measured defect: both keys ended 500/900 — layer B's — one of them labelled "A"."""
        from digitalearth.web import WebMap

        m = WebMap().choropleth(self._squares("pop", [1, 100]), column="pop", name="A")
        m.choropleth(self._squares("rain", [500, 900]), column="rain", name="B")
        m.colorbar("A", label="People")
        drawn = m._panels["legend"][0]
        assert ">1<" in drawn and ">100<" in drawn.replace(">1<", ""), drawn

    def test_the_other_layer_still_shows_its_own(self):
        """The same call for B is B's range, so this is selection rather than a reversed default."""
        from digitalearth.web import WebMap

        m = WebMap().choropleth(self._squares("pop", [1, 100]), column="pop", name="A")
        m.choropleth(self._squares("rain", [500, 900]), column="rain", name="B")
        m.colorbar("B", label="Rain")
        drawn = m._panels["legend"][0]
        assert ">500<" in drawn, drawn

    def test_a_layer_with_no_classification_says_so(self, points):
        """An unclassified layer has no ramp; naming it is answered rather than substituted.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        m = WebMap().choropleth(self._squares("pop", [1, 100]), column="pop", name="A")
        m.points(points, name="plain")
        with pytest.raises(ValueError, match="was not drawn with a classification"):
            m.colorbar("plain")


class TestACustomLayerIsAddedUnderTheIdItIsGiven:
    """Review H5 — the tree was renamed and the MapLibre object was not."""

    def test_a_colliding_custom_id_is_refused_without_touching_the_object(self, points):
        """What `layer_ids` advertises has to be what `addLayer` sends — and the object is the caller's.

        Args:
            points: The fixture points.

        Test scenario:
            Suffixing left the map naming `obs-2` while the object said `obs`. Rewriting the object closed
            that and opened a worse one: the object may already be on another map, which then advertised an
            id its own page never adds (review H6). A collision is refused, and nothing is mutated.
        """
        from maplibre.layer import Layer, LayerType

        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs")
        own = Layer(id="obs", type=LayerType.CIRCLE, source="s")
        with pytest.raises(ValueError, match="is already on this map"):
            m.add_layer(own)
        assert own.id == "obs", own.id
        assert m.layer_ids == ["obs"], m.layer_ids

    def test_a_layer_held_by_another_map_is_left_alone(self, points):
        """The measured defect: adding to a second map re-pointed the first map's layer.

        Args:
            points: The fixture points.
        """
        from maplibre.layer import Layer, LayerType

        from digitalearth.web import WebMap

        own = Layer(id="obs", type=LayerType.CIRCLE, source="s")
        first = WebMap().add_layer(own)
        second = WebMap().points(points, name="obs")
        with pytest.raises(ValueError, match="is already on this map"):
            second.add_layer(own)
        assert own.id == "obs", own.id
        assert first.layer_ids == ["obs"], first.layer_ids

    def test_one_object_cannot_be_added_twice(self, points):
        """One object is one layer on the page, and the queue cannot tell two registrations apart.

        Args:
            points: The fixture points.

        Test scenario:
            The measured defect: `add_layer(obj, name="a").add_layer(obj, name="b")` gave two tree entries
            matched by identity, so `remove_layer("a")` dropped both from the queue — the map still listed
            `b` and nothing drew it (review M13). With the allocated id written onto the object, the first
            entry also named an id the object no longer carried.
        """
        from maplibre.layer import Layer, LayerType

        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs")
        own = Layer(id="mine", type=LayerType.CIRCLE, source="s")
        m.add_layer(own)
        with pytest.raises(ValueError, match="already on the map as 'mine'"):
            m.add_layer(own)
        assert m.layer_ids == ["obs", "mine"], m.layer_ids

    def test_two_objects_are_removed_independently(self, points):
        """The ordinary case: two layers, and removing one leaves the other drawn.

        Args:
            points: The fixture points.
        """
        from maplibre.layer import Layer, LayerType

        from digitalearth.web import WebMap

        m = WebMap()
        m.add_layer(Layer(id="x", type=LayerType.CIRCLE, source="s"), name="x")
        m.add_layer(Layer(id="y", type=LayerType.CIRCLE, source="s"), name="y")
        m.remove_layer("x")
        assert m.layer_ids == ["y"], m.layer_ids
        assert len(m.layers) == 1, m.layers

    def test_an_uncontested_id_is_left_as_it_was(self, points):
        """A caller's own id survives when nothing is claiming it.

        Args:
            points: The fixture points.
        """
        from maplibre.layer import Layer, LayerType

        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs")
        own = Layer(id="mine", type=LayerType.CIRCLE, source="s")
        m.add_layer(own)
        assert own.id == "mine", own.id
        assert "mine" in m.layer_ids, m.layer_ids


class TestOneMapsSourcesAreItsOwn:
    """#296 / review H2 — the object registry is process-global; a figure's sources must not be shared."""

    def test_two_maps_in_one_session_keep_their_own_data(self, points):
        """Every map restarts its layer numbering, so two of them mint the same id.

        Args:
            points: The fixture points.

        Test scenario:
            The measured defect: both maps registered under `circle-2`, the second replaced the first, and
            map A's stored figure then described map B's data — in a notebook, where two maps in one
            session is the ordinary case.
        """
        from digitalearth.web import WebMap

        other = points.iloc[:1].copy()
        first = WebMap().points(points)
        second = WebMap().points(other)
        assert first.layer_ids == second.layer_ids, (
            "the ids must collide for this to mean anything"
        )
        assert len(first.figure_spec.sources[first.layer_ids[0]].open()) == len(
            points
        ), "the first map's source must still be the data it was given"
        assert len(second.figure_spec.sources[second.layer_ids[0]].open()) == len(
            other
        ), "and the second map's must be its own"

    def test_a_closed_map_lets_its_data_go(self, points):
        """A map that is finished with lets go of the data it registered.

        Args:
            points: The fixture points.

        Test scenario:
            The registry holds strong references, and namespacing made this *worse* — every layer got its
            own permanent key instead of colliding on one (review M2). `close()` is the caller saying they
            are finished with the data; a `with` block says it for them.
        """
        from digitalearth.base.registry import _OBJECTS
        from digitalearth.web import WebMap

        before = len(_OBJECTS)
        for _ in range(5):
            with WebMap() as m:
                m.points(points, name="obs")
        assert len(_OBJECTS) == before, (
            f"five closed maps left {len(_OBJECTS) - before} entries behind"
        )

    def test_an_open_map_still_resolves_its_sources(self, points):
        """Closing is the caller's decision, and until they make it the figure reads.

        Args:
            points: The fixture points.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs")
        assert m.figure_spec.sources["obs"].open() is not None, "still readable"

    def test_a_removed_layer_lets_its_data_go(self, points):
        """The registry holds strong references, so a drawn-and-removed layer would leak one.

        Args:
            points: The fixture points.
        """
        from digitalearth.base.registry import _OBJECTS
        from digitalearth.web import WebMap

        before = len(_OBJECTS)
        m = WebMap().points(points)
        m.remove_layer(m.layer_ids[0])
        assert len(_OBJECTS) == before, (
            f"the registry grew by {len(_OBJECTS) - before} after a draw and a remove"
        )


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


class TestACallersOwnLayer:
    """A MapLibre layer the caller built is addressable, like every other layer (#293)."""

    @staticmethod
    def _layer(layer_id):
        """Return a minimal MapLibre layer.

        Args:
            layer_id: The id the layer carries.

        Returns:
            A `maplibre` `Layer`.
        """
        from maplibre import Layer, LayerType

        return Layer(id=layer_id, type=LayerType.CIRCLE, source=f"{layer_id}-src")

    def test_a_layer_the_caller_built_is_addressable(self):
        """The reproduction from the issue: `layer_ids` listed nothing and `remove_layer` raised.

        Test scenario:
            The object never reached the tree, so a caller could add a layer and then not refer to it again.
        """
        from digitalearth.web import WebMap

        m = WebMap().add_layer(self._layer("wells"))
        assert m.layer_ids == ["wells"], m.layer_ids
        assert m.remove_layer("wells").layer_ids == [], m.layer_ids

    def test_removing_it_takes_the_object_off_the_map_too(self):
        """A `maplibre` `Layer` carries no marker attribute, so removal matches it by identity."""
        from digitalearth.web import WebMap

        m = WebMap().add_layer(self._layer("wells"))
        m.remove_layer("wells")
        assert m.layers == [], m.layers

    def test_it_is_recorded_as_a_custom_layer_of_its_engine(self):
        """The kind names the engine that built the object, which is what a renderer reads."""
        from digitalearth.web import WebMap

        m = WebMap().add_layer(self._layer("wells"))
        assert m._layer_tree.get("wells").kind == "custom:maplibre", m._layer_tree.get(
            "wells"
        )

    def test_a_name_that_contradicts_the_object_s_id_is_refused(self):
        """The object's id is what `addLayer` sends, so a different `name=` would name nothing.

        Test scenario:
            `name=` used to win, and the object kept its own id — so the map advertised one id while the
            page added another. Rewriting the object instead reached into every other map holding it
            (review H6), so the two are simply required to agree.
        """
        from digitalearth.web import WebMap

        with pytest.raises(
            ValueError, match="draws this layer under the id it carries"
        ):
            WebMap().add_layer(self._layer("wells"), name="Boreholes")

    def test_a_name_matching_the_object_s_id_is_accepted(self):
        """Saying the same thing twice is not a contradiction."""
        from digitalearth.web import WebMap

        m = WebMap().add_layer(self._layer("wells"), name="wells")
        assert m.layer_ids == ["wells"], m.layer_ids

    def test_an_unnamed_object_gets_a_generated_id(self):
        """Anything without an id of its own is still addressable."""
        from digitalearth.web import WebMap

        m = WebMap().add_layer(lambda widget: None)
        assert m.layer_ids == ["custom-1"], m.layer_ids

    def test_a_repeated_id_is_refused_rather_than_suffixed(self):
        """Two layers cannot share an id, and the object carries the one the page will use.

        Test scenario:
            Suffixing gave the tree `wells-2` while the object still said `wells`, so MapLibre dropped the
            second layer with a console error and the map named one that was never added (review H5/H6).
        """
        from digitalearth.web import WebMap

        m = WebMap().add_layer(self._layer("wells"))
        with pytest.raises(ValueError, match="is already on this map"):
            m.add_layer(self._layer("wells"))
        assert m.layer_ids == ["wells"], m.layer_ids

    def test_a_band_puts_it_under_the_data(self, points):
        """A caller's own ground cover is drawn first, and the tree says so.

        Args:
            points: The features drawn over it.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .points(points, name="obs")
            .add_layer(self._layer("tiles"), band="underlay")
        )
        assert _drawn(m) == ["tiles", "obs"], _drawn(m)
        assert list(m._layer_tree.ids) == _drawn(m), (m._layer_tree.ids, _drawn(m))

    def test_a_band_no_tier_could_draw_is_refused(self):
        """A misspelt band names the four that work."""
        from digitalearth.web import WebMap

        drawn = WebMap()
        wells = self._layer("wells")
        with pytest.raises(ValueError, match="band must be one of"):
            drawn.add_layer(wells, band="middle")

    def test_a_builder_s_own_layer_is_not_recorded_twice(self, points):
        """The package's builders describe what they draw, so they do not come through this entry point.

        Args:
            points: The features drawn.

        Test scenario:
            `points` records one `points` layer. Were the builder still queueing through `add_layer`, the same
            layer would also be recorded as a caller's own object, under a second id.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points, name="obs")
        assert m.layer_ids == ["obs"], m.layer_ids
        assert m._layer_tree.get("obs").kind == "points", m._layer_tree.get("obs")
