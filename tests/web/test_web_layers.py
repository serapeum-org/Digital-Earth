"""Addressing the layers on a web map (#188) — ids, visibility, removal and the viewer's switch.

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env. Assertions
read the exported page's call payload rather than the page as a whole, which also contains the MapLibre
library.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon


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
