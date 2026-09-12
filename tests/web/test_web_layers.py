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
