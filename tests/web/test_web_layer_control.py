"""The one ``layer_control`` signature, on the web half of it (#264).

``layer_control`` existed on two tiers and shared **not one parameter** between them: this tier took
``(position, layer_ids, theme)`` and the interactive one ``(opacity, reorder, basemap_switch)``, so no caller
could add a layer control without knowing which backend it was talking to. The settled signature is the
layers to include, the position, and the controls to expose — with both tiers returning the map.

This tier already returned the map and already took ``position``, so what changes here is the *spelling*:
``layer_ids=`` becomes ``layers=``, the Tier-2 name the contract declares, and keeps working for one release.
``controls=`` arrives as the third shared name; this tier can only build visibility rows, and says so rather
than accepting a control it cannot draw.

Every check reads the **exported page** where it can, because the switch is a control in a saved file rather
than a Python-side flag: what the request records is only interesting if it reaches the viewer.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

pytest.importorskip("maplibre")

from digitalearth.web import WebMap  # noqa: E402  (after the engine guard)


@pytest.fixture
def points():
    """Return two points with a numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"v": [1, 2]}, geometry=[Point(0.0, 0.0), Point(1.0, 1.0)], crs="EPSG:4326"
    )


@pytest.fixture
def polygons():
    """Return two disjoint squares with a numeric column.

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
def two_layers(points, polygons):
    """Return a map carrying a choropleth and a point overlay over a basemap.

    Args:
        points: The point fixture.
        polygons: The polygon fixture.

    Returns:
        A ``WebMap`` with two data layers and no layer control yet.
    """
    return WebMap().basemap().choropleth(polygons, column="pop").points(points)


def _payload(html: str) -> str:
    """Return the exported page's call payload — what this map does, not what the library contains.

    Args:
        html: A page from ``to_html``.

    Returns:
        The substring from the last ``var data =`` to the end of the page.
    """
    marker = html.rfind("var data = ")
    assert marker != -1, "the exported page carries no call payload"
    return html[marker:]


class TestTheSharedSpelling:
    """``layers`` and ``position`` are the two names the contract declares, and both must reach the page."""

    def test_layers_offers_the_named_subset(self, two_layers):
        """``layers=`` is the Tier-2 spelling of the ids to offer, and it has to reach the switch.

        Args:
            two_layers: The map under test.

        Test scenario:
            The excluded id is still *on* the map — offering a subset hides a layer from the switch, not
            from the drawing — so the switch's own ``layerIds`` list is what has to be read, not the page
            as a whole.
        """
        keep, dropped = two_layers.layer_ids[0], two_layers.layer_ids[1]
        payload = _payload(two_layers.layer_control(layers=[keep]).to_html())
        assert f'"layerIds": ["{keep}"]' in payload, payload[-400:]
        assert dropped in payload, "the excluded layer should still be drawn"

    def test_the_map_comes_back_so_calls_chain(self, two_layers):
        """Both tiers return the map, which is the half of #264 that let a call be written once.

        Args:
            two_layers: The map under test.
        """
        assert two_layers.layer_control() is two_layers

    def test_an_unknown_layer_is_refused_under_the_new_name(self, two_layers):
        """A typo would leave a dead row in the saved switch rather than raise.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError, match="not on this map"):
            two_layers.layer_control(layers=["circle-999"])


class TestTheControlsThisTierCanOffer:
    """``controls=`` is the third shared name, and this tier can honour one entry of it."""

    def test_visibility_is_what_it_offers(self, two_layers):
        """Naming the control this tier draws changes nothing, which is what "supported" means.

        Args:
            two_layers: The map under test.
        """
        payload = _payload(two_layers.layer_control(controls=("visibility",)).to_html())
        assert "LayerSwitcherControl" in payload

    @pytest.mark.parametrize("control", ["opacity", "basemap"])
    def test_a_control_this_tier_cannot_draw_is_refused(self, two_layers, control):
        """Accepting one and ignoring it is how the other tier's flags came to look inert.

        Args:
            two_layers: The map under test.
            control: The control this tier has no widget for.
        """
        with pytest.raises(ValueError) as refused:
            two_layers.layer_control(controls=("visibility", control))
        assert "visibility" in str(refused.value), refused.value

    def test_a_control_no_tier_has_is_refused_by_the_vocabulary(self, two_layers):
        """A name outside the shared vocabulary is a typo, and reads as one.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError) as refused:
            two_layers.layer_control(controls=("visibilty",))
        assert "visibilty" in str(refused.value), refused.value


class TestTheOldSpellingIsAPromise:
    """``layer_ids=`` is in every script already written against this tier."""

    def test_the_old_name_still_offers_the_subset(self, two_layers, recwarn):
        """It forwards unchanged, so the page it produces is the page it always produced.

        Args:
            two_layers: The map under test.
            recwarn: pytest's warning recorder, so the deprecation does not escape the test.
        """
        keep = two_layers.layer_ids[0]
        payload = _payload(two_layers.layer_control(layer_ids=[keep]).to_html())
        assert f'"layerIds": ["{keep}"]' in payload, payload[-400:]

    def test_the_old_name_warns_and_names_the_new_one(self, two_layers):
        """A deprecated spelling that says nothing is one nobody stops writing.

        Args:
            two_layers: The map under test.
        """
        with pytest.warns(DeprecationWarning, match="layers"):
            two_layers.layer_control(layer_ids=two_layers.layer_ids[:1])

    def test_both_spellings_at_once_is_refused(self, two_layers):
        """They name one parameter, so preferring either would silently drop the other.

        Args:
            two_layers: The map under test.
        """
        offered = two_layers.layer_ids[:1]
        with pytest.raises(TypeError):
            two_layers.layer_control(layers=offered, layer_ids=offered)
