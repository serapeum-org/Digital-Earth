"""Framing a web map on its data (#186) — the accumulated extent, and who wins when views disagree.

Named ``test_web_*`` so the ``test-web`` pixi task collects it in the ``web`` env. Nothing here needs the
network: the widget is built and inspected, never displayed.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

#: Two disjoint boxes over the Netherlands, so a union is visibly wider than either.
WEST_BOX = Polygon([(4.0, 51.0), (5.0, 51.0), (5.0, 52.0), (4.0, 52.0)])
EAST_BOX = Polygon([(6.0, 52.0), (7.0, 52.0), (7.0, 53.0), (6.0, 53.0)])


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


@pytest.fixture
def boxes():
    """Two polygons with a known combined extent of (4, 51, 7, 53).

    Returns:
        A GeoDataFrame in EPSG:4326 carrying a numeric column.
    """
    return gpd.GeoDataFrame(
        {"pop": [1, 2]}, geometry=[WEST_BOX, EAST_BOX], crs="EPSG:4326"
    )


class TestTheExtentAccumulates:
    """Every builder contributes through the shared choke points, not one by one."""

    def test_a_vector_layer_records_its_extent(self, boxes):
        """`_display_gdf` is the single vector choke point, so one hook covers every vector builder.

        Test scenario:
            Two disjoint boxes must union to their combined extent, not to the last one added.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(boxes, column="pop")
        assert m._data_bounds == [4.0, 51.0, 7.0, 53.0], m._data_bounds

    def test_two_layers_union_rather_than_replace(self, boxes):
        """A map of several layers has to frame on all of them."""
        from digitalearth.web import WebMap

        far = gpd.GeoDataFrame(
            {"pop": [3]},
            geometry=[Polygon([(0.0, 40.0), (1.0, 40.0), (1.0, 41.0), (0.0, 41.0)])],
            crs="EPSG:4326",
        )
        m = WebMap().basemap().choropleth(boxes, column="pop").points(far)
        assert m._data_bounds == [0.0, 40.0, 7.0, 53.0], m._data_bounds

    def test_a_basemap_alone_records_nothing(self):
        """A tile basemap covers the planet, so it says nothing about where to look."""
        from digitalearth.web import WebMap

        assert WebMap().basemap()._data_bounds is None

    @pytest.mark.parametrize("bad", [None, "not-an-extent", (1.0, 2.0)])
    def test_an_unusable_extent_is_ignored(self, bad):
        """One unusable layer must not decide where the map looks.

        Args:
            bad: Something that is not a four-number extent.
        """
        from digitalearth.web import WebMap

        m = WebMap()
        m._note_bounds(bad)
        assert m._data_bounds is None, m._data_bounds

    def test_a_non_finite_extent_is_ignored(self):
        """An empty frame reports `inf` bounds, which would frame the map on nothing.

        Test scenario:
            `GeoDataFrame.total_bounds` of an empty frame is all `inf`; passing that to `fitBounds` puts
            the viewer nowhere.
        """
        from digitalearth.web import WebMap

        m = WebMap()
        m._note_bounds((float("-inf"), float("-inf"), float("inf"), float("inf")))
        assert m._data_bounds is None


class TestWhoDecidesTheView:
    """Explicit beats implicit: a caller who chose a view keeps it."""

    def test_data_frames_the_map_when_the_caller_said_nothing(self, boxes):
        """The common case — plot something and look at it.

        Test scenario:
            Before this, `WebMap().choropleth(gdf)` opened at zoom 2 on the whole world, with the data an
            invisible speck.
        """
        from digitalearth.web import WebMap

        view = WebMap().basemap().choropleth(boxes, column="pop")._map_view()
        assert view["bounds"] == [4.0, 51.0, 7.0, 53.0], view

    @pytest.mark.parametrize(
        "kwargs",
        [{"zoom": 7}, {"center": (5.0, 52.0)}, {"zoom": 7, "center": (5.0, 52.0)}],
    )
    def test_an_explicit_constructor_view_is_not_overridden(self, boxes, kwargs):
        """Passing `center` or a non-default `zoom` means "I have chosen the view".

        Args:
            kwargs: The view the caller asked for.
        """
        from digitalearth.web import WebMap

        m = WebMap(**kwargs).basemap().choropleth(boxes, column="pop")
        assert m._map_view() is None, m._map_view()

    def test_an_explicit_fit_bounds_wins_over_everything(self, boxes):
        """A direct request is the strongest signal there is."""
        from digitalearth.web import WebMap

        m = (
            WebMap(zoom=7)
            .basemap()
            .choropleth(boxes, column="pop")
            .fit_bounds((0.0, 0.0, 1.0, 1.0))
        )
        assert m._map_view()["bounds"] == [0.0, 0.0, 1.0, 1.0]

    def test_fit_bounds_with_no_data_says_so(self):
        """Doing nothing quietly would look exactly like the call being ignored."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="nothing to frame on"):
            web_map.fit_bounds()

    def test_fit_bounds_with_no_argument_uses_the_data(self, boxes):
        """The no-argument form is the one a caller reaches for after adding their data."""
        from digitalearth.web import WebMap

        m = WebMap(zoom=7).basemap().choropleth(boxes, column="pop").fit_bounds()
        assert m._map_view()["bounds"] == [4.0, 51.0, 7.0, 53.0]

    def test_padding_and_animate_reach_the_view(self, boxes):
        """Both are the caller's to set, so neither may be dropped on the way through."""
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .fit_bounds((0.0, 0.0, 1.0, 1.0), padding=64, animate=True)
        )
        assert (m._map_view()["padding"], m._map_view()["animate"]) == (64, True)


def _payload(html):
    """Return the page's call payload — the part the map is actually built from.

    The exported page inlines the whole MapLibre library, which mentions ``fitBounds`` in its own source.
    Searching the page as a whole therefore matches whether or not the map was framed; only the
    ``var data = {...}`` payload at the end says what this map does.

    Args:
        html: A page from ``to_html``.

    Returns:
        The payload substring, from ``var data =`` to the end of the page.
    """
    marker = html.rfind("var data = ")
    assert marker != -1, "the exported page carries no call payload"
    return html[marker:]


class TestTheFramingReachesTheMap:
    """A view that never leaves Python would not move anything."""

    def test_the_built_widget_is_framed(self, boxes):
        """`_build_map_widget` is shared by render and save, so framing there covers both."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().choropleth(boxes, column="pop").to_html())
        assert '"fitBounds"' in payload, "the exported page does not frame itself"
        assert "[4.0, 51.0, 7.0, 53.0]" in payload, payload[-300:]

    def test_a_map_with_a_chosen_view_emits_no_fit(self, boxes):
        """The absence matters as much as the presence — otherwise the override does nothing."""
        from digitalearth.web import WebMap

        html = (
            WebMap(zoom=7, center=(5.0, 52.0))
            .basemap()
            .choropleth(boxes, column="pop")
            .to_html()
        )
        assert '"fitBounds"' not in _payload(html), (
            "an explicitly-framed map was re-framed"
        )
