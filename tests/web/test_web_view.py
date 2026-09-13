"""Framing a web map on its data (#186) — the accumulated extent, and who wins when views disagree.

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env. Nothing here
needs the network: the widget is built and inspected, never displayed.
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


class TestFramingInANonLonLatDisplayCrs:
    """M5: `fitBounds` takes degrees, and `WebMap(crs=)` is a supported public setting."""

    def test_a_projected_display_crs_is_converted_before_framing(self, boxes):
        """Handing metres to `fitBounds` frames the map on nothing, silently.

        Args:
            boxes: The fixture frame, in EPSG:4326.
        """
        from digitalearth.web import WebMap

        m = WebMap(crs=3857).basemap().choropleth(boxes.to_crs(3857), column="pop")
        view = m._map_view()
        assert view is not None, "a projected map lost its framing entirely"
        west, south, east, north = view["bounds"]
        assert -180.0 <= west <= 180.0 and -90.0 <= south <= 90.0, view
        assert round(west) == 4 and round(south) == 51, view

    def test_a_lonlat_display_crs_is_untouched(self, boxes):
        """The default path must not pay for a reprojection it does not need.

        Args:
            boxes: The fixture frame.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(boxes, column="pop")
        assert m._map_view()["bounds"] == [4.0, 51.0, 7.0, 53.0]


class TestAnAnnotationDoesNotDecideTheView:
    """M6: `text()` is decoration; a caption should not move the map."""

    def test_a_lone_annotation_does_not_frame_the_map(self):
        """A zero-area extent resolves to maximum zoom on a point."""
        from digitalearth.web import WebMap

        assert WebMap().basemap().text(4.9, 52.4, "Amsterdam")._map_view() is None

    def test_an_annotation_does_not_drag_the_extent(self, boxes):
        """One caption far away used to shrink the data to a speck — the bug fit_bounds cured.

        Args:
            boxes: The fixture frame, spanning (4, 51, 7, 53).
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .choropleth(boxes, column="pop")
            .text(-170.0, -80.0, "far away")
        )
        assert m._map_view()["bounds"] == [4.0, 51.0, 7.0, 53.0], m._map_view()


class TestRasterPlacementAndFramingAgree:
    """M5: the image source is positioned in lon/lat, so the framing must be the same numbers."""

    @pytest.fixture(autouse=True)
    def _need_pyramids(self):
        """Skip when pyramids is absent."""
        pytest.importorskip("pyramids")

    @pytest.mark.parametrize("crs", [4326, 3857])
    def test_the_corners_and_the_view_are_the_same_degrees(self, dataset, crs):
        """A projected display CRS framed on degrees while placing the image in metres.

        Args:
            dataset: The shared pyramids raster fixture.
            crs: The display CRS.
        """
        import re

        from digitalearth.web import WebMap

        payload = _payload(WebMap(crs=crs).add_raster(dataset).to_html())
        corners = re.search(r'"coordinates": \[\[([-\d.]+), ([-\d.]+)\]', payload)
        view = re.search(r'"fitBounds", \[\[([-\d.]+), ([-\d.]+)', payload)
        assert corners and view, payload[-400:]
        assert abs(float(corners.group(1)) - float(view.group(1))) < 0.01, (
            f"west differs between placement and framing at crs={crs}"
        )
        assert -180.0 <= float(corners.group(1)) <= 180.0, "corners are not lon/lat"


class TestTheViewChoiceIsRemembered:
    """N7: an explicit `zoom=2` is a choice, not the absence of one."""

    def test_an_explicit_default_zoom_still_counts_as_chosen(self, boxes):
        """Comparing by value cannot tell the caller's 2 from the default 2.

        Args:
            boxes: The fixture frame.
        """
        from digitalearth.web import WebMap

        m = WebMap(zoom=2).basemap().choropleth(boxes, column="pop")
        assert m._map_view() is None, "an explicitly-set view was overridden"

    def test_a_wrong_length_extent_names_the_argument(self):
        """A tuple-unpacking traceback points at the code, not at the call."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="west, south, east, north"):
            web_map.fit_bounds((1.0, 2.0, 3.0))


class TestFramingGivesUpRatherThanGuessing:
    """L5: both ways `_as_lonlat` can fail end in no framing at all, and neither was covered."""

    def test_an_unresolvable_crs_frames_nothing(self, boxes, monkeypatch):
        """A CRS pyproj cannot resolve leaves the extent unknown, not wrong.

        Args:
            boxes: The fixture frame.
            monkeypatch: pytest's patcher.
        """
        from digitalearth.web import WebMap, base as web_base

        def _refuse(*args, **kwargs):
            """Stand in for a CRS pyproj cannot resolve."""
            raise ValueError("unknown CRS")

        monkeypatch.setattr(web_base, "reproject_coordinates", _refuse)
        m = WebMap(crs=3857)
        m._note_bounds((1.0, 2.0, 3.0, 4.0))
        assert m._data_bounds is None

    def test_a_non_finite_reprojection_frames_nothing(self, monkeypatch):
        """pyproj answers `inf` for a point outside the projection's domain rather than raising.

        Args:
            monkeypatch: pytest's patcher.
        """
        from digitalearth.web import WebMap, base as web_base

        monkeypatch.setattr(
            web_base,
            "reproject_coordinates",
            lambda xs, ys, **kwargs: ([float("inf"), 1.0], [2.0, 3.0]),
        )
        m = WebMap(crs=3857)
        m._note_bounds((1.0, 2.0, 3.0, 4.0))
        assert m._data_bounds is None
