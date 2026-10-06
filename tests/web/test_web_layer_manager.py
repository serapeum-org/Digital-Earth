"""WB-8 — the layer manager: ``layer_control`` drives opacity and ordering, not only visibility.

``layer_control`` was a visibility switcher and nothing else: it refused ``controls=("opacity",)`` because
py-maplibregl's ``LayerSwitcherControl`` builds visibility rows and no opacity slider, and it had no say over
draw order at all. The reorder/restyle machinery now exists on the map — ``move_layer``, ``replace_layer`` and
``set_visible`` — so ``layer_control`` can *apply* opacity and ordering as a one-time, declarative transform of
the stack, baked into the saved page, while the live switcher stays visibility-only.

These tests pin the declarative half, because that is the half this tier can actually honour:

* ``opacity={id: value}`` rebuilds each named layer's paint through ``replace_layer`` so the page draws it at
  that opacity;
* ``order=[...]`` reorders the named layers through ``move_layer`` so the page draws them in that order;
* the live opacity slider / drag-to-reorder control stays **X-2-blocked** — ``controls=("opacity",)`` is still
  refused, because accepting a control this tier cannot build is exactly the inertness #242/#244 removed.
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
        A ``WebMap`` with a basemap and two data layers in the ``data`` band, no layer control yet. Its
        ``layer_ids`` run ``[basemap, fill, circle]`` — the basemap sits first, in the ``underlay`` band.
    """
    return WebMap().basemap().choropleth(polygons, column="pop").points(points)


class TestOpacityIsAppliedDeclaratively:
    """``opacity=`` bakes each named layer's opacity into the page through ``replace_layer``."""

    def test_the_paint_opacity_is_rewritten(self, two_layers):
        """A choropleth's ``fill-opacity`` and a point layer's ``circle-opacity`` take the asked value.

        Args:
            two_layers: The map under test.

        Test scenario:
            The opacity has to reach the layer's own recorded paint, not a widget flag, because that paint is
            what the drawer rebuilds the MapLibre layer from — the saved page draws what the layer carries.
        """
        fill_id, circle_id = two_layers.layer_ids[1], two_layers.layer_ids[2]
        mapped = two_layers.layer_control(opacity={fill_id: 0.25, circle_id: 0.5})
        assert (
            mapped.get_layer(fill_id).symbology.props["paint"]["fill-opacity"] == 0.25
        )
        assert (
            mapped.get_layer(circle_id).symbology.props["paint"]["circle-opacity"]
            == 0.5
        )

    def test_the_map_comes_back_so_calls_chain(self, two_layers):
        """``layer_control`` returns the map whether or not it was asked to restyle (the chaining contract).

        Args:
            two_layers: The map under test.
        """
        assert (
            two_layers.layer_control(opacity={two_layers.layer_ids[2]: 0.3})
            is two_layers
        )

    def test_an_unknown_layer_is_refused(self, two_layers):
        """An opacity keyed by an id that is not on the map is a typo, and reads as one.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError, match="not on this map"):
            two_layers.layer_control(opacity={"circle-999": 0.5})

    def test_an_out_of_range_opacity_is_refused(self, two_layers):
        """Opacity is a fraction; anything outside ``[0, 1]`` would draw wrong without saying why.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError, match="opacity"):
            two_layers.layer_control(opacity={two_layers.layer_ids[2]: 1.5})

    def test_a_layer_with_no_opacity_paint_is_refused(self, points):
        """A text annotation carries no opacity paint, so asking to dim it is refused rather than ignored.

        Args:
            points: The point fixture (unused geometry; a text layer needs none).
        """
        mapped = WebMap().text(4.9, 52.4, "Amsterdam", name="ams").points(points)
        with pytest.raises(ValueError, match="opacity"):
            mapped.layer_control(opacity={"ams": 0.5})


class TestOrderingIsAppliedDeclaratively:
    """``order=`` reorders the named data layers through ``move_layer``."""

    def test_the_draw_order_follows_the_list(self, two_layers):
        """The listed bottom-to-top order has to reach both the tree and what the page draws.

        Args:
            two_layers: The map under test.

        Test scenario:
            ``layer_ids`` is the tree's order and ``[d.id for d in layers]`` is the draw order the widget is
            built from; a reorder that moved one and not the other would draw a stale map.
        """
        basemap_id, fill_id, circle_id = two_layers.layer_ids
        mapped = two_layers.layer_control(order=[circle_id, fill_id])
        assert mapped.layer_ids == [basemap_id, circle_id, fill_id]
        assert [drawn.id for drawn in mapped.layers] == [basemap_id, circle_id, fill_id]

    def test_an_unknown_layer_is_refused(self, two_layers):
        """A reorder naming an id that is not on the map is a typo, and reads as one.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError, match="not on this map"):
            two_layers.layer_control(order=["circle-999", two_layers.layer_ids[1]])

    def test_crossing_a_band_is_refused(self, two_layers):
        """A basemap is the ground; dragging a data layer beneath it is the one move the bands forbid.

        Args:
            two_layers: The map under test.

        Test scenario:
            The basemap sits in the ``underlay`` band and the data layers in ``data``; ordering a data layer
            into the underlay's slot is what ``move_layer`` refuses, and ``layer_control`` must surface that
            rather than swallow it.
        """
        basemap_id = two_layers.layer_ids[0]
        data_id = two_layers.layer_ids[1]
        # Listing the data layer first drives it to the basemap's slot in the underlay band, which is the
        # move that crosses — the basemap below it stays put.
        with pytest.raises(IndexError):
            two_layers.layer_control(order=[data_id, basemap_id])


class TestTheLiveControlStaysVisibilityOnly:
    """The switcher widget is visibility rows; a live opacity control is X-2-blocked, so still refused."""

    def test_a_live_opacity_control_is_still_refused(self, two_layers):
        """``controls=`` selects live switcher controls, and this tier builds no live opacity slider.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError, match="opacity"):
            two_layers.layer_control(controls=("visibility", "opacity"))

    def test_declarative_opacity_and_a_visibility_switcher_coexist(self, two_layers):
        """Dimming a layer and still listing a visibility switcher is one call, not a contradiction.

        Args:
            two_layers: The map under test.

        Test scenario:
            The opacity is baked into the layer while the switcher furniture is recorded, so the page both
            dims the layer and ships the visibility rows.
        """
        circle_id = two_layers.layer_ids[2]
        mapped = two_layers.layer_control(opacity={circle_id: 0.4})
        switcher = next(
            item for item in mapped._furniture if item.kind == "layer_switcher"
        )
        assert switcher.options["controls"] == ("visibility",)
        assert (
            mapped.get_layer(circle_id).symbology.props["paint"]["circle-opacity"]
            == 0.4
        )


class TestWhatTheControlRecords:
    """The applied opacity and order are recorded on the switcher furniture, so the declaration reads true."""

    def test_the_furniture_records_the_applied_opacity_and_order(self, two_layers):
        """A reader of the furniture sees what the manager did, not only that one was added.

        Args:
            two_layers: The map under test.
        """
        fill_id, circle_id = two_layers.layer_ids[1], two_layers.layer_ids[2]
        mapped = two_layers.layer_control(
            opacity={circle_id: 0.6}, order=[circle_id, fill_id]
        )
        switcher = next(
            item for item in mapped._furniture if item.kind == "layer_switcher"
        )
        assert switcher.options["opacity"] == ((circle_id, 0.6),)
        assert switcher.options["order"] == (circle_id, fill_id)
