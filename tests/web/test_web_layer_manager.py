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

    def test_a_flat_opacity_layer_is_rewritten(self):
        """A graticule carries a flat ``opacity`` prop (no paint dict), so that flat value is rewritten.

        Test scenario:
            Vector layers dim through a ``*-opacity`` paint key, but a graticule / raster field records its
            opacity flat in ``props["opacity"]``; ``layer_control(opacity=)`` has to reach that flat prop, not
            only the paint branch, or those reference layers could never be dimmed.
        """
        mapped = WebMap().graticule(name="grid")
        grid_id = mapped.layer_ids[0]
        dimmed = mapped.layer_control(opacity={grid_id: 0.3})
        assert dimmed.get_layer(grid_id).symbology.props["opacity"] == 0.3

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
        bad_id = two_layers.layer_ids[2]
        with pytest.raises(ValueError, match="opacity"):
            two_layers.layer_control(opacity={bad_id: 1.5})

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
        known_id = two_layers.layer_ids[1]
        with pytest.raises(ValueError, match="not on this map"):
            two_layers.layer_control(order=["circle-999", known_id])

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


class TestARefusalLeavesTheMapUntouched:
    """A bad opacity/order request raises with the map unchanged — the atomicity the method's comment claims."""

    def test_a_bad_opacity_target_leaves_the_earlier_one_undimmed(self, points):
        """A dimmable layer named before a non-dimmable one must not be dimmed when the call then refuses.

        Args:
            points: The point fixture.

        Test scenario:
            ``opacity`` is iterated in order, so a good id followed by a text layer (no opacity to dim) used to
            dim the good one through ``replace_layer`` and only then raise — leaving the map half-transformed
            with no switcher recorded. The whole request must be validated before any layer is touched.
        """
        mapped = (
            WebMap().basemap().points(points).text(4.9, 52.4, "Amsterdam", name="ams")
        )
        good_id = mapped.layer_ids[1]
        before = mapped.get_layer(good_id).symbology.props["paint"]["circle-opacity"]
        with pytest.raises(ValueError, match="cannot dim layer 'ams'"):
            mapped.layer_control(opacity={good_id: 0.3, "ams": 0.5})
        assert (
            mapped.get_layer(good_id).symbology.props["paint"]["circle-opacity"]
            == before
        )

    def test_a_bad_opacity_target_records_no_switcher(self, points):
        """The refused call must record no switcher furniture — no half-built control is left behind.

        Args:
            points: The point fixture.
        """
        mapped = (
            WebMap().basemap().points(points).text(4.9, 52.4, "Amsterdam", name="ams")
        )
        good_id = mapped.layer_ids[1]
        with pytest.raises(ValueError, match="cannot dim layer 'ams'"):
            mapped.layer_control(opacity={good_id: 0.3, "ams": 0.5})
        assert not [item for item in mapped._furniture if item.kind == "layer_switcher"]

    def test_a_band_crossing_order_leaves_the_earlier_opacity_untouched(
        self, two_layers
    ):
        """An ``order`` that crosses a band must not leave an ``opacity`` named in the same call applied.

        Args:
            two_layers: The map under test.

        Test scenario:
            ``opacity`` used to be applied before ``order`` was validated, so a band-crossing move raised with
            the opacity already baked in. Both halves must be validated before either is applied.
        """
        basemap_id, fill_id, circle_id = two_layers.layer_ids
        before = two_layers.get_layer(circle_id).symbology.props["paint"][
            "circle-opacity"
        ]
        with pytest.raises(IndexError, match="data band"):
            two_layers.layer_control(
                opacity={circle_id: 0.2}, order=[fill_id, basemap_id]
            )
        assert (
            two_layers.get_layer(circle_id).symbology.props["paint"]["circle-opacity"]
            == before
        )

    def test_a_band_crossing_order_records_no_switcher(self, two_layers):
        """The refused band-crossing call must record no switcher furniture.

        Args:
            two_layers: The map under test.
        """
        basemap_id, fill_id, circle_id = two_layers.layer_ids
        with pytest.raises(IndexError, match="data band"):
            two_layers.layer_control(
                opacity={circle_id: 0.2}, order=[fill_id, basemap_id]
            )
        assert not [
            item for item in two_layers._furniture if item.kind == "layer_switcher"
        ]


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
