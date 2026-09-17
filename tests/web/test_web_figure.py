"""The web tier's description: a `FigureSpec` the map writes as it is built (DE-23b + U-1, #296).

The tier's figure was 18 `apply(widget)` closures and eighteen attributes beside them: the layer tree held ids
and labels but no source and no style, controls were closures nobody could name, a popup outlived the layer it
was bound to, and nothing said what the map was looking at. These cover the description that replaces that —
the layers with their sources, the view, the panel's title and furniture, and the tier's capability declaration.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

pytest.importorskip("maplibre")

from dataclasses import replace  # noqa: E402

from digitalearth.base.capabilities import Capabilities  # noqa: E402
from digitalearth.base.spec import DataRef, FigureSpec, Viewport  # noqa: E402
from digitalearth.web import WebMap  # noqa: E402
from digitalearth.web.capabilities import CAPABILITIES  # noqa: E402


@pytest.fixture
def points() -> gpd.GeoDataFrame:
    """Return two points with a numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"pop": [1, 9]},
        geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
        crs="EPSG:4326",
    )


@pytest.fixture
def polygons() -> gpd.GeoDataFrame:
    """Return one square with a numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"pop": [3]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )


class TestTheFigureAMapWrites:
    """`WebMap.figure_spec` is the map as data."""

    def test_a_layer_carries_its_source(self, points):
        """A layer names where its data came from, which the tree could not say before.

        Args:
            points: The features drawn.

        Test scenario:
            `_index_layer` wrote an id, a kind, a label and a visibility — the data lived in a closure, so a
            figure could describe a layer without being able to say what it drew.
        """
        figure = WebMap().points(points, name="obs").figure_spec
        layer = figure.layers.get("obs")
        assert layer.source_id == "obs", layer.to_dict()
        assert figure.sources["obs"].open() is not None, dict(figure.sources)

    def test_the_figure_is_valid_and_lists_its_layers_on_the_panel(self, points):
        """One panel, `"main"`, showing the layers in draw order."""
        figure = WebMap().points(points, name="obs").text(4.9, 52.4, "A", name="label")
        panel = figure.figure_spec.panels[0]
        assert (panel.id, panel.layers) == ("main", ("obs", "label")), panel.to_dict()

    def test_a_removed_layer_leaves_the_figure(self, points):
        """What a map no longer draws is not in the description it hands out."""
        m = WebMap().points(points, name="obs")
        m.remove_layer("obs")
        figure = m.figure_spec
        assert (figure.layers.ids, dict(figure.sources)) == ((), {}), figure.to_dict()


class TestTheView:
    """Where the map is looking is a `Viewport`, not four attributes."""

    def test_the_centre_and_zoom_are_the_view(self):
        """A map built to look somewhere says so in its description."""
        view = WebMap(center=(4.9, 52.4), zoom=7).viewport
        assert (view.center, view.zoom) == ((4.9, 52.4), 7.0), view.to_dict()

    def test_the_view_is_in_the_crs_the_data_is_placed_in(self):
        """EPSG:4326 — what `WebMap(crs=)` accepts — not the projection MapLibre draws."""
        assert WebMap().viewport.crs == 4326, WebMap().viewport.to_dict()

    def test_fitting_bounds_frames_the_view(self):
        """`fit_bounds` is a region, which is what `Viewport.bounds` holds."""
        view = WebMap().fit_bounds([3.0, 50.0, 7.0, 54.0]).viewport
        assert view.bounds.as_bbox() == [3.0, 50.0, 7.0, 54.0], view.to_dict()

    def test_the_globe_projection_is_part_of_the_view(self):
        """`globe()` switches the projection, which the view records."""
        assert WebMap().globe().viewport.globe is True, "the view must say so"

    def test_the_view_round_trips(self):
        """A stored view reads back as the view it was written from."""
        view = WebMap(center=(4.9, 52.4), zoom=7).viewport
        assert Viewport.from_dict(view.to_dict()) == view, view.to_dict()


class TestTheFurniture:
    """Controls are anchored to the frame, so they are the panel's furniture (#292)."""

    @pytest.mark.parametrize(
        "build, kind",
        [
            (lambda m: m.navigation(), "navigation"),
            (lambda m: m.scale_bar(), "scale_bar"),
            (lambda m: m.fullscreen(), "fullscreen"),
            (lambda m: m.measure(), "measure"),
        ],
    )
    def test_each_control_is_recorded(self, build, kind):
        """A control a caller asked for is named in the description.

        Args:
            build: The builder call under test.
            kind: The furniture kind it records.
        """
        panel = build(WebMap()).figure_spec.panels[0]
        assert [item.kind for item in panel.furniture] == [kind], panel.to_dict()

    def test_a_control_carries_the_corner_it_sits_in(self):
        """The anchor is the MapLibre position, which is one of the four corners."""
        panel = WebMap().scale_bar(position="bottom-right").figure_spec.panels[0]
        assert panel.furniture[0].anchor == "bottom-right", panel.to_dict()

    def test_asking_twice_records_one_control(self):
        """MapLibre would draw a second control in the same corner; the description says one."""
        panel = WebMap().navigation().navigation().figure_spec.panels[0]
        assert len(panel.furniture) == 1, panel.to_dict()

    def test_the_switcher_lists_the_layers_it_switches(self, points):
        """The switcher is furniture whose options say what it controls.

        Args:
            points: The features drawn.
        """
        m = WebMap().points(points, name="obs").layer_control()
        (item,) = [
            entry
            for entry in m.figure_spec.panels[0].furniture
            if entry.kind == "layer_switcher"
        ]
        assert item.options["layers"] == ("obs",), item.to_dict()

    def test_a_title_is_the_panel_s(self):
        """A heading belongs to the panel, not to a floating HTML box beside it."""
        assert WebMap().title("Rainfall").figure_spec.panels[0].title == "Rainfall", (
            "the panel must carry the title"
        )


class TestPerLayerInteraction:
    """A popup and a tooltip are the layer's `tooltip` channel (#292)."""

    def test_a_tooltip_is_recorded_on_its_layer(self, polygons):
        """The fields shown travel with the layer they are bound to.

        Args:
            polygons: The features drawn.
        """
        m = WebMap().choropleth(polygons, column="pop", name="area").tooltip(["pop"])
        layer = m.figure_spec.layers.get("area")
        assert layer.symbology.encoding("tooltip").resolve() == ("pop",), (
            layer.to_dict()
        )

    def test_a_popup_and_a_tooltip_differ_by_trigger(self, polygons):
        """One channel, two triggers: a click and a hover.

        Args:
            polygons: The features drawn.
        """
        m = WebMap().choropleth(polygons, column="pop", name="area").popup(["pop"])
        layer = m.figure_spec.layers.get("area")
        assert layer.symbology.props["tooltip_trigger"] == "click", layer.to_dict()

    def test_removing_the_layer_removes_its_popup(self, polygons):
        """The page no longer pops up over a layer that is not there.

        Test scenario:
            The probed defect: after `points(name="a").popup(layer="a").remove_layer("a")` the page still
            emitted `addPopup`. The popup is tagged with the layer it belongs to, so it goes with it.

        Args:
            polygons: The features drawn.
        """
        m = WebMap().choropleth(polygons, column="pop", name="area").popup(["pop"])
        m.remove_layer("area")
        assert m.layers == [], m.layers

    def test_removing_a_slider_layer_forgets_the_slider(self, points):
        """A control that stepped through a layer that is gone is not kept.

        Args:
            points: The features the slider steps through.
        """
        frames = points.assign(t=["2020", "2021"])
        m = WebMap().timeslider(frames, kdim="t")
        (layer_id,) = m.layer_ids
        m.remove_layer(layer_id)
        assert m.figure_spec.panels[0].furniture == (), m.figure_spec.panels[
            0
        ].to_dict()

    def test_removing_another_layer_leaves_the_slider_alone(self, points):
        """A slider steps through its own layers; removing a different one does not touch it.

        Args:
            points: The features drawn.
        """
        frames = points.assign(t=["2020", "2021"])
        m = WebMap().timeslider(frames, kdim="t")
        (stepped,) = m.layer_ids
        m.points(points, name="obs").remove_layer("obs")
        kinds = [item.kind for item in m.figure_spec.panels[0].furniture]
        assert kinds == ["time_slider"], kinds
        assert stepped in m.layer_ids, m.layer_ids


class TestTheDeclaration:
    """`capabilities.py` says what the tier can draw (#294)."""

    def test_the_declaration_is_this_backend(self):
        """The row is named as `quickmap(backend=...)` spells it."""
        assert CAPABILITIES.backend == "web", CAPABILITIES.backend

    def test_every_kind_a_builder_records_is_declared(self, points, polygons):
        """A builder drawing an undeclared kind would make the declaration a lie.

        Args:
            points: The point features drawn.
            polygons: The polygon features drawn.
        """
        m = (
            WebMap()
            .points(points, name="obs")
            .choropleth(polygons, column="pop", name="area")
            .text(4.9, 52.4, "A", name="label")
            .graticule()
            .heatmap(points)
        )
        drawn = {m.figure_spec.layers.get(name).kind for name in m.layer_ids}
        assert drawn <= CAPABILITIES.kinds, f"undeclared kinds drawn: {drawn}"

    def test_the_declaration_loads_without_maplibre(self):
        """A dispatcher reads it before it decides which backend to build.

        Test scenario:
            Run in a subprocess, since this session has MapLibre loaded already.
        """
        import subprocess
        import sys

        code = (
            "import sys; from digitalearth.web.capabilities import CAPABILITIES;"
            "print('maplibre' in sys.modules, CAPABILITIES.backend)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False web", result.stdout or result.stderr

    def test_what_the_tier_will_not_draw_says_why(self):
        """A field has no MapLibre primitive, and that is a decision rather than a gap."""
        assert "no arrow glyph" in CAPABILITIES.reason("vectors"), CAPABILITIES.absent

    def test_the_dispatcher_reads_the_declaration(self):
        """`api.BACKEND_CAPABILITIES["web"]` is derived from it, not written out beside it."""
        from digitalearth.api import BACKEND_CAPABILITIES

        assert sorted(BACKEND_CAPABILITIES["web"]) == ["basemap", "colorbar", "crs"], (
            BACKEND_CAPABILITIES["web"]
        )

    def test_a_refusal_carries_the_declared_reason(self, points):
        """A caller is told what the tier does instead of what they asked for."""
        from digitalearth import quickmap

        with pytest.raises(ValueError, match="pans and zooms"):
            quickmap(points, backend="web", domain="europe")

    def test_the_declaration_is_a_capabilities_value(self):
        """The shared type, so one support matrix can be built from every tier's row."""
        assert isinstance(CAPABILITIES, Capabilities), type(CAPABILITIES)


class TestTheFigureSurvivesStorage:
    """What the map describes can be written, as long as its data can be named."""

    def test_a_figure_over_a_named_source_round_trips(self, points):
        """A layer whose source can be named survives the JSON round trip.

        Args:
            points: The features drawn, referenced by a name rather than by the object.

        Test scenario:
            The web builders take pyramids objects today, so the reference is to this process's memory. A
            layer pointed at a path — what the builders will take when the renderer reads sources at draw
            time — stores and reads back, which is what the round trip needs.
        """
        m = WebMap().points(points, name="obs")
        figure = m.figure_spec
        portable = replace(
            figure, sources={"obs": DataRef("examples/data/points.geojson")}
        )
        assert FigureSpec.from_dict(portable.to_dict()) == portable, portable.to_dict()

    def test_a_figure_over_an_object_describes_but_does_not_store(self, points):
        """An in-memory source is deliberately not portable (#285), and says so.

        Args:
            points: The features drawn.
        """
        m = WebMap().points(points, name="obs")
        with pytest.raises(ValueError, match="only resolves in the process"):
            m.figure_spec.to_dict()
