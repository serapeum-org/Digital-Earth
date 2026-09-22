"""What the web tier declares, held against what it actually draws (U-1, #294).

A declaration is only worth reading if it is true. `api.BACKEND_CAPABILITIES` is derived from these rows, a
dispatcher routes a figure by them, and the support matrix the docs build will print them — so a row that
claims a kind nothing draws, or a channel nothing carries, is worse than no row at all
(`planning/refactor/backends/api-unification.md` §2, "`hasattr` capability lies").

These are the three questions that keep this tier's row honest:

* **every declared kind is drawn** — by the renderer from its description, or by a builder that draws it
  straight onto the widget. The two buckets below are exhaustive and disjoint, so a kind added to the
  declaration with nothing behind it lands outside both of them and fails here.
* **every declared channel reaches the layer** — each one is given to a builder and read back off the
  recorded symbology, and the `data_driven` ones are read back as MapLibre *expressions* over a column
  rather than as constants.
* **the schemes are the shared classifier's**, which is contract C4: one `scheme`/`k` pair cuts the same
  classes here as on every other tier.
"""

import subprocess
import sys

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from digitalearth.base.capabilities import Capabilities
from digitalearth.base.spec import Scale

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402
from digitalearth.web.capabilities import CAPABILITIES  # noqa: E402
from digitalearth.web.renderer import DRAWN_KINDS, drawer_for  # noqa: E402

#: The raster every field builder here is given. Read from the repo root, as the rest of the suite does.
DEM_PATH = "examples/data/acc4000.tif"

#: A contour interval `acc4000` actually has levels for — its band carries no variable `auto_style` knows,
#: so `contours()` asks for one rather than guessing.
CONTOUR_INTERVAL = 50.0


def _points() -> gpd.GeoDataFrame:
    """Return two points in EPSG:4326 with a numeric, a nominal and a height column.

    Returns:
        The collection every vector builder below is given.
    """
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0], "label": ["a", "b"], "storeys": [10.0, 20.0]},
        geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
        crs=4326,
    )


def _lines() -> gpd.GeoDataFrame:
    """Return two lines in EPSG:4326.

    Returns:
        The collection `lines()` is given.
    """
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[
            LineString([(4.0, 52.0), (5.0, 53.0)]),
            LineString([(5.0, 52.0), (6.0, 53.0)]),
        ],
        crs=4326,
    )


def _polygons() -> gpd.GeoDataFrame:
    """Return two polygons in EPSG:4326 with a numeric and a height column.

    Returns:
        The collection `polygons()`, `choropleth()` and `extrusion()` are given.
    """
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0], "storeys": [10.0, 20.0]},
        geometry=[
            Polygon([(4.0, 52.0), (5.0, 52.0), (5.0, 53.0), (4.0, 53.0)]),
            Polygon([(5.0, 52.0), (6.0, 52.0), (6.0, 53.0), (5.0, 53.0)]),
        ],
        crs=4326,
    )


def _dem():
    """Return the committed raster, for the two contour builders.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(DEM_PATH)


#: Kinds a builder records as a layer, which the renderer then draws from that description. This is
#: `DRAWN_KINDS`, resolved to the call that produces each one — so a kind whose builder stops recording it
#: fails on the call rather than on a table nobody runs.
DESCRIBED_AND_DRAWN = {
    "graticule": lambda m: m.graticule(spacing=10.0),
    "text": lambda m: m.text(4.9, 52.4, "here"),
    "raster": lambda m: m.field(_dem()),
    "rgb": lambda m: m.rgb_composite(_dem(), bands=(1, 1, 1)),
    "points": lambda m: m.points(_points()),
    "lines": lambda m: m.lines(_lines()),
    "polygons": lambda m: m.polygons(_polygons()),
    "choropleth": lambda m: m.choropleth(_polygons(), column="value"),
    "labels": lambda m: m.labels(_points(), "label"),
    "heatmap": lambda m: m.heatmap(_points()),
    "clusters": lambda m: m.cluster(_points()),
    "extrusion": lambda m: m.extrusion(_polygons(), height=30.0),
    "contours": lambda m: m.contours(_dem(), interval=CONTOUR_INTERVAL),
    "filled_contours": lambda m: m.contours(
        _dem(), interval=CONTOUR_INTERVAL, filled=True
    ),
    "custom:maplibre": lambda m: m.add_layer({"id": "own", "type": "background"}),
}

#: Kinds a builder draws straight onto the widget without recording a layer at all: a basemap is the map's
#: style, and a point cloud, a terrain source and a glTF model are deck.gl/terrain objects the widget takes
#: rather than MapLibre layers the description can rebuild. They are declared because the tier draws them,
#: and they are separated here because "declared, drawn, not described" is a different answer from the one
#: above — and the one a reader of the declaration is most likely to get wrong.
DRAWN_BUT_NOT_DESCRIBED = {
    "basemap": lambda m: m.basemap("CartoDark"),
    "point_cloud": lambda m: m.point_cloud([(4.9, 52.4, 10.0), (5.0, 52.2, 20.0)]),
    "terrain": lambda m: m.terrain_tiles(),
    "model": lambda m: m.gltf("https://example.invalid/model.glb", 4.9, 52.4),
}


def _kinds_recorded(build) -> list:
    """Return the kinds a builder call records on a fresh map.

    Args:
        build: A callable taking the map and calling one builder on it.

    Returns:
        The kind of every layer the call left in the figure, in draw order.
    """
    web_map = WebMap()
    build(web_map)
    return [layer.kind for layer in web_map.figure_spec.layers]


def _last_symbology(build):
    """Return the symbology of the last layer a builder call records.

    Args:
        build: A callable taking the map and calling one builder on it.

    Returns:
        The last layer's :class:`~digitalearth.base.spec.Symbology`.
    """
    web_map = WebMap()
    build(web_map)
    return list(web_map.figure_spec.layers)[-1].symbology


def _paint(build, key: str):
    """Return one MapLibre paint value off the last layer a builder call records.

    Args:
        build: A callable taking the map and calling one builder on it.
        key: The MapLibre paint property to read.

    Returns:
        The recorded value, which is a constant for a channel set from a keyword and a MapLibre expression
        (a list) for one driven by a column.
    """
    return _last_symbology(build).props["paint"][key]


class TestTheDeclaration:
    """The row itself: whose it is, what type it is, and what it costs to read."""

    def test_the_declaration_is_this_backend(self):
        """The row is named as `quickmap(backend=...)` spells it."""
        assert CAPABILITIES.backend == "web", CAPABILITIES.backend

    def test_the_declaration_is_a_capabilities_value(self):
        """The shared type, so one support matrix can be built from every tier's row."""
        assert isinstance(CAPABILITIES, Capabilities), type(CAPABILITIES)

    def test_the_declaration_loads_without_maplibre(self):
        """A dispatcher reads every tier's row before it decides which backend to build.

        Test scenario:
            Run in a subprocess, because this session has MapLibre imported already — asking `sys.modules`
            in-process would answer about the test run rather than about the import. This test only means
            something in the web environment, where MapLibre *is* installed and so *could* be pulled in;
            the module-level `importorskip` is what keeps it from passing vacuously elsewhere.
        """
        code = (
            "import sys; from digitalearth.web.capabilities import CAPABILITIES;"
            "print('maplibre' in sys.modules, CAPABILITIES.backend)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False web", result.stdout or result.stderr

    def test_the_dispatcher_reads_the_declaration(self):
        """`api.BACKEND_CAPABILITIES["web"]` is derived from this row, and says what it always said."""
        from digitalearth.api import BACKEND_CAPABILITIES

        assert sorted(BACKEND_CAPABILITIES["web"]) == [
            "basemap",
            "colorbar",
            "crs",
        ], BACKEND_CAPABILITIES["web"]

    def test_a_dispatcher_refusal_carries_the_declared_reason(self):
        """A caller is told what the tier does instead of what they asked for."""
        from digitalearth import quickmap

        dem = _dem()
        with pytest.raises(ValueError, match="pans and zooms"):
            quickmap(dem, backend="web", domain="europe")

    def test_a_kind_refusal_carries_the_declared_reason(self):
        """The renderer refuses an unsupported kind with the tier's own sentence, not a new one.

        Test scenario:
            `mesh` is declared absent with "a raster is drawn as an image rather than as cells". Before
            #294 closed, the refusal said only that the kind was not drawn — true, and no help at all to a
            caller holding a figure written for the static tier.
        """
        with pytest.raises(KeyError, match="drawn as an image rather than as cells"):
            drawer_for("mesh")

    def test_a_kind_the_tier_has_simply_not_reached_carries_no_reason(self):
        """The other half: no declared reason, no invented one.

        Test scenario:
            `terrain` is declared and drawn, just not from a description yet, so `absent` says nothing
            about it. If the refusal appended a reason anyway it would be making one up — this is what
            stops the clause above from being unconditional text.
        """
        assert CAPABILITIES.reason("terrain") is None, CAPABILITIES.absent
        with pytest.raises(KeyError) as refused:
            drawer_for("terrain")
        assert "—" not in str(refused.value), str(refused.value)


class TestEveryDeclaredKindIsDrawn:
    """Hold `CAPABILITIES.kinds` against what the tier actually draws and describes."""

    def test_the_two_ways_a_kind_is_drawn_account_for_every_declared_one(self):
        """A declared kind with nothing behind it is the capability lie this file exists to catch.

        Test scenario:
            The buckets are exhaustive: drawn from a description, or drawn onto the widget with no
            description. Adding a kind to the declaration puts it in neither, and removing one that is drawn
            leaves it in a bucket with nothing to declare it. There is no third bucket of kinds described but
            replayed from the queue: contours were the last, and are drawn from their description now (L3).
        """
        accounted = set(DRAWN_KINDS) | set(DRAWN_BUT_NOT_DESCRIBED)
        difference = sorted(accounted.symmetric_difference(CAPABILITIES.kinds))
        assert difference == [], (
            f"{difference} are declared with nothing drawing them, or drawn without being declared"
        )

    def test_no_kind_is_counted_two_ways(self):
        """A kind both described and queued would be drawn twice; the buckets have to be disjoint."""
        seen = list(DRAWN_KINDS) + list(DRAWN_BUT_NOT_DESCRIBED)
        repeated = sorted({kind for kind in seen if seen.count(kind) > 1})
        assert repeated == [], f"{repeated} are claimed by more than one drawing path"

    def test_the_described_bucket_is_the_renderer_s_own_list(self):
        """`DESCRIBED_AND_DRAWN` names a builder per drawable kind, so none can go untested."""
        difference = sorted(set(DESCRIBED_AND_DRAWN).symmetric_difference(DRAWN_KINDS))
        assert difference == [], f"{difference} have no builder call to exercise them"

    @pytest.mark.parametrize("kind", sorted(DESCRIBED_AND_DRAWN))
    def test_a_builder_records_the_kind_the_renderer_draws(self, kind):
        """Calling the builder leaves a layer of exactly that kind behind.

        Args:
            kind: The declared kind under test.

        Test scenario:
            Read off what the builder records rather than off a table beside it, so a builder that stops
            recording its kind — or records another tier's spelling — fails here.
        """
        assert kind in _kinds_recorded(DESCRIBED_AND_DRAWN[kind]), kind

    @pytest.mark.parametrize("kind", sorted(DRAWN_BUT_NOT_DESCRIBED))
    def test_a_widget_builder_draws_without_describing_a_layer(self, kind):
        """The four kinds that reach the widget directly: one drawing queued, and no layer recorded.

        Args:
            kind: The declared kind under test.

        Test scenario:
            This is the bucket a reader of the declaration gets wrong: `capabilities.supports("terrain")`
            is `True` and `WebMap.terrain_tiles()` really does drape terrain, but no `LayerSpec` carries
            the kind, so a saved figure does not round-trip it. Asserting *both* halves is what makes the
            claim testable — a builder that started describing its layer would move to another bucket, and
            one that stopped drawing anything would queue nothing.
        """
        web_map = WebMap()
        DRAWN_BUT_NOT_DESCRIBED[kind](web_map)
        assert len(web_map._queued) == 1, web_map._queued
        assert [layer.kind for layer in web_map.figure_spec.layers] == [], kind


class TestEveryDeclaredChannelReachesTheLayer:
    """A channel this tier declares has to arrive somewhere a drawer can read it."""

    #: Channel -> (the builder call that sets it, the MapLibre paint/layout key it lands under, the value
    #: that was asked for). `tooltip` is the one channel recorded as an `Encoding` rather than as a
    #: MapLibre property, and is checked on its own below.
    CARRIED = {
        "color": (
            lambda m: m.points(_points(), color="#ff0000"),
            "circle-color",
            "#ff0000",
        ),
        "opacity": (lambda m: m.points(_points(), opacity=0.5), "circle-opacity", 0.5),
        "size": (lambda m: m.points(_points(), size=8.0), "circle-radius", 8.0),
        "width": (lambda m: m.lines(_lines(), width=3.0), "line-width", 3.0),
        "height": (
            lambda m: m.extrusion(_polygons(), height=42.0),
            "fill-extrusion-height",
            42.0,
        ),
    }

    def test_every_declared_channel_is_covered_here(self):
        """The table below has to name every channel the row claims, or a claim goes unchecked."""
        covered = set(self.CARRIED) | {"text", "tooltip"}
        difference = sorted(covered.symmetric_difference(CAPABILITIES.channels))
        assert difference == [], (
            f"{difference} are declared and unchecked, or checked and undeclared"
        )

    @pytest.mark.parametrize("channel", sorted(CARRIED))
    def test_a_channel_set_from_a_keyword_reaches_the_recorded_paint(self, channel):
        """What the caller asked for is what the layer records, under the MapLibre key that paints it.

        Args:
            channel: The declared channel under test.
        """
        build, key, asked = self.CARRIED[channel]
        assert _paint(build, key) == asked, (channel, key)

    def test_the_text_channel_reaches_the_layout(self):
        """A label's string is a layout property, not a paint one, so it is read from there."""
        layout = _last_symbology(lambda m: m.labels(_points(), "label")).props["layout"]
        assert layout["text-field"] == ("get", "label"), layout

    def test_the_tooltip_channel_reaches_the_layer_as_an_encoding(self):
        """The one channel held as an `Encoding`: it names fields rather than painting anything."""
        symbology = _last_symbology(lambda m: m.points(_points()).tooltip(["value"]))
        assert symbology.encoding("tooltip").resolve() == ("value",), (
            symbology.to_dict()
        )


class TestOnlyTheDataDrivenChannelsTakeAField:
    """`data_driven` is a second, narrower claim, and is checked separately from `channels`."""

    def test_a_data_driven_colour_is_an_expression_over_the_column(self):
        """A classified fill is a MapLibre expression reading the column, not a constant colour."""
        painted = _paint(
            lambda m: m.choropleth(
                _polygons(), column="value", scheme="quantiles", k=2
            ),
            "fill-color",
        )
        assert "value" in str(painted), painted

    def test_a_data_driven_height_is_an_expression_over_the_column(self):
        """`height="storeys"` extrudes each polygon by its own value."""
        painted = _paint(
            lambda m: m.extrusion(_polygons(), height="storeys"),
            "fill-extrusion-height",
        )
        assert painted == ("get", "storeys"), painted

    def test_a_data_driven_text_is_an_expression_over_the_column(self):
        """A label reads its string from the feature rather than repeating one constant."""
        layout = _last_symbology(lambda m: m.labels(_points(), "label")).props["layout"]
        assert layout["text-field"] == ("get", "label"), layout

    @pytest.mark.parametrize(
        ("channel", "build", "key"),
        [
            ("opacity", lambda m: m.points(_points(), opacity=0.5), "circle-opacity"),
            ("size", lambda m: m.points(_points(), size=8.0), "circle-radius"),
            ("width", lambda m: m.lines(_lines(), width=3.0), "line-width"),
        ],
    )
    def test_a_channel_outside_data_driven_records_a_constant(
        self, channel, build, key
    ):
        """The narrower claim has to be narrower: these three take a number and nothing else.

        Args:
            channel: The declared channel under test.
            build: The builder call that sets it.
            key: The MapLibre paint property it lands under.

        Test scenario:
            The recorded value has to *be a number*. Asking instead that it is not a `list` asked nothing:
            a description freezes every sequence to a tuple, so a data-driven expression recorded here is a
            tuple and was never a list (review M12). `data_driven` is a promise about *which* channels take
            a field, and a promise that cannot be broken is not one — so if one of these ever grew a
            `column=`, the value stops being a number and this fails.
        """
        recorded = _paint(build, key)
        assert channel not in CAPABILITIES.data_driven, sorted(CAPABILITIES.data_driven)
        assert isinstance(recorded, (int, float)), (channel, key, recorded)


class TestTheSchemesAreTheSharedClassifiers:
    """Contract C4 — one `scheme`/`k` pair cuts the same classes here as on every other tier."""

    @pytest.mark.parametrize("scheme", sorted(CAPABILITIES.schemes - {"categorical"}))
    def test_every_declared_scheme_cuts_classes(self, scheme):
        """A scheme the shared classifier does not know would be a name nothing accepts.

        Args:
            scheme: The declared scheme under test.

        Test scenario:
            `Scale.breaks_of` is the one classifier every tier cuts with, and it raises for a name it does
            not know — so a typo, or a scheme borrowed from another library, fails here rather than on the
            caller's first classified map.
        """
        assert len(Scale.breaks_of(list(range(100)), scheme, 3)) >= 2, scheme

    def test_the_nominal_scheme_is_declared(self):
        """`categorical` is not a classifier scheme; it is the nominal path, and C4 pins its meaning."""
        assert "categorical" in CAPABILITIES.schemes, sorted(CAPABILITIES.schemes)

    def test_a_classified_layer_records_the_scheme_and_the_count(self):
        """C4's own shape: `scheme` + `k` reach the layer rather than being swallowed by `**kwargs`.

        Test scenario:
            `k=2` asks for two classes, so the key the builder recorded carries two colours and the three
            edges that bound them. A `k` that never reached the classifier would leave the tier's default
            of five.
        """
        legend = (
            WebMap()
            .choropleth(_polygons(), column="value", scheme="quantiles", k=2)
            .last_legend
        )
        assert len(legend["colors"]) == 2, legend
