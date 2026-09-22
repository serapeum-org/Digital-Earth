"""The static seam: the matplotlib tier describes what it draws (#303, the first half).

Until now this tier drew and kept nothing but the artists: `Scene.layers` held `(glyph, mappable)` pairs, and
a figure could say how *many* layers there were but never what any of them was. A raster and a choropleth
were both "a thing cleopatra drew", a basemap and a graticule were not layers at all, and nothing outside
matplotlib could read the result.

These cover the description that replaces that silence — every builder routed through one funnel, recording
the registry's engine-neutral kind, the source it was given and its style as values. They deliberately do
**not** cover rendering *from* that description: the renderer is #303's second half, and a test that drew
from a figure would be asserting against code that is not written yet.
"""

import json

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection

from digitalearth.base.custom import custom_kind
from digitalearth.base.registry import KIND_BANDS, band_of
from digitalearth.static import Map, Scene
from digitalearth.static.scene import PANEL_ID


@pytest.fixture
def features():
    """Return the point fixture as a pyramids ``FeatureCollection``.

    Returns:
        A point collection in EPSG:32618 with a numeric ``fid`` column.
    """
    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygons(features):
    """Return polygon features, built by buffering the points.

    Args:
        features: The point collection to buffer.

    Returns:
        A polygon collection in the same CRS, still carrying ``fid``.
    """
    features["geometry"] = features.geometry.buffer(500.0)
    return features


@pytest.fixture
def lines():
    """Return a two-line collection, which is what a flow map draws.

    Returns:
        A ``LineString`` collection in EPSG:4326 with a numeric ``flow`` column.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    frame = gpd.GeoDataFrame(
        {"flow": [1.0, 2.0]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(0, 1), (1, 2)])],
        crs="EPSG:4326",
    )
    return FeatureCollection(frame)


@pytest.fixture
def bands(dataset):
    """Return a three-band raster on the fixture's grid, which is what a composite draws.

    Args:
        dataset: The single-band fixture whose grid and CRS are reused.

    Returns:
        A three-band ``Dataset``.
    """
    base = np.nan_to_num(dataset.read_array(band=0).astype("float32"))
    return Dataset.from_array(
        arr=np.stack([base, base * 0.5, base * 0.25]),
        geo_ref=GeoReference(geo=dataset.geotransform, epsg=dataset.epsg),
    )


#: What each data builder draws, and the registered kind it must record. The kinds are the registry's own
#: nouns, never cleopatra's render names: ``contour`` is a matplotlib call, ``contours`` is what the layer
#: *is*. The pairs that look redundant are the ones worth having — ``block`` and ``pcolormesh`` draw the same
#: mesh under two entry points, and ``voronoi``/``cartogram`` are a choropleth or a plain polygon layer
#: depending only on whether a column was named.
DATA_BUILDERS = {
    "imshow": ("raster", lambda canvas, given: canvas.imshow(given["raster"])),
    "contour": ("contours", lambda canvas, given: canvas.contour(given["raster"])),
    "contourf": (
        "filled_contours",
        lambda canvas, given: canvas.contourf(given["raster"]),
    ),
    "pcolormesh": ("mesh", lambda canvas, given: canvas.pcolormesh(given["raster"])),
    "block": ("mesh", lambda canvas, given: canvas.block(given["raster"])),
    "rgb_composite": (
        "rgb",
        lambda canvas, given: canvas.rgb_composite(given["bands"]),
    ),
    "hsv_composite": (
        "rgb",
        lambda canvas, given: canvas.hsv_composite(given["bands"]),
    ),
    "scatter": ("points", lambda canvas, given: canvas.scatter(given["points"])),
    "grid_points": (
        "points",
        lambda canvas, given: canvas.grid_points(given["raster"]),
    ),
    "point_cloud": (
        "points",
        lambda canvas, given: canvas.point_cloud(given["raster"]),
    ),
    "grid_cells": (
        "choropleth",
        lambda canvas, given: canvas.grid_cells(given["raster"]),
    ),
    "quiver": (
        "vectors",
        lambda canvas, given: canvas.quiver(given["raster"], given["raster"]),
    ),
    "barbs": (
        "vectors",
        lambda canvas, given: canvas.barbs(given["raster"], given["raster"]),
    ),
    "streamplot": (
        "streamlines",
        lambda canvas, given: canvas.streamplot(given["raster"], given["raster"]),
    ),
    "tricontour": (
        "unstructured",
        lambda canvas, given: canvas.tricontour(given["points"]),
    ),
    "tricontourf": (
        "unstructured",
        lambda canvas, given: canvas.tricontourf(given["points"]),
    ),
    "tripcolor": (
        "unstructured",
        lambda canvas, given: canvas.tripcolor(given["points"]),
    ),
    "choropleth": (
        "choropleth",
        lambda canvas, given: canvas.choropleth(given["polygons"], column="fid"),
    ),
    "shapes": ("polygons", lambda canvas, given: canvas.shapes(given["polygons"])),
    "voronoi": (
        "choropleth",
        lambda canvas, given: canvas.voronoi(given["points"], column="fid"),
    ),
    "voronoi-outlines": (
        "polygons",
        lambda canvas, given: canvas.voronoi(given["points"]),
    ),
    "cartogram": (
        "choropleth",
        lambda canvas, given: canvas.cartogram(
            given["polygons"], scale="fid", column="fid"
        ),
    ),
    "cartogram-outlines": (
        "polygons",
        lambda canvas, given: canvas.cartogram(given["polygons"], scale="fid"),
    ),
    "quadtree": (
        "choropleth",
        lambda canvas, given: canvas.quadtree(given["points"], nmax=1),
    ),
    "kde": ("heatmap", lambda canvas, given: canvas.kde(given["points"])),
    "sankey": (
        "flow",
        lambda canvas, given: canvas.sankey(given["lines"], column="flow"),
    ),
}

#: What each decoration builder draws. These draw straight onto the axes and leave no mappable to register,
#: which is exactly why the description is a structure of its own rather than a label beside ``Scene.layers``.
DECORATION_BUILDERS = {
    "text": ("text", lambda canvas: canvas.text(4.9, 52.4, "Amsterdam")),
    "annotate": ("text", lambda canvas: canvas.annotate(4.9, 52.4, "Amsterdam")),
    "graticule": ("graticule", lambda canvas: canvas.graticule()),
    "coastlines": ("coastlines", lambda canvas: canvas.coastlines()),
    "borders": ("borders", lambda canvas: canvas.borders()),
    "land": ("land", lambda canvas: canvas.land()),
    "ocean": ("ocean", lambda canvas: canvas.ocean()),
    "lakes": ("lakes", lambda canvas: canvas.lakes()),
    "rivers": ("rivers", lambda canvas: canvas.rivers()),
}


@pytest.fixture
def given(dataset, features, polygons, lines, bands):
    """Bundle every input :data:`DATA_BUILDERS` draws from, so the table can stay a table.

    Args:
        dataset: The single-band raster fixture.
        features: A point collection.
        polygons: A polygon collection.
        lines: A line collection.
        bands: A three-band raster.

    Returns:
        The inputs, keyed by what they are rather than by which builder wants them.
    """
    # `polygons` buffers the very collection `features` yields, so the two fixtures cannot share one object:
    # a buffered geometry is no longer a point, and half the table needs points.
    return {
        "raster": dataset,
        "bands": bands,
        "points": FeatureCollection.read_file("tests/data/points.geojson"),
        "polygons": polygons,
        "lines": lines,
    }


@pytest.fixture
def offline_tiles(monkeypatch):
    """Replace the tile fetch so ``basemap`` draws without reaching the network.

    Args:
        monkeypatch: pytest's patcher, which restores the real call afterwards.

    Returns:
        The list the stand-in appends to, one entry per call that reached it.
    """
    from digitalearth.static.maps import decoration

    drawn = []

    def stand_in(ax, source=None, crs=None, **kwargs):
        """Record the call and hand back something artist-shaped.

        Args:
            ax: The axes tiles would be drawn on.
            source: The resolved provider.
            crs: The display CRS.
            **kwargs: Whatever else the builder forwarded.

        Returns:
            A marker standing for the tile artist.
        """
        drawn.append((ax, source, crs, kwargs))
        return "tiles"

    monkeypatch.setattr(decoration, "add_tiles", stand_in)
    return drawn


class TestEveryBuilderDescribesWhatItDrew:
    """The funnel: a builder that draws also says, in the registry's words, what it drew."""

    @pytest.mark.parametrize("builder", sorted(DATA_BUILDERS))
    def test_a_data_builder_records_its_registered_kind(self, builder, given):
        """The matplotlib artist cannot answer this: a choropleth and an outline are one ``PolyCollection``.

        Args:
            builder: The entry in :data:`DATA_BUILDERS` under test.
            given: The inputs the builders draw from.
        """
        kind, draw = DATA_BUILDERS[builder]
        canvas = Map(crs=given["raster"].epsg)
        draw(canvas, given)
        assert canvas.layer_ids != [], f"{builder} drew without describing anything"
        described = canvas.figure_spec.layers.get(canvas.layer_ids[-1])
        assert described.kind == kind, f"{builder} recorded {described.kind!r}"

    @pytest.mark.parametrize("builder", sorted(DECORATION_BUILDERS))
    def test_a_decoration_builder_records_its_registered_kind(self, builder):
        """Decoration is a layer too — its kind's band is what keeps it under or over the data.

        Args:
            builder: The entry in :data:`DECORATION_BUILDERS` under test.
        """
        kind, draw = DECORATION_BUILDERS[builder]
        canvas = Map()
        draw(canvas)
        assert canvas.layer_ids != [], f"{builder} drew without describing anything"
        described = canvas.figure_spec.layers.get(canvas.layer_ids[-1])
        assert described.kind == kind, f"{builder} recorded {described.kind!r}"

    def test_a_basemap_is_described_once_the_tiles_arrive(self, offline_tiles):
        """Tiles come off the network, so the record follows the draw rather than announcing it.

        Args:
            offline_tiles: The stand-in for cleopatra's tile call.
        """
        canvas = Map()
        canvas.basemap()
        assert offline_tiles != [], "the stand-in was never reached"
        assert canvas.layer_ids == ["basemap-1"], canvas.layer_ids

    def test_an_ensemble_describes_one_layer_per_member(self, dataset):
        """``spaghetti`` is the one builder that draws several layers from one call.

        Args:
            dataset: A raster standing in for each ensemble member.
        """
        collection = _FakeCollection([dataset, dataset, dataset])
        canvas = Map(crs=dataset.epsg)
        canvas.spaghetti(collection)
        kinds = [canvas.figure_spec.layers.get(i).kind for i in canvas.layer_ids]
        assert kinds == ["contours"] * 3, kinds

    def test_every_registered_artist_is_described(self, given):
        """The two structures are separate, but nothing may land in one and not the other.

        Args:
            given: The inputs the builders draw from.

        Test scenario:
            ``Scene.layers`` holds the mappables a colorbar is keyed to and the tree holds the descriptions.
            A builder that appended to the first without routing through the funnel would leave a figure
            describing fewer layers than it drew, which is the drift this seam exists to remove.
        """
        canvas = Map(crs=given["raster"].epsg)
        canvas.imshow(given["raster"])
        canvas.scatter(given["points"])
        canvas.choropleth(given["polygons"], column="fid")
        assert len(canvas.layer_ids) == len(canvas.layers), (
            f"{len(canvas.layers)} artists registered, {len(canvas.layer_ids)} described"
        )

    def test_a_builder_that_drew_nothing_describes_nothing(self, dataset, monkeypatch):
        """A described layer no artist backs is the same drift, seen from the other side.

        Args:
            dataset: The raster whose reprojection is made to fail.
            monkeypatch: Used to put the data outside what the display CRS can show.
        """
        from digitalearth.base.crs import OffLimbError

        def off_limb(self, data):
            """Report the data as unplaceable, the way a warp past the limb does.

            Args:
                self: The map.
                data: The dataset being placed.

            Raises:
                OffLimbError: always.
            """
            raise OffLimbError("nothing survives the warp")

        monkeypatch.setattr(Map, "_reproject", off_limb)
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        assert canvas.layer_ids == [], canvas.layer_ids


class TestAnIdFollowsTheLayer:
    """A layer is addressed by id, never by position — so the ids have to be unique and stable."""

    def test_two_layers_of_one_kind_get_two_ids(self, dataset):
        """Sharing an id would make the second layer unaddressable.

        Args:
            dataset: A raster drawn twice.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        canvas.imshow(dataset)
        assert canvas.layer_ids == ["raster-1", "raster-2"], canvas.layer_ids

    def test_an_id_reaches_the_layer_it_was_issued_for(self, dataset, features):
        """The id is the handle: looking it up must come back with that builder's own record.

        Args:
            dataset: A raster.
            features: Points drawn over it.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        canvas.scatter(features)
        found = {i: canvas.figure_spec.layers.get(i).kind for i in canvas.layer_ids}
        assert found == {"raster-1": "raster", "points-2": "points"}, found

    def test_the_counter_numbers_the_figure_rather_than_the_kind(
        self, dataset, features
    ):
        """One sequence, so an id read out of a figure says when the layer was added.

        Args:
            dataset: A raster.
            features: Points drawn over it.

        Test scenario:
            Numbering per kind would give a map with one raster and one point layer two layers both called
            ``-1``, which reads like a collision even though it is not one.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.scatter(features)
        canvas.imshow(dataset)
        assert canvas.layer_ids == ["points-1", "raster-2"], canvas.layer_ids

    def test_a_redrawn_frame_mints_the_same_ids_again(self, dataset):
        """An animation clears the axes per frame, and the description is cleared with it.

        Args:
            dataset: The raster each frame draws.

        Test scenario:
            ``_reset_layers`` runs between frames. Leaving the tree alone would have frame 50 describe fifty
            copies of one raster, each under a fresh id, and the figure would grow with the animation.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        canvas._reset_layers()
        canvas.imshow(dataset)
        assert canvas.layer_ids == ["raster-1"], canvas.layer_ids

    def test_a_second_graticule_replaces_the_first(self):
        """The map holds one set of graticule lines, so it describes one graticule.

        Test scenario:
            ``graticule()`` overwrites ``_graticule_lines``; a second call that described a second layer
            would say the map draws two grids where it draws one.
        """
        canvas = Map()
        canvas.graticule(lon_step=30.0)
        canvas.graticule(lon_step=15.0)
        assert canvas.layer_ids == ["graticule-1"], canvas.layer_ids
        held = canvas.figure_spec.layers.get("graticule-1")
        assert held.symbology.props["lon_step"] == 15.0, dict(held.symbology.props)


class TestTheSourcesAreWhatTheBuildersWereGiven:
    """A layer refers to its data instead of holding it, and the reference has to reach the real thing."""

    def test_a_raster_layer_references_the_dataset_it_was_handed(self, dataset):
        """Not a copy and not a look-alike: the reference resolves back to that object.

        Args:
            dataset: The raster the builder is given.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        reference = canvas.figure_spec.sources["raster-1"]
        assert reference.open() is dataset, reference.uri

    def test_each_layer_references_its_own_data(self, dataset, features):
        """Two layers, two sources — keyed by the layer id, so neither can pick up the other's.

        Args:
            dataset: The raster.
            features: The points.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        canvas.scatter(features)
        sources = canvas.figure_spec.sources
        assert sources["raster-1"].open() is dataset, sources["raster-1"].uri
        assert sources["points-2"].open() is features, sources["points-2"].uri

    def test_a_vector_field_references_both_components(self, dataset):
        """Neither component alone draws the field, so the pair is the source.

        Args:
            dataset: Stands in for both the u and the v component.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.quiver(dataset, dataset)
        held = canvas.figure_spec.sources["vectors-1"].open()
        assert held == (dataset, dataset), held

    def test_a_layer_drawn_from_no_data_has_no_source(self):
        """A graticule is computed from the projection; there is nothing to reference."""
        canvas = Map()
        canvas.graticule()
        described = canvas.figure_spec.layers.get("graticule-1")
        assert described.source_id is None, described.source_id

    def test_two_maps_do_not_share_one_source_entry(self, dataset):
        """The object table is process-wide, so each figure namespaces the keys it writes into it.

        Args:
            dataset: A raster drawn on both maps.

        Test scenario:
            Both maps mint ``raster-1``. Keyed by that alone, the second map's registration would re-point
            the first map's already-captured source at its own data.
        """
        first = Map(crs=dataset.epsg)
        first.imshow(dataset)
        second = Map(crs=dataset.epsg)
        second.imshow(dataset)
        held = first.figure_spec.sources["raster-1"].uri
        assert held != second.figure_spec.sources["raster-1"].uri, held

    def test_an_in_memory_source_is_refused_rather_than_written_as_a_dead_reference(
        self, dataset
    ):
        """A pyramids object does not know where it came from, and the figure says so.

        Args:
            dataset: A raster held in memory rather than referenced by path.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        figure = canvas.figure_spec
        with pytest.raises(ValueError, match="only resolves in the process"):
            figure.to_dict()


class TestTheBandOfALayerIsItsKinds:
    """Where a layer is drawn is a property of what it is, not of the order the calls were made in."""

    def test_decoration_sorts_itself_around_the_data(self, dataset, offline_tiles):
        """Added last to first, the layers still come back bottom-first.

        Args:
            dataset: The data the decoration is arranged around.
            offline_tiles: The stand-in for cleopatra's tile call.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.coastlines()
        canvas.imshow(dataset)
        canvas.graticule()
        canvas.basemap()
        assert canvas.layer_ids == [
            "basemap-4",
            "graticule-3",
            "raster-2",
            "coastlines-1",
        ], canvas.layer_ids

    def test_the_order_is_the_registry_s_and_not_this_tier_s(
        self, dataset, offline_tiles
    ):
        """The bands are read back off the registry, so the two cannot drift apart.

        Args:
            dataset: The data layer.
            offline_tiles: The stand-in for cleopatra's tile call.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.coastlines()
        canvas.imshow(dataset)
        canvas.graticule()
        canvas.basemap()
        drawn = [
            band_of(canvas.figure_spec.layers.get(i).kind) for i in canvas.layer_ids
        ]
        assert drawn == sorted(drawn, key=KIND_BANDS.index), drawn

    def test_a_backdrop_is_a_raster_drawn_under_the_data(self, dataset):
        """``stock_img`` sets a low ``zorder``; the description says the same thing in its own words.

        Args:
            dataset: The raster used as the backdrop.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        canvas.stock_img(dataset)
        assert canvas.layer_ids == ["raster-2", "raster-1"], canvas.layer_ids
        assert canvas.figure_spec.layers.get("raster-2").band == "underlay"


class TestACustomLayerIsDescribedButNotRebuildable:
    """An artist the caller built by hand has no source and no symbology — only an identity."""

    def test_an_unclaimed_artist_is_recorded_as_this_engine_s_custom_kind(self):
        """``custom:matplotlib`` names the engine that made it, which is all anyone can say."""
        scene = Scene()
        scene._add_layer(None, "an artist the caller drew")
        described = scene.figure_spec.layers.get(scene.layer_ids[-1])
        assert described.kind == custom_kind("matplotlib"), described.kind

    def test_a_custom_layer_is_numbered_without_its_namespace(self):
        """A colon cannot appear in the middle of an id, so the engine name is not counted."""
        scene = Scene()
        scene._add_layer(None, "an artist the caller drew")
        assert scene.layer_ids == ["custom-1"], scene.layer_ids

    def test_a_custom_layer_carries_no_source(self):
        """Nothing was handed in to reference: the caller drew it from whatever they liked."""
        scene = Scene()
        scene._add_layer(None, "an artist the caller drew")
        described = scene.figure_spec.layers.get("custom-1")
        assert described.source_id is None, described.source_id

    def test_a_custom_layer_leaves_the_figure_s_sources_empty(self):
        """The other half of the same statement, read off the figure rather than the layer."""
        scene = Scene()
        scene._add_layer(None, "an artist the caller drew")
        assert scene.figure_spec.sources == {}, scene.figure_spec.sources

    def test_a_custom_layer_is_still_addressable(self):
        """Which is the point of describing it at all — it can be named, hidden and reordered."""
        scene = Scene()
        scene._add_layer(None, "an artist the caller drew")
        described = scene.figure_spec.layers.get("custom-1")
        assert (described.label, described.visible) == ("custom-1", True), described


class TestTheFigureIsOneWholeThing:
    """`figure_spec` is the tier's answer, so it has to hold together as a value."""

    def test_the_panel_holds_the_layers_in_draw_order(self, dataset, features):
        """One axes is one panel, and its layer list is the tree's order rather than a second one.

        Args:
            dataset: A raster.
            features: Points drawn over it.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        canvas.scatter(features)
        panel = canvas.figure_spec.panels[0]
        assert list(panel.layers) == canvas.layer_ids, panel.layers

    def test_the_panel_is_named_the_way_every_tier_names_its_own(self, dataset):
        """A reader moving between tiers should not have to learn a new panel id each time.

        Args:
            dataset: A raster, so the figure is not empty.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        assert canvas.figure_spec.panels[0].id == PANEL_ID, PANEL_ID

    def test_the_view_is_the_display_crs_rather_than_the_axes_limits(self, dataset):
        """What the map was asked to show, not what the last autoscale left behind.

        Args:
            dataset: A raster in a projected CRS.
        """
        canvas = Map(crs=dataset.epsg, domain="europe")
        canvas.imshow(dataset)
        view = canvas.viewport
        assert view.crs == dataset.epsg, view.crs
        assert view.domain == "europe", view.domain

    def test_a_bare_scene_has_a_view_of_its_own(self):
        """A chart host has no display CRS to place anything in, and does not invent one."""
        assert Scene().figure_spec.panels[0].view.bounds is None

    def test_the_style_is_recorded_as_values(self, dataset):
        """A figure that named a layer but not its colours could never be drawn again.

        Args:
            dataset: The raster drawn with an explicit colormap.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset, cmap="terrain")
        props = canvas.figure_spec.layers.get("raster-1").symbology.props
        assert props["cmap"] == "terrain", dict(props)

    def test_the_style_names_the_builder_that_drew_it(self, dataset):
        """``via`` is how the renderer half will know which recipe to replay.

        Args:
            dataset: The raster drawn as filled contours.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.contourf(dataset)
        props = canvas.figure_spec.layers.get("filled_contours-1").symbology.props
        assert props["via"] == "contourf", dict(props)


#: A matplotlib dash pattern. Its type carries meaning: matplotlib reads the tuple as ``(offset, on-off)``
#: and refuses the list a JSON-shaped round trip turns it into.
DASHED = (0, (5, 5))

#: The calls the review found broken by freezing the caller's keywords into the description (H3). Each
#: worked before, and each passes a dash tuple somewhere matplotlib reads its type.
TYPED_KEYWORD_CALLS = {
    "coastlines(linestyle=)": lambda canvas: canvas.coastlines(linestyle=DASHED),
    "borders(linestyle=)": lambda canvas: canvas.borders(linestyle=DASHED),
    "text(bbox=)": lambda canvas: canvas.text(
        4.9, 52.4, "Amsterdam", bbox={"linestyle": DASHED}
    ),
    "annotate(arrowprops=)": lambda canvas: canvas.annotate(
        4.9,
        52.4,
        "Amsterdam",
        xytext=(6.0, 53.0),
        arrowprops={"arrowstyle": "->", "linestyle": DASHED},
    ),
}


#: The file the ``dataset`` fixture reads, which is what a saved figure references its raster by.
RASTER_PATH = "examples/data/acc4000.tif"


def _saved(figure):
    """Return the figure with each in-memory source referenced by the path it was read from instead.

    ``FigureSpec.to_dict`` refuses an ``object:`` source by design — it names memory in this process — so
    a figure is saved by referencing its data by path. Every source in these tests is the fixture raster.

    Args:
        figure: The figure a map reports.

    Returns:
        The same figure, storable.
    """
    from dataclasses import replace as with_fields

    from digitalearth.base.spec import DataRef

    return with_fields(
        figure, sources={key: DataRef(RASTER_PATH) for key in figure.sources}
    )


def _written_and_read_back(figure):
    """Return a figure after a real JSON round trip, the way a saved figure comes back.

    Args:
        figure: The figure to write; its sources must already be storable (see :func:`_saved`).

    Returns:
        The figure read back from the JSON text.
    """
    from digitalearth.base.spec import FigureSpec

    return FigureSpec.from_dict(
        json.loads(json.dumps(figure.to_dict(), allow_nan=False))
    )


class TestEngineKeywordsAreHeldBesideTheLayer:
    """The description records plain values; the caller's own engine keywords stay as they were passed.

    Freezing every keyword into the description turned a dash tuple into a list on the way back out, and
    wrote objects JSON cannot carry — a ``Normalize``, a ``FontProperties``, a per-pixel ``alpha`` array — into
    a figure that then could not be saved.
    """

    @pytest.mark.parametrize("call", sorted(TYPED_KEYWORD_CALLS))
    def test_a_keyword_whose_type_matters_reaches_matplotlib_intact(self, call):
        """Each of these raised once the dash tuple came back from the description as a list.

        Args:
            call: The entry in :data:`TYPED_KEYWORD_CALLS` under test.

        Test scenario:
            The map is lon/lat and framed on the whole world before drawing, so every label lands on the
            canvas: FreeType overflows rendering text millions of pixels off it, whatever its style.
        """
        canvas = Map(crs=4326)
        TYPED_KEYWORD_CALLS[call](canvas)
        canvas.ax.set_xlim(-180.0, 180.0)
        canvas.ax.set_ylim(-90.0, 90.0)
        canvas.fig.canvas.draw()
        assert canvas.layer_ids != [], canvas.layer_ids

    def test_a_keyword_object_is_held_as_the_object_passed(self, dataset):
        """A per-pixel ``alpha`` array is held, not copied into nested tuples of its cells.

        Args:
            dataset: The raster drawn.

        Test scenario:
            Freezing a ``1000 x 1000`` array cell by cell took ten times the render it belonged to (L6);
            held as passed, it costs nothing.
        """
        alpha = np.full(dataset.shape[-2:], 0.5)
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset, alpha=alpha)
        assert canvas._layer_opts["raster-1"]["alpha"] is alpha

    def test_the_description_records_no_engine_keywords(self, dataset):
        """What the caller passed through to matplotlib is not part of what the layer is.

        Args:
            dataset: The raster drawn.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset, vmin=0.0)
        props = canvas.figure_spec.layers.get("raster-1").symbology.props
        assert "opts" not in props, dict(props)

    def test_a_figure_holding_engine_objects_writes_to_json(self, dataset):
        """``Normalize`` and ``FontProperties`` have no JSON form, so they must not be in the description.

        Args:
            dataset: The raster drawn.
        """
        from matplotlib.colors import Normalize
        from matplotlib.font_manager import FontProperties

        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset, norm=Normalize(0.0, 10.0))
        canvas.text(
            4.9, 52.4, "Amsterdam", crs=dataset.epsg, fontproperties=FontProperties()
        )
        written = json.dumps(_saved(canvas.figure_spec).to_dict(), allow_nan=False)
        assert "raster-1" in written

    def test_a_figure_read_back_from_json_still_draws_with_defaults(self, dataset):
        """Another scene holds none of the caller's keywords, and must draw the layers anyway.

        Args:
            dataset: The raster drawn.

        Test scenario:
            A figure written to JSON keeps what the layers are and loses the engine keywords held beside
            them. Drawn on a scene that never held those, each layer draws with its defaults rather than
            failing on a keyword it cannot find.
        """
        from matplotlib.colors import Normalize

        drawn_from = Map(crs=dataset.epsg)
        drawn_from.imshow(dataset, norm=Normalize(0.0, 10.0))
        drawn_from.coastlines(linestyle=DASHED)
        figure = _written_and_read_back(_saved(drawn_from.figure_spec))
        target = Map(crs=dataset.epsg)
        target._renderer.apply(target.figure_spec, figure)
        assert sorted(target._renderer.drawn) == sorted(figure.layers.ids)


class TestANamedArgumentIsNormalisedForTheDescription:
    """A builder's own arguments are part of what the layer *is*, so they are recorded — as plain values.

    Each of these reached the description as an object with no JSON form, and took the whole figure with it.
    One that has a plain spelling is written in it; one that has none is held beside the layer like any
    other engine object.
    """

    def test_a_registered_colormap_is_recorded_by_name(self, dataset):
        """``colormaps["viridis"]`` and ``"viridis"`` name one colormap, and the description says so.

        Args:
            dataset: The raster drawn.
        """
        from matplotlib import colormaps

        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset, cmap=colormaps["viridis"])
        recorded = canvas.figure_spec.layers.get("raster-1").symbology.props["cmap"]
        drawn = canvas.ax.images[-1].get_cmap().name
        canvas.close()
        assert recorded == "viridis", recorded
        assert drawn == "viridis", drawn

    def test_a_colormap_of_the_callers_own_is_held_and_still_colours_the_layer(
        self, dataset
    ):
        """A hand-built colormap has no name to write down, so it travels beside the layer instead.

        Args:
            dataset: The raster drawn.
        """
        from matplotlib.colors import ListedColormap

        theirs = ListedColormap(["red", "blue"], name="two-tone")
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset, cmap=theirs)
        written = json.dumps(
            canvas.figure_spec.layers.to_dict(), allow_nan=False
        )  # the figure is storable
        drawn = canvas.ax.images[-1].get_cmap()
        canvas.close()
        assert drawn is theirs, drawn
        assert "raster-1" in written

    def test_a_crs_object_is_recorded_in_a_spelling_json_carries(self, dataset):
        """A label's CRS is what places it, so it is described — through the shared CRS spelling.

        Args:
            dataset: The raster whose CRS the map is drawn in.
        """
        from pyramids.base.crs import crs_from_user_input

        canvas = Map(crs=dataset.epsg)
        placed = canvas.text(4.9, 52.4, "Amsterdam", crs=crs_from_user_input(4326))
        by_code = Map(crs=dataset.epsg).text(4.9, 52.4, "Amsterdam", crs=4326)
        recorded = canvas.figure_spec.layers.get("text-1").symbology.props["crs"]
        json.dumps(canvas.figure_spec.layers.to_dict(), allow_nan=False)
        canvas.close()
        assert recorded == "EPSG:4326", recorded
        assert placed.get_position() == by_code.get_position()

    def test_a_channel_with_no_limits_to_freeze_is_recorded_as_none(self, dataset):
        """``channel_limits`` documents ``(nan, nan)`` for an unmeasurable channel; JSON has no nan.

        Args:
            dataset: The raster whose grid and CRS the composite is built on.

        Test scenario:
            The drawer reads ``None`` back as "no frozen bound for this channel", which is what a ``nan``
            pair meant, so the image is the one the caller's own limits draw.
        """
        base = np.nan_to_num(dataset.read_array(band=0).astype("float32"))
        stack = Dataset.from_array(
            arr=np.stack([base, base * 0.5, np.full(base.shape, np.nan, "float32")]),
            geo_ref=GeoReference(geo=dataset.geotransform, epsg=dataset.epsg),
        )
        limits = [(0.0, 1.0), (0.0, 1.0), (float("nan"), float("nan"))]
        canvas = Map(crs=dataset.epsg)
        canvas.rgb_composite(stack, limits=limits)
        recorded = canvas.figure_spec.layers.get("rgb-1").symbology.props["limits"]
        written = json.dumps(canvas.figure_spec.layers.to_dict(), allow_nan=False)
        canvas.close()
        assert recorded[-1] == (None, None), recorded
        assert "rgb-1" in written


class _FakeCollection:
    """A stand-in for a pyramids ``DatasetCollection``, which ``spaghetti`` reads one attribute of."""

    def __init__(self, datasets):
        """Hold the members ``spaghetti`` iterates.

        Args:
            datasets: The ensemble members, in order.
        """
        self.datasets = datasets
