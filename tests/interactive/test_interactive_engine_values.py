"""A figure holds plain values; the engine's own objects are held beside the layer (C1/H2/H3/M9).

The seam froze whatever a builder was handed into `Symbology.props` and thawed it back at the drawer. That
works for a number and a colour name and for nothing else: a `ListedColormap`, a Datashader reduction, a
`pandas.Timestamp` and an `xyzservices.TileProvider` have no JSON form, so the figure they built could not be
written down at all — and the provider, which is a `dict` subclass, was written down *with its API key* and
came back as a plain dict the engine could not draw.

These cover the answer: the description keeps the JSON-safe half, the map holds the engine values by layer id,
and a figure with nothing held still draws.
"""

import json
from collections.abc import Mapping

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")

#: A credential that is not a real key, used to show one never reaches a written figure.
FAKE_KEY = "FAKE-KEY-NOT-REAL"


def _written(interactive_map):
    """Return the map's described layers, written the way a figure is saved.

    The layers rather than the whole `FigureSpec`, because a figure whose data is an in-memory object
    refuses to write at all — `FigureSpec.to_dict` says so by name, and that is a property of the *source*,
    not of the symbology these tests are about. A source-less figure is written whole in
    `test_a_source_less_figure_writes_whole`.

    Args:
        interactive_map: The map whose description is written.

    Returns:
        The JSON text of `figure_spec.layers.to_dict()`.
    """
    return json.dumps(interactive_map.figure_spec.layers.to_dict())


@pytest.fixture
def point_fc():
    """Return a small point collection read through pyramids.

    Returns:
        A `FeatureCollection` of scattered points, with a numeric `fid` column.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygon_fc(point_fc):
    """Return polygons, by buffering the points.

    Args:
        point_fc: The point fixture.

    Returns:
        A polygon `FeatureCollection` carrying the same `fid` column.
    """
    features = point_fc.copy()
    features["geometry"] = features.geometry.buffer(500.0)
    return features


@pytest.fixture
def collection():
    """Return a two-member raster collection, for the time-slider builders.

    Returns:
        A pyramids `DatasetCollection`.
    """
    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(["examples/data/acc4000.tif"] * 2)


@pytest.fixture
def new_map():
    """Yield a factory for maps, closing every map it made on the way out.

    Yields:
        A callable taking the same keywords as `InteractiveMap` and returning a map.
    """
    built = []

    def make(**options):
        """Return a map that will be closed when the test ends.

        Args:
            **options: Passed straight to `InteractiveMap`.

        Returns:
            The map.
        """
        interactive_map = InteractiveMap(**options)
        built.append(interactive_map)
        return interactive_map

    yield make
    for interactive_map in built:
        interactive_map.close()


def _builder_calls(dataset, collection, point_fc, polygon_fc, **extra):
    """Return one call per builder this tier draws, by name.

    Args:
        dataset: A small raster.
        collection: A two-member raster collection.
        point_fc: Scattered points.
        polygon_fc: The same points buffered.
        **extra: Styling keywords added to every call, so a check about what a builder does with the
            caller's `**opts` can reuse this one table rather than restating thirty calls that would then
            drift from it.

    Returns:
        `{name: call(map)}`, where each call adds exactly one layer (or, for `features`, the named
        Natural-Earth layers) to the map it is given.
    """
    edges = [(0, 1), (1, 2)]
    return {
        "image": lambda m: m.image(dataset, **extra),
        "rgb": lambda m: m.rgb(dataset, bands=(1, 1, 1), **extra),
        "quadmesh": lambda m: m.quadmesh(dataset, **extra),
        "contours": lambda m: m.contours(dataset, levels=3, **extra),
        "filled_contours": lambda m: m.filled_contours(dataset, levels=3, **extra),
        "large_image": lambda m: m.large_image(dataset, dynamic=False, **extra),
        "timecube": lambda m: m.timecube(collection, **extra),
        "spaghetti": lambda m: m.spaghetti(collection, **extra),
        "points": lambda m: m.points(point_fc, **extra),
        "path": lambda m: m.path(point_fc, **extra),
        "polygons": lambda m: m.polygons(polygon_fc, **extra),
        "choropleth": lambda m: m.choropleth(polygon_fc, "fid", **extra),
        "choropleth-categorical": lambda m: m.choropleth(
            polygon_fc, "fid", scheme="categorical", **extra
        ),
        "choropleth-graduated": lambda m: m.choropleth(
            polygon_fc, "fid", scheme="quantiles", k=3, **extra
        ),
        "trimesh": lambda m: m.trimesh(point_fc, value_column="fid", **extra),
        "hexbin": lambda m: m.hexbin(point_fc, aggregator="count", **extra),
        "kde": lambda m: m.kde(point_fc, **extra),
        "graph": lambda m: m.graph(point_fc, edges, **extra),
        "vectorfield": lambda m: m.vectorfield(dataset, dataset, **extra),
        "streamlines": lambda m: m.streamlines(dataset, dataset, **extra),
        "barbs": lambda m: m.barbs(dataset, dataset, **extra),
        "rasterize": lambda m: m.rasterize(
            point_fc, dynamic=False, width=20, height=20, **extra
        ),
        "datashade": lambda m: m.datashade(
            point_fc, dynamic=False, width=20, height=20, **extra
        ),
        "trajectory": lambda m: m.trajectory(
            point_fc, dynamic=False, dynspread=False, width=20, height=20, **extra
        ),
        "tiles": lambda m: m.tiles("CartoLight", **extra),
        "coastlines": lambda m: m.coastlines(**extra),
        "features": lambda m: m.features(land=True, ocean=True, rivers=True, **extra),
        "graticule": lambda m: m.graticule(**extra),
        "text": lambda m: m.text(4.0, 52.0, "here", **extra),
        "labels": lambda m: m.labels(point_fc, "fid", **extra),
    }


class TestAHeldColormapStillReachesTheEngine:
    """Holding a value beside the layer must not stop the drawer from using it (round 2, H4)."""

    #: A colormap matplotlib does not know, so its *name* cannot stand in for the object anywhere.
    RAMP = ("#ff0000", "#00ff00", "#0000ff")

    @staticmethod
    def _engine_cmap(element):
        """Return the colormap the engine will draw an element with.

        Args:
            element: A HoloViews element the map has drawn.

        Returns:
            The `cmap` HoloViews holds for it, which is what Bokeh renders from.
        """
        import holoviews as hv

        return hv.Store.lookup_options("bokeh", element, "style").kwargs.get("cmap")

    def _drawn_from_the_ramp(self, element):
        """Whether an element is coloured by the caller's ramp, passed whole or sampled.

        Args:
            element: The drawn element.

        Returns:
            `True` when the engine holds the ramp itself, or colours taken from it — a graduated layer
            samples the ramp into one colour per class rather than handing the object over.
        """
        drawn = self._engine_cmap(element)
        if list(getattr(drawn, "colors", ())) == list(self.RAMP):
            return True
        sampled = list(drawn) if isinstance(drawn, (list, tuple)) else []
        return bool(sampled) and set(sampled) <= set(self.RAMP)

    @pytest.mark.parametrize(
        "builder", ["points", "polygons", "choropleth", "rasterize"]
    )
    def test_the_caller_s_colormap_reaches_the_drawn_element(
        self, new_map, point_fc, polygon_fc, builder
    ):
        """A colormap object held beside the layer must still colour the layer.

        Test scenario:
            Describing a held value put it at the top level of the held dict, while `draw_vector` and
            `draw_rasterize` build their style from `props["common"]` — so the object was held and never
            read, and HoloViews drew its own default. The two tests written alongside that code asserted
            on the *description* and used a registered colormap, whose name makes the description right
            while the draw stays wrong, so the whole suite stayed green. This one reads the engine, with a
            colormap matplotlib does not know.

        Args:
            new_map: The map factory.
            point_fc: A point collection.
            polygon_fc: A polygon collection.
            builder: The builder under test.
        """
        from matplotlib.colors import ListedColormap

        ramp = ListedColormap(list(self.RAMP), name="homemade-ramp")
        interactive_map = new_map()
        if builder == "points":
            interactive_map.points(point_fc, value_column="fid", cmap=ramp)
        elif builder == "polygons":
            interactive_map.polygons(polygon_fc, column="fid", cmap=ramp)
        elif builder == "choropleth":
            interactive_map.choropleth(
                polygon_fc, "fid", scheme="quantiles", k=2, cmap=ramp
            )
        else:
            # `dynamic=False` so the style lands on an element: a `DynamicMap` styles its frames, and
            # reading the map itself would report nothing whatever the drawer did.
            interactive_map.rasterize(point_fc, column="fid", cmap=ramp, dynamic=False)
        assert self._drawn_from_the_ramp(interactive_map.layers[-1]), (
            f"{builder} drew with {self._engine_cmap(interactive_map.layers[-1])!r}, "
            "not with the caller's colormap"
        )


class TestAStyleDictDescribesItsColormapByName:
    """A resolved style dict must spell a held colormap, whichever builder resolved it."""

    def test_a_point_layer_describes_a_colormap_object_by_name(self, new_map, point_fc):
        """A figure written from a point layer keeps the colormap, as every sibling builder's does.

        Test scenario:
            The builders that name each property one by one pass `cmap_name(cmap)` as the spelling, so a
            colormap object is described as `'magma'`. The three that describe a whole resolved style dict
            in one comprehension passed no spelling at all, so the same call through `points()` described
            `None` — the layer still drew, because the object is held beside it, but a figure written out
            and read back elsewhere lost the colormap for a point layer while keeping it for a polygon one.
        """
        from matplotlib import colormaps

        interactive_map = new_map()
        interactive_map.points(point_fc, value_column="fid", cmap=colormaps["magma"])
        props = dict(
            interactive_map.figure_spec.layers.get(
                interactive_map.layer_ids[-1]
            ).symbology.props
        )
        described = dict(props.get("common") or {})
        assert described.get("cmap") == "magma", (
            f"a held colormap must be described by name, got {described.get('cmap')!r}"
        )

    def test_a_polygon_layer_describes_it_the_same_way(self, new_map, polygon_fc):
        """The two builders must agree: the same colormap object describes the same way."""
        from matplotlib import colormaps

        interactive_map = new_map()
        interactive_map.polygons(polygon_fc, column="fid", cmap=colormaps["magma"])
        props = dict(
            interactive_map.figure_spec.layers.get(
                interactive_map.layer_ids[-1]
            ).symbology.props
        )
        described = dict(props.get("common") or {})
        assert described.get("cmap") == "magma", (
            f"a held colormap must be described by name, got {described.get('cmap')!r}"
        )


class TestEveryBuilderWritesAFigureThatCanBeSaved:
    """`figure_spec.to_dict()` is the seam's promise; a builder that breaks it breaks the seam."""

    @pytest.mark.parametrize("builder", sorted(_builder_calls(None, None, None, None)))
    def test_the_figure_a_builder_describes_is_json(
        self, builder, new_map, dataset, collection, point_fc, polygon_fc
    ):
        """Every builder, with ordinary arguments, describes a figure that writes down.

        Args:
            builder: The builder to call.
            new_map: The map factory.
            dataset: A small raster.
            collection: A two-member raster collection.
            point_fc: Scattered points.
            polygon_fc: The same points buffered.
        """
        interactive_map = new_map()
        _builder_calls(dataset, collection, point_fc, polygon_fc)[builder](
            interactive_map
        )
        assert interactive_map.layer_ids, "nothing was drawn"
        assert _written(interactive_map), "the description wrote down as nothing"

    def test_a_source_less_figure_writes_whole(self, new_map):
        """A figure drawing no in-memory data writes end to end, panels, view and all.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().tiles("CartoLight").graticule().text(4.0, 52.0, "x")
        assert json.dumps(interactive_map.figure_spec.to_dict()), "nothing was written"


class TestAnEngineValueIsHeldBesideTheLayer:
    """The four kinds of value that used to make a figure unwritable (review M9)."""

    def test_a_colormap_object_does_not_stop_the_figure_writing(self, new_map, dataset):
        """`image(cmap=ListedColormap(...))` draws, and its figure still writes down.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        from matplotlib.colors import ListedColormap

        interactive_map = new_map().image(
            dataset, cmap=ListedColormap(["#000000", "#ffffff"])
        )
        assert _written(interactive_map), "nothing was written"

    def test_a_registered_colormap_object_is_described_by_its_name(
        self, new_map, dataset
    ):
        """A colormap another reader can resolve is written down, rather than dropped.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        from matplotlib import colormaps

        interactive_map = new_map().image(dataset, cmap=colormaps["magma"])
        props = interactive_map.figure_spec.layers.get(
            interactive_map.layer_ids[0]
        ).symbology.props
        assert props["cmap"] == "magma", props

    def test_a_datashader_reduction_does_not_stop_the_figure_writing(
        self, new_map, point_fc
    ):
        """`rasterize(aggregator=ds.mean("fid"))` is a live object, and a common one to pass.

        Args:
            new_map: The map factory.
            point_fc: Scattered points with a numeric column.
        """
        import datashader as ds

        interactive_map = new_map().rasterize(
            point_fc, aggregator=ds.mean("fid"), dynamic=False, width=20, height=20
        )
        assert _written(interactive_map), "nothing was written"

    def test_a_hook_does_not_stop_the_figure_writing(self, new_map, dataset):
        """`hooks=[fn]` is a HoloViews keyword whose value is a function.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        interactive_map = new_map().image(dataset, hooks=[lambda plot, element: None])
        assert _written(interactive_map), "nothing was written"

    def test_timestamp_labels_do_not_stop_the_figure_writing(self, new_map, collection):
        """A time slider's labels are timestamps, which is the normal thing to label one with.

        Args:
            new_map: The map factory.
            collection: A two-member raster collection.
        """
        import pandas as pd

        interactive_map = new_map().timecube(
            collection, labels=pd.date_range("2024-01-01", periods=2, freq="D")
        )
        written = _written(interactive_map)
        assert "2024-01-01" in written, written

    def test_a_channel_with_no_finite_bound_does_not_stop_the_figure_writing(
        self, new_map, dataset
    ):
        """`channel_limits` documents `(nan, nan)` for a channel with no finite cell.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        limits = [(0.0, 1.0), (float("nan"), float("nan")), (0.0, 1.0)]
        interactive_map = new_map().rgb(dataset, bands=(1, 1, 1), limits=limits)
        assert _written(interactive_map), "nothing was written"


class TestAKeyedProviderKeepsItsCredentialOffTheFigure:
    """An `xyzservices.TileProvider` is a `dict`, so freezing it wrote every field it holds (review C1)."""

    @staticmethod
    def _keyed_provider():
        """Return a tile provider carrying an API key.

        Returns:
            The provider.
        """
        xyz = pytest.importorskip("xyzservices")
        return xyz.providers.Thunderforest.OpenCycleMap(apikey=FAKE_KEY)

    def test_the_key_is_not_in_the_written_figure(self, new_map):
        """The whole figure is searched, not the field the key would obviously sit in.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().tiles(self._keyed_provider())
        written = json.dumps(interactive_map.figure_spec.to_dict())
        assert FAKE_KEY not in written, written

    def test_the_provider_is_still_described(self, new_map):
        """What is written is enough for another reader to draw a basemap from.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().tiles(self._keyed_provider())
        props = interactive_map.figure_spec.layers.get(
            interactive_map.layer_ids[0]
        ).symbology.props
        assert "{x}" in str(props["provider"]), props


class TestWhatIsHeldReachesTheEngine:
    """Held or not, the value has to reach the drawer — and the drawer has to draw without it."""

    @staticmethod
    def _bokeh_plot(interactive_map):
        """Render a map through the Bokeh backend, as a notebook or `save()` does.

        Args:
            interactive_map: The map to render.

        Returns:
            The Bokeh plot HoloViews built.
        """
        return hv.renderer("bokeh").get_plot(interactive_map.render())

    def test_a_tile_provider_object_renders(self, new_map):
        """`frozen_value` rebuilt the provider as a plain dict, which the engine cannot draw (review H2).

        Args:
            new_map: The map factory.
        """
        xyz = pytest.importorskip("xyzservices")
        interactive_map = new_map().tiles(xyz.providers.OpenStreetMap.Mapnik)
        assert self._bokeh_plot(interactive_map) is not None, "nothing rendered"

    def test_a_hook_reaches_the_render_as_a_list(self, new_map, dataset):
        """Bokeh refuses `hooks` as a tuple, which is what a round trip through the description made it.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        fired = []

        def hook(plot, element):
            """Record that the render reached this hook.

            Args:
                plot: The Bokeh plot being built.
                element: The element being rendered.
            """
            fired.append(type(element).__name__)

        interactive_map = new_map().image(dataset, hooks=[hook])
        self._bokeh_plot(interactive_map)
        assert fired, "the hook never ran"

    def test_a_layer_with_nothing_held_still_draws(self, new_map, dataset):
        """A figure read from disk, or drawn on another map, carries no held values at all.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        from matplotlib.colors import ListedColormap

        built = new_map().image(dataset, cmap=ListedColormap(["#000000", "#ffffff"]))
        figure = built.figure_spec
        elsewhere = new_map()
        drawn = elsewhere._renderer.draw_layer(figure, built.layer_ids[0])
        assert isinstance(drawn.element, hv.Image), type(drawn.element)

    def test_what_is_held_goes_when_the_map_closes(self, new_map, dataset):
        """The held values are the caller's own objects, so a closed map must let them go.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        interactive_map = new_map().image(dataset, hooks=[lambda plot, element: None])
        assert interactive_map._layer_held, "nothing was held"
        interactive_map.close()
        assert interactive_map._layer_held == {}, interactive_map._layer_held


def test_a_held_value_is_not_the_figures_business(new_map, dataset):
    """The description says what a reader needs; the engine value stays out of it.

    Args:
        new_map: The map factory.
        dataset: A small raster.

    Test scenario:
        The one thing a caller can check without reaching into the map: what they passed is not written
        anywhere in the figure, however deeply it is nested. A plot hook is a **function**, which is the
        case that matters — this used to pass a `numpy` array instead, which JSON carries perfectly well
        and which the figure is now right to keep (round 2, M3). The keyword is absent rather than
        recorded as `null`, because this bag is splatted into `element.opts(**opts)`, where `hooks=None`
        is a value the engine is handed rather than a keyword it was never given.
    """
    interactive_map = new_map().image(dataset, hooks=[lambda plot, element: None])
    written = _written(interactive_map)
    assert "hooks" not in written, written


def _carries(props, value):
    """Whether a description records a value, at its top level or inside one of its mappings.

    Which key a builder records a keyword under is its own business — `image()` names `alpha` in its
    signature and writes `props["alpha"]`, the style-dict builders write `props["common"]`, and everything
    reaching a builder through `**opts` writes `props["opts"]`. What the seam promises is that the value is
    in the figure *somewhere*, so that is what this asks.

    Args:
        props: The layer's recorded properties.
        value: The value the caller passed.

    Returns:
        `True` when the description carries it.
    """
    for recorded in props.values():
        if recorded == value:
            return True
        if isinstance(recorded, Mapping) and any(
            item == value for item in recorded.values()
        ):
            return True
    return False


class TestAKeywordJsonCanCarryIsWrittenIntoTheFigure:
    """The description keeps what JSON can carry, and the map holds only what it cannot (round 2, M3).

    Round 1's decision was that a figure keeps every value JSON can carry. `describe()` implements exactly
    that, per value — and `**opts` went round it: every builder froze the whole bag into `held` untested,
    so which keywords a figure kept was decided by which ones the builder happened to name in its
    signature. `image(alpha=0.25)` survived because `alpha` is a parameter; `quadmesh(alpha=0.25)` did not,
    although both are plain floats and both are declared channels in the shared vocabulary. A figure read
    back drew the engine's defaults where the caller's styling had been, with no warning and nothing in the
    JSON to mark the omission.
    """

    #: The styling keyword each builder is given. `alpha` for almost all of them; the four element types
    #: HoloViews refuses it for get one their own options declare, so every builder is covered rather than
    #: four being quietly dropped from the sweep.
    KEYWORDS = {
        "graph": "node_alpha",
        "trimesh": "node_alpha",
        "labels": "text_alpha",
        "text": "text_alpha",
    }

    #: A plain float — JSON carries it, and no builder resolves it to something else.
    VALUE = 0.25

    @pytest.mark.parametrize("builder", sorted(_builder_calls(None, None, None, None)))
    def test_every_builder_writes_it_down(
        self, builder, new_map, dataset, collection, point_fc, polygon_fc
    ):
        """One call per builder, because the defect was per builder rather than per tier.

        Args:
            builder: The builder to call.
            new_map: The map factory.
            dataset: A small raster.
            collection: A two-member raster collection.
            point_fc: Scattered points.
            polygon_fc: The same points buffered.
        """
        keyword = self.KEYWORDS.get(builder, "alpha")
        interactive_map = new_map()
        _builder_calls(
            dataset, collection, point_fc, polygon_fc, **{keyword: self.VALUE}
        )[builder](interactive_map)
        layer = interactive_map.figure_spec.layers.get(interactive_map.layer_ids[-1])
        props = dict(layer.symbology.props)
        assert _carries(props, self.VALUE), (
            f"{builder}({keyword}={self.VALUE}) is not in the figure: {props}"
        )

    @pytest.mark.parametrize("builder", sorted(_builder_calls(None, None, None, None)))
    def test_it_survives_being_written_and_read_back(
        self, builder, new_map, dataset, collection, point_fc, polygon_fc
    ):
        """In the figure is not enough; it has to come back out of the JSON the same.

        Args:
            builder: The builder to call.
            new_map: The map factory.
            dataset: A small raster.
            collection: A two-member raster collection.
            point_fc: Scattered points.
            polygon_fc: The same points buffered.

        Test scenario:
            A value recorded as something JSON cannot spell would pass the check above and still be lost on
            the way to disk, which is the failure the whole seam exists to prevent.
        """
        from digitalearth.base.spec import LayerTree

        keyword = self.KEYWORDS.get(builder, "alpha")
        interactive_map = new_map()
        _builder_calls(
            dataset, collection, point_fc, polygon_fc, **{keyword: self.VALUE}
        )[builder](interactive_map)
        layer_id = interactive_map.layer_ids[-1]
        reread = LayerTree.from_dict(
            json.loads(json.dumps(interactive_map.figure_spec.layers.to_dict()))
        )
        props = dict(reread.get(layer_id).symbology.props)
        assert _carries(props, self.VALUE), (
            f"{builder}({keyword}={self.VALUE}) did not survive the round trip: {props}"
        )


class TestAKeywordJsonCannotCarryIsStillHeld:
    """The other half of the rule, which widening the description must not break (round 2, M3).

    Routing `**opts` through the per-value test is only right if the test is still applied. A bag recorded
    wholesale would write a `ListedColormap` into `Symbology.props`, which is what round 1's C1/H2 were
    about: the figure then refuses to write at all, or writes something a reader cannot draw from.
    """

    @staticmethod
    def _homemade():
        """Return a colormap matplotlib's registry does not know.

        Returns:
            A `ListedColormap`, whose name resolves nowhere, so it has no JSON spelling at all.
        """
        from matplotlib.colors import ListedColormap

        return ListedColormap(["#ff0000", "#00ff00", "#0000ff"], name="homemade-ramp")

    def test_it_is_not_in_the_description(self, new_map, dataset):
        """`contours()` does not name `cmap`, so the object arrives through `**opts`.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        ramp = self._homemade()
        interactive_map = new_map()
        interactive_map.contours(dataset, levels=3, cmap=ramp)
        props = dict(
            interactive_map.figure_spec.layers.get(
                interactive_map.layer_ids[-1]
            ).symbology.props
        )
        recorded = dict(props.get("opts") or {})
        assert recorded.get("cmap") is None, (
            f"a colormap object reached the description: {recorded.get('cmap')!r}"
        )

    def test_it_is_held_beside_the_layer(self, new_map, dataset):
        """Kept out of the figure, but not thrown away — the drawer still colours by it.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        ramp = self._homemade()
        interactive_map = new_map()
        interactive_map.contours(dataset, levels=3, cmap=ramp)
        held = interactive_map._held_for(interactive_map.layer_ids[-1])
        assert dict(held.get("opts") or {}).get("cmap") is ramp, held

    def test_the_figure_still_writes(self, new_map, dataset):
        """The point of holding it: a layer carrying an engine object is still a storable figure.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        interactive_map = new_map()
        interactive_map.contours(dataset, levels=3, cmap=self._homemade())
        assert _written(interactive_map), "the figure wrote nothing at all"


class TestAContainerKeywordIsHeldRatherThanDescribed:
    """A container travels no better than an engine object, so this tier holds it too (#322).

    This tier used to ask its own oracle, which stopped at what the figure *writer* accepts — so a container
    of plain values was described. A description freezes every sequence to a tuple, because JSON has no
    tuple, and the drawer then splatted that tuple into `element.opts(...)`: the engine was handed a
    spelling the caller had not written. The static tier had already measured the same trip the other way
    round, where a `linestyle` of `(0, (5, 5))` comes back `[0, [5, 5]]` and matplotlib refuses it outright.
    The rule is now the shared, narrower one.

    The dangerous half is holding: a value kept out of the description and then never read leaves the engine
    drawing its own default, silently — the defect that left four builders on this tier ignoring the
    caller's colormap. So the last check here reads HoloViews, not the description.
    """

    #: The dash pattern the caller passes, as a **list** — the spelling a caller writes and the one Bokeh is
    #: meant to receive. Described, it was frozen to the tuple `(4, 4)` on the way into `Symbology.props`.
    DASH = [4, 4]

    @staticmethod
    def _drawn_dash(element):
        """Return the dash pattern HoloViews will draw an element with.

        Args:
            element: A HoloViews element the map has drawn.

        Returns:
            The `line_dash` HoloViews holds for it in the Bokeh style options, which is what the renderer
            builds the glyph from.
        """
        return hv.Store.lookup_options("bokeh", element, "style").kwargs.get(
            "line_dash"
        )

    def _with_a_dash(self, new_map, point_fc):
        """Return a map holding one line layer the caller styled with a dash pattern.

        Args:
            new_map: The map factory.
            point_fc: A point collection, drawn as a path.

        Returns:
            The map. The pattern is passed as a **copy**, so nothing here can pass by identity alone.
        """
        return new_map().path(point_fc, line_dash=list(self.DASH))

    def test_the_description_does_not_carry_it(self, new_map, point_fc):
        """A figure written out names no dash pattern at all.

        Args:
            new_map: The map factory.
            point_fc: A point collection.
        """
        written = _written(self._with_a_dash(new_map, point_fc))
        assert "line_dash" not in written, written

    def test_the_map_holds_it_instead(self, new_map, point_fc):
        """Kept out of the figure, but not thrown away — the map keeps the caller's own list.

        Args:
            new_map: The map factory.
            point_fc: A point collection.
        """
        interactive_map = self._with_a_dash(new_map, point_fc)
        held = interactive_map._held_for(interactive_map.layer_ids[-1])
        assert dict(held.get("opts") or {}).get("line_dash") == self.DASH, held

    def test_the_engine_still_draws_the_caller_s_pattern(self, new_map, point_fc):
        """The drawn object is unchanged: HoloViews holds the list the caller wrote.

        Args:
            new_map: The map factory.
            point_fc: A point collection.

        Test scenario:
            This is the half that cannot be checked against the description. Held and never read, the layer
            draws Bokeh's default and the two checks above still pass. Described instead of held, HoloViews
            is handed `(4, 4)` — equal by nothing the caller wrote — which is what this refuses.
        """
        interactive_map = self._with_a_dash(new_map, point_fc)
        drawn = self._drawn_dash(interactive_map.layers[-1])
        assert drawn == self.DASH, (
            f"HoloViews was given {drawn!r}, not the dash pattern the caller passed"
        )

    def test_the_figure_still_writes(self, new_map, point_fc):
        """Holding more must not stop a figure being written down.

        Args:
            new_map: The map factory.
            point_fc: A point collection.
        """
        assert _written(self._with_a_dash(new_map, point_fc)), "nothing was written"
