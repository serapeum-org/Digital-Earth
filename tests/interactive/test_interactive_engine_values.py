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

import numpy as np
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


def _builder_calls(dataset, collection, point_fc, polygon_fc):
    """Return one call per builder this tier draws, by name.

    Args:
        dataset: A small raster.
        collection: A two-member raster collection.
        point_fc: Scattered points.
        polygon_fc: The same points buffered.

    Returns:
        `{name: call(map)}`, where each call adds exactly one layer (or, for `features`, the named
        Natural-Earth layers) to the map it is given.
    """
    edges = [(0, 1), (1, 2)]
    return {
        "image": lambda m: m.image(dataset),
        "rgb": lambda m: m.rgb(dataset, bands=(1, 1, 1)),
        "quadmesh": lambda m: m.quadmesh(dataset),
        "contours": lambda m: m.contours(dataset, levels=3),
        "filled_contours": lambda m: m.filled_contours(dataset, levels=3),
        "large_image": lambda m: m.large_image(dataset, dynamic=False),
        "timecube": lambda m: m.timecube(collection),
        "spaghetti": lambda m: m.spaghetti(collection),
        "points": lambda m: m.points(point_fc),
        "path": lambda m: m.path(point_fc),
        "polygons": lambda m: m.polygons(polygon_fc),
        "choropleth": lambda m: m.choropleth(polygon_fc, "fid"),
        "choropleth-categorical": lambda m: m.choropleth(
            polygon_fc, "fid", scheme="categorical"
        ),
        "choropleth-graduated": lambda m: m.choropleth(
            polygon_fc, "fid", scheme="quantiles", k=3
        ),
        "trimesh": lambda m: m.trimesh(point_fc, value_column="fid"),
        "hexbin": lambda m: m.hexbin(point_fc, aggregator="count"),
        "kde": lambda m: m.kde(point_fc),
        "graph": lambda m: m.graph(point_fc, edges),
        "vectorfield": lambda m: m.vectorfield(dataset, dataset),
        "streamlines": lambda m: m.streamlines(dataset, dataset),
        "barbs": lambda m: m.barbs(dataset, dataset),
        "rasterize": lambda m: m.rasterize(
            point_fc, dynamic=False, width=20, height=20
        ),
        "datashade": lambda m: m.datashade(
            point_fc, dynamic=False, width=20, height=20
        ),
        "trajectory": lambda m: m.trajectory(
            point_fc, dynamic=False, dynspread=False, width=20, height=20
        ),
        "tiles": lambda m: m.tiles("CartoLight"),
        "coastlines": lambda m: m.coastlines(),
        "features": lambda m: m.features(land=True, ocean=True, rivers=True),
        "graticule": lambda m: m.graticule(),
        "text": lambda m: m.text(4.0, 52.0, "here"),
        "labels": lambda m: m.labels(point_fc, "fid"),
    }


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
        anywhere in the figure, however deeply it is nested.
    """
    aggregate = np.arange(3.0)
    interactive_map = new_map().image(dataset, hooks=[aggregate])
    written = _written(interactive_map)
    assert "hooks" not in written, written
