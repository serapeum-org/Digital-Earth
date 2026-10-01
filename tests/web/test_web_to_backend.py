"""``to_backend(..., backend="web")`` — a web figure draws again into a fresh web map (U-6, order 33).

Runs in the ``web`` environment: the replay goes through MapLibre builders. The dispatch itself and the
matplotlib round trip live in ``tests/test_to_backend.py`` (core, no extra).
"""

from digitalearth import api
from digitalearth.base.spec import Bounds
from digitalearth.static import Map
from digitalearth.web import WebMap


class TestWebRoundTrip:
    """A figure a web map describes draws again into a fresh web map, view and all."""

    def test_a_projected_sources_bounds_are_reprojected_to_lonlat(self, dataset):
        """A foreign figure framed in a projected CRS has its bounds reprojected to lon/lat for the web tier.

        Args:
            dataset: The raster fixture (EPSG:32618, projected metres).

        Test scenario:
            The web tier renders in EPSG:4326 and ``set_bounds`` takes lon/lat degrees. A static source framed
            with a projected ``Bounds`` (metres) must be reprojected on replay, not fed in as raw metres — so
            the replayed web view's bounds must be sane lon/lat degrees, not six-figure eastings/northings.
        """
        assert dataset.epsg != 4326, (
            "fixture must be projected to exercise reprojection"
        )
        source = Map(crs=dataset.epsg)
        source.field(dataset)
        source.set_bounds(
            Bounds(432968.0, 468007.0, 488968.0, 520007.0, crs=dataset.epsg)
        )
        replayed = api.to_backend(source.figure_spec, backend="web")
        bbox = replayed.viewport.bounds.as_bbox()
        assert all(abs(value) <= 180 for value in bbox), (
            f"web bounds were applied as raw projected metres, not reprojected to lon/lat: {bbox}"
        )

    def test_the_replayed_map_carries_the_same_layers(self, dataset):
        """Every layer id and kind survives the ``figure_spec`` → ``to_backend`` round trip.

        Args:
            dataset: The raster fixture, drawn as a field.
        """
        source = WebMap(crs=4326)
        source.field(dataset)
        replayed = api.to_backend(source.figure_spec, backend="web")

        assert isinstance(replayed, WebMap)
        assert replayed.layer_ids == source.layer_ids, (
            f"replayed {replayed.layer_ids} from {source.layer_ids}"
        )
        for layer_id in source.layer_ids:
            before, after = source.get_layer(layer_id), replayed.get_layer(layer_id)
            assert after.kind == before.kind, (
                f"layer {layer_id!r} changed kind on replay: {after.kind} vs {before.kind}"
            )

    def test_the_centre_and_zoom_are_carried_by_the_figure(self, dataset):
        """The view the web map was built to stand at is carried across by the figure, not lost.

        Args:
            dataset: The raster fixture, so the figure holds a layer as well as a view.

        Test scenario:
            ``center``/``zoom`` live on the viewport this tier's figure records, so a replayed map should
            stand where the original did without being told — the web-specific half of ``from_figure``.
        """
        source = WebMap(center=(4.9, 52.4), zoom=7, crs=4326)
        source.field(dataset)
        replayed = api.to_backend(source.figure_spec, backend="web")
        assert replayed.viewport.center == source.viewport.center
        assert replayed.viewport.zoom == source.viewport.zoom

    def test_fitted_bounds_and_globe_projection_survive_the_round_trip(self, dataset):
        """A web map framed with ``set_bounds`` and set to the globe projection keeps both on a round trip.

        Args:
            dataset: The raster fixture, so the figure holds a layer as well as the framing.

        Test scenario:
            The viewport records fitted ``bounds`` and ``globe`` alongside center/zoom; ``from_figure`` must
            restore them (the constructor takes neither), or a ``set_bounds``/globe web map loses its framing
            silently on replay.
        """
        source = WebMap(crs=4326)
        source.field(dataset)
        source.set_bounds([3.0, 50.0, 7.0, 54.0])
        source.projection("globe")
        replayed = api.to_backend(source.figure_spec, backend="web")
        assert replayed.viewport.globe is True, "globe projection dropped on replay"
        assert replayed.viewport.bounds is not None, "fitted bounds dropped on replay"
        assert replayed.viewport.bounds.as_bbox() == [3.0, 50.0, 7.0, 54.0], (
            f"fitted bounds not carried: {replayed.viewport.bounds.as_bbox()}"
        )
