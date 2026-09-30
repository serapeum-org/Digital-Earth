"""``to_backend(..., backend="web")`` — a web figure draws again into a fresh web map (U-6, order 33).

Runs in the ``web`` environment: the replay goes through MapLibre builders. The dispatch itself and the
matplotlib round trip live in ``tests/test_to_backend.py`` (core, no extra).
"""

from digitalearth import api
from digitalearth.web import WebMap


class TestWebRoundTrip:
    """A figure a web map describes draws again into a fresh web map, view and all."""

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
