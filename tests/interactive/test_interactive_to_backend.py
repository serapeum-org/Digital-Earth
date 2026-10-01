"""``to_backend(..., backend="interactive")`` — a HoloViz figure draws again (U-6, order 33).

Runs in the ``interactive`` environment: the replay goes through the HoloViz builders. The dispatch itself
and the matplotlib round trip live in ``tests/test_to_backend.py`` (core, no extra).
"""

from digitalearth import api
from digitalearth.interactive import InteractiveMap


class TestInteractiveRoundTrip:
    """A figure an interactive map describes draws again into a fresh interactive map."""

    def test_the_replayed_map_carries_the_same_layers(self, dataset):
        """Every layer id and kind survives the ``figure_spec`` → ``to_backend`` round trip.

        Args:
            dataset: The raster fixture, drawn as a field.
        """
        source = InteractiveMap(crs=3857)
        source.field(dataset)
        replayed = api.to_backend(source.figure_spec, backend="interactive")

        assert isinstance(replayed, InteractiveMap)
        assert replayed.layer_ids == source.layer_ids, (
            f"replayed {replayed.layer_ids} from {source.layer_ids}"
        )
        for layer_id in source.layer_ids:
            before, after = source.get_layer(layer_id), replayed.get_layer(layer_id)
            assert after.kind == before.kind, (
                f"layer {layer_id!r} changed kind on replay: {after.kind} vs {before.kind}"
            )

    def test_the_display_crs_is_carried_by_the_figure(self, dataset):
        """The replayed map is drawn in the CRS the figure's viewport names, without being told.

        The source is built in ``4326``, not the tier default ``3857``, so a ``from_figure`` that dropped the
        CRS would default the replay to ``3857`` and this assertion would fail — the test can only pass when
        the CRS is genuinely carried across.
        """
        source = InteractiveMap(crs=4326)
        source.field(dataset)
        replayed = api.to_backend(source.figure_spec, backend="interactive")
        assert replayed.viewport.crs == source.viewport.crs == 4326, (
            f"CRS not carried: source {source.viewport.crs}, replayed {replayed.viewport.crs}"
        )

    def test_explicit_set_bounds_framing_survives_the_round_trip(self, dataset):
        """A ``set_bounds`` frame is carried across the interactive round trip, not silently dropped.

        Args:
            dataset: The raster fixture, drawn before the view is framed on a region.
        """
        source = InteractiveMap(crs=4326)
        source.field(dataset)
        source.set_bounds([3.0, 50.0, 7.0, 54.0])
        replayed = api.to_backend(source.figure_spec, backend="interactive")
        assert replayed.viewport.bounds is not None, (
            "set_bounds framing lost on round trip"
        )
        assert replayed.viewport.bounds.as_bbox() == source.viewport.bounds.as_bbox(), (
            f"framing changed: {replayed.viewport.bounds.as_bbox()} vs {source.viewport.bounds.as_bbox()}"
        )
