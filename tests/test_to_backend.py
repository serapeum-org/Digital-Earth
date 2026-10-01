"""``to_backend`` — replay one ``FigureSpec`` onto a chosen backend (U-6, order 33).

The dispatcher lives in :mod:`digitalearth.api` and is re-exported at the package root. These tests cover the
dispatch itself and the matplotlib tier's same-tier round trip, both of which run in the core ``dev``
environment. The other three tiers' round trips live beside their own suites, because each replays through
builders that need that tier's engine (``tests/web``, ``tests/interactive``, ``tests/three_d``).
"""

import pytest

import digitalearth
from digitalearth import api
from digitalearth.base.spec import FigureSpec, PanelSpec


def _one_panel_figure() -> FigureSpec:
    """A minimal valid figure, for the checks that never reach a renderer.

    Returns:
        A figure with a single empty panel — enough to satisfy ``FigureSpec`` for the unknown-backend guard,
        which refuses before anything is built.
    """
    return FigureSpec(panels=(PanelSpec("main"),))


class TestDispatch:
    """The backend switch: the public name, the refusal, and the default."""

    def test_the_package_level_name_is_the_api_function(self):
        """``digitalearth.to_backend`` resolves to the very function :mod:`digitalearth.api` defines."""
        assert digitalearth.to_backend is api.to_backend

    def test_an_unknown_backend_is_refused_by_name_before_building(self):
        """A backend outside the four raises ``ValueError`` naming it, touching no renderer."""
        # figure built above the block so only `to_backend` can raise inside it (refusal-block tree guard)
        figure = _one_panel_figure()
        with pytest.raises(ValueError, match="unknown backend 'nope'"):
            api.to_backend(figure, backend="nope")

    def test_a_non_figurespec_is_refused_with_a_clear_message(self):
        """A bad ``figure`` with a valid backend raises ``TypeError`` naming the expected type, not an opaque
        ``AttributeError`` from deep inside a tier's ``from_figure``.
        """
        argument = {"not": "a figure"}
        with pytest.raises(TypeError, match="to_backend draws a FigureSpec"):
            api.to_backend(argument, backend="matplotlib")

    def test_a_multi_panel_figure_is_refused(self):
        """A figure with more than one panel is refused: the tiers each render a single panel, so a
        multi-panel figure would otherwise be silently flattened into panel 0's view.
        """
        figure = FigureSpec(panels=(PanelSpec("a"), PanelSpec("b")))
        with pytest.raises(ValueError, match="single-panel figure"):
            api.to_backend(figure, backend="matplotlib")

    def test_the_default_backend_is_matplotlib(self, dataset):
        """``to_backend(figure)`` with no backend builds the static tier, the same default ``quickmap`` has."""
        from digitalearth.static import Map

        source = Map(crs=dataset.epsg)
        source.field(dataset)
        drawn = api.to_backend(source.figure_spec)
        assert isinstance(drawn, Map)


class TestMatplotlibRoundTrip:
    """A figure a static map describes draws again into a fresh static map."""

    def test_the_replayed_map_carries_the_same_layers(self, dataset):
        """Every layer id, kind and visibility survives the description → ``to_backend`` round trip.

        Args:
            dataset: The raster fixture, drawn as a field.

        Test scenario:
            The round trip the seam exists for, in memory: the source map publishes a ``figure_spec``,
            ``to_backend`` replays it onto a new static map, and the two describe the same layers. An
            in-memory source is enough here — what is under test is the replay, not ``FigureSpec``'s own JSON
            round trip, which its own suite covers.
        """
        from digitalearth.static import Map

        source = Map(crs=dataset.epsg)
        source.field(dataset, name="dem")
        replayed = api.to_backend(source.figure_spec, backend="matplotlib")

        assert replayed.layer_ids == source.layer_ids, (
            f"replayed {replayed.layer_ids} from {source.layer_ids}"
        )
        for layer_id in source.layer_ids:
            before, after = source.get_layer(layer_id), replayed.get_layer(layer_id)
            assert (after.kind, after.visible) == (before.kind, before.visible), (
                f"layer {layer_id!r} changed on replay: "
                f"{(after.kind, after.visible)} vs {(before.kind, before.visible)}"
            )

    def test_the_display_crs_is_carried_by_the_figure(self, dataset):
        """The replayed map is drawn in the CRS the figure's viewport names, without being told."""
        from digitalearth.static import Map

        source = Map(crs=dataset.epsg)
        source.field(dataset)
        replayed = api.to_backend(source.figure_spec, backend="matplotlib")
        assert replayed.crs == source.crs

    def test_explicit_set_bounds_framing_survives_the_round_trip(self, dataset):
        """A map framed on a subregion via ``set_bounds`` keeps that frame on replay.

        Args:
            dataset: The raster fixture, drawn then framed on a sub-box of its extent.

        Test scenario:
            ``set_bounds`` records a region on the viewport the figure carries; a round trip that dropped it
            would silently replay the full data extent instead of the asked-for subregion.
        """
        from digitalearth.static import Map

        source = Map(crs=dataset.epsg)
        source.field(dataset)
        source.set_bounds([440000.0, 470000.0, 480000.0, 510000.0])
        replayed = api.to_backend(source.figure_spec, backend="matplotlib")
        assert replayed.viewport.bounds is not None, (
            "set_bounds framing lost on round trip"
        )
        assert replayed.viewport.bounds.as_bbox() == source.viewport.bounds.as_bbox(), (
            f"framing changed: {replayed.viewport.bounds.as_bbox()} vs {source.viewport.bounds.as_bbox()}"
        )
