"""``quickmap(colorbar=...)`` on the web tier when the input is a **raster** (#254).

``tests/web/test_web_colorbar.py`` covers the vector half, where a ``column`` classifies the layer and there
is a key to build. The raster half takes the same branch in ``_quickmap_web`` and used to end it with
nothing: ``field`` published no colour encoding and recorded no classification, so the tier had nothing to
describe and the ``colorbar=True`` default was *inert* on a raster — on this tier and on no other. Static,
interactive and 3-D all key that same one-call raster, which made web the odd tier out on order 24's own
goal. It is not inert any more: a raster's colour varies with its band, the band has a name and the band's
values have a span, so the layer carries an encoding a guide hangs on like any other.

What this file still guards is the older regression underneath it: before the tier's recorded state was
consulted first, the default reached ``WebMap.legend()``, which calls ``_require_maplibre()`` before it
checks whether it has anything to key — so a raster map could fail on the way to discovering it had no key
to draw.
"""

import pytest

import digitalearth.api as qp


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


class TestARasterWebMapFollowsTheFlag:
    """The flag decides whether the key is *attempted*; the tier decides whether there is one."""

    @pytest.mark.parametrize("colorbar", [True, False])
    def test_a_raster_map_is_built_either_way(self, dataset, colorbar):
        """Neither spelling of the flag stops a raster reaching the map.

        Args:
            dataset: The raster to draw.
            colorbar: The flag under test.

        Test scenario:
            ``colorbar`` used to be refused outright on this tier, so the same one-call line worked on three
            backends and raised on the fourth. Both values must now produce a finished map carrying the
            raster layer — the flag is about the key, never about whether the data is drawn.
        """
        from digitalearth.web import WebMap

        scene = qp.quickmap(dataset, backend="web", colorbar=colorbar)
        assert isinstance(scene, WebMap), (
            f"expected a WebMap, got {type(scene).__name__}"
        )
        assert scene.layers, (
            "the raster layer must be on the map whatever the flag says"
        )

    def test_the_default_keys_a_raster_as_every_other_tier_does(self, dataset):
        """``quickmap(raster, backend="web")`` comes back with the key three other tiers already draw.

        Args:
            dataset: The raster to draw.

        Test scenario:
            The measured divergence. `field` published no colour encoding, so the layer carried nothing a
            guide could hang on, and the one-call raster map ended with `_panels` empty — while
            `WebMap.colorbar()` on that same map raised "no classified layer has been added yet" about a
            map whose only layer is coloured by data. Static, interactive and 3-D all key it. The band is
            the value its colour varies with; naming it is what closes the gap.
        """
        scene = qp.quickmap(dataset, backend="web")
        assert scene.last_legend is not None, (
            "a raster's band is a classification this tier can describe"
        )
        assert "legend" in scene._panels, (
            f"the one-call raster map carries no key: panels {sorted(scene._panels)}"
        )


class TestTheBandIsWhatARasterKeyDescribes:
    """A raster has no classified column, so what its colour varies with is the band itself."""

    def test_a_raster_layer_can_be_keyed_by_hand_too(self, dataset):
        """`colorbar()` on a raster map is the call the divergence was measured through.

        Args:
            dataset: The raster to draw.

        Test scenario:
            The refusal a caller actually hit — "legend() has nothing to describe: no classified layer has
            been added yet. Add a choropleth (or any builder given column=...) first." — on a map holding
            one value-coloured raster. Asserted through the public method rather than through `quickmap`,
            so the builder is pinned independently of the dispatcher that calls it.
        """
        from digitalearth.web import WebMap

        scene = WebMap().field(dataset).colorbar(label="Accumulation")
        panel = scene._panels.get("legend")
        assert panel is not None, "colorbar() drew no key for a value-coloured raster"
        assert "Accumulation" in panel[0], (
            f"the key's heading is not the label asked for: {panel[0][:160]}"
        )

    def test_a_tiled_raster_is_keyed_on_the_same_terms(self, dataset, tmp_path):
        """`field`'s two routes describe the band the same way, or the divergence moves inside one builder.

        Args:
            dataset: The raster to draw.
            tmp_path: Where the pyramid is written.

        Test scenario:
            The tiled route never reads the band whole — that is what it exists to avoid — so it resolves
            one ``(vmin, vmax)`` for the pyramid from a decimated read and colours every tile on it. That
            pair is the one the key must label, which is why the assertion compares the scale against the
            limits the route *recorded* rather than against a number written here: a key built from a second
            reading of the band would disagree with the tiles at the edges of the range.
        """
        from digitalearth.web import WebMap

        scene = WebMap().field(
            dataset, tiles="xyz", tiles_path=tmp_path / "acc", zooms=(9, 9)
        )
        symbology = scene.get_layer(scene.layer_ids[-1]).symbology
        encoding = symbology.encoding("color")
        assert encoding is not None, "a tiled band is no less coloured by its values"
        assert encoding.field == "Band_1", encoding.field
        assert [encoding.scale.vmin, encoding.scale.vmax] == [
            symbology.props["vmin"],
            symbology.props["vmax"],
        ], (
            f"the key's scale {(encoding.scale.vmin, encoding.scale.vmax)} is not the pair the tiles were "
            f"coloured on {(symbology.props['vmin'], symbology.props['vmax'])}"
        )
        scene.colorbar(label="Accumulation")
        assert "legend" in scene._panels, "a tiled raster must be keyable too"

    def test_a_band_with_nothing_to_colour_publishes_no_encoding(self, dataset):
        """The measured "cannot": no finite cell, no span, so nothing honest to say about the colour.

        Args:
            dataset: A real raster, so the band the builder would name is a real one — the point being
                that it is not named at all.

        Test scenario:
            The other side of the choice. A band of pure NoData has no limits to describe — the draw
            refuses it on its own terms — and publishing a scale-less encoding would make the layer
            *keyable* with nothing to put in the key, which is the empty box in the corner the tier refuses
            everywhere else. So the builder publishes none, and a key on that layer is refused for the same
            reason it is refused on a flat-coloured one.

            Both halves are asserted, because the first alone is only about the helper: that no domain is
            reported, and that the builder handed that answer files neither an encoding for a guide to hang
            on nor the key's content beside it. `last_legend`/`last_breaks` are the half a scale-less
            encoding would leave inconsistent — a layer describing a colour with a key whose rows were
            never derived.
        """
        import numpy as np

        from digitalearth.web import WebMap
        from digitalearth.web.raster import _colour_domain

        assert _colour_domain(np.full((3, 3), np.nan), vmin=None, vmax=None) is None, (
            "a band with no finite cell must report no colour domain at all"
        )
        web_map = WebMap()
        published = web_map._band_colour(dataset, band=1, cmap="viridis", limits=None)
        assert published == {}, (
            f"a band with no span published {sorted(published)} for a guide to hang on"
        )
        assert (web_map.last_legend, web_map.last_breaks) == (None, None), (
            "the key's content was filed for a band whose colours nobody can name: "
            f"{web_map.last_legend}, {web_map.last_breaks}"
        )

    @pytest.mark.parametrize(
        ("colorbar", "expected_calls"),
        [(True, 1), (False, 0)],
    )
    def test_the_flag_decides_whether_the_key_is_attempted(
        self, dataset, mocker, colorbar, expected_calls
    ):
        """``True`` asks the tier for a key exactly once; ``False`` never asks.

        Args:
            dataset: The raster to draw.
            mocker: Spies on the translation helper.
            colorbar: The flag under test.
            expected_calls: How many times the helper should be reached.

        Test scenario:
            The raster branch of the dispatcher, asserted where it is decided rather than by what the
            finished map happens to carry — so this still holds the wiring if the tier later learns to key a
            raster. ``colorbar=False`` must be a real suppression here, not an accepted no-op: a flag that
            is honoured on three backends and merely tolerated on the fourth is the divergence #254 closes.
        """
        spy = mocker.spy(qp, "_add_web_legend")
        qp.quickmap(dataset, backend="web", colorbar=colorbar)
        assert spy.call_count == expected_calls, (
            f"colorbar={colorbar} must reach the key builder {expected_calls}x, got {spy.call_count}"
        )
