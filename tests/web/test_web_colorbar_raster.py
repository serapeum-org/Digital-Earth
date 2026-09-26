"""``quickmap(colorbar=...)`` on the web tier when the input is a **raster** (#254).

``tests/web/test_web_colorbar.py`` covers the vector half, where a ``column`` classifies the layer and there
is a key to build. The raster half takes the same new branch in ``_quickmap_web`` and behaves differently at
the end of it — ``field`` records no classification, so the tier has nothing to describe — which makes
it the case where the ``colorbar=True`` default has to stay *inert* rather than raise.

That is the regression this file guards: before the tier's recorded state was consulted first, the default
reached ``WebMap.legend()``, which calls ``_require_maplibre()`` before it checks whether it has anything to
key — so a raster map could fail on the way to discovering it had no key to draw.
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

    def test_the_default_is_inert_on_a_raster_rather_than_a_failure(self, dataset):
        """``quickmap(raster, backend="web")`` succeeds without a classified layer to key.

        Args:
            dataset: The raster to draw.

        Test scenario:
            The default asks for a key *if there is one to draw*. This tier records a classification only
            for a classified vector layer, so a raster reaches the end of the build with nothing to
            describe. That is not a caller error and must not surface as one — neither as the builder's own
            "nothing to describe" refusal nor, where the extra is thin, as the ``ImportError`` raised on the
            way to it.
        """
        scene = qp.quickmap(dataset, backend="web")
        assert scene.last_legend is None, (
            f"a raster records no classification, so there is nothing to key: {scene.last_legend}"
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
