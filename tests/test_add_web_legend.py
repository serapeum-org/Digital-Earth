"""``_add_web_legend`` — how ``quickmap(colorbar=True)`` reaches the web tier's colour key (#254).

The other three backends toggle a colorbar; the web tier *builds* a legend from the classification its last
layer recorded. ``api`` translates between the two, and the translation is only honest if it holds three
lines at once: build the key when there is one, stay quiet when there is not, and never swallow a failure
that is really a missing install.

These are unit tests against a stand-in map, so they need no ``web`` extra and run in the default ``dev``
environment alongside the rest of ``tests/*.py`` — which is also the point of the first test below: the
helper must answer "nothing to describe" from recorded state, without touching the engine at all.
"""

import logging

import pytest

from digitalearth.api import _add_web_legend

#: A minimal ``WebMap.last_legend`` record: enough to say a layer was classified.
CLASSIFIED = {
    "kind": "graduated",
    "column": "pop",
    "values": [1, 5, 9],
    "colors": ["#000"],
}


class _FakeWebMap:
    """A stand-in ``WebMap`` that records whether its legend builder was reached.

    Args:
        last_legend: What the most recent layer recorded, or ``None`` for a map with nothing classified.
        error: An exception ``legend()`` should raise instead of building, or ``None`` to build.
    """

    def __init__(self, last_legend=None, error=None):
        self.last_legend = last_legend
        self.error = error
        self.legend_calls = 0

    def legend(self):
        """Stand in for the tier's builder, counting the call and raising ``error`` when one was given.

        Returns:
            The same map, as the real builder does so calls chain.

        Raises:
            BaseException: whatever ``error`` was constructed with.
        """
        self.legend_calls += 1
        if self.error is not None:
            raise self.error
        return self


class _NoLegendAttribute:
    """A scene that never recorded a classification at all — not even the attribute."""


class TestAddWebLegend:
    """The three outcomes of asking a web map for a colour key."""

    def test_a_map_with_nothing_classified_never_reaches_the_builder(self):
        """An unclassified map is answered from recorded state, without calling ``legend()``.

        Test scenario:
            The builder calls ``_require_maplibre()`` before it checks whether it has anything to describe,
            so reaching it to *learn* there is no key raises ``ImportError`` wherever the ``web`` extra is
            absent — turning the ``colorbar=True`` default into a failure the caller never asked for. Asking
            ``last_legend`` first is what keeps the default inert on a map with nothing to key.
        """
        scene = _FakeWebMap(last_legend=None)
        assert _add_web_legend(scene) is None, (
            "a map with nothing classified has no key, so None must come back"
        )
        assert scene.legend_calls == 0, (
            f"the builder must not be reached at all, but was called {scene.legend_calls}x"
        )

    def test_a_scene_without_the_attribute_is_a_defect_not_an_unkeyed_map(self):
        """A scene carrying no ``last_legend`` raises, rather than being read as "nothing to describe".

        Test scenario:
            ``WebMapBase.__init__`` always sets the attribute, so its absence means the map is not a web map
            or the field was renamed. Answering "no key" there would make a rename silently drop every web
            legend while the dev-tier fakes kept passing; the only thing that would notice is the web CI leg.
        """
        with pytest.raises(AttributeError):
            _add_web_legend(_NoLegendAttribute())

    def test_a_classified_map_gets_its_key_built(self):
        """A recorded classification is passed to the builder, whose result is handed back.

        Test scenario:
            The half that has to actually happen: ``colorbar=True`` on a choropleth must produce the key the
            thematic map is unreadable without, which means the guard above cannot be so eager that it
            short-circuits a map that *does* have something to describe.
        """
        scene = _FakeWebMap(last_legend=CLASSIFIED)
        assert _add_web_legend(scene) is scene, (
            "the builder's result must be returned unchanged"
        )
        assert scene.legend_calls == 1, (
            f"the builder must be called exactly once, got {scene.legend_calls}"
        )

    @pytest.mark.parametrize("error", [ValueError, AttributeError, TypeError])
    def test_a_malformed_classification_surfaces_rather_than_being_swallowed(
        self, error
    ):
        """A builder that refuses a recorded classification is a library defect, so it propagates.

        Args:
            error: The exception the stubbed builder raises.

        Test scenario:
            Once the ``last_legend`` guard went in, the only way ``legend()`` can still raise is a malformed
            spec written by a web builder. Catching that reported a bug in this package as "no key to draw",
            which is exactly the kind of silent swallow the off-limb work spent Wave 0 removing.
        """
        scene = _FakeWebMap(last_legend={"kind": "graduated"}, error=error("bad spec"))
        with pytest.raises(error):
            _add_web_legend(scene)

    @pytest.mark.parametrize(
        "error",
        [
            ImportError("the web tier needs MapLibre GL JS + deck.gl (maplibre)"),
            RuntimeError("the widget is already closed"),
        ],
    )
    def test_a_failure_this_helper_does_not_own_propagates(self, error):
        """A missing extra or an engine fault is not a colour-key decision, so it is not swallowed here.

        Args:
            error: A failure outside :data:`~digitalearth.api.UNMAPPABLE`.

        Test scenario:
            The tolerated set is deliberately narrow. Widening it to bare ``Exception`` would turn "you have
            not installed ``digitalearth[web]``" into a map that silently comes back without a key, which is
            the exact class of silent failure the capability checks exist to remove.
        """
        scene = _FakeWebMap(last_legend=CLASSIFIED, error=error)
        with pytest.raises(type(error)):
            _add_web_legend(scene)
