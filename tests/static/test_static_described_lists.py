"""A described list reaches matplotlib as the list the caller wrote (#330).

The shared rule a figure is written by refuses a value the round trip re-types, and takes one it does not:
a tuple is held beside the layer, a **list** is described. That is only half a fix. A spec freezes every
list to a tuple so it still hashes, so the tier has to thaw the described half back at the point it hands
keywords to the engine — otherwise the caller's `levels=[10, 20, 30]` reaches cleopatra as
`(10.0, 20.0, 30.0)`, a spelling the caller never wrote, and the widening has swapped one silent
mis-drawing for another.

This tier has two such points, and each is checked here against what the drawer actually handed over
rather than against the description:

* :func:`~digitalearth.static.scene.drawing_style` — the caller's own `**opts`, described and then laid
  under whatever the scene holds.
* :func:`~digitalearth.static.maps.raster.draw_field` — the builder's own recorded properties, `levels`
  among them, which is the key that used to be thawed by name while every other key was not.
"""

import json

import pytest

from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.static import Map, Scene
from digitalearth.static.maps.raster import draw_field
from digitalearth.static.scene import drawing_style

#: Contour edges a caller would write by hand — a list, and one the round trip returns unchanged.
LEVELS = [10.0, 20.0, 30.0]

#: A cleopatra keyword that reaches the glyph through ``**opts`` rather than through a builder parameter,
#: so it exercises the *caller's* half of the description rather than the builder's.
ALPHA_RANGE = [0.2, 0.8]


def _reloaded(layer):
    """Return a layer's description written to JSON and read back, as a figure from elsewhere arrives.

    Args:
        layer: The `LayerSpec` the builder recorded.

    Returns:
        An equal `LayerSpec` that has been through the text, carrying nothing the scene holds beside it.
    """
    return LayerSpec.from_dict(json.loads(json.dumps(layer.to_dict())))


@pytest.fixture
def render_calls(monkeypatch):
    """Record the keywords every drawer hands the glyph, without changing what is drawn.

    `Scene._render_glyph` is the one call between a drawer and cleopatra, so what passes through it is
    what the engine was asked for — which is the only place the description's spelling can be checked
    against the caller's.

    Args:
        monkeypatch: pytest's patcher, which restores the method afterwards.

    Yields:
        A list that fills with one keyword dict per glyph rendered.
    """
    calls = []
    original = Scene._render_glyph

    def recording(self, glyph, **kwargs):
        """Record the keywords, then render for real.

        Args:
            self: The scene rendering.
            glyph: The cleopatra glyph being rendered.
            **kwargs: What the drawer asked for.

        Returns:
            Whatever the real render returned.
        """
        calls.append(dict(kwargs))
        return original(self, glyph, **kwargs)

    monkeypatch.setattr(Scene, "_render_glyph", recording)
    yield calls


class TestTheCallersOwnKeywordsAreThawedAtTheReadBoundary:
    """`drawing_style` is where a caller's `**opts` reach the drawer, so it is where they are thawed.

    On the scene that built the layer the held copy wins and nothing is thawed at all — the caller's own
    object is handed straight over. The described half only decides what a scene holding *nothing* draws,
    which is every scene that read the figure from somewhere else.
    """

    def test_a_scene_that_holds_nothing_hands_the_drawer_a_list(self):
        """A hand-built description is what a reader has, and it must draw the caller's spelling.

        Test scenario:
            The description is written with a list; `Symbology` freezes it to a tuple, as it freezes every
            sequence so that a spec still hashes. Without the thaw the drawer forwards that tuple.
        """
        layer = LayerSpec(
            "raster-1",
            "raster",
            symbology=Symbology(props={"opts": {"alpha_range": ALPHA_RANGE}}),
        )
        with Scene() as scene:
            style = drawing_style(scene, layer)
        drawn = style["alpha_range"]
        assert drawn == ALPHA_RANGE, f"the drawer was handed {drawn!r}"
        assert isinstance(drawn, list), (
            f"and as a {type(drawn).__name__}; the description's tuple was not thawed back"
        )

    def test_a_figure_written_and_read_back_draws_with_the_caller_s_list(self, dataset):
        """The same, reached the way a user reaches it: through the text and onto another scene.

        Args:
            dataset: The raster drawn.
        """
        built = Map(crs=dataset.epsg)
        built.imshow(dataset, alpha_range=list(ALPHA_RANGE))
        layer = built.figure_spec.layers.get(built.layer_ids[-1])
        built.close()
        with Scene() as elsewhere:
            style = drawing_style(elsewhere, _reloaded(layer))
        assert style.get("alpha_range") == ALPHA_RANGE, (
            f"a reloaded figure drew with {style.get('alpha_range')!r}, not the caller's range"
        )

    def test_a_held_keyword_is_not_thawed_with_it(self, dataset):
        """Only the described half is thawed; the scene's own objects are handed over untouched.

        Args:
            dataset: The raster drawn.

        Test scenario:
            A tuple is held rather than described precisely so the engine is handed the caller's own
            object — a matplotlib dash pattern is the value the rule is argued from, since matplotlib
            reads `(0, (5, 5))` and refuses `[0, [5, 5]]` outright. Thawing the merged keywords rather
            than the described half would undo that for every held tuple at once.
        """
        given = tuple(ALPHA_RANGE)
        built = Map(crs=dataset.epsg)
        built.imshow(dataset, alpha_range=given)
        layer = built.figure_spec.layers.get(built.layer_ids[-1])
        described = dict(layer.symbology.props.get("opts") or {})
        style = drawing_style(built, layer)
        built.close()
        assert "alpha_range" not in described, (
            f"a tuple must be held rather than described, and the figure carries {described}"
        )
        assert style["alpha_range"] == given, (
            f"the drawer was handed {style['alpha_range']!r}"
        )
        assert isinstance(style["alpha_range"], tuple), (
            f"and as a {type(style['alpha_range']).__name__}; a held value must not be thawed"
        )


class TestABuildersOwnRecordedListIsThawedToo:
    """The raster drawer reads properties the *builder* recorded, and those are frozen just the same.

    `levels` used to be thawed by name, one key in one drawer, while every other recorded container was
    forwarded frozen. Thawing the properties once covers the keys nobody has thought of yet — which is the
    failure mode the per-key list had: a value only reaches the engine correctly once someone remembers to
    add it.
    """

    def test_the_glyph_is_given_the_edges_as_a_list(self, dataset, render_calls):
        """On the scene that built it: `levels` is recorded, so even here it arrives frozen.

        Args:
            dataset: The raster drawn.
            render_calls: The recorder of what each drawer handed the glyph.
        """
        built = Map(crs=dataset.epsg)
        built.contour(dataset, levels=list(LEVELS))
        recorded = dict(
            built.figure_spec.layers.get(built.layer_ids[-1]).symbology.props
        )
        built.close()
        assert recorded["levels"] == tuple(LEVELS), (
            f"the description is expected to hold the frozen spelling, and holds {recorded['levels']!r}"
        )
        drawn = render_calls[-1]["levels"]
        assert isinstance(drawn, list), (
            f"cleopatra was given {type(drawn).__name__} edges, not the list the caller wrote"
        )

    def test_a_reloaded_layer_draws_with_the_same_edges(self, dataset, render_calls):
        """And on a scene that holds nothing, which is the one a stored figure is opened on.

        Args:
            dataset: The raster drawn.
            render_calls: The recorder of what each drawer handed the glyph.
        """
        built = Map(crs=dataset.epsg)
        built.contour(dataset, levels=list(LEVELS))
        layer = built.figure_spec.layers.get(built.layer_ids[-1])
        built.close()
        render_calls.clear()
        elsewhere = Map(crs=dataset.epsg)
        draw_field(elsewhere, dataset, _reloaded(layer))
        elsewhere.close()
        drawn = render_calls[-1]["levels"]
        assert drawn == LEVELS, f"a reloaded layer drew with {drawn!r}"
        assert isinstance(drawn, list), (
            f"and as a {type(drawn).__name__}; the recorded tuple was not thawed back"
        )
