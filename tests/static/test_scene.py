"""Tests for digitalearth.static.Scene — the shared-axes glyph host."""

import numpy as np
import pytest
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

from digitalearth.static import Map, Scene


def _render(scene, arr, kind="imshow"):
    """Render an array on the scene's shared axes and register it as a layer."""
    glyph = ArrayGlyph(arr, ax=scene.ax, fig=scene.fig)
    glyph.plot(kind=kind, add_colorbar=False)
    return scene._add_layer(glyph, glyph.im)


@pytest.fixture
def grid_map():
    """A map carrying one field whose colour a key can explain, drawn from a bare array.

    A colour key hangs on the layer's colour encoding since order 24, and only a *builder* publishes one — an
    artist registered through `add_layer` is described as `custom:matplotlib` and says nothing about what
    colours it. So the key tests run on a `Map`, which is a `Scene` with builders, rather than on the bare
    `Scene` a hand-rendered glyph gives.

    Yields:
        The map, with the layer `grid` on it. Closed afterwards.
    """
    canvas = Map(globe=False)
    canvas.field(np.array([[0.0, 1.0], [2.0, 3.0]]), name="grid")
    yield canvas
    canvas.close()


def test_scene_creates_its_own_axes():
    """A Scene with no axes creates exactly one figure/axes."""
    scene = Scene()
    assert scene.fig is not None
    assert scene.ax is not None
    assert len(scene.fig.axes) == 1


def test_two_layers_share_one_axes():
    """Two glyphs render onto a single shared axes and both register as layers."""
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    _render(scene, np.random.rand(8, 8))
    # still a single axes before any colorbar is drawn
    assert len(scene.fig.axes) == 1
    assert len(scene.layers) == 2


def test_colorbar_adds_an_axes(grid_map):
    """Drawing a layer's colorbar adds a second axes to the figure.

    Args:
        grid_map: A map carrying one value-coloured field, `grid`.

    Test scenario:
        The bar is figure furniture, so it takes an axes of its own beside the data axes.
    """
    grid_map.colorbar()
    assert len(grid_map.fig.axes) == 2


def test_colorbar_with_label(grid_map):
    """colorbar(label=...) sets the colorbar's label text.

    Args:
        grid_map: A map carrying one value-coloured field, `grid`.

    Test scenario:
        The label reaches the drawn bar, which is read off the layer it explains — `colorbar` returns the
        scene now, as the Core declares.
    """
    grid_map.colorbar(label="discharge")
    bar = grid_map._renderer.drawn["grid"].guides[0]
    assert bar.ax.get_ylabel() == "discharge" or bar.ax.get_xlabel() == "discharge"


def test_colorbar_chains(grid_map):
    """`colorbar` hands the scene back, so a keyed map reads as one expression (#261, order 24).

    Args:
        grid_map: A map carrying one value-coloured field, `grid`.

    Test scenario:
        The Core declares `returns="self"` and the web and interactive tiers already answer that way; this
        tier returned the `Colorbar`, so `m.field(ds).colorbar().legend(...)` chained on two tiers and raised
        on this one. Asserted as identity, since any object at all would satisfy "not None" and still break
        the next call.
    """
    assert grid_map.colorbar() is grid_map


def test_colorbars_one_per_keyed_layer(grid_map):
    """colorbars() draws one colorbar for every layer that is coloured by a value.

    Args:
        grid_map: A map carrying one value-coloured field, `grid`.

    Test scenario:
        A second field is added, and a text label after it: the two fields are keyed and the label — which is
        coloured by nothing — is skipped rather than refused.
    """
    grid_map.field(np.array([[4.0, 5.0], [6.0, 7.0]]), name="second")
    grid_map.add_layer(grid_map.ax.imshow(np.array([[1.0, 2.0]])), name="mine")
    assert grid_map.colorbars() is grid_map
    assert len(grid_map.fig.axes) == 3, "one data axes and one bar per keyed layer"
    assert grid_map._renderer.drawn["mine"].guides == (), (
        "a caller's own mappable was keyed, although the tier cannot name what varies its colour"
    )


def test_colorbars_draws_nothing_without_a_keyed_layer():
    """colorbars() on a figure nothing colours by value draws nothing, and does not raise.

    Test scenario:
        The plural is the "key everything" call, so a figure with nothing to key is silence rather than an
        error — the singular is what refuses.
    """
    scene = Scene()
    assert scene.colorbars() is scene
    assert len(scene.fig.axes) == 1


def test_colorbar_without_a_keyed_layer_raises():
    """Asking for a colorbar where no layer is coloured by a value is refused, and says so.

    Test scenario:
        The refusal names what to draw instead. A bar over a colour nothing varies has no values to label,
        which is the web tier's answer for the same case.
    """
    scene = Scene()
    with pytest.raises(
        ValueError, match="no layer on this figure is coloured by a value"
    ):
        scene.colorbar()


def test_a_custom_artist_has_no_colour_key():
    """An artist the caller built themselves publishes no colour encoding, so a key on it is refused.

    Test scenario:
        `_add_layer` records the artist and nothing about what colours it — this tier cannot name the field a
        caller's own mappable varies with. That is the web tier's answer for its own custom layers, and it is
        what makes `layer_id=None` mean "the most recent layer that carries a colour encoding" rather than
        "the last artist registered".
    """
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    assert scene.layer_ids == ["custom-1"], (
        "the artist was not recorded as a custom layer"
    )
    with pytest.raises(
        ValueError, match="no layer on this figure is coloured by a value"
    ):
        scene.colorbar()


def test_legend_is_attached(grid_map):
    """A swatch legend derived from the layer's own scale is attached to the axes.

    Args:
        grid_map: A map carrying one value-coloured field, `grid`.

    Test scenario:
        `legend` took the colours and labels themselves until order 24; the rows now come from the scale the
        layer was drawn through, so a swatch equals the colour on the map by construction.
    """
    assert grid_map.legend(title="depth") is grid_map
    legend = grid_map._renderer.drawn["grid"].guides[0]
    assert grid_map.ax.get_legend() is legend
    assert legend.get_title().get_text() == "depth"


def test_legend_labels_replace_the_derived_rows(grid_map):
    """legend(labels=...) renames the rows without touching the colours they stand for.

    Args:
        grid_map: A map carrying one value-coloured field, `grid`.

    Test scenario:
        The override the Core declares. A count that does not match the rows is refused rather than drawn
        half-labelled.
    """
    grid_map.legend(labels=["a", "b", "c", "d", "e"])
    rows = [text.get_text() for text in grid_map.ax.get_legend().get_texts()]
    assert rows == ["a", "b", "c", "d", "e"]
    with pytest.raises(ValueError, match="labels= needs 5 labels"):
        grid_map.legend(labels=["only one"])


def test_save_writes_a_file(tmp_path):
    """Scene.save writes a PNG under Agg."""
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    out = tmp_path / "scene.png"
    scene.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_set_title():
    """Scene.set_title sets the axes title."""
    scene = Scene()
    scene.set_title("my map")
    assert scene.ax.get_title() == "my map"


def test_set_title_chains():
    """`set_title` hands the scene back, so figure decoration reads as one expression (order 27a, #265).

    Test scenario:
        The Core names this method and the web tier already returns `Self` from it; this tier returned
        `None`, so `Map(crs=3857).set_title("x").coastlines()` raised `AttributeError` on the default tier
        while the same line worked on web. Asserted as identity: a method returning any object at all would
        satisfy "not None" and could still break the next call in the chain.
    """
    scene = Scene()
    assert scene.set_title("my map") is scene


def test_set_title_still_draws_when_its_result_is_chained():
    """Returning `self` must not come at the cost of the title actually being set.

    Test scenario:
        The failure a return-type change invites: a body rewritten to `return self` that drops the call it
        was doing the work for. Read off the axes, which is where the title has to end up.
    """
    scene = Scene()
    assert scene.set_title("chained", loc="left").ax.get_title(loc="left") == "chained"


def test_show_invokes_pyplot(mocker):
    """Scene.show delegates to matplotlib.pyplot.show without raising under Agg."""
    spy = mocker.patch("matplotlib.pyplot.show")
    Scene().show()
    spy.assert_called_once()


def test_accepts_external_axes():
    """A Scene can wrap a caller-supplied fig/ax instead of creating one."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scene = Scene(ax=ax, fig=fig)
    assert scene.ax is ax
    assert scene.fig is fig


class _FakeGlyph:
    """Minimal cleopatra-glyph stand-in for _render_glyph tests.

    Records the args its ``plot`` received, exposes a mappable on ``im``, and returns the cleopatra
    ``(fig, ax, artist)`` tuple so both artist-resolution conventions can be exercised.
    """

    def __init__(self, im="IM", tuple_artist="ARTIST"):
        self.im = im
        self._tuple_artist = tuple_artist
        self.plot_args = None
        self.plot_kwargs = None

    def plot(self, *args, **kwargs):
        """Record the call and return a cleopatra-style (fig, ax, artist) triple."""
        self.plot_args = args
        self.plot_kwargs = kwargs
        return ("FIG", "AX", self._tuple_artist)


class TestRenderGlyph:
    """Tests for Scene._render_glyph (PA-3)."""

    def test_im_convention_registers_glyph_im(self):
        """artist='im' (default) registers glyph.im and reports it as what was drawn.

        Test scenario:
            The ArrayGlyph/MeshGlyph convention exposes the mappable on .im; that object is the layer.
            Since #303 the recipe answers in a ``DrawnLayer``, because its callers are drawers and what
            they hand back is what the renderer records.
        """
        scene = Scene()
        glyph = _FakeGlyph(im="THE_IMAGE")
        drawn = scene._render_glyph(glyph)
        assert drawn.artist == "THE_IMAGE", f"expected glyph.im, got {drawn.artist}"
        assert scene.layers[-1] == (glyph, "THE_IMAGE"), (
            f"layer not registered correctly: {scene.layers[-1]}"
        )

    def test_plot_convention_registers_third_element(self):
        """artist='plot' registers and reports the third element of plot()'s return.

        Test scenario:
            Scatter/Polygon/Vector/KDE/Flow glyphs return (fig, ax, artist); the artist is the layer.
        """
        scene = Scene()
        glyph = _FakeGlyph(tuple_artist="THE_COLLECTION")
        drawn = scene._render_glyph(glyph, artist="plot")
        assert drawn.artist == "THE_COLLECTION", (
            f"expected plot()[2], got {drawn.artist}"
        )
        assert scene.layers[-1] == (glyph, "THE_COLLECTION"), (
            f"layer wrong: {scene.layers[-1]}"
        )

    def test_forwards_positional_and_keyword_args(self):
        """_render_glyph forwards *plot_args and **plot_kwargs to glyph.plot.

        Test scenario:
            A positional data array and keyword options must reach glyph.plot unchanged, while the
            artist selector itself is consumed and not forwarded.
        """
        scene = Scene()
        glyph = _FakeGlyph()
        scene._render_glyph(
            glyph, [1, 2, 3], artist="im", kind="contourf", outline_only=True
        )
        assert glyph.plot_args == ([1, 2, 3],), (
            f"positional args not forwarded: {glyph.plot_args}"
        )
        assert glyph.plot_kwargs == {"kind": "contourf", "outline_only": True}, (
            f"kwargs not forwarded cleanly: {glyph.plot_kwargs}"
        )

    def test_returned_mappable_is_added_once(self):
        """Each _render_glyph call appends exactly one layer.

        Test scenario:
            Two renders produce two distinct registered layers in call order.
        """
        scene = Scene()
        scene._render_glyph(_FakeGlyph(im="A"))
        scene._render_glyph(_FakeGlyph(im="B"))
        ims = [mappable for _, mappable in scene.layers]
        assert ims == ["A", "B"], f"layers not registered in order: {ims}"


class TestContextManager:
    """Tests for Scene.__enter__/__exit__ (PC-2)."""

    def test_enter_returns_self(self):
        """``with Scene() as s`` binds the scene itself.

        Test scenario:
            __enter__ returns the same instance so it can be used as the ``as`` target.
        """
        scene = Scene()
        with scene as bound:
            assert bound is scene, "__enter__ should return the scene"

    def test_exit_closes_figure(self):
        """Leaving the context closes the scene's figure.

        Test scenario:
            After the with-block the figure number is no longer registered with pyplot.
        """
        import matplotlib.pyplot as plt

        with Scene() as scene:
            num = scene.fig.number
            assert plt.fignum_exists(num), "figure should be open inside the context"
        assert not plt.fignum_exists(num), "figure should be closed on exit"

    def test_exit_does_not_suppress_exceptions(self):
        """Exceptions raised in the body propagate (the figure is still closed).

        Test scenario:
            __exit__ returns False, so a body error is re-raised; the figure is closed regardless.
        """
        import matplotlib.pyplot as plt

        scene = Scene()
        num = scene.fig.number
        with pytest.raises(ValueError, match="boom"):
            with scene:
                raise ValueError("boom")
        assert not plt.fignum_exists(num), (
            "figure should be closed even when the body raised"
        )


def _widen(ax):
    """Blow the view out to the whole world from inside a block, the way a global backdrop does.

    **Setting the limits, not plotting into them.** These tests used to draw far-away points with
    ``ax.plot`` and assert the view had not moved — but ``set_xlim`` turns that axis' autoscaling off, so
    the plot could not have moved it in the first place, and all three passed with ``_preserve_view``
    gutted to a bare ``yield`` (measured: the class stays green under that mutation). What the context
    manager is actually there for is a decoration that *sets* the limits itself — ``add_tiles`` and
    ``add_features`` both do — which is what this does.

    Args:
        ax: The axes to widen.
    """
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)


def _widen_then_fail(scene):
    """Widen the view inside a preserved block and then raise, for the early-exit path.

    A helper rather than the body of the ``pytest.raises`` block: that block may hold one call that can
    raise (``tests/base/test_refusal_blocks.py``), and the block needs three.

    Args:
        scene: The scene whose view is preserved.

    Raises:
        ValueError: always, from inside the ``with`` block.
    """
    with scene._preserve_view():
        _widen(scene.ax)
        raise ValueError("boom")


class TestPreserveView:
    """Tests for Scene._preserve_view (PA-2)."""

    def test_restores_limits_when_layer_present(self):
        """A registered layer counts as data, so the block's autoscaling is undone.

        Test scenario:
            With a layer drawn and explicit limits set, a block that widens the view to the whole world
            must not leave it there — the pre-block limits are restored on exit.
        """
        scene = Scene()
        _render(scene, np.random.rand(8, 8))
        scene.ax.set_xlim(0, 10)
        scene.ax.set_ylim(0, 5)
        with scene._preserve_view():
            _widen(scene.ax)
        assert scene.ax.get_xlim() == pytest.approx((0.0, 10.0)), (
            f"xlim moved: {scene.ax.get_xlim()}"
        )
        assert scene.ax.get_ylim() == pytest.approx((0.0, 5.0)), (
            f"ylim moved: {scene.ax.get_ylim()}"
        )

    def test_restores_limits_when_image_present(self):
        """An axes image (no registered layer) still counts as data.

        Test scenario:
            ``ax.images`` being non-empty is enough for _preserve_view to restore the view.
        """
        scene = Scene()
        scene.ax.imshow(np.random.rand(4, 4))
        scene.ax.set_xlim(0, 3)
        scene.ax.set_ylim(0, 3)
        with scene._preserve_view():
            _widen(scene.ax)
        assert scene.ax.get_xlim() == pytest.approx((0.0, 3.0)), (
            f"xlim moved: {scene.ax.get_xlim()}"
        )

    def test_restores_limits_when_collection_present(self):
        """An axes collection (no registered layer) counts as data.

        Test scenario:
            ``ax.collections`` being non-empty (here a scatter) triggers limit restoration.
        """
        scene = Scene()
        scene.ax.scatter([1, 2], [1, 2])
        scene.ax.set_xlim(0, 3)
        scene.ax.set_ylim(0, 3)
        with scene._preserve_view():
            _widen(scene.ax)
        assert scene.ax.get_xlim() == pytest.approx((0.0, 3.0)), (
            f"xlim moved: {scene.ax.get_xlim()}"
        )

    def test_restores_limits_when_the_block_raises(self):
        """A block that fails part-way still gives the view back (SonarCloud S9152).

        Test scenario:
            The widening happens, then the block raises. The exception reaches the caller unchanged, and
            the axes is the regional one it was before — not the world the failed decoration left behind,
            which is what a ``yield`` outside ``try``/``finally`` skipped past.
        """
        scene = Scene()
        scene.ax.imshow(np.random.rand(4, 4))
        scene.ax.set_xlim(0, 3)
        scene.ax.set_ylim(0, 3)
        with pytest.raises(ValueError, match="boom"):
            _widen_then_fail(scene)
        assert scene.ax.get_xlim() == pytest.approx((0.0, 3.0)), (
            f"xlim left where the failed block put it: {scene.ax.get_xlim()}"
        )
        assert scene.ax.get_ylim() == pytest.approx((0.0, 3.0)), (
            f"ylim left where the failed block put it: {scene.ax.get_ylim()}"
        )

    def test_empty_axes_keeps_new_extent(self):
        """On an empty axes the block is free to set the initial extent.

        Test scenario:
            With no data present, limits set inside the block survive (no restoration).
        """
        scene = Scene()
        with scene._preserve_view():
            scene.ax.set_xlim(50, 60)
            scene.ax.set_ylim(70, 80)
        assert scene.ax.get_xlim() == pytest.approx((50.0, 60.0)), (
            f"xlim not kept: {scene.ax.get_xlim()}"
        )
        assert scene.ax.get_ylim() == pytest.approx((70.0, 80.0)), (
            f"ylim not kept: {scene.ax.get_ylim()}"
        )


class TestUnregisteringALayerThatRegisteredNoMappable:
    """The colorbar registry is matched by identity, so a pair of `None`s must match nothing."""

    def test_removing_it_leaves_the_earlier_registration_alone(self):
        """A layer that registered nothing recognisable must not take another layer's entry with it.

        Test scenario:
            `Scene.layers` holds only the layers that registered a mappable, so a layer's index there is
            not its index in the description and the two are matched by identity instead. A layer drawn
            with neither a glyph nor a mappable — a caller's artist that is not one, a glyph whose render
            produced no `im` — then matches ``(None, None)`` wherever that pair happens to sit, and
            removing it unregisters whichever layer was filed first. keying by position — which `colorbar` did until order 24 — would name the
            colorbar to the wrong layer from then on.
        """
        scene = Scene()
        try:
            scene._add_layer(None, None, "the first caller artist")
            scene._add_layer(None, None, "the second caller artist")
            scene._renderer.remove("custom-2")
            assert scene._layer_labels[0] == "the first caller artist", (
                f"removing one unregistered layer dropped another: {scene._layer_labels}"
            )
        finally:
            scene.close()
