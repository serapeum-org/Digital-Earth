"""Tests for digitalearth.scene.Scene — the shared-axes glyph host."""

import numpy as np
import pytest
from cleopatra.array_glyph import ArrayGlyph

from digitalearth.scene import Scene


def _render(scene, arr, kind="imshow"):
    """Render an array on the scene's shared axes and register it as a layer."""
    glyph = ArrayGlyph(arr, ax=scene.ax, fig=scene.fig)
    glyph.plot(kind=kind, add_colorbar=False)
    return scene._add_layer(glyph, glyph.im)


def test_scene_creates_its_own_axes():
    """A Scene with no axes creates exactly one figure/axes."""
    scene = Scene()
    assert scene.fig is not None and scene.ax is not None
    assert len(scene.fig.axes) == 1


def test_two_layers_share_one_axes():
    """Two glyphs render onto a single shared axes and both register as layers."""
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    _render(scene, np.random.rand(8, 8))
    # still a single axes before any colorbar is drawn
    assert len(scene.fig.axes) == 1
    assert len(scene.layers) == 2


def test_colorbar_adds_an_axes():
    """Drawing the aggregated colorbar adds a second axes to the figure."""
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    scene.colorbar()
    assert len(scene.fig.axes) == 2


def test_colorbar_with_label():
    """colorbar(label=...) sets the colorbar's label text."""
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    cbar = scene.colorbar(label="discharge")
    assert cbar.ax.get_ylabel() == "discharge" or cbar.ax.get_xlabel() == "discharge"


def test_colorbars_one_per_layer():
    """colorbars() draws one colorbar for every registered layer."""
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    _render(scene, np.random.rand(8, 8))
    cbars = scene.colorbars()
    assert len(cbars) == 2


def test_colorbars_empty_without_layers():
    """colorbars() returns an empty list when there are no layers."""
    assert Scene().colorbars() == []


def test_colorbar_without_layers_raises():
    """Asking for a colorbar with no layers is an error."""
    scene = Scene()
    with pytest.raises(ValueError, match="no layers"):
        scene.colorbar()


def test_legend_is_attached():
    """A categorical legend is attached to the axes."""
    scene = Scene()
    _render(scene, np.random.rand(8, 8))
    scene.legend(["red", "blue"], ["a", "b"])
    assert scene.ax.get_legend() is not None


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
    assert scene.ax is ax and scene.fig is fig


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
        """artist='im' (default) registers and returns glyph.im.

        Test scenario:
            The ArrayGlyph/MeshGlyph convention exposes the mappable on .im; that object is the layer.
        """
        scene = Scene()
        glyph = _FakeGlyph(im="THE_IMAGE")
        out = scene._render_glyph(glyph)
        assert out == "THE_IMAGE", f"expected glyph.im returned, got {out}"
        assert scene.layers[-1] == (glyph, "THE_IMAGE"), f"layer not registered correctly: {scene.layers[-1]}"

    def test_plot_convention_registers_third_element(self):
        """artist='plot' registers and returns the third element of plot()'s return.

        Test scenario:
            Scatter/Polygon/Vector/KDE/Flow glyphs return (fig, ax, artist); the artist is the layer.
        """
        scene = Scene()
        glyph = _FakeGlyph(tuple_artist="THE_COLLECTION")
        out = scene._render_glyph(glyph, artist="plot")
        assert out == "THE_COLLECTION", f"expected plot()[2] returned, got {out}"
        assert scene.layers[-1] == (glyph, "THE_COLLECTION"), f"layer wrong: {scene.layers[-1]}"

    def test_forwards_positional_and_keyword_args(self):
        """_render_glyph forwards *plot_args and **plot_kwargs to glyph.plot.

        Test scenario:
            A positional data array and keyword options must reach glyph.plot unchanged, while the
            artist selector itself is consumed and not forwarded.
        """
        scene = Scene()
        glyph = _FakeGlyph()
        scene._render_glyph(glyph, [1, 2, 3], artist="im", kind="contourf", outline_only=True)
        assert glyph.plot_args == ([1, 2, 3],), f"positional args not forwarded: {glyph.plot_args}"
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
        assert not plt.fignum_exists(num), "figure should be closed even when the body raised"


class TestPreserveView:
    """Tests for Scene._preserve_view (PA-2)."""

    def test_restores_limits_when_layer_present(self):
        """A registered layer counts as data, so the block's autoscaling is undone.

        Test scenario:
            With a layer drawn and explicit limits set, plotting far-away points inside the block must
            not move the view — the pre-block limits are restored on exit.
        """
        scene = Scene()
        _render(scene, np.random.rand(8, 8))
        scene.ax.set_xlim(0, 10)
        scene.ax.set_ylim(0, 5)
        with scene._preserve_view():
            scene.ax.plot([100, 200], [100, 200])
        assert scene.ax.get_xlim() == pytest.approx((0.0, 10.0)), f"xlim moved: {scene.ax.get_xlim()}"
        assert scene.ax.get_ylim() == pytest.approx((0.0, 5.0)), f"ylim moved: {scene.ax.get_ylim()}"

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
            scene.ax.plot([100, 200], [100, 200])
        assert scene.ax.get_xlim() == pytest.approx((0.0, 3.0)), f"xlim moved: {scene.ax.get_xlim()}"

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
            scene.ax.plot([100, 200], [100, 200])
        assert scene.ax.get_xlim() == pytest.approx((0.0, 3.0)), f"xlim moved: {scene.ax.get_xlim()}"

    def test_empty_axes_keeps_new_extent(self):
        """On an empty axes the block is free to set the initial extent.

        Test scenario:
            With no data present, limits set inside the block survive (no restoration).
        """
        scene = Scene()
        with scene._preserve_view():
            scene.ax.set_xlim(50, 60)
            scene.ax.set_ylim(70, 80)
        assert scene.ax.get_xlim() == pytest.approx((50.0, 60.0)), f"xlim not kept: {scene.ax.get_xlim()}"
        assert scene.ax.get_ylim() == pytest.approx((70.0, 80.0)), f"ylim not kept: {scene.ax.get_ylim()}"
