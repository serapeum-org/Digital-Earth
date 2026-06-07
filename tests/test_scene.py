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
