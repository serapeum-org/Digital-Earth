"""Tests for digitalearth.scene.Scene — the shared-axes glyph host."""
import matplotlib

matplotlib.use("Agg")

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
