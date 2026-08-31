"""Tests for Scene.stamp — the figure-level watermark / brand mark.

``stamp_mark`` itself is tested upstream in cleopatra; these cover that Digital-Earth wires it to the scene's
own figure, that it is inherited by every scene type, and that its failure modes surface unchanged.
"""
import numpy as np
import pytest

from digitalearth.scene import Map, Scene


@pytest.fixture
def mark() -> np.ndarray:
    """A small opaque white mark."""
    logo = np.zeros((40, 80, 4), dtype=np.uint8)
    logo[..., :3] = 255
    logo[..., 3] = 255
    return logo


def test_stamp_adds_an_axes_to_the_scene_figure(mark):
    scene = Scene(figsize=(8, 6))
    before = len(scene.fig.axes)
    scene.stamp(mark, frac=0.2, shadow=False)
    assert len(scene.fig.axes) == before + 1


def test_stamp_returns_the_mark_axes_positioned_in_the_corner(mark):
    """Default corner is lower-right at a 0.025 margin, so the bounds are predictable."""
    scene = Scene(figsize=(8, 6))
    mark_ax = scene.stamp(mark, frac=0.2, shadow=False)
    assert [round(float(v), 3) for v in mark_ax.get_position().bounds] == [0.775, 0.025, 0.2, 0.133]


def test_stamp_honours_the_corner(mark):
    scene = Scene(figsize=(8, 6))
    lower_left = scene.stamp(mark, frac=0.2, shadow=False, corner="lower left")
    assert lower_left.get_position().bounds[0] == pytest.approx(0.025)


def test_stamp_is_inherited_by_map(dataset, mark):
    """Map is a Scene, so it gets the brand mark without its own implementation."""
    m = Map(crs=dataset.epsg)
    m.imshow(dataset)
    before = len(m.fig.axes)
    m.stamp(mark, frac=0.1, shadow=False)
    assert len(m.fig.axes) == before + 1


def test_stamp_draws_above_the_data(dataset, mark):
    """A watermark that renders under the map is useless, so check the drawing order, not list order."""
    m = Map(crs=dataset.epsg)
    m.imshow(dataset)
    mark_ax = m.stamp(mark, frac=0.1, shadow=False)
    assert mark_ax.get_zorder() >= m.ax.get_zorder(), (
        f"the mark axes (zorder {mark_ax.get_zorder()}) must not sit under the data "
        f"(zorder {m.ax.get_zorder()})"
    )


def test_a_bad_corner_is_refused(mark):
    scene = Scene()
    with pytest.raises(ValueError):
        scene.stamp(mark, corner="middle")


def test_an_oversized_mark_is_refused(mark):
    """frac must stay in (0, 1] — a mark larger than the figure cannot be placed."""
    scene = Scene()
    with pytest.raises(ValueError):
        scene.stamp(mark, frac=1.5)


def test_a_missing_mark_file_is_refused():
    scene = Scene()
    with pytest.raises(FileNotFoundError):
        scene.stamp("no-such-logo.png")


def test_stamped_figure_saves(tmp_path, mark):
    """The documented bbox_inches=None save path must work end to end."""
    scene = Scene(figsize=(4, 3))
    scene.stamp(mark, frac=0.2, shadow=False)
    out = tmp_path / "stamped.png"
    scene.save(str(out), bbox_inches=None)
    assert out.exists() and out.stat().st_size > 0
