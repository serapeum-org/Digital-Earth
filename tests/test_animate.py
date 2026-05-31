"""Tests for Map.animate / Map.rotate — globe animations over a raster stack."""
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation, PillowWriter
from pyramids.dataset import Dataset

from digitalearth.scene import Map, projections


def _field(offset: float) -> Dataset:
    """A small global lon/lat field, shifted by ``offset`` so frames differ.

    Args:
        offset: Constant added to the field (distinguishes stack frames).

    Returns:
        Dataset: a 60x120 global EPSG:4326 raster.
    """
    ny, nx = 60, 120
    lat = np.linspace(90, -90, ny)[:, None]
    z = (np.cos(np.deg2rad(lat)) * 30 + offset) * np.ones((ny, nx), "float32")
    return Dataset.create_from_array(arr=z.astype("float32"), geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0),
                                     epsg=4326)


@pytest.fixture
def stack():
    """A 3-frame stack of small global fields.

    Returns:
        list[Dataset]: three distinct global rasters.
    """
    return [_field(o) for o in (0.0, 15.0, 30.0)]


class TestAnimate:
    """Tests for Map.animate."""

    def test_returns_funcanimation_with_frame_count(self, stack):
        """animate returns a FuncAnimation whose frame count matches the stack length (lazy, no render)."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(stack, fps=2, vmin=-40, vmax=70)
        assert isinstance(anim, FuncAnimation), f"expected FuncAnimation, got {type(anim)}"
        assert len(list(anim.new_frame_seq())) == len(stack), "frame count must match the stack length"

    def test_renders_gif_with_titles_and_ocean(self, stack, tmp_path):
        """Rendering writes a non-empty GIF; the last frame shows its title and the ocean disc.

        Test scenario:
            A globe animation with per-frame titles and the ocean disc renders all frames without error;
            after rendering, the axes hold the final frame's title and at least the ocean PolyCollection.
        """
        m = Map(crs=projections.orthographic(10, 15), globe=True, figsize=(4, 4))
        titles = ["Jan", "Feb", "Mar"]
        anim = m.animate(stack, fps=2, titles=titles, ocean=True, cmap="RdYlBu_r", vmin=-40, vmax=70)
        out = tmp_path / "anim.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.exists() and out.stat().st_size > 0, "animation GIF should be non-empty"
        assert m.ax.get_title() == titles[-1], f"last title not applied: {m.ax.get_title()!r}"
        assert m.ax.collections, "ocean disc PolyCollection should be present on the last frame"
        assert m.ax.images, "the raster frame should be drawn"

    def test_flat_map_animation(self, stack, tmp_path):
        """animate works on a non-globe map (no frame applied) and leaves globe False."""
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.animate(stack, fps=2, vmin=-40, vmax=70)
        out = tmp_path / "flat.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, "flat-map animation GIF should be non-empty"
        assert m.globe is False, "a flat map must not be turned into a globe by animate"
        assert not m.ax.patches, "a flat map adds no projection-boundary patch"

    def test_empty_stack_raises(self):
        """An empty stack raises a clear ValueError."""
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="empty stack"):
            m.animate([])

    def test_titles_length_mismatch_raises(self, stack):
        """A titles list of the wrong length raises ValueError."""
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="titles length"):
            m.animate(stack, titles=["only-one"])

    def test_coastlines_best_effort_swallows_failure(self, stack, tmp_path, mocker):
        """coastlines=True still renders when the (network) coastline fetch raises — the error is swallowed."""
        mocker.patch.object(Map, "coastlines", side_effect=RuntimeError("offline"))
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(stack, fps=2, coastlines=True, vmin=-40, vmax=70)
        out = tmp_path / "coast.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, "animation should still render when coastlines fail"
        assert Map.coastlines.called, "coastlines should have been attempted"


class TestRotate:
    """Tests for Map.rotate."""

    def test_frame_count_and_forces_globe(self):
        """rotate returns the requested frame count and forces the map into globe mode."""
        m = Map(crs=projections.orthographic(0, 15), figsize=(4, 4))  # starts non-globe
        anim = m.rotate(_field(5.0), n_frames=6, fps=4, vmin=-40, vmax=70)
        assert isinstance(anim, FuncAnimation), f"expected FuncAnimation, got {type(anim)}"
        assert len(list(anim.new_frame_seq())) == 6, "rotate frame count must equal n_frames"
        assert m.globe is True, "rotate must force globe mode"

    def test_renders_gif_and_sweeps_longitude(self, tmp_path):
        """Rendering writes a non-empty GIF and sweeps the centre longitude to the last frame's value.

        Test scenario:
            A 4-frame rotation from lon0=-180 steps the orthographic centre to -180, -90, 0, 90; after
            rendering, the display CRS holds the final centre longitude (+lon_0=90).
        """
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.rotate(_field(5.0), n_frames=4, fps=4, lon0=-180.0, ocean=True, cmap="terrain",
                        vmin=-40, vmax=70)
        out = tmp_path / "rot.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert out.stat().st_size > 0, "rotation GIF should be non-empty"
        assert "+lon_0=90" in m.crs, f"final centre longitude not swept to 90: {m.crs!r}"
        assert m.ax.images, "the rotated field should be drawn"

    def test_invalid_n_frames_raises(self):
        """rotate with fewer than one frame raises ValueError."""
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="n_frames"):
            m.rotate(_field(0.0), n_frames=0)

    def test_rotate_coastlines_best_effort(self, tmp_path, mocker):
        """rotate(coastlines=True) attempts coastlines each frame and still renders when they fail offline."""
        coast = mocker.patch.object(Map, "coastlines", side_effect=RuntimeError("offline"))
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.rotate(_field(5.0), n_frames=2, fps=4, coastlines=True, vmin=-40, vmax=70)
        out = tmp_path / "rotcoast.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert out.stat().st_size > 0, "rotation should still render when coastlines fail"
        assert coast.call_count >= 2, "coastlines should be attempted on each frame"
