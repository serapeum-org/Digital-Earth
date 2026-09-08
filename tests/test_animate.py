"""Tests for Map.animate / Map.rotate — globe animations over a raster stack."""
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation, PillowWriter
from pyramids.dataset import Dataset, GeoReference

from digitalearth.scene import Map, projections
from digitalearth.scene.maps.raster import _stretch_to_unit, channel_limits
from digitalearth.sources import get_stack


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
    return Dataset.from_array(
        arr=z.astype("float32"),
        geo_ref=GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326),
    )


def _rgb_field(offset: float, exposure: float = 1.0) -> Dataset:
    """A small 3-band global field, shifted by ``offset`` and scaled by ``exposure``.

    Args:
        offset: Constant added to every band (distinguishes stack frames).
        exposure: Multiplier applied to every band, standing in for a brighter or darker scene.

    Returns:
        Dataset: a 3-band 60x120 global EPSG:4326 raster.
    """
    ny, nx = 60, 120
    lat = np.linspace(90, -90, ny)[:, None]
    base = (np.cos(np.deg2rad(lat)) * 30 + offset) * np.ones((ny, nx), "float32")
    bands = np.stack([base * exposure, base * 0.6 * exposure, base * 0.3 * exposure])
    return Dataset.from_array(
        arr=bands.astype("float32"),
        geo_ref=GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326),
    )


@pytest.fixture
def rgb_stack():
    """A 3-frame stack of small 3-band global fields.

    Returns:
        list[Dataset]: three distinct 3-band rasters.
    """
    return [_rgb_field(o) for o in (0.0, 15.0, 30.0)]


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

    def test_unknown_kind_raises_up_front(self, stack):
        """An invalid kind fails fast at the animate() call, not later during render (N1)."""
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="unknown animation kind"):
            m.animate(stack, kind="imshowx")

    def test_animation_kept_on_self(self, stack):
        """The returned FuncAnimation is also held on self._animation so it is not GC'd (L3)."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(stack, fps=2, vmin=-40, vmax=70)
        assert m._animation is anim, "animate must keep a strong reference to the animation"

    def test_resolve_clim_caps_scan(self, mocker):
        """_resolve_animation_clim scans at most _CLIM_SCAN_CAP frames of a large stack (L2)."""
        from digitalearth.scene.maps import animation as anim_mod

        spy = mocker.spy(anim_mod.AnimationMixin, "_stack_clim")
        m = Map(crs=4326)
        big = [_field(float(o)) for o in range(anim_mod._CLIM_SCAN_CAP * 3)]
        opts = {}
        m._resolve_animation_clim(big, opts)
        scanned = len(spy.call_args.args[0])
        assert scanned <= anim_mod._CLIM_SCAN_CAP, f"scanned {scanned} frames, cap is {anim_mod._CLIM_SCAN_CAP}"
        assert "vmin" in opts and "vmax" in opts, "clim should still be resolved from the sampled frames"

    def test_titles_length_mismatch_raises(self, stack):
        """A titles list of the wrong length raises ValueError."""
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="titles length"):
            m.animate(stack, titles=["only-one"])

    def test_colorbar_adds_single_static_axes(self, stack, tmp_path):
        """colorbar=True adds exactly one colorbar axes that persists (not re-added) across all frames."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(stack, fps=2, colorbar=True, cbar_label="T", cmap="RdYlBu_r", vmin=-40, vmax=70)
        assert len(m.fig.axes) == 2, "colorbar should add one axes (data + colorbar)"
        out = tmp_path / "cbar.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert len(m.fig.axes) == 2, "the colorbar must not be re-added per frame"
        assert out.stat().st_size > 0

    def test_shares_clim_across_frames(self):
        """Every frame uses one global colour scale — even with no colorbar and no vmin/vmax (the fix)."""
        def fld(lo, hi):
            ny, nx = 30, 60
            z = np.linspace(lo, hi, ny * nx).reshape(ny, nx).astype("float32")
            return Dataset.from_array(arr=z, geo_ref=GeoReference(geo=(-180.0, 6.0, 0.0, 90.0, 0.0, -6.0), epsg=4326))

        frames = [fld(0, 10), fld(0, 100), fld(0, 1000)]   # wildly different per-frame ranges
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(frames, fps=2, cmap="viridis")    # no colorbar, no vmin/vmax
        clims = []
        for i in range(len(frames)):
            anim._func(i)
            clims.append(tuple(round(c) for c in m.ax.images[0].get_clim()))
        assert len(set(clims)) == 1, f"frames must share one colour scale, got {clims}"
        assert clims[0] == (0, 1000), f"shared clim should span the whole stack, got {clims[0]}"

    def test_resolve_clim_fills_missing_bounds(self, stack):
        """_resolve_animation_clim computes a global clim only for the missing bound(s)."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        lo, hi = Map._stack_clim(stack)
        both = {}
        m._resolve_animation_clim(stack, both)
        assert (both["vmin"], both["vmax"]) == (lo, hi), "both bounds should be filled from the stack"
        one = {"vmin": -100.0}
        m._resolve_animation_clim(stack, one)
        assert one["vmin"] == -100.0 and one["vmax"] == hi, "only the missing bound should be filled"
        explicit = {"vmin": -5.0, "vmax": 5.0}
        m._resolve_animation_clim(stack, explicit)
        assert (explicit["vmin"], explicit["vmax"]) == (-5.0, 5.0), "explicit bounds must be left untouched"

    def test_colorbar_computes_clim_from_stack(self, stack):
        """colorbar=True draws one bar over the resolved clim and adds a single colorbar axes."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        opts = {"cmap": "viridis"}
        m._resolve_animation_clim(stack, opts)
        lo, hi = Map._stack_clim(stack)
        assert opts["vmin"] == lo and opts["vmax"] == hi, "resolved clim should be injected into opts"
        m._animation_colorbar(opts, "auto")
        assert len(m.fig.axes) == 2, "a colorbar axes should be present"

    def test_stack_clim_ignores_nodata(self):
        """_stack_clim excludes nodata cells when computing the range (from_array defaults to -9999)."""
        arr = np.array([[0.0, 5.0], [10.0, -9999.0]], dtype="float32")
        ds = Dataset.from_array(arr=arr, geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326))
        assert ds.no_data_value[0] == -9999.0
        lo, hi = Map._stack_clim([ds])
        assert (lo, hi) == (0.0, 10.0), f"nodata -9999 should be ignored, got ({lo}, {hi})"

    def test_stack_clim_no_nodata_uses_full_range(self):
        """When a dataset declares no nodata, every finite cell counts toward the range."""
        from types import SimpleNamespace

        ds = SimpleNamespace(read_array=lambda band=0: np.array([[1.0, 2.0], [3.0, 4.0]]),
                             no_data_value=[None])
        assert Map._stack_clim([ds]) == (1.0, 4.0)

    def test_stack_clim_all_nodata_falls_back(self):
        """An all-nodata stack has no finite cells, so _stack_clim returns the (0, 1) default."""
        from types import SimpleNamespace

        ds = SimpleNamespace(read_array=lambda band=0: np.array([[-9999.0, -9999.0]]),
                             no_data_value=[-9999.0])
        assert Map._stack_clim([ds]) == (0.0, 1.0)

    def test_colorbar_without_label(self):
        """A colorbar with no label still adds exactly one colorbar axes."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        m._animation_colorbar({"cmap": "viridis", "vmin": 0, "vmax": 60}, None)
        assert len(m.fig.axes) == 2, "colorbar axes should be added even without a label"

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

    def test_unknown_kind_raises_up_front(self):
        """An invalid kind fails fast at the rotate() call (N1)."""
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="unknown animation kind"):
            m.rotate(_field(0.0), kind="bogus")

    def test_colorbar_static(self, tmp_path):
        """rotate(colorbar=True) adds one persistent colorbar axes across the rotation frames."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.rotate(_field(5.0), n_frames=3, fps=4, colorbar=True, cbar_label="elev",
                        cmap="terrain", vmin=-40, vmax=70)
        assert len(m.fig.axes) == 2, "rotate colorbar should add one axes"
        out = tmp_path / "rotcbar.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert len(m.fig.axes) == 2 and out.stat().st_size > 0, "colorbar must stay single after rendering"

    def test_rotate_coastlines_best_effort(self, tmp_path, mocker):
        """rotate(coastlines=True) attempts coastlines each frame and still renders when they fail offline."""
        coast = mocker.patch.object(Map, "coastlines", side_effect=RuntimeError("offline"))
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.rotate(_field(5.0), n_frames=2, fps=4, coastlines=True, vmin=-40, vmax=70)
        out = tmp_path / "rotcoast.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert out.stat().st_size > 0, "rotation should still render when coastlines fail"
        assert coast.call_count >= 2, "coastlines should be attempted on each frame"


class TestDrawAnimationFrame:
    """Tests for the shared Map._draw_animation_frame helper (PB-4)."""

    def test_sets_title_and_draws_field(self):
        """_draw_animation_frame draws the field and sets the title when given.

        Test scenario:
            On a flat map (globe=False) the field is drawn as one layer and the title is applied;
            ocean is skipped because it is globe-only.
        """
        m = Map(crs=4326)
        m._draw_animation_frame(_field(0.0), "imshow", {}, ocean=True, coastlines=False, title="frame-0")
        assert m.ax.get_title() == "frame-0", f"title not set, got {m.ax.get_title()!r}"
        assert len(m.layers) == 1, f"expected one drawn field layer, got {len(m.layers)}"

    def test_no_title_leaves_title_empty(self):
        """_draw_animation_frame leaves the title untouched when none is given.

        Test scenario:
            Omitting title draws the field without setting any axes title.
        """
        m = Map(crs=4326)
        m._draw_animation_frame(_field(0.0), "imshow", {}, ocean=False, coastlines=False)
        assert m.ax.get_title() == "", f"title should be empty, got {m.ax.get_title()!r}"


class TestAnimateComposites:
    """Tests for animating RGB/HSV composites (issue #150)."""

    @pytest.mark.parametrize("kind", ["rgb_composite", "hsv_composite"])
    def test_composite_is_an_accepted_kind(self, rgb_stack, kind):
        """A composite kind animates instead of being rejected as unknown."""
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.animate(rgb_stack, kind=kind, fps=2)
        assert isinstance(anim, FuncAnimation), f"expected FuncAnimation, got {type(anim)}"
        assert len(list(anim.new_frame_seq())) == len(rgb_stack), "frame count must match the stack length"

    def test_composite_renders_a_gif(self, rgb_stack, tmp_path):
        """A composite animation renders every frame to a non-empty GIF."""
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.animate(rgb_stack, kind="rgb_composite", fps=2, titles=["a", "b", "c"])
        out = tmp_path / "rgb.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.exists() and out.stat().st_size > 0, "composite GIF should be non-empty"
        assert m.ax.get_title() == "c", "the last frame's title should remain on the axes"

    def test_colorbar_on_a_composite_is_refused(self, rgb_stack):
        """A composite has no single value scale, so colorbar=True is rejected rather than drawn."""
        m = Map(crs=4326, figsize=(4, 4))
        with pytest.raises(ValueError, match="colorbar=True is not meaningful"):
            m.animate(rgb_stack, kind="rgb_composite", colorbar=True)

    def test_stretch_is_frozen_across_the_stack(self):
        """One stretch spans the whole stack, so a real exposure change survives instead of being erased."""
        dark, bright = _rgb_field(0.0, exposure=0.5), _rgb_field(0.0, exposure=1.0)
        opts: dict = {}
        Map(crs=4326)._resolve_composite_stretch([dark, bright], opts)
        limits = opts["stretch_limits"]
        rendered = [_stretch_to_unit(get_stack(ds, (1, 2, 3)), limits).mean() for ds in (dark, bright)]
        assert rendered[1] - rendered[0] > 0.05, (
            f"the frozen stretch flattened a 2x exposure difference to {rendered}; frames would not differ"
        )
        per_frame = [_stretch_to_unit(get_stack(ds, (1, 2, 3))).mean() for ds in (dark, bright)]
        assert abs(per_frame[1] - per_frame[0]) < 0.01, (
            "per-frame stretch is expected to erase the difference — that is the flicker this guards against"
        )

    def test_explicit_stretch_limits_are_kept(self, rgb_stack):
        """Caller-supplied stretch_limits are honoured, not overwritten by the stack scan."""
        given = [(0.0, 10.0), (0.0, 20.0), (0.0, 30.0)]
        opts = {"stretch_limits": given}
        Map(crs=4326)._resolve_composite_stretch(rgb_stack, opts)
        assert opts["stretch_limits"] is given, "an explicit stretch must not be recomputed"

    def test_composite_scan_is_capped(self, mocker):
        """The stack scan samples at most _CLIM_SCAN_CAP frames, however long the stack."""
        from digitalearth.scene.maps import animation as anim_mod

        spy = mocker.spy(anim_mod, "get_stack")
        big = [_rgb_field(float(o)) for o in range(anim_mod._CLIM_SCAN_CAP * 3)]
        opts: dict = {}
        Map(crs=4326)._resolve_composite_stretch(big, opts)
        assert spy.call_count <= anim_mod._CLIM_SCAN_CAP, (
            f"scanned {spy.call_count} frames, cap is {anim_mod._CLIM_SCAN_CAP}"
        )
        assert len(opts["stretch_limits"]) == 3, "one bound pair per band"


class TestChannelLimits:
    """Tests for the stretch-bounds helpers the composite animation path depends on."""

    def test_limits_are_reused_verbatim(self):
        """Passing limits bypasses the per-array percentile derivation."""
        stack = np.dstack([np.arange(100.0).reshape(10, 10) for _ in range(3)])
        out = _stretch_to_unit(stack, [(0.0, 99.0)] * 3)
        assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)

    def test_channel_limits_one_pair_per_band(self):
        """channel_limits returns a (low, high) pair for each channel, low < high."""
        stack = np.dstack([np.arange(100.0).reshape(10, 10) * s for s in (1.0, 2.0, 3.0)])
        limits = channel_limits(stack)
        assert len(limits) == 3
        assert all(high > low for low, high in limits)

    def test_flat_channel_does_not_divide_by_zero(self):
        """A constant channel gets a nudged upper bound so the stretch stays finite."""
        stack = np.dstack([np.full((4, 4), 7.0), np.zeros((4, 4)), np.ones((4, 4))])
        assert np.isfinite(_stretch_to_unit(stack, channel_limits(stack))).all()

    def test_wrong_number_of_limits_is_rejected(self):
        """Limits that do not match the channel count raise rather than silently mis-stretching."""
        stack = np.dstack([np.zeros((4, 4))] * 3)
        with pytest.raises(ValueError, match="channel bounds"):
            _stretch_to_unit(stack, [(0.0, 1.0)])
