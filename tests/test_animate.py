"""Tests for Map.animate / Map.rotate — globe animations over a raster stack and over composites."""

import numpy as np
import pytest
from matplotlib.animation import FuncAnimation, PillowWriter
from pyramids.dataset import Dataset, GeoReference

from digitalearth.base.sources import get_stack
from digitalearth.base.stretch import channel_limits, stretch_to_unit
from digitalearth.static import Map, projections


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


@pytest.fixture
def stack():
    """A 3-frame stack of small global fields.

    Returns:
        list[Dataset]: three distinct global rasters.
    """
    return [_field(o) for o in (0.0, 15.0, 30.0)]


def _rgb_field(
    shift: float = 0.0, exposure: float = 1.0, ny: int = 60, nx: int = 120
) -> Dataset:
    """A small 3-band global raster whose scene shifts and whose overall brightness scales.

    Args:
        shift: Phase shift of the pattern, so consecutive frames differ.
        exposure: Multiplies every channel — a whole-scene brightness change, which is exactly what a
            per-frame contrast stretch erases.
        ny: Row count.
        nx: Column count.

    Returns:
        Dataset: a 3-band global EPSG:4326 raster.
    """
    yy, xx = np.mgrid[0:ny, 0:nx]
    base = 0.4 + 0.3 * np.sin((xx + shift) / 13.0) + 0.2 * np.cos(yy / 9.0)
    bands = np.stack(
        [base * 3000.0 * exposure, base * 2600.0 * exposure, base * 2200.0 * exposure]
    )
    return Dataset.from_array(
        arr=bands.astype("float32"),
        geo_ref=GeoReference(
            geo=(-180.0, 360.0 / nx, 0.0, 90.0, 0.0, -180.0 / ny), epsg=4326
        ),
    )


def _rgb_field_with_dead_channel(
    shift: float = 0.0, ny: int = 60, nx: int = 120
) -> Dataset:
    """A 3-band raster whose **second** channel is entirely nodata.

    Args:
        shift: Phase shift of the pattern in the two live channels.
        ny: Row count.
        nx: Column count.

    Returns:
        Dataset: a 3-band raster with channel 2 fully nodata (-9999).
    """
    yy, xx = np.mgrid[0:ny, 0:nx]
    base = 0.4 + 0.3 * np.sin((xx + shift) / 13.0) + 0.2 * np.cos(yy / 9.0)
    bands = np.stack([base * 3000.0, np.full_like(base, -9999.0), base * 2200.0])
    return Dataset.from_array(
        arr=bands.astype("float32"),
        geo_ref=GeoReference(
            geo=(-180.0, 360.0 / nx, 0.0, 90.0, 0.0, -180.0 / ny), epsg=4326
        ),
        no_data_value=-9999.0,
    )


@pytest.fixture
def rgb_stack():
    """A 3-frame stack of 3-band rasters — a true-colour time-lapse.

    Returns:
        list[Dataset]: three distinct multiband global rasters.
    """
    return [_rgb_field(shift=s) for s in (0.0, 10.0, 20.0)]


class TestAnimateOffLimb:
    """A stack no frame of which is on the view still animates (issue #151, via the colour scan)."""

    @pytest.fixture
    def regional_stack(self):
        """Two frames of a small AOI that any far-side orthographic view hides.

        Returns:
            list[Dataset]: two single-band regional rasters.
        """
        frames = []
        for index in range(2):
            _, xx = np.mgrid[0:20, 0:24]
            values = (25 + 8 * np.sin((xx + index * 10) / 14.0)).astype("float32")
            frames.append(
                Dataset.from_array(
                    values,
                    geo_ref=GeoReference(
                        geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326
                    ),
                    no_data_value=-9999.0,
                )
            )
        return frames

    def test_a_swept_clim_ignores_the_views_that_see_nothing(self, regional_stack):
        """A rotation's colour scale comes from the views that can see the data, not the ones that cannot.

        Test scenario:
            rotate unions one measurement per swept projection. A view showing none of the data must
            contribute nothing; folding in the "no finite values" placeholder instead floors the scale at
            zero, and a regional AOI whose values sit around 250 then collapses into the top few percent
            of the ramp — visually the same bug the frozen-stretch work exists to prevent.
        """
        values = np.concatenate(
            [frame.read_array(band=0).ravel() for frame in regional_stack]
        )
        m = Map(crs=4326, globe=True, figsize=(4, 4))
        views = [
            projections.orthographic(lon=-180.0 + step * 15.0, lat=15.0)
            for step in range(24)
        ]
        opts = {}
        m._resolve_animation_clim(regional_stack, opts, views=views)
        assert opts["vmin"] == pytest.approx(float(np.nanmin(values)), rel=1e-3), (
            f"the swept clim floored at a placeholder instead of the data: {opts}"
        )
        assert opts["vmin"] > 1.0, (
            f"a placeholder (0, 1) must not leak into the union: {opts}"
        )

    def test_a_scalar_stack_animates_on_a_hiding_globe(self, regional_stack, tmp_path):
        """The colour-scale scan reprojects every frame, so it needed the guard too.

        Test scenario:
            The per-frame guard is unreachable for animate: _stack_clim warps each frame to derive one
            shared clim *before* any frame is drawn, so an entirely hidden stack raised out of the scan
            and never reached the draw. A frame that cannot be drawn contributes no colour range.
        """
        from matplotlib.animation import PillowWriter

        m = Map(
            crs=projections.orthographic(lon=-175, lat=15), globe=True, figsize=(4, 4)
        )
        anim = m.animate(regional_stack, fps=2)
        out = tmp_path / "hidden.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, (
            "a fully hidden stack should still render empty frames"
        )

    def test_a_composite_stack_animates_on_a_hiding_globe(
        self, regional_stack, tmp_path
    ):
        """The composite stretch scan warps each frame as well, and needed the same guard."""
        from matplotlib.animation import PillowWriter

        stack = []
        for frame in regional_stack:
            band = np.nan_to_num(frame.read_array(band=0)).astype("float32")
            stack.append(
                Dataset.from_array(
                    np.stack([band, band * 0.5, band * 0.25]),
                    geo_ref=GeoReference(
                        geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326
                    ),
                    no_data_value=-9999.0,
                )
            )
        m = Map(
            crs=projections.orthographic(lon=-175, lat=15), globe=True, figsize=(4, 4)
        )
        anim = m.animate(stack, kind="rgb_composite", fps=2)
        out = tmp_path / "hidden_rgb.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, (
            "a hidden composite stack should still render empty frames"
        )

    def test_a_visible_stack_still_gets_a_real_clim(self, regional_stack):
        """Skipping hidden frames must not skip visible ones: an on-limb stack keeps its colour range."""
        m = Map(crs=4326, figsize=(4, 4))
        opts = {}
        m._resolve_animation_clim(regional_stack, opts)
        values = np.concatenate(
            [frame.read_array(band=0).ravel() for frame in regional_stack]
        )
        assert opts["vmin"] == pytest.approx(float(np.nanmin(values)), rel=1e-3), (
            f"vmin should be the stack's own minimum, not a placeholder: {opts}"
        )
        assert opts["vmax"] == pytest.approx(float(np.nanmax(values)), rel=1e-3), (
            f"vmax should be the stack's own maximum: {opts}"
        )


class TestAnimateDatasetCollection:
    """animate accepts the DatasetCollection its docstring promises (issue #154)."""

    @pytest.fixture
    def cube(self, tmp_path):
        """A 3-member DatasetCollection written to disk.

        Returns:
            DatasetCollection: three single-band rasters read back from files.
        """
        from pyramids.dataset.collection import DatasetCollection

        paths = []
        for index in range(3):
            _, xx = np.mgrid[0:20, 0:24]
            values = (25 + 8 * np.sin((xx + index * 10) / 13.0)).astype("float32")
            path = tmp_path / f"f{index}.tif"
            Dataset.from_array(
                values,
                geo_ref=GeoReference(
                    geo=(58.2, 0.05, 0.0, 46.8, 0.0, -0.05), epsg=4326
                ),
                no_data_value=-9999.0,
            ).to_file(str(path))
            paths.append(str(path))
        return DatasetCollection.from_files(paths)

    def test_a_collection_iterates_to_arrays_not_datasets(self, cube):
        """The premise of the bug: plain iteration hands back numpy, so animate must not rely on it."""
        assert isinstance(next(iter(cube)), np.ndarray), (
            "if a collection ever iterates to Datasets, the unwrap in animate can go"
        )

    def test_animate_renders_a_collection(self, cube, tmp_path):
        """A DatasetCollection animates and renders, not just builds.

        Test scenario:
            animate did `list(stack)`, which on a collection yields the members' arrays rather than the
            Datasets, so the first thing to ask a frame for its CRS died with an AttributeError. Rendering
            (not merely constructing the lazy FuncAnimation) is what proves the frames are real Datasets.
        """
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.animate(cube, fps=2)
        assert len(list(anim.new_frame_seq())) == 3, "one frame per collection member"
        out = tmp_path / "cube.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, (
            "the collection animation should render a non-empty GIF"
        )
        assert m.ax.images, "each frame should draw its raster"

    def test_a_collection_and_its_members_animate_alike(self, cube):
        """Passing the collection and passing `.datasets` must produce the same animation."""
        by_collection = Map(crs=4326, figsize=(4, 4)).animate(cube, fps=2)
        by_members = Map(crs=4326, figsize=(4, 4)).animate(cube.datasets, fps=2)
        assert len(list(by_collection.new_frame_seq())) == len(
            list(by_members.new_frame_seq())
        )


class TestAnimate:
    """Tests for Map.animate."""

    def test_returns_funcanimation_with_frame_count(self, stack):
        """animate returns a FuncAnimation whose frame count matches the stack length (lazy, no render)."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(stack, fps=2, vmin=-40, vmax=70)
        assert isinstance(anim, FuncAnimation), (
            f"expected FuncAnimation, got {type(anim)}"
        )
        assert len(list(anim.new_frame_seq())) == len(stack), (
            "frame count must match the stack length"
        )

    def test_renders_gif_with_titles_and_ocean(self, stack, tmp_path):
        """Rendering writes a non-empty GIF; the last frame shows its title and the ocean disc.

        Test scenario:
            A globe animation with per-frame titles and the ocean disc renders all frames without error;
            after rendering, the axes hold the final frame's title and at least the ocean PolyCollection.
        """
        m = Map(crs=projections.orthographic(10, 15), globe=True, figsize=(4, 4))
        titles = ["Jan", "Feb", "Mar"]
        anim = m.animate(
            stack, fps=2, titles=titles, ocean=True, cmap="RdYlBu_r", vmin=-40, vmax=70
        )
        out = tmp_path / "anim.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.exists() and out.stat().st_size > 0, (
            "animation GIF should be non-empty"
        )
        assert m.ax.get_title() == titles[-1], (
            f"last title not applied: {m.ax.get_title()!r}"
        )
        assert m.ax.collections, (
            "ocean disc PolyCollection should be present on the last frame"
        )
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
        assert m._animation is anim, (
            "animate must keep a strong reference to the animation"
        )

    def test_resolve_clim_caps_scan(self, mocker):
        """_resolve_animation_clim scans at most _CLIM_SCAN_CAP frames of a large stack (L2)."""
        from digitalearth.static.maps import animation as anim_mod

        spy = mocker.spy(anim_mod.AnimationMixin, "_stack_clim")
        m = Map(crs=4326)
        big = [_field(float(o)) for o in range(anim_mod._CLIM_SCAN_CAP * 3)]
        opts = {}
        m._resolve_animation_clim(big, opts)
        scanned = len(
            spy.call_args.args[1]
        )  # args[0] is self, since the scan reprojects per frame
        assert scanned <= anim_mod._CLIM_SCAN_CAP, (
            f"scanned {scanned} frames, cap is {anim_mod._CLIM_SCAN_CAP}"
        )
        assert "vmin" in opts and "vmax" in opts, (
            "clim should still be resolved from the sampled frames"
        )

    def test_titles_length_mismatch_raises(self, stack):
        """A titles list of the wrong length raises ValueError."""
        m = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="titles length"):
            m.animate(stack, titles=["only-one"])

    def test_colorbar_adds_single_static_axes(self, stack, tmp_path):
        """colorbar=True adds exactly one colorbar axes that persists (not re-added) across all frames."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(
            stack,
            fps=2,
            colorbar=True,
            cbar_label="T",
            cmap="RdYlBu_r",
            vmin=-40,
            vmax=70,
        )
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
            return Dataset.from_array(
                arr=z,
                geo_ref=GeoReference(
                    geo=(-180.0, 6.0, 0.0, 90.0, 0.0, -6.0), epsg=4326
                ),
            )

        frames = [
            fld(0, 10),
            fld(0, 100),
            fld(0, 1000),
        ]  # wildly different per-frame ranges
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(frames, fps=2, cmap="viridis")  # no colorbar, no vmin/vmax
        clims = []
        for i in range(len(frames)):
            anim._func(i)
            clims.append(tuple(round(c) for c in m.ax.images[0].get_clim()))
        assert len(set(clims)) == 1, f"frames must share one colour scale, got {clims}"
        assert clims[0][0] == 0, (
            f"shared clim should start at the stack minimum, got {clims[0]}"
        )
        assert clims[0][1] > 800, (
            f"the scale spans what the globe renders, which drops the off-hemisphere extreme: {clims[0]}"
        )
        assert clims[0][1] <= 1000, (
            f"the scale cannot exceed the stack maximum: {clims[0]}"
        )

    def test_resolve_clim_fills_missing_bounds(self, stack):
        """_resolve_animation_clim computes a global clim only for the missing bound(s)."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        lo, hi = m._stack_clim(stack)
        both = {}
        m._resolve_animation_clim(stack, both)
        assert (both["vmin"], both["vmax"]) == (lo, hi), (
            "both bounds should be filled from the stack"
        )
        one = {"vmin": -100.0}
        m._resolve_animation_clim(stack, one)
        assert one["vmin"] == -100.0 and one["vmax"] == hi, (
            "only the missing bound should be filled"
        )
        explicit = {"vmin": -5.0, "vmax": 5.0}
        m._resolve_animation_clim(stack, explicit)
        assert (explicit["vmin"], explicit["vmax"]) == (-5.0, 5.0), (
            "explicit bounds must be left untouched"
        )

    def test_colorbar_computes_clim_from_stack(self, stack):
        """colorbar=True draws one bar over the resolved clim and adds a single colorbar axes."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        opts = {"cmap": "viridis"}
        m._resolve_animation_clim(stack, opts)
        lo, hi = m._stack_clim(stack)
        assert opts["vmin"] == lo and opts["vmax"] == hi, (
            "resolved clim should be injected into opts"
        )
        m._animation_colorbar(opts, "auto")
        assert len(m.fig.axes) == 2, "a colorbar axes should be present"

    def test_stack_clim_ignores_nodata(self):
        """_stack_clim excludes nodata cells when computing the range (from_array defaults to -9999)."""
        arr = np.array([[0.0, 5.0], [10.0, -9999.0]], dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        )
        assert ds.no_data_value[0] == -9999.0
        lo, hi = Map(crs=4326)._stack_clim([ds])
        assert (lo, hi) == (0.0, 10.0), (
            f"nodata -9999 should be ignored, got ({lo}, {hi})"
        )

    def test_stack_clim_no_nodata_uses_full_range(self):
        """When a dataset declares no nodata, every finite cell counts toward the range."""
        from types import SimpleNamespace

        ds = SimpleNamespace(
            read_array=lambda band=0: np.array([[1.0, 2.0], [3.0, 4.0]]),
            no_data_value=[None],
            epsg=4326,
        )
        assert Map(crs=4326)._stack_clim([ds]) == (1.0, 4.0)

    def test_stack_clim_all_nodata_falls_back(self):
        """An all-nodata stack has no finite cells, so _stack_clim returns the (0, 1) default."""
        from types import SimpleNamespace

        ds = SimpleNamespace(
            read_array=lambda band=0: np.array([[-9999.0, -9999.0]]),
            no_data_value=[-9999.0],
            epsg=4326,
        )
        assert Map(crs=4326)._stack_clim([ds]) == (0.0, 1.0)

    def test_colorbar_without_label(self):
        """A colorbar with no label still adds exactly one colorbar axes."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        m._animation_colorbar({"cmap": "viridis", "vmin": 0, "vmax": 60}, None)
        assert len(m.fig.axes) == 2, (
            "colorbar axes should be added even without a label"
        )

    def test_coastlines_best_effort_swallows_failure(self, stack, tmp_path, mocker):
        """coastlines=True still renders when the (network) coastline fetch raises — the error is swallowed."""
        mocker.patch.object(Map, "coastlines", side_effect=RuntimeError("offline"))
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.animate(stack, fps=2, coastlines=True, vmin=-40, vmax=70)
        out = tmp_path / "coast.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, (
            "animation should still render when coastlines fail"
        )
        assert Map.coastlines.called, "coastlines should have been attempted"


class TestAnimateBand:
    """Tests for the band the shared colour scale is measured from (#155)."""

    @pytest.fixture
    def two_band_stack(self):
        """A 3-frame stack whose two bands sit in very different ranges.

        Returns:
            list[Dataset]: frames whose band 1 spans 1..3 and band 2 spans 500..700.
        """
        geo_ref = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.02, epsg=4326)
        frames = []
        for k in range(3):
            low = np.full((40, 50), 1.0 + k, dtype="float32")
            high = np.full((40, 50), 500.0 + 100 * k, dtype="float32")
            frames.append(
                Dataset.from_array(
                    np.stack([low, high]), geo_ref=geo_ref, no_data_value=-9999.0
                )
            )
        return frames

    def test_scan_reads_the_band_being_animated(self, two_band_stack):
        """band=2 in opts scales against band 2's range, not band 1's.

        Test scenario:
            The stack's band 1 spans 1..3 and its band 2 spans 500..700. Resolving the clim with
            ``band=2`` must report 500..700; reporting 1..3 is the defect this covers.
        """
        opts = {"band": 2}
        Map(crs=4326)._resolve_animation_clim(two_band_stack, opts)
        assert (opts["vmin"], opts["vmax"]) == (500.0, 700.0), (
            f"band 2 spans 500..700, got {opts['vmin']}..{opts['vmax']}"
        )

    def test_scan_defaults_to_the_first_band(self, two_band_stack):
        """With no band in opts the scan keeps its band-1 default.

        Test scenario:
            The control for the case above — an unspecified band must still measure band 1.
        """
        opts = {}
        Map(crs=4326)._resolve_animation_clim(two_band_stack, opts)
        assert (opts["vmin"], opts["vmax"]) == (1.0, 3.0), (
            f"band 1 spans 1..3, got {opts['vmin']}..{opts['vmax']}"
        )

    def test_animate_applies_the_scanned_band_to_the_drawn_image(
        self, two_band_stack, tmp_path
    ):
        """End to end: animate(band=2) draws band 2 under band 2's clim.

        Test scenario:
            Renders every frame, then reads the clim matplotlib actually applied. The drawn data is band
            2's (700 in the last frame), so a clim of 1..3 would clip it to a flat saturated block.
        """
        m = Map(crs=4326, figsize=(3, 3))
        anim = m.animate(two_band_stack, kind="imshow", band=2, fps=2)
        anim.save(str(tmp_path / "band2.gif"), writer=PillowWriter(fps=2))
        image = m.ax.get_images()[-1]
        assert image.get_clim() == (500.0, 700.0), (
            f"drawn image should use band 2's range, got {image.get_clim()}"
        )
        assert float(np.nanmax(image.get_array())) == 700.0, (
            "the drawn data should be band 2's, confirming scale and draw agree"
        )

    def test_explicit_bounds_still_win_over_the_band_scan(self, two_band_stack):
        """A caller's own vmin/vmax is kept whatever band is animated.

        Test scenario:
            band=2 would scan 500..700, but explicit bounds must not be overwritten.
        """
        opts = {"band": 2, "vmin": -1.0, "vmax": 1.0}
        Map(crs=4326)._resolve_animation_clim(two_band_stack, opts)
        assert (opts["vmin"], opts["vmax"]) == (-1.0, 1.0), (
            f"explicit bounds should be kept, got {opts['vmin']}..{opts['vmax']}"
        )

    def test_rotate_scan_reads_the_band_too(self, two_band_stack):
        """The per-view union rotate() uses scans the animated band as well.

        Test scenario:
            rotate() turns one dataset under a sweep of projections, so it takes the
            ``_clim_across_views`` path and measures only the first frame. That frame's band 2 is a
            constant 500 and its band 1 a constant 1, so the reported bound says which band was read.
        """
        opts = {"band": 2}
        views = [projections.orthographic(lon, 30.0) for lon in (4.0, 5.0)]
        Map(crs=views[0], globe=True)._resolve_animation_clim(
            two_band_stack, opts, views=views
        )
        assert (opts["vmin"], opts["vmax"]) == (500.0, 500.0), (
            f"rotate's scan should read band 2 of the first frame, got "
            f"{opts['vmin']}..{opts['vmax']}"
        )


class TestAnimateComposites:
    """Tests for animating RGB/HSV composites — the kinds Map.animate used to refuse (issue #150)."""

    @pytest.mark.parametrize("kind", ["rgb_composite", "hsv_composite"])
    def test_composite_kind_accepted(self, rgb_stack, kind):
        """Both composite kinds are accepted and animate over the whole stack."""
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.animate(rgb_stack, kind=kind, fps=2)
        assert isinstance(anim, FuncAnimation), (
            f"expected FuncAnimation, got {type(anim)}"
        )
        assert len(list(anim.new_frame_seq())) == len(rgb_stack), (
            "frame count must match the stack length"
        )

    def test_composite_animation_renders_gif(self, rgb_stack, tmp_path):
        """A composite animation renders every frame to a non-empty GIF, titles included.

        Test scenario:
            A flat true-colour animation with per-frame titles renders without error; afterwards the axes
            hold the final frame's title and its RGB image.
        """
        m = Map(crs=4326, figsize=(4, 4))
        titles = ["t0", "t1", "t2"]
        anim = m.animate(rgb_stack, kind="rgb_composite", fps=2, titles=titles)
        out = tmp_path / "rgb.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, "composite animation GIF should be non-empty"
        assert m.ax.get_title() == titles[-1], (
            f"last title not applied: {m.ax.get_title()!r}"
        )
        assert m.ax.images, "each frame should draw the composite image"

    def test_composite_honours_band_order(self, tmp_path):
        """bands= reaches the freeze, so the frozen bounds come back in the order that was asked for.

        Test scenario:
            Three bands of clearly separated magnitude (10 / 100 / 1000) requested as (3, 2, 1). Asserting
            only that the render is three channels wide would hold for any order — and would still hold if
            the scan ignored bands= entirely and froze the default (1, 2, 3), which would silently attach
            each bound to the wrong channel. The magnitudes make the order observable.
        """
        ny, nx = 20, 40
        geo = GeoReference(
            geo=(-180.0, 360.0 / nx, 0.0, 90.0, 0.0, -180.0 / ny), epsg=4326
        )
        graded = Dataset.from_array(
            arr=np.stack(
                [np.full((ny, nx), scale, "float32") for scale in (10.0, 100.0, 1000.0)]
            ),
            geo_ref=geo,
        )
        m = Map(crs=4326, figsize=(4, 4))
        opts = {"bands": (3, 2, 1)}
        m._prime_animation(
            [graded], opts, kind="rgb_composite", colorbar=False, cbar_label=None
        )
        assert [round(lo) for lo, _ in opts["limits"]] == [1000, 100, 10], (
            f"frozen bounds should follow the requested band order, got {opts['limits']!r}"
        )

        anim = m.animate([graded], kind="rgb_composite", fps=2, bands=(3, 2, 1))
        anim.save(str(tmp_path / "bands.gif"), writer=PillowWriter(fps=2))
        assert m.ax.images[-1].get_array().shape[-1] == 3, (
            "a custom band order should still render RGB"
        )

    def test_rotate_accepts_composite(self, tmp_path):
        """rotate takes a composite kind too — it shares animate's kind validation."""
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.rotate(_rgb_field(), kind="rgb_composite", n_frames=3, fps=4)
        out = tmp_path / "rotrgb.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert out.stat().st_size > 0, "composite rotation GIF should be non-empty"

    def test_colorbar_refused_for_composite(self, rgb_stack):
        """colorbar=True on a composite is refused rather than answered with a meaningless bar."""
        m = Map(crs=4326)
        with pytest.raises(ValueError, match="colorbar=True is not supported"):
            m.animate(rgb_stack, kind="rgb_composite", colorbar=True)
        assert len(m.fig.axes) == 1, (
            "no colorbar axes should have been added before the refusal"
        )

    def test_priming_a_composite_skips_the_clim(self, rgb_stack):
        """Composite priming fills frozen limits and injects no clim (an RGB image ignores vmin/vmax)."""
        m = Map(crs=4326)
        opts = {}
        m._prime_animation(
            rgb_stack, opts, kind="rgb_composite", colorbar=False, cbar_label=None
        )
        assert "vmin" not in opts, f"composite priming injected a vmin: {opts}"
        assert "vmax" not in opts, f"composite priming injected a vmax: {opts}"
        assert len(opts["limits"]) == 3, (
            f"expected one (lo, hi) per channel, got {opts['limits']!r}"
        )

    def test_priming_a_field_still_resolves_the_clim(self, stack):
        """The scalar path is untouched — priming a field kind still fills one shared vmin/vmax."""
        m = Map(crs=4326)
        opts = {}
        m._prime_animation(stack, opts, kind="imshow", colorbar=False, cbar_label=None)
        assert opts["vmin"] is not None, "field priming must still set a vmin"
        assert opts["vmax"] is not None, "field priming must still set a vmax"
        assert "limits" not in opts, "a scalar field takes no composite stretch limits"

    def test_caller_limits_are_kept(self, rgb_stack):
        """An explicit limits= wins — the stack scan only fills a missing one."""
        m = Map(crs=4326)
        mine = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        opts = {"limits": mine}
        m._prime_animation(
            rgb_stack, opts, kind="rgb_composite", colorbar=False, cbar_label=None
        )
        assert opts["limits"] == mine, (
            "caller-supplied limits must not be overwritten by the scan"
        )

    def test_explicit_none_limits_are_still_frozen(self, rgb_stack):
        """An explicit ``limits=None`` is filled, not treated as "the caller already chose".

        Test scenario:
            Forwarding an optional through a wrapper passes the kwarg as None rather than omitting it. Key
            presence alone would leave that None in place, stretch_to_unit would fall back to its per-frame
            percentiles, and the clip would pump exactly as it did before the freeze existed — silently.
        """
        m = Map(crs=4326)
        opts = {"limits": None}
        m._prime_animation(
            rgb_stack, opts, kind="rgb_composite", colorbar=False, cbar_label=None
        )
        assert opts["limits"] is not None, (
            "an explicit limits=None must still be filled from the stack"
        )
        assert len(opts["limits"]) == 3, (
            f"expected one bound per channel, got {opts['limits']!r}"
        )

    @pytest.mark.parametrize("display", ["webmercator", "orthographic"])
    def test_frozen_limits_are_measured_on_the_display_crs(self, display):
        """The scan measures the reprojected frame, not the stored one.

        Test scenario:
            The composites stretch `get_stack(self._reproject(dataset))`, so bounds taken from the stored
            values freeze the wrong numbers under any non-trivial display CRS. Under both a Web Mercator and
            an orthographic display the frozen bounds must equal the warped frame's own, and must differ from
            the stored frame's — otherwise the animation renders on a different stretch from the still.
        """
        crs = projections.orthographic(0, 15) if display == "orthographic" else 3857
        frame = _rgb_field()
        m = Map(crs=crs)
        frozen = [
            value
            for pair in m._stack_channel_limits([frame], (1, 2, 3))
            for value in pair
        ]
        warped = [
            value
            for pair in channel_limits(get_stack(m._reproject(frame), (1, 2, 3)))
            for value in pair
        ]
        stored = [
            value
            for pair in channel_limits(get_stack(frame, (1, 2, 3)))
            for value in pair
        ]
        assert frozen == pytest.approx(warped), (
            f"frozen {frozen} should match the warped frame {warped}"
        )
        assert frozen != pytest.approx(stored), (
            f"frozen {frozen} must not be the stored bounds {stored}"
        )

    def test_composite_animation_matches_the_still_under_a_warp(self, tmp_path):
        """An animated composite renders like the equivalent still, even on a strong projection.

        Test scenario:
            The orthographic warp `rotate` performs moves the 2-98 percentiles a long way. Measuring the
            stored values instead of the warped ones left the animated frame about a third darker than the
            same data drawn as a still; both paths must now agree to within a few percent.
        """
        frames = [_rgb_field(shift=0.0), _rgb_field(shift=10.0)]
        crs = projections.orthographic(0, 15)
        animated_map = Map(crs=crs, globe=True, figsize=(4, 4))
        anim = animated_map.animate(frames, kind="rgb_composite", fps=2)
        anim.save(str(tmp_path / "warp.gif"), writer=PillowWriter(fps=2))
        animated = np.nanmean(
            np.asarray(animated_map.ax.images[-1].get_array(), dtype="float64")
        )

        still_map = Map(crs=crs, globe=True, figsize=(4, 4))
        still_map.rgb_composite(frames[-1])
        still = np.nanmean(
            np.asarray(still_map.ax.images[-1].get_array(), dtype="float64")
        )

        assert animated == pytest.approx(still, rel=0.05), (
            f"animated frame {animated:.4f} should render like the still {still:.4f}"
        )

    def test_rotate_freezes_on_the_views_it_sweeps(self, tmp_path):
        """rotate measures the projections it is about to draw, not the CRS it happened to be built with.

        Test scenario:
            rotate assigns its display CRS inside the frame callback, so priming against the Map's build-time
            CRS described a view no frame uses — and on the documented Map(crs=4326) usage that meant no warp
            at all. Driven end to end, the last rendered frame must match the still at that same projection;
            frozen on the unwarped bounds it comes out about 8% dark against a 2% residue for the union.
        """
        frame = _rgb_field()
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.rotate(frame, kind="rgb_composite", n_frames=4, lon0=-180.0, fps=4)
        anim.save(str(tmp_path / "spin.gif"), writer=PillowWriter(fps=4))
        rendered = np.nanmean(np.asarray(m.ax.images[-1].get_array(), dtype="float64"))

        still = Map(crs=m.crs, globe=True, figsize=(4, 4))
        still.rgb_composite(frame)
        expected = np.nanmean(
            np.asarray(still.ax.images[-1].get_array(), dtype="float64")
        )
        assert rendered == pytest.approx(expected, rel=0.05), (
            f"the swept frame {rendered:.4f} should render like the still at that view {expected:.4f}"
        )

    def test_the_view_scan_restores_the_display_crs(self):
        """Priming borrows the display CRS to measure each view and must hand it back untouched."""
        frame = _rgb_field()
        views = [
            projections.orthographic(lon=-180.0 + k * 90.0, lat=15.0) for k in range(4)
        ]
        spun = Map(crs=4326, figsize=(4, 4))
        primed = {}
        spun._prime_animation(
            [frame],
            primed,
            kind="rgb_composite",
            colorbar=False,
            cbar_label=None,
            views=views,
        )
        assert spun.crs == 4326, (
            f"the scan must restore the display CRS it borrowed, left {spun.crs!r}"
        )

        per_view = []
        for view in views:
            probe = Map(crs=view, figsize=(4, 4))
            per_view.append(
                channel_limits(get_stack(probe._reproject(frame), (1, 2, 3)))
            )
        for channel, (lo, hi) in enumerate(primed["limits"]):
            assert lo <= min(v[channel][0] for v in per_view) + 1e-6, (
                f"channel {channel} lo too high"
            )
            assert hi >= max(v[channel][1] for v in per_view) - 1e-6, (
                f"channel {channel} hi too low"
            )

    def test_a_sweep_that_never_shows_the_data_still_freezes(self):
        """Every sampled view failing to warp falls back to the data as stored, rather than raising.

        Test scenario:
            A small local raster spun on an orthographic globe spends part of the sweep entirely on the far
            side, where pyramids has nothing to transform and raises. Those views contribute no bound; if
            none of them shows anything the scan must still return a usable stretch.
        """
        ny, nx = 6, 6
        local = Dataset.from_array(
            arr=np.stack(
                [np.full((ny, nx), value, "float32") for value in (10.0, 20.0, 30.0)]
            ),
            geo_ref=GeoReference(geo=(100.0, 0.1, 0.0, 10.0, 0.0, -0.1), epsg=4326),
        )
        away = [
            projections.orthographic(lon=-80.0 + offset, lat=-60.0)
            for offset in (0.0, 5.0)
        ]
        frozen = Map(crs=4326)._stack_channel_limits([local], (1, 2, 3), views=away)
        assert len(frozen) == 3, f"a stretch must still come back, got {frozen!r}"
        assert all(np.isfinite(value) for pair in frozen for value in pair), (
            f"the stored-data fallback must produce real bounds: {frozen!r}"
        )

    def test_frozen_limits_span_the_whole_stack(self):
        """The frozen limits bracket every individual frame's own limits (widest lo/hi wins)."""
        frames = [_rgb_field(exposure=1.0), _rgb_field(exposure=0.4)]
        frozen = Map(crs=4326)._stack_channel_limits(frames, (1, 2, 3))
        per_frame = [channel_limits(get_stack(f, (1, 2, 3))) for f in frames]
        for channel, (lo, hi) in enumerate(frozen):
            assert lo <= min(p[channel][0] for p in per_frame), (
                f"channel {channel} lo is not the widest"
            )
            assert hi >= max(p[channel][1] for p in per_frame), (
                f"channel {channel} hi is not the widest"
            )

    @pytest.mark.parametrize("frames", [25, 47, 72])
    def test_stack_channel_limits_caps_scan(self, mocker, frames):
        """The scan never reads more than _CLIM_SCAN_CAP frames, including the awkward sizes.

        Test scenario:
            A floor-divided stride returns 1 for any stack under twice the cap, so 25 and 47 frames were
            both read in full while the docstring promised 24. Only an exact multiple of the cap (72) hid
            it, which is what the original test used.
        """
        from digitalearth.static.maps import animation as anim_mod

        spy = mocker.spy(anim_mod, "channel_limits")
        big = [_rgb_field(shift=float(s), ny=12, nx=24) for s in range(frames)]
        limits = Map(crs=4326)._stack_channel_limits(big, (1, 2, 3))
        cap = anim_mod._CLIM_SCAN_CAP
        assert spy.call_count <= cap, (
            f"scanned {spy.call_count} of {frames} frames, cap is {cap}"
        )
        assert len(limits) == 3, (
            "the capped scan must still yield one (lo, hi) per channel"
        )

    @pytest.mark.parametrize("kind", ["rgb_composite", "hsv_composite"])
    def test_colorbar_refused_for_either_composite(self, rgb_stack, kind):
        """Both composite kinds refuse a colorbar, not just the RGB one."""
        m = Map(crs=4326)
        with pytest.raises(ValueError, match=f"{kind}"):
            m.animate(rgb_stack, kind=kind, colorbar=True)

    @pytest.mark.parametrize("position", ["first", "last"])
    def test_one_dead_frame_does_not_poison_the_channel(self, position):
        """An all-nodata frame contributes no bound, wherever it sits in the stack.

        Test scenario:
            One frame's second channel is entirely nodata, placed either first or last in the stack. Its
            (nan, nan) must be dropped rather than folded in: min/max against nan is order-dependent, so a
            dead *first* frame would otherwise return (nan, nan) and blank that channel for every frame of
            the animation, while a dead *last* frame would happen to survive. Both orders must agree with
            the live frame's own bounds.
        """
        live = _rgb_field(shift=5.0)
        dead = _rgb_field_with_dead_channel(shift=5.0)
        stack = [dead, live] if position == "first" else [live, dead]
        frozen = Map(crs=4326)._stack_channel_limits(stack, (1, 2, 3))
        lo, hi = frozen[1]
        assert np.isfinite(lo), (
            f"a dead frame poisoned channel 2's low bound: {frozen[1]}"
        )
        assert np.isfinite(hi), (
            f"a dead frame poisoned channel 2's high bound: {frozen[1]}"
        )
        frozen_bounds = frozen[1]
        surviving = channel_limits(get_stack(live, (1, 2, 3)))[1]
        assert frozen_bounds == pytest.approx(surviving), (
            f"the surviving frame's own bounds {surviving} should be the frozen ones {frozen_bounds}"
        )

    def test_channel_dead_in_every_frame_reports_no_bound(self):
        """A channel the scan never saw alive reports (nan, nan) — "no frozen bound", not a span."""
        stack = [_rgb_field_with_dead_channel(shift=s) for s in (0.0, 10.0)]
        frozen = Map(crs=4326)._stack_channel_limits(stack, (1, 2, 3))
        assert np.isnan(frozen[1]).all(), (
            f"an unmeasurable channel should report (nan, nan): {frozen[1]}"
        )
        assert all(np.isfinite(v) for v in frozen[0] + frozen[2]), (
            f"live channels keep real bounds: {frozen}"
        )

    def test_strided_scan_does_not_clip_a_live_channel_flat(self):
        """A channel the stride never samples alive still renders on its own stretch, not saturated.

        Test scenario:
            The scan samples a subset of a long stack, so "dead in every scanned frame" is not "dead in
            every frame". Channel 2 is made nodata on exactly the frames the scan will sample (derived from
            _scan_subset rather than assumed, so a change of stride cannot quietly defuse this) and carries
            real data on the rest. Answering that with a fixed (0, 1) span would push every real value
            through clip((500 - 0) / 1) = 1.0 and blow the channel out in every frame that has data.
        """
        from digitalearth.static.maps import animation as anim_mod

        count = 50
        scanned_indices = set(anim_mod._scan_subset(list(range(count))))
        stack = []
        for index in range(count):
            dead = index in scanned_indices
            stack.append(
                _rgb_field_with_dead_channel(shift=float(index), ny=12, nx=24)
                if dead
                else _rgb_field(shift=float(index), ny=12, nx=24)
            )
        live_index = next(
            index for index in range(count) if index not in scanned_indices
        )
        frozen = Map(crs=4326)._stack_channel_limits(stack, (1, 2, 3))
        assert np.isnan(frozen[1]).all(), (
            f"the unscanned-alive channel should report no bound: {frozen[1]}"
        )

        live = get_stack(stack[live_index], (1, 2, 3))
        stretched = stretch_to_unit(live, frozen)
        channel = stretched[..., 1]
        assert channel.min() < channel.max(), (
            "a live channel must keep real contrast, not clip flat"
        )
        assert channel.mean() < 0.9, (
            f"the channel reads as blown out: mean {channel.mean():.3f}"
        )

    def test_dead_channel_still_renders(self, tmp_path):
        """A stack with one dead channel still animates instead of failing or blanking every frame."""
        stack = [_rgb_field_with_dead_channel(shift=0.0), _rgb_field(shift=10.0)]
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.animate(stack, kind="rgb_composite", fps=2)
        out = tmp_path / "dead.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, (
            "an animation with a dead channel should still render"
        )
        rendered = np.asarray(m.ax.images[-1].get_array(), dtype="float64")
        assert np.isfinite(rendered[..., 0]).any(), (
            "the live channels must not be blanked by the dead one"
        )

    def test_mask_nodata_is_threaded_into_the_frozen_limits(self):
        """mask_nodata=False reaches the stack scan, so the nodata sentinel widens the frozen limits.

        Test scenario:
            A frame carrying real -9999 nodata cells. Masked (the default) those cells are excluded and the
            low bound sits in the data range; unmasked the sentinel participates, dragging the low bound far
            below it — proving the flag is forwarded rather than defaulted.
        """
        stack = [_rgb_field_with_dead_channel(shift=0.0), _rgb_field(shift=10.0)]
        masked, unmasked = {}, {"mask_nodata": False}
        Map(crs=4326)._prime_animation(
            stack, masked, kind="rgb_composite", colorbar=False, cbar_label=None
        )
        Map(crs=4326)._prime_animation(
            stack, unmasked, kind="rgb_composite", colorbar=False, cbar_label=None
        )
        assert unmasked["limits"][1][0] < masked["limits"][1][0], (
            f"mask_nodata=False should widen the low bound: {unmasked['limits'][1]} vs {masked['limits'][1]}"
        )

    @pytest.mark.parametrize("bands", [(1, 2), (1, 2, 3, 4)])
    def test_wrong_band_count_is_refused_before_the_scan(
        self, rgb_stack, bands, mocker
    ):
        """A wrong band count fails up front, naming the caller — not mid-render as a bare IndexError.

        Test scenario:
            Priming used to scan up to 24 frames off disk and only then die inside FuncAnimation.save, where
            PillowWriter.finish() raises over the top of the real error. The check must therefore run before
            the scan, so no frame is read at all.
        """
        spy = mocker.spy(Map, "_stack_channel_limits")
        m = Map(crs=4326)
        with pytest.raises(ValueError, match="needs exactly three bands"):
            m.animate(rgb_stack, kind="rgb_composite", bands=bands)
        assert spy.call_count == 0, (
            "the stack must not be scanned before the band count is checked"
        )

    def test_a_refused_rotate_leaves_the_map_alone(self):
        """rotate validates before it mutates: a refused call must not switch the Map into globe mode.

        Test scenario:
            n_frames and kind are both checked before anything is touched, but globe was set before the
            priming that raises on a composite + colorbar, so the caller was left holding a Map that had
            been switched to a globe by a call that never ran.
        """
        m = Map(crs=4326)
        frame = _rgb_field()
        with pytest.raises(ValueError, match="colorbar=True is not supported"):
            m.rotate(frame, kind="rgb_composite", n_frames=3, colorbar=True)
        assert m.globe is False, (
            "a rotate that raised must not have switched the Map into globe mode"
        )

    def test_empty_stack_has_no_limits_to_derive(self):
        """The helper says so rather than raising IndexError off an empty scan (reachable directly)."""
        m = Map(crs=4326)
        with pytest.raises(ValueError, match="empty stack"):
            m._stack_channel_limits([], (1, 2, 3))

    def test_two_band_composite_limits(self):
        """The scan follows the bands it is given — two bands yield two channel bounds, not three."""
        stack = [_rgb_field(shift=s) for s in (0.0, 10.0)]
        frozen = Map(crs=4326)._stack_channel_limits(stack, (1, 2))
        assert len(frozen) == 2, (
            f"expected one bound per requested band, got {frozen!r}"
        )

    def test_frozen_stretch_keeps_a_real_brightness_change(self, tmp_path):
        """One frozen stretch preserves a genuine brightness drop that a per-frame stretch would erase.

        Test scenario:
            Two frames of the same scene, the second at 40% exposure. Animated, the second frame renders
            visibly darker because both frames share one stretch; rendered as a still each frame
            renormalises to its own 2-98 percentile, so the drop disappears — the flicker in issue #150.
        """
        frames = [_rgb_field(exposure=1.0), _rgb_field(exposure=0.4)]
        m = Map(crs=4326, figsize=(4, 4))
        anim = m.animate(frames, kind="rgb_composite", fps=2)
        anim.save(str(tmp_path / "dim.gif"), writer=PillowWriter(fps=2))
        animated_dim = np.nanmean(
            np.asarray(m.ax.images[-1].get_array(), dtype="float64")
        )

        stills = []
        for frame in frames:
            still = Map(crs=4326, figsize=(4, 4))
            still.rgb_composite(frame)
            stills.append(
                np.nanmean(np.asarray(still.ax.images[-1].get_array(), dtype="float64"))
            )

        assert stills[1] == pytest.approx(stills[0], abs=1e-6), (
            "per-frame stretch should erase the drop"
        )
        assert animated_dim < stills[0] * 0.75, (
            f"the dim frame must stay dim under a frozen stretch: {animated_dim:.3f} vs {stills[0]:.3f}"
        )


class TestRotate:
    """Tests for Map.rotate."""

    def test_frame_count_and_forces_globe(self):
        """rotate returns the requested frame count and forces the map into globe mode."""
        m = Map(crs=projections.orthographic(0, 15), figsize=(4, 4))  # starts non-globe
        anim = m.rotate(_field(5.0), n_frames=6, fps=4, vmin=-40, vmax=70)
        assert isinstance(anim, FuncAnimation), (
            f"expected FuncAnimation, got {type(anim)}"
        )
        assert len(list(anim.new_frame_seq())) == 6, (
            "rotate frame count must equal n_frames"
        )
        assert m.globe is True, "rotate must force globe mode"

    def test_renders_gif_and_sweeps_longitude(self, tmp_path):
        """Rendering writes a non-empty GIF and sweeps the centre longitude to the last frame's value.

        Test scenario:
            A 4-frame rotation from lon0=-180 steps the orthographic centre to -180, -90, 0, 90; after
            rendering, the display CRS holds the final centre longitude (+lon_0=90).
        """
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.rotate(
            _field(5.0),
            n_frames=4,
            fps=4,
            lon0=-180.0,
            ocean=True,
            cmap="terrain",
            vmin=-40,
            vmax=70,
        )
        out = tmp_path / "rot.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert out.stat().st_size > 0, "rotation GIF should be non-empty"
        assert "+lon_0=90" in m.crs, (
            f"final centre longitude not swept to 90: {m.crs!r}"
        )
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
        anim = m.rotate(
            _field(5.0),
            n_frames=3,
            fps=4,
            colorbar=True,
            cbar_label="elev",
            cmap="terrain",
            vmin=-40,
            vmax=70,
        )
        assert len(m.fig.axes) == 2, "rotate colorbar should add one axes"
        out = tmp_path / "rotcbar.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert len(m.fig.axes) == 2 and out.stat().st_size > 0, (
            "colorbar must stay single after rendering"
        )

    def test_rotate_coastlines_best_effort(self, tmp_path, mocker):
        """rotate(coastlines=True) attempts coastlines each frame and still renders when they fail offline."""
        coast = mocker.patch.object(
            Map, "coastlines", side_effect=RuntimeError("offline")
        )
        m = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(4, 4))
        anim = m.rotate(
            _field(5.0), n_frames=2, fps=4, coastlines=True, vmin=-40, vmax=70
        )
        out = tmp_path / "rotcoast.gif"
        anim.save(str(out), writer=PillowWriter(fps=4))
        assert out.stat().st_size > 0, (
            "rotation should still render when coastlines fail"
        )
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
        m._draw_animation_frame(
            _field(0.0), "imshow", {}, ocean=True, coastlines=False, title="frame-0"
        )
        assert m.ax.get_title() == "frame-0", f"title not set, got {m.ax.get_title()!r}"
        assert len(m.layers) == 1, (
            f"expected one drawn field layer, got {len(m.layers)}"
        )

    def test_no_title_leaves_title_empty(self):
        """_draw_animation_frame leaves the title untouched when none is given.

        Test scenario:
            Omitting title draws the field without setting any axes title.
        """
        m = Map(crs=4326)
        m._draw_animation_frame(
            _field(0.0), "imshow", {}, ocean=False, coastlines=False
        )
        assert m.ax.get_title() == "", (
            f"title should be empty, got {m.ax.get_title()!r}"
        )
