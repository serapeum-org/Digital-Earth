"""Tests for ST-9 — animating without clearing the axes, and blitting.

The capability under test is the per-frame strategy, not the picture: an animation whose ``kind`` draws one
updatable artist gives that artist the next frame's values (``AxesImage.set_data`` / ``QuadMesh.set_array``)
instead of clearing the axes and rebuilding every layer, which is what makes ``blit=True`` possible at all.
The proof that the strategy is only a strategy is :meth:`TestInPlaceFrames.
test_in_place_frames_match_a_cleared_redraw_pixel_for_pixel`, which renders the same stack both ways and
compares the canvases.
"""

import logging

import numpy as np
import pytest
from matplotlib.animation import PillowWriter
from matplotlib.image import AxesImage
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map, projections
from digitalearth.static.maps.animation import FrameUpdate

#: A clim every test passes explicitly, so the colour scale is never what a comparison is measuring.
CLIM = {"vmin": -10.0, "vmax": 40.0}


def _field(offset: float, *, bands: int = 1) -> Dataset:
    """A small global lon/lat raster, shifted by ``offset`` so consecutive frames differ.

    Args:
        offset: Constant added to every cell.
        bands: How many bands the raster carries.

    Returns:
        A 30x60 global EPSG:4326 raster.
    """
    ny, nx = 30, 60
    lat = np.linspace(90, -90, ny)[:, None]
    plane = (np.cos(np.deg2rad(lat)) * 20 + offset) * np.ones((ny, nx), "float32")
    arr = plane if bands == 1 else np.stack([plane * (1 + k) for k in range(bands)])
    return Dataset.from_array(
        arr=arr.astype("float32"),
        geo_ref=GeoReference(geo=(-180.0, 6.0, 0.0, 90.0, 0.0, -6.0), epsg=4326),
    )


@pytest.fixture
def stack():
    """A 3-frame stack of small global fields.

    Returns:
        list[Dataset]: three distinct global rasters.
    """
    return [_field(offset) for offset in (0.0, 8.0, 16.0)]


@pytest.fixture
def rgb_stack():
    """A 2-frame stack of 3-band rasters — a composite animation.

    Returns:
        list[Dataset]: two distinct 3-band global rasters.
    """
    return [_field(offset, bands=3) for offset in (0.0, 8.0)]


def _canvas(scene: Map) -> np.ndarray:
    """Render the scene and return its canvas as RGBA pixels.

    Args:
        scene: The map to draw.

    Returns:
        The figure's RGBA buffer, copied so a later draw cannot change it.
    """
    scene.fig.canvas.draw()
    return np.asarray(scene.fig.canvas.buffer_rgba()).copy()


def _frames_of(scene: Map, anim, count: int):
    """Drive ``count`` frames of ``anim`` and collect what each one left on the canvas.

    Args:
        scene: The map the animation draws on.
        anim: The ``FuncAnimation`` to drive.
        count: How many frames to draw.

    Returns:
        One RGBA buffer per frame, in frame order.
    """
    return [(anim._func(index), _canvas(scene))[1] for index in range(count)]


class TestFrameUpdateStrategy:
    """The strategy object itself: what it allows, what it refuses, and what it settles to."""

    def test_an_image_field_is_updatable(self, stack):
        """The capability's own case reports itself available.

        Test scenario:
            A ``FrameUpdate`` built for an ``imshow`` animation with plain options says the frames update
            one artist.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        strategy = FrameUpdate(scene, stack, dict(CLIM), kind="imshow")
        assert strategy.in_place is True, (
            f"an image field should be updatable, blocked by {strategy.blocker!r}"
        )
        scene.close()

    def test_a_blocked_strategy_names_the_kind_it_cannot_update(self, stack):
        """The blocker is a sentence, because the refusal and the log line both quote it.

        Test scenario:
            A composite's ``blocker`` names the kind, so the two callers of it cannot describe the same
            animation differently.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        strategy = FrameUpdate(scene, stack, {}, kind="hsv_composite")
        assert "hsv_composite" in str(strategy.blocker), (
            f"the blocker should name the kind, got {strategy.blocker!r}"
        )
        scene.close()

    def test_an_updatable_strategy_settles_to_itself(self, stack):
        """``settle`` hands back the strategy the frame loop should run.

        Test scenario:
            An updatable strategy asked for blitting returns itself, which is what the frame loop takes as
            "update rather than redraw".
        """
        scene = Map(crs=4326, figsize=(3, 3))
        strategy = FrameUpdate(scene, stack, dict(CLIM), kind="imshow")
        assert strategy.settle(blit=True) is strategy, (
            "an updatable strategy must settle to itself"
        )
        scene.close()

    def test_a_blocked_strategy_settles_to_nothing(self, stack):
        """And a blocked one settles to ``None``, which is the redrawing path.

        Test scenario:
            A contour strategy under the default mode falls back rather than refusing, so ``settle``
            reports no strategy at all.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        strategy = FrameUpdate(scene, stack, {}, kind="contour")
        assert strategy.settle(blit=False) is None, (
            "a blocked strategy must settle to no strategy"
        )
        scene.close()


class TestInPlaceFrames:
    """An animation that can update one artist does that instead of clearing the axes."""

    def test_the_data_artist_survives_every_frame(self, stack):
        """Every frame draws onto the artist frame 0 created, rather than a new one.

        Test scenario:
            A flat three-frame field animation is driven frame by frame; the image on the axes is the same
            object throughout, and there is only ever one of it.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        anim = scene.animate(stack, fps=2, **CLIM)
        anim._func(0)
        first = scene.ax.images[0]
        for index in (1, 2):
            anim._func(index)
        assert scene.ax.images[0] is first, "the frames must share one image artist"
        assert len(scene.ax.images) == 1, (
            f"an in-place animation adds no image per frame, got {len(scene.ax.images)}"
        )

    def test_the_decoration_is_drawn_once_not_once_per_frame(self, stack, mocker):
        """Globe decoration is background: it is drawn for frame 0 and left standing.

        Test scenario:
            A globe animation with the ocean disc is driven for three frames with ``Map.ocean`` spied on.
            Clearing the axes would throw the disc away and redraw it three times; keeping the axes draws
            it once.
        """
        scene = Map(crs=projections.orthographic(0, 15), globe=True, figsize=(3, 3))
        spy = mocker.spy(scene, "ocean")
        anim = scene.animate(stack, fps=2, ocean=True, **CLIM)
        for index in range(3):
            anim._func(index)
        assert spy.call_count == 1, (
            f"the ocean disc should be drawn once, not per frame; drawn {spy.call_count} times"
        )

    def test_the_axes_is_not_cleared_after_the_first_frame(self, stack, mocker):
        """``Axes.clear`` runs for frame 0 and never again.

        Test scenario:
            The same three-frame animation with ``Axes.clear`` spied on. One call is the first frame's; a
            second would mean the frame's artists, decoration and view were thrown away again.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        spy = mocker.spy(scene.ax, "clear")
        anim = scene.animate(stack, fps=2, **CLIM)
        for index in range(3):
            anim._func(index)
        assert spy.call_count == 1, (
            f"the axes should be cleared once, got {spy.call_count} clears"
        )

    def test_in_place_frames_match_a_cleared_redraw_pixel_for_pixel(self, stack):
        """Updating in place and clearing-and-redrawing put the same pixels on the canvas.

        Test scenario:
            The same stack is animated twice on two identical figures — once with ``update="in_place"``,
            once with ``update="redraw"`` — and each frame's RGBA buffer is compared. The two paths share
            no code after the first frame, so an equal canvas is the evidence that the strategy is only a
            strategy.
        """
        kept = Map(crs=4326, figsize=(3, 3))
        rebuilt = Map(crs=4326, figsize=(3, 3))
        in_place = _frames_of(
            kept, kept.animate(stack, fps=2, update="in_place", **CLIM), 3
        )
        redrawn = _frames_of(
            rebuilt, rebuilt.animate(stack, fps=2, update="redraw", **CLIM), 3
        )
        for index, (left, right) in enumerate(zip(in_place, redrawn)):
            assert np.array_equal(left, right), (
                f"frame {index} differs between the in-place and the redrawn path"
            )

    def test_consecutive_frames_really_change_the_canvas(self, stack):
        """The comparison above would also pass on an animation that never updated anything.

        Test scenario:
            The control for the pixel comparison: frames 0 and 2 of the in-place animation must differ,
            or a frame function that quietly did nothing would read as correct.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        drawn = _frames_of(
            scene, scene.animate(stack, fps=2, update="in_place", **CLIM), 3
        )
        assert not np.array_equal(drawn[0], drawn[2]), (
            "the first and last frame of a changing stack must not be the same picture"
        )

    def test_a_per_frame_title_is_restated_on_every_frame(self, stack):
        """A title is the one piece of per-frame decoration, so an updated frame sets it too.

        Test scenario:
            Three titled frames are driven one at a time and the axes title is read after each. Updating
            only the image left every frame carrying the first frame's title, which is the defect this
            covers.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        titles = ["Jan", "Feb", "Mar"]
        anim = scene.animate(stack, fps=2, titles=titles, **CLIM)
        shown = []
        for index in range(len(titles)):
            anim._func(index)
            shown.append(scene.ax.get_title())
        assert shown == titles, f"each frame should show its own title, got {shown}"

    def test_the_frame_function_hands_back_the_artist_it_changed(self, stack):
        """Blitting needs the frame function to return what it redrew, so it does.

        Test scenario:
            The frame function's return value is inspected directly: it is a sequence holding the image
            matplotlib must repaint for that frame.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        anim = scene.animate(stack, fps=2, **CLIM)
        anim._func(0)
        returned = list(anim._func(1))
        assert len(returned) == 1, f"one artist changes per frame, got {returned!r}"
        assert isinstance(returned[0], AxesImage), (
            f"a field frame repaints an AxesImage, got {type(returned[0]).__name__}"
        )

    def test_a_mesh_kind_updates_its_cells_in_place(self, stack):
        """``pcolormesh`` keeps its ``QuadMesh`` and swaps the cell values.

        Test scenario:
            A mesh animation is driven for two frames; the collection on the axes is the same object, and
            the values it holds are the second frame's.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        anim = scene.animate(stack, kind="pcolormesh", fps=2, **CLIM)
        anim._func(0)
        mesh = scene.ax.collections[0]
        anim._func(1)
        assert scene.ax.collections[0] is mesh, "the frames must share one QuadMesh"
        assert float(np.nanmax(mesh.get_array())) == pytest.approx(28.0, abs=0.5), (
            f"the mesh should hold frame 1's values, got max {np.nanmax(mesh.get_array())}"
        )

    def test_the_redraw_mode_rebuilds_the_artist_every_frame(self, stack):
        """``update="redraw"`` is the escape hatch, and still clears.

        Test scenario:
            The same animation under ``update="redraw"``: the image after frame 1 is a different object
            from the one frame 0 drew, which is what clearing the axes means.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        anim = scene.animate(stack, fps=2, update="redraw", **CLIM)
        anim._func(0)
        first = scene.ax.images[0]
        anim._func(1)
        assert scene.ax.images[0] is not first, (
            "update='redraw' must rebuild the layer rather than update it"
        )

    def test_an_unknown_update_mode_is_refused(self, stack):
        """A misspelled mode fails at the call, naming the modes there are.

        Test scenario:
            ``update="inplace"`` (no underscore) is refused up front rather than silently redrawing.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with pytest.raises(ValueError, match="unknown update mode"):
            scene.animate(stack, update="inplace")


class TestInPlaceRefusals:
    """What cannot be updated in place is named, not silently animated wrongly."""

    def test_a_contour_kind_cannot_be_updated_in_place(self, stack):
        """A contour set is rebuilt from the values, so there is no artist to hand new data to.

        Test scenario:
            ``update="in_place"`` with ``kind="contourf"`` is refused, and the message names the kind.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with pytest.raises(ValueError, match="contourf"):
            scene.animate(stack, kind="contourf", update="in_place", **CLIM)

    def test_a_composite_cannot_be_updated_in_place(self, rgb_stack):
        """A composite's pixels are a frozen three-channel stretch, not the band's values.

        Test scenario:
            ``update="in_place"`` with ``kind="rgb_composite"`` is refused, naming the kind.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with pytest.raises(ValueError, match="rgb_composite"):
            scene.animate(rgb_stack, kind="rgb_composite", update="in_place")

    def test_classes_cut_per_frame_block_the_in_place_path(self, stack):
        """``scheme=`` cuts its class edges from the frame being drawn, which a kept norm cannot follow.

        Test scenario:
            ``update="in_place"`` with ``scheme="quantiles"`` is refused, and the message names ``scheme``.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with pytest.raises(ValueError, match="scheme"):
            scene.animate(stack, scheme="quantiles", k=4, update="in_place", **CLIM)

    def test_auto_falls_back_to_a_redraw_naming_the_blocker(self, rgb_stack, caplog):
        """The default mode animates whatever it is given, and says which path it took.

        Test scenario:
            A composite animation under the default ``update="auto"`` renders both frames, and the log
            names the kind that cannot be updated in place rather than leaving the choice unexplained.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with caplog.at_level(logging.INFO, logger="digitalearth.static.maps.animation"):
            anim = scene.animate(rgb_stack, kind="rgb_composite", fps=2)
            anim._func(0)
            anim._func(1)
        assert "rgb_composite" in caplog.text, (
            f"the fallback must name what blocked it, logged: {caplog.text!r}"
        )
        assert len(scene.ax.images) == 1, (
            f"a redrawn composite leaves one image per frame, got {len(scene.ax.images)}"
        )


class TestBlit:
    """``blit=True`` is opt-in, and refused wherever it would freeze part of the frame."""

    def test_blitting_is_off_by_default(self, stack):
        """An animation nobody asked to blit does not blit.

        Test scenario:
            The returned animation's blit flag is off unless ``blit=True`` was passed.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        anim = scene.animate(stack, fps=2, **CLIM)
        assert anim._blit is False, "blitting must stay opt-in"

    def test_blitting_is_on_when_asked(self, stack):
        """``blit=True`` reaches the ``FuncAnimation``.

        Test scenario:
            The capability this row is about: the animation is built with blitting enabled.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        anim = scene.animate(stack, fps=2, blit=True, **CLIM)
        assert anim._blit is True, "blit=True must reach the FuncAnimation"

    def test_blit_is_refused_for_a_kind_that_cannot_update_in_place(self, stack):
        """Blitting a redrawn frame would leave the previous frame standing underneath.

        Test scenario:
            ``blit=True`` with ``kind="contourf"`` is refused, naming the kind.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with pytest.raises(ValueError, match="contourf"):
            scene.animate(stack, kind="contourf", blit=True, **CLIM)

    def test_blit_is_refused_with_per_frame_titles(self, stack):
        """A per-frame title lives outside the region blitting repaints, so it would freeze.

        Test scenario:
            ``blit=True`` together with ``titles=`` is refused, naming ``titles``.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with pytest.raises(ValueError, match="titles"):
            scene.animate(stack, titles=["a", "b", "c"], blit=True, **CLIM)

    def test_the_title_sits_outside_the_region_blitting_repaints(self, stack):
        """The measurement behind that refusal, rather than an assertion about matplotlib.

        Test scenario:
            matplotlib blits the axes bounding box. The title's drawn extent is measured and sits wholly
            above it, so nothing a blitted frame repaints can change the title.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        scene.field(stack[0], **CLIM)
        scene.set_title("January")
        scene.fig.canvas.draw()
        title = scene.ax.title.get_window_extent()
        assert title.y0 >= scene.ax.bbox.y1, (
            f"the title starts at y={title.y0} but the blitted axes box ends at {scene.ax.bbox.y1}"
        )

    def test_a_blitted_animation_saves_the_same_clip(self, stack, tmp_path):
        """Saving is unaffected: matplotlib draws every frame in full when writing a file.

        Test scenario:
            The same stack is written twice, with and without ``blit=True``, and the two files are
            compared byte for byte — the contract that ``blit=`` is a playback optimisation and never
            changes what is saved.
        """
        blitted, plain = tmp_path / "blit.gif", tmp_path / "plain.gif"
        with Map(crs=4326, figsize=(3, 3)) as scene:
            scene.animate(stack, fps=2, blit=True, **CLIM)
            scene.save_animation(str(blitted))
        with Map(crs=4326, figsize=(3, 3)) as scene:
            scene.animate(stack, fps=2, **CLIM)
            scene.save_animation(str(plain))
        assert blitted.stat().st_size > 0, "a blitted animation must still write a clip"
        assert blitted.read_bytes() == plain.read_bytes(), (
            "blitting must not change the frames that are written"
        )

    def test_rotate_refuses_to_blit(self, stack):
        """A rotation reprojects every frame, so no artist survives for blitting to leave alone.

        Test scenario:
            ``rotate(blit=True)`` is refused with a message naming the sweep, rather than the keyword
            reaching the renderer as unknown styling.
        """
        scene = Map(crs=4326, figsize=(3, 3))
        with pytest.raises(ValueError, match="reproject"):
            scene.rotate(stack[0], n_frames=3, blit=True)


class TestOffLimbFrames:
    """A frame the display CRS cannot show must draw nothing, not repeat the frame before it."""

    @pytest.fixture
    def half_hidden(self):
        """Two regional frames, the second of them at the antipode of the first.

        Returns:
            list[Dataset]: a visible frame followed by one no near-side orthographic view can show.
        """
        _, xx = np.mgrid[0:20, 0:24]
        values = (25 + 8 * np.sin(xx / 14.0)).astype("float32")
        return [
            Dataset.from_array(
                values,
                geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
                no_data_value=-9999.0,
            ),
            Dataset.from_array(
                values + 3.0,
                geo_ref=GeoReference(
                    geo=(-176.0, 0.02, 0.0, -53.0, 0.0, -0.02), epsg=4326
                ),
                no_data_value=-9999.0,
            ),
        ]

    def test_an_off_limb_frame_hides_the_image_rather_than_repeating_the_last(
        self, half_hidden
    ):
        """An un-drawable frame takes the kept image off the picture for that frame.

        Test scenario:
            A globe centred on the first frame animates a stack whose second frame is on the far side.
            Frame 0 shows the image; frame 1 hides it, which is what "draws nothing" means for an artist
            that is not rebuilt per frame.
        """
        scene = Map(
            crs=projections.orthographic(lon=4, lat=53), globe=True, figsize=(3, 3)
        )
        anim = scene.animate(half_hidden, fps=2, **CLIM)
        anim._func(0)
        assert scene.ax.images[0].get_visible() is True, (
            "the first frame is on the view and must be drawn"
        )
        anim._func(1)
        assert scene.ax.images[0].get_visible() is False, (
            "a frame behind the limb must not leave the previous frame on screen"
        )

    def test_a_clip_whose_first_frame_is_hidden_still_renders(
        self, half_hidden, tmp_path
    ):
        """With nothing drawn for frame 0 there is no artist to keep, so every frame is redrawn.

        Test scenario:
            The stack is animated from the antipode of its first frame, so frame 0 draws nothing. The
            animation must still write a clip rather than failing on the artist it never got.
        """
        scene = Map(
            crs=projections.orthographic(lon=-176, lat=-53), globe=True, figsize=(3, 3)
        )
        anim = scene.animate(half_hidden, fps=2, **CLIM)
        out = tmp_path / "hidden-first.gif"
        anim.save(str(out), writer=PillowWriter(fps=2))
        assert out.stat().st_size > 0, (
            "an animation whose first frame is hidden should still render"
        )


def _coarser(offset: float) -> Dataset:
    """The same global field on **half** the grid, so a stack can change shape between frames.

    Args:
        offset: Constant added to every cell, so the frame still differs by value as well as by shape.

    Returns:
        A 15x30 global EPSG:4326 raster — the same extent as :func:`_field`, at twice the cell size.
    """
    ny, nx = 15, 30
    lat = np.linspace(90, -90, ny)[:, None]
    plane = (np.cos(np.deg2rad(lat)) * 20 + offset) * np.ones((ny, nx), "float32")
    return Dataset.from_array(
        arr=plane.astype("float32"),
        geo_ref=GeoReference(geo=(-180.0, 12.0, 0.0, 90.0, 0.0, -12.0), epsg=4326),
    )


class TestAFrameOnAnotherGridFallsBackMidClip:
    """A stack whose frames are not all one shape: the in-place path gives up where it has to."""

    def test_the_later_frame_is_drawn_on_a_rebuilt_artist(self):
        """A frame on a coarser grid cannot refill the artist, so the artist is rebuilt for it.

        Test scenario:
            ``AxesImage.set_data`` with a differently shaped array leaves the image's extent describing
            the first frame's grid, so the picture would be stretched rather than redrawn. The strategy
            is chosen once, up front, from the *first* frame — nothing up there can know the third frame
            is a different shape — so the fallback has to happen mid-clip.
        """
        scene = Map(crs=4326)
        clip = scene.animate([_field(0.0), _coarser(8.0)], fps=2, **CLIM)
        first = clip._func(0)
        second = clip._func(1)
        scene.close()
        assert first[0] is not second[0], (
            "a frame on another grid must be drawn on a rebuilt artist, not the kept one"
        )

    def test_the_fallback_says_which_frame_changed_shape(self, caplog):
        """The switch is logged with both grids, so a slow clip is explained.

        Test scenario:
            The only other symptom is an animation that silently becomes as slow as the redrawing path.
            Naming the frame and the two shapes is what lets a caller fix the stack instead.
        """
        scene = Map(crs=4326)
        clip = scene.animate([_field(0.0), _coarser(8.0)], fps=2, **CLIM)
        clip._func(0)
        with caplog.at_level(logging.WARNING):
            clip._func(1)
        scene.close()
        assert (
            "holds a (15, 30) grid where the first frame held (30, 60)" in caplog.text
        ), f"the fallback should name both grids; log was {caplog.text!r}"

    def test_the_frames_after_the_change_keep_redrawing(self):
        """Once the strategy is dropped it stays dropped, rather than being retried per frame.

        Test scenario:
            Retrying would re-measure the mismatch on every remaining frame and log it again. Read off
            the artists: a third frame back on the *first* grid is still drawn on a new artist, because
            the kept one was given up two frames ago.
        """
        scene = Map(crs=4326)
        clip = scene.animate([_field(0.0), _coarser(8.0), _field(16.0)], fps=2, **CLIM)
        drawn = [clip._func(index)[0] for index in range(3)]
        scene.close()
        assert drawn[2] is not drawn[0], (
            "a frame after the fallback must be redrawn, not handed back the first artist"
        )

    def test_the_clip_still_renders_every_frame(self, tmp_path):
        """And the clip saves — the fallback is a slower path, not a failure.

        Args:
            tmp_path: Where the GIF is written.

        Test scenario:
            The mid-clip switch happens inside matplotlib's frame function, where an exception would be
            swallowed into a half-written file rather than reported. Writing the clip is what proves it
            was not.
        """
        scene = Map(crs=4326)
        clip = scene.animate([_field(0.0), _coarser(8.0)], fps=2, **CLIM)
        out = tmp_path / "mixed-grids.gif"
        clip.save(str(out), writer=PillowWriter(fps=2))
        scene.close()
        assert out.stat().st_size > 0, "a stack of mixed grids should still save a clip"


class TestAKindWithNoReasonOnRecord:
    """``FrameUpdate`` is built directly by the tiers, so it answers for a kind ``animate`` never passes."""

    def test_an_unlisted_kind_is_blocked_with_a_general_reason(self):
        """A kind in neither table is blocked rather than assumed updatable.

        Test scenario:
            ``animate`` validates ``kind`` against ``_ANIMATION_KINDS`` first, and every member of that
            tuple is either in ``SETTERS`` or in ``UNUPDATABLE`` — so this default is the answer for a
            kind added upstream before its row here, and the safe default is "cannot update in place".
        """
        scene = Map(crs=4326)
        update = FrameUpdate(scene, [_field(0.0)], {}, kind="hexbin")
        scene.close()
        assert (
            update.blocker
            == "kind='hexbin' draws no artist whose values can be replaced"
        ), f"an unlisted kind should be blocked by name; got {update.blocker!r}"

    def test_an_unlisted_kind_is_not_updated_in_place(self):
        """Which is to say the strategy reports itself unavailable.

        Test scenario:
            The blocker above is the message; this is the behaviour it stands for, so a change that kept
            the string and let the kind through fails here.
        """
        scene = Map(crs=4326)
        update = FrameUpdate(scene, [_field(0.0)], {}, kind="hexbin")
        scene.close()
        assert not update.in_place, (
            "a kind with no setter must not take the in-place path"
        )
