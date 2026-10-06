"""Animation export for a temporal web map (#194).

The frame enumeration and the GIF encoder are tested without a browser — the headless browser the frames
themselves need is an optional, deliberately un-declared dependency, so nothing here renders a page.

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env.
"""

import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

from digitalearth.web import WebMap
from digitalearth.web.export import _write_gif, _write_mp4


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


@pytest.fixture
def raster_stack(tmp_path):
    """A 3-member ``DatasetCollection``, one file per step.

    Args:
        tmp_path: pytest's per-test directory.

    Returns:
        The collection.
    """
    pytest.importorskip("pyramids")
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset
    from pyramids.dataset.collection import DatasetCollection

    geo_ref = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.02, epsg=4326)
    paths = []
    for step in range(3):
        _, xx = np.mgrid[0:20, 0:25]
        values = (10.0 * step + xx / 10.0).astype("float32")
        path = tmp_path / f"step{step}.tif"
        Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0).to_file(
            str(path)
        )
        paths.append(str(path))
    return DatasetCollection.from_files(paths)


@pytest.fixture
def points_with_time():
    """Two points carrying a time column, for the vector slider.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"time": pd.to_datetime(["2020-01-01", "2020-02-01"]), "v": [1, 2]},
        geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
        crs="EPSG:4326",
    )


@pytest.fixture
def frames(tmp_path):
    """Three distinct 8x8 PNGs standing in for rendered map frames.

    Args:
        tmp_path: pytest's per-test directory.

    Returns:
        The frame paths, in order.
    """
    paths = []
    for step in range(3):
        path = tmp_path / f"frame{step}.png"
        Image.fromarray(np.full((8, 8, 3), step * 100, dtype="uint8")).save(path)
        paths.append(str(path))
    return paths


class TestTheGifEncoder:
    """Pillow arrives with matplotlib, so an animation needs no dependency the tier lacks."""

    def test_every_frame_reaches_the_gif(self, frames, tmp_path):
        """A GIF holding one frame is a still image with extra steps.

        Args:
            frames: The synthetic frame paths.
            tmp_path: pytest's per-test directory.
        """
        out = tmp_path / "out.gif"
        _write_gif(frames, str(out), duration=0.5, loop=0)
        with Image.open(out) as animation:
            assert animation.n_frames == 3, animation.n_frames

    def test_frame_order_is_preserved(self, frames, tmp_path):
        """Time runs one way; a reordered animation is simply wrong.

        Args:
            frames: The synthetic frame paths, dark to light.
            tmp_path: pytest's per-test directory.
        """
        out = tmp_path / "out.gif"
        _write_gif(frames, str(out), duration=0.5, loop=0)
        seen = []
        with Image.open(out) as animation:
            for index in range(animation.n_frames):
                animation.seek(index)
                seen.append(int(np.asarray(animation.convert("RGB")).mean()))
        assert seen == sorted(seen), f"frames are out of order: {seen}"

    def test_the_frame_duration_is_carried(self, frames, tmp_path):
        """Otherwise every animation plays at the encoder's default speed.

        Args:
            frames: The synthetic frame paths.
            tmp_path: pytest's per-test directory.
        """
        out = tmp_path / "out.gif"
        _write_gif(frames, str(out), duration=0.25, loop=0)
        with Image.open(out) as animation:
            assert animation.info["duration"] == 250, animation.info

    def test_the_frames_are_closed(self, frames, tmp_path):
        """A still-open frame keeps a Windows file handle on the caller's temporary directory.

        Args:
            frames: The synthetic frame paths.
            tmp_path: pytest's per-test directory.

        Test scenario:
            ``animate`` renders its frames into a ``TemporaryDirectory``; a leaked handle makes that
            directory fail to delete with ``PermissionError`` on Windows.
        """
        out = tmp_path / "out.gif"
        _write_gif(frames, str(out), duration=0.5, loop=0)
        for frame in frames:
            pathlib.Path(
                frame
            ).unlink()  # raises PermissionError on Windows if still open

    def test_no_frames_is_refused(self, tmp_path):
        """An empty GIF is not a thing; the caller needs to know nothing was rendered.

        Args:
            tmp_path: pytest's per-test directory.
        """
        destination = str(tmp_path / "out.gif")
        with pytest.raises(ValueError, match="no frames"):
            _write_gif([], destination, duration=0.5, loop=0)


class TestTheMp4Encoder:
    """MP4 reuses the GIF's frame pipeline and swaps only the encoder (ffmpeg via imageio-ffmpeg)."""

    @pytest.fixture(autouse=True)
    def _need_encoder(self):
        """Skip when the ffmpeg encoder is absent (present in the web env, not in the bare dev env)."""
        pytest.importorskip("imageio_ffmpeg")

    def test_frames_become_a_real_mp4(self, frames, tmp_path):
        """The bytes have to be a real MP4 — an ``ftyp`` box — not a mislabelled GIF or empty file.

        Args:
            frames: The synthetic frame paths.
            tmp_path: pytest's per-test directory.

        Test scenario:
            ISO base-media (MP4) files open with an ``ftyp`` box in their first atom; its presence is the
            proof the encoder produced a genuine container rather than writing nothing or a wrong format.
        """
        out = tmp_path / "out.mp4"
        _write_mp4(frames, str(out), fps=3.0)
        data = out.read_bytes()
        assert b"ftyp" in data[:32], data[:32]

    def test_no_frames_is_refused(self, tmp_path):
        """An empty video is not a thing; the caller needs to know nothing was rendered.

        Args:
            tmp_path: pytest's per-test directory.
        """
        destination = str(tmp_path / "out.mp4")
        with pytest.raises(ValueError, match="no frames"):
            _write_mp4([], destination, fps=3.0)

    def test_the_frames_are_closed(self, frames, tmp_path):
        """A still-open frame keeps a Windows handle on the caller's temporary directory.

        Args:
            frames: The synthetic frame paths.
            tmp_path: pytest's per-test directory.

        Test scenario:
            ``save_animation`` renders its frames into a ``TemporaryDirectory``; a leaked handle makes
            that directory fail to delete with ``PermissionError`` on Windows.
        """
        out = tmp_path / "out.mp4"
        _write_mp4(frames, str(out), fps=3.0)
        for frame in frames:
            pathlib.Path(
                frame
            ).unlink()  # raises PermissionError on Windows if still open

    def test_a_missing_ffmpeg_encoder_is_refused(self, frames, tmp_path, monkeypatch):
        """Without ``imageio-ffmpeg`` the encoder is absent, so the caller gets an actionable ImportError.

        Args:
            frames: The synthetic frame paths, so the empty-frames guard is passed before the import.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's monkeypatch fixture.

        Test scenario:
            ``imageio-ffmpeg`` is an optional dependency ``digitalearth[web]`` does not pull. Blanking it in
            ``sys.modules`` makes the in-method ``import imageio_ffmpeg`` raise, and the message must name the
            missing package and the GIF fallback rather than failing obscurely inside ffmpeg.
        """
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
        destination = str(tmp_path / "out.mp4")
        with pytest.raises(ImportError, match="imageio-ffmpeg"):
            _write_mp4(frames, destination, fps=3.0)


class TestWhichStepsAreAnimated:
    """One frame per step, each showing that step alone."""

    def test_a_raster_series_yields_one_frame_per_step(self, raster_stack):
        """The stack's layers are the frames; anything else would animate the wrong thing.

        Args:
            raster_stack: The 3-member collection fixture.
        """
        m = WebMap().basemap().timeslider(raster_stack)
        assert m._temporal_frames() == [[i] for i in m._temporal["layer_ids"]]

    def test_a_map_with_no_series_says_what_to_add(self):
        """Nothing to animate is a caller error, and the message should name the fix."""
        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="timeslider"):
            web_map._temporal_frames()

    def test_the_vector_slider_is_named_as_unsupported(self, points_with_time):
        """Its steps are a filter over one layer, so they cannot be rendered separately.

        Args:
            points_with_time: A GeoDataFrame carrying a time column.
        """
        web_map = WebMap().basemap().timeslider(points_with_time, kdim="time")
        with pytest.raises(ValueError, match="vector time-slider"):
            web_map._temporal_frames()


class TestSaveDispatch:
    """``save`` routes by suffix, and a new one must not be read as HTML."""

    def test_a_gif_suffix_routes_to_the_animation(self, tmp_path, monkeypatch):
        """Before this, ``save("out.gif")`` silently wrote an HTML page into a .gif file.

        Args:
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.

        Test scenario:
            Patched on `save_animation`, which is the contract's name for it. `save` used to call the
            `animate` alias, so a caller who wrote `save()` was told to write `save_animation()` from a
            line inside the library (review M14).
        """
        called = {}

        def fake_gif(self, path, **kwargs):
            """Record that the animation path was taken."""
            called["path"] = path
            return str(path)

        monkeypatch.setattr(WebMap, "save_animation", fake_gif)
        out = tmp_path / "out.gif"
        WebMap().basemap().save(str(out))
        assert called["path"] == str(out), called

    def test_an_html_suffix_is_unaffected(self, tmp_path):
        """The common case must keep working exactly as it did.

        Args:
            tmp_path: pytest's per-test directory.
        """
        out = tmp_path / "out.html"
        WebMap().basemap().save(str(out))
        assert out.read_text(encoding="utf-8").lstrip().startswith("<")


class TestTheAnimationOrchestration:
    """`animate` drives the step visibility, renders each frame and encodes them.

    The screenshot itself needs a headless browser, which is deliberately not a declared dependency — so
    only that one step is faked here. Everything around it is the real code: the frame enumeration, the
    per-frame visibility, the temporary directory, and the encode.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _fake_renderer(seen):
        """Return a stand-in for ``_render_png`` that records its widget and writes a real PNG.

        Args:
            seen: A list the stand-in appends one record to per frame.

        Returns:
            A function with ``_render_png``'s signature.
        """

        def render(self, path, *, title="", **kwargs):
            """Write a 4x4 PNG and record which layers the handed-in widget left visible."""
            widget = kwargs.get("widget")
            calls = getattr(widget, "_visibility_calls", None)
            seen.append({"path": str(path), "title": title, "visibility": calls})
            shade = len(seen) * 40
            Image.fromarray(np.full((4, 4, 3), shade, dtype="uint8")).save(path)
            return str(path)

        return render

    @pytest.fixture
    def spy_widget(self, monkeypatch):
        """Make built widgets record their ``set_visibility`` calls.

        Args:
            monkeypatch: pytest's patcher.

        Returns:
            None; the recording is read off each widget as ``_visibility_calls``.
        """
        from digitalearth.web import WebMap

        original = WebMap._build_map_widget

        def build(self, *, with_controls=True):
            """Build the real widget, then wrap its visibility setter to record calls.

            Args:
                with_controls: Passed through; export paths build frames without the step picker.
            """
            widget = original(self, with_controls=with_controls)
            widget._visibility_calls = {}
            inner = widget.set_visibility

            def set_visibility(layer_id, visible=True):
                """Record the call, then apply it."""
                widget._visibility_calls[layer_id] = visible
                return inner(layer_id, visible)

            widget.set_visibility = set_visibility
            return widget

        monkeypatch.setattr(WebMap, "_build_map_widget", build)

    def test_it_writes_one_frame_per_step(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """Three steps must produce a three-frame GIF, not one frame or four.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.
        """
        from digitalearth.web import WebMap

        seen = []
        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer(seen))
        out = tmp_path / "series.gif"
        m = WebMap().basemap().timeslider(raster_stack)
        m.save_animation(str(out))

        assert len(seen) == 3, f"expected one frame per step, rendered {len(seen)}"
        with Image.open(out) as animation:
            assert animation.n_frames == 3, animation.n_frames

    def test_each_frame_shows_exactly_one_step(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """A frame showing two steps at once is the stack piled up, not an animation.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.
        """
        from digitalearth.web import WebMap

        seen = []
        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer(seen))
        m = WebMap().basemap().timeslider(raster_stack)
        steps = list(m._temporal["layer_ids"])
        m.save_animation(str(tmp_path / "series.gif"))

        for index, record in enumerate(seen):
            visible = [layer for layer, shown in record["visibility"].items() if shown]
            assert visible == [steps[index]], (
                f"frame {index} showed {visible}, expected only {steps[index]}"
            )

    def test_it_returns_the_path_and_leaves_no_temporary_files(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """The frames live in a TemporaryDirectory that has to be gone afterwards.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.

        Test scenario:
            An unclosed frame keeps a Windows handle on that directory, so cleanup failing is the
            symptom this guards.
        """
        from digitalearth.web import WebMap

        seen = []
        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer(seen))
        out = tmp_path / "series.gif"
        m = WebMap().basemap().timeslider(raster_stack)

        assert m.save_animation(str(out)) == pathlib.Path(out), (
            "save/animate return the pathlib.Path written (C1)"
        )
        assert out.exists()
        for record in seen:
            assert not pathlib.Path(record["path"]).exists(), (
                "a frame outlived its directory"
            )

    def test_the_title_reaches_every_frame(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """Each frame is its own rendered page, so the title has to be passed through per frame.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.
        """
        from digitalearth.web import WebMap

        seen = []
        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer(seen))
        m = WebMap().basemap().timeslider(raster_stack)
        m.save_animation(str(tmp_path / "series.gif"), title="Rainfall")

        assert {record["title"] for record in seen} == {"Rainfall"}, seen

    def test_save_with_a_gif_suffix_runs_the_real_animation(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """The dispatch test elsewhere fakes `animate`; this one runs it for real through `save`.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.
        """
        from digitalearth.web import WebMap

        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer([]))
        out = tmp_path / "viasave.gif"
        WebMap().basemap().timeslider(raster_stack).save(str(out))

        with Image.open(out) as animation:
            assert animation.n_frames == 3, animation.n_frames

    def test_an_mp4_suffix_encodes_a_real_mp4(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """``save_animation("x.mp4")`` runs the same frame pipeline and swaps to the MP4 encoder.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.

        Test scenario:
            The ``.mp4`` suffix must route past the GIF encoder to ffmpeg, so the file carries an MP4
            ``ftyp`` box rather than a ``GIF89a`` header.
        """
        pytest.importorskip("imageio_ffmpeg")
        from digitalearth.web import WebMap

        seen = []
        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer(seen))
        out = tmp_path / "series.mp4"
        m = WebMap().basemap().timeslider(raster_stack)
        assert m.save_animation(str(out)) == pathlib.Path(out)

        assert len(seen) == 3, f"expected one frame per step, rendered {len(seen)}"
        data = out.read_bytes()
        assert b"ftyp" in data[:32], data[:32]

    def test_format_mp4_overrides_a_gif_suffix(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """An explicit ``format="mp4"`` must win over the path's ``.gif`` suffix.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.

        Test scenario:
            A caller who names the format explicitly gets that encoder regardless of the filename, proving
            the switch is the format argument and not merely the suffix.
        """
        pytest.importorskip("imageio_ffmpeg")
        from digitalearth.web import WebMap

        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer([]))
        out = tmp_path / "despite-the-name.gif"
        WebMap().basemap().timeslider(raster_stack).save_animation(
            str(out), format="mp4"
        )
        data = out.read_bytes()
        assert b"ftyp" in data[:32], data[:32]

    def test_an_unknown_format_is_refused(
        self, raster_stack, tmp_path, monkeypatch, spy_widget
    ):
        """A format that is neither GIF nor MP4 is a caller error, named before anything is rendered.

        Args:
            raster_stack: The 3-member collection fixture.
            tmp_path: pytest's per-test directory.
            monkeypatch: pytest's patcher.
            spy_widget: Records each widget's visibility calls.
        """
        from digitalearth.web import WebMap

        rendered = []
        monkeypatch.setattr(WebMap, "_render_png", self._fake_renderer(rendered))
        m = WebMap().basemap().timeslider(raster_stack)
        destination = str(tmp_path / "series.webm")
        with pytest.raises(ValueError, match="gif.*mp4|mp4.*gif"):
            m.save_animation(destination, format="webm")
        assert rendered == [], (
            "nothing should be rendered before the format is validated"
        )
