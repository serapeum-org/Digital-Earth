"""Round-2 review M9/M10 — where ``WebMap.to_gif``'s warning lands, and what ``duration=`` may convert to.

Two defects in the one place the package converts a deprecated parameter rather than renaming it:

* **M9** — ``to_gif`` forwarded ``**kwargs`` to ``animate``, which resolved ``duration=`` itself. The shared
  helper's hand-counted ``stacklevel`` then landed the warning about the caller's keyword inside
  ``export.py``, a file the caller does not own. The assertion here is on the warning's **filename**, since a
  misattributed warning still satisfies ``pytest.warns``.
* **M10** — the conversion is a reciprocal, and ``nan`` slipped *both* positivity guards (``nan <= 0`` is
  ``False`` on either side of ``1 / nan``), so a frame rate of ``nan`` reached the encoder in silence.
"""

import pathlib
import warnings

import pytest

from digitalearth.web import WebMap

HERE = __file__


@pytest.fixture
def two_step_map(monkeypatch, tmp_path):
    """A ``WebMap`` whose animation renders two stub frames through a stub encoder.

    Args:
        monkeypatch: pytest's patcher, standing in for the browser and the GIF encoder.
        tmp_path: pytest's per-test directory.

    Returns:
        ``(map, out, recorded)`` — the map, the GIF path, and the dict the encoder records its hold in.
    """
    recorded: dict = {}

    def fake_write(frames, path, *, duration, loop):
        """Record the per-frame hold the encoder was handed."""
        recorded["duration"] = duration
        pathlib.Path(path).write_bytes(b"GIF89a")

    from digitalearth.web import export as web_export

    monkeypatch.setattr(web_export, "_write_gif", fake_write)
    monkeypatch.setattr(WebMap, "_temporal_frames", lambda self: [["a"], ["b"]])
    monkeypatch.setattr(WebMap, "_frame_png", lambda self, path, visible, title: path)
    return WebMap(), tmp_path / "series.gif", recorded


class TestTheAliasWarningLandsOnTheCallersLine:
    """M9 — a warning attributed to ``export.py`` names a line the caller cannot edit."""

    def test_duration_through_the_alias_is_attributed_to_the_caller(self, two_step_map):
        """``WebMap.to_gif(duration=...)`` must point at the user's file, not at ``export.py``.

        Args:
            two_step_map: The stubbed map, output path and encoder record.

        Test scenario:
            The alias added a frame the shared helper's ``stacklevel`` never counted, so the deprecation
            notice about the caller's own keyword pointed inside the package. ``duration=`` is now resolved
            in ``to_gif`` itself, one frame below the caller.
        """
        scene, out, _ = two_step_map
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene.to_gif(str(out), duration=0.5)
        by_message = {
            str(record.message).split(":")[0]: record
            for record in caught
            if issubclass(record.category, DeprecationWarning)
        }
        assert len(by_message) == 2, [str(record.message) for record in caught]
        for label, record in by_message.items():
            assert record.filename == HERE, f"{label} landed in {record.filename}"

    def test_the_alias_still_converts_the_rate_it_was_given(self, two_step_map):
        """Moving the resolution into the alias must not change what the old call renders.

        Args:
            two_step_map: The stubbed map, output path and encoder record.
        """
        scene, out, recorded = two_step_map
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert scene.to_gif(str(out), duration=0.5) == out
        assert recorded["duration"] == pytest.approx(0.5), recorded

    def test_the_alias_without_a_rate_uses_the_shared_default(self, two_step_map):
        """An old call naming no rate animates at the package default, as ``animate`` does.

        Args:
            two_step_map: The stubbed map, output path and encoder record.
        """
        from digitalearth.base.animation import DEFAULT_FPS

        scene, out, recorded = two_step_map
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene.to_gif(str(out))
        assert recorded["duration"] == pytest.approx(1.0 / DEFAULT_FPS), recorded

    def test_both_spellings_through_the_alias_is_still_refused(self, two_step_map):
        """``to_gif(fps=..., duration=...)`` names one rate twice, so it is a ``TypeError``.

        Args:
            two_step_map: The stubbed map, output path and encoder record.
        """
        scene, out, _ = two_step_map
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(TypeError, match="pass only fps="):
                scene.to_gif(str(out), fps=4.0, duration=0.5)


class TestANonFiniteRateIsRefusedByName:
    """M10 — a conversion that can produce a nonsense rate silently is worse than the parameter it replaced."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_duration_is_refused(self, bad, two_step_map):
        """``duration=nan`` passed both positivity guards and produced ``fps=nan``.

        Args:
            bad: A hold that is not a finite number of seconds.
            two_step_map: The stubbed map, output path and encoder record.

        Test scenario:
            ``nan <= 0`` is ``False``, so the guard in the converter let it through; ``1 / nan`` is ``nan``,
            so the guard on the resolved ``fps`` let it through too; and the encoder was handed a hold of
            ``nan`` seconds. The failure the caller eventually saw was about something else entirely.
        """
        scene, out, recorded = two_step_map
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as excinfo:
                scene.animate(str(out), duration=bad)
        assert "duration=" in str(excinfo.value), str(excinfo.value)
        assert "recorded" not in recorded, "the encoder must never be reached"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_fps_is_refused(self, bad, two_step_map):
        """The canonical spelling gets the same check, so neither route can reach the encoder with a ``nan``.

        Args:
            bad: A rate that is not a finite number.
            two_step_map: The stubbed map, output path and encoder record.
        """
        scene, out, _ = two_step_map
        with pytest.raises(ValueError) as excinfo:
            scene.animate(str(out), fps=bad)
        assert "fps=" in str(excinfo.value), str(excinfo.value)

    def test_the_converter_names_the_parameter_the_caller_wrote(self):
        """The message has to say ``duration=``, because that is the keyword in the caller's source."""
        from digitalearth.web.export import _fps_from_duration

        with pytest.raises(
            ValueError, match=r"duration= must be a finite number of seconds"
        ):
            _fps_from_duration(float("nan"))

    def test_a_finite_rate_still_converts(self):
        """The guard must not disturb the conversion it protects."""
        from digitalearth.web.export import _fps_from_duration

        assert _fps_from_duration(0.5) == pytest.approx(2.0)
