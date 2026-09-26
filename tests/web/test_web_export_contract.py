"""Round-2 review M10 — what a frame rate may be before it reaches the GIF encoder.

The web tier's animation entry point once took a seconds-per-frame ``duration=`` beside ``fps=`` and converted
one into the other. Both the conversion and the second spelling are gone; what the review found underneath
them is not, and is what this module pins:

* **M10** — the rate is used as a reciprocal on the way to the encoder, and ``nan`` slips an ordinary
  positivity guard (``nan <= 0`` is ``False``), so a frame rate of ``nan`` reached the encoder in silence. A
  non-finite rate is refused **by name**, before anything is rendered.
"""

import pathlib

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


class TestANonFiniteRateIsRefusedByName:
    """M10 — a rate that means nothing must not reach the encoder, and must be refused by its own name."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_fps_is_refused(self, bad, two_step_map):
        """``fps=nan`` passed the positivity guard and produced a hold of ``nan`` seconds.

        Args:
            bad: A rate that is not a finite number.
            two_step_map: The stubbed map, output path and encoder record.

        Test scenario:
            ``nan <= 0`` is ``False``, so a plain positivity check lets it through, and ``1 / nan`` is
            ``nan`` — so the encoder was handed a hold that means nothing. The failure the caller eventually
            saw was about something else entirely.
        """
        scene, out, recorded = two_step_map
        destination = str(out)
        with pytest.raises(ValueError) as excinfo:
            scene.save_animation(destination, fps=bad)
        assert "fps=" in str(excinfo.value), str(excinfo.value)
        assert "duration" not in recorded, "the encoder must never be reached"

    def test_a_finite_rate_still_reaches_the_encoder(self, two_step_map):
        """The guard must not disturb the rate it protects.

        Args:
            two_step_map: The stubbed map, output path and encoder record.

        Test scenario:
            A check that refused everything would satisfy the case above and animate nothing. The hold the
            encoder is handed is the reciprocal of the rate, which is the arithmetic the guard exists for.
        """
        scene, out, recorded = two_step_map
        assert scene.save_animation(str(out), fps=2.0) == out
        assert recorded["duration"] == pytest.approx(0.5), recorded
