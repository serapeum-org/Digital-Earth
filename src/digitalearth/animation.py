"""Saving animations — write a clip once, derive every other format from that file.

Drawing frames is far more expensive than encoding them: a long scientific animation can take hours to render
and seconds to encode. cleopatra 0.33 added ``gif_from_video``, which reads frames back off an already-written
video instead of re-rendering them, so a clip can be rendered **once** and delivered as both a video and a GIF.

:func:`save_animation` is the Digital-Earth wrapper over that pair. It is shared by
:class:`~digitalearth.scene.maps.animation.AnimationMixin` (``Map.animate`` / ``Map.rotate``) and
:class:`~digitalearth.scene.textured_globe.TexturedGlobe`, so every animated scene saves the same way.

The one piece of real judgement here is the intermediate's pixel format. cleopatra's ``save_animation``
defaults to ``pix_fmt="yuv420p"`` for universal playback, but that halves the colour resolution before the GIF
palette ever sees the frames — and ``gif_from_video`` warns when handed such a file. So when a GIF is being
derived, this wrapper raises the intermediate to ``yuv444p`` unless the caller asks for something else: the
video stays perfectly playable and the GIF gets full-chroma frames to quantise.
"""
import math
import os
from typing import Any, Optional, Tuple, Union

from cleopatra.glyphs.base.animation import gif_from_video
from cleopatra.glyphs.base.animation import save_animation as _cleopatra_save_animation

#: Pixel format used for the intermediate video when a GIF is derived from it — full chroma, so the GIF
#: palette is built from unsubsampled colour (see the module docstring).
FULL_CHROMA_PIX_FMT = "yuv444p"

#: Frames per second used when neither the caller nor the scene supplies one.
DEFAULT_FPS = 12.0

#: Container suffixes a GIF can be derived from. Anything else would reach ffmpeg as an
#: intermediate it cannot decode, and fail there rather than here.
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".webp", ".mkv", ".m4v")


def _encoder_fps(fps: Any) -> int:
    """Round a frame rate to the whole number both encoders will actually use.

    cleopatra's ``save_animation`` types ``fps`` as an ``int``, so a fractional rate has to be rounded
    somewhere. Rounding it only for the video and handing the raw float to ``gif_from_video`` is the trap:
    ``fps=2.5`` then wrote a 2 fps video beside a 2.5 fps GIF, two files of the same animation playing at
    different speeds. Rounding once, here, keeps them identical.

    Args:
        fps: The requested frame rate.

    Returns:
        The frame rate as a positive whole number of frames per second.

    Raises:
        ValueError: If ``fps`` is not a finite number, or rounds to less than one frame per second — a rate
            below 0.5 would otherwise silently become ``fps=0``, which no encoder can use.
    """
    try:
        rate = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fps must be a number, got {fps!r}") from exc
    if not math.isfinite(rate):
        raise ValueError(f"fps must be finite, got {fps!r}")
    rounded = int(round(rate))
    if rounded < 1:
        raise ValueError(
            f"fps={fps!r} rounds to {rounded} frames per second, which no encoder accepts. Pass fps >= 0.5, "
            "or slow the animation down with more frames instead of a fractional rate."
        )
    return rounded


def save_animation(anim: Any, path: Union[str, "os.PathLike[str]"], *, fps: Optional[float] = None,
                   gif: Optional[Union[str, "os.PathLike[str]"]] = None,
                   gif_options: Optional[dict] = None, **kwargs: Any) -> Union[str, Tuple[str, str]]:
    """Save ``anim`` to ``path``, optionally deriving a GIF from the written file without re-rendering.

    Args:
        anim: The matplotlib ``FuncAnimation`` to save.
        path: Output path. The extension picks the format — ``gif``, ``mp4``, ``mov``, ``avi`` or ``webp``.
        fps: Frames per second. Defaults to :data:`DEFAULT_FPS`; callers that know the scene's own rate
            (``Map.animate(fps=...)``) pass it through so the file matches what was previewed.
        gif: When given, a second output path to derive a GIF at, by reading the frames back off ``path``
            instead of re-rendering them. ``path`` must be a video for this — deriving a GIF from a GIF is
            pointless and raises.
        gif_options: Extra keyword arguments for ``gif_from_video`` (``width``, ``max_colors``, ``loop``,
            ``optimize``, ``quantize_method``). ``fps`` defaults to the same rate as the video.
        **kwargs: Forwarded to cleopatra's ``save_animation`` (``crf``, ``bitrate``, ``codec``, ``preset``,
            ``pix_fmt``, ``dpi``, ``optimize``, ``loop``, ``quantize_method``, ``extra_args``).

    Returns:
        The written path as a ``str``, or a ``(video, gif)`` pair when ``gif`` was requested.

    Raises:
        ValueError: if ``gif`` is given but ``path`` is itself a GIF, if ``gif`` does not end in ``.gif``, or
            if ``fps`` is not a finite number or rounds to less than one frame per second.

    Note:
        The rate is rounded to a whole number **once**, and the same value is used for the video and for the
        derived GIF, so the two files always play at the same speed.

    Examples:
        - Rendering once and delivering two formats, without drawing the frames twice:
            ```python
            >>> from digitalearth.animation import save_animation
            >>> video, gif = save_animation(anim, "clip.mp4", fps=12, gif="clip.gif")  # doctest: +SKIP

            ```
    """
    rate = _encoder_fps(DEFAULT_FPS if fps is None else fps)
    video_path = os.fspath(path)
    if gif is None:
        _cleopatra_save_animation(anim, video_path, fps=rate, **kwargs)
        return video_path

    gif_path = os.fspath(gif)
    if video_path.lower().endswith(".gif"):
        raise ValueError(
            "gif= derives a GIF from a video; path is already a GIF. Save to a video (e.g. .mp4) and let "
            "gif= produce the GIF, or drop gif= to write the GIF directly."
        )
    if not video_path.lower().endswith(VIDEO_SUFFIXES):
        raise ValueError(
            f"gif= reads the frames back off a video, so path must be one of {', '.join(VIDEO_SUFFIXES)}; "
            f"got {video_path!r}."
        )
    if not gif_path.lower().endswith(".gif"):
        raise ValueError(f"gif must end in '.gif', got {gif_path!r}")

    # Full chroma in the intermediate so the GIF palette is not built from subsampled colour.
    kwargs.setdefault("pix_fmt", FULL_CHROMA_PIX_FMT)
    _cleopatra_save_animation(anim, video_path, fps=rate, **kwargs)
    gif_from_video(video_path, gif_path, **{"fps": rate, **(gif_options or {})})
    return video_path, gif_path


__all__ = ["save_animation", "FULL_CHROMA_PIX_FMT", "DEFAULT_FPS", "VIDEO_SUFFIXES"]
