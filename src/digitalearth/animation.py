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
import os
from typing import Any, Optional, Tuple, Union

from cleopatra.glyphs.base.animation import gif_from_video
from cleopatra.glyphs.base.animation import save_animation as _cleopatra_save_animation

#: Pixel format used for the intermediate video when a GIF is derived from it — full chroma, so the GIF
#: palette is built from unsubsampled colour (see the module docstring).
FULL_CHROMA_PIX_FMT = "yuv444p"

#: Frames per second used when neither the caller nor the scene supplies one.
DEFAULT_FPS = 12.0


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
        ValueError: if ``gif`` is given but ``path`` is itself a GIF, or if ``gif`` does not end in ``.gif``.

    Examples:
        - Rendering once and delivering two formats, without drawing the frames twice:
            ```python
            >>> from digitalearth.animation import save_animation
            >>> video, gif = save_animation(anim, "clip.mp4", fps=12, gif="clip.gif")  # doctest: +SKIP

            ```
    """
    rate = DEFAULT_FPS if fps is None else float(fps)
    video_path = os.fspath(path)
    if gif is None:
        _cleopatra_save_animation(anim, video_path, fps=int(round(rate)), **kwargs)
        return video_path

    gif_path = os.fspath(gif)
    if video_path.lower().endswith(".gif"):
        raise ValueError(
            "gif= derives a GIF from a video; path is already a GIF. Save to a video (e.g. .mp4) and let "
            "gif= produce the GIF, or drop gif= to write the GIF directly."
        )
    if not gif_path.lower().endswith(".gif"):
        raise ValueError(f"gif must end in '.gif', got {gif_path!r}")

    # Full chroma in the intermediate so the GIF palette is not built from subsampled colour.
    kwargs.setdefault("pix_fmt", FULL_CHROMA_PIX_FMT)
    _cleopatra_save_animation(anim, video_path, fps=int(round(rate)), **kwargs)
    gif_from_video(video_path, gif_path, **{"fps": rate, **(gif_options or {})})
    return video_path, gif_path


__all__ = ["save_animation", "FULL_CHROMA_PIX_FMT", "DEFAULT_FPS"]
