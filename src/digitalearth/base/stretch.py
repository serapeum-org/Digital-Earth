"""Contrast-stretch helpers shared by the backends' composite renders.

A composite (RGB or HSV) turns three bands into an image by stretching each channel into ``[0, 1]``, and
every backend that draws one needs the same three things: the per-channel bounds, the stretch itself, and
the argument checks around them. Keeping one copy here is what stops the tiers drifting — the static and
interactive backends had already grown two stretches that disagreed about an all-nodata channel.

Engine-neutral by construction: pure numpy, no renderer import (enforced by
``tests/test_base_is_engine_neutral.py``). The bounds are ordinary numbers, so a caller can compute them
once over a whole animation stack and hand the same pair to every frame — which is how a composite
time-lapse holds one stretch instead of pumping as each frame re-derives its own.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

from digitalearth.base.arrays import finite


#: Percentiles clipped off each channel by the default composite contrast stretch.
_STRETCH_PERCENTILES = (2, 98)

#: The three bands a composite maps to its channels when the caller names none. Defined here so the
#: renderers and the animation scan cannot drift apart about what a bare composite means.
DEFAULT_COMPOSITE_BANDS = (1, 2, 3)

#: One ``(lo, hi)`` stretch bound per composite channel, in channel order.
ChannelLimits = Sequence[Tuple[float, float]]


def require_three_bands(caller: str, bands: Sequence[int]) -> None:
    """Reject a composite band list that is not exactly three long.

    A composite maps its bands to three channels (R/G/B or H/S/V), so any other count is a caller mistake.
    Checked up front because the failure otherwise surfaces deep inside the renderer: from an animation it
    lands after the whole stack has been scanned, as a bare ``IndexError`` raised while matplotlib's writer
    is already unwinding, with the real complaint buried in a chained traceback.

    Args:
        caller: The method name to name in the message (e.g. ``"rgb_composite"``).
        bands: The band indices to check. ``None`` — what a wrapper forwarding an optional passes — is
            rejected by name rather than as a ``TypeError`` from trying to iterate it.

    Raises:
        ValueError: when ``bands`` is ``None`` or does not hold exactly three indices.

    Examples:
        - Three bands pass silently:
            ```python
            >>> from digitalearth.base.stretch import require_three_bands
            >>> require_three_bands("rgb_composite", (1, 2, 3)) is None
            True

            ```
        - Any other count names the caller, the count and the value it got:
            ```python
            >>> from digitalearth.base.stretch import require_three_bands
            >>> require_three_bands("rgb_composite", (1, 2))
            Traceback (most recent call last):
                ...
            ValueError: rgb_composite() needs exactly three bands, got 2: (1, 2)

            ```
        - A generator is consumed once, so it is materialised before it is counted:
            ```python
            >>> from digitalearth.base.stretch import require_three_bands
            >>> bands = (index for index in (1, 2, 3))
            >>> require_three_bands("rgb_composite", bands) is None
            True

            ```
    """
    if bands is None:
        raise ValueError(f"{caller}() needs exactly three bands, got None")
    named = tuple(bands)  # materialise once: a caller may hand over a one-shot iterable
    if len(named) != 3:
        raise ValueError(f"{caller}() needs exactly three bands, got {len(named)}: {named!r}")


def channel_limits(stack: np.ndarray) -> List[Tuple[float, float]]:
    """Return the per-channel 2-98 percentile ``(lo, hi)`` of an ``(rows, cols, n)`` stack.

    These are the bounds :func:`stretch_to_unit` derives on every call. Computing them **once** over a
    whole animation stack and passing them back in freezes the stretch, so a composite time-lapse does not
    pulse as each frame re-derives its own black and white point.

    A channel with no finite cell at all (fully nodata, or all ``inf``) has no percentile to take and
    yields ``(nan, nan)`` rather than a warning; callers treat that as "this channel contributes no
    bound" — see :meth:`~digitalearth.static.maps.animation.AnimationMixin._stack_channel_limits`, which
    drops it, and :func:`stretch_to_unit`, which falls back rather than dividing by it.

    Args:
        stack: An ``(rows, cols, n)`` channel stack (nodata already NaN).

    Returns:
        One ``(lo, hi)`` tuple per channel, in channel order; ``(nan, nan)`` for an all-nodata channel.

    Raises:
        ValueError: when ``stack`` is not 3-D.

    Examples:
        - Bound each channel of a stack whose channels are scaled copies of one another:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.stretch import channel_limits
            >>> stack = np.dstack([np.arange(100.0).reshape(10, 10) * scale for scale in (1, 2, 3)])
            >>> limits = channel_limits(stack)
            >>> len(limits)
            3
            >>> [round(hi) for _, hi in limits]
            [97, 194, 291]

            ```
        - A channel with no finite cell reports ``(nan, nan)`` rather than raising or warning, and
          leaves its neighbours' bounds intact:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.stretch import channel_limits
            >>> stack = np.dstack([np.arange(16.0).reshape(4, 4), np.full((4, 4), np.nan), np.ones((4, 4))])
            >>> limits = channel_limits(stack)
            >>> limits[1]
            (nan, nan)
            >>> round(limits[0][0], 1)
            0.3

            ```

    See Also:
        stretch_to_unit: Applies these bounds (or derives its own when none are given).
        digitalearth.static.maps.animation.AnimationMixin._stack_channel_limits: Combines them across a
            whole animation stack so every frame shares one stretch.
    """
    if stack.ndim != 3:
        raise ValueError(
            f"channel_limits needs an (rows, cols, n) channel stack, got a {stack.ndim}-D array "
            f"with shape {stack.shape}"
        )
    bounds: List[Tuple[float, float]] = []
    for index in range(stack.shape[2]):
        values = finite(stack[..., index])  # drops NaN *and* inf, which nanpercentile would keep
        if values.size == 0:
            bounds.append((float("nan"), float("nan")))
            continue
        # finite() already returned a fresh array, so numpy may sort it in place instead of copying
        # again — the scan reads three channels per frame, up to _CLIM_SCAN_CAP frames.
        lo, hi = np.percentile(values, _STRETCH_PERCENTILES, overwrite_input=True)
        bounds.append((float(lo), float(hi)))
    return bounds


def _check_limits(limits: Optional[ChannelLimits], channels: int) -> None:
    """Reject a ``limits`` argument that is not one ``(lo, hi)`` pair per channel.

    Both composites take ``limits`` from the caller, so the shapes people actually get wrong are worth
    naming: too few pairs used to raise ``IndexError``, a bare ``(lo, hi)`` tuple ``TypeError: cannot unpack
    non-iterable float``, and too many were accepted silently — the worst of the three, since it hides a
    mismatch between the caller's band list and their limits list.

    Args:
        limits: The caller's limits, or ``None`` (which is always valid — it means "derive them").
        channels: How many channels the stack holds.

    Raises:
        ValueError: when ``limits`` is not ``None`` and does not hold exactly ``channels`` ``(lo, hi)`` pairs.
    """
    if limits is None:
        return
    if len(limits) != channels:
        raise ValueError(
            f"limits has {len(limits)} entries but the stack has {channels} channels; pass one (lo, hi) "
            f"pair per channel"
        )
    for index, pair in enumerate(limits):
        if np.shape(pair) != (2,):
            raise ValueError(f"limits[{index}] must be a (lo, hi) pair, got {pair!r}")


def stretch_to_unit(stack: np.ndarray, limits: Optional[ChannelLimits] = None) -> np.ndarray:
    """Per-channel contrast stretch of an ``(rows, cols, n)`` stack into ``[0, 1]``.

    Args:
        stack: The ``(rows, cols, n)`` channel stack to stretch.
        limits: Optional precomputed ``(lo, hi)`` per channel, in channel order. When omitted each channel
            is stretched to its **own** 2-98 percentile — right for a still, but it makes the brightness
            pump across an animation, since every frame gets its own black and white point. Pass frozen
            limits (see :func:`channel_limits`) to hold one stretch across a sequence of frames. A
            non-finite entry means "no frozen bound for this channel": that one channel falls back to this
            frame's own percentile, so a channel the freeze could not measure degrades to the per-frame
            behaviour rather than clipping flat against an invented span.

    Returns:
        The stretched stack: same shape, ``float64``, clipped into ``[0, 1]``.

    Raises:
        ValueError: when ``limits`` is given but does not hold one ``(lo, hi)`` pair per channel. A silently
            ignored extra pair would hide a real mismatch between the caller's bands and their limits.

    Examples:
        - Each channel spans the full range on its own percentiles:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.stretch import stretch_to_unit
            >>> stack = np.dstack([np.arange(100.0).reshape(10, 10) * scale for scale in (1, 2, 3)])
            >>> out = stretch_to_unit(stack)
            >>> float(out.min()), float(out.max())
            (0.0, 1.0)
            >>> [round(float(out[..., i].mean()), 3) for i in range(3)]
            [0.5, 0.5, 0.5]

            ```
        - One shared set of limits keeps a real brightness difference visible, which is what holds an
          animation steady; each channel's own percentiles would erase it:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.stretch import channel_limits, stretch_to_unit
            >>> bright = np.dstack([np.arange(100.0).reshape(10, 10) for _ in range(3)])
            >>> dim = bright * 0.4
            >>> shared = channel_limits(bright)
            >>> round(float(stretch_to_unit(bright, shared).mean()), 3)
            0.5
            >>> round(float(stretch_to_unit(dim, shared).mean()), 3)
            0.188
            >>> round(float(stretch_to_unit(dim).mean()), 3)
            0.5

            ```
        - A channel with no bound to freeze falls back to that frame's own, rather than an invented span:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.stretch import stretch_to_unit
            >>> stack = np.dstack([np.full((4, 4), 500.0) for _ in range(3)])
            >>> out = stretch_to_unit(stack, [(float("nan"), float("nan"))] * 3)
            >>> float(out.max())
            0.0

            ```
    """
    _check_limits(limits, stack.shape[2])
    out = np.empty(stack.shape, dtype="float64")
    derived = channel_limits(stack) if limits is None else limits
    for i in range(stack.shape[2]):
        band = stack[..., i].astype("float64")
        lo, hi = derived[i]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            # No frozen bound for this channel — fall back to what this frame alone says rather than to a
            # fixed span, which would clip a live channel flat if the freeze simply never saw it (M2).
            lo, hi = channel_limits(band[..., None])[0]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = 0.0, 1.0  # this frame's channel is nodata too: any span, its cells stay NaN
        elif hi <= lo:
            hi = lo + 1.0  # a constant channel: widen rather than divide by zero
        out[..., i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out
