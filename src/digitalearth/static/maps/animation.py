"""AnimationMixin — animate a stack of rasters and rotate an orthographic globe.

Drives per-frame redraws on the shared axes as a matplotlib ``FuncAnimation``, with one shared colour
treatment so colours do not flicker between frames: a scalar field gets one ``vmin``/``vmax`` (and an
optional single static colorbar), an RGB/HSV composite gets one frozen per-channel contrast stretch.
"""

import logging
from math import isfinite
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple

from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from digitalearth.base.animation import DEFAULT_FPS
from digitalearth.base.arrays import read_masked_band
from digitalearth.base.autostyle import auto_style
from digitalearth.base.clim import (
    DEFAULT_CLIM_SCAN_CAP,
    measure_clim,
    sample_evenly,
)
from digitalearth.base.sources import get_source, get_stack
from digitalearth.base.stretch import (
    DEFAULT_COMPOSITE_BANDS,
    ChannelLimits,
    channel_limits,
    require_three_bands,
)
from digitalearth.static import projections
from digitalearth.static.animation import save_animation
from digitalearth.static.maps.base import OffLimbError
from digitalearth.static.maps.raster import DEFAULT_FIELD_CMAP

logger = logging.getLogger(__name__)

# The cap and the sampling rule are `digitalearth.base.clim`'s: one number and one strategy shared with the
# interactive and web tiers, so the same collection is read at the same members, the same way, whichever
# tier draws it. The *values* can still differ -- each tier warps to its own display CRS first.
_CLIM_SCAN_CAP = DEFAULT_CLIM_SCAN_CAP

#: What "this frame cannot be read" looks like to :meth:`AnimationMixin._frame_style`, which moves on to the
#: next frame rather than failing the colorbar. A stack member that is not a plottable input at all is
#: refused by type (``TypeError``, from the ``get_source`` dispatch); a frame that has no such band — or a
#: NetCDF frame with no variable to plot — is refused by value (``ValueError``, from pyramids); and a frame
#: whose bytes cannot be fetched surfaces as the I/O failure the reader reports (``OSError``, or the
#: ``RuntimeError`` GDAL raises with exceptions enabled, which is what a corrupt or unreachable remote frame
#: looks like). Anything else — an ``AttributeError`` from a typo in the style lookup, say — is a bug in
#: this package, not an unreadable frame, and must reach the caller instead of being reported as "no style".
UNREADABLE_FRAME = (OSError, RuntimeError, TypeError, ValueError)

# `DEFAULT_FPS` is imported above rather than declared here: the rate every tier's animation entry point
# defaults to lives in `digitalearth.base.animation`, so `animate`/`rotate` play a clip built with defaults at
# the same speed as the interactive, 3-D and web tiers. It stays importable from this module because that is
# where this tier's callers and tests already reach for it. Not to be confused with
# `digitalearth.static.animation.FALLBACK_SAVE_FPS`, the rate the *encoder* assumes for a clip of unknown rate.


def _scan_subset(datasets: Sequence[Any]) -> List[Any]:
    """Return at most :data:`_CLIM_SCAN_CAP` evenly-spaced frames of ``datasets``.

    Both stack scans — the scalar clim and the composite stretch — sample rather than read every frame, and
    they must sample the same way; deferring to :func:`~digitalearth.base.clim.sample_evenly` is what
    guarantees that, and what makes this tier agree with the other two.

    Args:
        datasets: The animation stack.

    Returns:
        At most :data:`_CLIM_SCAN_CAP` frames, spread across the whole stack and always including its
        first and last.
    """
    return sample_evenly(datasets, cap=_CLIM_SCAN_CAP)


def _as_frames(stack: Any) -> List[Any]:
    """Return an animation stack as a list of pyramids ``Dataset`` frames.

    A ``DatasetCollection`` is documented as an accepted stack, but iterating one yields the members'
    **numpy arrays** rather than the ``Dataset`` objects themselves, so everything downstream that reads a
    frame's CRS or bands broke on it. The collection exposes its members under ``.datasets``, which is how
    the rest of the package reads one; anything else (a list, a tuple) is already a sequence of frames.
    The check is by attribute rather than by type so a collection-like object works too — nothing in
    the package exposes a non-sequence ``.datasets``, and a caller that did would be handing over
    something that is not a stack in the first place.

    Args:
        stack: A ``DatasetCollection``, or any ordered collection of ``Dataset`` frames.

    Returns:
        The frames as a list.
    """
    return list(getattr(stack, "datasets", stack))


def _union_channel_limits(
    scanned: Sequence[Sequence[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Combine per-scan channel bounds into the widest ``(lo, hi)`` each channel takes.

    Non-finite bounds are dropped rather than folded in: ``min``/``max`` against ``nan`` returns whichever
    argument came first, so a scan that could not measure a channel would otherwise blank or keep it
    depending purely on where it sat in the sequence. A channel no scan could measure reports
    ``(nan, nan)`` — "no frozen bound" — which the stretch answers per frame.

    Args:
        scanned: One per-channel bound list per scanned frame or view.

    Returns:
        One ``(lo, hi)`` tuple per channel, in channel order.
    """
    limits: List[Tuple[float, float]] = []
    for index in range(len(scanned[0])):
        lows = [bounds[index][0] for bounds in scanned if isfinite(bounds[index][0])]
        highs = [bounds[index][1] for bounds in scanned if isfinite(bounds[index][1])]
        limits.append(
            (min(lows), max(highs)) if lows and highs else (float("nan"), float("nan"))
        )
    return limits


#: Composite renderers accepted as an animation ``kind``. They draw an RGB image rather than a scalar
#: field, so they take a frozen per-channel stretch instead of a clim, and admit no colorbar.
_COMPOSITE_KINDS = ("rgb_composite", "hsv_composite")

#: Render methods accepted as the ``kind`` of an animation frame (validated up front, N1).
_ANIMATION_KINDS = (
    "field",
    "imshow",
    "contourf",
    "contour",
    "pcolormesh",
    "block",
) + _COMPOSITE_KINDS

#: Where the method that draws a ``kind`` differs from the kind itself, as ``{kind: (method, keywords)}``.
#:
#: A ``kind`` is the *renderer* a caller names, and order 27a moved the methods it dispatches to — so
#: dispatching on the kind alone would reach a method that has been renamed, and fail from inside the
#: never wrote. The old kinds stay in the vocabulary because they are what a caller may already have
#: written; the keywords are what tells ``contour`` and ``contourf`` apart now that they are one method.
_KIND_METHODS = {
    "imshow": ("field", {}),
    "contour": ("contours", {"filled": False}),
    "contourf": ("contours", {"filled": True}),
}


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _animated_band(opts: dict) -> int:
    """Read the band an animation will draw from its render options.

    Checked here rather than at the first rendered frame: bands are counted from 1, and passing 0 used to
    reach pyramids as the 0-based band -1, which reported a band the caller never asked for.

    Args:
        opts: The render options the frames will be drawn with.

    Returns:
        The 1-based band index, defaulting to the first band.

    Raises:
        ValueError: if ``band`` is not a whole number of 1 or more.

    Examples:
        - The first band is the default:
            ```python
            >>> from digitalearth.static.maps.animation import _animated_band
            >>> _animated_band({}), _animated_band({"band": 3})
            (1, 3)

            ```
        - Counting from zero is refused, naming what was passed:
            ```python
            >>> from digitalearth.static.maps.animation import _animated_band
            >>> _animated_band({"band": 0})
            Traceback (most recent call last):
                ...
            ValueError: band must be a whole number of 1 or more, got 0

            ```
    """
    asked = opts.get("band", 1)
    try:
        band = int(asked)
    except (TypeError, ValueError):
        band = 0
    if band < 1 or band != asked:
        raise ValueError(f"band must be a whole number of 1 or more, got {asked!r}")
    return band


class AnimationMixin(_MixinBase):
    """Stack animation and globe rotation for :class:`~digitalearth.static.map.Map`."""

    def _animate_frames(
        self, draw_one: Any, n_frames: int, fps: float
    ) -> FuncAnimation:
        """Drive ``n_frames`` of ``draw_one(i)`` on this Map's axes as a :class:`FuncAnimation`.

        Each frame clears the axes and resets the per-frame layer/frame state, calls ``draw_one(i)`` to draw
        frame ``i``, then (on a globe) sets the full-domain extent and applies the projection frame. No
        colorbar is added per frame — pass a fixed ``vmin``/``vmax`` to keep colours stable instead.
        """

        def _f(i: int) -> None:
            self.ax.clear()
            self._reset_layers()
            self._framed = False
            draw_one(i)
            if self.globe:
                self.set_global()
                self._apply_frame()

        anim = FuncAnimation(
            self.fig, _f, frames=n_frames, interval=1000.0 / fps, blit=False
        )
        self._animation = anim  # keep a strong reference so it isn't garbage-collected before save (L3)
        self._animation_fps = float(
            fps
        )  # so save_animation writes at the rate the scene was built for
        return anim

    def save_animation(
        self,
        path: str,
        *,
        fps: Optional[float] = None,
        gif: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Save the animation built by :meth:`animate` / :meth:`rotate`, optionally also deriving a GIF.

        Passing ``gif`` draws the frames **once**: the clip is encoded to ``path``, then the GIF is derived
        by reading those frames back off that file rather than redrawing every one. On a long animation that
        roughly halves the wall-clock cost of shipping both formats.

        Args:
            path: Output path; the extension picks the format (``mp4`` / ``mov`` / ``avi`` / ``webp`` /
                ``gif``).
            fps: Frames per second. Defaults to the rate the animation was built with.
            gif: Optional second path to derive a GIF at. Requires ``path`` to be a video.
            **kwargs: Forwarded to :func:`digitalearth.static.animation.save_animation` (and on to cleopatra) —
                ``crf``, ``bitrate``, ``codec``, ``dpi``, ``gif_options``, and the rest.

        Returns:
            The written path, or a ``(video, gif)`` pair when ``gif`` was requested.

        Raises:
            RuntimeError: if no animation has been built yet — call :meth:`animate` or :meth:`rotate` first.

        Examples:
            - Animate a stack, then write it straight to a GIF at the rate it was built with:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import tempfile
                >>> from pathlib import Path
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> ref = GeoReference(geo=geo, epsg=4326)
                >>> frames = [Dataset.from_array(arr=np.full((4, 4), v, dtype="float32"), geo_ref=ref)
                ...           for v in (1.0, 2.0)]
                >>> m = Map(crs=4326)
                >>> anim = m.animate(frames, fps=2.0)
                >>> out = Path(tempfile.mkdtemp()) / "clip.gif"
                >>> written = m.save_animation(str(out))
                >>> Path(written).exists()
                True

                ```
            - Render once and deliver both a video and a GIF derived from that file:
                ```python
                >>> video, gif = m.save_animation("clip.mp4", gif="clip.gif")   # doctest: +SKIP

                ```
            - Saving before animating is refused, rather than writing an empty clip:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> Map(crs=4326).save_animation("clip.gif")
                Traceback (most recent call last):
                    ...
                RuntimeError: no animation to save; call animate() or rotate() first

                ```
        """
        anim = self._animation
        if anim is None:
            raise RuntimeError("no animation to save; call animate() or rotate() first")
        rate = self._animation_fps if fps is None else fps
        return save_animation(anim, path, fps=rate, gif=gif, **kwargs)

    def _stack_clim(
        self, datasets: Sequence[Any], band: int = 1
    ) -> Tuple[float, float]:
        """Return the ``(min, max)`` of ``band`` across ``datasets``, ignoring nodata/non-finite.

        Each frame is reprojected to the display CRS before it is measured, for the same reason the
        composite scan does it: the frame renders warped, so the stored values are not the ones on screen.
        The effect is far smaller here than on a composite — the clim is shared across frames either way, so
        nothing flickers, and only the animation-vs-still comparison moves — but the two scans measuring
        different things is exactly the drift worth not having.

        Args:
            datasets: The frames to measure (already sampled by :func:`_scan_subset`).
            band: 1-based index of the band being animated — the one whose range must set the scale.

        Returns:
            The ``(min, max)`` across them, or ``(0, 1)`` when no frame holds a finite value —
            which includes the case where no frame is on the view at all, since a frame that
            cannot be warped draws nothing and so contributes no colour range.
        """
        measured = self._measured_clim(datasets, band=band)
        return measured if measured is not None else (0.0, 1.0)

    def _measured_clim(
        self, datasets: Sequence[Any], band: int = 1
    ) -> Optional[Tuple[float, float]]:
        """Return the ``(min, max)`` actually measured across ``datasets``, or ``None`` if nothing was.

        The difference from :meth:`_stack_clim` matters to :meth:`_clim_across_views`, which unions one
        result per swept projection: a view showing none of the data must contribute *nothing*, not the
        ``(0, 1)`` placeholder, or the union floors at zero and the real data collapses into the top of
        the colour ramp.

        Args:
            datasets: The frames to measure.
            band: 1-based index of the band to read from each frame.

        Returns:
            The ``(min, max)`` across every frame that could be warped and held a finite value, or
            ``None`` when no frame did.
        """
        return measure_clim(self._frame_values(datasets, band=band))

    def _frame_values(self, datasets: Sequence[Any], band: int = 1) -> Iterator[Any]:
        """Yield the display-CRS values of ``band`` for each frame that can be warped onto the view.

        The engine-specific half of the stack scan: warping and band reading are this tier's business, while
        reducing the arrays to one range is :mod:`digitalearth.base.clim`'s.

        Args:
            datasets: The frames to read.
            band: 1-based index of the band to read from each frame.

        Yields:
            One array per readable frame. A frame that warps to nothing is skipped rather than yielded, so it
            contributes no colour range.

        Note:
            **Must be drained before the display CRS moves.** Each frame is warped at the moment it is
            pulled, against whatever ``self.crs`` is then — and :meth:`_clim_across_views` reassigns that in
            a loop. Today every caller drains it synchronously inside
            :func:`~digitalearth.base.clim.measure_clim`, so the frames are warped under the intended view;
            storing or chaining the generator would silently measure the wrong projection.
        """
        for ds in datasets:
            try:
                warped = self._reproject(ds)
            except OffLimbError:
                continue  # this frame draws nothing, so it contributes no colour range
            yield read_masked_band(warped, band=band)

    def _clim_across_views(
        self, dataset: Any, views: Sequence[Any], band: int = 1
    ) -> Tuple[float, float]:
        """Return the ``(min, max)`` of one dataset measured under each sampled display CRS.

        The scalar counterpart of :meth:`_scan_across_views`: :meth:`rotate` redraws one dataset under a
        sweep of projections, so a scale taken from the CRS in force at build time would describe a single
        hemisphere and then be applied to all of them. The display CRS is restored afterwards.

        Args:
            dataset: The raster every frame draws.
            views: The display CRSs the animation will sweep; sampled by :func:`_scan_subset`.
            band: 1-based index of the band being animated.

        Returns:
            The widest ``(min, max)`` across the sampled views, or ``(0, 1)`` when no view shows any of the
            data — a full sweep passes the far side of the globe, where the warp has nothing to transform.
        """
        original = self.crs
        try:
            bounds = []
            for view in _scan_subset(views):
                self.crs = view
                measured = self._measured_clim([dataset], band=band)
                if measured is None:
                    continue  # this view shows none of the data, so it bounds nothing
                bounds.append(measured)
            return (
                (min(lo for lo, _ in bounds), max(hi for _, hi in bounds))
                if bounds
                else (0.0, 1.0)
            )
        finally:
            self.crs = original

    def _resolve_animation_clim(
        self,
        datasets: Sequence[Any],
        opts: dict,
        *,
        views: Optional[Sequence[Any]] = None,
    ) -> None:
        """Ensure ``opts`` carries a shared ``vmin``/``vmax`` so every animation frame uses one colour scale.

        Without this, each frame's renderer auto-scales to its own data range, so the colours (and any
        colorbar) flicker between frames. Any ``vmin``/``vmax`` already in ``opts`` is kept; a missing bound
        is filled once from the stack (ignoring nodata/non-finite) and written back, so all frames — and the
        colorbar — share it. Passing both ``vmin`` and ``vmax`` skips the scan entirely. To bound the cost on
        large stacks, at most :data:`_CLIM_SCAN_CAP` evenly-spaced frames are scanned.

        ``views`` mirrors the composite scan: given the projections :meth:`rotate` is about to sweep, the
        scale spans what all of them show rather than what the build-time CRS happens to show.

        The scan reads the band the frames will be **drawn** from. ``band`` rides in ``opts`` on its way to
        the renderer, so scanning band 1 regardless would scale every other band against the wrong range —
        usually a fully saturated clip under a colorbar labelled with band 1's numbers.

        Args:
            datasets: The animation's frames.
            opts: The render options every frame will be drawn with. Read for ``vmin``, ``vmax`` and
                ``band``; the resolved ``vmin``/``vmax`` are written back into it.
            views: The display CRSs :meth:`rotate` will sweep, or ``None`` for an :meth:`animate` whose
                frames share the current CRS.

        Examples:
            - The scale is measured from the band being animated, not from band 1:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.1, epsg=4326)
                >>> frames = [
                ...     Dataset.from_array(
                ...         np.stack([np.full((4, 5), 1.0 + k), np.full((4, 5), 500.0 + 100 * k)])
                ...         .astype("float32"),
                ...         geo_ref=geo,
                ...     )
                ...     for k in range(3)
                ... ]
                >>> opts = {"band": 2}
                >>> Map(crs=4326)._resolve_animation_clim(frames, opts)
                >>> opts["vmin"], opts["vmax"]
                (500.0, 700.0)

                ```
            - A bound the caller already set is kept, and only the missing one is measured:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(top_left_corner=(4.0, 53.0), cell_size=0.1, epsg=4326)
                >>> frames = [
                ...     Dataset.from_array(
                ...         np.stack([np.full((4, 5), 1.0 + k), np.full((4, 5), 500.0 + 100 * k)])
                ...         .astype("float32"),
                ...         geo_ref=geo,
                ...     )
                ...     for k in range(3)
                ... ]
                >>> opts = {"band": 2, "vmin": 0.0}
                >>> Map(crs=4326)._resolve_animation_clim(frames, opts)
                >>> opts["vmin"], opts["vmax"]
                (0.0, 700.0)

                ```
        """
        vmin, vmax = opts.get("vmin"), opts.get("vmax")
        if vmin is None or vmax is None:
            band = _animated_band(opts)
            if views is None:
                lo, hi = self._stack_clim(_scan_subset(datasets), band=band)
            else:
                lo, hi = self._clim_across_views(next(iter(datasets)), views, band=band)
            opts["vmin"] = lo if vmin is None else vmin
            opts["vmax"] = hi if vmax is None else vmax

    def _stack_channel_limits(
        self,
        datasets: Sequence[Any],
        bands: Sequence[int],
        *,
        mask_nodata: bool = True,
        views: Optional[Sequence[Any]] = None,
    ) -> ChannelLimits:
        """Return one ``(lo, hi)`` stretch bound per composite channel, spanning the whole stack.

        The composite counterpart of :meth:`_resolve_animation_clim`. Each scanned frame contributes its own
        per-channel 2-98 percentiles and the widest ``lo``/``hi`` per channel wins, so every frame is
        stretched on one fixed black and white point. Without this each frame re-derives its own and the clip
        pumps between frames even where the scene is unchanged — the artefact a viewer reads as the data
        having changed. Each frame is reprojected to the display CRS before it is measured, exactly as the
        composite does when it renders it — the warp resamples values, and on a strong projection it moves the
        percentiles a long way (an orthographic warp shifts the white point by roughly a third), so measuring
        the stored values would put the animation on a visibly different stretch from the equivalent still.
        That costs one extra warp per scanned frame, bounded by :data:`_CLIM_SCAN_CAP` evenly-spaced frames.

        ``views`` covers :meth:`rotate`, whose frames differ by projection rather than by data: it sweeps the
        display CRS as it renders, so measuring the CRS in force at build time would describe a view none of
        its frames use. Given the projections it is about to sweep, the scan measures the dataset under a
        sample of them and unions the result, at one warp per sampled view.

        A frame whose channel is entirely nodata contributes no bound for that channel rather than a ``nan``
        that would swallow the others (``min``/``max`` against ``nan`` is order-dependent, so a dead **first**
        frame would otherwise blank the channel for the whole animation). A channel dead in every *scanned*
        frame reports ``(nan, nan)``, which :func:`~digitalearth.base.stretch.stretch_to_unit` reads as
        "no frozen bound for this channel" and answers per frame. Reporting a fixed span here instead would be
        wrong whenever the sampled positions happen to miss the frames that carry the channel — dead in every
        scanned frame is not dead in every frame, and the frames that do carry data would then clip flat
        against that span.

        Args:
            datasets: The animation stack.
            bands: The 1-based band indices the composite maps to its channels.
            mask_nodata: Whether nodata cells are excluded from the percentiles, mirroring the composite's
                own ``mask_nodata`` so both read the same cells. The result is still not any single frame's
                own stretch — it is the union across the scanned frames, which is the whole point.
            views: Optional display CRSs the frames will actually be drawn in. When given, the first dataset
                is measured under a sample of them instead of under the current display CRS — the shape
                :meth:`rotate` needs, where every frame is the same data in a different projection.

        Returns:
            One ``(lo, hi)`` tuple per channel, in channel order.

        Raises:
            ValueError: when ``datasets`` is empty, so there is nothing to derive a stretch from.
        """
        seq = list(datasets)
        if not seq:
            raise ValueError("cannot derive composite limits from an empty stack")
        # Measure what the frame will actually render: the composites stretch get_stack(self._reproject(ds)),
        # so scanning the stored values would freeze the wrong bounds under any non-trivial display CRS.
        if views is None:
            scanned = []
            for ds in _scan_subset(seq):
                try:
                    warped = self._reproject(ds)
                except OffLimbError:
                    continue  # this frame draws nothing, so it contributes no stretch bound
                scanned.append(
                    channel_limits(get_stack(warped, bands, mask=mask_nodata))
                )
            if not scanned:  # no frame is on the view at all
                scanned.append(
                    channel_limits(get_stack(seq[0], bands, mask=mask_nodata))
                )
        else:
            scanned = self._scan_across_views(
                seq[0], bands, views, mask_nodata=mask_nodata
            )
        return _union_channel_limits(scanned)

    def _scan_across_views(
        self,
        dataset: Any,
        bands: Sequence[int],
        views: Sequence[Any],
        *,
        mask_nodata: bool = True,
    ) -> List[List[Tuple[float, float]]]:
        """Measure ``dataset`` once under each sampled display CRS in ``views``.

        The rotation counterpart of the per-frame scan: :meth:`rotate` redraws one dataset under a sweep of
        projections, so the bounds that matter are the ones those projections produce, not the ones the CRS
        happening to sit on the Map at build time produces. The display CRS is restored afterwards, so a
        Map primed but never rendered is left as it was found.

        Args:
            dataset: The raster every frame draws.
            bands: The 1-based band indices the composite maps to its channels.
            views: The display CRSs the animation will sweep; sampled by :func:`_scan_subset`.
            mask_nodata: Whether nodata cells are excluded from the percentiles.

        Returns:
            One per-channel bound list per sampled view that shows any of the data. A sweep passes the far
            side of the globe, where the warp has nothing to transform; if no view shows anything at all the
            data is measured as stored, so the caller still gets a usable stretch.
        """
        original = self.crs
        try:
            measured = []
            for view in _scan_subset(views):
                self.crs = view
                try:
                    stack = get_stack(self._reproject(dataset), bands, mask=mask_nodata)
                except OffLimbError:  # this view shows none of the data
                    continue
                measured.append(channel_limits(stack))
            if not measured:  # no view showed anything: fall back to the data as stored
                measured.append(
                    channel_limits(get_stack(dataset, bands, mask=mask_nodata))
                )
            return measured
        finally:
            self.crs = original

    def _frame_style(self, datasets: Sequence[Any], band: int) -> dict:
        """Resolve the per-variable style (``cmap``/``levels``/``units``) the frames will be drawn with.

        The frames resolve theirs one at a time inside
        :meth:`~digitalearth.static.maps.raster.RasterMixin._field`; the persistent colorbar is built once,
        before any frame is drawn, so it has to look the same style up here or it would key a ``viridis``
        bar to frames drawn in whatever colormap the variable actually selects.

        Args:
            datasets: The animation's frames; the first readable one carries the variable metadata (every
                frame of an animation is the same variable).
            band: 1-based band the animation draws.

        Returns:
            The :func:`~digitalearth.base.autostyle.auto_style` dict for the animated variable, or an empty
            dict when no frame could be read.

        Raises:
            Exception: anything that is not a read failure (see :data:`UNREADABLE_FRAME`) — a style lookup
                that breaks for its own reasons is a bug, and reporting it as "no style" would hide it
                behind a silently default-coloured bar.
        """
        for dataset in datasets:
            # An unreadable frame is the next frame's problem, not the colorbar's: styling is decoration,
            # and refusing to build the bar would fail an animation that renders perfectly well.
            try:
                return auto_style(get_source(dataset, band=band))
            except UNREADABLE_FRAME as error:
                logger.warning(
                    "animation colorbar: unreadable frame skipped — %s: %s",
                    type(error).__name__,
                    error,
                )
        return {}

    def _animation_colorbar(
        self, opts: dict, label: Optional[str], style: Optional[dict] = None
    ) -> Any:
        """Add one static colorbar for an animation from the already-resolved ``cmap``/``vmin``/``vmax``.

        The colorbar lives on its own figure axes (not the data axes that each frame clears), so it persists
        across frames. Call :meth:`_resolve_animation_clim` first so ``opts`` has the shared clim.

        Args:
            opts: The render options every frame is drawn with; the resolved ``cmap`` is written back into
                it so the frames and this bar cannot disagree.
            label: Caller-supplied colorbar label, or ``None`` to fall back to the variable's ``units``.
            style: The variable's :func:`~digitalearth.base.autostyle.auto_style` dict, when one was
                resolved — the source of both the default colormap and the fallback label.

        Returns:
            The created ``matplotlib.colorbar.Colorbar``.
        """
        style = style or {}
        if opts.get("cmap") is None:
            opts["cmap"] = style.get("cmap") or DEFAULT_FIELD_CMAP
        mappable = ScalarMappable(
            norm=Normalize(vmin=opts.get("vmin"), vmax=opts.get("vmax")),
            cmap=opts["cmap"],
        )
        mappable.set_array([])
        cbar = self.fig.colorbar(mappable, ax=self.ax)
        if label is None:
            label = style.get("units")  # the variable's own units (T6.2)
        if label is not None:
            cbar.set_label(label)
        return cbar

    def _prime_animation(
        self,
        datasets: Sequence[Any],
        opts: dict,
        *,
        kind: str,
        colorbar: bool,
        cbar_label: Optional[str],
        views: Optional[Sequence[Any]] = None,
    ) -> None:
        """Resolve the one shared colour treatment into ``opts``, and if asked add the static colorbar.

                The setup shared by :meth:`animate` and :meth:`rotate`, branching on what ``kind`` draws:

                - A **scalar field** gets a missing ``vmin``/``vmax`` filled once from the stack so colours don't
                  flicker between frames, then optionally one persistent colorbar.
                - A **composite** (:data:`_COMPOSITE_KINDS`) gets frozen per-channel stretch ``limits`` instead. A
                  clim is meaningless for an RGB image — the composite runs its own per-channel stretch, so nothing
                  in its render path reads ``vmin``/``vmax`` (they are still forwarded to the glyph with every other
                  kwarg; they simply have no colour scale to move). There is likewise no single mappable to key a
                  colorbar to, so an explicit ``colorbar=True`` is refused rather than answered with a useless bar.

        Real limits already in ``opts`` are kept, so a caller can pass their own ``limits=`` to override the
                scan. A ``limits`` of ``None`` counts as absent and is filled, matching how
                :meth:`_resolve_animation_clim` treats a ``None`` ``vmin``/``vmax`` — an explicit ``limits=None`` is
                what a wrapper forwarding an optional passes, and silently skipping the freeze there would put the
                flicker back with nothing to notice.

                Raises:
                    ValueError: when ``colorbar=True`` is combined with a composite ``kind``, or when a composite's
                        ``bands`` does not hold exactly three indices.
        """
        if kind in _COMPOSITE_KINDS:
            if colorbar:
                raise ValueError(
                    f"colorbar=True is not supported for a {kind!r} animation: a composite renders an RGB "
                    "image, which has no single scalar mappable to key a colorbar to"
                )
            # absent or None -> the default (M2)
            bands = opts.get("bands") or DEFAULT_COMPOSITE_BANDS
            opts["bands"] = bands
            require_three_bands(kind, bands)  # before the scan, not after it (M3)
            if opts.get("limits") is None:  # absent *or* explicitly None (H1)
                opts["limits"] = self._stack_channel_limits(
                    datasets,
                    bands,
                    mask_nodata=opts.get("mask_nodata", True),
                    views=views,
                )
            return
        self._resolve_animation_clim(datasets, opts, views=views)
        if colorbar:
            style = self._frame_style(datasets, _animated_band(opts))
            self._animation_colorbar(opts, cbar_label, style)

    def _draw_animation_frame(
        self,
        data: Any,
        kind: str,
        opts: dict,
        *,
        ocean: bool,
        coastlines: bool,
        title: Optional[str] = None,
    ) -> None:
        """Draw one animation frame: optional ocean disc, the field, optional coastlines, optional title.

        The per-frame body shared by :meth:`animate` and :meth:`rotate`. Ocean fill and coastlines are
        decoration: the ocean disc is drawn only on a globe, and a coastline failure (no network/data) is
        swallowed so the animation still renders.
        """
        if ocean and self.globe:
            self.ocean()
        method, picked = _KIND_METHODS.get(kind, (kind, {}))
        getattr(self, method)(data, **picked, **opts)
        if coastlines:
            try:
                self.coastlines()
            except Exception:  # network/data unavailable — decoration is best-effort
                pass
        if title is not None:
            self.set_title(title)

    def animate(
        self,
        stack: Any,
        *,
        kind: str = "imshow",
        fps: float = DEFAULT_FPS,
        titles: Optional[Sequence[str]] = None,
        ocean: bool = False,
        coastlines: bool = False,
        colorbar: bool = False,
        cbar_label: Optional[str] = None,
        **kwargs,
    ) -> FuncAnimation:
        """Animate a stack of rasters over this map, returning a matplotlib :class:`FuncAnimation`.

        Each frame reprojects ``stack[i]`` to the display CRS (pyramids), renders it with the ``kind`` method
        (cleopatra), optionally draws the ocean disc / coastlines, and — on a globe — applies the projection
        frame, so every frame gets the boundary, graticule, and limb-clipping for free. The returned
        animation is lazy: call ``anim.save("out.gif", writer=PillowWriter(fps=...))`` or display it.

        All frames share **one colour treatment**, so nothing flickers between them. A scalar field takes
        ``vmin``/``vmax`` from ``kwargs`` if given, else measured once over the stack. A composite
        (``"rgb_composite"`` / ``"hsv_composite"``) instead takes one per-channel contrast stretch, frozen
        once over the stack — without which every frame would re-derive its own 2-98 percentile and the
        clip would pump. Pass your own ``limits=[(lo, hi), ...]`` to override that scan.

        Either scan measures frames **as they will be drawn**, reprojected into the display CRS, so the
        animation lands on the same scale as the equivalent still. On a globe that means the scale spans
        what the globe shows: an extreme value on the hidden hemisphere does not set the top of it.

        Args:
            stack: An ordered collection of pyramids ``Dataset`` frames (e.g. a list, or a
                ``DatasetCollection`` datacube) — one raster per animation frame.
            kind: The method used to draw each frame — a scalar field (``"imshow"`` / ``"contourf"`` /
                ``"contour"`` / ``"pcolormesh"`` / ``"block"``) or a true/false-colour composite
                (``"rgb_composite"`` / ``"hsv_composite"``, which take ``bands=(r, g, b)`` through
                ``**kwargs`` to pick their channels).
            fps: Frames per second (sets the inter-frame interval). Defaults to :data:`DEFAULT_FPS`, the one
                rate every animation entry point — here, :meth:`rotate`, and the other backends — starts from.
            titles: Optional per-frame titles; must match the stack length when given.
            ocean: When True, fill the ocean disc behind each frame (globe maps only).
            coastlines: When True, overlay coastlines each frame (best-effort; ignored if unreachable).
            colorbar: When True, add one static colorbar (drawn once, not per frame) using the shared
                colour scale. Not available on a composite ``kind`` — an RGB image has no scalar mappable.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method. A scalar field takes ``band``, ``cmap``, ``vmin``,
                ``vmax`` and the rest of its styling — the shared colour scale is measured from that same
                ``band`` (1 by default); a composite takes ``bands``, ``mask_nodata`` and ``limits``.
                ``vmin``/``vmax`` are accepted on a composite because they reach the glyph like any other
                kwarg, but an RGB image has no colour scale for them to move — use ``limits``.

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``len(stack)`` frames (also kept on
            ``self._animation`` so it is not garbage-collected before you save/display it).

        Raises:
            ValueError: if ``kind`` is not a known renderer, ``stack`` is empty, ``titles`` is given with a
                mismatched length, or ``colorbar=True`` is combined with a composite ``kind``.

        Examples:
            - Animate a two-frame scalar stack; the animation is lazy, so no frame is drawn yet:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> stack = [Dataset.from_array(arr=np.full((60, 120), value, "float32"), geo_ref=geo)
                ...          for value in (1.0, 2.0)]
                >>> m = Map(crs=4326)
                >>> anim = m.animate(stack, fps=2)
                >>> len(list(anim.new_frame_seq()))
                2

                ```
            - A true-colour stack animates as a composite, on one stretch frozen over the whole stack:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> stack = [Dataset.from_array(arr=np.stack([np.full((60, 120), value, "float32")] * 3),
                ...                             geo_ref=geo) for value in (1.0, 2.0)]
                >>> m = Map(crs=4326)
                >>> anim = m.animate(stack, kind="rgb_composite", fps=2)
                >>> len(list(anim.new_frame_seq()))
                2

                ```
            - A composite has no scalar mappable, so asking for a colorbar is refused rather than
                answered with a meaningless bar:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> stack = [Dataset.from_array(arr=np.stack([np.full((60, 120), 1.0, "float32")] * 3),
                ...                             geo_ref=geo)]
                >>> Map(crs=4326).animate(stack, kind="rgb_composite",
                ...                       colorbar=True)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: colorbar=True is not supported for a 'rgb_composite' animation: ...

                ```

        See Also:
            rotate: Spin one field on an orthographic globe instead of stepping through a stack.
            save_animation: Write the returned animation to an mp4/GIF.
        """
        if kind not in _ANIMATION_KINDS:
            raise ValueError(
                f"unknown animation kind {kind!r}; choose one of {_ANIMATION_KINDS}"
            )
        frames = _as_frames(
            stack
        )  # a DatasetCollection iterates to arrays, not Datasets
        if not frames:
            raise ValueError("animate got an empty stack (nothing to animate)")
        if titles is not None and len(titles) != len(frames):
            raise ValueError(
                f"titles length ({len(titles)}) must match the stack length ({len(frames)})"
            )
        self._prime_animation(
            frames, kwargs, kind=kind, colorbar=colorbar, cbar_label=cbar_label
        )

        def draw_one(i: int) -> None:
            title = titles[i] if titles is not None else None
            self._draw_animation_frame(
                frames[i], kind, kwargs, ocean=ocean, coastlines=coastlines, title=title
            )

        return self._animate_frames(draw_one, len(frames), fps)

    def rotate(
        self,
        dataset: Any,
        *,
        lat: float = 15.0,
        n_frames: int = 24,
        fps: float = DEFAULT_FPS,
        lon0: float = -180.0,
        kind: str = "imshow",
        ocean: bool = False,
        coastlines: bool = False,
        colorbar: bool = False,
        cbar_label: Optional[str] = None,
        **kwargs,
    ) -> FuncAnimation:
        """Spin an orthographic globe over a single field by sweeping the centre longitude.

        Forces a globe map and redraws ``dataset`` on ``n_frames`` orthographic projections whose centre
        longitude steps a full 360 degrees from ``lon0``.

        **Terminal for this Map's projection:** ``rotate`` sets ``globe=True`` and sweeps the display CRS
        (:attr:`crs`) as the animation renders, leaving the Map centred on the **final** frame. Treat a
        rotated Map as consumed by the animation — create a fresh ``Map`` if you need the original projection.

        The colour treatment is measured across the projections the sweep will actually use, not the one the
        Map was built with, so a rotation is not scaled to whichever hemisphere happened to face front when
        it started. A view that shows none of the data — a full sweep passes the far side — contributes
        nothing rather than failing the animation.

        Args:
            dataset: The pyramids ``Dataset`` to spin (reprojected per frame).
            lat: Centre latitude of every orthographic view.
            n_frames: Number of frames spanning the full 360-degree turn.
            fps: Frames per second. Defaults to :data:`DEFAULT_FPS` — the same rate :meth:`animate` starts
                from, so a rotation and a stack animation built with defaults play at one speed. This method
                used to default to ``8.0``; a call that names no rate now renders slower, and ``fps=8.0``
                restores the previous speed.
            lon0: Starting centre longitude.
            kind: The method used to draw the data — a scalar field (``"imshow"`` / ``"contourf"`` /
                ``"pcolormesh"`` / ``"contour"`` / ``"block"``) or a composite (``"rgb_composite"`` /
                ``"hsv_composite"``).
            ocean: When True, fill the ocean disc behind the data each frame.
            coastlines: When True, overlay coastlines each frame (best-effort).
            colorbar: When True, add one static colorbar (drawn once) using the shared colour scale. Not
                available on a composite ``kind``.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method. A scalar field takes ``band``, ``cmap``, ``vmin``,
                ``vmax`` and the rest of its styling — the shared colour scale is measured from that same
                ``band`` (1 by default); a composite takes ``bands``, ``mask_nodata`` and ``limits``.
                ``vmin``/``vmax`` are accepted on a composite because they reach the glyph like any other
                kwarg, but an RGB image has no colour scale for them to move — use ``limits``.

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``n_frames`` frames (also kept on
            ``self._animation`` so it is not garbage-collected before you save/display it).

        Raises:
            ValueError: if ``n_frames`` is less than 1, ``kind`` is not a known renderer, or
                ``colorbar=True`` is combined with a composite ``kind``.

        Examples:
            - Spin one field over four frames; the map is forced into globe mode:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> field = Dataset.from_array(arr=np.full((60, 120), 1.0, "float32"), geo_ref=geo)
                >>> m = Map(crs=4326)
                >>> anim = m.rotate(field, n_frames=4, fps=4)
                >>> len(list(anim.new_frame_seq()))
                4
                >>> m.globe
                True

                ```
            - A composite spins too, sharing animate's kind validation:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> rgb = Dataset.from_array(arr=np.stack([np.full((60, 120), 1.0, "float32")] * 3),
                ...                          geo_ref=geo)
                >>> anim = Map(crs=4326).rotate(rgb, kind="rgb_composite", n_frames=3)
                >>> len(list(anim.new_frame_seq()))
                3

                ```

        See Also:
            animate: Step through a stack of rasters instead of spinning one.
            save_animation: Write the returned animation to an mp4/GIF.
        """
        if n_frames < 1:
            raise ValueError("rotate needs n_frames >= 1")
        if kind not in _ANIMATION_KINDS:
            raise ValueError(
                f"unknown animation kind {kind!r}; choose one of {_ANIMATION_KINDS}"
            )
        lons = [lon0 + k * (360.0 / n_frames) for k in range(n_frames)]
        views = [projections.orthographic(lon=lon, lat=lat) for lon in lons]
        # Prime first: it is the last thing that can refuse the call (a composite with colorbar=True, or a
        # wrong band count), and a refused rotate must not leave the Map switched into globe mode. Hand it
        # the projections about to be swept, so a composite freezes on the views it will really draw.
        self._prime_animation(
            [dataset],
            kwargs,
            kind=kind,
            colorbar=colorbar,
            cbar_label=cbar_label,
            views=views,
        )
        self.globe = True

        def draw_one(i: int) -> None:
            self.crs = views[i]
            self._draw_animation_frame(
                dataset, kind, kwargs, ocean=ocean, coastlines=coastlines
            )

        return self._animate_frames(draw_one, n_frames, fps)
