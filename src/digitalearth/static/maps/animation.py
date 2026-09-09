"""AnimationMixin — animate a stack of rasters and rotate an orthographic globe.

Drives per-frame redraws on the shared axes as a matplotlib ``FuncAnimation``, with one shared colour
treatment so colours do not flicker between frames: a scalar field gets one ``vmin``/``vmax`` (and an
optional single static colorbar), an RGB/HSV composite gets one frozen per-channel contrast stretch.
"""
from math import isfinite
from typing import Any, List, Optional, Sequence, Tuple

from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from digitalearth.base.arrays import finite, read_masked_band
from digitalearth.base.sources import get_stack
from digitalearth.static import projections
from digitalearth.static.animation import save_animation
from digitalearth.static.maps.raster import ChannelLimits, channel_limits

#: Cap on how many stack frames are scanned to derive a shared animation colour scale (L2).
_CLIM_SCAN_CAP = 24

#: Composite renderers accepted as an animation ``kind``. They draw an RGB image rather than a scalar
#: field, so they take a frozen per-channel stretch instead of a clim, and admit no colorbar.
_COMPOSITE_KINDS = ("rgb_composite", "hsv_composite")

#: Default channels a composite maps to R/G/B when the caller passes no ``bands``.
_DEFAULT_COMPOSITE_BANDS = (1, 2, 3)

#: Render methods accepted as the ``kind`` of an animation frame (validated up front, N1).
_ANIMATION_KINDS = ("imshow", "contourf", "contour", "pcolormesh", "block") + _COMPOSITE_KINDS


class AnimationMixin:
    """Stack animation and globe rotation for :class:`~digitalearth.static.map.Map`."""

    def _animate_frames(self, draw_one: Any, n_frames: int, fps: float) -> FuncAnimation:
        """Drive ``n_frames`` of ``draw_one(i)`` on this Map's axes as a :class:`FuncAnimation`.

        Each frame clears the axes and resets the per-frame layer/frame state, calls ``draw_one(i)`` to draw
        frame ``i``, then (on a globe) sets the full-domain extent and applies the projection frame. No
        colorbar is added per frame — pass a fixed ``vmin``/``vmax`` to keep colours stable instead.
        """
        def _f(i: int) -> None:
            self.ax.clear()
            self.layers = []
            self._framed = False
            draw_one(i)
            if self.globe:
                self.set_global()
                self._apply_frame()

        anim = FuncAnimation(self.fig, _f, frames=n_frames, interval=1000.0 / fps, blit=False)
        self._animation = anim  # keep a strong reference so it isn't garbage-collected before save (L3)
        self._animation_fps = float(fps)  # so save_animation writes at the rate the scene was built for
        return anim

    def save_animation(self, path: str, *, fps: Optional[float] = None, gif: Optional[str] = None,
                       **kwargs: Any) -> Any:
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
        anim = getattr(self, "_animation", None)
        if anim is None:
            raise RuntimeError("no animation to save; call animate() or rotate() first")
        rate = getattr(self, "_animation_fps", None) if fps is None else fps
        return save_animation(anim, path, fps=rate, gif=gif, **kwargs)

    @staticmethod
    def _stack_clim(datasets: Sequence[Any]) -> Tuple[float, float]:
        """Return the ``(min, max)`` of the first band across ``datasets``, ignoring nodata/non-finite."""
        lows: List[float] = []
        highs: List[float] = []
        for ds in datasets:
            arr = finite(read_masked_band(ds, band=1))
            if arr.size:
                lows.append(float(arr.min()))
                highs.append(float(arr.max()))
        return (min(lows), max(highs)) if lows else (0.0, 1.0)

    def _resolve_animation_clim(self, datasets: Sequence[Any], opts: dict) -> None:
        """Ensure ``opts`` carries a shared ``vmin``/``vmax`` so every animation frame uses one colour scale.

        Without this, each frame's renderer auto-scales to its own data range, so the colours (and any
        colorbar) flicker between frames. Any ``vmin``/``vmax`` already in ``opts`` is kept; a missing bound
        is filled once from the stack (ignoring nodata/non-finite) and written back, so all frames — and the
        colorbar — share it. Passing both ``vmin`` and ``vmax`` skips the scan entirely. To bound the cost on
        large stacks, at most :data:`_CLIM_SCAN_CAP` evenly-spaced frames are scanned (L2).
        """
        vmin, vmax = opts.get("vmin"), opts.get("vmax")
        if vmin is None or vmax is None:
            seq = list(datasets)
            stride = max(1, len(seq) // _CLIM_SCAN_CAP)  # cap the scan to ~_CLIM_SCAN_CAP frames
            lo, hi = self._stack_clim(seq[::stride])
            opts["vmin"] = lo if vmin is None else vmin
            opts["vmax"] = hi if vmax is None else vmax

    @staticmethod
    def _stack_channel_limits(datasets: Sequence[Any], bands: Sequence[int],
                              *, mask_nodata: bool = True) -> ChannelLimits:
        """Return one ``(lo, hi)`` stretch bound per composite channel, spanning the whole stack.

        The composite counterpart of :meth:`_resolve_animation_clim`. Each scanned frame contributes its own
        per-channel 2-98 percentiles and the widest ``lo``/``hi`` per channel wins, so every frame is
        stretched on one fixed black and white point. Without this each frame re-derives its own and the clip
        pumps between frames even where the scene is unchanged — the artefact a viewer reads as the data
        having changed. Frames are read as stored rather than reprojected first: the display warp resamples
        values but does not move the percentiles enough to matter, and skipping it keeps a second warp of the
        whole stack off the critical path. At most :data:`_CLIM_SCAN_CAP` evenly-spaced frames are scanned.

        A frame whose channel is entirely nodata contributes no bound for that channel rather than a ``nan``
        that would swallow the others (``min``/``max`` against ``nan`` is order-dependent, so a dead **first**
        frame would otherwise blank the channel for the whole animation). A channel dead in *every* scanned
        frame falls back to ``(0, 1)`` — the same fallback :meth:`_stack_clim` uses for an all-nodata stack.

        Args:
            datasets: The animation stack.
            bands: The 1-based band indices the composite maps to its channels.
            mask_nodata: Whether nodata cells are excluded from the percentiles (mirrors the composite's own
                ``mask_nodata``, so the frozen limits match what an unfrozen frame would have computed).

        Returns:
            One ``(lo, hi)`` tuple per channel, in channel order.
        """
        seq = list(datasets)
        stride = max(1, len(seq) // _CLIM_SCAN_CAP)  # cap the scan to ~_CLIM_SCAN_CAP frames
        scanned = [channel_limits(get_stack(ds, bands, mask=mask_nodata)) for ds in seq[::stride]]
        limits: List[Tuple[float, float]] = []
        for index in range(len(scanned[0])):
            lows = [frame[index][0] for frame in scanned if isfinite(frame[index][0])]
            highs = [frame[index][1] for frame in scanned if isfinite(frame[index][1])]
            limits.append((min(lows), max(highs)) if lows and highs else (0.0, 1.0))
        return limits

    def _animation_colorbar(self, opts: dict, label: Optional[str]) -> Any:
        """Add one static colorbar for an animation from the already-resolved ``cmap``/``vmin``/``vmax``.

        The colorbar lives on its own figure axes (not the data axes that each frame clears), so it persists
        across frames. Call :meth:`_resolve_animation_clim` first so ``opts`` has the shared clim.
        """
        cmap = opts.setdefault("cmap", "viridis")
        mappable = ScalarMappable(norm=Normalize(vmin=opts.get("vmin"), vmax=opts.get("vmax")), cmap=cmap)
        mappable.set_array([])
        cbar = self.fig.colorbar(mappable, ax=self.ax)
        if label is not None:
            cbar.set_label(label)
        return cbar

    def _prime_animation(self, datasets: Sequence[Any], opts: dict, *, kind: str, colorbar: bool,
                         cbar_label: Optional[str]) -> None:
        """Resolve the one shared colour treatment into ``opts``, and if asked add the static colorbar.

        The setup shared by :meth:`animate` and :meth:`rotate`, branching on what ``kind`` draws:

        - A **scalar field** gets a missing ``vmin``/``vmax`` filled once from the stack so colours don't
          flicker between frames, then optionally one persistent colorbar.
        - A **composite** (:data:`_COMPOSITE_KINDS`) gets frozen per-channel stretch ``limits`` instead. A
          clim is meaningless for an RGB image — the composite runs its own per-channel stretch and never
          consults ``vmin``/``vmax`` — and there is no single mappable to key a colorbar to, so an explicit
          ``colorbar=True`` is refused rather than answered with a meaningless bar.

        Limits already in ``opts`` are kept, so a caller can pass their own ``limits=`` to override the scan.

        Raises:
            ValueError: when ``colorbar=True`` is combined with a composite ``kind``.
        """
        if kind in _COMPOSITE_KINDS:
            if colorbar:
                raise ValueError(
                    f"colorbar=True is not supported for a {kind!r} animation: a composite renders an RGB "
                    "image, which has no single scalar mappable to key a colorbar to"
                )
            opts.setdefault("limits", self._stack_channel_limits(
                datasets, opts.get("bands", _DEFAULT_COMPOSITE_BANDS),
                mask_nodata=opts.get("mask_nodata", True),
            ))
            return
        self._resolve_animation_clim(datasets, opts)
        if colorbar:
            self._animation_colorbar(opts, cbar_label)

    def _draw_animation_frame(self, data: Any, kind: str, opts: dict, *, ocean: bool, coastlines: bool,
                              title: Optional[str] = None) -> None:
        """Draw one animation frame: optional ocean disc, the field, optional coastlines, optional title.

        The per-frame body shared by :meth:`animate` and :meth:`rotate`. Ocean fill and coastlines are
        decoration: the ocean disc is drawn only on a globe, and a coastline failure (no network/data) is
        swallowed so the animation still renders.
        """
        if ocean and self.globe:
            self.ocean()
        getattr(self, kind)(data, **opts)
        if coastlines:
            try:
                self.coastlines()
            except Exception:  # network/data unavailable — decoration is best-effort
                pass
        if title is not None:
            self.set_title(title)

    def animate(self, stack: Any, *, kind: str = "imshow", fps: float = 3.0,
                titles: Optional[Sequence[str]] = None, ocean: bool = False, coastlines: bool = False,
                colorbar: bool = False, cbar_label: Optional[str] = None, **kwargs) -> FuncAnimation:
        """Animate a stack of rasters over this map, returning a matplotlib :class:`FuncAnimation`.

        Each frame reprojects ``stack[i]`` to the display CRS (pyramids), renders it with the ``kind`` method
        (cleopatra), optionally draws the ocean disc / coastlines, and — on a globe — applies the projection
        frame, so every frame gets the boundary, graticule, and limb-clipping for free. The returned
        animation is lazy: call ``anim.save("out.gif", writer=PillowWriter(fps=...))`` or display it.

        All frames share **one colour treatment**, so nothing flickers between them. A scalar field takes
        ``vmin``/``vmax`` from ``kwargs`` if given, else computed once from the whole stack. A composite
        (``"rgb_composite"`` / ``"hsv_composite"``) instead takes one per-channel contrast stretch, frozen
        once over the stack — without which every frame would re-derive its own 2-98 percentile and the
        clip would pump. Pass your own ``limits=[(lo, hi), ...]`` to override that scan.

        Args:
            stack: An ordered, indexable collection of pyramids ``Dataset`` frames (e.g. a list, or a
                ``DatasetCollection`` datacube) — one raster per animation frame.
            kind: The method used to draw each frame — a scalar field (``"imshow"`` / ``"contourf"`` /
                ``"contour"`` / ``"pcolormesh"`` / ``"block"``) or a true/false-colour composite
                (``"rgb_composite"`` / ``"hsv_composite"``, which take ``bands=(r, g, b)`` through
                ``**kwargs`` to pick their channels).
            fps: Frames per second (sets the inter-frame interval).
            titles: Optional per-frame titles; must match the stack length when given.
            ocean: When True, fill the ocean disc behind each frame (globe maps only).
            coastlines: When True, overlay coastlines each frame (best-effort; ignored if unreachable).
            colorbar: When True, add one static colorbar (drawn once, not per frame) using the shared
                colour scale. Not available on a composite ``kind`` — an RGB image has no scalar mappable.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method (e.g. ``cmap``, ``vmin``, ``vmax``; or ``bands``,
                ``mask_nodata``, ``limits`` for a composite).

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
            raise ValueError(f"unknown animation kind {kind!r}; choose one of {_ANIMATION_KINDS}")
        frames = list(stack)
        if not frames:
            raise ValueError("animate got an empty stack (nothing to animate)")
        if titles is not None and len(titles) != len(frames):
            raise ValueError(f"titles length ({len(titles)}) must match the stack length ({len(frames)})")
        self._prime_animation(frames, kwargs, kind=kind, colorbar=colorbar, cbar_label=cbar_label)

        def draw_one(i: int) -> None:
            title = titles[i] if titles is not None else None
            self._draw_animation_frame(frames[i], kind, kwargs, ocean=ocean, coastlines=coastlines,
                                       title=title)

        return self._animate_frames(draw_one, len(frames), fps)

    def rotate(self, dataset: Any, *, lat: float = 15.0, n_frames: int = 24, fps: float = 8.0,
               lon0: float = -180.0, kind: str = "imshow", ocean: bool = False, coastlines: bool = False,
               colorbar: bool = False, cbar_label: Optional[str] = None, **kwargs) -> FuncAnimation:
        """Spin an orthographic globe over a single field by sweeping the centre longitude.

        Forces a globe map and redraws ``dataset`` on ``n_frames`` orthographic projections whose centre
        longitude steps a full 360 degrees from ``lon0``.

        **Terminal for this Map's projection:** ``rotate`` sets ``globe=True`` and sweeps the display CRS
        (:attr:`crs`) as the animation renders, leaving the Map centred on the **final** frame. Treat a
        rotated Map as consumed by the animation — create a fresh ``Map`` if you need the original projection.

        Args:
            dataset: The pyramids ``Dataset`` to spin (reprojected per frame).
            lat: Centre latitude of every orthographic view.
            n_frames: Number of frames spanning the full 360-degree turn.
            fps: Frames per second.
            lon0: Starting centre longitude.
            kind: The method used to draw the data — a scalar field (``"imshow"`` / ``"contourf"`` /
                ``"pcolormesh"`` / ``"contour"`` / ``"block"``) or a composite (``"rgb_composite"`` /
                ``"hsv_composite"``).
            ocean: When True, fill the ocean disc behind the data each frame.
            coastlines: When True, overlay coastlines each frame (best-effort).
            colorbar: When True, add one static colorbar (drawn once) using the shared colour scale. Not
                available on a composite ``kind``.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method (e.g. ``cmap``, ``vmin``, ``vmax``; or ``bands``,
                ``mask_nodata``, ``limits`` for a composite).

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
            raise ValueError(f"unknown animation kind {kind!r}; choose one of {_ANIMATION_KINDS}")
        self.globe = True
        self._prime_animation([dataset], kwargs, kind=kind, colorbar=colorbar, cbar_label=cbar_label)
        lons = [lon0 + k * (360.0 / n_frames) for k in range(n_frames)]

        def draw_one(i: int) -> None:
            self.crs = projections.orthographic(lon=lons[i], lat=lat)
            self._draw_animation_frame(dataset, kind, kwargs, ocean=ocean, coastlines=coastlines)

        return self._animate_frames(draw_one, n_frames, fps)
