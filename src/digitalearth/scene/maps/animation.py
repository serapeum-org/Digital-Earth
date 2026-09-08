"""AnimationMixin — animate a stack of rasters and rotate an orthographic globe.

Drives per-frame redraws on the shared axes as a matplotlib ``FuncAnimation``, with one shared colour scale
(and an optional single static colorbar) so colours do not flicker between frames.
"""
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from digitalearth._arrays import finite, read_masked_band
from digitalearth.animation import save_animation
from digitalearth.scene import projections
from digitalearth.scene.maps.raster import channel_limits
from digitalearth.sources import get_stack

#: Cap on how many stack frames are scanned to derive a shared animation colour scale (L2).
_CLIM_SCAN_CAP = 24

#: Composite render methods accepted as an animation ``kind``. They paint colour directly rather than
#: mapping one band through a colormap, so they take the frozen-stretch path below instead of a shared
#: ``vmin``/``vmax``, and they have no mappable a colorbar could key to.
_COMPOSITE_KINDS = ("rgb_composite", "hsv_composite")

#: Field-render methods accepted as the ``kind`` of an animation frame (validated up front, N1).
_ANIMATION_KINDS = ("imshow", "contourf", "contour", "pcolormesh", "block") + _COMPOSITE_KINDS


class AnimationMixin:
    """Stack animation and globe rotation for :class:`~digitalearth.scene.map.Map`."""

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
            **kwargs: Forwarded to :func:`digitalearth.animation.save_animation` (and on to cleopatra) —
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
                >>> from digitalearth.scene import Map
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
                >>> from digitalearth.scene import Map
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

    @staticmethod
    def _resolve_composite_stretch(datasets: Sequence[Any], opts: dict) -> None:
        """Freeze one per-channel contrast stretch into ``opts`` for a composite animation.

        The composite renderers stretch each channel between its own 2-98 percentiles. Left per frame that
        gives every frame its own black and white point, so a scene that never changes still pulses in
        brightness — the artefact a viewer reads as the data changing. Scanning the stack once and pinning
        ``stretch_limits`` keeps all frames on one scale, the composite counterpart of
        :meth:`_resolve_animation_clim`.

        An explicit ``stretch_limits`` from the caller is honoured untouched. As with the clim scan, at most
        :data:`_CLIM_SCAN_CAP` evenly-spaced frames are sampled so the cost does not grow with the stack.
        """
        if opts.get("stretch_limits") is not None:
            return
        bands = tuple(opts.get("bands", (1, 2, 3)))
        mask = opts.get("mask_nodata", True)
        seq = list(datasets)
        stride = max(1, len(seq) // _CLIM_SCAN_CAP)
        sampled = [get_stack(ds, bands, mask=mask) for ds in seq[::stride]]
        # Percentiles over the concatenated frames, so the bounds describe the series and not whichever
        # frame happened to be sampled first.
        opts["stretch_limits"] = channel_limits(np.concatenate(sampled, axis=0))

    def _prime_animation(self, datasets: Sequence[Any], opts: dict, *, kind: str, colorbar: bool,
                         cbar_label: Optional[str]) -> None:
        """Resolve one shared colour scale into ``opts`` and, if asked, add the single static colorbar.

        The setup shared by :meth:`animate` and :meth:`rotate`: fix the colour treatment once across the
        whole stack so it does not flicker between frames, then optionally draw one persistent colorbar.

        Which treatment depends on ``kind``. A scalar field gets a shared ``vmin``/``vmax``
        (:meth:`_resolve_animation_clim`); a composite paints colour directly, so it gets a frozen
        per-channel stretch instead (:meth:`_resolve_composite_stretch`) and has no mappable to key a
        colorbar to.

        Raises:
            ValueError: if ``colorbar=True`` is asked for on a composite animation, which has no single
                mappable a colorbar could describe.
        """
        if kind in _COMPOSITE_KINDS:
            if colorbar:
                raise ValueError(
                    f"colorbar=True is not meaningful for kind={kind!r}: a composite paints colour from "
                    "several bands at once, so there is no single value scale to draw. Drop colorbar=, or "
                    "animate a single band with kind='imshow' to get one."
                )
            self._resolve_composite_stretch(datasets, opts)
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

        All frames share **one colour treatment**, so the colours do not flicker between frames. For a
        scalar ``kind`` that is a shared ``vmin``/``vmax`` — taken from ``kwargs`` if given, else computed
        once from the whole stack. For a composite ``kind`` it is a shared per-channel contrast stretch,
        computed once over the stack (or supplied as ``stretch_limits``); a composite has no single value
        scale, so ``colorbar=True`` is rejected rather than silently drawing a misleading one.

        Args:
            stack: An ordered collection of pyramids ``Dataset`` frames — one raster per animation frame.
                A plain list, or a ``DatasetCollection`` datacube (its ``.datasets`` are used, since the
                collection itself iterates to arrays rather than to its members).
            kind: The method used to draw each frame — a scalar field renderer (``"imshow"`` /
                ``"contourf"`` / ``"contour"`` / ``"pcolormesh"`` / ``"block"``) or a colour composite
                (``"rgb_composite"`` / ``"hsv_composite"``), which takes ``bands=`` in ``**kwargs``.
            fps: Frames per second (sets the inter-frame interval).
            titles: Optional per-frame titles; must match the stack length when given.
            ocean: When True, fill the ocean disc behind each frame (globe maps only).
            coastlines: When True, overlay coastlines each frame (best-effort; ignored if unreachable).
            colorbar: When True, add one static colorbar (drawn once, not per frame) using the shared
                colour scale. Not available for a composite ``kind``.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method (e.g. ``cmap``, ``vmin``, ``vmax`` for a scalar
                field; ``bands``, ``stretch_limits`` for a composite).

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``len(stack)`` frames (also kept on
            ``self._animation`` so it is not garbage-collected before you save/display it).

        Raises:
            ValueError: if ``kind`` is not a known renderer, ``stack`` is empty, ``titles`` is given with a
                mismatched length, or ``colorbar=True`` is asked for on a composite ``kind``.
        """
        if kind not in _ANIMATION_KINDS:
            raise ValueError(f"unknown animation kind {kind!r}; choose one of {_ANIMATION_KINDS}")
        # A DatasetCollection iterates to numpy arrays, not to its members, so unwrap `.datasets` before
        # listing — otherwise the documented datacube input reaches the renderers as bare arrays and dies
        # on the first `.read_array` (#154).
        frames = list(getattr(stack, "datasets", stack))
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
            kind: The field method used to draw the data (``"imshow"`` / ``"contourf"`` / ``"pcolormesh"`` /
                ``"contour"`` / ``"block"``).
            ocean: When True, fill the ocean disc behind the data each frame.
            coastlines: When True, overlay coastlines each frame (best-effort).
            colorbar: When True, add one static colorbar (drawn once) using the shared colour scale.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method (e.g. ``cmap``, ``vmin``, ``vmax``).

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``n_frames`` frames (also kept on
            ``self._animation`` so it is not garbage-collected before you save/display it).

        Raises:
            ValueError: if ``n_frames`` is less than 1, or ``kind`` is not a known field renderer.
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
