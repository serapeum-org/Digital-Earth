"""AnimationMixin — animate a stack of rasters and rotate an orthographic globe.

Drives per-frame redraws on the shared axes as a matplotlib ``FuncAnimation``, with one shared colour scale
(and an optional single static colorbar) so colours do not flicker between frames.
"""
from typing import Any, List, Optional, Sequence, Tuple

from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from digitalearth._arrays import finite, read_masked_band
from digitalearth.scene import projections

#: Cap on how many stack frames are scanned to derive a shared animation colour scale (L2).
_CLIM_SCAN_CAP = 24

#: Field-render methods accepted as the ``kind`` of an animation frame (validated up front, N1).
_ANIMATION_KINDS = ("imshow", "contourf", "contour", "pcolormesh", "block")


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
        return anim

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

    def _prime_animation(self, datasets: Sequence[Any], opts: dict, *, colorbar: bool,
                         cbar_label: Optional[str]) -> None:
        """Resolve one shared colour scale into ``opts`` and, if asked, add the single static colorbar.

        The setup shared by :meth:`animate` and :meth:`rotate`: fill a missing ``vmin``/``vmax`` once from the
        stack so colours don't flicker between frames, then optionally draw one persistent colorbar.
        """
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

        All frames share **one colour scale**: ``vmin``/``vmax`` from ``kwargs`` if given, else computed once
        from the whole stack — so the colours (and the colorbar) do not flicker between frames.

        Args:
            stack: An ordered, indexable collection of pyramids ``Dataset`` frames (e.g. a list, or a
                ``DatasetCollection`` datacube) — one raster per animation frame.
            kind: The field method used to draw each frame — one of ``"imshow"`` / ``"contourf"`` /
                ``"contour"`` / ``"pcolormesh"`` / ``"block"``.
            fps: Frames per second (sets the inter-frame interval).
            titles: Optional per-frame titles; must match the stack length when given.
            ocean: When True, fill the ocean disc behind each frame (globe maps only).
            coastlines: When True, overlay coastlines each frame (best-effort; ignored if unreachable).
            colorbar: When True, add one static colorbar (drawn once, not per frame) using the shared
                colour scale.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method (e.g. ``cmap``, ``vmin``, ``vmax``).

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``len(stack)`` frames (also kept on
            ``self._animation`` so it is not garbage-collected before you save/display it).

        Raises:
            ValueError: if ``kind`` is not a known field renderer, ``stack`` is empty, or ``titles`` is given
                with a mismatched length.
        """
        if kind not in _ANIMATION_KINDS:
            raise ValueError(f"unknown animation kind {kind!r}; choose one of {_ANIMATION_KINDS}")
        frames = list(stack)
        if not frames:
            raise ValueError("animate got an empty stack (nothing to animate)")
        if titles is not None and len(titles) != len(frames):
            raise ValueError(f"titles length ({len(titles)}) must match the stack length ({len(frames)})")
        self._prime_animation(frames, kwargs, colorbar=colorbar, cbar_label=cbar_label)

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
        self._prime_animation([dataset], kwargs, colorbar=colorbar, cbar_label=cbar_label)
        lons = [lon0 + k * (360.0 / n_frames) for k in range(n_frames)]

        def draw_one(i: int) -> None:
            self.crs = projections.orthographic(lon=lons[i], lat=lat)
            self._draw_animation_frame(dataset, kind, kwargs, ocean=ocean, coastlines=coastlines)

        return self._animate_frames(draw_one, n_frames, fps)
