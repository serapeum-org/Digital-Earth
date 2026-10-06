"""AnimationMixin — animate a stack of rasters and rotate an orthographic globe.

Drives the frames on the shared axes as a matplotlib ``FuncAnimation``, with one shared colour treatment so
colours do not flicker between frames: a scalar field gets one ``vmin``/``vmax`` (and an optional single
static colorbar), an RGB/HSV composite gets one frozen per-channel contrast stretch.

**What a frame costs is decided here (ST-9).** The first frame draws the scene; every frame after it gives the
artist that frame drew the next frame's values — ``AxesImage.set_data``, ``QuadMesh.set_array`` — and leaves
the decoration, the projection frame, the colorbar and the view exactly as they are. Clearing the axes per
frame threw all of that away and rebuilt it, which is both the per-frame cost and the reason blitting was off:
blitting repaints only what a frame hands back and needs the rest of the picture to stand still. Not every
frame can be updated that way, so :class:`FrameUpdate` says which cannot and why, in words the refusal,
the log line and the fallback all share.
"""

import logging
from math import isfinite
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from digitalearth.base.animation import DEFAULT_FPS
from digitalearth.base.arrays import read_masked_band
from digitalearth.base.autostyle import auto_style
from digitalearth.base.clim import (
    DEFAULT_CLIM_SCAN_CAP,
    frozen_scale,
    measure_clim,
    sample_evenly,
)
from digitalearth.base.sources import get_source, get_stack
from digitalearth.base.spec import DEFAULT_BAND
from digitalearth.base.stretch import (
    DEFAULT_COMPOSITE_BANDS,
    ChannelLimits,
    channel_limits,
    require_three_bands,
)
from digitalearth.static import projections
from digitalearth.static.animation import save_animation
from digitalearth.static.maps.base import OffLimbError
from digitalearth.static.maps.raster import (
    DEFAULT_FIELD_CMAP,
    FieldColors,
    _field_source,
)
from digitalearth.static.render_compat import CENTER_KEY

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
#: A `kind` is the *renderer* a caller names, and order 27a moved the methods it dispatches to — so
#: dispatching on the kind alone would reach a method that no longer exists. Measured: of the eight
#: `_ANIMATION_KINDS`, `Map` carries no `imshow`, `contour` or `contourf` method at all, and
#: `getattr(self, kind)` for one of them raises `AttributeError` at the first frame — inside matplotlib's
#: frame loop, where a caller has nothing to act on. The old kinds stay in the vocabulary because they
#: are what a caller may already have written; the keywords are what tells `contour` and `contourf`
#: apart now that `Map.contours` draws both.
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


class FrameUpdate:
    """How an animation draws the frames after the first — ST-9's per-frame strategy, as an object.

    One of these is built per :meth:`AnimationMixin.animate` call and answers three questions that have to
    agree with each other:

    - **Can one artist be updated?** :attr:`in_place`, and when it cannot, :attr:`blocker` — the reason as a
      sentence, because every reader of it needs the words: ``update="in_place"`` refuses with them,
      ``blit=True`` refuses with them, and the default ``update="auto"`` logs them as why it is redrawing.
      "Unsupported" would have been none of the three.
    - **May this call have what it asked for?** :meth:`settle`, which refuses a ``blit``/``update`` that
      cannot be honoured and otherwise reports the strategy the frame loop should run.
    - **What happens on frame i?** :meth:`apply`, which gives the kept artist that frame's values.

    Holding the three on one object is the point: a refusal that said one thing while the frame loop quietly
    did another is exactly how an animation comes out wrong with nothing to notice.

    Attributes:
        scene: The map being animated, read for the display CRS, the canvas and the strict flag.
        frames: The animation's frames, indexed by :meth:`apply`.
        kind: The renderer the frames are drawn with.
        mode: The caller's ``update``, one of :attr:`MODES`.
        titles: The per-frame titles, or ``None``.
        blocker: Why the in-place path is unavailable, or ``None`` when it is available.
        band: The 1-based band the frames draw; :data:`~digitalearth.base.spec.DEFAULT_BAND` and unread
            when :attr:`blocker` says no frame will be updated in place.

    Examples:
        - An image field is the case the capability is for, so nothing blocks it and there is no reason
          to report:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.maps.animation import FrameUpdate
            >>> scene = Map(crs=4326)
            >>> plan = FrameUpdate(scene, [], {}, kind="imshow")
            >>> plan.in_place, plan.blocker
            (True, None)
            >>> scene.close()

            ```
        - A kind with no array to refill says why, in the words the refusal and the log line share; so
          does an option whose value belongs to the frame being drawn:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.maps.animation import FrameUpdate
            >>> scene = Map(crs=4326)
            >>> FrameUpdate(scene, [], {}, kind="contourf").blocker
            "kind='contourf' draws a contour set, which is cut from the values rather than handed new ones"
            >>> FrameUpdate(scene, [], {"scheme": "quantiles"}, kind="imshow").blocker
            "scheme='quantiles' is derived from the frame being drawn, which a kept artist cannot follow"
            >>> scene.close()

            ```
        - An `update=` outside :attr:`MODES` is refused where the caller is, rather than at the first
          frame:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.maps.animation import FrameUpdate
            >>> scene = Map(crs=4326)
            >>> FrameUpdate(scene, [], {}, kind="imshow", mode="inplace")
            Traceback (most recent call last):
                ...
            ValueError: unknown update mode 'inplace'; choose one of ('auto', 'in_place', 'redraw')
            >>> scene.close()

            ```
    """

    #: How each animation ``kind`` hands the next frame's values to the artist it already drew — the update
    #: that makes clearing the axes unnecessary, as ``{kind: artist setter}``. ``field``/``imshow`` draw an
    #: ``AxesImage``, whose ``set_data`` swaps the array; ``pcolormesh``/``block`` draw a ``QuadMesh``, whose
    #: ``set_array`` swaps the cell values. A kind absent from here draws something no setter can refill and
    #: is redrawn per frame instead (see :attr:`UNUPDATABLE`).
    SETTERS = {
        "field": "set_data",
        "imshow": "set_data",
        "pcolormesh": "set_array",
        "block": "set_array",
    }

    #: Why each remaining :data:`_ANIMATION_KINDS` member cannot be updated in place. Said per kind rather
    #: than as one "unsupported", because the two reasons are different facts a caller can act on: a contour
    #: set has no array to refill, while a composite's pixels are a frozen three-channel stretch and so are
    #: not the band values a frame carries at all.
    UNUPDATABLE = {
        "contour": "a contour set, which is cut from the values rather than handed new ones",
        "contourf": "a contour set, which is cut from the values rather than handed new ones",
        "rgb_composite": "a frozen three-channel stretch rather than a band's values",
        "hsv_composite": "a frozen three-channel stretch rather than a band's values",
    }

    #: Render options derived from *the frame being drawn*, which an artist kept across frames cannot
    #: follow: `scheme` cuts its class edges from that frame's own values, and `cyclic` adds a column, so
    #: the array stops matching the grid the artist was built on. Either sends the animation down the
    #: redrawing path.
    #:
    #: `cyclic`'s mechanism, measured: a `(60, 120)` band drawn with `cyclic=True` leaves a `(60, 121)`
    #: image, while :meth:`apply` reads the next frame through
    #: :func:`~digitalearth.static.maps.raster._field_source`, which does not close the seam and so hands
    #: back `(60, 120)`. Pointed at that artist, :meth:`apply` answers `False` and logs
    #: "animation: frame 1 holds a (60, 120) grid where the first frame held (60, 121), so the axes is
    #: cleared and every layer redrawn from here on" — the degrade this entry gets ahead of.
    #:
    #: A stated `center=` (or `color_scale="midpoint", midpoint=`) reads the same way and is deliberately
    #: **not** listed: its diverging ramp is a colour decision, and
    #: :meth:`AnimationMixin._freeze_animation_ramp` takes it once over the stack's shared scale before any
    #: frame is drawn — so there is nothing left for a frame to restate, and listing it here would buy
    #: agreement between the paths at the price of a ramp that flickers along the clip.
    #: Measured over a 2-frame stack running -5..5 then 1..5, drawn with `center=0.0`: the colormap per
    #: frame is `['viridis-diverging', 'viridis-diverging']` under both `update='redraw'` and
    #: `update='auto'`. `robust`, `vmin`/`vmax`, `cmap`, `norm` and the `missing`/`over`/`under` colours
    #: need no entry either: each is settled before the frames run, and the two paths were re-measured to
    #: agree on all of them — per frame, the colormap name, the clim, the norm class and the bad/over/under
    #: colours of the artist each path hands back are identical for `robust=True`, for an explicit
    #: `vmin`/`vmax`, for `cmap="magma"`, for `norm=LogNorm(...)` and for
    #: `missing="black", over="red", under="blue"`.
    PER_FRAME_OPTS = ("scheme", "cyclic")

    #: The accepted spellings of :meth:`AnimationMixin.animate`'s ``update``: take the cheapest path that is
    #: correct for the call (``"auto"``), insist on the in-place one (``"in_place"``, which refuses rather
    #: than falls back), or always clear and rebuild (``"redraw"``, what every animation did before ST-9).
    MODES = ("auto", "in_place", "redraw")

    def __init__(
        self,
        scene: Any,
        frames: Sequence[Any],
        opts: Mapping[str, Any],
        *,
        kind: str,
        mode: str = "auto",
        titles: Optional[Sequence[str]] = None,
    ):
        """Resolve the strategy from what the call asks for and what the renderer can do.

        Args:
            scene: The map being animated.
            frames: The animation's frames.
            opts: The render options every frame is drawn with; read for the band and for the options that
                block the in-place path.
            kind: The renderer the frames are drawn with.
            mode: The caller's ``update``, one of :attr:`MODES`.
            titles: The per-frame titles, or ``None``.

        Raises:
            ValueError: when `mode` is not one of :attr:`MODES`; and, **on the in-place path only**, when
                `opts` names a band that is not a whole number of 1 or more (from :func:`_animated_band`,
                so the refusal is the same one a still makes). A blocked strategy does not read the band
                at all — see :attr:`band` — so a bad one gets past this constructor there. Measured:
                `FrameUpdate(scene, frames, {"band": 0}, kind="imshow")` refuses, while the same call with
                `kind="contour"` or `kind="rgb_composite"` does not and leaves `band` at its default.

                Where a blocked strategy ends up refusing anyway is
                :meth:`AnimationMixin._prime_animation` — but only when it has a reason to read the band:
                the stack scan (a missing `vmin` or `vmax`), a `colorbar=True`, or a stated `center=`.
                Give a blocked call both bounds and neither of the other two and `band=0` reaches the
                frame loop. Measured on this stack, `Map.animate(frames, ...)`:
                `kind="contourf", band=0` refuses; `kind="contourf", band=0, vmin=0.0, vmax=1.0` does
                **not**; adding `colorbar=True` or `center=0.5` to that call refuses again; and
                `kind="imshow", band=0, vmin=0.0, vmax=1.0, update="redraw"` does not either, because
                `update="redraw"` is itself recorded as a blocker. A composite never reads the value at
                all: it names `bands`, not `band`.
        """
        if mode not in self.MODES:
            raise ValueError(
                f"unknown update mode {mode!r}; choose one of {self.MODES}"
            )
        self.scene = scene
        self.frames = frames
        self.kind = kind
        self.mode = mode
        self.titles = titles
        blocker = self._blocked_by(kind, opts)
        if blocker is None and mode == "redraw":
            blocker = "update='redraw' asks for the axes to be cleared and every layer rebuilt"
        self.blocker: Optional[str] = blocker
        # Read only on the in-place path, so a blocked strategy leaves it at the default rather than
        # validating a band no frame of it will use: a composite names `bands`, not `band`, and a redrawn
        # frame has its band checked where every still checks it (`_prime_animation`).
        self.band: int = DEFAULT_BAND if blocker else _animated_band(dict(opts))

    @classmethod
    def _blocked_by(cls, kind: str, opts: Mapping[str, Any]) -> Optional[str]:
        """Return why ``kind`` drawn with ``opts`` cannot update one artist, or ``None`` when it can.

        Args:
            kind: The renderer the frames are drawn with.
            opts: The render options every frame is drawn with.

        Returns:
            A sentence naming the blocker, or ``None``.
        """
        if kind not in cls.SETTERS:
            drawn = cls.UNUPDATABLE.get(kind, "no artist whose values can be replaced")
            return f"kind={kind!r} draws {drawn}"
        for name in cls.PER_FRAME_OPTS:
            if opts.get(name):
                return (
                    f"{name}={opts[name]!r} is derived from the frame being drawn, which a kept artist "
                    "cannot follow"
                )
        return None

    @property
    def in_place(self) -> bool:
        """Whether the frames after the first update one artist instead of clearing the axes.

        Returns:
            ``True`` when nothing blocks the in-place path.

        Examples:
            - It is the one question :attr:`blocker` answers twice over, so the two always agree:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.animation import FrameUpdate
                >>> scene = Map(crs=4326)
                >>> FrameUpdate(scene, [], {}, kind="pcolormesh").in_place
                True
                >>> FrameUpdate(scene, [], {}, kind="rgb_composite").in_place
                False
                >>> FrameUpdate(scene, [], {}, kind="imshow", mode="redraw").in_place
                False
                >>> scene.close()

                ```
        """
        return self.blocker is None

    def settle(self, *, blit: bool) -> Optional["FrameUpdate"]:
        """Refuse what this call cannot have, and report the strategy its frame loop should run.

        Args:
            blit: Whether the caller asked for blitting.

        Returns:
            ``self`` when the frames update in place, or ``None`` when every frame must be redrawn — which
            is what :meth:`AnimationMixin._animate_frames` takes as "clear and rebuild each time".

        Raises:
            ValueError: when ``blit=True`` or ``update="in_place"`` was asked for and this animation cannot
                be drawn that way, with :attr:`blocker` as the reason; when ``blit=True`` is combined
                with per-frame ``titles``, which blitting cannot repaint; and when either of those two —
                ``blit=True`` **or** ``update="in_place"`` — is asked for a stack that would take the clip
                into the rebuilding path at playback, with :meth:`_stack_blocker` as the reason. The two
                modes that never promised the fast path, ``"auto"`` and ``"redraw"``, are not refused for
                such a stack and do not pay the read either (round 2, M4).

        Examples:
            - The two answers the frame loop reads: `self` for a call nothing blocks, or `None` for
              "clear and rebuild each time" — which is what `update="auto"` falls back to once something
              does block it:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.animation import FrameUpdate
                >>> scene = Map(crs=4326)
                >>> plan = FrameUpdate(scene, [], {}, kind="imshow")
                >>> plan.settle(blit=False) is plan
                True
                >>> print(FrameUpdate(scene, [], {}, kind="contour").settle(blit=False))
                None
                >>> scene.close()

                ```
            - Each refusal names what the caller asked for and why it cannot be had:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.animation import FrameUpdate
                >>> scene = Map(crs=4326)
                >>> refused = FrameUpdate(scene, [], {}, kind="contour", mode="in_place")
                >>> refused.settle(blit=False)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: update='in_place' is not available for this animation: kind='contour' draws...
                >>> titled = FrameUpdate(scene, [], {}, kind="imshow", titles=["Jan"])
                >>> titled.settle(blit=True)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: blit=True cannot be combined with titles=...
                >>> scene.close()

                ```
        """
        if blit and self.blocker is not None:
            raise ValueError(
                f"blit=True needs frames that update one artist in place, and this animation cannot: "
                f"{self.blocker}. Blitting leaves everything the frame function does not return standing, "
                "so a rebuilt frame would stack on the one before it"
            )
        if blit and self.titles is not None:
            raise ValueError(
                "blit=True cannot be combined with titles=: matplotlib repaints only the axes box when it "
                "blits, and the title is drawn above it, so every frame would show the first frame's "
                "title. Drop one of the two"
            )
        if self.blocker is not None:
            if self.mode == "in_place":
                raise ValueError(self._in_place_refusal(self.blocker))
            logger.info(
                "animate: clearing the axes and redrawing every frame — %s",
                self.blocker,
            )
            return None
        # Nothing the kind and the options decide stands in the way, so the *stack* is read — but only
        # for the two callers that promised not to degrade. `"auto"` and `"redraw"` pay neither the read
        # nor the refusal: clearing and rebuilding repaints the whole axes, so for them a frame on
        # another grid is a slower path rather than a broken picture or a broken promise.
        if blit or self.mode == "in_place":
            standing = self._stack_blocker()
            if standing is not None:
                if blit:
                    raise ValueError(
                        f"blit=True needs every frame to refill the artist the first frame drew, and "
                        f"this stack does not: {standing}. Blitting leaves everything the frame function "
                        "does not return standing, so a rebuilt frame would stack on the one before it. "
                        "Drop blit=True, or animate frames that share one grid"
                    )
                raise ValueError(self._in_place_refusal(standing))
        return self

    @staticmethod
    def _in_place_refusal(reason: str) -> str:
        """Return the sentence ``update="in_place"`` is refused with, for ``reason``.

        One phrasing for both halves of the refusal, so the caller meets the same words whether the
        in-place path is unavailable because of the ``kind`` and the options (:attr:`blocker`, a static
        fact) or because of the stack (:meth:`_stack_blocker`, read once before the first frame). They are
        one refusal to a caller — ``update="in_place"`` could not be honoured — and differ only in which
        reason is named.

        Args:
            reason: Why the in-place path is unavailable, as a sentence.

        Returns:
            The refusal message.

        Examples:
            - The mode that was asked for opens the sentence, the reason is quoted as it stands, and the
              way out closes it:
                ```python
                >>> from digitalearth.static.maps.animation import FrameUpdate
                >>> message = FrameUpdate._in_place_refusal("frame 1 is on another grid")
                >>> message.split(":")[0]
                "update='in_place' is not available for this animation"
                >>> message.split(". ")[-1]
                "Leave update at 'auto' to redraw the frames instead"

                ```
        """
        return (
            f"update='in_place' is not available for this animation: {reason}. Leave update at 'auto' "
            "to redraw the frames instead"
        )

    def _display_grid(self, index: int) -> Optional[Tuple[int, ...]]:
        """Return the grid frame ``index`` is **drawn** on, or ``None`` when it draws nothing.

        Read through :func:`~digitalearth.static.maps.raster._field_source`, the same reader
        :meth:`apply` compares against, so the answer here and the answer at playback cannot disagree.
        The raster's own shape is not that answer: a frame is warped into the display CRS and decimated to
        the canvas before it reaches an artist, so two frames of one source shape can be drawn on two
        grids and two source shapes can be drawn on one.

        Args:
            index: Which frame to measure.

        Returns:
            The drawn array's shape, or ``None`` for a frame the display CRS cannot show — which draws
            nothing and so matches no grid.
        """
        try:
            src, _ = _field_source(self.scene, self.frames[index], self.band)
        except OffLimbError:
            return None
        return tuple(np.shape(src.z.values))

    def _stack_blocker(self) -> Optional[str]:
        """Return why this stack cannot be held in place all the way through, or ``None`` when it can.

        :meth:`settle` refuses ``blit=True`` and ``update="in_place"`` for what the *kind and the options*
        make impossible, which is a static fact. A stack can take the same animation into the
        clear-and-rebuild path at **runtime** instead: a frame on another grid cannot refill the artist
        (:meth:`apply` answers ``False``, and :meth:`AnimationMixin._animate_frames` rebuilds every frame
        from there on), and a first frame the display CRS cannot show leaves no artist to keep at all, so
        every frame draws itself. Under ``blit=True`` the rebuilt decoration is then absent from the artist
        list a blitted frame hands back and is never repainted — which is exactly what :meth:`settle`'s own
        refusal text says must not happen; under ``update="in_place"`` the picture is sound but the mode's
        promise is not, since insisting is the one thing that mode is for. Either way the stack is read
        here, before the first frame, and the caller is told.

        This costs one read of every frame, which is why it is behind a ``blit=True``/``"in_place"`` check
        rather than done for every animation. ``"auto"`` and ``"redraw"`` need no such scan: clearing and
        rebuilding repaints the whole axes, and neither mode promised otherwise, so the fallback is a
        slower path rather than a broken picture or a broken contract (round 2, M4).

        Returns:
            A sentence naming what stands in the way, or ``None``.
        """
        first = self._display_grid(0) if self.frames else None
        if self.frames and first is None:
            return (
                "its first frame draws nothing — the display CRS cannot show it — so there is no artist "
                "to keep and every frame after it draws itself"
            )
        for index in range(1, len(self.frames)):
            grid = self._display_grid(index)
            if grid is not None and grid != first:
                return (
                    f"frame {index} is drawn on a {grid} grid where the first frame is drawn on {first}, "
                    "so the axes would be cleared and every layer redrawn from there on"
                )
        return None

    def apply(self, index: int, artist: Any) -> bool:
        """Give ``artist`` frame ``index``'s values instead of drawing that frame's layer again.

        The values are read through the field drawer's own reader
        (:func:`~digitalearth.static.maps.raster._field_source`), not through a second pipeline of this
        class's own: a frame must arrive decimated, warped and masked exactly as the drawer would have
        delivered it, or the kept artist would show a different picture from the one a redraw puts up.

        The title goes with it. A title is the one piece of per-frame *decoration*, so an updated frame has
        to restate it just as a redrawn one does — leaving it to the first frame is how an in-place
        animation came out with every frame titled "Jan". It is set straight on the axes rather than
        through :meth:`~digitalearth.static.scene.Scene.set_title`, which would also write it into the
        figure's description: a frame's caption is not the figure's heading, and ``titles=`` leaves that
        heading to the caller *in the record*. On the axes there is only one title to draw, so a
        recorded heading is painted over here — :meth:`AnimationMixin.animate` says so once, at
        ``WARNING``, when a call asks for both (round 2, L11).

        Args:
            index: Which frame to show.
            artist: The artist the first frame drew — an ``AxesImage`` or a ``QuadMesh``, by :attr:`kind`.

        Returns:
            ``True`` when the artist now holds this frame, ``False`` when it cannot and the caller must
            clear and redraw instead — a frame whose grid differs from the one the artist was built on,
            which ``set_data`` would accept and then draw at the wrong resolution.

        Raises:
            OffLimbError: only under the scene's ``strict=True``, from
                :meth:`~digitalearth.static.maps.base.GeoLayerBase._skipped_off_limb` — the same refusal a
                strict still makes. Otherwise a frame the display CRS cannot show hides the artist for that
                frame, because "draws nothing" has to mean nothing *is* drawn rather than the previous
                frame staying up.

        Examples:
            - The artist the first frame drew is given the second frame's values, with nothing else on
              the axes touched:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.animation import FrameUpdate
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> frames = [Dataset.from_array(arr=np.full((60, 120), v, "float32"), geo_ref=geo)
                ...           for v in (1.0, 2.0)]
                >>> m = Map(crs=4326)
                >>> _ = m.field(frames[0], name="f")
                >>> image = m.artist("f")
                >>> float(np.asarray(image.get_array()).max())
                1.0
                >>> FrameUpdate(m, frames, {}, kind="imshow").apply(1, image)
                True
                >>> float(np.asarray(image.get_array()).max())
                2.0
                >>> m.close()

                ```
            - A frame on a different grid answers ``False`` instead of refilling the artist at the wrong
              resolution, which is the frame loop's signal to clear and redraw from there on:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.animation import FrameUpdate
                >>> fine = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> coarse = GeoReference(geo=(-180.0, 6.0, 0.0, 90.0, 0.0, -6.0), epsg=4326)
                >>> frames = [
                ...     Dataset.from_array(arr=np.full((60, 120), 1.0, "float32"), geo_ref=fine),
                ...     Dataset.from_array(arr=np.full((30, 60), 9.0, "float32"), geo_ref=coarse),
                ... ]
                >>> m = Map(crs=4326)
                >>> _ = m.field(frames[0], name="f")
                >>> FrameUpdate(m, frames, {}, kind="imshow").apply(1, m.artist("f"))
                False
                >>> m.close()

                ```
        """
        method = _KIND_METHODS.get(self.kind, (self.kind, {}))[0]
        try:
            src, _ = _field_source(self.scene, self.frames[index], self.band)
        except OffLimbError:
            artist.set_visible(False)
            self.scene._skipped_off_limb(f"Map.{method}()")  # raises under strict
            return True
        values = src.z.values
        held = artist.get_array()
        if held is not None and np.shape(held) != np.shape(values):
            logger.warning(
                "animation: frame %s holds a %s grid where the first frame held %s, so the axes is cleared "
                "and every layer redrawn from here on",
                index,
                np.shape(values),
                np.shape(held),
            )
            return False
        getattr(artist, self.SETTERS[self.kind])(values)
        artist.set_visible(True)
        if self.titles is not None:
            # Drawn, not recorded. `Scene.set_title` writes the heading onto the figure's description
            # (ST-18), and a frame's caption is not the figure's heading: restating it through there left
            # a clip describing itself as headed with its *last* frame's caption, over whatever heading
            # the caller had set.
            self.scene.ax.set_title(self.titles[index])
        return True


class AnimationMixin(_MixinBase):
    """Stack animation and globe rotation for :class:`~digitalearth.static.map.Map`."""

    def _animate_frames(
        self,
        draw_one: Callable[[int], Any],
        n_frames: int,
        fps: float,
        *,
        blit: bool = False,
        updates: Optional[FrameUpdate] = None,
    ) -> FuncAnimation:
        """Drive ``n_frames`` of ``draw_one(i)`` on this Map's axes as a :class:`FuncAnimation`.

        The first frame clears the axes, resets the per-frame layer/frame state, calls ``draw_one(i)`` to
        draw frame ``i``, then (on a globe) sets the full-domain extent and applies the projection frame.

        Every frame after it does the same **unless** an ``updates`` strategy was given, in which case the
        artist the first frame drew is given the new frame's values and nothing else is touched (ST-9). That
        is what keeps the decoration, the projection frame and the view from being thrown away and rebuilt
        sixty times, and what lets the frame function hand matplotlib the one artist it changed — without
        which ``blit=True`` has nothing to blit over.

        No colorbar is added per frame — pass a fixed ``vmin``/``vmax`` to keep colours stable instead.

        Args:
            draw_one: Draws frame ``i`` from scratch, returning the data artist it drew (or ``None``).
            n_frames: How many frames the animation runs for.
            fps: Frames per second, as the inter-frame interval.
            blit: Passed to :class:`FuncAnimation`. Only ever ``True`` alongside ``updates``, and only for
                a stack :meth:`FrameUpdate.settle` has already read as one every frame can update:
                ``FuncAnimation._blit`` cannot be turned off mid-run, so the degrade below must be
                unreachable under blitting rather than compensated for.
            updates: The :class:`FrameUpdate` to run for the frames after the first, or ``None`` to redraw
                every frame. A ``False`` from :meth:`FrameUpdate.apply` degrades the rest of the animation
                to redrawing, rather than letting one odd frame fail the run.

        Returns:
            The :class:`FuncAnimation`, also held on ``self._animation``.
        """
        kept: List[Any] = []
        strategy = {"updates": updates}

        def _redraw(i: int) -> List[Any]:
            self.ax.clear()
            self._reset_layers()
            self._framed = False
            drawn = draw_one(i)
            if self.globe:
                self.set_global()
                self._apply_frame()
            return [] if drawn is None else [drawn]

        def _f(i: int) -> List[Any]:
            # `kept` empty means the first frame drew no data artist at all — an off-limb frame on a globe —
            # so there is nothing to update and each frame draws itself.
            running = strategy["updates"]
            if running is not None and kept:
                if running.apply(i, kept[0]):
                    return list(kept)
                strategy["updates"] = None
            kept[:] = _redraw(i)
            return list(kept)

        anim = FuncAnimation(
            self.fig, _f, frames=n_frames, interval=1000.0 / fps, blit=blit
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
                >>> m.close()

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
                >>> bare = Map(crs=4326)
                >>> bare.save_animation("clip.gif")
                Traceback (most recent call last):
                    ...
                RuntimeError: no animation to save; call animate() or rotate() first
                >>> bare.close()

                ```
        """
        anim = self._animation
        if anim is None:
            raise RuntimeError("no animation to save; call animate() or rotate() first")
        rate = self._animation_fps if fps is None else fps
        return save_animation(anim, path, fps=rate, gif=gif, **kwargs)

    def _stack_clim(
        self, datasets: Sequence[Any], band: int = DEFAULT_BAND
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
            cannot be warped draws nothing and so contributes no colour range. Both come from the frozen
            :class:`~digitalearth.base.spec.scale.Scale` the measurement builds, so a stack of identical
            frames reports a domain with width rather than the zero-width pair a normaliser clamps.
        """
        return frozen_scale(self._measured_clim(datasets, band=band)).as_limits()

    def _measured_clim(
        self, datasets: Sequence[Any], band: int = DEFAULT_BAND
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

    def _frame_values(
        self, datasets: Sequence[Any], band: int = DEFAULT_BAND
    ) -> Iterator[Any]:
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
        self, dataset: Any, views: Sequence[Any], band: int = DEFAULT_BAND
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
            The union goes through the same frozen :class:`~digitalearth.base.spec.scale.Scale` as the
            single-view scan, so a sweep over a constant cell answers the same domain that stack does.
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
            union = (
                (min(lo for lo, _ in bounds), max(hi for _, hi in bounds))
                if bounds
                else None
            )
            return frozen_scale(union).as_limits()
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

        What the scan reports is a :class:`~digitalearth.base.spec.scale.Scale` domain, so a stack whose
        frames all hold one value comes back widened rather than zero-width. That widening applies to the
        *scan*, not to the caller: a bound already in ``opts`` is still kept verbatim, so passing
        ``vmin=0.0`` over a stack of constant 7 resolves to ``(0.0, 8.0)`` — the scan's widened upper bound
        beside the caller's floor.

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
                >>> scene = Map(crs=4326)
                >>> scene._resolve_animation_clim(frames, opts)
                >>> opts["vmin"], opts["vmax"]
                (500.0, 700.0)
                >>> scene.close()

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
                >>> scene = Map(crs=4326)
                >>> scene._resolve_animation_clim(frames, opts)
                >>> opts["vmin"], opts["vmax"]
                (0.0, 700.0)
                >>> scene.close()

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
        """Resolve the one shared colour treatment into `opts`, and if asked add the static colorbar.

        The setup shared by :meth:`animate` and :meth:`rotate`, branching on what `kind` draws:

        - A **scalar field** gets a missing `vmin`/`vmax` filled once from the stack so colours do not
          flicker between frames, then optionally one persistent colorbar.
        - A **composite** (:data:`_COMPOSITE_KINDS`) gets frozen per-channel stretch `limits` instead. A
          clim is meaningless for an RGB image — the composite runs its own per-channel stretch, so nothing
          in its render path reads `vmin`/`vmax`. They are still forwarded to the glyph with every other
          kwarg; they simply have no colour scale to move. Measured, the first frame's image reports the
          **same** `get_clim()` with and without `vmin=-100.0, vmax=100.0`, whichever stack it is drawn
          from — `(0.0, 1.0)` on a two-frame true-colour stack, the unit range the frozen three-channel
          stretch maps into, and `(0.0, 0.0)` on the degenerate constant stack the `limits` measurement
          below uses, whose channels stretch to a single number. It is the *agreement* that is the claim;
          the pair itself is the stretch's, not the bounds'. There is likewise no single mappable to key
          a colorbar to, so an explicit `colorbar=True` is refused rather than answered with a useless
          bar.

        Real limits already in `opts` are kept, so a caller can pass their own `limits=` to override the
        scan. A `limits` of `None` counts as absent and is filled, matching how
        :meth:`_resolve_animation_clim` treats a `None` `vmin`/`vmax` — an explicit `limits=None` is what a
        wrapper forwarding an optional passes, and silently skipping the freeze there would put the
        flicker back with nothing to notice. Measured over a stack running 1 then 5 in every channel:
        `limits` absent and `limits=None` both resolve to `[(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)]`, while
        `limits=[(0.0, 1.0)] * 3` is left exactly as passed.

        Args:
            datasets: The animation's frames — the stack the shared treatment is measured over.
            opts: The render options every frame will be drawn with. The resolved `vmin`/`vmax` (scalar) or
                `bands`/`limits` (composite) are written back into it, and so is the `cmap` a colorbar or a
                stated `center=` resolves.
            kind: The renderer the frames are drawn with, which decides the branch.
            colorbar: Whether to add one static colorbar. Refused on a composite `kind`.
            cbar_label: Caller-supplied colorbar label, or `None` to fall back to the variable's `units`.
            views: The display CRSs :meth:`rotate` will sweep, or `None` for an :meth:`animate` whose
                frames share the current CRS.

        Raises:
            ValueError: when `colorbar=True` is combined with a composite `kind`; when a composite's
                `bands` does not hold exactly three indices (measured:
                `rgb_composite() needs exactly three bands, got 2: (1, 2)`); and, whenever a band is read
                at all here, when `opts` names one that is not a whole number of 1 or more.
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
        # Read off a copy: `stated_on` takes the extreme-colour keywords out of the dict it is given, and
        # every frame still needs them in `opts`. A `scheme=` is left out of it: it cuts its class colours
        # from the frame being drawn, which is why it blocks the in-place path outright
        # (:attr:`FrameUpdate.PER_FRAME_OPTS`) — both paths then redraw and so already agree, and pinning a
        # ramp there would take the shared categorical palette away from a band of class codes.
        colors = None if opts.get("scheme") else FieldColors.stated_on(dict(opts))
        center = None if colors is None else colors.center
        # One read of the stack's variable serves both the ramp and the bar, and is paid for only when one
        # of them is asked for — resolving a style costs a frame read.
        style = (
            self._frame_style(datasets, _animated_band(opts))
            if colorbar or center is not None
            else {}
        )
        if colors is not None and center is not None:
            self._freeze_animation_ramp(opts, colors, style)
        if colorbar:
            self._animation_colorbar(opts, cbar_label, style)

    def _freeze_animation_ramp(
        self, opts: dict, colors: FieldColors, style: Mapping[str, Any]
    ) -> None:
        """Resolve the diverging ramp a stated ``center=`` asks for **once over the stack**.

        The ramp is a colour decision, and an animation takes its colour decisions once: a frame drawing
        itself asks :meth:`~digitalearth.static.maps.raster.FieldColors.ramp_over` whether *that frame*
        straddles the centre, which made the ramp flicker between diverging and sequential across a clip —
        and made the answer depend on ``update=``, because an artist kept across frames froze frame 0's
        answer while a redrawn one re-decided it. Asking once, of the shared ``vmin``/``vmax``
        :meth:`_resolve_animation_clim` has just measured, is what makes the two paths one animation.

        The resolved colormap is written into ``opts["cmap"]``. For the ``center=`` spelling the symmetric
        limits cleopatra would compute are then baked into ``opts["vmin"]``/``opts["vmax"]`` — reused from
        cleopatra's own rule through :meth:`~digitalearth.static.maps.raster.FieldColors.centered_limits`,
        not recomputed here — and the ``center=`` keyword is dropped, so each frame reaches ``ramp_over``
        with no centre, draws the same widened domain, and adds no second warning: the one above is the
        whole clip's (issue #390). cleopatra does not widen the limits for a ``color_scale="midpoint"``
        scale, so there is nothing to bake there and the keyword stays; those frames decline through
        cleopatra's own named-cmap no-op, the colormap already being the one resolved here.

        Args:
            opts: The render options every frame is drawn with. Read for ``cmap``, ``vmin`` and ``vmax``;
                the resolved colormap is written back, and for a ``center=`` call ``vmin``/``vmax`` are
                rewritten to the symmetric limits and ``center`` is removed.
            colors: The colour decisions the call stated, carrying the centre and which spelling named it.
            style: The animated variable's :func:`~digitalearth.base.autostyle.auto_style` dict, the source
                of the colormap a call that named none resolves to.

        Examples:
            - A stack straddling the centre gets one diverging ramp, built from the resolved map's ends:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.raster import FieldColors
                >>> scene = Map(crs=4326)
                >>> opts = {"vmin": -5.0, "vmax": 5.0}
                >>> scene._freeze_animation_ramp(opts, FieldColors(center=0.0), {"cmap": "viridis"})
                >>> opts["cmap"].name
                'viridis-diverging'
                >>> scene.close()

                ```
            - A caller's own colormap is left exactly as it was asked for:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.raster import FieldColors
                >>> scene = Map(crs=4326)
                >>> opts = {"vmin": -5.0, "vmax": 5.0, "cmap": "RdBu_r"}
                >>> scene._freeze_animation_ramp(opts, FieldColors(center=0.0), {"cmap": "viridis"})
                >>> opts["cmap"]
                'RdBu_r'
                >>> scene.close()

                ```
            - An off-band ``center=`` bakes the widened limits in and drops the keyword, so the frames need
              no centre to draw the same domain:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> from digitalearth.static.maps.raster import FieldColors
                >>> scene = Map(crs=4326)
                >>> opts = {"vmin": -3.0, "vmax": 8.0, "center": 100.0, "cmap": "RdBu_r"}
                >>> colors = FieldColors(center=100.0, from_center=True)
                >>> scene._freeze_animation_ramp(opts, colors, {})
                >>> (opts["vmin"], opts["vmax"]), "center" in opts
                ((-3.0, 203.0), False)
                >>> scene.close()

                ```
        """
        requested = opts.get("cmap")
        resolved = requested or style.get("cmap") or DEFAULT_FIELD_CMAP
        opts["cmap"] = colors.ramp_over(
            resolved,
            np.asarray([opts["vmin"], opts["vmax"]], dtype="float64"),
            requested,
        )
        if colors.from_center:
            opts["vmin"], opts["vmax"] = colors.centered_limits(
                opts["vmin"], opts["vmax"]
            )
            del opts[CENTER_KEY]

    def _draw_animation_frame(
        self,
        data: Any,
        kind: str,
        opts: dict,
        *,
        ocean: bool,
        coastlines: bool,
        title: Optional[str] = None,
    ) -> Any:
        """Draw one animation frame: optional ocean disc, the field, optional coastlines, optional title.

        The per-frame body shared by :meth:`animate` and :meth:`rotate`. Ocean fill and coastlines are
        decoration: the ocean disc is drawn only on a globe, and a coastline failure (no network/data) is
        swallowed so the animation still renders. So is the title, which is drawn on the axes and left out
        of the figure's description — see :meth:`FrameUpdate.apply`, which restates it the other way.

        Returns:
            The data layer's mappable — the artist a later frame updates in place rather than drawing again,
            and the one a blitted frame repaints. ``None`` for a frame the display CRS cannot show, which
            draws nothing.
        """
        if ocean and self.globe:
            self.ocean()
        method, picked = _KIND_METHODS.get(kind, (kind, {}))
        # The builders hand back the map rather than the artist since ST-20, so the frame's mappable is read
        # off the layer the call has just drawn. It is the **last** layer id: the ocean disc above is drawn
        # first, and the coastlines below have not run yet, so the data layer is the newest one on the
        # figure. A frame the display CRS cannot show draws nothing, is forgotten again, and so leaves
        # `layer_ids` as it was — which is the `None` this used to get back from the builder.
        before = set(self.layer_ids)
        getattr(self, method)(data, **picked, **opts)
        added = [layer_id for layer_id in self.layer_ids if layer_id not in before]
        record = self._renderer.drawn.get(added[-1]) if added else None
        drawn = None if record is None else record.artist
        if coastlines:
            try:
                self.coastlines()
            except Exception:  # network/data unavailable — decoration is best-effort
                pass
        if title is not None:
            # On the axes only, for the reason `FrameUpdate.apply` gives: the same caption reaches the
            # picture by this path, and the two must be as quiet as each other about the description.
            self.ax.set_title(title)
        return drawn

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
        update: str = "auto",
        blit: bool = False,
        **kwargs: Any,
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

        A stated ``center=`` belongs to that one treatment too. The diverging ramp it asks for is resolved
        **once, over the shared scale**, rather than by each frame against its own values: a stack that
        straddles the centre diverges for the whole clip, and one that does not keeps the sequential ramp
        and says so once. Deciding it per frame made the ramp flicker between the two along a clip — and
        made the picture depend on ``update=``, since an artist kept across frames froze the first frame's
        answer while a redrawn one re-asked it.

        Either scan measures frames **as they will be drawn**, reprojected into the display CRS, so the
        animation lands on the same scale as the equivalent still. On a globe that means the scale spans
        what the globe shows: an extreme value on the hidden hemisphere does not set the top of it.

        **Only what changes is redrawn (ST-9).** The first frame draws the whole scene; every frame after it
        gives the artist that frame drew the new frame's values (``AxesImage.set_data`` /
        ``QuadMesh.set_array``) and touches nothing else, so the ocean disc, the coastlines, the projection
        frame, the colorbar and the view survive instead of being thrown away and rebuilt per frame. That is
        a strategy, not a different picture: the frames are pixel-identical to the cleared-and-redrawn ones,
        which ``tests/static/test_animate_in_place.py`` compares buffer by buffer. ``update=`` chooses the
        path, and a ``kind`` or an option that cannot be updated this way (a contour set, a composite, a
        per-frame ``scheme=``) is named rather than animated wrongly. A frame the display CRS cannot show
        hides the artist for that frame, so an off-limb frame still draws nothing.

        Args:
            stack: An ordered collection of pyramids ``Dataset`` frames (e.g. a list, or a
                ``DatasetCollection`` datacube) — one raster per animation frame.
            kind: The method used to draw each frame — a scalar field (``"imshow"`` / ``"contourf"`` /
                ``"contour"`` / ``"pcolormesh"`` / ``"block"``) or a true/false-colour composite
                (``"rgb_composite"`` / ``"hsv_composite"``, which take ``bands=(r, g, b)`` through
                ``**kwargs`` to pick their channels).
            fps: Frames per second (sets the inter-frame interval). Defaults to :data:`DEFAULT_FPS`, the one
                rate every animation entry point — here, :meth:`rotate`, and the other backends — starts from.
            titles: Optional per-frame captions; must match the stack length when given. They are drawn
                on the axes and left out of the figure's description, so a heading
                :meth:`~digitalearth.static.scene.Scene.set_title` recorded survives in the record — but
                it is **painted over** on the axes by every frame, which is a disagreement between the
                drawn and the described title and is said once, at ``WARNING``, when both are asked for
                (round 2, L11).
            ocean: When True, fill the ocean disc behind each frame (globe maps only).
            coastlines: When True, overlay coastlines each frame (best-effort; ignored if unreachable).
            colorbar: When True, add one static colorbar (drawn once, not per frame) using the shared
                colour scale. Not available on a composite ``kind`` — an RGB image has no scalar mappable.
            cbar_label: Optional label for the colorbar.
            update: How the frames after the first are drawn — one of :attr:`FrameUpdate.MODES`. ``"auto"``
                (default) updates the artist the first frame drew when the ``kind`` and the options allow
                it and clears the axes when they do not, logging which — including mid-clip, at
                ``WARNING``, for a stack whose later frame is drawn on another grid. ``"in_place"``
                insists, and refuses the call rather than falling back: naming
                :attr:`FrameUpdate.blocker` for what the ``kind`` and the options settle, and — read off
                the stack once, before the first frame — naming the frame and the two grids for a stack
                that would leave the fast path mid-clip, or the first frame the display CRS cannot show.
                ``"redraw"`` always clears, which is what every animation did before ST-9, and is refused
                for nothing.
            blit: When True, build the animation with matplotlib's blitting, which repaints only the artist
                each frame returns. It requires the in-place path and is refused without it, and it is
                refused alongside ``titles`` (see the note below). It is also refused for a **stack** that
                would leave that path mid-clip — one whose frames are not all drawn on a single grid, or
                whose first frame the display CRS cannot show, either of which clears the axes and rebuilds
                every layer from there on. That is read off the stack once, before the first frame, so the
                refusal reaches the caller rather than the playback. It changes **playback only**: saving a
                clip draws every frame in full whatever this says.
            **kwargs: Forwarded to the ``kind`` method. A scalar field takes ``band``, ``cmap``, ``vmin``,
                ``vmax`` and the rest of its styling — the shared colour scale is measured from that same
                ``band`` (1 by default); a composite takes ``bands``, ``mask_nodata`` and ``limits``.
                ``vmin``/``vmax`` are accepted on a composite because they reach the glyph like any other
                kwarg, but an RGB image has no colour scale for them to move — use ``limits``.

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``len(stack)`` frames (also kept on
            ``self._animation`` so it is not garbage-collected before you save/display it).

        Raises:
            ValueError: if `kind` is not a known renderer, `update` is not one of
                :attr:`FrameUpdate.MODES`, `stack` is empty, `titles` is given with a mismatched length,
                `colorbar=True` is combined with a composite `kind`, a composite's `bands` does not hold
                exactly three indices (measured: `rgb_composite() needs exactly three bands, got 2:
                (1, 2)`), a band that is read is not a whole number of 1 or more, or `blit=True` /
                `update="in_place"` is asked for where the frames cannot update one artist — including,
                for **either** of those two, a stack whose frames are not all drawn on one grid or whose
                first frame the display CRS cannot show.

        Note:
            ``blit=True`` repaints only what the frame function returns, which is the data artist and
            nothing else. A per-frame **title** is therefore refused: matplotlib blits the axes bounding box
            and the title is drawn above it, so it would hold the first frame's text. A colorbar is fine —
            an animation's colorbar is static by construction, drawn once on its own axes.

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
                >>> m.close()

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
                >>> m.close()

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
                >>> refused = Map(crs=4326)
                >>> refused.animate(stack, kind="rgb_composite", colorbar=True)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: colorbar=True is not supported for a 'rgb_composite' animation: ...
                >>> refused.close()

                ```
            - A field animation can be blitted, because its frames update one image:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> fields = [Dataset.from_array(arr=np.full((60, 120), v, "float32"), geo_ref=geo)
                ...           for v in (1.0, 2.0)]
                >>> scene = Map(crs=4326)
                >>> len(list(scene.animate(fields, fps=2, blit=True).new_frame_seq()))
                2
                >>> scene.close()

                ```
            - A contour animation cuts its lines from each frame, so blitting it is refused by name:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> geo = GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326)
                >>> stack = [Dataset.from_array(arr=np.full((60, 120), v, "float32"), geo_ref=geo)
                ...          for v in (1.0, 2.0)]
                >>> refused = Map(crs=4326)
                >>> refused.animate(stack, kind="contourf", blit=True)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: blit=True needs frames that update one artist in place, ...
                >>> refused.close()

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
        if titles is not None and self.title is not None:
            # Said here rather than where the caption is painted on, so one clip says it once however
            # many frames it runs for, and both `update=` paths say it alike.
            logger.warning(
                "animate: titles= captions each frame on the axes, which paints over the heading "
                "set_title() recorded (%r). The figure's description keeps that heading, so the drawn "
                "and the described title will disagree for the whole clip — drop one of the two",
                self.title,
            )
        updates = FrameUpdate(
            self, frames, kwargs, kind=kind, mode=update, titles=titles
        ).settle(blit=blit)
        self._prime_animation(
            frames, kwargs, kind=kind, colorbar=colorbar, cbar_label=cbar_label
        )

        def draw_one(i: int) -> Any:
            title = titles[i] if titles is not None else None
            return self._draw_animation_frame(
                frames[i], kind, kwargs, ocean=ocean, coastlines=coastlines, title=title
            )

        return self._animate_frames(
            draw_one, len(frames), fps, blit=blit, updates=updates
        )

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
        blit: bool = False,
        **kwargs,
    ) -> FuncAnimation:
        """Spin an orthographic globe over a single field by sweeping the centre longitude.

        Forces a globe map and redraws ``dataset`` on ``n_frames`` orthographic projections whose centre
        longitude steps a full 360 degrees from ``lon0``.

        **Terminal for this Map's projection:** `rotate` sets `globe=True` immediately and sweeps the
        display CRS (:attr:`crs`) *as the animation renders*, leaving the Map centred on the **final** frame.
        The two do not land together, which matters if you read :attr:`crs` back: measured on a Map built
        with `crs=4326`, `globe` is already `True` when `rotate` returns while `crs` is still `4326` — the
        scan restores what it borrowed — and only after the frames have played does `crs` read
        `+proj=ortho +lat_0=15.0 +lon_0=90.0 ...`, the last view of the sweep. Treat a rotated Map as
        consumed by the animation — create a fresh `Map` if you need the original projection.

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
            blit: Accepted only as ``False``. A rotation has no artist to blit over: it warps the data into
                a new orthographic view every frame, so the image, its extent and the globe's limb all
                change together and the axes is cleared and rebuilt each time. Declared rather than left to
                ``**kwargs`` so ``blit=True`` is refused here instead of reaching the renderer as unknown
                styling. :meth:`animate` is the method that blits.
            **kwargs: Forwarded to the ``kind`` method. A scalar field takes ``band``, ``cmap``, ``vmin``,
                ``vmax`` and the rest of its styling — the shared colour scale is measured from that same
                ``band`` (1 by default); a composite takes ``bands``, ``mask_nodata`` and ``limits``.
                ``vmin``/``vmax`` are accepted on a composite because they reach the glyph like any other
                kwarg, but an RGB image has no colour scale for them to move — use ``limits``.

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``n_frames`` frames (also kept on
            ``self._animation`` so it is not garbage-collected before you save/display it).

        Raises:
            ValueError: if `n_frames` is less than 1, `kind` is not a known renderer, `colorbar=True` is
                combined with a composite `kind`, a composite's `bands` does not hold exactly three indices
                (measured: `rgb_composite() needs exactly three bands, got 2: (1, 2)`), or `blit=True` is
                asked for.

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
                >>> m.close()

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
                >>> spun = Map(crs=4326)
                >>> anim = spun.rotate(rgb, kind="rgb_composite", n_frames=3)
                >>> len(list(anim.new_frame_seq()))
                3
                >>> spun.close()

                ```

        See Also:
            animate: Step through a stack of rasters instead of spinning one.
            save_animation: Write the returned animation to an mp4/GIF.
        """
        if n_frames < 1:
            raise ValueError("rotate needs n_frames >= 1")
        if blit:
            raise ValueError(
                "blit=True is not available for rotate(): every frame reprojects the data into a new "
                "orthographic view, so the image, its extent and the globe's limb all change together and "
                "no artist survives for blitting to leave standing. Use Map.animate(..., blit=True) for a "
                "stack drawn in one projection"
            )
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

        def draw_one(i: int) -> Any:
            self.crs = views[i]
            return self._draw_animation_frame(
                dataset, kind, kwargs, ocean=ocean, coastlines=coastlines
            )

        return self._animate_frames(draw_one, n_frames, fps)
