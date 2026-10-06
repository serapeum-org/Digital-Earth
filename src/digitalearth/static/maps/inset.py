"""inset — the locator map: a small map inside the main one, with the main map's extent marked on it.

A locator map answers one question — *where on the wider area is this?* — and it is geospatial composition
rather than rendering, so it belongs here: ``ax.inset_axes()`` is matplotlib's own, a borrowed axes is
already what :class:`~digitalearth.static.scene.Scene` draws into, and the only part neither of them
answers is placing one map's extent in another map's CRS. cleopatra and pyramids need nothing for it.

Two value objects carry the work, so neither the arithmetic nor the refusals live in the method:

- :class:`_ExtentBox` — the rectangle a map is looking at, in that map's own CRS, and how it lands in
  another CRS. **It reprojects its edges, not its corners**, which is the whole correctness question (see
  the class docstring for the measured polar case, and #375).
- :class:`_InsetFrame` — where the inset axes sits inside its parent, as a fraction of it.

:class:`InsetMixin` is then two methods: :meth:`~InsetMixin.mark_extent`, which draws one map's extent on
another, and :meth:`~InsetMixin.inset`, which builds the inset axes, puts reference geography in it and
marks the parent's extent — the one call.
"""

import logging
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Self, Sequence, Tuple

import numpy as np
from matplotlib.patches import Polygon
from pyramids.base.crs import reproject_coordinates

from digitalearth.base.spec.bounds import same_crs

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.map import Map
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

logger = logging.getLogger(__name__)

#: Samples per edge of an extent box. 21 is the count
#: :func:`~digitalearth.static.maps.decoration._edge_samples` uses for the neighbouring job (bounding a
#: reprojected box), so an outline drawn here and an envelope measured there agree about the same rectangle.
_EDGE_SAMPLES = 21

#: The named corners, as ``(right, upper)`` flags — 0 for the near edge of the axes, 1 for the far one.
_CORNERS: Dict[str, Tuple[int, int]] = {
    "upper right": (1, 1),
    "upper left": (0, 1),
    "lower right": (1, 0),
    "lower left": (0, 0),
}

#: Gap between the inset and the edge of the map it sits in, as a fraction of the map.
_CORNER_PAD = 0.03

#: The largest side a *named corner* can hold, since the pad is kept on both sides of the inset. Measured:
#: ``_InsetFrame.resolve("upper right", 1.0)`` used to answer ``(-0.03, -0.03, 1.0, 1.0)`` — inside the
#: documented ``(0, 1]`` and outside the map (round 1, L1). At this value the four corners all answer
#: ``(0.03, 0.03, 0.94, 0.94)``, which is the whole axes less its two pads.
_LARGEST_CORNER = 1.0 - 2.0 * _CORNER_PAD

#: The inset's side, as a fraction of the map it sits in.
_DEFAULT_SIZE = 0.28

#: Decimal places a placement is reported to by :meth:`_InsetFrame.as_bounds`. A thousandth of the axes is
#: well under a pixel, and it is also the floor a placement has to clear: a side that rounds to ``0.0``
#: here reaches ``ax.inset_axes`` as zero, whatever it was asked as (round 2, L10).
_PRECISION = 3

#: The reference geography a locator draws by default: filled land under a coastline.
_DEFAULT_REFERENCE: Tuple[str, ...] = ("land", "coastlines")

#: The Natural-Earth layers ``inset(reference=)`` may name. A whitelist rather than a plain attribute
#: lookup: the names are resolved to methods on the locator, so an unchecked one would reach *any*
#: attribute — ``reference=("close",)`` would close the figure instead of drawing on it.
_REFERENCE_LAYERS: Tuple[str, ...] = (
    "land",
    "ocean",
    "coastlines",
    "borders",
    "lakes",
    "rivers",
)

#: How the extent box is drawn unless the caller says otherwise: a hollow outline, over the geography.
#: Hollow because a filled box would cover exactly the part of the world the reader is being pointed at.
_BOX_STYLE: Dict[str, Any] = {
    "facecolor": "none",
    "edgecolor": "red",
    "linewidth": 1.2,
    "zorder": 4.0,
}


@dataclass(frozen=True, eq=False)
class _ExtentBox:
    """The rectangle a map is looking at, in that map's own CRS, and how it lands in another one.

    The four numbers and the CRS they are in are born together (off one axes), travel together (into the
    reprojection) and die together (as the ring a patch is built from), which is what makes them one value
    rather than five arguments threaded through a method. Reading them is also where the two refusals
    belong — something that is not a map, and a map that has not been framed — because both are answered
    before any axes is created.

    **The edges are reprojected, not the corners.** Outside the cylindrical projections a projected
    rectangle's outline is not the outline of its corners. Measured for the EPSG:3031 square
    ``(-2e6, -2e6, 2e6, 2e6)`` taken to EPSG:4326: its four corners all land on latitude ``-64.386``, so a
    corner-only box is a **flat line**, while the sampled edges span ``-71.743`` to ``-64.386``. That is the
    lesson recorded in #375, and the reason :meth:`outline` exists at all.

    Attributes:
        west: Western edge, in :attr:`crs`.
        south: Southern edge.
        east: Eastern edge.
        north: Northern edge.
        crs: The CRS those four numbers are in — the map's display CRS.

    Examples:
        - Read off a framed map, then placed in another CRS; the box is its own rectangle when there is no
          reprojection to make:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.maps.inset import _ExtentBox
            >>> m = Map(crs=4326)
            >>> _ = m.set_bounds([2.0, 3.0, 8.0, 9.0])
            >>> box = _ExtentBox.of(m)
            >>> (box.west, box.south, box.east, box.north)
            (2.0, 3.0, 8.0, 9.0)
            >>> ring = box.placed_in(4326)
            >>> [float(ring[:, 0].min()), float(ring[:, 1].max())]
            [2.0, 9.0]
            >>> m.close()

            ```
        - A map that has not been framed holds matplotlib's unit square, which is refused rather than
          marked as if it were an extent:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.maps.inset import _ExtentBox
            >>> m = Map(crs=4326)
            >>> try:
            ...     _ExtentBox.of(m)
            ... except ValueError as error:
            ...     print(str(error).split(",")[0])
            mark_extent(): the map has not been framed — its axes still hold matplotlib's default unit square
            >>> m.close()

            ```
    """

    west: float
    south: float
    east: float
    north: float
    crs: Any

    @classmethod
    def of(cls, scene: Any, caller: str = "mark_extent") -> "_ExtentBox":
        """Read the extent a map is holding off its axes.

        Args:
            scene: The map whose extent is wanted — anything carrying a matplotlib ``ax`` and a ``crs``.
            caller: The method to name in the not-framed refusal, since both
                :meth:`InsetMixin.mark_extent` and :meth:`InsetMixin.inset` read their extent here and a
                caller can only act on the name they typed (round 1, L2). The `TypeError` below keeps
                ``mark_extent``'s own wording because only that call can reach it: :meth:`InsetMixin.inset`
                reads the extent off ``self``, which is a map by construction.

        Returns:
            The box, in that map's display CRS.

        Raises:
            TypeError: when ``scene`` has no axes to read an extent off, which a bare sequence of four
                numbers (the plausible mistake) does not.
            ValueError: when the axes is still the one matplotlib made — autoscaling on both axes, no
                artist on it — and so still holds the default ``(0, 1)`` unit square. Nothing has been
                drawn or framed, and marking that would draw a one-by-one box off the coast of Africa and
                label it the map's extent — a wrong picture rather than an empty one. The state is what
                is read and **not** the four numbers: a map framed on one degree square holds exactly
                those limits legitimately, and used to be refused with a message that was simply false
                (round 1, M5).
        """
        axes: Any = getattr(scene, "ax", None)
        if not hasattr(axes, "get_xlim"):
            raise TypeError(
                "mark_extent() takes the map whose extent is to be marked (anything with .ax and .crs); "
                f"got {type(scene).__name__}. Pass the map itself, as in locator.mark_extent(main)"
            )
        west, east = (float(value) for value in axes.get_xlim())
        south, north = (float(value) for value in axes.get_ylim())
        # The two tests are kept together so the message stays literally true: an untouched axes holds the
        # unit square, so the coordinate half can only ever fire on one that is also in the default state.
        if cls._untouched(axes) and (west, south, east, north) == (0.0, 0.0, 1.0, 1.0):
            raise ValueError(
                f"{caller}(): the map has not been framed — its axes still hold matplotlib's default "
                "unit square, so there is no extent to mark. Draw a layer or call set_bounds() first"
            )
        return cls(west, south, east, north, getattr(scene, "crs", None))

    @staticmethod
    def _untouched(axes: Any) -> bool:
        """Whether ``axes`` is still the axes matplotlib made, rather than one framed on a real extent.

        Both halves are matplotlib's own state rather than anything this package records, so the answer
        holds for any axes a scene hands over — including one framed by ``ax.set_xlim`` directly, which a
        scene-level flag would not see.

        Args:
            axes: The matplotlib axes to read.

        Returns:
            ``True`` when neither limit has been set and nothing has been drawn. ``set_xlim`` /
            ``set_ylim`` turn autoscaling off, which covers every framing call; and an artist leaves
            ``has_data()`` ``True`` even when it autoscaled the view to exactly ``(0, 1)``, which an
            image drawn with ``extent=(0, 1, 0, 1)`` does (its sticky edges take the margins away while
            autoscaling stays on), so neither half answers alone.
        """
        return (
            bool(axes.get_autoscalex_on())
            and bool(axes.get_autoscaley_on())
            and not bool(axes.has_data())
        )

    def outline(self) -> np.ndarray:
        """Return the box's outline, sampled along all four edges and closed.

        The samples run anticlockwise from the south-west corner — south edge west to east, east edge
        south to north, north edge east to west, west edge north to south — so the points are in *ring*
        order and can be drawn as a polygon. That ordering is the difference from
        :func:`~digitalearth.static.maps.decoration._edge_samples`, which samples the same four edges for
        an envelope and so interleaves them.

        Returns:
            An ``(N, 2)`` array of ``(x, y)`` in :attr:`crs`, first point repeated last.
        """
        steps = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
        edge = np.full(_EDGE_SAMPLES, 1.0)
        xs = np.concatenate(
            [
                self.west + (self.east - self.west) * steps,
                edge * self.east,
                self.east + (self.west - self.east) * steps,
                edge * self.west,
            ]
        )
        ys = np.concatenate(
            [
                edge * self.south,
                self.south + (self.north - self.south) * steps,
                edge * self.north,
                self.north + (self.south - self.north) * steps,
            ]
        )
        ring = np.column_stack([xs, ys])
        return np.vstack([ring, ring[:1]])

    def placed_in(self, crs: Any) -> Optional[np.ndarray]:
        """Return the outline in ``crs``, or ``None`` when that CRS cannot place the whole box.

        Args:
            crs: The CRS to draw the box in — the locator's display CRS.

        Returns:
            The ring as an ``(N, 2)`` array in ``crs``; :meth:`outline` unchanged when the two CRSs are the
            same reference system however either is spelled. ``None`` when the reprojection fails, or when
            **any** vertex comes back non-finite — pyproj answers ``inf`` for a point outside the
            projection's domain rather than raising, and an outline missing part of itself closes across
            the gap, drawing a chord no edge of the extent follows. The whole box is dropped for that
            reason rather than partly drawn.
        """
        ring = self.outline()
        if self.crs is None or crs is None or same_crs(self.crs, crs):
            return ring
        try:
            xs, ys = reproject_coordinates(
                ring[:, 0].tolist(), ring[:, 1].tolist(), from_crs=self.crs, to_crs=crs
            )
        except (ValueError, RuntimeError):
            # A CRS pyproj cannot resolve, or a transform it refuses: no box rather than a wrong one.
            return None
        placed = np.column_stack(
            [np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")]
        )
        return placed if bool(np.isfinite(placed).all()) else None


@dataclass(frozen=True, eq=False)
class _InsetFrame:
    """Where an inset axes sits inside its parent, as a fraction of it.

    The placement is four numbers in axes-fraction coordinates, and a caller names them either by corner
    plus a size or outright. Resolving the one into the other is arithmetic with two refusals attached, so
    it is answered here — before the axes exists — rather than inside
    :meth:`InsetMixin.inset`.

    Attributes:
        x0: Left edge, as a fraction of the parent axes.
        y0: Bottom edge.
        width: Width, as a fraction of the parent axes.
        height: Height.

    Examples:
        - A corner and a size become the rectangle, padded off the map's edge:
            ```python
            >>> from digitalearth.static.maps.inset import _InsetFrame
            >>> _InsetFrame.resolve("upper right", 0.3).as_bounds()
            (0.67, 0.67, 0.3, 0.3)
            >>> _InsetFrame.resolve("lower left", 0.3).as_bounds()
            (0.03, 0.03, 0.3, 0.3)

            ```
        - Four fractions are taken as given, and a word that is not a corner is refused by name:
            ```python
            >>> from digitalearth.static.maps.inset import _InsetFrame
            >>> _InsetFrame.resolve((0.1, 0.2, 0.3, 0.4), 0.28).as_bounds()
            (0.1, 0.2, 0.3, 0.4)
            >>> try:
            ...     _InsetFrame.resolve("top right", 0.28)
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            inset(position='top right') is not a corner

            ```
        - The largest corner inset is the axes less its two pads, and a size that cannot keep them is
          refused rather than hung off the edge:
            ```python
            >>> from digitalearth.static.maps.inset import _InsetFrame
            >>> _InsetFrame.resolve("upper right", 0.94).as_bounds()
            (0.03, 0.03, 0.94, 0.94)
            >>> try:
            ...     _InsetFrame.resolve("upper right", 1.0)
            ... except ValueError as error:
            ...     print(str(error).split(":")[0])
            inset(size=1.0) does not fit in the 'upper right' corner

            ```
    """

    x0: float
    y0: float
    width: float
    height: float

    @classmethod
    def resolve(cls, position: Any, size: Any) -> "_InsetFrame":
        """Read a caller's placement into one rectangle.

        Args:
            position: One of :data:`_CORNERS`, or four axes fractions ``(x0, y0, width, height)``.
            size: The inset's side as a fraction of the map, used for the named corners. Checked whichever
                form ``position`` takes, so a value that could not be a fraction is refused rather than
                quietly ignored beside an explicit rectangle.

        Returns:
            The rectangle, in axes-fraction coordinates.

        Raises:
            ValueError: for a ``size`` that is not a fraction in ``(0, 1]`` — ``0`` draws an inset with no
                area and anything above ``1`` hangs outside the map, neither of which matplotlib refuses
                on its own — for a ``size`` positive but too small to survive :meth:`as_bounds`'
                thousandth-of-the-axes rounding, which is the same invisible locator reached through the
                rounding instead of through the input; for a string that is not one of the four corners;
                for a ``size`` above :data:`_LARGEST_CORNER` *beside a named corner*, which cannot keep
                its pad on both sides and so hangs outside the map from the other end (round 1, L1); for
                a sequence that does not hold four numbers; and, from :meth:`_placed`, for four numbers
                that describe no inset at all (round 2, L10). The corner bound is not applied to an
                explicit rectangle: that spelling places the inset itself, pad and all, and ``size`` is
                then only read to refuse a value that could not be a fraction at all.
        """
        fraction = float(size) if isinstance(size, Real) else None
        if fraction is None or isinstance(size, bool) or not 0.0 < fraction <= 1.0:
            # `bool` is a `Real`, and `size=True` is a mistake rather than a request for a full-size inset.
            raise ValueError(
                "inset(size=) is the inset's side as a fraction of the map it sits in, in (0, 1]; got "
                f"{size!r}"
            )
        if round(fraction, _PRECISION) <= 0.0:
            # In (0, 1] and still nothing: `as_bounds` reports the placement to a thousandth of the axes,
            # so a side below 0.0005 reaches `inset_axes` as 0.0 -- the zero-area locator an explicit
            # rectangle is refused for, reached through the rounding rather than through the input.
            raise ValueError(
                f"inset(size={size!r}) is too small to place: a placement is taken to a thousandth of "
                f"the axes, so this side reaches matplotlib as {round(fraction, _PRECISION)} and the "
                "locator has no area. Ask for a larger size"
            )
        if isinstance(position, str):
            if position not in _CORNERS:
                raise ValueError(
                    f"inset(position={position!r}) is not a corner; use one of {sorted(_CORNERS)}, or "
                    "four axes fractions (x0, y0, width, height)"
                )
            if fraction > _LARGEST_CORNER:
                raise ValueError(
                    f"inset(size={size!r}) does not fit in the {position!r} corner: the inset keeps a "
                    f"{_CORNER_PAD} pad on both sides of itself, so a named corner holds a side of at "
                    f"most {_LARGEST_CORNER}. Ask for a smaller size, or place the inset yourself with "
                    "four axes fractions as position=(x0, y0, width, height)"
                )
            right, upper = _CORNERS[position]
            span = _LARGEST_CORNER - fraction
            return cls(
                x0=_CORNER_PAD + right * span,
                y0=_CORNER_PAD + upper * span,
                width=fraction,
                height=fraction,
            )
        try:
            x0, y0, width, height = (float(value) for value in position)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"inset(position=) takes one of {sorted(_CORNERS)} or four axes fractions "
                f"(x0, y0, width, height); got {position!r}"
            ) from error
        return cls._placed(x0, y0, width, height, asked=position)

    @classmethod
    def _placed(
        cls, x0: float, y0: float, width: float, height: float, *, asked: Any
    ) -> "_InsetFrame":
        """Return the explicit rectangle ``asked`` for, refusing the ones that place no inset.

        The named-corner spelling is bounded arithmetic — round 1's L1 capped it at
        :data:`_LARGEST_CORNER` and the four corners then sit inside the axes by construction. The
        explicit spelling is the caller's own four numbers, and it is where that refusal *sends* them
        ("or place the inset yourself with four axes fractions"), so it is the one that has to be read
        (round 2, L10). Measured before this existed: ``position=(0.9, 0.9, 0.5, 0.5)`` drew an inset
        spanning ``x=[0.90, 1.40]``, ``(0.0, -0.5, 0.3, 0.3)`` one entirely below the map,
        ``(0.6, 0.6, 0.0, 0.3)`` one of zero width, and ``(nan, 0.0, 0.3, 0.3)`` one whose position was
        ``nan`` — all four silently.

        **The line drawn here**: a rectangle that *overlaps* the axes and hangs over an edge is a
        placement, honoured as written and only reported at ``WARNING`` — that is what the escape hatch
        is for, matplotlib allows it, and clamping it would hand back a different figure from the one
        asked for. A rectangle that could not be a placement at all is refused.

        Args:
            x0: Left edge, as a fraction of the parent axes.
            y0: Bottom edge.
            width: Width, as a fraction of the parent axes.
            height: Height.
            asked: What the caller wrote, so the refusal quotes it rather than the unpacked numbers.

        Returns:
            The rectangle, in axes-fraction coordinates.

        Raises:
            ValueError: for a non-finite edge or side; for a width or height of zero or less, or one too
                small to survive :meth:`as_bounds`' thousandth-of-the-axes rounding, either of which is
                an invisible locator rather than a placed one; and for a rectangle that does not meet the
                axes anywhere, which is a locator drawn off the map it locates. A negative side was
                already refused — by *matplotlib*, as "Width and height specified must be non-negative",
                which names neither this call nor the keyword; it is answered here instead.

        Examples:
            - A contained rectangle is taken as given, and each impossible one is refused naming the
              call and the keyword:
                ```python
                >>> from digitalearth.static.maps.inset import _InsetFrame
                >>> _InsetFrame.resolve((0.1, 0.2, 0.3, 0.4), 0.28).as_bounds()
                (0.1, 0.2, 0.3, 0.4)
                >>> for bad in ((0.6, 0.6, 0.0, 0.3), (float("nan"), 0.0, 0.3, 0.3),
                ...             (0.0, -0.5, 0.3, 0.3)):
                ...     try:
                ...         _InsetFrame.resolve(bad, 0.28)
                ...     except ValueError as error:
                ...         print(str(error).split(":")[0])
                inset(position=(0.6, 0.6, 0.0, 0.3)) has a side of zero or less
                inset(position=(nan, 0.0, 0.3, 0.3)) is not a rectangle
                inset(position=(0.0, -0.5, 0.3, 0.3)) places the inset outside the map altogether

                ```
        """
        if not all(isfinite(value) for value in (x0, y0, width, height)):
            raise ValueError(
                f"inset(position={asked!r}) is not a rectangle: every one of (x0, y0, width, height) "
                "has to be a finite fraction of the map the inset sits in"
            )
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"inset(position={asked!r}) has a side of zero or less: width={width}, "
                f"height={height}, so there is no locator to draw. Ask for a positive width and height"
            )
        if round(width, _PRECISION) <= 0.0 or round(height, _PRECISION) <= 0.0:
            raise ValueError(
                f"inset(position={asked!r}) is too small to place: a placement is taken to a thousandth "
                "of the axes, so this rectangle reaches matplotlib with a side of 0.0 and the locator "
                "has no area. Ask for a larger width and height"
            )
        if x0 >= 1.0 or y0 >= 1.0 or x0 + width <= 0.0 or y0 + height <= 0.0:
            raise ValueError(
                f"inset(position={asked!r}) places the inset outside the map altogether: (x0, y0) and "
                "(x0 + width, y0 + height) are fractions of the axes the inset sits in, so the "
                "rectangle has to meet [0, 1] in both directions. Ask for a rectangle on the map, or "
                f"name a corner — one of {sorted(_CORNERS)}"
            )
        if x0 < 0.0 or y0 < 0.0 or x0 + width > 1.0 or y0 + height > 1.0:
            logger.warning(
                "inset(position=%r): the inset reaches x=[%s, %s], y=[%s, %s] as a fraction of the map, "
                "so part of it is drawn outside the map it locates. It is placed as asked for — name a "
                "corner, or keep the rectangle inside [0, 1], if that was not intended",
                asked,
                round(x0, _PRECISION),
                round(x0 + width, _PRECISION),
                round(y0, _PRECISION),
                round(y0 + height, _PRECISION),
            )
        return cls(x0=x0, y0=y0, width=width, height=height)

    def as_bounds(self) -> Tuple[float, float, float, float]:
        """Return the rectangle as ``ax.inset_axes`` reads it.

        Returns:
            ``(x0, y0, width, height)`` in axes-fraction coordinates — the tuple ``Axes.inset_axes`` is
            typed for — rounded to :data:`_PRECISION`, the precision the arithmetic is meaningful at (a
            thousandth of the axes is well under a pixel). That rounding is also a floor on the sides,
            which is why :meth:`resolve` refuses a placement that would not survive it.
        """
        return (
            round(self.x0, _PRECISION),
            round(self.y0, _PRECISION),
            round(self.width, _PRECISION),
            round(self.height, _PRECISION),
        )


class InsetMixin(_MixinBase):
    """Locator-map capability for :class:`~digitalearth.static.map.Map`: an inset, and an extent marked on it.

    The fourth of the six capability mixins `Map` is composed from; the methods here call sibling methods
    (`land`, `coastlines`, `set_global`, `set_bounds`, `add_layer`) through `self`, so they only run
    inside a composed `Map`.

    **Both methods return `self`**, like every other builder on the tier, and what each one built is read
    back off an accessor rather than off the call: the locator from :attr:`locator`, the extent box from
    :meth:`~digitalearth.static.scene.Scene.artist` by its layer id. The alternative — handing the thing
    back directly — reads well for one call but breaks the chain every other method keeps, and
    `tests/test_mixin_contract.py` holds the tier to it. Neither is a carve-out, which is what keeps the
    tier's censuses true:
    :meth:`~digitalearth.static.maps.decoration.DecorationMixin.stock_img` is still the only one
    (round 2, L7).
    """

    #: The locator :meth:`inset` built, or `None` before the first call. Set on the *main* map, so
    #: `m.inset().locator` is the inset and `m.inset()` is still `m`.
    _locator: Optional["Map"] = None

    @property
    def locator(self) -> Optional["Map"]:
        """The locator map :meth:`inset` built, or ``None`` if it has not been called.

        Calling :meth:`inset` again replaces it: a second locator is a second axes on the figure, and the
        attribute names the most recent one. The earlier locator keeps working — it is an ordinary `Map` —
        it is simply no longer what this attribute returns.

        Returns:
            The inset `Map`, or ``None``.

        Examples:
            - Before and after one call:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> _ = m.set_bounds([2.0, 3.0, 8.0, 9.0])
                >>> m.locator is None
                True
                >>> m.inset() is m
                True
                >>> m.locator.layer_ids
                ['land-1', 'coastlines-1', 'custom-1']
                >>> m.close()

                ```
            - A second call replaces it, and the inset it names is the most recent one — the earlier
              locator is an ordinary `Map` and goes on working, it is simply no longer what this returns:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> _ = m.set_bounds([2.0, 3.0, 8.0, 9.0])
                >>> _ = m.inset(size=0.2)
                >>> first = m.locator
                >>> _ = m.inset(size=0.2, position="lower left")
                >>> m.locator is first, len(m.ax.child_axes)
                (False, 2)
                >>> first.set_title("still drawable").ax.get_title()
                'still drawable'
                >>> m.close()

                ```
        """
        return self._locator

    def _mark(
        self, box: _ExtentBox, name: Optional[str] = None, **style: Any
    ) -> Optional[Polygon]:
        """Draw an already-read extent box on this map, in **this** map's CRS.

        Args:
            box: The extent to mark, carrying the CRS its four numbers are in.
            name: The id to address the mark by; ``None`` generates ``custom-1``, ``custom-2``, ….
            **style: Patch keywords laid over :data:`_BOX_STYLE`.

        Returns:
            The :class:`~matplotlib.patches.Polygon` put on the axes, or ``None`` when this map's CRS
            cannot place the box (see :meth:`_ExtentBox.placed_in`) — logged as a warning, because the
            only other symptom is a locator with no box on it, which looks like the call never happened.
        """
        ring = box.placed_in(self.crs)
        if ring is None:
            logger.warning(
                "mark_extent: the extent (%s, %s, %s, %s) in %r has no finite image in %r, so no box was "
                "drawn — use a locator CRS that covers the extent, or the map's own",
                box.west,
                box.south,
                box.east,
                box.north,
                box.crs,
                self.crs,
            )
            return None
        patch = Polygon(ring, closed=True, **{**_BOX_STYLE, **style})
        self.ax.add_patch(patch)
        # Registered rather than just drawn, so the mark is addressable like any other layer:
        # `set_visible`, `remove_layer` and `get_layer` all reach it. A matplotlib artist carries no
        # description, so the figure names it as `custom:matplotlib` and cannot rebuild it elsewhere.
        self.add_layer(patch, name=name, band="overlay")
        return patch

    def mark_extent(
        self, other: Any, *, name: Optional[str] = None, **style: Any
    ) -> Self:
        """Draw the extent ``other`` is looking at as an outline on **this** map.

        The outline is placed in **this** map's display CRS, by reprojecting the extent's four *edges*
        rather than its corners — outside the cylindrical projections those are different shapes, and the
        corner-only box of a polar extent is a flat line (measured in :class:`_ExtentBox`). It is a true
        rectangle exactly when the two maps place their coordinates the same way, which is the case
        :meth:`inset` defaults to; under a reprojection it is the extent's projected outline.

        Useful on its own as well as through :meth:`inset` — two panels of a
        :func:`~digitalearth.static.figure.grid` can mark each other's extents, with no inset involved.

        Args:
            other: The map whose extent is to be marked — the detail map, when this one is the locator.
            name: The id to address the mark by, as every builder's ``name=`` works; ``None`` (default)
                generates ``custom-1``, ``custom-2``, ….
            **style: Patch keywords laid over the default hollow red outline (``edgecolor``,
                ``linewidth``, ``linestyle``, ``facecolor``, ``zorder``, …).

        Returns:
            This map, so the call chains like every other builder on the tier — including
            :meth:`inset`, the sibling on this mixin. The box itself is read back by id through
            :meth:`~digitalearth.static.scene.Scene.artist`: `custom-1` by default, or whatever
            ``name=`` asked for. :meth:`~digitalearth.static.scene.Scene.set_visible`,
            :meth:`~digitalearth.static.scene.Scene.get_layer` and
            :meth:`~digitalearth.static.scene.Scene.remove_layer` reach it by the same id.

            It returned the ``Polygon`` until round 2's L7, which made this a **second** carve-out handing
            back a *registered layer's* artist — the kind
            :meth:`~digitalearth.static.maps.decoration.DecorationMixin.stock_img` is singled out as in
            ``tests/static/test_decoration_chaining.py`` ("the one decoration method that keeps its artist
            return"), and which ``raster.py``'s canonical chaining note names beside
            :meth:`~digitalearth.static.maps.vector.VectorMixin.quiverkey` — the second entry in that
            census, whose ``QuiverKey`` is furniture registering no layer rather than a layer's artist. The
            rationale it carried ("one patch a caller restyles afterwards") is the reasoning this package
            rejected for ``text``/``annotate``, and it bought nothing, since the patch was already
            registered and already reachable by id.

            A box this map's CRS has no finite image of **registers no layer**, so there is no id to read
            and :meth:`~digitalearth.static.scene.Scene.artist` refuses the name — the signal the old
            ``None`` return carried, in the tier's own words. The skip is logged at ``WARNING`` as well,
            because a locator with no box on it looks exactly like a call that never happened.

        Raises:
            TypeError: when ``other`` is not a map (nothing to read an extent off).
            ValueError: when ``other`` has not been framed and still holds matplotlib's unit square.

        Examples:
            - A Web Mercator extent marked on a lon/lat locator lands in degrees, and the box is read
              back by its id:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> main = Map(crs=3857)
                >>> _ = main.set_bounds([0.0, 0.0, 1113194.9, 1118889.97])
                >>> locator = Map(crs=4326)
                >>> locator.mark_extent(main) is locator
                True
                >>> ring = locator.artist("custom-1").get_xy()
                >>> [round(float(ring[:, 0].max()), 3), round(float(ring[:, 1].max()), 3)]
                [10.0, 10.0]
                >>> locator.layer_ids
                ['custom-1']
                >>> main.close()
                >>> locator.close()

                ```
            - An extent with no image in the locator's CRS draws nothing rather than a part of itself.
              The call still chains, and the absent box is the id that is not there:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map, projections
                >>> main = Map(crs=projections.orthographic(0, 0))
                >>> _ = main.set_bounds([2.0e7, 2.0e7, 3.0e7, 3.0e7])   # far off the limb
                >>> locator = Map(crs=4326)
                >>> locator.mark_extent(main).layer_ids
                []
                >>> locator.artist("custom-1")
                Traceback (most recent call last):
                    ...
                KeyError: "no layer 'custom-1' on this figure; its layers are []"
                >>> main.close()
                >>> locator.close()

                ```
        """
        self._mark(_ExtentBox.of(other), name=name, **style)
        return self

    def inset(
        self,
        *,
        crs: Any = None,
        globe: Optional[bool] = None,
        position: Any = "upper right",
        size: Any = _DEFAULT_SIZE,
        extent: Optional[Sequence[float]] = None,
        reference: Sequence[str] = _DEFAULT_REFERENCE,
    ) -> Self:
        """Add a locator map: a small inset showing where this map's extent sits on a wider area.

        One call does three things — creates the inset axes inside this map's own axes, draws reference
        geography into it, and marks this map's extent on it with :meth:`mark_extent`. The call itself
        hands back **this** map, so it chains; the locator it built is an ordinary
        :class:`~digitalearth.static.map.Map` read back from :attr:`locator`, and anything else this
        package draws can then go on it (a point for the site, a second box, a basemap).

        It is a method on `Map` rather than a figure-level helper because both halves of it are this
        map's own business: the inset axes is a *child* of this map's axes, so it moves and scales with it,
        and the extent being marked is this map's extent in this map's CRS. A figure-level function would
        have to be handed both anyway, and would have no way to place the inset inside one panel of a
        :func:`~digitalearth.static.figure.grid`.

        Note:
            **A globe locator is framed as it is built**, not at render time. Nothing else ever renders
            it: `render`, `save` and `show` are called on the *map*, and the locator is a scene of its
            own, so its frame would never go on at all. The frame therefore lands after the reference
            geography and the extent box, which is what clips both at the limb — and a layer drawn on
            the locator *afterwards* is not clipped, because `_apply_frame` is applied once per axes.
            Draw everything the locator shows through `inset(reference=)`, or keep the locator flat.

        Warning:
            **The locator shares this map's figure, but closing the locator no longer closes the map.**
            The locator is built on an inset axes of this map's figure, so it does not own that figure
            (:attr:`~digitalearth.static.scene.Scene._owns_fig` is `False`):
            :meth:`~digitalearth.static.scene.Scene.close` spares a borrowed figure, so after
            `m.inset(); m.locator.close()`, `m.fig.number` is still in `pyplot.get_fignums()` — the
            locator's own state is released and the map's figure is left open (#371, resolved). Close the
            *map* when you are done, which closes the figure both sit on.

        Args:
            crs: Display CRS of the locator. `None` (default) uses **this map's own** display CRS, which
                is the one case where the extent box needs no reprojection and so is a true rectangle. A
                different CRS is honoured and the box is reprojected edge-wise into it; note a geographic
                locator cannot draw an extent that straddles its antimeridian without the ring wrapping.
            globe: Whether the locator is drawn on a globe frame — its reference geography re-closed at
                the projection limb, its boundary drawn, its layers clipped to it. `None` (default)
                **follows this map**, because a locator that renders differently from the map it sits in
                is a second picture rather than a smaller one; before #389 it was always flat, so a globe's
                locator went through the flat path with no limb clipping and no boundary. `True` draws a
                globe locator inside a flat map, and `False` the flat locator every caller used to get.
                Naming an `extent` implies `False` (see below).
            position: Where the inset sits — `"upper right"` (default), `"upper left"`,
                `"lower right"`, `"lower left"`, or four axes fractions
                `(x0, y0, width, height)` used as given. The rectangle is the spelling the corner bound
                below points callers at, so it is **read rather than trusted** (round 2, L10): a
                non-finite edge, a side of zero or less, a side too small to survive the
                thousandth-of-the-axes precision a placement is taken to, and a rectangle that does not
                meet the axes anywhere are each refused by name. A rectangle that overlaps the axes and
                hangs over an edge is a placement, drawn exactly as asked for and reported at `WARNING`.
            size: The inset's side as a fraction of this map, for the named corners. Default `0.28`, and
                at most `0.94` there — the inset keeps a `0.03` pad on both sides of itself, so a larger
                side would hang outside the map. Positive but below `0.0005` is refused too: a placement
                is taken to a thousandth of the axes, so such a side reaches matplotlib as `0.0`. An
                explicit `position` rectangle places the inset itself and is not bounded by the corner
                pad.
            extent: The wider area the locator shows, as `(west, south, east, north)` in the locator's
                CRS. `None` (default) shows the whole projection domain. A globe frame cannot hold one
                — it sets the projection's own limits, measured: a globe locator asked for
                `[-4e6, -4e6, 4e6, 4e6]` came back holding the whole disc — so naming an extent is the
                caller asking for a flat locator, and naming it beside `globe=True` is refused rather
                than half-honoured.
            reference: The Natural-Earth layers drawn into the locator, in order. Default
                `("land", "coastlines")`; `()` draws none, for a caller supplying their own geography
                on the locator afterwards.

        Returns:
            This map, so the call chains like every other builder on the tier. The locator it built is
            read back from :attr:`locator` — a :class:`~digitalearth.static.map.Map` bound to the inset
            axes and sharing this map's figure.

        Raises:
            ValueError: for a call that cannot be drawn — listed by what is wrong with it, and
                deliberately **not** counted. The count that used to open this list ("each of the
                four") went stale the moment a refusal was added and nothing updated the numeral, so
                there is no numeral to keep in step any more (round 2, L13). What is said instead holds
                however many there are: **every one of them names `inset()`**, and every one of them
                happens *before* the inset axes is created, so a refused call leaves nothing half-built
                on the figure. `tests/static/test_inset.py` keeps that honest as a census rather than as
                prose — a refusal added here and not there leaves the claim unproven.

                By keyword:

                * `size` — not a real number; not a fraction in `(0, 1]`; positive but too small to
                  survive the thousandth-of-the-axes precision a placement is taken to; or above `0.94`
                  beside a named corner, which cannot keep its pad on both sides (round 1, L1).
                * `position` — a string that is not one of the four corners; anything that does not
                  unpack into four numbers; or four numbers that place no inset at all: a non-finite
                  edge, a side of zero or less, a side that rounds away, or a rectangle that does not
                  meet the axes anywhere. A rectangle that merely *hangs over* an edge is a placement
                  and is drawn as asked for, reported at `WARNING` (round 2, L10).
                * `reference` — a name that is not one of the Natural-Earth layers a locator can draw.
                * `extent` — not four finite numbers; or a rectangle that frames no area, its east not
                  east of its west or its north not north of its south. `set_bounds` honours such a flip as
                  an inverted axis, but a locator's extent is the wider rectangle it shows, not an axis to
                  invert, so it is refused here rather than forwarded.
                * `globe=True` beside an `extent` — the globe frame sets the projection's own limits, so
                  the two cannot both be honoured.

                And one that belongs to no keyword: a map that **has not been framed**, which has no
                extent to mark. It names `inset()` like the rest even though the extent is read by the
                same value object :meth:`mark_extent` reads it with — the object is told which call to
                name (round 1, L2).

        Examples:
            - It chains, because it hands back the map: the locator is a side effect read from
              :attr:`locator` afterwards, never the call's value:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.static import Map
                >>> m = Map(crs=4326)
                >>> _ = m.set_bounds([2.0, 3.0, 8.0, 9.0])
                >>> m.inset(size=0.2).set_title("Rhine delta").ax.get_title()
                'Rhine delta'
                >>> m.locator.layer_ids
                ['land-1', 'coastlines-1', 'custom-1']
                >>> m.close()

                ```
            - A locator on a lon/lat map: the inset is a child of the map's axes, shows the whole world,
              and carries the geography plus the extent box:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> m = Map(crs=4326)
                >>> _ = m.set_bounds([2.0, 3.0, 8.0, 9.0])
                >>> locator = m.inset().locator
                >>> locator.ax in m.ax.child_axes, locator.fig is m.fig
                (True, True)
                >>> [float(v) for v in locator.ax.get_xlim()]
                [-180.0, 180.0]
                >>> locator.layer_ids
                ['land-1', 'coastlines-1', 'custom-1']
                >>> m.close()   # one figure holds both, so one close is the whole picture

                ```
            - A locator in a CRS and a corner of its own, framed on a continent, with no geography but a
              coastline:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> m = Map(crs=3857)
                >>> _ = m.set_bounds([0.0, 0.0, 1113194.9, 1118889.97])
                >>> locator = m.inset(
                ...     crs=4326,
                ...     position="lower left",
                ...     extent=[-20.0, -40.0, 60.0, 40.0],
                ...     reference=("coastlines",),
                ... ).locator
                >>> [float(v) for v in locator.ax.get_ylim()]
                [-40.0, 40.0]
                >>> locator.layer_ids
                ['coastlines-1', 'custom-1']
                >>> m.close()

                ```
            - Each refusal lands before anything is drawn:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> m = Map(crs=4326)
                >>> _ = m.set_bounds([2.0, 3.0, 8.0, 9.0])
                >>> for kwargs in ({"size": 0.0}, {"position": "top right"},
                ...                {"reference": ("close",)}):
                ...     try:
                ...         m.inset(**kwargs)
                ...     except ValueError as error:
                ...         print(str(error).split(";")[0])
                inset(size=) is the inset's side as a fraction of the map it sits in, in (0, 1]
                inset(position='top right') is not a corner
                inset(reference=('close',)) names 'close', which is not a reference layer
                >>> m.ax.child_axes
                []
                >>> m.close()

                ```

            - A globe's locator is a globe as well: its land comes back as the limb-clipped rings the
              globe path fills, and the frame puts a boundary on the inset axes:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from matplotlib.patches import PathPatch
                >>> from digitalearth.static import Map, projections
                >>> m = Map(crs=projections.orthographic(10, 20), globe=True)
                >>> _ = m.set_bounds([-1.0e6, -1.0e6, 1.0e6, 1.0e6])
                >>> locator = m.inset(reference=("land",)).locator
                >>> locator.globe
                True
                >>> type(locator.artist("land-1")).__name__
                'PolyCollection'
                >>> any(isinstance(patch, PathPatch) for patch in locator.ax.patches)
                True
                >>> m.close()

                ```

        See Also:
            mark_extent: the box on its own, for marking one map's extent on any other.
        """
        # Everything a caller can get wrong is answered first, so a refused call leaves no axes behind.
        frame = _InsetFrame.resolve(position, size)
        for layer in reference:
            if layer not in _REFERENCE_LAYERS:
                raise ValueError(
                    f"inset(reference={tuple(reference)!r}) names {layer!r}, which is not a reference "
                    f"layer; use any of {list(_REFERENCE_LAYERS)}"
                )
        on_a_globe = self._locator_globe(globe, extent)
        checked_extent = None if extent is None else self._checked_extent(extent)
        box = _ExtentBox.of(self, caller="inset")
        locator = type(self)(
            crs=self.crs if crs is None else crs,
            ax=self.ax.inset_axes(frame.as_bounds()),
            fig=self.fig,
            globe=on_a_globe,
        )
        # A locator is a thumbnail, not a plot: at a quarter of the map's size its tick labels overlap
        # into illegibility.
        locator.ax.set_xticks([])
        locator.ax.set_yticks([])
        # Geography first, then the frame: on an otherwise-empty axes a Natural-Earth layer autoscales the
        # view to itself, so framing afterwards is what makes the asked-for extent the one that stands.
        for layer in reference:
            getattr(locator, layer)()
        if checked_extent is None:
            locator.set_global()
        else:
            locator.set_bounds(checked_extent, caller="inset")
        locator._mark(box)
        # The frame goes on here rather than at render time: nothing else renders the locator, and the
        # geography and the box have to be on the axes before they can be clipped to the limb. `render`
        # answers a flat locator with nothing, so this is the globe case only.
        locator.render()
        self._locator = locator
        return self

    def _checked_extent(self, extent: Sequence[float]) -> List[float]:
        """Read a caller's locator extent into four display-CRS numbers, refusing the ones that frame nothing.

        :meth:`~digitalearth.static.maps.projection.ProjectionMixin.set_bounds` honours a flipped pair as an
        inverted axis — a deliberate 2-D-tier contract — but a locator's `extent` is the wider rectangle the
        inset shows, not an axis to invert, so an inverted or zero-area one is a caller's mistake. It is read
        here, in front of the method with the method's other refusals, so a bad extent leaves nothing
        half-built (the same invariant every inset refusal keeps) and the message names `inset()` rather
        than the `set_bounds` it used to forward to.

        Args:
            extent: The `(west, south, east, north)` the caller named, in the locator's CRS.

        Returns:
            The four numbers as a list, ready for `set_bounds`.

        Raises:
            ValueError: for an extent that is not four finite numbers, whose east is not east of its west,
                or whose north is not north of its south — each naming `inset(extent=...)`.
        """
        try:
            values = [float(value) for value in extent]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"inset(extent={extent!r}) takes four numbers (west, south, east, north) in the "
                "locator's CRS; that is not a sequence of numbers"
            ) from error
        if len(values) != 4:
            raise ValueError(
                f"inset(extent={values!r}) needs exactly 4 values as (west, south, east, north); got "
                f"{len(values)}"
            )
        if not all(isfinite(value) for value in values):
            raise ValueError(
                f"inset(extent={values!r}) has a non-finite edge: every one of (west, south, east, "
                "north) has to be a finite number in the locator's CRS"
            )
        west, south, east, north = values
        if east <= west or north <= south:
            raise ValueError(
                f"inset(extent={values!r}) frames no area: west has to be below east and south below "
                "north — the extent is the wider rectangle the locator shows, not an inverted axis. Give "
                "the corners the right way round, or drop extent= to show the whole projection domain"
            )
        return values

    def _locator_globe(self, globe: Optional[bool], extent: Any) -> bool:
        """Settle whether the locator is drawn on a globe frame, refusing the pair that cannot be.

        Args:
            globe: What the caller wrote — ``True``, ``False``, or ``None`` for "the same frame as this
                map".
            extent: The region the caller framed the locator on, or ``None``.

        Returns:
            Whether the locator is a globe: what the caller asked for; else this map's own frame, unless
            an ``extent`` was named, which only a flat frame can hold.

        Raises:
            ValueError: for ``globe=True`` beside an ``extent``. The globe frame sets the projection's own
                limits, so the extent would be dropped in silence — measured, a globe locator asked for
                ``[-4e6, -4e6, 4e6, 4e6]`` came back holding the whole disc.
        """
        if globe and extent is not None:
            raise ValueError(
                "inset(globe=True, extent=...) cannot both be drawn: the globe frame sets the "
                "projection's own limits, so the extent would be thrown away (measured: a locator asked "
                f"for {list(extent)!r} came back holding the whole disc). Ask for one or the other — an "
                "extent draws a flat locator framed on it, and globe=True draws the whole disc"
            )
        if globe is not None:
            return bool(globe)
        # A caller who named a region asked for the one frame that can hold it, whatever this map is.
        return False if extent is not None else bool(self.globe)
