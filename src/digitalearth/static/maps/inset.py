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
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, Optional, Self, Sequence, Tuple

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
                on its own — for a string that is not one of the four corners, for a ``size`` above
                :data:`_LARGEST_CORNER` *beside a named corner*, which cannot keep its pad on both sides
                and so hangs outside the map from the other end (round 1, L1), and for a sequence that
                does not hold four numbers. The corner bound is not applied to an explicit rectangle:
                that spelling places the inset itself, pad and all, and ``size`` is then only read to
                refuse a value that could not be a fraction at all.
        """
        fraction = float(size) if isinstance(size, Real) else None
        if fraction is None or isinstance(size, bool) or not 0.0 < fraction <= 1.0:
            # `bool` is a `Real`, and `size=True` is a mistake rather than a request for a full-size inset.
            raise ValueError(
                "inset(size=) is the inset's side as a fraction of the map it sits in, in (0, 1]; got "
                f"{size!r}"
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
        return cls(x0=x0, y0=y0, width=width, height=height)

    def as_bounds(self) -> Tuple[float, float, float, float]:
        """Return the rectangle as ``ax.inset_axes`` reads it.

        Returns:
            ``(x0, y0, width, height)`` in axes-fraction coordinates — the tuple ``Axes.inset_axes`` is
            typed for — rounded to the precision the arithmetic is meaningful at (a thousandth of the
            axes is well under a pixel).
        """
        return (
            round(self.x0, 3),
            round(self.y0, 3),
            round(self.width, 3),
            round(self.height, 3),
        )


class InsetMixin(_MixinBase):
    """Locator-map capability for :class:`~digitalearth.static.map.Map`: an inset, and an extent marked on it.

    The fourth of the six capability mixins `Map` is composed from; the methods here call sibling methods
    (`land`, `coastlines`, `set_global`, `set_bounds`, `add_layer`) through `self`, so they only run
    inside a composed `Map`.

    :meth:`inset` returns `self` like every other builder on the tier, and the locator it builds is read
    back from :attr:`locator`. The alternative — handing the locator back directly — reads well for one call
    but breaks the chain every other method keeps, and `tests/test_mixin_contract.py` holds the tier to it.
    :meth:`mark_extent` is the exception, and for the reason
    :meth:`~digitalearth.static.maps.decoration.DecorationMixin.stock_img` is: the one patch it draws is
    what a caller restyles afterwards, so handing it back is the point of the call.
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
                >>> m.set_bounds([2.0, 3.0, 8.0, 9.0])  # doctest: +ELLIPSIS
                <....Map object at ...>
                >>> m.locator is None
                True
                >>> m.inset() is m
                True
                >>> type(m.locator).__name__
                'Map'
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
    ) -> Optional[Polygon]:
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
            The :class:`~matplotlib.patches.Polygon` drawn, or `None` when this map's CRS has no finite
            image of the extent at all — the skip is logged and the figure is left without a box rather
            than given a wrong one.

            **An artist rather than `self`**, unlike the data builders and unlike :meth:`inset`: the box
            is one patch a caller restyles afterwards, so handing it back is the point of the call. It is
            registered as a layer all the same — `custom-1` by default — so
            :meth:`~digitalearth.static.scene.Scene.set_visible`,
            :meth:`~digitalearth.static.scene.Scene.get_layer` and
            :meth:`~digitalearth.static.scene.Scene.remove_layer` reach it, and
            :meth:`~digitalearth.static.scene.Scene.artist` hands back this same patch by that id.

        Raises:
            TypeError: when ``other`` is not a map (nothing to read an extent off).
            ValueError: when ``other`` has not been framed and still holds matplotlib's unit square.

        Examples:
            - A Web Mercator extent marked on a lon/lat locator lands in degrees:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> main = Map(crs=3857)
                >>> _ = main.set_bounds([0.0, 0.0, 1113194.9, 1118889.97])
                >>> locator = Map(crs=4326)
                >>> ring = locator.mark_extent(main).get_xy()
                >>> [round(float(ring[:, 0].max()), 3), round(float(ring[:, 1].max()), 3)]
                [10.0, 10.0]
                >>> locator.layer_ids
                ['custom-1']
                >>> main.close()
                >>> locator.close()

                ```
            - An extent with no image in the locator's CRS draws nothing rather than a part of itself:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map, projections
                >>> main = Map(crs=projections.orthographic(0, 0))
                >>> _ = main.set_bounds([2.0e7, 2.0e7, 3.0e7, 3.0e7])   # far off the limb
                >>> locator = Map(crs=4326)
                >>> print(locator.mark_extent(main))
                None
                >>> main.close()
                >>> locator.close()

                ```
        """
        return self._mark(_ExtentBox.of(other), name=name, **style)

    def inset(
        self,
        *,
        crs: Any = None,
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

        Warning:
            **The locator shares this map's figure, so do not close it or context-manage it.**
            :meth:`~digitalearth.static.scene.Scene.close` calls `pyplot.close` on whatever figure a
            scene holds, and a borrowed figure is not spared (#371) — measured: after
            `m.inset().locator.close()`, `m.fig.number` is no longer in `pyplot.get_fignums()`, i.e.
            closing the locator closed the map. Close the *map* when you are done, which closes both.

        Args:
            crs: Display CRS of the locator. `None` (default) uses **this map's own** display CRS, which
                is the one case where the extent box needs no reprojection and so is a true rectangle. A
                different CRS is honoured and the box is reprojected edge-wise into it; note a geographic
                locator cannot draw an extent that straddles its antimeridian without the ring wrapping.
            position: Where the inset sits — `"upper right"` (default), `"upper left"`,
                `"lower right"`, `"lower left"`, or four axes fractions
                `(x0, y0, width, height)` used as given.
            size: The inset's side as a fraction of this map, for the named corners. Default `0.28`, and
                at most `0.94` there — the inset keeps a `0.03` pad on both sides of itself, so a larger
                side would hang outside the map. An explicit `position` rectangle places the inset
                itself and is not bounded that way.
            extent: The wider area the locator shows, as `(west, south, east, north)` in the locator's
                CRS. `None` (default) shows the whole projection domain.
            reference: The Natural-Earth layers drawn into the locator, in order. Default
                `("land", "coastlines")`; `()` draws none, for a caller supplying their own geography
                on the locator afterwards.

        Returns:
            This map, so the call chains like every other builder on the tier. The locator it built is
            read back from :attr:`locator` — a :class:`~digitalearth.static.map.Map` bound to the inset
            axes and sharing this map's figure.

        Raises:
            ValueError: for a `size` that is not a fraction in `(0, 1]` or one above `0.94` beside a
                named corner, a `position` that is neither a corner nor four numbers, a `reference` naming something that is not a
                Natural-Earth layer, or a map that has not been framed — there is then no extent to mark,
                and the refusal happens **before** the inset axes is created, so nothing half-built is
                left on the figure. Each of the four names `inset()`, including the last: the extent is
                read by the same value object :meth:`mark_extent` reads it with, and it is told which
                call to name.

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
        box = _ExtentBox.of(self, caller="inset")
        locator = type(self)(
            crs=self.crs if crs is None else crs,
            ax=self.ax.inset_axes(frame.as_bounds()),
            fig=self.fig,
        )
        # A locator is a thumbnail, not a plot: at a quarter of the map's size its tick labels overlap
        # into illegibility.
        locator.ax.set_xticks([])
        locator.ax.set_yticks([])
        # Geography first, then the frame: on an otherwise-empty axes a Natural-Earth layer autoscales the
        # view to itself, so framing afterwards is what makes the asked-for extent the one that stands.
        for layer in reference:
            getattr(locator, layer)()
        if extent is None:
            locator.set_global()
        else:
            locator.set_bounds(list(extent))
        locator._mark(box)
        self._locator = locator
        return self
