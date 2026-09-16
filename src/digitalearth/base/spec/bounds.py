"""A rectangle that carries its own CRS, and names its ordering instead of implying it.

Four floats cannot say what they mean. matplotlib's axes want ``[xmin, xmax, ymin, ymax]``; cleopatra and
pyramids want ``[xmin, ymin, xmax, ymax]``; a named domain is the second order but always in EPSG:4326. All
three were live in the static tier at once, none of them stated in a signature, so a call site that passed the
wrong one got a silently wrong extent rather than an error.

`Bounds` makes the ordering a **method name** — :meth:`Bounds.as_mpl` or :meth:`Bounds.as_bbox` — and keeps the
CRS with the numbers rather than beside them.

The name is `Bounds`, not `Extent`, because :data:`digitalearth.base.basemaps.Extent` already binds that name
for the plain tuple a basemap's coverage is declared with.

Reprojection is pyramids' job, not this package's: :meth:`Bounds.to_crs` delegates to
``pyramids.base.crs.reproject_coordinates`` rather than doing coordinate maths here.
"""

import warnings
from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from digitalearth.base.spec._serial import (
    crs_to_json,
    finite_number,
    refuse_unknown,
    require,
)

__all__ = ["Bounds"]


def same_crs(one: Any, other: Any) -> bool:
    """Whether two CRS spellings name the same reference system.

    Args:
        one: A CRS in any spelling pyramids accepts — an EPSG int, an ``"EPSG:4326"`` string, a proj4 string.
        other: The CRS to compare it with.

    Returns:
        ``True`` when they denote the same system. Compared by meaning rather than by ``==``, because the
        whole point of carrying a CRS is to stop a rectangle being measured against the wrong one — and
        ``4326 != "EPSG:4326"`` would refuse a great many *right* ones. pyramids owns the normalisation, so
        it is asked; it answers ``False`` for a spelling it cannot read rather than raising, which is what
        this needs — the question is whether reprojection is required, not whether the input is valid.

        Nothing short-circuits in front of it, not even identity. `crs_equal` carries a guard that refuses
        values naming no reference system, and CPython interns ``0``, ``''`` and ``True`` — so an ``is`` or
        ``==`` shortcut would answer "same CRS" for two rectangles built with ``crs=0``, letting them union
        as though they agreed and letting ``to_crs(0)`` no-op instead of refusing. It already answers
        ``True`` for two unset CRSs, which is the one case a shortcut would have been for.

        A CRS **object** — a pyproj `CRS`, which is what `GeoDataFrame.crs` holds — is first written in the
        spelling a figure stores it in (``"EPSG:<code>"``, or WKT when it has no code), because `crs_equal` reads
        only ``int``/``str``/``None`` and answers ``False`` for an object compared even with itself. Without that,
        a `Viewport` in an object CRS could never hold bounds, and `to_crs` reprojected a rectangle into the CRS it
        was already in.

        A value with no written form — a float, a boolean, a list, an object pyramids cannot read — compares as
        different, and is never handed to `crs_equal`. Its answers are cached by value regardless of type, so a
        float `4326.0` would leave a "not the same" answer that `4326` then reads back, and a list cannot be a
        cache key at all.

    Examples:
        - Two spellings of one system, and a CRS object against its own EPSG code:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> from digitalearth.base.spec.bounds import same_crs
            >>> same_crs(4326, "EPSG:4326"), same_crs(crs_from_user_input(3857), 3857)
            (True, True)

            ```
    """
    from pyramids.base.crs import crs_equal

    try:
        first, second = crs_to_json(one, "crs"), crs_to_json(other, "crs")
    except TypeError:
        return False
    return bool(crs_equal(first, second))


@dataclass(frozen=True)
class Bounds:
    """An axis-aligned rectangle in one CRS.

    Attributes:
        xmin: Western edge, in `crs` units.
        ymin: Southern edge.
        xmax: Eastern edge.
        ymax: Northern edge.
        crs: The CRS the four values are expressed in — an EPSG code, a WKT string, or anything pyramids
            resolves. Kept with the numbers so a rectangle cannot be measured against the wrong one.

    Raises:
        ValueError: if any edge is not finite, or if an edge pair is inverted (``xmax < xmin``). A rectangle
            whose corners are the wrong way round draws nothing and reports no error, which is the failure
            this refuses up front.

    Examples:
        - Build one and read back the two orderings that used to be positional conventions:
            ```python
            >>> from digitalearth.base.spec import Bounds
            >>> box = Bounds(0.0, 10.0, 4.0, 20.0, crs=4326)
            >>> box.as_bbox()
            [0.0, 10.0, 4.0, 20.0]
            >>> box.as_mpl()
            [0.0, 4.0, 10.0, 20.0]

            ```
        - The union of two rectangles is what auto-framing needs:
            ```python
            >>> from digitalearth.base.spec import Bounds
            >>> west = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
            >>> east = Bounds(5.0, 2.0, 6.0, 3.0, crs=4326)
            >>> west.union(east)
            Bounds(xmin=0.0, ymin=0.0, xmax=6.0, ymax=3.0, crs=4326)

            ```
        - An inverted rectangle is refused rather than drawn empty:
            ```python
            >>> from digitalearth.base.spec import Bounds
            >>> Bounds(10.0, 0.0, 1.0, 1.0, crs=4326)
            Traceback (most recent call last):
                ...
            ValueError: Bounds needs xmin <= xmax; got xmin=10.0, xmax=1.0

            ```
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: Any

    def __post_init__(self) -> None:
        """Refuse a rectangle that cannot bound anything.

        Raises:
            ValueError: for a non-finite edge, or an inverted pair.
        """
        for name, value in (
            ("xmin", self.xmin),
            ("ymin", self.ymin),
            ("xmax", self.xmax),
            ("ymax", self.ymax),
        ):
            if not isfinite(value):
                raise ValueError(f"Bounds needs finite edges; got {name}={value!r}")
        if self.xmax < self.xmin:
            raise ValueError(
                f"Bounds needs xmin <= xmax; got xmin={self.xmin}, xmax={self.xmax}"
            )
        if self.ymax < self.ymin:
            raise ValueError(
                f"Bounds needs ymin <= ymax; got ymin={self.ymin}, ymax={self.ymax}"
            )

    # ------------------------------------------------------------------ builders

    @classmethod
    def from_bbox(cls, bbox: Sequence[float], crs: Any) -> "Bounds":
        """Build from the ``[xmin, ymin, xmax, ymax]`` order cleopatra and pyramids use.

        Args:
            bbox: Four edges in bbox order.
            crs: The CRS they are expressed in.

        Returns:
            The rectangle.

        Raises:
            ValueError: if `bbox` does not hold exactly four values, or the rectangle is invalid.

        Examples:
            - The order pyramids hands back:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds.from_bbox([0.0, 1.0, 2.0, 3.0], crs=4326).as_bbox()
                [0.0, 1.0, 2.0, 3.0]

                ```
        """
        xmin, ymin, xmax, ymax = cls._four(bbox, "bbox")
        return cls(xmin, ymin, xmax, ymax, crs)

    @classmethod
    def from_mpl(cls, extent: Sequence[float], crs: Any) -> "Bounds":
        """Build from the ``[xmin, xmax, ymin, ymax]`` order matplotlib's axes use.

        Args:
            extent: Four edges in matplotlib axes order.
            crs: The CRS they are expressed in.

        Returns:
            The rectangle.

        Raises:
            ValueError: if `extent` does not hold exactly four values, or the rectangle is invalid.

        Examples:
            - The order `set_xlim`/`set_ylim` read:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds.from_mpl([0.0, 2.0, 1.0, 3.0], crs=4326).as_bbox()
                [0.0, 1.0, 2.0, 3.0]

                ```
        """
        xmin, xmax, ymin, ymax = cls._four(extent, "extent")
        return cls(xmin, ymin, xmax, ymax, crs)

    @classmethod
    def from_points(cls, x: Any, y: Any, crs: Any) -> "Bounds":
        """Build the rectangle enclosing coordinate arrays.

        Args:
            x: X coordinates, any sequence or array.
            y: Y coordinates.
            crs: The CRS they are expressed in.

        Returns:
            The enclosing rectangle.

        Raises:
            ValueError: if either array is empty, or the rectangle is invalid.

        Examples:
            - Enclose a scattering of points:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds.from_points([3.0, 1.0, 2.0], [9.0, 7.0, 8.0], crs=4326).as_bbox()
                [1.0, 7.0, 3.0, 9.0]

                ```
        """
        # numpy's reductions rather than a Python loop: this runs per render over every coordinate of a
        # raster's axes, and materialising a Python list of them was a cost the previous np.min/np.max had
        # not. It also keeps what those accepted — a 2-D array, and NaN coordinates, which nanmin ignores
        # rather than turning into a "needs finite edges" error from the constructor.
        xs = np.asarray(x, dtype="float64")
        ys = np.asarray(y, dtype="float64")
        if xs.size == 0 or ys.size == 0:
            raise ValueError("Bounds.from_points needs at least one coordinate pair")
        with warnings.catch_warnings():
            # An all-NaN axis warns and yields NaN, which the constructor then refuses by name.
            warnings.simplefilter("ignore", RuntimeWarning)
            return cls(
                float(np.nanmin(xs)),
                float(np.nanmin(ys)),
                float(np.nanmax(xs)),
                float(np.nanmax(ys)),
                crs,
            )

    @staticmethod
    def _four(values: Sequence[float], label: str) -> Tuple[float, float, float, float]:
        """Return exactly four floats, or say which argument was the wrong length.

        Args:
            values: The candidate sequence.
            label: Argument name to blame in the error.

        Returns:
            The four values as floats.

        Raises:
            ValueError: when there are not exactly four.
        """
        seq = [float(v) for v in values]
        if len(seq) != 4:
            raise ValueError(f"{label} needs exactly 4 values; got {len(seq)}")
        return seq[0], seq[1], seq[2], seq[3]

    # ------------------------------------------------------------------ readers

    def as_bbox(self) -> List[float]:
        """Return ``[xmin, ymin, xmax, ymax]`` — the order cleopatra and pyramids take.

        Returns:
            The four edges in bbox order.

        Examples:
            - The corners come back in the order a bounding box is written in:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds(0.0, 10.0, 4.0, 20.0, crs=4326).as_bbox()
                [0.0, 10.0, 4.0, 20.0]

                ```
            - Which is what a pyramids call wants, unpacked:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> xmin, ymin, xmax, ymax = Bounds(-5.0, 40.0, 5.0, 50.0, crs=4326).as_bbox()
                >>> xmax - xmin, ymax - ymin
                (10.0, 10.0)

                ```
        """
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def as_mpl(self) -> List[float]:
        """Return ``[xmin, xmax, ymin, ymax]`` — the order matplotlib's axes take.

        Returns:
            The four edges in matplotlib axes order.

        Examples:
            - The x pair comes first here, unlike :meth:`as_bbox`:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds(0.0, 10.0, 4.0, 20.0, crs=4326).as_mpl()
                [0.0, 4.0, 10.0, 20.0]

                ```
            - The two orderings disagree for the same rectangle, which is the bug this type removes:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> box = Bounds(0.0, 10.0, 4.0, 20.0, crs=4326)
                >>> box.as_bbox() == box.as_mpl()
                False

                ```
        """
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    # ------------------------------------------------------------------ operations

    def union(self, other: "Bounds") -> "Bounds":
        """Return the smallest rectangle containing both.

        Args:
            other: The rectangle to merge in. Must already be in the same CRS — reproject it first with
                :meth:`to_crs` rather than relying on a silent reinterpretation of the numbers.

        Returns:
            The enclosing rectangle, in this rectangle's CRS.

        Raises:
            ValueError: if the two carry different CRSs. Unioning across CRSs by ignoring them is how an
                extent ends up describing a region nobody asked for.

        Examples:
            - Two layers' extents, framed together:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> a = Bounds(0.0, 0.0, 2.0, 2.0, crs=3857)
                >>> b = Bounds(1.0, -5.0, 3.0, 1.0, crs=3857)
                >>> a.union(b).as_bbox()
                [0.0, -5.0, 3.0, 2.0]

                ```
        """
        if not same_crs(self.crs, other.crs):
            raise ValueError(
                f"union needs both rectangles in one CRS; got {self.crs!r} and {other.crs!r} — "
                "reproject one with to_crs() first"
            )
        return Bounds(
            min(self.xmin, other.xmin),
            min(self.ymin, other.ymin),
            max(self.xmax, other.xmax),
            max(self.ymax, other.ymax),
            self.crs,
        )

    def padded(self, fraction: float) -> "Bounds":
        """Return this rectangle grown by `fraction` of its own span on every side.

        Args:
            fraction: How much to grow, as a proportion of width and height. ``0.05`` adds a 5% margin;
                a negative value shrinks.

        Returns:
            The padded rectangle.

        Raises:
            ValueError: if the padding would invert the rectangle.

        Examples:
            - A 10% margin around the data:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds(0.0, 0.0, 10.0, 10.0, crs=4326).padded(0.1).as_bbox()
                [-1.0, -1.0, 11.0, 11.0]

                ```
        """
        dx = (self.xmax - self.xmin) * fraction
        dy = (self.ymax - self.ymin) * fraction
        if dx * 2 < -(self.xmax - self.xmin) or dy * 2 < -(self.ymax - self.ymin):
            # Constructing the rectangle would raise, but its message names only the numbers that came out
            # — leaving the caller to work out where a 6.0 and a 4.0 came from when they wrote -0.6.
            raise ValueError(
                f"padded({fraction}) would invert this rectangle: a fraction below -0.5 removes more than "
                "the rectangle has. Use a fraction above -0.5 to shrink it"
            )
        return Bounds(
            self.xmin - dx, self.ymin - dy, self.xmax + dx, self.ymax + dy, self.crs
        )

    def to_crs(self, crs: Any) -> "Bounds":
        """Return this rectangle expressed in another CRS.

        The corner transform is pyramids' (``pyramids.base.crs.reproject_coordinates``); this package does no
        coordinate maths of its own, per `CLAUDE.md`. The result is the **enclosing** rectangle of the
        transformed corners: a warped rectangle is not a rectangle, so its axis-aligned hull is what an extent
        can hold.

        Args:
            crs: The target CRS.

        Returns:
            The enclosing rectangle in `crs`, or this rectangle unchanged when `crs` already matches.

        Raises:
            ValueError: naming the rectangle and both CRSs, when a corner falls outside the area `crs` can show —
                the far side of an orthographic globe, say — so the reprojection gives no finite corner to enclose;
                or, as pyramids' `CRSError` (a `ValueError` subclass), when `crs` is not a CRS pyramids can read.

        Examples:
            - Reprojecting to the CRS it already has is a no-op:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> box = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
                >>> box.to_crs(4326) is box
                True

                ```
            - A rectangle on the far side of an orthographic globe has no corners to enclose:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds(100.0, -10.0, 170.0, 10.0, crs=4326).to_crs("+proj=ortho +lat_0=0 +lon_0=0")
                Traceback (most recent call last):
                    ...
                ValueError: Bounds [100.0, -10.0, 170.0, 10.0] in 4326 cannot be reprojected into ...

                ```
        """
        if same_crs(crs, self.crs):
            return self
        from pyramids.base.crs import reproject_coordinates

        # (x, y) order throughout, matching pyramids' own contract — the four corners, not the four edges.
        xs, ys = reproject_coordinates(
            [self.xmin, self.xmax, self.xmin, self.xmax],
            [self.ymin, self.ymin, self.ymax, self.ymax],
            from_crs=self.crs,
            to_crs=crs,
            # pyramids rounds to 6 decimals by default, which collapses a rectangle finer than that step
            # into a degenerate one — and set_extent would hand matplotlib a singular limit.
            precision=None,
        )
        if not all(isfinite(value) for value in (*xs, *ys)):
            # pyramids answers a corner the target cannot show with an infinity. Handed on, the constructor would
            # report an infinite edge without saying a reprojection produced it.
            raise ValueError(
                f"Bounds {self.as_bbox()} in {self.crs!r} cannot be reprojected into {crs!r}: part of the "
                "rectangle falls outside the area that projection can show"
            )
        return Bounds(min(xs), min(ys), max(xs), max(ys), crs)

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            The four edges, as held, and the CRS. A CRS object is written as `"EPSG:<code>"` (or WKT when it has no
            code), so the CRS survives `json.dumps`; `None`, an EPSG integer or a string is written as given, and a
            string is not checked.

        Raises:
            TypeError: if the CRS is a boolean, or an object pyramids cannot read as a CRS.

        Examples:
            - The edges keep their names, so the ordering cannot be misread:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds(0.0, 1.0, 2.0, 3.0, crs=4326).to_dict()
                {'xmin': 0.0, 'ymin': 1.0, 'xmax': 2.0, 'ymax': 3.0, 'crs': 4326}

                ```
            - A CRS object is written by its authority code:
                ```python
                >>> from pyramids.base.crs import crs_from_user_input
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds(0.0, 1.0, 2.0, 3.0, crs=crs_from_user_input(4326)).to_dict()["crs"]
                'EPSG:4326'

                ```
        """
        return {
            # The constructor has already refused a non-finite edge, but not a numpy float, which json cannot write.
            "xmin": float(self.xmin),
            "ymin": float(self.ymin),
            "xmax": float(self.xmax),
            "ymax": float(self.ymax),
            "crs": crs_to_json(self.crs, "Bounds.crs"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Bounds":
        """Rebuild a rectangle from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`. A missing `crs` means `None`.

        Returns:
            The rectangle, validated as the constructor validates it.

        Raises:
            TypeError: if `data` is not a mapping.
            ValueError: if an edge is missing or is not a finite number — `None` and a numeric string are refused
                by name — a key is unknown, or the edges do not bound a rectangle.

        Examples:
            - A stored rectangle reads back to the same value:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> box = Bounds.from_dict({"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 5, "crs": 3857})
                >>> box.as_bbox(), box.crs
                ([0.0, 0.0, 10.0, 5.0], 3857)

                ```
            - A missing edge is named:
                ```python
                >>> from digitalearth.base.spec import Bounds
                >>> Bounds.from_dict({"xmin": 0, "ymin": 0, "xmax": 1, "crs": 4326})
                Traceback (most recent call last):
                    ...
                ValueError: Bounds.from_dict needs 'ymax'; got keys ['crs', 'xmax', 'xmin', 'ymin']

                ```
        """
        refuse_unknown("Bounds", data, ("xmin", "ymin", "xmax", "ymax", "crs"))
        return cls(
            finite_number("Bounds", "xmin", require("Bounds", data, "xmin")),
            finite_number("Bounds", "ymin", require("Bounds", data, "ymin")),
            finite_number("Bounds", "xmax", require("Bounds", data, "xmax")),
            finite_number("Bounds", "ymax", require("Bounds", data, "ymax")),
            data.get("crs"),
        )
