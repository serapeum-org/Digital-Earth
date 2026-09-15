"""Point geometry as coordinate arrays — one path, instead of one per call site.

Turning a `FeatureCollection`'s geometry into numpy arrays is the single most-repeated operation in the
package: **22 extraction lines across 9 files** on the commit this replaced. That would be unremarkable if the
copies agreed, and they do not. Three spellings are in use — ``gdf.geometry.x.to_numpy()``,
``np.asarray(geom.x, dtype=float)``, ``np.asarray(geom.x.to_numpy(), dtype="float64")`` — differing in whether
a dtype is forced at all, and in what happens to geometry that is not a point.

The part that actually bites is **non-finite coordinates**. A globe or a clipped display CRS reprojects the
far side of the world to ``inf``/``nan``; a site that forgets to drop those hands them to a tessellator, a
binner or matplotlib, which fail somewhere else entirely. One tier had already noticed and built
``_finite_point_xy`` locally — extraction plus filtering, keeping a values array in step with the points it
drops — and used it at three of its own call sites. This is that helper, lifted so every tier has it.

`z` is here because the 3-D tier and the web tier's deck.gl path need a third coordinate and each derived it
their own way; and because point clouds (#206) will need the same shape carrying an attribute table.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

__all__ = ["PointArrays"]


@dataclass(frozen=True, eq=False)
class PointArrays:
    """Point coordinates as parallel float arrays, with the CRS they are measured in.

    Attributes:
        x: X coordinates, ``float64``.
        y: Y coordinates, ``float64``.
        z: Z coordinates, ``float64``. Zeros when the source geometry is 2-D, so a consumer that needs three
            columns always has them — which is what the 3-D tier was filling in by hand.
        crs: What the coordinates are measured in, or ``None`` when the source declared none. Carried, not
            normalised: :meth:`of` keeps whatever the caller passed (typically an EPSG ``int``) and
            :meth:`from_features` keeps whatever the frame declared (typically a ``pyproj.CRS``). Compare it
            with :func:`pyramids.base.crs.crs_equal` rather than ``==`` if the spelling could differ.

    The three arrays are **copies**, marked read-only. Building one from a caller's buffer and then sharing
    memory with it would mean a "frozen" value that changes underneath whoever holds it, and `x`/`y`/`z` and
    :meth:`as_xy` hand the arrays straight back out.

    Raises:
        ValueError: if the three arrays are not the same length. They are index-aligned with each other and
            with any attribute column a caller filters alongside them, so a mismatch is a defect that would
            otherwise surface as points drawn at another point's coordinates.

    Examples:
        - Coordinates and a CRS, with z defaulting to zeros:
            ```python
            >>> from digitalearth.base.points import PointArrays
            >>> pts = PointArrays.of([0.0, 1.0], [10.0, 11.0], crs=4326)
            >>> pts.x.tolist(), pts.z.tolist()
            ([0.0, 1.0], [0.0, 0.0])

            ```
        - Non-finite points are dropped, and anything aligned with them is dropped too:
            ```python
            >>> from digitalearth.base.points import PointArrays
            >>> pts = PointArrays.of([0.0, float("inf"), 2.0], [5.0, 6.0, 7.0])
            >>> kept, (names,) = pts.finite(["a", "b", "c"])
            >>> kept.x.tolist(), [str(n) for n in names]
            ([0.0, 2.0], ['a', 'c'])

            ```
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    crs: Any = None

    def __post_init__(self) -> None:
        """Refuse arrays that are not index-aligned.

        Raises:
            ValueError: if `x`, `y` and `z` differ in length.
        """
        lengths = {len(self.x), len(self.y), len(self.z)}
        if len(lengths) != 1:
            raise ValueError(
                f"PointArrays needs x, y and z the same length; got {len(self.x)}, "
                f"{len(self.y)}, {len(self.z)}"
            )

    # ------------------------------------------------------------------ builders

    @classmethod
    def of(cls, x: Any, y: Any, z: Any = None, *, crs: Any = None) -> "PointArrays":
        """Build from coordinate sequences, forcing the dtype every call site used to choose for itself.

        Args:
            x: X coordinates.
            y: Y coordinates.
            z: Z coordinates, or ``None`` for zeros.
            crs: What the coordinates are measured in.

        Returns:
            The arrays as ``float64``, which is what every consumer wanted and only some asked for.

        Raises:
            ValueError: if the sequences are not the same length.

        Examples:
            - Integer input comes back as floats, with z filled in:
                ```python
                >>> from digitalearth.base.points import PointArrays
                >>> pts = PointArrays.of([0, 1], [2, 3], crs=4326)
                >>> pts.x.tolist(), pts.z.tolist()
                ([0.0, 1.0], [0.0, 0.0])

                ```
        """
        # np.array, not np.asarray: asarray does not copy a contiguous float64 input, so the "frozen" value
        # would share memory with the caller's buffer and change underneath it. The arrays are then marked
        # read-only, because `x`/`y`/`z` and `as_xy()` hand them straight back out.
        xs = np.array(x, dtype="float64").ravel()
        ys = np.array(y, dtype="float64").ravel()
        zs = np.zeros_like(xs) if z is None else np.array(z, dtype="float64").ravel()
        for array in (xs, ys, zs):
            array.flags.writeable = False
        return cls(xs, ys, zs, crs)

    @classmethod
    def from_features(
        cls, features: Any, *, centroids: bool = True, crs: Any = None
    ) -> "PointArrays":
        """Extract coordinates from a pyramids ``FeatureCollection`` or a geopandas frame.

        Args:
            features: The collection, frame or geometry series to read. A ``FeatureCollection`` is converted
                through ``to_geodataframe()``, which is the only way this package touches vector data.
            centroids: What to do with geometry that is not a point. ``True`` (default) falls back to the
                centroid, which is what the source extractor does; ``False`` raises instead, for a caller
                whose maths is only meaningful on real points.
            crs: Override the CRS to record. Defaults to whatever the input declares.

        Returns:
            The coordinates, with `z` taken from the geometry when **every** point is 3-D, and zeros
            otherwise. A frame mixing 2-D and 3-D points therefore reads as flat: `has_z.all()` is what
            geopandas can answer cheaply, and a partly-3-D frame has no elevation for some of its points
            anyway.

        Raises:
            TypeError: if `features` exposes no geometry to read.
            ValueError: if `centroids` is ``False`` and the geometry is not all points.

        Examples:
            - The CRS comes from the data unless overridden:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from digitalearth.base.points import PointArrays
                >>> gdf = gpd.GeoDataFrame(geometry=[Point(0, 1), Point(2, 3)], crs="EPSG:4326")
                >>> pts = PointArrays.from_features(gdf)
                >>> pts.x.tolist(), pts.crs.to_epsg()
                ([0.0, 2.0], 4326)

                ```
        """
        frame = (
            features.to_geodataframe()
            if hasattr(features, "to_geodataframe")
            else features
        )
        geom = getattr(frame, "geometry", frame)
        # geom_type first: geopandas raises ValueError from `.x` on a non-point series rather than
        # answering, so `hasattr(geom, "x")` raises too instead of returning False.
        if not hasattr(geom, "geom_type"):
            raise TypeError(
                f"PointArrays.from_features needs vector data with a geometry; got "
                f"{type(features).__name__}"
            )
        if not bool((geom.geom_type == "Point").all()):
            if not centroids:
                raise ValueError(
                    "PointArrays.from_features got geometry that is not all points, and centroids=False. "
                    "Pass centroids=True to use each geometry's centre point"
                )
            geom = geom.centroid
        has_z = bool(getattr(geom, "has_z", None) is not None and geom.has_z.all())
        return cls.of(
            geom.x.to_numpy(),
            geom.y.to_numpy(),
            geom.z.to_numpy() if has_z else None,
            crs=crs if crs is not None else getattr(geom, "crs", None),
        )

    # ------------------------------------------------------------------ readers

    def __eq__(self, other: Any) -> bool:
        """Compare by coordinates and CRS, element by element.

        Args:
            other: The value to compare with.

        Returns:
            ``True`` when both hold the same points in the same order, in the same CRS.

            The dataclass-generated ``__eq__`` compared the arrays with ``==``, which yields an array and
            then raises *"The truth value of an array with more than one element is ambiguous"* — so two
            perfectly ordinary instances could not be compared at all. Points are bulk data, so this type is
            deliberately **not** hashable: `Bounds` and `Selection` key caches, this one does not.

        Examples:
            - Two identical readings compare equal:
                ```python
                >>> from digitalearth.base.points import PointArrays
                >>> PointArrays.of([0.0, 1.0], [2.0, 3.0]) == PointArrays.of([0.0, 1.0], [2.0, 3.0])
                True

                ```
        """
        if not isinstance(other, PointArrays):
            return NotImplemented
        return (
            self.crs == other.crs
            and np.array_equal(self.x, other.x)
            and np.array_equal(self.y, other.y)
            and np.array_equal(self.z, other.z)
        )

    def __len__(self) -> int:
        """Return how many points there are.

        Returns:
            The point count.
        """
        return len(self.x)

    def finite(
        self, *aligned: Any, dims: str = "xy"
    ) -> Tuple["PointArrays", Tuple[Optional[np.ndarray], ...]]:
        """Drop points whose coordinates are not finite, taking aligned arrays with them.

        The far side of a clipped or globe display CRS reprojects to ``inf``/``nan``. Passing those on does
        not raise where they are made — it fails later, inside a tessellator, a binner or matplotlib, with a
        message about neither the points nor the projection.

        Args:
            *aligned: Optional per-point arrays to filter by the same mask — a value column, an id column.
                A ``None`` entry passes through as ``None``, so a caller with an optional column needs no
                branch of its own.
            dims: Which coordinates have to be finite. The default ``"xy"`` is deliberate: most consumers are
                2-D, and a point with a perfectly good position but an unknown *elevation* is still a point
                they can draw. Folding `z` in by default silently changed the topology of a 2-D Delaunay
                tessellation. Pass ``"xyz"`` where a missing z really does make the point unusable.

        Returns:
            A ``(points, aligned)`` pair: the surviving points, and the filtered arrays in the order given.

        Examples:
            - Points on the far side of a globe are removed:
                ```python
                >>> from digitalearth.base.points import PointArrays
                >>> pts = PointArrays.of([0.0, float("nan")], [1.0, 2.0])
                >>> kept, _ = pts.finite()
                >>> len(kept)
                1

                ```
            - An unknown elevation does not remove a point a 2-D consumer can still draw:
                ```python
                >>> from digitalearth.base.points import PointArrays
                >>> pts = PointArrays.of([0.0, 1.0], [2.0, 3.0], [9.0, float("nan")])
                >>> len(pts.finite()[0]), len(pts.finite(dims="xyz")[0])
                (2, 1)

                ```
        """
        axes = {"x": self.x, "y": self.y, "z": self.z}
        unknown = sorted(set(dims) - set(axes))
        if unknown:
            raise ValueError(
                f"finite() takes dims made of 'x', 'y' and 'z'; got {dims!r}, which names {unknown}"
            )
        mask = np.ones(len(self.x), dtype=bool)
        for name in dims:
            mask &= np.isfinite(axes[name])
        # Boolean indexing already copies, so these are fresh buffers; marking them read-only keeps the
        # guarantee `of` makes, since finite() is the other way an instance is built.
        kept_x, kept_y, kept_z = self.x[mask], self.y[mask], self.z[mask]
        for array in (kept_x, kept_y, kept_z):
            array.flags.writeable = False
        kept = PointArrays(kept_x, kept_y, kept_z, self.crs)
        filtered = tuple(
            None if item is None else np.asarray(item)[mask] for item in aligned
        )
        return kept, filtered

    def as_columns(self) -> np.ndarray:
        """Return the points as an ``(N, 3)`` array, the shape the 3-D tiers stack by hand.

        Returns:
            One row per point, columns ``x``, ``y``, ``z``.

        Examples:
            - The shape PyVista and deck.gl both want:
                ```python
                >>> from digitalearth.base.points import PointArrays
                >>> PointArrays.of([0.0, 1.0], [2.0, 3.0]).as_columns().shape
                (2, 3)

                ```
        """
        return np.column_stack([self.x, self.y, self.z])

    def as_xy(self) -> List[np.ndarray]:
        """Return ``[x, y]`` for the many 2-D consumers that want exactly that.

        Returns:
            The two coordinate arrays, so a 2-D call site never has to know `z` exists.

        Examples:
            - Unpacked straight into a 2-D consumer:
                ```python
                >>> from digitalearth.base.points import PointArrays
                >>> x, y = PointArrays.of([0.0, 1.0], [2.0, 3.0]).as_xy()
                >>> x.tolist(), y.tolist()
                ([0.0, 1.0], [2.0, 3.0])

                ```
        """
        return [self.x, self.y]
