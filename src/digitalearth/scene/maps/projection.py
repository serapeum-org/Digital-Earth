"""ProjectionMixin — extent/domain, the globe projection frame, and render/save/show hooks.

Sets the axes extent from a bbox or named domain, builds and caches the projection boundary/graticule for a
globe map, and overrides ``save``/``show`` to apply that frame before output.
"""
from typing import Any, Optional, Sequence

from cleopatra.projection import apply_projection_frame
from pyramids.base.crs import reproject_coordinates

from digitalearth.scene import projections
from digitalearth.scene.domains import DomainLike, resolve_domain


class ProjectionMixin:
    """Extent/domain and globe projection-frame behaviour for :class:`~digitalearth.scene.map.Map`."""

    def set_extent(self, bbox: Sequence[float]) -> None:
        """Set the axes extent.

        Args:
            bbox: ``[xmin, xmax, ymin, ymax]`` in the display CRS.
        """
        self.ax.set_xlim(bbox[0], bbox[1])
        self.ax.set_ylim(bbox[2], bbox[3])

    def set_domain(self, domain: Optional[DomainLike] = None) -> None:
        """Set the axes extent from a named region or bbox, reprojected to the display CRS via pyramids.

        Args:
            domain: A registered region name (e.g. ``"Europe"``), an explicit ``(west, south, east, north)``
                bbox in EPSG:4326, or ``None`` to fall back to the domain passed at construction. A no-op
                when neither resolves to a domain.

        Examples:
            - In a geographic CRS the axes limits equal the named region's bounds:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.scene import Map
                >>> m = Map(crs=4326)
                >>> m.set_domain("europe")
                >>> [float(v) for v in m.ax.get_xlim()]
                [-25.0, 45.0]
                >>> [float(v) for v in m.ax.get_ylim()]
                [34.0, 72.0]

                ```
        """
        bbox = resolve_domain(domain if domain is not None else self.domain)
        if bbox is None:
            return
        west, south, east, north = bbox
        xs, ys = reproject_coordinates(
            [west, east, west, east], [south, south, north, north], from_crs=4326, to_crs=self.crs
        )
        self.set_extent([min(xs), max(xs), min(ys), max(ys)])

    # ------------------------------------------------------------------ globe / projection frame

    def graticule(self, lon_step: float = 30.0, lat_step: float = 30.0) -> None:
        """Add a lon/lat graticule to a projected map (drawn when the frame is applied).

        Args:
            lon_step: Meridian spacing in degrees.
            lat_step: Parallel spacing in degrees.
        """
        self._graticule_lines = projections.graticule(self.crs, lon_step=lon_step, lat_step=lat_step)

    def _frame(self) -> tuple:
        """Return the cached ``(boundary, xlim, ylim)`` for the display CRS (computed once per CRS).

        ``projection_frame`` reprojects a dense lon/lat sample of the whole sphere, so it is memoised here to
        avoid recomputing it for both ``set_global`` and ``_apply_frame``. The cache is keyed on the display
        CRS and recomputed only when the CRS changes.

        Returns:
            The ``(boundary_xy, (xmin, xmax), (ymin, ymax))`` tuple from
            :func:`digitalearth.scene.projections.projection_frame` for the current display CRS — a closed
            ``(N, 2)`` boundary ring plus the projected x/y limits.
        """
        if self._frame_cache is None or self._frame_cache[0] != self.crs:
            self._frame_cache = (self.crs, projections.projection_frame(self.crs))
        return self._frame_cache[1]

    def set_global(self) -> None:
        """Set the axes extent to the full projection domain (the whole globe/world)."""
        _, xlim, ylim = self._frame()
        self.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])

    def _apply_frame(self) -> Any:
        """Draw the projection boundary + graticule and clip the layers to it (once, at render time)."""
        if not self.globe or self._framed:
            return None
        boundary, xlim, ylim = self._frame()
        patch = apply_projection_frame(
            self.ax, boundary_xy=boundary, xlim=xlim, ylim=ylim,
            graticule_lines=self._graticule_lines,
        )
        self._framed = True
        return patch

    def render(self) -> None:
        """Apply the projection frame if this is a globe map (idempotent). Call before showing/saving."""
        self._apply_frame()

    def save(self, path: str, **kwargs) -> None:
        """Apply the projection frame (for a globe map) then save the figure."""
        self._apply_frame()
        super().save(path, **kwargs)

    def show(self) -> None:
        """Apply the projection frame (for a globe map) then show the figure."""
        self._apply_frame()
        super().show()

