"""ProjectionMixin — extent/domain, the globe projection frame, and render/save/show hooks.

Sets the axes extent from a bbox or named domain, builds and caches the projection boundary/graticule for a
globe map, and overrides ``save``/``show`` to apply that frame before output.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from cleopatra.basemap.projection import apply_projection_frame

from digitalearth.base.spec import Bounds
from digitalearth.static import projections
from digitalearth.static.domains import DomainLike, resolve_domain

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class ProjectionMixin(_MixinBase):
    """Extent/domain and globe projection-frame behaviour for :class:`~digitalearth.static.map.Map`.

    A capability mixin of :class:`~digitalearth.static.map.Map`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the shared figure/axes, the layer registry and the
    display CRS — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.static.maps.base.GeoLayerBase`, the state ``Map``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``Map`` MRO exactly what
    it was before the annotation.

    See Also:
        digitalearth.static.map.Map: the composition that supplies the state these methods use.
        digitalearth.static.maps.base.GeoLayerBase: the typing-only base declared above the class.
    """

    def set_extent(self, bbox: Union[Bounds, Sequence[float]]) -> None:
        """Set the axes extent.

        Args:
            bbox: A :class:`~digitalearth.base.spec.bounds.Bounds` in **any** CRS — it is reprojected to the
                display CRS, which is the point of passing one — or a bare ``[xmin, xmax, ymin, ymax]``
                sequence in matplotlib axes order, assumed to be in the display CRS already. The sequence
                form is accepted because that ordering was this method's contract; prefer `Bounds`, which
                states both the ordering and the CRS instead of leaving them to position and assumption.

        Raises:
            ValueError: if the sequence form does not hold exactly four values.

        Notes:
            A **flipped** pair is honoured in the sequence form: ``[10, 0, 0, 10]`` inverts the x axis, which
            is how matplotlib expresses ``invert_xaxis`` through the limits. A `Bounds` cannot express that —
            it refuses corners the wrong way round, because for a rectangle handed to pyramids or cleopatra
            that is a defect rather than an intent — so invert the axis directly if you need both.
        """
        if isinstance(bbox, Bounds):
            # to_crs is a no-op when the CRSs already match. Without it a rectangle that carries its CRS
            # would be trusted to be in the display one, which is exactly the mistake Bounds exists to stop.
            xmin, xmax, ymin, ymax = bbox.to_crs(self.crs).as_mpl()
        else:
            # Not routed through Bounds: axes limits may legitimately run backwards, and Bounds refuses that.
            values = [float(value) for value in bbox]
            if len(values) != 4:
                raise ValueError(
                    f"set_extent needs exactly 4 values as [xmin, xmax, ymin, ymax]; got {len(values)}"
                )
            xmin, xmax, ymin, ymax = values
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

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
                >>> from digitalearth.static import Map
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
        # A resolved domain is always EPSG:4326 (see `static/domains.py`), which is exactly the assumption
        # a bare 4-tuple used to carry implicitly. Bounds makes it a value, and does the warp through pyramids.
        self.set_extent(Bounds.from_bbox(bbox, crs=4326).to_crs(self.crs))

    # ------------------------------------------------------------------ globe / projection frame

    def graticule(self, lon_step: float = 30.0, lat_step: float = 30.0) -> None:
        """Add a lon/lat graticule to a projected map (drawn when the frame is applied).

        Args:
            lon_step: Meridian spacing in degrees.
            lat_step: Parallel spacing in degrees.
        """
        self._graticule_lines = projections.graticule(
            self.crs, lon_step=lon_step, lat_step=lat_step
        )

    def _frame(self) -> tuple:
        """Return the cached ``(boundary, xlim, ylim)`` for the display CRS (computed once per CRS).

        ``projection_frame`` reprojects a dense lon/lat sample of the whole sphere, so it is memoised here to
        avoid recomputing it for both ``set_global`` and ``_apply_frame``. The cache is keyed on the display
        CRS and recomputed only when the CRS changes.

        Returns:
            The ``(boundary_xy, (xmin, xmax), (ymin, ymax))`` tuple from
            :func:`digitalearth.static.projections.projection_frame` for the current display CRS — a closed
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
            self.ax,
            boundary_xy=boundary,
            xlim=xlim,
            ylim=ylim,
            graticule_lines=self._graticule_lines,
        )
        self._framed = True
        return patch

    def render(self) -> None:
        """Apply the projection frame if this is a globe map (idempotent). Call before showing/saving."""
        self._apply_frame()

    def save(self, path: Union[str, "os.PathLike[str]"], **kwargs) -> Path:
        """Apply the projection frame (for a globe map) then save the figure.

        Args:
            path: Destination file path; the extension picks the format matplotlib writes.
            **kwargs: Forwarded to :meth:`~digitalearth.static.scene.Scene.save` / ``Figure.savefig``.

        Returns:
            The path that was written, as a :class:`pathlib.Path`.
        """
        self._apply_frame()
        return super().save(path, **kwargs)

    def show(self) -> None:
        """Apply the projection frame (for a globe map) then show the figure."""
        self._apply_frame()
        super().show()
