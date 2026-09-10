"""GeoLayerBase — a Scene with a display CRS plus the shared reproject/extract plumbing.

The protected base every Map capability mixin builds on: it owns the display-CRS state (``crs`` / ``domain`` /
``globe`` and the projection-frame caches set up in ``__init__``) and the reproject-to-display-CRS and
Source-extraction helpers the plotting mixins consume via ``self``.
"""

import re
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib.animation import FuncAnimation

from digitalearth.base.sources import get_source
from digitalearth.base.sources.source import Source
from digitalearth.static.scene import Scene

#: GDAL's complaint when a warp cannot place the data in the target CRS. It fires as soon as too few sample
#: points survive to bound an output — its own threshold is ``failed > total - 10``, not all of them — and by
#: then it has already refused to compute those bounds. So *any* occurrence means there is no output raster,
#: whatever the counts say; a warp that does produce output never raises it, which is why the counts are not
#: read here.
_POINTS_FAILED = re.compile(r"Too many points \(\d+ out of \d+\) failed to transform")


class OffLimbError(RuntimeError):
    """The data lies entirely outside the area the display CRS can represent.

    Raised in place of GDAL's opaque "Too many points ... failed to transform", which on an orthographic
    globe means the data sits behind the visible limb. Layer methods
    treat it as "there is nothing to draw here" and render an empty frame; it is a distinct type so that a
    caller can tell it apart from a real projection failure.

    Examples:
        - The reprojection reports a hidden dataset as such, rather than as a GDAL failure:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static import Map, projections
            >>> from digitalearth.static.maps.base import OffLimbError
            >>> ds = Dataset.from_array(
            ...     np.ones((20, 20), "float32"),
            ...     geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
            ... )
            >>> hidden = Map(crs=projections.orthographic(lon=-175, lat=15), globe=True)
            >>> try:
            ...     hidden._reproject(ds)
            ... except OffLimbError as error:
            ...     print(str(error).split(':')[-1].strip())
            too few sample points survive the warp for it to produce any output

            ```
        - Callers rarely see it: the layer methods answer it by drawing nothing:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static import Map, projections
            >>> ds = Dataset.from_array(
            ...     np.ones((20, 20), "float32"),
            ...     geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
            ... )
            >>> hidden = Map(crs=projections.orthographic(lon=-175, lat=15), globe=True)
            >>> hidden.imshow(ds) is None
            True
            >>> len(hidden.ax.images)
            0

            ```
    """


class GeoLayerBase(Scene):
    """A :class:`~digitalearth.static.scene.Scene` with a display CRS and reproject/extract helpers."""

    def __init__(
        self,
        crs: Any = 3857,
        domain: Any = None,
        ax: Any = None,
        fig: Any = None,
        figsize: Tuple[float, float] = (8, 8),
        globe: bool = False,
    ):
        super().__init__(ax=ax, fig=fig, figsize=figsize)
        self.crs = crs
        self.domain = domain
        self.globe = globe
        self._graticule_lines: Optional[List[np.ndarray]] = None  # set by graticule()
        self._last_vector: Optional[tuple] = (
            None  # (glyph, artist, kind) of the most recent vector layer
        )
        self._animation: Optional[FuncAnimation] = (
            None  # last animate()/rotate() result (kept alive, L3)
        )
        self._framed = False
        self._frame_cache: Optional[tuple] = None  # (crs, (boundary, xlim, ylim)) memo

    def _needs_reproject(self, dataset: Any) -> bool:
        """Whether ``dataset`` must be reprojected to the display CRS.

        Only an EPSG-int display CRS can be compared cheaply against ``dataset.epsg``. For a proj4/string
        display CRS (e.g. an orthographic globe) we always reproject — and ``dataset.epsg`` is unreliable for
        non-EPSG results anyway (pyramids returns 4326 for a no-code projection), so we never compare against
        a proj4 CRS structurally here.

        Args:
            dataset: A pyramids ``Dataset`` whose ``.epsg`` is compared against the display CRS.

        Returns:
            ``False`` only when the display CRS is an ``int`` equal to ``dataset.epsg`` (data already in the
            display CRS); ``True`` otherwise — i.e. for a differing EPSG code or any proj4/string CRS.
        """
        return not (isinstance(self.crs, int) and dataset.epsg == self.crs)

    def _prepare(self, dataset: Any, band: int = 1) -> Source:
        """Reproject ``dataset`` to the display CRS (if needed) and wrap it as a :class:`Source`.

        Raises:
            OffLimbError: when the data lies entirely outside what the display CRS can show.
        """
        return get_source(self._reproject(dataset), band=band)

    def _reproject(self, dataset: Any) -> Any:
        """Reproject a pyramids ``Dataset`` to the display CRS (returns it unchanged when already there).

        Args:
            dataset: The pyramids ``Dataset`` to place in the display CRS.

        Returns:
            The reprojected dataset, or ``dataset`` itself when it is already in the display CRS.

        Raises:
            OffLimbError: when the warp reports too few surviving sample points to bound an output, i.e. the
                data is outside the projection's visible area. Any *other* ``RuntimeError`` is re-raised as
                it came — a real projection failure must not be mistaken for an empty view.
        """
        if not self._needs_reproject(dataset):
            return dataset
        try:
            return dataset.to_crs(self.crs)
        except RuntimeError as error:
            if _POINTS_FAILED.search(str(error)):
                raise OffLimbError(
                    f"the data lies outside what {self.crs!r} can show: too few sample points survive the "
                    f"warp for it to produce any output"
                ) from error
            raise
