"""GeoLayerBase — a Scene with a display CRS plus the shared reproject/extract plumbing.

The protected base every Map capability mixin builds on: it owns the display-CRS state (``crs`` / ``domain`` /
``globe`` and the projection-frame caches set up in ``__init__``) and the reproject-to-display-CRS and
Source-extraction helpers the plotting mixins consume via ``self``.
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib.animation import FuncAnimation

from digitalearth.base.crs import OffLimbError, reproject
from digitalearth.base.sources import get_source
from digitalearth.base.sources.source import Source
from digitalearth.static.scene import Scene

logger = logging.getLogger(__name__)


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

    def _skipped_off_limb(self, layer: str) -> None:
        """Record that ``layer`` drew nothing because its data is outside the display CRS.

        The severity depends on whether hiding the data is a normal thing for this map to do. On a globe it
        is: a clipped projection shows one hemisphere, and a rotation sweeps past the far side on every
        run, so those skips are logged at debug and stay out of the way. On an unclipped display CRS there
        is no limb to be behind, so a warp that places *none* of the data almost always means the raster is
        mislabelled or its geo-transform is wrong — that is worth a warning, which is visible without any
        logging setup, because otherwise the only symptom is a blank figure.

        The guard cannot tell the two apart from the warp alone: GDAL reports that too few points survived,
        not why. This is the signal that lets a reader tell a hidden hemisphere from a broken raster.

        Args:
            layer: The public layer method that drew nothing, named for the log line.
        """
        if self.globe:
            logger.debug("%s: data lies outside %r; nothing drawn", layer, self.crs)
        else:
            logger.warning(
                "%s: none of the data could be placed in %r, so nothing was drawn — on an unclipped "
                "projection this usually means the raster's CRS or geo-transform is wrong",
                layer,
                self.crs,
            )

    def _prepare(self, dataset: Any, band: int = 1) -> Source:
        """Reproject ``dataset`` to the display CRS (if needed) and wrap it as a :class:`Source`.

        Args:
            dataset: The pyramids ``Dataset`` to place in the display CRS and read.
            band: 1-based band to extract.

        Returns:
            The dataset as a uniform :class:`Source` view.

        Raises:
            OffLimbError: when the data lies outside what the display CRS can show.
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
        return reproject(dataset, self.crs)
