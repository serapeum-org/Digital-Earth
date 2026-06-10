"""GeoLayerBase — a Scene with a display CRS plus the shared reproject/extract plumbing.

The protected base every Map capability mixin builds on: it owns the display-CRS state (``crs`` / ``domain`` /
``globe`` and the projection-frame caches set up in ``__init__``) and the reproject-to-display-CRS and
Source-extraction helpers the plotting mixins consume via ``self``.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib.animation import FuncAnimation

from digitalearth.scene.scene import Scene
from digitalearth.sources import get_source
from digitalearth.sources.source import Source


class GeoLayerBase(Scene):
    """A :class:`~digitalearth.scene.scene.Scene` with a display CRS and reproject/extract helpers."""

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
        self._last_vector: Optional[tuple] = None  # (glyph, artist, kind) of the most recent vector layer
        self._animation: Optional[FuncAnimation] = None  # last animate()/rotate() result (kept alive, L3)
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
        """Reproject ``dataset`` to the display CRS (if needed) and wrap it as a :class:`Source`."""
        ds = dataset.to_crs(self.crs) if self._needs_reproject(dataset) else dataset
        return get_source(ds, band=band)


    def _reproject(self, dataset: Any) -> Any:
        """Reproject a pyramids ``Dataset`` to the display CRS (returns it unchanged when already there)."""
        return dataset.to_crs(self.crs) if self._needs_reproject(dataset) else dataset

