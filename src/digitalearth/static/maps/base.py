"""GeoLayerBase — a Scene with a display CRS plus the shared reproject/extract plumbing.

The protected base every Map capability mixin builds on: it owns the display-CRS state (``crs`` / ``domain`` /
``globe`` and the projection-frame caches set up in ``__init__``) and the reproject-to-display-CRS and
Source-extraction helpers the plotting mixins consume via ``self``.
"""

import logging
from typing import Any

import numpy as np
from matplotlib.animation import FuncAnimation

from digitalearth.base.crs import OffLimbError, reproject
from digitalearth.base.display import needs_reproject
from digitalearth.base.sources import get_source
from digitalearth.base.sources.source import Source
from digitalearth.base.spec import DEFAULT_BAND, Viewport
from digitalearth.static.scene import Scene

logger = logging.getLogger(__name__)


class GeoLayerBase(Scene):
    """A :class:`~digitalearth.static.scene.Scene` with a display CRS and reproject/extract helpers.

    The protected base `Map` inherits and every capability mixin under `static/maps/` is typed against. It
    adds two things to a `Scene`: the **display-CRS state** a geospatial figure is drawn in, and the
    **reproject-then-extract plumbing** every builder funnels its data through — `_reproject` to place a
    pyramids `Dataset` in that CRS, then `_prepare` to read it as a uniform
    `digitalearth.base.sources.source.Source`. A builder therefore never warps or reads a raster itself; it
    asks `self._prepare(dataset, band)` and hands what comes back to a cleopatra glyph.

    It is not instantiated directly. `Map` is the composition callers use, and at runtime the mixins' own
    `_MixinBase` is plain `object`, so this class appears in the MRO once, through `Map` alone.

    Attributes:
        crs: The display CRS every layer is placed in before drawing. Anything pyramids resolves — an EPSG
            code, an `"EPSG:3857"` string, a proj string from `digitalearth.static.projections`. Reassigning
            it does not redraw what is already on the axes.
        domain: The named region or bbox the map was built on, or `None`. It is what the view *declares*;
            `ProjectionMixin.set_domain` is what frames the axes from it.
        globe: Whether the map is drawn on a globe frame — the projection boundary, the limb clipping and
            the graticule applied once, at `render()`/`save()`/`show()` rather than as each layer is built.

    Examples:
        - The state a `Map` carries from here. The axes limits are deliberately not consulted, so an
          unframed figure says `bounds=None` rather than reporting matplotlib's unit square as an extent:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth import Map
            >>> m = Map(crs=4326, domain="europe")
            >>> m.crs, m.domain, m.globe
            (4326, 'europe', False)
            >>> view = m.viewport
            >>> view.crs, view.bounds, view.domain
            (4326, None, 'europe')
            >>> m.close()

            ```
        - `Map.viewport` is **not** this property: `ProjectionMixin` overrides it and comes first in the
          MRO. Framing the map shows the two apart — the override reports the region and drops the domain,
          this one still reports the domain, and the `domain` attribute is untouched by either:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth import Map
            >>> from digitalearth.static.maps.base import GeoLayerBase
            >>> m = Map(crs=4326, domain="europe")
            >>> _ = m.set_domain()
            >>> m.viewport.bounds.as_bbox(), m.viewport.domain
            ([-25.0, 34.0, 45.0, 72.0], None)
            >>> base_view = GeoLayerBase.viewport.fget(m)
            >>> base_view.bounds, base_view.domain, m.domain
            (None, 'europe', 'europe')
            >>> m.close()

            ```
        - The extract half, on a bare array. It carries no CRS, so there is nothing to warp from and it is
          placed at its own indices — with the row axis handed over **descending**, which is why both
          `field` and `contours` draw row 0 at the top (#343):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from digitalearth import Map
            >>> m = Map(crs=4326)
            >>> arr = np.arange(12.0).reshape(3, 4)
            >>> m._reproject(arr) is arr
            True
            >>> source = m._prepare(arr)
            >>> [float(value) for value in source.y.values]
            [2.0, 1.0, 0.0]
            >>> [float(value) for value in source.x.values]
            [0.0, 1.0, 2.0, 3.0]
            >>> source.z.values.shape
            (3, 4)
            >>> m.close()

            ```
        - A non-2-D array is refused by the extractor rather than drawn flat:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from digitalearth import Map
            >>> m = Map(crs=4326)
            >>> try:
            ...     m._prepare(np.arange(8.0).reshape(2, 2, 2))
            ... except ValueError as error:
            ...     print(error)
            numpy Source expects a 2-D array, got 3-D
            >>> m.close()

            ```

    See Also:
        digitalearth.static.map.Map: the composition callers build, which this class supplies the state for.
        digitalearth.static.scene.Scene: the engine-level figure/axes host underneath it.
    """

    def __init__(
        self,
        crs: Any = 3857,
        domain: Any = None,
        ax: Any = None,
        fig: Any = None,
        figsize: tuple[float, float] = (8, 8),
        globe: bool = False,
        strict: bool = False,
    ):
        """Build the geospatial layer host: a scene plus a display CRS to reproject into.

        Args:
            crs: Display CRS every layer is reprojected to before drawing (EPSG code or anything pyramids
                resolves). Defaults to Web Mercator.
            domain: Optional named region or bbox setting the initial extent.
            ax: An existing axes to draw on; the scene then does not own the figure. One axes holds one map:
                a second map on the same axes wipes the first one's drawing (see `Scene`).
            fig: The figure `ax` belongs to; taken from `ax` when omitted.
            figsize: Size of the figure created when `ax` is None, in inches.
            globe: Draw on a globe frame rather than a flat projection.
            strict: Raise `OffLimbError` for a layer with nothing to draw instead of skipping it with a
                warning. See `Scene.__init__` for the trade-off.
        """
        super().__init__(ax=ax, fig=fig, figsize=figsize, strict=strict)
        self.crs = crs
        self.domain = domain
        self.globe = globe
        self._graticule_lines: list[np.ndarray] | None = None  # set by graticule()
        self._last_vector: tuple | None = (
            None  # (glyph, artist, kind) of the most recent vector layer
        )
        self._animation: FuncAnimation | None = (
            None  # last animate()/rotate() result (kept alive, L3)
        )
        self._animation_fps: float | None = (
            None  # rate the last animate()/rotate() was built at; save_animation's default
        )
        self._framed = False
        self._frame_cache: tuple | None = None  # (crs, (boundary, xlim, ylim)) memo
        self._graticule_id: str | None = (
            None  # the described graticule, so a redraw replaces it
        )

    @property
    def viewport(self) -> Viewport:
        """Where the map is looking, as a value — the declared half of it.

        The axes limits are deliberately *not* read here. They are whatever the last draw autoscaled them to
        — matplotlib's unit square on an empty figure — so a view taken from them would say a map showed a
        one-by-one metre rectangle off the coast of Africa. What the map was *asked* to show is the display
        CRS and the domain, and those are what a figure carries.

        **This is not the property a `Map` answers with.** `ProjectionMixin` overrides it and sits ahead of
        this class in the MRO, so `Map.viewport` is that one: it starts from the view built here and then
        adds the region the last framing call settled on. Three classes in the `Map` MRO define the name —
        `ProjectionMixin`, this class and `Scene` — and only the first is reached through an instance. Read
        this one directly (`GeoLayerBase.viewport.fget(m)`) and the two can disagree about the same map,
        which is the next paragraph.

        Because a `Viewport` holds a region **or** a named domain and never both, framing a map **clears the
        domain from the view** while leaving the `domain` attribute alone. Measured on
        `Map(crs=4326, domain="europe")` after `set_domain()`: `Map.viewport` answers
        `bounds=Bounds(-25.0, 34.0, 45.0, 72.0, crs=4326)` with `domain=None`, this property still answers
        `domain='europe'` with `bounds=None`, and `m.domain` is still `'europe'`. So the domain does not
        travel beside a frame through a stored figure — the resolved rectangle replaces it, which is what
        keeps a stale domain from sitting next to a live frame.

        Returns:
            A :class:`~digitalearth.base.spec.Viewport` in the CRS this tier places data in, carrying the
            declared domain and whether the map is drawn on a globe frame. Never a `bounds` — that is the
            override's to add.

        See Also:
            digitalearth.static.maps.projection.ProjectionMixin.viewport: what `Map.viewport` actually is.
        """
        return Viewport(crs=self.crs, domain=self.domain, globe=bool(self.globe))

    def _needs_reproject(self, dataset: Any) -> bool:
        """Whether `dataset` must be reprojected to the display CRS.

        A thin alias for :func:`digitalearth.base.display.needs_reproject`, kept because tier code and tests
        call it.

        Args:
            dataset: A pyramids `Dataset` whose CRS is compared against the display CRS.

        Returns:
            `False` when the dataset's CRS and the display CRS name the same reference system, however either
            is spelled (`3857`, `"EPSG:3857"`, a pyproj `CRS`); `True` otherwise.
        """
        return needs_reproject(dataset, self.crs)

    def _skipped_off_limb(self, layer: str) -> None:
        """Record that ``layer`` drew nothing because its data is outside the display CRS.

        Under ``strict=True`` (set on the constructor) nothing is recorded: the skip is refused and an
        :class:`~digitalearth.base.crs.OffLimbError` is raised naming the layer, so a pipeline that must
        not produce a silently empty figure fails at the layer that would have been dropped.

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

        Raises:
            OffLimbError: when this scene was built with ``strict=True``.
        """
        if self.strict:
            raise OffLimbError(
                f"{layer}: none of the data can be placed in {self.crs!r}, so the layer would draw "
                "nothing (strict=True; pass strict=False to skip it with a warning instead)"
            )
        if self.globe:
            logger.debug("%s: data lies outside %r; nothing drawn", layer, self.crs)
        else:
            logger.warning(
                "%s: none of the data could be placed in %r, so nothing was drawn — on an unclipped "
                "projection this usually means the raster's CRS or geo-transform is wrong",
                layer,
                self.crs,
            )

    def _prepare(self, dataset: Any, band: int = DEFAULT_BAND) -> Source:
        """Reproject ``dataset`` to the display CRS (if needed) and wrap it as a :class:`Source`.

        **A bare numpy array is placed at its own indices**, not in the display CRS: it carries no CRS and no
        geo-transform, so there is nothing to warp from and nothing that says where its cells are. The
        column index becomes ``x`` and the row index ``y``, which is what
        :func:`~digitalearth.base.sources.extractors._from_numpy` already does — except that its rows run
        *up* (``y = 0`` for row 0), and every render on this tier draws an array's first row at the **top**.
        Left ascending, one array came out mirrored between two renders of it: ``field`` places its image by
        extent and matplotlib draws row 0 at ``ymax``, while ``pcolormesh``/``contours`` place their cells by
        the coordinates and drew row 0 at ``ymin``. So the row axis is handed over descending, which is the
        convention every pyramids raster already arrives in (north-up, ``y`` decreasing) and the one that
        makes the tier's renders agree (#343).

        The convention is set here, in this tier's own choke point, rather than in ``_from_numpy``: that
        extractor's ascending axis is what the 3-D tier's ``terrain`` and the web tier's placeholder read,
        and flipping it there would flip surfaces those tiers already draw.

        Args:
            dataset: The pyramids ``Dataset`` to place in the display CRS and read, or a bare 2-D numpy
                array to place at its own indices.
            band: 1-based band to extract.

        Returns:
            The dataset as a uniform :class:`Source` view.

        Raises:
            OffLimbError: when the data lies outside what the display CRS can show.
            ValueError: when `dataset` is a numpy array that is not 2-D, from the extractor.
        """
        placed = self._reproject(dataset)
        if isinstance(placed, np.ndarray) and placed.ndim == 2:
            rows = placed.shape[0]
            return get_source(
                placed, band=band, y=np.arange(rows - 1, -1, -1, dtype="float64")
            )
        return get_source(placed, band=band)

    def _reproject(self, dataset: Any) -> Any:
        """Reproject a pyramids ``Dataset`` to the display CRS (returns it unchanged when already there).

        **Data with no reprojection to make is handed straight back.** A bare numpy array declares no CRS and
        has no ``to_crs``, so there is nothing to warp *from*: asking anyway is what leaked
        ``AttributeError: 'numpy.ndarray' object has no attribute 'to_crs'`` out of ``Map.field`` (#343). It
        is the carve-out :func:`~digitalearth.base.display.to_display_source` already makes for the same
        input, said here because this tier reprojects through its own choke point rather than through that
        one. What a builder will and will not accept is not decided here — that is
        :func:`~digitalearth.base.sources.require_drawable`, at the call — so an input this passes over
        still meets whatever the reader after it needs.

        Args:
            dataset: The pyramids ``Dataset`` to place in the display CRS.

        Returns:
            The reprojected dataset, or ``dataset`` itself when it is already in the display CRS or has no
            CRS to be warped out of.

        Raises:
            OffLimbError: when the warp reports too few surviving sample points to bound an output, i.e. the
                data is outside the projection's visible area. Any *other* ``RuntimeError`` is re-raised as
                it came — a real projection failure must not be mistaken for an empty view.
        """
        if not hasattr(dataset, "to_crs") or not self._needs_reproject(dataset):
            return dataset
        return reproject(dataset, self.crs)
