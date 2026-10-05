"""figure — multi-panel layout: a grid of :class:`~digitalearth.static.map.Map` panels sharing one figure.

earthkit-plots models a figure as ``Figure → Subplot/Map → Layer``. Digital-Earth keeps the panel itself
(``Map``) as the unit and adds a thin :func:`grid` that creates the matplotlib figure + axes grid and binds a
``Map`` to each axes, plus :func:`shared_colorbar` for one colorbar spanning the panels, and :func:`facet`, which
lays a raster stack out as small multiples on one shared colour scale. This is orchestration only — the
rendering stays in each ``Map`` (pyramids + cleopatra).

What the panels can share is therefore in three places, and they are independent: one **colour** scale
(:func:`shared_colorbar`, and :func:`facet`, which resolves one over a whole stack), one pair of **axis**
scales (:func:`grid`'s ``sharex``/``sharey``, in matplotlib's own vocabulary) and one **title** over the
figure (:func:`grid`'s ``suptitle``). :func:`facet` asks for none of the last two: its frames need not cover
one area, and it titles each panel by what that panel is.
"""

import logging
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pyramids.dataset import Dataset

from digitalearth.base.arrays import read_masked_band
from digitalearth.base.clim import frozen_scale, measure_clim
from digitalearth.base.crs import OffLimbError
from digitalearth.base.raster_classes import (
    asks_categorical,
    code_edges,
    raster_categories,
)
from digitalearth.base.spec import DEFAULT_BAND, Scale
from digitalearth.base.symbology import categorical_colors, resolve_categorical_cmap
from digitalearth.static.map import Map
from digitalearth.static.scene import Scene

logger = logging.getLogger(__name__)

__all__ = ["facet", "grid", "shared_colorbar"]

#: The render each ``facet(kind=)`` names, as ``(Map method, keywords it is called with)``.
_FACET_KINDS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "field": ("field", {}),
    "pcolormesh": ("pcolormesh", {}),
    "contourf": ("contours", {"filled": True}),
    "contour": ("contours", {"filled": False}),
}

#: What ``facet(stack=)`` takes, named in every refusal of something else.
_STACK_FORMS = (
    "a multi-band Dataset (its bands are the panels), a DatasetCollection (anything with a .datasets "
    "sequence), or a sequence of Dataset"
)


def grid(
    nrows: int,
    ncols: int,
    *,
    crs: Any = 3857,
    globe: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: Union[bool, Literal["all", "row", "col", "none"]] = False,
    sharey: Union[bool, Literal["all", "row", "col", "none"]] = False,
    suptitle: Optional[str] = None,
    **kwargs,
) -> Tuple[Figure, List[Map]]:
    """Create an ``nrows`` × ``ncols`` grid of :class:`Map` panels sharing one figure.

    Each cell of a ``matplotlib`` subplot grid is wrapped in a ``Map`` (all the same ``crs``/``globe``), so
    panels can be drawn on independently while sharing one figure for a single ``savefig`` / colorbar /
    title. ``sharex``/``sharey`` additionally put the panels on **one pair of axis scales**, and ``suptitle``
    titles the figure rather than any panel.

    **The sharing vocabulary is matplotlib's own, not a second one.** ``sharex``/``sharey`` are forwarded
    verbatim to ``plt.subplots``, so they take ``False`` (the default — nothing shared), ``True`` or
    ``"all"`` (every panel in one group), ``"row"``, ``"col"`` and ``"none"``, and a word outside that list
    is refused by matplotlib with its own message naming the keyword. There is therefore a *global* and a
    *per-row/per-column* form, and which one you get is the word you pass.

    **Sharing moves the ticks as well as linking the scales.** matplotlib drops the tick labels its sharing
    makes redundant: ``sharex="all"``/``"col"`` leaves x labels on the bottom row only, and
    ``sharey="all"``/``"row"`` leaves y labels on the leftmost column only (both measured below). The
    labels are dropped where sharing makes them redundant *along that axis*, so ``sharex="row"`` and
    ``sharey="col"`` link their groups and leave every panel's labels in place.

    Args:
        nrows: Number of panel rows.
        ncols: Number of panel columns.
        crs: Display CRS for every panel (passed to each ``Map``).
        globe: When True, every panel is a globe (``Map(globe=True)``).
        figsize: Figure size in inches; ``None`` uses the matplotlib default.
        sharex: How the panels share their x axis — ``False`` (default), ``True``/``"all"``, ``"row"``,
            ``"col"`` or ``"none"``, exactly as ``plt.subplots`` reads them, and exactly what the
            signature's own ``Literal`` admits. Sharing links the *limits*:
            framing or autoscaling one panel of a group frames every panel in it, which is the point of
            asking for it, and means a grid whose panels show different regions should leave it off.
        sharey: The same for the y axis.
        suptitle: One title over the whole figure (``Figure.suptitle``). ``None`` (default) adds none,
            and so do ``""`` and ``"   "``: a blank heading is the one request "no heading" the panels'
            own ``set_title`` reads it as, rather than an empty ``Text`` on the figure.
            This is **not** a panel title: every panel's own ``set_title`` is untouched, so a shared
            heading and one caption per panel coexist. Style it by calling ``fig.suptitle`` yourself on
            the returned figure — which each panel describes as its figure's heading either way, since
            every panel reads the heading off the figure it shares
            (:attr:`~digitalearth.static.scene.Scene.figure_spec`).
        **kwargs: Forwarded to each ``Map`` (e.g. ``domain``).

    Returns:
        ``(fig, maps)`` — the shared :class:`~matplotlib.figure.Figure` and the list of ``Map`` panels in
        row-major (left-to-right, top-to-bottom) order, length ``nrows * ncols``.

    Raises:
        ValueError: from ``plt.subplots``, for a ``sharex``/``sharey`` outside the vocabulary above.

    Examples:
        - A 2×2 grid yields four Maps sharing one figure:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static.figure import grid
            >>> fig, maps = grid(2, 2, crs=4326)
            >>> len(maps)
            4
            >>> all(m.fig is fig for m in maps)
            True
            >>> maps[0].close()

            ```
        - Draw on each panel independently (they share the figure):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static.figure import grid
            >>> fig, maps = grid(1, 2, crs=4326)
            >>> _ = maps[0].set_title("left")
            >>> _ = maps[1].set_title("right")
            >>> [m.ax.get_title() for m in maps]
            ['left', 'right']
            >>> maps[0].close()

            ```
        - Shared axes: framing one panel frames its group, and the redundant tick labels go. In a 2×2
          under ``sharex="all"`` only the bottom row keeps x tick labels:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static.figure import grid
            >>> fig, maps = grid(2, 2, crs=4326, sharex="all", sharey="all")
            >>> _ = maps[0].set_bounds([2.0, 3.0, 8.0, 9.0])
            >>> [float(v) for v in maps[3].ax.get_xlim()]
            [2.0, 8.0]
            >>> [m.ax.xaxis.get_tick_params(which="major")["labelbottom"] for m in maps]
            [False, False, True, True]
            >>> [m.ax.yaxis.get_tick_params(which="major")["labelleft"] for m in maps]
            [True, False, True, False]
            >>> maps[0].close()

            ```
        - Sharing per row and per column drops no labels at all, because neither makes a label redundant
          along its own axis:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static.figure import grid
            >>> fig, maps = grid(2, 2, crs=4326, sharex="row", sharey="col")
            >>> [m.ax.xaxis.get_tick_params(which="major")["labelbottom"] for m in maps]
            [True, True, True, True]
            >>> [m.ax.yaxis.get_tick_params(which="major")["labelleft"] for m in maps]
            [True, True, True, True]
            >>> maps[0].close()

            ```
        - Per-column sharing groups the panels of a column — in a 2×2 that is the row-major pair
          ``(0, 2)`` — and a word outside matplotlib's vocabulary is refused by name:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static.figure import grid
            >>> fig, maps = grid(2, 2, crs=4326, sharex="col")
            >>> group = maps[0].ax.get_shared_x_axes().get_siblings(maps[0].ax)
            >>> sorted(id(ax) for ax in group) == sorted(id(m.ax) for m in (maps[0], maps[2]))
            True
            >>> maps[0].close()
            >>> try:
            ...     grid(2, 2, crs=4326, sharex="both")
            ... except ValueError as error:
            ...     print(error)
            'both' is not a valid value for sharex. Supported values are 'all', 'row', 'col', 'none', False, True

            ```
        - A figure title sits beside the panels' own titles rather than replacing one:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static.figure import grid
            >>> fig, maps = grid(1, 2, crs=4326, suptitle="rainfall, 2020")
            >>> _ = maps[0].set_title("January")
            >>> [t.get_text() for t in fig.texts], maps[0].ax.get_title()
            (['rainfall, 2020'], 'January')
            >>> maps[0].close()

            ```
        - Both headings are described, each in its own place, so a stored figure keeps them (M4):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static.figure import grid
            >>> fig, maps = grid(1, 2, crs=4326, suptitle="rainfall, 2020")
            >>> _ = maps[0].set_title("January")
            >>> spec = maps[0].figure_spec
            >>> spec.title, spec.panels[0].title
            ('rainfall, 2020', 'January')
            >>> maps[0].close()

            ```
    """
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axs).ravel()
    maps = [Map(crs=crs, globe=globe, ax=ax, fig=fig, **kwargs) for ax in axes]
    # Through the panels' own normaliser, not a guard of its own: `None`, `""` and `"   "` are the one
    # request "no heading" on this tier (ST-18), and a bare `is not None` put an empty `Text` on the figure
    # where `set_title("")` draws and records none (L4).
    heading = Scene._recorded_title(suptitle)
    if heading is not None:
        fig.suptitle(heading)
    return fig, maps


def shared_colorbar(
    fig: Figure,
    mappable: Any,
    maps: Optional[List[Map]] = None,
    *,
    label: Optional[str] = None,
    **kwargs,
) -> Any:
    """Add one colorbar to ``fig`` spanning the given panels (or every axes when ``maps`` is ``None``).

    Args:
        fig: The figure created by :func:`grid`.
        mappable: A drawn mappable (e.g. an ``AxesImage`` / ``QuadMesh`` returned by a panel's ``field``)
            whose colour scale the bar represents. ``None`` is accepted and adds no bar, since a layer
            whose data lies outside the display CRS draws nothing and so has no scale to represent.
        maps: Panels the colorbar should steal space from; ``None`` spans all of the figure's axes.
        label: Optional colorbar label.
        **kwargs: Forwarded to ``Figure.colorbar`` (e.g. ``orientation``, ``shrink``, ``fraction``).

    Returns:
        The :class:`~matplotlib.colorbar.Colorbar` added to the figure, or ``None`` when ``mappable`` is
        ``None`` — there is no colour scale to draw a bar for.
    """
    if mappable is None:  # the layer it would describe was never drawn (e.g. off-limb)
        return None
    axes = [m.ax for m in maps] if maps is not None else None
    cbar = fig.colorbar(mappable, ax=axes, **kwargs)
    if label is not None:
        cbar.set_label(label)
    return cbar


def _frames_of(stack: Any) -> List[Any]:
    """Return the frames of a stack that is not a single multi-band dataset, refusing anything else by name.

    Every other input is refused *here*, before an axes exists, because each of the shapes a caller gets
    wrong iterates into something: a path is a sequence of characters, a dict is a sequence of its keys, and
    both used to fail four frames later with ``AttributeError: 'str' object has no attribute 'read_array'``
    — naming neither the argument nor what it takes.

    Args:
        stack: What the caller passed as ``facet(stack=)``, already known not to be a ``Dataset``.

    Returns:
        One frame per panel.

    Raises:
        ValueError: for a path or URL (``facet`` opens nothing), a mapping (it iterates as its keys),
            anything that is not iterable at all, and a sequence holding something that is not a
            ``Dataset`` — named by its position in the stack.
    """
    if isinstance(stack, (str, bytes, os.PathLike)):
        raise ValueError(
            f"facet(stack=) takes {_STACK_FORMS}; got a path or URL. facet opens nothing — read it with "
            "Dataset.read_file() (or DatasetCollection) and pass what that returns"
        )
    if isinstance(stack, Mapping):
        raise ValueError(
            f"facet(stack=) takes {_STACK_FORMS}; got {type(stack).__name__}, which iterates as its keys "
            "rather than its frames. Pass its values, and title the panels with labels="
        )
    members = getattr(stack, "datasets", stack)
    try:
        frames = list(members)
    except TypeError as error:
        raise ValueError(
            f"facet(stack=) takes {_STACK_FORMS}; got {type(stack).__name__}, which holds no frames"
        ) from error
    for position, frame in enumerate(frames):
        if not isinstance(frame, Dataset):
            raise ValueError(
                f"facet(stack=) takes {_STACK_FORMS}; frame {position} is {type(frame).__name__}. Every "
                "frame is read and warped as a dataset, so a bare array (which carries no CRS) cannot be "
                "a panel of a facet — draw one with Map.field"
            )
    return frames


def _panels_of(stack: Any, band: int) -> Tuple[List[Any], List[int], str]:
    """Return what each panel of a facet draws: its frame, the band it reads, and the facet's default name.

    Args:
        stack: A multi-band ``Dataset`` (one panel per band), a ``DatasetCollection`` (one per member) or any
            sequence of ``Dataset`` frames.
        band: The band every frame of a collection or sequence is read at.

    Returns:
        ``(frames, bands, name)``: one frame and one band per panel, and ``"band"`` or ``"frame"`` — what the
        panels are numbered by when the caller names no ``col``.

    Raises:
        ValueError: from :func:`_frames_of`, for a ``stack`` that is none of the three shapes it takes.
    """
    if isinstance(stack, Dataset):
        count = int(stack.band_count)
        return [stack] * count, list(range(1, count + 1)), "band"
    frames = _frames_of(stack)
    return frames, [band] * len(frames), "frame"


@dataclass(frozen=True, eq=False)
class _PanelPlan:
    """What a facet's panels are, how they are laid out, and what each one is titled.

    Everything a caller says about the *panels* — ``stack``, ``band``, ``col``, ``col_wrap``, ``labels`` —
    is answered here, in one place, before any axes exist: the frames and the band each reads, the grid
    shape they are laid out on, and one title per panel. These values are born together, travel together
    into :func:`grid` and the draw loop, and die when the panels are drawn, which is what makes them one
    object rather than seven locals threaded through :func:`facet`. The colour scale is the other half of
    a facet and is resolved separately, by :func:`_shared_style`.

    Attributes:
        frames: One frame per panel, in panel order. A multi-band dataset is the *same* dataset repeated,
            which is why :meth:`panels` hands the band over alongside it.
        bands: The band each frame is read at, index-aligned with :attr:`frames`.
        labels: One label per panel, index-aligned with :attr:`frames` — the caller's own, or the band
            numbers (1-based) for a multi-band dataset and the positions (0-based) otherwise.
        name: What the panels are, as each title names it: the caller's ``col``, else ``"band"`` or
            ``"frame"``.
        nrows: Rows of panels.
        ncols: Panels per row.

    Examples:
        - Four frames wrapped onto rows of three, titled by position:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static.figure import _PanelPlan
            >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
            >>> frames = [
            ...     Dataset.from_array(arr=np.full((2, 2), float(i)), geo_ref=geo, no_data_value=-9999.0)
            ...     for i in range(4)
            ... ]
            >>> plan = _PanelPlan.resolve(frames, 1, col=None, col_wrap=3, labels=None)
            >>> plan.count, (plan.nrows, plan.ncols)
            (4, (2, 3))
            >>> plan.titles()
            ['frame = 0', 'frame = 1', 'frame = 2', 'frame = 3']

            ```
        - A label per panel and a name for what they are:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static.figure import _PanelPlan
            >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
            >>> frames = [
            ...     Dataset.from_array(arr=np.full((2, 2), float(i)), geo_ref=geo, no_data_value=-9999.0)
            ...     for i in range(2)
            ... ]
            >>> plan = _PanelPlan.resolve(
            ...     frames, 1, col="month", col_wrap=None, labels=["Jan", "Feb"]
            ... )
            >>> plan.titles()
            ['month = Jan', 'month = Feb']

            ```
    """

    frames: List[Any]
    bands: List[int]
    labels: List[Any]
    name: str
    nrows: int
    ncols: int

    @property
    def count(self) -> int:
        """How many panels the facet draws.

        Returns:
            The number of frames, which is the number of panels.
        """
        return len(self.frames)

    @classmethod
    def resolve(
        cls,
        stack: Any,
        band: int,
        *,
        col: Optional[str],
        col_wrap: Optional[int],
        labels: Optional[Sequence[Any]],
    ) -> "_PanelPlan":
        """Read a caller's panel arguments into one plan, refusing what cannot be laid out.

        Args:
            stack: What the panels are, as :func:`_panels_of` takes it.
            band: The band each frame of a collection or sequence is read at.
            col: What the panels are, as each title names it, or ``None`` for ``"band"``/``"frame"``.
            col_wrap: Panels per row, or ``None`` to put them all on one row.
            labels: One label per panel, or ``None`` to number them. Any iterable, a generator included —
                spent into a list before the panels are counted, so a generator is counted rather than
                crashing ``len()``.

        Returns:
            The plan, with the grid shape and one label per panel already resolved.

        Raises:
            ValueError: for a ``stack`` that is none of the shapes :func:`_frames_of` takes, or that has no
                frames; for a ``col_wrap`` below 1; and for a ``labels`` that does not number the panels.
                All three are raised before :func:`facet` makes a figure, so a refused call leaves none
                open.

        Examples:
            - A multi-band dataset is one frame read at each band, and the bands are what the panels are
              named by — numbered from 1, as a band is:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static.figure import _PanelPlan
                >>> cube = Dataset.from_array(
                ...     arr=np.arange(12.0).reshape(3, 2, 2),
                ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
                ...     no_data_value=-9999.0,
                ... )
                >>> plan = _PanelPlan.resolve(cube, 1, col=None, col_wrap=None, labels=None)
                >>> plan.name, plan.bands
                ('band', [1, 2, 3])
                >>> plan.titles()
                ['band = 1', 'band = 2', 'band = 3']

                ```
            - A generator of labels is spent before the panels are counted, so a wrong number of them
              gets the refusal that names both counts rather than a ``len()`` failure:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static.figure import _PanelPlan
                >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
                >>> frames = [
                ...     Dataset.from_array(arr=np.full((2, 2), 1.0), geo_ref=geo, no_data_value=-9999.0)
                ...     for _ in range(3)
                ... ]
                >>> _PanelPlan.resolve(
                ...     frames, 1, col=None, col_wrap=None, labels=(name for name in ("a", "b"))
                ... )
                Traceback (most recent call last):
                    ...
                ValueError: facet() draws 3 panels, so labels= needs 3 labels; got 2

                ```
            - An empty stack and a ``col_wrap`` that cannot be panels per row are each refused by name:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static.figure import _PanelPlan
                >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
                >>> one = Dataset.from_array(arr=np.full((2, 2), 1.0), geo_ref=geo, no_data_value=-9999.0)
                >>> for stack, wrap in (([], None), ([one], 0)):
                ...     try:
                ...         _PanelPlan.resolve(stack, 1, col=None, col_wrap=wrap, labels=None)
                ...     except ValueError as error:
                ...         print(error)
                facet() was given no frames to draw
                facet(col_wrap=) is panels per row and must be at least 1; got 0

                ```
        """
        frames, bands, default_col = _panels_of(stack, band)
        count = len(frames)
        if count == 0:
            raise ValueError("facet() was given no frames to draw")
        if col_wrap is not None and col_wrap < 1:
            raise ValueError(
                f"facet(col_wrap=) is panels per row and must be at least 1; got {col_wrap}"
            )
        spent = None if labels is None else list(labels)
        if spent is not None and len(spent) != count:
            raise ValueError(
                f"facet() draws {count} panels, so labels= needs {count} labels; got {len(spent)}"
            )
        if spent is None:
            spent = bands if default_col == "band" else list(range(count))
        ncols = count if col_wrap is None else min(col_wrap, count)
        return cls(
            frames=frames,
            bands=bands,
            labels=spent,
            name=default_col if col is None else col,
            nrows=math.ceil(count / ncols),
            ncols=ncols,
        )

    def titles(self) -> List[str]:
        """Return the title of each panel, in panel order.

        Returns:
            One ``"<name> = <label>"`` string per panel.
        """
        return [f"{self.name} = {label}" for label in self.labels]

    def panels(self, maps: Sequence[Any]) -> Iterator[Tuple[Any, Any, int, str]]:
        """Pair each panel's ``Map`` with what it draws.

        Args:
            maps: The panels, from :func:`grid`, in panel order — as many as :attr:`count`.

        Returns:
            An iterator of ``(panel, frame, band, title)``, one per panel.

        Examples:
            - The draw loop's four values arrive together, so the panel, what it reads and what it is
              called never drift apart:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static.figure import _PanelPlan
                >>> cube = Dataset.from_array(
                ...     arr=np.arange(8.0).reshape(2, 2, 2),
                ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
                ...     no_data_value=-9999.0,
                ... )
                >>> plan = _PanelPlan.resolve(cube, 1, col="month", col_wrap=None, labels=["Jan", "Feb"])
                >>> [(panel, band, title) for panel, _frame, band, title in plan.panels(["left", "right"])]
                [('left', 1, 'month = Jan'), ('right', 2, 'month = Feb')]

                ```
        """
        return zip(maps, self.frames, self.bands, self.titles())


def _stack_values(
    panel: Map, frames: Sequence[Any], bands: Sequence[int]
) -> List[np.ndarray]:
    """Read every panel's values as they will be drawn: warped into the display CRS, nodata as ``NaN``.

    **Each frame is warped once, not once per panel.** A multi-band dataset is one panel per band, and
    :func:`_panels_of` hands the *same* dataset over once per band, so warping per panel warped the whole
    cube ``n`` times to read one band out of each — quadratic in the band count, and the cost of a warp is
    the whole cube, not the band. The warps are therefore kept by frame identity, which is what tells the
    repeated cube apart from a sequence of separate frames (each of those is still read once, as it was).
    A frame the display CRS cannot place is remembered as ``None`` so the refusal is not re-run either.

    Args:
        panel: A map in the display CRS the panels share, used for its warp.
        frames: One frame per panel.
        bands: The band each frame is read at.

    Returns:
        One array per frame that lands on the view; a frame the display CRS cannot place contributes none.
    """
    values = []
    # frame identity -> its warp, or None when the display CRS places none of it
    warps: Dict[int, Any] = {}
    for frame, frame_band in zip(frames, bands):
        key = id(frame)
        if key not in warps:
            try:
                warps[key] = panel._reproject(frame)
            except OffLimbError:
                warps[key] = None  # it draws nothing, so it bounds nothing
        if warps[key] is not None:
            values.append(read_masked_band(warps[key], band=frame_band))
    return values


def _pooled_values(values: Sequence[np.ndarray]) -> np.ndarray:
    """Return every frame's finite cells as one flat array — the stack a classifier cuts its classes over.

    Args:
        values: Every panel's values, from :func:`_stack_values`.

    Returns:
        One ``float64`` array of the finite cells of every frame, empty when there are none.
    """
    finite = [np.asarray(a, dtype=float)[np.isfinite(a)] for a in values]
    return np.concatenate(finite) if finite else np.array([], dtype=float)


def _pooled_codes(pooled: np.ndarray, cmap: Any) -> Dict[str, Any]:
    """Cut one categorical scale over the whole stack: every code the frames hold, each with its own colour.

    ``scheme="categorical"`` classifies a band by the codes *that band* holds, so forwarding the word to each
    panel gives each one a different class list — two frames holding ``{1, 2}`` and ``{3, 4}`` both paint
    their lowest code in the palette's first colour, and the two classes read as one. The codes are therefore
    pooled here, the way a graduated scheme's edges are, and handed to every panel as explicit class edges
    plus the matching palette — the spelling a shared scale has to travel in, since the word ``"categorical"``
    carries no class list with it.

    Explicit edges are a *graduated* scheme, so each panel's layer records a graduated scale and its key reads
    the class ranges (``0.5 – 1.5``) rather than the codes (``1``). The drawn colours are right either way —
    one code is one colour on every panel, and a replay of a panel paints the same hexes — so what is wrong is
    the recorded scale's identity and the legend text it produces; #377 asks for a way to state the categories
    outright and tracks the fix.

    Args:
        pooled: Every finite value of every frame that lands on the view, concatenated.
        cmap: The caller's ``cmap``, or ``None`` — resolved through
            :func:`~digitalearth.base.symbology.resolve_categorical_cmap`, so a facet takes the same
            qualitative default a single categorical ``Map`` takes.

    Returns:
        The ``scheme``/``cmap`` keywords every panel is drawn with: one class per pooled code, cut by
        :func:`~digitalearth.base.raster_classes.code_edges` — half a step below the first code, midway
        between each pair of neighbours, half a step above the last, so a gap in the codes widens a class
        rather than letting it swallow a code — and one colour per class in code order.

    Raises:
        ValueError: from :func:`~digitalearth.base.raster_classes.raster_categories` when the pooled values
            are not class codes — no finite cell at all, a non-integer value, or more distinct codes than a
            key can show (24).

    Examples:
        - Three codes, one class and one colour each; the gap between 2 and 5 widens the middle class:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.figure import _pooled_codes
            >>> shared = _pooled_codes(np.array([1.0, 2.0, 2.0, 5.0]), None)
            >>> shared["scheme"]
            [0.5, 1.5, 3.5, 5.5]
            >>> shared["cmap"]
            ['#1f77b4', '#ff7f0e', '#2ca02c']

            ```
        - The codes of two frames pool into one scale, so the second frame's codes carry on the palette
          instead of restarting it:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.figure import _pooled_codes
            >>> low, high = np.array([1.0, 2.0]), np.array([3.0, 4.0])
            >>> shared = _pooled_codes(np.concatenate([low, high]), None)
            >>> len(shared["cmap"]), shared["cmap"][2:]
            (4, ['#2ca02c', '#d62728'])

            ```
        - Values that are not class codes are refused, naming what to use instead:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.figure import _pooled_codes
            >>> try:
            ...     _pooled_codes(np.array([1.5, 2.5]), None)
            ... except ValueError as error:
            ...     print(str(error)[:59])
            scheme='categorical' needs integer class codes (land cover,

            ```
    """
    codes = raster_categories(pooled)
    _, colors = categorical_colors(codes, resolve_categorical_cmap(cmap))
    return {"scheme": code_edges(codes), "cmap": colors}


#: The classes a named ``facet(scheme=)`` cuts when the caller names no count — the classifier's own default.
_DEFAULT_CLASSES = 5


def _checked_classes(style: Dict[str, Any]) -> Dict[str, Any]:
    """Return the caller's styling with ``k`` given one meaning: the classes a named scheme cuts.

    ``k`` is forwarded to a render that ignores it, so every spelling it cannot apply to has to be refused
    here or it is a silent no-op — the defect this package has now fixed several times. There are four:
    no ``scheme`` at all, an explicit list of edges (which *is* the classification), ``"categorical"``
    (where a code is its own class), and a value that is not a count. An unset ``k`` is dropped rather than
    refused, so a caller forwarding its own optional arguments can pass ``k=None`` for "the default" — the
    reading ``vmin=None``/``vmax=None`` already get a few lines below, and the one spelling the first
    version of this guard let through to ``int(None)``.

    Args:
        style: The caller's styling keywords. Not mutated.

    Returns:
        A copy with ``k`` absent when it was unset, and a plain ``int`` otherwise.

    Raises:
        ValueError: naming ``facet(k=)`` and why the value cannot count classes here.

    Examples:
        - Beside a named scheme ``k`` is kept as the plain count it is; unset or ``None`` it is dropped, so
          :func:`_shared_style` falls back to :data:`_DEFAULT_CLASSES`:
            ```python
            >>> from digitalearth.static.figure import _checked_classes
            >>> _checked_classes({"scheme": "quantiles", "k": 3})
            {'scheme': 'quantiles', 'k': 3}
            >>> _checked_classes({"scheme": "quantiles", "k": None})
            {'scheme': 'quantiles'}
            >>> _checked_classes({"cmap": "viridis"})
            {'cmap': 'viridis'}

            ```
        - Each spelling ``k`` could not apply to is refused, naming what the value would have counted:
            ```python
            >>> from digitalearth.static.figure import _checked_classes
            >>> for style in (
            ...     {"k": 4},
            ...     {"k": 4, "scheme": [0.0, 1.0, 2.0]},
            ...     {"k": 4, "scheme": "categorical"},
            ... ):
            ...     try:
            ...         _checked_classes(style)
            ...     except ValueError as error:
            ...         print(str(error).split("; ")[0])
            facet(k=4) counts the classes a scheme cuts, so it classifies nothing without scheme=
            facet(k=4) counts the classes a scheme cuts, so it classifies nothing beside scheme=[0.0, 1.0, 2.0], which gives the class edges outright
            facet(k=4) counts the classes a scheme cuts, so it classifies nothing under scheme='categorical', where a class code is its own class

            ```
        - A value that is not a count is refused rather than truncated or coerced, ``True`` included:
            ```python
            >>> from digitalearth.static.figure import _checked_classes
            >>> for value in (2.7, "4", True, 0):
            ...     try:
            ...         _checked_classes({"scheme": "quantiles", "k": value})
            ...     except ValueError as error:
            ...         print(error)
            facet() needs k= as a whole number >= 1; got 2.7
            facet() needs k= as a whole number >= 1; got '4'
            facet() needs k= as a whole number >= 1; got True
            facet() needs k= as a whole number >= 1; got 0

            ```
    """
    checked = dict(style)
    k = checked.pop("k", None)
    if k is None:
        return checked
    scheme = checked.get("scheme")
    counts = f"facet(k={k!r}) counts the classes a scheme cuts, so it"
    if scheme is None:
        raise ValueError(
            f"{counts} classifies nothing without scheme=; pass scheme= ('quantiles', "
            "'equal_interval', 'categorical', ...) or drop k="
        )
    if not isinstance(scheme, str):
        raise ValueError(
            f"{counts} classifies nothing beside scheme={scheme!r}, which gives the class edges outright; "
            "drop k=, or name a scheme ('quantiles', 'equal_interval', ...) for k= to cut"
        )
    if asks_categorical(scheme):
        raise ValueError(
            f"{counts} classifies nothing under scheme={scheme!r}, where a class code is its own class; "
            "drop k=, or name a scheme that cuts classes ('quantiles', 'equal_interval', ...)"
        )
    if not isinstance(k, Integral) or isinstance(k, bool) or int(k) < 1:
        # `bool` is an `Integral`, but `k=True` is a mistake rather than a request for one class.
        raise ValueError(f"facet() needs k= as a whole number >= 1; got {k!r}")
    checked["k"] = int(k)
    return checked


def _shared_style(
    values: Sequence[np.ndarray],
    style: Dict[str, Any],
    kind: str,
    measured: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    """Resolve the colour scale every panel shares, once, over the whole stack.

    Args:
        values: Every panel's values, from :func:`_stack_values`.
        style: The caller's styling keywords. Not mutated.
        kind: The render, one of :data:`_FACET_KINDS`.
        measured: What the stack measures — :func:`~digitalearth.base.clim.measure_clim` over `values`, or
            ``None`` when no frame holds a finite value. Passed in rather than measured here because the
            caller reads the same answer to decide whether a colorbar can be keyed, and reducing every
            frame twice is the one thing this function exists to avoid.

    Returns:
        The keywords each panel is drawn with: a named ``scheme`` replaced by the class edges it cuts over
        the whole stack (``"categorical"`` by the pooled codes, see :func:`_pooled_codes`), and
        ``vmin``/``vmax`` filled from the stack's range wherever the caller left them unset — so no panel
        picks a scale from its own frame.

    Raises:
        ValueError: for a contour render without explicit ``levels``, which each panel would otherwise pick
            from its own frame; or from the classifier, when a named ``scheme`` cannot cut the pooled values.
    """
    shared = dict(style)
    if kind in ("contourf", "contour") and shared.get("levels") is None:
        raise ValueError(
            f"facet(kind={kind!r}) needs explicit levels=: without them every panel picks its own from its "
            "own frame, so the panels stop sharing one scale"
        )
    scheme = shared.get("scheme")
    # The classifiers below are the only readers of the pooled values, and they are the only reason to
    # compact every frame's finite cells into one array. An unclassified facet is scaled by `measure_clim`,
    # which reduces each frame where it lies, so building the pool for it was a whole second pass thrown
    # away: 0.14 s of it on twelve 1000 x 1000 frames.
    if isinstance(scheme, str):
        pooled = _pooled_values(values)
        if asks_categorical(scheme):
            # A code is its own class, so there is no class count to cut — and `k` beside a categorical
            # scheme never reaches here, `_checked_classes` having refused it by name.
            shared.update(_pooled_codes(pooled, shared.get("cmap")))
        else:
            classes = Scale.from_values(
                pooled, scheme=scheme, k=shared.pop("k", _DEFAULT_CLASSES)
            )
            shared["scheme"] = [float(edge) for edge in classes.breaks]
    low, high = frozen_scale(measured).as_limits()
    shared.setdefault("vmin", low)
    shared.setdefault("vmax", high)
    if shared["vmin"] is None:
        shared["vmin"] = low
    if shared["vmax"] is None:
        shared["vmax"] = high
    return shared


def facet(
    stack: Any,
    *,
    col: Optional[str] = None,
    col_wrap: Optional[int] = None,
    labels: Optional[Sequence[Any]] = None,
    kind: str = "field",
    band: int = DEFAULT_BAND,
    crs: Any = 3857,
    globe: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    colorbar: bool = True,
    cbar_label: Optional[str] = None,
    **style: Any,
) -> Tuple[Figure, List[Map]]:
    """Draw a raster stack as small multiples — one panel per frame — on one shared colour scale.

    Small multiples are read against each other, so the scale is resolved **once over the whole stack**
    rather than per panel: the frames are warped into the display CRS and measured together, ``vmin``/``vmax``
    become the stack's range, and a named ``scheme`` is cut into one set of class edges over every frame —
    ``scheme="categorical"`` included, whose class codes are pooled over the stack so one code is one colour
    on every panel. Each panel is an ordinary :class:`Map` from :func:`grid`, so its frame is a described
    layer and the panel takes a basemap, coastlines or a graticule like any other map. One colorbar spans the
    panels.

    Args:
        stack: A multi-band ``Dataset`` (one panel per band), a ``DatasetCollection`` (one per member), or
            a sequence of ``Dataset`` frames. Anything else is refused by name — ``facet`` opens nothing, so
            a path or URL is read with ``Dataset.read_file()`` first.
        col: What the panels are, as each title names it — ``"time"`` titles them ``"time = 0"``,
            ``"time = 1"``, … ``None`` (default) says ``"band"`` for a multi-band dataset and ``"frame"``
            otherwise.
        col_wrap: Panels per row; the rest wrap onto further rows, and the unused slots of the last row are
            hidden. ``None`` (default) puts every panel on one row.
        labels: One label per panel for its title, in place of the index (``1``-based band numbers for a
            multi-band dataset, ``0``-based positions otherwise). Any iterable, a generator included — it is
            spent into a list before the panels are counted.
        kind: The render — ``"field"`` (default), ``"pcolormesh"``, ``"contourf"`` or ``"contour"``. The two
            contour renders need explicit ``levels=``.
        band: The band each frame of a collection or sequence is drawn from. Ignored for a multi-band
            dataset, whose bands are the panels.
        crs: Display CRS of every panel.
        globe: Draw every panel on a globe frame.
        figsize: Figure size in inches; ``None`` uses the matplotlib default.
        colorbar: Draw one colorbar spanning the panels (default ``True``). It is left off, with a logged
            warning, when no frame holds a finite value: the shared scale is then the ``(0, 1)`` fallback a
            colour domain takes for an unmeasurable stack, and a bar labelled 0-1 over a figure where no
            cell holds a value reads as data that is not there.
        cbar_label: The colorbar's label.
        **style: Styling for every panel, forwarded to the render (``cmap``, ``vmin``, ``vmax``, ``scheme``,
            ``k``, ``levels``, …). A ``vmin``/``vmax`` given here is the shared scale instead of the stack's
            range. ``k`` counts the classes a **named** ``scheme`` cuts — five when it is left unset or
            passed as ``None`` — and is refused wherever it cannot do that rather than forwarded as a
            no-op: with no ``scheme``, beside an explicit list of class edges, under
            ``scheme="categorical"`` (a code is its own class), and when it is not a whole number of at
            least one (``k=2.7`` and ``k="4"`` were silently truncated and coerced).

    Returns:
        ``(fig, maps)`` — the figure and one ``Map`` per panel, in panel order.

    Raises:
        ValueError: when ``stack`` is none of the three shapes it takes (a path, a mapping, a frame that is
            not a ``Dataset``) or has no frames, ``col_wrap`` is below 1, ``labels`` does not number the
            panels, ``kind`` is not one of the four, a contour ``kind`` has no ``levels``, or ``k`` cannot
            count classes as given (see ``**style``). Also from the classifier, when a named ``scheme``
            cannot cut the pooled values: a stack with no spread (every cell the same), one with no finite
            cell at all, or — under ``scheme="categorical"`` — pooled values that are not class codes (a
            non-integer value, or more distinct codes than a key can show).

    Examples:
        - Four frames on one row, one scale, titled by month:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static import facet
            >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)
            >>> frames = [
            ...     Dataset.from_array(arr=np.full((4, 4), 10.0 * i), geo_ref=geo, no_data_value=-9999.0)
            ...     for i in range(4)
            ... ]
            >>> fig, maps = facet(frames, crs=4326, col="month", labels=["Jan", "Feb", "Mar", "Apr"])
            >>> [m.ax.get_title() for m in maps]
            ['month = Jan', 'month = Feb', 'month = Mar', 'month = Apr']
            >>> {m.layers[-1][1].get_clim() for m in maps}
            {(0.0, 30.0)}
            >>> maps[0].close()  # one figure holds every panel, so one close is the whole facet

            ```
        - Wrap onto rows of three; the last row's spare slots are hidden:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static import facet
            >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)
            >>> frames = [
            ...     Dataset.from_array(arr=np.full((4, 4), float(i)), geo_ref=geo, no_data_value=-9999.0)
            ...     for i in range(4)
            ... ]
            >>> fig, maps = facet(frames, crs=4326, col_wrap=3, colorbar=False)
            >>> maps[0].ax.get_subplotspec().get_geometry()[:2], sum(not ax.get_visible() for ax in fig.axes)
            ((2, 3), 2)
            >>> maps[0].close()

            ```
        - A categorical stack cuts its classes over every frame, so the second frame's codes carry on the
          palette rather than restarting it — code 1 is its first colour and code 3 its third:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from matplotlib.colors import to_hex
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static import facet
            >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
            >>> early = Dataset.from_array(
            ...     arr=np.array([[1.0, 1.0], [2.0, 2.0]]), geo_ref=geo, no_data_value=-9999.0
            ... )
            >>> late = Dataset.from_array(
            ...     arr=np.array([[3.0, 3.0], [4.0, 4.0]]), geo_ref=geo, no_data_value=-9999.0
            ... )
            >>> fig, maps = facet([early, late], crs=4326, scheme="categorical", colorbar=False)
            >>> art = [m.layers[-1][1] for m in maps]
            >>> [to_hex(art[0].cmap(art[0].norm(code))) for code in (1.0, 2.0)]
            ['#1f77b4', '#ff7f0e']
            >>> [to_hex(art[1].cmap(art[1].norm(code))) for code in (3.0, 4.0)]
            ['#2ca02c', '#d62728']
            >>> [m.layers[-1][1].get_clim() for m in maps]
            [(0.5, 4.5), (0.5, 4.5)]
            >>> maps[0].close()

            ```
        - A stack where every cell is nodata still draws its panels, but no colorbar: the figure holds the
          two panels and nothing else, where a measurable stack of the same shape holds a third axes for the
          bar. ``k`` without a ``scheme`` is refused outright, since it would classify nothing:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static import facet
            >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
            >>> blank = [
            ...     Dataset.from_array(
            ...         arr=np.full((2, 2), -9999.0), geo_ref=geo, no_data_value=-9999.0
            ...     )
            ...     for _ in range(2)
            ... ]
            >>> fig, maps = facet(blank, crs=4326)
            >>> len(maps), len(fig.axes)
            (2, 2)
            >>> maps[0].layers[-1][1].get_clim()
            (0.0, 1.0)
            >>> try:
            ...     facet(blank, crs=4326, k=4)
            ... except ValueError as error:
            ...     print(str(error)[:69])
            facet(k=4) counts the classes a scheme cuts, so it classifies nothing
            >>> maps[0].close()

            ```

    See Also:
        grid: the panels alone, for a layout that draws something different on each.
        shared_colorbar: the spanning bar on its own.
    """
    if kind not in _FACET_KINDS:
        raise ValueError(
            f"facet(kind={kind!r}) is not a render; use one of {sorted(_FACET_KINDS)}"
        )
    style = _checked_classes(style)
    # Every panel argument — the frames, the grid shape, the titles — is answered here, before any axes
    # exist, so a refused call leaves no figure open.
    plan = _PanelPlan.resolve(stack, band, col=col, col_wrap=col_wrap, labels=labels)
    fig, slots = grid(plan.nrows, plan.ncols, crs=crs, globe=globe, figsize=figsize)
    maps, spare = slots[: plan.count], slots[plan.count :]
    for empty in spare:
        empty.ax.set_visible(False)
    values = _stack_values(maps[0], plan.frames, plan.bands)
    # The one reduction of the stack: it fills the shared vmin/vmax below and decides the colorbar.
    measured = measure_clim(values)
    shared = _shared_style(values, style, kind, measured)
    method, fixed = _FACET_KINDS[kind]
    drawn = None
    for panel, frame, frame_band, title in plan.panels(maps):
        getattr(panel, method)(frame, band=frame_band, **fixed, **shared)
        # The builders hand back the panel rather than the artist since ST-20, so the mappable the shared
        # bar is keyed to is asked for by name — through `Map.artist`, the public accessor that replaced
        # the read of the renderer's private record here (L9). `layer_ids` lists only layers that *were*
        # drawn — a skip is forgotten again — so an off-limb panel contributes nothing (there is no id to
        # pass, and `artist()` would refuse the id of a layer that drew nothing) and the first panel that
        # drew something still wins, exactly as the artist-or-`None` pick did.
        artist = panel.artist(panel.layer_ids[-1]) if panel.layer_ids else None
        if drawn is None and artist is not None:
            drawn = artist
        panel.set_title(title)
    if colorbar and drawn is not None and measured is None:
        # The shared scale fell back to the (0, 1) a `Scale` uses for an unmeasurable domain, so a bar here
        # would label the figure 0-1 while no cell on it holds a value at all. The panels are legitimate —
        # an empty frame is a real datum — and `facet` has no `strict` to refuse under, so only the bar goes.
        logger.warning(
            "facet(): no frame holds a finite value, so the stack has no colour scale to key; the "
            "colorbar is left off rather than drawn over the (0, 1) fallback limits"
        )
    elif colorbar:
        shared_colorbar(fig, drawn, maps, label=cbar_label)
    return fig, maps
