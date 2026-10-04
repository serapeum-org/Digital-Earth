"""figure — multi-panel layout: a grid of :class:`~digitalearth.static.map.Map` panels sharing one figure.

earthkit-plots models a figure as ``Figure → Subplot/Map → Layer``. Digital-Earth keeps the panel itself
(``Map``) as the unit and adds a thin :func:`grid` that creates the matplotlib figure + axes grid and binds a
``Map`` to each axes, plus :func:`shared_colorbar` for one colorbar spanning the panels, and :func:`facet`, which
lays a raster stack out as small multiples on one shared colour scale. This is orchestration only — the
rendering stays in each ``Map`` (pyramids + cleopatra).
"""

import logging
import math
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    **kwargs,
) -> Tuple[Figure, List[Map]]:
    """Create an ``nrows`` × ``ncols`` grid of :class:`Map` panels sharing one figure.

    Each cell of a ``matplotlib`` subplot grid is wrapped in a ``Map`` (all the same ``crs``/``globe``), so
    panels can be drawn on independently while sharing one figure for a single ``savefig`` / colorbar / title.

    Args:
        nrows: Number of panel rows.
        ncols: Number of panel columns.
        crs: Display CRS for every panel (passed to each ``Map``).
        globe: When True, every panel is a globe (``Map(globe=True)``).
        figsize: Figure size in inches; ``None`` uses the matplotlib default.
        **kwargs: Forwarded to each ``Map`` (e.g. ``domain``).

    Returns:
        ``(fig, maps)`` — the shared :class:`~matplotlib.figure.Figure` and the list of ``Map`` panels in
        row-major (left-to-right, top-to-bottom) order, length ``nrows * ncols``.

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

            ```
    """
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axs).ravel()
    maps = [Map(crs=crs, globe=globe, ax=ax, fig=fig, **kwargs) for ax in axes]
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


def _pooled_codes(pooled: np.ndarray, cmap: Any) -> Dict[str, Any]:
    """Cut one categorical scale over the whole stack: every code the frames hold, each with its own colour.

    ``scheme="categorical"`` classifies a band by the codes *that band* holds, so forwarding the word to each
    panel gives each one a different class list — two frames holding ``{1, 2}`` and ``{3, 4}`` both paint
    their lowest code in the palette's first colour, and the two classes read as one. The codes are therefore
    pooled here, the way a graduated scheme's edges are, and handed to every panel as explicit class edges
    plus the matching palette — the spelling a shared scale has to travel in, since the word ``"categorical"``
    carries no class list with it.

    Args:
        pooled: Every finite value of every frame that lands on the view, concatenated.
        cmap: The caller's ``cmap``, or ``None`` — resolved through
            :func:`~digitalearth.base.symbology.resolve_categorical_cmap`, so a facet takes the same
            qualitative default a single categorical ``Map`` takes.

    Returns:
        The ``scheme``/``cmap`` keywords every panel is drawn with: one class per pooled code, bounded half a
        step either side of it, and one colour per class in code order.

    Raises:
        ValueError: from :func:`~digitalearth.base.raster_classes.raster_categories` when the pooled values
            are not class codes — no finite cell at all, a non-integer value, or more distinct codes than a
            key can show.
    """
    codes = raster_categories(pooled)
    _, colors = categorical_colors(codes, resolve_categorical_cmap(cmap))
    return {"scheme": code_edges(codes), "cmap": colors}


def _shared_style(
    values: Sequence[np.ndarray], style: Dict[str, Any], kind: str
) -> Dict[str, Any]:
    """Resolve the colour scale every panel shares, once, over the whole stack.

    Args:
        values: Every panel's values, from :func:`_stack_values`.
        style: The caller's styling keywords. Not mutated.
        kind: The render, one of :data:`_FACET_KINDS`.

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
    finite = [np.asarray(a, dtype=float)[np.isfinite(a)] for a in values]
    pooled = np.concatenate(finite) if finite else np.array([], dtype=float)
    scheme = shared.get("scheme")
    if asks_categorical(scheme):
        # A code is its own class, so there is no class count to cut and `k` says nothing.
        shared.pop("k", None)
        shared.update(_pooled_codes(pooled, shared.get("cmap")))
    elif isinstance(scheme, str):
        classes = Scale.from_values(pooled, scheme=scheme, k=int(shared.pop("k", 5)))
        shared["scheme"] = [float(edge) for edge in classes.breaks]
    low, high = frozen_scale(measure_clim(values)).as_limits()
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
            range, and ``k`` counts the classes ``scheme`` cuts, so it is refused without one rather than
            forwarded as a no-op.

    Returns:
        ``(fig, maps)`` — the figure and one ``Map`` per panel, in panel order.

    Raises:
        ValueError: when ``stack`` is none of the three shapes it takes (a path, a mapping, a frame that is
            not a ``Dataset``) or has no frames, ``col_wrap`` is below 1, ``labels`` does not number the
            panels, ``kind`` is not one of the four, a contour ``kind`` has no ``levels``, or ``k`` is given
            without ``scheme``.

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

            ```

    See Also:
        grid: the panels alone, for a layout that draws something different on each.
        shared_colorbar: the spanning bar on its own.
    """
    if kind not in _FACET_KINDS:
        raise ValueError(
            f"facet(kind={kind!r}) is not a render; use one of {sorted(_FACET_KINDS)}"
        )
    if style.get("k") is not None and style.get("scheme") is None:
        raise ValueError(
            f"facet(k={style['k']!r}) counts the classes a scheme cuts, so it classifies nothing without "
            "scheme=; pass scheme= ('quantiles', 'equal_interval', 'categorical', ...) or drop k="
        )
    frames, bands, default_col = _panels_of(stack, band)
    count = len(frames)
    if count == 0:
        raise ValueError("facet() was given no frames to draw")
    if col_wrap is not None and col_wrap < 1:
        raise ValueError(
            f"facet(col_wrap=) is panels per row and must be at least 1; got {col_wrap}"
        )
    # Spent into a list before it is counted, so a generator of labels is counted rather than crashing
    # `len()` — the refusal below is the one a wrong number of labels is meant to get.
    labels = None if labels is None else list(labels)
    if labels is not None and len(labels) != count:
        raise ValueError(
            f"facet() draws {count} panels, so labels= needs {count} labels; got {len(labels)}"
        )
    ncols = count if col_wrap is None else min(col_wrap, count)
    nrows = math.ceil(count / ncols)
    fig, slots = grid(nrows, ncols, crs=crs, globe=globe, figsize=figsize)
    maps, spare = slots[:count], slots[count:]
    for empty in spare:
        empty.ax.set_visible(False)
    values = _stack_values(maps[0], frames, bands)
    shared = _shared_style(values, style, kind)
    method, fixed = _FACET_KINDS[kind]
    name = default_col if col is None else col
    if labels is None:
        labels = bands if default_col == "band" else list(range(count))
    drawn = None
    for panel, frame, frame_band, label in zip(maps, frames, bands, labels):
        artist = getattr(panel, method)(frame, band=frame_band, **fixed, **shared)
        drawn = artist if drawn is None else drawn
        panel.set_title(f"{name} = {label}")
    if colorbar and drawn is not None and measure_clim(values) is None:
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
