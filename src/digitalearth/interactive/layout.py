"""Multi-panel layouts and linked views for the interactive tier (IN-7).

A single :class:`~digitalearth.interactive.map.InteractiveMap` composes its layers with ``*`` into one
overlay. Comparing two maps side by side, laying several out as small multiples, tabbing between them or
swiping one over another is a composition **across scenes**, so — like the static tier's ``grid`` /
``shared_colorbar`` and the package's ``quickmap`` — it lives in module-level functions rather than on the
map class (the house rule: a capability that acts on one map is a method; one that composes several is a
function).

Each function takes either an ``InteractiveMap`` (its ``render()`` is called) or an already-rendered
HoloViews/Panel object, so a caller can mix built maps with hand-made panels. Panels in the same display CRS
share their axes under Bokeh, so panning one pans the others — the linked-view half of the gap.
"""

from typing import Any

from digitalearth.interactive.base import _require_holoviz

__all__ = ["panels", "tabs", "swipe", "linked_views", "grid"]

#: The axis specifications `linked_views` accepts, mapped to the axes a `RangeToolLink` drives.
_LINK_AXES: dict[str, list[str]] = {
    "both": ["x", "y"],
    "x": ["x"],
    "y": ["y"],
}


def _panel() -> Any:
    """Import and return the ``panel`` module behind the HoloViz extra.

    Returns:
        The imported ``panel`` module.

    Raises:
        ImportError: when the ``interactive`` extra is not installed (via :func:`_require_holoviz`).
    """
    _require_holoviz()  # panel ships with the same extra; reuse its actionable message
    import panel as pn

    return pn


def _rendered(obj: Any) -> Any:
    """Return a drawable for ``obj``: a map's ``render()``, or ``obj`` itself.

    Args:
        obj: An :class:`~digitalearth.interactive.map.InteractiveMap`, or an already-rendered object.

    Returns:
        The HoloViews/Panel object to lay out.
    """
    render = getattr(obj, "render", None)
    return render() if callable(render) else obj


def panels(*maps: Any, cols: int | None = None) -> Any:
    """Lay several maps out as one multi-panel figure, axes linked where the CRS matches (IN-7).

    Composes the maps with HoloViews' ``+`` into a ``Layout`` — side by side, or wrapped into a grid when
    ``cols`` is given. Panels drawn in the same display CRS share their axes (Bokeh's default for matching
    dimensions), so panning or zooming one pans the others; that is the "linked views" half of the gap, and
    it needs no extra wiring.

    Args:
        *maps: Two or more ``InteractiveMap`` instances (or already-rendered objects) to lay out.
        cols: Wrap the row into a grid this many panels wide; ``None`` keeps a single row.

    Returns:
        A ``holoviews.Layout`` of the panels.

    Raises:
        ValueError: when fewer than two panels are given — one panel is not a layout.

    Examples:
        - Two maps become a two-panel layout (needs the engine):
            ```python
            >>> from digitalearth.interactive import InteractiveMap          # doctest: +SKIP
            >>> from digitalearth.interactive.layout import panels           # doctest: +SKIP
            >>> layout = panels(InteractiveMap().field(a), InteractiveMap().field(b))  # doctest: +SKIP
            >>> len(layout)                                                   # doctest: +SKIP
            2

            ```
    """
    _require_holoviz()
    if len(maps) < 2:
        raise ValueError(
            f"panels() lays out two or more maps; got {len(maps)}. One panel is a plain render()."
        )
    from functools import reduce
    from operator import add

    layout = reduce(add, (_rendered(m) for m in maps))
    return layout.cols(cols) if cols is not None else layout


def tabs(**named_maps: Any) -> Any:
    """Put each named map on its own tab of a Panel ``Tabs`` (IN-7).

    Args:
        **named_maps: ``title=map`` pairs — each value an ``InteractiveMap`` or a rendered object, each key
            the tab label.

    Returns:
        A ``panel.Tabs`` with one tab per map, in the order given.

    Raises:
        ValueError: when no maps are given.

    Examples:
        - Two named maps become a two-tab layout, labelled in the order given:
            ```python
            >>> from pyramids.dataset import Dataset                      # doctest: +SKIP
            >>> from digitalearth.interactive import InteractiveMap, tabs # doctest: +SKIP
            >>> dem = Dataset.read_file("examples/data/acc4000.tif")      # doctest: +SKIP
            >>> layout = tabs(                                           # doctest: +SKIP
            ...     rain=InteractiveMap().field(dem),
            ...     temp=InteractiveMap().field(dem),
            ... )
            >>> len(layout)                                              # doctest: +SKIP
            2

            ```
    """
    pn = _panel()
    if not named_maps:
        raise ValueError("tabs() needs at least one named map")
    return pn.Tabs(*[(title, _rendered(m)) for title, m in named_maps.items()])


def swipe(before: Any, after: Any) -> Any:
    """Swipe one map over another for a before/after comparison (IN-7).

    The geospatial before/after a change-detection or two-date comparison asks for: a draggable divider
    reveals ``after`` over ``before``.

    Args:
        before: The map shown on the left of the divider (an ``InteractiveMap`` or rendered object).
        after: The map shown on the right.

    Returns:
        A ``panel.layout.Swipe`` of the two.

    Examples:
        - Reveal an "after" raster over a "before" one with a draggable divider:
            ```python
            >>> from pyramids.dataset import Dataset                      # doctest: +SKIP
            >>> from digitalearth.interactive import InteractiveMap, swipe  # doctest: +SKIP
            >>> before = Dataset.read_file("examples/data/before.tif")    # doctest: +SKIP
            >>> after = Dataset.read_file("examples/data/after.tif")      # doctest: +SKIP
            >>> compare = swipe(                                         # doctest: +SKIP
            ...     InteractiveMap().field(before), InteractiveMap().field(after)
            ... )
            >>> type(compare).__name__                                  # doctest: +SKIP
            'Swipe'

            ```
    """
    pn = _panel()
    return pn.Swipe(_rendered(before), _rendered(after))


def linked_views(detail: Any, overview: Any, *, axes: str = "both") -> Any:
    """Link an overview map to a detail map with a declarative ``RangeToolLink`` (IN-7).

    The overview↔detail pattern the gap names: the overview carries a Bokeh range tool whose box
    **declaratively** drives the detail's axes, so navigating the small overview pans/zooms the big detail
    without a Python round-trip (it works in static HTML, unlike a stream-driven link). This is the explicit
    linked view `panels` only approximates through Bokeh's implicit shared axes.

    Args:
        detail: The detail (zoomed) map — an ``InteractiveMap`` or a rendered object; the link's target.
        overview: The overview (context) map; the link's source, which grows the range tool.
        axes: Which axes the range tool drives — ``"both"`` (default), ``"x"`` or ``"y"``.

    Returns:
        A ``holoviews.Layout`` of ``detail + overview`` with the ``RangeToolLink`` established between them.

    Raises:
        ValueError: for an ``axes`` other than ``"both"``/``"x"``/``"y"``.

    Examples:
        - An overview drives a detail on both axes (needs the engine):
            ```python
            >>> from digitalearth.interactive import InteractiveMap             # doctest: +SKIP
            >>> from digitalearth.interactive.layout import linked_views        # doctest: +SKIP
            >>> layout = linked_views(                                          # doctest: +SKIP
            ...     InteractiveMap().field(a), InteractiveMap().field(a)
            ... )
            >>> len(layout)                                                     # doctest: +SKIP
            2

            ```
    """
    _require_holoviz()
    if axes not in _LINK_AXES:
        raise ValueError(f"unknown axes {axes!r}; choose 'both'/'x'/'y'")
    from holoviews.plotting.links import RangeToolLink

    detail_obj = _rendered(detail)
    overview_obj = _rendered(overview)
    RangeToolLink(overview_obj, detail_obj, axes=_LINK_AXES[axes])
    return detail_obj + overview_obj


def grid(*maps: Any, cols: int = 2) -> Any:
    """Lay several maps out as a ``GridSpace`` of linked small multiples (IN-7).

    Unlike :func:`panels`' ``+`` ``Layout``, a ``GridSpace`` shares one set of axes across **every** cell
    declaratively, so panning any cell pans them all — the small-multiples linked view. Cells fill left to
    right, wrapping every ``cols`` maps; ``row=0`` is the **bottom** row, since a ``GridSpace`` places its y
    origin at the bottom (so the first row of maps renders along the bottom, not the top).

    Args:
        *maps: Two or more ``InteractiveMap`` instances (or already-rendered objects).
        cols: How many cells wide the grid is before it wraps to the next row.

    Returns:
        A ``holoviews.GridSpace`` keyed by ``(col, row)`` grid position.

    Raises:
        ValueError: when fewer than two maps are given, or ``cols`` is below one.

    Examples:
        - Four maps become a 2x2 linked grid (needs the engine):
            ```python
            >>> from digitalearth.interactive import InteractiveMap             # doctest: +SKIP
            >>> from digitalearth.interactive.layout import grid                # doctest: +SKIP
            >>> gs = grid(m1, m2, m3, m4, cols=2)                               # doctest: +SKIP
            >>> sorted(gs.keys())                                               # doctest: +SKIP
            [(0, 0), (0, 1), (1, 0), (1, 1)]

            ```
    """
    hv = _require_holoviz()[1]
    if len(maps) < 2:
        raise ValueError(
            f"grid() lays out two or more maps; got {len(maps)}. One panel is a plain render()."
        )
    if cols < 1:
        raise ValueError(f"cols must be at least 1; got {cols}")
    cells = {
        (index % cols, index // cols): _rendered(m) for index, m in enumerate(maps)
    }
    return hv.GridSpace(cells, kdims=["col", "row"])
