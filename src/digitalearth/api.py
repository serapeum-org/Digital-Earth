"""quickplot — one-call entry points that build a decorated :class:`~digitalearth.scene.map.Map`.

``quickmap`` (and its alias ``quickplot``) dispatch on the input type, auto-style the data, draw it on a
``Map``, optionally decorate (basemap/coastlines/domain), and add a colorbar — returning the finished
``Map`` so callers can further tweak or ``save`` it. Module-level functions (``contourf``, ``imshow``,
``scatter``, ``choropleth``, …) mirror the ``Map`` methods for a terse functional API.
"""

from typing import Any, Optional

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

from digitalearth._types import PlottableData
from digitalearth.scene import Map

__all__ = [
    "quickmap",
    "quickplot",
    "imshow",
    "contourf",
    "contour",
    "pcolormesh",
    "scatter",
    "grid_cells",
    "choropleth",
    "voronoi",
    "cartogram",
    "quadtree",
    "kde",
    "sankey",
]


def _draw(scene: Map, data: PlottableData, kind: str, **kwargs) -> None:
    """Draw ``data`` on ``scene`` using the renderer implied by its type and ``kind``.

    Dispatch is by input type: a ``FeatureCollection`` of polygons becomes a ``choropleth`` (when a
    ``column`` kwarg is given) or outline ``shapes``; any other geometry becomes a ``scatter``; a
    ``Dataset`` is rendered with the ``kind`` method (``imshow`` for ``"auto"``). An empty
    ``FeatureCollection`` is rejected up front — without the guard its all-``True`` empty ``geom_type``
    check would misclassify it as polygons and silently draw nothing.

    Args:
        scene: The :class:`Map` to draw on.
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (vector).
        kind: Raster renderer name (``"auto"`` → ``imshow``); ignored for vector input.
        **kwargs: Forwarded to the chosen ``Map`` draw method (e.g. ``column``, ``cmap``, ``levels``).

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection`` (nothing to draw).
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    if isinstance(data, FeatureCollection):
        if len(data) == 0:
            raise ValueError(
                "quickmap got an empty FeatureCollection (nothing to draw)"
            )
        if (data.geometry.geom_type.isin(["Polygon", "MultiPolygon"])).all():
            column = kwargs.pop("column", None)
            if column is not None:
                scene.choropleth(data, column=column, **kwargs)
            else:
                scene.shapes(data, **kwargs)
        else:
            scene.scatter(data, **kwargs)
        return
    if isinstance(data, Dataset):
        method = "imshow" if kind == "auto" else kind
        getattr(scene, method)(data, **kwargs)
        return
    raise TypeError(f"quickmap cannot draw a {type(data).__name__}")


def quickmap(
    data: PlottableData,
    *,
    crs: Any = 3857,
    kind: str = "auto",
    domain: Any = None,
    basemap: bool = False,
    coastlines: bool = False,
    colorbar: bool = True,
    backend: str = "matplotlib",
    **kwargs,
) -> Any:
    """Build a finished map from a single pyramids object in one call.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (points/polygons).
        crs: Display CRS for the map.
        kind: Renderer for raster input (``"auto"`` → ``imshow``; or ``contourf``/``contour``/``pcolormesh``).
        domain: Optional named region / bbox to set the extent.
        basemap: When True, add an XYZ-tile basemap (best-effort; ignored if tiles are unreachable).
        coastlines: When True, overlay coastlines (best-effort; ignored if unreachable).
        colorbar: When True, add a colorbar for the drawn layer (skipped if there is nothing mappable).
        backend: ``"matplotlib"`` (default) returns a static :class:`Map`; ``"interactive"`` returns a
            pan/zoom :class:`~digitalearth.interactive.map.InteractiveMap` (needs the ``interactive``
            extra) fed from the same ``Source``/auto-style pipeline.
        **kwargs: Forwarded to the underlying draw method (e.g. ``cmap``, ``levels``, ``column``).

    Returns:
        The decorated :class:`Map`, or an :class:`InteractiveMap` when ``backend="interactive"``.

    Raises:
        ValueError: for an unknown ``backend``.

    Examples:
        - One call turns a raster into a finished map with a colorbar:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.api import quickmap
            >>> ds = Dataset.read_file("examples/data/acc4000.tif")
            >>> m = quickmap(ds, crs=ds.epsg)
            >>> len(m.layers)
            1
            >>> len(m.fig.axes)  # main axes + colorbar
            2

            ```
        - Disable the colorbar to keep a single axes:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.api import quickmap
            >>> ds = Dataset.read_file("examples/data/acc4000.tif")
            >>> m = quickmap(ds, crs=ds.epsg, colorbar=False)
            >>> len(m.fig.axes)
            1

            ```
    """
    if backend == "interactive":
        return _quickmap_interactive(
            data, crs=crs, kind=kind, basemap=basemap, coastlines=coastlines, **kwargs
        )
    if backend != "matplotlib":
        raise ValueError(
            f"unknown backend {backend!r}; choose 'matplotlib' (static) or 'interactive' (HoloViz)"
        )
    scene = Map(crs=crs, domain=domain)
    _draw(scene, data, kind, **kwargs)
    if coastlines:
        try:
            scene.coastlines()
        except Exception:  # network/data unavailable — decoration is best-effort
            pass
    if basemap:
        try:
            scene.basemap()
        except Exception:  # tile servers unavailable — decoration is best-effort
            pass
    if domain is not None:
        scene.set_domain()
    if colorbar and scene.layers and scene.layers[-1][1] is not None:
        try:
            scene.colorbar()
        except Exception:  # outline-only / unmappable layer
            pass
    return scene


def quickplot(data: PlottableData, **kwargs) -> Any:
    """Alias of :func:`quickmap` — build a finished map from ``data`` in one call."""
    return quickmap(data, **kwargs)


def _quickmap_interactive(
    data: PlottableData,
    *,
    crs: Any,
    kind: str,
    basemap: bool,
    coastlines: bool,
    **kwargs,
) -> Any:
    """Build a finished ``InteractiveMap`` from ``data`` (the ``backend="interactive"`` path, DX.1).

    Dispatches by input type exactly like :func:`_draw`: a polygon ``FeatureCollection`` with a
    ``column`` becomes a ``choropleth`` (else ``polygons``); other vectors become ``points``; a raster
    is drawn with the ``kind`` method (``"auto"`` → ``image``). The ``InteractiveMap`` import is lazy so
    the core ``api`` works without the ``interactive`` extra.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (vector).
        crs: Display CRS for the interactive map.
        kind: Raster renderer (``"auto"`` → ``image``; or ``contourf``/``contour``/``pcolormesh``).
        basemap: When True, add a tile basemap.
        coastlines: When True, overlay a coastline.
        **kwargs: Forwarded to the chosen builder (e.g. ``cmap``, ``column``).

    Returns:
        The decorated :class:`~digitalearth.interactive.map.InteractiveMap`.

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection``.
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    from digitalearth.interactive import InteractiveMap

    _raster_kind = {
        "auto": "image",
        "imshow": "image",
        "contourf": "filled_contours",
        "contour": "contours",
        "pcolormesh": "quadmesh",
    }
    scene = InteractiveMap(crs=crs)
    if isinstance(data, FeatureCollection):
        if len(data) == 0:
            raise ValueError(
                "quickplot got an empty FeatureCollection (nothing to draw)"
            )
        if (data.geometry.geom_type.isin(["Polygon", "MultiPolygon"])).all():
            column = kwargs.pop("column", None)
            if column is not None:
                scene.choropleth(data, column=column, **kwargs)
            else:
                scene.polygons(data, **kwargs)
        else:
            scene.points(data, **kwargs)
    elif isinstance(data, Dataset):
        getattr(scene, _raster_kind.get(kind, "image"))(data, **kwargs)
    else:
        raise TypeError(f"quickplot cannot draw a {type(data).__name__}")
    if basemap:
        scene.tiles()
    if coastlines:
        scene.coastlines()
    return scene


def _finish(scene: Map, *, colorbar: bool) -> Map:
    """Add an aggregated colorbar to ``scene`` when requested and a mappable layer exists; return ``scene``.

    The shared tail of the one-call wrappers: a colorbar is added only when ``colorbar`` is true and a layer
    was drawn, and an outline-only / unmappable layer (which cannot carry a colorbar) is swallowed rather
    than raised.

    Args:
        scene: The :class:`Map` a wrapper has already drawn on.
        colorbar: Whether this plot kind should carry a colorbar (e.g. only when a value ``column`` was set).

    Returns:
        The same ``scene`` (so wrappers can ``return _finish(...)``).
    """
    if colorbar and scene.layers:
        try:
            scene.colorbar()
        except Exception:  # outline-only / unmappable layer
            pass
    return scene


def _method(name: str):
    """Build a module-level function that quick-draws via the ``Map`` method ``name``."""

    def _fn(data: PlottableData, **kwargs) -> Map:
        return quickmap(data, kind=name, **kwargs)

    _fn.__name__ = name
    _fn.__doc__ = (
        f"Quick-draw ``data`` with :meth:`Map.{name}` and return the finished Map."
    )
    return _fn


imshow = _method("imshow")
contourf = _method("contourf")
contour = _method("contour")
pcolormesh = _method("pcolormesh")


def scatter(data: PlottableData, **kwargs) -> Map:
    """Quick-draw a FeatureCollection of points as a scatter map; returns the finished Map."""
    return quickmap(data, **kwargs)


def grid_cells(data: PlottableData, **kwargs) -> Map:
    """Quick-draw raster cells as coloured polygons; returns the finished Map."""
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.grid_cells(data, **kwargs)
    return _finish(scene, colorbar=True)


def choropleth(data: PlottableData, column: str, **kwargs) -> Map:
    """Quick-draw a polygon FeatureCollection coloured by ``column``; returns the finished Map."""
    return quickmap(data, column=column, **kwargs)


def voronoi(data: PlottableData, column: Optional[str] = None, **kwargs) -> Map:
    """Quick-draw the Voronoi diagram of a point FeatureCollection; returns the finished Map.

    With ``column`` the cells are filled and coloured by that value (a colorbar is added); without it the cell
    outlines are drawn. ``clip`` and styling kwargs are forwarded to :meth:`Map.voronoi`.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.voronoi(data, column=column, **kwargs)
    return _finish(scene, colorbar=column is not None)


def cartogram(
    data: PlottableData, scale: str, column: Optional[str] = None, **kwargs
) -> Map:
    """Quick-draw a cartogram (polygons scaled by ``scale``); returns the finished Map.

    With ``column`` the scaled polygons are filled and coloured by that value (a colorbar is added); without it
    the outlines are drawn. ``limits`` and styling kwargs are forwarded to :meth:`Map.cartogram`.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.cartogram(data, scale=scale, column=column, **kwargs)
    return _finish(scene, colorbar=column is not None)


def quadtree(data: PlottableData, column: Optional[str] = None, **kwargs) -> Map:
    """Quick-draw a quadtree choropleth of a point FeatureCollection; returns the finished Map.

    Cells are coloured by an aggregate of ``column`` (or point count when ``None``) and a colorbar is added.
    ``agg``/``nmax``/``nmin``/``clip`` and styling kwargs are forwarded to :meth:`Map.quadtree`.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.quadtree(data, column=column, **kwargs)
    return _finish(scene, colorbar=True)


def kde(data: PlottableData, **kwargs) -> Map:
    """Quick-draw a 2-D kernel-density surface of a point FeatureCollection; returns the finished Map.

    ``clip`` and styling kwargs (``levels``/``shade``/``gridsize``/``cmap``/…) are forwarded to :meth:`Map.kde`.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.kde(data, **kwargs)
    return _finish(scene, colorbar=True)


def sankey(
    data: PlottableData,
    column: Optional[str] = None,
    scale: Optional[str] = None,
    **kwargs,
) -> Map:
    """Quick-draw a spatial flow / Sankey map of a line FeatureCollection; returns the finished Map.

    ``column`` colours each path and ``scale`` sets its width (both optional); styling kwargs are forwarded to
    :meth:`Map.sankey`. A colorbar is added when ``column`` is given.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.sankey(data, column=column, scale=scale, **kwargs)
    return _finish(scene, colorbar=column is not None)
