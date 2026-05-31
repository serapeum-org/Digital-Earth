"""quickplot — one-call entry points that build a decorated :class:`~digitalearth.scene.map.Map`.

``quickmap`` (and its alias ``quickplot``) dispatch on the input type, auto-style the data, draw it on a
``Map``, optionally decorate (basemap/coastlines/domain), and add a colorbar — returning the finished
``Map`` so callers can further tweak or ``save`` it. Module-level functions (``contourf``, ``imshow``,
``scatter``, ``choropleth``, …) mirror the ``Map`` methods for a terse functional API.
"""
from typing import Any, Optional

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

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
]


def _draw(scene: Map, data: Any, kind: str, **kwargs) -> None:
    """Draw ``data`` on ``scene`` using the renderer implied by its type and ``kind``."""
    if isinstance(data, FeatureCollection):
        if len(data) == 0:
            raise ValueError("quickmap got an empty FeatureCollection (nothing to draw)")
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
    data: Any,
    *,
    crs: Any = 3857,
    kind: str = "auto",
    domain: Any = None,
    basemap: bool = False,
    coastlines: bool = False,
    colorbar: bool = True,
    **kwargs,
) -> Map:
    """Build a finished :class:`Map` from a single pyramids object in one call.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (points/polygons).
        crs: Display CRS for the map.
        kind: Renderer for raster input (``"auto"`` → ``imshow``; or ``contourf``/``contour``/``pcolormesh``).
        domain: Optional named region / bbox to set the extent.
        basemap: When True, add an XYZ-tile basemap (best-effort; ignored if tiles are unreachable).
        coastlines: When True, overlay coastlines (best-effort; ignored if unreachable).
        colorbar: When True, add a colorbar for the drawn layer (skipped if there is nothing mappable).
        **kwargs: Forwarded to the underlying ``Map`` draw method (e.g. ``cmap``, ``levels``, ``column``).

    Returns:
        The decorated :class:`Map` (with ``.fig`` / ``.ax`` / ``.save``).

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


def quickplot(data: Any, **kwargs) -> Map:
    """Alias of :func:`quickmap` — build a finished map from ``data`` in one call."""
    return quickmap(data, **kwargs)


def _method(name: str):
    """Build a module-level function that quick-draws via the ``Map`` method ``name``."""

    def _fn(data: Any, **kwargs) -> Map:
        return quickmap(data, kind=name, **kwargs)

    _fn.__name__ = name
    _fn.__doc__ = f"Quick-draw ``data`` with :meth:`Map.{name}` and return the finished Map."
    return _fn


imshow = _method("imshow")
contourf = _method("contourf")
contour = _method("contour")
pcolormesh = _method("pcolormesh")


def scatter(data: Any, **kwargs) -> Map:
    """Quick-draw a FeatureCollection of points as a scatter map; returns the finished Map."""
    return quickmap(data, **kwargs)


def grid_cells(data: Any, **kwargs) -> Map:
    """Quick-draw raster cells as coloured polygons; returns the finished Map."""
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.grid_cells(data, **kwargs)
    if scene.layers:
        try:
            scene.colorbar()
        except Exception:
            pass
    return scene


def choropleth(data: Any, column: str, **kwargs) -> Map:
    """Quick-draw a polygon FeatureCollection coloured by ``column``; returns the finished Map."""
    return quickmap(data, column=column, **kwargs)
