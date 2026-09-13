"""quickplot — one-call entry points that build a decorated :class:`~digitalearth.static.map.Map`.

``quickmap`` (and its alias ``quickplot``) dispatch on the input type, auto-style the data, draw it on a
``Map``, optionally decorate (basemap/coastlines/domain), and add a colorbar — returning the finished
``Map`` so callers can further tweak or ``save`` it. Module-level functions (``contourf``, ``imshow``,
``scatter``, ``choropleth``, …) mirror the ``Map`` methods for a terse functional API.

**Decoration is best-effort, not error-proof.** A basemap or coastline overlay needs assets this process may
not be able to reach, so those steps tolerate *unavailability* — ``OSError`` and its network subclasses
(``ConnectionError``, ``urllib``'s ``URLError``), and ``ImportError`` when the optional tiles/reference extra
is not installed — and log a warning naming the step, so a bare figure is never silent. Everything else
propagates: a ``ValueError``/``TypeError`` out of these steps is the caller being told what to fix (a missing
credential, an empty extent, an unknown preset), and swallowing it left ``quickmap`` returning an
undecorated figure with no explanation.

**A parameter the chosen backend cannot honour is an error, not a no-op.** The four backends are not the same
map: only the matplotlib one has an extent setter, only the 2-D ones have a display CRS, and the 3-D scene
projects nothing at all. ``quickmap`` used to accept every keyword for every backend and quietly drop the ones
that did not apply, so ``quickmap(ds, backend="web", domain="europe")`` returned a world map with no hint that
the domain had been ignored. Now each such parameter is checked against
:data:`BACKEND_CAPABILITIES` and refused by name — see :func:`_reject_unsupported`. Everything a backend *does*
support is forwarded to it.
"""

import logging
from typing import Any

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

from digitalearth.base.types import PlottableData
from digitalearth.static import Map

logger = logging.getLogger(__name__)

#: What "the decoration could not be fetched" actually looks like. ``ConnectionError`` and
#: ``urllib.error.URLError`` are both ``OSError`` subclasses, so cleopatra's tile/reference failures land
#: here; ``ImportError`` covers the optional extras those helpers import lazily.
UNAVAILABLE = (OSError, ImportError)

#: What "this layer cannot carry a colorbar" looks like: an outline-only artist has no ``cmap``
#: (``AttributeError``), a non-mappable one is refused by matplotlib (``TypeError``), and cleopatra raises
#: ``ValueError`` when it cannot find an axes to steal space from.
UNMAPPABLE = (AttributeError, TypeError, ValueError)


class _Unset:
    """Sentinel type for "the caller did not pass this keyword"."""

    def __repr__(self) -> str:
        """Return the placeholder spelling used in signatures and messages."""
        return "<unset>"


#: "Not passed." Needed because every one of the checked parameters has a meaningful default — ``crs=3857``
#: and ``colorbar=True`` in particular — so the default value cannot itself signal absence: without this,
#: ``quickmap(ds, backend="3d")`` would be refused for a ``crs`` the caller never asked for.
_UNSET = _Unset()

#: Which of ``quickmap``'s map-shaped parameters each backend can actually honour. Anything a caller passes
#: that is not listed for their backend is refused by name rather than dropped (:func:`_reject_unsupported`).
#:
#: * ``matplotlib`` is the only tier with an extent setter, so it is the only one that takes ``domain``.
#: * ``interactive`` pans and zooms, so it has no fixed extent; it does have a display CRS, coastlines and a
#:   colorbar toggle.
#: * ``3d`` has no display CRS at all (:attr:`digitalearth.three_d.base.Scene3DBase.display_crs` is ``None``
#:   and says so) and no coastline or extent concept; its scalar bar is the ``colorbar`` toggle.
#: * ``web`` places inline data in lon/lat and carries a ``crs`` of its own, which it validates. It has no
#:   coastline layer and no colorbar — its key is ``WebMap.legend``, a different thing with a different
#:   arity, and wiring ``colorbar=`` to it is deliberately left to the ``colorbar``/``legend`` rename
#:   (TODO(#254)) rather than guessed at here.
BACKEND_CAPABILITIES: dict[str, frozenset[str]] = {
    "matplotlib": frozenset(
        {"crs", "kind", "domain", "basemap", "coastlines", "colorbar"}
    ),
    "interactive": frozenset({"crs", "kind", "basemap", "coastlines", "colorbar"}),
    "3d": frozenset({"colorbar"}),
    "web": frozenset({"crs", "basemap"}),
}

#: The value of a checked parameter that asks for **nothing**, where one exists. Passing it to a backend that
#: cannot honour the parameter is not a dropped request — there was no request — so it is allowed through.
#:
#: ``colorbar=False`` is one of them (review L9). The rule here is about *dropped requests*, and what
#: ``False`` asks for is a map with no colorbar — which is exactly what the one tier that cannot honour the
#: parameter (``web``) draws anyway. Nothing is lost, so refusing it was pedantry of the same shape as
#: refusing ``coastlines=False`` there, and it broke the caller who passes one kwargs dict through to
#: whichever backend they picked. ``colorbar=True`` stays a real request and is still refused by name:
#: ``web`` keys itself with :meth:`~digitalearth.web.decoration.DecorationMixin.legend`, a different thing
#: with a different arity (TODO(#254)).
#:
#: ``crs`` is absent on purpose and stays absent: it has no value that asks for nothing — every CRS names a
#: projection to display in, and there is no "no CRS" to pass.
_INERT: dict[str, Any] = {
    "domain": None,
    "coastlines": False,
    "kind": "auto",
    "basemap": False,
    "colorbar": False,
}


def _asks_for_nothing(value: Any, inert: Any) -> bool:
    """Return whether ``value`` is the "asks for nothing" value recorded in :data:`_INERT`.

    The comparison is deliberately *not* ``==``. Two things go wrong with equality here. An array-valued
    ``domain`` (a bbox as an ``ndarray``) compared against ``None`` returns an array, and the ``if`` over it
    raises numpy's "truth value ... is ambiguous" instead of this module's own message (review L3). And
    ``0 == False`` is ``True`` in Python, so ``basemap=0`` / ``colorbar=0`` were read as the inert ``False``
    even though the caller wrote a number (review L4) — the check means *identity* with the inert value, and
    only the string ``kind="auto"`` needs value equality, because a string literal is not interned by
    identity across every source.

    Args:
        value: The value the caller passed for the parameter.
        inert: The parameter's entry in :data:`_INERT` — the value that asks for nothing.

    Returns:
        ``True`` when ``value`` is that inert value, so nothing was dropped by ignoring it.

    Examples:
        - ``False`` is inert, but the number zero is a value the caller typed:
            ```python
            >>> from digitalearth.api import _asks_for_nothing
            >>> _asks_for_nothing(False, False), _asks_for_nothing(0, False)
            (True, False)

            ```
        - An array never reaches an ambiguous comparison:
            ```python
            >>> import numpy as np
            >>> from digitalearth.api import _asks_for_nothing
            >>> _asks_for_nothing(np.array([0.0, 0.0, 1.0, 1.0]), None)
            False

            ```
    """
    if isinstance(inert, str):
        return isinstance(value, str) and value == inert
    return value is inert


def _reject_unsupported(backend: str, **passed: Any) -> None:
    """Refuse any keyword the chosen ``backend`` cannot honour, naming the parameter and the backend.

    Args:
        backend: The backend name ``quickmap`` was called with; already validated against
            :data:`BACKEND_CAPABILITIES`.
        **passed: The checked parameters as the caller left them — :data:`_UNSET` for the ones they never
            named.

    Raises:
        ValueError: for the first parameter that was actually requested and that ``backend`` cannot honour.
            The message names both, so the caller learns which half to change.

    Examples:
        - A domain means nothing to the web tier, and saying so beats returning a world map:
            ```python
            >>> from digitalearth.api import _reject_unsupported, _UNSET
            >>> try:
            ...     _reject_unsupported("web", domain="europe", crs=_UNSET)
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            domain= is not supported by backend='web'

            ```
        - A parameter left at a value that asks for nothing is not a dropped request, so it passes:
            ```python
            >>> from digitalearth.api import _reject_unsupported, _UNSET
            >>> _reject_unsupported("3d", domain=None, coastlines=False, crs=_UNSET) is None
            True

            ```
        - A bbox handed over as an array is refused by name too, not by numpy's ambiguity error:
            ```python
            >>> import numpy as np
            >>> from digitalearth.api import _reject_unsupported, _UNSET
            >>> try:
            ...     _reject_unsupported("web", domain=np.array([0.0, 0.0, 1.0, 1.0]), crs=_UNSET)
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            domain= is not supported by backend='web'

            ```
        - ``basemap=0`` is a number the caller typed, not the inert ``False``, so it is refused:
            ```python
            >>> from digitalearth.api import _reject_unsupported, _UNSET
            >>> try:
            ...     _reject_unsupported("3d", basemap=0, crs=_UNSET)
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            basemap= is not supported by backend='3d'

            ```
    """
    supported = BACKEND_CAPABILITIES[backend]
    for name, value in passed.items():
        if value is _UNSET or name in supported:
            continue
        if name in _INERT and _asks_for_nothing(value, _INERT[name]):
            continue  # they asked for nothing, so nothing was dropped
        honoured = ", ".join(
            repr(other)
            for other in sorted(BACKEND_CAPABILITIES)
            if name in BACKEND_CAPABILITIES[other]
        )
        raise ValueError(
            f"{name}= is not supported by backend={backend!r}; it is honoured by {honoured} — "
            f"drop the argument, or pick one of those backends"
        )


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


def _best_effort(step: str, call, *args, **kwargs) -> Any:
    """Run one best-effort decoration step, tolerating only its unavailability.

    Args:
        step: Human-readable name of the step, used in the warning (e.g. ``"basemap"``).
        call: The bound method to run (e.g. ``scene.basemap``).
        *args: Positional arguments for ``call``.
        **kwargs: Keyword arguments for ``call``.

    Returns:
        Whatever ``call`` returned, or ``None`` when the assets it needs were unreachable.

    Raises:
        Exception: anything that is not an unavailability (see :data:`UNAVAILABLE`) — in particular the
            ``ValueError``/``TypeError`` family that tells the caller what to fix.
    """
    try:
        return call(*args, **kwargs)
    except UNAVAILABLE as error:
        logger.warning(
            "quickmap: %s skipped — %s: %s", step, type(error).__name__, error
        )
        return None


def _add_colorbar(scene: Map) -> Any:
    """Add a colorbar to ``scene``, tolerating only a layer that cannot carry one.

    Args:
        scene: The :class:`Map` whose last layer should get a colorbar.

    Returns:
        The created colorbar, or ``None`` when the layer is outline-only / unmappable.
    """
    try:
        return scene.colorbar()
    except UNMAPPABLE as error:
        logger.warning(
            "quickmap: colorbar skipped — %s: %s", type(error).__name__, error
        )
        return None


def _vector_kind(data: FeatureCollection, caller: str) -> str:
    """Classify a vector input as the family of renderer it needs, refusing an empty collection.

    Args:
        data: The ``FeatureCollection`` about to be drawn.
        caller: Public function name to blame in the error, e.g. ``"quickmap"``.

    Returns:
        ``"polygons"`` when every geometry is a (multi)polygon, ``"points"`` otherwise — the split the
        static and web dispatchers both make. A mixed collection reads as ``"points"``, which is what the
        marker renderers already accepted.

    Raises:
        ValueError: when the collection is empty. Without this guard the all-``True`` result of an empty
            ``geom_type`` check classifies it as polygons and the map draws nothing, silently.
    """
    if len(data) == 0:
        raise ValueError(f"{caller} got an empty FeatureCollection (nothing to draw)")
    if data.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        return "polygons"
    return "points"


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
        ValueError: if ``data`` is an empty ``FeatureCollection`` (nothing to draw), or if ``column`` was
            given for point input — it names a polygon fill and ``Map.scatter`` has no such parameter, so
            forwarding it produced an opaque cleopatra error instead of naming the keyword (review M19).
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    if isinstance(data, FeatureCollection):
        if _vector_kind(data, "quickmap") == "polygons":
            column = kwargs.pop("column", None)
            if column is not None:
                scene.choropleth(data, column=column, **kwargs)
            else:
                scene.shapes(data, **kwargs)
            return
        if "column" in kwargs:
            # `Map.scatter` has no fill column: it sizes markers by `size_column` and colours them from
            # the collection's own value column. Forwarding `column` reached cleopatra, which answered
            # with its own keyword list and never named the caller's parameter (review M19).
            raise ValueError(
                f"column={kwargs['column']!r} fills polygons and has no meaning for point input; "
                "size the markers with size_column=, or drop column="
            )
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
    crs: Any = _UNSET,
    kind: str = "auto",
    domain: Any = _UNSET,
    basemap: bool | str | Any = False,
    coastlines: Any = _UNSET,
    colorbar: Any = _UNSET,
    backend: str = "matplotlib",
    **kwargs,
) -> Any:
    """Build a finished map from a single pyramids object in one call.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (points/polygons).
        crs: Display CRS for the map (``backend="matplotlib"``/``"interactive"``, where it defaults to
            ``3857``, and ``"web"``, which accepts only ``4326``). ``backend="3d"`` has no display CRS, so
            passing it there is refused rather than ignored — reproject with pyramids before plotting.
        kind: Renderer for raster input (``"auto"`` → ``imshow``; or ``contourf``/``contour``/``pcolormesh``).
            ``backend="matplotlib"``/``"interactive"`` only — the web tier picks its own renderer and the 3-D
            tier has no 2-D analogue — so naming a renderer on those is refused, while ``"auto"`` (asking for
            nothing) is accepted anywhere.
        domain: Optional named region / bbox to set the extent. ``backend="matplotlib"`` only — it is the one
            tier with a fixed extent to set — and passing it to any other backend is refused by name.
        basemap: The basemap to draw beneath the data: ``False`` for none, ``True`` for the backend's
            default source, or the source itself — a cleopatra provider name, an ``xyzservices``
            ``TileProvider``, or a **keyed** preset name such as ``"Planet.NICFI"`` — forwarded to the
            backend's basemap method. Unreachable tiles are tolerated and warned about; a request the
            backend cannot satisfy (an unknown preset, a missing credential) is raised. Honoured by
            ``"matplotlib"``, ``"interactive"`` and ``"web"``; ``backend="3d"`` has no basemap, so asking for
            one there is refused rather than ignored, while ``basemap=False`` is accepted anywhere.
        coastlines: When True, overlay coastlines (tolerated and warned about if the assets are
            unreachable). ``backend="matplotlib"``/``"interactive"`` only; ``coastlines=True`` on another
            backend is refused, while ``coastlines=False`` — which asks for nothing — is accepted anywhere.
        colorbar: When True, add a colorbar for the drawn layer (skipped if there is nothing mappable);
            defaults to ``True``. ``colorbar=True`` is not supported by ``backend="web"``, whose key is
            ``WebMap.legend``; ``colorbar=False`` is accepted there, because that tier draws no colorbar to
            begin with, so suppressing one drops no request.
        backend: ``"matplotlib"`` (default) returns a static :class:`Map`; ``"interactive"`` returns a
            pan/zoom :class:`~digitalearth.interactive.map.InteractiveMap` (needs the ``interactive``
            extra); ``"3d"`` returns a :class:`~digitalearth.three_d.scene3d.Scene3D` (needs the ``3d``
            extra); ``"web"`` returns a :class:`~digitalearth.web.map.WebMap` (MapLibre + deck.gl; needs the
            ``web`` extra) — all fed from the same input-type dispatch. Which of the arguments above each one
            honours is :data:`BACKEND_CAPABILITIES`; anything else you pass it is refused, not dropped.
        **kwargs: Forwarded to the underlying draw method (e.g. ``cmap``, ``levels``, ``column``;
            ``z_exaggeration``/``height``/``size`` for ``backend="3d"``). ``column`` names a **polygon**
            fill on ``backend="matplotlib"``: point input there is a ``Map.scatter``, which sizes markers
            by ``size_column`` instead, so ``column`` on points is refused by name rather than forwarded.

    Returns:
        The decorated :class:`Map`, an :class:`InteractiveMap` when ``backend="interactive"``, a
        :class:`Scene3D` when ``backend="3d"``, or a :class:`WebMap` when ``backend="web"``.

    Raises:
        ValueError: for an unknown ``backend``, or for a ``crs``/``domain``/``coastlines``/``colorbar`` the
            chosen backend cannot honour — the message names both the parameter and the backend. Also for a
            ``kind`` naming a renderer the chosen backend does not have, and for ``column`` on point input.

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
        - A parameter the backend cannot honour is refused by name, instead of being silently dropped:
            ```python
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.api import quickmap
            >>> ds = Dataset.read_file("examples/data/acc4000.tif")
            >>> try:
            ...     quickmap(ds, backend="3d", crs=4326)
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            crs= is not supported by backend='3d'

            ```
    """
    if backend not in BACKEND_CAPABILITIES:
        raise ValueError(
            f"unknown backend {backend!r}; choose 'matplotlib' (static), 'interactive' (HoloViz), "
            f"'3d' (PyVista), or 'web' (MapLibre + deck.gl)"
        )
    # Refuse before anything is built, so a rejected call never leaves a half-made figure or VTK plotter.
    _reject_unsupported(
        backend,
        crs=crs,
        kind=kind,
        domain=domain,
        basemap=basemap,
        coastlines=coastlines,
        colorbar=colorbar,
    )
    # Past the gate, fill in the defaults the surviving backends expect.
    colorbar = True if colorbar is _UNSET else colorbar
    coastlines = False if coastlines is _UNSET else coastlines
    domain = None if domain is _UNSET else domain
    if backend == "3d":
        return _quickmap_3d(data, colorbar=colorbar, **kwargs)
    if backend == "interactive":
        return _quickmap_interactive(
            data,
            crs=3857 if crs is _UNSET else crs,
            kind=kind,
            basemap=basemap,
            coastlines=coastlines,
            colorbar=colorbar,
            **kwargs,
        )
    if backend == "web":
        return _quickmap_web(data, crs=crs, basemap=basemap, **kwargs)
    scene = Map(crs=3857 if crs is _UNSET else crs, domain=domain)
    _draw(scene, data, kind, **kwargs)
    if coastlines:
        _best_effort("coastlines", scene.coastlines)
    if basemap:
        # `True` means "the backend's default source"; anything else names the source and is forwarded, so a
        # keyed preset reaches Map.basemap instead of being silently reduced to the default.
        _best_effort("basemap", scene.basemap, _basemap_source(basemap))
    if domain is not None:
        scene.set_domain()
    if (
        colorbar
        and scene.layers
        and scene.layers[-1][1] is not None
        and not _last_layer_is_categorical(scene)
    ):
        _add_colorbar(scene)
    return scene


def _basemap_source(basemap: Any) -> Any:
    """Translate the ``basemap`` argument into the source a backend's basemap method takes.

    Args:
        basemap: The ``quickmap`` argument — ``True`` for the backend default, or a provider name /
            ``TileProvider`` / keyed preset name.

    Returns:
        ``None`` (the backend's own default) for a plain ``True``, else ``basemap`` unchanged.
    """
    return None if basemap is True else basemap


def quickplot(data: PlottableData, **kwargs) -> Any:
    """Alias of :func:`quickmap` — build a finished map from ``data`` in one call."""
    return quickmap(data, **kwargs)


def _quickmap_interactive(
    data: PlottableData,
    *,
    crs: Any,
    kind: str,
    basemap: bool | str | Any,
    coastlines: bool,
    colorbar: bool = True,
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
        basemap: ``True`` for the backend's default tile source, or the source itself (provider name or
            keyed preset), forwarded to ``InteractiveMap.tiles``.
        coastlines: When True, overlay a coastline.
        colorbar: When False, drop the colorbar the builder draws by default (mirrors the
            matplotlib backend's ``colorbar`` toggle); ``True`` leaves the builder default in place.
        **kwargs: Forwarded to the chosen builder (e.g. ``cmap``, ``column``).

    Returns:
        The decorated :class:`~digitalearth.interactive.map.InteractiveMap`.

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection``, or if ``kind`` names a renderer this
            tier does not have — it is refused by name rather than quietly drawn as an ``image``.
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
        if _vector_kind(data, "quickplot") == "polygons":
            column = kwargs.pop("column", None)
            if column is not None:
                scene.choropleth(data, column=column, **kwargs)
            else:
                scene.polygons(data, **kwargs)
        else:
            scene.points(data, **kwargs)
    elif isinstance(data, Dataset):
        if kind not in _raster_kind:
            # A renderer this tier does not have used to fall back to `image` and draw something the caller
            # never asked for; the matplotlib backend has always refused it (review M7).
            renderers = ", ".join(repr(name) for name in sorted(_raster_kind))
            raise ValueError(
                f"kind={kind!r} is not a renderer of backend='interactive'; use one of {renderers}"
            )
        getattr(scene, _raster_kind[kind])(data, **kwargs)
    else:
        raise TypeError(f"quickplot cannot draw a {type(data).__name__}")
    if (
        not colorbar and scene.layers
    ):  # builders draw a colorbar by default; drop it on the data layer
        scene.colorbar(False)
    if basemap:
        source = _basemap_source(basemap)
        if source is None:
            scene.tiles()
        else:
            scene.tiles(source)
    if coastlines:
        scene.coastlines()
    return scene


def _quickmap_web(
    data: PlottableData,
    *,
    crs: Any = _UNSET,
    basemap: bool | str | Any = False,
    **kwargs,
) -> Any:
    """Build a finished ``WebMap`` from ``data`` (the ``backend="web"`` path, DX.1).

    Dispatches by input type, mirroring :func:`_draw`: a polygon ``FeatureCollection`` with a ``column``
    becomes a ``choropleth`` (else outline ``polygons``); other vectors become ``points``; a raster
    ``Dataset`` becomes ``add_raster``. The ``WebMap`` import is lazy so the core ``api`` works without the
    ``web`` extra. The web tier normalises data to lon/lat itself, so the matplotlib-oriented ``kind`` /
    ``domain`` / ``coastlines`` kwargs have no counterpart here and :func:`_reject_unsupported` refuses them
    upstream. ``crs`` **is** forwarded: ``WebMap`` carries one and validates it, so a CRS it cannot place
    inline is reported by the tier that knows why, rather than dropped here.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (points/polygons).
        crs: Display CRS, forwarded to ``WebMap``; :data:`_UNSET` leaves the web tier's own default.
        basemap: ``True`` for the web tier's default dark tile basemap, or the source itself (provider name
            or keyed preset), forwarded to ``WebMap.basemap``.
        **kwargs: Forwarded to the chosen ``WebMap`` builder (e.g. ``cmap``, ``column``, ``scheme``, ``k``).

    Returns:
        The built :class:`~digitalearth.web.map.WebMap`.

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection``, or if ``WebMap`` refuses ``crs``.
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    from digitalearth.web import WebMap

    scene = WebMap() if crs is _UNSET else WebMap(crs=crs)
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
        scene.add_raster(data, **kwargs)
    else:
        raise TypeError(f"quickplot cannot draw a {type(data).__name__}")
    if basemap:
        source = _basemap_source(basemap)
        scene.basemap() if source is None else scene.basemap(source)
    return scene


def _quickmap_3d(data: PlottableData, *, colorbar: bool = True, **kwargs) -> Any:
    """Build a finished ``Scene3D`` from ``data`` (the ``backend="3d"`` path, DX.1).

    Dispatches by input type, mirroring :func:`_draw`: a raster ``Dataset`` becomes 3-D relief
    (``terrain``); a point ``FeatureCollection`` becomes a ``point_cloud`` (coloured by ``column`` when
    given); a polygon ``FeatureCollection`` becomes ``extruded_polygons`` (extruded by, and coloured by,
    ``column``). The ``Scene3D`` import is lazy so the core ``api`` works without the ``3d`` extra. The
    map-only kwargs (``crs``/``kind``/``domain``/``basemap``/``coastlines``) have no 3-D analogue and are
    not accepted here — ``crs``/``domain``/``coastlines`` are refused by name in :func:`quickmap` before this
    is reached, since :attr:`~digitalearth.three_d.base.Scene3DBase.display_crs` records that this tier
    projects nothing.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (points/polygons).
        colorbar: When False, hide the scalar bar (``show_scalar_bar=False``); True leaves PyVista's
            default (a bar iff the layer carries scalars).
        **kwargs: Forwarded to the chosen ``Scene3D`` builder (e.g. ``cmap``, ``z_exaggeration``,
            ``column``, ``height``, ``point_size``).

    Returns:
        The built :class:`~digitalearth.three_d.scene3d.Scene3D`.

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection``.
        TypeError: for a line ``FeatureCollection`` (no 3-D builder) or a non-raster/vector input.
    """
    from digitalearth.three_d import Scene3D

    # Validate the input up front: a bad type / unsupported geometry must raise BEFORE a Scene3D (which opens a
    # VTK plotter) is constructed — otherwise every error path leaks an open plotter.
    geom_kind = None
    if isinstance(data, FeatureCollection):
        if len(data) == 0:
            raise ValueError(
                "quickplot got an empty FeatureCollection (nothing to draw)"
            )
        geom_type = data.geometry.geom_type
        if geom_type.isin(["Polygon", "MultiPolygon"]).all():
            geom_kind = "polygons"
        elif geom_type.isin(["Point", "MultiPoint"]).all():
            geom_kind = "points"
        else:
            raise TypeError(
                "backend='3d' needs a uniformly point, polygon, or raster input "
                "(point_cloud / extruded_polygons / terrain); got geometry types "
                f"{sorted(geom_type.unique())}"
            )
    elif not isinstance(data, Dataset):
        raise TypeError(f"quickplot cannot draw a {type(data).__name__}")

    if (
        not colorbar
    ):  # PyVista shows a scalar bar by default when scalars exist; force it off here
        kwargs.setdefault("show_scalar_bar", False)
    scene = (
        Scene3D()
    )  # constructed only after validation — the error paths above never leak a plotter
    if isinstance(data, Dataset):
        scene.terrain(data, **kwargs)
    elif geom_kind == "polygons":
        column = kwargs.pop("column", None)
        height = kwargs.pop("height", column if column is not None else 1.0)
        scene.extruded_polygons(data, height=height, column=column, **kwargs)
    else:  # points
        scene.point_cloud(data, value_column=kwargs.pop("column", None), **kwargs)
    return scene


def _last_layer_is_categorical(scene: Map) -> bool:
    """Return whether ``scene``'s most recent layer is a categorical fill (keyed by a swatch legend).

    A categorical fill's mappable carries opaque integer class codes, so an aggregated colorbar over it would
    read ``0, 1, 2 …`` instead of the category labels — the glyph draws its own swatch legend instead
    (``PolygonGlyph.category_legend`` is set, non-categorical glyphs leave it ``None`` or absent). The one-call
    wrappers must not add a colorbar on top of that legend.

    Args:
        scene: The :class:`Map` whose last layer is inspected.

    Returns:
        ``True`` when the last layer's glyph exposes a drawn ``category_legend``, else ``False``.
    """
    if not scene.layers:
        return False
    return getattr(scene.layers[-1][0], "category_legend", None) is not None


def _finish(scene: Map, *, colorbar: bool) -> Map:
    """Add an aggregated colorbar to ``scene`` when requested and a mappable layer exists; return ``scene``.

    The shared tail of the one-call wrappers: a colorbar is added only when ``colorbar`` is true and a layer
    was drawn, and an outline-only / unmappable layer (which cannot carry a colorbar) is warned about rather
    than raised (see :data:`UNMAPPABLE`; anything else propagates). A categorical fill is skipped too — it
    keys itself with a swatch legend, and a colorbar over its integer class codes would be a meaningless
    second key (see :func:`_last_layer_is_categorical`).

    Args:
        scene: The :class:`Map` a wrapper has already drawn on.
        colorbar: Whether this plot kind should carry a colorbar (e.g. only when a value ``column`` was set).

    Returns:
        The same ``scene`` (so wrappers can ``return _finish(...)``).
    """
    if colorbar and scene.layers and not _last_layer_is_categorical(scene):
        _add_colorbar(scene)
    return scene


def _method(name: str):
    """Build a module-level function that quick-draws via the ``Map`` method ``name``.

    The wrapper *is* the ``kind``: it injects ``kind=name`` on the caller's behalf. So when the chosen
    backend has no renderer selector, :func:`_reject_unsupported`'s "drop the argument" message named a
    keyword the caller never typed (review L5). The wrapper answers for its own injection instead, naming
    itself and the call that does work.

    Args:
        name: The ``Map`` renderer the wrapper draws with, also the wrapper's own name.

    Returns:
        The module-level quick-draw function.
    """

    def _fn(data: PlottableData, **kwargs) -> Map:
        backend = kwargs.get("backend", "matplotlib")
        supported = BACKEND_CAPABILITIES.get(backend)
        if supported is not None and "kind" not in supported:
            honoured = ", ".join(
                repr(other)
                for other in sorted(BACKEND_CAPABILITIES)
                if "kind" in BACKEND_CAPABILITIES[other]
            )
            raise ValueError(
                f"{name}() draws with the {name!r} renderer, which backend={backend!r} does not have; "
                f"it is honoured by {honoured} — call quickmap(data, backend={backend!r}) instead"
            )
        return quickmap(data, kind=name, **kwargs)

    _fn.__name__ = name
    _fn.__doc__ = f"""Quick-draw ``data`` with :meth:`Map.{name}` and return the finished Map.

    The wrapper *is* the renderer choice: it calls :func:`quickmap` with ``kind={name!r}`` on the caller's
    behalf, so everything else :func:`quickmap` accepts is written here unchanged.

    Args:
        data: A pyramids ``Dataset`` or ``FeatureCollection`` to draw.
        **kwargs: Forwarded to :func:`quickmap` (``crs``, ``domain``, ``basemap``, ``coastlines``,
            ``colorbar``, ``backend``, plus styling kwargs). ``kind`` is not among them — this wrapper
            supplies it.

    Returns:
        The finished map :func:`quickmap` built.

    Raises:
        ValueError: when ``backend=`` names a backend that has no renderer selector, since the
            {name!r} renderer is this wrapper's own injection rather than something the caller
            asked for; the message names the backends that do honour it, and points at
            :func:`quickmap` for the chosen one.
    """
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


def voronoi(data: PlottableData, column: str | None = None, **kwargs) -> Map:
    """Quick-draw the Voronoi diagram of a point FeatureCollection; returns the finished Map.

    With ``column`` the cells are filled and coloured by that value (a colorbar is added); without it the cell
    outlines are drawn. ``clip`` and styling kwargs are forwarded to :meth:`Map.voronoi`.

    Args:
        data: A pyramids ``FeatureCollection`` of point geometries.
        column: Numeric column whose value colours each cell, or ``None`` (default) for
            outlines only — which is also what decides whether a colorbar is added.
        **kwargs: ``crs`` sets the display CRS (default ``3857``); everything else is forwarded to
            :meth:`Map.voronoi` (``clip``, ``cmap``, ``scheme``, ``k``, …).

    Returns:
        The finished :class:`Map`, with the cells registered as its single layer.

    Examples:
        - Outlines only: one layer, and a single axes, because there is nothing to key:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import voronoi
            >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
            >>> m = voronoi(fc, crs=fc.epsg)
            >>> len(m.layers)
            1
            >>> len(m.fig.axes)
            1

            ```
        - Naming a column fills the cells and adds the colorbar axes alongside the map:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import voronoi
            >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
            >>> m = voronoi(fc, column="fid", crs=fc.epsg)
            >>> len(m.fig.axes)
            2

            ```

    See Also:
        digitalearth.static.maps.vector.VectorMixin.voronoi: the ``Map`` method this wraps.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.voronoi(data, column=column, **kwargs)
    return _finish(scene, colorbar=column is not None)


def cartogram(
    data: PlottableData, scale: str, column: str | None = None, **kwargs
) -> Map:
    """Quick-draw a cartogram (polygons scaled by ``scale``); returns the finished Map.

    With ``column`` the scaled polygons are filled and coloured by that value (a colorbar is added); without it
    the outlines are drawn. ``limits`` and styling kwargs are forwarded to :meth:`Map.cartogram`.

    Args:
        data: A pyramids ``FeatureCollection`` of polygon geometries.
        scale: Numeric column whose value **sizes** each polygon — it is scaled about its own
            centroid by a factor normalised across the layer. Required, and distinct from
            ``column``: this one distorts area, ``column`` only colours.
        column: Numeric column whose value colours each scaled polygon, or ``None``
            (default) for outlines only — which also decides whether a colorbar is added.
        **kwargs: ``crs`` sets the display CRS (default ``3857``); everything else is forwarded to
            :meth:`Map.cartogram` (``limits``, ``cmap``, ``scheme``, ``k``, …).

    Returns:
        The finished :class:`Map`, with the scaled polygons registered as its single layer.

    Examples:
        - Size the polygons by a column and draw their outlines — one layer, one axes:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import cartogram
            >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
            >>> fc["geometry"] = fc.geometry.buffer(500.0)
            >>> m = cartogram(fc, scale="fid", crs=fc.epsg)
            >>> len(m.layers)
            1
            >>> len(m.fig.axes)
            1

            ```
        - Size *and* colour by the same column: the fill is keyed by the colorbar axes:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import cartogram
            >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
            >>> fc["geometry"] = fc.geometry.buffer(500.0)
            >>> m = cartogram(fc, scale="fid", column="fid", crs=fc.epsg)
            >>> len(m.fig.axes)
            2

            ```

    See Also:
        digitalearth.static.maps.vector.VectorMixin.cartogram: the ``Map`` method this wraps.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.cartogram(data, scale=scale, column=column, **kwargs)
    return _finish(scene, colorbar=column is not None)


def quadtree(data: PlottableData, column: str | None = None, **kwargs) -> Map:
    """Quick-draw a quadtree choropleth of a point FeatureCollection; returns the finished Map.

    Cells are coloured by an aggregate of ``column`` (or point count when ``None``) and a colorbar is added.
    ``agg``/``nmax``/``nmin``/``clip`` and styling kwargs are forwarded to :meth:`Map.quadtree`.

    Args:
        data: A pyramids ``FeatureCollection`` of point geometries.
        column: Numeric column aggregated per cell, or ``None`` (default) to colour by point count.
            Unlike the other wrappers here, it does **not** decide the colorbar — a quadtree
            is always a filled choropleth, so the bar is always drawn.
        **kwargs: ``crs`` sets the display CRS (default ``3857``); everything else is forwarded to
            :meth:`Map.quadtree` (``agg``, ``nmax``, ``nmin``, ``clip``, ``cmap``, ``scheme``,
            ``k``, …).

    Returns:
        The finished :class:`Map`, with the aggregated cells registered as its single layer.

    Examples:
        - Without a column the cells carry density; ``nmax`` sets how finely the box splits:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import quadtree
            >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
            >>> m = quadtree(fc, crs=fc.epsg, nmax=1)
            >>> len(m.layers)
            1
            >>> len(m.fig.axes)  # always filled, so always keyed
            2

            ```
        - With a column the cells carry an aggregate of it instead of a count:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import quadtree
            >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
            >>> m = quadtree(fc, column="fid", agg="max", crs=fc.epsg, nmax=1)
            >>> len(m.fig.axes)
            2

            ```

    See Also:
        digitalearth.static.maps.vector.VectorMixin.quadtree: the ``Map`` method this wraps.
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
    column: str | None = None,
    scale: str | None = None,
    **kwargs,
) -> Map:
    """Quick-draw a spatial flow / Sankey map of a line FeatureCollection; returns the finished Map.

    ``column`` colours each path and ``scale`` sets its width (both optional); styling kwargs are forwarded to
    :meth:`Map.sankey`. A colorbar is added when ``column`` is given.

    Args:
        data: A pyramids ``FeatureCollection`` of ``LineString``/``MultiLineString`` geometries; a
            MultiLineString contributes one path per part.
        column: Numeric column whose value **colours** each path, or ``None`` (default) for
            one colour — which is also what decides whether a colorbar is added.
        scale: Numeric column whose value sets each path's **line width**, or ``None``
            (default) for a uniform width. Independent of ``column``: either, both, neither.
        **kwargs: ``crs`` sets the display CRS (default ``3857``); everything else is forwarded to
            :meth:`Map.sankey` (``width_limits``, ``width_scale``, ``cmap``, ``size_legend``, …).

    Returns:
        The finished :class:`Map`, with the flow paths registered as its single layer.

    Examples:
        - Bare paths at a uniform width: nothing is encoded, so nothing is keyed:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import LineString
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import sankey
            >>> gdf = gpd.GeoDataFrame(
            ...     {"flow": [1.0, 2.0]},
            ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(0, 1), (1, 2)])],
            ...     crs="EPSG:4326",
            ... )
            >>> m = sankey(FeatureCollection(gdf), crs=4326)
            >>> len(m.layers)
            1
            >>> len(m.fig.axes)
            1

            ```
        - Colour by one column and widen by another (the same one here) — colour gets the bar:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import LineString
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.api import sankey
            >>> gdf = gpd.GeoDataFrame(
            ...     {"flow": [1.0, 2.0]},
            ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(0, 1), (1, 2)])],
            ...     crs="EPSG:4326",
            ... )
            >>> m = sankey(FeatureCollection(gdf), column="flow", scale="flow", crs=4326)
            >>> len(m.fig.axes)
            2

            ```

    See Also:
        digitalearth.static.maps.vector.VectorMixin.sankey: the ``Map`` method this wraps.
    """
    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.sankey(data, column=column, scale=scale, **kwargs)
    return _finish(scene, colorbar=column is not None)
