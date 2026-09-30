"""quickplot — one-call entry points that build a decorated :class:`~digitalearth.static.map.Map`.

``quickmap`` (and its alias ``quickplot``) dispatch on the input type, auto-style the data, draw it on a
``Map``, optionally decorate (basemap/coastlines/domain), and add a colorbar — returning the finished
``Map`` so callers can further tweak or ``save`` it. Module-level functions (``field``, ``contours``,
``points``, ``choropleth``, …) mirror the ``Map`` methods for a terse functional API.

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

**The refusal is the declared one.** Both gates — :func:`_reject_unsupported` and the renderer wrappers
:func:`field`/:func:`contours`/:func:`pcolormesh` — decide from the tier's own
:class:`~digitalearth.base.capabilities.Capabilities`, carry the reason that declaration gave, and raise its
:class:`~digitalearth.base.capabilities.CapabilityError`. It subclasses ``ValueError``, so a caller catching
``ValueError`` around ``quickmap`` keeps catching it, and one that wants only this refusal can now name it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

from digitalearth.base.capabilities import Capabilities, CapabilityError
from digitalearth.base.types import PlottableData
from digitalearth.interactive.capabilities import (
    CAPABILITIES as CAPABILITIES_INTERACTIVE,
)
from digitalearth.three_d.capabilities import CAPABILITIES as CAPABILITIES_3D
from digitalearth.web.capabilities import CAPABILITIES as CAPABILITIES_WEB

if TYPE_CHECKING:
    # A type checker resolves `Map` from here; at runtime it is imported lazily inside each function that
    # builds one, so importing `api` — and a `web`/`interactive`/`3d` `quickmap` — never loads the
    # matplotlib tier, which `digitalearth.static` imports eagerly with `Map` (DE-24).
    from digitalearth.base.spec import FigureSpec
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
#: Which capability each of `quickmap`'s map-shaped parameters needs, in the vocabulary every tier now
#: declares itself in (#294).
#: A keyword is honoured when the backend supports any of the capabilities listed for it: `colorbar=` is the
#: web tier's legend and the 3-D tier's scalar bar, which are one request with two names.
_KEYWORD_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "crs": ("display_crs",),
    "kind": ("raster_renderer",),
    "domain": ("domain",),
    "basemap": ("basemap",),
    # The *overlay*, not the kind: the 3-D tier draws a `coastlines` layer — a globe puts its own shoreline
    # on the sphere — while having no builder to add one to a scene, and `quickmap(coastlines=True)` asks for
    # the second. One name answering both questions accepted a request the tier cannot honour (review M6).
    "coastlines": ("coastline_overlay",),
    "colorbar": ("colorbar", "legend"),
}

_V = TypeVar("_V")


class _LazyBackendMap(Mapping[str, _V]):
    """A backend→value table whose ``matplotlib`` row is resolved, and cached, on first read.

    Reading the ``matplotlib`` row is what loads the matplotlib tier: ``digitalearth.static`` imports
    ``Map`` eagerly, so its :class:`~digitalearth.base.capabilities.Capabilities` cannot be read without
    loading matplotlib. The three tiers that declare their capabilities engine-free (``3d``, ``web``,
    ``interactive``) are held eagerly; ``matplotlib`` is deferred. Membership and iteration name all four
    backends **without** resolving any row, so ``import digitalearth.api`` and ``backend in
    BACKEND_CAPABILITIES`` — asked on every ``quickmap`` — and a *successful* ``web``/``interactive``/``3d``
    call never load matplotlib (DE-24). A refusal *message* is the one thing that does: it names the backends
    that honour the refused keyword from their real declared rows (:func:`_honoured_by`), and matplotlib's
    cannot be read without importing the tier — a rare error path where correctness beats the laziness N3
    tried to keep there (review L2). Only reading the matplotlib row itself loads the tier.

    **The cache is not thread-synchronized, and deliberately so.** ``__getitem__`` fills ``_cache`` with an
    unguarded read-modify-write, so two threads racing on the first read of a deferred row can both run its
    builder. That is benign: the deferred builder is **idempotent** — a deterministic ``frozenset`` over an
    import Python itself caches — so the worst a race does is build the same row twice and converge on an
    equal value, and after the first read the row is hot and no builder runs at all. A lock would guard a
    table read once and then never rebuilt, which was judged more machinery than the race is worth
    (review N2); a future deferred row whose builder stops being idempotent would need one.

    Args:
        eager: The rows that need no engine to build, by backend.
        lazy: The zero-argument builders for the deferred rows, by backend — each called once and cached.
    """

    def __init__(
        self, eager: Mapping[str, _V], lazy: Mapping[str, Callable[[], _V]]
    ) -> None:
        self._eager = dict(eager)
        self._lazy = dict(lazy)
        self._cache: dict[str, _V] = {}

    def __getitem__(self, backend: str) -> _V:
        """Return a backend's row, building and caching a deferred one on first read."""
        if backend in self._eager:
            return self._eager[backend]
        if backend not in self._lazy:
            raise KeyError(backend)
        if backend not in self._cache:
            self._cache[backend] = self._lazy[backend]()
        return self._cache[backend]

    def __contains__(self, backend: object) -> bool:
        """Answer membership from the backend names alone, resolving no deferred row."""
        return backend in self._eager or backend in self._lazy

    def __iter__(self) -> Iterator[str]:
        """Yield the backend names — the eager ones, then the deferred — resolving nothing."""
        yield from self._eager
        yield from self._lazy

    def __len__(self) -> int:
        """Return the number of backends, deferred rows included, resolving nothing."""
        return len(self._eager) + len(self._lazy)


def _static_declaration() -> Capabilities:
    """Return the matplotlib tier's :class:`Capabilities`, importing ``digitalearth.static`` lazily.

    That import loads matplotlib — the tier imports ``Map`` eagerly — so it is deferred to the moment a
    caller actually needs the matplotlib row, never at ``import digitalearth.api`` (DE-24).
    """
    from digitalearth.static.capabilities import CAPABILITIES

    return CAPABILITIES


#: Every tier's declaration, read from the tier's own `capabilities.py`. The three engine-free tiers are
#: held eagerly; `matplotlib` is deferred through :func:`_static_declaration`, because reading it loads the
#: matplotlib tier and importing `api` must not (DE-24). It still names all four backends, and reading its
#: `matplotlib` entry gives the same `Capabilities` the hand-written row derived from before.
_DECLARATIONS: Mapping[str, Capabilities] = _LazyBackendMap(
    eager={
        "3d": CAPABILITIES_3D,
        "web": CAPABILITIES_WEB,
        "interactive": CAPABILITIES_INTERACTIVE,
    },
    lazy={"matplotlib": _static_declaration},
)


def _declared_row(declaration: Capabilities) -> frozenset[str]:
    """Return the `quickmap` keywords a declaration says the backend honours.

    Args:
        declaration: The tier's `Capabilities`.

    Returns:
        The keyword names, as :data:`BACKEND_CAPABILITIES` holds them.

    Examples:
        - The 3-D tier's row is read from its own declaration rather than written here:
            ```python
            >>> from digitalearth.api import BACKEND_CAPABILITIES
            >>> sorted(BACKEND_CAPABILITIES["3d"])
            ['colorbar', 'crs']

            ```
    """
    return frozenset(
        keyword
        for keyword, capabilities in _KEYWORD_CAPABILITIES.items()
        if any(declaration.supports(capability) for capability in capabilities)
    )


def _refusal_reason(backend: str, keyword: str) -> str:
    """Return the tier's own reason for not honouring a keyword, when it declared one.

    Args:
        backend: The backend that was asked.
        keyword: The `quickmap` keyword it cannot honour.

    Returns:
        The declared reason, or `""` when the tier keeps no declaration yet or said nothing about it.
    """
    declaration = _DECLARATIONS.get(backend)
    if declaration is None:
        return ""
    reasons = [
        declaration.reason(capability)
        for capability in _KEYWORD_CAPABILITIES.get(keyword, ())
    ]
    return next((reason for reason in reasons if reason), "")


#: Which of ``quickmap``'s map-shaped parameters each backend can actually honour. Anything a caller passes
#: that is not listed for their backend is refused by name rather than dropped (:func:`_reject_unsupported`).
#: Every row is **derived** from that tier's own :class:`~digitalearth.base.capabilities.Capabilities`
#: (#294): the rows are read here, and decided there.
#:
#: * ``matplotlib`` is the only tier with an extent setter, so it is the only one that takes ``domain``.
#: * ``interactive`` pans and zooms, so it has no fixed extent; it does have a display CRS, coastlines and a
#:   colorbar toggle.
#: * ``3d`` draws every layer in one display CRS, given as ``crs`` or taken from the first layer
#:   (:attr:`digitalearth.three_d.base.Scene3DBase.display_crs`); it has no coastline or extent concept. Its
#:   ``colorbar`` toggle reaches the tier's own `colorbar`/`legend` through :func:`_add_3d_key` (order 24):
#:   PyVista's scalar bar for a layer coloured by a ramp, its keyed legend for one coloured by classes, and
#:   nothing at all — without an error — for a layer whose colour is flat.
#: * ``web`` places inline data in lon/lat and carries a ``crs`` of its own, which it validates. It has no
#:   coastline layer. Its colour key is ``WebMap.legend``, which is a builder rather than a toggle, so
#:   ``colorbar=`` is translated here rather than forwarded: ``True`` builds the key only when a layer
#:   recorded a classification, and nothing is tolerated once the builder is reached (#254). Renaming the
#:   tier methods themselves — a builder that takes content vs a visibility flag — is Core-contract work
#:   and stays with U-3.
BACKEND_CAPABILITIES: Mapping[str, frozenset[str]] = _LazyBackendMap(
    eager={
        backend: _declared_row(_DECLARATIONS[backend])
        for backend in ("3d", "web", "interactive")
    },
    lazy={"matplotlib": lambda: _declared_row(_static_declaration())},
)

#: The value of a checked parameter that asks for **nothing**, where one exists. Passing it to a backend that
#: cannot honour the parameter is not a dropped request — there was no request — so it is allowed through.
#:
#: ``colorbar=False`` was one of them (review L9). The rule here is about *dropped requests*, and what
#: ``False`` asked for was a map with no colorbar — exactly what the one tier that could not honour the
#: parameter (``web``) drew anyway. Nothing was lost, so refusing it was pedantry of the same shape as
#: refusing ``coastlines=False`` there, and it broke the caller who passes one kwargs dict through to
#: whichever backend they picked. The entry is now moot rather than wrong: ``web`` declares ``colorbar``
#: too (#254), so no value of it reaches this table on any backend.
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


def _honoured_by(keyword: str) -> str:
    """List the backends that honour ``keyword``, for a refusal message, from each tier's real declared row.

    Reads every backend's row from :data:`BACKEND_CAPABILITIES`, resolving the deferred ``matplotlib`` one
    too, so a backend is named **iff** its own declaration honours the keyword. ``matplotlib`` is the only
    deferred row, and its declaration cannot be read as data without importing the tier —
    ``digitalearth.static`` imports ``Map`` eagerly, so importing ``digitalearth.static.capabilities`` loads
    matplotlib — so there is no engine-free row to read here.

    N3 avoided that load by naming ``matplotlib`` from the standing fact that the default full tier honours
    every checked keyword; but it applied that fact *unconditionally* (``keyword in _KEYWORD_CAPABILITIES``
    is true at every call site), so it named ``matplotlib`` for a keyword matplotlib does **not** honour as
    readily as for one it does — a message that would lie the moment such a keyword were added (review L2).
    A refusal is a rare error path — an exception is about to be raised — so resolving the real row here, at
    the cost of loading the tier on that path alone, buys a message that cannot lie. Import and a *successful*
    ``web``/``interactive``/``3d`` dispatch still never resolve the ``matplotlib`` row, which is the laziness
    that matters (DE-24).

    Args:
        keyword: The ``quickmap`` keyword a backend could not honour.

    Returns:
        The backends whose declaration honours it, ``repr``-quoted and comma-joined in sorted order. The
        refusing backend is absent for free: it does not honour ``keyword``, which is why it was refused.
    """
    honoured = {
        name for name in BACKEND_CAPABILITIES if keyword in BACKEND_CAPABILITIES[name]
    }
    return ", ".join(repr(name) for name in sorted(honoured))


def _reject_unsupported(backend: str, **passed: Any) -> None:
    """Refuse any keyword the chosen ``backend`` cannot honour, naming the parameter and the backend.

    Args:
        backend: The backend name ``quickmap`` was called with; already validated against
            :data:`BACKEND_CAPABILITIES`.
        **passed: The checked parameters as the caller left them — :data:`_UNSET` for the ones they never
            named.

    Raises:
        CapabilityError: for the first parameter that was actually requested and that ``backend`` cannot
            honour — a ``ValueError``, so a caller catching that keeps catching this. The message names both
            halves, so the caller learns which one to change, and carries the tier's own reason when it
            declared one.

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
        honoured = _honoured_by(name)
        reason = _refusal_reason(backend, name)
        raise CapabilityError(
            f"{name}= is not supported by backend={backend!r}; "
            + (f"{reason}. " if reason else "")
            + f"It is honoured by {honoured} — drop the argument, or pick one of those backends"
        )


__all__ = [
    "quickmap",
    "quickplot",
    "to_backend",
    "field",
    "contours",
    "pcolormesh",
    "points",
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


def _add_static_key(scene: Map, *, visible: bool) -> None:
    """Ask the matplotlib tier for a layer's colour key, tolerating only a layer that cannot carry one.

    **Which key is a property of the layer, not of the flag** — the branch :func:`_add_3d_key` already had.
    The rule is stated in full at :func:`~digitalearth.static.guides.guide_kind` and is the same on all four
    backends: a scale is explained by what it is made of. Here that means a **categorical** fill is keyed by
    a swatch list and every other scale by a bar — a classified one banded on its class edges, a continuous
    one as a ramp. ``Map.colorbar`` refuses the categorical case outright, because a bar over it would read
    the class codes cleopatra assigned rather than the class names. Calling only ``colorbar()`` and
    swallowing that refusal is what left the default backend recording no key at all for a fill the web and
    3-D paths both keyed (review M6); the choice is asked of
    :func:`~digitalearth.static.guides.guide_kind`, which is where this tier already answers it.

    This docstring said "a continuous ramp is explained by a bar and a *categorical* fill by a keyed list"
    while :func:`_add_3d_key`'s said a *classified* fill gets the list, so the two stated different rules
    for the same layer a few hundred lines apart (review M2). Neither tier's behaviour changed: what
    differs is the furniture, and the table at ``guide_kind`` is the one place that says so.

    **Whether it is drawn is the caller's, and it is carried through rather than read as "skip the call".**
    This tier's builders draw a swatch legend of their own for ``scheme="categorical"``, so a path that only
    ever *added* a key could not take that one off: ``quickmap(colorbar=False)`` returned a map with a
    legend on it and a description that said nothing about a key at all, while the interactive and 3-D
    paths recorded ``Guide(show=False)`` and drew none. Passing the flag down records the decision either
    way — which is what lets a switcher offer the key back — and keeps the layer resolved and checked before
    the flag is read, the ordering all four tiers share.

    Args:
        scene: The :class:`Map` whose most recent colour-keyed layer the key describes. Which layer that is
            is ``Map.colorbar``/``Map.legend``'s own question since order 24 — the most recent one that
            publishes a colour encoding, never simply the last one added, so a coastline or a basemap drawn
            after the data cannot take the key. It is read here through the same
            :meth:`~digitalearth.static.scene.Scene._color_keyed` list those methods resolve through, so the
            layer this branches on and the layer they key cannot differ.
        visible: Whether the key is drawn. ``False`` records the guide switched **off** and takes off the
            key the builder drew, as ``quickmap(colorbar=False)`` asks for. The two calls that takes are
            tolerated separately, so an engine failure in the first — the only one that draws anything —
            cannot cost the second its record of the decision (review N1).

    Returns:
        Nothing. Both methods return the map itself, as the Core declares, so there is no artist to hand
        back; the drawn key is reachable per layer from the renderer.
    """
    from digitalearth.static.guides import guide_kind

    keyed = scene._color_keyed()
    if not keyed:  # pragma: no cover - `_has_a_key_to_draw` is asked first
        return
    kind = guide_kind(scene.get_layer(keyed[-1]))
    ask = scene.legend if kind == "legend" else scene.colorbar
    if not visible:
        # Take the key over before switching it off. `Renderer.draw_guide` removes the key **it** drew
        # — `DrawnLayer.guides` — and the swatch legend a categorical glyph draws for itself is not one
        # of those, so asking for `visible=False` alone recorded the decision and left that legend on
        # the axes. Drawing the key once through the tier's own method makes it the layer's (matplotlib
        # holds one legend per axes, so this replaces rather than stacks), and the call below then takes
        # it off. Both calls resolve and check the layer first, so a refusal still happens before
        # anything is recorded.
        #
        # Tolerated on its own rather than under one `except` with the call below (review N1). This is the
        # **only** one of the two that reaches `colorbar_legend`/`disjoint_legend`, because the other draws
        # nothing — so it is the only one an engine failure can come out of, and sharing a guard meant that
        # failure took the `Guide(show=False)` record with it. The decision is the thing `colorbar=False`
        # exists to record, and the take-over is housekeeping over a key the builder drew: worth doing,
        # not worth losing the record over.
        try:
            ask()
        except UNMAPPABLE as error:
            logger.warning(
                "quickmap: taking the %s over first skipped — %s: %s",
                kind,
                type(error).__name__,
                error,
            )
    try:
        ask(visible=visible)
    except UNMAPPABLE as error:
        logger.warning(
            "quickmap: %s skipped — %s: %s", kind, type(error).__name__, error
        )


#: The static tier's renderer for each raster ``kind`` whose method is spelled differently, as
#: ``{kind: (method, keywords)}``.
#:
#: The companion of :data:`_INTERACTIVE_RASTER_KINDS`, and here for the same reason: a ``kind`` is a
#: **renderer** a caller names — matplotlib's own vocabulary, which is also what cleopatra's ``ArrayGlyph``
#: takes — and it is not the method's name. The static tier's Core renames (order 27a) moved those methods,
#: so dispatching on the kind alone would now reach nothing at all.
#:
#: The keywords are why this is a pair rather than a name. ``contour`` and ``contourf`` are one method now,
#: told apart by ``filled=`` (#262), so the kind carries the argument that picks the render. A kind not
#: listed here is the method's own name with no keywords — ``pcolormesh``, ``block``.
_STATIC_RASTER_KINDS = {
    "auto": ("field", {}),
    "imshow": ("field", {}),
    "contour": ("contours", {"filled": False}),
    "contourf": ("contours", {"filled": True}),
}


def _renderer_for(table: dict, kind: str, kwargs: dict, caller: str) -> tuple:
    """Resolve a raster ``kind`` to its method and keywords, refusing one the caller contradicted.

    A ``kind`` in either table can carry keywords, so a caller who names both the kind and one of its
    keywords has said one thing twice and disagreed with themselves. Splatting both reached Python's own
    collision — ``RasterMixin.contours() got multiple values for keyword argument 'filled'`` — which names a
    private mixin the caller never wrote and never says that ``kind="contourf"`` *is* ``filled=True``
    (review R2-L4). Shared by the two tiers that take a ``kind``, because both tables carry keywords now and
    a refusal written on one of them would have left the other leaking the mixin.

    Args:
        table: The tier's ``{kind: (method, keywords)}`` table.
        kind: The renderer the caller named.
        kwargs: The caller's remaining keywords, read but not consumed.
        caller: The public function to blame, e.g. ``"quickmap"``.

    Returns:
        ``(method, keywords)`` — the method to call and the keywords the kind implies, exactly as the table
        holds them. A kind the table does not list is the method's own name with no keywords.

    Raises:
        ValueError: when the caller passed a keyword the kind already settles. The message names the kind, what
            it draws with and what was asked instead, and — where the table holds a kind that means what they
            asked for — that kind, since the contradiction is usually a caller who wanted the other render.
    """
    method, picked = table.get(kind, (kind, {}))
    clashing = sorted(set(picked) & set(kwargs))
    if not clashing:
        return method, picked
    instead = sorted(
        spelling
        for spelling, (spelled, keywords) in table.items()
        if spelled == method
        and all(keywords.get(name) == kwargs[name] for name in clashing)
    )
    asked = ", ".join(f"{name}={kwargs[name]!r}" for name in clashing)
    implied = ", ".join(f"{name}={picked[name]!r}" for name in clashing)
    dropped = ", ".join(f"{name}=" for name in clashing)
    remedy = (
        f"ask for kind={instead[0]!r}"
        if instead
        else f"name the kind that draws {asked}"
    )
    raise ValueError(
        f"{caller} got kind={kind!r} with {asked}, which contradicts it: kind={kind!r} is itself "
        f"{implied}. Drop {dropped} and let the kind say it, or {remedy}"
    )


def _input_kind(data: PlottableData, caller: str) -> str:
    """Classify a ``quickmap`` input into the base layer kind its geometry draws — the decision, made once.

    This is the single input-type→kind decision every backend routes through (DE-24). The ``isinstance``
    ladder, the empty-collection guard and the geometry check used to be re-derived once per backend — four
    hand-written ladders that could drift apart; here they are one. Each family returned is a data family in
    the shared kind registry's vocabulary (:data:`digitalearth.base.registry.KIND_TAKES`): a ``Dataset`` is
    ``"raster"``, and a uniform ``FeatureCollection`` is ``"points"``, ``"lines"`` or ``"polygons"``.

    A collection whose geometries disagree is ``"mixed"``. No tier draws one uniformly — the 2-D tiers drew
    every non-polygon vector as a marker map, and the 3-D tier refuses anything that is not uniformly point,
    polygon or raster — so ``"mixed"`` (and ``"lines"``) is *named* here and answered by each backend rather
    than guessed once for all of them.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (vector), or an unsupported type.
        caller: Public function name to blame in a refusal — ``"quickmap"`` or ``"quickplot"``.

    Returns:
        ``"raster"`` for a ``Dataset``; ``"points"``, ``"lines"`` or ``"polygons"`` for a uniform
        ``FeatureCollection``; ``"mixed"`` for one whose geometry types disagree.

    Raises:
        ValueError: when a ``FeatureCollection`` is empty. Without this guard the all-``True`` result of an
            empty ``geom_type`` check reads as polygons and the map draws nothing, silently.
        TypeError: when ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    if isinstance(data, Dataset):
        return "raster"
    if isinstance(data, FeatureCollection):
        if len(data) == 0:
            raise ValueError(
                f"{caller} got an empty FeatureCollection (nothing to draw)"
            )
        geometry = data.geometry.geom_type
        if geometry.isin(["Polygon", "MultiPolygon"]).all():
            return "polygons"
        if geometry.isin(["Point", "MultiPoint"]).all():
            return "points"
        if geometry.isin(["LineString", "MultiLineString"]).all():
            return "lines"
        return "mixed"
    raise TypeError(f"{caller} cannot draw a {type(data).__name__}")


def _draw_polygons(scene: Any, data: FeatureCollection, kwargs: dict) -> None:
    """Draw a polygon collection as a ``choropleth`` when a ``column`` selects one, else as outlines.

    The one branch the three 2-D tiers share once :func:`_input_kind` has said the input is polygons: a
    ``column`` picks ``choropleth``, its absence ``polygons``. Consumed in place from ``kwargs`` so the
    remaining styling keywords forward unchanged. The 3-D tier extrudes polygons instead, so it keeps its
    own polygon draw rather than this one.

    Args:
        scene: The map (static, interactive or web) being built.
        data: The polygon ``FeatureCollection`` to draw.
        kwargs: The caller's remaining keywords; ``column`` is popped here, the rest are forwarded.
    """
    column = kwargs.pop("column", None)
    if column is not None:
        scene.choropleth(data, column=column, **kwargs)
    else:
        scene.polygons(data, **kwargs)


def _draw(scene: Map, data: PlottableData, kind: str, **kwargs) -> None:
    """Draw ``data`` on ``scene`` using the renderer implied by its type and ``kind``.

    Dispatch is by input type: a ``FeatureCollection`` of polygons becomes a ``choropleth`` (when a
    ``column`` kwarg is given) or outline ``polygons``; any other geometry becomes ``points``; a
    ``Dataset`` is rendered with the method the ``kind`` names (``field`` for ``"auto"``). An empty
    ``FeatureCollection`` is rejected up front — without the guard its all-``True`` empty ``geom_type``
    check would misclassify it as polygons and silently draw nothing.

    Args:
        scene: The :class:`Map` to draw on.
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (vector).
        kind: Raster renderer name (``"auto"`` → ``field``); ignored for vector input.
        **kwargs: Forwarded to the chosen ``Map`` draw method (e.g. ``column``, ``cmap``, ``levels``).

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection`` (nothing to draw); if ``column`` was
            given for point input — it names a polygon fill and ``Map.points`` has no such parameter, so
            forwarding it produced an opaque cleopatra error instead of naming the keyword (review M19); or,
            from :func:`_renderer_for`, if a keyword contradicts the ``kind`` that already settles it.
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    family = _input_kind(data, "quickmap")
    if family == "raster":
        method, picked = _renderer_for(_STATIC_RASTER_KINDS, kind, kwargs, "quickmap")
        getattr(scene, method)(data, **picked, **kwargs)
        return
    if family == "polygons":
        _draw_polygons(scene, data, kwargs)
        return
    # points / lines / mixed → a marker map, as this tier has always drawn every non-polygon vector.
    if "column" in kwargs:
        # `Map.points` has no fill column: it sizes markers by `size_column` and colours them from
        # the collection's own value column. Forwarding `column` reached cleopatra, which answered
        # with its own keyword list and never named the caller's parameter (review M19).
        raise ValueError(
            f"column={kwargs['column']!r} fills polygons and has no meaning for point input; "
            "size the markers with size_column=, or drop column="
        )
    scene.points(data, **kwargs)


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
        crs: Display CRS for the map (`backend="matplotlib"`/`"interactive"`, where it defaults to `3857`, and
            `"web"`, which accepts only `4326`). With `backend="3d"` the scene is drawn in it; left out, the 3-D
            scene takes the data's own CRS.
        kind: Renderer for raster input (``"auto"`` → ``imshow``, which the ``field`` method draws; or
            ``contourf``/``contour``/``pcolormesh``).
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
        colorbar: Whether the map carries a colour key for the drawn layer. ``True`` is the default and
            adds one **if there is one to draw**; ``False`` draws none. Every backend accepts it, so the
            same call is valid everywhere (#254), but each reaches its own mechanism: ``matplotlib`` builds
            a colorbar, ``interactive`` toggles one, ``3d`` shows or hides the scalar bar, and ``web``
            builds :meth:`~digitalearth.web.decoration.DecorationMixin.legend`.

            ``False`` is the half that is genuinely uniform: it reliably means "no key" on all four.
            ``True`` is not one behaviour but two. On ``matplotlib`` and ``web`` it **actively builds** a
            key; on ``interactive`` and ``3d`` it is **passive** — it leaves the builder's or engine's own
            default in place, which for PyVista means a scalar bar iff the layer carries scalars. Only
            ``False`` acts on those two.

            What "nothing to draw" means also differs, and both active tiers skip **silently** rather than
            warning. ``matplotlib`` builds a colorbar for almost any mappable layer, and skips when the last
            layer has no mappable or is categorical. ``web`` keys only a **classification** — what a builder
            recorded in :attr:`~digitalearth.web.base.WebMapBase.last_legend` — so an unclassified layer gets
            nothing. The split is not raster-vs-vector: the same unclassified polygon layer gets a colorbar
            on ``matplotlib`` and no key on ``web``, and every raster falls on the empty side there because
            ``field`` records no classification. Giving the web tier a continuous ramp key for a raster
            is tier work, not part of this argument's contract. Neither active tier logs when it skips: the
            ``matplotlib`` path warns only if its builder refuses, and the ``web`` path lets a refusal
            surface, since past the guard only a malformed classification can raise.

            The *tier methods* also still differ in shape — a builder that takes content on
            ``matplotlib``/``web``, a visibility flag on ``interactive`` — and unifying those names is
            Core-contract work (U-3).
        backend: ``"matplotlib"`` (default) returns a static :class:`Map`; ``"interactive"`` returns a
            pan/zoom :class:`~digitalearth.interactive.map.InteractiveMap` (needs the ``interactive``
            extra); ``"3d"`` returns a :class:`~digitalearth.three_d.scene3d.Scene3D` (needs the ``3d``
            extra); ``"web"`` returns a :class:`~digitalearth.web.map.WebMap` (MapLibre + deck.gl; needs the
            ``web`` extra) — all fed from the same input-type dispatch. Which of the arguments above each one
            honours is :data:`BACKEND_CAPABILITIES`; anything else you pass it is refused, not dropped.
        **kwargs: Forwarded to the underlying draw method (e.g. ``cmap``, ``levels``, ``column``;
            ``z_exaggeration``/``height``/``size`` for ``backend="3d"``). ``column`` names a **polygon**
            fill on ``backend="matplotlib"``: point input there is a ``Map.points``, which sizes markers
            by ``size_column`` instead, so ``column`` on points is refused by name rather than forwarded.

    Returns:
        The decorated :class:`Map`, an :class:`InteractiveMap` when ``backend="interactive"``, a
        :class:`Scene3D` when ``backend="3d"``, or a :class:`WebMap` when ``backend="web"``.

    Raises:
        CapabilityError: for a ``crs``/``domain``/``basemap``/``coastlines``/``kind`` the chosen backend
            cannot honour — the message names both the parameter and the backend, and adds the reason that
            tier's own declaration gave. It subclasses ``ValueError``, so existing handlers still catch it.
        ValueError: for an unknown ``backend``, for ``column`` on point input, for an empty
            ``FeatureCollection``, which would otherwise draw nothing in silence, and for a keyword that
            contradicts the ``kind`` it was written beside — ``kind="contourf"`` *is* ``filled=True``, so
            naming both says one thing twice and disagrees. ``colorbar`` is never refused, since every
            backend honours it.
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection`` — and, on
            ``backend="3d"``, for a line ``FeatureCollection`` too, which has no 3-D builder.

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
            ...     quickmap(ds, backend="3d", domain="europe")
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            domain= is not supported by backend='3d'

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
        return _quickmap_3d(
            data, colorbar=colorbar, crs=None if crs is _UNSET else crs, **kwargs
        )
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
        return _quickmap_web(
            data, crs=crs, basemap=basemap, colorbar=colorbar, **kwargs
        )
    return _quickmap_matplotlib(
        data,
        crs=3857 if crs is _UNSET else crs,
        kind=kind,
        domain=domain,
        basemap=basemap,
        coastlines=coastlines,
        colorbar=colorbar,
        **kwargs,
    )


def to_backend(
    figure: "FigureSpec", backend: str = "matplotlib", **scene_kwargs: Any
) -> Any:
    """Render an engine-neutral :class:`~digitalearth.base.spec.FigureSpec` on a chosen ``backend`` (U-6).

    The inverse of every tier's :attr:`figure_spec`: where that describes a built scene *as data*, this
    replays the description onto a fresh scene of the named ``backend`` by handing the whole figure to that
    tier's own ``from_figure``. So nothing here renders — it only dispatches, reusing the same lazy backend
    seam ``quickmap`` does (DE-24), so importing ``api`` still loads no renderer until one is asked for.

    **Every backend renders a figure it itself described** — the round trip the seam exists for: a scene's
    :attr:`figure_spec`, optionally through ``to_dict()``/``from_dict()``, drawn again on the same backend,
    carrying kinds, sources, draw order, visibility and the portable half of each layer's colour encoding.

    **Cross-tier — a figure one tier described drawn on *another* — is supported into the interactive tier**,
    whose drawers fall back to their own defaults for the style a portable figure does not carry; the foreign
    recipe name (``via``) is retargeted to the drawing tier's own. The other three tiers refuse a foreign
    figure and are **same-tier only**, for two distinct reasons: the **matplotlib** and **3-D** tiers keep a
    layer's style flat in ``props`` their builders write and fold nothing into portable channels (the
    ``NO_PORTABLE_CHANNELS`` set, which names exactly those two), and the **web** tier's drawers require
    per-layer style ``props`` a portable figure does not carry (e.g. a raster's ``vmin``/``vmax``/``opacity``).
    Drawing a *foreign* figure on any of the three may raise from inside a drawer. A figure with an ``object:``
    (in-memory) source replays in-process but cannot be stored (``to_dict`` refuses it).

    Args:
        figure: The :class:`~digitalearth.base.spec.FigureSpec` to draw, from another scene's
            :attr:`figure_spec` (optionally through ``to_dict()``/``from_dict()``).
        backend: ``"matplotlib"`` (static, the default), ``"interactive"`` (HoloViz), ``"web"``
            (MapLibre + deck.gl) or ``"3d"`` (PyVista). Validated against :data:`BACKEND_CAPABILITIES`.
        **scene_kwargs: Forwarded to the tier's constructor (e.g. ``strict``, ``figsize``, ``off_screen``).
            A ``crs=`` here overrides the one the figure's view carries on the static, interactive and 3-D
            tiers; the web tier renders in EPSG:4326 only, so a ``crs=`` other than that is refused, not
            applied.

    Returns:
        The built tier scene (``Map`` / ``InteractiveMap`` / ``WebMap`` / ``Scene3D``), the same return
        contract as :func:`quickmap`.

    Raises:
        ValueError: if ``backend`` is not one of the four, or if ``figure`` has more than one panel (each
            tier renders a single panel, so a multi-panel figure is refused rather than silently flattened).
        TypeError: if ``figure`` is not a :class:`~digitalearth.base.spec.FigureSpec`.

    Examples:
        - An unknown backend is refused by name, before anything is built:
            ```python
            >>> from digitalearth.api import to_backend
            >>> to_backend("anything", backend="nope")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: unknown backend 'nope'; choose 'matplotlib' (static), ...

            ```

        A round-trip example — build a scene, store it, draw it again — lives on each tier's ``from_figure``
        (e.g. :meth:`digitalearth.three_d.Scene3D.from_figure`).
    """
    if backend not in BACKEND_CAPABILITIES:
        raise ValueError(
            f"unknown backend {backend!r}; choose 'matplotlib' (static), 'interactive' (HoloViz), "
            f"'3d' (PyVista), or 'web' (MapLibre + deck.gl)"
        )
    from digitalearth.base.spec import FigureSpec

    if not isinstance(figure, FigureSpec):
        # Caught here, not deep inside a tier's from_figure, where a non-figure fails with an opaque
        # `AttributeError` on `.panels` that names neither the argument nor the type expected.
        raise TypeError(
            f"to_backend draws a FigureSpec; got {type(figure).__name__}. Pass a scene's `figure_spec` "
            f"(optionally through to_dict()/from_dict())."
        )
    if len(figure.panels) != 1:
        # Each of the four tiers renders one panel, and a tier's `from_figure` reads only `panels[0]` for the
        # view while drawing the whole layer tree — so a valid multi-panel figure (panels in different CRSs,
        # each showing a subset) would be silently flattened into panel 0's view. Refuse it by name rather
        # than draw the wrong picture; a caller splits a `grid()`-composed figure into one per panel.
        raise ValueError(
            f"to_backend draws a single-panel figure; got {len(figure.panels)} panels. Each tier renders "
            f"one panel, so split a multi-panel figure (e.g. from grid()) into one figure per panel."
        )
    if backend == "3d":
        from digitalearth.three_d import Scene3D

        return Scene3D.from_figure(figure, **scene_kwargs)
    if backend == "interactive":
        from digitalearth.interactive import InteractiveMap

        return InteractiveMap.from_figure(figure, **scene_kwargs)
    if backend == "web":
        from digitalearth.web import WebMap

        return WebMap.from_figure(figure, **scene_kwargs)
    from digitalearth.static import Map

    return Map.from_figure(figure, **scene_kwargs)


def _quickmap_matplotlib(
    data: PlottableData,
    *,
    crs: Any,
    kind: str,
    domain: Any,
    basemap: bool | str | Any,
    coastlines: bool,
    colorbar: bool,
    **kwargs,
) -> Map:
    """Build a finished :class:`~digitalearth.static.map.Map` from ``data`` (the default backend's path).

    The fourth of the per-tier builders, so `quickmap` itself is one dispatch table rather than a dispatch
    table with one tier's assembly inlined after it. Nothing here is new: it is the steps the default path
    always took, in the order it took them.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (vector).
        crs: Display CRS for the map.
        kind: Raster renderer (``"auto"`` → ``imshow``, drawn by ``Map.field``).
        domain: A named region / bbox to frame on, or `None` to leave the extent to the data.
        basemap: ``True`` for the backend's default tile source, or the source itself.
        coastlines: When True, overlay coastlines.
        colorbar: Whether the map carries a colour key, where the drawn layer has one to carry. ``False``
            records the key switched off and takes off the swatch legend a categorical fill's glyph drew of
            its own accord, rather than leaving a key the caller asked against — the same thing the
            interactive and 3-D paths mean by it.
        **kwargs: Forwarded to the chosen builder (e.g. ``cmap``, ``column``).

    Returns:
        The decorated map.
    """
    from digitalearth.static import Map

    scene = Map(crs=crs, domain=domain)
    _draw(scene, data, kind, **kwargs)
    if coastlines:
        _best_effort("coastlines", scene.coastlines)
    if basemap:
        # `True` means "the backend's default source"; anything else names the source and is forwarded, so a
        # keyed preset reaches Map.basemap instead of being silently reduced to the default.
        _best_effort("basemap", scene.basemap, _basemap_source(basemap))
    if domain is not None:
        scene.set_domain()
    if _has_a_key_to_draw(scene):
        # Asked either way, because `colorbar=False` is a decision to record and a key to take off rather
        # than a call to skip. A map with nothing coloured by a value has no key to switch, so the gate
        # above still stands between the two. Reached through the one seam the four backends share.
        _add_key("matplotlib", scene, visible=colorbar)
    return scene


def _has_a_key_to_draw(scene: Map) -> bool:
    """Whether any layer on the map is one a colour key can describe.

    Asked of the **description** rather than of the last registered artist (order 24). The old reading was
    ``scene.layers[-1]``, which is the last layer that registered a mappable — so a coastline or a basemap
    drawn after the data answered for the data, and ``quickmap(ds, coastlines=True)`` decided whether to key
    a raster by looking at a coastline.

    It used to ask :func:`~digitalearth.static.guides.bar_refusal` — "can a **bar** describe this?" — and so
    answered `False` for a categorical fill, which is a layer with a key, just not that one. Choosing
    between the two is :func:`_add_static_key`'s job; the question here is only whether there is anything to
    explain (review M6).

    Args:
        scene: The map that was just drawn.

    Returns:
        `False` for a map where nothing is coloured by a value — no layers, a globe fill, an outline-only
        polygon draw.
    """
    return bool(scene._color_keyed())


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
    """Alias of :func:`quickmap` — build a finished map from ``data`` in one call.

    Args:
        data: A pyramids ``Dataset`` or ``FeatureCollection`` to draw.
        **kwargs: Forwarded to :func:`quickmap` unchanged — ``crs``, ``domain``, ``basemap``,
            ``coastlines``, ``colorbar``, ``kind``, ``backend`` and the styling kwargs the chosen
            builder takes.

    Returns:
        Whatever :func:`quickmap` built for the chosen backend: a :class:`Map` by default, else an
        ``InteractiveMap``, a ``Scene3D`` or a ``WebMap``.

    Raises:
        CapabilityError: as :func:`quickmap` raises it, for an argument the chosen backend cannot honour.
        ValueError: as :func:`quickmap` raises it, for an unknown ``backend``, an empty collection, or a
            keyword that contradicts the ``kind`` it was written beside.
        TypeError: as :func:`quickmap` raises it, for input that is neither a ``Dataset`` nor a
            ``FeatureCollection``.
    """
    return quickmap(data, **kwargs)


#: The interactive tier's renderer for each ``kind`` `quickmap` accepts, as ``{kind: (method, keywords)}``.
#: A kind absent from this table is refused by name rather than drawn as something the caller did not ask
#: for.
#:
#: The same shape as :data:`_STATIC_RASTER_KINDS`, and for the reason that table gives: a ``kind`` names a
#: **renderer**, and telling ``contour`` from ``contourf`` is an *argument* to one method rather than a
#: second method's name. Naming a method alone was not enough to say it, so ``contourf`` reached a second
#: spelling this tier used to carry, which took no ``interval=`` — and ``interval=`` then fell through
#: ``**opts`` to HoloViews, which refused it as an unknown style option (review H3). That second spelling has
#: since been deleted, so there is one ``contours`` here as well. The tables stay separate because the tiers
#: really do spell their methods differently; what they now share is the ability to carry the keyword that
#: picks the render.
_INTERACTIVE_RASTER_KINDS = {
    "auto": ("field", {}),
    "imshow": ("field", {}),
    "contour": ("contours", {"filled": False}),
    "contourf": ("contours", {"filled": True}),
    "pcolormesh": ("quadmesh", {}),
}


def _draw_interactive_raster(
    scene: Any, data: Dataset, kind: str, kwargs: dict
) -> None:
    """Draw a raster with the interactive renderer `kind` names.

    Args:
        scene: The ``InteractiveMap`` being built.
        data: The raster to draw.
        kind: The renderer's `quickmap` spelling.
        kwargs: The caller's remaining keywords.

    Raises:
        ValueError: if `kind` names a renderer this tier does not have. It used to fall back to the
            tier's own default raster builder — `field` — and draw something the caller never asked
            for; the matplotlib backend has always refused it (review M7). And, from
            :func:`_renderer_for`, if a keyword contradicts the ``kind`` that already settles it.
    """
    if kind not in _INTERACTIVE_RASTER_KINDS:
        renderers = ", ".join(repr(name) for name in sorted(_INTERACTIVE_RASTER_KINDS))
        raise ValueError(
            f"kind={kind!r} is not a renderer of backend='interactive'; use one of {renderers}"
        )
    method, picked = _renderer_for(_INTERACTIVE_RASTER_KINDS, kind, kwargs, "quickmap")
    getattr(scene, method)(data, **picked, **kwargs)


def _decorate_interactive(scene: Any, basemap: Any, coastlines: bool) -> None:
    """Add the decoration `quickmap` asks for, in the order the tier draws it.

    Args:
        scene: The ``InteractiveMap`` being built.
        basemap: `True` for the tier default, or the source itself.
        coastlines: Whether to overlay a coastline.
    """
    if basemap:
        source = _basemap_source(basemap)
        if source is None:
            scene.tiles()
        else:
            scene.tiles(source)
    if coastlines:
        scene.coastlines()


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

    Dispatches by input type as :func:`_draw` does, with **one deliberate difference in how point input
    treats** ``column``: a polygon ``FeatureCollection`` with a ``column`` becomes a ``choropleth`` (else
    ``polygons``); other vectors become ``points``; a raster is drawn with the ``kind`` method
    (``"auto"`` → ``field``). Unlike the matplotlib path, ``column`` on **point** input is **not refused**
    here — it is forwarded, because ``InteractiveMap.points`` takes a ``column`` and colours the markers by
    it, whereas ``Map.points`` has no such parameter (it sizes by ``size_column``), which is why
    :func:`_draw` raises for it (review L3). The ``InteractiveMap`` import is lazy so the core ``api`` works
    without the ``interactive`` extra.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (vector).
        crs: Display CRS for the interactive map.
        kind: Raster renderer (``"auto"`` → ``imshow``, drawn by ``InteractiveMap.field``; or
            ``contourf``/``contour``/``pcolormesh``).
        basemap: ``True`` for the backend's default tile source, or the source itself (provider name or
            keyed preset), forwarded to ``InteractiveMap.tiles``.
        coastlines: When True, overlay a coastline.
        colorbar: Whether the map carries a colour key, where the drawn layer has one to carry. Routed
            through the tier's own ``colorbar``/``legend`` (:func:`_add_interactive_key`) rather than left
            to the builder's option, so the key is recorded on the layer it explains either way — and so a
            categorical fill, which Bokeh draws no colorbar for, is keyed by its legend instead of not at
            all.
        **kwargs: Forwarded to the chosen builder (e.g. ``cmap``, ``column``).

    Returns:
        The decorated :class:`~digitalearth.interactive.map.InteractiveMap`.

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection``, or if ``kind`` names a renderer this
            tier does not have — it is refused by name rather than quietly drawn with ``field``.
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    from digitalearth.interactive import InteractiveMap

    scene = InteractiveMap(crs=crs)
    family = _input_kind(data, "quickplot")
    if family == "raster":
        _draw_interactive_raster(scene, data, kind, kwargs)
    elif family == "polygons":
        _draw_polygons(scene, data, kwargs)
    else:  # points / lines / mixed → a marker map, as this tier draws every non-polygon vector
        scene.points(data, **kwargs)
    _add_key("interactive", scene, visible=colorbar)
    _decorate_interactive(scene, basemap, coastlines)
    return scene


def _add_interactive_key(scene: Any, *, visible: bool) -> None:
    """Ask the interactive tier for the colour key of whatever it just drew, tolerating a layer with none.

    The HoloViz counterpart of :func:`_add_static_key`, and here for the same reason. This path used to act
    only on ``colorbar=False``, leaving the builder's own option in place for ``True`` — so a one-call map
    carried no :class:`~digitalearth.base.spec.encoding.Guide` at all, and a **categorical** fill, whose
    builder sets ``colorbar: False`` for itself, came back with no key in the picture either. That is review
    M6 on this tier.

    **Which key it is, is the layer's property, and the categorical answer is measured rather than assumed.**
    A categorical fill is coloured through a ``{label: colour}`` map, which Bokeh reads as a
    `CategoricalColorMapper` and draws no ``ColorBar`` from at all: rendering one with ``colorbar=True``
    produces zero bars, while ``legend()`` produces one keyed box. So the builder's ``colorbar: False`` is
    load-bearing and stays; what changes is that the key asked for is the one the scale calls for, which is
    :func:`~digitalearth.interactive.style_fold.guide_kind`'s question.

    Args:
        scene: The `InteractiveMap` whose most recent colour-driven layer the key describes.
        visible: Whether the key is drawn. ``False`` records the guide switched **off** — under the kind the
            layer actually calls for, so switching it back on asks Bokeh for furniture it can build.

    Returns:
        Nothing. Both methods return the map itself, as the Core declares.
    """
    from digitalearth.interactive.style_fold import guide_kind

    # A layer whose colour varies with nothing — plain `points`, an outline-only `polygons`, a map with no
    # data layer at all — has no key to speak about, and `colorbar=False` asking for a map without one is
    # met by a map that already has none. Refusing the whole `quickplot` over that would be review L9 again.
    if not scene._guidable():
        return
    target = scene._guided_layer("quickplot", None)
    kind = guide_kind(scene.get_layer(target).symbology)
    ask = scene.legend if kind == "legend" else scene.colorbar
    try:
        ask(target, visible=visible)
    except UNMAPPABLE as error:
        logger.warning(
            "quickplot: %s skipped — %s: %s", kind, type(error).__name__, error
        )


def _add_web_legend(scene: Any, *, visible: bool) -> Any:
    """Add the web tier's colour key, tolerating a map with nothing classified to describe.

    The web counterpart of :func:`_add_static_key`. ``WebMap.legend`` is a *builder* — it reads the
    classification the last layer recorded — so on a map with no classified layer it raises rather than
    drawing an empty box. Under ``quickmap(colorbar=True)`` that is not a caller error: the default asks for
    a key *if there is one to draw*, exactly as the matplotlib path treats an unmappable layer.

    **The flag is carried through rather than read as "skip the call"**, which is the rule the other three
    backends already follow. This path gated the whole call on it, so ``quickmap(colorbar=False)`` left the
    layer carrying no :class:`~digitalearth.base.spec.encoding.Guide` at all while the matplotlib,
    interactive and 3-D paths each recorded ``Guide(show=False)`` — and a web figure written out after
    ``colorbar=False`` was indistinguishable from one where nobody asked (review M1). The picture agrees
    either way; the *record* is what travels through ``FigureSpec``, and recording the decision is what lets
    a switcher offer the key back.

    The layer is resolved through the tier's own ``_guide_target`` with ``visible=True``, for both values of
    the flag, and named in the call — the shape :func:`_add_interactive_key` already has. That resolution
    answers "which layer would a key describe"; the flag answers the different question of whether the key
    is drawn. Leaving it to ``legend(visible=False)`` to resolve instead asks for "the key this map already
    has", which on a one-call map is nothing at all, and the decision goes unrecorded again.

    Args:
        scene: The ``WebMap`` whose most recent classified layer should get a key.
        visible: Whether the key is drawn. ``False`` records the guide switched **off**, as
            ``quickmap(colorbar=False)`` asks for.

    Returns:
        The same map, or ``None`` when there was no classification to describe — including a **stale**
        ``last_legend``, which is a sticky side record: a map whose classified layer has since been removed
        or restyled to a constant colour still carries one while nothing on it can be keyed. Naming the
        layer moves that resolution out of ``legend()`` and in here, so its "nothing to describe" refusal is
        tolerated here as well; otherwise ``quickmap`` would start failing on a map it used to return
        unkeyed, which is the same tolerance the matplotlib path gives an unmappable artist.

    Raises:
        AttributeError: if ``scene`` carries no ``last_legend`` — that is a defect, not an unkeyed map.
        Exception: whatever ``legend()`` raises for a malformed classification, which is a web-tier bug and
            must surface rather than be reported as "no key to draw".
    """
    # Answered from recorded state before the engine is touched. `legend()` is a builder: it refuses a map
    # with nothing classified to describe, and reaching that refusal first calls `_require_maplibre()`. Under
    # the `colorbar=True` default neither is a caller error -- they asked for a key *if there is one* -- so a
    # map with no classification must not be the reason an ImportError surfaces.
    # Read straight off the map, not through getattr: `WebMapBase.__init__` always sets this, so a scene
    # without it is a defect rather than a map with nothing to key, and a rename must fail here loudly
    # instead of silently dropping every web key.
    if not scene.last_legend:
        return None
    try:
        target = scene._guide_target(None, visible=True, caller="quickmap()")
    except ValueError as error:
        # The same "no mappable layer" tolerance one line up, for the case `last_legend` cannot answer.
        # It is a *sticky* side record: a map whose classified layer has since been removed, or restyled to
        # a constant colour, still carries one while nothing on it can be keyed any more. Naming the layer
        # is what records the decision either way, and it moves this resolution out of `legend()` and in
        # here -- so its refusal has to be tolerated here too, or `quickmap` would start failing on a map
        # it used to return unkeyed.
        logger.warning(
            "quickmap: web key skipped -- %s: %s", type(error).__name__, error
        )
        return None
    # No `except` around the builder. The guards above leave only a malformed `last_legend` able to raise --
    # a bug in a web builder -- and swallowing that would report a library defect as "no key to draw".
    return scene.legend(layer_id=target, visible=visible)


def _quickmap_web(
    data: PlottableData,
    *,
    crs: Any = _UNSET,
    basemap: bool | str | Any = False,
    colorbar: bool = True,
    **kwargs,
) -> Any:
    """Build a finished ``WebMap`` from ``data`` (the ``backend="web"`` path, DX.1).

    Dispatches by input type as :func:`_draw` does, with **one deliberate difference in how point input
    treats** ``column``: a polygon ``FeatureCollection`` with a ``column`` becomes a ``choropleth`` (else
    outline ``polygons``); other vectors become ``points``; a raster ``Dataset`` becomes ``field``. Unlike
    the matplotlib path, ``column`` on **point** input is **not refused** here — it is forwarded, because
    ``WebMap.points`` takes a ``column`` and colours the markers by it, whereas ``Map.points`` has no such
    parameter (it sizes by ``size_column``), which is why :func:`_draw` raises for it (review L3). The
    ``WebMap`` import is lazy so the core ``api`` works without the
    ``web`` extra. The web tier normalises data to lon/lat itself, so the matplotlib-oriented ``kind`` /
    ``domain`` / ``coastlines`` kwargs have no counterpart here and :func:`_reject_unsupported` refuses them
    upstream. ``crs`` **is** forwarded: ``WebMap`` carries one and validates it, so a CRS it cannot place
    inline is reported by the tier that knows why, rather than dropped here.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (points/polygons).
        crs: Display CRS, forwarded to ``WebMap``; :data:`_UNSET` leaves the web tier's own default.
        basemap: ``True`` for the web tier's default dark tile basemap, or the source itself (provider name
            or keyed preset), forwarded to ``WebMap.basemap``.
        colorbar: Whether this tier's colour key is drawn, carried through :func:`_add_web_legend` rather
            than read as "skip the call": ``True`` (the default) draws it and ``False`` records the guide
            switched **off**, so the figure says either way that the key was decided. The helper checks
            before it builds rather than catching afterwards, so a map with no classified layer to describe
            gets no key and no error, while a failure inside the builder still surfaces.
        **kwargs: Forwarded to the chosen ``WebMap`` builder (e.g. ``cmap``, ``column``, ``scheme``, ``k``).

    Returns:
        The built :class:`~digitalearth.web.map.WebMap`.

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection``, or if ``WebMap`` refuses ``crs``.
        TypeError: if ``data`` is neither a ``Dataset`` nor a ``FeatureCollection``.
    """
    from digitalearth.web import WebMap

    scene = WebMap() if crs is _UNSET else WebMap(crs=crs)
    family = _input_kind(data, "quickplot")
    if family == "raster":
        scene.field(data, **kwargs)
    elif family == "polygons":
        _draw_polygons(scene, data, kwargs)
    else:  # points / lines / mixed → a marker map
        scene.points(data, **kwargs)
    if basemap:
        source = _basemap_source(basemap)
        if source is None:
            scene.basemap()
        else:
            scene.basemap(source)
    # Called whichever way the flag is set, so the decision is recorded either way -- the rule the other
    # three backends follow. This tier's key is a builder, not a toggle, and it refuses a map with nothing
    # classified to describe; that is this tier's "no mappable layer" case, which the matplotlib path
    # tolerates too, so it is tolerated inside the helper rather than turning `colorbar=True` into an error
    # the caller did not cause. Reached through the one seam the four backends share.
    _add_key("web", scene, visible=colorbar)
    return scene


def _add_3d_key(scene: Any, *, visible: bool) -> Any:
    """Ask the 3-D tier for the colour key of whatever it just drew, tolerating a layer with none.

    The 3-D counterpart of :func:`_add_web_legend`, and the reason this path no longer reaches past the tier
    into a PyVista keyword: `Scene3D` has the methods now (order 24), and a `quickmap` flag routes through
    them. Which of the two is a property of the layer rather than of the flag, under the rule stated in full
    at :func:`~digitalearth.static.guides.guide_kind`: a scale is explained by what it is made of, and a
    **classified** scale means its classes. Here that is the keyed list, because this tier paints a
    classified fill by class **index** — `classified_scalars` assigns `0, 1, 2 …` — so a scalar bar over it
    would read those indices where the class ranges belong. The matplotlib and interactive tiers answer the
    same case with a bar, and it is the same list: matplotlib's `BoundaryNorm` bands the strip and ticks it
    on the very edges this tier's list spells out (review M2, which found the two docstrings stating two
    different rules). A **continuous** scale gets PyVista's scalar bar, which reads the values themselves.

    Args:
        scene: The `Scene3D` whose most recent colour-driven layer should get a key.
        visible: Whether the key is drawn. `False` takes off the bar PyVista draws of its own accord, which is
            what ``colorbar=False`` has always meant here.

    Returns:
        The same scene, or `None` when nothing it drew is coloured by data — a flat-coloured surface, or a
        point cloud given no values. Under the `colorbar=True` default that is not a caller error: they asked
        for a key *if there is one to draw*, which is how the matplotlib and web paths treat it too.

    Raises:
        Exception: whatever the tier's own methods raise for a layer that is coloured by data but cannot be
            keyed, which would be a 3-D-tier defect rather than an unkeyed scene.
    """
    for layer_id in reversed(scene.layer_ids):
        encoding = scene.get_layer(layer_id).symbology.encoding("color")
        if encoding is None or encoding.field is None:
            continue
        scale = encoding.scale
        if scale is not None and (scale.is_classified or scale.is_categorical):
            return scene.legend(layer_id, visible=visible)
        return scene.colorbar(layer_id, visible=visible)
    return None


#: Each backend's colour-key adder — the one seam the four `_quickmap_*` builders reach a key through,
#: rather than four call sites naming four different functions. Each impl stays per-tier because each
#: engine's key is a different thing: a matplotlib colorbar or swatch legend, a Bokeh toggle, a MapLibre
#: legend builder, a PyVista scalar bar. What they share is the point they are called from and the
#: `visible=` flag carried through from `quickmap(colorbar=)` (DE-24).
_KEY_ADDERS: dict[str, Callable[..., Any]] = {
    "matplotlib": _add_static_key,
    "interactive": _add_interactive_key,
    "web": _add_web_legend,
    "3d": _add_3d_key,
}


def _add_key(backend: str, scene: Any, *, visible: bool) -> None:
    """Add ``scene``'s colour key through ``backend``'s own key impl — the single seam for all four tiers.

    The four impls differ (each engine's key is its own thing) and each guards internally against a map with
    nothing to key; this is the one place they are dispatched from, so a fifth backend or a changed key
    contract is wired here once rather than at four call sites (DE-24). The impl's own return value — some
    return the scene, some ``None`` — is discarded: every ``_quickmap_*`` builder returns its scene itself.

    Args:
        backend: The backend whose key impl to use — a key of :data:`_KEY_ADDERS`.
        scene: The finished map to key.
        visible: Whether the key is drawn, carried through from ``quickmap(colorbar=)``.
    """
    # Dispatch through this module's namespace *by name* rather than through the table's stored reference, so
    # a test that spies or patches the named impl on the module — `mocker.spy(api, "_add_web_legend")`, as the
    # web tier's colorbar test does — observes the call. `_KEY_ADDERS` still holds the real callables (the
    # introspectable seam `test_api_seam` asserts identity on), and a per-backend override installed *into the
    # table* with `monkeypatch.setitem(_KEY_ADDERS, ...)` is still honoured: such an override's ``__name__`` is
    # not a global of this module, so the lookup falls back to the stored object. Re-collapsing this to
    # ``_KEY_ADDERS[backend](...)`` silently re-breaks the spy path (CI: tests/web/test_web_colorbar_raster.py).
    impl = _KEY_ADDERS[backend]
    globals().get(impl.__name__, impl)(scene, visible=visible)


def _quickmap_3d(
    data: PlottableData, *, colorbar: bool = True, crs: Any = None, **kwargs
) -> Any:
    """Build a finished ``Scene3D`` from ``data`` (the ``backend="3d"`` path, DX.1).

    Dispatches by input type, mirroring :func:`_draw`: a raster ``Dataset`` becomes 3-D relief
    (``terrain``); a point ``FeatureCollection`` becomes a ``point_cloud`` (coloured by ``column`` when
    given); a polygon ``FeatureCollection`` becomes ``extruded_polygons`` (extruded by, and coloured by,
    ``column``). The ``Scene3D`` import is lazy so the core ``api`` works without the ``3d`` extra. ``crs``
    becomes the scene's display CRS; the map-only kwargs (``kind``/``domain``/``basemap``/``coastlines``) have
    no 3-D analogue and are refused by name in :func:`quickmap` before this is reached.

    Args:
        data: A pyramids ``Dataset`` (raster) or ``FeatureCollection`` (points/polygons).
        colorbar: Whether the scene carries a colour key, routed through the tier's own
            :meth:`~digitalearth.three_d.guides.GuideMixin.colorbar` /
            :meth:`~digitalearth.three_d.guides.GuideMixin.legend` rather than through a PyVista keyword
            (order 24), so the key is recorded on the layer it explains. ``True`` (the default) keys the
            layer that was drawn, where its colour comes from data: a ramp gets PyVista's scalar bar, which
            is the bar the engine already drew, and a **classified** fill gets the keyed list instead of a
            scalar bar reading `0, 1, 2 …` over its class indices. ``False`` takes the bar off. Either way a
            layer whose colour is flat gets no key and no error, as on the matplotlib and web paths.
        crs: The scene's display CRS; ``None`` takes the data's own.
        **kwargs: Forwarded to the chosen ``Scene3D`` builder (e.g. ``cmap``, ``z_exaggeration``,
            ``column``, ``height``, ``size``).

    Returns:
        The built :class:`~digitalearth.three_d.scene3d.Scene3D`.

    Raises:
        ValueError: if ``data`` is an empty ``FeatureCollection``.
        TypeError: for a line ``FeatureCollection`` (no 3-D builder) or a non-raster/vector input.
    """
    from digitalearth.three_d import Scene3D

    # Classify the input up front, through the same decision the other three backends make (DE-24): a bad
    # type, an empty collection or a geometry this tier has no builder for must raise BEFORE a Scene3D
    # (which opens a VTK plotter) is constructed, or every error path leaks an open plotter. The families
    # this tier cannot draw — lines and mixed — are refused here, in this tier's own words.
    family = _input_kind(data, "quickplot")
    if isinstance(data, FeatureCollection) and family in ("lines", "mixed"):
        # Only a FeatureCollection yields the `lines`/`mixed` families, so this is the geometry this tier has
        # no builder for; the isinstance also narrows `data` for the geom-type read in the message.
        raise TypeError(
            "backend='3d' needs a uniformly point, polygon, or raster input "
            "(point_cloud / extruded_polygons / terrain); got geometry types "
            f"{sorted(data.geometry.geom_type.unique())}"
        )

    scene = Scene3D(
        crs=crs
    )  # constructed only after validation, so the error paths above leak no scene
    if family == "raster":
        scene.terrain(data, **kwargs)
    elif family == "polygons":
        column = kwargs.pop("column", None)
        height = kwargs.pop("height", column if column is not None else 1.0)
        scene.extruded_polygons(data, height=height, column=column, **kwargs)
    else:  # points
        scene.point_cloud(data, value_column=kwargs.pop("column", None), **kwargs)
    # After the layer, not before it as a `show_scalar_bar=` among its keywords: the key is a guide on that
    # layer's encoding, so there has to be a layer to put it on (order 24). Reached through the one seam the
    # four backends share.
    _add_key("3d", scene, visible=colorbar)
    return scene


def _finish(scene: Map, *, colorbar: bool) -> Map:
    """Add the colour key to ``scene`` when this plot kind has one and a keyed layer exists; return it.

    The shared tail of the one-call wrappers: the key is added only when ``colorbar`` is true and some layer
    publishes a colour to explain (:func:`_has_a_key_to_draw`), and an outline-only / unmappable layer is
    warned about rather than raised (see :data:`UNMAPPABLE`; anything else propagates). Which key it is —
    a bar or a swatch legend — is :func:`_add_static_key`'s question.

    ``colorbar`` here is **the wrapper's own answer to "does this kind carry a key at all"**, not a caller's
    request: every call site spells it ``True`` or ``column is not None``, and a wrapper drawn without a
    column publishes no colour encoding, so the ``False`` arm and a keyed layer never meet. That is why this
    passes ``visible=True`` rather than the flag: a caller's ``colorbar=False`` — which *is* a request, and
    one that has to switch a drawn key off — reaches :func:`_quickmap_matplotlib` instead.

    Args:
        scene: The :class:`Map` a wrapper has already drawn on.
        colorbar: Whether this plot kind carries a colour key (e.g. only when a value ``column`` was set).

    Returns:
        The same ``scene`` (so wrappers can ``return _finish(...)``).
    """
    if colorbar and _has_a_key_to_draw(scene):
        _add_static_key(scene, visible=True)
    return scene


def _method(name: str, kind: Optional[str] = None):
    """Build a module-level function that quick-draws via the ``Map`` method ``name``.

    The wrapper *is* the ``kind``: it injects one on the caller's behalf. So when the chosen
    backend has no renderer selector, :func:`_reject_unsupported`'s "drop the argument" message named a
    keyword the caller never typed (review L5). The wrapper answers for its own injection instead, naming
    itself and the call that does work.

    Its refusal is the declared one, like :func:`_reject_unsupported`'s: the tier that has no
    ``raster_renderer`` said why in its own `absent`, and that sentence is carried here rather than
    rewritten. A wrapper is the one gate a caller reaches without naming ``kind=``, so without the reason
    they were told a renderer is missing and never what the tier does instead.

    Args:
        name: The ``Map`` method the wrapper draws with, and the wrapper's own name.
        kind: The ``quickmap`` renderer it injects, where that is spelled differently from the
            method — ``field`` draws matplotlib's ``imshow`` kind, and ``contours`` draws
            ``contour`` or ``contourf`` depending on ``filled=``. ``None`` (default) means the
            two agree, which is the case for ``pcolormesh``.

    Returns:
        The module-level quick-draw function.
    """
    renderer = name if kind is None else kind

    def _fn(data: PlottableData, **kwargs) -> Map:
        backend = kwargs.get("backend", "matplotlib")
        supported = BACKEND_CAPABILITIES.get(backend)
        if supported is not None and "kind" not in supported:
            honoured = _honoured_by("kind")
            reason = _refusal_reason(backend, "kind")
            raise CapabilityError(
                f"{name}() draws with the {renderer!r} renderer, which backend={backend!r} does not have; "
                + (f"{reason}. " if reason else "")
                + f"It is honoured by {honoured} — call quickmap(data, backend={backend!r}) instead"
            )
        return quickmap(data, kind=renderer, **kwargs)

    _fn.__name__ = name
    _fn.__doc__ = f"""Quick-draw ``data`` with :meth:`Map.{name}` and return the finished Map.

    The wrapper *is* the renderer choice: it calls :func:`quickmap` with ``kind={renderer!r}`` on the
    caller's behalf, so everything else :func:`quickmap` accepts is written here unchanged.

    Args:
        data: A pyramids ``Dataset`` or ``FeatureCollection`` to draw.
        **kwargs: Forwarded to :func:`quickmap` (``crs``, ``domain``, ``basemap``, ``coastlines``,
            ``colorbar``, ``backend``, plus styling kwargs). ``kind`` is not among them — this wrapper
            supplies it.

    Returns:
        The finished map :func:`quickmap` built.

    Raises:
        CapabilityError: when ``backend=`` names a backend that has no renderer selector, since the
            {renderer!r} renderer is this wrapper's own injection rather than something the caller
            asked for. A ``ValueError``, so catching that still works. The message carries the
            tier's own reason for having no renderer selector, names the backends that do honour
            one, and points at :func:`quickmap` for the chosen one.
    """
    return _fn


field = _method("field", kind="imshow")
pcolormesh = _method("pcolormesh")

#: The two renderers :func:`contours` picks between, built once rather than per call so each keeps the
#: capability refusal every other wrapper has. The *kinds* are matplotlib's own two function names; the
#: method they reach is one ``contours`` on **both** tiers that take a ``kind``, because each tier's kind
#: table carries the ``filled=`` that tells the two renders apart (:data:`_STATIC_RASTER_KINDS`,
#: :data:`_INTERACTIVE_RASTER_KINDS`). Until the interactive table could carry it, ``contourf`` reached a
#: second method there and the rest of the call did not fit it (review H3).
_UNFILLED_CONTOURS = _method("contours", kind="contour")
_FILLED_CONTOURS = _method("contours", kind="contourf")


def contours(data: PlottableData, *, filled: bool = False, **kwargs) -> Map:
    """Quick-draw a raster as iso-value lines, or as filled bands between them.

    One function for both renders, which is what the tiers offer: ``filled=`` picks the render rather
    than the function's name doing it.

    Args:
        data: A pyramids ``Dataset`` to trace.
        filled: ``False`` (default) traces the levels as lines; ``True`` fills between them.
        **kwargs: Forwarded to :func:`quickmap` (``crs``, ``domain``, ``basemap``, ``coastlines``,
            ``colorbar``, ``backend``, plus the contour keywords ``levels`` and ``interval``).
            ``kind`` is not among them — this function supplies it.

    Returns:
        The finished map :func:`quickmap` built.

    Raises:
        CapabilityError: when ``backend=`` names a backend with no renderer selector, as every other
            renderer wrapper raises it.
    """
    wrapper = _FILLED_CONTOURS if filled else _UNFILLED_CONTOURS
    return wrapper(data, **kwargs)


def points(data: PlottableData, **kwargs) -> Map:
    """Quick-draw a FeatureCollection of points as a marker map; returns the finished Map.

    Point input is what makes it a marker map: this adds no ``kind``, so :func:`quickmap`'s own
    input-type dispatch chooses the builder. Sizing by an attribute is ``size_column``, not
    ``column`` — the latter names a polygon fill and is refused on points by name.

    Args:
        data: A pyramids ``FeatureCollection`` of point geometries.
        **kwargs: Forwarded to :func:`quickmap` (``crs``, ``domain``, ``basemap``, ``coastlines``,
            ``colorbar``, ``backend``, plus the styling kwargs ``Map.points`` takes).

    Returns:
        The finished :class:`Map` — or the other tier's map when ``backend=`` names one.

    Raises:
        ValueError: as :func:`quickmap` raises it, including for ``column`` on point input and for an
            empty collection.
        CapabilityError: as :func:`quickmap` raises it, for an argument the chosen backend cannot honour.
    """
    return quickmap(data, **kwargs)


def grid_cells(data: PlottableData, **kwargs) -> Map:
    """Quick-draw raster cells as coloured polygons; returns the finished Map.

    Unlike :func:`points` and :func:`choropleth`, this does not go through :func:`quickmap`: it builds a
    :class:`Map` and calls :meth:`Map.grid_cells` on it, so it is **matplotlib only**. There is no
    ``backend=`` to choose, and passing one reaches the cleopatra glyph as an unknown styling keyword and
    is refused there.

    Args:
        data: A pyramids ``Dataset`` whose cells become one coloured polygon each.
        **kwargs: ``crs`` sets the display CRS (default ``3857``); everything else goes to
            :meth:`Map.grid_cells` — ``band``, a ``scheme``, and the caller's own ``PolygonGlyph``
            styling.

    Returns:
        The finished :class:`Map`, with a colorbar added when the drawn layer has one to draw.

    Raises:
        ValueError: from ``PolygonGlyph`` for a styling keyword it does not accept — which is also what a
            stray ``backend=`` becomes here.
    """
    from digitalearth.static import Map

    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.grid_cells(data, **kwargs)
    return _finish(scene, colorbar=True)


def choropleth(data: PlottableData, column: str, **kwargs) -> Map:
    """Quick-draw a polygon FeatureCollection coloured by ``column``; returns the finished Map.

    Args:
        data: A pyramids ``FeatureCollection`` of polygon geometries.
        column: The attribute whose value fills each polygon. It is what makes the map a choropleth, so
            it is positional here rather than a keyword; on **point** input :func:`quickmap` refuses it by
            name, since points are sized by ``size_column`` instead.
        **kwargs: Forwarded to :func:`quickmap` (``crs``, ``domain``, ``basemap``, ``coastlines``,
            ``colorbar``, ``backend``, plus ``scheme``/``k``/``cmap`` and the rest of the builder's
            styling).

    Returns:
        The finished :class:`Map` — or the other tier's map when ``backend=`` names one.

    Raises:
        ValueError: as :func:`quickmap` raises it, including for ``column`` on point input and for an
            empty collection.
        CapabilityError: as :func:`quickmap` raises it, for an argument the chosen backend cannot honour.
    """
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
    from digitalearth.static import Map

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
    from digitalearth.static import Map

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
    from digitalearth.static import Map

    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.quadtree(data, column=column, **kwargs)
    return _finish(scene, colorbar=True)


def kde(data: PlottableData, **kwargs) -> Map:
    """Quick-draw a 2-D kernel-density surface of a point FeatureCollection; returns the finished Map.

    ``clip`` and styling kwargs (``levels``/``shade``/``gridsize``/``cmap``/…) are forwarded to :meth:`Map.kde`.

    Like :func:`grid_cells`, this builds a :class:`Map` and draws on it rather than going through
    :func:`quickmap`, so it is **matplotlib only**: there is no ``backend=`` to choose.

    Args:
        data: A pyramids ``FeatureCollection`` of point geometries.
        **kwargs: ``crs`` sets the display CRS (default ``3857``); everything else goes to
            :meth:`Map.kde` — ``clip``, and the caller's own ``KDEGlyph`` styling.

    Returns:
        The finished :class:`Map`, with a colorbar added when the drawn layer has one to draw.

    See Also:
        digitalearth.static.maps.vector.VectorMixin.kde: the ``Map`` method this wraps.
    """
    from digitalearth.static import Map

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
    from digitalearth.static import Map

    scene = Map(crs=kwargs.pop("crs", 3857))
    scene.sankey(data, column=column, scale=scale, **kwargs)
    return _finish(scene, colorbar=column is not None)
