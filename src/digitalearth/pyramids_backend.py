"""Serve pyramids' ``plot(backend=…)`` seam from digitalearth's three non-matplotlib tiers.

pyramids-core resolves a non-matplotlib ``backend=`` through the ``pyramids.plot_backends``
entry-point group: ``Dataset.plot(backend="3d")``, ``NetCDF.plot(backend="interactive")`` and the
rest hand the object to whichever callable is registered under that name. This module is the
registered end of that seam — one callable per tier, each matching the contract shape
``fn(obj, *, backend, **kwargs)``.

Registering it is not merely cosmetic. Without an entry point pyramids falls back to *importing*
``digitalearth`` by name and calling ``quickmap`` directly, which forwards its own vocabulary
verbatim. Each tier then answers in its own terms, naming its internal builder rather than the call
the caller made — and on one tier the complaint is not even digitalearth's:

    ds.plot(backend="web", title="DEM")
    TypeError: RasterMixin.field() got an unexpected keyword argument 'title'. Did you mean 'tiles'?

    ds.plot(backend="interactive", rgb_options={})
    ValueError: 'rgb_options' is not an option Image takes on the bokeh backend; did you mean
    ['xrotation', 'yrotation']?

    ds.plot(backend="3d", title="DEM")
    TypeError: "title" is an invalid keyword argument for `_common_arg_parser`

The last is PyVista's own argument parser, surfacing to someone who wrote ``ds.plot(backend="3d",
title=…)``. So the adapter's job is to translate between the two vocabularies, to apply a frame
keyword through the route that tier actually has, and to refuse **the keywords it knows about** by
name, before a scene is half-built — not to convert silence into a message, since the tiers were
never silent.

That last qualifier is the honest scope. This module checks the keywords the facades are documented
to forward: the ones no tier can serve (:data:`_UNSERVED`), the animation-only ones, the frame ones,
and — on the animate path, where it knows the builder — every remaining keyword against that
builder's signature. A keyword outside that set still reaches the tier on a **still** plot and is
answered there, in the terms quoted above, because ``quickmap`` takes ``**kwargs`` and there is no
signature to check it against (round-2 M2).

Three translations are genuinely necessary, because the facade and this package count and spell
things differently:

* **``band`` is 0-based upstream and 1-based here.** pyramids documents the forwarded selector as
  "a **0-based** raster band index (band ``0`` is the first band, not band ``1``)", while
  digitalearth's public ``band=`` is 1-based everywhere. Forwarded unchanged, ``band=0`` — what a
  facade sends for the *first* band — was refused on every tier, though not in the same words: the
  web and HoloViz tiers answered ``band -1 is out of range for a 1-band dataset`` and the 3-D tier
  ``Selection bands are 1-based; got 0`` (round-2 M9). That is the off-by-one reported in #501.
* **``variable_name`` is this package's ``variable``.** The ``UgridDataset`` facade spells the mesh
  selector ``variable_name``; ``quickmap`` spells it ``variable``.
* **``interval`` is milliseconds; the tiers take ``fps``.** The animation facades forward a frame
  interval, and every digitalearth animation verb takes a rate instead.

The module import is deliberately **light** — standard library only. Discovery imports every
registered entry point, so a heavy import here would pull a drawing stack into a bare
``import pyramids``; ``quickmap`` and the tier classes are imported inside the callables.
"""

from __future__ import annotations

import inspect
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = [
    "ANIMATION_BACKENDS",
    "BAND_ORIGIN_SHIFT",
    "interactive",
    "three_d",
    "web",
]

#: Added to a forwarded ``band`` to turn pyramids' 0-based index into digitalearth's 1-based one.
BAND_ORIGIN_SHIFT = 1

#: The backends whose tier owns a verb that animates *data*. The 3-D tier's ``record`` needs a
#: caller-supplied ``update`` callable and its ``orbit`` sweeps the camera over a static scene, so
#: neither can be driven from the facade's forwarded keywords alone — an animation request for it is
#: refused by name instead of answered with a still picture or a camera spin.
ANIMATION_BACKENDS: frozenset[str] = frozenset({"interactive", "web"})

#: Forwarded keywords digitalearth has no way to honour, each with the reason given in the refusal.
#: pyramids drops a defaulted argument before forwarding, so anything arriving here was set
#: deliberately. Refusing it here names the pyramids-side keyword and what to do instead; left to
#: the tier it is still refused, but in terms of whichever builder it reached — or, on the 3-D tier,
#: in PyVista's terms.
_UNSERVED: Mapping[str, str] = {
    "exclude_value": (
        "no digitalearth entry point takes a nodata value — the matplotlib path sets its own and "
        "answers a caller's with `got multiple values for keyword argument 'exclude_value'` — so "
        "mask the raster in pyramids before plotting"
    ),
    "rgb_options": (
        "a three-band composite is built with Map.rgb_composite on the static tier, which this "
        "seam does not reach"
    ),
    # Both say "0-based" deliberately: a caller following this advice passes `band=` as an extra
    # keyword, and this seam shifts whatever it is given. Advising the keyword without naming its
    # origin reintroduced #501's own off-by-one, one layer up (round-2 M7).
    "selectors": (
        "a NetCDF slice is chosen here with variable= and a 0-based band= (band 0 is the first), "
        "not with read-time selectors"
    ),
    "axes": (
        "a NetCDF slice is chosen here with variable= and a 0-based band= (band 0 is the first), "
        "not by naming its axes"
    ),
}

#: Keywords that only mean something in an animation call. Refused on a still plot in terms of the
#: call that was made, rather than left to the tier, which answers for the builder they reached
#: (`RasterMixin.field() … 'interval'` on web, `'interval' is not an option Image takes on the bokeh
#: backend` on interactive). `path` is here because a still render has nothing to write — only the
#: animation verbs take a destination.
_ANIMATION_ONLY = ("bands", "animation_axis_values", "interval", "path")

#: Keywords this module routes itself rather than handing to a builder: the scene frame. They are
#: applied through each tier's own constructor argument or verb, which differ — so they must not be
#: offered to the data builder, whose signature does not carry them. This is the **animate** path's
#: rule: a still plot delegates the whole call to `quickmap`, which routes the frame itself, and
#: only `title` is lifted out there (round-2 L2).
_FRAME_KWARGS = ("crs", "title", "basemap")

#: Tiers whose raster builder refuses `title` but whose scene owns `set_title`, so a still plot's
#: title is applied after the scene is built. The HoloViz tier is absent because it has no
#: `set_title` and its builder does accept `title` (executed on all three tiers, round-1 H2).
_TITLE_AFTER_BUILD = frozenset({"web", "3d"})

#: Keywords a tier can honour on the **built scene** when its data builder has no parameter for
#: them. `colorbar` is in every facade's forwarded set and the still web path honours it, but
#: `WebMap.timeslider` has no such parameter — so refusing it made a documented call impossible on
#: the animate path while it worked on the still one (round-2 H2). Both animating tiers own a
#: `colorbar()` verb, so it is applied there instead. A falsy value needs no call: neither tier
#: shows a key unless one is asked for.
_SCENE_FALLBACK = ("colorbar",)


def _accepted_by(verb: Callable[..., Any]) -> tuple[frozenset[str], bool]:
    """The keyword names a callable accepts, and whether it absorbs the rest through ``**kwargs``.

    Read from the live signature rather than a list kept here, because a list kept here goes stale
    silently: it both offers a builder a keyword it does not take (an immediate ``TypeError``) and
    drops a keyword it does take without a word.

    Args:
        verb: The builder or constructor to inspect.

    Returns:
        tuple: the accepted keyword names, and ``True`` when a ``**kwargs`` parameter absorbs any
            further keyword.

    Examples:
        - A plain keyword function accepts exactly its own names:
            ```python
            >>> from digitalearth.pyramids_backend import _accepted_by
            >>> def draw(data, *, band=1, cmap=None):
            ...     ...
            >>> names, absorbs = _accepted_by(draw)
            >>> sorted(names), absorbs
            (['band', 'cmap'], False)

            ```
        - A function with ``**opts`` absorbs anything, so nothing needs refusing on its behalf:
            ```python
            >>> from digitalearth.pyramids_backend import _accepted_by
            >>> def draw(data, *, band=1, **opts):
            ...     ...
            >>> names, absorbs = _accepted_by(draw)
            >>> sorted(names), absorbs
            (['band'], True)

            ```

    See Also:
        _for_builder: Splits a forwarded set against what this reports.
    """
    parameters = inspect.signature(verb).parameters.values()
    # Only parameters with a default. The rule is "optional", not "is not the data" — those are
    # different properties, and this package has a counter-example in `Scene3D.basemap`, whose
    # keyword-only `bbox` is required and would be dropped here (round-2 N1). It holds for the
    # verbs this routes to, whose one required parameter is the data, and it is what keeps a
    # refusal from inviting a caller to pass their raster as a keyword. A required keyword the
    # builder genuinely needs would be refused as unknown, which is loud rather than silent.
    named = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and parameter.default is not inspect.Parameter.empty
    }
    absorbs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    return frozenset(named), absorbs


def _for_builder(
    verb: Callable[..., Any],
    candidates: Mapping[str, Any],
    named: str,
    fallback: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``candidates`` against what ``verb`` accepts, refusing what it cannot take.

    Args:
        verb: The builder the keywords are destined for.
        candidates: The forwarded keywords left after the frame ones are routed.
        named: How to describe the builder in a refusal, e.g. ``"animate(backend='web')"``.
        fallback: Names that may be applied to the built scene instead of the builder, and so are
            handed back rather than refused.

    Returns:
        tuple: the keywords to pass to the builder, and those left for the scene.

    Raises:
        ValueError: when a candidate is one neither the builder nor the scene can take — named
            here, with the ones the builder does accept, instead of dropped in silence or sent on
            to a ``TypeError``.

    Examples:
        - Keywords the builder takes are passed through:
            ```python
            >>> from digitalearth.pyramids_backend import _for_builder
            >>> def slider(data, *, band=1, cmap="viridis"):
            ...     ...
            >>> _for_builder(slider, {"band": 2, "cmap": "magma"}, "animate(backend='web')")
            ({'band': 2, 'cmap': 'magma'}, {})

            ```
        - One it does not take, but the scene can, is handed back for the scene:
            ```python
            >>> from digitalearth.pyramids_backend import _for_builder
            >>> def slider(data, *, band=1):
            ...     ...
            >>> _for_builder(slider, {"colorbar": True}, "animate(…)", fallback=("colorbar",))
            ({}, {'colorbar': True})

            ```
        - One neither can take is refused by name, listing what the builder will accept:
            ```python
            >>> from digitalearth.pyramids_backend import _for_builder
            >>> def slider(data, *, band=1, cmap="viridis"):
            ...     ...
            >>> try:
            ...     _for_builder(slider, {"opacity": 0.5}, "animate(backend='web')")
            ... except ValueError as exc:
            ...     print(exc)
            animate(backend='web') cannot apply opacity=…; this render's parameters are band, cmap

            ```

    See Also:
        _accepted_by: Reads the signature this checks against.
        _refuse_unserved: The earlier gate, for keywords no tier could serve.
    """
    accepted, absorbs = _accepted_by(verb)
    taken: dict[str, Any] = {}
    scened: dict[str, Any] = {}
    refused: list[str] = []
    for name, value in candidates.items():
        if name in accepted or absorbs:
            taken[name] = value
        elif name in fallback:
            scened[name] = value
        else:
            refused.append(name)
    if refused:
        # "takes", not "accepts for this data": the list is the builder's whole parameter set, and
        # a raster series will not use its vector-choropleth half (round-2 L7).
        offered = ", ".join(sorted(accepted))
        wanted = ", ".join(f"{name}=…" for name in sorted(refused))
        raise ValueError(
            f"{named} cannot apply {wanted}; this render's parameters are {offered}"
        )
    return taken, scened


def _refuse_unserved(backend: str, forwarded: Mapping[str, Any]) -> None:
    """Refuse a forwarded keyword digitalearth cannot act on, naming it and the reason.

    Args:
        backend: The backend the facade selected, named in the message.
        forwarded: The keywords pyramids forwarded.

    Raises:
        ValueError: when any keyword in :data:`_UNSERVED` was forwarded.

    Examples:
        - A keyword this package cannot honour is refused in its own words, naming the alternative:
            ```python
            >>> from digitalearth.pyramids_backend import _refuse_unserved
            >>> try:
            ...     _refuse_unserved("web", {"rgb_options": {"bands": [1, 2, 3]}})
            ... except ValueError as exc:
            ...     print("rgb_options" in str(exc), "Map.rgb_composite" in str(exc))
            True True

            ```
        - Anything else passes, returning ``None``:
            ```python
            >>> _refuse_unserved("web", {"title": "DEM", "band": 0}) is None
            True

            ```
    """
    # Every match, not the first: `NetCDF.plot` forwards both `selectors` and `axes`, so stopping
    # at one had a caller fix it and be refused again for the other (round-2 N3).
    hit = [name for name in _UNSERVED if name in forwarded]
    if hit:
        reasons = "; ".join(f"{name}= — {_UNSERVED[name]}" for name in hit)
        raise ValueError(
            f"plot(…) cannot serve {len(hit)} argument(s) on backend={backend!r}: {reasons}. Drop "
            f"them for this backend, or render on backend='matplotlib'."
        )


def _basemap_asked(value: Any, backend: str) -> Any:
    """The basemap the tiers can actually act on, out of the shapes a facade may forward.

    The facades type ``basemap`` as ``bool | str | dict | Basemap | None``; the tiers take ``True``
    for their default, a provider name, an ``xyzservices`` provider or a keyed preset. A dict
    reached the web tier and came back as ``AttributeError: 'dict' object has no attribute
    'replace'``, while the 3-D and HoloViz tiers accepted it and drew no basemap at all — a
    requested basemap silently missing, which is worse than the error (round-2 M3).

    Args:
        value: The ``basemap`` the facade forwarded.
        backend: The backend the facade selected, named in a refusal.

    Returns:
        The value to hand the tier: ``True``, or a provider name.

    Raises:
        ValueError: for a shape no tier can act on, named rather than dropped.

    Examples:
        - ``True`` and a provider name pass straight through:
            ```python
            >>> from digitalearth.pyramids_backend import _basemap_asked
            >>> _basemap_asked(True, "web"), _basemap_asked("CartoDB.Positron", "web")
            (True, 'CartoDB.Positron')

            ```
        - A mapping carrying a provider name is unwrapped to that name:
            ```python
            >>> from digitalearth.pyramids_backend import _basemap_asked
            >>> _basemap_asked({"provider": "CartoDB.Positron"}, "web")
            'CartoDB.Positron'

            ```
        - A mapping naming no provider is refused, saying what the tiers take:
            ```python
            >>> from digitalearth.pyramids_backend import _basemap_asked
            >>> try:
            ...     _basemap_asked({"zoom": 3}, "web")
            ... except ValueError as exc:
            ...     print(str(exc).split("can apply: ")[1])
            pass True for the tier's default, or the provider name (optionally as {'provider': '…'})

            ```
    """
    if not isinstance(value, Mapping):
        return value
    named = value.get("provider") or value.get("name")
    if named is None:
        raise ValueError(
            f"basemap={value!r} is not a shape backend={backend!r} can apply: pass True for the "
            f"tier's default, or the provider name (optionally as {{'provider': '…'}})"
        )
    return named


def _shifted_band(band: Any) -> int:
    """Turn the facade's 0-based band index into digitalearth's 1-based one.

    Args:
        band: The forwarded ``band`` value — a 0-based index per the plot-backend contract.

    Returns:
        int: the same band as a 1-based index.

    Raises:
        ValueError: when ``band`` is not a non-negative integer — a 0-based index is always one.

    Examples:
        - The facade's first band is this package's band 1:
            ```python
            >>> from digitalearth.pyramids_backend import _shifted_band
            >>> _shifted_band(0), _shifted_band(3)
            (1, 4)

            ```
        - A NumPy integer is a band index too, and comes back as a plain ``int``:
            ```python
            >>> import numpy as np
            >>> from digitalearth.pyramids_backend import _shifted_band
            >>> shifted = _shifted_band(np.int64(0))
            >>> shifted, type(shifted).__name__
            (1, 'int')

            ```
        - A negative index is refused rather than silently reaching the reader:
            ```python
            >>> try:
            ...     _shifted_band(-1)
            ... except ValueError as exc:
            ...     print(str(exc))
            band=-1 is not a 0-based band index; the first band is 0, so there is no negative band

            ```
    """
    # `numbers.Integral`, not `int`: a NumPy-native stack hands out `np.int64` everywhere — a band
    # read off an array's shape or an index is one — and refusing those made the seam unusable with
    # the arrays it exists to draw (round-1 M4). `bool` is an Integral too, and `band=True` is a
    # mistake worth naming rather than reading as band 1.
    if isinstance(band, bool) or not isinstance(band, numbers.Integral):
        raise ValueError(
            f"band={band!r} is not a 0-based band index; pass an integer, where the first band "
            f"is 0"
        )
    if band < 0:
        raise ValueError(
            f"band={band} is not a 0-based band index; the first band is 0, so there is no "
            f"negative band"
        )
    # `int()` so a NumPy scalar leaves here as a plain int: it is handed to a builder that may put
    # it in a message or a dict key, and `np.int64(1)` prints as `np.int64(1)` in both.
    return int(band) + BAND_ORIGIN_SHIFT


def _rate(interval: Any) -> float:
    """Turn a frame interval in milliseconds into frames per second.

    Args:
        interval: The forwarded ``interval``, in milliseconds per frame.

    Returns:
        float: the equivalent rate in frames per second.

    Raises:
        ValueError: when ``interval`` is not a positive number, which no frame interval is.

    Examples:
        - A 200 ms interval is five frames a second:
            ```python
            >>> from digitalearth.pyramids_backend import _rate
            >>> _rate(200)
            5.0

            ```
        - A NumPy number is a frame interval too:
            ```python
            >>> import numpy as np
            >>> from digitalearth.pyramids_backend import _rate
            >>> _rate(np.int64(200))
            5.0

            ```
        - A non-positive interval is refused rather than dividing by zero:
            ```python
            >>> try:
            ...     _rate(0)
            ... except ValueError as exc:
            ...     print(str(exc))
            interval=0 is not a frame interval in milliseconds; it must be positive

            ```
    """
    if isinstance(interval, bool) or not isinstance(interval, numbers.Real):
        raise ValueError(
            f"interval={interval!r} is not a frame interval in milliseconds; pass a number"
        )
    if interval <= 0:
        raise ValueError(
            f"interval={interval} is not a frame interval in milliseconds; it must be positive"
        )
    return 1000.0 / float(interval)


@dataclass(frozen=True)
class _Forwarded:
    """One facade call's forwarded keywords, translated into digitalearth's own vocabulary.

    The facade hands over a flat ``**kwargs`` in which a few names mean something other than they
    do here, a few are animation-only, and the rest pass straight through. Those values arrive
    together, are translated together and are consumed together by one render call, so they are
    carried as one record with the translation on it rather than threaded through helpers as a
    growing parameter list.

    Attributes:
        backend: The backend name the facade selected, as the seam passes it.
        animate: Whether this is an animation call (the facade forwards ``animate=True``).
        draw: The keywords to hand ``quickmap`` — translated, with the animation-only ones removed.
            A read-only mapping, so the record really is the fixed record it claims to be. That
            makes the record unpicklable and un-deep-copyable, which nothing here needs: it is
            built, read and dropped inside one call (round-2 L4).
        fps: Playback rate in frames per second, or ``None`` when the call named no ``interval``.
        path: Where to write the animation, or ``None`` when the caller named no ``path``.
        labels: The per-step labels the facade forwarded as ``animation_axis_values``.
        axis: The dimension to step along, when the facade named one. ``NetCDF.plot`` forwards
            ``animate`` as the *name* of an axis, not only as a flag, and both tiers' temporal
            builders take that name as ``kdim`` — so reading it as a bool alone lost it.

    Examples:
        - A still plot's record carries the translated draw keywords and nothing else:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded
            >>> plan = _Forwarded.from_facade("web", {"band": 0, "title": "DEM"})
            >>> plan.animate, plan.title, dict(plan.draw)["band"]
            (False, 'DEM', 1)

            ```
        - An animation's record separates the frame keywords from the draw ones:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded
            >>> plan = _Forwarded.from_facade(
            ...     "interactive", {"animate": True, "interval": 200, "cmap": "viridis"}
            ... )
            >>> plan.animate, plan.fps, dict(plan.draw)
            (True, 5.0, {'cmap': 'viridis'})

            ```

    See Also:
        _Forwarded.from_facade: Builds one from a facade call, applying every translation.
        _Forwarded.destination: The resolved animation path, for the tiers that write one.
    """

    backend: str
    animate: bool
    # A read-only view, not a bare dict: `frozen=True` stops the field being rebound but says
    # nothing about what it points at, so a plain dict left this record mutable through it after
    # every translation had been applied (round-1 L5).
    draw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    fps: float | None = None
    path: str | None = None
    labels: Any = None
    axis: str | None = None

    @property
    def crs(self) -> Any:
        """The forwarded display CRS, or ``None`` when the call named none.

        Returns:
            The ``crs`` the facade forwarded.
        """
        return self.draw.get("crs")

    @property
    def title(self) -> Any:
        """The forwarded title, or ``None`` when the call named none.

        Returns:
            The ``title`` the facade forwarded.
        """
        return self.draw.get("title")

    def destination(self) -> str:
        """The path to write the animation to, refusing when the call named none.

        A tier that animates only to a file has nowhere to put the frames without one, and the
        facades forward no destination of their own — so the caller has to pass ``path=``. Checked
        here rather than at the call site because the answer is also the narrowed type the export
        takes.

        Returns:
            str: the destination path.

        Raises:
            ValueError: when the call named no ``path``.

        Examples:
            - A named destination is handed back:
                ```python
                >>> from digitalearth.pyramids_backend import _Forwarded
                >>> plan = _Forwarded.from_facade("web", {"animate": True, "path": "out.gif"})
                >>> plan.destination()
                'out.gif'

                ```
            - Without one, the refusal says what to pass:
                ```python
                >>> plan = _Forwarded.from_facade("web", {"animate": True})
                >>> try:
                ...     plan.destination()
                ... except ValueError as exc:
                ...     print("writes the animation to a file" in str(exc))
                True

                ```
        """
        if self.path is None:
            raise ValueError(
                f"animate(backend={self.backend!r}) writes the animation to a file, so it needs a "
                f"path; pass path='…gif' (or '…mp4') to the animate call."
            )
        return self.path

    @property
    def basemap(self) -> Any:
        """The forwarded basemap request, or ``None`` when the call named none.

        ``True`` asks for the tier's default tiles; anything else names the source. Read through a
        property rather than lifted out of :attr:`draw`, because a still plot forwards all three of
        these to ``quickmap`` unchanged — only the animation path, which builds the scene itself,
        needs them by name.

        Returns:
            The ``basemap`` the facade forwarded.
        """
        return self.draw.get("basemap")

    @classmethod
    def from_facade(cls, backend: str, forwarded: Mapping[str, Any]) -> _Forwarded:
        """Translate one facade call's keywords, refusing what cannot be served.

        Args:
            backend: The backend name the seam passed.
            forwarded: The keywords pyramids forwarded, in its own vocabulary.

        Returns:
            _Forwarded: the translated record.

        Raises:
            ValueError: for a keyword digitalearth cannot honour, an animation-only keyword on a
                still plot, a per-band animation (``bands``), a band that is not a 0-based index, or
                an unusable ``interval``.

        Examples:
            - A still plot translates the band and keeps the rest untouched:
                ```python
                >>> from digitalearth.pyramids_backend import _Forwarded
                >>> plan = _Forwarded.from_facade("web", {"band": 0, "title": "DEM"})
                >>> plan.animate, sorted(dict(plan.draw).items())
                (False, [('band', 1), ('title', 'DEM')])

                ```
            - An animation call lifts the animation keywords out of the draw kwargs:
                ```python
                >>> plan = _Forwarded.from_facade(
                ...     "interactive", {"animate": True, "interval": 250, "path": "out.gif"}
                ... )
                >>> plan.animate, plan.fps, plan.path, dict(plan.draw)
                (True, 4.0, 'out.gif', {})

                ```
            - A named animation axis is kept, not reduced to a flag:
                ```python
                >>> plan = _Forwarded.from_facade("interactive", {"animate": "time"})
                >>> plan.animate, plan.axis
                (True, 'time')

                ```
            - A per-band animation is refused: both tiers step through a stack's members:
                ```python
                >>> try:
                ...     _Forwarded.from_facade("web", {"animate": True, "bands": [0, 1]})
                ... except ValueError as exc:
                ...     print("animates a stack" in str(exc))
                True

                ```
            - An animation-only keyword on a still plot is refused, not ignored:
                ```python
                >>> try:
                ...     _Forwarded.from_facade("web", {"interval": 200})
                ... except ValueError as exc:
                ...     print(str(exc).split(". ")[0])
                interval= only applies to an animation; this is a still plot on backend='web'

                ```
        """
        _refuse_unserved(backend, forwarded)
        rest = dict(forwarded)
        # `NetCDF.plot` forwards `animate` as the name of the axis to step along, not just a flag,
        # so a bool conversion alone silently discarded which dimension to animate (round-1 M3).
        asked = rest.pop("animate", False)
        animate = bool(asked)
        axis = asked if isinstance(asked, str) and asked else None
        if not animate:
            for name in _ANIMATION_ONLY:
                if name in rest:
                    # Worded as the keyword alone, not `plot(path=…)`: no facade offers a
                    # `path` argument to `plot`, so quoting that call shape would describe a call
                    # the caller could not have made (round-1 N3).
                    raise ValueError(
                        f"{name}= only applies to an animation; this is a still plot on "
                        f"backend={backend!r}. Call animate(backend={backend!r}) instead, or drop "
                        f"it."
                    )
        if "bands" in rest:
            # Lifting this out and ignoring it would animate the wrong thing in silence: both tiers'
            # temporal builders take a *stack* and step through its members, so there is nowhere to
            # honour a per-band selection within one raster.
            raise ValueError(
                f"animate(bands=…) is not served on backend={backend!r}: this seam animates a "
                f"stack, stepping through its members. Pass a DatasetCollection of the steps to "
                f"animate, or animate the bands on backend='matplotlib'."
            )
        interval = rest.pop("interval", None)
        labels = rest.pop("animation_axis_values", None)
        path = rest.pop("path", None)
        # The mesh facade spells the selector `variable_name`; `quickmap` spells it `variable`.
        # Inert as things stand, and kept deliberately: only `digitalearth.api.MESH_BACKENDS`
        # ({"matplotlib"}) has a triangular render, so `quickmap` refuses a `UgridDataset` on all
        # three of these tiers before any keyword is read — executed on each through the repo's mesh
        # fixture (round-1 M1). It is the right translation the moment a tier gains that render, and
        # `test_no_plugin_backend_draws_a_mesh_yet` fails here if one does.
        if "variable_name" in rest:
            rest["variable"] = rest.pop("variable_name")
        if "band" in rest:
            rest["band"] = _shifted_band(rest["band"])
        return cls(
            backend=backend,
            animate=animate,
            draw=MappingProxyType(rest),
            fps=None if interval is None else _rate(interval),
            path=path,
            labels=labels,
            axis=axis,
        )


def _refuse_unanimatable(plan: _Forwarded) -> None:
    """Refuse an animation the selected tier has no data-animation verb for.

    Args:
        plan: The translated call.

    Raises:
        ValueError: when the backend is not in :data:`ANIMATION_BACKENDS`. A missing destination is
            refused separately, by :meth:`_Forwarded.destination`.

    Examples:
        - The 3-D tier has no verb the seam can drive, and says which ones to use directly:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _refuse_unanimatable
            >>> plan = _Forwarded.from_facade("3d", {"animate": True})
            >>> try:
            ...     _refuse_unanimatable(plan)
            ... except ValueError as exc:
            ...     print("Scene3D.record" in str(exc), "orbit" in str(exc))
            True True

            ```
    """
    if plan.backend not in ANIMATION_BACKENDS:
        raise ValueError(
            f"animate(backend={plan.backend!r}) is not served through the pyramids plot-backend "
            f"seam: the 3-D tier animates a scene with Scene3D.record (which needs an update "
            f"callable) or sweeps the camera with Scene3D.orbit, and neither can be driven from a "
            f"plot call's keywords. Build the scene with digitalearth and call one of them "
            f"directly."
        )


def _animatable(obj: Any, plan: _Forwarded) -> tuple[Any, Any, tuple[str, ...]]:
    """The stack a temporal builder can step through, and the labels for its steps.

    Asked by **type**, not by attribute. The duck-typed version of this (``hasattr(obj,
    "datasets")``) misread a ``FeatureCollection``: it subclasses ``GeoDataFrame``, and pandas
    resolves a column name as an attribute, so a vector layer with a column called ``datasets``
    was refused as a multi-member stack and told to call a facade pyramids does not expose for
    vectors (round-2 H1). The import is inside the function, so this module still imports nothing
    from pyramids at discovery time.

    Args:
        obj: The object the facade was called on.
        plan: The translated call.

    Returns:
        tuple: the object to hand the temporal builder, the step labels to pass with it (``None``
            when the data carries none of its own), and the keywords this consumed in choosing
            them — so they are neither offered to the builder nor dropped in silence.

    Raises:
        ValueError: when ``obj`` is nothing either tier can step through.

    Examples:
        - Something that is neither a stack nor a NetCDF is refused, naming both shapes:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _animatable
            >>> class Neither:
            ...     pass
            >>> plan = _Forwarded.from_facade("interactive", {"animate": True})
            >>> try:
            ...     _animatable(Neither(), plan)
            ... except ValueError as exc:
            ...     print(str(exc).split(". ")[0].split(") ")[1])
            steps through a stack, which Neither is not

            ```

    See Also:
        _netcdf_stack: Slices the NetCDF form into the stack this returns.
        _is_vector: The web-only second form.
    """
    from pyramids.dataset.collection import DatasetCollection
    from pyramids.netcdf import NetCDF

    if isinstance(obj, DatasetCollection):
        return obj, None, ()
    if isinstance(obj, NetCDF):
        stack, labels = _netcdf_stack(obj, plan)
        # `variable` picked which variable to slice, so it has done its job here; left in, it
        # reaches `timecube`, whose `**opts` forwards it to HoloViews: `'variable' is not an option
        # Image takes on the bokeh backend` (round-2 H3/M1).
        return stack, labels, ("variable",)
    if plan.backend == "web" and _is_vector(obj):
        # `timeslider` documents a vector series as one of the two forms it takes: one layer per
        # step, sharing a colour classification across the series.
        return obj, None, ()
    raise ValueError(
        f"animate(backend={plan.backend!r}) steps through a stack, which "
        f"{type(obj).__name__} is not. Pass a DatasetCollection of the steps, or a NetCDF whose "
        f"variable carries the axis to step along"
    )


def _is_vector(obj: Any) -> bool:
    """Whether ``obj`` is a vector layer the web tier's slider can step through.

    Args:
        obj: The object the facade was called on.

    Returns:
        bool: ``True`` for a ``FeatureCollection`` or the ``GeoDataFrame`` it subclasses.

    Examples:
        - A plain object is no vector layer:
            ```python
            >>> from digitalearth.pyramids_backend import _is_vector
            >>> _is_vector(object())
            False

            ```
        - A GeoDataFrame is, which is what makes it the web slider's second accepted form:
            ```python
            >>> from geopandas import GeoDataFrame
            >>> from digitalearth.pyramids_backend import _is_vector
            >>> _is_vector(GeoDataFrame({"a": [1]}))
            True

            ```
    """
    try:
        from geopandas import GeoDataFrame
    except ImportError:  # pragma: no cover - geopandas arrives with pyramids
        return False
    return isinstance(obj, GeoDataFrame)


def _netcdf_stack(nc: Any, plan: _Forwarded) -> tuple[Any, Any]:
    """Slice a NetCDF variable's extra axis into a collection of steps, with its own labels.

    Neither tier's temporal builder takes a NetCDF: both step through a collection's members. So
    the variable's bands — its time steps — are read into one in-memory ``Dataset`` each. Before
    this, every NetCDF animation was refused with a message naming pyramids' internal ``Container``
    class and claiming a 624-step variable carried no steps (round-2 H3).

    The whole axis is materialised, which is the cost of animating it: roughly 40 MB for the 624
    steps of ``examples/data/global/ersstv5.nc``.

    Args:
        nc: The ``NetCDF`` container, or a variable already pinned out of one.
        plan: The translated call. ``variable`` names the variable to animate; ``axis`` names the
            dimension whose values label the steps.

    Returns:
        tuple: a ``DatasetCollection`` of the steps, and the axis values that label them.

    Raises:
        ValueError: when the container exposes no variable to animate, or the chosen variable
            carries a single band and so has no axis to step along.
    """
    from pyramids.dataset import Dataset, GeoReference
    from pyramids.dataset.collection import DatasetCollection

    variable = _netcdf_variable(nc, plan)
    steps = int(variable.band_count)
    if steps < 2:
        raise ValueError(
            f"animate(backend={plan.backend!r}): the variable carries one band, so it has no axis "
            f"to step along. Animate a variable with a time or level dimension"
        )
    reference = GeoReference(geo=variable.geotransform, epsg=variable.epsg)

    def member(step: int) -> Any:
        """Read one step of the axis into its own in-memory Dataset.

        Args:
            step: The 0-based band to read — pyramids' own reader indexing, not the facade's
                1-based selector.

        Returns:
            The step as a Dataset sharing the variable's geobox.
        """
        return Dataset.from_array(variable.read_array(band=step), geo_ref=reference)

    # The first member is built on its own and the collection is seeded with it, rather than
    # indexing the list afterwards: `steps >= 2` is checked above, but only the explicit form says
    # so to a reader (and to a static analyser).
    first = member(0)
    members = [first] + [member(step) for step in range(1, steps)]
    return DatasetCollection(first, steps, datasets=members), _axis_values(
        variable, plan
    )


def _netcdf_variable(nc: Any, plan: _Forwarded) -> Any:
    """The variable to animate: the one named, the one ``nc`` already is, or the container's first.

    Args:
        nc: The container, or a variable already pinned out of one.
        plan: The translated call.

    Returns:
        The variable to slice.

    Raises:
        ValueError: when the container exposes no variables.

    Examples:
        - A container hands back the variable the call named:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _netcdf_variable
            >>> class Container:
            ...     variable_names = ["time_bnds", "sst"]
            ...     def get_variable(self, name):
            ...         return f"<{name}>"
            >>> plan = _Forwarded.from_facade(
            ...     "interactive", {"animate": True, "variable": "sst"}
            ... )
            >>> _netcdf_variable(Container(), plan)
            '<sst>'

            ```
        - With none named it takes the first, which is what the still path draws too:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _netcdf_variable
            >>> class Container:
            ...     variable_names = ["time_bnds", "sst"]
            ...     def get_variable(self, name):
            ...         return f"<{name}>"
            >>> plan = _Forwarded.from_facade("interactive", {"animate": True})
            >>> _netcdf_variable(Container(), plan)
            '<time_bnds>'

            ```
        - A variable already pinned out of its container is used as it is:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _netcdf_variable
            >>> class Pinned:
            ...     variable_names = []
            >>> pinned = Pinned()
            >>> plan = _Forwarded.from_facade("interactive", {"animate": True})
            >>> _netcdf_variable(pinned, plan) is pinned
            True

            ```
    """
    named = plan.draw.get("variable")
    carried = list(getattr(nc, "variable_names", None) or [])
    if not carried:
        # A pinned subset reports no variable names of its own because it *is* the variable, which
        # is the same test `base.sources.extractors` makes of a subset.
        return nc
    if named is None:
        # The same choice the still path makes, so `plot` and `animate` draw the same variable.
        named = next(iter(carried))
    return nc.get_variable(named)


def _axis_values(variable: Any, plan: _Forwarded) -> Any:
    """The values along the animated axis, used to label the steps.

    Args:
        variable: The variable being sliced.
        plan: The translated call, whose ``axis`` names the dimension when the caller did.

    Returns:
        The axis values, or ``None`` when the variable exposes none.
    """
    if plan.axis is not None:
        return variable.get_dimension_values(plan.axis)
    return variable.get_time_values()


def _animated(obj: Any, plan: _Forwarded) -> Any:
    """Build the tier's temporal scene and return its animation.

    Args:
        obj: The pyramids object the facade was called on.
        plan: The translated call.

    Returns:
        The interactive tier's player, or the path the animation was written to.

    Raises:
        ValueError: propagated from :func:`_refuse_unanimatable` for a tier with no verb to drive,
            from :func:`_animatable` for data with no steps, or from
            :meth:`_Forwarded.destination` for a web animation with no destination.
    """
    _refuse_unanimatable(plan)
    if plan.backend == "interactive":
        stack, labels, consumed = _animatable(obj, plan)
        return _animated_interactive(stack, plan, labels, consumed)
    # The destination is resolved first, before the tier is imported and before the payload is
    # examined: it is a property of the call rather than of the data, so a pathless animation is
    # answered the same way whatever was passed, and never pays for the engine.
    destination = plan.destination()
    stack, labels, consumed = _animatable(obj, plan)
    return _animated_web(stack, plan, destination, labels, consumed)


def _animated_interactive(
    obj: Any, plan: _Forwarded, labels: Any, consumed: tuple[str, ...] = ()
) -> Any:
    """Animate a stack on the HoloViz tier: a time cube, then its player or its export.

    Args:
        obj: The stack to animate.
        plan: The translated call.
        labels: Step labels the data carried, used when the call named none.
        consumed: Keywords already spent choosing the data, so the builder is not offered them.

    Returns:
        A Panel player when the call named no ``path``, else whatever ``save_animation`` returns
        for the destination it wrote.

    Raises:
        ValueError: for a keyword neither ``timecube`` nor the scene can take.
    """
    from digitalearth.interactive import InteractiveMap

    frame = _scene_kwargs(plan, InteractiveMap)
    scene = InteractiveMap(**frame)
    drawn, applied = _animation_kwargs(plan, scene.timecube, labels, consumed, frame)
    scene.timecube(obj, **drawn)
    _framed(scene, plan, frame, applied)
    if plan.path is not None:
        if plan.fps is None:
            return scene.save_animation(plan.path)
        return scene.save_animation(plan.path, fps=plan.fps)
    return scene.play() if plan.fps is None else scene.play(fps=plan.fps)


def _animated_web(
    obj: Any,
    plan: _Forwarded,
    destination: str,
    labels: Any,
    consumed: tuple[str, ...] = (),
) -> Any:
    """Animate a stack on the MapLibre tier: a time slider, then the written file.

    Args:
        obj: The stack to animate.
        plan: The translated call.
        destination: Where to write the animation, already resolved by
            :meth:`_Forwarded.destination` so the refusal precedes the engine import.
        labels: Step labels the data carried, used when the call named none.
        consumed: Keywords already spent choosing the data, so the builder is not offered them.

    Returns:
        Whatever ``save_animation`` returns for the destination it wrote.

    Raises:
        ValueError: for a keyword neither ``timeslider`` nor the scene can take.
    """
    from digitalearth.web import WebMap

    frame = _scene_kwargs(plan, WebMap)
    scene = WebMap(**frame)
    drawn, applied = _animation_kwargs(plan, scene.timeslider, labels, consumed, frame)
    scene.timeslider(obj, **drawn)
    _framed(scene, plan, frame, applied)
    if plan.fps is None:
        return scene.save_animation(destination)
    return scene.save_animation(destination, fps=plan.fps)


def _animation_kwargs(
    plan: _Forwarded,
    verb: Callable[..., Any],
    labels: Any = None,
    consumed: tuple[str, ...] = (),
    scened: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The data keywords for a temporal builder, checked against what it really accepts.

    Args:
        plan: The translated call.
        verb: The tier's temporal builder, bound to its scene.
        labels: Step labels the data carried; used only when the call named none of its own, so a
            caller's ``animation_axis_values`` always wins over a NetCDF's time axis.
        consumed: Keywords already spent choosing the data, left out rather than passed on.
        scened: Keywords the scene constructor already took, likewise left out.

    Returns:
        tuple: the keywords to pass the builder, and those the scene applies instead.

    Raises:
        ValueError: for a forwarded keyword neither can apply, or an ``animate="…"`` axis that
            contradicts an explicit ``kdim``.

    Examples:
        - The frame keywords are held back and the step labels folded in:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _animation_kwargs
            >>> def timecube(collection, *, band=1, labels=None, cmap=None):
            ...     ...
            >>> plan = _Forwarded.from_facade(
            ...     "interactive",
            ...     {"animate": True, "title": "T", "cmap": "magma", "animation_axis_values": [1, 2]},
            ... )
            >>> drawn, scened = _animation_kwargs(plan, timecube)
            >>> sorted(drawn.items()), scened
            ([('cmap', 'magma'), ('labels', [1, 2])], {})

            ```
        - A named animation axis arrives as the builder's ``kdim``:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _animation_kwargs
            >>> def timecube(collection, *, kdim="time", band=1):
            ...     ...
            >>> plan = _Forwarded.from_facade("interactive", {"animate": "step"})
            >>> _animation_kwargs(plan, timecube)[0]
            {'kdim': 'step'}

            ```
        - A keyword the builder has no parameter for, but the scene can apply, is handed to it:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _animation_kwargs
            >>> def slider(features, *, band=1):
            ...     ...
            >>> plan = _Forwarded.from_facade("web", {"animate": True, "colorbar": True})
            >>> _animation_kwargs(plan, slider)
            ({}, {'colorbar': True})

            ```

    See Also:
        _for_builder: Checks these against what the builder accepts.
        _scene_kwargs: The frame half, which this one deliberately leaves out.
    """
    held_back = set(_FRAME_KWARGS) | set(consumed) | set(scened or {})
    candidates = {
        name: value for name, value in plan.draw.items() if name not in held_back
    }
    held = plan.labels if plan.labels is not None else labels
    if held is not None:
        candidates["labels"] = held
    if plan.axis is not None:
        asked = candidates.get("kdim")
        if asked is not None and asked != plan.axis:
            # Overwriting it would animate a different dimension than the caller named in the
            # keyword they passed explicitly (round-2 M4).
            raise ValueError(
                f"animate={plan.axis!r} and kdim={asked!r} name different axes to step along; "
                f"pass one of them"
            )
        candidates["kdim"] = plan.axis
    return _for_builder(
        verb, candidates, f"animate(backend={plan.backend!r})", _SCENE_FALLBACK
    )


def _scene_kwargs(plan: _Forwarded, constructor: Callable[..., Any]) -> dict[str, Any]:
    """The forwarded keywords the tier's scene **constructor** takes.

    Every keyword the constructor accepts is offered to it, not just the frame ones. A tier scene is
    built inside this call, so a keyword it only takes at construction — the HoloViz tier's
    ``width`` / ``height`` / ``tiles`` / ``big_data_threshold``, the web tier's ``center`` / ``zoom``
    / ``style``, the matplotlib tier's ``figsize`` / ``domain`` / ``globe`` — is otherwise
    unreachable through ``plot(backend=…)``: there is no setter for it on the finished object. The
    facade forwards any further keyword verbatim, so they already arrive here; before this they were
    offered to the data builder, which has no such parameter, and refused.

    The constructor is asked first and the builder second, which is safe because the two parameter
    sets are disjoint on every tier (asserted by ``test_no_constructor_builder_collision``).

    Args:
        plan: The translated call.
        constructor: The tier's scene class, whose signature decides what it is handed.

    Returns:
        dict: the constructor keywords. One the constructor does not take is left for the builder,
            or for :func:`_framed` to apply through the tier's own verb — never dropped.

    Examples:
        - A constructor that takes a title is handed it; one that does not, is not:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _scene_kwargs
            >>> def takes_title(*, crs=None, title=None):
            ...     ...
            >>> def no_title(*, crs=None):
            ...     ...
            >>> plan = _Forwarded.from_facade("web", {"animate": True, "title": "Flood"})
            >>> sorted(_scene_kwargs(plan, takes_title))
            ['title']
            >>> sorted(_scene_kwargs(plan, no_title))
            []

            ```
        - A construction-time option reaches the constructor that declares it:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _scene_kwargs
            >>> def web_scene(*, center=None, zoom=None, style=None, crs=None):
            ...     ...
            >>> plan = _Forwarded.from_facade(
            ...     "web", {"animate": True, "zoom": 6, "style": "dark", "band": 0}
            ... )
            >>> sorted(_scene_kwargs(plan, web_scene).items())
            [('style', 'dark'), ('zoom', 6)]

            ```

    See Also:
        digitalearth.api._built: The same split on the still path, where `quickmap` builds the scene.
        _framed: Applies the frame keywords this one's constructor could not take.
    """
    accepted, _absorbs = _accepted_by(constructor)
    taken = {
        name: value
        for name, value in plan.draw.items()
        if name in accepted and value is not None
    }
    # `crs` and `title` are read off the record rather than `draw`, because `_framed` needs to know
    # whether the constructor consumed the title before it reaches for `set_title`.
    for name, value in (("crs", plan.crs), ("title", plan.title)):
        if name in accepted and value is not None:
            taken[name] = value
    return taken


def _framed(
    scene: Any,
    plan: _Forwarded,
    applied: Mapping[str, Any],
    scened: Mapping[str, Any] | None = None,
) -> None:
    """Apply the frame keywords the constructor could not take, through the scene's own verbs.

    ``basemap`` is a verb on every tier that has one, and a title is a verb on the tiers whose
    constructor has no ``title``. Routed here so neither is offered to a data builder, which is
    what raised ``RasterMixin.field() got an unexpected keyword argument 'title'`` on the web tier
    and a raw PyVista ``"title" is an invalid keyword argument`` on the 3-D one.

    Two of the refusals below cannot fire through the tiers this routes to today — both animating
    tiers own ``basemap`` and ``colorbar``, and the web scene owns ``set_title`` — so they guard a
    scene that loses a verb rather than describing live behaviour (round-2 L3).

    Args:
        scene: The built tier scene.
        plan: The translated call.
        applied: The keywords the constructor already consumed, so a title is not set twice.
        scened: Keywords the data builder had no parameter for, which this tier applies through a
            verb on the built scene instead — ``colorbar`` is the one the facades forward.

    Raises:
        ValueError: when a frame keyword has no route at all on this tier — named rather than
            dropped.

    Examples:
        - A requested basemap is applied through the scene's own verb:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _framed
            >>> class Scene:
            ...     def __init__(self):
            ...         self.asked = []
            ...     def basemap(self, source=None):
            ...         self.asked.append(source)
            >>> scene = Scene()
            >>> plan = _Forwarded.from_facade("web", {"animate": True, "basemap": "CartoDB"})
            >>> _framed(scene, plan, {})
            >>> scene.asked
            ['CartoDB']

            ```
        - A scene with no route for a frame keyword is told, not quietly skipped:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _framed
            >>> class Bare:
            ...     pass
            >>> plan = _Forwarded.from_facade("web", {"animate": True, "title": "Flood"})
            >>> try:
            ...     _framed(Bare(), plan, {})
            ... except ValueError as exc:
            ...     print(str(exc).split(": ")[1])
            this tier has neither a title argument nor a set_title verb

            ```

    See Also:
        _scene_kwargs: The frame keywords the constructor took, which this one skips.
    """
    if plan.basemap:
        basemap = getattr(scene, "basemap", None)
        if basemap is None:
            raise ValueError(
                f"basemap= cannot be applied on backend={plan.backend!r}: this tier has no basemap"
            )
        asked = _basemap_asked(plan.basemap, plan.backend)
        basemap() if asked is True else basemap(asked)
    if plan.title is not None and "title" not in applied:
        set_title = getattr(scene, "set_title", None)
        if set_title is None:
            raise ValueError(
                f"title= cannot be applied on backend={plan.backend!r}: this tier has neither a "
                f"title argument nor a set_title verb"
            )
        set_title(plan.title)
    for name, value in (scened or {}).items():
        if not value:
            # Neither tier shows a key unless asked, so a falsy `colorbar` needs no call — and
            # calling the verb to say "no" would add the very thing it is refusing.
            continue
        verb = getattr(scene, name, None)
        if verb is None:
            raise ValueError(
                f"{name}= cannot be applied on backend={plan.backend!r}: neither this render nor "
                f"the scene takes it"
            )
        verb()


def _categorical(obj: Any, column: str) -> bool:
    """Whether ``column`` holds labels rather than numbers.

    Read off the column's dtype kind rather than through pandas' type helpers, so this stays a
    two-line check on the array interface every vector layer already exposes.

    Args:
        obj: The vector layer the facade was called on.
        column: The attribute column named by the caller.

    Returns:
        bool: ``True`` when the column's values are strings or objects.

    Examples:
        - A string column is categorical; a numeric one is not:
            ```python
            >>> import numpy as np
            >>> from digitalearth.pyramids_backend import _categorical
            >>> layer = {"name": np.array(["a", "b"]), "pop": np.array([1.0, 2.0])}
            >>> _categorical(layer, "name"), _categorical(layer, "pop")
            (True, False)

            ```
        - A column the layer does not carry is not claimed either way:
            ```python
            >>> import numpy as np
            >>> from digitalearth.pyramids_backend import _categorical
            >>> _categorical({"pop": np.array([1.0])}, "absent")
            False

            ```

    See Also:
        _classified: Acts on this, per tier.
    """
    try:
        values = obj[column]
    except Exception:  # noqa: BLE001 - any lookup failure means "nothing to classify here"
        return False
    kind = getattr(getattr(values, "dtype", None), "kind", "")
    return kind in ("O", "U", "S")


def _classified(obj: Any, backend: str, draw: dict[str, Any]) -> None:
    """Make a label-valued ``column`` drawable, or refuse it on the tier that cannot.

    ``column`` is the ``FeatureCollection.plot`` facade's own selector, and colouring by a *label*
    column is the ordinary reason to pass it. Forwarded alone it reached the tier's classifier as a
    number: ``could not convert string to float: 'Baden-Württemberg'`` on web and 3-D. The web tier
    draws it once told the scale is categorical; the 3-D tier has no categorical fill at all, and
    both ways round it produced the same unreadable message (round-2 M10).

    Args:
        obj: The vector layer the facade was called on.
        backend: The backend the facade selected.
        draw: The keywords bound for the builder, modified in place.

    Raises:
        ValueError: on ``backend="3d"``, which cannot colour a label column by any route.

    Examples:
        - The web tier is told the scale is categorical, so a label column draws:
            ```python
            >>> import numpy as np
            >>> from digitalearth.pyramids_backend import _classified
            >>> layer = {"name": np.array(["a", "b"])}
            >>> draw = {"column": "name"}
            >>> _classified(layer, "web", draw)
            >>> sorted(draw.items())
            [('column', 'name'), ('scheme', 'categorical')]

            ```
        - A numeric column is left exactly as it came:
            ```python
            >>> import numpy as np
            >>> from digitalearth.pyramids_backend import _classified
            >>> draw = {"column": "pop"}
            >>> _classified({"pop": np.array([1.0, 2.0])}, "web", draw)
            >>> draw
            {'column': 'pop'}

            ```
        - The 3-D tier colours by a number, so a label column is refused there:
            ```python
            >>> import numpy as np
            >>> from digitalearth.pyramids_backend import _classified
            >>> try:
            ...     _classified({"name": np.array(["a"])}, "3d", {"column": "name"})
            ... except ValueError as exc:
            ...     print(str(exc).split(": ")[1].split(".")[0])
            it colours by a number

            ```

    See Also:
        _categorical: The dtype test this acts on.
    """
    column = draw.get("column")
    if column is None or not _categorical(obj, column):
        return
    if backend == "3d":
        raise ValueError(
            f"column={column!r} holds labels, and backend='3d' has no categorical fill: it "
            f"colours by a number. Draw the labels on backend='web' or 'interactive', or pass a "
            f"numeric column"
        )
    if backend == "web":
        # Executed: the web tier draws a label column once the scale is named categorical, and the
        # facade forwards no `scheme`, so there is no way for the caller to say it themselves.
        draw.setdefault("scheme", "categorical")


def _render(obj: Any, backend: str, forwarded: Mapping[str, Any]) -> Any:
    """Translate the facade's keywords and render ``obj`` on ``backend``.

    Args:
        obj: The pyramids object the facade was called on.
        backend: The backend name the seam passed.
        forwarded: The keywords pyramids forwarded.

    Returns:
        The tier's own object — a scene, a player, or a written animation path.

    Raises:
        ValueError: for a keyword this seam cannot serve, an animation the tier cannot drive, a
            still plot of a multi-member stack, which draws one field, or a label-valued
            ``column`` on the tier with no categorical fill.
    """
    plan = _Forwarded.from_facade(backend, forwarded)
    if plan.animate:
        return _animated(obj, plan)
    # Imported here, not at module scope: discovery imports this module, and `quickmap` reaches the
    # tier packages, whose engines must not be pulled into a bare `import pyramids`.
    from pyramids.dataset.collection import DatasetCollection

    from digitalearth import quickmap

    # Asked by type, for the reason given in `_animatable`: the attribute test this used to make
    # misread a `FeatureCollection`, whose columns resolve as attributes (round-2 H1).
    #
    # Not reachable through the facades as they stand: `DatasetCollection.plot(backend=…)` forwards
    # `animate=True` either way, so a collection always arrives on the animate path (round-2 M6).
    # Kept for a direct call, and because `quickmap` answers a collection with
    # `TypeError: quickplot cannot draw a DatasetCollection` — a function the caller never called.
    if isinstance(obj, DatasetCollection):
        raise ValueError(
            f"plot(backend={backend!r}) draws one field, and {type(obj).__name__} carries "
            f"several. Select one member and plot that, or call "
            f"animate(backend={backend!r}) to step through them"
        )
    draw = dict(plan.draw)
    # `title` is a frame keyword, not a data one: the web tier's raster builder answers it with
    # `RasterMixin.field() got an unexpected keyword argument 'title'` and the 3-D tier leaks
    # PyVista's `"title" is an invalid keyword argument for '_common_arg_parser'`. It is lifted out
    # here and set on the finished scene through `set_title`, which both of those tiers own. The
    # HoloViz tier has no `set_title`, so there it stays a builder keyword, which is the route that
    # already works (round-1 H2/L6).
    held = draw.pop("title", None) if backend in _TITLE_AFTER_BUILD else None
    if held is not None and not isinstance(held, str):
        # Checked before the render, not after. Set on the finished scene, a bad title cost the
        # whole picture: `three_d(ds, title=3)` built the Scene3D and *then* raised, while the web
        # tier took the int and returned a map — two answers to the same call (round-2 L5).
        raise ValueError(
            f"title={held!r} is not a heading; pass a string (backend={backend!r} sets it on the "
            f"finished scene)"
        )
    _classified(obj, backend, draw)
    if "basemap" in draw:
        draw["basemap"] = _basemap_asked(draw["basemap"], backend)
    scene = quickmap(obj, backend=backend, **draw)
    if held is not None:
        scene.set_title(held)
    return scene


def interactive(obj: Any, *, backend: str = "interactive", **kwargs: Any) -> Any:
    """Render a pyramids object on the HoloViz tier, for ``plot(backend="interactive")``.

    Args:
        obj: The ``Dataset`` / ``NetCDF`` / ``FeatureCollection`` / ``UgridDataset`` the facade was
            called on.
        backend: The backend name the seam passes; defaults to the name this is registered under.
        **kwargs: The keywords pyramids forwarded, in its own vocabulary.

    Returns:
        An :class:`~digitalearth.interactive.InteractiveMap`, or — for an animation call — its
        player, or the path the animation was written to.

    Raises:
        ValueError: for a forwarded keyword this seam cannot serve, or an unusable band/interval.

    Examples:
        - Reached through pyramids, which forwards its own keywords (needs the ``interactive``
          extra, so this one is shown rather than run):
            ```python
            from pyramids.dataset import Dataset

            ds = Dataset.read_file("examples/data/acc4000.tif")
            scene = ds.plot(backend="interactive", band=0, title="DEM")
            ```
        - A keyword this seam cannot serve is refused in digitalearth's own words, before any
          engine is imported, and says where the capability does live:
            ```python
            >>> from digitalearth.pyramids_backend import interactive
            >>> try:
            ...     interactive("a-dataset", backend="interactive", rgb_options={})
            ... except ValueError as exc:
            ...     print(str(exc).split("— ")[1].split(", which")[0])
            a three-band composite is built with Map.rgb_composite on the static tier

            ```
        - An animation keyword on a still plot is named rather than ignored, which is what it used
          to be:
            ```python
            >>> from digitalearth.pyramids_backend import interactive
            >>> try:
            ...     interactive("a-dataset", backend="interactive", interval=200)
            ... except ValueError as exc:
            ...     print(str(exc).split(". ")[0])
            interval= only applies to an animation; this is a still plot on backend='interactive'

            ```

    See Also:
        three_d: The same seam on the PyVista tier.
        web: The same seam on the MapLibre tier.
        ANIMATION_BACKENDS: Which tiers serve an ``animate`` call.
    """
    return _render(obj, backend, kwargs)


def three_d(obj: Any, *, backend: str = "3d", **kwargs: Any) -> Any:
    """Render a pyramids object on the PyVista tier, for ``plot(backend="3d")``.

    Args:
        obj: The pyramids object the facade was called on.
        backend: The backend name the seam passes; defaults to the name this is registered under.
        **kwargs: The keywords pyramids forwarded, in its own vocabulary.

    Returns:
        A :class:`~digitalearth.three_d.Scene3D`.

    Raises:
        ValueError: for a forwarded keyword this seam cannot serve, an unusable band, or an
            animation request — the 3-D tier's animation verbs cannot be driven from a plot call
            (see :data:`ANIMATION_BACKENDS`).

    Examples:
        - Reached through pyramids (needs the ``3d`` extra, so this one is shown rather than run):
            ```python
            from pyramids.dataset import Dataset

            ds = Dataset.read_file("examples/data/acc4000.tif")
            scene = ds.plot(backend="3d", band=0)
            ```
        - An animation request is refused by name, and names the two verbs that do animate a scene:
            ```python
            >>> from digitalearth.pyramids_backend import three_d
            >>> try:
            ...     three_d("a-dataset", backend="3d", animate=True)
            ... except ValueError as exc:
            ...     print(str(exc).split(":")[0])
            animate(backend='3d') is not served through the pyramids plot-backend seam

            ```
        - A band that is not a 0-based index is refused here rather than by the reader, far away:
            ```python
            >>> from digitalearth.pyramids_backend import three_d
            >>> try:
            ...     three_d("a-dataset", backend="3d", band=1.5)
            ... except ValueError as exc:
            ...     print(exc)
            band=1.5 is not a 0-based band index; pass an integer, where the first band is 0

            ```

    See Also:
        interactive: The same seam on the HoloViz tier, which does serve an animation.
        web: The same seam on the MapLibre tier.
        ANIMATION_BACKENDS: Why this tier is absent from it.
    """
    return _render(obj, backend, kwargs)


def web(obj: Any, *, backend: str = "web", **kwargs: Any) -> Any:
    """Render a pyramids object on the MapLibre tier, for ``plot(backend="web")``.

    Args:
        obj: The pyramids object the facade was called on.
        backend: The backend name the seam passes; defaults to the name this is registered under.
        **kwargs: The keywords pyramids forwarded, in its own vocabulary.

    Returns:
        A :class:`~digitalearth.web.WebMap`, or — for an animation call — the path the animation
        was written to.

    Raises:
        ValueError: for a forwarded keyword this seam cannot serve, an unusable band, or an
            animation call that named no ``path`` (the web tier animates only to a file).

    Examples:
        - Reached through pyramids (needs the ``web`` extra, so this one is shown rather than run):
            ```python
            from pyramids.dataset import Dataset

            ds = Dataset.read_file("examples/data/acc4000.tif")
            scene = ds.plot(backend="web", band=0, colorbar=True)
            ```
        - This tier animates only to a file, so an animation with no destination says what to pass:
            ```python
            >>> from digitalearth.pyramids_backend import web
            >>> try:
            ...     web("a-stack", backend="web", animate=True)
            ... except ValueError as exc:
            ...     print(str(exc).split(";")[0])
            animate(backend='web') writes the animation to a file, so it needs a path

            ```
        - Animating a single raster's bands is refused: this seam steps through a stack's members:
            ```python
            >>> from digitalearth.pyramids_backend import web
            >>> try:
            ...     web("a-dataset", backend="web", animate=True, bands=[0, 1])
            ... except ValueError as exc:
            ...     print(str(exc).split(": ")[1].split(".")[0])
            this seam animates a stack, stepping through its members

            ```

    See Also:
        interactive: The same seam on the HoloViz tier, whose animation needs no destination.
        three_d: The same seam on the PyVista tier.
        BAND_ORIGIN_SHIFT: The offset between the facade's band index and this package's.
    """
    return _render(obj, backend, kwargs)
