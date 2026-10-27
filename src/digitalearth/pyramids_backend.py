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
keyword through the route that tier actually has, and to refuse the rest **by name**, before a
scene is half-built — not to convert silence into a message, since the tiers were never silent.

Three translations are genuinely necessary, because the facade and this package count and spell
things differently:

* **``band`` is 0-based upstream and 1-based here.** pyramids documents the forwarded selector as
  "a **0-based** raster band index (band ``0`` is the first band, not band ``1``)", while
  digitalearth's public ``band=`` is 1-based everywhere. Forwarded unchanged, ``band=0`` reached the
  reader as index ``-1`` and raised ``band -1 is out of range for a 1-band dataset`` — the
  off-by-one reported in #501.
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
    "selectors": (
        "a NetCDF slice is chosen here with variable= and band=, not with read-time selectors"
    ),
    "axes": (
        "a NetCDF slice is chosen here with variable= and band=, not by naming its axes"
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
#: offered to the data builder, whose signature does not carry them.
_FRAME_KWARGS = ("crs", "title", "basemap")

#: Tiers whose raster builder refuses `title` but whose scene owns `set_title`, so a still plot's
#: title is applied after the scene is built. The HoloViz tier is absent because it has no
#: `set_title` and its builder does accept `title` (executed on all three tiers, round-1 H2).
_TITLE_AFTER_BUILD = frozenset({"web", "3d"})


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
    """
    parameters = inspect.signature(verb).parameters.values()
    # Only parameters with a default: the one without is the data itself, and listing it in a
    # refusal would invite a caller to pass their raster as a keyword.
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
    return frozenset(named) - {"self"}, absorbs


def _for_builder(
    verb: Callable[..., Any], candidates: Mapping[str, Any], named: str
) -> dict[str, Any]:
    """Split ``candidates`` against what ``verb`` accepts, refusing what it cannot take.

    Args:
        verb: The builder the keywords are destined for.
        candidates: The forwarded keywords left after the frame ones are routed.
        named: How to describe the builder in a refusal, e.g. ``"animate(backend='web')"``.

    Returns:
        dict: the keywords to pass on.

    Raises:
        ValueError: when a candidate is one the builder does not accept — named here, with the ones
            it does accept, instead of dropped in silence or sent on to a ``TypeError``.

    Examples:
        - Keywords the builder takes are passed through:
            ```python
            >>> from digitalearth.pyramids_backend import _for_builder
            >>> def slider(data, *, band=1, cmap="viridis"):
            ...     ...
            >>> _for_builder(slider, {"band": 2, "cmap": "magma"}, "animate(backend='web')")
            {'band': 2, 'cmap': 'magma'}

            ```
        - One it does not take is refused by name, listing what it will accept:
            ```python
            >>> from digitalearth.pyramids_backend import _for_builder
            >>> def slider(data, *, band=1, cmap="viridis"):
            ...     ...
            >>> try:
            ...     _for_builder(slider, {"colorbar": True}, "animate(backend='web')")
            ... except ValueError as exc:
            ...     print(exc)
            animate(backend='web') cannot apply colorbar=…; this render accepts band, cmap

            ```
    """
    accepted, absorbs = _accepted_by(verb)
    if absorbs:
        return dict(candidates)
    taken: dict[str, Any] = {}
    refused: list[str] = []
    for name, value in candidates.items():
        if name in accepted:
            taken[name] = value
        else:
            refused.append(name)
    if refused:
        offered = ", ".join(sorted(accepted))
        wanted = ", ".join(f"{name}=…" for name in sorted(refused))
        raise ValueError(
            f"{named} cannot apply {wanted}; this render accepts {offered}"
        )
    return taken


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
    for name, reason in _UNSERVED.items():
        if name in forwarded:
            raise ValueError(
                f"plot({name}=…) cannot be served on backend={backend!r}: {reason}. Drop it for "
                f"this backend, or render on backend='matplotlib'."
            )


def _shifted_band(band: Any) -> int:
    """Turn the facade's 0-based band index into digitalearth's 1-based one.

    Args:
        band: The forwarded ``band`` value — a 0-based index per the plot-backend contract.

    Returns:
        int: the same band as a 1-based index.

    Raises:
        ValueError: when ``band`` is not a non-negative integer, which no 0-based index is.

    Examples:
        - The facade's first band is this package's band 1:
            ```python
            >>> from digitalearth.pyramids_backend import _shifted_band
            >>> _shifted_band(0), _shifted_band(3)
            (1, 4)

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
    # bool is an int subclass, and `band=True` is a mistake worth naming rather than reading as 1.
    if isinstance(band, bool) or not isinstance(band, int):
        raise ValueError(
            f"band={band!r} is not a 0-based band index; pass an integer, where the first band "
            f"is 0"
        )
    if band < 0:
        raise ValueError(
            f"band={band} is not a 0-based band index; the first band is 0, so there is no "
            f"negative band"
        )
    return band + BAND_ORIGIN_SHIFT


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
        - A non-positive interval is refused rather than dividing by zero:
            ```python
            >>> try:
            ...     _rate(0)
            ... except ValueError as exc:
            ...     print(str(exc))
            interval=0 is not a frame interval in milliseconds; it must be positive

            ```
    """
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
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
            >>> plan.animate, plan.title, plan.draw["band"]
            (False, 'DEM', 1)

            ```
        - An animation's record separates the frame keywords from the draw ones:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded
            >>> plan = _Forwarded.from_facade(
            ...     "interactive", {"animate": True, "interval": 200, "cmap": "viridis"}
            ... )
            >>> plan.animate, plan.fps, plan.draw
            (True, 5.0, {'cmap': 'viridis'})

            ```

    See Also:
        _Forwarded.from_facade: Builds one from a facade call, applying every translation.
        _Forwarded.destination: The resolved animation path, for the tiers that write one.
    """

    backend: str
    animate: bool
    draw: Mapping[str, Any] = field(default_factory=dict)
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
                >>> plan.animate, sorted(plan.draw.items())
                (False, [('band', 1), ('title', 'DEM')])

                ```
            - An animation call lifts the animation keywords out of the draw kwargs:
                ```python
                >>> plan = _Forwarded.from_facade(
                ...     "interactive", {"animate": True, "interval": 250, "path": "out.gif"}
                ... )
                >>> plan.animate, plan.fps, plan.path, plan.draw
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
                ...     print("only applies to an animation" in str(exc))
                True

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
                    raise ValueError(
                        f"plot({name}=…) only applies to an animation; this is a still plot on "
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
        if "variable_name" in rest:
            rest["variable"] = rest.pop("variable_name")
        if "band" in rest:
            rest["band"] = _shifted_band(rest["band"])
        return cls(
            backend=backend,
            animate=animate,
            draw=rest,
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


def _is_stack(obj: Any) -> bool:
    """Whether ``obj`` carries the members a temporal builder steps through.

    A ``DatasetCollection`` holds its members on ``datasets``; a single ``Dataset`` does not, which
    is the difference the stack builders depend on. Asked by attribute rather than by type so this
    module keeps importing nothing from pyramids — discovery imports it on every
    ``import pyramids``.

    Args:
        obj: The object the facade was called on.

    Returns:
        bool: ``True`` when it exposes members to step through.

    Examples:
        - An object holding members reads as a stack; a bare one does not:
            ```python
            >>> from digitalearth.pyramids_backend import _is_stack
            >>> class Stack:
            ...     datasets = ["a", "b"]
            >>> class Single:
            ...     pass
            >>> _is_stack(Stack()), _is_stack(Single())
            (True, False)

            ```
    """
    return hasattr(obj, "datasets")


def _refuse_unstacked(obj: Any, plan: _Forwarded) -> None:
    """Refuse an animation of something with no steps, naming what this seam animates.

    Args:
        obj: The object the facade was called on.
        plan: The translated call.

    Raises:
        ValueError: when ``obj`` carries no members to step through.

    Examples:
        - A single raster has nothing to step through, and is told so:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _refuse_unstacked
            >>> plan = _Forwarded.from_facade("interactive", {"animate": True})
            >>> class Single:
            ...     pass
            >>> try:
            ...     _refuse_unstacked(Single(), plan)
            ... except ValueError as exc:
            ...     print(str(exc).split(":")[1].split(".")[0].strip())
            a Single carries no steps to animate

            ```
    """
    if not _is_stack(obj):
        raise ValueError(
            f"animate(backend={plan.backend!r}): a {type(obj).__name__} carries no steps to "
            f"animate. Pass a DatasetCollection of the steps, or animate on "
            f"backend='matplotlib', which animates a single raster's bands"
        )


def _animated(obj: Any, plan: _Forwarded) -> Any:
    """Build the tier's temporal scene and return its animation.

    Args:
        obj: The pyramids object the facade was called on — a stack, since a single field has no
            steps to animate.
        plan: The translated call.

    Returns:
        The interactive tier's player, or the path the animation was written to.

    Raises:
        ValueError: propagated from :func:`_refuse_unanimatable` for a tier with no verb to drive,
            from :func:`_refuse_unstacked` for an object with no steps, or from
            :meth:`_Forwarded.destination` for a web animation with no destination.
    """
    _refuse_unanimatable(plan)
    if plan.backend == "interactive":
        _refuse_unstacked(obj, plan)
        return _animated_interactive(obj, plan)
    # The destination is resolved first, before the tier is imported and before the payload is
    # examined: it is a property of the call rather than of the data, so a pathless animation is
    # answered the same way whatever was passed, and never pays for the engine.
    destination = plan.destination()
    _refuse_unstacked(obj, plan)
    return _animated_web(obj, plan, destination)


def _animated_interactive(obj: Any, plan: _Forwarded) -> Any:
    """Animate a stack on the HoloViz tier: a time cube, then its player or its export.

    Args:
        obj: The stack to animate.
        plan: The translated call.

    Returns:
        A Panel player when the call named no ``path``, else whatever ``save_animation`` returns
        for the destination it wrote.

    Raises:
        ValueError: for a keyword neither the scene nor ``timecube`` can take.
    """
    from digitalearth.interactive import InteractiveMap

    frame = _scene_kwargs(plan, InteractiveMap)
    scene = InteractiveMap(**frame)
    scene.timecube(obj, **_animation_kwargs(plan, scene.timecube))
    _framed(scene, plan, frame)
    if plan.path is not None:
        if plan.fps is None:
            return scene.save_animation(plan.path)
        return scene.save_animation(plan.path, fps=plan.fps)
    return scene.play() if plan.fps is None else scene.play(fps=plan.fps)


def _animated_web(obj: Any, plan: _Forwarded, destination: str) -> Any:
    """Animate a stack on the MapLibre tier: a time slider, then the written file.

    Args:
        obj: The stack to animate.
        plan: The translated call.
        destination: Where to write the animation, already resolved by
            :meth:`_Forwarded.destination` so the refusal precedes the engine import.

    Returns:
        Whatever ``save_animation`` returns for the destination it wrote.

    Raises:
        ValueError: for a keyword neither the scene nor ``timeslider`` can take — ``colorbar`` is
            the one the facades forward that this tier's slider has no parameter for.
    """
    from digitalearth.web import WebMap

    frame = _scene_kwargs(plan, WebMap)
    scene = WebMap(**frame)
    scene.timeslider(obj, **_animation_kwargs(plan, scene.timeslider))
    _framed(scene, plan, frame)
    if plan.fps is None:
        return scene.save_animation(destination)
    return scene.save_animation(destination, fps=plan.fps)


def _animation_kwargs(plan: _Forwarded, verb: Callable[..., Any]) -> dict[str, Any]:
    """The data keywords for a temporal builder, checked against what it really accepts.

    Args:
        plan: The translated call.
        verb: The tier's temporal builder, bound to its scene.

    Returns:
        dict: the keywords to pass, including the step ``labels`` and the ``kdim`` a named
            animation axis asks for.

    Raises:
        ValueError: for a forwarded keyword the builder cannot apply.
    """
    candidates = {
        name: value for name, value in plan.draw.items() if name not in _FRAME_KWARGS
    }
    if plan.labels is not None:
        candidates["labels"] = plan.labels
    if plan.axis is not None:
        candidates["kdim"] = plan.axis
    return _for_builder(verb, candidates, f"animate(backend={plan.backend!r})")


def _scene_kwargs(plan: _Forwarded, constructor: Callable[..., Any]) -> dict[str, Any]:
    """The frame keywords the tier's scene constructor really takes.

    Args:
        plan: The translated call.
        constructor: The tier's scene class, whose signature decides what it is handed.

    Returns:
        dict: the constructor keywords. A frame keyword the constructor does not take is left for
            :func:`_framed` to apply through the tier's own verb, not dropped.

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
    """
    accepted, _absorbs = _accepted_by(constructor)
    offered = {"crs": plan.crs, "title": plan.title}
    return {
        name: value
        for name, value in offered.items()
        if name in accepted and value is not None
    }


def _framed(scene: Any, plan: _Forwarded, applied: Mapping[str, Any]) -> None:
    """Apply the frame keywords the constructor could not take, through the scene's own verbs.

    ``basemap`` is a verb on every tier that has one, and a title is a verb on the tiers whose
    constructor has no ``title``. Routed here so neither is offered to a data builder, which is
    what raised ``RasterMixin.field() got an unexpected keyword argument 'title'`` on the web tier
    and a raw PyVista ``"title" is an invalid keyword argument`` on the 3-D one.

    Args:
        scene: The built tier scene.
        plan: The translated call.
        applied: The keywords the constructor already consumed, so a title is not set twice.

    Raises:
        ValueError: when a frame keyword has no route at all on this tier — named rather than
            dropped.
    """
    if plan.basemap:
        basemap = getattr(scene, "basemap", None)
        if basemap is None:
            raise ValueError(
                f"basemap= cannot be applied on backend={plan.backend!r}: this tier has no basemap"
            )
        basemap() if plan.basemap is True else basemap(plan.basemap)
    if plan.title is not None and "title" not in applied:
        set_title = getattr(scene, "set_title", None)
        if set_title is None:
            raise ValueError(
                f"title= cannot be applied on backend={plan.backend!r}: this tier has neither a "
                f"title argument nor a set_title verb"
            )
        set_title(plan.title)


def _render(obj: Any, backend: str, forwarded: Mapping[str, Any]) -> Any:
    """Translate the facade's keywords and render ``obj`` on ``backend``.

    Args:
        obj: The pyramids object the facade was called on.
        backend: The backend name the seam passed.
        forwarded: The keywords pyramids forwarded.

    Returns:
        The tier's own object — a scene, a player, or a written animation path.

    Raises:
        ValueError: for a keyword this seam cannot serve, an animation the tier cannot drive, or a
            still plot of a multi-member stack, which draws one field.
    """
    plan = _Forwarded.from_facade(backend, forwarded)
    if plan.animate:
        return _animated(obj, plan)
    # Imported here, not at module scope: discovery imports this module, and `quickmap` reaches the
    # tier packages, whose engines must not be pulled into a bare `import pyramids`.
    from digitalearth import quickmap

    if _is_stack(obj):
        # `DatasetCollection.plot(backend=…)` is one of the facades, and `quickmap` cannot draw a
        # collection — it answered with `TypeError: quickplot cannot draw a DatasetCollection`,
        # naming a function the caller never called. Said here in terms of the call they did make
        # (round-1 M5).
        raise ValueError(
            f"plot(backend={backend!r}) draws one field, and a "
            f"{type(obj).__name__} carries several. Select one member and plot that, or call "
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
            ...     print(str(exc).split(": ")[1].split(". Drop")[0])
            a three-band composite is built with Map.rgb_composite on the static tier, which this seam does not reach

            ```
        - An animation keyword on a still plot is named rather than ignored, which is what it used
          to be:
            ```python
            >>> from digitalearth.pyramids_backend import interactive
            >>> try:
            ...     interactive("a-dataset", backend="interactive", interval=200)
            ... except ValueError as exc:
            ...     print(str(exc).split(". ")[0])
            plot(interval=…) only applies to an animation; this is a still plot on backend='interactive'

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
