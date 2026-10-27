"""Serve pyramids' ``plot(backend=…)`` seam from digitalearth's three non-matplotlib tiers.

pyramids-core resolves a non-matplotlib ``backend=`` through the ``pyramids.plot_backends``
entry-point group: ``Dataset.plot(backend="3d")``, ``NetCDF.plot(backend="interactive")`` and the
rest hand the object to whichever callable is registered under that name. This module is the
registered end of that seam — one callable per tier, each matching the contract shape
``fn(obj, *, backend, **kwargs)``.

Registering it is not merely cosmetic. Without an entry point pyramids falls back to *importing*
``digitalearth`` by name and calling ``quickmap`` directly, which forwards its own vocabulary
verbatim; three of those keywords then reach cleopatra and are refused in cleopatra's words, naming
cleopatra's allowlist to someone who only ever wrote ``ds.plot(backend="web")``. The adapter exists
to translate between the two vocabularies, and to refuse what digitalearth cannot honour **by name**
rather than leaking a lower layer's complaint or silently dropping the argument.

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

from collections.abc import Mapping
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
#: Silence is the alternative this avoids: pyramids drops a defaulted argument before forwarding, so
#: anything arriving here was set deliberately and must not be ignored.
_UNSERVED: Mapping[str, str] = {
    "exclude_value": (
        "digitalearth masks a raster through its own nodata handling and passes its own "
        "exclude_value to the renderer"
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

#: Keywords that only mean something in an animation call; passed to a still plot they would be
#: accepted and ignored, which is the silent failure this names instead. `path` is here because a
#: still render has nothing to write — only the animation verbs take a destination.
_ANIMATION_ONLY = ("bands", "animation_axis_values", "interval", "path")

#: What each tier's scene **constructor** accepts out of the forwarded set. Both tiers take `crs`;
#: only the interactive one takes a `title` (the web tier sets it on the built scene), and neither
#: takes a `basemap` — that is a verb on the scene in both, as `_quickmap_*` already does it.
_SCENE_KWARGS: Mapping[str, tuple[str, ...]] = {
    "interactive": ("crs", "title"),
    "web": ("crs",),
}

#: What each tier's temporal builder accepts out of the forwarded set. `column` is a web-tier
#: selector: `timeslider` colours a vector series by it, while `timecube` has no such argument.
_TEMPORAL_KWARGS: Mapping[str, tuple[str, ...]] = {
    "interactive": ("band", "cmap", "colorbar"),
    "web": ("band", "cmap", "colorbar", "column"),
}


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
    """

    backend: str
    animate: bool
    draw: Mapping[str, Any] = field(default_factory=dict)
    fps: float | None = None
    path: str | None = None
    labels: Any = None

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
        animate = bool(rest.pop("animate", False))
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
            or a web animation with no destination.
    """
    _refuse_unanimatable(plan)
    if plan.backend == "interactive":
        return _animated_interactive(obj, plan)
    # Resolved before the tier is imported and before any scene is built: this tier animates only to
    # a file, so a call with no destination can never succeed and should not pay for the engine.
    return _animated_web(obj, plan, plan.destination())


def _animated_interactive(obj: Any, plan: _Forwarded) -> Any:
    """Animate a stack on the HoloViz tier: a time cube, then its player or its export.

    Args:
        obj: The stack to animate.
        plan: The translated call.

    Returns:
        A Panel player when the call named no ``path``, else whatever ``save_animation`` returns
        for the destination it wrote.
    """
    from digitalearth.interactive import InteractiveMap

    scene = InteractiveMap(**_scene_kwargs(plan))
    scene.timecube(obj, **_temporal_kwargs(plan))
    # The tier applies a basemap through its own verb rather than a constructor argument, exactly as
    # `_quickmap_interactive` does — the constructor takes only the frame (crs/title).
    if plan.basemap:
        scene.basemap() if plan.basemap is True else scene.basemap(plan.basemap)
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
    """
    from digitalearth.web import WebMap

    scene = WebMap(**_scene_kwargs(plan))
    scene.timeslider(obj, **_temporal_kwargs(plan))
    if plan.basemap:
        scene.basemap() if plan.basemap is True else scene.basemap(plan.basemap)
    # This tier's constructor carries no title; it is set on the built scene instead.
    if plan.title is not None:
        scene.set_title(plan.title)
    if plan.fps is None:
        return scene.save_animation(destination)
    return scene.save_animation(destination, fps=plan.fps)


def _scene_kwargs(plan: _Forwarded) -> dict[str, Any]:
    """The forwarded keywords the selected tier's scene **constructor** accepts.

    Args:
        plan: The translated call.

    Returns:
        dict: the constructor keywords, dropping the ones that tier takes through a verb instead.

    Examples:
        - The interactive constructor carries the title; the web one does not:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _scene_kwargs
            >>> call = {"animate": True, "title": "Flood", "basemap": True, "band": 0}
            >>> sorted(_scene_kwargs(_Forwarded.from_facade("interactive", dict(call))))
            ['title']
            >>> sorted(_scene_kwargs(_Forwarded.from_facade("web", dict(call))))
            []

            ```
    """
    accepted = _SCENE_KWARGS.get(plan.backend, ())
    taken = {"crs": plan.crs, "title": plan.title}
    return {
        name: value
        for name, value in taken.items()
        if name in accepted and value is not None
    }


def _temporal_kwargs(plan: _Forwarded) -> dict[str, Any]:
    """The forwarded keywords the selected tier's temporal builder accepts.

    Args:
        plan: The translated call.

    Returns:
        dict: the builder keywords, including the per-step ``labels`` when the facade sent them.

    Examples:
        - The colour keywords and the step labels carry over; the frame ones do not:
            ```python
            >>> from digitalearth.pyramids_backend import _Forwarded, _temporal_kwargs
            >>> plan = _Forwarded.from_facade(
            ...     "interactive",
            ...     {"animate": True, "cmap": "viridis", "title": "T", "animation_axis_values": [1, 2]},
            ... )
            >>> sorted(_temporal_kwargs(plan).items())
            [('cmap', 'viridis'), ('labels', [1, 2])]

            ```
        - A vector ``column`` is a web-tier selector only, since the time cube has no such argument:
            ```python
            >>> call = {"animate": True, "column": "depth"}
            >>> _temporal_kwargs(_Forwarded.from_facade("web", dict(call)))
            {'column': 'depth'}
            >>> _temporal_kwargs(_Forwarded.from_facade("interactive", dict(call)))
            {}

            ```
    """
    accepted = _TEMPORAL_KWARGS.get(plan.backend, ())
    taken = {name: value for name, value in plan.draw.items() if name in accepted}
    if plan.labels is not None:
        taken["labels"] = plan.labels
    return taken


def _render(obj: Any, backend: str, forwarded: Mapping[str, Any]) -> Any:
    """Translate the facade's keywords and render ``obj`` on ``backend``.

    Args:
        obj: The pyramids object the facade was called on.
        backend: The backend name the seam passed.
        forwarded: The keywords pyramids forwarded.

    Returns:
        The tier's own object — a scene, a player, or a written animation path.

    Raises:
        ValueError: for a keyword this seam cannot serve, or an animation the tier cannot drive.
    """
    plan = _Forwarded.from_facade(backend, forwarded)
    if plan.animate:
        return _animated(obj, plan)
    # Imported here, not at module scope: discovery imports this module, and `quickmap` reaches the
    # tier packages, whose engines must not be pulled into a bare `import pyramids`.
    from digitalearth import quickmap

    return quickmap(obj, backend=backend, **plan.draw)


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
        - The callable conforms to the seam's render shape, so discovery never skips it:
            ```python
            >>> import inspect
            >>> from digitalearth.pyramids_backend import interactive
            >>> bound = inspect.signature(interactive).bind("a-dataset", backend="interactive")
            >>> sorted(bound.arguments)
            ['backend', 'obj']

            ```
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
        - The callable conforms to the seam's render shape:
            ```python
            >>> import inspect
            >>> from digitalearth.pyramids_backend import three_d
            >>> bound = inspect.signature(three_d).bind("a-dataset", backend="3d")
            >>> sorted(bound.arguments)
            ['backend', 'obj']

            ```
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
        - The callable conforms to the seam's render shape:
            ```python
            >>> import inspect
            >>> from digitalearth.pyramids_backend import web
            >>> bound = inspect.signature(web).bind("a-dataset", backend="web")
            >>> sorted(bound.arguments)
            ['backend', 'obj']

            ```
    """
    return _render(obj, backend, kwargs)
