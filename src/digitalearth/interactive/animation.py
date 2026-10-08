"""AnimationMixin — animation playback & export for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``play`` and ``save_animation`` (DI.11), the interactive counterparts of ``Map.animate``/``rotate``.
They operate on the ``hv.DynamicMap`` a :meth:`~digitalearth.interactive.temporal.TemporalMixin.timecube`
registered: ``play`` binds a ``panel.widgets.Player`` to its time kdim for auto-advancing playback, and
``save_animation`` materialises the (lazy) DynamicMap to a finite ``HoloMap`` and writes a GIF/MP4 via the
matplotlib backend or a client-side **scrubber** HTML that animates offline with no server.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from digitalearth.base.animation import DEFAULT_FPS
from digitalearth.interactive.base import _require_holoviz

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class AnimationMixin(_MixinBase):
    """Animation builders (DI.11): Player playback + GIF/MP4/scrubber export of a time cube.

    A capability mixin of :class:`~digitalearth.interactive.map.InteractiveMap`: it is only ever composed into that
    map class, never instantiated or subclassed on its own. Its methods reach the element registry, the display CRS
    and the render/save lifecycle — and the sibling mixins' methods — through ``self``, and only the composition
    supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.interactive.base.InteractiveMapBase`, the state
    ``InteractiveMap`` inherits. At runtime that base is plain ``object``, so composing this mixin leaves the
    ``InteractiveMap`` MRO exactly what it was before the annotation.

    See Also:
        digitalearth.interactive.map.InteractiveMap: the composition that supplies the state these methods use.
        digitalearth.interactive.base.InteractiveMapBase: the typing-only base declared above the class.
    """

    def frames(
        self,
        frames: Any,
        *,
        dimension: str = "frame",
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Register any sequence of elements as one animatable **temporal layer** (IN-12).

        The Selection-frame layer model the gap names: ``play``/``save_animation`` could only drive the
        raster ``hv.DynamicMap`` a :meth:`~digitalearth.interactive.temporal.TemporalMixin.timecube` built,
        so "animate any temporal layer" was unreachable for, say, a time-varying *vector* layer. This folds
        an arbitrary frame set — a mapping ``{key: element}`` or a plain sequence — into one ``hv.HoloMap``
        keyed by ``dimension``, a selectable-by-key layer that :meth:`play` and :meth:`save_animation` then
        animate exactly like a time cube, whatever the element type.

        Args:
            frames: A ``{key: element}`` mapping (keys become the frame selector values, e.g. datetimes) or a
                sequence of elements (enumerated 0, 1, 2, …). Each element is any HoloViews/GeoViews element.
            dimension: The key dimension's name — the axis the player/scrubber steps along.
            name: The caller's own name for the layer, used as its id and label; ``None`` generates one, and
                a name already on the map is suffixed ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn; ``False`` builds and describes it hidden (#327).

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``frames`` is empty — there is nothing to animate.
        """
        _gv, hv = _require_holoviz()
        items = dict(frames) if isinstance(frames, Mapping) else dict(enumerate(frames))
        if not items:
            raise ValueError("frames() needs at least one frame to animate")
        holomap = hv.HoloMap(items, kdims=[dimension])
        self.add_layer(holomap, name=name, visible=visible, kind="custom:holoviews")
        return self

    def _animatable_layers(self) -> list:
        """Return every temporal layer that can be animated, as ``(layer_id, element)`` (IN-12).

        A temporal layer is a registered ``hv.DynamicMap`` **or ``hv.HoloMap``** with key dimensions — a
        ``timecube`` or a :meth:`frames` layer today, and any other slider-driven layer a future builder
        registers. Listing **all** of them is what lifts the old "first ``DynamicMap`` only" limit: a map
        with two temporal layers (of any element type) can animate either by id.

        Returns:
            The animatable layers in draw order, each as ``(layer_id, element)``.
        """
        gv, hv = _require_holoviz()
        return [
            (layer_id, element)
            for layer_id, element in zip(self.layer_ids, self.layers)
            # `DynamicMap` subclasses `HoloMap`, so this one check catches a timecube's DynamicMap and a
            # frames() HoloMap alike; `.kdims` excludes a static element with no selector.
            if isinstance(element, hv.HoloMap) and element.kdims
        ]

    def _time_dynamicmap(self, layer: str | None = None) -> Any:
        """Return the temporal ``hv.DynamicMap`` to animate — a named one, or the first (IN-12).

        Args:
            layer: The id of the temporal layer to animate; ``None`` takes the first animatable layer in
                draw order (the behaviour before a map could carry more than one).

        Returns:
            The chosen ``hv.DynamicMap``.

        Raises:
            ValueError: when no layer is animatable, or when ``layer`` names one that is not — the message
                lists the layers that are.
        """
        animatable = self._animatable_layers()
        if layer is not None:
            for layer_id, element in animatable:
                if layer_id == layer:
                    return element
            raise ValueError(
                f"layer {layer!r} is not an animatable temporal layer; the animatable layers are "
                f"{[layer_id for layer_id, _ in animatable]}"
            )
        if not animatable:
            raise ValueError(
                "no time cube to animate — call timecube(collection) before play()/save_animation()"
            )
        return animatable[0][1]

    def _to_holomap(self, dmap: Any) -> Any:
        """Materialise a lazy ``DynamicMap`` into a finite ``HoloMap`` (every frame evaluated).

        Args:
            dmap: The time-cube ``DynamicMap``.

        Returns:
            An ``hv.HoloMap`` over the same kdim — exportable as a finite animation.
        """
        gv, hv = _require_holoviz()
        # A frames() layer is already a finite HoloMap — return it unchanged. Only a lazy DynamicMap (a
        # timecube) needs materialising, and its frame keys live on the kdim's explicit `.values`.
        if isinstance(dmap, hv.HoloMap) and not isinstance(dmap, hv.DynamicMap):
            return dmap
        keys = list(dmap.kdims[0].values)
        return hv.HoloMap({key: dmap[key] for key in keys}, kdims=dmap.kdims)

    def play(
        self, *, fps: float = DEFAULT_FPS, loop: bool = True, layer: str | None = None
    ) -> Any:
        """Wrap the time cube in a Panel layout with an auto-advancing ``Player`` widget.

        Args:
            fps: Playback frames per second (the Player interval). Defaults to
                :data:`~digitalearth.base.animation.DEFAULT_FPS`, the one rate every tier reads (#256), so
                the same animation plays at the same speed on every backend.
            loop: Loop at the end (``True``) or stop (``False``).
            layer: The id of the temporal layer to animate (IN-12); ``None`` animates the first one, which
                is the only one most maps carry.

        Returns:
            A ``panel.viewable.Viewable`` hosting the map + a bound time ``Player``.

        Raises:
            ValueError: when no ``timecube`` layer has been added.
            ImportError: when the ``interactive`` extra (which provides panel) is absent.

        Examples:
            - Play a time cube at the shared default speed; the player steps through the cube's
              own time keys, not a frame index:
                ```python
                >>> from pyramids.dataset.collection import DatasetCollection     # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> cube = DatasetCollection.from_files(["jan.tif", "feb.tif"])   # doctest: +SKIP
                >>> app = InteractiveMap().timecube(cube).play()                  # doctest: +SKIP
                >>> app[1].value == app[1].options[0]   # starts on step one  # doctest: +SKIP
                True

                ```
            - ``fps`` is the playback rate, translated into the Player's millisecond interval —
              two frames a second is a 500 ms step:
                ```python
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> m = InteractiveMap().timecube(cube)                          # doctest: +SKIP
                >>> app = m.play(fps=2.0, loop=False)                            # doctest: +SKIP
                >>> app[1].interval                                               # doctest: +SKIP
                500
                >>> app[1].loop_policy                                            # doctest: +SKIP
                'once'

                ```
            - Playback needs a time cube; anything else is refused rather than played empty:
                ```python
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> try:                                                          # doctest: +SKIP
                ...     InteractiveMap().play()
                ... except ValueError as error:
                ...     print(str(error).split(" — ")[0])
                no time cube to animate

                ```
        """
        import panel as pn

        gv, hv = _require_holoviz()
        dmap = self._time_dynamicmap(layer)
        # A timecube DynamicMap carries its frame keys on the kdim's explicit `.values`; a frames() HoloMap
        # carries them as its own keys instead. Take whichever is populated so both kinds play (IN-12).
        values = list(dmap.kdims[0].values) or list(dmap.keys())
        # DiscretePlayer (not Player) steps through arbitrary labelled values (ints / datetimes).
        player = pn.widgets.DiscretePlayer(
            options=values,
            value=values[0],
            interval=max(1, int(1000 / fps)),
            loop_policy="loop" if loop else "once",
        )
        view = pn.bind(lambda value: dmap[value], player)
        return pn.Column(pn.panel(view), player)

    def save_animation(
        self,
        path: Any,
        *,
        fps: float = DEFAULT_FPS,
        layer: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Export the time cube as a GIF/MP4 (matplotlib backend) or a scrubber HTML.

        ``.gif``/``.mp4`` materialise the DynamicMap to a finite ``HoloMap`` and render via the
        matplotlib backend (``.mp4`` needs ffmpeg). ``.html`` writes a client-side **scrubber** that
        plays offline with no server.

        Args:
            path: Output file (``.gif`` / ``.mp4`` / ``.html``), as ``str`` or ``pathlib.Path``.
            fps: Frames per second. Defaults to :data:`~digitalearth.base.animation.DEFAULT_FPS`, the
                one rate every tier reads (#256).
            layer: The id of the temporal layer to export (IN-12); ``None`` exports the first one.
            **kwargs: Forwarded to :func:`holoviews.save`.

        Returns:
            pathlib.Path: the file written, matching :meth:`~digitalearth.interactive.base.\
InteractiveMapBase.save` (#248).

        Raises:
            ValueError: when no ``timecube`` layer has been added.

        Examples:
            - Export a GIF; the returned ``Path`` can be inspected or moved without re-wrapping it
              (#248 — this used to come back as a bare ``str``):
                ```python
                >>> from pyramids.dataset.collection import DatasetCollection     # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> cube = DatasetCollection.from_files(["jan.tif", "feb.tif"])   # doctest: +SKIP
                >>> m = InteractiveMap().timecube(cube)                          # doctest: +SKIP
                >>> out = m.save_animation("rain.gif")                           # doctest: +SKIP
                >>> out.suffix, out.exists()                                      # doctest: +SKIP
                ('.gif', True)

                ```
            - The suffix picks the writer: ``.html`` is the client-side scrubber, which needs no
              server and no ffmpeg, and ``fps`` sets its playback rate:
                ```python
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> out = InteractiveMap().timecube(cube).save_animation(         # doctest: +SKIP
                ...     "rain.html", fps=8.0
                ... )
                >>> out.name                                                      # doctest: +SKIP
                'rain.html'
                >>> "<html" in out.read_text(encoding="utf-8")[:200].lower()      # doctest: +SKIP
                True

                ```
            - Exporting needs a time cube, same as :meth:`play`:
                ```python
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> try:                                                          # doctest: +SKIP
                ...     InteractiveMap().save_animation("nothing.gif")
                ... except ValueError as error:
                ...     print(str(error).split(" — ")[0])
                no time cube to animate

                ```
        """
        gv, hv = _require_holoviz()
        holomap = self._to_holomap(self._time_dynamicmap(layer))
        suffix = str(path).lower().rsplit(".", 1)[-1]
        if suffix == "html":
            hv.save(holomap, path, fmt="scrubber", fps=fps, **kwargs)
        else:
            hv.save(holomap, path, backend="matplotlib", fps=fps, **kwargs)
        return Path(path)
