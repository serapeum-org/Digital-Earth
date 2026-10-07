"""LiveMixin — streaming / live-updating layers for :class:`~digitalearth.interactive.map.InteractiveMap` (IN-17).

A normal builder draws a fixed snapshot. A **live** layer is backed by a HoloViews ``Pipe`` or ``Buffer``:
the caller pushes new data into it with :meth:`LiveMixin.push`, and only the changed glyphs redraw — Bokeh
patches the layer's ``ColumnDataSource`` in place rather than the map rebuilding — which is the IN-2
reconcile applied to a data feed (a gauge, a moving vehicle, a sensor stream). The push reaches the browser
only under a live kernel/server; a saved HTML captures the current snapshot.

A ``Pipe`` carries whatever you send it (the push replaces the layer's data); a ``Buffer`` **appends** and
keeps a sliding window of the last ``length`` rows, for a feed you want to tail. Data is taken in the display
CRS, like a hand-built element handed to ``add_layer`` — a live feed is pushed too often to reproject through
pyramids on every tick, so reproject once at the source.
"""

from typing import TYPE_CHECKING, Any, Self, cast

from digitalearth.interactive.base import _require_holoviz

#: The element kinds a live layer can draw — geospatial point feeds and growing tracks.
_LIVE_KINDS: tuple[str, ...] = ("points", "path")


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class LiveMixin(_MixinBase):
    """Streaming builders (IN-17): layers backed by a HoloViews ``Pipe``/``Buffer`` you push into.

    A capability mixin of :class:`~digitalearth.interactive.map.InteractiveMap`: it is only ever composed into that
    map class, never instantiated or subclassed on its own. Its methods reach the element registry, the display CRS
    and the render/save lifecycle — and the sibling mixins' methods — through ``self``, and only the composition
    supplies those.

    See Also:
        digitalearth.interactive.map.InteractiveMap: the composition that supplies the state these methods use.
        digitalearth.interactive.base.InteractiveMapBase: the typing-only base declared above the class.
    """

    def live(
        self,
        *,
        kind: str = "points",
        data: Any = None,
        buffer: bool = False,
        length: int = 1000,
        name: str | None = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a live layer backed by a ``Pipe`` (replace) or ``Buffer`` (append) you push into (IN-17).

        The layer is a ``DynamicMap`` that re-renders from the stream, so :meth:`push` updates only its glyphs
        rather than rebuilding the map. Needs a live kernel/server for a push to reach the browser.

        Args:
            kind: ``"points"`` (a point feed — moving vehicles, sensors) or ``"path"`` (a growing track).
            data: The initial data, in the **display CRS** — a DataFrame (``x``/``y`` columns) or anything the
                GeoViews element accepts; ``None`` starts empty (``Pipe`` only).
            buffer: ``False`` (default) uses a ``Pipe`` — a push **replaces** the layer's data; ``True`` uses a
                ``Buffer`` that **appends** each push and keeps the last ``length`` rows.
            length: The sliding-window size for ``buffer=True`` (ignored for a ``Pipe``).
            name: The caller's own layer id/label; ``None`` generates one from the element kind.
            visible: Whether the layer is drawn.
            **opts: HoloViews style options applied to every frame of the live element.

        Returns:
            The same map instance, so builder calls chain. The stream is reached through :meth:`live_stream`
            and fed through :meth:`push`, both by the layer's id (:attr:`InteractiveMap.layer_ids`).

        Raises:
            ValueError: for an unknown ``kind``, or for ``buffer=True`` with no initial ``data`` (a ``Buffer``
                needs a DataFrame to infer its columns).

        Examples:
            - A point feed starts empty; pushing replaces its data, and the layer is a live ``DynamicMap``:
                ```python
                >>> import pandas as pd                                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> import holoviews as hv                                     # doctest: +SKIP
                >>> m = InteractiveMap().live(kind="points", name="cars")      # doctest: +SKIP
                >>> isinstance(m.layers[-1], hv.DynamicMap)                    # doctest: +SKIP
                True
                >>> m.push("cars", pd.DataFrame({"x": [0.0], "y": [0.0]}))     # doctest: +SKIP
                >>> len(m.live_stream("cars").data)                           # doctest: +SKIP
                1

                ```
            - A ``Buffer`` tails the last ``length`` rows across pushes:
                ```python
                >>> import pandas as pd                                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> start = pd.DataFrame({"x": [0.0], "y": [0.0]})            # doctest: +SKIP
                >>> m = InteractiveMap().live(                                # doctest: +SKIP
                ...     kind="points", data=start, buffer=True, length=2, name="track"
                ... )
                >>> m.push("track", pd.DataFrame({"x": [1.0], "y": [1.0]}))   # doctest: +SKIP
                >>> m.push("track", pd.DataFrame({"x": [2.0], "y": [2.0]}))   # doctest: +SKIP
                >>> len(m.live_stream("track").data)  # kept the last 2       # doctest: +SKIP
                2

                ```
        """
        gv, hv = _require_holoviz()
        from holoviews import streams

        if kind not in _LIVE_KINDS:
            raise ValueError(
                f"unknown live kind {kind!r}; choose from {list(_LIVE_KINDS)}"
            )
        crs = gv.util.process_crs(self.crs)
        element_cls = {"points": gv.Points, "path": gv.Path}[kind]

        def _draw(data: Any) -> Any:
            """Build one frame of the live element from the stream's current data.

            Args:
                data: Whatever the ``Pipe``/``Buffer`` currently holds, or ``None`` before the first push.

            Returns:
                The GeoViews element for this frame, styled with the caller's ``**opts``.
            """
            element = element_cls([] if data is None else data, crs=crs)
            return element.opts(**opts) if opts else element

        if buffer:
            if data is None:
                raise ValueError(
                    "live(buffer=True) needs an initial data= frame so the Buffer can infer its columns; "
                    "pass the first batch, or use a Pipe (buffer=False) which starts empty"
                )
            stream: Any = streams.Buffer(data, length=length)
        else:
            stream = streams.Pipe(data=data)
        dmap = hv.DynamicMap(_draw, streams=[stream])
        self.add_layer(dmap, kind=None, name=name, visible=visible)
        # `add_layer` has just registered this layer, so `_last_layer_id` is its id (never None here); the
        # cast tells the checker what the call guarantees.
        self._live_streams[cast(str, self._last_layer_id)] = stream
        return self

    def live_stream(self, layer_id: str) -> Any:
        """Return the ``Pipe``/``Buffer`` backing a live layer, to inspect or ``.send()`` to directly (IN-17).

        Args:
            layer_id: The live layer's id (:attr:`InteractiveMap.layer_ids`).

        Returns:
            The HoloViews stream — a ``Pipe`` for ``buffer=False`` or a ``Buffer`` for ``buffer=True``.

        Raises:
            KeyError: when ``layer_id`` is not a live layer, naming the ones that are.
        """
        self._require_live(layer_id)
        return self._live_streams[layer_id]

    def push(self, layer_id: str, data: Any) -> Self:
        """Push new data into a live layer — replace (``Pipe``) or append (``Buffer``) — and redraw it (IN-17).

        A thin, chainable wrapper over the stream's ``.send()``. Needs a live kernel/server for the update to
        reach the browser; off-server it still updates the stream's data so the next ``render()`` reflects it.

        Args:
            layer_id: The live layer's id.
            data: The new data, in the display CRS — a DataFrame for a ``Buffer`` (same columns as the seed),
                or anything the layer's element accepts for a ``Pipe``.

        Returns:
            The same map instance, so pushes chain.

        Raises:
            KeyError: when ``layer_id`` is not a live layer, naming the ones that are.
        """
        self._require_live(layer_id)
        self._live_streams[layer_id].send(data)
        return self

    def _require_live(self, layer_id: Any) -> None:
        """Refuse an id that is not a live layer, naming the ones that are.

        Args:
            layer_id: The id the caller passed.

        Raises:
            KeyError: when no live layer has that id.
        """
        if layer_id not in self._live_streams:
            raise KeyError(
                f"no live layer {layer_id!r} on this map; its live layers are "
                f"{sorted(self._live_streams)}"
            )
