"""AnimationMixin — animation playback & export for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``play`` and ``save_animation`` (DI.11), the interactive counterparts of ``Map.animate``/``rotate``.
They operate on the ``hv.DynamicMap`` a :meth:`~digitalearth.interactive.temporal.TemporalMixin.timecube`
registered: ``play`` binds a ``panel.widgets.Player`` to its time kdim for auto-advancing playback, and
``save_animation`` materialises the (lazy) DynamicMap to a finite ``HoloMap`` and writes a GIF/MP4 via the
matplotlib backend or a client-side **scrubber** HTML that animates offline with no server.
"""

from typing import Any

from digitalearth.interactive.base import _require_holoviz


class AnimationMixin:
    """Animation builders (DI.11): Player playback + GIF/MP4/scrubber export of a time cube."""

    def _time_dynamicmap(self) -> Any:
        """Return the registered time-cube ``hv.DynamicMap`` layer.

        Returns:
            The first ``hv.DynamicMap`` among the layers (a ``timecube``).

        Raises:
            ValueError: when no ``timecube`` layer has been added.
        """
        gv, hv = _require_holoviz()
        for layer in self.layers:
            if isinstance(layer, hv.DynamicMap) and layer.kdims:
                return layer
        raise ValueError(
            "no time cube to animate — call timecube(collection) before play()/save_animation()"
        )

    def _to_holomap(self, dmap: Any) -> Any:
        """Materialise a lazy ``DynamicMap`` into a finite ``HoloMap`` (every frame evaluated).

        Args:
            dmap: The time-cube ``DynamicMap``.

        Returns:
            An ``hv.HoloMap`` over the same kdim — exportable as a finite animation.
        """
        gv, hv = _require_holoviz()
        keys = list(dmap.kdims[0].values)
        return hv.HoloMap({key: dmap[key] for key in keys}, kdims=dmap.kdims)

    def play(self, *, fps: int = 3, loop: bool = True) -> Any:
        """Wrap the time cube in a Panel layout with an auto-advancing ``Player`` widget.

        Args:
            fps: Playback frames per second (the Player interval).
            loop: Loop at the end (``True``) or stop (``False``).

        Returns:
            A ``panel.viewable.Viewable`` hosting the map + a bound time ``Player``.

        Raises:
            ValueError: when no ``timecube`` layer has been added.
            ImportError: when the ``interactive`` extra (which provides panel) is absent.
        """
        import panel as pn

        gv, hv = _require_holoviz()
        dmap = self._time_dynamicmap()
        values = list(dmap.kdims[0].values)
        # DiscretePlayer (not Player) steps through arbitrary labelled values (ints / datetimes).
        player = pn.widgets.DiscretePlayer(
            options=values,
            value=values[0],
            interval=max(1, int(1000 / fps)),
            loop_policy="loop" if loop else "once",
        )
        view = pn.bind(lambda value: dmap[value], player)
        return pn.Column(pn.panel(view), player)

    def save_animation(self, path: str, *, fps: int = 3, **kwargs: Any) -> str:
        """Export the time cube as a GIF/MP4 (matplotlib backend) or a scrubber HTML.

        ``.gif``/``.mp4`` materialise the DynamicMap to a finite ``HoloMap`` and render via the
        matplotlib backend (``.mp4`` needs ffmpeg). ``.html`` writes a client-side **scrubber** that
        plays offline with no server.

        Args:
            path: Output file (``.gif`` / ``.mp4`` / ``.html``).
            fps: Frames per second.
            **kwargs: Forwarded to :func:`holoviews.save`.

        Returns:
            The ``path`` written.

        Raises:
            ValueError: when no ``timecube`` layer has been added.
        """
        gv, hv = _require_holoviz()
        holomap = self._to_holomap(self._time_dynamicmap())
        suffix = str(path).lower().rsplit(".", 1)[-1]
        if suffix == "html":
            hv.save(holomap, path, fmt="scrubber", fps=fps, **kwargs)
        else:
            hv.save(holomap, path, backend="matplotlib", fps=fps, **kwargs)
        return str(path)
