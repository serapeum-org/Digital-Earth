"""TemporalMixin — time-slider datacubes for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``timecube`` (DI.3): render a multi-band / time-stacked ``DatasetCollection`` as one
``hv.DynamicMap`` with a slider over the members. Each frame is an I1 ``image`` built from the *t*-th
member's ``Source`` (reusing the collection extractor — no new datacube model). The colour range is
**frozen** across frames (a global ``clim`` computed once over the stack, or an explicit ``clim``) so the
colormap and colorbar do not jump as the slider moves.

``DynamicMap`` callbacks are lazy — they evaluate a frame only when the slider lands on it; tests
materialise a frame (``dmap[0]``) to assert on it.
"""

from typing import Any, Optional, Sequence, Tuple

from digitalearth.interactive.base import _masked_to_nan, _require_holoviz


class TemporalMixin:
    """Time-slider datacube builder (DI.3)."""

    def _global_clim(self, collection: Any, band: int) -> Tuple[float, float]:
        """Compute one ``(vmin, vmax)`` over every member so the colour range never jumps.

        Note: this pass is **eager** — it reprojects and extracts every member once at ``timecube``
        construction time (the per-frame ``DynamicMap`` callback warps them again lazily). For a very
        large datacube, pass an explicit ``clim`` to ``timecube`` to skip this whole-stack scan.

        Args:
            collection: A pyramids ``DatasetCollection``.
            band: 1-based band read from each member.

        Returns:
            ``(vmin, vmax)`` finite colour limits across the whole stack.
        """
        import numpy as np

        lows, highs = [], []
        for member in collection.datasets:
            arr = _masked_to_nan(self._to_display_source(member, band=band).z.values)
            if np.isfinite(arr).any():
                lows.append(np.nanmin(arr))
                highs.append(np.nanmax(arr))
        return (float(min(lows)), float(max(highs))) if lows else (0.0, 1.0)

    def timecube(
        self,
        collection: Any,
        *,
        kdim: str = "time",
        labels: Optional[Sequence] = None,
        band: int = 1,
        cmap: str = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        colorbar: bool = True,
        **opts: Any,
    ) -> "TemporalMixin":
        """Render a ``DatasetCollection`` as an interactive time-slider map.

        Builds an ``hv.DynamicMap`` whose ``frame(t)`` constructs an I1 ``hv.Image`` from the *t*-th
        member; ``redim.values`` drives a Bokeh slider. The colour range is frozen across frames so
        the colorbar is identical on the first and last frame.

        Args:
            collection: A pyramids ``DatasetCollection`` whose members are ordered time steps.
            kdim: Slider dimension name.
            labels: Optional per-member labels (e.g. datetimes) shown on the slider instead of the
                integer index; must match the member count.
            band: 1-based band rendered in every frame.
            cmap: Colormap name.
            clim: Frozen ``(vmin, vmax)`` colour limits; ``None`` computes a global range once over
                the whole stack.
            colorbar: Whether to draw a colorbar.
            **opts: Extra HoloViews style options applied to every frame.

        Returns:
            This map (chainable) — one ``DynamicMap`` layer is registered.

        Raises:
            ValueError: when ``labels`` is given but its length differs from the member count.

        Examples:
            - Scrub a 3-step collection with a frozen colour range:
                ```python
                >>> from pyramids.dataset.collection import DatasetCollection  # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dc = DatasetCollection.from_files(["a.tif", "b.tif"])      # doctest: +SKIP
                >>> m = InteractiveMap().timecube(dc, cmap="inferno")          # doctest: +SKIP
                >>> [d.name for d in m.layers[0].kdims]                        # doctest: +SKIP
                ['time']

                ```
            - Label the slider with real datetimes:
                ```python
                >>> import datetime as dt                                     # doctest: +SKIP
                >>> from pyramids.dataset.collection import DatasetCollection  # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dc = DatasetCollection.from_files(["a.tif", "b.tif"])      # doctest: +SKIP
                >>> stamps = [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)]  # doctest: +SKIP
                >>> InteractiveMap().timecube(dc, labels=stamps).save("t.html")  # doctest: +SKIP
                't.html'

                ```
        """
        gv, hv = _require_holoviz()
        members = collection.datasets
        n = len(members)
        if labels is not None:
            if len(labels) != n:
                raise ValueError(
                    f"labels has {len(labels)} entries but the collection has {n} members"
                )
            if len(set(labels)) != n:
                raise ValueError(
                    "timecube labels must be unique — duplicate labels collapse the slider and "
                    "make the matching frames unreachable"
                )
        frozen_clim = clim if clim is not None else self._global_clim(collection, band)
        keys = list(labels) if labels is not None else list(range(n))
        key_to_index = {key: index for index, key in enumerate(keys)}

        def frame(value: Any) -> Any:
            src = self._to_display_source(members[key_to_index[value]], band=band)
            arr = _masked_to_nan(src.z.values)
            name = self._vdim_name(src)
            image = hv.Image(
                (src.x.values, src.y.values, arr), kdims=["x", "y"], vdims=[name]
            )
            return self._styled(
                image,
                common={
                    "cmap": cmap,
                    "clim": frozen_clim,
                    "colorbar": colorbar,
                    **opts,
                },
                bokeh={"tools": ["hover"]},
            )

        dmap = hv.DynamicMap(frame, kdims=[kdim]).redim.values(**{kdim: keys})
        return self.add_element(dmap)
