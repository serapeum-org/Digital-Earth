"""TemporalMixin — time-slider datacubes for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``timecube`` (DI.3): render a multi-band / time-stacked ``DatasetCollection`` as one
``hv.DynamicMap`` with a slider over the members. Each frame is an I1 ``image`` built from the *t*-th
member's ``Source`` (reusing the collection extractor — no new datacube model). The colour range is
**frozen** across frames (a global ``clim`` computed once over the stack, or an explicit ``clim``) so the
colormap and colorbar do not jump as the slider moves.

``DynamicMap`` callbacks are lazy — they evaluate a frame only when the slider lands on it; tests
materialise a frame (``dmap[0]``) to assert on it.
"""

from typing import TYPE_CHECKING, Any, Optional, Self, Sequence, Tuple

from digitalearth.base.clim import sample_evenly, stack_clim
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.base.spec._serial import thawed_value
from digitalearth.interactive.base import (
    _masked_to_nan,
    _require_holoviz,
    _skips_off_limb,
)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def draw_timecube(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the time-slider layer a described cube asks for.

    Args:
        interactive_map: The map being drawn.
        data: The `DatasetCollection` whose members are the frames.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    _, hv = _require_holoviz()
    props = dict(layer.symbology.props)
    band = props["band"]
    members = data.datasets
    # One colour range and one colormap for the whole cube, resolved from the collection and its first
    # member, so the colorbar on the last frame is the colorbar on the first.
    clim = props.get("clim")
    frozen_clim = clim if clim is not None else interactive_map._global_clim(data, band)
    cmap = props.get("cmap")
    if cmap is None and members:
        cmap = interactive_map._auto_cmap(
            interactive_map._to_display_source(members[0], band=band), None
        )
    cmap = cmap or "viridis"
    labels = props.get("labels")
    # Only the slider keys are thawed: a symbology stores every sequence as a tuple, and `clim` beside
    # them *is* a pair — HoloViews reads it as one, and a list where a tuple belongs is not honoured.
    keys = thawed_value(labels) if labels is not None else list(range(len(members)))
    key_to_index = {key: index for index, key in enumerate(keys)}
    common = {
        "cmap": cmap,
        "clim": frozen_clim,
        "colorbar": props.get("colorbar"),
        **dict(props.get("opts") or {}),
    }

    def frame(value: Any) -> Any:
        """Draw one member of the cube.

        Args:
            value: The slider key naming the member.

        Returns:
            The styled image for that member.
        """
        src = interactive_map._to_display_source(
            members[key_to_index[value]], band=band
        )
        image = hv.Image(
            (src.x.values, src.y.values, _masked_to_nan(src.z.values)),
            kdims=["x", "y"],
            vdims=[interactive_map._vdim_name(src)],
        )
        return interactive_map._styled(image, common=common, bokeh={"tools": ["hover"]})

    dmap = hv.DynamicMap(frame, kdims=[props["kdim"]]).redim.values(
        **{props["kdim"]: keys}
    )
    return DrawnLayer(element=dmap, style=common)


class TemporalMixin(_MixinBase):
    """Time-slider datacube builder (DI.3).

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

    def _global_clim(self, collection: Any, band: int) -> Tuple[float, float]:
        """Compute one ``(vmin, vmax)`` for the whole series so the colour range never jumps between frames.

        Note: this pass is **eager** — it reprojects and extracts the sampled members once at ``timecube``
        construction time (the per-frame ``DynamicMap`` callback warps them again lazily). At most
        :data:`~digitalearth.base.clim.DEFAULT_CLIM_SCAN_CAP` members are read, evenly spaced across the
        series, which is the rule the static and web tiers follow too. For a very large datacube, pass an
        explicit ``clim`` to ``timecube`` to skip the scan entirely.

        This tier also stopped letting an infinity set a limit. It reduced with ``np.nanmin``/``np.nanmax``,
        which skip ``NaN`` but keep ``±inf``, so a single overflowed cell pinned the whole ramp to one end.
        The shared rule drops every non-finite value, so such a frame now contributes its real extremes.

        Args:
            collection: A pyramids ``DatasetCollection``.
            band: 1-based band read from each member.

        Returns:
            ``(vmin, vmax)`` finite colour limits taken from the sampled members, or ``(0.0, 1.0)`` when none
            holds a finite value.
        """
        return stack_clim(
            _masked_to_nan(self._to_display_source(member, band=band).z.values)
            for member in sample_evenly(collection.datasets)
        )

    @_skips_off_limb
    def timecube(
        self,
        collection: Any,
        *,
        kdim: str = "time",
        labels: Optional[Sequence] = None,
        band: int = 1,
        cmap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        colorbar: bool = True,
        **opts: Any,
    ) -> Self:
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
            cmap: Colormap name; ``None`` (default) resolves it from the variable through
                ``autostyle.auto_style`` (#249), exactly as ``image`` does.
            clim: Frozen ``(vmin, vmax)`` colour limits; ``None`` computes one range for the whole series
                once, from at most :data:`~digitalearth.base.clim.DEFAULT_CLIM_SCAN_CAP` members sampled
                evenly across it.
            colorbar: Whether to draw a colorbar.
            **opts: Extra HoloViews style options applied to every frame.

        Returns:
            The same map instance, so builder calls chain — one ``DynamicMap`` layer is registered.

        Raises:
            ValueError: when ``labels`` is given but its length differs from the member count, or two of
                its entries are equal — duplicate keys collapse the slider and make the matching frames
                unreachable.

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
                >>> m = InteractiveMap().timecube(dc, labels=stamps)          # doctest: +SKIP
                >>> m.save("t.html").name                                     # doctest: +SKIP
                't.html'

                ```
        """
        _require_holoviz()
        # Both refusals stay with the builder: each names the `labels` the caller wrote.
        n = len(collection.datasets)
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
        return self.add_element(
            None,
            kind="raster",
            source=collection,
            symbology=Symbology(
                props={
                    "via": "timecube",
                    "kdim": kdim,
                    "labels": None if labels is None else list(labels),
                    "band": band,
                    "cmap": cmap,
                    "clim": clim,
                    "colorbar": colorbar,
                    "opts": dict(opts),
                }
            ),
        )
