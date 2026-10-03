"""RasterMixin — raster builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``field`` / ``rgb`` / ``quadmesh`` / ``contours`` / ``spaghetti`` (DI.1a); ``large_image`` viewport
loading lands later (DI.14).

Every builder records what it draws; its drawer funnels through ``_to_display_source`` (reproject in
**pyramids**, option A), then emits a **plain HoloViews** element (``hv.Image``/``hv.RGB``/``hv.QuadMesh``)
whose coordinates are already in the display CRS — deliberately *not* ``gv.Image``, whose default
PlateCarree ``crs`` would re-project already-projected coordinates at render time. NoData arrives as a
masked array from pyramids and renders transparent (``NaN``).

**Naming note** — the interactive builders use HoloViews-idiomatic names that differ from the static
``Map``: ``rgb`` (static ``rgb_composite``). ``field``/``spaghetti``/``quadmesh``/``contours`` match —
``contours`` because this tier's own second spelling ``filled_contours`` was deleted in review M3, leaving
one method taking ``filled=`` on every tier that draws it. The remaining divergence is intentional (this
tier reads as HoloViews to its users); the static↔interactive mapping is documented in the tier plan's
feature-parity matrix.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Self, Sequence, Tuple

from digitalearth.base.ask import UNSET, Ask, Maybe
from digitalearth.base.crs import reproject
from digitalearth.base.levels import levels_every
from digitalearth.base.raster_classes import (
    BandClasses,
    asks_categorical,
    classes_of,
    classify_band,
)
from digitalearth.base.sources.view import SourceView
from digitalearth.base.spec import (
    DEFAULT_BAND,
    Bounds,
    DataRef,
    Encoding,
    LayerSpec,
    RenderTarget,
    Selection,
    Symbology,
    Viewport,
)
from digitalearth.base.stretch import (
    DEFAULT_COMPOSITE_BANDS,
    ChannelLimits,
    require_three_bands,
    stretch_to_unit,
)
from digitalearth.interactive.base import (
    _masked_to_nan,
    _require_holoviz,
    _skips_off_limb,
    cmap_name,
    describe,
    describe_opts,
    held_props,
)
from digitalearth.interactive.style_fold import TIER_BUCKET, limits_scale

#: What an unstyled colour-mapped raster is drawn at on this tier. The value left `field`'s signature with
#: #334, for the reason `POINT_SIZE` gives on the vector side, and this is where it lives instead.
FIELD_ALPHA = 1.0

#: What a raster layer's colour is said to vary with when the raster names neither a band nor a variable.
#:
#: The same fallback :meth:`~digitalearth.interactive.base.InteractiveMapBase._vdim_name` lands on — a
#: `Source` built from a bare array records ``variable: ""`` and its value axis is named ``z`` — so the name
#: the description publishes is the name the drawn element's value dimension carries. Written once because
#: the two would otherwise be two spellings of one thing.
UNNAMED_VALUE = "z"

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _travelling_pair(pair: Any) -> Any:
    """Return a ``(low, high)`` argument in the spelling a figure can be written with.

    ``clim`` is declared as a pair — "Colour limits, as (low, high)" — and a **tuple** is exactly what
    :func:`~digitalearth.base.spec._serial.travels_in_a_figure` refuses, so the documented spelling was
    held beside the layer and dropped from every saved figure while the undocumented list form travelled
    (review R2-M13). Normalising here is what makes the two spellings one recorded value;
    :func:`_engine_pair` restores the engine's spelling at draw time.

    Args:
        pair: What the caller passed — a tuple, a list, or `None` to decline the keyword.

    Returns:
        A list for a two-item sequence, and the argument untouched for anything else, so a value this
        tier does not recognise still reaches `describe` and is held rather than mangled.
    """
    if isinstance(pair, (tuple, list)) and len(pair) == 2:
        return list(pair)
    return pair


def coloured_by(data: Any, band: int = DEFAULT_BAND) -> str:
    """Return the name of the value a raster layer's colour varies with, **without warping the raster**.

    Order 24 hangs a colour key on the layer's own colour encoding, and an encoding bound to a field needs
    that field's name at the moment the layer is *described* — which is before any drawer has run. The drawer
    reads the name off the reprojected :class:`~digitalearth.base.sources.source.Source`
    (:meth:`~digitalearth.interactive.base.InteractiveMapBase._vdim_name`), and reprojecting here to ask the
    same question would do the drawer's work a second time and throw the result away — the duplication review
    M8 took out of the vector builders. So the name is read off the raster's own metadata instead, the way
    :meth:`~digitalearth.interactive.base.InteractiveMapBase._auto_cmap_for_band` already reads a band name to
    resolve a colormap without touching the band.

    The two agree by construction for every input this tier draws: the extractor records a `Dataset`'s
    ``band_names[band - 1]`` as the source's ``variable`` and a `NetCDF`'s first variable name likewise, and
    those are exactly what is read here. Measured on ``examples/data/acc4000.tif``: this answers ``'Band_1'``
    and the drawn ``hv.Image``'s value dimension is ``'Band_1'``.

    Args:
        data: The raster the layer draws — a pyramids ``Dataset`` / ``NetCDF`` / ``Source``, a path, or a bare
            array. Read with `getattr`, so an input declaring neither list simply falls back.
        band: The 1-based band the layer colours by.

    Returns:
        The band's name, the first variable's name, or :data:`UNNAMED_VALUE` — never an empty string, which is
        what an :class:`~digitalearth.base.spec.encoding.Encoding` refuses as a field name.
    """
    names = list(getattr(data, "band_names", None) or [])
    if 1 <= band <= len(names) and names[band - 1]:
        return str(names[band - 1])
    # A `NetCDF` has variables rather than bands, and `to_display_source` extracts the **first** of them —
    # it forwards only `band`, never a variable — so that is the one a colour key would explain.
    # `next(iter(...), None)` rather than `variables[0]` guarded by `if variables`: the two are the same
    # value, and the second reads as an unguarded subscript to a reviewer scanning for one (SonarCloud
    # raises S6552 on it as a possible `IndexError`, which the short-circuit already makes impossible). It
    # is still the **first** variable specifically — not the first non-empty one — because that is the one
    # `to_display_source` extracts, which is what makes the name agree with the drawn value dimension.
    first = next(iter(getattr(data, "variable_names", None) or []), None)
    if first:
        return str(first)
    return UNNAMED_VALUE


def _one_spacing_only(caller: str, levels: Any, interval: Optional[float]) -> None:
    """Refuse a contour call that asks for its levels two ways at once.

    Written once for the tier's two contour spellings, so neither could answer the pair differently from
    the other; the second spelling is gone (review M3) and this is :meth:`RasterMixin.contours`' refusal
    alone. It still takes the caller's name rather than hard-coding one, which is the rule every builder
    refusal on this tier follows: the message names the method the caller actually wrote.

    Args:
        caller: The public method to name in the message.
        levels: The ``levels=`` argument, or `None` for a caller who did not give one.
        interval: The ``interval=`` argument, or `None` for a caller who did not give one.

    Raises:
        ValueError: when both arrived — two ways of asking for one thing, so neither can be silently
            preferred.
    """
    if levels is not None and interval is not None:
        raise ValueError(
            f"{caller}() takes at most one of interval= or levels=; "
            f"got interval={interval!r} and levels={levels!r}"
        )


def _engine_pair(pair: Any) -> Any:
    """Return a recorded ``(low, high)`` in the spelling HoloViews declares the option with.

    Args:
        pair: The recorded value, thawed by :func:`~digitalearth.interactive.base.held_props`.

    Returns:
        A tuple for a two-item sequence, and the value untouched otherwise — including `None`, which is a
        caller declining the keyword and must stay a decline.
    """
    if isinstance(pair, (tuple, list)) and len(pair) == 2:
        return tuple(pair)
    return pair


#: The drawing options that classify a band rather than style its element. The static tier records them
#: among its options; this tier records them beside the layer — either way they are read for the classes the
#: colour encoding holds, never handed to HoloViews, which has no option of either name.
_CLASS_KEYWORDS = frozenset({"scheme", "k"})


def _class_options(classes: BandClasses) -> Dict[str, Any]:
    """Return the HoloViews options that draw a band class by class.

    One colour per class (``cmap``) and the edges between them (``color_levels``) — the same pair this tier
    gives a graduated choropleth — over limits spanning the edges, so the bar steps exactly at them. A
    categorical band's bar is ticked at its codes rather than at the half-steps between them, which are where
    its classes meet but are no value a cell holds.

    Args:
        classes: The band's classes.

    Returns:
        The options, laid over the element's continuous ones.

    Examples:
        - Three graduated classes step at their edges:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> from digitalearth.interactive.raster import _class_options
            >>> options = _class_options(classify_band(np.arange(10.0), "equal_interval", 3, "viridis"))
            >>> options["color_levels"], options["clim"]
            ([0.0, 3.0, 6.0, 9.0], (0.0, 9.0))

            ```
    """
    edges = list(classes.edges)
    options: Dict[str, Any] = {
        "cmap": list(classes.colors),
        "color_levels": edges,
        "clim": (edges[0], edges[-1]),
    }
    if classes.is_categorical:
        from bokeh.models import FixedTicker

        codes = [int(code) for code in classes.scale.categories]
        options["colorbar_opts"] = {"ticker": FixedTicker(ticks=sorted(codes))}
    return options


def _band_classes(
    interactive_map: Any, data: Any, band: int, scheme: Any, k: Optional[int], cmap: Any
) -> BandClasses:
    """Classify a band at build time, so the layer's description carries its classes.

    A colour key hangs on the layer's colour encoding, and the encoding is written when the layer is
    *described* — before any drawer runs — so the classes have to be known here. The band is read through the
    same display-CRS choke point the drawer reads it through, so the classes are cut on the cells that are
    drawn; that is a second read of the band, paid only by a classified field.

    Args:
        interactive_map: The map the layer is added to.
        data: The raster, or a path or URL naming one.
        band: The 1-based band.
        scheme: ``"categorical"``, a named graduated scheme, or explicit edges.
        k: Classes for a named graduated scheme.
        cmap: The caller's colormap — a categorical palette for codes, the colormap graduated classes are
            sampled from otherwise — or ``None``.

    Returns:
        The band's classes.

    Raises:
        ValueError: as :func:`~digitalearth.base.raster_classes.classify_band` refuses.
    """
    if isinstance(data, (str, os.PathLike, DataRef)):
        # A path or URL is what the figure records; the drawer is handed it already opened, the builder
        # is not.
        data = DataRef.of(data).open()
    src = interactive_map._to_display_source(data, band=band)
    # Codes take the categorical palette when no `cmap` is given; graduated classes the colormap the drawer
    # will resolve for this variable, so the colours a key derives agree with the ones painted.
    palette = (
        cmap if asks_categorical(scheme) else interactive_map._auto_cmap(src, cmap)
    )
    return classify_band(src.z.values, scheme, k, palette)


def draw_image(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the colour-mapped raster a described image layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    props = held_props(interactive_map, layer)
    src = interactive_map._to_display_source(data, band=props["band"])
    # The static tier records `scheme`/`k` among its drawing options; they are cleopatra's words, not
    # HoloViews', so a figure that tier described carries them here, where HoloViews refused the whole element
    # ("Unexpected option 'scheme' for Image"). What they decided travels in the colour encoding, read below.
    opts = {
        key: value
        for key, value in dict(props.get("opts") or {}).items()
        if key not in _CLASS_KEYWORDS
    }
    common = {
        "cmap": interactive_map._auto_cmap(src, props.get("cmap")),
        "clim": _engine_pair(props.get("clim")),
        "alpha": props.get("alpha"),
        "colorbar": props.get("colorbar"),
        "clabel": interactive_map._auto_clabel(src, props.get("clabel")),
        **opts,
    }
    described = dict(layer.symbology.encodings).get("color")
    classes = classes_of(getattr(described, "scale", None), common["cmap"])
    if classes is not None:
        common.update(_class_options(classes))
    element = interactive_map._styled(
        interactive_map._image_from_source(src),
        common=common,
        bokeh={"tools": ["hover"]},
    )
    return DrawnLayer(element=element, style=common)


def draw_rgb(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Compose the three-band true-colour image a described RGB layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The multiband raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        ValueError: when the recorded ``limits`` do not hold one ``(lo, hi)`` pair per channel. The band
            count is refused by the builder, which names the caller's own argument; the limits are refused
            here, where the stretch that reads them runs.
    """
    from digitalearth.base.sources import get_stack
    from digitalearth.interactive.renderer import DrawnLayer

    _, hv = _require_holoviz()
    props = held_props(interactive_map, layer)
    bands = list(props["bands"])
    # Reproject once into a local handle, then feed both the coordinate extraction (get_source, via
    # _to_display_source) and the band stack (get_stack) from it — get_stack needs the same
    # already-reprojected dataset, so a single warp here keeps them consistent (H1).
    if hasattr(data, "to_crs") and interactive_map._needs_reproject(data):
        data = reproject(data, interactive_map.crs)
    src = interactive_map._to_display_source(data, band=bands[0])
    # One shared stretch for every backend (base/stretch.py). A recorded `limits` holds it fixed across a
    # sequence of frames, the same way the matplotlib tier freezes an animation.
    limits = props.get("limits")
    stretched = stretch_to_unit(get_stack(data, bands), limits)
    channels = [stretched[:, :, index] for index in range(3)]
    element = hv.RGB(
        (src.x.values, src.y.values, *channels),
        kdims=["x", "y"],
        vdims=["R", "G", "B"],
    )
    common = dict(props.get("opts") or {})
    return DrawnLayer(
        element=interactive_map._styled(element, common=common or None), style=common
    )


def draw_quadmesh(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the quadrilateral mesh a described mesh layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    _, hv = _require_holoviz()
    props = held_props(interactive_map, layer)
    src = interactive_map._to_display_source(data, band=props["band"])
    arr = _masked_to_nan(src.z.values)
    element = hv.QuadMesh(
        (src.x.values, src.y.values, arr),
        kdims=["x", "y"],
        vdims=[interactive_map._vdim_name(src)],
    )
    common = {
        "cmap": interactive_map._auto_cmap(src, props.get("cmap")),
        "clabel": interactive_map._auto_clabel(src, props.get("clabel")),
        **dict(props.get("opts") or {}),
    }
    element = interactive_map._styled(
        element, common=common, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=element, style=common)


def draw_contours(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Trace the contours a described contour layer asks for, filled or as lines.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    # Guarded before holoviews is reached, so a missing stack is refused by the message that names the
    # extra to install rather than by holoviews' own ImportError.
    _require_holoviz()
    from holoviews.operation import contours as contour_op

    props = held_props(interactive_map, layer)
    src = interactive_map._to_display_source(data, band=props["band"])
    # `interval=` is resolved here rather than in the builder, because it is the *band* that decides which
    # multiples of the spacing lie inside it — and the band is only read once the layer is drawn. The
    # arithmetic is `base.levels`, shared so this tier and the static one cannot disagree about which
    # multiples are inside (#262). The builder has already refused `interval=` together with `levels=`, so
    # only one of the two branches can be taken.
    interval = props.get("interval")
    if interval is not None:
        resolved = levels_every(src.z.values, interval)
    else:
        # A caller's `levels` always wins; `None` consults autostyle for the variable's canonical contour
        # levels (#230) before falling back to the tier's 10.
        resolved = interactive_map._auto_levels(src, props.get("levels"))
    element = contour_op(
        interactive_map._image_from_source(src),
        # Already a list: `held_props` thaws the described half once, so a caller's `levels=[0, 5, 10]`
        # reaches the contour operation in the spelling it was written in.
        levels=10 if resolved is None else resolved,
        filled=props.get("filled", False),
    )
    # The derived half first, the caller's own over it — the precedence every other drawer on this tier
    # applies, and the reason `spaghetti`'s per-member colour can sit in `common` without outranking a
    # `color=` the caller wrote (review R2-H3).
    common = {
        **dict(props.get(TIER_BUCKET) or {}),
        **dict(props.get("opts") or {}),
    }
    element = interactive_map._styled(
        element, common=common or None, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=element, style=common)


def draw_large_image(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the windowed-read layer a described large-raster layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    props = held_props(interactive_map, layer)
    element, style = interactive_map._draw_large_image(data, props)
    return DrawnLayer(element=element, style=style)


class RasterMixin(_MixinBase):
    """Raster builders (DI.1a): colour-mapped fields, composites and ensemble spaghetti."""

    def _image_from_source(self, src: Any, *, vname: Optional[str] = None) -> Any:
        """Build the I1 image from an already display-CRS :class:`Source`.

        Takes an already-reprojected source rather than the raw data, so a builder that also needs the
        source itself — for the autostyle ``cmap``/``levels``/``units`` lookup (#230) — reprojects once
        instead of twice.

        Args:
            src: The display-CRS source.
            vname: Value-dimension name; defaults to the source's variable/z name.

        Returns:
            holoviews.Image: the raster as a plain HoloViews image in the display CRS.

            A view that read a window is placed on the rectangle it read, rather than on one derived from
            its axes: a zoom that lands on a single cell leaves an axis with no spacing to derive from,
            and HoloViews answers that with `nan` bounds and a raised frame (#300).
        """
        # Called for its actionable ImportError; the element itself comes from `_raster_element`, which is
        # the one place either engine module is named.
        _require_holoviz()
        arr = _masked_to_nan(src.z.values)
        name = vname or self._vdim_name(src)
        window = getattr(src, "window", None)
        return self._raster_element(
            src.x.values,
            src.y.values,
            arr,
            name,
            bounds=None if window is None else window.as_bbox(),
        )

    @_skips_off_limb
    def field(
        self,
        data: Any,
        *,
        band: int = DEFAULT_BAND,
        cmap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        scheme: Optional[Any] = None,
        k: Optional[int] = None,
        alpha: Maybe[float] = UNSET,
        colorbar: bool = True,
        clabel: Optional[str] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a colour-mapped raster layer with hover readout.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected to the display CRS
                through pyramids when needed.
            band: 1-based band to render.
            cmap: Colormap name; ``None`` (default) resolves it from the variable via
                ``autostyle.auto_style`` (DI.12) — the same lookup the static ``Map`` uses.
            clim: Optional ``(vmin, vmax)`` colour limits; ``None`` auto-scales. A list is taken too,
                and is the spelling the figure records — a tuple is what the travel rule refuses, so the
                documented pair used to be dropped from every saved figure (review R2-M13). The drawer
                hands HoloViews the tuple its own option is declared with either way. Ignored when
                ``scheme`` classifies the band, whose limits are its class edges.
            scheme: Colour the band class by class rather than along a ramp. A named graduated scheme
                (``"quantiles"``, ``"equal_interval"``, ``"fisher_jenks"``, …) or an explicit list of edges cuts
                classes coloured from ``cmap``; ``"categorical"`` gives each integer code a class of its own,
                coloured from the shared categorical palette (``cmap`` picks another). The classes are cut by
                :mod:`digitalearth.base.raster_classes`, the classifier the static and web tiers use, so one
                band is classed — and coloured — alike on all three, and they are recorded in the layer's
                colour encoding, so the colorbar steps at them and a figure drawn on another tier keeps them.
                ``None`` (default) draws the continuous ramp.
            k: How many classes a named graduated scheme cuts; ``None`` takes 5. Ignored without ``scheme``.
            alpha: Layer opacity in ``[0, 1]``; not passed leaves :data:`FIELD_ALPHA`. The keyword is
                recorded when it is passed, so a caller asking for exactly that value publishes it like any
                other (#334).
            colorbar: Whether to draw a colorbar.
            clabel: Colorbar label; ``None`` (default) takes the variable's ``units`` from
                ``autostyle.auto_style`` (#230) and leaves the colorbar unlabelled when it knows none.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Render a DEM as a pan/zoom raster with explicit colour limits:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem, cmap="terrain", clim=(0, 60))  # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                1

                ```

        Returns:
            This map (chainable).

        Raises:
            ValueError: for a ``scheme`` that cannot classify the band — an unknown name, ``k`` below one, or,
                under ``"categorical"``, a band holding non-integer values, more than
                :data:`~digitalearth.base.raster_classes.MAX_RASTER_CATEGORIES` codes, or no valid cell.
        """
        scale = limits_scale(clim)
        if scheme is not None:
            scale = _band_classes(self, data, band, scheme, k, cmap).scale
        # The caller's raw HoloViews keywords are split per value (review M3): the JSON-safe half goes
        # into the description, and what has no JSON form — a colormap object, a callable — is held beside
        # the layer, because a figure is saved as JSON.
        held: Dict[str, Any] = {}
        described_opts = describe_opts(held, opts)
        ask = Ask()
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="raster",
            source=data,
            held=held,
            symbology=Symbology(
                # The layer's colour varies with the band it draws, so that is published as a binding rather
                # than left for a reader to infer from `cmap`. It is what a colorbar hangs on (order 24): a
                # `Guide` explains an encoding, so a layer with no colour encoding has no key, and a layer
                # with this one has a key that moves, hides and disappears with it.
                encodings={
                    "color": Encoding.by_field(
                        "color", coloured_by(data, band), scale=scale
                    )
                },
                props={
                    "via": "image",
                    "band": band,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                    "clim": describe(held, "clim", _travelling_pair(clim)),
                    "scheme": scheme
                    if scheme is None or isinstance(scheme, str)
                    else list(scheme),
                    "k": k,
                    "alpha": ask("alpha", alpha, FIELD_ALPHA),
                    "colorbar": colorbar,
                    "clabel": clabel,
                    "opts": described_opts,
                    **ask.record,
                },
            ),
        )

    @_skips_off_limb
    def rgb(
        self,
        data: Any,
        *,
        bands: Sequence[int] = DEFAULT_COMPOSITE_BANDS,
        limits: Optional[ChannelLimits] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a true-colour composite from three raster bands (2–98 % percentile stretch).

        Args:
            data: A pyramids multiband ``Dataset``; reprojected to the display CRS first.
            bands: The three 1-based band indices composing ``(R, G, B)``.
            limits: Optional frozen ``(lo, hi)`` stretch bounds, one pair per channel — skips the per-call
                percentile scan, so a sequence of frames can share one black and white point.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Examples:
            - Compose three bands of a satellite stack into true colour:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> stack = Dataset.read_file("sentinel.tif")                   # doctest: +SKIP
                >>> m = InteractiveMap().rgb(stack, bands=(4, 3, 2))            # doctest: +SKIP
                >>> [d.name for d in m.layers[0].vdims]                         # doctest: +SKIP
                ['R', 'G', 'B']

                ```

        Raises:
            ValueError: when ``bands`` does not name exactly three bands, or ``limits`` is given without
                one ``(lo, hi)`` pair per channel.
        """
        _require_holoviz()
        # Refused here rather than in the drawer, because the message names the argument the caller wrote.
        require_three_bands("rgb", bands)
        held: Dict[str, Any] = {}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="rgb",
            source=data,
            held=held,
            symbology=Symbology(
                props={
                    "via": "rgb",
                    "bands": tuple(bands),
                    # A frozen stretch whose channel had no finite cell holds `(nan, nan)`, which JSON has
                    # no spelling for. Such limits are held beside the layer and described as not given,
                    # which is what a reader without them derives per frame.
                    "limits": describe(held, "limits", limits),
                    "opts": described_opts,
                }
            ),
        )

    @_skips_off_limb
    def quadmesh(
        self,
        data: Any,
        *,
        band: int = DEFAULT_BAND,
        cmap: Optional[str] = None,
        clabel: Optional[str] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a quadrilateral-mesh raster layer (handles non-uniform / curvilinear coordinates).

        Unlike :meth:`field` (regular grid), a ``QuadMesh`` draws each cell from its coordinate
        arrays, so irregularly spaced or 2-D (curvilinear) coordinates render without resampling.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to render.
            cmap: Colormap name; ``None`` (default) resolves it from the variable via
                ``autostyle.auto_style`` (DI.12) — consistent with :meth:`field`.
            clabel: Colorbar label; ``None`` (default) takes the variable's ``units`` from
                ``autostyle.auto_style`` (#230), as :meth:`field` does.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Draw an irregular grid without resampling to axis-aligned:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> grid = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> InteractiveMap().quadmesh(grid).save("mesh.html").name      # doctest: +SKIP
                'mesh.html'

                ```

        Returns:
            This map (chainable).
        """
        _require_holoviz()
        held: Dict[str, Any] = {}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="mesh",
            source=data,
            held=held,
            symbology=Symbology(
                # Coloured by its band, exactly as `field` is; it takes no `clim=`, so the domain is the
                # engine's to measure and the encoding carries no scale rather than a guessed one.
                encodings={
                    "color": Encoding.by_field("color", coloured_by(data, band))
                },
                props={
                    "via": "quadmesh",
                    "band": band,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                    "clabel": clabel,
                    "opts": described_opts,
                },
            ),
        )

    @_skips_off_limb
    def contours(
        self,
        data: Any,
        *,
        band: int = DEFAULT_BAND,
        levels: Any = None,
        interval: Optional[float] = None,
        filled: bool = False,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Trace iso-value lines of a raster band, or fill the bands between them.

        The three keywords the Tier-2 contract declares — ``levels``, ``interval`` and ``filled`` — on the
        one method that answers to the name, so a call written against the web or static tier runs here
        unchanged (#262). This tier also had a second method, ``filled_contours``, for what ``filled=True``
        says here; it was deleted in review M3, and ``interval=`` used to fall through ``**opts`` to
        HoloViews, which refused it as an unknown *style* option.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to contour.
            levels: Contour levels — an int (count) or explicit sequence; ``None`` takes the
                variable's canonical levels from ``autostyle.auto_style`` (#230), falling back to 10.
                Give at most one of this or ``interval``.
            interval: Spacing between levels — one level every N, in the band's own units. Resolved
                against the band when the layer is drawn, because it is the band that decides which
                multiples of the spacing lie inside it, and only the ones *strictly* inside are traced: a
                level on an extreme draws the frame's edge or nothing. The arithmetic is
                :func:`~digitalearth.base.levels.levels_every`, shared with the static tier so the two
                cannot disagree about which multiples are inside.
            filled: Fill the bands between the levels instead of drawing the levels as lines. It is also
                what the layer's kind records — ``"filled_contours"`` against ``"contours"`` — so a figure
                says which of the two renders it holds.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **opts: Extra HoloViews style options applied to the element.

        Raises:
            ValueError: when both ``levels`` and ``interval`` are given — two ways of asking for one thing,
                so neither can be silently preferred; when ``interval`` is not a positive finite number; or
                when no multiple of it lies inside the band's range. The last two are refused as the layer
                is drawn, which is where the band is read.

        Examples:
            - Contour a raster at five levels:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().contours(dem, levels=5)                # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                1

                ```
            - One level every 100 m, filled, from the same method:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().contours(dem, interval=100, filled=True)  # doctest: +SKIP
                >>> type(m.layers[0]).__name__                                  # doctest: +SKIP
                'Polygons'

                ```

        Returns:
            This map (chainable).

        See Also:
            digitalearth.base.levels.levels_every: the shared interval-to-levels arithmetic.
        """
        _one_spacing_only("contours", levels, interval)
        return self._contour_layer(
            data,
            band=band,
            levels=levels,
            interval=interval,
            filled=filled,
            name=name,
            visible=visible,
            opts=opts,
        )

    def _contour_layer(
        self,
        data: Any,
        *,
        band: int,
        levels: Any,
        filled: bool,
        interval: Optional[float] = None,
        name: Optional[str] = None,
        visible: bool = True,
        derived: Optional[Mapping[str, Any]] = None,
        opts: Optional[Mapping[str, Any]] = None,
        coloured_by_value: bool = True,
    ) -> Self:
        """Record the shared contour recipe: I1 image → ``holoviews.operation.contours`` → styled layer.

        The one place :meth:`contours` and :meth:`spaghetti` describe their layer, so the two cannot drift
        in what they record; :func:`draw_contours` traces it. A caller's ``levels`` always wins there;
        ``None`` consults ``autostyle.auto_style`` for the variable's canonical contour levels (#230) before
        falling back to the tier's 10.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to contour.
            levels: Contour levels — an int (count) or explicit sequence; ``None`` auto-resolves.
            filled: Whether the bands between the levels are filled, which is also what the layer's kind
                records: ``"filled_contours"`` against ``"contours"``.
            interval: Spacing between levels, or ``None``. Recorded rather than resolved: which multiples of
                it lie inside the band is a property of the band, which only the drawer reads. It reaches
                the description mutually exclusive with ``levels`` — :func:`_one_spacing_only` refuses both,
                for either spelling — so the drawer never has to choose between them.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            derived: Style this tier worked out for the caller rather than style the caller wrote —
                :meth:`spaghetti`'s per-member colour is the only one today. Filed under the derived
                bucket, because the bucket a builder files style in is what says whether anyone asked
                for it: `portable_encodings` lifts ``opts`` unfiltered, so a derived value routed
                through ``**opts`` was published as the caller's own style (review R2-H3).
            opts: The caller's own HoloViews style options, taken as a mapping rather than ``**opts``
                so a keyword named like one of this method's own parameters cannot collide with it.
            coloured_by_value: Whether the traced levels are coloured **by the value they trace**, which is
                what decides whether the layer publishes a colour encoding for a key to hang on (order 24).
                `True` for :meth:`contours`, whose levels are coloured along the band's own range. `False` is
                :meth:`spaghetti` cycling one flat colour per ensemble member: a member's strands all draw in
                that one colour, so there is nothing a colour key could label — and a `Guide` on it is
                refused rather than drawn as an empty box.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        held: Dict[str, Any] = {}
        described_opts = describe_opts(held, dict(opts or {}))
        # The tier's own half is described into its own bucket, and anything that cannot travel is held
        # under the same key — which is how `held_props` merges the two back together for the drawer.
        tier_held: Dict[str, Any] = {}
        described_tier = {
            key: describe(tier_held, key, value)
            for key, value in dict(derived or {}).items()
        }
        if tier_held:
            held[TIER_BUCKET] = tier_held
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="filled_contours" if filled else "contours",
            source=data,
            held=held,
            symbology=Symbology(
                # The traced levels are coloured along the band's range, so the layer's colour varies with the
                # band — the binding a legend or a colorbar is hung on. No scale: the levels are resolved when
                # the layer is drawn (a caller's `interval=` is a property of the band, not of this call), so
                # the domain is the engine's to measure.
                encodings=(
                    {"color": Encoding.by_field("color", coloured_by(data, band))}
                    if coloured_by_value
                    else {}
                ),
                props={
                    "via": "contours",
                    "band": band,
                    "levels": describe(held, "levels", levels),
                    "interval": describe(held, "interval", interval),
                    "filled": filled,
                    TIER_BUCKET: described_tier,
                    "opts": described_opts,
                },
            ),
        )

    #: Colour cycle used to distinguish ensemble members in :meth:`spaghetti` (Category10-ish).
    _SPAGHETTI_COLORS = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )

    def spaghetti(
        self,
        collection: Any,
        *,
        band: int = DEFAULT_BAND,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Overlay each member of a ``DatasetCollection`` as line contours (ensemble spaghetti).

        Each member gets a distinct colour from a cycling palette so the strands are
        distinguishable, unless the caller passes an explicit ``color``/``cmap`` in ``opts``.

        Args:
            collection: A pyramids ``DatasetCollection`` whose members share a grid.
            band: 1-based band contoured in every member.
            name: The caller's own name for the layers, used as their ids and labels; ``None``
                (default) generates one per member. One call draws **one layer per member**, so a
                single name is shared and the second and later take ``-2``, ``-3``, … (#321).
            visible: Whether the members are drawn. ``False`` builds them hidden **and** describes
                them hidden (#327).
            **opts: Extra HoloViews style options applied to each member's contour element. An
                explicit ``color`` or ``cmap`` here disables the per-member colour cycle.

        Examples:
            - Overlay a three-member ensemble as spaghetti contours:
                ```python
                >>> from pyramids.dataset.collection import DatasetCollection   # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)  # doctest: +SKIP
                >>> m = InteractiveMap().spaghetti(dc, levels=4)                # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                3

                ```

        Returns:
            This map (chainable) — one contour layer registered per member.
        """
        cycle_colour = "color" not in opts and "cmap" not in opts
        # `levels` is a parameter of the shared funnel rather than a style option, so it is taken out of
        # the caller's keywords here instead of being handed to it twice.
        levels = opts.pop("levels", None)
        for index, member in enumerate(collection.datasets):
            # The cycle goes in the *derived* bucket, not in the caller's. Passing it through `**opts`
            # filed a colour nobody asked for under the one bucket `portable_encodings` lifts unfiltered,
            # so three unstyled members published three different colours as three caller intents — and a
            # figure carried to another tier repainted itself in this tier's cycle (review R2-H3).
            derived = (
                {"color": self._SPAGHETTI_COLORS[index % len(self._SPAGHETTI_COLORS)]}
                if cycle_colour
                else {}
            )
            self._contour_layer(
                member,
                band=band,
                levels=levels,
                filled=False,
                name=name,
                visible=visible,
                derived=derived,
                # A member's strands are one flat colour — the cycle's, or the caller's own `color=` — so the
                # layer publishes no colour encoding and takes no colour key. A `cmap=` is the one keyword
                # that makes the strands vary with the value they trace, and only then is there a key to draw.
                coloured_by_value="cmap" in opts,
                opts=opts,
            )
        return self

    @_skips_off_limb
    def large_image(
        self,
        dataset: Any,
        *,
        band: int = DEFAULT_BAND,
        max_pixels: int = 4_000_000,
        dynamic: bool = True,
        cmap: Optional[str] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a large raster / COG by loading only the viewport at a decimated overview (DI.14).

        The raster analogue of the vector Datashader path: instead of materialising a multi-GB raster,
        it reads only the visible window at a suitable overview via **pyramids** and re-reads on pan/
        zoom. Cloud-native partial reads for remote COGs over ``/vsicurl/`` come for free.

        **Every frame is a read through the data tier** (#297): the window, the canvas and the budget are a
        :class:`~digitalearth.base.spec.RenderTarget` request, and
        :class:`~digitalearth.base.sources.view.SourceView` answers it — snapping the window to source pixel
        edges, never asking for more cells than the window holds, masking what is missing, and drawing an
        empty frame for a window off the data. A raster in another CRS is read through pyramids'
        ``Dataset.warped_view``, so each frame warps only its own window rather than the whole raster once.

        Args:
            dataset: A pyramids ``Dataset`` (ideally a COG with overviews). Must expose ``read_part``.
            band: 1-based band to read.
            max_pixels: Cell budget per rendered frame. The canvas follows the map's own width and height
                within it, so a wide map reads a wide window rather than a square one.
            dynamic: Re-read the viewport on pan/zoom via a ``RangeXY`` stream (needs a live server);
                ``False`` renders one frame of the whole raster within the budget (deterministic — what
                tests assert).
            cmap: Colormap; ``None`` (default) resolves from the band's variable name via
                ``autostyle.auto_style`` (#249), with ``"viridis"`` behind the lookup as the fallback.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when ``band`` is below 1 (the tier's band numbering is 1-based, like GDAL).
            AttributeError: when ``dataset`` lacks the pyramids COG/overview read surface
                (``read_part``/``preview``) — file a pyramids issue rather than reaching around it.

        Examples:
            - ``dynamic=False`` reads one frame of the whole raster — deterministic, and the one
              form that works with no live server behind it. A raster already under the budget comes back
              whole, so the budget only ever *caps* what is materialised:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> (dem.rows, dem.columns)                                    # doctest: +SKIP
                (13, 14)
                >>> m = InteractiveMap().large_image(dem, dynamic=False)       # doctest: +SKIP
                >>> m.layers[0].dimension_values(2, flat=False).shape          # doctest: +SKIP
                (13, 14)

                ```
            - The default ``dynamic=True`` registers a viewport-driven layer instead, carrying the
              one stream that re-reads the window from the axes ranges on every pan/zoom:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().large_image(dem)                      # doctest: +SKIP
                >>> len(m.layers[0].streams)                                   # doctest: +SKIP
                1
                >>> sorted(m.layers[0].streams[0].contents)                    # doctest: +SKIP
                ['x_range', 'y_range']

                ```
            - A raster without pyramids' windowed-read surface is refused by name, rather than
              being quietly read whole:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> try:                                                       # doctest: +SKIP
                ...     InteractiveMap().large_image(object())
                ... except AttributeError as error:
                ...     print(str(error).split(" (")[0])
                large_image needs pyramids' COG/overview read surface

                ```
        """
        _require_holoviz()
        # Both refusals stay with the builder, because each names an argument the caller wrote.
        if band < 1:
            raise ValueError(f"band is 1-based; got {band!r}")
        if not hasattr(dataset, "read_part") or not hasattr(dataset, "preview"):
            raise AttributeError(
                "large_image needs pyramids' COG/overview read surface (Dataset.read_part / "
                ".preview); upgrade pyramids or use field() for a small raster"
            )
        held: Dict[str, Any] = {}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="raster",
            source=dataset,
            held=held,
            symbology=Symbology(
                # Coloured by its band like `field`, and named the same warp-free way — which matters more
                # here than anywhere else, since the whole point of this builder is never to read the raster
                # whole. No scale: each frame is stretched to the window it read.
                encodings={
                    "color": Encoding.by_field("color", coloured_by(dataset, band))
                },
                props={
                    "via": "large_image",
                    "band": band,
                    "max_pixels": max_pixels,
                    "dynamic": dynamic,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                    "opts": described_opts,
                },
            ),
        )

    def _draw_large_image(self, dataset: Any, props: dict) -> Tuple[Any, dict]:
        """Build the windowed-read layer a `large_image` description asks for.

        Args:
            dataset: The raster the layer draws.
            props: The layer's recorded properties.

        Returns:
            ``(element, style)``: the element — a single frame, or the `DynamicMap` that re-reads the window
            on pan and zoom — and the backend-agnostic options every frame is drawn with, which is what the
            layer's `DrawnLayer.style` records.
        """
        _, hv = _require_holoviz()
        band = props["band"]
        opts = dict(props.get("opts") or {})
        dynamic = props.get("dynamic", True)
        ds = self._windowable(dataset)
        # Resolved once, from the band's name only: reading the array to build a full Source would
        # defeat the whole point of a windowed reader.
        cmap = self._auto_cmap_for_band(ds, band, props.get("cmap"))
        common = {"cmap": cmap, "colorbar": True, **opts}
        frame_opts = {"tools": ["hover"]}
        target = RenderTarget(
            "window", width=self.width, height=self.height, budget=props["max_pixels"]
        )
        # Filled in below with the `DynamicMap` the frames belong to, so each frame's style is filed
        # against the layer a caller holds rather than against whichever layer happened to register last
        # (review H3). It stays `None` for the static path, where the frame *is* the layer. `held` carries
        # the view between frames: the first read makes it, every later one re-reads through it.
        owner: Dict[str, Any] = {}
        held: Dict[str, Any] = {}

        def _read(request: Any) -> Any:
            """Answer one read request, making the layer's view the first time it is asked.

            Args:
                request: What this frame wants — the region, the canvas and the budget.

            Returns:
                The view of that window.
            """
            if "view" not in held:
                held["view"] = SourceView.of(
                    ds,
                    ref=DataRef.to_object(ds),
                    selection=Selection.of(band),
                    request=request,
                )
                return held["view"]
            return held["view"].reread(request)

        def _frame(x_range: Any = None, y_range: Any = None) -> Any:
            """Read the window the viewport asks for, and draw it.

            Args:
                x_range: The viewport's x range, or `None` for the first frame.
                y_range: Its y range, or `None`.

            Returns:
                The styled element for that window.
            """
            if x_range is None or y_range is None:
                # The first frame is the whole raster within the budget: a canvas with no region windows
                # the source itself, which is what `preview` used to approximate.
                request = target.view_request(Viewport(self.crs))
            else:
                request = target.view_request(
                    Viewport(self.crs),
                    bounds=Bounds(
                        x_range[0], y_range[0], x_range[1], y_range[1], crs=self.crs
                    ),
                )
            return self._styled(
                self._image_from_source(_read(request)),
                common=common,
                bokeh=frame_opts,
                owner=owner.get("layer"),
            )

        if not dynamic:
            return _frame(), common
        from holoviews.streams import RangeXY

        dmap = hv.DynamicMap(_frame, streams=[RangeXY()])
        owner["layer"] = dmap
        # The style every frame will be drawn with, filed against the layer a caller holds before any frame
        # exists, so `style_of` answers for it at once (review M17). It used to be filed by drawing a first
        # frame here and throwing it away, which is why the window was read at build time at all: a
        # `DynamicMap` is lazy, and now so is its layer's first read (review M8).
        self._record_style(dmap, self._style_record(common, frame_opts))
        return dmap, common

    def _windowable(self, dataset: Any) -> Any:
        """Return the raster to read windows from, warped lazily when the display CRS asks for it.

        Args:
            dataset: The raster the caller gave.

        Returns:
            The dataset itself when it is already in the display CRS; otherwise pyramids' lazily warped view
            of it (`Dataset.warped_view`), so each frame warps only the window it reads rather than the whole
            raster once, up front, at full resolution. A pyramids without `warped_view` falls back to warping
            the dataset, which is what this did before.
        """
        if not self._needs_reproject(dataset):
            return dataset
        warped_view = getattr(dataset, "warped_view", None)
        if warped_view is None:  # pragma: no cover - older pyramids
            return reproject(dataset, self.crs)
        return warped_view(self.crs)
