"""RasterMixin — web-tier raster builder (DW.1b, recipe W1).

``field`` puts a pyramids raster on the web map as a MapLibre **image source**: the band is reprojected
to lon/lat through pyramids, colour-mapped to an RGBA PNG (NoData → transparent), embedded as a ``data:`` URI,
and placed by its lon/lat corner coordinates. This is the offline, size-limited path; the large-raster
COG/XYZ-tile path (pyramids ``to_cog``/``to_xyz``) is a follow-up — it needs a tile server or PMTiles and is
tracked in the plan (DW.1b risks).

``rgb_composite`` and ``hsv_composite`` put *three* bands on the map the same way, and are one builder and one
drawer with the recipe recorded beside the layer: they share every step but what the three stretched channels
mean (#266). The stretch itself is ``digitalearth.base.stretch``'s, which states the rule both follow — "a
composite (RGB or HSV) turns three bands into an image by stretching each channel into ``[0, 1]``" — so a
composite looks the same here as on the static tier.

matplotlib (the colormap → RGBA → PNG encoding, and the HSV → RGB conversion) and numpy are imported lazily
inside the methods, so importing the tier needs neither the ``web`` extra nor matplotlib at module load.
"""

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Self, Sequence, Tuple

from loguru import logger

from digitalearth.base.deprecation import renamed_method
from digitalearth.base.spec import DEFAULT_BAND, Bounds, LayerSpec, Scale, Symbology
from digitalearth.base.stretch import DEFAULT_COMPOSITE_BANDS, require_three_bands
from digitalearth.web.base import _require_layer_api, as_finite

#: Pixel count above which the inline image-source path is warned against (use COG/XYZ tiles for big rasters).
_LARGE_RASTER_PIXELS = 4_000_000

#: The recipe that maps the three stretched channels straight to red, green and blue. It is also what an
#: ``rgb`` layer whose description records no recipe at all means: every figure written before this tier had a
#: second composite is one of those, and read as anything else they would be refused or quietly recoloured.
_RGB_RECIPE = "rgb_composite"

#: The recipe that reads the same three channels as hue, saturation and value.
_HSV_RECIPE = "hsv_composite"

#: The composites this tier draws, each mapped to the prefix its generated layer ids take. One structure, so
#: the recipes the tier *has* and the ids it mints for them cannot come to name different sets.
_COMPOSITE_RECIPES: Mapping[str, str] = MappingProxyType(
    {_RGB_RECIPE: "rgb", _HSV_RECIPE: "hsv"}
)


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _colour_limits(
    limits: Optional[Sequence[float]], vmin: Any, vmax: Any, *, caller: str
) -> Tuple[Any, Any]:
    """Resolve the contract's `limits=` against this tier's own `vmin`/`vmax`.

    `limits` is the one spelling every tier answers to (#299); `vmin`/`vmax` are what this tier took before
    it, and both still work. What is refused is naming the same thing twice, which has no right answer.

    Args:
        limits: `(vmin, vmax)`, or `None`.
        vmin: The lower limit, or `None`.
        vmax: The upper limit, or `None`.
        caller: The method, for the message.

    Returns:
        The `(vmin, vmax)` pair to colour with.

    Raises:
        ValueError: when `limits` is given alongside either of the others, or is not a pair.
    """
    if limits is None:
        return vmin, vmax
    if vmin is not None or vmax is not None:
        raise ValueError(
            f"{caller} takes limits= or vmin=/vmax=, not both; they name the same thing"
        )
    # A sequence of two numbers, checked as such. `low, high = limits` unpacks a two-character string and a
    # two-element iterator just as happily, and the failure surfaced further down as numpy's own message
    # about a value this function had already seen (review L9).
    if isinstance(limits, (str, bytes)) or not isinstance(limits, Sequence):
        raise ValueError(
            f"{caller} limits must be a (vmin, vmax) pair of numbers; got {limits!r}"
        )
    if len(limits) != 2:
        raise ValueError(
            f"{caller} limits must be a (vmin, vmax) pair of numbers; got {len(limits)} values"
        )
    try:
        return float(limits[0]), float(limits[1])
    except (TypeError, ValueError):
        raise ValueError(
            f"{caller} limits must be a (vmin, vmax) pair of numbers; got {limits!r}"
        ) from None


def _grid_pixels(data: Any) -> Optional[int]:
    """Return how many cells a raster's grid holds, or ``None`` when it does not report one.

    Read from the grid rather than from a band's values: the composite builder holds the warped dataset and
    not a band, and reading one only to measure it would cost a whole band read.

    Args:
        data: A pyramids ``Dataset`` in the display CRS.

    Returns:
        ``rows * columns``, or ``None`` for an input that reports neither as a plain integer — a raster is
        the only thing that does, and a size that cannot be measured is not worth guessing at.
    """
    rows, columns = getattr(data, "rows", None), getattr(data, "columns", None)
    if isinstance(rows, int) and isinstance(columns, int):
        return rows * columns
    return None


def _warn_if_large(caller: str, noun: str, pixels: Optional[int]) -> None:
    """Warn, at the call, when a raster is too large to inline as a ``data:`` image source.

    Both raster builders embed their pixels in the page as a base64 PNG, so both have the same ceiling and
    the same advice. The warning belongs to the **builder**: a drawer runs again every time the layer is
    drawn back from its description, so a stored composite repeated the advice once per draw for an input
    the caller chose once — while its sibling ``field`` warned from the call (review N7).

    Args:
        caller: The builder's name, which opens the message.
        noun: What is being inlined — a ``"band"`` or a ``"composite"``.
        pixels: How many cells it holds, or ``None`` when that could not be measured.
    """
    if pixels is None or pixels <= _LARGE_RASTER_PIXELS:
        return
    logger.warning(
        "{}: inlining a {}-pixel {} as a data-URI image source bloats the page; for large rasters serve "
        "COG/XYZ tiles from pyramids instead",
        caller,
        pixels,
        noun,
    )


def _limit_pairs(limits: Any) -> Optional[List[Tuple[Any, ...]]]:
    """Return per-channel limits as a list of pairs, or `None` when they are not shaped like any.

    The one guard both limit translators need, and the one each had its own copy of. Neither of them is the
    place a malformed ``limits=`` is diagnosed: :func:`~digitalearth.base.stretch.stretch_to_unit` owns that
    message and names the argument, so a value this cannot read is handed back to the caller untouched to
    reach it — rather than failing here, on iterating a float or unpacking a triple, as a `TypeError` about a
    private helper nobody called.

    Args:
        limits: The caller's `limits=`, already known not to be `None`.

    Returns:
        One tuple per channel when `limits` is an iterable of two-element pairs; `None` for anything else,
        including a bare number and a sequence of triples.
    """
    try:
        pairs = [tuple(pair) for pair in limits]
    except TypeError:
        return None
    if any(len(pair) != 2 for pair in pairs):
        return None
    return pairs


def _recorded_limits(limits: Any) -> Any:
    """Return per-channel stretch limits in the spelling a figure can be written in.

    :func:`~digitalearth.base.stretch.channel_limits` answers `(nan, nan)` for a channel it could not
    measure — a normal result, and the one a caller freezing a series on its first frame passes to every
    later frame. NaN has no JSON form, so a description holding one cannot be written at all: `to_dict`
    refuses the whole figure. A bound the freeze does not carry is recorded as `None` instead, which says
    the same thing, and :func:`_stretch_limits` reads it back as the NaN the stretch expects.

    Args:
        limits: The caller's `limits=`, or `None`.

    Returns:
        The limits with every non-finite bound as `None` and every finite one as a plain `float`; `None`
        unchanged; and anything not shaped as a sequence of pairs unchanged, so that
        :func:`~digitalearth.base.stretch.stretch_to_unit` refuses it in its own words rather than this
        failing on it first.
    """
    if limits is None:
        return None
    pairs = _limit_pairs(limits)
    if pairs is None:
        return limits
    try:
        return tuple(
            tuple(
                None
                if value is None or not math.isfinite(float(value))
                else float(value)
                for value in pair
            )
            for pair in pairs
        )
    except (TypeError, ValueError):
        return limits


def _stretch_limits(limits: Any) -> Any:
    """Return recorded limits as :func:`~digitalearth.base.stretch.stretch_to_unit` takes them.

    Args:
        limits: The recorded `limits` prop.

    Returns:
        The limits with every `None` bound back as `nan`, which is how the stretch spells "no frozen bound
        for this channel"; anything else unchanged.
    """
    if limits is None:
        return None
    pairs = _limit_pairs(limits)
    if pairs is None:
        return limits
    return [
        tuple(float("nan") if value is None else value for value in pair)
        for pair in pairs
    ]


def _composed_rgb(via: str, unit_stack: Any) -> Any:
    """Return three stretched channels as the RGB the image source is encoded from.

    The one step the two composites do not share. Everything up to here — one warp, one band stack, one
    per-channel stretch into ``[0, 1]`` — is :mod:`digitalearth.base.stretch`'s shared rule for both; what
    differs is only what those three numbers per cell *mean*.

    Args:
        via: The layer's recorded recipe, one of :data:`_COMPOSITE_RECIPES`.
        unit_stack: The ``(rows, cols, 3)`` stretched stack; NaN marks a cell a band has no value for.

    Returns:
        The stack unchanged for an RGB composite. For an HSV one, the same three channels read as hue,
        saturation and value and converted to RGB — with every cell that was incomplete left NaN, so the
        encoder still draws it transparent.
    """
    if via != _HSV_RECIPE:
        return unit_stack
    import numpy as np
    from matplotlib.colors import hsv_to_rgb

    stack = np.asarray(unit_stack, dtype=float)
    complete = np.isfinite(stack).all(axis=-1)[..., None]
    # Masked before the conversion and restored after it, rather than handed over as it is. `hsv_to_rgb` reads
    # the hue with `(h * 6.0).astype(int)`, and casting NaN yields a garbage index rather than NaN, so a cell
    # with no hue comes back *finite* — measured: `(0.0, 0.0, 0.0)` at full saturation and value,
    # `(0.5, 1.04e-311, 1.04e-311)` at half — and therefore opaque, with only a RuntimeWarning about an
    # invalid cast to say anything had happened. The tier's NoData contract (a cell missing any band is
    # transparent) has to survive the colour space.
    return np.where(complete, hsv_to_rgb(np.where(complete, stack, 0.0)), np.nan)


def _placed_corners(web_map: Any, source: Any, caller: str) -> Any:
    """Return a raster's lon/lat corners and frame the map on them, or report that it cannot be placed.

    Computed here rather than in the builder so a source's corners are read **once**: the builder records
    what was asked for, and everything derived from the data is derived where the data is read.

    Args:
        web_map: The map being drawn.
        source: The display source whose corners are wanted.
        caller: The builder's name, for the skip message.

    Returns:
        The four lon/lat corners, or `None` when they cannot be expressed — reported through the map's own
        skip log, so an unplaceable layer is refused rather than drawn somewhere wrong.

    Raises:
        OffLimbError: when the map is `strict`, which is what that flag asks for: a pipeline that must not
            publish a half-drawn map gets the refusal back instead of a warning.
    """
    corners = web_map._lonlat_corners(source)
    if corners is None:
        web_map._skipped(
            caller,
            "the raster's corners cannot be expressed in lon/lat, which a MapLibre image source "
            "needs; reproject the dataset so its extent is representable",
        )
        return None
    # Already lon/lat, so the framing takes them as they are.
    web_map._note_lonlat_bounds(
        (corners[0][0], corners[2][1], corners[1][0], corners[0][1])
    )
    return corners


def _image_layer(_web_map: Any, layer: LayerSpec, url: str, coordinates: Any) -> Any:
    """Package an image source and the raster layer reading it.

    Both raster kinds draw the same way — a ``data:`` PNG placed on four lon/lat corners — and differ only
    in how the image is made, so the MapLibre half is written once.

    Args:
        _web_map: Unused — taken so this reads with the same first argument as the two drawers that call
            it, which do need the map.
        layer: The layer's description. Its `opacity` prop and its visibility are what MapLibre is given;
            everything else about the image is already in `url`.
        url: The ``data:image/png;base64,`` URI to place.
        coordinates: The four lon/lat corners, clockwise from the north-west.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`.

    Raises:
        ValueError: when the description records no `opacity` for the raster, naming the layer, its kind
            and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, layer_types = _require_layer_api()
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec={"type": "image", "url": url, "coordinates": coordinates},
        layer=layer_cls(
            id=layer.id,
            type=layer_types.RASTER,
            source=source_id,
            paint={
                "raster-opacity": float(required_props(layer, "opacity")["opacity"])
            },
            layout={"visibility": "visible" if layer.visible else "none"},
        ),
    )


def draw_field(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Encode one band as a coloured image and place it on the map.

    Args:
        web_map: The map being drawn.
        data: The layer's source — a pyramids dataset or an array.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`, or `None` when the band cannot be placed — it
        lies outside what the display CRS can show, or its corners will not express as lon/lat. Either way
        it has been reported through the map's skip log before this answers.

    Raises:
        ValueError: when the description records none of the values the image is encoded from — its
            band, its colour map or its colour limits — naming the layer, its kind and what is missing.
        OffLimbError: when the map is `strict` and the band cannot be placed, in place of the `None`.
    """
    import numpy as np

    from digitalearth.web.renderer import required_props

    props = required_props(layer, "band", "cmap", "vmin", "vmax", "opacity")
    source = web_map._display_source_or_skip(data, band=props["band"], layer="field")
    if source is None:
        return None
    values = source.z.values
    y = np.asarray(source.y.values, dtype=float)
    if y.size > 1 and y[0] < y[-1]:
        # Ascending y → flip so PNG row 0 is the northern edge.
        values = values[::-1]
    url = web_map._rgba_png_datauri(
        values, props["cmap"], vmin=props["vmin"], vmax=props["vmax"]
    )
    coordinates = _placed_corners(web_map, source, "field")
    if coordinates is None:
        return None
    return _image_layer(web_map, layer, url, coordinates)


def _drawn_recipe(layer: LayerSpec, props: dict) -> str:
    """Return which composite a description asks for, refusing one this tier does not draw.

    Args:
        layer: The layer being drawn, named in the refusal.
        props: Its recorded props, already thawed.

    Returns:
        The recorded recipe, or :data:`_RGB_RECIPE` when the description records none — which is every
        ``rgb`` layer written before this tier had a second composite.

    Raises:
        ValueError: when the recipe is one this tier has no composite for. Falling back to the default
            instead would draw a figure written by another version as something it does not claim to be;
            the static renderer refuses an unknown recipe the same way.
    """
    via = props.get("via", _RGB_RECIPE)
    if via not in _COMPOSITE_RECIPES:
        raise ValueError(
            f"layer {layer.id!r} records {via!r} as how its composite was drawn; the web tier draws one of "
            f"{sorted(_COMPOSITE_RECIPES)}"
        )
    return str(via)


def draw_rgb_composite(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Encode three bands as one colour composite image and place it on the map.

    Both composites are drawn here — the recipe is the layer's own, recorded under ``via`` — because they
    differ in one step out of six: the warp, the band stack, the stretch, the north-up flip, the PNG and the
    placement are the same work either way. It keeps the name it is registered under in
    :func:`~digitalearth.web.renderer.drawer_for`.

    Args:
        web_map: The map being drawn.
        data: The layer's source — the dataset the bands are read from, in whatever CRS it was recorded
            in. It is warped to the display CRS here, so a figure handed straight to the renderer draws
            the same image as the builder call that recorded it.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`, or `None` when the composite cannot be placed —
        it lies outside what the display CRS can show, or its corners will not express as lon/lat. Either
        way it has been reported through the map's skip log before this answers.

    Raises:
        ValueError: when the description records none of the values the composite is built from — its
            three bands, its stretch limits or whether NoData is masked — naming the layer, its kind and
            what is missing; when it records a recipe this tier has no composite for; when it records a band
            count other than three, in :func:`~digitalearth.base.stretch.require_three_bands`' words; when
            the recorded limits do not hold one `(lo, hi)` pair per band; or when no pixel is finite in all
            three, since there is nothing to draw and an empty image would read as a rendering failure
            rather than as an empty input.
        OffLimbError: when the map is `strict` and the composite cannot be placed, in place of the `None`.
    """
    import numpy as np

    from digitalearth.base.sources import get_stack
    from digitalearth.web.renderer import required_props

    props = required_props(layer, "bands", "mask_nodata", "limits", "opacity")
    via = _drawn_recipe(layer, props)
    bands = list(props["bands"])
    # The same guard the builder runs, because this is the other way in: a figure is drawn from its
    # description — by a redraw onto another view, or from JSON written elsewhere — with no builder in front
    # of it. Without it the count reaches `get_stack`, which answers with numpy's "could not broadcast input
    # array from shape (4,9,2) into shape (4,9,3)": a complaint about an array the caller never named.
    require_three_bands(via, bands)
    # One warp for both halves: the pixels are read from the warped dataset and the corners are taken from
    # that same grid. Stacking `data` itself drew the source-CRS grid stretched over the warped grid's
    # extent — a different shape at a different resolution (review H1).
    data = web_map._display_raster_or_skip(data, layer=via)
    if data is None:
        return None
    stack = get_stack(data, bands, mask=props["mask_nodata"])
    source = web_map._to_display_source(data, band=bands[0])
    y = np.asarray(source.y.values, dtype=float)
    if y.size > 1 and y[0] < y[-1]:
        # Ascending y → flip so PNG row 0 is the north edge.
        stack = stack[::-1]
    from digitalearth.base.stretch import stretch_to_unit

    url = web_map._composite_png_datauri(
        _composed_rgb(via, stretch_to_unit(stack, _stretch_limits(props["limits"]))),
        caller=via,
    )
    coordinates = _placed_corners(web_map, source, via)
    if coordinates is None:
        return None
    return _image_layer(web_map, layer, url, coordinates)


class RasterMixin(_MixinBase):
    """Raster builder for :class:`~digitalearth.web.map.WebMap` (image-source path)."""

    def field(
        self,
        data: Any,
        *,
        band: int = DEFAULT_BAND,
        cmap: Any = None,
        units: Optional[str] = None,
        opacity: float = 1.0,
        limits: Optional[Sequence[float]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        visible: bool = True,
        name: Optional[str] = None,
    ) -> Self:
        """Overlay a pyramids raster band as a colour-mapped MapLibre image source (recipe W1).

        The band is reprojected to the display CRS (lon/lat) through pyramids, normalised over its finite
        range (or the explicit ``limits``/``vmin``/``vmax``), colour-mapped with ``cmap`` (autostyle default when
        ``None``), and embedded as an RGBA PNG data-URI placed by its lon/lat corners. Masked / non-finite
        cells become fully transparent.

        Args:
            data: A pyramids ``Dataset`` (or anything ``get_source`` accepts).
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            band: 1-based band to draw.
            cmap: A registered matplotlib colormap name, or a ``Colormap`` itself — the classified
                builders take either and so does this one (#315, review H3). A colormap built on the spot
                colours the image but has no name a figure read elsewhere could resolve. ``None`` resolves
                the autostyle default for the variable.
            limits: The contract's name for the colour limits, as ``(vmin, vmax)`` — the one spelling every
                tier answers to (#299). ``vmin``/``vmax`` remain, and naming both is refused rather than
                silently resolved one way.
            units: What the band's values are measured in, recorded as
                :attr:`~digitalearth.web.base.WebMapBase.last_units` so a key built from them can say so.
                ``None`` (the default) takes the variable's units from
                :func:`~digitalearth.base.autostyle.auto_style`, and leaves them unknown when it carries
                none — a unit is never guessed. Pass one to correct a band the library mis-identifies, or
                to name the units of a variable it does not know.
            opacity: Raster layer opacity in ``[0, 1]``.
            vmin: Lower colour limit; ``None`` uses the band's finite minimum.
            vmax: Upper colour limit; ``None`` uses the band's finite maximum.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible. ``False`` builds it hidden, which is how
                :meth:`~digitalearth.web.temporal.TemporalMixin.timeslider` stacks time steps without
                every frame showing at once — including in a saved page, which carries no slider.

        Returns:
            This map (chainable). When the band cannot be placed — it lies outside what the display CRS
            can show, or its corners will not express as lon/lat — nothing is added: the layer is skipped
            with a warning, or the error is raised when the map was built with ``strict=True``.

        Raises:
            ValueError: when `limits` is given alongside `vmin`/`vmax` — they name the same thing — or is
                not a `(vmin, vmax)` pair of numbers; and when ``opacity`` is not a finite number, refused
                at this call because a figure holding NaN or infinity could not be written down.
            KeyError: when `cmap` names no registered colormap, or `dataset` is a URL with no resolver.
            TypeError: when `cmap` is neither a name, a ``Colormap``, nor a sequence of colours.
            FileNotFoundError: when `data` is a path that names nothing.
            OffLimbError: only when the map was built with ``strict=True`` and the band cannot be
                placed; by default that layer is skipped with a warning instead, so one unplaceable
                raster does not cost the map the layers around it.

        Examples:
            - Colour-map a band and address the layer afterwards by the name it was given (needs
              the ``web`` extra, so the block is skipped without it):
                ```python
                >>> import numpy as np                               # doctest: +SKIP
                >>> from digitalearth.base.sources import get_source  # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> src = get_source(                                # doctest: +SKIP
                ...     np.arange(12.0).reshape(3, 4),
                ...     x=np.array([0.0, 1.0, 2.0, 3.0]),
                ...     y=np.array([2.0, 1.0, 0.0]),
                ... )
                >>> m = WebMap().field(src, cmap="viridis", name="dem")       # doctest: +SKIP
                >>> m.layer_ids, len(m.layers)                       # doctest: +SKIP
                (['dem'], 1)

                ```
            - The band also hands the map its extent, so the view frames itself and
              :meth:`~digitalearth.web.base.WebMapBase.set_bounds` has something to frame on;
              on an empty map the same call raises instead:
                ```python
                >>> m.set_bounds() is m                              # doctest: +SKIP
                True

                ```
            - ``visible=False`` builds the layer hidden, which is how
              :meth:`~digitalearth.web.temporal.TemporalMixin.timeslider` stacks one layer per time
              step without every frame showing at once — in a saved page too, which has no slider:
                ```python
                >>> m = WebMap().field(src, visible=False, name="t0")       # doctest: +SKIP
                >>> m.layer_ids                                      # doctest: +SKIP
                ['t0']

                ```

        See Also:
            digitalearth.web.raster.RasterMixin.rgb_composite: the three-band composite path.
            digitalearth.web.vector.VectorMixin.contours: draws the same field as vectors.
        """
        vmin, vmax = _colour_limits(limits, vmin, vmax, caller="WebMap.field()")
        opacity = as_finite(opacity, "opacity", "WebMap.field()")
        _require_layer_api()
        source = self._display_source_or_skip(data, band=band, layer="field")
        if source is None:
            return self
        cmap_name = self._auto_cmap(source, cmap)
        # Carried for a key built from this band's values (see `_auto_units`); `None` when unknown.
        self.last_units = self._auto_units(source, units)

        _warn_if_large("field", "band", int(getattr(source.z.values, "size", 0)))
        layer_id = self._layer_id("raster", name)
        # The colour map is resolved here because `_auto_cmap` reads the band's own metadata, which is the
        # caller's request as much as `cmap=` is. The image — orientation included — is encoded by
        # `draw_field`, which is handed the band already warped above rather than warping the recorded
        # `data` a second time (review M8).
        if self._index_layer(
            layer_id,
            name,
            kind="raster",
            visible=visible,
            source=data,
            placed=source,
            symbology=Symbology(
                props={
                    "cmap": cmap_name,
                    "vmin": vmin,
                    "vmax": vmax,
                    "opacity": float(opacity),
                    "band": band,
                }
            ),
        ):
            self._last_layer_id = layer_id
        return self

    #: Deprecated spelling of :meth:`field`, the contract's name for a raster band drawn as a coloured field
    #: (#299). It forwards and warns.
    add_raster = renamed_method(new="field", old="add_raster", owner="WebMap")

    def rgb_composite(
        self,
        dataset: Any,
        bands: Any = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[Any] = None,
        opacity: float = 1.0,
        visible: bool = True,
        name: Optional[str] = None,
    ) -> Self:
        """Overlay three bands as a true- or false-colour image (recipe W1).

        Satellite imagery is a headline use of a web map, and the tier could only draw one band through a
        colormap. The stretch comes from :mod:`digitalearth.base.stretch` — the same engine-neutral code
        the static tier's ``rgb_composite`` uses — so the same three bands look the same on both tiers.

        The three channels are read straight as red, green and blue; :meth:`hsv_composite` reads the same
        three as hue, saturation and value.

        Args:
            dataset: A pyramids ``Dataset`` (or anything ``get_stack`` accepts).
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            bands: The three 1-based band numbers, in red-green-blue order.
            mask_nodata: Whether NoData becomes NaN (and so transparent) rather than a real value.
            limits: Per-channel ``(lo, hi)`` stretch limits in band order. ``None`` derives them from this
                image; pass a fixed set to keep a series comparable across frames.
            opacity: Raster layer opacity in ``[0, 1]``.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.

        Returns:
            The same map instance, so builder calls chain. A composite that cannot be placed — off-limb
            in the display CRS, or with corners that will not express as lon/lat — is skipped with a
            warning instead, unless the map was built with ``strict=True``.

        Raises:
            ValueError: when ``bands`` is not exactly three, when ``limits`` does not match them, when the
                composite has no finite pixels to draw, or when ``opacity`` is not a finite number —
                refused at this call, because a figure holding NaN or infinity could not be written down.
            FileNotFoundError: when `dataset` is a path that names nothing, or KeyError when no resolver
                is registered for its URL scheme — from :meth:`~digitalearth.web.base.WebMapBase._opened`.

        Examples:
            - A true-colour composite from a Landsat-ordered dataset:
                ```python
                >>> from digitalearth.web import WebMap                       # doctest: +SKIP
                >>> WebMap().basemap().rgb_composite(ds, bands=(4, 3, 2))     # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.raster.RasterMixin.hsv_composite: the same three bands read as hue/sat/value.
            digitalearth.base.stretch.channel_limits: derives the ``limits`` this accepts.
            digitalearth.web.raster.RasterMixin.add_raster: the single-band, colormapped path.
        """
        return self._composite(
            _RGB_RECIPE,
            dataset,
            bands,
            mask_nodata=mask_nodata,
            limits=limits,
            opacity=opacity,
            visible=visible,
            name=name,
        )

    def hsv_composite(
        self,
        dataset: Any,
        bands: Any = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[Any] = None,
        opacity: float = 1.0,
        visible: bool = True,
        name: Optional[str] = None,
    ) -> Self:
        """Overlay three bands as an HSV composite — hue, saturation and value → RGB (#266).

        The static tier has drawn both composites since it drew one, and this tier drew only RGB without
        declaring HSV absent, so it was missing rather than refused. It is the same concept through this
        tier's own engine: the three bands are stretched into ``[0, 1]`` by the shared
        :mod:`digitalearth.base.stretch` — "a composite (RGB or HSV) turns three bands into an image by
        stretching each channel into ``[0, 1]``" — then read as hue, saturation and value and encoded as the
        one RGBA PNG a MapLibre image source takes.

        It is the natural reading for a magnitude-and-direction pair (direction as hue, magnitude as value)
        and for anything whose interesting variable is a *ratio* rather than a colour.

        Args:
            dataset: A pyramids ``Dataset`` (or anything ``get_stack`` accepts).
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            bands: The three 1-based band numbers, in hue-saturation-value order.
            mask_nodata: Whether NoData becomes NaN (and so transparent) rather than a real value. A cell
                missing *any* of the three has no colour and is drawn transparent, as in ``rgb_composite``
                — including a missing hue, which the conversion would otherwise turn into a real colour.
            limits: Per-channel ``(lo, hi)`` stretch limits in band order. ``None`` derives them from this
                image; pass a fixed set to keep a series comparable across frames.
            opacity: Raster layer opacity in ``[0, 1]``.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.

        Returns:
            The same map instance, so builder calls chain. A composite that cannot be placed — off-limb
            in the display CRS, or with corners that will not express as lon/lat — is skipped with a
            warning instead, unless the map was built with ``strict=True``.

        Raises:
            ValueError: when ``bands`` is not exactly three, when ``limits`` does not match them, when the
                composite has no finite pixels to draw, or when ``opacity`` is not a finite number —
                refused at this call, because a figure holding NaN or infinity could not be written down.
            FileNotFoundError: when `dataset` is a path that names nothing, or KeyError when no resolver
                is registered for its URL scheme — from :meth:`~digitalearth.web.base.WebMapBase._opened`.

        Examples:
            - Three bands read as hue, saturation and value (needs the ``web`` extra, so the block is
              skipped without it):
                ```python
                >>> from digitalearth.web import WebMap                       # doctest: +SKIP
                >>> WebMap().basemap().hsv_composite(ds, bands=(1, 2, 3))     # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.raster.RasterMixin.rgb_composite: the same three bands as red/green/blue.
            digitalearth.static.maps.raster.RasterMixin.hsv_composite: the matplotlib tier's counterpart,
                which this answers to on every parameter the two share.
        """
        return self._composite(
            _HSV_RECIPE,
            dataset,
            bands,
            mask_nodata=mask_nodata,
            limits=limits,
            opacity=opacity,
            visible=visible,
            name=name,
        )

    def _composite(
        self,
        via: str,
        dataset: Any,
        bands: Any,
        *,
        mask_nodata: bool,
        limits: Optional[Any],
        opacity: float,
        visible: bool,
        name: Optional[str],
    ) -> Self:
        """Record a three-band composite and draw it, in whichever colour space ``via`` names.

        Everything the two composites do before the colour space is here: the argument checks, the one warp
        that refuses an off-limb dataset before anything is recorded, the size warning and the description.
        The recipe goes into that description rather than staying on the map, because a figure is redrawn
        without its builder — a redraw that fell back to RGB would be a different image with no other symptom
        (see :func:`_composed_rgb`).

        Args:
            via: The recipe, one of :data:`_COMPOSITE_RECIPES`. It names the builder in every message the
                call can raise or log, and is what the drawer reads back.
            dataset: The multiband source, as the caller gave it.
            bands: The three 1-based band numbers, in the order the recipe reads them.
            mask_nodata: Whether NoData becomes NaN (and so transparent).
            limits: Frozen per-channel ``(lo, hi)`` stretch bounds, or ``None`` for a per-call scan.
            opacity: Raster layer opacity in ``[0, 1]``.
            visible: Whether the layer starts visible.
            name: The caller's name for the layer; ``None`` generates one under the recipe's own prefix.

        Returns:
            The map, so builder calls chain.
        """
        opacity = as_finite(opacity, "opacity", f"WebMap.{via}()")
        _require_layer_api()
        require_three_bands(via, bands)
        # Warped here only so an off-limb dataset is refused before anything is recorded, and so the first
        # draw is handed the warped dataset rather than warping it again. The stack, the size warning, the
        # north-up flip and the placement are all the drawer's, which is the one place they are computed
        # (review M8).
        data = self._display_raster_or_skip(dataset, layer=via)
        if data is None:
            return self
        # At the call, like `field`'s: the drawer runs again on every redraw, and the size of the input is
        # the caller's one-time choice (review N7).
        _warn_if_large(via, "composite", _grid_pixels(data))
        layer_id = self._layer_id(_COMPOSITE_RECIPES[via], name)
        if self._index_layer(
            layer_id,
            name,
            # The engine-neutral kind for a multi-band colour composite, which is what both of these are —
            # the recipe below says which. The static tier records both under it too.
            kind="rgb",
            visible=visible,
            source=dataset,
            placed=data,
            symbology=Symbology(
                props={
                    "via": via,
                    "bands": tuple(int(band) for band in bands),
                    # A bound the caller's freeze could not measure is recorded as `None`, not as the NaN
                    # `channel_limits` answers with: a description holds only what a figure can be written
                    # as, and NaN has no JSON form (review M9's web instance).
                    "limits": _recorded_limits(limits),
                    "opacity": float(opacity),
                    "mask_nodata": bool(mask_nodata),
                }
            ),
        ):
            self._last_layer_id = layer_id
        return self

    @staticmethod
    def _composite_png_datauri(unit_stack: Any, *, caller: str = "a composite") -> str:
        """Encode a stretched ``(rows, cols, 3)`` stack as a ``data:image/png;base64,`` URI.

        Args:
            unit_stack: Channel values already stretched to ``[0, 1]`` and read as red, green and blue by
                here (an HSV composite has been converted — see :func:`_composed_rgb`); NaN marks NoData.
            caller: What to name in the refusal below. Both composites reach this encoder, so a builder
                written into the message would send the reader of the other one's failure to the wrong call.

        Returns:
            The PNG data-URI string.

        Raises:
            ValueError: when no pixel is finite in all three channels — there is nothing to draw, and an
                empty image would look like a rendering failure instead of an empty input.
        """
        import base64
        import io

        import numpy as np
        from matplotlib import image as mpimage

        stack = np.asarray(unit_stack, dtype=float)
        valid = np.isfinite(stack).all(axis=-1)
        if not valid.any():
            raise ValueError(
                f"{caller} got a stack with no pixel finite in all three bands"
            )
        rgba = np.zeros(stack.shape[:2] + (4,), dtype=float)
        rgba[..., :3] = np.clip(np.where(np.isfinite(stack), stack, 0.0), 0.0, 1.0)
        rgba[..., 3] = valid.astype(float)  # NoData in any channel → transparent
        buffer = io.BytesIO()
        mpimage.imsave(buffer, (rgba * 255).astype("uint8"), format="png")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def _lonlat_corners(self, source: Any) -> Optional[list]:
        """Return a raster's corner coordinates as the lon/lat an image source is placed by.

        MapLibre positions an ``image`` source with ``[[lng, lat], …]``, so a display CRS that is not
        lon/lat has to be converted — otherwise the map frames on the right degrees while the image sits in
        metre-space a long way off.

        Args:
            source: The display-CRS source whose ``x``/``y`` coordinate arrays give the corners.

        Returns:
            ``[TL, TR, BR, BL]`` in lon/lat, or ``None`` when the corners cannot be converted.
        """
        corners = self._image_coordinates(source.x.values, source.y.values)
        west, north = corners[0]
        east, south = corners[2]
        converted = self._as_lonlat(west, south, east, north)
        if converted is None:
            return None
        west, south, east, north = converted
        return [[west, north], [east, north], [east, south], [west, south]]

    @staticmethod
    def _image_coordinates(x: Any, y: Any) -> List[List[float]]:
        """Return the image-source corner coordinates ``[TL, TR, BR, BL]`` in ``[lng, lat]``.

        The corners are the cell *edges* — half the outermost spacing beyond the centres on each side
        (:meth:`Bounds.cell_edges`). Taken from the centres themselves, an image source is drawn half a cell
        inside the data all the way round, and a map framed on those corners is inset by the same amount
        (#301).

        Args:
            x: 1-D x / longitude cell-centre coordinates (display CRS, lon/lat).
            y: 1-D y / latitude cell-centre coordinates.

        Returns:
            The four corners top-left, top-right, bottom-right, bottom-left as ``[lng, lat]`` pairs — the
            order MapLibre's image source expects.
        """
        west, south, east, north = Bounds.cell_edges(x, y, crs=None).as_bbox()
        return [[west, north], [east, north], [east, south], [west, south]]

    @staticmethod
    def _rgba_png_datauri(
        values: Any,
        cmap: Any,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> str:
        """Colour-map a 2-D array to an RGBA PNG and return it as a ``data:image/png;base64,`` URI.

        Non-finite / masked cells are rendered fully transparent (the tier's NoData contract). matplotlib
        is imported lazily here so the tier imports without it.

        Args:
            values: A 2-D (possibly masked) array of band values, already oriented north-up.
            cmap: A registered matplotlib colormap name, or a ``Colormap`` itself. Resolved through
                :func:`~digitalearth.base.symbology.as_colormap`, the one resolver every tier reads a
                colormap through, so the unclassified raster path accepts the same object the classified
                ones do: looking the argument up as a dict key raised ``TypeError: unhashable type`` for a
                ``Colormap``, which defines ``__eq__`` and so has no hash (#315, review H3).
            vmin: Lower colour limit; ``None`` uses the finite minimum.
            vmax: Upper colour limit; ``None`` uses the finite maximum.

        Returns:
            The PNG data-URI string.

        Raises:
            ValueError: when the array has no finite values to colour.
            KeyError: when ``cmap`` is hashable but names no colormap matplotlib's registry holds, which
                is matplotlib's own message naming it.
            TypeError: when ``cmap`` is unhashable — a list of colours, say — which
                :func:`~digitalearth.base.symbology.as_colormap` reports as ``unhashable type: 'list'``.
        """
        import base64
        import io

        import numpy as np
        from matplotlib import image as mpimage
        from matplotlib.colors import Normalize

        from digitalearth.base.symbology import as_colormap

        array = np.ma.asarray(values).astype(float)
        data = (
            array.filled(np.nan)
            if np.ma.isMaskedArray(array)
            else np.asarray(array, dtype=float)
        )
        valid = np.isfinite(data)
        if not valid.any():
            raise ValueError("field() got a band with no finite values to colour")
        # The domain, the explicit-limit override and the constant-band widening are one rule, in base/spec.
        # `valid` is already computed above, so the finite subset is handed over rather than derived twice —
        # this is the tier with the explicit inline-pixel budget.
        lo, hi = Scale.from_finite(data[valid], vmin=vmin, vmax=vmax).as_limits()
        norm = Normalize(vmin=lo, vmax=hi)
        rgba = as_colormap(cmap)(norm(np.where(valid, data, lo)))
        rgba[~valid, 3] = 0.0  # NoData → transparent
        rgba8 = (rgba * 255).astype("uint8")

        buffer = io.BytesIO()
        mpimage.imsave(buffer, rgba8, format="png")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
