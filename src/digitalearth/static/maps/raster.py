"""RasterMixin — raster field rendering: field/contours/pcolormesh/block, composites, spaghetti.

Wires a pyramids ``Dataset`` (reprojected to the display CRS by the base) into cleopatra ``ArrayGlyph`` field
renders, plus the RGB/HSV composites and the ensemble spaghetti overlay.

**A field is read at the size the figure will draw it (ST-5).** Every cell of every raster used to be decoded,
warped whole, and then handed to matplotlib to be resampled down onto a few hundred thousand device pixels. A
band well past the canvas is now read through :class:`~digitalearth.base.sources.view.SourceView` instead — the
canvas and the budget as a :class:`~digitalearth.base.spec.target.RenderTarget` request, answered by pyramids'
windowed read, and warped one window at a time through ``Dataset.warped_view`` — which is the read the
interactive tier's viewport loop has driven all along. Below :data:`_DECIMATE_ABOVE` times the canvas on a side
the full read stands, so nothing already drawn moves.
"""

import logging
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, RgbBands
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap

from digitalearth.base.autostyle import auto_style
from digitalearth.base.display import auto_cmap
from digitalearth.base.levels import levels_every
from digitalearth.base.preprocess import add_cyclic_column
from digitalearth.base.sources import Source, get_stack, require_drawable
from digitalearth.base.spec import (
    DEFAULT_BAND,
    Bounds,
    LayerSpec,
    RenderTarget,
    Scale,
    Symbology,
)
from digitalearth.base.spec._serial import thawed_value
from digitalearth.base.stretch import (
    DEFAULT_COMPOSITE_BANDS,
    ChannelLimits,
    require_three_bands,
    stretch_to_unit,
)
from digitalearth.base.symbology import categorical_colors, resolve_categorical_cmap
from digitalearth.static.guides import source_field
from digitalearth.static.maps.base import OffLimbError
from digitalearth.static.render_compat import relocate_flat_style
from digitalearth.static.renderer import DrawnLayer
from digitalearth.static.scene import LayerRecord, drawing_style

#: The tier logs through the standard library, as `static/maps/base.py` does — not loguru, which is the web
#: tier's choice. One logger per tier, so a skip and a fallback from the same draw land in the same place.
logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: Colormap a field falls back to when ``cmap=None`` and ``auto_style`` resolves none — the tier's literal,
#: kept *behind* the lookup rather than in a signature, so one variable-driven default decides the colour of
#: every backend's render of the same data.
DEFAULT_FIELD_CMAP = "viridis"

#: The most distinct codes a raster may carry under ``scheme="categorical"``. A categorical key draws one swatch
#: per code, so a band past this is a continuous field in disguise, and is refused toward a graduated scheme
#: rather than drawn as a legend nobody can read.
MAX_RASTER_CATEGORIES = 24


def _asks_categorical(scheme: Any) -> bool:
    """Whether a ``scheme`` asks for one class per code, in any spelling of ``"categorical"``.

    Args:
        scheme: The caller's ``scheme`` — a name, a list of edges, or ``None``.

    Returns:
        ``True`` for ``"categorical"`` in any case; ``False`` for every other scheme, an edge list and ``None``.

    Examples:
        - The spelling is case-insensitive, and an edge list is never categorical:
            ```python
            >>> from digitalearth.static.maps.raster import _asks_categorical
            >>> _asks_categorical("Categorical"), _asks_categorical("quantiles"), _asks_categorical([0, 1])
            (True, False, False)

            ```
    """
    return isinstance(scheme, str) and scheme.lower() == "categorical"


def _raster_categories(values: Any) -> List[int]:
    """Return a raster band's distinct integer class codes, refusing a band that is not nominal.

    Args:
        values: The band's cell values; ``NaN`` and masked cells are nodata and fall in no class.

    Returns:
        The distinct codes, ascending, as ``int``.

    Raises:
        ValueError: when every cell is nodata, when a value is not a whole number (a magnitude, not a code),
            or when there are more than :data:`MAX_RASTER_CATEGORIES` distinct codes.

    Examples:
        - Codes come back sorted and distinct; a NaN cell is no code:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.maps.raster import _raster_categories
            >>> _raster_categories(np.array([[3.0, 1.0], [np.nan, 3.0]]))
            [1, 3]

            ```
        - A fractional value is a magnitude, so the band is refused:
            ```python
            >>> import numpy as np
            >>> from digitalearth.static.maps.raster import _raster_categories
            >>> _raster_categories(np.array([0.5, 1.0]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: scheme='categorical' needs integer class codes ... such as 0.5; ...

            ```
    """
    valid = np.ma.compressed(np.ma.masked_invalid(np.ma.asarray(values, dtype=float)))
    if valid.size == 0:
        raise ValueError(
            "scheme='categorical' found no valid cells to classify: every cell is nodata"
        )
    codes = np.unique(valid)
    fractional = codes[codes != np.round(codes)]
    if fractional.size:
        raise ValueError(
            "scheme='categorical' needs integer class codes (land cover, zone ids), but this band holds "
            f"non-integer values such as {fractional[0]:g}; classify a continuous magnitude with a graduated "
            "scheme ('quantiles', 'equal_interval', ...) instead"
        )
    if codes.size > MAX_RASTER_CATEGORIES:
        raise ValueError(
            f"scheme='categorical' draws one swatch per distinct code, and this band has {codes.size} "
            f"(at most {MAX_RASTER_CATEGORIES}); classify it with a graduated scheme instead"
        )
    return [int(code) for code in codes]


def _code_edges(codes: Sequence[int]) -> List[float]:
    """Return class edges that put each integer code in a class of its own.

    Half a step below the first code, midway between each pair of neighbours, and half a step above the last
    — so a gap in the codes (``1, 2, 5``) widens a class without letting it swallow a code.

    Args:
        codes: The distinct codes, ascending.

    Returns:
        ``len(codes) + 1`` ascending edges.

    Examples:
        - A gap between codes is split down the middle:
            ```python
            >>> from digitalearth.static.maps.raster import _code_edges
            >>> _code_edges([1, 2, 5])
            [0.5, 1.5, 3.5, 5.5]

            ```
    """
    middles = [(low + high) / 2 for low, high in zip(codes, codes[1:])]
    return [codes[0] - 0.5, *middles, codes[-1] + 0.5]


#: cleopatra's render kind -> the registered layer kind that render *is* (#303). The two vocabularies are not
#: the same word for the same thing: ``"contour"`` names a matplotlib call, ``"contours"`` names iso-value
#: lines traced from a band, and ``block`` draws a ``pcolormesh`` while being its own entry point. The figure
#: records the second, so a description written here is readable by a tier that has never heard of cleopatra.
FIELD_KINDS = {
    "imshow": "raster",
    "pcolormesh": "mesh",
    "contour": "contours",
    "contourf": "filled_contours",
}

#: Render kinds that draw contour lines/bands, i.e. the ones a resolved ``levels`` describes. ``auto_style``
#: contributes canonical contour levels for the operational fields it recognises (ECMWF Magics), and those
#: belong to a contour render — applying them to ``imshow``/``pcolormesh`` would quietly band a continuous
#: field the caller asked to see continuously.
_CONTOUR_KINDS = frozenset({"contour", "contourf"})

#: The public method each render kind is reached through, for a message that has to name something a caller
#: can call. The kinds are matplotlib's own spellings and two of them share one method, so a skip reported
#: under its kind would name ``imshow`` at a tier whose method is ``field``.
_FIELD_METHODS = {
    "imshow": "field",
    "pcolormesh": "pcolormesh",
    "contour": "contours",
    "contourf": "contours",
}

#: How much larger than the canvas a band must be before a field render decimates it instead of decoding it
#: whole — measured on a side, so this squared is the ratio in cells. Below it the decimated read and the full
#: read differ by less than one output pixel of detail, so the resample would cost more than the saving is
#: worth; above it most of the decoded cells are thrown away by matplotlib's own downsampling.
_DECIMATE_ABOVE = 2.0

#: Device pixels assumed when a scene's figure cannot be measured — matplotlib's own default figure at its
#: default resolution. A budget is needed either way; guessing this one only ever decides *whether* to
#: decimate, never where the data is.
_FALLBACK_CANVAS = (640, 480)


def _canvas_pixels(scene: Any) -> Tuple[int, int]:
    """Return the device pixels a scene's figure will be drawn at.

    Device pixels, not inches: a figure saved at 200 dpi draws four times the pixels of the same figure at
    100, and a budget measured in inches would ask for a ten-thousandth of what is actually drawn.

    Args:
        scene: The map being drawn on.

    Returns:
        ``(width, height)`` in pixels, at least one each; :data:`_FALLBACK_CANVAS` for a scene whose figure
        cannot be measured.
    """
    figure = getattr(scene, "fig", None)
    if figure is None:
        return _FALLBACK_CANVAS
    try:
        width, height = figure.get_size_inches()
        dpi = float(figure.dpi)
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
        return _FALLBACK_CANVAS
    return max(int(float(width) * dpi), 1), max(int(float(height) * dpi), 1)


def _worth_decimating(data: Any, target: RenderTarget) -> bool:
    """Whether a band is far enough past the canvas to be read at a lower resolution.

    Args:
        data: The raster the read would go through.
        target: Where the figure is going, which carries the canvas.

    Returns:
        ``True`` only when the raster reports an integer grid, exposes pyramids' windowed read
        (``read_part``), and holds more than :data:`_DECIMATE_ABOVE` times the canvas on a side. Each of the
        three is a reason *not* to decimate on its own: an unmeasurable grid is not one to decide a read on, a
        source with no window has nothing to read, and a raster near the canvas gains nothing.
    """
    rows, columns = getattr(data, "rows", None), getattr(data, "columns", None)
    if not isinstance(rows, int) or not isinstance(columns, int):
        return False
    if not hasattr(data, "read_part"):
        return False
    canvas = (target.width or 0) * (target.height or 0)
    return rows * columns > canvas * _DECIMATE_ABOVE**2


def _windowable(scene: Any, data: Any) -> Any:
    """Return the raster to read windows from, warped lazily when the display CRS asks for it.

    Args:
        scene: The map being drawn on, for its display CRS.
        data: The raster the caller gave.

    Returns:
        The raster itself when it is already in the display CRS; pyramids' lazily warped view of it
        (``Dataset.warped_view``) otherwise, so each window warps only itself. This mirrors the interactive
        tier's ``_windowable``, which faces the same choice.

    Raises:
        Exception: whatever pyramids raises for a warp it cannot set up — a raster wholly behind a clipped
            projection's limb reports ``RuntimeError: Too many points ... failed to transform``. It is raised
            rather than answered, because only ``GeoLayerBase._reproject`` knows that report means "nothing to
            draw here" and turns it into an :class:`~digitalearth.base.crs.OffLimbError`; see
            :func:`_windowed_field`, which hands the decision back to it.
    """
    if not scene._needs_reproject(data):
        return data
    warped_view = getattr(data, "warped_view", None)
    return None if warped_view is None else warped_view(scene.crs)


def _windowed_field(scene: Any, data: Any, band: int, target: RenderTarget) -> Any:
    """Read one band at the canvas, or answer ``None`` when this raster cannot be read that way.

    Args:
        scene: The map being drawn on, which carries the display CRS.
        data: The raster the layer draws.
        band: The 1-based band.
        target: Where the figure is going, which carries the canvas and the budget.

    Returns:
        A :class:`~digitalearth.base.sources.view.SourceView` of the band at the canvas, or ``None`` when the
        windowed read is not available for it — pyramids cannot warp it lazily, or the read itself failed.

        ``None`` rather than an exception, because the full read is not merely a fallback: it is the path that
        *owns* the off-limb decision. A raster wholly behind a clipped projection's limb makes the lazy warp
        report a failed transform, which only ``GeoLayerBase._prepare`` reads as "nothing to draw" — so a read
        that cannot be done here is handed back to it rather than reinterpreted here. Nothing is hidden:
        whatever is really wrong surfaces from that read, in the tier's own words.
    """
    try:
        windowable = _windowable(scene, data)
        if windowable is None:
            return None
        from digitalearth.base.sources.view import SourceView
        from digitalearth.base.spec import Selection, Viewport

        # No `ref=`: this view is never re-read. A still is drawn at one canvas, so the read that answers the
        # request is the only one — a moving region is `SourceView.reread`'s job, and the tier that has one
        # (interactive) registers its dataset once per layer and drives the re-reads per frame. Registering
        # here instead would mint a `DataRef` per *draw* and pin the warped view in the process-global object
        # table for good, so an animation of a hundred frames would hold a hundred of them alive.
        return SourceView.of(
            windowable,
            selection=Selection.of(band),
            # An unframed viewport names no region, so the request windows the source's own extent at the
            # canvas — which is what a still wants.
            request=target.view_request(Viewport(scene.crs)),
        )
    except Exception as error:  # noqa: BLE001 - the full read decides what it means; see Returns
        logger.debug(
            "field: reading band %s at the canvas failed (%s), falling back to the full read",
            band,
            error,
        )
        return None


def _band_identity(data: Any, band: int) -> Tuple[str, Optional[str]]:
    """Return a band's name and units, read from the raster's metadata rather than from its values.

    A windowed read answers with a bare array, which carries neither — pyramids' own docstring says *"pixel
    values only"* — so a decimated field would lose the variable :func:`~digitalearth.base.autostyle.auto_style`
    matches on and the units a colorbar labels itself with. Both are attributes of the raster, so they cost no
    read. The interactive tier makes the same trade for the same reason
    (``InteractiveMap._auto_cmap_for_band``), for the colormap alone.

    Args:
        data: The raster.
        band: The 1-based band.

    Returns:
        ``(variable, units)``; an empty variable and ``None`` units for a band that names neither, which is
        exactly what the full read answers for one.
    """
    names = list(getattr(data, "band_names", None) or [])
    units = list(getattr(data, "band_units", None) or [])
    variable = names[band - 1] if 1 <= band <= len(names) else ""
    unit = units[band - 1] if 1 <= band <= len(units) else None
    return str(variable or ""), str(unit) if unit else None


def _field_source(
    scene: Any, data: Any, band: int, *, exact: bool = False
) -> Tuple[Any, Any]:
    """Read one band for a field render, at the resolution the figure will draw it (ST-5).

    Args:
        scene: The map being drawn on, which carries the display CRS and the canvas.
        data: The raster the layer draws.
        band: The 1-based band.
        exact: Read every cell, however far past the canvas the raster is. A decimated read *combines*
            neighbouring cells, which is right for a magnitude and wrong for nominal class codes — averaging
            codes ``1`` and ``9`` invents a ``5`` that no cell holds — so a categorical field asks for this.

    Returns:
        ``(values, identity)`` — the source the render reads its cells and coordinates from, and the source the
        style lookup reads the band's identity from. They are the same object for a full read; for a decimated
        one the identity is restored from the raster's own metadata, because a windowed read answers with a
        bare array (see :func:`_band_identity`).

    Raises:
        OffLimbError: when the data lies outside what the display CRS can show, from
            ``GeoLayerBase._prepare`` on the full-read path.
    """
    target = RenderTarget("image", *_canvas_pixels(scene))
    # Measured on the raster the caller gave, before anything is warped: a raster near the canvas must not pay
    # for a lazy warp it will not use, and the full read below is the path that owns the off-limb decision.
    view = (
        _windowed_field(scene, data, band, target)
        if not exact and _worth_decimating(data, target)
        else None
    )
    if view is None:
        whole = scene._prepare(data, band)
        return whole, whole
    variable, units = _band_identity(data, band)
    identity = Source(
        view.z,
        view.x,
        view.y,
        view.crs,
        {"kind": "raster", "band": band, "variable": variable},
        units,
    )
    return view, identity


def _described_cmap(cmap: Any) -> Tuple[Any, Any]:
    """Split a colormap argument into the name a description records and the object held beside the layer.

    Args:
        cmap: What the caller asked for — a name, a ``Colormap``, or ``None`` to resolve one from the data.

    Returns:
        ``(recorded, held)``. A registered colormap is recorded by **name**: the name resolves to the same
        colours wherever the figure is read, and a `Colormap` object has no JSON form. One the caller built
        themselves resolves nowhere, so it is recorded as ``None`` — a figure read back elsewhere draws with
        the variable's own colormap — and held beside the layer, which is what colours this drawing.
    """
    if cmap is None or isinstance(cmap, str):
        return cmap, None
    name = getattr(cmap, "name", None)
    registered = (
        colormaps[name] if isinstance(name, str) and name in colormaps else None
    )
    if registered is not None and registered == cmap:
        return name, None
    return None, cmap


def _described_limits(limits: Optional[ChannelLimits]) -> Any:
    """Return stretch limits in the spelling a description carries: an unmeasurable bound as ``None``.

    :func:`~digitalearth.base.stretch.channel_limits` documents ``(nan, nan)`` for a channel it could not
    measure, and JSON has no spelling for ``nan`` — so a composite frozen on such a stack could not be
    written at all. ``None`` says the same thing in a form the description carries.

    Args:
        limits: The caller's per-channel ``(lo, hi)`` pairs, or ``None`` for a per-call scan.

    Returns:
        The pairs with every non-finite bound as ``None``, or ``None``.
    """
    if limits is None:
        return None
    return tuple(
        tuple(
            None if value is None or not isfinite(float(value)) else float(value)
            for value in pair
        )
        for pair in limits
    )


def _drawing_limits(limits: Any) -> Any:
    """Return recorded limits in the spelling ``stretch_to_unit`` reads, ``None`` back as ``nan``.

    Args:
        limits: The recorded pairs, or ``None``.

    Returns:
        The pairs as floats, a recorded ``None`` bound back as ``nan`` — which is what tells the stretch to
        fall back to this frame's own percentile for that channel.
    """
    if limits is None:
        return None
    return [
        tuple(float("nan") if value is None else float(value) for value in pair)
        for pair in limits
    ]


def draw_field(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Render the raster field a described layer asks for, through ``cleopatra.ArrayGlyph``.

    The one drawer behind ``imshow``/``pcolormesh``/``contour``/``contourf`` — four registered kinds drawn
    by one recipe, told apart by the ``via`` the builder recorded, which is cleopatra's own render kind.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the glyph's mappable.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows. The builder answers
            it by skipping the layer (or, under ``strict``, letting it out).
    """
    # Thawed whole rather than key by key: a spec freezes every list to a tuple so it still hashes, and
    # `levels` was the one key this drawer knew to thaw back. Nothing held is in `props` on this tier —
    # the scene holds the caller's own objects, and `drawing_style` lays them over the description — so
    # every container here is a described list, and thawing is the exact inverse of the freeze.
    props = thawed_value(dict(layer.symbology.props))
    kind = props["via"]
    opts = drawing_style(scene, layer)
    # Read at the size the figure will draw it: a band far past the canvas comes back decimated through
    # pyramids' windowed read, and `identity` carries the band's name and units a bare windowed array lacks
    # (ST-5). For everything smaller the two are one object and the read is the one this always did.
    src, identity = _field_source(
        scene, data, props["band"], exact=_asks_categorical(opts.get("scheme"))
    )
    z_values, x_values, y_values = src.z.values, src.x.values, src.y.values
    if opts.pop(
        "cyclic", False
    ):  # close the antimeridian seam for global fields (T5.2)
        z_values, x_values = add_cyclic_column(z_values, x_values)
    # Per-variable defaults (T6.2): the colormap, the canonical contour levels, and the units that label a
    # colorbar the caller did not label itself.
    style = auto_style(identity)
    # A colormap the caller built themselves has no name a description can carry, so it is held beside the
    # layer; a named one was recorded. Either way the caller's choice wins over the variable's own.
    # `.get` rather than `[]` on the style props the builder records: a figure another tier described (drawn
    # here through `to_backend`/`Map.from_figure`) carries the kind, source and portable encoding but not this
    # tier's resolved `cmap`/`default_cmap`/`levels`, so each falls back to the variable's own style (via
    # `auto_cmap`'s lookup) or the tier default. A same-tier figure always carries them, so this changes nothing
    # for it.
    requested = opts.pop("cmap", props.get("cmap"))
    opts["cmap"] = auto_cmap(
        src,
        requested,
        # `auto_cmap`'s fallback, reached only if the lookup answers no cmap. `auto_style` always answers one,
        # so this is belt-and-suspenders — it just keeps the fallback a real name rather than the `None` a
        # foreign figure's absent `default_cmap` would otherwise pass.
        props.get("default_cmap") or DEFAULT_FIELD_CMAP,
        lookup=lambda _: style,
    )
    levels = props.get("levels")
    if levels is None and kind in _CONTOUR_KINDS:
        # A caller's `interval=` wins over the variable's canonical levels, because it is the one the caller
        # wrote. Resolved here rather than in the builder: "one level every N" is a fact about the band's
        # range, and the builder records a description without reading the data.
        interval = props.get("interval")
        levels = (
            levels_every(z_values, interval)
            if interval is not None
            else style.get("levels")  # the variable's canonical contour levels
        )
    if levels is not None:
        opts["levels"] = levels
    # Geo-reference the data. cleopatra honours `extent` only for imshow (bbox order
    # [xmin, ymin, xmax, ymax]); contour/contourf/pcolormesh plot in array-index space unless given
    # `coords=(x, y)`. The two are mutually exclusive, so pick by kind — otherwise the field lands at the
    # wrong location/zoom.
    if kind == "imshow":
        placement = {"extent": scene._extent_of(x_values, y_values)}
    else:
        placement = {"coords": (x_values, y_values)}
    categorical = _categorical_scale(opts, z_values, requested)
    # cleopatra's glyph constructors reject the regrouped styling keys (levels/style/color_scale/…);
    # relocate them onto the plot() call, where `_render_glyph` folds them into their group objects.
    plot_style = relocate_flat_style(opts)
    glyph = ArrayGlyph(
        z_values,
        exclude_value=[float("nan")],
        ax=scene.ax,
        fig=scene.fig,
        **placement,
        **opts,
    )
    drawn: DrawnLayer = scene._render_glyph(
        glyph,
        kind=kind,
        # `.get` so a figure another tier described draws here: it carries no static `add_colorbar`/`zorder`.
        # `add_colorbar=False` is the ordinary path on this tier — the `Scene` (not the glyph) owns the
        # colorbar and adds it from the layer's colour key — so a cross-tier field is drawn the same way a
        # same-tier one is; it does not mean no colorbar appears.
        add_colorbar=props.get("add_colorbar", False),
        label=style.get("units"),  # the Scene's colorbar labels itself with it (T6.2)
        **plot_style,
    )
    zorder = props.get("zorder")
    if zorder is not None and drawn.artist is not None:
        # Recorded rather than set by the builder afterwards: a backdrop drawn again from its
        # description has to land behind the data again, and the builder is not there the second time.
        drawn.artist.set_zorder(zorder)
    if categorical is not None:
        drawn = replace(drawn, scale=categorical)
    # The band's own name, which only the drawer can answer: the builder records a band *number* and never
    # opens the source. It is the same `identity` `auto_style` matched the colormap and the units on.
    return drawn.colored_by(source_field(identity, props["band"]))


def _categorical_scale(
    opts: Dict[str, Any], values: Any, requested: Any
) -> Optional[Scale]:
    """Turn a ``scheme="categorical"`` raster into what cleopatra draws, and return its categorical scale.

    cleopatra's ``ArrayGlyph`` refuses a categorical scheme — to it a raster value is a magnitude — so a band
    of integer class codes (land cover, zone ids) is drawn as what that glyph already supports: explicit
    class edges putting each code in a class of its own, coloured with the same categorical palette the
    vector layers use, so one code reads as one colour on either. ``opts`` is rewritten in place.

    Args:
        opts: The drawing options; read for ``scheme``, and given the edges, the palette and no ``k``.
        values: The band's cell values.
        requested: The caller's ``cmap``, or ``None`` for the shared default categorical palette.

    Returns:
        The :class:`~digitalearth.base.spec.scale.Scale` of the codes and their colours, or ``None`` when the
        scheme is not categorical and ``opts`` is left alone.

    Raises:
        ValueError: from :func:`_raster_categories`, for a band that is not nominal.
    """
    if not _asks_categorical(opts.get("scheme")):
        return None
    codes = _raster_categories(values)
    categories, colors = categorical_colors(codes, resolve_categorical_cmap(requested))
    edges = _code_edges(codes)
    opts["scheme"] = edges
    opts.pop("k", None)
    opts["cmap"] = _class_lookup(colors, edges, codes)
    return Scale.categorical(categories, colors)


#: How many slots cleopatra's classifying ``BoundaryNorm`` spreads the classes over (``ncolors=256``).
_CLASS_SLOTS = 256


def _class_lookup(
    colors: Sequence[str], edges: Sequence[float], codes: Sequence[int]
) -> ListedColormap:
    """Return a colormap that paints each code's class in that class's colour.

    cleopatra classifies through ``BoundaryNorm(edges, ncolors=256)``, which spreads ``n`` classes over 256
    colour slots rather than onto slots ``0..n-1`` — so a plain ``n``-colour map paints only the first and last
    classes in their own colours, and every class between them in whatever colour its slot lands on (for four
    codes, all the last one). This builds a 256-slot map and puts each colour on the slot its code lands in,
    read off that same norm rather than re-derived, filling forward so every slot is a class colour.

    Args:
        colors: One colour per code, in code order.
        edges: The class edges from :func:`_code_edges`.
        codes: The codes, ascending.

    Returns:
        A ``ListedColormap`` of :data:`_CLASS_SLOTS` entries; its "bad" colour stays transparent, so nodata
        cells are left blank.

    Examples:
        - Three codes land on the first, middle and last slots, each in its own colour:
            ```python
            >>> from matplotlib.colors import to_hex
            >>> from digitalearth.static.maps.raster import _class_lookup
            >>> lookup = _class_lookup(["#ff0000", "#00ff00", "#0000ff"], [0.5, 1.5, 2.5, 3.5], [1, 2, 3])
            >>> lookup.N, [to_hex(lookup(slot)) for slot in (0, 127, 255)]
            (256, ['#ff0000', '#00ff00', '#0000ff'])

            ```
    """
    slots = np.asarray(
        BoundaryNorm(edges, ncolors=_CLASS_SLOTS)(np.asarray(codes, dtype=float))
    )
    lookup = [colors[0]] * _CLASS_SLOTS
    for slot, color in zip(slots, colors):
        lookup[int(slot) :] = [color] * (_CLASS_SLOTS - int(slot))
    return ListedColormap(lookup)


def _composite_bands(scene: Any, data: Any, props: Dict[str, Any]) -> tuple:
    """Return the reprojected dataset and its three stretched channels, band-last.

    The half ``rgb_composite`` and ``hsv_composite`` share: reproject once, read the three bands the
    description names, and stretch them onto ``0-1`` with whatever limits it froze.

    Args:
        scene: The map being drawn on.
        data: The multiband pyramids ``Dataset``.
        props: The layer's recorded properties, holding ``bands``, ``mask_nodata`` and ``limits``.

    Returns:
        ``(dataset, stretched)`` — the display-CRS dataset and a ``(rows, cols, 3)`` float array in ``0-1``.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    bands = list(props["bands"])
    ds = scene._reproject(data)
    # (rows, cols, n); nodata -> NaN unless mask_nodata=False
    stack = get_stack(ds, bands, mask=props["mask_nodata"])
    return ds, stretch_to_unit(stack, _drawing_limits(props["limits"]))


def _composite_glyph(scene: Any, ds: Any, band_first: Any, opts: Dict[str, Any]) -> Any:
    """Build the ``ArrayGlyph`` that draws a three-channel composite as one image.

    Args:
        scene: The map being drawn on.
        ds: The display-CRS dataset, whose cells place the image.
        band_first: The channels as ``(3, rows, cols)`` — cleopatra's ``RgbBands`` path transposes them
            back to band-last for ``imshow``.
        opts: The constructor keywords, already stripped of the regrouped styling keys.

    Returns:
        The glyph, ready to plot.
    """
    return ArrayGlyph(
        band_first,
        rgb_bands=RgbBands(list(range(band_first.shape[0]))),
        extent=scene._extent(ds),
        ax=scene.ax,
        fig=scene.fig,
        **opts,
    )


def draw_rgb_composite(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Compose the three bands a described true/false-colour layer asks for into one image.

    Args:
        scene: The map being drawn on.
        data: The multiband pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the image mappable.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    props = thawed_value(dict(layer.symbology.props))
    opts = drawing_style(scene, layer)
    ds, stretched = _composite_bands(scene, data, props)
    # cleopatra's RgbBands path is band-FIRST: it does array[indices].transpose(1, 2, 0), so feed
    # (n, rows, cols) and let it transpose back to (rows, cols, n) for imshow.
    band_first = np.moveaxis(stretched, -1, 0)
    plot_style = relocate_flat_style(opts)
    drawn: DrawnLayer = scene._render_glyph(
        _composite_glyph(scene, ds, band_first, opts), **plot_style
    )
    return drawn


def draw_hsv_composite(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Read the three bands a described HSV layer asks for as hue/saturation/value and draw the RGB.

    Args:
        scene: The map being drawn on.
        data: The multiband pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the image mappable.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    from matplotlib.colors import hsv_to_rgb

    props = thawed_value(dict(layer.symbology.props))
    opts = drawing_style(scene, layer)
    ds, stretched = _composite_bands(scene, data, props)
    rgb = hsv_to_rgb(stretched)  # (rows, cols, 3) RGB
    # band-FIRST for cleopatra's RgbBands path (see draw_rgb_composite); it transposes back to band-last.
    band_first = np.moveaxis(rgb, -1, 0)
    plot_style = relocate_flat_style(opts)
    drawn: DrawnLayer = scene._render_glyph(
        _composite_glyph(scene, ds, band_first, opts), **plot_style
    )
    return drawn


class RasterMixin(_MixinBase):
    """Raster field renders and composites for :class:`~digitalearth.static.map.Map`.

    Every builder here takes the caller's styling as ``**kwargs``/``**opts`` and passes it on to the
    cleopatra glyph untouched. Those keywords **go two ways**
    (see :attr:`~digitalearth.static.scene.LayerRecord.opts`). The plain ones — a string, a boolean, a
    finite number, ``None`` — are written into the layer's description as well, so the same figure drawn
    on another scene comes back with the ``vmin``, ``alpha`` or ``title`` the caller asked for. Everything
    else is held on the scene and nowhere else: a ``Normalize``, a per-pixel ``alpha`` array or a dash
    tuple is either unwritable or comes back as something matplotlib refuses, and a figure read elsewhere
    draws those with the engine's defaults. What the description records besides them is the call the
    caller made — the band, the colormap name, the levels, the draw order.
    """

    def _field(
        self,
        dataset: Any,
        *,
        kind: str,
        band: int = DEFAULT_BAND,
        cmap: Optional[str] = None,
        levels: Any = None,
        interval: Optional[float] = None,
        add_colorbar: bool = False,
        default_cmap: str = DEFAULT_FIELD_CMAP,
        draw_band: Optional[str] = None,
        zorder: Optional[float] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Render a raster ``dataset`` on the shared axes via ``cleopatra.ArrayGlyph`` (the canonical recipe).

        The colour treatment is resolved here rather than defaulted in a signature: with ``cmap=None`` the
        field's own metadata decides it, through :func:`~digitalearth.base.autostyle.auto_style`, and
        ``default_cmap`` is what answers when that lookup has no opinion. The same lookup supplies the
        contour ``levels`` and the colorbar label (``units``) a caller did not pass — a caller-supplied
        value always wins.

        Args:
            dataset: A pyramids ``Dataset``, or a path or URL to one (reprojected to :attr:`crs`
                first). It is recorded as the layer's source as given, so only a path-backed layer can
                be written down.
            kind: ArrayGlyph render kind (``auto``/``imshow``/``pcolormesh``/``contour``/``contourf``).
            band: 1-based band index.
            cmap: Colormap name, or ``None`` (default) to resolve one from the variable via ``auto_style``.
            levels: Discrete levels (int or sequence of edges), or ``None`` to take the canonical levels
                ``auto_style`` resolved for the variable — on a contour render only.
            interval: Spacing between contour levels — "one level every N" — resolved against the band's own
                range by the drawer, which is the only place the data is read. Give at most one of this or
                ``levels``; ``None`` (default) leaves the levels to ``levels``/``auto_style``.
            add_colorbar: When ``False`` (default) the Scene owns the colorbar, not the glyph.
            default_cmap: Colormap used when ``cmap`` is ``None`` *and* ``auto_style`` resolves none.
            draw_band: Where the layer is drawn in the figure's description, overriding the band its kind
                declares — ``"underlay"`` for a backdrop, which is what
                :meth:`~digitalearth.static.maps.decoration.DecorationMixin.stock_img` draws. ``None``
                (default) takes the kind's own band.
            zorder: The matplotlib draw order to give the render, or ``None`` (default) for whichever
                one the glyph leaves it with. The companion of `draw_band` on the engine's side, and
                recorded for the same reason: a backdrop drawn again from its description — a redraw, a
                rollback, a figure read back — has to come back *behind* the data rather than at the
                default 0.
            name: The caller's own name for the layer, used as its id and its label. ``None`` (default)
                generates one from the kind. A name already on the figure is suffixed ``-2``, ``-3``, …
                (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it hidden,
                so a switcher reading the figure agrees with what is on the axes (#327).
            **opts: The caller's own engine keywords, handed to ``ArrayGlyph`` as passed. The plain ones
                are described as well as held; the rest are held beside the layer alone (see the class
                docstring), so a figure drawn on a scene that does not hold them draws those with the
                engine's defaults. ``ArrayGlyph`` refuses a name it does not know rather than dropping it.

        Returns:
            The glyph's mappable (also registered as a Scene layer), or ``None`` when the data lies
            entirely outside what the display CRS shows — an off-limb frame draws nothing rather than
            raising, so a rotation past the far side of a globe still renders.

        Raises:
            TypeError: when `dataset` is something no field render can read — see
                :func:`~digitalearth.base.sources.require_drawable`, which names this call, the type it was
                handed and what it takes. It used to leak ``AttributeError: ... has no attribute 'to_crs'``
                from inside the reprojection instead (#343).
            ValueError: when `dataset` is a bare numpy array and this map is drawn on a globe frame; or from
                ``ArrayGlyph`` for a keyword in ``**opts`` it does not accept, naming the ones it does. The
                layer is dropped from the description again before it propagates, so a refused call leaves
                the scene exactly as it found it.
        """
        require_drawable(
            dataset,
            caller=f"Map.{_FIELD_METHODS[kind]}()",
            accepts=("raster", "array"),
        )
        if isinstance(dataset, np.ndarray) and self.globe:
            # A globe frame is a clipped projection with a limb, measured in the display CRS's own units —
            # metres, for every projection this tier draws a globe in. An array is placed at its indices, so
            # a 3 x 4 grid is a speck at the centre of a disc 12,000 km across, and a larger one is a square
            # swallowing the globe. Neither is a picture, and neither raises on its own.
            raise ValueError(
                f"Map.{_FIELD_METHODS[kind]}() cannot draw a bare numpy array on a globe frame: the frame is "
                "a projection limb in the display CRS's units and an array is placed at its own indices, so "
                "nothing relates the two. Wrap the array in a pyramids Dataset to place it, or build the Map "
                "with globe=False"
            )
        recorded_cmap, held_cmap = _described_cmap(cmap)
        if held_cmap is not None:
            opts["cmap"] = held_cmap
        record = LayerRecord(
            FIELD_KINDS[kind],
            source=dataset,
            name=name,
            band=draw_band,
            visible=visible,
            symbology=Symbology(
                props={
                    "via": kind,
                    "band": band,
                    "cmap": recorded_cmap,
                    "levels": levels,
                    "interval": interval,
                    "add_colorbar": add_colorbar,
                    "default_cmap": default_cmap,
                    "zorder": zorder,
                }
            ),
            # What the figure records is the call the caller made, not the call cleopatra receives:
            # `draw_field` resolves the colormap and the levels from it and hands cleopatra the result. The
            # caller's own engine keywords travel beside the description, exactly as passed.
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb(_FIELD_METHODS[kind])
            return None

    def field(
        self,
        dataset: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Render a raster band as a coloured field — a pixel grid (``ArrayGlyph`` ``kind="imshow"``).

        **A bare 2-D numpy array is drawn too** — the "just show me this grid" case, which was
        ``StaticGlyph.plot(arr)`` until that class was deleted and which nothing took over (#343). Such a
        layer is **not georeferenced**, and that is not a detail:

        * its coordinates are its **own indices** — the column index on x, the row index on y, cell centres
          on the integers and the image covering ``[-0.5, cols - 0.5] x [-0.5, rows - 0.5]``, with the first
          row drawn at the **top** as matplotlib draws every array;
        * **nothing is reprojected.** An array declares no CRS, so :attr:`crs` says nothing about where it
          is and no warp is attempted — :meth:`~digitalearth.static.maps.base.GeoLayerBase._reproject` hands
          it straight back and the layer's `Source` carries ``crs=None``;
        * **no georeferenced layer may share the figure.** A basemap, a coastline, a graticule and a
          reprojected raster are all placed in the display CRS, so mixing one in is refused by
          :meth:`~digitalearth.static.scene.Scene._require_one_placement` rather than drawn with the array
          as an invisible speck beside it;
        * **a globe frame is refused** for the same reason, at this call;
        * :meth:`~digitalearth.static.maps.projection.ProjectionMixin.set_bounds` still frames the figure,
          and its ``bounds`` are read in the display CRS — which for such a figure means *array indices*.
          ``set_bounds(None)`` fits the array, and ``set_bounds((0, 0, 4, 3))`` frames columns 0-4 and rows
          0-3. Geographic bounds would frame empty axes, and there is nothing to tell them apart by.

        **Nodata in a bare array is a masked array.** There is no sidecar to carry a ``no_data_value``, so
        ``np.ma.masked_equal(grid, -9999)`` (or ``NaN`` in the values) is how a cell says it has none, and
        those cells are left blank.

        Args:
            dataset: A pyramids ``Dataset``, a bare 2-D numpy array, or a path or URL to a raster
                (reprojected to :attr:`crs` first, except for the array — see above). Only a path-backed
                layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`_field`, which documents the named ones (``band``, ``cmap``,
                ``levels``, ``add_colorbar``, ``default_cmap``, ``draw_band``, ``zorder``). Anything
                left over is the caller's own engine styling, split between the layer's description and
                the scene (see the class docstring). ``scheme`` and ``k`` classify the band into discrete
                classes, as on the vector layers: a named scheme (``"quantiles"``, ``"equal_interval"``,
                ``"fisher_jenks"``, …) cut into ``k`` classes, or a list of explicit class edges. The
                classes are drawn as steps, published as :attr:`last_breaks` and on the layer's colour
                encoding, and keyed by :meth:`colorbar` (ticks on the edges) or :meth:`legend` (one swatch
                per class). Cut points that coincide — tied values under ``"quantiles"`` — merge, so fewer
                than ``k`` classes can be drawn; nodata cells fall in no class and stay blank. Explicit edges
                narrower than the data leave cells outside them, and those are painted the colormap's end
                colours — the first and last classes' colours when there are two or more classes — while
                the key lists only the given ranges, so span the data's range when that matters. Without
                ``scheme`` the field is a continuous ramp, and ``k`` alone is ignored — it counts the classes
                a named scheme cuts, so it does nothing without one. ``scheme="categorical"`` is for a band of
                integer class codes (land cover, zone ids): each distinct code becomes its own class,
                coloured from the same categorical palette the vector layers use (``cmap`` picks the
                palette), keyed by :meth:`legend` with one swatch per code and published as the layer's
                categories rather than class edges — so :attr:`last_breaks` stays ``None`` and a
                colorbar is refused in favour of the legend.

        Returns:
            The image mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            TypeError: when `dataset` is neither a raster, an array nor a reference to one — refused by name
                (see :meth:`_field`).
            ValueError: when `dataset` is an array and the map is drawn on a globe frame, or when it is an
                array of any rank but two; or from ``ArrayGlyph`` for a styling keyword it does not accept or
                an unknown ``scheme`` name; or, under ``scheme="categorical"``, when the band holds
                non-integer values, more than :data:`MAX_RASTER_CATEGORIES` distinct codes, or no valid cell.

        Examples:
            - Draw a bare grid, and read back where its image was placed:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.static import Map
                >>> with Map() as canvas:
                ...     image = canvas.field(np.arange(12.0).reshape(3, 4))
                ...     image.get_extent()
                [-0.5, 3.5, -0.5, 2.5]

                ```
            - Classify a raster into four equal-interval classes, and read back the edges it was cut at:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> with Map(crs=ds.epsg) as m:
                ...     _ = m.field(ds, scheme="equal_interval", k=4)
                ...     m.last_breaks
                [0.0, 22.0, 44.0, 66.0, 88.0]

                ```
            - Draw a band of class codes as categories, and read the key's labels — one per code:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.static import Map
                >>> landcover = np.array([[1.0, 1.0, 2.0], [3.0, 3.0, 2.0]])
                >>> with Map() as m:
                ...     _ = m.field(landcover, scheme="categorical")
                ...     _ = m.legend()
                ...     [label.get_text() for label in m.ax.get_legend().get_texts()]
                ['1', '2', '3']

                ```
            - A band of fractional values is a magnitude, so the categorical scheme refuses it:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from digitalearth.static import Map
                >>> with Map() as m:
                ...     m.field(np.array([[0.5, 1.0]]), scheme="categorical")  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: scheme='categorical' needs integer class codes ...

                ```
        """
        return self._field(dataset, kind="imshow", name=name, visible=visible, **kwargs)

    def contours(
        self,
        dataset: Any,
        *,
        levels: Any = None,
        interval: Optional[float] = None,
        filled: bool = False,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Trace iso-value lines through a raster band, or fill between them.

        One method for both renders, which is what the Tier-2 contract declares and what the web and
        interactive tiers offer. This tier had two — ``contour`` and ``contourf`` — so ``filled=`` meant
        nothing here and a script moving between tiers had to know which of the two to write. Neither
        spelling survives: ``filled=`` is the argument that picks the render.

        Args:
            dataset: A pyramids ``Dataset``, or a path or URL to one (reprojected to :attr:`crs`
                first). Only a path-backed layer can be written down.
            levels: The iso-values to trace — an int (a count) or a sequence of values. ``None``
                (default) takes the variable's canonical levels from ``auto_style``, and leaves the choice
                to matplotlib when it carries none. Give at most one of this or ``interval``.
            interval: Spacing between levels — one level every N, in the band's own units. Resolved
                against the band's range when it is drawn, so only the multiples *inside* the data are
                traced. Give at most one of this or ``levels``.
            filled: ``False`` (default) draws the levels as lines; ``True`` fills the bands between them.
                The two are different matplotlib renders, so this is the argument that picks one.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`_field`, which documents the named ones. Anything left over is
                the caller's own engine styling, split between the layer's description and the scene
                (see the class docstring).

        Returns:
            The contour mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when both ``levels`` and ``interval`` are given — two ways of asking for one
                thing, so neither can be silently preferred; when ``interval`` is not a positive finite
                spacing or crosses no level inside the band; or from ``ArrayGlyph`` for a styling keyword
                it does not accept.
        """
        if levels is not None and interval is not None:
            raise ValueError(
                "contours() takes at most one of interval= or levels=; "
                f"got interval={interval!r} and levels={levels!r}"
            )
        return self._field(
            dataset,
            kind="contourf" if filled else "contour",
            levels=levels,
            interval=interval,
            name=name,
            visible=visible,
            **kwargs,
        )

    def pcolormesh(
        self,
        dataset: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Render a raster as a quadrilateral mesh (``ArrayGlyph`` ``kind="pcolormesh"``).

        Args:
            dataset: A pyramids ``Dataset``, or a path or URL to one (reprojected to :attr:`crs`
                first). Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`_field`, which documents the named ones. Anything left over is
                the caller's own engine styling, split between the layer's description and the scene
                (see the class docstring).

        Returns:
            The ``QuadMesh`` mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: from ``ArrayGlyph`` for a styling keyword it does not accept.
        """
        return self._field(
            dataset, kind="pcolormesh", name=name, visible=visible, **kwargs
        )

    def block(
        self,
        dataset: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Render a raster as a filled cell mesh — currently an alias of :meth:`pcolormesh`.

        ``block`` is meant for *discrete* per-cell rectangles aligned to cell **edges**. cleopatra's
        ``pcolormesh`` path accepts cell-**centre** coordinates only (it rejects edge arrays of length
        ``n + 1``), so true edge-aligned blocks need an upstream cleopatra option; until that lands ``block``
        draws the same filled mesh as :meth:`pcolormesh`. It is kept as a distinct, stable entry point so
        callers and examples can switch to true blocks transparently once cleopatra supports them.

        Args:
            dataset: A pyramids ``Dataset``, or a path or URL to one (reprojected to :attr:`crs`
                first). Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`_field`, which documents the named ones. Anything left over is
                the caller's own engine styling, split between the layer's description and the scene
                (see the class docstring).

        Returns:
            The ``QuadMesh`` mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: from ``ArrayGlyph`` for a styling keyword it does not accept.
        """
        return self._field(
            dataset, kind="pcolormesh", name=name, visible=visible, **kwargs
        )

    @staticmethod
    def _extent_of(x: Any, y: Any) -> List[float]:
        """Return the bbox-order extent of the cells 1-D x/y centres describe, as cleopatra takes it.

        The coordinates name the middle of each cell, and the extent is the rectangle those cells *cover* —
        half the outermost spacing further out on every side (:meth:`Bounds.cell_edges`). Taken from the
        smallest and largest centre instead, the image would be drawn one cell narrower and one shorter than
        the data, with every pixel at ``(n - 1) / n`` of its size (#301).

        Args:
            x: X coordinates, already in the display CRS.
            y: Y coordinates, already in the display CRS.

        Returns:
            ``[xmin, ymin, xmax, ymax]``. The ordering comes from
            :meth:`~digitalearth.base.spec.bounds.Bounds.as_bbox` rather than a list written out here, so it
            cannot drift from the ordering the axes path uses. The `Bounds` is built only to name that
            ordering and is unwrapped immediately, so it carries no CRS: the caller's coordinates are already
            in the display one, and a CRS that changes nothing is a parameter every caller must think about
            for no benefit.
        """
        return Bounds.cell_edges(x, y, crs=None).as_bbox()

    def _extent(self, ds: Any) -> List[float]:
        """Return the bbox-order extent of the cells a dataset's coordinates describe, as cleopatra takes it.

        Args:
            ds: The display-CRS source whose ``x``/``y`` cell centres bound the image.

        Returns:
            ``[xmin, ymin, xmax, ymax]`` — the rectangle the cells cover, as :meth:`_extent_of` places it.
        """
        return self._extent_of(ds.x, ds.y)

    def rgb_composite(
        self,
        dataset: Any,
        bands: Sequence[int] = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[ChannelLimits] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Render three raster bands as a true/false-colour RGB image (``ArrayGlyph`` RGB path).

        Args:
            dataset: A multiband pyramids ``Dataset``, or a path or URL to one (reprojected to the
                display CRS first). Only a path-backed layer can be written down.
            bands: Three 1-based band indices mapped to R, G, B. Defaults to ``(1, 2, 3)``.
            mask_nodata: When ``True`` (default) each band's nodata cells are excluded from the 2-98
                percentile stretch (and render transparent). Pass ``False`` for the raw values (the
                pre-mask behaviour, where the nodata sentinel participates in the stretch).
            limits: Optional frozen ``(lo, hi)`` stretch bounds, one pair per channel — skips the per-call
                percentile scan. :meth:`~digitalearth.static.maps.animation.AnimationMixin.animate` fills
                these from the whole stack so a composite animation holds one stretch across its frames.
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when ``bands`` does not hold exactly three indices, or ``limits`` is given without
                one ``(lo, hi)`` pair per channel.

        Examples:
            - Composite three bands of a multiband raster into one RGB image:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> rgb = Dataset.from_array(arr=np.stack([base, base, base]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=rgb.epsg)
                >>> _ = m.rgb_composite(rgb)
                >>> len(m.ax.images)
                1

                ```
            - Frozen ``limits`` replace the per-call stretch: a white point far above the data renders it
                black, which is what holds a sequence of frames on one scale:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> rgb = Dataset.from_array(arr=np.stack([base, base, base]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=rgb.epsg)
                >>> _ = m.rgb_composite(rgb, limits=[(0.0, 1e6)] * 3)
                >>> round(float(np.nanmax(m.ax.images[-1].get_array())), 3)
                0.0

                ```

        See Also:
            hsv_composite: The same three bands read as hue/saturation/value instead.
            digitalearth.base.stretch.channel_limits: Derives the ``limits`` this accepts.
        """
        require_three_bands("rgb_composite", bands)
        return self._composite(
            "rgb_composite",
            dataset,
            bands,
            mask_nodata,
            limits,
            opts,
            name=name,
            visible=visible,
        )

    def _composite(
        self,
        via: str,
        dataset: Any,
        bands: Sequence[int],
        mask_nodata: bool,
        limits: Optional[ChannelLimits],
        opts: dict,
        name: Optional[str] = None,
        visible: bool = True,
    ) -> Any:
        """Record a three-band composite and draw it, answering an off-limb warp with a skipped layer.

        The half :meth:`rgb_composite` and :meth:`hsv_composite` share once the band count is validated —
        they differ only in the recipe they record, which is what picks the drawer.

        Args:
            via: The recipe the figure records, ``"rgb_composite"`` or ``"hsv_composite"``.
            dataset: The multiband pyramids ``Dataset``, or a path or URL to one. It is recorded as the
                layer's source as given, so only a path-backed layer can be written down.
            bands: The three 1-based band indices.
            mask_nodata: Whether each band's nodata cells are excluded from the stretch.
            limits: Frozen per-channel ``(lo, hi)`` stretch bounds, or ``None`` for a per-call scan.
            opts: The caller's styling keywords, as they were written — described where
                :func:`~digitalearth.base.spec._serial.travels_in_a_figure` accepts them, held beside the layer
                otherwise.
            name: The caller's own name for the layer, used as its id and its label; ``None`` generates
                one from the kind (#321).
            visible: Whether the layer is drawn — ``False`` builds it hidden and describes it hidden
                (#327).

        Returns:
            The image mappable, or ``None`` when the data lies outside what the display CRS shows.

        Raises:
            TypeError: when `dataset` is not a multiband raster or a reference to one. A bare numpy array is
                refused here where :meth:`field` takes one: a composite reads three *bands*, and an array is
                one grid with no band dimension for :func:`~digitalearth.base.sources.get_stack` to index.
        """
        require_drawable(dataset, caller=f"Map.{via}()", accepts=("raster",))
        record = LayerRecord(
            "rgb",
            source=dataset,
            name=name,
            visible=visible,
            symbology=Symbology(
                props={
                    "via": via,
                    "bands": tuple(bands),
                    "mask_nodata": mask_nodata,
                    "limits": _described_limits(limits),
                }
            ),
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb(via)
            return None

    def hsv_composite(
        self,
        dataset: Any,
        bands: Sequence[int] = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[ChannelLimits] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Render three raster bands as an HSV composite (hue/sat/value → RGB → image).

        Args:
            dataset: A multiband pyramids ``Dataset``, or a path or URL to one (reprojected to the
                display CRS first). Only a path-backed layer can be written down.
            bands: Three 1-based band indices mapped to H, S, V. Defaults to ``(1, 2, 3)``.
            mask_nodata: When ``True`` (default) each band's nodata cells are excluded from the 2-98
                percentile stretch (and render transparent). Pass ``False`` for the raw pre-mask behaviour.
            limits: Optional frozen ``(lo, hi)`` stretch bounds, one pair per channel (H, S, V) — skips the
                per-call percentile scan, so an animation can hold one stretch across its frames.
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when ``bands`` does not hold exactly three indices, or ``limits`` is given without
                one ``(lo, hi)`` pair per channel.

        Examples:
            - Read three bands as hue/saturation/value and render the resulting RGB image:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> hsv = Dataset.from_array(arr=np.stack([base, base * 0.5, base * 0.25]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=hsv.epsg)
                >>> _ = m.hsv_composite(hsv)
                >>> len(m.ax.images)
                1

                ```
            - The rendered image is band-last RGB, one value per channel per cell:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> hsv = Dataset.from_array(arr=np.stack([base, base * 0.5, base * 0.25]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=hsv.epsg)
                >>> _ = m.hsv_composite(hsv)
                >>> m.ax.images[-1].get_array().shape[-1]
                3

                ```

        See Also:
            rgb_composite: The same three bands mapped straight to red/green/blue.
            digitalearth.base.stretch.channel_limits: Derives the ``limits`` this accepts.
        """
        require_three_bands("hsv_composite", bands)
        return self._composite(
            "hsv_composite",
            dataset,
            bands,
            mask_nodata,
            limits,
            opts,
            name=name,
            visible=visible,
        )

    def spaghetti(
        self,
        collection: Any,
        band: int = DEFAULT_BAND,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> List[Any]:
        """Overlay each member of a ``DatasetCollection`` as line contours on one axes (ensemble spaghetti).

        Args:
            collection: A pyramids ``DatasetCollection`` whose members share a grid. Unlike the
                single-raster builders this takes no path: a collection is a set of rasters rather
                than one file.
            band: 1-based band read from each member.
            name: The caller's own name for the layers, used as their ids and labels; ``None``
                (default) generates one per member. One call draws **one layer per member**, so a
                single name is shared and the second and later take ``-2``, ``-3``, … (#321).
            visible: Whether the members are drawn. ``False`` builds them hidden **and** describes
                them hidden (#327).
            **opts: Styling kwargs forwarded to the per-member contour call.

        Returns:
            The list of per-member contour mappables (each also registered as a Scene layer). Members
            lying outside what the display CRS shows draw nothing and are absent from the list, so it
            stays one entry per *drawn* member and never contains ``None``. That means the list cannot be
            zipped against ``collection.datasets`` when some members are hidden — pair by drawing members
            individually if a per-member legend needs to know which is which.
        """
        drawn = [
            self._field(
                member,
                kind="contour",
                band=band,
                add_colorbar=False,
                name=name,
                visible=visible,
                **opts,
            )
            for member in collection.datasets
        ]
        return [artist for artist in drawn if artist is not None]
