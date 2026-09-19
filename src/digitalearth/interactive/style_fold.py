"""Folding a declared `Symbology` into HoloViews options — checked against the engine, not against a guess.

The interactive tier spells style in HoloViews' own keywords, one call at a time: nineteen builders write
`cmap`, `clim`, `alpha`, `colorbar`, `clabel`, `color`, `color_levels`, `size`, `fill_alpha`, `edge_color` and
`edge_cmap` into `.opts()`, and a caller's `**opts` goes through untouched. That works until a keyword is
misspelt, or belongs to another element, or to another backend: `.opts()` checks names, but only once the
element exists, only in HoloViews' spelling, and — without a backend named — across every backend that happens
to be registered, so `interpolation="nearest"` on a Bokeh image is accepted and then ignored.

This module is the fold that replaces that, and the second engine the Wave 1 style vocabulary is tested against:

* :data:`INTERACTIVE_STYLE_SCHEMA` declares the flat keywords the builders accept, each tied to the channel it
  drives, so a misspelling is answered with a suggestion before anything is built.
* :func:`fold_symbology` turns a `Symbology` into `{"style": ..., "plot": ...}` for one element type, and says
  which channels that element cannot express and why — `size` on a polygon, `height` on anything here.
* :func:`allowed_options` reads the engine's own table (`hv.Store.options`), so every key the fold produces is
  checked against what the element actually takes on the backend being drawn.

HoloViews is imported inside the functions that need it: a dispatcher reads this module's schema to answer what
the tier accepts, and that must not cost a backend import.
"""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Tuple

from digitalearth.base.spec import Encoding, Scale, StyleKey, StyleSchema, Symbology

__all__ = [
    "CHANNEL_OPTIONS",
    "INTERACTIVE_STYLE_SCHEMA",
    "ChannelOption",
    "allowed_options",
    "fold_symbology",
    "route_flat_style",
]

#: The flat keywords the interactive builders accept, each declared once. A keyword that drives a visual
#: channel names it, so `route_flat_style` can lift it into a `Symbology`; the rest are engine options carried
#: as they are. The names are HoloViews' own, because that is what a caller of this tier writes today and the
#: fold is not a rename.
INTERACTIVE_STYLE_SCHEMA: StyleSchema = StyleSchema.of(
    # -- the visual channels of a layer
    StyleKey("alpha", "Layer opacity, 0 transparent to 1 opaque.", channel="opacity"),
    StyleKey("size", "Marker size, in points.", channel="size"),
    StyleKey("line_width", "Line width, in points.", channel="width"),
    StyleKey("angle", "Rotation applied to the glyph, in degrees.", channel="rotation"),
    StyleKey(
        "color",
        "A constant colour, or the column the layer is coloured by.",
        channel="color",
    ),
    # -- how a field-driven colour is drawn
    StyleKey("cmap", "Colormap a field-driven colour is ramped through."),
    StyleKey("clim", "Colour limits, as (low, high)."),
    StyleKey("color_levels", "Class edges a classified colour is drawn with."),
    StyleKey("cnorm", "How colour values are normalised — linear, log, eq_hist."),
    StyleKey("colorbar", "Whether a colour key is drawn beside the layer."),
    StyleKey("clabel", "The colour key's label."),
    # -- the rest of what the builders write
    StyleKey("fill_alpha", "Opacity of a filled shape's interior."),
    StyleKey("line_color", "Colour of a shape's outline."),
    StyleKey("edge_color", "Colour of a mesh's edges."),
    StyleKey("edge_cmap", "Colormap a mesh's edge values are ramped through."),
    StyleKey("muted", "Whether the layer is drawn muted."),
    StyleKey("visible", "Whether the layer is drawn at all."),
    StyleKey("tools", "Bokeh tools attached to the frame — hover, tap, box_select."),
)


@dataclass(frozen=True)
class ChannelOption:
    """How one visual channel is spelled for HoloViews, and which elements can express it.

    Attributes:
        channel: The channel, a key of :data:`~digitalearth.base.spec.CHANNELS`.
        option: The HoloViews keyword it becomes.
        group: The option group the keyword belongs to — `"style"` or `"plot"`. One layer's style spans both:
            `cmap` and `alpha` are style, `clim` and `colorbar` are plot, and HoloViews validates them
            separately.
        elements: The element types that take it, as HoloViews names them.
        reason: Why the other elements cannot, shown to a caller who asks for it there.

    Examples:
        - Marker size is a point's, and a polygon's answer says so:
            ```python
            >>> from digitalearth.interactive.style_fold import CHANNEL_OPTIONS
            >>> size = next(entry for entry in CHANNEL_OPTIONS if entry.channel == "size")
            >>> size.option, "Points" in size.elements, "Polygons" in size.elements
            ('size', True, False)

            ```
    """

    channel: str
    option: str
    group: str
    elements: FrozenSet[str]
    reason: str


#: Which HoloViews option each channel becomes, and where it is taken. Measured against the registered Bokeh
#: options of every element the builders emit — a test walks this table against `hv.Store` so it cannot drift
#: from the engine. Two channels are in it nowhere: `height` has no HoloViews option at all, and `text` is a
#: `Labels` value dimension rather than a style keyword, so both are reported as unsupported with that reason.
CHANNEL_OPTIONS: Tuple[ChannelOption, ...] = (
    ChannelOption(
        "color",
        "color",
        "style",
        frozenset(
            {
                "Points",
                "Path",
                "Polygons",
                "Contours",
                "QuadMesh",
                "HexTiles",
                "VectorField",
                "Rectangles",
            }
        ),
        "an image is coloured by its own values through cmap, not by a colour keyword",
    ),
    ChannelOption(
        "opacity",
        "alpha",
        "style",
        frozenset(
            {
                "Image",
                "RGB",
                "QuadMesh",
                "Points",
                "Path",
                "Polygons",
                "Contours",
                "HexTiles",
                "VectorField",
                "Rectangles",
                "WMTS",
            }
        ),
        "this element has no whole-layer alpha; its parts are faded separately",
    ),
    ChannelOption(
        "size",
        "size",
        "style",
        frozenset({"Points"}),
        "only a marker has a size; a shape is as big as its geometry",
    ),
    ChannelOption(
        "width",
        "line_width",
        "style",
        frozenset(
            {
                "Path",
                "Polygons",
                "Contours",
                "Points",
                "QuadMesh",
                "HexTiles",
                "VectorField",
                "Rectangles",
            }
        ),
        "an image has no outline to set a width on",
    ),
    ChannelOption(
        "rotation",
        "angle",
        "style",
        frozenset({"Points", "Labels"}),
        "only a glyph can be turned; a path is drawn where its coordinates are",
    ),
    ChannelOption(
        "tooltip",
        "tools",
        "plot",
        frozenset(
            {
                "Points",
                "Path",
                "Polygons",
                "Contours",
                "QuadMesh",
                "HexTiles",
                "Rectangles",
                "Image",
                "RGB",
                "VectorField",
                "Labels",
                "WMTS",
                "TriMesh",
            }
        ),
        "this element has no hover inspector in Bokeh",
    ),
)

#: The channels no element of this tier can express, and why — the finding the Core contract (#299) starts from.
UNEXPRESSIBLE: Mapping[str, str] = {
    "height": "HoloViews has no z-height option: an extrusion is a 3-D or web layer",
    "text": "text is a Labels value dimension, not a style keyword, so it is drawn rather than styled",
}


#: The option groups HoloViews sorts an element's keywords into, in the order a key is looked for. `style` is
#: first because that is where a colour or a width lives; `norm` holds `framewise`/`axiswise`, and `output`
#: the renderer's own settings.
OPTION_GROUPS: Tuple[str, ...] = ("style", "plot", "norm", "output")


def route_flat_style(flat: Mapping[str, Any]) -> Tuple[Symbology, Dict[str, Any]]:
    """Split a builder's flat keywords into a `Symbology` and the engine options left over.

    Args:
        flat: The keywords a builder was called with.

    Returns:
        A `(symbology, leftover)` pair, where `leftover` is **always empty** — the shape is the one every
        tier's `route_flat_style` returns, and this tier has nothing to leave over. A keyword that drives a
        channel becomes an `Encoding`; every other keyword becomes a property, whether the schema declares it
        or not, because only :func:`fold_symbology` knows the element and can tell a raw HoloViews option
        from a misspelling. Handing an undeclared keyword back here would refuse the first and the second
        alike.

    Examples:
        - A channel keyword and an engine option travel in one dict and are separated:
            ```python
            >>> from digitalearth.interactive.style_fold import route_flat_style
            >>> symbology, leftover = route_flat_style({"alpha": 0.4, "cmap": "viridis"})
            >>> symbology.encoding("opacity").resolve(), dict(symbology.props)
            (0.4, {'cmap': 'viridis'})

            ```
        - A keyword the schema does not declare is carried as a property, for the fold to check against the
          element it will be applied to:
            ```python
            >>> from digitalearth.interactive.style_fold import fold_symbology, route_flat_style
            >>> symbology, _ = route_flat_style({"alhpa": 0.4})
            >>> fold_symbology(symbology, "Points")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: 'alhpa' is not an option Points takes on the bokeh backend; did you mean ['alpha']...

            ```
    """
    symbology, leftover = INTERACTIVE_STYLE_SCHEMA.route(flat)
    # A keyword the schema does not declare is not dropped and not guessed at here: it is carried as a
    # property, so `fold_symbology` can check it against the element it will be applied to — where a raw
    # HoloViews option is legitimate and a misspelling is not, and only the element can tell them apart.
    if not leftover:
        return symbology, {}
    return Symbology(
        encodings=dict(symbology.encodings),
        props={**dict(symbology.props), **leftover},
    ), {}


def allowed_options(
    element: str, backend: str = "bokeh"
) -> Mapping[str, FrozenSet[str]]:
    """Return the options HoloViews accepts for one element type, by group.

    Args:
        element: The element type, as HoloViews names it — `"Image"`, `"Points"`, `"Polygons"`.
        backend: The backend to ask. The backend is loaded before it is asked, so the answer is what *that*
            backend takes rather than whatever happens to be registered.

    Returns:
        `{group: keywords}` for the groups HoloViews validates — `style`, `plot`, `norm`, `output`.

    Raises:
        KeyError: for an element type the backend does not know, naming it.

    Examples:
        - A Bokeh image takes a colormap as style and colour limits as a plot option:
            ```python
            >>> from digitalearth.interactive.style_fold import allowed_options
            >>> options = allowed_options("Image")
            >>> "cmap" in options["style"], "clim" in options["plot"]
            (True, True)

            ```
        - A marker size belongs to points, and nothing else here:
            ```python
            >>> from digitalearth.interactive.style_fold import allowed_options
            >>> "size" in allowed_options("Points")["style"], "size" in allowed_options("Polygons")["style"]
            (True, False)

            ```
    """
    import holoviews as hv

    # Registers the backend's option tree without making it the current one. `hv.extension(backend)` is
    # `Store.set_current_backend`, so asking what options an element takes — a read — left the whole process
    # rendering through whichever backend was asked about last (review M15). `base.py` already uses this
    # idiom, with the comment "register mpl opts before applying them".
    hv.renderer(backend)
    store = hv.Store.options(backend=backend)
    try:
        options = store[element]
    except KeyError:
        raise KeyError(
            f"holoviews' {backend} backend has no element type {element!r}"
        ) from None
    return {
        group: frozenset(options.groups[group].allowed_keywords.values)
        for group in options.groups
    }


def fold_symbology(
    symbology: Symbology, element: str, *, backend: str = "bokeh"
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Fold a declared symbology into the options one HoloViews element takes.

    This is the single point where a declared style becomes this renderer's own form — which is what keeps the
    vocabulary in `base/spec/` free of HoloViews, and HoloViews' spellings out of every builder.

    Args:
        symbology: The style to fold.
        element: The element type it will be applied to, as HoloViews names it.
        backend: The backend the options are checked against.

    Returns:
        A `(grouped, unsupported)` pair. `grouped` is `{"style": {...}, "plot": {...}}`, ready for two
        `.opts()` calls; `unsupported` maps each channel this element cannot express to the reason, so a
        builder can report rather than silently drop it.

    Raises:
        ValueError: for a channel driven by a field where the element needs the column resolved, and for any
            key — a channel's or a raw prop — the element does not accept on this backend, naming the closest
            option it does accept. The check runs before any element is built.

    Examples:
        - Opacity and size fold onto a point's own keywords:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.interactive.style_fold import fold_symbology
            >>> grouped, unsupported = fold_symbology(Symbology.of(opacity=0.5, size=6), "Points")
            >>> grouped["style"], unsupported
            ({'alpha': 0.5, 'size': 6}, {})

            ```
        - A polygon has no marker size, and says so rather than dropping it:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.interactive.style_fold import fold_symbology
            >>> _, unsupported = fold_symbology(Symbology.of(opacity=0.5, size=6), "Polygons")
            >>> unsupported["size"]
            'only a marker has a size; a shape is as big as its geometry'

            ```
        - A colour driven by a field becomes the column plus its limits:
            ```python
            >>> from digitalearth.base.spec import Encoding, Scale, Symbology
            >>> from digitalearth.interactive.style_fold import fold_symbology
            >>> by_pop = Encoding.by_field("color", "pop", scale=Scale.from_limits(0.0, 100.0))
            >>> grouped, _ = fold_symbology(Symbology.of(color=by_pop), "Points")
            >>> grouped["style"]["color"], grouped["plot"]["clim"]
            ('pop', (0.0, 100.0))

            ```
    """
    allowed = allowed_options(element, backend)
    # Every group `_group_of` can name, so a standard option that belongs to one of the other two — HoloViews
    # puts `framewise` and `axiswise` under `norm` — lands in it instead of raising a bare `KeyError('norm')`
    # where the did-you-mean belonged (review M16). Empty groups are dropped before this returns.
    grouped: Dict[str, Dict[str, Any]] = {name: {} for name in OPTION_GROUPS}
    unsupported: Dict[str, str] = {}
    for channel, encoding in dict(symbology.encodings).items():
        if channel in UNEXPRESSIBLE:
            unsupported[channel] = UNEXPRESSIBLE[channel]
            continue
        entry = next(
            (option for option in CHANNEL_OPTIONS if option.channel == channel), None
        )
        if (
            entry is None
        ):  # pragma: no cover - a channel added to CHANNELS before this table knows it
            unsupported[channel] = f"this tier folds no {channel!r} channel"
            continue
        if element not in entry.elements:
            unsupported[channel] = entry.reason
            continue
        _fold_channel(entry, encoding, grouped, unsupported)
    for key, value in dict(symbology.props).items():
        grouped[_group_of(key, allowed, element)][key] = value
    for group, options in grouped.items():
        for key in options:
            if key not in allowed.get(group, frozenset()):
                raise ValueError(_refusal(key, element, backend))
    # `style` and `plot` are always present — a caller reads them without asking whether they are there —
    # and the other two only when something landed in them.
    return {
        group: options
        for group, options in grouped.items()
        if options or group in ("style", "plot")
    }, unsupported


def _fold_channel(
    entry: ChannelOption,
    encoding: Encoding,
    grouped: Dict[str, Dict[str, Any]],
    unexpressible: Dict[str, str],
) -> None:
    """Write one channel's option into the grouped result.

    Args:
        entry: The channel's row in :data:`CHANNEL_OPTIONS`.
        encoding: What drives the channel.
        grouped: The result, mutated in place.
        unexpressible: What this tier cannot express, mutated in place — reported rather than raised, as
            the rest of the fold reports it.

    Raises:
        ValueError: for a field-driven channel that is not colour — HoloViews colours by a column, but sizes
            and widths take a number, so the builder must resolve those itself.
    """
    if entry.channel == "tooltip":
        grouped["plot"].setdefault("tools", []).append("hover")
        # The hover tool reads the element's own dimensions, so *which* columns it shows is not something
        # `.opts()` can be told. The fields were dropped in silence; the module's contract is that anything
        # it cannot express is reported, so they are (review L17).
        shown = encoding.value if encoding.is_constant else None
        if shown:
            unexpressible["tooltip"] = (
                f"HoloViews' hover tool shows the element's own dimensions, so it cannot be limited to "
                f"{list(shown)}; build the element with those columns as its vdims"
            )
        return
    if encoding.is_constant:
        grouped[entry.group][entry.option] = encoding.value
        return
    if entry.channel != "color":
        raise ValueError(
            f"the {entry.channel!r} channel is driven by the field {encoding.field!r}, which HoloViews takes "
            f"as a number rather than a column; resolve it and pass the values"
        )
    # HoloViews colours by a dimension name, and the scale becomes the limits and the class edges around it.
    grouped["style"]["color"] = encoding.field
    scale = encoding.scale
    if isinstance(scale, Scale):
        grouped["plot"]["clim"] = (scale.vmin, scale.vmax)
        if scale.breaks:
            grouped["plot"]["color_levels"] = list(scale.breaks)


def _group_of(key: str, allowed: Mapping[str, FrozenSet[str]], element: str) -> str:
    """Return the option group a raw key belongs to.

    Args:
        key: The option a caller wrote.
        allowed: What the element accepts, by group.
        element: The element type, for the message.

    Returns:
        Whichever of :data:`OPTION_GROUPS` declares the key, style first, since that is where a colour or a
        width lives. A key nothing declares is returned as `"style"` so the check below refuses it there,
        with a suggestion.
    """
    for group in OPTION_GROUPS:
        if key in allowed.get(group, frozenset()):
            return group
    return "style"


def _refusal(key: str, element: str, backend: str) -> str:
    """Return the message for an option this element does not take.

    The accepted options are not passed in: the suggestion comes from HoloViews' own fuzzy match over the
    element's keywords, which is a richer answer than the flat set the caller checked against.

    Args:
        key: The option that was refused.
        element: The element type it was folded for.
        backend: The backend it was checked against.

    Returns:
        A sentence naming the key, the element and the backend, with the closest accepted options when
        HoloViews can suggest any — the same did-you-mean `.opts()` gives, but before anything is built.
    """
    import holoviews as hv

    options = hv.Store.options(backend=backend)[element]
    close = []
    for group in options.groups:
        close.extend(options.groups[group].allowed_keywords.fuzzy_match(key))
    declared = INTERACTIVE_STYLE_SCHEMA.suggest(key)
    if declared is not None:
        close.append(declared)
    hint = f"; did you mean {sorted(set(close))}?" if close else ""
    return f"{key!r} is not an option {element} takes on the {backend} backend{hint}"
