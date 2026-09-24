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
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Tuple

from digitalearth.base.spec import Encoding, Scale, StyleKey, StyleSchema, Symbology
from digitalearth.base.spec.style import asked_constants, portable_constants

__all__ = [
    "ASKED_BUCKET",
    "CHANNEL_KEYWORDS",
    "CHANNEL_OPTIONS",
    "DERIVED_BUCKET",
    "INTERACTIVE_STYLE_SCHEMA",
    "STYLE_BUCKETS",
    "TIER_BUCKET",
    "UNASKED_STYLE",
    "ChannelOption",
    "builder_of",
    "allowed_options",
    "fold_symbology",
    "portable_encodings",
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
    StyleKey(
        "clim",
        "Colour limits, as (low, high). A list is taken too, and is the spelling a figure carries: a "
        "tuple is what the travel rule refuses, so the documented pair was dropped from every saved "
        "figure while the list survived (review R2-M13).",
    ),
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


#: Which declared visual channel each of this tier's own style keywords drives.
#:
#: The inverse view of :data:`INTERACTIVE_STYLE_SCHEMA`, derived from it rather than written out again, so a
#: keyword cannot be tied to one channel for routing and another for recording. Only the keywords that drive
#: a channel appear; `cmap`, `clim` and the rest are engine options with no portable reading.
CHANNEL_KEYWORDS: Mapping[str, str] = MappingProxyType(
    {
        key.name: key.channel
        for key in INTERACTIVE_STYLE_SCHEMA.keys.values()
        if key.channel is not None
    }
)

#: The bucket a builder files its **resolved parameters** under — defaults included, so a value here is not
#: by itself evidence that anyone asked for it.
#:
#: ``points(size=7.0)`` and ``points(features)`` both land here as ``{"size": …}``, which is why the
#: subtraction below exists at all: the bucket records what the parameter *resolved to* and not whether it
#: was passed. :data:`UNASKED_STYLE` is the only attribution available until a builder records the ask
#: itself (#334).
DERIVED_BUCKET: str = "common"

#: The bucket a builder files the caller's **own** ``**opts`` under. Every key here was written by the
#: caller, which is what makes it liftable without a second question.
#:
#: The invariant is load bearing — :func:`portable_encodings` lifts this bucket **unfiltered** — and it was
#: broken once: ``spaghetti`` passed its per-member colour cycle through ``**opts``, so three unstyled
#: ensemble members published three different colours as three caller intents (review R2-H3). Style a
#: builder works out for itself belongs in :data:`TIER_BUCKET`, which nothing lifts;
#: ``tests/interactive/test_interactive_unasked_style.py`` holds every builder to that.
ASKED_BUCKET: str = "opts"

#: The bucket a builder files style **it** invented under — never a parameter, never an ask, never lifted.
#:
#: The third case the two buckets above could not hold. ``spaghetti`` colours each ensemble member from a
#: cycle so the strands are distinguishable: that colour is neither a keyword the caller wrote
#: (:data:`ASKED_BUCKET`) nor a parameter of the builder that happened to resolve to it
#: (:data:`DERIVED_BUCKET`) — no caller can pass it, and it exists only because the tier drew several
#: layers from one call. It reaches the element like any other option and is absent from
#: :data:`STYLE_BUCKETS`, so :func:`portable_encodings` never publishes it and no table has to enumerate
#: the ten colours to keep it out (review R2-H3).
TIER_BUCKET: str = "tier"

#: The props a builder files its resolved HoloViews style under, in the precedence the drawer applies.
#:
#: A builder writes the style it derived into ``common`` and the caller's own keywords into ``opts``, and
#: every drawer merges the second over the first — so an explicit keyword outranks a derived one, and the
#: lift below must read them in the same order. A few builders (``image``, ``rgb``) write their flat style
#: at the top level of ``props`` instead, which is why the mapping itself is read first.
STYLE_BUCKETS: Tuple[str, ...] = (DERIVED_BUCKET, ASKED_BUCKET)

#: What each of this tier's builders derives when the caller styled nothing, by the builder that wrote it.
#:
#: **Why this table has to exist.** :data:`ASKED_BUCKET` is the caller's own, but a builder also writes its
#: resolved parameters into :data:`DERIVED_BUCKET` and into the flat top level — ``points`` writes ``size``
#: there whether or not ``size=`` was passed, and ``image`` writes ``alpha`` the same way. Publishing those
#: made an unstyled ``points(features)`` claim ``{'size': 6.0}`` as the caller's own style, against the web
#: tier's ``{'color': '#3388ff', 'opacity': 0.9, 'size': 5.0}`` for the same call — two tiers publishing two
#: sets of defaults into the one field `to_backend()` (order 33) will read as intent (review R-H2).
#:
#: **Why it is keyed by builder.** The first version was keyed by style keyword, with every builder's
#: defaults pooled under one tuple, so each builder was also charged with its siblings' (review R2-H2). This
#: tier can do better than the web tier can, because every builder already records **which builder it was**:
#: ``props["via"]``, with ``hv_type`` separating the five kinds that share the geometry funnel. So a row is
#: looked up rather than guessed at, and adding a builder with its own default cannot silently swallow
#: another's explicit ask.
#:
#: **What is listed.** Measured by drawing every builder on this tier that records a layer with no style
#: keywords at all — **twenty-seven** of them, the twenty-three data builders and the four decorations, as
#: ``tests/interactive/test_interactive_unasked_style.py`` enumerates them — and reading back
#: ``symbology.props``: only two derive anything that drives a declared channel — ``points`` records
#: ``common['size'] == 6.0`` and ``image`` records a flat ``alpha == 1.0``. ``rgb`` is **not** one of them
#: (it records ``via``/``bands``/``limits``/``opts`` and no ``alpha``), ``polygons`` derives
#: ``fill_alpha``, which drives no channel, and a classified ``color`` is a value dimension and is refused
#: below. ``tests/interactive/test_interactive_unasked_style.py`` re-measures every builder, so a row that
#: stops matching fails rather than rots (review R2-M5).
#:
#: **What it costs, deliberately.** A caller who asks for exactly the default — ``points(size=6.0)`` —
#: publishes no ``size``, so a figure carried elsewhere is drawn at that tier's own default. The conformance
#: suite already calls that acceptable: "Which default a tier picks is not what a round trip has to agree
#: about". Publishing it anyway is not, because it repaints a layer nobody styled. A caller who writes the
#: same value through ``**opts`` is unaffected — that bucket is never filtered.
UNASKED_STYLE: Mapping[Tuple[str, str], Mapping[str, Any]] = MappingProxyType(
    {
        ("geometry", "Points"): MappingProxyType({"size": 6.0}),
        ("image", ""): MappingProxyType({"alpha": 1.0}),
    }
)


def builder_of(props: Mapping[str, Any]) -> Tuple[str, str]:
    """Return the key :data:`UNASKED_STYLE` is looked up by, for one recorded style.

    Args:
        props: A layer's recorded `Symbology.props`.

    Returns:
        ``(via, hv_type)``. Five builders share ``via == "geometry"`` — the vector funnel records the
        HoloViews element type beside it, which is what tells a point layer from a path — and every other
        builder's ``via`` is its own, so the second half is an empty string there. A description carrying
        no ``via`` at all answers ``("", "")``, which matches no row and is therefore charged nothing.

    Note:
        **Deliberately not the layer's `kind`,** which is what the web tier's table is keyed by. A kind is
        coarser than a builder here: ``image``, ``large_image`` and ``rasterize`` all record ``"raster"``,
        ``hexbin`` and ``kde`` both record ``"heatmap"``, and ``points`` and ``datashade`` both record
        ``"points"``. Keying on it would pool those builders' defaults — the exact defect keying by kind
        *removes* on the web tier, where one kind has one builder (review R2-H2). So the two tiers look
        asymmetric and are answering the same question: which builder wrote this style.

    Examples:
        - The geometry funnel is told apart by its element type; every other builder by its own name:
            ```python
            >>> from digitalearth.interactive.style_fold import builder_of
            >>> builder_of({"via": "geometry", "hv_type": "Points"})
            ('geometry', 'Points')
            >>> builder_of({"via": "image"})
            ('image', '')

            ```
    """
    return (str(props.get("via") or ""), str(props.get("hv_type") or ""))


def portable_encodings(symbology: Symbology) -> Dict[str, Encoding]:
    """Return the declared channels an interactive layer's recorded options say the **caller** asked for.

    Additive by construction: the resolved HoloViews options stay exactly where every drawer reads them, and
    this writes a second, portable reading of the same style beside them — the one another tier, and
    `to_backend()`, can act on (#328). Nothing here renames a keyword: `alpha` is still `alpha` in the
    props a drawer applies, and the channel it drives is recorded as `opacity` because that is what the
    vocabulary calls it.

    The three buckets are read differently, because they mean different things. :data:`ASKED_BUCKET` holds
    the caller's own ``**opts`` and is lifted as it stands; :data:`DERIVED_BUCKET` and the flat top level
    hold the builder's *resolved parameters*, defaults included, so a value :data:`UNASKED_STYLE` names for
    **that builder** is passed over rather than published; and :data:`TIER_BUCKET` holds style the tier
    invented for itself and is not read at all. The figure records no list of the keywords a caller named,
    so the comparison against the builder's own defaults is the only attribution available — and publishing
    an unattributed value as an `Encoding` is what made an unstyled layer carry one tier's defaults onto
    another (review R-H2).

    Args:
        symbology: The layer's recorded style, as the builder wrote it.

    Returns:
        Channel name -> a constant :class:`~digitalearth.base.spec.encoding.Encoding`, for the style values
        that drive a declared channel and are constants a channel can portably hold.

        A `color` that names one of the layer's value dimensions is **not** one of them. HoloViews reads a
        colour naming a dimension as "colour by that column", so recording it as a constant would say the
        layer is painted the literal string `"pop"`; the classification behind it is published portably as
        ``last_breaks`` instead.

    Examples:
        - A marker size and an opacity read back as the channels a caller asked for:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.interactive.style_fold import portable_encodings
            >>> props = {"via": "geometry", "hv_type": "Points", "common": {"size": 7.0}}
            >>> lifted = portable_encodings(Symbology(props=dict(props, opts={"alpha": 0.5})))
            >>> sorted(lifted), lifted["size"].resolve(), lifted["opacity"].resolve()
            (['opacity', 'size'], 7.0, 0.5)

            ```
        - The same options as `points()` writes them when nobody styled the layer publish nothing at all:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.interactive.style_fold import portable_encodings
            >>> unstyled = {"via": "geometry", "hv_type": "Points", "common": {"size": 6.0}, "opts": {}}
            >>> sorted(portable_encodings(Symbology(props=unstyled)))
            []

            ```
        - 6.0 is `points`' default and nobody else's, so the same number from another builder is an ask:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.interactive.style_fold import portable_encodings
            >>> other = {"via": "trimesh", "common": {"size": 6.0}}
            >>> portable_encodings(Symbology(props=other))["size"].resolve()
            6.0

            ```
        - The caller's own bucket is never filtered, so asking for the default there still publishes it:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.interactive.style_fold import portable_encodings
            >>> asked = portable_encodings(Symbology(props={"via": "image", "opts": {"alpha": 1.0}}))
            >>> sorted(asked), asked["opacity"].resolve()
            (['opacity'], 1.0)

            ```
        - A colour that names a value dimension is a column, not a colour, so it is left unclaimed:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.interactive.style_fold import portable_encodings
            >>> classified = {"vdims": ("pop",), "common": {"color": "pop", "colorbar": True}}
            >>> sorted(portable_encodings(Symbology(props=classified)))
            []

            ```
    """
    props = dict(symbology.props)
    unasked = {
        key: (default,)
        for key, default in UNASKED_STYLE.get(builder_of(props), {}).items()
    }
    lifted = asked_constants(props, CHANNEL_KEYWORDS, unasked)
    for bucket in STYLE_BUCKETS:
        held = props.get(bucket)
        if not isinstance(held, dict):
            continue
        if bucket == ASKED_BUCKET:
            lifted.update(portable_constants(held, CHANNEL_KEYWORDS))
        else:
            lifted.update(asked_constants(held, CHANNEL_KEYWORDS, unasked))
    coloured = lifted.get("color")
    dimensions = {str(name) for name in (props.get("vdims") or ())}
    if coloured is not None and coloured.value in dimensions:
        del lifted["color"]
    return lifted


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


def _named_fields(value: Any) -> list:
    """Return the column names a tooltip encoding asks for, however it was written.

    The `tooltip` channel is a plain text channel, so nothing in the vocabulary stops a caller writing one
    column as a bare string or passing something that is not a sequence at all. Listing the value directly
    spelled a string out letter by letter and raised `TypeError` on a number (review M9).

    Args:
        value: What the encoding carries — a sequence of names, one name, or anything else.

    Returns:
        The names as a list: a string becomes one name, a sequence keeps its order, and `None` or an empty
        value becomes `[]`. Anything else is named as the single thing it is, so the message can still say
        what could not be honoured.
    """
    if value is None or value == ():
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(name) for name in value]
    except TypeError:
        return [str(value)]


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
        shown = _named_fields(encoding.value if encoding.is_constant else None)
        if shown:
            unexpressible["tooltip.fields"] = (
                f"HoloViews' hover tool shows the element's own dimensions, so it cannot be limited to "
                f"{shown}; build the element with those columns as its vdims"
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
