"""The one place a style is folded into the flat form cleopatra's glyphs take.

cleopatra >=0.30 removed the loose ``plot``/``animate`` styling keywords (``levels``, ``scheme``, ``style``,
``color_scale``, ``points``, …) in favour of small typed *group objects* (``Contour``, ``Classify``,
``DataStyle``, ``ColorScaling``, ``CellValues``, ``PointOverlay``). Digital-Earth keeps accepting the flat
kwargs as its public surface and folds them here, so callers stay insulated from the upstream regrouping.

Those flat kwargs used to be **undeclared** — 27 keys, in no signature anywhere, so a caller could not ask
what a builder accepts and a typo was a silently ignored keyword rather than an error.
:data:`STATIC_STYLE_SCHEMA` declares them: each with what it controls, and with the visual channel it drives
where it drives one. That is what :func:`route_flat_style` routes against and what
:func:`digitalearth.base.spec.style.StyleSchema.suggest` reads to name the key a typo was meant to be.

Entry points. The first two are **Wave-1 scaffolding**: they are the declared-style half of the seam
Wave 3's `LayerSpec` plugs into, and no builder calls them yet — the builders still pass flat kwargs
through the last three. Said plainly because a docstring describing a pipeline that does not exist is
the kind of claim that survives unchecked into the wave that was supposed to build it.

- :func:`route_flat_style` — flat public kwargs -> a declared
  :class:`~digitalearth.base.spec.style.Symbology`, plus whatever was not style at all.
- :func:`fold_symbology` — the inverse: a declared symbology -> the flat kwargs, plus any channel this
  renderer's flat surface cannot express (returned as a value, not raised).
- :func:`relocate_flat_style` — pop the flat members out of a *constructor* kwargs dict (the cleopatra glyph
  constructors now reject them) so they can be forwarded to ``plot`` instead. Also folds the ``size``
  channel onto the constructor spelling a point glyph wants.
- :func:`group_render_kwargs` — the glyph-agnostic fold into group objects (idempotent; an already-built
  group object, or an unrelated kwarg, passes through untouched). Every group but one is folded by mapping
  keys onto fields here, because no other group has a builder upstream; the colour group is built by
  :func:`digitalearth.static.style_fold.fold_color_scaling`, which calls cleopatra's own
  ``ColorScaling.from_options``.
- :func:`prepare_plot_kwargs` — fold the flat members a *given glyph* supports into their group objects, and
  hand back any ``alpha`` the glyph cannot take for the caller to apply to the artist. Applied centrally in
  :meth:`~digitalearth.static.scene.Scene._render_glyph`.

What *is* true today: no builder holds a style-mapping decision of its own — every cleopatra spelling,
including the ``size`` -> ``point_size`` fold, is decided here.
"""

import inspect
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Set, Tuple

from cleopatra.glyphs.gridded.array_glyph import PointOverlay
from cleopatra.styling.params import CellValues, Classify, Contour, DataStyle

from digitalearth.base.spec import StyleKey, StyleSchema, Symbology
from digitalearth.static.style_fold import (
    COLOR_GROUP_MEMBERS,
    COLOR_GROUP_PARAM,
    COLOR_SCALE_ALIASES,
    coerce_color_scale,
    fold_color_scaling,
)

__all__ = [
    "COLOR_SCALE_ALIASES",
    "STATIC_STYLE_SCHEMA",
    "coerce_color_scale",
    "fold_color_scaling",
    "fold_symbology",
    "group_render_kwargs",
    "plot_takes",
    "prepare_plot_kwargs",
    "relocate_flat_style",
    "route_flat_style",
]

#: Marker/label styling that folds into a ``PointOverlay`` wrapping the ``points`` array.
_POINT_FIELDS = {
    "point_color": "color",
    "point_size": "size",
    "point_label_color": "label_color",
    "point_label_size": "label_size",
    "pid_color": "label_color",  # oldest spelling
    "pid_size": "label_size",
}

#: ``plot`` group parameter -> (group class, {flat kwarg: group field}). Ordered for stable construction.
_GROUP_SPECS = (
    (
        "contour",
        Contour,
        {"levels": "levels", "labels": "labels", "label_kw": "label_kw"},
    ),
    (
        "cells",
        CellValues,
        {
            "display_cell_value": "show",
            "num_size": "size",
            "background_color_threshold": "background_threshold",
        },
    ),
    (
        "data_style",
        DataStyle,
        {
            "style": "style",
            "hillshade": "hillshade",
            "bands": "bands",
            "alpha": "alpha",
            "alpha_range": "alpha_range",
        },
    ),
    (
        "classify",
        Classify,
        {
            "scheme": "scheme",
            "k": "k",
            "category_legend_kwargs": "category_legend_kwargs",
        },
    ),
)

#: The typed group parameters themselves (``color``/``contour``/``data_style``/``classify``/``cells``/``points``)
#: — the constructors reject these too, so a caller passing a group object directly must also route to ``plot``.
#: ``color`` is not in :data:`_GROUP_SPECS`: its members fold through
#: :func:`~digitalearth.static.style_fold.fold_color_scaling` rather than through a field map here.
_GROUP_PARAMS = frozenset(
    {param for param, _, _ in _GROUP_SPECS} | {"points", COLOR_GROUP_PARAM}
)

#: Every styling key cleopatra's glyph constructors now reject — the flat members, plus the typed group params.
FLAT_STYLE_KEYS = frozenset(
    {"points", *_POINT_FIELDS}
    | {flat for _, _, field_map in _GROUP_SPECS for flat in field_map}
    | set(COLOR_GROUP_MEMBERS)
    | _GROUP_PARAMS
)


#: The public spelling of a marker's visual size. It is not one of the regrouped flat keys — a point glyph
#: takes its size on the *constructor* — but it is part of the same declared vocabulary, and the ``size``
#: channel it drives is what makes it the demonstration that a channel costs a fold rule, not a parameter on
#: every builder.
MARKER_SIZE_KEY = "size"

#: Every style keyword the static tier accepts, declared: what it controls, and the visual channel it drives
#: where it drives one. That is the **27** flat members cleopatra's constructors reject, the 6 typed group
#: parameters they fold into, and :data:`MARKER_SIZE_KEY` — 34 keywords that were in no signature anywhere.
#:
#: Most of them are static properties — a threshold, a preset name, a nested kwargs dict — and say so by
#: declaring no channel. Only two vary a visual variable of the layer as a whole today, and both route
#: through :class:`~digitalearth.base.spec.encoding.Encoding` rather than through a keyword of their own.
#: ``tests/static/test_style_schema.py`` pins this table against :data:`FLAT_STYLE_KEYS`, so a key added
#: upstream cannot quietly go undeclared again.
STATIC_STYLE_SCHEMA: StyleSchema = StyleSchema.of(
    # -- visual channels of the layer itself
    StyleKey("alpha", "Layer opacity, 0 transparent to 1 opaque.", channel="opacity"),
    StyleKey(MARKER_SIZE_KEY, "Marker size, in points.", channel="size"),
    # -- the point overlay drawn over a field, and its labels
    StyleKey("points", "Point coordinates to overlay on the field."),
    StyleKey("point_color", "Marker colour of the point overlay."),
    StyleKey("point_size", "Marker size of the point overlay, in points."),
    StyleKey("point_label_color", "Colour of the point overlay's labels."),
    StyleKey("point_label_size", "Size of the point overlay's labels, in points."),
    StyleKey(
        "pid_color",
        "Older spelling of point_label_color; ignored when both are passed.",
    ),
    StyleKey(
        "pid_size", "Older spelling of point_label_size; ignored when both are passed."
    ),
    # -- contours
    StyleKey("levels", "Contour levels: a count, or the explicit values to draw."),
    StyleKey("labels", "Whether to label the contour lines."),
    StyleKey("label_kw", "Extra keyword arguments for the contour labels."),
    # -- printed cell values
    StyleKey("display_cell_value", "Print each cell's value inside the cell."),
    StyleKey("num_size", "Font size of the printed cell values, in points."),
    StyleKey(
        "background_color_threshold",
        "Value above which a printed cell value switches to the contrasting ink.",
    ),
    # -- the data-style preset and its per-call overrides
    StyleKey("style", "Named data-style preset the surface is drawn with."),
    StyleKey("hillshade", "Shade the surface with relief."),
    StyleKey("bands", "Discrete band count partitioning the preset's value range."),
    StyleKey("alpha_range", "(low, high) opacity the preset's scale is drawn across."),
    # -- classification
    StyleKey("scheme", "Classification scheme cutting the values into classes."),
    StyleKey("k", "Number of classes the scheme cuts."),
    StyleKey(
        "category_legend_kwargs", "Extra keyword arguments for a category legend."
    ),
    # -- colour scaling
    StyleKey(
        "color_scale",
        "Colour scaling: linear, power, lognorm, sym_log, boundary, midpoint or equalize.",
    ),
    StyleKey("gamma", "Exponent of the power colour scale."),
    StyleKey("line_threshold", "Half-width of a symmetric-log scale's linear region."),
    StyleKey("line_scale", "Width scaling of a symmetric-log scale's linear region."),
    StyleKey("bounds", "Explicit class bounds for a boundary colour scale."),
    StyleKey("midpoint", "The value a diverging colour scale centres on."),
    StyleKey(
        "samples",
        "Number of samples the equalize colour scale reads the value distribution at.",
    ),
    # -- already-built group objects, passed straight through
    # NOT the `color` visual channel, despite the name. This is cleopatra's `plot(color=...)` parameter,
    # which takes a ColorScaling group object describing how values are *scaled* onto a ramp. The static
    # tier has no flat keyword for a constant layer colour at all — colour comes from `cmap` plus the data —
    # so this key deliberately declares no channel, and fold_symbology says so when asked for one.
    StyleKey(
        "color",
        "A built ColorScaling group object — the colour *scaling*, not a colour.",
    ),
    StyleKey("contour", "A built Contour group object."),
    StyleKey("data_style", "A built DataStyle group object."),
    StyleKey("classify", "A built Classify group object."),
    StyleKey("cells", "A built CellValues group object."),
)

#: Channel -> the flat keyword that carries it, derived from the declaration so the two cannot disagree.
_CHANNEL_KEYS: Dict[str, str] = {
    key.channel: key.name for key in STATIC_STYLE_SCHEMA.keys.values() if key.channel
}


def route_flat_style(flat: Mapping[str, Any]) -> Tuple[Symbology, Dict[str, Any]]:
    """Route flat public kwargs to the channels and properties that own them.

    The half :func:`prepare_plot_kwargs` cannot do: it can only *reject* a keyword it has no home for, whereas
    routing gives every declared keyword its home and hands back the rest.

    Args:
        flat: A caller's keyword dict. Not mutated.

    Returns:
        A ``(symbology, rest)`` pair — the declared style as a value, and the keywords that are not style at
        all (``cmap``, ``add_colorbar``, the glyph's own constructor options). `rest` is returned rather than
        refused because this schema covers the styling surface, not everything a builder accepts; use
        :meth:`~digitalearth.base.spec.style.StyleSchema.suggest` on a key that reaches nothing.

    Examples:
        - A channel keyword, a static property and a constructor option travel in one dict and are separated:
            ```python
            >>> from digitalearth.static.render_compat import route_flat_style
            >>> sym, rest = route_flat_style({"alpha": 0.4, "hillshade": True, "cmap": "viridis"})
            >>> sym.encoding("opacity").resolve()
            0.4
            >>> dict(sym.props), rest
            ({'hillshade': True}, {'cmap': 'viridis'})

            ```
        - A misspelt key reaches nothing, and the schema names what it was probably meant to be:
            ```python
            >>> from digitalearth.static.render_compat import STATIC_STYLE_SCHEMA, route_flat_style
            >>> _, rest = route_flat_style({"hillshde": True})
            >>> list(rest)
            ['hillshde']
            >>> STATIC_STYLE_SCHEMA.suggest("hillshde")
            'hillshade'

            ```
    """
    return STATIC_STYLE_SCHEMA.route(flat)


def fold_symbology(symbology: Symbology) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Fold a declared symbology into the flat kwargs the cleopatra glyphs take.

    The inverse of :func:`route_flat_style`, and the single point where a declared style becomes this
    renderer's own form — which is what keeps the vocabulary in ``base/spec/`` free of cleopatra, and
    cleopatra's spellings out of every builder.

    Args:
        symbology: The style to fold.

    Returns:
        A ``(flat, unsupported)`` pair. `flat` carries every constant channel under the keyword that
        expresses it, plus the static properties unchanged. `unsupported` maps a channel this flat surface
        cannot express to the reason — a channel with no keyword here, or one driven by a data field, which
        the flat kwargs have no expression for (the glyph takes resolved ``values`` instead). It is
        **returned, not raised**, so a caller can degrade rather than fail.

    Examples:
        - Constants fold onto their keywords; the properties pass through:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.static.render_compat import fold_symbology
            >>> flat, unsupported = fold_symbology(Symbology.of(opacity=0.4).with_props(hillshade=True))
            >>> flat == {"alpha": 0.4, "hillshade": True}, unsupported
            (True, {})

            ```
        - A channel this surface has no keyword for is reported rather than dropped:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> from digitalearth.static.render_compat import fold_symbology
            >>> _, unsupported = fold_symbology(Symbology.of(height=30.0))
            >>> sorted(unsupported)
            ['height']

            ```
    """
    flat: Dict[str, Any] = dict(symbology.props)
    unsupported: Dict[str, str] = {}
    for channel, encoding in symbology.encodings.items():
        keyword = _CHANNEL_KEYS.get(channel)
        if keyword is None:
            unsupported[channel] = (
                f"the static tier has no flat styling keyword for the {channel!r} channel"
                + (
                    " — a matplotlib layer takes its colours from cmap plus the data values, and the flat"
                    " `color=` key is cleopatra's ColorScaling group, which is the scaling rather than a"
                    " colour"
                    if channel == "color"
                    else ""
                )
            )
        elif not encoding.is_constant:
            unsupported[channel] = (
                f"the {channel!r} channel is driven by field {encoding.field!r}; the flat kwargs take a "
                "constant, so the builder must resolve it and pass the values itself"
            )
        else:
            flat[keyword] = encoding.value
    return flat, unsupported


def _fold_marker_size(opts: Dict[str, Any]) -> None:
    """Fold the ``size`` channel onto the ``point_size`` a cleopatra point glyph takes (in place).

    ``size`` is what a marker's visual size is called on every backend, so it is the spelling the static tier
    accepts too; ``point_size`` is cleopatra's own name for it, on the glyph's **constructor**. One place
    decides that mapping, so no builder holds it.

    Args:
        opts: The glyph constructor kwargs, mutated in place: the ``size`` becomes ``point_size``.
    """
    size = opts.pop(MARKER_SIZE_KEY, None)
    if size is not None:
        opts["point_size"] = size


def relocate_flat_style(
    opts: Dict[str, Any], *, folds_marker_size: bool = False
) -> Dict[str, Any]:
    """Pop cleopatra-regrouped style keys out of a constructor kwargs dict, returning them.

    The glyph constructors reject these keys now — both the flat members (``levels``/``scheme``/``style``/…) and
    the typed group parameters (``color``/``contour``/``data_style``/…) they fold into. Pop them here so the
    constructor keeps only the options it still accepts (``cmap``, ``add_colorbar``, ``size_*``, …), and forward
    the returned dict to ``plot`` (where :func:`group_render_kwargs` folds any flat members into group objects and
    leaves an already-built group object untouched).

    Args:
        opts: The constructor keyword dict; mutated in place (matched keys are removed).
        folds_marker_size: ``True`` when the glyph being built is a point glyph, which takes its marker
            size on the constructor: the ``size`` channel is then folded onto the ``point_size`` it wants.
            Leave it ``False`` for every other glyph, where ``point_size`` belongs to the raster point
            overlay and travels on to ``plot()``.

    Returns:
        The removed style keys as a new dict.

    Examples:
        - The styling keys move out, and the constructor keeps what it still accepts:
            ```python
            >>> from digitalearth.static.render_compat import relocate_flat_style
            >>> opts = {"scheme": "quantiles", "k": 4, "cmap": "viridis"}
            >>> moved = relocate_flat_style(opts)
            >>> sorted(moved), opts
            (['k', 'scheme'], {'cmap': 'viridis'})

            ```
        - For a point glyph, the `size` channel folds onto the constructor spelling cleopatra wants:
            ```python
            >>> from digitalearth.static.render_compat import relocate_flat_style
            >>> opts = {"size": 12, "cmap": "viridis"}
            >>> relocate_flat_style(opts, folds_marker_size=True)
            {}
            >>> opts["point_size"]
            12

            ```
    """
    moved = {key: opts[key] for key in opts if key in FLAT_STYLE_KEYS}
    for key in moved:
        del opts[key]
    if folds_marker_size:
        _fold_marker_size(opts)
    return moved


#: Flat member kwargs per group parameter (used to spot styling a target glyph cannot accept).
_GROUP_MEMBERS = {
    **{param: frozenset(field_map) for param, _, field_map in _GROUP_SPECS},
    COLOR_GROUP_PARAM: frozenset(COLOR_GROUP_MEMBERS),
}

#: The point-overlay keys (the ``points`` array plus its ``point_*`` styling), rejected on a glyph with no
#: ``points`` parameter the same way an unsupported group is.
_POINT_OVERLAY_KEYS = frozenset({"points", *_POINT_FIELDS})


def _fold_points(out: Dict[str, Any]) -> None:
    """Wrap a bare ``points`` array plus any ``point_*`` styling into a ``PointOverlay`` (in place).

    Args:
        out: The keyword dict being prepared, mutated in place: ``points`` and every ``point_*`` key are
            taken out of it and one built ``points`` overlay put back. A dict whose ``points`` is already
            a ``PointOverlay`` is left alone, stray ``point_*`` keys included, so nothing is dropped in
            silence; one carrying neither an array nor any styling is a no-op.

    Raises:
        ValueError: when ``point_*`` styling was written with no ``points=`` array to attach it to,
            naming the key the caller actually spelled.
    """
    if isinstance(out.get("points"), PointOverlay):
        return  # already a built overlay; leave any stray point_* keys in place, don't silently drop them
    points = out.pop("points", None)
    # Keep the caller's spelling beside the PointOverlay field it folds into: the error below has to name
    # `point_color`, which the caller wrote, not `color`, which is a *different* declared style key here
    # (cleopatra's ColorScaling group) and would send them to the wrong keyword entirely.
    written = [key for key in _POINT_FIELDS if key in out]
    point_kw = {
        field: out.pop(key) for key, field in _POINT_FIELDS.items() if key in out
    }
    if points is None:
        if not point_kw:
            # No array and no styling — nothing to fold. `points=None` is how a wrapper forwards an
            # optional argument it was not given, and it was a no-op before; raising here broke every
            # such wrapper.
            return
        # Styling with no array to attach it to. Dropping it was the old behaviour and is the exact
        # silence this module's schema exists to remove: `point_color="red"` on a layer with no `points=`
        # did nothing and said nothing.
        raise ValueError(
            f"{sorted(written)} style the point overlay, but no points= array was given for them to "
            "apply to; pass points=, or drop the styling"
        )
    out["points"] = PointOverlay(points, **point_kw)


def _fold_group(
    out: Dict[str, Any], param: str, cls: type, field_map: Dict[str, str]
) -> None:
    """Fold one group's flat members in ``out`` into a ``param`` group object (in place).

    Args:
        out: The ``plot()`` keyword dict, mutated: the flat members are removed and the built group put in
            their place.
        param: The ``plot()`` parameter the group is passed under.
        cls: The cleopatra group class to build.
        field_map: ``{flat keyword: group field}`` for this group.

    Note:
        A caller who passed a built group under ``param`` keeps it, and their flat members are left in place
        for :func:`prepare_plot_kwargs` to refuse by name. The colour group does not come through here — it
        is built by :func:`~digitalearth.static.style_fold.fold_color_scaling`, which refuses that pair
        rather than forwarding it.
    """
    if out.get(param) is not None:
        return  # caller already passed a built group object under this name; leave any flat members in place
    members = {field: out.pop(key) for key, field in field_map.items() if key in out}
    if not members:
        return
    out[param] = cls(**members)


def group_render_kwargs(
    kwargs: Dict[str, Any], accepted: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Fold cleopatra's flat render keywords in ``kwargs`` into typed group objects.

    A bare ``points`` array (with any ``point_*`` styling) becomes a ``PointOverlay``; ``levels``/``labels``
    become a ``Contour``; ``scheme``/``k`` a ``Classify``; and so on. Keys that are not flat members — including
    an already-built group object passed under its group name — pass through untouched, so the function is safe
    to apply once, centrally, and idempotent on its own output.

    The colour group is the one built upstream, by ``ColorScaling.from_options`` — see
    :func:`~digitalearth.static.style_fold.fold_color_scaling`, which also refuses a built ``color=`` passed
    beside a flat colour keyword instead of forwarding both.

    Args:
        kwargs: The ``plot`` keyword dict to fold (not mutated).
        accepted: When given, only fold the groups whose parameter name is in this set (a target glyph's
            ``plot`` parameters); flat members of an unsupported group are left in place for the caller to
            handle. ``None`` (default) folds every group.

    Returns:
        A new dict with the folded flat members replaced by their group objects.
    """
    out = dict(kwargs)
    if (accepted is None or "points" in accepted) and (
        "points" in out or any(key in out for key in _POINT_FIELDS)
    ):
        _fold_points(out)
    for param, cls, field_map in _GROUP_SPECS:
        if (
            accepted is None or param in accepted
        ):  # skip a group this glyph's plot() cannot take
            _fold_group(out, param, cls, field_map)
    if accepted is None or COLOR_GROUP_PARAM in accepted:
        fold_color_scaling(out)
    return out


@lru_cache(maxsize=None)
def _plot_params(glyph_cls: type) -> frozenset:
    """The parameter names of a glyph class's ``plot`` method (cached per class).

    Args:
        glyph_cls: The glyph class — not an instance, since the signature is the class's and the cache is
            keyed on it.

    Returns:
        Every named parameter of ``glyph_cls.plot``, ``self`` and the trailing ``**kwargs`` name included.
        The membership test is what callers want it for, so the names are returned as a frozenset.
    """
    return frozenset(inspect.signature(glyph_cls.plot).parameters)


def plot_takes(glyph: Any, param: str) -> bool:
    """Whether ``glyph``'s ``plot`` declares `param` as a named parameter of its own.

    The same question :func:`prepare_plot_kwargs` asks of the style groups, asked of one keyword: cleopatra's
    glyphs do not all take the same ones, and every one of them ends in ``**kwargs``, so a keyword the glyph
    has no parameter for is not refused — it is forwarded to matplotlib, where it fails as something else
    entirely. Asking first is what keeps a keyword meant for the glyphs that understand it from reaching the
    ones that do not.

    Args:
        glyph: The cleopatra glyph about to be drawn.
        param: The parameter name to look for.

    Returns:
        ``True`` when `param` is an explicit parameter of ``type(glyph).plot``.

    Examples:
        - ``ArrayGlyph`` can compose over an existing axes; ``ScatterGlyph`` has no such parameter (it never
          clears anything, so it has no need of one):
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
            >>> from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
            >>> from digitalearth.static.render_compat import plot_takes
            >>> plot_takes(ArrayGlyph(np.zeros((2, 2))), "compose")
            True
            >>> plot_takes(ScatterGlyph(np.zeros(2), np.zeros(2)), "compose")
            False

            ```
    """
    return param in _plot_params(type(glyph))


def prepare_plot_kwargs(
    glyph: Any, kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Fold flat styling into the groups ``glyph`` supports, returning ``(plot_kwargs, deferred_alpha)``.

    Only the groups the glyph's ``plot`` accepts are built (the vector glyphs take ``color``/``contour``/
    ``classify`` but not ``data_style``/``cells``, so folding those blindly would raise an opaque ``TypeError``).
    A leftover flat member the glyph cannot take (including an unsupported ``points`` overlay) raises a clear
    ``ValueError`` naming it — except ``alpha``, which every layer should honour: it is returned as
    ``deferred_alpha`` for the caller to apply to the rendered artist, since the vector glyphs expose no
    ``alpha`` parameter upstream.

    This assumes each glyph advertises its supported groups as *explicit named* ``plot`` parameters — true for
    every cleopatra glyph today (``ArrayGlyph`` names ``color``/``contour``/``cells``/``data_style``/``points``;
    the vector glyphs name ``color``/``contour``/``classify``). A glyph that exposed its groups only through
    ``**kwargs`` would need this rejection revisited.

    Args:
        glyph: The cleopatra glyph about to be drawn.
        kwargs: The flat ``plot`` keyword dict.

    Returns:
        A ``(plot_kwargs, deferred_alpha)`` pair — the grouped ``plot`` kwargs, and the ``alpha`` to apply to
        the artist after drawing (``None`` when the glyph folded it into a ``DataStyle`` itself or none was given).

    Raises:
        ValueError: if a styling kwarg has no home on this glyph type (e.g. ``style=`` on a scatter layer), or
            if a built ``color=`` group is passed beside a flat colour keyword.
    """
    accepted = _plot_params(type(glyph))
    grouped = group_render_kwargs(kwargs, accepted)
    leftover = {}
    for param, members in _GROUP_MEMBERS.items():
        if param in accepted:
            continue
        for key in [k for k in grouped if k in members]:
            leftover[key] = grouped.pop(key)
    if (
        "points" not in accepted
    ):  # a point overlay on a glyph with no `points` parameter is unsupported too
        for key in [k for k in grouped if k in _POINT_OVERLAY_KEYS]:
            leftover[key] = grouped.pop(key)
    deferred_alpha = leftover.pop("alpha", None)
    if leftover:
        raise ValueError(
            f"{type(glyph).__name__} does not support the styling option(s) {sorted(leftover)}; "
            "they apply to raster/mesh layers, not this layer type"
        )
    return grouped, deferred_alpha
