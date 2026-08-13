"""Translate Digital-Earth's flat styling kwargs into cleopatra's typed render-group objects.

cleopatra >=0.30 removed the loose ``plot``/``animate`` styling keywords (``levels``, ``scheme``, ``style``,
``color_scale``, ``points``, …) in favour of small typed *group objects* (``Contour``, ``Classify``,
``DataStyle``, ``ColorScaling``, ``CellValues``, ``PointOverlay``). Digital-Earth keeps accepting the flat
kwargs as its public surface and folds them here, so callers stay insulated from the upstream regrouping.

Two entry points:

- :func:`group_render_kwargs` — fold any flat members in a ``plot`` kwargs dict into their group objects
  (idempotent: an already-built group object, or an unrelated kwarg, is passed through untouched). Applied
  once, centrally, in :meth:`~digitalearth.scene.scene.Scene._render_glyph`.
- :func:`relocate_flat_style` — pop the flat members out of a *constructor* kwargs dict (the cleopatra glyph
  constructors now reject them) so they can be forwarded to ``plot`` instead.
"""
from typing import Any, Dict

from cleopatra.glyphs.gridded.array_glyph import PointOverlay
from cleopatra.styling.params import CellValues, Classify, Contour, DataStyle
from cleopatra.styling.scaling import ColorScale, ColorScaling

#: Flat ``color_scale=`` spellings -> the ``ColorScale`` enum the renderer needs (its ``.value``s carry dashes,
#: so a bare string like ``"power"`` constructs a ColorScaling but blows up at render time). Digital-Earth keeps
#: accepting the friendly names.
_COLOR_SCALE_ALIASES = {
    "linear": ColorScale.LINEAR,
    "power": ColorScale.POWER,
    "sym_log": ColorScale.SYM_LOGNORM,
    "symlog": ColorScale.SYM_LOGNORM,
    "sym_lognorm": ColorScale.SYM_LOGNORM,
    "lognorm": ColorScale.SYM_LOGNORM,
    "boundary": ColorScale.BOUNDARY_NORM,
    "boundary_norm": ColorScale.BOUNDARY_NORM,
    "midpoint": ColorScale.MIDPOINT,
}


def _coerce_color_scale(value: Any) -> Any:
    """Coerce a friendly ``color_scale=`` string to a ``ColorScale`` member; pass through anything unrecognised."""
    if isinstance(value, ColorScale):
        return value
    key = str(value).strip().lower().replace("-", "_")
    if key in _COLOR_SCALE_ALIASES:
        return _COLOR_SCALE_ALIASES[key]
    try:
        return ColorScale(str(value))  # exact enum value (e.g. "sym-lognorm")
    except ValueError:
        return value  # leave it; cleopatra raises a clear error listing the valid scales

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
    ("contour", Contour, {"levels": "levels", "labels": "labels", "label_kw": "label_kw"}),
    ("cells", CellValues, {
        "display_cell_value": "show", "num_size": "size", "background_color_threshold": "background_threshold",
    }),
    ("data_style", DataStyle, {
        "style": "style", "hillshade": "hillshade", "bands": "bands", "alpha": "alpha",
        "alpha_range": "alpha_range",
    }),
    ("classify", Classify, {"scheme": "scheme", "k": "k", "category_legend_kwargs": "category_legend_kwargs"}),
    ("color", ColorScaling, {
        "color_scale": "kind", "gamma": "gamma", "line_threshold": "line_threshold",
        "line_scale": "line_scale", "bounds": "bounds", "midpoint": "center",
    }),
)

#: The typed group parameters themselves (``color``/``contour``/``data_style``/``classify``/``cells``/``points``)
#: — the constructors reject these too, so a caller passing a group object directly must also route to ``plot``.
_GROUP_PARAMS = frozenset({param for param, _, _ in _GROUP_SPECS} | {"points"})

#: Every styling key cleopatra's glyph constructors now reject — the flat members, plus the typed group params.
FLAT_STYLE_KEYS = frozenset(
    {"points", *_POINT_FIELDS} | {flat for _, _, field_map in _GROUP_SPECS for flat in field_map} | _GROUP_PARAMS
)


def relocate_flat_style(opts: Dict[str, Any]) -> Dict[str, Any]:
    """Pop cleopatra-regrouped style keys out of a constructor kwargs dict, returning them.

    The glyph constructors reject these keys now — both the flat members (``levels``/``scheme``/``style``/…) and
    the typed group parameters (``color``/``contour``/``data_style``/…) they fold into. Pop them here so the
    constructor keeps only the options it still accepts (``cmap``, ``add_colorbar``, ``size_*``, …), and forward
    the returned dict to ``plot`` (where :func:`group_render_kwargs` folds any flat members into group objects and
    leaves an already-built group object untouched).

    Args:
        opts: The constructor keyword dict; mutated in place (matched keys are removed).

    Returns:
        The removed style keys as a new dict.
    """
    return {key: opts.pop(key) for key in list(opts) if key in FLAT_STYLE_KEYS}


def group_render_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Fold cleopatra's flat render keywords in ``kwargs`` into typed group objects.

    A bare ``points`` array (with any ``point_*`` styling) becomes a ``PointOverlay``; ``levels``/``labels``
    become a ``Contour``; ``scheme``/``k`` a ``Classify``; and so on. Keys that are not flat members — including
    an already-built group object passed under its group name — pass through untouched, so the function is safe
    to apply once, centrally, and idempotent on its own output.

    Args:
        kwargs: The ``plot`` keyword dict to fold (not mutated).

    Returns:
        A new dict with flat members replaced by their group objects.
    """
    out = dict(kwargs)
    if "points" in out or any(key in out for key in _POINT_FIELDS):
        points = out.pop("points", None)
        point_kw = {field: out.pop(key) for key, field in _POINT_FIELDS.items() if key in out}
        if points is not None and not isinstance(points, PointOverlay):
            out["points"] = PointOverlay(points, **point_kw)
        elif points is not None:
            out["points"] = points  # already a PointOverlay; stray flat point_* (if any) dropped above
    for param, cls, field_map in _GROUP_SPECS:
        members = {field: out.pop(key) for key, field in field_map.items() if key in out}
        if members and out.get(param) is None:
            if "kind" in members:  # color_scale=: coerce the friendly string to the ColorScale enum
                members["kind"] = _coerce_color_scale(members["kind"])
            out[param] = cls(**members)
    return out
