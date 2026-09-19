"""The static tier's style fold: the colour group, built by cleopatra's own builder (DE-43, #302).

The first piece of the static seam's ``style_fold`` to land. :mod:`digitalearth.static.render_compat` folds
every other flat style family (``Contour``, ``Classify``, ``DataStyle``, ``CellValues``, ``PointOverlay``) by
mapping keys onto fields itself, because none of those groups has a builder upstream. ``ColorScaling`` does —
:meth:`cleopatra.styling.scaling.ColorScaling.from_options` — and the hand-rolled copy of it had already
drifted: the ``equalize`` scale's ``samples`` had no flat key at all, so ``color_scale="equalize", samples=64``
reached cleopatra as a stray keyword and raised *"the 'samples' option moved onto a grouped parameter
object"* — advice the caller had no way to follow through this tier's own surface.

So the colour group is built by the builder, with three rules that are this tier's rather than cleopatra's:

- **The friendly spellings stay.** ``sym_log``, ``symlog``, ``sym_lognorm``, ``boundary``, ``boundary_norm``
  are Digital-Earth's documented public names and ``from_options`` refuses every one of them (it validates
  through ``ColorScale(raw)``, which matches the enum's dashed values case-insensitively but does not map
  ``_`` to ``-``). :func:`coerce_color_scale` canonicalises first; the builder validates what is left.
- **No colour keyword means no colour group.** ``from_options({})`` returns ``ColorScaling()`` — a *linear*
  scale — and ``ColorScaling.to_options`` resets ``"norm"`` to ``None``, so passing that default group on
  every call would quietly turn ``Map.imshow(ds, norm=LogNorm())`` into a plain ``Normalize``. A default is
  not harmless here; it is a silent override of the caller's own norm.
- **A built group and a flat key is an error.** Passing ``color=ColorScaling.power()`` beside ``gamma=0.3``
  used to forward both, and cleopatra answered with *"pass color=ColorScaling.power(gamma=...)"* — which is
  what the caller already did. The two spellings style the same thing, so the fold names both and stops.
"""

from typing import Any, Dict, Mapping

from cleopatra.styling.scaling import ColorScale, ColorScaling

__all__ = [
    "COLOR_GROUP_MEMBERS",
    "COLOR_GROUP_PARAM",
    "COLOR_SCALE_ALIASES",
    "coerce_color_scale",
    "fold_color_scaling",
]

#: Flat ``color_scale=`` spellings -> the ``ColorScale`` member they name. The enum's own values carry dashes
#: (``"sym-lognorm"``, ``"boundary-norm"``), so the underscored spellings this tier documents reach nothing
#: upstream; these are what keeps them working.
COLOR_SCALE_ALIASES = {
    "linear": ColorScale.LINEAR,
    "power": ColorScale.POWER,
    "lognorm": ColorScale.LOGNORM,
    "sym_log": ColorScale.SYM_LOGNORM,
    "symlog": ColorScale.SYM_LOGNORM,
    "sym_lognorm": ColorScale.SYM_LOGNORM,
    "boundary": ColorScale.BOUNDARY_NORM,
    "boundary_norm": ColorScale.BOUNDARY_NORM,
    "midpoint": ColorScale.MIDPOINT,
    "equalize": ColorScale.EQUALIZE,
}

#: The ``plot()`` parameter the colour group is passed under.
COLOR_GROUP_PARAM = "color"

#: The flat keywords that fold into it — the seven ``from_options`` reads, in the spelling this tier accepts.
#: ``midpoint`` is the flat name for ``ColorScaling.center``, and ``samples`` is the ``equalize`` scale's
#: resolution, which had no flat key before #302.
COLOR_GROUP_MEMBERS = (
    "color_scale",
    "gamma",
    "line_threshold",
    "line_scale",
    "bounds",
    "midpoint",
    "samples",
)


def coerce_color_scale(value: Any) -> ColorScale:
    """Coerce a friendly ``color_scale=`` spelling (or a ``ColorScale``) to the enum member it names.

    Args:
        value: What the caller wrote — one of :data:`COLOR_SCALE_ALIASES`, an exact enum value such as
            ``"sym-lognorm"``, or a ``ColorScale`` member.

    Returns:
        The member. Unrecognised input raises here rather than reaching cleopatra as a bare string and
        crashing opaquely at render time (``'str' object has no attribute 'value'``).

    Raises:
        ValueError: naming the spellings there are, when the value is not one of them.

    Examples:
        - The underscored spelling this tier documents reaches the dashed enum value upstream:
            ```python
            >>> from digitalearth.static.style_fold import coerce_color_scale
            >>> coerce_color_scale("sym_log").value
            'sym-lognorm'

            ```
        - Case is not significant, and a member passes through:
            ```python
            >>> from cleopatra.styling.scaling import ColorScale
            >>> from digitalearth.static.style_fold import coerce_color_scale
            >>> coerce_color_scale("MidPoint") is coerce_color_scale(ColorScale.MIDPOINT)
            True

            ```
    """
    if isinstance(value, ColorScale):
        return value
    key = str(value).strip().lower().replace("-", "_")
    if key in COLOR_SCALE_ALIASES:
        return COLOR_SCALE_ALIASES[key]
    try:
        return ColorScale(str(value))  # exact enum value (e.g. "sym-lognorm")
    except ValueError:
        raise ValueError(
            f"color_scale={value!r} is not a recognised colour scale; use one of "
            f"{sorted(COLOR_SCALE_ALIASES)} (or a cleopatra ColorScale member)"
        ) from None


def _written_colour_keys(kwargs: Mapping[str, Any]) -> list:
    """Return the flat colour keywords present in ``kwargs``, in the declared order.

    Args:
        kwargs: A ``plot()`` keyword dict.

    Returns:
        The subset of :data:`COLOR_GROUP_MEMBERS` the caller wrote.
    """
    return [key for key in COLOR_GROUP_MEMBERS if key in kwargs]


def fold_color_scaling(out: Dict[str, Any]) -> None:
    """Fold the flat colour keywords in ``out`` into a ``ColorScaling`` under ``color=`` (in place).

    Args:
        out: The ``plot()`` keyword dict, mutated: the flat members are removed and the built group put in
            their place. A dict with no colour keyword is left exactly as it is — see the module docstring
            for why a default group is not harmless.

    Raises:
        ValueError: if a built ``color=`` group is passed together with a flat colour keyword (they style the
            same thing), or if ``color_scale=`` names no scale.

    Examples:
        - The flat keywords become the group, through cleopatra's own builder:
            ```python
            >>> from digitalearth.static.style_fold import fold_color_scaling
            >>> kwargs = {"color_scale": "power", "gamma": 0.7, "cmap": "viridis"}
            >>> fold_color_scaling(kwargs)
            >>> kwargs["color"].kind.value, kwargs["color"].gamma, kwargs["cmap"]
            ('power', 0.7, 'viridis')

            ```
        - The ``equalize`` scale's resolution is one of them, which is what the hand-rolled fold had lost:
            ```python
            >>> from digitalearth.static.style_fold import fold_color_scaling
            >>> kwargs = {"color_scale": "equalize", "samples": 64}
            >>> fold_color_scaling(kwargs)
            >>> kwargs["color"].samples
            64

            ```
        - Nothing about colour, nothing built — so a caller's own ``norm=`` is left in force:
            ```python
            >>> from digitalearth.static.style_fold import fold_color_scaling
            >>> kwargs = {"cmap": "viridis"}
            >>> fold_color_scaling(kwargs)
            >>> kwargs
            {'cmap': 'viridis'}

            ```
    """
    written = _written_colour_keys(out)
    built = out.get(COLOR_GROUP_PARAM)
    if built is not None:
        if written:
            scale = getattr(getattr(built, "kind", None), "value", "linear")
            builder = str(scale).replace("-", "_")
            raise ValueError(
                f"{COLOR_GROUP_PARAM}={type(built).__name__}(...) and {written} style the same thing; pass "
                f"one or the other — the flat keywords are the group's own fields, so "
                f"{COLOR_GROUP_PARAM}={type(built).__name__}.{builder}({written[0]}=...) carries them"
            )
        return
    if not written:
        return
    options = {key: out.pop(key) for key in written}
    if "color_scale" in options:
        options["color_scale"] = coerce_color_scale(options["color_scale"]).value
    out[COLOR_GROUP_PARAM] = ColorScaling.from_options(options)
