"""Shared symbology helpers — categorical (distinct-value → colour) mapping for thematic maps (DC.8).

Graduated / continuous classification lives upstream in ``cleopatra.styles.classify`` (numeric breaks); this
module is the **categorical** counterpart: map each distinct value of a field to a colour from a qualitative
colormap, to colour by an unordered attribute (land-use class, region name, …).

This module has two distinct jobs, with **different scopes** — do not conflate them:

1. :func:`resolve_categorical_cmap` — the **cmap sentinel, shared by all three tiers** (static, interactive,
   web). Every tier resolves ``cmap`` through it before colouring, so one ``cmap`` cannot mean two different
   things depending on which tier renders it. The static tier must call it *before* handing off to cleopatra
   (see the function's own note on the sentinel mismatch).
2. :func:`categorical_colors` (and :func:`_categories`) — the category→colour **compiler, for the web and
   interactive tiers only**. The static tier takes its mapping from cleopatra instead (``scheme="categorical"``
   → ``Glyph._prepare_categorical_mapping``, cleopatra >=0.26.0), the canonical home for this colour work. What
   remains here is not a duplicate of it but a different *output shape*: cleopatra builds a matplotlib
   ``ListedColormap`` + ``BoundaryNorm`` bound to a glyph, whereas MapLibre needs a literal ``["match", …]``
   colour expression and GeoViews a plain ``{value: colour}`` dict — neither can consume a matplotlib mappable,
   so both tiers need the category→colour pairs as plain data. The ordering rule (sorted when sortable, else
   first-seen) is shared with cleopatra's ``styles.categorize``, so the classes agree across tiers.
"""

from typing import Any, List, Tuple

import numpy as np

#: Default qualitative colormap for categorical symbology (10 distinct hues; cycled if more categories).
_DEFAULT_CATEGORICAL_CMAP = "tab10"

#: The continuous-colour default ``choropleth`` carries in its signature (right for graduated/continuous,
#: a poor fit for categorical). The categorical path swaps it for ``_DEFAULT_CATEGORICAL_CMAP``.
_CONTINUOUS_DEFAULT_CMAP = "viridis"


def resolve_categorical_cmap(cmap: Any = None) -> Any:
    """Pick the colormap for categorical symbology, swapping the continuous default for a qualitative one.

    ``choropleth``'s ``cmap`` defaults to the continuous ``"viridis"`` (right for graduated/continuous
    colouring). That sequential map reads poorly as discrete categories, so when the caller leaves the default
    in place it is swapped for :data:`_DEFAULT_CATEGORICAL_CMAP`; any other explicit ``cmap`` is honoured.

    **This is the single sentinel for every tier** — static, interactive and web all resolve ``cmap`` through
    here before colouring, so one ``cmap`` cannot mean two different things depending on the tier. The static
    tier must resolve *before* handing off to cleopatra, whose ``PolygonGlyph`` applies the same idea against a
    **different** sentinel (its own continuous default, ``"coolwarm_r"``): left to itself it would honour an
    explicit ``"viridis"`` that the sibling tiers swap for ``tab10``.

    Note this cannot distinguish "caller left the default" from "caller explicitly passed ``"viridis"``" — both
    are swapped to the qualitative default. To force a viridis categorical map, pass a distinct spelling such as
    ``"viridis_r"`` (or sample viridis yourself); this is deliberate, since viridis rarely reads well as
    discrete categories.

    .. warning::
        One divergence remains and is **not** fixable from this side: ``cmap="coolwarm_r"`` is cleopatra's own
        sentinel, so the static tier swaps it for ``tab10`` while the web/interactive tiers honour it. Passing
        cleopatra's continuous default explicitly to a categorical map is a pathological case (it renders
        near-identically either way — see the qualitative-colormap note below); it needs an upstream change to
        settle. Tracked in `planning/geolibre-parity/upstream-cleopatra-categorical.md`.

    A **qualitative** (``ListedColormap``) colormap is what belongs here — ``tab10``, ``tab20``, ``Set1``,
    ``Set2``, ``Paired``. A continuous map (``viridis``, ``plasma``, ``coolwarm``) is honoured as given but
    reads poorly: category colours are drawn at the colormap's first ``n`` LUT entries, which for a 256-entry
    continuous map are near-identical shades. To build categories from a continuous map, sample it yourself and
    pass the resulting ``ListedColormap``.

    Args:
        cmap: The colormap passed to ``choropleth`` (a qualitative map is preferred for categorical), or
            ``None``/omitted when the caller supplied none — which resolves to the qualitative default.

    Returns:
        ``cmap`` unchanged, unless it is ``None`` or the continuous default — then the qualitative categorical
        default.

    Examples:
        - The continuous default is swapped for a qualitative colormap:
            ```python
            >>> from digitalearth._symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap("viridis")
            'tab10'

            ```
        - No colormap at all resolves to the same qualitative default:
            ```python
            >>> from digitalearth._symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap()
            'tab10'

            ```
        - An explicitly chosen colormap is left untouched:
            ```python
            >>> from digitalearth._symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap("Set2")
            'Set2'

            ```
    """
    left_at_default = cmap is None or cmap == _CONTINUOUS_DEFAULT_CMAP
    return _DEFAULT_CATEGORICAL_CMAP if left_at_default else cmap


def _categories(values: Any) -> List[Any]:
    """Return the distinct, non-null values of ``values`` in a stable order (sorted when sortable).

    Args:
        values: An array-like of category labels (strings or numbers); ``None``/``NaN`` are dropped.

    Returns:
        The unique categories, sorted ascending when they are mutually comparable, else in first-seen order.
    """
    seen: List[Any] = []
    seen_set: set = set()  # O(1) membership so dedup stays O(n), not O(n·k), for large columns
    for value in np.asarray(values, dtype=object).ravel():
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if value not in seen_set:
            seen_set.add(value)
            seen.append(value)
    try:
        return sorted(seen)
    except TypeError:  # mixed/unsortable types — keep first-seen order
        return seen


def categorical_colors(
    values: Any, cmap: str = _DEFAULT_CATEGORICAL_CMAP
) -> Tuple[List[Any], List[str]]:
    """Map the distinct values of a field to colours from a qualitative colormap (DC.8).

    The categorical analog of ``cleopatra.styles.classify``: instead of binning a continuous range, it assigns
    one colour per distinct value. Colours cycle through ``cmap`` when there are more categories than the
    colormap has entries.

    Args:
        values: An array-like of category labels (strings or numbers); ``None``/``NaN`` are ignored.
        cmap: A matplotlib (preferably qualitative) colormap name, e.g. ``"tab10"``/``"tab20"``/``"Set2"``.

    Returns:
        tuple[list, list[str]]: ``(categories, colors)`` — the distinct categories (sorted when sortable) and a
        ``#rrggbb`` hex colour per category, aligned by index.

    Raises:
        ValueError: if there are no non-null values to colour.

    Examples:
        - Distinct string categories get distinct colours:
            ```python
            >>> from digitalearth._symbology import categorical_colors
            >>> cats, colors = categorical_colors(["a", "b", "a", "c"])
            >>> cats
            ['a', 'b', 'c']
            >>> len(colors) == 3 and all(c.startswith("#") for c in colors)
            True

            ```
        - More categories than the colormap → colours cycle (never runs out):
            ```python
            >>> from digitalearth._symbology import categorical_colors
            >>> cats, colors = categorical_colors(list(range(12)), cmap="tab10")
            >>> len(colors)
            12

            ```
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    categories = _categories(values)
    if not categories:
        raise ValueError("no non-null values to colour categorically")
    colormap = colormaps[cmap]
    n_colors = getattr(colormap, "N", len(categories)) or len(categories)
    colors = [to_hex(colormap(i % n_colors)) for i in range(len(categories))]
    return categories, colors
