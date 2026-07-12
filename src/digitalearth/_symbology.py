"""Shared symbology helpers — categorical (distinct-value → colour) mapping for thematic maps (DC.8).

Graduated / continuous classification lives upstream in ``cleopatra.styles.classify`` (numeric breaks); this
module is the **categorical** counterpart: map each distinct value of a field to a colour from a qualitative
colormap. It is pure numpy + matplotlib colour work (styling, not GIS), consumed by the tiers' ``choropleth``
``scheme="categorical"`` path to colour by an unordered attribute (land-use class, region name, …).

The mapping itself is generic colour work and is a candidate to move into cleopatra alongside ``classify``
(tracked in ``planning/geolibre-parity/ISSUE-TRACKER.md`` / upstream note); it lives here for now because no
cleopatra equivalent exists yet.
"""

from typing import Any, List, Tuple

import numpy as np

#: Default qualitative colormap for categorical symbology (10 distinct hues; cycled if more categories).
_DEFAULT_CATEGORICAL_CMAP = "tab10"

#: The continuous-colour default ``choropleth`` carries in its signature (right for graduated/continuous,
#: a poor fit for categorical). The categorical path swaps it for ``_DEFAULT_CATEGORICAL_CMAP``.
_CONTINUOUS_DEFAULT_CMAP = "viridis"


def resolve_categorical_cmap(cmap: str) -> str:
    """Pick the colormap for categorical symbology, swapping the continuous default for a qualitative one.

    ``choropleth``'s ``cmap`` defaults to the continuous ``"viridis"`` (right for graduated/continuous
    colouring). That sequential map reads poorly as discrete categories, so when the caller leaves the default
    in place it is swapped for :data:`_DEFAULT_CATEGORICAL_CMAP`; any other explicit ``cmap`` is honoured.

    Note this cannot distinguish "caller left the default" from "caller explicitly passed ``"viridis"``" — both
    are swapped to the qualitative default. To force a viridis categorical map, pass a distinct spelling such as
    ``"viridis_r"`` (or sample viridis yourself); this is deliberate, since viridis rarely reads well as
    discrete categories.

    Args:
        cmap: The colormap name passed to ``choropleth`` (a qualitative map is preferred for categorical).

    Returns:
        ``cmap`` unchanged, unless it is the continuous default — then the qualitative categorical default.

    Examples:
        - The continuous default is swapped for a qualitative colormap:
            ```python
            >>> from digitalearth._symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap("viridis")
            'tab10'

            ```
        - An explicitly chosen colormap is left untouched:
            ```python
            >>> from digitalearth._symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap("Set2")
            'Set2'

            ```
    """
    return _DEFAULT_CATEGORICAL_CMAP if cmap == _CONTINUOUS_DEFAULT_CMAP else cmap


def _categories(values: Any) -> List[Any]:
    """Return the distinct, non-null values of ``values`` in a stable order (sorted when sortable).

    Args:
        values: An array-like of category labels (strings or numbers); ``None``/``NaN`` are dropped.

    Returns:
        The unique categories, sorted ascending when they are mutually comparable, else in first-seen order.
    """
    seen: List[Any] = []
    for value in np.asarray(values, dtype=object).ravel():
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if value not in seen:
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
