"""Shared symbology helpers — categorical (distinct-value → colour) mapping for thematic maps (DC.8).

Graduated / continuous classification lives upstream in ``cleopatra.styling.styles.classify`` (numeric breaks); this
module is the **categorical** counterpart: map each distinct value of a field to a colour from a qualitative
colormap, to colour by an unordered attribute (land-use class, region name, …).

This module has four distinct jobs, with **different scopes** — do not conflate the first two:

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
   first-seen) is **independently reimplemented** here (:func:`_categories`) and in cleopatra's
   ``styles.categorize``, kept in agreement by ``tests/base/test_symbology.py::test_matches_cleopatra_categorize``
   — not shared code, so a change on either side must be mirrored (the test is what catches a drift).
3. :func:`sample_cmap` — the **graduated** counterpart of that compiler, for every tier that has to hand a
   renderer literal colours rather than a colormap object: a colormap name plus a class count become ``n``
   evenly-spaced ``#rrggbb`` stops. The interactive tier feeds them to a HoloViews ``cmap`` list, the web tier
   to a MapLibre ``["step", …]`` paint expression and the 3-D tier to a PyVista lookup table — none of which
   can consume a matplotlib mappable. Each tier used to carry its own copy of the sampling, so "the same
   ``column``/``scheme``/``k`` colours identically everywhere" rested on three functions staying in step.
4. :func:`as_colormap` — the **resolver the two compilers above read through**, so a ``cmap`` given as a
   name and the same ``cmap`` given as a ``matplotlib.colors.Colormap`` classify identically (#315). It is
   not in ``__all__``; ``digitalearth.web.raster`` imports it by name.
"""

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "MISSING_COLOR",
    "categorical_colors",
    "is_null",
    "nulls_to_none",
    "resolve_categorical_cmap",
    "sample_cmap",
]

#: Default qualitative colormap for categorical symbology (10 distinct hues; cycled if more categories).
_DEFAULT_CATEGORICAL_CMAP = "tab10"

#: Neutral colour for a feature whose value is missing (``NaN``/``None``/``pd.NA``). Shared by every tier —
#: the web tier's ``["match", …]`` fallback, the interactive tier's dict-cmap fallback, the static tier's
#: colormap "bad" colour and the 3-D tier's lookup-table NaN colour — so missing data reads as *missing*
#: rather than as absent, identically everywhere. Keep this a single constant: two tiers spelling the same
#: idea differently is exactly how the ``cmap`` sentinels drifted apart.
#:
#: The rule is **classified layers**, not categorical ones: a graduated scheme puts a missing value outside
#: every class exactly as a categorical one does, so it is painted with this colour too. A *continuous* ramp
#: has no classes and is left to the renderer — no tier repaints it.
MISSING_COLOR = "#cccccc"

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
        cleopatra's continuous default explicitly to a categorical map is a pathological edge; it needs an
        upstream change to settle. Tracked in `planning/geolibre-parity/upstream-cleopatra-categorical.md`.

    A **qualitative** (``ListedColormap``) colormap is what belongs here — ``tab10``, ``tab20``, ``Set1``,
    ``Set2``, ``Paired``. A continuous map is honoured as given, and how well it reads depends on its matplotlib
    type: a ``LinearSegmentedColormap`` (``coolwarm``, ``RdBu``, ``jet``) is sampled at ``n`` evenly-spaced
    points, so the categories stay distinct; a perceptual ``ListedColormap`` (``viridis``, ``plasma`` — 256
    discrete entries) contributes its first ``n``, which are near-identical shades and read poorly. Either way
    :func:`categorical_colors` and cleopatra sample it identically, so the tiers agree. To spread a perceptual
    map across the categories, sample it into a shorter ``ListedColormap`` yourself.

    Args:
        cmap: The colormap passed to ``choropleth`` (a qualitative map is preferred for categorical), or
            ``None``/omitted when the caller supplied none — which resolves to the qualitative default.

    Returns:
        ``cmap`` unchanged, unless it is ``None`` or the continuous default — then the qualitative categorical
        default.

    Examples:
        - The continuous default is swapped for a qualitative colormap:
            ```python
            >>> from digitalearth.base.symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap("viridis")
            'tab10'

            ```
        - No colormap at all resolves to the same qualitative default:
            ```python
            >>> from digitalearth.base.symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap()
            'tab10'

            ```
        - An explicitly chosen colormap is left untouched:
            ```python
            >>> from digitalearth.base.symbology import resolve_categorical_cmap
            >>> resolve_categorical_cmap("Set2")
            'Set2'

            ```
    """
    left_at_default = cmap is None or cmap == _CONTINUOUS_DEFAULT_CMAP
    return _DEFAULT_CATEGORICAL_CMAP if left_at_default else cmap


def is_null(value: Any) -> bool:
    """Return whether ``value`` is missing, in any of the spellings a dataframe column can produce.

    A category column can carry its nulls as ``None`` (object dtype), ``float("nan")`` (numeric dtype), or
    ``pd.NA``/``pd.NaT`` (pandas' nullable ``string``/``Int64``/``boolean``/datetime dtypes — increasingly the
    default for anything round-tripped through modern pandas or pyarrow). All three mean "no data" and must
    never become a category: a bogus ``<NA>`` class would take a palette colour and shift every subsequent
    category's colour by one, so the same logical data would render differently purely by dtype.

    Args:
        value: A single scalar from a category column.

    Returns:
        ``True`` when the value is any flavour of null, else ``False``.

    Examples:
        - Every spelling of missing is caught, and real categories are kept:
            ```python
            >>> import numpy as np
            >>> import pandas as pd
            >>> from digitalearth.base.symbology import is_null
            >>> [is_null(v) for v in (None, np.nan, pd.NA, pd.NaT)]
            [True, True, True, True]
            >>> [is_null(v) for v in ("urban", 0, False)]
            [False, False, False]

            ```
    """
    try:
        result = bool(pd.isna(value))
    except (TypeError, ValueError):
        # A list-like or otherwise non-scalar value: `pd.isna` returns an elementwise array whose truth value
        # is ambiguous. Such a value is not a null, it is (an unhashable) category — let it flow on.
        result = False
    return result


def nulls_to_none(values: Any) -> np.ndarray:
    """Return ``values`` as an object array with every flavour of null spelled ``None``.

    The static tier's mapping is built by cleopatra, whose null test is ``value is None`` plus a float-``NaN``
    check — so a ``pd.NA`` from a pandas nullable dtype survives it and becomes a real, coloured ``<NA>``
    category. Normalizing the nulls to the one spelling cleopatra recognises keeps the static tier's classes
    identical to the web/interactive tiers' (which drop them via :func:`is_null`), without reaching into
    cleopatra. The underlying gap is upstream and is reported there, not patched here.

    Args:
        values: An array-like of category labels.

    Returns:
        An object-dtype array with all nulls replaced by ``None`` and non-null values untouched. Copied only
        when a null is actually rewritten; a null-free input may be returned as-is.

    Examples:
        - `pd.NA` is normalized to `None`, so it cannot become a category:
            ```python
            >>> import pandas as pd
            >>> from digitalearth.base.symbology import nulls_to_none
            >>> nulls_to_none(pd.array(["urban", pd.NA], dtype="string")).tolist()
            ['urban', None]

            ```
    """
    array = np.asarray(values, dtype=object)
    # `pd.isna` on an object array is reliably elementwise (a list/tuple/dict/set/ndarray cell reads as a
    # non-null value, matching :func:`is_null`), so one vectorized call suffices — no per-element loop.
    missing = np.asarray(pd.isna(array), dtype=bool)
    if missing.any():
        array = array.copy()
        array[missing] = None
    return array


def _categories(values: Any) -> List[Any]:
    """Return the distinct, non-null values of ``values`` in a stable order (sorted when sortable).

    Args:
        values: An array-like of category labels (strings or numbers); nulls (``None``/``NaN``/``pd.NA``) are
            dropped — see :func:`is_null`.

    Returns:
        The unique categories, sorted ascending when they are mutually comparable, else in first-seen order.
    """
    seen: List[Any] = []
    seen_set: set = (
        set()
    )  # O(1) membership so dedup stays O(n), not O(n·k), for large columns
    for value in np.asarray(values, dtype=object).ravel():
        if is_null(value):
            continue
        if value not in seen_set:
            seen_set.add(value)
            seen.append(value)
    try:
        return sorted(seen)
    except TypeError:  # mixed/unsortable types — keep first-seen order
        return seen


def as_colormap(cmap: Any) -> Any:
    """Return `cmap` as a matplotlib colormap, whether it was named or handed over as one.

    Every classifying helper in this module resolves a colormap through here — :func:`categorical_colors`
    and :func:`sample_cmap`, and so the web, interactive and 3-D tiers that call them; the static tier
    takes its own route, resolving a name against ``matplotlib.colormaps`` in
    :func:`~digitalearth.static.maps.raster._described_cmap` and handing cleopatra the object. A caller may
    hold one either way: `"magma"` is what a keyword argument usually carries, while `colormaps["magma"]` is
    what code that built or modified a ramp has. Both helpers below used to accept only the name — the
    categorical one looked the argument up as a dict key, which a `Colormap` cannot be (it defines `__eq__`
    and so has no hash), and the graduated one treated any non-string as an already-built sequence of
    colours, which a `Colormap` is not either. So a classified layer refused the very object the
    unclassified builders accept (#315).

    Args:
        cmap: A registered colormap name, or a `matplotlib.colors.Colormap`. Not a sequence of colours —
            that is a palette, and the two callers that accept one resolve it before they ask here.

    Returns:
        The colormap. An object is returned as it is; a name is looked up.

    Raises:
        KeyError: for anything hashable that is not a registered name — a name that is not registered, and
            equally a tuple of colours, which reads as ``('#f00', '#0f0') is not a valid value for
            colormap``. Either way it is matplotlib's own message naming what it was given.
        TypeError: when `cmap` is unhashable, because the lookup uses it as a dict key — a **list** of
            colours reads as ``unhashable type: 'list'`` (review L9).

    Examples:
        - A name and the colormap it names resolve to the same ramp:
            ```python
            >>> from matplotlib import colormaps
            >>> from digitalearth.base.symbology import as_colormap
            >>> as_colormap("magma")(0.5) == as_colormap(colormaps["magma"])(0.5)
            True

            ```
    """
    from matplotlib import colormaps
    from matplotlib.colors import Colormap

    return cmap if isinstance(cmap, Colormap) else colormaps[cmap]


def categorical_colors(
    values: Any, cmap: Any = _DEFAULT_CATEGORICAL_CMAP
) -> Tuple[List[Any], List[str]]:
    """Map the distinct values of a field to colours from a colormap, or from a palette given outright (DC.8).

    The categorical analog of ``cleopatra.styling.styles.classify``: instead of binning a continuous range, it assigns
    one colour per distinct value. Given a colormap — by name or as the object — the sampling mirrors
    ``cleopatra.styling.styles.categorize`` **exactly** (a sequence of colours is a spelling cleopatra's
    ``categorize`` does not take, and is cycled here as it was given), so
    the web/interactive tiers (which consume this) and the static tier (which consumes cleopatra) render one
    ``cmap`` as the same colours — a qualitative ``ListedColormap`` (``tab10``/``Set2``/…) contributes its
    palette entries in order, cycling when there are more categories than colours, while a continuous
    ``LinearSegmentedColormap`` (``coolwarm``/``RdBu``/…) is sampled at ``n`` **evenly-spaced** points so the
    categories stay visually distinct rather than collapsing to the first few near-identical LUT entries. This
    parity is pinned by ``tests/base/test_symbology.py::test_matches_cleopatra_categorize``.

    Args:
        values: An array-like of category labels (strings or numbers); nulls (``None``/``NaN``/``pd.NA``) are
            ignored (see :func:`is_null`).
        cmap: A matplotlib colormap name, a `Colormap` itself, or an already-built sequence of colours; a
            qualitative map (``"tab10"``/``"tab20"``/``"Set2"``) is preferred, but a continuous map is
            sampled evenly and honoured too. The three spellings are the ones :func:`sample_cmap` takes,
            deliberately: the tiers hand a caller's ``cmap`` to whichever of the two the scheme picks, so
            one argument has to mean one thing in both (review L9).

    Returns:
        tuple[list, list[str]]: ``(categories, colors)`` — the distinct categories (sorted when sortable) and a
        ``#rrggbb`` hex colour per category, aligned by index.

    Raises:
        ValueError: if there are no non-null values to colour, or if `cmap` is an empty sequence — cycling
            an empty palette would divide by zero, naming neither the argument nor this helper.
        KeyError: for a `cmap` **name** that is not registered, from :func:`as_colormap` via
            :func:`_palette` — matplotlib's own message naming it (``'nosuch' is not a valid value for
            colormap.``).
        TypeError: for a `cmap` that is neither a name, a `Colormap`, nor iterable — :func:`_palette` reads
            anything else as a sequence of colours (``'int' object is not iterable``).

    Examples:
        - Distinct string categories get distinct colours:
            ```python
            >>> from digitalearth.base.symbology import categorical_colors
            >>> cats, colors = categorical_colors(["a", "b", "a", "c"])
            >>> cats
            ['a', 'b', 'c']
            >>> len(colors) == 3 and all(c.startswith("#") for c in colors)
            True

            ```
        - More categories than the colormap → colours cycle (never runs out):
            ```python
            >>> from digitalearth.base.symbology import categorical_colors
            >>> cats, colors = categorical_colors(list(range(12)), cmap="tab10")
            >>> len(colors)
            12

            ```
        - An already-built sequence of colours is the palette, exactly as it is for :func:`sample_cmap`:
            ```python
            >>> from digitalearth.base.symbology import categorical_colors
            >>> categorical_colors(["a", "b", "c"], ["#ff0000", "#00ff00"])[1]
            ['#ff0000', '#00ff00', '#ff0000']

            ```
    """
    from matplotlib.colors import to_hex

    categories = _categories(values)
    if not categories:
        raise ValueError("no non-null values to colour categorically")
    n = len(categories)
    base_colors = _palette(cmap, n)
    colors = [to_hex(base_colors[i % len(base_colors)]) for i in range(n)]
    return categories, colors


def _palette(cmap: Any, n: int) -> List[Any]:
    """Return the colours `cmap` offers `n` categories, in the order they are handed out.

    Args:
        cmap: A colormap name, a `Colormap`, or an already-built sequence of colours.
        n: How many categories are being coloured, for the continuous case.

    Returns:
        The palette to cycle: a sequence of colours as given; a ``ListedColormap``'s own entries; or a
        continuous map sampled at `n` evenly-spaced points.

    Raises:
        ValueError: when `cmap` is an empty sequence, which the caller would then cycle by ``% 0``.
        KeyError: when `cmap` is a name no colormap is registered under, from :func:`as_colormap`.
        TypeError: when `cmap` is neither a name, a `Colormap`, nor iterable — anything else is read as a
            sequence of colours, and ``list()`` refuses it.
    """
    from matplotlib.colors import Colormap

    # Asked in the order `sample_cmap` asks it, because a colormap is neither a name nor a sequence of
    # colours and the two helpers have to read one argument the same way (review L9).
    if not isinstance(cmap, (str, Colormap)):
        given = list(cmap)
        if not given:
            raise ValueError(
                "cmap is an empty sequence, so there is no colour to give a category; pass a colormap "
                "name, a Colormap, or a non-empty sequence of colours"
            )
        return given
    colormap = as_colormap(cmap)
    # Match cleopatra.styling.styles.categorize exactly: a ListedColormap exposes its palette as `.colors` and cycles;
    # a LinearSegmentedColormap has none, so sample it at n evenly-spaced points (not the first n LUT entries,
    # which on a 256-entry map are near-identical). Keeping this identical to cleopatra is what makes one cmap
    # render the same colours on the static tier (via cleopatra) and the web/interactive tiers (via here).
    base_colors = getattr(colormap, "colors", None)
    if base_colors is None:
        base_colors = [colormap(x) for x in np.linspace(0.0, 1.0, n)]
    return list(base_colors)


def sample_cmap(cmap: Any, n: int) -> List[str]:
    """Sample ``cmap`` at ``n`` evenly-spaced stops and return them as hex colour strings.

    The colour side of graduated symbology, shared by every tier that has to hand its renderer literal
    colours: a colormap name becomes the concrete ``#rrggbb`` strings a HoloViews ``cmap`` list, a MapLibre
    ``["step", …]`` expression or a PyVista lookup table needs — one per class. matplotlib is imported lazily
    (only when a builder actually classifies something), so importing a tier stays engine-free.

    An already-built sequence of colours is returned as a plain list, untouched. That is what lets a caller
    pass either spelling of ``cmap`` down one code path, rather than branching before every call —
    :func:`categorical_colors` reads a sequence as the same thing, a palette, so the tiers can hand one
    caller's ``cmap`` to whichever of the two the scheme picks (review L9).

    Args:
        cmap: A matplotlib colormap name, a `Colormap` itself, or an already-built sequence of colours.
        n: How many colours to draw — one per class. Nothing enforces a floor: ``n=0`` returns ``[]``, and
            the callers all pass a class count they have already built. Ignored for a sequence of colours,
            which is taken as given.

    Returns:
        list[str]: ``n`` ``#rrggbb`` colours in ramp order, or ``cmap`` itself as a list when it already was
        a sequence of colours.

    Raises:
        KeyError: when `cmap` is a name no colormap is registered under, from :func:`as_colormap` —
            matplotlib's own message naming it.
        TypeError: when `cmap` is neither a name, a `Colormap`, nor iterable, because anything else is read
            as a sequence of colours (``'int' object is not iterable``).

    Examples:
        - One colour per class, spanning the whole ramp:
            ```python
            >>> from digitalearth.base.symbology import sample_cmap
            >>> sample_cmap("viridis", 3)
            ['#440154', '#21918c', '#fde725']

            ```
        - A single class takes the ramp's middle, not its dark end — one dark-purple polygon reads as
          missing data rather than as a category:
            ```python
            >>> from digitalearth.base.symbology import sample_cmap
            >>> sample_cmap("viridis", 1)
            ['#21918c']

            ```
        - An explicit list of colours is honoured as written:
            ```python
            >>> from digitalearth.base.symbology import sample_cmap
            >>> sample_cmap(["#ff0000", "#00ff00"], 2)
            ['#ff0000', '#00ff00']

            ```
    """
    from matplotlib.colors import Colormap, to_hex

    # Asked in this order because a colormap is neither a name nor a sequence of colours: taking "not a
    # string" to mean "a list of colours" is what made a `Colormap` fall through to `list(cmap)` (#315).
    if not isinstance(cmap, (str, Colormap)):
        return list(cmap)
    colormap = as_colormap(cmap)
    # One stop per class, spread across the whole ramp — for n == 1 that is the middle of it, since
    # linspace(0, 1, 1) would otherwise pin the single class to the ramp's dark end.
    stops = [0.5] if n == 1 else list(np.linspace(0.0, 1.0, n))
    return [to_hex(colormap(stop)) for stop in stops]
