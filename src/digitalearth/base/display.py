"""Getting data into the display CRS, and choosing its colours — once, rather than once per tier.

Three helpers were defined on each tier's own base class. The roadmap described them as *"byte-identical in
two tiers"*; measured at ``b9e067e4`` (the ``main`` this module replaced them on) by parsing each definition
and comparing the AST with docstrings stripped, only one of the three actually was:

| Helper | Definitions | Distinct bodies |
|---|---|---|
| ``_to_display_source`` | 2 | **1** — genuinely identical |
| ``_needs_reproject`` | 3 | **2** — interactive and web agree; static differs |
| ``_auto_cmap`` | 3 | **3** — all differ |

The `_auto_cmap` case is the interesting one, because all three claim in their own docstrings to *mirror*
the others. They differ in the name of the lookup they reach through — ``self._auto_style`` on one tier,
``self._style_for`` on another — and in whether the last-resort colormap is a parameter. Underneath, all three
call :func:`digitalearth.base.autostyle.auto_style` and take its ``cmap``. Three implementations kept in step
by hand and by docstring is the drift this refactor exists to remove.

They are **equivalent in practice today**: ``auto_style`` seeds every answer from the library's ``default``
group, which carries ``viridis``, so the ``cmap`` key is always present and truthy and the three spellings
cannot diverge. The one lifted here is the defensive one, which stays right if that ever stops being true —
not a live defect any of them had.

These are free functions taking the display CRS as an argument rather than reading ``self.crs``, so a tier
that is not a scene class can use them too, and so they can be tested without one.
"""

from typing import Any, Callable, Dict, Optional

from digitalearth.base.crs import reproject
from digitalearth.base.sources import Source, get_source

__all__ = ["auto_cmap", "needs_reproject", "to_display_source"]

#: Colormap used when the autostyle lookup recognises nothing. Every tier wrote this literal out; sharing it
#: is what stops a fifth appearing.
DEFAULT_CMAP = "viridis"


def needs_reproject(data: Any, crs: Any) -> bool:
    """Whether `data` must be reprojected to reach the display CRS.

    Args:
        data: A pyramids object exposing ``.epsg`` — a ``Dataset`` or ``FeatureCollection``.
        crs: The display CRS.

    Returns:
        ``False`` only when `crs` is an ``int`` equal to ``data.epsg``; ``True`` otherwise.

        Only an EPSG-int display CRS can be compared cheaply. For a proj4/WKT display CRS — an orthographic
        globe, say — the answer is always ``True``: ``data.epsg`` is unreliable for a projection with no
        authority code (pyramids reports 4326 for one), so comparing against it structurally would answer
        "already there" for data that is not.

        Read with ``getattr``, which is how the interactive and web tiers wrote it: the static copy used
        ``data.epsg`` directly and raised ``AttributeError`` on an input that declares no CRS at all. The
        tolerant reading is a superset, so lifting it changes no answer that was previously returned.

    Examples:
        - Matching EPSG ints need no warp:
            ```python
            >>> from digitalearth.base.display import needs_reproject
            >>> class Ds:
            ...     epsg = 4326
            >>> needs_reproject(Ds(), 4326)
            False

            ```
        - A proj4 display CRS always warps, because the comparison cannot be trusted:
            ```python
            >>> from digitalearth.base.display import needs_reproject
            >>> class Ds:
            ...     epsg = 4326
            >>> needs_reproject(Ds(), "+proj=ortho +lat_0=53 +lon_0=4")
            True

            ```
    """
    return not (isinstance(crs, int) and getattr(data, "epsg", None) == crs)


def to_display_source(data: Any, crs: Any, *, band: int = 1) -> Source:
    """Reproject `data` into the display CRS through pyramids, and wrap it as a :class:`Source`.

    The single display-CRS choke point a raster or vector builder calls. It settles the projection decision
    the same way on every tier: **pre-reproject in pyramids**, no cartopy anywhere. Data already in the
    display CRS passes through untouched.

    Args:
        data: A pyramids ``Dataset`` / ``FeatureCollection``, or anything
            :func:`~digitalearth.base.sources.get_source` accepts. A `Source` passes straight back — it has
            already been placed.
        crs: The display CRS.
        band: 1-based band to extract for raster inputs.

    Returns:
        The display-CRS source.

    Raises:
        Exception: whatever pyramids raises for a warp it cannot perform, or the extractor for data it
            cannot read.

    Examples:
        - A plain array has no CRS to warp from, so it goes straight to extraction:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.display import to_display_source
            >>> to_display_source(np.arange(6.0).reshape(2, 3), 4326).z.values.shape
            (2, 3)

            ```
    """
    if isinstance(data, Source):
        return data
    if hasattr(data, "epsg") and hasattr(data, "to_crs") and needs_reproject(data, crs):
        data = reproject(data, crs)
    return get_source(data, band=band)


def auto_cmap(
    source: Any,
    cmap: Optional[str],
    fallback: str = DEFAULT_CMAP,
    *,
    lookup: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> str:
    """Return the colormap to draw `source` with, resolving from its variable when the caller named none.

    Defers to :func:`digitalearth.base.autostyle.auto_style` — the same variable→style table the static
    ``Map`` consults, ECMWF-Magics match included — so a recognised field is coloured the same way whichever
    tier renders it.

    Args:
        source: The display-CRS source whose variable drives the lookup.
        cmap: The caller's colormap, or ``None`` to resolve one.
        fallback: Colormap to use when the lookup recognises nothing.
        lookup: The style lookup to consult, taking `source` and returning the style dict. ``None`` calls
            :func:`~digitalearth.base.autostyle.auto_style` directly. A tier passes its own lookup method
            here — the interactive tier's ``_auto_style``, the web tier's ``_style_for`` — because those
            methods are documented as the tier's *single* entry into the style table, and the tier's
            ``levels`` and ``units`` readers go through them. Calling ``auto_style`` directly instead made
            the colormap a separate lookup from the other two, and took away the one hook a subclass or a
            test had for replacing the lookup across all three.

    Returns:
        The caller's `cmap` when they named one, else the lookup's answer, else `fallback`.

        The fallback is reached with ``or`` rather than as a ``dict.get`` default. Both spellings give the
        same answer for every input the shipped library can produce — it seeds each result from a ``default``
        group carrying ``viridis``, so ``cmap`` is always present and truthy. ``or`` is chosen because it also
        holds if a library ever answers with ``cmap=None`` or ``""``, which ``.get("cmap", "viridis")``
        would hand back as the colormap.

    Examples:
        - A caller's choice always wins:
            ```python
            >>> from digitalearth.base.display import auto_cmap
            >>> auto_cmap(None, "magma")
            'magma'

            ```
    """
    if cmap is not None:
        return cmap
    if lookup is None:
        from digitalearth.base.autostyle import auto_style

        lookup = auto_style
    style: Dict[str, Any] = lookup(source)
    return style.get("cmap") or fallback
