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
from digitalearth.base.spec.bounds import same_crs

__all__ = ["auto_cmap", "needs_reproject", "to_display_source"]

#: Colormap used when the autostyle lookup recognises nothing. Every tier wrote this literal out; sharing it
#: is what stops a fifth appearing.
DEFAULT_CMAP = "viridis"


def needs_reproject(data: Any, crs: Any) -> bool:
    """Whether `data` must be reprojected to reach the display CRS.

    Args:
        data: A pyramids object — a ``Dataset`` or ``FeatureCollection`` — exposing ``.crs`` and/or ``.epsg``.
        crs: The display CRS, in any spelling pyramids reads: an EPSG ``int``, an ``"EPSG:<code>"`` string, a
            proj4 or WKT string, or a pyproj ``CRS``.

    Returns:
        ``False`` when the data's own CRS and `crs` name the same reference system; ``True`` otherwise,
        including for data that declares no CRS at all.

        The two are compared by meaning, through :func:`~digitalearth.base.spec.bounds.same_crs`, the rule
        `Bounds` and `Viewport` already use. Comparing Python values instead skipped a warp only for an
        ``int`` display CRS, so a view holding ``"EPSG:4326"`` — the spelling `Viewport` writes for a CRS
        object — warped data that was already in EPSG:4326.

        The data's CRS **definition** (``.crs``) is read before its EPSG code, which is only the fallback. An
        EPSG code says nothing about a projection with no authority code, while the definition does: an
        orthographic dataset still warps to EPSG:4326, and needs no warp to its own orthographic CRS.

        Both attributes are read with ``getattr``, so an input declaring no CRS gets an answer rather than an
        ``AttributeError``.

    Examples:
        - The same system needs no warp, however the display CRS is spelled:
            ```python
            >>> from types import SimpleNamespace
            >>> from digitalearth.base.display import needs_reproject
            >>> data = SimpleNamespace(epsg=4326)
            >>> needs_reproject(data, 4326), needs_reproject(data, "EPSG:4326")
            (False, False)

            ```
        - A different system warps, and so does data that declares no CRS:
            ```python
            >>> from types import SimpleNamespace
            >>> from digitalearth.base.display import needs_reproject
            >>> needs_reproject(SimpleNamespace(epsg=4326), "+proj=ortho +lat_0=53 +lon_0=4")
            True
            >>> needs_reproject(SimpleNamespace(), 4326)
            True

            ```
    """
    own = getattr(data, "crs", None)
    if own is None:
        own = getattr(data, "epsg", None)
    if own is None:
        return True
    return not same_crs(own, crs)


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

    Defers to `lookup`, which by default is :func:`digitalearth.base.autostyle.auto_style` — the same
    variable→style table the static ``Map`` consults, ECMWF-Magics match included — so a recognised field is
    coloured the same way whichever tier renders it. Neither `source` nor the lookup is consulted when the
    caller named a colormap.

    Args:
        source: The display-CRS source whose variable drives the lookup.
        cmap: The caller's colormap, or ``None`` to resolve one.
        fallback: Colormap to use when the lookup's answer has no `cmap`, or an empty one. The default
            lookup always answers one — `viridis` for a variable it does not recognise — so with
            `lookup=None` this is not reached.
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
        - A tier hands in its own lookup, and the colormap is whatever that lookup answers:
            ```python
            >>> from digitalearth.base.display import auto_cmap
            >>> auto_cmap(None, None, lookup=lambda source: {"cmap": "RdBu_r", "units": "K"})
            'RdBu_r'

            ```
        - `fallback` is used only when the answer names no colormap; the default lookup always names one:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.display import auto_cmap
            >>> from digitalearth.base.sources import get_source
            >>> auto_cmap(None, None, "gray", lookup=lambda source: {})
            'gray'
            >>> auto_cmap(get_source(np.zeros((2, 2))), None, "gray")
            'viridis'

            ```
    """
    if cmap is not None:
        return cmap
    if lookup is None:
        from digitalearth.base.autostyle import auto_style

        lookup = auto_style
    style: Dict[str, Any] = lookup(source)
    return style.get("cmap") or fallback
