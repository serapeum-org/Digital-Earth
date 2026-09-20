"""Named geographic domains — region name → bounding box, for setting map extents.

This is pure bounding-box bookkeeping (no GIS computation): a small registry of well-known regions as
``(west, south, east, north)`` in EPSG:4326. Reprojecting a resolved bbox to a display CRS is delegated to
pyramids, through :meth:`digitalearth.base.spec.Bounds.to_crs` or the static tier's ``Map.set_domain``.

**It lives in ``base`` because a name is data, not rendering.** It began in ``static/``, which meant
``base`` could not read it without importing a backend — so ``RenderTarget.view_request`` refused a named
domain outright and a view framed by ``domain="europe"`` read its whole source instead of that region. The
table has no matplotlib in it and no reason to belong to one tier; every tier resolves the same name to the
same box from here. ``digitalearth.static.domains`` re-exports it, so the old import path still works.
"""

from typing import Optional, Sequence, Tuple, Union

#: Built-in named regions as ``(west, south, east, north)`` in EPSG:4326 (lon/lat degrees).
DOMAINS = {
    "global": (-180.0, -90.0, 180.0, 90.0),
    "europe": (-25.0, 34.0, 45.0, 72.0),
    "africa": (-20.0, -38.0, 55.0, 38.0),
    "asia": (40.0, 5.0, 150.0, 78.0),
    "north-america": (-170.0, 10.0, -50.0, 75.0),
    "south-america": (-85.0, -57.0, -32.0, 14.0),
    "oceania": (110.0, -50.0, 180.0, 0.0),
    "north-atlantic": (-80.0, 20.0, 10.0, 70.0),
}

DomainLike = Union[str, Sequence[float]]


def resolve_domain(
    domain: Optional[DomainLike],
) -> Optional[Tuple[float, float, float, float]]:
    """Resolve a domain name or explicit bbox to a ``(west, south, east, north)`` tuple in EPSG:4326.

    Args:
        domain: A registered region name (case-insensitive), an explicit ``(west, south, east, north)``
            bbox, or ``None``.

    Returns:
        The resolved bbox tuple, or ``None`` when ``domain`` is ``None``.

    Raises:
        KeyError: If ``domain`` is a string that is not a registered region.
        ValueError: If ``domain`` is a sequence that is not length 4.

    Examples:
        - Resolve a named region to its lon/lat bounds:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain("Europe")
            (-25.0, 34.0, 45.0, 72.0)

            ```
        - Pass an explicit bbox through unchanged:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain([0, 40, 20, 60])
            (0.0, 40.0, 20.0, 60.0)

            ```
        - An unknown region name is rejected:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain("atlantis")
            Traceback (most recent call last):
                ...
            KeyError: "unknown domain 'atlantis'; known: ['africa', 'asia', 'europe', 'global', 'north-america', 'north-atlantic', 'oceania', 'south-america']"

            ```
    """
    if domain is None:
        return None
    if isinstance(domain, str):
        key = domain.strip().lower()
        if key not in DOMAINS:
            raise KeyError(f"unknown domain {domain!r}; known: {sorted(DOMAINS)}")
        return DOMAINS[key]
    bbox = tuple(float(v) for v in domain)
    if len(bbox) != 4:
        raise ValueError(
            f"a bbox domain must be (west, south, east, north); got {domain!r}"
        )
    return bbox
