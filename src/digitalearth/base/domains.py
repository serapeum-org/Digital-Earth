"""Named geographic domains — region name → bounding box, for setting map extents.

This is pure bounding-box bookkeeping (no GIS computation): a registry of well-known regions as
``(west, south, east, north)`` in EPSG:4326. Reprojecting a resolved bbox to a display CRS is delegated to
pyramids, through :meth:`digitalearth.base.spec.Bounds.to_crs` or the static tier's ``Map.set_domain``.

**It lives in ``base`` because a name is data, not rendering.** It began in ``static/``, which meant
``base`` could not read it without importing a backend — so ``RenderTarget.view_request`` refused a named
domain outright and a view framed by ``domain="europe"`` read its whole source instead of that region. The
table has no matplotlib in it and no reason to belong to one tier; every tier resolves the same name to the
same box from here. ``digitalearth.static.domains`` re-exports it, so the old import path still works.

Three ways to name a region:

- a **registered name** or **alias** (case-insensitive), e.g. ``"europe"``, ``"germany"``, ``"uk"``;
- an explicit ``(west, south, east, north)`` **bbox**;
- a **union** — a sequence of any of the above — resolved to the single enclosing bbox by
  :func:`union_domains` (or by passing the sequence straight to :func:`resolve_domain`).
"""

import numbers
from collections.abc import Sequence
from typing import cast

#: Built-in named regions as ``(west, south, east, north)`` in EPSG:4326 (lon/lat degrees). Longitudes stay
#: in ``[-180, 180]``; a region that would cross the antimeridian is not listed, because a lon/lat bbox cannot
#: express it (see :func:`union_domains`).
DOMAINS = {
    # whole world and latitude bands
    "global": (-180.0, -90.0, 180.0, 90.0),
    "arctic": (-180.0, 66.5, 180.0, 90.0),
    "antarctic": (-180.0, -90.0, 180.0, -60.0),
    "tropics": (-180.0, -23.5, 180.0, 23.5),
    # continents
    "europe": (-25.0, 34.0, 45.0, 72.0),
    "africa": (-20.0, -38.0, 55.0, 38.0),
    "asia": (40.0, 5.0, 150.0, 78.0),
    "north-america": (-170.0, 10.0, -50.0, 75.0),
    "south-america": (-85.0, -57.0, -32.0, 14.0),
    "oceania": (110.0, -50.0, 180.0, 0.0),
    "north-atlantic": (-80.0, 20.0, 10.0, 70.0),
    # sub-regions
    "mediterranean": (-6.0, 30.0, 37.0, 47.0),
    "scandinavia": (4.0, 54.0, 32.0, 71.0),
    "iberia": (-10.0, 36.0, 4.0, 44.0),
    "middle-east": (34.0, 12.0, 63.0, 42.0),
    "east-asia": (100.0, 20.0, 146.0, 54.0),
    "southeast-asia": (92.0, -11.0, 141.0, 29.0),
    "south-asia": (60.0, 5.0, 97.0, 38.0),
    "central-america": (-92.0, 7.0, -77.0, 22.0),
    "caribbean": (-85.0, 9.0, -59.0, 27.0),
    "west-africa": (-18.0, 4.0, 16.0, 20.0),
    "southern-africa": (11.0, -35.0, 41.0, -15.0),
    # countries (continental/metropolitan extents)
    "germany": (5.8, 47.2, 15.1, 55.1),
    "france": (-5.2, 41.3, 9.6, 51.1),
    "united-kingdom": (-8.7, 49.9, 1.8, 60.9),
    "spain": (-9.4, 35.9, 4.4, 43.8),
    "italy": (6.6, 36.6, 18.6, 47.1),
    "united-states": (-125.0, 24.0, -66.9, 49.4),
    "canada": (-141.0, 41.7, -52.6, 83.1),
    "brazil": (-74.0, -34.0, -34.8, 5.3),
    "india": (68.0, 6.5, 97.4, 35.5),
    "china": (73.5, 18.0, 135.0, 53.6),
    "japan": (122.0, 24.0, 146.0, 46.0),
    "australia": (112.9, -43.7, 153.6, -10.1),
}

#: Alternative spellings that resolve to a canonical :data:`DOMAINS` key. Kept separate so the catalogue
#: stays one name per region while common shorthands still work.
ALIASES = {
    "uk": "united-kingdom",
    "britain": "united-kingdom",
    "great-britain": "united-kingdom",
    "usa": "united-states",
    "us": "united-states",
    "conus": "united-states",
    "med": "mediterranean",
    "antarctica": "antarctic",
    "north-pole": "arctic",
    "south-pole": "antarctic",
}

DomainLike = str | Sequence[float]


def _is_number(value: object) -> bool:
    """True for any real number (``numpy`` scalars included) but not a bool.

    ``bool`` is excluded because ``True`` would otherwise read as the longitude ``1``; ``numbers.Real``
    (rather than ``(int, float)``) is used so a numpy ``int64``/``float64`` bbox resolves on every platform
    — on Windows ``np.int64`` is not a Python ``int`` subclass.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _resolve_name(name: str) -> tuple[float, float, float, float]:
    """Resolve one registered name or alias to its bbox, case-insensitively."""
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in DOMAINS:
        raise KeyError(
            f"unknown domain {name!r}; not a registered region — see digitalearth.base.domains.DOMAINS"
        )
    return DOMAINS[key]


def resolve_domain(
    domain: "DomainLike | Sequence[DomainLike] | None",
) -> tuple[float, float, float, float] | None:
    """Resolve a domain name, alias, explicit bbox, or union to a ``(west, south, east, north)`` tuple.

    Args:
        domain: One of — a registered region name or alias (case-insensitive); an explicit
            ``(west, south, east, north)`` bbox of four numbers; a sequence of any of those, resolved to the
            single enclosing bbox (a union); or ``None``.

    Returns:
        The resolved bbox tuple in EPSG:4326, or ``None`` when ``domain`` is ``None``.

    Raises:
        KeyError: If a string is not a registered region or alias.
        ValueError: If a numeric bbox is not length 4, or a union sequence is empty.

    Examples:
        - Resolve a named region to its lon/lat bounds:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain("Europe")
            (-25.0, 34.0, 45.0, 72.0)

            ```
        - An alias resolves to its canonical region:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain("UK") == resolve_domain("united-kingdom")
            True

            ```
        - Pass an explicit bbox through unchanged:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain([0, 40, 20, 60])
            (0.0, 40.0, 20.0, 60.0)

            ```
        - A union of regions resolves to the enclosing bbox:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain(["iberia", "france"])
            (-10.0, 36.0, 9.6, 51.1)

            ```
        - An unknown region name is rejected:
            ```python
            >>> from digitalearth.base.domains import resolve_domain
            >>> resolve_domain("atlantis")
            Traceback (most recent call last):
                ...
            KeyError: "unknown domain 'atlantis'; not a registered region — see digitalearth.base.domains.DOMAINS"

            ```
    """
    if domain is None:
        return None
    if isinstance(domain, str):
        return _resolve_name(domain)
    parts = list(domain)
    if parts and all(_is_number(v) for v in parts):
        numbers = cast("list[float]", parts)
        bbox = tuple(float(v) for v in numbers)
        if len(bbox) != 4:
            raise ValueError(
                f"a bbox domain must be (west, south, east, north); got {domain!r}"
            )
        return bbox
    # A union's members are names (str) or bboxes (sub-sequences). A scalar member that _is_number did not
    # accept — a bool, or a number type it does not recognise — means this was meant to be a bbox, so give
    # the bbox message rather than routing the scalar into union_domains where list() would raise a cryptic
    # TypeError.
    if any(not isinstance(v, (str, Sequence)) for v in parts):
        raise ValueError(
            f"a bbox domain must be (west, south, east, north); got {domain!r}"
        )
    return union_domains(*cast("list[DomainLike]", parts))


def union_domains(*domains: DomainLike) -> tuple[float, float, float, float]:
    """Resolve each domain and return the single ``(west, south, east, north)`` bbox that encloses them all.

    Args:
        *domains: Two or more region names, aliases, or explicit bboxes (one is allowed and returns itself).

    Returns:
        The enclosing bbox: ``(min west, min south, max east, max north)`` over the resolved parts.

    Raises:
        ValueError: If no domains are given.

    Notes:
        The union is a plain lon/lat envelope, so two regions on opposite sides of the antimeridian (e.g.
        ``"oceania"`` and ``"north-america"``) enclose to a box that spans the globe the wrong way. The
        registry lists no antimeridian-crossing region for the same reason.

    Examples:
        - The box around two adjacent regions:
            ```python
            >>> from digitalearth.base.domains import union_domains
            >>> union_domains("iberia", "france")
            (-10.0, 36.0, 9.6, 51.1)

            ```
    """
    if not domains:
        raise ValueError("union_domains needs at least one domain")
    boxes = [resolve_domain(d) for d in domains]
    wests, souths, easts, norths = zip(*boxes)
    return (min(wests), min(souths), max(easts), max(norths))
