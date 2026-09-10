"""Keyed-XYZ basemap definitions — the *provider* half of a tile basemap, with no renderer in sight.

Every backend can already render a tile source it is handed: the static tier passes an
``xyzservices.TileProvider`` to ``cleopatra.basemap.tiles.add_tiles``, the web tier emits a MapLibre raster
source from a URL, and the interactive tier builds a GeoViews ``WMTS`` from either. What none of them owns is
the *definition* of a *keyed* service — one that needs a credential, resolves a time-varying dataset id, and
covers only part of the world. That definition is pure data, so it lives here and every backend reads the same
one.

A :class:`KeyedTileSource` carries a URL template, the environment variable its credential comes from, the
attribution to display, and an optional lon/lat bounding box. It hands out a resolved URL (``tile_url``) with
``{z}/{x}/{y}`` left intact for whichever engine substitutes them, so nothing here imports a tile library —
the static backend is the only place an ``xyzservices.TileProvider`` is constructed.

**Planet NICFI** is the first preset (:func:`planet_nicfi`): ~4.77 m monthly mosaics of the tropics, free for
non-commercial use under NICFI terms. Two things about it are worth knowing before you rely on it:

* **The credential ends up in the tile URL.** That is how the service authenticates, but it means the URL is
  sensitive. ``cleopatra.basemap.tiles`` logs the built URL at ``DEBUG`` when a tile fetch fails, so the
  static backend suppresses that logger for the duration of a keyed fetch — see
  ``digitalearth.static.maps.decoration``. Do not otherwise run tile fetches at ``DEBUG`` into shared logs.
* **The mosaic ids are Planet's, and they have varied.** The pattern here matches Planet's published
  normalized-analytic / visual naming, but the ids should be confirmed against Planet's ``/mosaics`` API for
  any date you depend on; ``mosaic=`` overrides the derived id when they differ.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

#: ``(west, south, east, north)`` in lon/lat — the shape every bounding box here uses.
Extent = Tuple[float, float, float, float]

#: NICFI publishes the tropics only, roughly 30°N–30°S. Outside this band the tiles 404, and cleopatra
#: raises ``ConnectionError`` after its retries — an opaque way to learn the basemap does not cover you.
_NICFI_BOUNDS: Extent = (-180.0, -30.0, 180.0, 30.0)

#: Planet's Basemaps tile service, templated on the mosaic id and the API key.
_NICFI_URL = (
    "https://tiles.planet.com/basemaps/v1/planet-tiles/"
    "{mosaic}/gmap/{z}/{x}/{y}.png?api_key={api_key}"
)

#: ``flavour`` → the mosaic-id stem Planet publishes it under.
_NICFI_FLAVOURS = {
    "analytic": "planet_medres_normalized_analytic",
    "visual": "planet_medres_visual",
}

#: A ``YYYY-MM`` month, the granularity NICFI mosaics are published at.
_MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class KeyedTileSource:
    """A tile service that needs a credential, described without reference to any renderer.

    Attributes:
        name: A display name, also used as the ``xyzservices`` provider name in the static backend.
        url_template: The tile URL, containing ``{z}``/``{x}``/``{y}`` plus ``{api_key}`` and any keys of
            :attr:`params`. Only the latter are substituted here — the tile coordinates are left for the
            rendering engine.
        attribution: The text a map must display when using this service.
        credential_env: The environment variable :meth:`resolve_key` reads when no key is passed.
        max_zoom: The deepest zoom level the service serves.
        bounds: Optional lon/lat ``(west, south, east, north)`` the service covers; ``None`` means global.
        params: Extra template placeholders to substitute, e.g. the resolved mosaic id.
    """

    name: str
    url_template: str
    attribution: str
    credential_env: str
    max_zoom: int = 20
    bounds: Optional[Extent] = None
    params: Dict[str, str] = field(default_factory=dict)

    def resolve_key(self, api_key: Optional[str] = None) -> str:
        """Return the credential to use, preferring an explicit one over the environment.

        Args:
            api_key: A key supplied by the caller; ``None`` reads :attr:`credential_env`.

        Returns:
            The credential. Never logged or echoed by this module.

        Raises:
            ValueError: when no key was passed and the environment variable is unset or empty, naming the
                variable so the caller knows what to set.
        """
        if api_key:
            return api_key
        from_env = os.environ.get(self.credential_env, "")
        if not from_env:
            raise ValueError(
                f"{self.name} needs a credential: set the {self.credential_env} environment variable, "
                f"or pass api_key=... . The key is never read from anywhere else and never hard-coded."
            )
        return from_env

    def tile_url(self, api_key: Optional[str] = None) -> str:
        """Return the tile URL with the credential and any preset placeholders filled in.

        ``{z}``/``{x}``/``{y}`` are deliberately left in place: MapLibre, GeoViews and ``xyzservices`` each
        substitute them at fetch time, so the same string serves all three backends.

        Args:
            api_key: A key supplied by the caller; ``None`` reads the environment.

        Returns:
            The URL template, ready for a tile engine.

        Raises:
            ValueError: when no credential is available (see :meth:`resolve_key`).
        """
        url = self.url_template
        for placeholder, value in self.params.items():
            url = url.replace("{" + placeholder + "}", value)
        return url.replace("{api_key}", self.resolve_key(api_key))

    def check_bounds(self, extent: Optional[Extent]) -> None:
        """Raise when ``extent`` lies entirely outside the service's coverage.

        A partial overlap passes: a map spanning the tropics and beyond still shows tiles where they exist.
        Only a request with no overlap at all is refused, because that renders an empty basemap after a
        round of failed fetches.

        Args:
            extent: The area about to be drawn, as lon/lat ``(west, south, east, north)``; ``None`` skips
                the check, as does a source with no declared :attr:`bounds`.

        Raises:
            ValueError: when the two boxes do not overlap, quoting both so the mismatch is obvious.
        """
        if extent is None or self.bounds is None:
            return
        west, south, east, north = extent
        b_west, b_south, b_east, b_north = self.bounds
        if east < b_west or west > b_east or north < b_south or south > b_north:
            raise ValueError(
                f"{self.name} covers only {self.bounds} (lon/lat), and the requested extent {extent} "
                f"lies entirely outside it — this basemap would render blank. Use a global basemap, or "
                f"plot an area within the covered band."
            )


def planet_nicfi(
    date: str,
    *,
    flavour: str = "analytic",
    mosaic: Optional[str] = None,
) -> KeyedTileSource:
    """Build the Planet NICFI basemap for one monthly mosaic.

    NICFI (Norway's International Climate and Forest Initiative) funds Planet to publish ~4.77 m mosaics of
    the tropics, free for non-commercial use. Monthly mosaics run from 2020-09; earlier periods are biannual
    and named differently, so pass ``mosaic=`` for those.

    Args:
        date: The mosaic month as ``"YYYY-MM"``.
        flavour: ``"analytic"`` (surface reflectance, for index work) or ``"visual"`` (colour-corrected RGB,
            the better choice under a plot).
        mosaic: An explicit Planet mosaic id, overriding the one derived from ``date`` and ``flavour``. Use
            it when Planet's naming differs from the derived pattern.

    Returns:
        The configured source. The credential is *not* read here — only when a URL is built.

    Raises:
        ValueError: when ``date`` is not ``YYYY-MM``, or ``flavour`` is not one of the two published kinds.

    Examples:
        - Build a source and see the mosaic id it derived:
            ```python
            >>> from digitalearth.base.basemaps import planet_nicfi
            >>> src = planet_nicfi("2024-01")
            >>> src.params["mosaic"]
            'planet_medres_normalized_analytic_2024-01_mosaic'
            >>> src.bounds
            (-180.0, -30.0, 180.0, 30.0)

            ```
        - The visual flavour is a different Planet product, not a styling flag:
            ```python
            >>> from digitalearth.base.basemaps import planet_nicfi
            >>> planet_nicfi("2024-01", flavour="visual").params["mosaic"]
            'planet_medres_visual_2024-01_mosaic'

            ```
        - A malformed month is refused rather than silently requesting a mosaic that cannot exist:
            ```python
            >>> from digitalearth.base.basemaps import planet_nicfi
            >>> planet_nicfi("Jan 2024")
            Traceback (most recent call last):
                ...
            ValueError: date must be a 'YYYY-MM' month, got 'Jan 2024'

            ```

    See Also:
        digitalearth.base.basemaps.get_keyed_basemap: resolve this by name from a ``basemap()`` call.
    """
    if not _MONTH.match(date):
        raise ValueError(f"date must be a 'YYYY-MM' month, got {date!r}")
    if flavour not in _NICFI_FLAVOURS:
        raise ValueError(
            f"flavour must be one of {sorted(_NICFI_FLAVOURS)}, got {flavour!r} — 'analytic' is surface "
            f"reflectance, 'visual' is the colour-corrected RGB product"
        )
    resolved = mosaic or f"{_NICFI_FLAVOURS[flavour]}_{date}_mosaic"
    return KeyedTileSource(
        name=f"Planet.NICFI.{flavour}.{date}",
        url_template=_NICFI_URL,
        attribution="Imagery © Planet Labs PBC / NICFI — non-commercial use",
        credential_env="PLANET_API_KEY",
        max_zoom=20,
        bounds=_NICFI_BOUNDS,
        params={"mosaic": resolved},
    )


#: Keyed basemaps a ``basemap()`` call can name, lower-cased. Each value takes the preset's own keywords.
KEYED_BASEMAPS: Dict[str, Callable[..., KeyedTileSource]] = {
    "planet.nicfi": planet_nicfi,
}


def is_keyed_basemap(name: object) -> bool:
    """Whether ``name`` names a keyed basemap preset.

    Args:
        name: The value a caller passed as a basemap; anything not a string is not a preset name.

    Returns:
        ``True`` when :func:`get_keyed_basemap` would resolve it.

    Examples:
        - The preset name is matched case-insensitively:
            ```python
            >>> from digitalearth.base.basemaps import is_keyed_basemap
            >>> is_keyed_basemap("Planet.NICFI")
            True

            ```
        - An ordinary basemap name, a URL or a built provider object is not one:
            ```python
            >>> from digitalearth.base.basemaps import is_keyed_basemap
            >>> [is_keyed_basemap(v) for v in ("CartoDark", "https://a/{z}/{x}/{y}.png", None)]
            [False, False, False]

            ```
    """
    return isinstance(name, str) and name.lower() in KEYED_BASEMAPS


def get_keyed_basemap(name: str, **kwargs: object) -> KeyedTileSource:
    """Resolve a keyed basemap preset by name.

    Args:
        name: The preset name, e.g. ``"Planet.NICFI"`` (case-insensitive).
        **kwargs: The preset's own keywords — for NICFI, ``date`` / ``flavour`` / ``mosaic``.

    Returns:
        The configured :class:`KeyedTileSource`.

    Raises:
        ValueError: when ``name`` is not a known preset, listing the ones that are.
        TypeError: when the preset's required keywords are missing, from the preset itself.

    Examples:
        - Resolve NICFI by the name a ``basemap()`` caller would type:
            ```python
            >>> from digitalearth.base.basemaps import get_keyed_basemap
            >>> get_keyed_basemap("Planet.NICFI", date="2024-01").credential_env
            'PLANET_API_KEY'

            ```
        - An unknown name says what is available:
            ```python
            >>> from digitalearth.base.basemaps import get_keyed_basemap
            >>> get_keyed_basemap("Mapbox.Satellite", date="2024-01")
            Traceback (most recent call last):
                ...
            ValueError: unknown keyed basemap 'Mapbox.Satellite'; available: Planet.NICFI

            ```
    """
    factory = KEYED_BASEMAPS.get(name.lower())
    if factory is None:
        available = ", ".join(
            sorted(k.title().replace("Nicfi", "NICFI") for k in KEYED_BASEMAPS)
        )
        raise ValueError(f"unknown keyed basemap {name!r}; available: {available}")
    return factory(**kwargs)


def upper_placeholders(url: str) -> str:
    """Return ``url`` with ``{z}``/``{x}``/``{y}`` upper-cased, the spelling GeoViews' ``WMTS`` expects.

    Args:
        url: A tile URL template using the lower-case placeholders.

    Returns:
        The same template with the three tile-coordinate placeholders upper-cased; every other placeholder
        and the rest of the URL are untouched.

    Examples:
        - Only the tile coordinates change — a substituted key is left alone:
            ```python
            >>> from digitalearth.base.basemaps import upper_placeholders
            >>> upper_placeholders("https://a/{z}/{x}/{y}.png?api_key=abc")
            'https://a/{Z}/{X}/{Y}.png?api_key=abc'

            ```
    """
    for lower, upper in (("{z}", "{Z}"), ("{x}", "{X}"), ("{y}", "{Y}")):
        url = url.replace(lower, upper)
    return url
