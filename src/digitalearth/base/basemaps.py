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

* **The credential ends up in the tile URL, and therefore in anything that records that URL.** That is how
  an XYZ service authenticates a browser, so it is inherent rather than a bug — but it has consequences
  worth stating plainly:

  - **A saved web map contains the key.** ``WebMap.basemap("Planet.NICFI").save("map.html")`` writes the
    tile URL, key and all, into the HTML. So does a rendered interactive map, which means a committed
    notebook with saved output carries it too. Treat such a file as a secret: do not commit it, publish
    it, or paste it into an issue. The static tier does not have this exposure — it fetches tiles at plot
    time and embeds only the resulting image.
  - **A failed fetch can log it.** ``cleopatra.basemap.tiles`` logs the built URL at ``DEBUG``, so the
    static backend suppresses that logger for the duration of a keyed fetch (see
    ``digitalearth.static.maps.decoration``). Do not otherwise run tile fetches at ``DEBUG`` into shared
    logs.
  - **A non-``http(s)`` URL is refused here**, before it reaches a tile engine, because cleopatra reports
    that case by raising with the offending URL in the message — which the log suppression does not
    cover, since it is an exception rather than a log record.
* **The mosaic ids are Planet's, and they have varied.** The pattern here matches Planet's published
  normalized-analytic / visual naming, but the ids should be confirmed against Planet's ``/mosaics`` API for
  any date you depend on; ``mosaic=`` overrides the derived id when they differ.
"""

import inspect
import os
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

#: ``(west, south, east, north)`` in lon/lat — the shape every bounding box here uses.
Extent = tuple[float, float, float, float]

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

#: Characters that would end the placeholder they were substituted into and start something else —
#: a new query parameter, a fragment, or another path segment. ``#`` is the dangerous one: it truncates
#: the query string, so the ``api_key`` after it is never sent and the request goes out anonymous.
_URL_DELIMITERS = ("#", "?", "&", "/", " ")


#: A ``YYYY-MM`` month, the granularity NICFI mosaics are published at.
_MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")

#: The first monthly NICFI mosaic. Earlier periods are biannual and named differently, so the derived
#: id would be wrong for them — those need an explicit ``mosaic=``.
_NICFI_FIRST_MONTH = "2020-09"


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

    Examples:
        - Describe a keyed service of your own and build its URL:
            ```python
            >>> from digitalearth.base.basemaps import KeyedTileSource
            >>> source = KeyedTileSource(
            ...     name="Example.Imagery",
            ...     url_template="https://tiles.example/{z}/{x}/{y}.png?key={api_key}",
            ...     attribution="© Example",
            ...     credential_env="EXAMPLE_TILE_KEY",
            ... )
            >>> source.tile_url(api_key="abc123")
            'https://tiles.example/{z}/{x}/{y}.png?key=abc123'

            ```
        - Extra placeholders come from ``params``, so a dataset id can vary per call:
            ```python
            >>> from digitalearth.base.basemaps import KeyedTileSource
            >>> source = KeyedTileSource(
            ...     name="Example.Dated",
            ...     url_template="https://tiles.example/{layer}/{z}/{x}/{y}.png?key={api_key}",
            ...     attribution="© Example",
            ...     credential_env="EXAMPLE_TILE_KEY",
            ...     params={"layer": "2024-01"},
            ... )
            >>> source.tile_url(api_key="abc123").split("?")[0]
            'https://tiles.example/2024-01/{z}/{x}/{y}.png'

            ```
    """

    name: str
    url_template: str
    attribution: str
    credential_env: str
    max_zoom: int = 20
    bounds: Extent | None = None
    params: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Wrap :attr:`params` so the frozen dataclass is actually immutable.

        ``frozen=True`` stops the *field* being reassigned but says nothing about the dict it points at,
        so ``source.params["mosaic"] = ...`` would otherwise edit a preset in place.
        """
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __hash__(self) -> int:
        """Hash on the same values :meth:`__eq__` compares, with :attr:`params` as a sorted tuple.

        The hash a frozen dataclass generates hashes the field tuple, and :attr:`params` is a
        ``mappingproxy``, which delegates to the dict underneath and is therefore unhashable. Without
        this, a source could be compared but not put in a set, used as a dict key, or memoised.

        Returns:
            A hash consistent with equality.
        """
        return hash(
            (
                self.name,
                self.url_template,
                self.attribution,
                self.credential_env,
                self.max_zoom,
                self.bounds,
                tuple(sorted(self.params.items())),
            )
        )

    def resolve_key(self, api_key: str | None = None) -> str:
        """Return the credential to use, preferring an explicit one over the environment.

        Args:
            api_key: A key supplied by the caller; ``None`` reads :attr:`credential_env`.

        Returns:
            The credential, stripped of surrounding whitespace — a trailing newline from
            ``KEY=$(cat key.txt)`` is the usual way this arrives malformed. Never logged or echoed by this
            module.

        Raises:
            ValueError: when no key was passed and the environment variable is unset or empty, naming the
                variable so the caller knows what to set.

        Examples:
            - An explicit key is returned as given:
                ```python
                >>> from digitalearth.base.basemaps import KeyedTileSource
                >>> source = KeyedTileSource(
                ...     name="Example",
                ...     url_template="https://t/{z}/{x}/{y}.png?key={api_key}",
                ...     attribution="© Example",
                ...     credential_env="EXAMPLE_TILE_KEY",
                ... )
                >>> source.resolve_key("abc123")
                'abc123'

                ```
            - With nothing to read, the error names the variable to set:
                ```python
                >>> import os
                >>> from digitalearth.base.basemaps import KeyedTileSource
                >>> source = KeyedTileSource(
                ...     name="Example",
                ...     url_template="https://t/{z}/{x}/{y}.png?key={api_key}",
                ...     attribution="© Example",
                ...     credential_env="EXAMPLE_TILE_KEY_UNSET",
                ... )
                >>> os.environ.pop("EXAMPLE_TILE_KEY_UNSET", None) and None
                >>> try:
                ...     source.resolve_key()
                ... except ValueError as err:
                ...     print(str(err).split(",")[0])
                Example needs a credential: set the EXAMPLE_TILE_KEY_UNSET environment variable

                ```
        """
        if api_key:
            return api_key.strip()
        from_env = os.environ.get(self.credential_env, "").strip()
        if not from_env:
            raise ValueError(
                f"{self.name} needs a credential: set the {self.credential_env} environment variable, "
                f"or pass api_key=... . The key is never read from anywhere else and never hard-coded."
            )
        return from_env

    def tile_url(self, api_key: str | None = None) -> str:
        """Return the tile URL with the credential and any preset placeholders filled in.

        ``{z}``/``{x}``/``{y}`` are deliberately left in place: MapLibre, GeoViews and ``xyzservices`` each
        substitute them at fetch time, so the same string serves all three backends.

        Args:
            api_key: A key supplied by the caller; ``None`` reads the environment.

        Returns:
            The URL template, ready for a tile engine.

        Raises:
            ValueError: when no credential is available (see :meth:`resolve_key`), when a substituted
                value — a preset placeholder or the credential itself — contains a URL delimiter that
                would rewrite the request, or when the template is not ``http(s)``.

        Examples:
            - The credential is filled in and the tile coordinates are left for the engine:
                ```python
                >>> from digitalearth.base.basemaps import planet_nicfi
                >>> url = planet_nicfi("2024-01").tile_url(api_key="abc123")
                >>> "{z}/{x}/{y}" in url
                True
                >>> url.endswith("?api_key=abc123")
                True

                ```
            - The preset's own placeholders are resolved too, so nothing is left templated but the tiles:
                ```python
                >>> from digitalearth.base.basemaps import planet_nicfi
                >>> url = planet_nicfi("2024-01", flavour="visual").tile_url(api_key="abc123")
                >>> "planet_medres_visual_2024-01_mosaic" in url
                True
                >>> "{mosaic}" in url or "{api_key}" in url
                False

                ```
        """
        url = self.url_template
        for placeholder, value in self.params.items():
            self._check_substitution(placeholder, value)
            url = url.replace("{" + placeholder + "}", value)
        if not url.lower().startswith(("http://", "https://")):
            # Refuse here rather than let a tile engine complain: cleopatra reports this by raising with
            # the offending URL in the message, and that message would carry the credential.
            raise ValueError(
                f"{self.name} has a non-http(s) tile URL, which is refused because the error a tile "
                f"engine would raise for it quotes the URL — and the URL carries the credential."
            )
        key = self.resolve_key(api_key)
        # The credential is a substitution like any other, and the one most likely to arrive malformed:
        # it comes from an environment variable, often populated from a file.
        self._check_substitution("api_key", key, quote=False)
        return url.replace("{api_key}", key)

    def _check_is_lonlat(self, extent: Extent) -> None:
        """Refuse an extent that cannot be a lon/lat box, whatever this source's coverage is.

        This validates the *caller's* input, so it runs whether or not the source declares
        :attr:`bounds` — a source that covers the globe is not a reason to accept metres.

        Args:
            extent: The extent as passed to :meth:`check_bounds`.

        Raises:
            ValueError: when a latitude is outside ``[-90, 90]``, when the extent is inverted in latitude,
                or when the longitudes are too large or too far apart to be degrees. All three usually
                mean the extent is in a projected CRS.
        """
        west, south, east, north = extent
        if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
            raise ValueError(
                f"{self.name}: the extent {extent} has latitudes outside [-90, 90], so it is not lon/lat "
                f"— reproject it before checking coverage."
            )
        if abs(west) > 360.0 or abs(east) > 360.0 or abs(east - west) > 360.0:
            # Northings can land inside [-90, 90] by coincidence, so latitude alone does not settle it;
            # eastings and a span wider than the planet do.
            raise ValueError(
                f"{self.name}: the extent {extent} has longitudes that cannot be degrees, so it is not "
                f"lon/lat — reproject it before checking coverage."
            )
        if south > north:
            raise ValueError(
                f"{self.name}: the extent {extent} is inverted (south {south} is north of north {north})."
            )

    def _check_substitution(
        self, placeholder: str, value: str, *, quote: bool = True
    ) -> None:
        """Refuse a value that would rewrite the request rather than fill the placeholder it replaces.

        Args:
            placeholder: The placeholder being filled, named in the error so the caller knows which value
                to look at.
            value: The value about to be substituted.
            quote: Whether the value may be echoed in the error. ``False`` for the credential, whose whole
                point is not to appear in a message that may be logged or displayed.

        Raises:
            ValueError: when the value contains one of :data:`_URL_DELIMITERS`.
        """
        for delimiter in _URL_DELIMITERS:
            if delimiter in value:
                shown = repr(value) if quote else "the resolved credential"
                raise ValueError(
                    f"{self.name}: the {placeholder!r} value ({shown}) contains {delimiter!r}, which "
                    f"would rewrite the tile request rather than fill a placeholder."
                )

    def check_bounds(self, extent: Extent | None) -> None:
        """Raise when ``extent`` lies entirely outside the service's coverage.

        A partial overlap passes: a map spanning the tropics and beyond still shows tiles where they exist.
        Only a request with no overlap at all is refused, because that renders an empty basemap after a
        round of failed fetches.

        Longitude is compared modulo 360, so a ``0–360`` domain (200°E, say) is understood as the same
        place as ``-160``, and a box crossing the antimeridian is treated as the two spans it really
        covers rather than as an inverted one. Latitude is compared directly — it has no such convention.

        That makes the two axes deliberately asymmetric: ``south > north`` is an error, while
        ``west > east`` is read as a crossing rather than a mistake. The numbers cannot tell a swapped
        longitude pair from a genuine Pacific extent, and refusing the crossing would break the real case
        to catch the typo — so the crossing wins. A swapped tuple is caught by its latitudes instead.

        Args:
            extent: The area about to be drawn, as lon/lat ``(west, south, east, north)``; ``None`` skips
                the check, as does a source with no declared :attr:`bounds`.

        Raises:
            ValueError: when the two boxes do not overlap, quoting both so the mismatch is obvious; or
                when ``extent`` cannot be a lon/lat box at all — latitudes outside ``[-90, 90]``,
                longitudes too large or too far apart to be degrees, or an extent inverted in latitude,
                all of which usually mean it is in a projected CRS. An extent whose *longitudes* are
                inverted is read as an antimeridian crossing, not an error; see above.

        Examples:
            - A tropical extent passes, and so does one that merely straddles the edge:
                ```python
                >>> from digitalearth.base.basemaps import planet_nicfi
                >>> source = planet_nicfi("2024-01")
                >>> source.check_bounds((-60.0, -5.0, -55.0, 0.0))
                >>> source.check_bounds((-60.0, -40.0, -55.0, -25.0))

                ```
            - An extent with no overlap at all is refused before anything is fetched:
                ```python
                >>> from digitalearth.base.basemaps import planet_nicfi
                >>> try:
                ...     planet_nicfi("2024-01").check_bounds((5.0, 52.0, 6.0, 53.0))
                ... except ValueError as err:
                ...     print(str(err).split(",")[0])
                Planet.NICFI.analytic.2024-01 covers only (-180.0

                ```
            - A domain expressed 0-360 names the same place as its negative counterpart, and passes:
                ```python
                >>> from digitalearth.base.basemaps import planet_nicfi
                >>> source = planet_nicfi("2024-01")
                >>> source.check_bounds((200.0, -10.0, 300.0, 10.0))
                >>> source.check_bounds((-160.0, -10.0, -60.0, 10.0))

                ```
            - Metres are not degrees, and are reported as such rather than as missing coverage:
                ```python
                >>> from digitalearth.base.basemaps import planet_nicfi
                >>> try:
                ...     planet_nicfi("2024-01").check_bounds((500000.0, 5800000.0, 510000.0, 5810000.0))
                ... except ValueError as err:
                ...     print(str(err).split(",")[-1].strip())
                so it is not lon/lat — reproject it before checking coverage.

                ```
        """
        if extent is None:
            return
        self._check_is_lonlat(extent)
        if self.bounds is None:
            return
        west, south, east, north = extent
        b_west, b_south, b_east, b_north = self.bounds
        if not _spans_latitude(south, north, b_south, b_north) or not _spans_longitude(
            west, east, b_west, b_east
        ):
            raise ValueError(
                f"{self.name} covers only {self.bounds} (lon/lat), and the requested extent {extent} "
                f"lies entirely outside it — this basemap would render blank. Use a global basemap, or "
                f"plot an area within the covered band."
            )


def _spans_latitude(south: float, north: float, low: float, high: float) -> bool:
    """Whether two latitude ranges overlap at all.

    Args:
        south: Southern edge of the requested extent.
        north: Northern edge of the requested extent.
        low: Southern edge of the service's coverage.
        high: Northern edge of the service's coverage.

    Returns:
        ``True`` when the ranges share any latitude, touching edges included.
    """
    return not (north < low or south > high)


def _wrap_longitude(lon: float) -> float:
    """Normalise a longitude to ``[-180, 180)`` so 0–360 and -180–180 conventions compare equal.

    Args:
        lon: A longitude in either convention.

    Returns:
        The same meridian expressed in ``[-180, 180)``.
    """
    return (lon + 180.0) % 360.0 - 180.0


def _spans_longitude(west: float, east: float, low: float, high: float) -> bool:
    """Whether two longitude ranges overlap, allowing for wrap-around in either.

    A range whose normalised west is greater than its east crosses the antimeridian, and is treated as the
    two spans it actually covers. A service spanning the whole globe overlaps everything.

    Args:
        west: Western edge of the requested extent, in either longitude convention.
        east: Eastern edge of the requested extent.
        low: Western edge of the service's coverage.
        high: Eastern edge of the service's coverage.

    Returns:
        ``True`` when the ranges share any meridian.
    """
    if east - west >= 360.0 or high - low >= 360.0:
        return True  # one of them is global in longitude

    def segments(start: float, end: float) -> list[tuple[float, float]]:
        """Split a possibly-wrapping range into non-wrapping ``(start, end)`` pieces."""
        a, b = _wrap_longitude(start), _wrap_longitude(end)
        if a <= b:
            return [(a, b)]
        return [(a, 180.0), (-180.0, b)]

    return any(
        not (b_end < a_start or b_start > a_end)
        for a_start, a_end in segments(west, east)
        for b_start, b_end in segments(low, high)
    )


def planet_nicfi(
    date: str,
    *,
    flavour: str = "analytic",
    mosaic: str | None = None,
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
        ValueError: when ``date`` is not ``YYYY-MM``; when it predates the monthly series and no explicit
            ``mosaic`` is given, since the derived id would name a mosaic that does not exist; or when
            ``flavour`` is not one of the two published kinds.

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
        - A month before the monthly series says so, and points at the way to reach those periods:
            ```python
            >>> from digitalearth.base.basemaps import planet_nicfi
            >>> try:
            ...     planet_nicfi("2019-06")
            ... except ValueError as err:
            ...     print(str(err).split(";")[0])
            NICFI monthly mosaics start at 2020-09
            >>> older = planet_nicfi(
            ...     "2019-06", mosaic="planet_medres_normalized_analytic_2019-06_2019-11_mosaic"
            ... )
            >>> older.params["mosaic"]
            'planet_medres_normalized_analytic_2019-06_2019-11_mosaic'

            ```

    See Also:
        digitalearth.base.basemaps.get_keyed_basemap: resolve this by name from a ``basemap()`` call.
    """
    if not _MONTH.match(date):
        raise ValueError(f"date must be a 'YYYY-MM' month, got {date!r}")
    if mosaic is None and date < _NICFI_FIRST_MONTH:
        raise ValueError(
            f"NICFI monthly mosaics start at {_NICFI_FIRST_MONTH}; {date!r} predates them, and the "
            f"earlier biannual mosaics use a different id — pass mosaic=... for those."
        )
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
KEYED_BASEMAPS: dict[str, Callable[..., KeyedTileSource]] = {
    "planet.nicfi": planet_nicfi,
}

#: How each preset name is spelled back to the user, since the lookup key is lower-cased.
KEYED_BASEMAP_NAMES: dict[str, str] = {
    "planet.nicfi": "Planet.NICFI",
}

#: Every keyword any registered preset accepts. A backend uses this to tell a preset keyword apart from
#: one of its own, so it is derived from the registry rather than written out — a preset added with a new
#: keyword is understood by all three backends without touching them.
PRESET_KEYWORDS: frozenset[str] = frozenset(
    keyword
    for factory in KEYED_BASEMAPS.values()
    for keyword in inspect.signature(factory).parameters
)


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
        TypeError: when a keyword is missing or misspelled — re-raised naming the preset the caller asked
            for and the keywords it takes, since the factory behind it is an implementation detail.

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
            sorted(KEYED_BASEMAP_NAMES.get(k, k) for k in KEYED_BASEMAPS)
        )
        raise ValueError(f"unknown keyed basemap {name!r}; available: {available}")
    try:
        # Bind first and call outside the handler: only a *binding* failure is a keyword problem, and
        # renaming every TypeError the factory body raises would bury the real one.
        inspect.signature(factory).bind(**kwargs)
    except TypeError as err:
        # The factory is an implementation detail; report the preset the caller actually named.
        accepted = sorted(inspect.signature(factory).parameters)
        raise TypeError(
            f"{KEYED_BASEMAP_NAMES.get(name.lower(), name)}: {err}. Its keywords are {accepted}."
        ) from err
    return factory(**kwargs)
