"""RasterMixin — web-tier raster builder (DW.1b, recipe W1).

``field`` puts a pyramids raster on the web map as a MapLibre **image source**: the band is reprojected
to lon/lat through pyramids, colour-mapped to an RGBA PNG (NoData → transparent), embedded as a ``data:`` URI,
and placed by its lon/lat corner coordinates. That is the default, and the only path a page can carry on its
own — every pixel is inside the HTML.

**A raster too big to inline takes a tiled route instead (#189).** ``tiles="xyz"`` writes a
``{z}/{x}/{y}.png`` pyramid beside the page and points a MapLibre ``raster`` source at it, needing nothing of
the page; ``tiles="cog"`` writes one Cloud-Optimized GeoTIFF and addresses it as ``cog://<file>``, which only
resolves on a page that has registered a ``cog://`` protocol — MapLibre has none built in, and
``WebMapBase.save`` has no hook to add the script, so that route is for a map embedded in a page somebody
else writes. Both change what this tier *produces*: the output is a **folder** — the HTML plus the pyramid or
the COG — rather than one file, which is why neither is the default and why the caller names the destination. Above
:data:`_LARGE_RASTER_PIXELS` the inline path is **refused** rather than warned about: 4 M pixels is a
2000x2000 raster, so a national-scale band used to be warned about and then inlined anyway, into a
multi-hundred-megabyte page.

The pixels of a tile come from pyramids (``Dataset.read_tile``, which computes the tile's EPSG:3857 bounds and
resamples through the overviews) and the colour from this tier. The only arithmetic here is the slippy-map
address itself — ``(lon, lat, zoom) → (x, y)`` — which is MapLibre's own tile scheme, the same one
``{z}/{x}/{y}`` templates are written in. Note that pyramids' ``Dataset.to_xyz`` is **not** a tile writer: it
exports lon/lat/value rows, so it has no part in this.

``rgb_composite`` and ``hsv_composite`` put *three* bands on the map the same way, and are one builder and one
drawer with the recipe recorded beside the layer: they share every step but what the three stretched channels
mean (#266). The stretch itself is ``digitalearth.base.stretch``'s, which states the rule both follow — "a
composite (RGB or HSV) turns three bands into an image by stretching each channel into ``[0, 1]``" — so a
composite looks the same here as on the static tier.

matplotlib (the colormap → RGBA → PNG encoding, and the HSV → RGB conversion) and numpy are imported lazily
inside the methods, so importing the tier needs neither the ``web`` extra nor matplotlib at module load.
"""

import math
import pathlib
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Self,
    Sequence,
    Tuple,
)

from digitalearth.base.ask import UNSET, Ask, Maybe, asked_record
from digitalearth.base.raster_classes import (
    BandClasses,
    asks_categorical,
    class_index,
    classes_of,
    classify_band,
)
from digitalearth.base.spec import (
    DEFAULT_BAND,
    DEFAULT_RAMP_STOPS,
    Bounds,
    Encoding,
    LayerSpec,
    LegendSpec,
    Scale,
    Symbology,
)
from digitalearth.base.stretch import DEFAULT_COMPOSITE_BANDS, require_three_bands
from digitalearth.web.base import _require_layer_api, as_finite

#: Pixel count above which the inline image-source path is refused (name a ``tiles=`` route for big rasters).
_LARGE_RASTER_PIXELS = 4_000_000

#: What a band with no name of its own is called, so a colour encoding has a field name rather than the
#: empty string an :class:`~digitalearth.base.spec.encoding.Encoding` refuses. The interactive tier's
#: ``UNNAMED_VALUE``, spelled the same, so one unnamed raster reads the same on both.
_UNNAMED_BAND = "z"

#: The tiled routes ``tiles=`` accepts, each mapped to the key its MapLibre ``raster`` source addresses the
#: data under: ``xyz`` lists tile-URL templates, ``cog`` names one file a client-side protocol reads windows
#: from. One structure, so the routes this tier *offers* and the routes its refusals *name* cannot drift.
_TILE_ROUTES: Mapping[str, str] = MappingProxyType({"xyz": "tiles", "cog": "url"})

#: Tile edge in pixels for a written pyramid — the web's default, and what MapLibre assumes.
_TILE_SIZE = 256

#: Highest zoom a derived range writes to, whatever the source resolution asks for. MapLibre's own ceiling is
#: 24; stopping earlier keeps a mislabelled geo-transform from asking for millions of tiles.
_MAX_TILE_ZOOM = 20

#: The latitude Web Mercator ends at. Beyond it the projection runs to infinity, so a tile address there has
#: to be clamped rather than computed.
_MERCATOR_LIMIT = 85.05112878

#: Cells the colour-limit scan reads. A tiled route colours each tile on its own, so every tile has to be
#: coloured on the *same* limits or neighbours disagree across their shared edge. They are measured from a
#: decimated read — reading the band whole to measure it is the cost the route exists to avoid.
_LIMIT_SCAN_BUDGET = 250_000

#: What an unstyled raster layer is drawn at. The value left the signatures with #334 — a style keyword's
#: default is now :data:`~digitalearth.base.ask.UNSET`, so a builder can tell an ask from its own default —
#: and this is where it lives instead, cited by name from the docstrings that used to show it.
RASTER_OPACITY = 1.0

#: How the single-band builder names itself in a refusal. The colour-limit check, the opacity check, the
#: route check and the tiled writer all speak for the same public call, so they share the one spelling —
#: named once, because a second spelling of the same method name would read as a second method.
_FIELD_CALLER = "WebMap.field()"

#: The recipe that maps the three stretched channels straight to red, green and blue. It is also what an
#: ``rgb`` layer whose description records no recipe at all means: every figure written before this tier had a
#: second composite is one of those, and read as anything else they would be refused or quietly recoloured.
_RGB_RECIPE = "rgb_composite"

#: The recipe that reads the same three channels as hue, saturation and value.
_HSV_RECIPE = "hsv_composite"

#: The composites this tier draws, each mapped to the prefix its generated layer ids take. One structure, so
#: the recipes the tier *has* and the ids it mints for them cannot come to name different sets.
_COMPOSITE_RECIPES: Mapping[str, str] = MappingProxyType(
    {_RGB_RECIPE: "rgb", _HSV_RECIPE: "hsv"}
)


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _colour_limits(
    limits: Optional[Sequence[float]], vmin: Any, vmax: Any, *, caller: str
) -> Tuple[Any, Any]:
    """Resolve the contract's `limits=` against this tier's own `vmin`/`vmax`.

    `limits` is the one spelling every tier answers to (#299); `vmin`/`vmax` are what this tier took before
    it, and both still work. What is refused is naming the same thing twice, which has no right answer.

    Args:
        limits: `(vmin, vmax)`, or `None`.
        vmin: The lower limit, or `None`.
        vmax: The upper limit, or `None`.
        caller: The method, for the message.

    Returns:
        The `(vmin, vmax)` pair to colour with.

    Raises:
        ValueError: when `limits` is given alongside either of the others, or is not a pair.
    """
    if limits is None:
        return vmin, vmax
    if vmin is not None or vmax is not None:
        raise ValueError(
            f"{caller} takes limits= or vmin=/vmax=, not both; they name the same thing"
        )
    # A sequence of two numbers, checked as such. `low, high = limits` unpacks a two-character string and a
    # two-element iterator just as happily, and the failure surfaced further down as numpy's own message
    # about a value this function had already seen (review L9).
    if isinstance(limits, (str, bytes)) or not isinstance(limits, Sequence):
        raise ValueError(
            f"{caller} limits must be a (vmin, vmax) pair of numbers; got {limits!r}"
        )
    if len(limits) != 2:
        raise ValueError(
            f"{caller} limits must be a (vmin, vmax) pair of numbers; got {len(limits)} values"
        )
    try:
        return float(limits[0]), float(limits[1])
    except (TypeError, ValueError):
        raise ValueError(
            f"{caller} limits must be a (vmin, vmax) pair of numbers; got {limits!r}"
        ) from None


def _grid_pixels(data: Any) -> Optional[int]:
    """Return how many cells a raster's grid holds, or ``None`` when it does not report one.

    Read from the grid rather than from a band's values: the composite builder holds the warped dataset and
    not a band, and reading one only to measure it would cost a whole band read.

    Args:
        data: A pyramids ``Dataset`` in the display CRS.

    Returns:
        ``rows * columns``, or ``None`` for an input that reports neither as a plain integer — a raster is
        the only thing that does, and a size that cannot be measured is not worth guessing at.
    """
    rows, columns = getattr(data, "rows", None), getattr(data, "columns", None)
    if isinstance(rows, int) and isinstance(columns, int):
        return rows * columns
    return None


def _refuse_if_large(caller: str, noun: str, pixels: Optional[int]) -> None:
    """Refuse, at the call, a raster too large to inline as a ``data:`` image source.

    Both raster builders embed their pixels in the page as a base64 PNG, so both have the same ceiling and
    the same two ways out. It was a **warning**, and a warning does not stop anything: 4 M pixels is a
    2000x2000 raster, so a national-scale band was warned about and then inlined anyway, into a page of
    several hundred megabytes. The refusal names both tiled routes, because a caller who is refused has to
    be told which parameter to reach for (#189).

    It belongs to the **builder**: a drawer runs again every time the layer is drawn back from its
    description, so a stored composite repeated the old advice once per draw for an input the caller chose
    once — while its sibling ``field`` warned from the call (review N7). A figure written before this change
    records no route and is drawn back unrefused, because its pixels are already in it.

    Args:
        caller: The builder's name, which opens the message.
        noun: What is being inlined — a ``"band"`` or a ``"composite"``.
        pixels: How many cells it holds, or ``None`` when that could not be measured — a size nobody
            measured is not one to refuse on.

    Raises:
        ValueError: when the raster holds more than :data:`_LARGE_RASTER_PIXELS` cells.
    """
    if pixels is None or pixels <= _LARGE_RASTER_PIXELS:
        return
    raise ValueError(
        f"{caller}: a {pixels}-pixel {noun} is too large to inline in the page as a data-URI image source "
        f'(the ceiling is {_LARGE_RASTER_PIXELS} cells). Name a tiled route instead: tiles="xyz" writes a '
        '{z}/{x}/{y}.png pyramid beside the page, tiles="cog" writes one Cloud-Optimized GeoTIFF the page '
        "reads windows from — both need tiles_path= to say where, and both make the output a folder rather "
        "than one file"
    )


def _tile_route(tiles: Any, caller: str) -> Optional[str]:
    """Resolve the caller's ``tiles=`` to one of the routes this tier writes.

    Args:
        tiles: The caller's ``tiles=`` — ``None`` for the inline path, or a route name.
        caller: The builder, for the message.

    Returns:
        The route in lower case, or ``None`` for the inline path.

    Raises:
        ValueError: for anything else, naming what was asked for and the routes that exist. Falling back to
            inlining would answer a request to *tile* a raster by embedding it, which is the page weight the
            caller asked to avoid. ``pmtiles`` is the likely near miss and is refused with the rest: pyramids
            writes PMTiles for vector data only, so it is not a raster route.
    """
    if tiles is None:
        return None
    route = str(tiles).strip().lower() if isinstance(tiles, str) else tiles
    if route not in _TILE_ROUTES:
        raise ValueError(
            f"{caller} tiles must be one of {sorted(_TILE_ROUTES)}, or None to inline the pixels in the "
            f"page; got {tiles!r}"
        )
    return str(route)


def _lonlat_bounds(dataset: Any) -> Bounds:
    """Return a raster's extent in lon/lat, without warping a single pixel.

    A tiled route needs the extent twice — to enumerate the tile addresses that cover it, and to frame the
    map on it — and reprojecting the whole raster to find out would defeat the route. Only the rectangle is
    converted, through :meth:`~digitalearth.base.spec.bounds.Bounds.to_crs`, which is pyramids' own corner
    transform.

    Args:
        dataset: A pyramids ``Dataset``.

    Returns:
        The enclosing rectangle in EPSG:4326.

    Raises:
        ValueError: when the extent cannot be expressed in lon/lat, in ``Bounds.to_crs``' own words.
    """
    return Bounds.from_bbox(list(dataset.bbox), dataset.epsg).to_crs(4326)


def _tile_index(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Return the slippy-map tile ``(x, y)`` a lon/lat falls in at ``zoom``.

    This is MapLibre's tile scheme, which is why it is here rather than asked of pyramids: a ``{z}/{x}/{y}``
    template *is* this addressing, and pyramids owns the other half — :meth:`Dataset.read_tile` turns one
    address back into the EPSG:3857 rectangle and the pixels in it.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees, clamped to :data:`_MERCATOR_LIMIT` — Web Mercator has no finite row beyond
            it, and a pole would otherwise index off the grid.
        zoom: The zoom level.

    Returns:
        The column and row, both clamped into the grid the zoom holds.
    """
    side = 2**zoom
    x = int((lon + 180.0) / 360.0 * side)
    latitude = math.radians(min(max(lat, -_MERCATOR_LIMIT), _MERCATOR_LIMIT))
    y = int((1.0 - math.asinh(math.tan(latitude)) / math.pi) / 2.0 * side)
    return min(max(x, 0), side - 1), min(max(y, 0), side - 1)


def _native_zoom(bounds: Bounds, columns: Any) -> int:
    """Return the zoom at which a tile pixel is about the size of a source cell.

    Writing beyond it invents detail the raster does not hold, and stopping short of it throws detail away —
    so it is where a derived range ends.

    Args:
        bounds: The raster's lon/lat extent.
        columns: How many columns the raster has.

    Returns:
        The zoom, clamped into ``[0, _MAX_TILE_ZOOM]``. Zero for an extent or a column count that cannot be
        measured, which is the one answer that asks for no tiles that do not exist. A width that is not a
        finite positive number is one of those: ``Bounds`` guarantees each edge is finite, not that their
        difference is, and a width of zero or one that overflows to infinity asks for ``log2`` of zero.
    """
    span = float(bounds.xmax) - float(bounds.xmin)
    if (
        not isinstance(columns, int)
        or columns < 1
        or not math.isfinite(span)
        or span <= 0.0
    ):
        return 0
    per_cell = span / columns
    zoom = math.ceil(math.log2(360.0 / (_TILE_SIZE * per_cell)))
    return min(max(zoom, 0), _MAX_TILE_ZOOM)


def _zoom_range(
    bounds: Bounds, dataset: Any, zooms: Any, caller: str
) -> Tuple[int, int]:
    """Resolve which zoom levels a pyramid is written for.

    Args:
        bounds: The raster's lon/lat extent.
        dataset: The raster, read for its column count.
        zooms: The caller's ``zooms=`` — a ``(lowest, highest)`` pair, or ``None`` to derive the range.
        caller: The builder, for the message.

    Returns:
        ``(lowest, highest)``. ``None`` derives ``(0, native)``: the low zooms are one tile each and make the
        raster visible when the map is zoomed out, which a pyramid starting at its native zoom is not.

    Raises:
        ValueError: when ``zooms`` is not a pair of whole numbers in ``[0, _MAX_TILE_ZOOM]`` with the lower
            first. A reversed pair would write nothing at all and look like a silent failure.
    """
    if zooms is None:
        return 0, _native_zoom(bounds, getattr(dataset, "columns", None))
    if (
        isinstance(zooms, (str, bytes))
        or not isinstance(zooms, Sequence)
        or len(zooms) != 2
    ):
        raise ValueError(
            f"{caller} zooms must be a (lowest, highest) pair of zoom levels; got {zooms!r}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in zooms):
        raise ValueError(f"{caller} zooms must be whole zoom levels; got {zooms!r}")
    lowest, highest = int(zooms[0]), int(zooms[1])
    if not 0 <= lowest <= highest <= _MAX_TILE_ZOOM:
        raise ValueError(
            f"{caller} zooms must run from the lowest to the highest, within 0..{_MAX_TILE_ZOOM}; "
            f"got {zooms!r}"
        )
    return lowest, highest


def _tile_addresses(bounds: Bounds, zoom: int) -> Iterator[Tuple[int, int]]:
    """Yield every tile ``(x, y)`` at ``zoom`` that the extent touches.

    Args:
        bounds: The lon/lat extent to cover.
        zoom: The zoom level.

    Yields:
        The tile column and row, north-west first — the order a pyramid is written in.
    """
    west, north = _tile_index(float(bounds.xmin), float(bounds.ymax), zoom)
    east, south = _tile_index(float(bounds.xmax), float(bounds.ymin), zoom)
    for x in range(west, east + 1):
        for y in range(north, south + 1):
            yield x, y


def _off_tile(error: BaseException) -> bool:
    """Whether a failed tile read means the tile misses the raster rather than that the read broke.

    Args:
        error: What the reader raised.

    Returns:
        ``True`` for pyramids' out-of-bounds report, matched by **type name** so this module needs no import
        of it — :func:`digitalearth.base.sources.view._off_source` matches the same way, and for the same
        reason: the exception has moved package once already. The message is deliberately not searched, since
        "out of bounds" is also the text of the commonest indexing bug in Python, and swallowing one would
        turn a real defect into a silently missing tile.
    """
    return "OutOfBounds" in type(error).__name__


def _png_bytes(rgba: Any) -> bytes:
    """Encode a float RGBA array as PNG bytes.

    Args:
        rgba: An ``(rows, cols, 4)`` array of channel values in ``[0, 1]``.

    Returns:
        The PNG file contents. matplotlib is imported here so the tier imports without it.
    """
    import io

    import numpy as np
    from matplotlib import image as mpimage

    buffer = io.BytesIO()
    mpimage.imsave(buffer, (np.asarray(rgba) * 255).astype("uint8"), format="png")
    return buffer.getvalue()


def _band_name(data: Any, band: int) -> str:
    """Return the name of the value a raster layer's colour varies with.

    A raster has no classified column, so the honest name for the field its colour is bound to is the
    **band's own** — read straight off the raster's ``band_names``/``variable_names``, never invented, and
    never obtained by warping the band a second time to look.

    The tier's other metadata reader is **not** the precedent for that, although the interactive tier's is:
    :meth:`~digitalearth.web.base.WebMapBase._auto_cmap` is handed the display-CRS
    :class:`~digitalearth.base.sources.source.Source` the builder already warped — measured, `field()` calls
    it with a ``Source``, and it raises ``AttributeError: 'Dataset' object has no attribute 'metadata'`` if
    given the raster — so it reads the band name only as the extractor already recorded it under the
    source's ``variable``. The method that reads ``band_names`` off the raster the way this does is
    :meth:`~digitalearth.interactive.base.InteractiveMapBase._auto_cmap_for_band`, which this tier has no
    counterpart of.

    Args:
        data: The raster the layer draws — a pyramids ``Dataset`` / ``NetCDF``. Read with `getattr`, so an
            input declaring neither list simply falls back.
        band: The 1-based band the layer colours by.

    Returns:
        The band's name, the first variable's name, or :data:`_UNNAMED_BAND` — never an empty string, which
        an :class:`~digitalearth.base.spec.encoding.Encoding` refuses as a field name.

    Note:
        The interactive tier answers the same question, by the same two lookups and with the same fallback,
        in its own ``coloured_by`` — measured on ``examples/data/acc4000.tif``, both answer ``'Band_1'``.
        They are not one import on purpose: the four backends are peers, and a backend importing another
        would tie this tier's ``web`` extra to the ``interactive`` one.
    """
    names = list(getattr(data, "band_names", None) or [])
    if 1 <= band <= len(names) and names[band - 1]:
        return str(names[band - 1])
    # A `NetCDF` has variables rather than bands, and the display extractor takes the **first** of them, so
    # that is the one a colour key would explain.
    first = next(iter(getattr(data, "variable_names", None) or []), None)
    return str(first) if first else _UNNAMED_BAND


def _colour_domain(
    values: Any, *, vmin: Any, vmax: Any, nodata: Any = None
) -> Optional[Tuple[Any, Any, Tuple[float, float]]]:
    """Return which of a band's cells have a value, and the span its colours are stretched across.

    The one place this tier decides what a band's colours *mean*, so the image and the key that explains
    it cannot be built on two different answers. :func:`_coloured_png` needs all three parts; the `field`
    builder needs only the limits, and needs them at a point where no image has been encoded yet — order 24
    describes a layer before it is drawn, and a colour key hangs on that description.

    Args:
        values: A 2-D (possibly masked) array of band values.
        vmin: The caller's lower limit, or ``None`` to measure one from these cells.
        vmax: The caller's upper limit, or ``None`` to measure one.
        nodata: A sentinel read as missing alongside the non-finite cells, or ``None``. A windowed tile read
            hands back the raster's own NoData rather than NaN — including in the padding of an edge tile.

    Returns:
        ``(data, valid, (lo, hi))`` — the values as plain floats with missing cells as ``nan``, the mask of
        cells that have one, and the limits those cells are coloured between. ``None`` when no cell has a
        value: a pyramid legitimately contains tiles the raster does not reach, and a band of pure NoData
        has no span to describe or to draw.
    """
    import numpy as np

    array = np.ma.asarray(values).astype(float)
    data = (
        array.filled(np.nan)
        if np.ma.isMaskedArray(array)
        else np.asarray(array, dtype=float)
    )
    valid = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        valid &= data != float(nodata)
    if not valid.any():
        return None
    # The domain, the explicit-limit override and the constant-band widening are one rule, in base/spec.
    # `valid` is already computed above, so the finite subset is handed over rather than derived twice —
    # this is the tier with the explicit inline-pixel budget.
    return data, valid, Scale.from_finite(data[valid], vmin=vmin, vmax=vmax).as_limits()


def _coloured_png(
    values: Any,
    cmap: Any,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nodata: Any = None,
) -> Optional[bytes]:
    """Colour-map a 2-D array to an RGBA PNG, drawing what it has no value for transparent.

    The one encoder behind both the inlined image and every written tile, so a tile is coloured by exactly
    the rule the inline path colours a whole band by. Its ``vmin``/``vmax`` are what hold a pyramid together:
    a tile normalised over its *own* range disagrees with its neighbour across their shared edge.

    Args:
        values: A 2-D (possibly masked) array of band values, already oriented north-up.
        cmap: A registered colormap name, or a ``Colormap`` itself, resolved through
            :func:`~digitalearth.base.symbology.as_colormap`.
        vmin: Lower colour limit; ``None`` uses the finite minimum of *these* values.
        vmax: Upper colour limit; ``None`` uses their finite maximum.
        nodata: A sentinel value drawn transparent alongside the non-finite cells, or ``None``. A windowed
            tile read hands back the raster's own NoData rather than NaN — including in the padding of an
            edge tile — so without it the whole margin of a pyramid is coloured as data.

    Returns:
        The PNG bytes, or ``None`` when no cell has a value to colour. ``None`` rather than an exception
        because a pyramid legitimately contains tiles the raster does not reach: an extent's bounding box is
        not the extent.

    Raises:
        KeyError: when ``cmap`` is hashable but names no colormap matplotlib's registry holds.
        TypeError: when ``cmap`` is unhashable — a list of colours, say.
    """
    import numpy as np
    from matplotlib.colors import Normalize

    from digitalearth.base.symbology import as_colormap

    domain = _colour_domain(values, vmin=vmin, vmax=vmax, nodata=nodata)
    if domain is None:
        return None
    data, valid, (lo, hi) = domain
    rgba = as_colormap(cmap)(Normalize(vmin=lo, vmax=hi)(np.where(valid, data, lo)))
    rgba[~valid, 3] = 0.0  # NoData → transparent
    return _png_bytes(rgba)


def _classed_png(values: Any, classes: BandClasses) -> Optional[bytes]:
    """Colour a 2-D array class by class to an RGBA PNG, drawing what it has no value for transparent.

    The classed counterpart of :func:`_coloured_png`: each cell takes its class's colour — binned by
    :func:`~digitalearth.base.raster_classes.class_index`, the rule the other tiers bin by — rather than a
    position along a ramp.

    Args:
        values: A 2-D (possibly masked) array of band values, already oriented north-up.
        classes: The band's classes and their colours.

    Returns:
        The PNG bytes, or ``None`` when no cell has a value to colour.

    Examples:
        - Two classes; the NaN cell is transparent:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> from digitalearth.web.raster import _classed_png
            >>> classes = classify_band(np.array([1, 2]), "categorical", None, None)
            >>> _classed_png(np.array([[1.0, 2.0], [np.nan, 1.0]]), classes)[:8]
            b'\\x89PNG\\r\\n\\x1a\\n'

            ```
    """
    import numpy as np
    from matplotlib.colors import to_rgba_array

    index = class_index(values, classes.edges)
    if not (index >= 0).any():
        return None
    table = to_rgba_array(list(classes.colors))
    rgba = table[np.clip(index, 0, None)]
    rgba[index < 0, 3] = 0.0  # NoData → transparent
    return _png_bytes(rgba)


def _written_pyramid(
    directory: pathlib.Path,
    encode: Callable[[int, int, int], Optional[bytes]],
    *,
    bounds: Bounds,
    zooms: Tuple[int, int],
    caller: str,
) -> int:
    """Write a ``{z}/{x}/{y}.png`` pyramid over an extent and return how many tiles it holds.

    Args:
        directory: The pyramid's root, created by the first tile written along with every level it needs — so
            a range that writes nothing creates nothing (review R2-L11).
        encode: Called with ``(zoom, x, y)``; answers the tile's PNG bytes, or ``None`` for a tile the raster
            does not reach — which is not written at all, rather than written blank.
        bounds: The lon/lat extent to cover.
        zooms: ``(lowest, highest)`` zoom levels, inclusive.
        caller: The builder, for the message.

    Returns:
        How many tiles were written.

    Raises:
        ValueError: when the whole range produced nothing. A pyramid with no tile in it, beside a page that
            points into it, is a map that draws nothing with no other symptom — and since the levels are made
            tile by tile, the refusal leaves no directory at all.
    """
    lowest, highest = zooms
    written = 0
    for zoom in range(lowest, highest + 1):
        for x, y in _tile_addresses(bounds, zoom):
            payload = encode(zoom, x, y)
            if payload is None:
                continue
            tile = directory / str(zoom) / str(x) / f"{y}.png"
            tile.parent.mkdir(parents=True, exist_ok=True)
            tile.write_bytes(payload)
            written += 1
    if not written:
        raise ValueError(
            f"{caller}: zooms {lowest}..{highest} over {bounds.as_bbox()} produced no tile with any value "
            "in it, so the page would point at an empty pyramid"
        )
    return written


def _tile_props(
    route: str,
    destination: pathlib.Path,
    *,
    bounds: Bounds,
    zooms: Optional[Tuple[int, int]],
) -> dict:
    """Return the props a tiled layer is described by, all of them writable as JSON.

    The reference is recorded **relative to the destination's own name**, not as the absolute path it was
    written to. A saved page is shared, and an absolute path names the machine that made it — a home
    directory carries a user name — so the page addresses what sits beside it and nothing more. This is why
    a tiled route makes the output a folder: the HTML and the pyramid (or the COG) travel together.

    Args:
        route: The resolved route, a key of :data:`_TILE_ROUTES`.
        destination: Where the tiles or the COG were written.
        bounds: The raster's lon/lat extent, so the source declares where it has data and the map can frame
            itself on a layer whose pixels it never reads.
        zooms: The pyramid's ``(lowest, highest)`` levels, or ``None`` for the COG route, which has none.

    Returns:
        The ``tiles``/``tiles_url``/``tiles_bounds``/``tiles_size``/``tiles_zooms`` props.
    """
    reference = (
        f"{destination.name}/{{z}}/{{x}}/{{y}}.png"
        if route == "xyz"
        else f"cog://{destination.name}"
    )
    return {
        "tiles": route,
        "tiles_url": reference,
        "tiles_bounds": tuple(float(value) for value in bounds.as_bbox()),
        "tiles_size": _TILE_SIZE,
        "tiles_zooms": None if zooms is None else (int(zooms[0]), int(zooms[1])),
    }


def _windowed_raster(web_map: Any, data: Any, *, route: str, caller: str) -> Any:
    """Open what a tiled route was handed, refusing an input it cannot read one window at a time.

    Args:
        web_map: The map, whose :meth:`~digitalearth.web.base.WebMapBase._opened` reads a path, a URL or a
            `DataRef` into the raster it names.
        data: The caller's ``data=``/``dataset=``, as they gave it.
        route: The resolved route, named in the refusal.
        caller: The builder, for the refusal.

    Returns:
        The opened pyramids ``Dataset`` the tiled route reads its tiles out of.

    Raises:
        TypeError: when the input is not a raster pyramids can window — a `Source`, a bare array, a
            `GeoDataFrame`. The inline route takes the first two (they are values already in memory, and it
            encodes the whole band as one image), so this is the one place the two routes' inputs differ, and
            it is refused by name here rather than as ``cannot build a Source from Source`` out of the
            colour-limit scan or ``no attribute 'bbox'`` out of the extent (review R2-L12).
    """
    from pyramids.dataset import Dataset

    dataset = web_map._opened(data)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f'{caller} tiles="{route}" reads its pixels one window at a time through pyramids\' overviews, '
            f"so it needs a raster it can window: a pyramids Dataset, or a path or URL to one; got "
            f"{type(data).__name__}. Drop tiles= to draw the pixels this input already holds"
        )
    return dataset


def _destination(tiles_path: Any, route: str, caller: str) -> pathlib.Path:
    """Return where a tiled route writes.

    Nothing is created here. A refused call must leave the filesystem as it found it, and every other reason
    a tiled route can refuse — the zoom range, an extent that will not express as lon/lat, a range holding no
    value at all — is checked after this one, so making room up front left an empty pyramid root behind beside
    a page that was never written (review R2-L11). The writers make their own room:
    :func:`_written_pyramid` makes each tile's parents, and the COG route makes the file's parent.

    Args:
        tiles_path: The caller's ``tiles_path=``.
        route: The resolved route, named in the message.
        caller: The builder, for the message.

    Returns:
        The destination as a path.

    Raises:
        ValueError: when ``tiles_path`` is ``None``. A route that writes files cannot invent a place to put
            them, and defaulting to the working directory would scatter pyramids wherever a script was run
            from.
    """
    if tiles_path is None:
        raise ValueError(
            f'{caller} tiles="{route}" writes its pixels beside the page, so it needs tiles_path= to say '
            "where — a directory for the tile pyramid, or the .tif to write for a COG. Save the page into "
            "the same folder: the layer addresses what sits next to it"
        )
    return pathlib.Path(tiles_path)


def _tiled_reference(
    route: str,
    dataset: Any,
    *,
    tiles_path: Any,
    zooms: Any,
    encode: Callable[[int, int, int], Optional[bytes]],
    caller: str,
) -> dict:
    """Write a raster's pixels beside the page and return the props that address them.

    The write happens **once**, in the builder. A drawer runs again on every redraw and on every figure read
    back from JSON, so writing there would re-encode a pyramid the caller asked for once — the same reason
    the size check belongs to the builder (review N7).

    Args:
        route: The resolved route, a key of :data:`_TILE_ROUTES`.
        dataset: The opened pyramids ``Dataset``.
        tiles_path: Where to write.
        zooms: The caller's ``zooms=``, used by the ``xyz`` route only.
        encode: One tile's PNG bytes, by ``(zoom, x, y)``; unused by the ``cog`` route, which hands the
            browser the data and lets the page's protocol colour it.
        caller: The builder, for every message this can raise.

    Returns:
        The tile props, as :func:`_tile_props` builds them.

    Raises:
        ValueError: from :func:`_destination`, :func:`_zoom_range` or :func:`_written_pyramid`. None of them
            has created anything when it refuses: the destination is only resolved here, and each writer makes
            its own room (review R2-L11).
    """
    destination = _destination(tiles_path, route, caller)
    bounds = _lonlat_bounds(dataset)
    levels: Optional[Tuple[int, int]] = None
    if route == "xyz":
        levels = _zoom_range(bounds, dataset, zooms, caller)
        _written_pyramid(
            destination, encode, bounds=bounds, zooms=levels, caller=caller
        )
    else:
        # A COG is one file, so the directory it goes in has to exist before pyramids writes it.
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_cog(destination)
    return _tile_props(route, destination, bounds=bounds, zooms=levels)


def _band_nodata(dataset: Any, band: int) -> Any:
    """Return the NoData sentinel of one band, or ``None`` when it has none.

    Args:
        dataset: A pyramids ``Dataset``.
        band: The 1-based band.

    Returns:
        The sentinel, or ``None``. A windowed tile read hands back the sentinel rather than NaN, so without
        it the NoData padding around an edge tile is coloured as data.
    """
    values = getattr(dataset, "no_data_value", None) or ()
    return values[band - 1] if 1 <= band <= len(values) else None


def _styling_source(dataset: Any, band: int) -> Any:
    """Return a metadata-only source the autostyle lookup can read a band's variable from.

    A tiled route must not materialise the band just to style it, and
    :func:`~digitalearth.base.autostyle.auto_style` reads the variable name and the CF attributes, never the
    values — so a 1x1 placeholder carrying the band's name drives it. This is the interactive tier's
    ``_auto_cmap_for_band`` trick, which exists for the same windowed reader (#249).

    Args:
        dataset: A pyramids ``Dataset`` whose ``band_names`` name the variable.
        band: The 1-based band being drawn.

    Returns:
        A :class:`~digitalearth.base.sources.source.Source` holding one placeholder cell and the band's name.
    """
    import numpy as np

    from digitalearth.base.sources import get_source

    names = list(getattr(dataset, "band_names", None) or [])
    variable = names[band - 1] if 1 <= band <= len(names) else ""
    return get_source(np.zeros((1, 1)), metadata={"variable": variable or ""})


def _decimated(dataset: Any, band: int) -> Any:
    """Read one band at :data:`_LIMIT_SCAN_BUDGET` cells, through pyramids' overviews.

    Args:
        dataset: A pyramids ``Dataset``.
        band: The 1-based band.

    Returns:
        A :class:`~digitalearth.base.sources.view.SourceView` of the whole band within the budget — the same
        windowed read the interactive tier's viewport loop drives, asked once for the whole extent. A raster
        already under the budget comes back whole, so the budget only ever *caps* what is materialised.
    """
    from digitalearth.base.sources.view import SourceView
    from digitalearth.base.spec import Selection, ViewRequest

    return SourceView.of(
        dataset,
        selection=Selection.of(band),
        request=ViewRequest(budget=_LIMIT_SCAN_BUDGET),
    )


def _scan_limits(
    dataset: Any, band: int, *, vmin: Optional[float], vmax: Optional[float]
) -> Tuple[float, float]:
    """Resolve the colour limits every tile of a pyramid is coloured on.

    Args:
        dataset: A pyramids ``Dataset``.
        band: The 1-based band.
        vmin: The caller's lower limit, or ``None`` to measure one.
        vmax: The caller's upper limit, or ``None``.

    Returns:
        The ``(lo, hi)`` pair. A limit the caller named is used as it is; a missing one is measured from a
        decimated read rather than from the full band, because reading the band whole is what a tiled route
        exists to avoid. One pair for the whole pyramid, since a tile normalised over its own range disagrees
        with its neighbour across their shared edge.
    """
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    import numpy as np

    values = np.asarray(_decimated(dataset, band).z.values, dtype=float)
    return Scale.from_finite(
        values[np.isfinite(values)], vmin=vmin, vmax=vmax
    ).as_limits()


def _tile_values(dataset: Any, zoom: int, x: int, y: int, *, band: int) -> Any:
    """Read one slippy-map tile of one band, or answer ``None`` for a tile off the raster.

    Args:
        dataset: A pyramids ``Dataset``.
        zoom: The zoom level.
        x: The tile column.
        y: The tile row.
        band: The 1-based band; pyramids' ``read_tile`` counts bands from zero.

    Returns:
        A ``(_TILE_SIZE, _TILE_SIZE)`` array of values, or ``None`` when the tile does not meet the raster —
        an extent's bounding box is not the extent, so a pyramid legitimately has holes.

    Raises:
        Exception: whatever pyramids raised for anything other than a missed tile, unchanged. Swallowing an
            indexing defect here would turn it into a silently absent tile.
    """
    try:
        return dataset.read_tile(zoom, x, y, tilesize=_TILE_SIZE, band=band - 1)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the tile simply missed
        if _off_tile(error):
            return None
        raise


def _tile_stack(
    dataset: Any, zoom: int, x: int, y: int, *, bands: Sequence[int]
) -> Any:
    """Read one slippy-map tile of three bands as the ``(rows, cols, 3)`` stack a composite stretches.

    Args:
        dataset: A pyramids ``Dataset``.
        zoom: The zoom level.
        x: The tile column.
        y: The tile row.
        bands: The three 1-based band numbers, in the order the recipe reads them.

    Returns:
        The channel stack, band-last, with every cell any band has no value for as NaN — the tier's NoData
        contract, which the encoder then draws transparent. ``None`` for a tile that does not meet the raster.

    Raises:
        Exception: whatever pyramids raised for anything other than a missed tile, unchanged.
    """
    import numpy as np

    try:
        read = dataset.read_tile(zoom, x, y, tilesize=_TILE_SIZE)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the tile simply missed
        if _off_tile(error):
            return None
        raise
    values = np.asarray(read, dtype=float)
    if values.ndim == 2:
        # pyramids answers a single-band raster with `(tilesize, tilesize)` and a multiband one with
        # `(bands, tilesize, tilesize)`. Indexed without this, a one-band dataset drawn as three copies of
        # its only band — the smallest composite there is — handed one *row* per channel.
        values = values[np.newaxis, ...]
    sentinels = getattr(dataset, "no_data_value", None) or ()
    channels = []
    for band in bands:
        channel = values[band - 1]
        sentinel = sentinels[band - 1] if 1 <= band <= len(sentinels) else None
        if sentinel is not None and np.isfinite(sentinel):
            channel = np.where(channel == float(sentinel), np.nan, channel)
        channels.append(channel)
    return np.stack(channels, axis=-1)


def _composite_limits(dataset: Any, bands: Sequence[int]) -> Any:
    """Measure one set of per-channel stretch bounds for a whole pyramid.

    Args:
        dataset: A pyramids ``Dataset``.
        bands: The three 1-based band numbers.

    Returns:
        One ``(lo, hi)`` pair per channel, from a decimated read of each band — not from the full stack,
        which is the cost a tiled route exists to avoid. Frozen for every tile, because a tile stretched over
        its own percentiles disagrees with its neighbour across their shared edge, exactly as an unfrozen
        animation pulses frame to frame (:func:`~digitalearth.base.stretch.channel_limits`).
    """
    import numpy as np

    from digitalearth.base.stretch import channel_limits

    scans = [
        np.asarray(_decimated(dataset, band).z.values, dtype=float) for band in bands
    ]
    return channel_limits(np.stack(scans, axis=-1))


def _composite_png(unit_stack: Any) -> Optional[bytes]:
    """Encode a stretched ``(rows, cols, 3)`` stack as PNG bytes.

    Args:
        unit_stack: Channel values already stretched to ``[0, 1]`` and read as red, green and blue by here
            (an HSV composite has been converted — see :func:`_composed_rgb`); NaN marks NoData.

    Returns:
        The PNG bytes, or ``None`` when no pixel is finite in all three channels — nothing to draw, which for
        one tile of a pyramid is an ordinary hole rather than a failure.
    """
    import numpy as np

    stack = np.asarray(unit_stack, dtype=float)
    valid = np.isfinite(stack).all(axis=-1)
    if not valid.any():
        return None
    rgba = np.zeros(stack.shape[:2] + (4,), dtype=float)
    rgba[..., :3] = np.clip(np.where(np.isfinite(stack), stack, 0.0), 0.0, 1.0)
    rgba[..., 3] = valid.astype(float)  # NoData in any channel → transparent
    return _png_bytes(rgba)


def _limit_pairs(limits: Any) -> Optional[List[Tuple[Any, ...]]]:
    """Return per-channel limits as a list of pairs, or `None` when they are not shaped like any.

    The one guard both limit translators need, and the one each had its own copy of. Neither of them is the
    place a malformed ``limits=`` is diagnosed: :func:`~digitalearth.base.stretch.stretch_to_unit` owns that
    message and names the argument, so a value this cannot read is handed back to the caller untouched to
    reach it — rather than failing here, on iterating a float or unpacking a triple, as a `TypeError` about a
    private helper nobody called.

    Args:
        limits: The caller's `limits=`, already known not to be `None`.

    Returns:
        One tuple per channel when `limits` is an iterable of two-element pairs; `None` for anything else,
        including a bare number and a sequence of triples.
    """
    try:
        pairs = [tuple(pair) for pair in limits]
    except TypeError:
        return None
    if any(len(pair) != 2 for pair in pairs):
        return None
    return pairs


def _recorded_limits(limits: Any) -> Any:
    """Return per-channel stretch limits in the spelling a figure can be written in.

    :func:`~digitalearth.base.stretch.channel_limits` answers `(nan, nan)` for a channel it could not
    measure — a normal result, and the one a caller freezing a series on its first frame passes to every
    later frame. NaN has no JSON form, so a description holding one cannot be written at all: `to_dict`
    refuses the whole figure. A bound the freeze does not carry is recorded as `None` instead, which says
    the same thing, and :func:`_stretch_limits` reads it back as the NaN the stretch expects.

    Args:
        limits: The caller's `limits=`, or `None`.

    Returns:
        The limits with every non-finite bound as `None` and every finite one as a plain `float`; `None`
        unchanged; and anything not shaped as a sequence of pairs unchanged, so that
        :func:`~digitalearth.base.stretch.stretch_to_unit` refuses it in its own words rather than this
        failing on it first.
    """
    if limits is None:
        return None
    pairs = _limit_pairs(limits)
    if pairs is None:
        return limits
    try:
        return tuple(
            tuple(
                None
                if value is None or not math.isfinite(float(value))
                else float(value)
                for value in pair
            )
            for pair in pairs
        )
    except (TypeError, ValueError):
        return limits


def _stretch_limits(limits: Any) -> Any:
    """Return recorded limits as :func:`~digitalearth.base.stretch.stretch_to_unit` takes them.

    Args:
        limits: The recorded `limits` prop.

    Returns:
        The limits with every `None` bound back as `nan`, which is how the stretch spells "no frozen bound
        for this channel"; anything else unchanged.
    """
    if limits is None:
        return None
    pairs = _limit_pairs(limits)
    if pairs is None:
        return limits
    return [
        tuple(float("nan") if value is None else value for value in pair)
        for pair in pairs
    ]


def _composed_rgb(via: str, unit_stack: Any) -> Any:
    """Return three stretched channels as the RGB the image source is encoded from.

    The one step the two composites do not share. Everything up to here — one warp, one band stack, one
    per-channel stretch into ``[0, 1]`` — is :mod:`digitalearth.base.stretch`'s shared rule for both; what
    differs is only what those three numbers per cell *mean*.

    Args:
        via: The layer's recorded recipe, one of :data:`_COMPOSITE_RECIPES`.
        unit_stack: The ``(rows, cols, 3)`` stretched stack; NaN marks a cell a band has no value for.

    Returns:
        The stack unchanged for an RGB composite. For an HSV one, the same three channels read as hue,
        saturation and value and converted to RGB — with every cell that was incomplete left NaN, so the
        encoder still draws it transparent.
    """
    if via != _HSV_RECIPE:
        return unit_stack
    import numpy as np
    from matplotlib.colors import hsv_to_rgb

    stack = np.asarray(unit_stack, dtype=float)
    complete = np.isfinite(stack).all(axis=-1)[..., None]
    # Masked before the conversion and restored after it, rather than handed over as it is. `hsv_to_rgb` reads
    # the hue with `(h * 6.0).astype(int)`, and casting NaN yields a garbage index rather than NaN, so a cell
    # with no hue comes back *finite* — measured: `(0.0, 0.0, 0.0)` at full saturation and value,
    # `(0.5, 1.04e-311, 1.04e-311)` at half — and therefore opaque, with only a RuntimeWarning about an
    # invalid cast to say anything had happened. The tier's NoData contract (a cell missing any band is
    # transparent) has to survive the colour space.
    return np.where(complete, hsv_to_rgb(np.where(complete, stack, 0.0)), np.nan)


def _placed_corners(web_map: Any, source: Any, caller: str) -> Any:
    """Return a raster's lon/lat corners and frame the map on them, or report that it cannot be placed.

    Computed here rather than in the builder so a source's corners are read **once**: the builder records
    what was asked for, and everything derived from the data is derived where the data is read.

    Args:
        web_map: The map being drawn.
        source: The display source whose corners are wanted.
        caller: The builder's name, for the skip message.

    Returns:
        The four lon/lat corners, or `None` when they cannot be expressed — reported through the map's own
        skip log, so an unplaceable layer is refused rather than drawn somewhere wrong.

    Raises:
        OffLimbError: when the map is `strict`, which is what that flag asks for: a pipeline that must not
            publish a half-drawn map gets the refusal back instead of a warning.
    """
    corners = web_map._lonlat_corners(source)
    if corners is None:
        web_map._skipped(
            caller,
            "the raster's corners cannot be expressed in lon/lat, which a MapLibre image source "
            "needs; reproject the dataset so its extent is representable",
        )
        return None
    # Already lon/lat, so the framing takes them as they are.
    web_map._note_lonlat_bounds(
        (corners[0][0], corners[2][1], corners[1][0], corners[0][1])
    )
    return corners


def _image_layer(_web_map: Any, layer: LayerSpec, url: str, coordinates: Any) -> Any:
    """Package an image source and the raster layer reading it.

    Both raster kinds draw the same way — a ``data:`` PNG placed on four lon/lat corners — and differ only
    in how the image is made, so the MapLibre half is written once.

    Args:
        _web_map: Unused — taken so this reads with the same first argument as the two drawers that call
            it, which do need the map.
        layer: The layer's description. Its `opacity` prop and its visibility are what MapLibre is given;
            everything else about the image is already in `url`.
        url: The ``data:image/png;base64,`` URI to place.
        coordinates: The four lon/lat corners, clockwise from the north-west.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`.

    Raises:
        ValueError: when the description records no `opacity` for the raster, naming the layer, its kind
            and what is missing.
    """
    return _raster_layer(
        layer, {"type": "image", "url": url, "coordinates": coordinates}
    )


def _raster_layer(layer: LayerSpec, source_spec: dict) -> Any:
    """Package a MapLibre raster layer over whichever source spec holds its pixels.

    Every raster this tier draws is the same MapLibre layer — a ``raster`` type reading one source at the
    described opacity — and differs only in how that source addresses the pixels: inline as a ``data:`` image,
    as a tile-URL template, or as a file a client-side protocol reads windows from. The layer half is
    therefore written once, so a route cannot lose the opacity or the visibility the description records.

    Args:
        layer: The layer's description. Its ``opacity`` prop and its visibility are what MapLibre is given.
        source_spec: The source, in the shape the widget's ``add_source`` takes it.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`.

    Raises:
        ValueError: when the description records no ``opacity`` for the raster, naming the layer, its kind
            and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, layer_types = _require_layer_api()
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec=source_spec,
        layer=layer_cls(
            id=layer.id,
            type=layer_types.RASTER,
            source=source_id,
            paint={
                "raster-opacity": float(required_props(layer, "opacity")["opacity"])
            },
            layout={"visibility": "visible" if layer.visible else "none"},
        ),
    )


def _tiled_layer(web_map: Any, layer: LayerSpec, props: dict) -> Any:
    """Point a MapLibre raster source at the pyramid or the COG the builder already wrote.

    The writing is the **builder's**, once, and only the reference is described — so a figure redrawn from its
    description, or read back from JSON on another machine, re-uses the tiles rather than writing them again.

    Args:
        web_map: The map being drawn, told the layer's extent so it can frame itself on pixels it never reads.
        layer: The layer's description.
        props: Its recorded props, already thawed.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`.

    Raises:
        ValueError: when the description names a route this tier does not draw — the same refusal the builder
            gives, because a figure written by another version is the other way in.
    """
    route = _tile_route(props.get("tiles"), f"layer {layer.id!r}")
    spec: dict = {
        "type": "raster",
        "tileSize": int(props.get("tiles_size", _TILE_SIZE)),
    }
    reference = str(props["tiles_url"])
    spec[_TILE_ROUTES[str(route)]] = [reference] if route == "xyz" else reference
    zooms = props.get("tiles_zooms")
    if zooms is not None:
        spec["minzoom"], spec["maxzoom"] = int(zooms[0]), int(zooms[1])
    bounds = props.get("tiles_bounds")
    if bounds is not None:
        spec["bounds"] = [float(value) for value in bounds]
        web_map._note_lonlat_bounds(tuple(float(value) for value in bounds))
    return _raster_layer(layer, spec)


def draw_field(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Encode one band as a coloured image and place it on the map.

    A layer whose colour encoding records classes — graduated edges or categorical codes, written by this
    tier's :meth:`RasterMixin.field` or by another tier's — is coloured class by class (:func:`_classed_png`);
    a continuous or absent scale is coloured along the recorded ``cmap`` between ``vmin`` and ``vmax``. When
    the map has no key filed for the layer yet — a layer drawn from a figure rather than built here — one is
    filed from that colour scale (:meth:`RasterMixin._band_key`), so ``legend()`` can describe it.

    Args:
        web_map: The map being drawn.
        data: The layer's source — a pyramids dataset or an array.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`, or `None` when the band cannot be placed — it
        lies outside what the display CRS can show, or its corners will not express as lon/lat. Either way
        it has been reported through the map's skip log before this answers.

    Raises:
        ValueError: when the description records none of the values the image is encoded from — its
            band, its colour map or its colour limits — naming the layer, its kind and what is missing.
        OffLimbError: when the map is `strict` and the band cannot be placed, in place of the `None`.
    """
    import numpy as np

    from digitalearth.web.renderer import required_props

    props = required_props(layer, "band", "cmap", "vmin", "vmax", "opacity")
    if props.get("tiles") is not None:
        return _tiled_layer(web_map, layer, props)
    source = web_map._display_source_or_skip(data, band=props["band"], layer="field")
    if source is None:
        return None
    values = source.z.values
    y = np.asarray(source.y.values, dtype=float)
    if y.size > 1 and y[0] < y[-1]:
        # Ascending y → flip so PNG row 0 is the northern edge.
        values = values[::-1]
    scale = getattr(dict(layer.symbology.encodings).get("color"), "scale", None)
    classes = classes_of(scale, props["cmap"])
    url = web_map._rgba_png_datauri(
        values, props["cmap"], vmin=props["vmin"], vmax=props["vmax"], classes=classes
    )
    coordinates = _placed_corners(web_map, source, "field")
    if coordinates is None:
        return None
    if layer.id not in web_map._legends and scale is not None:
        # A layer this map's own `field()` built had its key filed as it was indexed. One drawn from a figure —
        # another tier's, or this tier's read back — arrives with only its colour encoding, so the key is
        # filed from that here, or `legend()` would find nothing to describe on a map with a coloured band.
        web_map._legends[layer.id] = web_map._band_key(
            str(layer.symbology.encoding("color").field), scale, props["cmap"], classes
        )
    return _image_layer(web_map, layer, url, coordinates)


def _drawn_recipe(layer: LayerSpec, props: dict) -> str:
    """Return which composite a description asks for, refusing one this tier does not draw.

    Args:
        layer: The layer being drawn, named in the refusal.
        props: Its recorded props, already thawed.

    Returns:
        The recorded recipe, or :data:`_RGB_RECIPE` when the description records none — which is every
        ``rgb`` layer written before this tier had a second composite.

    Raises:
        ValueError: when the recipe is one this tier has no composite for. Falling back to the default
            instead would draw a figure written by another version as something it does not claim to be;
            the static renderer refuses an unknown recipe the same way.
    """
    via = props.get("via", _RGB_RECIPE)
    if via not in _COMPOSITE_RECIPES:
        raise ValueError(
            f"layer {layer.id!r} records {via!r} as how its composite was drawn; the web tier draws one of "
            f"{sorted(_COMPOSITE_RECIPES)}"
        )
    return str(via)


def draw_rgb_composite(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Encode three bands as one colour composite image and place it on the map.

    Both composites are drawn here — the recipe is the layer's own, recorded under ``via`` — because they
    differ in one step out of six: the warp, the band stack, the stretch, the north-up flip, the PNG and the
    placement are the same work either way. It keeps the name it is registered under in
    :func:`~digitalearth.web.renderer.drawer_for`.

    Args:
        web_map: The map being drawn.
        data: The layer's source — the dataset the bands are read from, in whatever CRS it was recorded
            in. It is warped to the display CRS here, so a figure handed straight to the renderer draws
            the same image as the builder call that recorded it.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`, or `None` when the composite cannot be placed —
        it lies outside what the display CRS can show, or its corners will not express as lon/lat. Either
        way it has been reported through the map's skip log before this answers.

    Raises:
        ValueError: when the description records none of the values the composite is built from — its
            three bands, its stretch limits or whether NoData is masked — naming the layer, its kind and
            what is missing; when it records a recipe this tier has no composite for; when it records a band
            count other than three, in :func:`~digitalearth.base.stretch.require_three_bands`' words; when
            the recorded limits do not hold one `(lo, hi)` pair per band; or when no pixel is finite in all
            three, since there is nothing to draw and an empty image would read as a rendering failure
            rather than as an empty input.
        OffLimbError: when the map is `strict` and the composite cannot be placed, in place of the `None`.
    """
    import numpy as np

    from digitalearth.base.sources import get_stack
    from digitalearth.web.renderer import required_props

    props = required_props(layer, "bands", "mask_nodata", "limits", "opacity")
    if props.get("tiles") is not None:
        return _tiled_layer(web_map, layer, props)
    via = _drawn_recipe(layer, props)
    bands = list(props["bands"])
    # The same guard the builder runs, because this is the other way in: a figure is drawn from its
    # description — by a redraw onto another view, or from JSON written elsewhere — with no builder in front
    # of it. Without it the count reaches `get_stack`, which answers with numpy's "could not broadcast input
    # array from shape (4,9,2) into shape (4,9,3)": a complaint about an array the caller never named.
    require_three_bands(via, bands)
    # One warp for both halves: the pixels are read from the warped dataset and the corners are taken from
    # that same grid. Stacking `data` itself drew the source-CRS grid stretched over the warped grid's
    # extent — a different shape at a different resolution (review H1).
    data = web_map._display_raster_or_skip(data, layer=via)
    if data is None:
        return None
    stack = get_stack(data, bands, mask=props["mask_nodata"])
    source = web_map._to_display_source(data, band=bands[0])
    y = np.asarray(source.y.values, dtype=float)
    if y.size > 1 and y[0] < y[-1]:
        # Ascending y → flip so PNG row 0 is the north edge.
        stack = stack[::-1]
    from digitalearth.base.stretch import stretch_to_unit

    url = web_map._composite_png_datauri(
        _composed_rgb(via, stretch_to_unit(stack, _stretch_limits(props["limits"]))),
        caller=via,
    )
    coordinates = _placed_corners(web_map, source, via)
    if coordinates is None:
        return None
    return _image_layer(web_map, layer, url, coordinates)


class RasterMixin(_MixinBase):
    """Raster builder for :class:`~digitalearth.web.map.WebMap` (image-source path)."""

    def _band_colour(
        self,
        data: Any,
        *,
        band: int,
        cmap: str,
        limits: Optional[Tuple[float, float]],
        classes: Optional[BandClasses] = None,
    ) -> dict:
        """Describe what a band's colour varies with, and record the key that would explain it.

        The raster half of order 24. A colour key is a ``Guide`` on the layer's ``color``
        :class:`~digitalearth.base.spec.encoding.Encoding`, so a layer that publishes none cannot be keyed
        at all — which is what left a one-call raster map unkeyable on **this** tier and on no other, while
        the static, interactive and 3-D tiers all keyed the same raster.

        A raster is not classified by a column, so there is no column to bind to; the band is what its
        colour varies with, and the span its cells are coloured across is the scale. Both are things the
        tier already knows at build time, so neither is invented.

        Args:
            data: The raster the layer draws, for the band's name.
            band: The 1-based band being coloured.
            cmap: The colormap already resolved for it, sampled for the key's ramp.
            limits: The ``(lo, hi)`` its cells are coloured between, or ``None`` when the band has none —
                nothing finite to colour, which the draw refuses on its own terms.
            classes: The band's classes when ``scheme=`` classified it, or ``None`` for a ramp. A classified
                band's scale is its classes' and its key one swatch per class.

        Returns:
            The ``{"color": Encoding}`` mapping for the layer's symbology, or ``{}`` when `limits` is
            ``None`` and the band is not classified (`classes` is ``None``). Empty rather than a scale-less
            encoding: a colour key drawn from a span nobody knows would be an invented one, and this tier
            would rather publish nothing than that.

        Note:
            Also sets :attr:`~digitalearth.web.base.WebMapBase.last_breaks` and
            :attr:`~digitalearth.web.base.WebMapBase.last_legend`, which is how the key's **content** —
            the swatches and the colours they were drawn with — reaches the layer:
            :meth:`~digitalearth.web.base.WebMapBase._index_layer` files a fresh `last_legend` under the
            layer being built, exactly as it does for a classifying vector builder. The two are set
            together, as every other builder sets them (review L7).

            The units go on the key rather than into the encoding, the way `contours` puts them there:
            ``last_units`` is only set when the band said what it is measured in, so a heading gains
            ``(mm)`` only when that is the band's own word for it and never a guess.
        """
        if classes is not None:
            scale = classes.scale
        elif limits is not None:
            scale = Scale.from_limits(*limits)
        else:
            return {}
        column = _band_name(data, band)
        self.last_legend = self._band_key(column, scale, cmap, classes)
        # The values the key was built on, as every builder publishes them: the ramp's stops, a graduated
        # band's class edges, a categorical band's codes.
        self.last_breaks = list(self.last_legend["values"])
        if self.last_units:
            self.last_legend["units"] = self.last_units
        return {"color": Encoding.by_field("color", column, scale=scale)}

    def _band_key(
        self, column: str, scale: Scale, cmap: Any, classes: Optional[BandClasses]
    ) -> dict:
        """Build the key a coloured band is explained by — a ramp, one swatch per class, or one per code.

        Written once for the two ways a band reaches this tier: the builder, which files it as the layer is
        indexed, and the drawer, which files it for a band drawn from a figure, carrying only its colour
        encoding. One body, so a band keyed either way reads the same.

        Args:
            column: The band's name, the key's heading.
            scale: The band's colour scale.
            cmap: The colormap a ramp's stops are sampled from.
            classes: The band's classes, or ``None`` for a ramp.

        Returns:
            The legend dict this tier stores per layer (:meth:`~digitalearth.web.base.WebMapBase._legend_dict`):
            its ``values`` are the ramp's stops, a graduated band's class edges, or a categorical band's codes.

        Examples:
            - A graduated band is keyed one swatch per class, valued at its edges:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.raster_classes import classify_band
                >>> from digitalearth.web import WebMap
                >>> classes = classify_band(np.arange(10.0), "equal_interval", 3, "viridis")
                >>> key = WebMap()._band_key("dem", classes.scale, "viridis", classes)
                >>> key["kind"], key["values"], key["colors"]
                ('graduated', [0.0, 3.0, 6.0, 9.0], ['#440154', '#21918c', '#fde725'])

                ```
            - A categorical band is keyed one swatch per code:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.raster_classes import classify_band
                >>> from digitalearth.web import WebMap
                >>> classes = classify_band(np.array([1, 3, 3]), "categorical", None, None)
                >>> key = WebMap()._band_key("landcover", classes.scale, None, classes)
                >>> key["kind"], key["values"], key["colors"]
                ('categorical', [1, 3], ['#1f77b4', '#ff7f0e'])

                ```
            - With no classes the key is a ramp sampled at the shared number of stops:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> from digitalearth.web import WebMap
                >>> key = WebMap()._band_key("dem", Scale.from_limits(0.0, 4.0), "viridis", None)
                >>> key["kind"], key["values"]
                ('continuous', [0.0, 1.0, 2.0, 3.0, 4.0])

                ```
        """
        import numpy as np

        if classes is not None and classes.is_categorical:
            return self._legend_dict(LegendSpec.from_scale(scale, title=column), column)
        if classes is not None:
            edges = list(classes.edges)
            return self._legend_dict(
                LegendSpec.from_scale(scale, colors=list(classes.colors), title=column),
                column,
                values=edges,
            )
        # `DEFAULT_RAMP_STOPS`, not a number of this module's own: the vector ramp
        # (:meth:`~digitalearth.web.vector.VectorMixin._ramp_color_expr`) samples at the same count, so a
        # continuous key reads the same whether the values came from a column or from a band — and that was
        # a coincidence between two spellings of `5` until both read the constant `LegendSpec` itself
        # defaults to (review N6).
        lo, hi = scale.as_limits()
        stops = [float(stop) for stop in np.linspace(lo, hi, DEFAULT_RAMP_STOPS)]
        return self._legend_dict(
            LegendSpec.from_scale(
                scale,
                colors=list(self._cmap_hex(cmap, len(stops))),
                title=column,
                values=stops,
            ),
            column,
        )

    def field(
        self,
        data: Any,
        *,
        band: int = DEFAULT_BAND,
        cmap: Any = None,
        units: Optional[str] = None,
        opacity: Maybe[float] = UNSET,
        limits: Optional[Sequence[float]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        scheme: Optional[Any] = None,
        k: Optional[int] = None,
        visible: bool = True,
        name: Optional[str] = None,
        tiles: Optional[str] = None,
        tiles_path: Any = None,
        zooms: Any = None,
    ) -> Self:
        """Overlay a pyramids raster band as a colour-mapped MapLibre image source (recipe W1).

        The band is reprojected to the display CRS (lon/lat) through pyramids, normalised over its finite
        range (or the explicit ``limits``/``vmin``/``vmax``), colour-mapped with ``cmap`` (autostyle default when
        ``None``), and embedded as an RGBA PNG data-URI placed by its lon/lat corners. Masked / non-finite
        cells become fully transparent.

        Args:
            data: A pyramids ``Dataset`` (or anything ``get_source`` accepts).
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            band: 1-based band to draw.
            cmap: A registered matplotlib colormap name, or a ``Colormap`` itself — the classified
                builders take either and so does this one (#315, review H3). A colormap built on the spot
                colours the image but has no name a figure read elsewhere could resolve. ``None`` resolves
                the autostyle default for the variable.
            limits: The contract's name for the colour limits, as ``(vmin, vmax)`` — the one spelling every
                tier answers to (#299). ``vmin``/``vmax`` remain, and naming both is refused rather than
                silently resolved one way.
            units: What the band's values are measured in, recorded as
                :attr:`~digitalearth.web.base.WebMapBase.last_units` so a key built from them can say so.
                ``None`` (the default) takes the variable's units from
                :func:`~digitalearth.base.autostyle.auto_style`, and leaves them unknown when it carries
                none — a unit is never guessed. Pass one to correct a band the library mis-identifies, or
                to name the units of a variable it does not know.
            opacity: Raster layer opacity in ``[0, 1]``.
            vmin: Lower colour limit; ``None`` uses the band's finite minimum.
            vmax: Upper colour limit; ``None`` uses the band's finite maximum.
            scheme: Colour the band class by class rather than along a ramp. A named graduated scheme
                (``"quantiles"``, ``"equal_interval"``, ``"fisher_jenks"``, …) or an explicit list of edges cuts
                classes coloured from ``cmap``; ``"categorical"`` gives each integer code a class of its own,
                coloured from the shared categorical palette (``cmap`` picks another). The classes are cut by
                :mod:`digitalearth.base.raster_classes`, the classifier the static and interactive tiers use,
                so one band is classed — and coloured — alike on all three; they are recorded in the layer's
                colour encoding, and :meth:`~digitalearth.web.decoration.DecorationMixin.legend` keys them
                one swatch per class. ``limits``/``vmin``/``vmax`` do not apply to a classified band. ``None``
                (default) draws the continuous ramp. Not taken by a ``tiles`` route.
            k: How many classes a named graduated scheme cuts; ``None`` takes 5. Ignored without ``scheme``.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible. ``False`` builds it hidden, which is how
                :meth:`~digitalearth.web.temporal.TemporalMixin.timeslider` stacks time steps without
                every frame showing at once — including in a saved page, which carries no slider.
            tiles: How the pixels reach the page. ``None`` (the default) inlines them as a ``data:`` PNG, so
                the page is one self-contained file — and is refused above :data:`_LARGE_RASTER_PIXELS`
                cells, because that page would be hundreds of megabytes (#189). ``"xyz"`` writes a
                ``{z}/{x}/{y}.png`` pyramid under ``tiles_path`` and points a MapLibre ``raster`` source at
                it, with **no client-side dependency** — it is the route that works as written. ``"cog"``
                writes one Cloud-Optimized GeoTIFF there instead and addresses it as ``cog://<file>``: fewer,
                larger files, read by range request, but MapLibre has no built-in reader for one, so the page
                showing it must register a ``cog://`` protocol itself. This tier cannot put that script in a
                page it writes — :meth:`~digitalearth.web.base.WebMapBase.save` has no hook for one — so take
                the COG route when you control the page the map is embedded in, and ``"xyz"`` when
                :meth:`~digitalearth.web.base.WebMapBase.save` is the whole pipeline. **Either route makes
                the output a folder** — the HTML plus the pyramid or the COG — rather than a single file, and
                the page addresses what sits beside it, so save it into the same directory.
            tiles_path: Where a tiled route writes: the pyramid's root directory for ``"xyz"``, the ``.tif``
                to write for ``"cog"``. Required by both, and ignored by the inline default.
            zooms: ``(lowest, highest)`` zoom levels for the ``"xyz"`` pyramid. ``None`` derives
                ``(0, native)``, where *native* is the zoom at which a tile pixel is about the size of a
                source cell — beyond it a pyramid invents detail the raster does not hold.

        Returns:
            This map (chainable). When the band cannot be placed — it lies outside what the display CRS
            can show, or its corners will not express as lon/lat — nothing is added: the layer is skipped
            with a warning, or the error is raised when the map was built with ``strict=True``.

        Raises:
            ValueError: when `limits` is given alongside `vmin`/`vmax` — they name the same thing — or is
                not a `(vmin, vmax)` pair of numbers; when ``opacity`` is not a finite number, refused
                at this call because a figure holding NaN or infinity could not be written down; when the
                band is too large to inline and no ``tiles`` route was named; and, for a tiled route, when
                ``tiles`` names no route this tier writes, ``tiles_path`` is missing, ``zooms`` is not an
                ordered pair of levels, or the range produced no tile with any value in it; when ``scheme``
                is given with a ``tiles`` route, or cannot classify the band — an unknown name, ``k`` below
                one, or, under ``"categorical"``, a band holding non-integer values, more than
                :data:`~digitalearth.base.raster_classes.MAX_RASTER_CATEGORIES` codes, or no valid cell.
            KeyError: when `cmap` names no registered colormap, or `dataset` is a URL with no resolver.
            TypeError: when `cmap` is neither a name, a ``Colormap``, nor a sequence of colours; and, for a
                tiled route, when `data` is not a raster pyramids can read windows out of — a `Source` or a
                bare array is drawn by the inline route, which encodes the whole band at once, but a pyramid
                is written one window at a time and needs a ``Dataset``, or a path or URL to one.
            FileNotFoundError: when `data` is a path that names nothing.
            OffLimbError: only when the map was built with ``strict=True`` and the band cannot be
                placed; by default that layer is skipped with a warning instead, so one unplaceable
                raster does not cost the map the layers around it.

        Examples:
            - Colour-map a band and address the layer afterwards by the name it was given (needs
              the ``web`` extra, so the block is skipped without it):
                ```python
                >>> import numpy as np                               # doctest: +SKIP
                >>> from digitalearth.base.sources import get_source  # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> src = get_source(                                # doctest: +SKIP
                ...     np.arange(12.0).reshape(3, 4),
                ...     x=np.array([0.0, 1.0, 2.0, 3.0]),
                ...     y=np.array([2.0, 1.0, 0.0]),
                ... )
                >>> m = WebMap().field(src, cmap="viridis", name="dem")       # doctest: +SKIP
                >>> m.layer_ids, len(m.layers)                       # doctest: +SKIP
                (['dem'], 1)

                ```
            - The band also hands the map its extent, so the view frames itself and
              :meth:`~digitalearth.web.base.WebMapBase.set_bounds` has something to frame on;
              on an empty map the same call raises instead:
                ```python
                >>> m.set_bounds() is m                              # doctest: +SKIP
                True

                ```
            - ``visible=False`` builds the layer hidden, which is how
              :meth:`~digitalearth.web.temporal.TemporalMixin.timeslider` stacks one layer per time
              step without every frame showing at once — in a saved page too, which has no slider:
                ```python
                >>> m = WebMap().field(src, visible=False, name="t0")       # doctest: +SKIP
                >>> m.layer_ids                                      # doctest: +SKIP
                ['t0']

                ```
            - Classify the band into three equal-interval classes; the key is one swatch per class:
                ```python
                >>> import numpy as np                               # doctest: +SKIP
                >>> from digitalearth.base.sources import get_source  # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> src = get_source(                                # doctest: +SKIP
                ...     np.arange(10.0).reshape(2, 5), x=np.arange(5.0), y=np.array([1.0, 0.0])
                ... )
                >>> m = WebMap().field(src, scheme="equal_interval", k=3)  # doctest: +SKIP
                >>> m.last_breaks, m.last_legend["colors"]           # doctest: +SKIP
                ([0.0, 3.0, 6.0, 9.0], ['#440154', '#21918c', '#fde725'])

                ```
            - Draw a band of class codes as categories, keyed one swatch per code:
                ```python
                >>> import numpy as np                               # doctest: +SKIP
                >>> from digitalearth.base.sources import get_source  # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> src = get_source(                                # doctest: +SKIP
                ...     np.array([[1.0, 1.0, 2.0], [3.0, 3.0, 2.0]]),
                ...     x=np.array([0.0, 1.0, 2.0]),
                ...     y=np.array([1.0, 0.0]),
                ... )
                >>> m = WebMap().field(src, scheme="categorical")    # doctest: +SKIP
                >>> m.last_legend["kind"], m.last_breaks             # doctest: +SKIP
                ('categorical', [1, 2, 3])

                ```

        See Also:
            digitalearth.web.raster.RasterMixin.rgb_composite: the three-band composite path.
            digitalearth.web.vector.VectorMixin.contours: draws the same field as vectors.
        """
        vmin, vmax = _colour_limits(limits, vmin, vmax, caller=_FIELD_CALLER)
        ask = Ask()
        opacity = as_finite(
            ask("opacity", opacity, RASTER_OPACITY), "opacity", _FIELD_CALLER
        )
        route = _tile_route(tiles, _FIELD_CALLER)
        if scheme is not None and route is not None:
            # A tile is coloured from pyramids' overviews, which average neighbouring cells — inventing codes
            # no cell holds — and the route never reads the band whole, which a graduated scheme's edges need.
            raise ValueError(
                f"{_FIELD_CALLER} classifies a band with scheme= only on the inline route; a tiles= route "
                "colours each tile from averaged overviews and never reads the whole band to cut classes from"
            )
        _require_layer_api()
        if route is not None:
            return self._tiled_field(
                route,
                data,
                band=band,
                cmap=cmap,
                units=units,
                opacity=opacity,
                asked=ask.named,
                vmin=vmin,
                vmax=vmax,
                visible=visible,
                name=name,
                tiles_path=tiles_path,
                zooms=zooms,
            )
        source = self._display_source_or_skip(data, band=band, layer="field")
        if source is None:
            return self
        # Before the colour map and the units, so a refused call leaves nothing of itself behind: those two
        # write `last_units`, which a later `colorbar()` would then label with a layer that was never built.
        # Named by the constant every other refusal this builder gives is named by, so the ceiling does not
        # open with `field:` where its siblings open with `WebMap.field()` (review R2-N6).
        _refuse_if_large(
            _FIELD_CALLER, "band", int(getattr(source.z.values, "size", 0))
        )
        cmap_name = self._auto_cmap(source, cmap)
        # Carried for a key built from this band's values (see `_auto_units`); `None` when unknown.
        self.last_units = self._auto_units(source, units)
        # Resolved here rather than left wholly to the draw, because order 24 hangs the colour key on the
        # layer's **description** and a description is written before anything is drawn. Through
        # `_colour_domain`, which is also what `_coloured_png` colours the image through, so the key labels
        # the ramp that is drawn rather than a second reading of the same band. The limits stay out of
        # `props`: what a figure records is what the caller *asked* for (#334), and this tier's own
        # resolution of an unset `vmin`/`vmax` is its business — the tiled route records its pair only
        # because one span has to hold a whole pyramid together.
        domain = _colour_domain(source.z.values, vmin=vmin, vmax=vmax)
        classes: Optional[BandClasses] = None
        if scheme is not None:
            # Codes take the categorical palette unless the caller names one; graduated classes the colormap
            # the band resolved, the one a ramp would have used.
            palette = cmap if asks_categorical(scheme) else cmap_name
            classes = classify_band(source.z.values, scheme, k, palette)
        layer_id = self._layer_id("raster", name)
        # The colour map is resolved here because `_auto_cmap` reads the band's own metadata, which is the
        # caller's request as much as `cmap=` is. The image — orientation included — is encoded by
        # `draw_field`, which is handed the band already warped above rather than warping the recorded
        # `data` a second time (review M8).
        if self._index_layer(
            layer_id,
            name,
            kind="raster",
            visible=visible,
            source=data,
            placed=source,
            symbology=Symbology(
                encodings=self._band_colour(
                    data,
                    band=band,
                    cmap=cmap_name,
                    limits=None if domain is None else domain[2],
                    classes=classes,
                ),
                props={
                    "cmap": cmap_name,
                    "vmin": vmin,
                    "vmax": vmax,
                    "opacity": float(opacity),
                    "band": band,
                    "scheme": (
                        scheme
                        if scheme is None or isinstance(scheme, str)
                        else list(scheme)
                    ),
                    "k": k,
                    **ask.record,
                },
            ),
        ):
            self._last_layer_id = layer_id
        return self

    def _tiled_field(
        self,
        route: str,
        data: Any,
        *,
        band: int,
        cmap: Any,
        units: Optional[str],
        opacity: float,
        vmin: Optional[float],
        vmax: Optional[float],
        visible: bool,
        name: Optional[str],
        tiles_path: Any,
        zooms: Any,
        asked: Sequence[str] = (),
    ) -> Self:
        """Write one band's pixels beside the page and describe the layer that reads them (#189).

        Nothing here reads the band whole, which is the point: the colour map comes from the band's *name*,
        the colour limits from a decimated read, the extent from a corner transform, and the pixels one tile
        at a time out of pyramids' overviews.

        Args:
            route: The resolved route, a key of :data:`_TILE_ROUTES`.
            data: The caller's raster, as they gave it — a ``Dataset``, a path or a URL. It is what the
                figure records, so a description opens the caller's own reference.
            band: The 1-based band to draw.
            cmap: The caller's colormap, or ``None`` to resolve one from the band's name.
            units: The caller's units, or ``None`` to resolve them from the band's name.
            opacity: Raster layer opacity, already checked finite.
            vmin: The caller's lower colour limit, or ``None`` to measure one.
            vmax: The caller's upper colour limit, or ``None``.
            visible: Whether the layer starts visible.
            name: The caller's name for the layer, or ``None`` to generate one.
            tiles_path: Where to write.
            zooms: The pyramid's zoom range, or ``None`` to derive it.
            asked: The style keys the public builder recorded the caller as having named. Threaded rather
                than derived, because the sentinel is seen at that call and this route is reached with the
                keyword already resolved.

        Returns:
            The map, so builder calls chain.

        Raises:
            ValueError: as :meth:`field` documents for a tiled route.
            TypeError: as :meth:`field` documents, when the input is not a raster pyramids can window.
        """
        caller = _FIELD_CALLER
        dataset = _windowed_raster(self, data, route=route, caller=caller)
        styling = _styling_source(dataset, band)
        cmap_name = self._auto_cmap(styling, cmap)
        low, high = _scan_limits(dataset, band, vmin=vmin, vmax=vmax)
        nodata = _band_nodata(dataset, band)

        def encode(zoom: int, x: int, y: int) -> Optional[bytes]:
            """Colour one tile of the band.

            Args:
                zoom: The zoom level.
                x: The tile column.
                y: The tile row.

            Returns:
                The tile's PNG bytes, or ``None`` when it holds no value to colour.
            """
            values = _tile_values(dataset, zoom, x, y, band=band)
            return (
                None
                if values is None
                else _coloured_png(
                    values, cmap_name, vmin=low, vmax=high, nodata=nodata
                )
            )

        props: dict = {
            "cmap": cmap_name,
            "vmin": low,
            "vmax": high,
            "opacity": float(opacity),
            "band": band,
            **asked_record(asked),
        }
        props.update(
            _tiled_reference(
                route,
                dataset,
                tiles_path=tiles_path,
                zooms=zooms,
                encode=encode,
                caller=caller,
            )
        )
        # After the write, so a call refused for want of a destination leaves nothing of itself behind: a
        # later `colorbar()` would otherwise label itself from a layer that was never built.
        self.last_units = self._auto_units(styling, units)
        layer_id = self._layer_id("raster", name)
        # No source, the way `terrain` records none: `_tiled_layer` builds the MapLibre source out of the
        # `tiles_*` props alone and never opens the caller's raster, so filing it described a read that does
        # not happen. It also cost the route the two things a description is for — an in-memory `Dataset`'s
        # only honest reference is `object:<process>:<id>`, which `FigureSpec.to_dict` refuses, so the figure
        # could not be written down at all; and a replacement carrying the `source_id` the tree now holds as
        # `None` was refused as a raster that had lost its data (review R2-M13).
        if self._index_layer(
            layer_id,
            name,
            kind="raster",
            visible=visible,
            # The same description the inline route publishes, off the pair `_scan_limits` already resolved
            # above — a tiled band is no less coloured by its values than an inlined one, and a key that
            # appeared on one route and not the other would be a second divergence inside one builder.
            symbology=Symbology(
                encodings=self._band_colour(
                    dataset, band=band, cmap=cmap_name, limits=(low, high)
                ),
                props=props,
            ),
        ):
            self._last_layer_id = layer_id
        return self

    def rgb_composite(
        self,
        dataset: Any,
        bands: Any = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[Any] = None,
        opacity: Maybe[float] = UNSET,
        visible: bool = True,
        name: Optional[str] = None,
        tiles: Optional[str] = None,
        tiles_path: Any = None,
        zooms: Any = None,
    ) -> Self:
        """Overlay three bands as a true- or false-colour image (recipe W1).

        Satellite imagery is a headline use of a web map, and the tier could only draw one band through a
        colormap. The stretch comes from :mod:`digitalearth.base.stretch` — the same engine-neutral code
        the static tier's ``rgb_composite`` uses — so the same three bands look the same on both tiers.

        The three channels are read straight as red, green and blue; :meth:`hsv_composite` reads the same
        three as hue, saturation and value.

        Args:
            dataset: A pyramids ``Dataset`` (or anything ``get_stack`` accepts).
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            bands: The three 1-based band numbers, in red-green-blue order.
            mask_nodata: Whether NoData becomes NaN (and so transparent) rather than a real value.
            limits: Per-channel ``(lo, hi)`` stretch limits in band order. ``None`` derives them from this
                image; pass a fixed set to keep a series comparable across frames.
            opacity: Raster layer opacity in ``[0, 1]``.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            tiles: How the pixels reach the page, as on :meth:`field`: ``None`` (the default) inlines them,
                and is refused above :data:`_LARGE_RASTER_PIXELS` cells; ``"xyz"`` writes a
                ``{z}/{x}/{y}.png`` pyramid beside the page and ``"cog"`` one Cloud-Optimized GeoTIFF the
                page reads windows from. Either makes the output a **folder** rather than one file. A tiled
                composite is stretched on one set of bounds for the whole pyramid — ``limits`` when given,
                otherwise measured from a decimated read — because a tile stretched over its own percentiles
                disagrees with its neighbour across their shared edge.
            tiles_path: Where a tiled route writes: the pyramid's root directory, or the ``.tif`` for a COG.
            zooms: ``(lowest, highest)`` zoom levels for the ``"xyz"`` pyramid; ``None`` derives them.

        Returns:
            The same map instance, so builder calls chain. A composite that cannot be placed — off-limb
            in the display CRS, or with corners that will not express as lon/lat — is skipped with a
            warning instead, unless the map was built with ``strict=True``.

        Raises:
            ValueError: when ``bands`` is not exactly three, when ``limits`` does not match them, when the
                composite has no finite pixels to draw, or when ``opacity`` is not a finite number —
                refused at this call, because a figure holding NaN or infinity could not be written down.
            FileNotFoundError: when `dataset` is a path that names nothing, or KeyError when no resolver
                is registered for its URL scheme — from :meth:`~digitalearth.web.base.WebMapBase._opened`.
            TypeError: for a tiled route, when `dataset` is not a raster pyramids can read windows out of;
                see :meth:`field`, whose tiled route shares this one's writer.

        Examples:
            - A true-colour composite from a Landsat-ordered dataset:
                ```python
                >>> from digitalearth.web import WebMap                       # doctest: +SKIP
                >>> WebMap().basemap().rgb_composite(ds, bands=(4, 3, 2))     # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.raster.RasterMixin.hsv_composite: the same three bands read as hue/sat/value.
            digitalearth.base.stretch.channel_limits: derives the ``limits`` this accepts.
            digitalearth.web.raster.RasterMixin.field: the single-band, colormapped path.
        """
        return self._composite(
            _RGB_RECIPE,
            dataset,
            bands,
            mask_nodata=mask_nodata,
            limits=limits,
            opacity=opacity,
            visible=visible,
            name=name,
            tiles=tiles,
            tiles_path=tiles_path,
            zooms=zooms,
        )

    def hsv_composite(
        self,
        dataset: Any,
        bands: Any = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[Any] = None,
        opacity: Maybe[float] = UNSET,
        visible: bool = True,
        name: Optional[str] = None,
        tiles: Optional[str] = None,
        tiles_path: Any = None,
        zooms: Any = None,
    ) -> Self:
        """Overlay three bands as an HSV composite — hue, saturation and value → RGB (#266).

        The static tier has drawn both composites since it drew one, and this tier drew only RGB without
        declaring HSV absent, so it was missing rather than refused. It is the same concept through this
        tier's own engine: the three bands are stretched into ``[0, 1]`` by the shared
        :mod:`digitalearth.base.stretch` — "a composite (RGB or HSV) turns three bands into an image by
        stretching each channel into ``[0, 1]``" — then read as hue, saturation and value and encoded as the
        one RGBA PNG a MapLibre image source takes.

        It is the natural reading for a magnitude-and-direction pair (direction as hue, magnitude as value)
        and for anything whose interesting variable is a *ratio* rather than a colour.

        Args:
            dataset: A pyramids ``Dataset`` (or anything ``get_stack`` accepts).
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            bands: The three 1-based band numbers, in hue-saturation-value order.
            mask_nodata: Whether NoData becomes NaN (and so transparent) rather than a real value. A cell
                missing *any* of the three has no colour and is drawn transparent, as in ``rgb_composite``
                — including a missing hue, which the conversion would otherwise turn into a real colour.
            limits: Per-channel ``(lo, hi)`` stretch limits in band order. ``None`` derives them from this
                image; pass a fixed set to keep a series comparable across frames.
            opacity: Raster layer opacity in ``[0, 1]``.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            tiles: How the pixels reach the page, as on :meth:`field`: ``None`` (the default) inlines them,
                and is refused above :data:`_LARGE_RASTER_PIXELS` cells; ``"xyz"`` writes a
                ``{z}/{x}/{y}.png`` pyramid beside the page and ``"cog"`` one Cloud-Optimized GeoTIFF the
                page reads windows from. Either makes the output a **folder** rather than one file. A tiled
                composite is stretched on one set of bounds for the whole pyramid — ``limits`` when given,
                otherwise measured from a decimated read — because a tile stretched over its own percentiles
                disagrees with its neighbour across their shared edge.
            tiles_path: Where a tiled route writes: the pyramid's root directory, or the ``.tif`` for a COG.
            zooms: ``(lowest, highest)`` zoom levels for the ``"xyz"`` pyramid; ``None`` derives them.

        Returns:
            The same map instance, so builder calls chain. A composite that cannot be placed — off-limb
            in the display CRS, or with corners that will not express as lon/lat — is skipped with a
            warning instead, unless the map was built with ``strict=True``.

        Raises:
            ValueError: when ``bands`` is not exactly three, when ``limits`` does not match them, when the
                composite has no finite pixels to draw, or when ``opacity`` is not a finite number —
                refused at this call, because a figure holding NaN or infinity could not be written down.
            FileNotFoundError: when `dataset` is a path that names nothing, or KeyError when no resolver
                is registered for its URL scheme — from :meth:`~digitalearth.web.base.WebMapBase._opened`.
            TypeError: for a tiled route, when `dataset` is not a raster pyramids can read windows out of;
                see :meth:`field`, whose tiled route shares this one's writer.

        Examples:
            - Three bands read as hue, saturation and value (needs the ``web`` extra, so the block is
              skipped without it):
                ```python
                >>> from digitalearth.web import WebMap                       # doctest: +SKIP
                >>> WebMap().basemap().hsv_composite(ds, bands=(1, 2, 3))     # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.raster.RasterMixin.rgb_composite: the same three bands as red/green/blue.
            digitalearth.static.maps.raster.RasterMixin.hsv_composite: the matplotlib tier's counterpart,
                which this answers to on every parameter the two share.
        """
        return self._composite(
            _HSV_RECIPE,
            dataset,
            bands,
            mask_nodata=mask_nodata,
            limits=limits,
            opacity=opacity,
            visible=visible,
            name=name,
            tiles=tiles,
            tiles_path=tiles_path,
            zooms=zooms,
        )

    def _composite(
        self,
        via: str,
        dataset: Any,
        bands: Any,
        *,
        mask_nodata: bool,
        limits: Optional[Any],
        opacity: Maybe[float],
        visible: bool,
        name: Optional[str],
        tiles: Optional[str] = None,
        tiles_path: Any = None,
        zooms: Any = None,
    ) -> Self:
        """Record a three-band composite and draw it, in whichever colour space ``via`` names.

        Everything the two composites do before the colour space is here: the argument checks, the one warp
        that refuses an off-limb dataset before anything is recorded, the size ceiling and the description.
        The recipe goes into that description rather than staying on the map, because a figure is redrawn
        without its builder — a redraw that fell back to RGB would be a different image with no other symptom
        (see :func:`_composed_rgb`).

        Args:
            via: The recipe, one of :data:`_COMPOSITE_RECIPES`. It names the builder in every message the
                call can raise or log, and is what the drawer reads back.
            dataset: The multiband source, as the caller gave it.
            bands: The three 1-based band numbers, in the order the recipe reads them.
            mask_nodata: Whether NoData becomes NaN (and so transparent).
            limits: Frozen per-channel ``(lo, hi)`` stretch bounds, or ``None`` for a per-call scan.
            opacity: Raster layer opacity in ``[0, 1]``, or :data:`~digitalearth.base.ask.UNSET` for
                :data:`RASTER_OPACITY`. Both composites hand their keyword straight over, so the ask is
                resolved and recorded once here rather than twice above.
            visible: Whether the layer starts visible.
            name: The caller's name for the layer; ``None`` generates one under the recipe's own prefix.
            tiles: The route the pixels reach the page by — see :meth:`field`. Both composites answer to it,
                because both inline their pixels and so both meet the same ceiling.
            tiles_path: Where a tiled route writes.
            zooms: The ``"xyz"`` pyramid's zoom range, or ``None`` to derive it.

        Returns:
            The map, so builder calls chain.
        """
        ask = Ask()
        opacity = as_finite(
            ask("opacity", opacity, RASTER_OPACITY), "opacity", f"WebMap.{via}()"
        )
        route = _tile_route(tiles, f"WebMap.{via}()")
        _require_layer_api()
        require_three_bands(via, bands)
        if route is not None:
            return self._tiled_composite(
                route,
                via,
                dataset,
                bands,
                mask_nodata=mask_nodata,
                limits=limits,
                opacity=opacity,
                asked=ask.named,
                visible=visible,
                name=name,
                tiles_path=tiles_path,
                zooms=zooms,
            )
        # Warped here only so an off-limb dataset is refused before anything is recorded, and so the first
        # draw is handed the warped dataset rather than warping it again. The stack, the size warning, the
        # north-up flip and the placement are all the drawer's, which is the one place they are computed
        # (review M8).
        data = self._display_raster_or_skip(dataset, layer=via)
        if data is None:
            return self
        # At the call, like `field`'s: the drawer runs again on every redraw, and the size of the input is
        # the caller's one-time choice (review N7).
        _refuse_if_large(via, "composite", _grid_pixels(data))
        layer_id = self._layer_id(_COMPOSITE_RECIPES[via], name)
        if self._index_layer(
            layer_id,
            name,
            # The engine-neutral kind for a multi-band colour composite, which is what both of these are —
            # the recipe below says which. The static tier records both under it too.
            kind="rgb",
            visible=visible,
            source=dataset,
            placed=data,
            symbology=Symbology(
                props={
                    "via": via,
                    "bands": tuple(int(band) for band in bands),
                    # A bound the caller's freeze could not measure is recorded as `None`, not as the NaN
                    # `channel_limits` answers with: a description holds only what a figure can be written
                    # as, and NaN has no JSON form (review M9's web instance).
                    "limits": _recorded_limits(limits),
                    "opacity": float(opacity),
                    "mask_nodata": bool(mask_nodata),
                    **ask.record,
                }
            ),
        ):
            self._last_layer_id = layer_id
        return self

    def _tiled_composite(
        self,
        route: str,
        via: str,
        dataset: Any,
        bands: Any,
        *,
        mask_nodata: bool,
        limits: Optional[Any],
        opacity: float,
        visible: bool,
        name: Optional[str],
        tiles_path: Any,
        zooms: Any,
        asked: Sequence[str] = (),
    ) -> Self:
        """Write a composite's pixels beside the page and describe the layer that reads them (#189).

        The composites share the ceiling with :meth:`field` because they share the inline path, so they share
        the way out of it too. The one difference from a single band is the stretch: it is measured once, over
        the whole image, and every tile is stretched on **those** bounds — a tile stretched over its own
        percentiles disagrees with its neighbour across their shared edge, which is the same defect an
        unfrozen animation shows as a pulse from frame to frame.

        Args:
            route: The resolved route, a key of :data:`_TILE_ROUTES`.
            via: The recipe, one of :data:`_COMPOSITE_RECIPES`.
            dataset: The multiband source, as the caller gave it — what the figure records.
            bands: The three 1-based band numbers, in the order the recipe reads them.
            mask_nodata: Recorded as the description's own, and applied per tile: a cell any band has no value
                for is drawn transparent either way, because a tile read hands back the raster's sentinel.
            limits: The caller's frozen per-channel bounds, or ``None`` to measure them from a decimated read.
            opacity: Raster layer opacity, already checked finite.
            visible: Whether the layer starts visible.
            name: The caller's name for the layer, or ``None`` to generate one.
            tiles_path: Where to write.
            zooms: The pyramid's zoom range, or ``None`` to derive it.
            asked: The style keys the public builder recorded the caller as having named. Threaded rather
                than derived, because the sentinel is seen at that call and this route is reached with the
                keyword already resolved.

        Returns:
            The map, so builder calls chain.

        Raises:
            ValueError: as :meth:`field` documents for a tiled route.
            TypeError: as :meth:`field` documents, when the input is not a raster pyramids can window.
        """
        from digitalearth.base.stretch import stretch_to_unit

        caller = f"WebMap.{via}()"
        opened = _windowed_raster(self, dataset, route=route, caller=caller)
        channels = [int(band) for band in bands]
        frozen = _composite_limits(opened, channels) if limits is None else limits

        def encode(zoom: int, x: int, y: int) -> Optional[bytes]:
            """Stretch and compose one tile of the three bands.

            Args:
                zoom: The zoom level.
                x: The tile column.
                y: The tile row.

            Returns:
                The tile's PNG bytes, or ``None`` when no pixel in it is finite in all three bands.
            """
            stack = _tile_stack(opened, zoom, x, y, bands=channels)
            if stack is None:
                return None
            return _composite_png(
                _composed_rgb(via, stretch_to_unit(stack, _stretch_limits(frozen)))
            )

        props: dict = {
            "via": via,
            "bands": tuple(channels),
            "limits": _recorded_limits(frozen),
            "opacity": float(opacity),
            "mask_nodata": bool(mask_nodata),
            **asked_record(asked),
        }
        props.update(
            _tiled_reference(
                route,
                opened,
                tiles_path=tiles_path,
                zooms=zooms,
                encode=encode,
                caller=caller,
            )
        )
        layer_id = self._layer_id(_COMPOSITE_RECIPES[via], name)
        # No source, for the reason `_tiled_field` records none: `draw_rgb_composite` takes the same
        # `tiles`-prop branch into `_tiled_layer`, which reads the template and never the bands (review
        # R2-M13).
        if self._index_layer(
            layer_id,
            name,
            kind="rgb",
            visible=visible,
            symbology=Symbology(props=props),
        ):
            self._last_layer_id = layer_id
        return self

    @staticmethod
    def _composite_png_datauri(unit_stack: Any, *, caller: str = "a composite") -> str:
        """Encode a stretched ``(rows, cols, 3)`` stack as a ``data:image/png;base64,`` URI.

        Args:
            unit_stack: Channel values already stretched to ``[0, 1]`` and read as red, green and blue by
                here (an HSV composite has been converted — see :func:`_composed_rgb`); NaN marks NoData.
            caller: What to name in the refusal below. Both composites reach this encoder, so a builder
                written into the message would send the reader of the other one's failure to the wrong call.

        Returns:
            The PNG data-URI string.

        Raises:
            ValueError: when no pixel is finite in all three channels — there is nothing to draw, and an
                empty image would look like a rendering failure instead of an empty input.
        """
        import base64

        payload = _composite_png(unit_stack)
        if payload is None:
            raise ValueError(
                f"{caller} got a stack with no pixel finite in all three bands"
            )
        return "data:image/png;base64," + base64.b64encode(payload).decode()

    def _lonlat_corners(self, source: Any) -> Optional[list]:
        """Return a raster's corner coordinates as the lon/lat an image source is placed by.

        MapLibre positions an ``image`` source with ``[[lng, lat], …]``, so a display CRS that is not
        lon/lat has to be converted — otherwise the map frames on the right degrees while the image sits in
        metre-space a long way off.

        Args:
            source: The display-CRS source whose ``x``/``y`` coordinate arrays give the corners.

        Returns:
            ``[TL, TR, BR, BL]`` in lon/lat, or ``None`` when the corners cannot be converted.
        """
        corners = self._image_coordinates(source.x.values, source.y.values)
        west, north = corners[0]
        east, south = corners[2]
        converted = self._as_lonlat(west, south, east, north)
        if converted is None:
            return None
        west, south, east, north = converted
        return [[west, north], [east, north], [east, south], [west, south]]

    @staticmethod
    def _image_coordinates(x: Any, y: Any) -> List[List[float]]:
        """Return the image-source corner coordinates ``[TL, TR, BR, BL]`` in ``[lng, lat]``.

        The corners are the cell *edges* — half the outermost spacing beyond the centres on each side
        (:meth:`Bounds.cell_edges`). Taken from the centres themselves, an image source is drawn half a cell
        inside the data all the way round, and a map framed on those corners is inset by the same amount
        (#301).

        Args:
            x: 1-D x / longitude cell-centre coordinates (display CRS, lon/lat).
            y: 1-D y / latitude cell-centre coordinates.

        Returns:
            The four corners top-left, top-right, bottom-right, bottom-left as ``[lng, lat]`` pairs — the
            order MapLibre's image source expects.
        """
        west, south, east, north = Bounds.cell_edges(x, y, crs=None).as_bbox()
        return [[west, north], [east, north], [east, south], [west, south]]

    @staticmethod
    def _rgba_png_datauri(
        values: Any,
        cmap: Any,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        classes: Optional[BandClasses] = None,
    ) -> str:
        """Colour-map a 2-D array to an RGBA PNG and return it as a ``data:image/png;base64,`` URI.

        Non-finite / masked cells are rendered fully transparent (the tier's NoData contract). matplotlib
        is imported lazily here so the tier imports without it.

        Args:
            values: A 2-D (possibly masked) array of band values, already oriented north-up.
            cmap: A registered matplotlib colormap name, or a ``Colormap`` itself. Resolved through
                :func:`~digitalearth.base.symbology.as_colormap`, the one resolver every tier reads a
                colormap through, so the unclassified raster path accepts the same object the classified
                ones do: looking the argument up as a dict key raised ``TypeError: unhashable type`` for a
                ``Colormap``, which defines ``__eq__`` and so has no hash (#315, review H3).
            vmin: Lower colour limit; ``None`` uses the finite minimum.
            vmax: Upper colour limit; ``None`` uses the finite maximum.
            classes: The band's classes, to colour it class by class; ``None`` colours it along `cmap`'s
                ramp between `vmin` and `vmax`.

        Returns:
            The PNG data-URI string.

        Raises:
            ValueError: when the array has no finite values to colour.
            KeyError: when ``cmap`` is hashable but names no colormap matplotlib's registry holds, which
                is matplotlib's own message naming it. Only on the ramp path: with `classes` the colours are
                the classes' own and ``cmap`` is not resolved.
            TypeError: when ``cmap`` is unhashable — a list of colours, say — which
                :func:`~digitalearth.base.symbology.as_colormap` reports as ``unhashable type: 'list'``;
                likewise only on the ramp path.

        Examples:
            - Colour a band along a ramp, and get back an inline PNG:
                ```python
                >>> import numpy as np
                >>> from digitalearth.web import WebMap
                >>> WebMap._rgba_png_datauri(np.array([[0.0, 1.0]]), "viridis")[:22]
                'data:image/png;base64,'

                ```
            - Colour it class by class; the classes carry their colours, so ``cmap`` is not looked up:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.raster_classes import classify_band
                >>> from digitalearth.web import WebMap
                >>> classes = classify_band(np.array([1, 2]), "categorical", None, None)
                >>> WebMap._rgba_png_datauri(np.array([[1.0, 2.0]]), "no-such-cmap", classes=classes)[:22]
                'data:image/png;base64,'

                ```
            - A band with nothing to colour is refused, classified or not:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.raster_classes import classify_band
                >>> from digitalearth.web import WebMap
                >>> classes = classify_band(np.array([1, 2]), "categorical", None, None)
                >>> WebMap._rgba_png_datauri(np.array([[np.nan]]), "viridis", classes=classes)
                Traceback (most recent call last):
                    ...
                ValueError: field() got a band with no finite values to colour

                ```
        """
        import base64

        payload = (
            _coloured_png(values, cmap, vmin=vmin, vmax=vmax)
            if classes is None
            else _classed_png(values, classes)
        )
        if payload is None:
            raise ValueError("field() got a band with no finite values to colour")
        return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
