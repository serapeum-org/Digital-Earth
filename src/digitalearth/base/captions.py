"""Captions from a Source's metadata — a human title and a units-aware value label.

The counterpart of :mod:`digitalearth.base.autostyle`: that module maps a Source's metadata to *how* a field
is coloured, and this one maps the same metadata to *what it is called* — the title over a panel and the label
beside a colorbar. Both are domain mappings over CF/GRIB-style metadata (``variable``/``standard_name``/
``long_name``/``units`` plus any time or level keys), so both are engine-neutral and live in ``base``; the
static, interactive, web and 3-D tiers all compose the same caption.

It is **read-only over metadata and imports no renderer**, so it stays inside ``base``'s engine-neutral rule.
A tier applies a caption by calling :func:`caption` (the ready-made title/label pair) or
:func:`format_metadata` (an arbitrary template), and passing the strings to its own ``set_title`` / colorbar.
"""

from dataclasses import dataclass

from digitalearth.base.sources.source import Source

__all__ = ["Caption", "caption", "format_metadata"]

#: Display spellings for a few common unit strings, so a label reads ``[°C]`` rather than ``[celsius]``.
#: Everything not listed is shown verbatim — the map only tidies the handful that recur.
_UNIT_DISPLAY = {
    "celsius": "°C",
    "degc": "°C",
    "degree_celsius": "°C",
    "degrees_celsius": "°C",
    "kelvin": "K",
    "degk": "K",
    "fahrenheit": "°F",
    "degf": "°F",
    "percent": "%",
    "degrees": "°",
    "degree": "°",
}


def _prettify_units(units: str | None) -> str | None:
    """Return a display spelling for a unit string, tidying the common ones; ``None``/empty stay as given."""
    if not units:
        return units
    return _UNIT_DISPLAY.get(units.strip().lower(), units)


def _name(source: Source) -> str:
    """The most human name a source offers: long_name, then standard_name, then variable, else ``value``."""
    for key in ("long_name", "standard_name", "variable"):
        value = source.metadata(key)
        if value:
            return str(value).replace("_", " ").strip()
    return "value"


def _units(source: Source, style_units: str | None) -> str | None:
    """The source's own units, falling back to a style's declared units; prettified for display."""
    return _prettify_units(source.units or style_units)


@dataclass(frozen=True)
class Caption:
    """A field's title and value label, both derived from its metadata.

    Attributes:
        title: The panel heading — the field's human name, plus a time stamp when the source carries one.
        label: The value label for a colorbar/legend — the name with its units in brackets when known.

    Examples:
        - Build a caption and read its two parts:
            ```python
            >>> from digitalearth.base.captions import Caption
            >>> cap = Caption(title="2 metre temperature", label="2 metre temperature [°C]")
            >>> cap.title
            '2 metre temperature'
            >>> cap.label
            '2 metre temperature [°C]'

            ```
    """

    title: str
    label: str


def caption(source: Source, *, style_units: str | None = None) -> Caption:
    """Build a :class:`Caption` — a title and a units-aware value label — from a source's metadata.

    The name is the most human of ``long_name`` → ``standard_name`` → ``variable`` (underscores spaced),
    falling back to ``value``. The label appends the units in brackets when the source (or, failing it, a
    resolved style via ``style_units``) names any. The title appends a ``time`` / ``valid_time`` stamp when
    the source carries one — the field named, and when it is for.

    Args:
        source: The data source whose metadata drives the caption.
        style_units: Units from a resolved :func:`~digitalearth.base.autostyle.auto_style` (its canonical
            units win over the file's where it has them), used only when the source names none itself.

    Returns:
        A :class:`Caption` with ``title`` and ``label``.

    Examples:
        - A named field with units and a time becomes a full caption:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, Source
            >>> from digitalearth.base.captions import caption
            >>> src = Source(
            ...     DimensionInfo(np.zeros((2, 2)), "z"),
            ...     DimensionInfo(np.array([0.0]), "x"),
            ...     DimensionInfo(np.array([0.0]), "y"),
            ...     metadata={"long_name": "2 metre temperature", "time": "2020-01-01"},
            ...     units="celsius",
            ... )
            >>> cap = caption(src)
            >>> cap.title
            '2 metre temperature — 2020-01-01'
            >>> cap.label
            '2 metre temperature [°C]'

            ```
        - With no units and no time, the label is just the name:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, Source
            >>> from digitalearth.base.captions import caption
            >>> bare = Source(
            ...     DimensionInfo(np.zeros((2, 2)), "z"),
            ...     DimensionInfo(np.array([0.0]), "x"),
            ...     DimensionInfo(np.array([0.0]), "y"),
            ...     metadata={"variable": "mystery"},
            ... )
            >>> caption(bare)
            Caption(title='mystery', label='mystery')

            ```
    """
    name = _name(source)
    units = _units(source, style_units)
    label = f"{name} [{units}]" if units else name
    when = source.metadata("time") or source.metadata("valid_time")
    title = f"{name} — {when}" if when else name
    return Caption(title=title, label=label)


class _MetadataMap(dict):
    """A ``format_map`` backing that resolves unknown keys from the source, missing ones to ``""``."""

    def __init__(self, source: Source, seed: dict[str, str]) -> None:
        super().__init__(seed)
        self._source = source

    def __missing__(self, key: str) -> str:
        value = self._source.metadata(key)
        return "" if value is None else str(value)


def format_metadata(
    source: Source, template: str, *, style_units: str | None = None
) -> str:
    """Fill a ``str.format``-style template from a source's metadata, missing keys resolving to ``""``.

    The templating counterpart of :func:`caption`, for a caller who wants a bespoke string — e.g.
    ``"{long_name} at {valid_time} [{units}]"``. Any ``{key}`` is looked up with
    :meth:`Source.metadata`, except ``{name}`` (the human name :func:`caption` uses) and ``{units}`` (the
    prettified units, source-or-style). A key the source does not carry becomes an empty string rather than
    raising, so a template degrades cleanly on data that lacks a field.

    Args:
        source: The data source whose metadata fills the template.
        template: A ``str.format`` template referencing metadata keys, plus ``{name}`` and ``{units}``.
        style_units: Units from a resolved style, used for ``{units}`` only when the source names none.

    Returns:
        The filled template. Surrounding whitespace is stripped, but internal gaps left by empty fields are
        the caller's template to design around.

    Examples:
        - A template pulls named keys, and silently empties the ones the data lacks:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, Source
            >>> from digitalearth.base.captions import format_metadata
            >>> src = Source(
            ...     DimensionInfo(np.zeros((2, 2)), "z"),
            ...     DimensionInfo(np.array([0.0]), "x"),
            ...     DimensionInfo(np.array([0.0]), "y"),
            ...     metadata={"variable": "t2m", "level": "2m"},
            ...     units="K",
            ... )
            >>> format_metadata(src, "{variable} at {level} [{units}]")
            't2m at 2m [K]'
            >>> format_metadata(src, "{variable} {step}")
            't2m'

            ```
    """
    seed = {"name": _name(source), "units": _units(source, style_units) or ""}
    return template.format_map(_MetadataMap(source, seed)).strip()
