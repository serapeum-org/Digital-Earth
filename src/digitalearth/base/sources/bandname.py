"""The one answer to "what is this raster's band called?", shared by every tier that asks it.

A raster layer's colour is bound to a field, and a colour key is titled after that field's name. Two 2-D tiers
need the name at the moment the layer is *described* — before any drawer has run, so before the reprojected
:class:`~digitalearth.base.sources.source.Source` exists to read it off — and both answer it from the raster's
own metadata, by the same two lookups and the same fallback. They were written independently, days apart, and
agreed only by maintenance (#351). The lookup lives here now, so the *same raster keys the same field name on
whichever backend draws it* — which is what order 24's guide-on-the-encoding model needs to be true across
tiers, and what a figure written on one tier and read on another relies on.

It reads the name off the raster, never by warping the band a second time to look: the drawer already warps,
and asking again here would do its work twice and throw the result away. It is a leaf like
:mod:`digitalearth.base.sources.source` — pyramids metadata in through ``getattr``, a plain string out, no
renderer and no cycle.
"""

from typing import Any

from digitalearth.base.sources.variable import default_variable
from digitalearth.base.spec import DEFAULT_BAND

#: What a band with no name of its own is called, so a colour :class:`~digitalearth.base.spec.encoding.Encoding`
#: has a field name rather than the empty string it refuses. One constant, read by every tier, because the
#: cross-tier promise fails exactly where the metadata is thinnest — a nameless band must read the same name on
#: every backend, or a figure written on one and read on another carries a field the second does not recognise.
UNNAMED_BAND = "z"


def band_field_name(data: Any, band: int = DEFAULT_BAND) -> str:
    """Return the name of the value a raster layer's colour varies with.

    Read straight off the raster's ``band_names``/``variable_names``, never invented and never obtained by
    warping the band again. A pyramids ``Dataset``'s band carries its own name; a ``NetCDF`` has variables
    rather than bands, and the display extractor takes the **first** of them, so that is the one a colour key
    would explain. Both are read with ``getattr``, so an input declaring neither list simply falls back.

    Args:
        data: The raster the layer draws — a pyramids ``Dataset`` / ``NetCDF`` / ``Source``, a path, or a bare
            array. Anything declaring neither ``band_names`` nor ``variable_names`` falls back.
        band: The 1-based band the layer colours by.

    Returns:
        The band's name, the first drawable variable's name, or :data:`UNNAMED_BAND` — never an empty string, which an
        :class:`~digitalearth.base.spec.encoding.Encoding` refuses as a field name.

    Examples:
        - A band with a name returns it:
            ```python
            >>> from digitalearth.base.sources.bandname import band_field_name
            >>> class _Raster:
            ...     band_names = ["elevation", "slope"]
            >>> band_field_name(_Raster(), 2)
            'slope'

            ```
        - With no band names, the first drawable variable's name stands in:
            ```python
            >>> from digitalearth.base.sources.bandname import band_field_name
            >>> class _NetCDF:
            ...     variable_names = ["t2m", "sp"]
            >>> band_field_name(_NetCDF(), 1)
            't2m'

            ```
        - With neither, the shared placeholder keeps the field name non-empty:
            ```python
            >>> from digitalearth.base.sources.bandname import band_field_name, UNNAMED_BAND
            >>> band_field_name(object(), 1) == UNNAMED_BAND
            True

            ```
    """
    names = list(getattr(data, "band_names", None) or [])
    if 1 <= band <= len(names) and names[band - 1]:
        return str(names[band - 1])
    # The variable the display extractor draws, read through the same helper rather than re-deriving it, so
    # this name agrees with the drawn value dimension by construction. It is deliberately not
    # `variable_names[0]`: pyramids 0.75 returns the store's declaration order, so the first name can be a CF
    # bounds array the extractor skips (see base/sources/variable.py).
    first = default_variable(data)
    return str(first) if first else UNNAMED_BAND
