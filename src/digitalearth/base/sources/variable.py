"""The one answer to "which variable does this container draw?", shared by every reader that asks it.

A NetCDF container holds several named arrays and a call that names none has to pick one. Three readers ask
that question — the display extractor that reads the values, the animated path that slices them, and the
colour-key label — and they must agree, or the same call draws one field and titles the key with another.

They used to agree by each taking ``variable_names[0]``, which held while pyramids returned the CF
classification's *sorted* list. pyramids-core 0.75 returns the store's own declaration order instead: the set
of names is unchanged and nothing warns, but the first one moved. A store is free to declare a CF bounds array
before its data — the repository's own ``ersstv5.nc`` declares ``["time_bnds", "sst"]`` — so taking the
declared first began drawing the bounds array.

A bounds array has no spatial plane, so it is not something a map can draw at all. The answer here is
therefore the first variable that *is* drawable, which is stable under either ordering.

Drawability is decided from the container's own public metadata: a variable is drawable when its dimensions
include a dimension spanned by a coordinate the store marks as its eastings *and* one spanned by a
coordinate it marks as its northings (CF ``axis`` of ``X``/``Y``, or a ``standard_name`` naming either).
Reading the dimensions a horizontal coordinate spans, rather than the coordinate's own name, is what lets a
curvilinear store classify: ROMS' ``lat_rho``/``lon_rho`` span ``(eta_rho, xi_rho)`` and nothing is named
``lat``. Requiring one of *each* axis rather than any two horizontal names is what refuses a staggered
store's ``(xi_rho, xi_u)`` variable, which spans no northing at all.

pyramids answers this exact question internally — on the installed 0.75.0, ``NetCDF._spatial_variable_names()``
returns ``['sst']`` for the container above — but the method is private, so it is not consumed here. If it is
ever published, this module collapses to a delegation.

When nothing can be classified — a container exposing no metadata, or one marking neither axis — the first
declared name is returned, which is what every reader did before. That keeps this strictly no worse than the
previous behaviour on a store it cannot read.

It is a leaf like :mod:`digitalearth.base.sources.bandname`: pyramids metadata in through ``getattr``, a plain
string out, no renderer and no cycle.
"""

from collections.abc import Mapping
from typing import Any

#: CF ``axis`` values naming the two horizontal axes. ``T`` and ``Z`` are deliberately absent: a variable
#: over time and depth alone has no plane to draw.
X_AXIS = "X"
Y_AXIS = "Y"

#: CF ``standard_name`` values for each horizontal axis, for a store that names its coordinates but declares
#: no ``axis``. Geographic, rotated-pole and projected spellings, since any of them can carry the map plane.
X_STANDARD_NAMES = frozenset({"longitude", "grid_longitude", "projection_x_coordinate"})
Y_STANDARD_NAMES = frozenset({"latitude", "grid_latitude", "projection_y_coordinate"})


def _bare(name: Any) -> str:
    """One name with any group path stripped, so the two sides of a lookup can be compared.

    A grouped store names the same variable two ways: a sub-group view reports a bare ``sst`` from
    ``variable_names`` while ``metadata.variables`` is keyed by the full store path ``forecast/sst``, and the
    mismatch can run either way. Every side of the comparison — the declared names, the metadata keys and the
    dimension lists — goes through here, so none of them is normalised while another is not.

    Args:
        name: A variable, coordinate or dimension name, with or without a leading group path.

    Returns:
        The name after the last ``/``.
    """
    return str(name).rsplit("/", maxsplit=1)[-1]


def _described(variables: Any) -> list[tuple[str, Any]]:
    """The ``(name, info)`` pairs of a container's variable metadata, from either shape it comes in.

    pyramids hands back a mapping of name to ``VariableInfo``, but a container may expose a plain sequence
    of infos instead — :func:`digitalearth.base.sources.extractors._variable_attributes` already tolerates
    both, so this package expects both. A sequence's names are read off the infos themselves.

    Args:
        variables: A mapping of name to info, a sequence of infos, or ``None``.

    Returns:
        The pairs, or an empty list when there is no metadata to read.
    """
    if variables is None:
        return []
    if isinstance(variables, Mapping):
        return list(variables.items())
    return [
        (str(getattr(info, "full_name", None) or getattr(info, "name", "") or ""), info)
        for info in variables
    ]


def _attributes(info: Any) -> dict[str, Any]:
    """The CF attributes pyramids read off one variable, as a plain mapping.

    Args:
        info: A pyramids ``VariableInfo``, or anything exposing ``attributes``.

    Returns:
        The attribute mapping, or an empty dict when the object exposes none.
    """
    return dict(getattr(info, "attributes", None) or {})


def _horizontal_dimensions(variables: Any) -> tuple[set[str], set[str]]:
    """The dimension names carrying each of the two horizontal axes.

    Args:
        variables: The container's ``metadata.variables`` mapping, name to ``VariableInfo``.

    Returns:
        ``(x_names, y_names)`` — the coordinate names the store marks as its eastings and its northings.
        Either may be empty when the store marks none, which is the signal to fall back rather than to
        conclude nothing is drawable.
    """
    x_names: set[str] = set()
    y_names: set[str] = set()
    for name, info in _described(variables):
        attributes = _attributes(info)
        axis = str(attributes.get("axis", "")).upper()
        standard = str(attributes.get("standard_name", "")).lower()
        # The *dimensions* the coordinate spans, not its own name, because those are what a data variable
        # references. For a 1-D `lat` over `lat` the two are the same; for a curvilinear store they are not
        # — ROMS' `lat_rho` spans `(eta_rho, xi_rho)` and nothing is called `lat` at all, so matching on the
        # coordinate's name found no horizontal dimension and every variable fell through to the fallback.
        # A coordinate whose dimensions are not declared contributes its own name, as before.
        spanned = _dimensions_of(info) or {_bare(name)}
        if axis == X_AXIS or standard in X_STANDARD_NAMES:
            x_names |= spanned
        elif axis == Y_AXIS or standard in Y_STANDARD_NAMES:
            y_names |= spanned
    return x_names, y_names


def _dimensions_of(info: Any) -> set[str]:
    """One variable's dimension names, with any group path stripped.

    Args:
        info: A pyramids ``VariableInfo``, or anything exposing ``dimensions``.

    Returns:
        The bare dimension names, so a grouped store's ``/time`` compares equal to a coordinate named
        ``time``.
    """
    return {
        str(dim).rsplit("/", maxsplit=1)[-1]
        for dim in (getattr(info, "dimensions", None) or [])
    }


def drawable_variable_names(data: Any) -> list[str]:
    """The container's variables that have a map plane, in the order the container declares them.

    Args:
        data: A pyramids ``NetCDF`` container, or anything exposing ``variable_names`` and ``metadata``.

    Returns:
        The drawable names. Empty when the container declares none, or when its metadata does not mark a
        horizontal axis — the caller cannot tell those two apart and should fall back either way.

    Examples:
        - A store declaring a bounds array beside its data reports only the data:
            ```python
            >>> class Info:
            ...     def __init__(self, dimensions, attributes=None):
            ...         self.dimensions = dimensions
            ...         self.attributes = attributes or {}
            >>> class Metadata:
            ...     variables = {
            ...         "lat": Info(["lat"], {"axis": "Y"}),
            ...         "lon": Info(["lon"], {"axis": "X"}),
            ...         "time_bnds": Info(["time", "nbnds"]),
            ...         "sst": Info(["time", "lat", "lon"]),
            ...     }
            >>> class Container:
            ...     variable_names = ["time_bnds", "sst"]
            ...     metadata = Metadata()
            >>> drawable_variable_names(Container())
            ['sst']

            ```
        - A projected store is classified from its ``standard_name`` when it declares no ``axis``:
            ```python
            >>> class Info:
            ...     def __init__(self, dimensions, attributes=None):
            ...         self.dimensions = dimensions
            ...         self.attributes = attributes or {}
            >>> class Metadata:
            ...     variables = {
            ...         "x": Info(["x"], {"standard_name": "projection_x_coordinate"}),
            ...         "y": Info(["y"], {"standard_name": "projection_y_coordinate"}),
            ...         "temp": Info(["time", "y", "x"]),
            ...     }
            >>> class Container:
            ...     variable_names = ["temp"]
            ...     metadata = Metadata()
            >>> drawable_variable_names(Container())
            ['temp']

            ```
        - A curvilinear store classifies too, because a coordinate lends the dimensions it spans rather
          than its own name — nothing here is called ``lat`` or ``lon``:
            ```python
            >>> class Info:
            ...     def __init__(self, dimensions, attributes=None):
            ...         self.dimensions = dimensions
            ...         self.attributes = attributes or {}
            >>> class Metadata:
            ...     variables = {
            ...         "lat_rho": Info(["eta_rho", "xi_rho"], {"standard_name": "latitude"}),
            ...         "lon_rho": Info(["eta_rho", "xi_rho"], {"standard_name": "longitude"}),
            ...         "time_bnds": Info(["time", "nbnds"]),
            ...         "temp": Info(["time", "eta_rho", "xi_rho"]),
            ...     }
            >>> class Container:
            ...     variable_names = ["time_bnds", "temp"]
            ...     metadata = Metadata()
            >>> drawable_variable_names(Container())
            ['temp']

            ```
        - A container whose metadata marks no horizontal axis reports nothing, which is the signal to fall
          back rather than a claim that nothing is drawable:
            ```python
            >>> class Bare:
            ...     variable_names = ["first", "second"]
            >>> drawable_variable_names(Bare())
            []

            ```

    See Also:
        default_variable: Picks the first of these, falling back to the first declared name.
    """
    return _drawable(
        _declared(data), getattr(getattr(data, "metadata", None), "variables", None)
    )


def _declared(data: Any) -> list[str]:
    """The variable names a container declares, in its own order.

    Args:
        data: Anything exposing ``variable_names``.

    Returns:
        The names as strings. Empty for a subset already pinned to one variable, which reports none.
    """
    return [str(name) for name in (getattr(data, "variable_names", None) or [])]


def _drawable(names: list[str], variables: Any) -> list[str]:
    """Which of `names` the metadata shows to have a map plane.

    Args:
        names: The declared names, in the container's order.
        variables: The container's variable metadata, in either shape :func:`_described` accepts.

    Returns:
        The drawable names, in declaration order.
    """
    if not names:
        return []
    x_names, y_names = _horizontal_dimensions(variables)
    if not (x_names and y_names):
        return []
    described: dict[str, Any] = {}
    for key, info in _described(variables):
        described.setdefault(_bare(key), info)
    drawable: list[str] = []
    for name in names:
        info = described.get(_bare(name))
        if info is None:
            continue
        dimensions = _dimensions_of(info)
        if dimensions & x_names and dimensions & y_names:
            drawable.append(name)
    return drawable


def default_variable(data: Any) -> str | None:
    """The variable a call draws when it names none.

    The first drawable variable, falling back to the first declared one when drawability cannot be decided —
    a curvilinear store, or a container exposing no metadata.

    Args:
        data: A pyramids ``NetCDF`` container, or anything exposing ``variable_names``.

    Returns:
        The variable's name, or ``None`` when the container declares no variables at all (which is how a
        subset already pinned to one variable answers, since it *is* the variable).

    Examples:
        - A store declaring a bounds array first still draws its data:
            ```python
            >>> class Info:
            ...     def __init__(self, dimensions, attributes=None):
            ...         self.dimensions = dimensions
            ...         self.attributes = attributes or {}
            >>> class Metadata:
            ...     variables = {
            ...         "lat": Info(["lat"], {"axis": "Y"}),
            ...         "lon": Info(["lon"], {"axis": "X"}),
            ...         "time_bnds": Info(["time", "nbnds"]),
            ...         "sst": Info(["time", "lat", "lon"]),
            ...     }
            >>> class Container:
            ...     variable_names = ["time_bnds", "sst"]
            ...     metadata = Metadata()
            >>> default_variable(Container())
            'sst'

            ```
        - A container whose metadata marks no horizontal axis falls back to the declared first:
            ```python
            >>> class Bare:
            ...     variable_names = ["first", "second"]
            >>> default_variable(Bare())
            'first'

            ```
        - A subset already pinned to one variable declares no names, and answers with nothing:
            ```python
            >>> class Pinned:
            ...     variable_names = []
            >>> default_variable(Pinned()) is None
            True

            ```
    """
    # `variable_names` is read once and handed to the classifier: on a large or remote store each read
    # walks it, and the fallback path used to read it a second time after `drawable_variable_names` had.
    names = _declared(data)
    if not names:
        return None
    drawable = _drawable(
        names, getattr(getattr(data, "metadata", None), "variables", None)
    )
    return drawable[0] if drawable else names[0]
