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
include at least two of the axes the store marks as horizontal (CF ``axis`` of ``X``/``Y``, or a
``standard_name`` naming a horizontal coordinate). pyramids answers this exact question internally — on the
installed 0.75.0, ``NetCDF._spatial_variable_names()`` returns ``['sst']`` for the container above — but the
method is private, so it is not consumed here. If it is ever published, this module collapses to a delegation.

When nothing can be classified — a curvilinear store whose coordinates are 2-D auxiliary arrays, or a
container exposing no metadata — the first declared name is returned, which is what every reader did before.
That keeps this strictly no worse than the previous behaviour on a store it cannot read.

It is a leaf like :mod:`digitalearth.base.sources.bandname`: pyramids metadata in through ``getattr``, a plain
string out, no renderer and no cycle.
"""

from typing import Any

#: CF ``axis`` values that mark a horizontal (map-plane) coordinate. ``T`` and ``Z`` are deliberately absent:
#: a variable over time and depth alone has no plane to draw.
HORIZONTAL_AXES = frozenset({"X", "Y"})

#: CF ``standard_name`` values for a horizontal coordinate, for a store that names them but declares no
#: ``axis``. Both the geographic and the projected spellings, since either can carry the map plane.
HORIZONTAL_STANDARD_NAMES = frozenset(
    {
        "latitude",
        "longitude",
        "grid_latitude",
        "grid_longitude",
        "projection_x_coordinate",
        "projection_y_coordinate",
    }
)

#: How many horizontal axes a variable needs before it can be drawn: both of them, or it has no plane.
_PLANE = 2


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


def _attributes(info: Any) -> dict:
    """The CF attributes pyramids read off one variable, as a plain mapping.

    Args:
        info: A pyramids ``VariableInfo``, or anything exposing ``attributes``.

    Returns:
        The attribute mapping, or an empty dict when the object exposes none.
    """
    return dict(getattr(info, "attributes", None) or {})


def _horizontal_dimensions(variables: Any) -> set:
    """The names of the dimensions that carry the map plane.

    Args:
        variables: The container's ``metadata.variables`` mapping, name to ``VariableInfo``.

    Returns:
        The coordinate names the store marks as horizontal. Empty when it marks none, which is the signal to
        fall back rather than to conclude nothing is drawable.
    """
    horizontal = set()
    for name, info in (variables or {}).items():
        attributes = _attributes(info)
        axis = str(attributes.get("axis", "")).upper()
        standard = str(attributes.get("standard_name", "")).lower()
        if axis in HORIZONTAL_AXES or standard in HORIZONTAL_STANDARD_NAMES:
            horizontal.add(_bare(name))
    return horizontal


def _dimensions_of(info: Any) -> set:
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


def drawable_variable_names(data: Any) -> list:
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
    names = [str(name) for name in (getattr(data, "variable_names", None) or [])]
    if not names:
        return []
    variables = getattr(getattr(data, "metadata", None), "variables", None)
    horizontal = _horizontal_dimensions(variables)
    if not horizontal:
        return []
    described = {}
    for key, info in (variables or {}).items():
        described.setdefault(_bare(key), info)
    drawable = []
    for name in names:
        info = described.get(_bare(name))
        if info is None:
            continue
        if len(_dimensions_of(info) & horizontal) >= _PLANE:
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
    drawable = drawable_variable_names(data)
    if drawable:
        return drawable[0]
    names = [str(name) for name in (getattr(data, "variable_names", None) or [])]
    return names[0] if names else None
