"""Engine-neutral input guards shared by the 2-D vector tiers (web, interactive).

A vector builder (``points`` / ``polygons`` / ``choropleth`` / …) wants a vector layer. Handed a pyramids
raster instead, it used to fail wherever it first touched the input — ``object of type 'Dataset' has no
len()`` from one builder, ``argument of type 'int' is not iterable`` from another (#152) — never with a hint
that the method wanted a vector. Both 2-D tiers have the same shape, so rather than copy a guard per tier the
check lives here, where it belongs: it inspects only ``geometry`` and ``columns``, imports no renderer, and so
keeps ``base/`` engine-neutral (``pyramids`` stays the only GIS dependency — a raster is recognised by duck
type, never by importing ``Dataset``). Each tier wraps these with its own raster entry points, so the message
routes the caller to ``field()`` / ``timecube()`` on the interactive tier and ``field()`` / ``timeslider()``
on the web tier.
"""

from typing import Any

#: The most column names a geometry-less-table error lists before truncating, so a wide frame does not print
#: a hundred names. The one place both tiers agree on the cap.
MAX_COLUMNS_IN_ERROR = 12


def reject_raster(data: Any, method: str, *, single: str, stack: str) -> None:
    """Raise ``TypeError`` when ``data`` is a pyramids raster, leaving anything else alone.

    For a builder that legitimately takes something other than a vector layer — a raw ``xyz`` point cloud, a
    UGRID mesh exposing ``node_x``/``node_y`` rather than ``geometry`` — where :func:`require_vector` would be
    too strict. A raster is recognised by reporting ``columns`` as an ``int`` (the grid width in cells); a
    table reports an ``Index`` of names and a bare sequence has no ``columns`` at all.

    Args:
        data: The caller's input.
        method: The calling builder name (quoted in the error, without parentheses).
        single: The tier's single-raster entry point, named in the message (e.g. ``"field"``).
        stack: The tier's raster-time-stack entry point, named in the message (e.g. ``"timecube"``).

    Raises:
        TypeError: when ``data`` is a pyramids raster.
    """
    if isinstance(getattr(data, "columns", None), int):
        raise TypeError(
            f"{method}() does not take a raster; got {type(data).__name__}. For a single raster use "
            f"{single}(); for a raster time stack pass a DatasetCollection to {stack}()."
        )


def require_vector(features: Any, method: str, *, single: str, stack: str) -> None:
    """Raise ``TypeError`` unless ``features`` is a vector layer.

    The shared input guard a tier calls so every vector builder rejects a raster the same way. Without it a
    pyramids raster reaches whichever expression the builder happens to run first and fails obscurely there —
    a ``Dataset`` reports ``columns`` as an ``int`` (the grid width in cells), so an attribute-membership test
    raises ``TypeError: argument of type 'int' is not iterable``, while other builders raise ``has no len()``,
    an ``AttributeError``, or nothing at all.

    Args:
        features: The caller's input, expected to be a pyramids ``FeatureCollection`` / GeoDataFrame.
        method: The calling builder name (quoted in the error, without parentheses).
        single: The tier's single-raster entry point, named in the message (e.g. ``"field"``).
        stack: The tier's raster-time-stack entry point, named in the message (e.g. ``"timecube"``).

    Raises:
        TypeError: when ``features`` exposes no ``geometry``. A table that has columns but no *active*
            geometry (``set_geometry`` never called) gets its own message naming ``set_geometry``, rather
            than being misreported as a raster.

    Examples:
        - A vector-like input passes the guard silently (the check returns nothing):
            ```python
            >>> from digitalearth.base.guards import require_vector
            >>> layer = type("Layer", (), {"geometry": ()})()
            >>> print(require_vector(layer, "points", single="field", stack="timecube"))
            None

            ```
        - A raster is turned away, and the message names the builder and the tier's raster entry points:
            ```python
            >>> from digitalearth.base.guards import require_vector
            >>> try:
            ...     require_vector(42, "points", single="field", stack="timecube")
            ... except TypeError as err:
            ...     print(str(err).split("(")[0], "|", "field" in str(err), "timecube" in str(err))
            points | True True

            ```
    """
    if hasattr(features, "geometry"):
        return
    # A raster reports `columns` as an int (the grid width); a table reports an Index of names. Telling the
    # two apart keeps a GeoDataFrame whose geometry was never activated from being called a raster.
    reject_raster(features, method, single=single, stack=stack)
    columns = getattr(features, "columns", None)
    if columns is not None:
        names = [str(name) for name in columns]
        shown = names[:MAX_COLUMNS_IN_ERROR]
        if len(names) > MAX_COLUMNS_IN_ERROR:
            shown.append(f"... (+{len(names) - MAX_COLUMNS_IN_ERROR} more)")
        raise TypeError(
            f"{method}() needs a layer with an active geometry column; got a "
            f"{type(features).__name__} whose columns are {shown}. Call set_geometry(...) on it first."
        )
    raise TypeError(
        f"{method}() needs a vector layer (a pyramids FeatureCollection / GeoDataFrame); got "
        f"{type(features).__name__}. For a single raster use {single}(); for a raster time stack "
        f"use {stack}() with a DatasetCollection."
    )
