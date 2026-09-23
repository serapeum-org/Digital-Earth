"""How a 3-D layer is described: its id, and the engine options the renderer draws it with.

A builder in this tier used to hand a mesh straight to the plotter, and what the layer *was* existed only inside
PyVista afterwards. Now each builder writes a :class:`~digitalearth.base.spec.LayerSpec` and the renderer draws
from it, which needs two small things this module provides.

**An id per layer.** Layers are addressed by id, not by position, so a builder that was not given a name gets
one: `terrain-1`, `point_cloud-2`. A name a caller gives is used as it is, suffixed when it is already taken.

**A place for the caller's keywords.** ``scene.terrain(dem, cmap="terrain", show_edges=True)`` passes its extra
keywords straight to PyVista, and they have to survive in the description or the renderer could not redraw the
layer. They live in :attr:`~digitalearth.base.spec.Symbology.props`, which holds JSON values — so an array
among them (a point cloud's `values`, a vector field's arrows) is stored as a **reference** to the object
rather than as a few hundred thousand numbers: :func:`ref` puts it in the process-local object registry and
:func:`resolved` reads it back. That keeps `to_dict()` small, and keeps what it writes a description rather
than a copy of the data.
"""

from typing import Any, Dict, Mapping

import numpy as np

from digitalearth.base.spec import DataRef, LayerTree, free_layer_id

__all__ = ["REFERENCE_KEY", "next_layer_id", "ref", "resolved"]

#: The key that marks a property holding a reference to an object rather than a value.
REFERENCE_KEY: str = "$ref"


def next_layer_id(tree: LayerTree, kind: str, name: Any = None) -> str:
    """Return the id a new layer takes.

    Args:
        tree: The layers already in the scene, whose ids the new one must not clash with.
        kind: The layer's kind, used as the stem of a generated id.
        name: The caller's name for the layer, if any.

    Returns:
        `name` when it is a non-empty string and free, `name` suffixed (`"wells-2"`) when it is taken —
        through :func:`~digitalearth.base.spec.layer.free_layer_id`, the rule all four tiers share (#321) —
        else `"<kind>-<n>"` with the lowest `n` that is free. A `custom:pyvista` kind generates `custom-1`, since a
        colon cannot appear in the middle of an id.

    Examples:
        - An unnamed layer is numbered by its kind:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, LayerTree
            >>> from digitalearth.three_d.layer import next_layer_id
            >>> next_layer_id(LayerTree(), "terrain")
            'terrain-1'

            ```
        - A name already on the scene is suffixed rather than replacing what is there:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, LayerTree
            >>> from digitalearth.three_d.layer import next_layer_id
            >>> tree = LayerTree().add(LayerSpec("dem", "terrain"))
            >>> next_layer_id(tree, "terrain", "dem")
            'dem-2'

            ```
    """
    taken = set(tree.ids)
    if isinstance(name, str) and name.strip():
        return free_layer_id(name, taken.__contains__)
    stem = kind.split(":")[0]
    number = 1
    while f"{stem}-{number}" in taken:
        number += 1
    return f"{stem}-{number}"


def ref(obj: Any) -> Dict[str, str]:
    """Return a property value that points at `obj` instead of copying it.

    Args:
        obj: The object to reference — an array of per-point values, a field of arrows.

    Returns:
        ``{"$ref": "object:<id>"}``, a JSON value a figure can store. The object stays in this process; a
        figure written to disk carries the reference, and reading it back in another process finds nothing,
        exactly as a custom layer's object does (:mod:`digitalearth.base.custom`).

    Examples:
        - An array is stored as a reference, and read back as itself:
            ```python
            >>> import numpy as np
            >>> from digitalearth.three_d.layer import ref, resolved
            >>> stored = ref(np.array([1.0, 2.0, 3.0]))
            >>> sorted(stored)
            ['$ref']
            >>> resolved(stored).tolist()
            [1.0, 2.0, 3.0]

            ```
    """
    return {REFERENCE_KEY: DataRef.to_object(obj).uri}


def resolved(value: Any) -> Any:
    """Return what a property value stands for, following a reference when it is one.

    Args:
        value: A property value, as stored in a layer's `Symbology.props`.

    Returns:
        The referenced object for ``{"$ref": ...}``, else `value` unchanged. A list is returned as a list, so a
        `levels=[0.5]` that a figure stored as a tuple reaches PyVista in the shape it expects.

    Raises:
        KeyError: if the reference names an object this process does not hold — a figure read back where the
            array it points at was never registered.

    Examples:
        - A plain value passes through:
            ```python
            >>> from digitalearth.three_d.layer import resolved
            >>> resolved("terrain")
            'terrain'

            ```
        - A stored tuple comes back as a list, which is what PyVista's keywords take:
            ```python
            >>> from digitalearth.three_d.layer import resolved
            >>> resolved((0.25, 0.5))
            [0.25, 0.5]

            ```
    """
    if isinstance(value, Mapping) and set(value) == {REFERENCE_KEY}:
        return DataRef(value[REFERENCE_KEY]).open()
    if isinstance(value, tuple):
        return list(value)
    return value


def drawing_props(props: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a layer's stored properties as the keywords its builder was given.

    Args:
        props: The layer's `Symbology.props`.

    Returns:
        A fresh dict with every reference resolved and every stored tuple back as a list.

    Examples:
        - What a builder stored is what the renderer passes on:
            ```python
            >>> import numpy as np
            >>> from digitalearth.three_d.layer import drawing_props, ref
            >>> stored = {"cmap": "terrain", "values": ref(np.array([1.0, 2.0]))}
            >>> drawn = drawing_props(stored)
            >>> drawn["cmap"], drawn["values"].tolist()
            ('terrain', [1.0, 2.0])

            ```
    """
    return {key: resolved(value) for key, value in dict(props).items()}


def stored_props(**kwargs: Any) -> Dict[str, Any]:
    """Return keywords in the form a layer stores them, referencing arrays rather than copying them.

    Args:
        **kwargs: The builder's own keywords.

    Returns:
        A dict where every numpy array is a reference (:func:`ref`) and everything else is as given.

    Examples:
        - A colormap is stored as it is; an array of values is stored as a reference:
            ```python
            >>> import numpy as np
            >>> from digitalearth.three_d.layer import stored_props
            >>> stored = stored_props(cmap="terrain", values=np.array([1.0, 2.0]))
            >>> stored["cmap"], sorted(stored["values"])
            ('terrain', ['$ref'])

            ```
    """
    return {
        key: ref(value) if isinstance(value, np.ndarray) else value
        for key, value in kwargs.items()
    }
