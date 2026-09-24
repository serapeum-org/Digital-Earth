"""Custom layers — a caller's own engine object, described well enough to address it.

Every tier lets a caller hand it something the package did not build: a PyVista mesh, a HoloViews element, a
MapLibre layer, a callable that draws one. That object cannot be described — it has no `to_dict`, no CRS, no
symbology — but the *layer* it draws can be: an id, a kind naming the engine that made it, a label, a band, and
whether it is visible. That is enough to show it, hide it, remove it, reorder it and save the figure around it.

The kind is ``custom:<engine>`` — `custom:pyvista`, `custom:holoviews`, `custom:maplibre` — which is why the kind
grammar allows one namespace (#288). The object itself stays with the renderer that was handed it, in a plain
``{layer_id: object}`` table, and is never written to the figure.

So a figure can name a layer whose object is not here: one loaded from a dict, or one handed to another backend.
:func:`held_object` is the single rule for that case, and it says which of the two it is. The tier then answers
it the way contract C7 asks, stated once in :data:`digitalearth.base.contract_clauses.CLAUSES`. What it raises is
:class:`MissingObject`, which is an :class:`~digitalearth.base.crs.OffLimbError`: one catchable type for a layer
that could not be drawn, whichever way it could not (#325).

**Which half of C7 a missing object is (#320).** This module does not restate the clause; it says only where a
missing object falls under it. A missing object is a *layer with nothing to draw* — a fact about the data, not
about the call — which is why leniency is useful here and ``strict=True`` is how a caller opts out of it. A kind
the tier does not draw at all (``terrain`` asked of the static tier, ``choropleth`` asked of the 3-D one) is the
clause's other half: ``drawer_for`` refuses it by name before anything reaches this module. That one is a caller
error, and a silent one costs more than it saves — the figure claims a layer, nothing appears, nothing says why.
"""

from typing import Any, Mapping, Optional

from digitalearth.base.crs import OffLimbError
from digitalearth.base.registry import KIND_PATTERN

__all__ = [
    "CUSTOM_PREFIX",
    "MissingObject",
    "custom_engine",
    "custom_kind",
    "held_object",
    "is_custom",
]

#: What a custom layer's kind starts with. The rest of the kind names the engine that built the object.
CUSTOM_PREFIX: str = "custom:"


class MissingObject(OffLimbError, LookupError):
    """Raised when a custom layer's engine object is not here to draw.

    **An `OffLimbError`, so every tier answers "this layer could not be drawn" with one catchable type
    (#325).** A caller does not care *how* a layer came to have nothing in it — off-limb data, an empty
    geometry set, an engine object a figure read back from a dict cannot carry — only that the layer drew
    nothing and whether that is fatal. The 3-D and web tiers already said so: they catch this and re-raise
    through their own ``_skip_empty``/``_skipped``, which raises `OffLimbError` under ``strict``. The static
    tier re-raises *this class*, and static is the default tier — so before this base was added,
    ``except OffLimbError`` covered three tiers out of four and silently missed the one most callers use.

    **The base is not only for a caller's own `except`.** Every ``except OffLimbError`` already written in
    the package now reaches this too, and one of them is a guard the interactive tier puts on its builders
    (``_skips_off_limb``). So a builder that meets a missing object skips the layer with a warning on a
    lenient map and re-raises *this class*, with its own message, on a ``strict=True`` one — measured, not
    inferred (`R2-L9`). That is the answer this module already prescribes two paragraphs down: a missing
    object is a layer with nothing to draw, a fact about the data, and leniency is the useful default with
    ``strict`` as the way out. It is latent today, because no builder that draws a custom layer carries the
    guard, and `tests/base/test_undrawable_layer.py` pins it for the day one does.

    **Still a `LookupError`**, because that is what it is: a lookup in the renderer's table of held objects
    that came back empty. Nothing in the package catches it that way today — it is kept because narrowing a
    public exception's bases breaks callers that nothing in this repo can see, and because the reading is
    true. The two bases linearise to `MissingObject` -> `OffLimbError` -> `RuntimeError` -> `LookupError` ->
    `Exception`, so a clause naming any of them catches it.

    It is still an error a tier is expected to catch and turn into its own skip-or-raise answer, rather than
    one that reaches a caller unhandled from a lenient map.

    Examples:
        - One ``except`` clause now reaches both ways a layer can have nothing to draw:
            ```python
            >>> from digitalearth.base.crs import OffLimbError
            >>> from digitalearth.base.custom import MissingObject
            >>> issubclass(MissingObject, OffLimbError), issubclass(MissingObject, LookupError)
            (True, True)

            ```
        - The message says which case it is, so a caller is not left guessing:
            ```python
            >>> from digitalearth.base.custom import MissingObject, held_object
            >>> try:
            ...     held_object("wells", "custom:pyvista", {}, engine="maplibre", backend="web")
            ... except MissingObject as error:
            ...     print(error)
            layer 'wells' is a custom:pyvista layer and cannot be drawn by the web backend

            ```
    """


def custom_kind(engine: str) -> str:
    """Return the layer kind for an object built with `engine`.

    Args:
        engine: The engine that made the object — `"pyvista"`, `"holoviews"`, `"maplibre"`.

    Returns:
        ``"custom:<engine>"``.

    Raises:
        ValueError: if `engine` is not a plain lowercase name, since the result would not be a spellable kind.

    Examples:
        - The kind a PyVista mesh is recorded under:
            ```python
            >>> from digitalearth.base.custom import custom_kind
            >>> custom_kind("pyvista")
            'custom:pyvista'

            ```
        - A name that could not be spelled as a kind is refused here, not by `LayerSpec`:
            ```python
            >>> from digitalearth.base.custom import custom_kind
            >>> custom_kind("PyVista")
            Traceback (most recent call last):
                ...
            ValueError: 'PyVista' is not an engine name: use a lowercase identifier such as 'pyvista'

            ```
    """
    if (
        not isinstance(engine, str)
        or ":" in engine
        or KIND_PATTERN.fullmatch(engine) is None
    ):
        raise ValueError(
            f"{engine!r} is not an engine name: use a lowercase identifier such as 'pyvista'"
        )
    return f"{CUSTOM_PREFIX}{engine}"


def is_custom(kind: Any) -> bool:
    """Whether a layer kind names a caller's own object.

    Args:
        kind: The kind, as a `LayerSpec` holds it.

    Returns:
        `True` for ``"custom:<engine>"``; `False` for anything else, a non-string included.

    Examples:
        - A custom layer and a drawn one:
            ```python
            >>> from digitalearth.base.custom import is_custom
            >>> is_custom("custom:maplibre"), is_custom("raster")
            (True, False)

            ```
        - A namespaced kind that is not custom belongs to a plugin, and is drawn like any other:
            ```python
            >>> from digitalearth.base.custom import is_custom
            >>> is_custom("mypkg:hexbin")
            False

            ```
    """
    return isinstance(kind, str) and kind.startswith(CUSTOM_PREFIX)


def custom_engine(kind: Any) -> Optional[str]:
    """Return the engine a custom layer's object was built with.

    Args:
        kind: The kind, as a `LayerSpec` holds it.

    Returns:
        The engine name, or `None` when `kind` is not a custom layer's.

    Examples:
        - Read which engine holds the object:
            ```python
            >>> from digitalearth.base.custom import custom_engine
            >>> custom_engine("custom:holoviews")
            'holoviews'

            ```
        - A drawn layer has no engine of its own — every tier can draw it:
            ```python
            >>> from digitalearth.base.custom import custom_engine
            >>> print(custom_engine("points"))
            None

            ```
    """
    return kind[len(CUSTOM_PREFIX) :] if is_custom(kind) else None


def held_object(
    layer_id: str,
    kind: str,
    held: Mapping[str, Any],
    *,
    engine: str,
    backend: str,
) -> Any:
    """Return the engine object a custom layer draws, or say why it cannot be drawn.

    Args:
        layer_id: The layer being drawn, named in the message.
        kind: Its kind — ``"custom:<engine>"``.
        held: The renderer's table of objects it was handed, keyed by layer id.
        engine: The engine this renderer draws with — `"pyvista"`, `"holoviews"`, `"maplibre"`.
        backend: What this renderer is called in `quickmap(backend=...)`, for the message.

    Returns:
        The object to draw.

    Raises:
        MissingObject: when this renderer has no object for the layer. There are two ways to get there and the
            message says which: the layer was built with another engine, so no table here could hold it, or it
            is this engine's but the object is not here — the figure was loaded from a dict, and a description
            cannot rebuild a mesh somebody built in a notebook.

    Examples:
        - The object is handed straight back when it is there:
            ```python
            >>> from digitalearth.base.custom import held_object
            >>> held_object("wells", "custom:maplibre", {"wells": "a layer"}, engine="maplibre", backend="web")
            'a layer'

            ```
        - A figure loaded from a dict carries the description but not the object:
            ```python
            >>> from digitalearth.base.custom import MissingObject, held_object
            >>> try:
            ...     held_object("wells", "custom:maplibre", {}, engine="maplibre", backend="web")
            ... except MissingObject as error:
            ...     print(error)
            layer 'wells' is a maplibre object this figure does not carry; add it again on the map that draws it

            ```
    """
    # The engine first: a layer id that collides across tiers would otherwise hand a renderer an
    # object built with another engine, which is the one case this function exists to tell apart (review L4).
    if custom_engine(kind) != engine:
        raise MissingObject(
            f"layer {layer_id!r} is a {kind} layer and cannot be drawn by the {backend} backend"
        )
    try:
        return held[layer_id]
    except KeyError:
        pass
    raise MissingObject(
        f"layer {layer_id!r} is a {engine} object this figure does not carry; add it again on the map that "
        "draws it"
    )
