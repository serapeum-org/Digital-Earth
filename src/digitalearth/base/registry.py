"""Open registries — how a reference becomes data, without `base/` knowing any reader.

A :class:`~digitalearth.base.spec.dataref.DataRef` names data; something has to turn that name into an object.
That something is a **resolver**, keyed by URI scheme, and this module is where resolvers are registered and
looked up.

Two kinds of entry get in:

* **Built in** — registered by the package at import time (the default file resolver, which hands the path to
  pyramids).
* **Third party** — declared under the ``digitalearth.sources`` entry-point group, which
  :mod:`digitalearth.ops.plugins` already discovers. That group name is a **public contract string** a plugin
  writes verbatim in its own ``pyproject.toml``; it is deliberately not tied to where our code lives.

The in-memory case has its own scheme. A pyramids object a caller already holds should not have to be written
to disk to be referenced, so :func:`register_object` puts it in a process-local table and hands back an
``object:`` URI that resolves to it. That is what lets a figure describe layers over an object built in a
notebook.

Nothing here imports a renderer, or pyramids at module level: the default resolver reaches for pyramids lazily
so `base/` stays importable without it.
"""

import uuid
from typing import Any, Callable, Dict

__all__ = [
    "OBJECT_SCHEME",
    "get_classifier",
    "register_classifier",
    "SOURCES_GROUP",
    "clear_objects",
    "register_object",
    "register_resolver",
    "resolve_uri",
    "resolvers",
]

#: Entry-point group third-party resolvers declare themselves under. Shared verbatim with
#: :data:`digitalearth.ops.plugins.GROUPS`, because it is the same public contract string.
SOURCES_GROUP: str = "digitalearth.sources"

#: URI scheme for an object held in this process rather than on disk.
OBJECT_SCHEME: str = "object"

_RESOLVERS: Dict[str, Callable[[str], Any]] = {}
_CLASSIFIER: Dict[str, Callable[..., Any]] = {}
_OBJECTS: Dict[str, Any] = {}


def register_resolver(scheme: str, resolver: Callable[[str], Any]) -> None:
    """Register the callable that opens URIs of one scheme.

    Args:
        scheme: The URI scheme, without ``://`` — ``"file"``, ``"s3"``, ``"object"``.
        resolver: Called with the full URI, returning the opened object.

    Raises:
        ValueError: if `scheme` is empty. A resolver under no scheme could never be reached, and registering
            one is more likely a typo than an intent.

    Examples:
        - Register a resolver and read it back:
            ```python
            >>> from digitalearth.base.registry import register_resolver, resolvers
            >>> register_resolver("demo", lambda uri: uri.upper())
            >>> "demo" in resolvers()
            True

            ```
    """
    if not scheme:
        raise ValueError("a resolver needs a non-empty scheme")
    _RESOLVERS[scheme] = resolver


def resolvers() -> Dict[str, Callable[[str], Any]]:
    """Return the registered resolvers, keyed by scheme.

    Returns:
        A copy of the table, so a caller inspecting it cannot edit the registry by accident.

    Examples:
        - Reach the resolver a scheme is served by:
            ```python
            >>> from digitalearth.base.registry import resolvers
            >>> resolvers()["object"].__name__
            '_resolve_object'

            ```
        - Editing what comes back leaves the registry alone, because it is a copy:
            ```python
            >>> from digitalearth.base.registry import resolvers
            >>> table = resolvers()
            >>> del table["file"]
            >>> "file" in resolvers()
            True

            ```
    """
    return dict(_RESOLVERS)


def register_object(obj: Any, *, name: str = "") -> str:
    """Hold an in-memory object in this process and return the URI that reaches it.

    Args:
        obj: The object to register — typically a pyramids ``Dataset`` or ``FeatureCollection``.
        name: An optional stable id. Reusing one replaces what it pointed at, which is how a caller refreshes
            the data behind a figure without rebuilding the figure. Omitted, a unique id is generated.

    Returns:
        An ``object:<id>`` URI.

    Examples:
        - An object gets a URI without touching the filesystem:
            ```python
            >>> from digitalearth.base.registry import register_object, resolve_uri
            >>> uri = register_object([1, 2, 3], name="demo-rows")
            >>> uri
            'object:demo-rows'
            >>> resolve_uri(uri)
            [1, 2, 3]

            ```
    """
    key = name or uuid.uuid4().hex
    _OBJECTS[key] = obj
    return f"{OBJECT_SCHEME}:{key}"


def clear_objects() -> None:
    """Forget every registered in-memory object.

    The table is process-local and holds strong references, so a long-lived session that registers many
    datasets keeps them all alive. This is the release valve, and what a test suite calls between cases.

    Examples:
        - A cleared id no longer resolves:
            ```python
            >>> from digitalearth.base.registry import clear_objects, register_object, resolve_uri
            >>> uri = register_object([1, 2], name="doc-rows")
            >>> resolve_uri(uri)
            [1, 2]
            >>> clear_objects()
            >>> resolve_uri("object:doc-rows")
            Traceback (most recent call last):
                ...
            KeyError: "no object is registered as 'doc-rows'. An object: reference only resolves in the process that registered it — save the data and reference it by path to share the figure"

            ```
    """
    _OBJECTS.clear()


def _resolve_object(uri: str) -> Any:
    """Return the in-memory object a ``object:`` URI names.

    Args:
        uri: The full ``object:<id>`` URI.

    Returns:
        The registered object.

    Raises:
        KeyError: if nothing is registered under that id — including the case where it was registered in
            another process, which is why a figure holding one is not portable.
    """
    key = uri.split(":", 1)[1]
    if key not in _OBJECTS:
        raise KeyError(
            f"no object is registered as {key!r}. An object: reference only resolves in the process that "
            "registered it — save the data and reference it by path to share the figure"
        )
    return _OBJECTS[key]


def _resolve_file(uri: str) -> Any:
    """Open a path through pyramids, choosing the reader by what the path holds.

    Args:
        uri: A filesystem path, or a ``file:`` URI.

    Returns:
        A pyramids ``Dataset`` for a raster, or a ``FeatureCollection`` for a vector.

    Raises:
        Exception: whatever pyramids raises when it can read the path as neither.
    """
    path = uri[len("file:") :] if uri.startswith("file:") else uri
    path = path.lstrip("/") if path.startswith("///") else path
    from pyramids.dataset import Dataset

    try:
        return Dataset.read_file(path)
    except Exception as raster_error:
        from pyramids.feature import FeatureCollection

        try:
            return FeatureCollection.read_file(path)
        except Exception as vector_error:
            raise vector_error from raster_error


def resolve_uri(uri: str) -> Any:
    """Open a URI with the resolver registered for its scheme.

    Args:
        uri: The reference to open.

    Returns:
        The opened object.

    Raises:
        KeyError: if no resolver is registered for the scheme, listing the ones that are — a plugin that
            failed to install looks exactly like a typo otherwise.

    Examples:
        - A bare path uses the default file resolver:
            ```python
            >>> from digitalearth.base.registry import resolve_uri
            >>> resolve_uri("object:missing-id")
            Traceback (most recent call last):
                ...
            KeyError: "no object is registered as 'missing-id'. An object: reference only resolves in the process that registered it — save the data and reference it by path to share the figure"

            ```
    """
    scheme = uri.split(":", 1)[0] if ":" in uri else "file"
    # A Windows drive letter is not a scheme: "C:/data/x.tif" must read as a path, not as scheme "C".
    if len(scheme) == 1:
        scheme = "file"
    if scheme not in _RESOLVERS:
        raise KeyError(
            f"no resolver registered for scheme {scheme!r}; known schemes are "
            f"{sorted(_RESOLVERS)}. A third-party resolver declares itself under the "
            f"{SOURCES_GROUP!r} entry-point group"
        )
    return _RESOLVERS[scheme](uri)


register_resolver("file", _resolve_file)
register_resolver(OBJECT_SCHEME, _resolve_object)


def register_classifier(classifier: Callable[[Any, str, int], Any]) -> None:
    """Register the function that cuts class edges from values.

    `base/` may not import a renderer, and the classifier this package uses lives in cleopatra. So `base/`
    declares the seam and something above it fills it — :mod:`digitalearth` registers a lazily-importing
    adapter at package import. The *policy* around classification (the default class count, the error that
    names the scheme and `k`, turning edges into classes) stays in
    :class:`~digitalearth.base.spec.scale.Scale`; only the arithmetic is injected.

    Args:
        classifier: Called as ``classifier(values, scheme, k)``, returning ``(edges, _)`` — cleopatra's
            ``styles.classify`` signature.

    Examples:
        - Swap in a classifier of your own, then put the package's back:
            ```python
            >>> from digitalearth.base.registry import get_classifier, register_classifier
            >>> original = get_classifier()
            >>> register_classifier(lambda values, scheme, k: ([0.0, 0.5, 1.0], None))
            >>> from digitalearth.base.spec import Scale
            >>> Scale.from_values([0.0, 1.0], scheme="anything").breaks
            (0.0, 0.5, 1.0)
            >>> register_classifier(original)

            ```
    """
    _CLASSIFIER["fn"] = classifier


def get_classifier() -> Callable[..., Any]:
    """Return the registered classifier.

    Returns:
        The function registered by :func:`register_classifier`.

    Raises:
        RuntimeError: if none is registered. That means `digitalearth` was not imported, or something
            replaced its registration — not a caller error, so it says so rather than reading as a bad scheme.

    Examples:
        - Cut class edges with whatever is registered — ``k`` classes give ``k + 1`` edges:
            ```python
            >>> import digitalearth  # registers the adapter
            >>> from digitalearth.base.registry import get_classifier
            >>> edges, _ = get_classifier()([0.0, 5.0, 10.0], "equal_interval", 2)
            >>> [float(edge) for edge in edges]
            [0.0, 5.0, 10.0]

            ```
    """
    if "fn" not in _CLASSIFIER:
        raise RuntimeError(
            "no classifier is registered. digitalearth registers one at import; import the package, or "
            "call base.registry.register_classifier() with a classify(values, scheme, k) callable"
        )
    return _CLASSIFIER["fn"]
