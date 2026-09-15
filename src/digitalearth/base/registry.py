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

Nothing here imports a renderer. pyramids is imported inside the default resolver rather than at module level
to keep importing this module cheap — not to make `base/` pyramids-free, which it is not: `base/crs.py` and
`base/sources/extractors.py` both import it at module level and `pyramids-gis` is a hard dependency.
"""

import os
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

__all__ = [
    "OBJECT_SCHEME",
    "temporary_classifier",
    "temporary_resolver",
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
_CLASSIFIER: Optional[Callable[..., Any]] = None
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
        - Register a resolver and reach it through the scheme it serves. A plain call is **permanent** —
          there is no unregister — so this shows it inside :func:`temporary_resolver`, which is what to
          reach for anywhere the registration should not outlive the code that made it:
            ```python
            >>> from digitalearth.base.registry import resolve_uri, resolvers, temporary_resolver
            >>> with temporary_resolver("demo", lambda uri: uri.upper()):
            ...     "demo" in resolvers(), resolve_uri("demo:x")
            (True, 'DEMO:X')
            >>> "demo" in resolvers()
            False

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


def _path_of(uri: str) -> str:
    """Return the filesystem path a ``file:`` URI names, or the string unchanged.

    Args:
        uri: A ``file:`` URI, or a path that is already one.

    Returns:
        The local path. ``url2pathname`` is what knows the per-platform rules — that ``file:///C:/x.tif``
        carries a leading slash Windows must drop but POSIX must keep, and that ``%20`` is a space. Hand-rolling
        that with ``lstrip("/")`` rescues the drive letter and breaks every absolute POSIX path, which is the bug
        this replaced.

    Examples:
        - A POSIX absolute URI keeps the root it names:
            ```python
            >>> import os
            >>> from digitalearth.base.registry import _path_of
            >>> _path_of("file:///home/me/x.tif").replace(os.sep, "/")
            '/home/me/x.tif'

            ```
        - A bare path is already a path:
            ```python
            >>> from digitalearth.base.registry import _path_of
            >>> _path_of("data/dem.tif")
            'data/dem.tif'

            ```
    """
    if not uri.startswith("file:"):
        return uri
    parsed = urlparse(uri)
    path = url2pathname(parsed.path)
    if parsed.netloc:
        # file://server/share/x.tif is a UNC path; the host is part of it, not an authority to drop.
        return f"//{parsed.netloc}{path}"
    return path


def _resolve_file(uri: str) -> Any:
    """Open a path through pyramids, choosing the reader by what the path holds.

    Args:
        uri: A filesystem path, or a ``file:`` URI.

    Returns:
        A pyramids ``Dataset`` for a raster, or a ``FeatureCollection`` for a vector.

    Raises:
        FileNotFoundError: if the path does not exist. Checked before the readers are tried, because a typo
            would otherwise surface as whatever the *vector* reader says about a file that was never there —
            pointing at the format instead of at the path.
        Exception: whatever pyramids raises when the path exists but reads as neither.
    """
    path = _path_of(uri)
    # A GDAL virtual path (/vsizip/, /vsicurl/, …) is not a filesystem entry, so only a plain path is checked.
    if not path.startswith("/vsi") and not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path!r} (from {uri!r})")
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
    # Nor is anything holding a separator. A GDAL virtual path carries its own colon
    # ("/vsicurl/https://host/x.tif"), which split on the first colon yields "/vsicurl/https" — a scheme no
    # plugin could ever register, so the error blamed a missing plugin for what is a path.
    if "/" in scheme or "\\" in scheme:
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
        - Swap in a classifier of your own. :func:`temporary_classifier` is the form to reach for: there
          is one classifier for the whole process, so a replacement that is not restored silently changes
          every later classification:
            ```python
            >>> from digitalearth.base.registry import temporary_classifier
            >>> from digitalearth.base.spec import Scale
            >>> with temporary_classifier(lambda values, scheme, k: ([0.0, 0.5, 1.0], None)):
            ...     Scale.from_values([0.0, 1.0], scheme="anything").breaks
            (0.0, 0.5, 1.0)

            ```
    """
    global _CLASSIFIER
    _CLASSIFIER = classifier


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
    if _CLASSIFIER is None:
        raise RuntimeError(
            "no classifier is registered. digitalearth registers one at import; import the package, or "
            "call base.registry.register_classifier() with a classify(values, scheme, k) callable"
        )
    return _CLASSIFIER


@contextmanager
def temporary_resolver(scheme: str, resolver: Callable[[str], Any]) -> Iterator[None]:
    """Register a resolver for the duration of a block, then put the table back as it was.

    The registry is process-global and has no unregister, so a test or an example that registers a scheme
    changes what every later one sees. This is the scoped form, so demonstrating the registry does not leave
    a scheme behind in the session that ran the demonstration.

    Args:
        scheme: The URI scheme to register under.
        resolver: The resolver to install for the block.

    Yields:
        Nothing; the registration is in effect inside the block.

    Examples:
        - The scheme resolves inside the block and is gone afterwards:
            ```python
            >>> from digitalearth.base.registry import resolve_uri, resolvers, temporary_resolver
            >>> with temporary_resolver("scratch", lambda uri: uri.upper()):
            ...     resolve_uri("scratch:x")
            'SCRATCH:X'
            >>> "scratch" in resolvers()
            False

            ```
    """
    had = scheme in _RESOLVERS
    previous = _RESOLVERS.get(scheme)
    register_resolver(scheme, resolver)
    try:
        yield
    finally:
        if had:
            # `had`, not `previous is not None`: register_resolver validates the scheme but not the
            # resolver, so None can be registered — and testing the value would delete it instead of
            # putting it back.
            _RESOLVERS[scheme] = previous  # type: ignore[assignment]
        else:
            _RESOLVERS.pop(scheme, None)


@contextmanager
def temporary_classifier(classifier: Callable[..., Any]) -> Iterator[None]:
    """Swap the registered classifier for the duration of a block.

    There is one classifier for the process, so replacing it without restoring leaves every later
    classification using the replacement. Where that happens inside a doctest it is worse than a wrong
    answer: the failure surfaces in some unrelated example further down the session.

    Args:
        classifier: The ``classify(values, scheme, k)`` callable to install for the block.

    Yields:
        Nothing; the classifier is in effect inside the block.

    Examples:
        - The swap applies inside and is undone after, even if the block raises:
            ```python
            >>> import digitalearth  # registers the real adapter
            >>> from digitalearth.base.registry import get_classifier, temporary_classifier
            >>> real = get_classifier()
            >>> with temporary_classifier(lambda values, scheme, k: ([0.0, 0.5, 1.0], None)):
            ...     get_classifier()([0.0], "anything", 2)[0]
            [0.0, 0.5, 1.0]
            >>> get_classifier() is real
            True

            ```
    """
    global _CLASSIFIER
    previous = _CLASSIFIER
    _CLASSIFIER = classifier
    try:
        yield
    finally:
        _CLASSIFIER = previous
