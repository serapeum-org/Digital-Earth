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
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import url2pathname

__all__ = [
    "KIND_BANDS",
    "KIND_PATTERN",
    "KIND_TAKES",
    "KindInfo",
    "band_of",
    "is_kind_name",
    "kind_info",
    "kinds",
    "register_kind",
    "temporary_kind",
    "FURNITURE_ANCHORS",
    "FurnitureInfo",
    "furniture_info",
    "furniture_kinds",
    "register_furniture",
    "temporary_furniture",
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


# ---------------------------------------------------------------------------------------------------------------------
# Layer kinds
# ---------------------------------------------------------------------------------------------------------------------

#: What a layer kind may be spelled as: a lowercase identifier (a letter, then letters, digits, `_` or `-`),
#: optionally after **one** namespace of the same form and a colon — `points`, `custom:pyvista`,
#: `mypkg:hexbin`. The namespace keeps a plugin's kinds and an engine's custom layers out of the built-in names.
KIND_PATTERN = re.compile(r"(?:[a-z][a-z0-9_-]*:)?[a-z][a-z0-9_-]*")

#: The data a kind draws. A renderer reads it to know which extractor a layer's source goes through.
KIND_TAKES = ("raster", "points", "lines", "polygons", "mesh", "volume", "none")

_KINDS: Dict[str, "KindInfo"] = {}


def is_kind_name(value: Any) -> bool:
    """Whether `value` is spelled as a layer kind.

    Args:
        value: The candidate name.

    Returns:
        `True` for a string matching :data:`KIND_PATTERN` in full; `False` for anything else, a non-string
        included.

    Examples:
        - A plain and a namespaced kind are names; an upper-case one and a doubly namespaced one are not:
            ```python
            >>> from digitalearth.base.registry import is_kind_name
            >>> [is_kind_name(n) for n in ("points", "custom:pyvista", "Points", "a:b:c")]
            [True, True, False, False]

            ```
        - A value that is not a string is never a kind name, so it is answered rather than raising:
            ```python
            >>> from digitalearth.base.registry import is_kind_name
            >>> is_kind_name(42), is_kind_name(None)
            (False, False)

            ```
    """
    return isinstance(value, str) and KIND_PATTERN.fullmatch(value) is not None


#: The draw-order bands a layer kind belongs to, **bottom first**. A band says where a kind sits relative to the
#: data, which every tier already decides its own way: the web tier counts underlays and references onto its layer
#: queue, the interactive tier inserts ground cover at position 0, and the static tier spells the same thing as
#: `zorder`. `LayerTree.add` places a new layer at the top of its band, so a graticule added last still draws
#: beneath the data and a label added first still draws over it.
#:
#: * `underlay` — ground cover the data is drawn on: tiles, land, ocean, lakes.
#: * `reference` — geography drawn over the ground but under the data, so it locates the map without hiding it.
#: * `data` — what the figure is about. The default for a kind that does not say, and for an unregistered one.
#: * `overlay` — drawn over the data: text and labels, and the line geography (coastlines, borders, rivers) the
#:   static and interactive tiers draw above the field so it stays visible over an opaque raster.
KIND_BANDS: Tuple[str, ...] = ("underlay", "reference", "data", "overlay")


@dataclass(frozen=True)
class KindInfo:
    """One registered layer kind: its name, the data it draws, where it is drawn, and what it is.

    Attributes:
        name: The kind, as a `LayerSpec` writes it.
        takes: The data class a layer of this kind draws — one of :data:`KIND_TAKES`.
        doc: A one-line description, shown wherever the registry is listed.
        band: Where a layer of this kind is drawn relative to the data — one of :data:`KIND_BANDS`. `"data"` by
            default, which is what a kind that draws the figure's subject wants.

    Raises:
        ValueError: for a name that is not spelled as a kind, a `takes` outside :data:`KIND_TAKES`, a `band`
            outside :data:`KIND_BANDS`, or an empty description — each refused where the entry is built rather
            than when a renderer looks it up.

    Examples:
        - Build an entry for a plugin's kind, drawn among the data:
            ```python
            >>> from digitalearth.base.registry import KindInfo
            >>> entry = KindInfo("mypkg:hexbin", "points", "binned point density")
            >>> entry.takes, entry.band
            ('points', 'data')

            ```
        - A name that could not be looked up is refused where the entry is built:
            ```python
            >>> from digitalearth.base.registry import KindInfo
            >>> KindInfo("Hexbin", "points", "binned point density")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: 'Hexbin' is not a layer-kind name: use a lowercase identifier, ...

            ```
    """

    name: str
    takes: str
    doc: str
    band: str = "data"

    def __post_init__(self) -> None:
        """Refuse an entry that could not be looked up, drawn, or placed.

        Raises:
            ValueError: as described on the class.
        """
        if not is_kind_name(self.name):
            raise ValueError(
                f"{self.name!r} is not a layer-kind name: use a lowercase identifier, optionally after one "
                "namespace and a colon, such as 'points' or 'mypkg:hexbin'"
            )
        if self.takes not in KIND_TAKES:
            raise ValueError(
                f"layer kind {self.name!r}: takes must be one of {list(KIND_TAKES)}; got {self.takes!r}"
            )
        if self.band not in KIND_BANDS:
            raise ValueError(
                f"layer kind {self.name!r}: band must be one of {list(KIND_BANDS)}; got {self.band!r}"
            )
        if not isinstance(self.doc, str) or not self.doc.strip():
            raise ValueError(f"layer kind {self.name!r} needs a description")


def register_kind(info: KindInfo) -> None:
    """Register a layer kind, so a renderer can look it up by the name a `LayerSpec` holds.

    Args:
        info: The entry to register.

    Raises:
        TypeError: if `info` is not a :class:`KindInfo`.
        ValueError: if a *different* entry is already registered under the same name. Two plugins claiming one
            name would otherwise overwrite each other silently, and whichever imported last would win.
            Registering an identical entry again — a module imported twice — is accepted.

    Examples:
        - Registration is permanent, so the demonstration uses :func:`temporary_kind`:
            ```python
            >>> from digitalearth.base.registry import KindInfo, kind_info, temporary_kind
            >>> with temporary_kind(KindInfo("demo:dots", "points", "a demonstration kind")):
            ...     kind_info("demo:dots").doc
            'a demonstration kind'

            ```
        - A different meaning for a built-in name is refused, and the built-in stays as it was:
            ```python
            >>> from digitalearth.base.registry import KindInfo, kind_info, register_kind
            >>> register_kind(KindInfo("raster", "points", "another meaning"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: layer kind 'raster' is already registered as KindInfo(name='raster', ...
            >>> kind_info("raster").takes
            'raster'

            ```
    """
    if not isinstance(info, KindInfo):
        raise TypeError(f"register_kind needs a KindInfo; got {type(info).__name__}")
    held = _KINDS.get(info.name)
    if held is not None and held != info:
        raise ValueError(
            f"layer kind {info.name!r} is already registered as {held!r}; register a namespaced kind "
            "such as 'mypkg:<name>' instead"
        )
    _KINDS[info.name] = info


def kind_info(name: str) -> KindInfo:
    """Return the registered entry for a layer kind.

    Args:
        name: The kind, as a `LayerSpec` holds it.

    Returns:
        The :class:`KindInfo` registered under `name`.

    Raises:
        KeyError: for a kind nobody registered, naming the kinds that are.

    Examples:
        - Look up a built-in kind:
            ```python
            >>> from digitalearth.base.registry import kind_info
            >>> kind_info("choropleth").takes
            'polygons'

            ```
        - A misspelt kind is refused with the registered names in the message:
            ```python
            >>> from digitalearth.base.registry import kind_info
            >>> kind_info("polygon")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "no layer kind 'polygon' is registered; registered kinds are ['basemap', 'borders', ...]"

            ```
    """
    try:
        return _KINDS[name]
    except (KeyError, TypeError):
        raise KeyError(
            f"no layer kind {name!r} is registered; registered kinds are {sorted(_KINDS)}"
        ) from None


def kinds() -> Tuple[str, ...]:
    """Return the registered layer kinds, sorted.

    Returns:
        The names, as a tuple, so the listing cannot be used to edit the registry.

    Examples:
        - The built-in vocabulary is registered at import:
            ```python
            >>> from digitalearth.base.registry import kinds
            >>> "polygons" in kinds() and "choropleth" in kinds()
            True

            ```
        - The names come back sorted:
            ```python
            >>> from digitalearth.base.registry import kinds
            >>> kinds()[:3]
            ('basemap', 'borders', 'choropleth')

            ```
    """
    return tuple(sorted(_KINDS))


def band_of(kind: str) -> str:
    """Return the draw-order band a layer kind belongs to.

    Args:
        kind: The kind, as a `LayerSpec` holds it.

    Returns:
        One of :data:`KIND_BANDS`. A kind nobody registered — a plugin's, or one whose plugin is not installed —
        is `"data"`, so an unknown layer is drawn where the figure is about rather than under the basemap.

    Examples:
        - The built-in vocabulary already sorts a map into bands:
            ```python
            >>> from digitalearth.base.registry import band_of
            >>> [band_of(kind) for kind in ("basemap", "graticule", "raster", "labels")]
            ['underlay', 'reference', 'data', 'overlay']

            ```
        - An unregistered kind is data:
            ```python
            >>> from digitalearth.base.registry import band_of
            >>> band_of("mypkg:hexbin")
            'data'

            ```
    """
    info = _KINDS.get(kind) if isinstance(kind, str) else None
    return "data" if info is None else info.band


@contextmanager
def temporary_kind(info: KindInfo) -> Iterator[None]:
    """Register a layer kind for the duration of a block, then put the registry back as it was.

    Unlike :func:`register_kind`, this may replace an entry already registered under the same name — that is
    what a test swapping a built-in needs — and the original is restored when the block ends.

    Args:
        info: The entry to install for the block.

    Yields:
        Nothing; the entry is registered inside the block.

    Raises:
        TypeError: if `info` is not a :class:`KindInfo`.

    Examples:
        - The kind resolves inside the block and is gone afterwards:
            ```python
            >>> from digitalearth.base.registry import KindInfo, kinds, temporary_kind
            >>> with temporary_kind(KindInfo("demo:scratch", "none", "a scratch kind")):
            ...     "demo:scratch" in kinds()
            True
            >>> "demo:scratch" in kinds()
            False

            ```
        - Swapping a built-in for a block puts the built-in back afterwards:
            ```python
            >>> from digitalearth.base.registry import KindInfo, kind_info, temporary_kind
            >>> with temporary_kind(KindInfo("points", "points", "a stand-in description")):
            ...     kind_info("points").doc
            'a stand-in description'
            >>> kind_info("points").doc
            'point features — static scatter/grid_points, interactive/web points'

            ```
    """
    if not isinstance(info, KindInfo):
        raise TypeError(f"temporary_kind needs a KindInfo; got {type(info).__name__}")
    previous = _KINDS.get(info.name)
    _KINDS[info.name] = info
    try:
        yield
    finally:
        if previous is None:
            _KINDS.pop(info.name, None)
        else:
            _KINDS[info.name] = previous


#: Where an anchored item sits in the panel frame. The four corners are what every tier can place: matplotlib's
#: `loc`, MapLibre's control positions and PyVista's corner widgets all name them, so a figure that asks for one
#: is drawable everywhere rather than on the tier it was written for.
FURNITURE_ANCHORS: Tuple[str, ...] = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
)

_FURNITURE: Dict[str, "FurnitureInfo"] = {}


@dataclass(frozen=True)
class FurnitureInfo:
    """One registered piece of panel furniture: an item fixed to the frame rather than drawn in map coordinates.

    A scale bar, a north arrow, a navigation control or a time slider does not move when the map is panned, so it
    is not a layer. It is registered here and placed by :class:`~digitalearth.base.spec.Furniture` on a panel.

    Attributes:
        name: The kind, spelled as a layer kind is (:data:`KIND_PATTERN`).
        anchor: Where it sits unless the item says otherwise — one of :data:`FURNITURE_ANCHORS`.
        doc: A one-line description, shown wherever the registry is listed.

    Raises:
        ValueError: for a name that is not spelled as a kind, an anchor outside :data:`FURNITURE_ANCHORS`, or an
            empty description.

    Examples:
        - Read where a built-in item sits by default:
            ```python
            >>> from digitalearth.base.registry import furniture_info
            >>> furniture_info("scale_bar").anchor
            'bottom-left'

            ```
        - An anchor no tier could place is refused where the entry is built:
            ```python
            >>> from digitalearth.base.registry import FurnitureInfo
            >>> FurnitureInfo("compass", "middle", "a compass")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: furniture 'compass': anchor must be one of ['top-left', ...]; got 'middle'

            ```
    """

    name: str
    anchor: str
    doc: str

    def __post_init__(self) -> None:
        """Refuse an entry that could not be looked up or placed.

        Raises:
            ValueError: as described on the class.
        """
        if not is_kind_name(self.name):
            raise ValueError(
                f"{self.name!r} is not a furniture name: use a lowercase identifier, optionally after one "
                "namespace and a colon, such as 'scale_bar' or 'mypkg:compass'"
            )
        if self.anchor not in FURNITURE_ANCHORS:
            raise ValueError(
                f"furniture {self.name!r}: anchor must be one of {list(FURNITURE_ANCHORS)}; got {self.anchor!r}"
            )
        if not isinstance(self.doc, str) or not self.doc.strip():
            raise ValueError(f"furniture {self.name!r} needs a description")


def register_furniture(info: FurnitureInfo) -> None:
    """Register a piece of panel furniture, so a panel can ask for it by name.

    Args:
        info: The entry to register.

    Raises:
        TypeError: if `info` is not a :class:`FurnitureInfo`.
        ValueError: if a *different* entry is already registered under the same name, exactly as
            :func:`register_kind` refuses one.

    Examples:
        - Registration is permanent, so the demonstration uses :func:`temporary_furniture`:
            ```python
            >>> from digitalearth.base.registry import FurnitureInfo, furniture_info, temporary_furniture
            >>> with temporary_furniture(FurnitureInfo("demo:compass", "top-left", "a demonstration compass")):
            ...     furniture_info("demo:compass").doc
            'a demonstration compass'

            ```
        - A different meaning for a built-in name is refused, and the built-in stays as it was:
            ```python
            >>> from digitalearth.base.registry import FurnitureInfo, furniture_info, register_furniture
            >>> register_furniture(FurnitureInfo("scale_bar", "top-left", "another meaning"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: furniture 'scale_bar' is already registered as FurnitureInfo(name='scale_bar', ...
            >>> furniture_info("scale_bar").anchor
            'bottom-left'

            ```
    """
    if not isinstance(info, FurnitureInfo):
        raise TypeError(
            f"register_furniture needs a FurnitureInfo; got {type(info).__name__}"
        )
    held = _FURNITURE.get(info.name)
    if held is not None and held != info:
        raise ValueError(
            f"furniture {info.name!r} is already registered as {held!r}; register a namespaced name "
            "such as 'mypkg:<name>' instead"
        )
    _FURNITURE[info.name] = info


def furniture_info(name: str) -> FurnitureInfo:
    """Return the registered entry for a piece of panel furniture.

    Args:
        name: The kind, as a `Furniture` holds it.

    Returns:
        The :class:`FurnitureInfo` registered under `name`.

    Raises:
        KeyError: for furniture nobody registered, naming what is.

    Examples:
        - Look up a built-in item:
            ```python
            >>> from digitalearth.base.registry import furniture_info
            >>> furniture_info("north_arrow").doc
            'an arrow pointing to north — static north_arrow'

            ```
        - A misspelt name is refused with the registered names in the message:
            ```python
            >>> from digitalearth.base.registry import furniture_info
            >>> furniture_info("scalebar")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "no furniture 'scalebar' is registered; registered furniture is ['attribution', ...]"

            ```
    """
    try:
        return _FURNITURE[name]
    except (KeyError, TypeError):
        raise KeyError(
            f"no furniture {name!r} is registered; registered furniture is {sorted(_FURNITURE)}"
        ) from None


def furniture_kinds() -> Tuple[str, ...]:
    """Return the registered furniture names, sorted.

    Returns:
        The names, as a tuple, so the listing cannot be used to edit the registry.

    Examples:
        - The built-in vocabulary is registered at import:
            ```python
            >>> from digitalearth.base.registry import furniture_kinds
            >>> "scale_bar" in furniture_kinds() and "time_slider" in furniture_kinds()
            True

            ```
        - The names come back sorted:
            ```python
            >>> from digitalearth.base.registry import furniture_kinds
            >>> furniture_kinds()[:3]
            ('attribution', 'fullscreen', 'layer_switcher')

            ```
    """
    return tuple(sorted(_FURNITURE))


@contextmanager
def temporary_furniture(info: FurnitureInfo) -> Iterator[None]:
    """Register a piece of furniture for the duration of a block, then put the registry back as it was.

    Unlike :func:`register_furniture`, this may replace an entry already registered under the same name, and the
    original is restored when the block ends.

    Args:
        info: The entry to install for the block.

    Yields:
        Nothing; the entry is registered inside the block.

    Raises:
        TypeError: if `info` is not a :class:`FurnitureInfo`.

    Examples:
        - The name resolves inside the block and is gone afterwards:
            ```python
            >>> from digitalearth.base.registry import FurnitureInfo, furniture_kinds, temporary_furniture
            >>> with temporary_furniture(FurnitureInfo("demo:inset", "top-right", "a scratch inset map")):
            ...     "demo:inset" in furniture_kinds()
            True
            >>> "demo:inset" in furniture_kinds()
            False

            ```
    """
    if not isinstance(info, FurnitureInfo):
        raise TypeError(
            f"temporary_furniture needs a FurnitureInfo; got {type(info).__name__}"
        )
    previous = _FURNITURE.get(info.name)
    _FURNITURE[info.name] = info
    try:
        yield
    finally:
        if previous is None:
            _FURNITURE.pop(info.name, None)
        else:
            _FURNITURE[info.name] = previous


#: The furniture every tier draws in the panel frame. Each entry's anchor is where the item sits unless the panel
#: says otherwise; the tier that cannot draw one declares it absent (#294) and skips it rather than failing.
_BUILT_IN_FURNITURE = (
    (
        "scale_bar",
        "bottom-left",
        "a bar showing how far a screen distance is — static scalebar, web scale_bar",
    ),
    ("north_arrow", "top-right", "an arrow pointing to north — static north_arrow"),
    ("attribution", "bottom-right", "the data and tile credits — web attribution"),
    ("navigation", "top-right", "zoom and compass buttons — web navigation"),
    ("fullscreen", "top-right", "a button that fills the screen — web fullscreen"),
    (
        "layer_switcher",
        "top-right",
        "a list of the layers, to switch each on and off — web controls",
    ),
    (
        "time_slider",
        "bottom-left",
        "a slider over the time steps — web timeslider, interactive player",
    ),
    ("measure", "top-left", "a tool that measures distances — web measure"),
)

for _name, _anchor, _doc in _BUILT_IN_FURNITURE:
    register_furniture(FurnitureInfo(_name, _anchor, _doc))

#: The engine-neutral vocabulary every tier writes. Kind names are nouns for *what is drawn*, not method names or
#: engine layer types; the builders that draw each one today are listed in its description.
_BUILT_IN_KINDS = (
    (
        "raster",
        "raster",
        "a band drawn as an image — static imshow, interactive image/large_image, web add_raster",
        "data",
    ),
    (
        "mesh",
        "raster",
        "a band drawn as cells — static pcolormesh/block, interactive quadmesh",
        "data",
    ),
    (
        "rgb",
        "raster",
        "a multi-band colour composite — rgb_composite, static hsv_composite, interactive rgb",
        "data",
    ),
    (
        "contours",
        "raster",
        "iso-value lines traced from a band — static contour, interactive/web contours",
        "data",
    ),
    (
        "filled_contours",
        "raster",
        "bands between iso-values — static contourf, interactive filled_contours",
        "data",
    ),
    (
        "vectors",
        "raster",
        "a u/v vector field — static quiver/barbs, interactive vectorfield/barbs, 3-D vectors",
        "data",
    ),
    (
        "streamlines",
        "raster",
        "flow lines through a u/v field — static streamplot, interactive streamlines",
        "data",
    ),
    (
        "terrain",
        "raster",
        "a DEM drawn as a surface — 3-D terrain, web terrain",
        "data",
    ),
    (
        "volume",
        "volume",
        "a 3-D cube, ray-cast — 3-D volume",
        "data",
    ),
    (
        "isosurface",
        "volume",
        "the surface at one value of a cube — 3-D isosurface",
        "data",
    ),
    (
        "points",
        "points",
        "point features — static scatter/grid_points, interactive/web points",
        "data",
    ),
    (
        "point_cloud",
        "points",
        "positioned 3-D points — 3-D point_cloud, web point_cloud",
        "data",
    ),
    (
        "heatmap",
        "points",
        "point density — static kde, web heatmap",
        "data",
    ),
    (
        "clusters",
        "points",
        "points grouped by proximity — web cluster",
        "data",
    ),
    (
        "labels",
        "points",
        "text taken from a feature column — interactive/web labels",
        "overlay",
    ),
    (
        "lines",
        "lines",
        "line features — interactive path, web lines",
        "data",
    ),
    (
        "flow",
        "lines",
        "flows between places — static sankey",
        "data",
    ),
    (
        "polygons",
        "polygons",
        "polygon features, outlined or flat-filled — static shapes, interactive/web polygons",
        "data",
    ),
    (
        "choropleth",
        "polygons",
        "polygons coloured by a column — choropleth, static grid_cells/cartogram",
        "data",
    ),
    (
        "extrusion",
        "polygons",
        "polygons raised by a column — 3-D extruded_polygons, web extrusion",
        "data",
    ),
    (
        "unstructured",
        "mesh",
        "a UGRID mesh — static tripcolor/tricontour/tricontourf",
        "data",
    ),
    (
        "text",
        "none",
        "a string placed at a coordinate — text, annotate",
        "overlay",
    ),
    (
        "graticule",
        "none",
        "meridians and parallels — graticule",
        "reference",
    ),
    (
        "basemap",
        "none",
        "tiles drawn under the data — basemap, tiles, static stock_img",
        "underlay",
    ),
    (
        "coastlines",
        "none",
        "Natural Earth coastlines — coastlines",
        "overlay",
    ),
    (
        "borders",
        "none",
        "Natural Earth country borders — borders",
        "overlay",
    ),
    (
        "land",
        "none",
        "Natural Earth land — land",
        "underlay",
    ),
    (
        "ocean",
        "none",
        "Natural Earth ocean — ocean",
        "underlay",
    ),
    (
        "lakes",
        "none",
        "Natural Earth lakes — lakes",
        "underlay",
    ),
    (
        "rivers",
        "none",
        "Natural Earth rivers — rivers",
        "overlay",
    ),
    (
        "model",
        "none",
        "a 3-D model placed on the map — web gltf",
        "data",
    ),
)

for _name, _takes, _doc, _band in _BUILT_IN_KINDS:
    register_kind(KindInfo(_name, _takes, _doc, _band))
