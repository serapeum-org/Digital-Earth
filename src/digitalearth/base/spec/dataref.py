"""A reference to data, so a layer can describe what it draws without holding it.

A layer currently owns its data object. That one fact blocks three things at once:

* **dynamic tiling** for a large raster — the renderer has to be able to ask for a different region at a
  different resolution, which means asking the source again, not slicing what it was handed
* **level of detail and decimation** on the 3-D tier — same reason
* **point clouds** too large to hold eagerly at all

It also makes a scene undescribable. A figure that could round-trip through a dict cannot contain a live
``Dataset``, and MapLibre's cardinal decision — *N layers over one source* — has no expression when every layer
owns a copy.

`DataRef` is the name; :mod:`digitalearth.base.registry` turns a name into data. A figure then holds
``sources: {id: DataRef}`` and a layer holds a `source_id`, which is what makes the source/layer split real.

The payoff lands with `SourceView` in Wave 2: ``SourceView = materialise(DataRef, Selection, ViewRequest)``.
Until then this is the reference and the registry, with no tier obliged to use it yet.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from digitalearth.base.registry import register_object, resolve_uri

__all__ = ["DataRef"]


@dataclass(frozen=True)
class DataRef:
    """Where a layer's data comes from, as a value rather than the data itself.

    Attributes:
        uri: What to open — a filesystem path, a URL, or an ``object:<id>`` naming something registered in
            this process by :meth:`DataRef.to_object`.
        driver: Optional reader hint, for a path whose extension does not settle which reader to use.
        version: Optional version or revision, so a figure can record *which* vintage of a moving dataset it
            described. Carried, not interpreted — a resolver that understands versioning reads it.

    Examples:
        - A path, and the dict a figure would store:
            ```python
            >>> from digitalearth.base.spec import DataRef
            >>> ref = DataRef("data/dem.tif")
            >>> ref.to_dict()
            {'uri': 'data/dem.tif'}

            ```
        - A round trip through a dict, which is what makes a figure serialisable:
            ```python
            >>> from digitalearth.base.spec import DataRef
            >>> DataRef.from_dict({'uri': 's3://bucket/x.tif', 'driver': 'COG'})
            DataRef(uri='s3://bucket/x.tif', driver='COG', version=None)

            ```
        - An in-memory object is referenced without being written to disk:
            ```python
            >>> from digitalearth.base.spec import DataRef
            >>> ref = DataRef.to_object([1, 2, 3], name="demo-doc")
            >>> ref.uri
            'object:demo-doc'
            >>> ref.open()
            [1, 2, 3]

            ```
    """

    uri: str
    driver: Optional[str] = None
    version: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse a reference that names nothing, or holds a hint a figure could not store.

        Raises:
            ValueError: if `uri` is not a string or is empty or blank — an empty reference resolves to whatever the
                working directory happens to be, which is a failure that only shows on someone else's machine —
                or if `driver` or `version` is neither a string nor ``None``. A `pathlib.Path` is refused as a
                `uri` too: pass ``str(path)``.
        """
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError(
                f"DataRef needs uri as a non-empty string; got {self.uri!r}"
            )
        for hint in ("driver", "version"):
            value = getattr(self, hint)
            if value is not None and not isinstance(value, str):
                # Written by `to_dict` as it is, a non-string hint failed inside `json.dumps`, naming neither
                # the type nor the field — or, for a number, read back as a number.
                raise ValueError(
                    f"DataRef needs {hint} as a string or None; got {value!r}"
                )
        if self.uri != self.uri.strip():
            # Blank was already refused; surrounding whitespace was not, and " a.tif" is a path that does
            # not exist on any filesystem that would have opened "a.tif".
            object.__setattr__(self, "uri", self.uri.strip())

    @classmethod
    def to_object(cls, obj: Any, *, name: str = "", **rest: Any) -> "DataRef":
        """Register an in-memory object and return a reference to it.

        Args:
            obj: The object to reference — typically a pyramids ``Dataset`` or ``FeatureCollection``.
            name: An optional stable id; reusing one replaces what it points at.
            **rest: Any other field — `driver`, `version`.

        Returns:
            A reference whose `uri` resolves to `obj` **in this process**. It is deliberately not portable:
            see :func:`digitalearth.base.registry.register_object`.

        Examples:
            - The reference reaches the same object back:
                ```python
                >>> from digitalearth.base.spec import DataRef
                >>> rows = [1, 2]
                >>> DataRef.to_object(rows, name="demo-rows").open() is rows
                True

                ```
        """
        return cls(register_object(obj, name=name), **rest)

    def open(self) -> Any:
        """Open the data this reference names.

        Returns:
            Whatever the registered resolver returns for this URI's scheme — a pyramids ``Dataset`` or
            ``FeatureCollection`` for the built-in file resolver.

        Raises:
            KeyError: if no resolver is registered for the scheme, or an ``object:`` id is not in this
                process.

        Examples:
            - An in-memory reference hands back the data it names:
                ```python
                >>> from digitalearth.base.spec import DataRef
                >>> ref = DataRef.to_object({"rows": 3}, name="doc-open")
                >>> ref.open()["rows"]
                3

                ```
            - A scheme nothing is registered for says so, and lists the ones that are — a plugin that failed
              to install looks exactly like a typo otherwise:
                ```python
                >>> from digitalearth.base.spec import DataRef
                >>> DataRef("weirdscheme://x").open()  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                KeyError: "no resolver registered for scheme 'weirdscheme'; known schemes are [...]..."

                ```
        """
        return resolve_uri(self.uri)

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            The fields that are set. `driver` and `version` are omitted when unset, so the common case is one
            key and a stored figure does not fill with nulls.

        Examples:
            - The common case is a single key:
                ```python
                >>> from digitalearth.base.spec import DataRef
                >>> DataRef("data/dem.tif").to_dict()
                {'uri': 'data/dem.tif'}

                ```
            - Hints appear only when they were set:
                ```python
                >>> from digitalearth.base.spec import DataRef
                >>> stored = DataRef("s3://bucket/x.tif", version="2024-01").to_dict()
                >>> sorted(stored)
                ['uri', 'version']
                >>> stored["version"]
                '2024-01'

                ```
        """
        out: Dict[str, Any] = {"uri": self.uri}
        if self.driver is not None:
            out["driver"] = self.driver
        if self.version is not None:
            out["version"] = self.version
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRef":
        """Rebuild a reference from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reference.

        Raises:
            ValueError: if `uri` is missing or empty, or the mapping carries a key this type does not know —
                silently dropping an unknown key would lose data a newer writer meant to keep.

        Examples:
            - Rebuild a reference a figure stored, and read its fields back:
                ```python
                >>> from digitalearth.base.spec import DataRef
                >>> ref = DataRef.from_dict({"uri": "s3://bucket/x.tif", "driver": "COG"})
                >>> ref.uri, ref.driver
                ('s3://bucket/x.tif', 'COG')

                ```
            - A key this version does not know is refused rather than dropped:
                ```python
                >>> from digitalearth.base.spec import DataRef
                >>> DataRef.from_dict({"uri": "a.tif", "bbox": [0, 0, 1, 1]})
                Traceback (most recent call last):
                    ...
                ValueError: DataRef.from_dict got unknown keys ['bbox']; known keys are ['driver', 'uri', 'version']

                ```
        """
        known = {"uri", "driver", "version"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"DataRef.from_dict got unknown keys {unknown}; known keys are {sorted(known)}"
            )
        return cls(
            uri=data.get("uri", ""),
            driver=data.get("driver"),
            version=data.get("version"),
        )
