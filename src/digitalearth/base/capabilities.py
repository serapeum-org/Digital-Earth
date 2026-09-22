"""What a backend can draw, declared rather than discovered.

Four tiers draw the same figures with different engines, and they do not draw the same things. Today that is
answered in as many places as there are questions: `api.py` keeps one table of the `quickmap` keywords each
backend honours, the web tier refuses a CRS that is not EPSG:4326 in its own words, the interactive tier guards
six call sites with a Web-Mercator check, and nothing anywhere says which visual channels a tier can fold or
which layer kinds it can build. A caller who wants to know asks by trying.

A :class:`Capabilities` is one tier's answer to all of it, as data: the layer kinds it builds, the channels it
folds and which of those may be driven by a field, the classification schemes it offers, the features it has —
a display CRS, a colorbar, a scale bar — and, deliberately, what it does **not** have and why. The last part is
the reason this is a declaration and not a set of `hasattr` probes: a gap that is intentional reads differently
from a gap nobody has filled, and only the tier can say which one it is.

Each tier exports its own `CAPABILITIES` from `<tier>/capabilities.py`, importable without that tier's engine,
so a dispatcher can read every tier's answer before deciding which one to build — and refuse a request the
chosen backend cannot honour, by name, before anything is drawn.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional

from digitalearth.base.registry import KIND_PATTERN, furniture_kinds, kinds
from digitalearth.base.spec.encoding import CHANNELS

__all__ = ["FEATURES", "CapabilityError", "Capabilities", "is_feature"]


class CapabilityError(ValueError):
    """Raised when a backend is asked for something it does not have.

    A `ValueError`, because that is what the dispatcher raised for the same refusal before capabilities were
    declared, and a caller catching `ValueError` around `quickmap` keeps catching it.

    Examples:
        - The message names the capability, the backend and, when the tier said why, the reason:
            ```python
            >>> from digitalearth.base.capabilities import Capabilities, CapabilityError
            >>> flat = Capabilities("flat", absent={"display_crs": "it draws in the data's own CRS"})
            >>> try:
            ...     flat.require("display_crs", caller="quickmap")
            ... except CapabilityError as error:
            ...     print(error)
            quickmap needs 'display_crs', which backend='flat' does not support; it draws in the data's own CRS

            ```
    """


#: What a tier can *have*, beyond the kinds it draws and the channels it folds — name to one line. It is the
#: growth axis for anything that is neither a layer nor a visual variable: a display CRS, a raster renderer, an
#: export target. The furniture kinds (:func:`~digitalearth.base.registry.furniture_kinds`) are features too,
#: under their own names, so a tier declares its scale bar and its zoom buttons the same way it declares a
#: colorbar — and a figure asking for furniture the tier lists as absent is drawn without it, quietly.
FEATURES: Mapping[str, str] = MappingProxyType(
    {
        "display_crs": "every layer is drawn in one CRS, which the caller may choose",
        "domain": "the map can be framed to a named region",
        "coastline_overlay": "a coastline layer can be added to any figure the tier draws",
        "raster_renderer": "a raster can be drawn as an image, as cells or as contours, chosen by name",
        "colorbar": "a continuous colour key",
        "legend": "a keyed list of classes",
        "animation": "a sequence of frames over a dimension",
        "export_image": "the figure can be written as a raster image",
        "export_vector": "the figure can be written as a vector image",
        "export_html": "the figure can be written as a page that opens in a browser",
    }
)


def is_feature(name: Any) -> bool:
    """Whether `name` is a feature a tier can declare.

    Args:
        name: The candidate feature.

    Returns:
        `True` for a key of :data:`FEATURES` or a registered furniture kind; `False` for anything else.

    Examples:
        - A feature and a piece of furniture are both declarable; a layer kind is not:
            ```python
            >>> from digitalearth.base.capabilities import is_feature
            >>> [is_feature(name) for name in ("display_crs", "scale_bar", "raster")]
            [True, True, False]

            ```
    """
    return isinstance(name, str) and (name in FEATURES or name in furniture_kinds())


def _named_set(owner: str, name: str, values: Any) -> FrozenSet[str]:
    """Return a set of names, refusing a bare string and anything that is not a name.

    Args:
        owner: The field's owner, for the message.
        name: The field being read.
        values: What the caller gave.

    Returns:
        The names, frozen.

    Raises:
        ValueError: if `values` is a string — `frozenset("crs")` would be three one-letter capabilities — or
            holds something that is not a non-empty string.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{owner} {name} must be a collection of names; got the string {values!r}"
        )
    frozen = frozenset(values)
    for value in frozen:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{owner} {name} must be names; got {value!r}")
    return frozen


@dataclass(frozen=True)
class Capabilities:
    """One backend's declaration of what it can draw.

    Attributes:
        backend: The tier's name, as `quickmap(backend=...)` spells it — `"matplotlib"`, `"interactive"`,
            `"3d"`, `"web"`.
        kinds: The layer kinds it builds, each registered (:func:`~digitalearth.base.registry.kinds`). A tier
            that holds a caller's own objects lists its `custom:<engine>` kind here too.
        channels: The visual channels it folds, each a key of :data:`~digitalearth.base.spec.encoding.CHANNELS`.
        data_driven: The channels it can drive from a field rather than only from a constant — a subset of
            `channels`, because a channel that cannot be set at all cannot be set from data.
        schemes: The classification schemes it offers: `"categorical"` plus the classifier names it accepts.
            `scheme=None` is continuous everywhere (contract C4) and is not listed.
        features: What it has beyond kinds and channels — keys of :data:`FEATURES`, and the furniture kinds it
            draws.
        absent: What it deliberately does **not** have, name to reason. A name here is disjoint from everything
            declared, and the reason is shown to a caller who asks for it. This is the field that makes the
            declaration honest: "no colorbar" and "no colorbar *yet*" are different answers, and only a reason
            tells them apart.

    Raises:
        ValueError: for a backend that is not a non-empty string, a kind nobody registered, a channel outside
            `CHANNELS`, a `data_driven` channel that is not also in `channels`, a scheme that is not a name, a
            feature that is neither a :data:`FEATURES` key nor a furniture kind, an `absent` entry naming
            something declared, or an `absent` reason that is empty.

    Examples:
        - A tier declares what it draws, and what it does not:
            ```python
            >>> from digitalearth.base.capabilities import Capabilities
            >>> flat = Capabilities(
            ...     "flat",
            ...     kinds={"raster", "points"},
            ...     channels={"color", "size"},
            ...     data_driven={"color"},
            ...     features={"colorbar"},
            ...     absent={"domain": "it has no extent to set"},
            ... )
            >>> flat.supports("points"), flat.supports("domain")
            (True, False)

            ```
        - A channel that can be set but not driven by a field answers both questions separately:
            ```python
            >>> from digitalearth.base.capabilities import Capabilities
            >>> flat = Capabilities("flat", channels={"color", "size"}, data_driven={"color"})
            >>> "size" in flat.channels, "size" in flat.data_driven
            (True, False)

            ```
        - The declaration is checked where it is written, so a typo is not a silent "no":
            ```python
            >>> from digitalearth.base.capabilities import Capabilities
            >>> Capabilities("flat", kinds={"rastre"})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: Capabilities('flat') lists layer kinds nobody registered: ['rastre']...

            ```
    """

    backend: str
    kinds: FrozenSet[str] = frozenset()
    channels: FrozenSet[str] = frozenset()
    data_driven: FrozenSet[str] = frozenset()
    schemes: FrozenSet[str] = frozenset()
    features: FrozenSet[str] = frozenset()
    absent: Mapping[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash the declaration by what it declares.

        A frozen dataclass presents as a value — the tiers' declarations are compared and passed around as
        one — but `absent` is a mapping, which the generated `__hash__` cannot hash. Hashing its items
        instead lets a declaration go in a set or serve as a dict key, which is what "frozen" promised
        (review L2).

        Returns:
            The hash of all seven fields, with `absent` taken as its sorted items.
        """
        return hash(
            (
                self.backend,
                self.kinds,
                self.channels,
                self.data_driven,
                self.schemes,
                self.features,
                tuple(sorted(self.absent.items())),
            )
        )

    def __post_init__(self) -> None:
        """Check every name against the vocabulary it belongs to, then freeze the declaration.

        Raises:
            ValueError: as described on the class.
        """
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError(
                f"Capabilities backend must be a non-empty name; got {self.backend!r}"
            )
        owner = f"Capabilities({self.backend!r})"
        for name in ("kinds", "channels", "data_driven", "schemes", "features"):
            object.__setattr__(self, name, _named_set(owner, name, getattr(self, name)))
        self._check_kinds(owner)
        unknown_channels = sorted(self.channels.difference(CHANNELS))
        if unknown_channels:
            raise ValueError(
                f"{owner} lists channels that are not declared: {unknown_channels}; the channels are "
                f"{sorted(CHANNELS)}"
            )
        undriveable = sorted(self.data_driven.difference(self.channels))
        if undriveable:
            raise ValueError(
                f"{owner} lists {undriveable} as data-driven but does not list them as channels; a channel it "
                "cannot set at all cannot be set from a field"
            )
        for scheme in self.schemes:
            if KIND_PATTERN.fullmatch(scheme) is None:
                raise ValueError(
                    f"{owner} scheme {scheme!r} is not a name: use a lowercase identifier such as 'quantiles'"
                )
        unknown_features = sorted(
            name for name in self.features if not is_feature(name)
        )
        if unknown_features:
            raise ValueError(
                f"{owner} lists features that are neither declared nor furniture: {unknown_features}; the "
                f"features are {sorted(FEATURES)} and the furniture is {list(furniture_kinds())}"
            )
        self._check_absent(owner)

    def _check_kinds(self, owner: str) -> None:
        """Refuse a layer kind nobody registered.

        Args:
            owner: The declaration, for the message.

        Raises:
            ValueError: naming the unregistered kinds. A tier that draws something new registers the kind
                first (#288), so an unknown name here is a typo or a kind whose plugin is not installed —
                either way the declaration would claim something no caller could ask for by name.
        """
        registered = set(kinds())
        unknown = sorted(self.kinds.difference(registered))
        if unknown:
            raise ValueError(
                f"{owner} lists layer kinds nobody registered: {unknown}; register the kind first, or check "
                "the spelling against digitalearth.base.registry.kinds()"
            )

    def _check_absent(self, owner: str) -> None:
        """Refuse an `absent` entry that is empty, unnamed, or contradicts what is declared.

        Args:
            owner: The declaration, for the message.

        Raises:
            ValueError: for a non-mapping, a name that is not a string, an empty reason, or a name that is also
                declared — "we have it and we deliberately do not" is not an answer.
        """
        if not isinstance(self.absent, Mapping):
            raise ValueError(
                f"{owner} absent must be a mapping of name to reason; got {type(self.absent).__name__}"
            )
        entries: Dict[str, str] = {}
        for name, reason in dict(self.absent).items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{owner} absent must be keyed by name; got {name!r}")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"{owner} absent[{name!r}] needs a reason: what the tier does instead, or why it cannot"
                )
            entries[name] = reason
        declared = self.kinds | self.channels | self.schemes | self.features
        contradicted = sorted(set(entries).intersection(declared))
        if contradicted:
            raise ValueError(
                f"{owner} lists {contradicted} as absent and as supported; a declaration says one or the other"
            )
        object.__setattr__(self, "absent", MappingProxyType(entries))

    def supports(self, name: str) -> bool:
        """Whether the backend can honour a capability.

        Args:
            name: A layer kind, a channel, a scheme or a feature.

        Returns:
            `True` when the declaration lists it anywhere.

        Examples:
            - One question, whatever kind of capability is being asked about:
                ```python
                >>> from digitalearth.base.capabilities import Capabilities
                >>> flat = Capabilities("flat", kinds={"raster"}, features={"colorbar"})
                >>> [flat.supports(name) for name in ("raster", "colorbar", "graticule")]
                [True, True, False]

                ```
        """
        return (
            name in self.kinds
            or name in self.channels
            or name in self.schemes
            or name in self.features
        )

    def reason(self, name: str) -> Optional[str]:
        """Return why the backend does not have a capability, when it said.

        Args:
            name: The capability asked about.

        Returns:
            The declared reason, or `None` — either because the capability is supported, or because the tier
            has not said anything about it.

        Examples:
            - A declared absence explains itself; an undeclared one has nothing to say:
                ```python
                >>> from digitalearth.base.capabilities import Capabilities
                >>> flat = Capabilities("flat", absent={"domain": "it has no extent to set"})
                >>> flat.reason("domain")
                'it has no extent to set'
                >>> print(flat.reason("graticule"))
                None

                ```
        """
        return self.absent.get(name)

    def require(self, name: str, *, caller: str) -> None:
        """Refuse a request the backend cannot honour.

        Args:
            name: The capability the caller is asking for.
            caller: What is asking — a builder or `"quickmap"` — so the message says where the request came
                from rather than only what was missing.

        Raises:
            CapabilityError: when the capability is not declared, carrying the reason when the tier gave one.

        Examples:
            - A supported capability passes silently:
                ```python
                >>> from digitalearth.base.capabilities import Capabilities
                >>> print(Capabilities("flat", features={"colorbar"}).require("colorbar", caller="quickmap"))
                None

                ```
            - An undeclared one is refused by name:
                ```python
                >>> from digitalearth.base.capabilities import Capabilities, CapabilityError
                >>> try:
                ...     Capabilities("flat").require("colorbar", caller="quickmap")
                ... except CapabilityError as error:
                ...     print(error)
                quickmap needs 'colorbar', which backend='flat' does not support

                ```

        See Also:
            reason: what to use where the refusal already has a first clause of its own. `api.py`'s
                `"<name>= is not supported by backend=..."` and a renderer's `"the web tier does not draw
                '<kind>' layers"` are both pinned wordings a caller reads a keyword or a kind out of, so
                those sites compose the message themselves and take only the declared sentence from here.
                This method is for a refusal that has no such clause to keep.
        """
        if self.supports(name):
            return
        reason = self.reason(name)
        raise CapabilityError(
            f"{caller} needs {name!r}, which backend={self.backend!r} does not support"
            + (f"; {reason}" if reason else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form, for the support matrix the docs build.

        Returns:
            Every field, with each set sorted so two runs write the same table, and `absent` as a plain dict.

        Examples:
            - The declaration reads as a table row:
                ```python
                >>> from digitalearth.base.capabilities import Capabilities
                >>> Capabilities("flat", kinds={"points", "raster"}).to_dict()["kinds"]
                ['points', 'raster']

                ```
            - What is missing is part of the row, with its reason:
                ```python
                >>> from digitalearth.base.capabilities import Capabilities
                >>> Capabilities("flat", absent={"domain": "it has no extent"}).to_dict()["absent"]
                {'domain': 'it has no extent'}

                ```
        """
        return {
            "backend": self.backend,
            "kinds": sorted(self.kinds),
            "channels": sorted(self.channels),
            "data_driven": sorted(self.data_driven),
            "schemes": sorted(self.schemes),
            "features": sorted(self.features),
            "absent": dict(self.absent),
        }
