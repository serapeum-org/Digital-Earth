"""Furniture — what is fixed to a panel's frame rather than drawn in map coordinates.

A scale bar, a north arrow, a navigation control, a time slider: none of them has a position on the ground. They
sit in a corner of the panel and stay there while the map is panned, so they are not layers, however much the
tiers have treated them as such — the web tier queues its controls as closures beside its layers, and the static
tier draws its scale bar as one more artist.

They are described here instead: a :class:`Furniture` names a registered kind
(:func:`~digitalearth.base.registry.furniture_kinds`), the corner it is anchored to, and whatever options the
tier that draws it reads. What a tier cannot draw it declares absent and skips, so a figure written for the web
still loads and still draws its map on a static image — without its zoom buttons, which mean nothing there.

This is one of the four homes a decoration can have; the other three are a layer in the tree (anything drawn in
map coordinates), a guide on an `Encoding` (anything that explains a channel) and the panel's own `title`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from digitalearth.base.registry import (
    FURNITURE_ANCHORS,
    KIND_PATTERN,
    furniture_info,
)
from digitalearth.base.spec._serial import (
    FrozenDict,
    as_mapping,
    frozen_value,
    plain_text,
    refuse_unknown,
    require,
    to_json_value,
)

__all__ = ["Furniture"]


@dataclass(frozen=True)
class Furniture:
    """One item anchored to a panel's frame: what it is, which corner it sits in, and how it is drawn.

    Attributes:
        kind: A registered furniture name — `"scale_bar"`, `"north_arrow"`, `"navigation"`, … Refused when
            nobody registered it, because a name no tier knows would be silently dropped at draw time.
        anchor: The corner it sits in, one of :data:`~digitalearth.base.registry.FURNITURE_ANCHORS`. `None`
            takes the corner the kind is registered with, and is stored as that corner: a panel says where its
            furniture is without the reader having to consult the registry.
        options: What the tier that draws it reads — a scale bar's units, a slider's step. Held read-only, with
            every list in it stored as a tuple, exactly as `Symbology.props` are.

    Raises:
        ValueError: for an unregistered kind, an anchor that is not one of the four corners, options that are not
            a mapping, or an option key that is not a non-empty string.

    Examples:
        - A scale bar sits where its registration says, without the panel repeating it:
            ```python
            >>> from digitalearth.base.spec import Furniture
            >>> Furniture("scale_bar").anchor
            'bottom-left'

            ```
        - Put it somewhere else, with an option for the tier that draws it:
            ```python
            >>> from digitalearth.base.spec import Furniture
            >>> item = Furniture("navigation", anchor="top-left", options={"show_compass": False})
            >>> item.anchor, item.options["show_compass"]
            ('top-left', False)

            ```
        - A stored item reads back as the item it was written from:
            ```python
            >>> from digitalearth.base.spec import Furniture
            >>> stored = Furniture("north_arrow", options={"scale": 0.5}).to_dict()
            >>> stored
            {'kind': 'north_arrow', 'anchor': 'top-right', 'options': {'scale': 0.5}}
            >>> Furniture.from_dict(stored).options["scale"]
            0.5

            ```
        - A name no tier could draw is refused where it is written, not at draw time:
            ```python
            >>> from digitalearth.base.spec import Furniture
            >>> Furniture("scalebar")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: no furniture 'scalebar' is registered; registered furniture is ['attribution', ...]

            ```
    """

    kind: str
    anchor: Optional[str] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve the anchor against the registry, then freeze the options.

        Raises:
            ValueError: as described on the class.
        """
        try:
            registered = furniture_info(self.kind)
        # The registry's message already names what is registered; a `Furniture` is a value, and a value refuses
        # with ValueError, so the caller does not have to catch two exception types to validate a figure.
        except KeyError as error:
            # Except for a *namespaced* kind — `mypkg:compass` — which is a plugin's furniture read on a
            # machine where the plugin is not installed. `LayerSpec` checks only the spelling for the same
            # reason, so a figure whose panel carried one could not be read back at all while its layers
            # loaded fine (review L3). A bare name nobody registered is a typo, and keeps the registry's
            # message, which lists what there is. A plugin's piece has no default anchor here, so it needs
            # its own.
            if ":" not in self.kind or KIND_PATTERN.fullmatch(self.kind) is None:
                raise ValueError(error.args[0]) from None
            if self.anchor is None:
                raise ValueError(
                    f"furniture {self.kind!r} is not registered here, so it has no default anchor; pass "
                    f"anchor= to place it"
                ) from None
            registered = None
        anchor = (
            self.anchor
            if registered is None or self.anchor is not None
            else registered.anchor
        )
        if anchor not in FURNITURE_ANCHORS:
            raise ValueError(
                f"furniture {self.kind!r}: anchor must be one of {list(FURNITURE_ANCHORS)}; got {self.anchor!r}"
            )
        if not isinstance(self.options, Mapping):
            raise ValueError(
                f"furniture {self.kind!r}: options must be a mapping; got {type(self.options).__name__}"
            )
        for key in self.options:
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"furniture {self.kind!r}: option names must be non-empty strings; got {key!r}"
                )
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(
            self,
            "options",
            FrozenDict(
                {key: frozen_value(value) for key, value in dict(self.options).items()}
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            `kind` and the resolved `anchor`, plus `options` when there are any. The anchor is always written:
            a figure read back years later places its furniture where it was placed when it was written, even if
            the registered default has moved since.

        Raises:
            TypeError: if an option value has no JSON form.

        Examples:
            - The anchor is written even when it was never given:
                ```python
                >>> from digitalearth.base.spec import Furniture
                >>> Furniture("fullscreen").to_dict()
                {'kind': 'fullscreen', 'anchor': 'top-right'}

                ```
            - Options are written as JSON values, so a tuple comes back as a list:
                ```python
                >>> from digitalearth.base.spec import Furniture
                >>> Furniture("scale_bar", options={"units": ("km", "mi")}).to_dict()["options"]
                {'units': ['km', 'mi']}

                ```
        """
        out: Dict[str, Any] = {
            "kind": plain_text(self.kind),
            "anchor": plain_text(self.anchor),
        }
        if self.options:
            out["options"] = to_json_value(
                dict(self.options), f"Furniture[{self.kind!r}].options"
            )
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Furniture":
        """Rebuild an item from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The item, validated as the constructor validates it.

        Raises:
            TypeError: if `data` or its `options` is not a mapping, naming the field.
            ValueError: for a missing kind, an unknown key, or an item the constructor refuses.

        Examples:
            - An item stored without options reads back:
                ```python
                >>> from digitalearth.base.spec import Furniture
                >>> Furniture.from_dict({"kind": "measure", "anchor": "top-left"}).kind
                'measure'

                ```
            - A key nobody writes is refused rather than ignored:
                ```python
                >>> from digitalearth.base.spec import Furniture
                >>> Furniture.from_dict({"kind": "measure", "corner": "top-left"})  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: Furniture.from_dict got unknown keys ['corner']; known keys are ...

                ```
        """
        refuse_unknown("Furniture", data, ("kind", "anchor", "options"))
        return cls(
            kind=require("Furniture", data, "kind"),
            anchor=data.get("anchor"),
            options=as_mapping("Furniture", "options", data.get("options", {})),
        )
