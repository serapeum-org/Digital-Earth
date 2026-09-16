"""How one layer looks, as a declared value instead of undeclared keyword soup.

`CLAUDE.md` states the position plainly: most styling kwargs *"are passed straight through ``**kwargs``"* and
are not handled in this repo. The static tier alone has **26** such keys — the flat members
:mod:`digitalearth.static.render_compat` folds into cleopatra's typed groups — and not one of them appears in
any signature. Three things follow, and all three are why this module exists:

* **Nothing can be discovered.** A caller cannot ask what a builder accepts, and neither can tooling.
* **A typo is not an error.** ``cmpa="viridis"`` is a silently ignored keyword and a map in the wrong colours.
  `IN-13` ("option validation with did-you-mean") is a task only because there is no schema to check against.
* **Every visual channel costs N × M.** Without :class:`~digitalearth.base.spec.encoding.Encoding`, each new
  channel is a new keyword on every builder on every tier.

Two pieces answer it. :class:`StyleSchema` **declares** the keys — each with the channel it drives, or none if
it is a static property — and :class:`Symbology` is what a declared style resolves *to*: the encodings for one
layer, plus the static properties that are not channels.

:meth:`StyleSchema.route` is the half that did not exist before. Today
:func:`digitalearth.static.render_compat.prepare_plot_kwargs` can only *reject* a keyword it has no home for;
routing means a flat public kwarg **finds** the channel that owns it, and only a genuinely homeless key is
reported back — with :meth:`StyleSchema.suggest` naming the nearest declared key, which is what turns a
silently-ignored typo into an answerable error.

Nothing here knows a renderer. Folding a `Symbology` into the flat form a particular engine wants is the
backend's job, and in the static tier there is exactly one place it happens.
"""

from dataclasses import dataclass, field
from difflib import get_close_matches
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from digitalearth.base.spec._serial import (
    as_mapping,
    frozen_value,
    refuse_unknown,
    to_json_value,
)
from digitalearth.base.spec.encoding import CHANNELS, Encoding

__all__ = ["StyleKey", "StyleSchema", "Symbology"]


@dataclass(frozen=True)
class StyleKey:
    """One declared style keyword: its name, what it does, and whether it drives a channel.

    Attributes:
        name: The public keyword, exactly as a caller writes it.
        doc: One line saying what it controls. This is the text a discovery surface shows, so it is part of
            the declaration rather than a comment beside it.
        channel: The visual channel the keyword sets, or ``None`` when it is a static property — a flag,
            a threshold, a nested kwargs dict — that varies with nothing.

    Raises:
        ValueError: if `channel` names something that is not a declared channel.

    Examples:
        - A keyword that drives a channel says which one:
            ```python
            >>> from digitalearth.base.spec import StyleKey
            >>> key = StyleKey("alpha", "Layer opacity, 0 transparent to 1 opaque.", channel="opacity")
            >>> key.name, key.channel
            ('alpha', 'opacity')

            ```
        - A static property declares no channel, and that is how routing tells the two apart:
            ```python
            >>> from digitalearth.base.spec import StyleKey
            >>> StyleKey("hillshade", "Shade the surface with relief.").channel is None
            True

            ```
    """

    name: str
    doc: str
    channel: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse a key pointing at a channel that does not exist.

        Raises:
            ValueError: for an undeclared channel name.
        """
        if self.channel is not None and self.channel not in CHANNELS:
            raise ValueError(
                f"style key {self.name!r} names channel {self.channel!r}, which is not declared; "
                f"declared channels are {sorted(CHANNELS)}"
            )


@dataclass(frozen=True)
class Symbology:
    """The look of one layer: what drives each channel, plus the properties that drive nothing.

    Attributes:
        encodings: Channel name -> :class:`~digitalearth.base.spec.encoding.Encoding`. Each entry is stored
            under the channel it drives, so a renderer asks by channel rather than searching.
        props: Static style properties that are not visual channels — a contour level list, a hillshade flag,
            a legend kwargs dict. Declared, unlike the keyword soup they replace, but not resolved per datum.

    Raises:
        ValueError: if an encoding is filed under a channel it does not drive. That mismatch would make
            :meth:`encoding` answer with something that styles a different channel.

    Examples:
        - Constants, which is what plain styling keywords mean:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> sym = Symbology.of(color="#f00", opacity=0.5)
            >>> sym.encoding("color").resolve()
            '#f00'

            ```
        - Defaults fill the gaps a caller left, and never overwrite what they set:
            ```python
            >>> from digitalearth.base.spec import Symbology
            >>> caller = Symbology.of(color="#f00")
            >>> theme = Symbology.of(color="#000", opacity=0.8)
            >>> merged = caller.merged_over(theme)
            >>> merged.encoding("color").resolve(), merged.encoding("opacity").resolve()
            ('#f00', 0.8)

            ```
    """

    encodings: Mapping[str, Encoding] = field(default_factory=dict)
    props: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check every encoding is filed under its own channel, then freeze both mappings.

        The dataclass is frozen, but a plain ``dict`` field is not — a caller holding the dict it passed in
        could still re-key the symbology afterwards. Replacing both with read-only views closes that.

        Property values are stored in their canonical form: every list and tuple in them, however nested,
        becomes a tuple (see :func:`~digitalearth.base.spec._serial.frozen_value`). That copies a caller's list —
        appending to it afterwards no longer changes the symbology — and it is what lets a symbology written with
        `to_dict`, where tuples become JSON lists, read back equal and hashable. Other mutable values (a numpy
        array, a custom object) are still held as given.

        Raises:
            ValueError: if a key does not match its encoding's channel.
        """
        for key, encoding in dict(self.encodings).items():
            if encoding.channel != key:
                raise ValueError(
                    f"encoding filed under {key!r} drives channel {encoding.channel!r}; "
                    "a Symbology stores each encoding under the channel it drives"
                )
        object.__setattr__(self, "encodings", MappingProxyType(dict(self.encodings)))
        object.__setattr__(
            self,
            "props",
            MappingProxyType(
                {key: frozen_value(value) for key, value in dict(self.props).items()}
            ),
        )

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        """Pickle and copy by rebuilding through the constructor.

        Returns:
            ``(Symbology, (dict(encodings), dict(props)))``.

            The read-only mapping views this type stores cannot be pickled, so
            `pickle`, `copy.copy` and `copy.deepcopy` raised ``cannot pickle 'mappingproxy' object`` — for this
            type and for everything holding one, a `LayerTree` and a web map with a layer included. Rebuilding
            from plain dicts goes through the same validation and freezing as any other construction.

        Examples:
            - A copy is equal to the original, and is a separate object:
                ```python
                >>> import copy
                >>> from digitalearth.base.spec import Symbology
                >>> original = Symbology.of(color="#f00").with_props(levels=(1, 2))
                >>> clone = copy.deepcopy(original)
                >>> clone == original, clone is original
                (True, False)

                ```
        """
        return Symbology, (dict(self.encodings), dict(self.props))

    def __hash__(self) -> int:
        """Hash by the channels driven and the properties set, so a style can key a cache.

        Returns:
            A hash over the encodings and the properties. `Bounds`, `Scale`, `Selection` and `Encoding` all
            hash; a symbology that did not would be the odd one out the moment Wave 2 wants to key a set or
            a cache on a layer's style.

        Raises:
            TypeError: if any value in it is itself unhashable — a dict or a numpy array held as a property,
                say. A list is not among them: lists are stored as tuples, so ``Symbology.of(color=[1, 0, 0])``
                hashes.
        """
        return hash(
            (
                tuple(sorted(self.encodings.items(), key=lambda item: item[0])),
                tuple(sorted(self.props.items(), key=lambda item: item[0])),
            )
        )

    # ------------------------------------------------------------------ builders

    @classmethod
    def of(cls, **channels: Any) -> "Symbology":
        """Build a symbology from constant channel values — the common, simple case.

        Args:
            **channels: Channel name -> constant, or an already-built
                :class:`~digitalearth.base.spec.encoding.Encoding` to use as is.

        Returns:
            The symbology.

        Raises:
            ValueError: if a name is not a declared channel.

        Examples:
            - A constant and a field-driven encoding side by side:
                ```python
                >>> from digitalearth.base.spec import Encoding, Symbology
                >>> sym = Symbology.of(opacity=0.5, color=Encoding.by_field("color", "class"))
                >>> sorted(sym.encodings)
                ['color', 'opacity']

                ```
        """
        encodings = {
            name: value
            if isinstance(value, Encoding)
            else Encoding.constant(name, value)
            for name, value in channels.items()
        }
        return cls(encodings=encodings)

    # ------------------------------------------------------------------ readers

    def encoding(self, channel: str) -> Optional[Encoding]:
        """Return what drives one channel.

        Args:
            channel: The channel to look up.

        Returns:
            Its encoding, or ``None`` when the layer does not style that channel — which is the renderer's
            cue to use its own default rather than to raise.

        Examples:
            - Ask what drives a channel, then resolve it:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> Symbology.of(color="#f00").encoding("color").resolve()
                '#f00'

                ```
            - An unstyled channel answers ``None``, so the renderer keeps its own default:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> Symbology.of(color="#f00").encoding("width") is None
                True

                ```
        """
        return self.encodings.get(channel)

    def merged_over(self, defaults: "Symbology") -> "Symbology":
        """Return this symbology laid over `defaults`, channel by channel and property by property.

        Args:
            defaults: The symbology to fall back to — a theme, a layer-kind default, a figure-wide style.

        Returns:
            A new symbology where this one wins every key it sets and `defaults` supplies the rest. Merging
            per key rather than per object is the point: a caller who names one colour must not lose the
            theme's opacity.

        Examples:
            - The caller's channel wins; the default's survives where the caller was silent:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> Symbology.of(color="#f00").merged_over(Symbology.of(size=6)).encoding("size").resolve()
                6

                ```
        """
        encodings = dict(defaults.encodings)
        encodings.update(self.encodings)
        props = dict(defaults.props)
        props.update(self.props)
        return Symbology(encodings=encodings, props=props)

    def with_props(self, **props: Any) -> "Symbology":
        """Return a copy carrying extra static properties.

        Args:
            **props: The properties to add or replace.

        Returns:
            A new symbology; this one is unchanged.

        Examples:
            - Properties accumulate, and the original is untouched:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> one = Symbology().with_props(scheme="quantiles")
                >>> both = one.with_props(k=7)
                >>> sorted(both.props), sorted(one.props)
                (['k', 'scheme'], ['scheme'])

                ```
            - Naming a property again replaces it:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> Symbology().with_props(k=5).with_props(k=9).props["k"]
                9

                ```
        """
        merged = dict(self.props)
        merged.update(props)
        return Symbology(encodings=dict(self.encodings), props=merged)

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            ``encodings`` (channel -> encoding dict) and ``props``, each omitted when empty — so a layer with no
            styling stores an empty dict.

        Raises:
            TypeError: if a property, an encoding's constant, or the scheme or a category of an encoding's scale
                has no JSON form.

        Examples:
            - Encodings are stored under the channel they drive:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> Symbology.of(color="#f00").to_dict()
                {'encodings': {'color': {'channel': 'color', 'value': '#f00'}}}

                ```
            - No styling at all is an empty dict:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> Symbology().to_dict()
                {}

                ```
        """
        out: Dict[str, Any] = {}
        if self.encodings:
            out["encodings"] = {
                channel: encoding.to_dict()
                for channel, encoding in self.encodings.items()
            }
        if self.props:
            out["props"] = to_json_value(dict(self.props), "Symbology.props")
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Symbology":
        """Rebuild a symbology from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The symbology, with every encoding checked against the channel it is filed under.

        Raises:
            TypeError: if `data`, `encodings`, `props` or a stored encoding is not a mapping, naming the field.
            ValueError: for an unknown key, an encoding `Encoding.from_dict` refuses — an undeclared channel, say —
                or an encoding filed under a channel it does not drive.

        Examples:
            - A stored style reads back channel by channel:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> stored = {"encodings": {"opacity": {"channel": "opacity", "value": 0.4}}, "props": {"k": 5}}
                >>> sym = Symbology.from_dict(stored)
                >>> sym.encoding("opacity").resolve(), sym.props["k"]
                (0.4, 5)

                ```
            - An encoding filed under a channel it does not drive is refused:
                ```python
                >>> from digitalearth.base.spec import Symbology
                >>> misfiled = {"encodings": {"color": {"channel": "opacity", "value": 0.4}}}
                >>> Symbology.from_dict(misfiled)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: encoding filed under 'color' drives channel 'opacity'; ...

                ```
        """
        refuse_unknown("Symbology", data, ("encodings", "props"))
        return cls(
            encodings={
                channel: Encoding.from_dict(encoding)
                for channel, encoding in as_mapping(
                    "Symbology", "encodings", data.get("encodings", {})
                ).items()
            },
            props=as_mapping("Symbology", "props", data.get("props", {})),
        )


@dataclass(frozen=True)
class StyleSchema:
    """The declared style keywords of one builder surface, and the routing they enable.

    Attributes:
        keys: Keyword name -> :class:`StyleKey`.

    Examples:
        - A declared channel keyword becomes an encoding; a declared property becomes a property; an
          undeclared keyword is handed back rather than swallowed:
            ```python
            >>> from digitalearth.base.spec import StyleKey, StyleSchema
            >>> schema = StyleSchema.of(
            ...     StyleKey("alpha", "Opacity.", channel="opacity"),
            ...     StyleKey("hillshade", "Shade the surface."),
            ... )
            >>> sym, leftover = schema.route({"alpha": 0.5, "hillshade": True, "cmpa": "viridis"})
            >>> sym.encoding("opacity").resolve(), dict(sym.props), leftover
            (0.5, {'hillshade': True}, {'cmpa': 'viridis'})

            ```
        - And the nearest declared name is available for the error a caller should get:
            ```python
            >>> from digitalearth.base.spec import StyleKey, StyleSchema
            >>> StyleSchema.of(StyleKey("hillshade", "Shade the surface.")).suggest("hillshde")
            'hillshade'

            ```
    """

    keys: Mapping[str, StyleKey] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the table, so a schema handed around cannot be extended behind a caller's back."""
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        """Pickle and copy by rebuilding through the constructor.

        Returns:
            ``(StyleSchema, (dict(keys),))``.

            The read-only mapping views this type stores cannot be pickled, so
            `pickle`, `copy.copy` and `copy.deepcopy` raised ``cannot pickle 'mappingproxy' object`` — for this
            type and for everything holding one, a `LayerTree` and a web map with a layer included. Rebuilding
            from plain dicts goes through the same validation and freezing as any other construction.

        Examples:
            - A copy is equal to the original, and is a separate object:
                ```python
                >>> import copy
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> original = StyleSchema.of(StyleKey("cmap", "Colormap name."))
                >>> clone = copy.deepcopy(original)
                >>> clone == original, clone is original
                (True, False)

                ```
        """
        return StyleSchema, (dict(self.keys),)

    @classmethod
    def of(cls, *keys: StyleKey) -> "StyleSchema":
        """Build a schema from its keys.

        Args:
            *keys: The declared keywords.

        Returns:
            The schema.

        Raises:
            ValueError: if a name is declared twice — two rows for one keyword means one of them is dead, and
                which one wins would depend on argument order.

        Examples:
            - Declare a surface, then ask what it accepts and what a keyword controls:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> schema = StyleSchema.of(
                ...     StyleKey("alpha", "Layer opacity.", channel="opacity"),
                ...     StyleKey("levels", "Contour levels."),
                ... )
                >>> schema.names()
                ('alpha', 'levels')
                >>> schema.keys["levels"].doc
                'Contour levels.'

                ```
            - Declaring one name twice is refused, because one of the two rows would be dead:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> StyleSchema.of(StyleKey("alpha", "One."), StyleKey("alpha", "Two."))
                Traceback (most recent call last):
                    ...
                ValueError: style key 'alpha' is declared twice

                ```
        """
        table: Dict[str, StyleKey] = {}
        driving: Dict[str, str] = {}
        for key in keys:
            if key.name in table:
                raise ValueError(f"style key {key.name!r} is declared twice")
            if key.channel is not None:
                if key.channel in driving:
                    # route() writes encodings[key.channel], so a second keyword on one channel would
                    # overwrite the first in whatever order the caller's kwargs happened to iterate.
                    raise ValueError(
                        f"style keys {driving[key.channel]!r} and {key.name!r} both drive the "
                        f"{key.channel!r} channel; only one may, or routing would silently drop one"
                    )
                driving[key.channel] = key.name
            table[key.name] = key
        return cls(keys=table)

    def names(self) -> Tuple[str, ...]:
        """Return every declared keyword, sorted.

        Returns:
            The names a caller may pass — the answer to "what does this builder accept?" that did not exist
            before.

        Examples:
            - The declared keywords, sorted:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> schema = StyleSchema.of(StyleKey("levels", "Contour levels."), StyleKey("k", "Classes."))
                >>> schema.names()
                ('k', 'levels')

                ```
            - Which is enough to spot a keyword the surface does not accept:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> schema = StyleSchema.of(StyleKey("hillshade", "Shade the surface."))
                >>> [name for name in ("hillshade", "cmpa") if name not in schema.names()]
                ['cmpa']

                ```
        """
        return tuple(sorted(self.keys))

    def channels(self) -> Tuple[str, ...]:
        """Return the channels this surface can drive, sorted.

        Returns:
            The distinct channel names its keys point at.

        Examples:
            - Only the keys that drive a channel are counted:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> schema = StyleSchema.of(
                ...     StyleKey("alpha", "Layer opacity.", channel="opacity"),
                ...     StyleKey("size", "Marker size.", channel="size"),
                ...     StyleKey("hillshade", "Shade the surface."),
                ... )
                >>> schema.channels()
                ('opacity', 'size')

                ```
            - A surface of static properties alone drives nothing:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> StyleSchema.of(StyleKey("levels", "Contour levels.")).channels()
                ()

                ```
        """
        return tuple(sorted({key.channel for key in self.keys.values() if key.channel}))

    def route(self, flat: Mapping[str, Any]) -> Tuple[Symbology, Dict[str, Any]]:
        """Split flat keyword arguments into a declared symbology and whatever is left over.

        Args:
            flat: The caller's keyword dict. Not mutated.

        Returns:
            A ``(symbology, leftover)`` pair. A keyword declared with a channel becomes a constant
            :class:`~digitalearth.base.spec.encoding.Encoding` on it; a keyword declared without one becomes
            a static property; anything undeclared is returned in `leftover`.

            Leftovers are **returned, not raised**: this surface is not the only thing a builder's kwargs
            feed (the glyph constructor takes its own), so deciding that a key is homeless needs more
            knowledge than a schema has. :meth:`suggest` is what makes that decision answerable when a caller
            does reach it.

        Examples:
            - A ``None`` is a caller declining a key, so it routes nowhere rather than becoming a binding:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> schema = StyleSchema.of(StyleKey("alpha", "Opacity.", channel="opacity"))
                >>> sym, leftover = schema.route({"alpha": None})
                >>> dict(sym.encodings), leftover
                ({}, {})

                ```
        """
        encodings: Dict[str, Encoding] = {}
        props: Dict[str, Any] = {}
        leftover: Dict[str, Any] = {}
        for name, value in flat.items():
            key = self.keys.get(name)
            if key is None:
                leftover[name] = value
                continue
            if value is None:
                continue  # a caller declining a key, not a binding to None
            if key.channel is None:
                props[name] = value
            else:
                encodings[key.channel] = Encoding.constant(key.channel, value)
        return Symbology(encodings=encodings, props=props), leftover

    def suggest(self, name: str) -> Optional[str]:
        """Return the declared keyword `name` was most likely meant to be.

        Args:
            name: The keyword a caller wrote.

        Returns:
            The closest declared name, or ``None`` when nothing is close enough to be worth guessing. A wrong
            guess is worse than none — it sends the caller to a key that was never their intent.

        Examples:
            - A near miss is named, which is what turns a silently ignored typo into an answerable error:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> schema = StyleSchema.of(StyleKey("hillshade", "Shade the surface."))
                >>> schema.suggest("hillshde")
                'hillshade'

                ```
            - Something unlike every declared key is not guessed at:
                ```python
                >>> from digitalearth.base.spec import StyleKey, StyleSchema
                >>> StyleSchema.of(StyleKey("hillshade", "Shade the surface.")).suggest("zzzzzz") is None
                True

                ```
        """
        matches = get_close_matches(name, self.keys, n=1, cutoff=0.7)
        return matches[0] if matches else None
