"""A visual channel, bound to a constant or to a data field — so a new channel is a key, not a parameter.

Colour is the only visual variable this package treats as first class. Everything else is a keyword bolted
onto whichever builders happened to need it: marker size is ``size`` on the point builders, ``num_size`` for
cell values, ``point_label_size`` for labels; line width is ``line_scale`` here and a ``widths`` array there.
The three channels already asked for — **height/extrusion** (#199), **text** (#191) and rotation — would each
repeat that, once per builder, on four tiers.

`Encoding` is the shape that stops it. A channel is a *key*, so adding one adds a row to :data:`CHANNELS`
rather than a parameter to every method, and a channel is bound the same way whether the caller wants a
constant (``color="#f00"``) or a data-driven mapping (colour by the ``elevation`` column through a
:class:`~digitalearth.base.spec.scale.Scale`). Both are the same type, which is what lets a renderer ask one
question — "what drives this channel?" — instead of checking a different keyword per case.

What :meth:`Encoding.resolve` returns is deliberately *not* a rendered value. A continuous colour encoding
resolves to a position in ``[0, 1]``, which every backend already knows how to put through its own ramp;
turning that into an RGBA tuple would need a colormap, and colormaps are the renderer's business. A numeric
channel with an `output_range` resolves to real units, because ``(4, 20)`` pixels means the same thing
everywhere.
"""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from digitalearth.base.spec.scale import Scale

__all__ = ["CHANNELS", "Channel", "Encoding"]


@dataclass(frozen=True)
class Channel:
    """One visual variable a layer can drive.

    Attributes:
        name: The channel's name, and the key it is stored under.
        kind: What a resolved value *is* — ``"color"`` for a colour or a ramp position, ``"number"`` for a
            magnitude in the channel's own units, ``"text"`` for a label. A renderer reads this to know
            whether to hand the result to a colormap or use it directly.
        doc: One line saying what the channel controls, so the declared vocabulary is self-describing.

    Examples:
        - Look a channel up and read what it controls:
            ```python
            >>> from digitalearth.base.spec import CHANNELS
            >>> CHANNELS["size"].kind
            'number'
            >>> CHANNELS["size"].doc
            'Marker or glyph size, in points.'

            ```
        - The table is the growth axis, so the channels a feature will need are already in it:
            ```python
            >>> from digitalearth.base.spec import CHANNELS
            >>> sorted(name for name in CHANNELS if CHANNELS[name].kind == "number")
            ['height', 'opacity', 'rotation', 'size', 'width']

            ```
    """

    name: str
    kind: str
    doc: str


#: The visual channels a layer can drive. **This table is the growth axis** — `height` and `text` are the
#: capabilities #199 and #191 ask for, and adding rotation or a stroke channel is a row here plus a fold rule
#: in the backend, not a parameter on every builder.
CHANNELS: Mapping[str, Channel] = MappingProxyType(
    {
        channel.name: channel
        for channel in (
            Channel("color", "color", "Fill/ramp colour of the drawn feature or cell."),
            Channel("opacity", "number", "Alpha, 0 transparent to 1 opaque."),
            Channel("size", "number", "Marker or glyph size, in points."),
            Channel("width", "number", "Line width, in points."),
            Channel(
                "height", "number", "Extrusion or z-height, in the layer's z units."
            ),
            Channel("text", "text", "Label drawn with the feature."),
            Channel("rotation", "number", "Rotation applied to the glyph, in degrees."),
        )
    }
)


def _clamp_unit(value: float) -> float:
    """Clamp a normalised position into ``[0, 1]``.

    Args:
        value: The position to clamp.

    Returns:
        The position, bounded. A value outside the scale's domain is pinned to the nearest end rather than
        dropped, matching what a renderer does with a value outside its colour limits.
    """
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


@dataclass(frozen=True)
class Encoding:
    """What drives one visual channel: a constant, or a field read through a scale.

    Attributes:
        channel: Which channel this drives — a key of :data:`CHANNELS`.
        value: The constant, when the channel does not vary with the data.
        field: The column or band name the channel varies with. Exactly one of `value` and `field` is set.
        scale: How a field's values map onto the channel. ``None`` passes the values through untouched, which
            is what a renderer wants when it holds its own mapping.
        output_range: ``(low, high)`` the normalised position is stretched into, for a numeric channel — the
            marker sizes or line widths the data should span. Meaningless for a colour channel, whose
            positions the backend's ramp consumes.

    Raises:
        ValueError: if the channel is not declared, if neither or both of `value`/`field` are given, or if a
            scale or an output range is attached to a constant — a constant that carries a mapping is a
            caller who expected the mapping to apply, and silently ignoring it would draw one colour.

    Examples:
        - A constant, which is what a plain ``color="#f00"`` means:
            ```python
            >>> from digitalearth.base.spec import Encoding
            >>> Encoding.constant("color", "#f00").resolve()
            '#f00'

            ```
        - Driven by a field, resolving to ramp positions the backend colours:
            ```python
            >>> from digitalearth.base.spec import Encoding, Scale
            >>> enc = Encoding.by_field("color", "elevation", scale=Scale.from_limits(0.0, 10.0))
            >>> enc.resolve([0.0, 5.0, 10.0])
            [0.0, 0.5, 1.0]

            ```
        - The same shape drives size, in real units — the reason a new channel costs a row, not a parameter:
            ```python
            >>> from digitalearth.base.spec import Encoding, Scale
            >>> enc = Encoding.by_field(
            ...     "size", "population", scale=Scale.from_limits(0.0, 100.0), output_range=(4.0, 20.0)
            ... )
            >>> enc.resolve([0.0, 50.0, 100.0])
            [4.0, 12.0, 20.0]

            ```
    """

    channel: str
    value: Any = None
    field: Optional[str] = None
    scale: Optional[Scale] = None
    output_range: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        """Refuse an encoding that names no channel, or names both a constant and a field.

        Raises:
            ValueError: for an undeclared channel, an ambiguous or empty binding, or a mapping attached to a
                constant.
        """
        if self.channel not in CHANNELS:
            raise ValueError(
                f"{self.channel!r} is not a visual channel; declared channels are {sorted(CHANNELS)}"
            )
        if (self.value is None) == (self.field is None):
            raise ValueError(
                f"an Encoding for {self.channel!r} needs exactly one of value= (a constant) or field= "
                "(driven by the data); a constant of None reads as no binding at all"
            )
        if self.field is not None and not str(self.field).strip():
            # `by_field` refuses this, but the constructor is public too — and an empty field name reports
            # is_constant == False while naming a column nobody can look up.
            raise ValueError(
                f"an Encoding for {self.channel!r} needs a non-empty field name; got {self.field!r}"
            )
        if self.output_range is not None:
            if isinstance(self.output_range, (str, bytes)) or not hasattr(
                self.output_range, "__iter__"
            ):
                # tuple(5) raises TypeError about ints, which names neither the argument nor the channel.
                raise ValueError(
                    f"the {self.channel!r} encoding needs output_range as a (low, high) pair; got "
                    f"{self.output_range!r}"
                )
            object.__setattr__(self, "output_range", tuple(self.output_range))
            if len(self.output_range) != 2:
                raise ValueError(
                    f"the {self.channel!r} encoding needs output_range as a (low, high) pair; got "
                    f"{self.output_range!r}"
                )
        if self.field is None and (
            self.scale is not None or self.output_range is not None
        ):
            raise ValueError(
                f"a constant Encoding for {self.channel!r} cannot carry a scale or an output range; "
                "they only apply to a field"
            )
        if (
            self.output_range is not None
            and self.scale is not None
            and self.scale.is_categorical
        ):
            # A categorical scale resolves to its own colours, so there is no position for a range to
            # stretch. Ignoring the argument returned colours from a channel the caller asked numbers of.
            raise ValueError(
                f"the {self.channel!r} encoding cannot combine a categorical scale with an output range: "
                "a category resolves to its assigned colour, not to a position a range can stretch"
            )

    # ------------------------------------------------------------------ builders

    @classmethod
    def constant(cls, channel: str, value: Any) -> "Encoding":
        """Bind a channel to one value for every feature.

        Args:
            channel: The channel to drive.
            value: The constant.

        Returns:
            The encoding.

        Raises:
            ValueError: if the channel is undeclared, or `value` is ``None`` — which would read as "no
                binding" rather than "this channel is off".

        Examples:
            - A constant colour, resolved back out:
                ```python
                >>> from digitalearth.base.spec import Encoding
                >>> Encoding.constant("color", "#1e90ff").resolve()
                '#1e90ff'

                ```
            - Any declared channel works the same way, and knows it does not vary:
                ```python
                >>> from digitalearth.base.spec import Encoding
                >>> width = Encoding.constant("width", 2.5)
                >>> width.is_constant, width.resolve([1, 2])
                (True, [2.5, 2.5])

                ```
        """
        return cls(channel=channel, value=value)

    @classmethod
    def by_field(
        cls,
        channel: str,
        field: str,
        *,
        scale: Optional[Scale] = None,
        output_range: Optional[Tuple[float, float]] = None,
    ) -> "Encoding":
        """Bind a channel to a data field.

        Args:
            channel: The channel to drive.
            field: The column or band the channel varies with.
            scale: How its values map onto the channel; ``None`` passes them through.
            output_range: ``(low, high)`` for a numeric channel.

        Returns:
            The encoding.

        Raises:
            ValueError: if the channel is undeclared or `field` is empty.

        Examples:
            - Name the column the channel varies with, and read it back:
                ```python
                >>> from digitalearth.base.spec import Encoding
                >>> enc = Encoding.by_field("color", "landcover")
                >>> enc.field, enc.is_constant
                ('landcover', False)

                ```
            - With a scale and an output range, the values land in the channel's own units:
                ```python
                >>> from digitalearth.base.spec import Encoding, Scale
                >>> enc = Encoding.by_field(
                ...     "size", "magnitude", scale=Scale.from_limits(0.0, 8.0), output_range=(2.0, 18.0)
                ... )
                >>> enc.resolve([0.0, 4.0, 8.0])
                [2.0, 10.0, 18.0]

                ```
        """
        if not field:
            raise ValueError(
                f"an Encoding for {channel!r} needs a non-empty field name"
            )
        return cls(channel=channel, field=field, scale=scale, output_range=output_range)

    # ------------------------------------------------------------------ readers

    @property
    def is_constant(self) -> bool:
        """Whether this channel holds one value rather than varying with the data.

        Returns:
            ``True`` for a constant binding — the case a renderer can set once instead of per feature.

        Examples:
            - A constant is set once; a field-driven channel is not:
                ```python
                >>> from digitalearth.base.spec import Encoding
                >>> Encoding.constant("opacity", 0.5).is_constant
                True
                >>> Encoding.by_field("opacity", "confidence").is_constant
                False

                ```
            - Which is what lets a renderer take the cheap path when it can:
                ```python
                >>> from digitalearth.base.spec import Encoding
                >>> enc = Encoding.constant("color", "#333")
                >>> enc.resolve() if enc.is_constant else enc.resolve(["a", "b"])
                '#333'

                ```
        """
        return self.field is None

    def resolve(self, values: Any = None) -> Any:
        """Return what the channel should be, per datum.

        Args:
            values: The field's values, one per feature or cell. Required for a field-driven encoding;
                optional for a constant, where omitting it returns the constant itself rather than a list.

        Returns:
            For a constant with no `values`, the constant. Otherwise one entry per value:

            * no scale — the values unchanged, for a renderer that holds its own mapping
            * a categorical scale — each value's colour, or the scale's missing colour
            * any other scale — the normalised position in ``[0, 1]``, stretched into `output_range` when
                one is set. A classified scale steps: every value in a class resolves to that class's
                position, so the picture matches the legend's swatches.

            A value the scale cannot place — not finite — resolves to ``None``, which is the renderer's cue
            to draw it as missing rather than as some other class.

        Raises:
            ValueError: if the encoding is field-driven and no `values` were given.

        Examples:
            - A constant broadcasts when values are supplied, so one code path serves both bindings:
                ```python
                >>> from digitalearth.base.spec import Encoding
                >>> Encoding.constant("opacity", 0.5).resolve([1, 2, 3])
                [0.5, 0.5, 0.5]

                ```
            - A classified scale steps, so the drawn colours match the legend's classes:
                ```python
                >>> from digitalearth.base.spec import Encoding, Scale
                >>> scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)
                >>> Encoding.by_field("color", "v", scale=scale).resolve([1.0, 9.0])
                [0.0, 1.0]

                ```
        """
        if self.is_constant:
            return self.value if values is None else [self.value] * len(list(values))
        if values is None:
            raise ValueError(
                f"the {self.channel!r} encoding is driven by field {self.field!r}, so resolve() needs its "
                "values"
            )
        items = list(values)
        if self.scale is None:
            return items
        if self.scale.is_categorical:
            return [self.scale.color_for(item) for item in items]
        return [self._position(self.scale, item) for item in items]

    def _position(self, scale: Scale, value: Any) -> Optional[float]:
        """Place one numeric value on the channel, in ``[0, 1]`` or in `output_range`.

        Args:
            scale: The scale to place against. Passed in rather than read from `self` because the caller has
                already established there is one — taking it as an argument is what makes that a fact here
                rather than a second check that can never fail.
            value: The datum to place.

        Returns:
            Its position, or ``None`` when it is not a finite number the scale can place.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(number):
            return None
        if scale.is_classified:
            classes = max(len(scale.class_ranges()) - 1, 1)
            # A finite value on a classified scale always places: `class_of` clamps rather than declining,
            # and a Scale cannot hold fewer than two edges. The fallback is for the type, not the data.
            index = scale.class_of(number)
            position = (0 if index is None else index) / classes
        else:
            lo, hi = scale.as_limits()
            position = _clamp_unit((number - lo) / (hi - lo))
        if self.output_range is None:
            return position
        low, high = self.output_range
        return low + position * (high - low)
