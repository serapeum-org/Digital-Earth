"""What a legend says, derived from the scale that was drawn — not computed a second time beside it.

Every tier builds a legend its own way. The web tier keeps two parallel attributes, ``last_breaks`` (the
edges, categories or ramp stops) and ``last_legend`` (those values *plus the colours they were drawn with*),
and assembles the second one three separate times in a single module — once per scheme kind. The interactive
tier records ``last_breaks`` for *"legend parity with the web tier"*, which is parity maintained by hand.

The consequence is the one #185 named: a legend can disagree with the picture. Nothing structurally ties a
swatch to the colour actually drawn — they are two computations from the same inputs, and they agree only
while nobody edits one of them.

:class:`~digitalearth.base.spec.scale.Scale` resolves colour once and can be frozen, so a legend derived
*from the scale* is agreement by construction rather than by maintenance. That is what
:meth:`LegendSpec.from_scale` is for.

This says what a legend contains. Drawing it stays with each tier — a MapLibre control, a matplotlib
colorbar and a Bokeh panel have nothing in common below this line.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from digitalearth.base.spec.scale import Scale

__all__ = ["LegendEntry", "LegendSpec"]

#: The legend kinds every tier can draw. A module constant, not a dataclass field: declared as a field it
#: became the 7th constructor parameter, so a caller could hand the type its own allow-list and defeat the
#: check `__post_init__` exists for — and it leaked into `dataclasses.asdict()`. `Scale` and `Selection` keep
#: their constants at module level for the same reason.
LEGEND_KINDS: Tuple[str, ...] = ("categorical", "graduated", "continuous")

#: How many stops a continuous ramp is described with. Five is what the web tier already used; sharing the
#: number is what stops a sixth appearing when another tier grows a ramp legend.
DEFAULT_RAMP_STOPS: int = 5


@dataclass(frozen=True)
class LegendEntry:
    """One row of a legend: a swatch, and what it means.

    Attributes:
        label: The text shown beside the swatch.
        color: The colour drawn, as the renderer drew it.
        value: The underlying value or range this row stands for, kept so a tier can build an interactive
            legend that filters or highlights. ``None`` when the row is decorative.

    Examples:
        - A row carries what it stands for, not just its label:
            ```python
            >>> from digitalearth.base.spec import LegendEntry
            >>> row = LegendEntry("land", "#8b4513", "land")
            >>> row.label, row.color, row.value
            ('land', '#8b4513', 'land')

            ```
    """

    label: str
    color: Optional[str]
    value: Any = None


@dataclass(frozen=True)
class LegendSpec:
    """A legend as a value: what it is called, what its rows are, and how it is laid out.

    Attributes:
        entries: The rows, in the order they are drawn.
        kind: ``"categorical"``, ``"graduated"`` or ``"continuous"`` — the three the tiers already
            distinguish, kept because a ramp is drawn as a bar and the other two as swatches.
        title: The legend's heading, usually the column name.
        units: Unit string appended to the labels, or shown once as a subtitle.
        format: Format spec for numeric labels, e.g. ``".1f"``.
        orientation: ``"vertical"`` or ``"horizontal"``.

    Examples:
        - A categorical legend takes its colours from the scale that drew them:
            ```python
            >>> from digitalearth.base.spec import LegendSpec, Scale
            >>> scale = Scale.categorical(["land", "sea"], ["#8b4513", "#1e90ff"])
            >>> legend = LegendSpec.from_scale(scale, title="cover")
            >>> [(e.label, e.color) for e in legend.entries]
            [('land', '#8b4513'), ('sea', '#1e90ff')]

            ```
        - A classified scale becomes one row per class, labelled by the range it covers:
            ```python
            >>> from digitalearth.base.spec import LegendSpec, Scale
            >>> scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=2)
            >>> legend = LegendSpec.from_scale(scale, colors=["#eee", "#111"])
            >>> len(legend.entries), legend.kind
            (2, 'graduated')

            ```
    """

    entries: Tuple[LegendEntry, ...] = ()
    kind: str = "continuous"
    title: Optional[str] = None
    units: Optional[str] = None
    format: Optional[str] = None
    orientation: str = "vertical"

    def __post_init__(self) -> None:
        """Refuse a kind or orientation no tier can draw.

        Raises:
            ValueError: for an unknown `kind` or `orientation`. Both select a drawing path in every tier, so
                an unrecognised one is silently ignored at render time and the legend simply does not appear.
        """
        if self.kind not in LEGEND_KINDS:
            raise ValueError(
                f"LegendSpec kind must be one of {list(LEGEND_KINDS)}; got {self.kind!r}"
            )
        if self.orientation not in ("vertical", "horizontal"):
            raise ValueError(
                f"LegendSpec orientation must be 'vertical' or 'horizontal'; got {self.orientation!r}"
            )
        object.__setattr__(self, "entries", tuple(self.entries))

    @classmethod
    def from_scale(
        cls,
        scale: Scale,
        *,
        colors: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        units: Optional[str] = None,
        format: Optional[str] = None,
        orientation: str = "vertical",
        stops: int = DEFAULT_RAMP_STOPS,
    ) -> "LegendSpec":
        """Build a legend from the scale a layer was drawn with.

        This is the load-bearing method. Deriving the rows *from the scale* is what makes a swatch equal the
        colour that was drawn, instead of a second computation that happens to agree.

        Args:
            scale: The resolved scale the layer used.
            colors: The colours drawn, one per class or stop. Required for a graduated or continuous scale,
                where the colours come from the renderer's colormap rather than from the scale. Ignored for a
                categorical scale, which carries its own — passing them there would reintroduce exactly the
                second source of truth this removes.
            title: Legend heading, usually the column name.
            units: Unit string for the values.
            format: Format spec for numeric labels.
            orientation: ``"vertical"`` or ``"horizontal"``.
            stops: How many stops describe a continuous ramp.

        Returns:
            The legend.

        Raises:
            ValueError: if `colors` is missing or the wrong length for a scale that needs it. A short list
                leaves classes sharing a colour and a long one leaves colours unused — either way the legend
                stops matching the picture, which is the failure this type exists to prevent.

        Examples:
            - A categorical scale needs no colours passed, because it has them:
                ```python
                >>> from digitalearth.base.spec import LegendSpec, Scale
                >>> scale = Scale.categorical(["a"], ["#f00"], missing="#ccc")
                >>> LegendSpec.from_scale(scale).entries[0].color
                '#f00'

                ```
            - A graduated scale given the wrong number of colours is refused:
                ```python
                >>> from digitalearth.base.spec import LegendSpec, Scale
                >>> scale = Scale.from_values([0.0, 10.0], scheme="equal_interval", k=3)
                >>> LegendSpec.from_scale(scale, colors=["#eee", "#111"])
                Traceback (most recent call last):
                    ...
                ValueError: a graduated legend needs one colour per class; got 2 colours for 3 classes

                ```
        """
        if stops < 2:
            # Fewer than two stops cannot describe a ramp: one divides by zero computing the spacing, none
            # yields an empty legend. Every other field on this type is checked, and this one reaches the
            # arithmetic directly.
            raise ValueError(
                f"a continuous legend needs at least two stops to describe a ramp; got {stops}"
            )
        if scale.is_categorical:
            entries = tuple(
                LegendEntry(str(category), scale.color_for(category), category)
                for category in scale.categories
            )
            return cls(entries, "categorical", title, units, format, orientation)

        if scale.is_classified:
            ranges = scale.class_ranges()
            colors = cls._checked(colors, len(ranges), "graduated", "class")
            entries = tuple(
                LegendEntry(cls._range_label(low, high, format), color, (low, high))
                for (low, high), color in zip(ranges, colors)
            )
            return cls(entries, "graduated", title, units, format, orientation)

        lo, hi = scale.as_limits()
        values = [lo + (hi - lo) * i / (stops - 1) for i in range(stops)]
        colors = cls._checked(colors, stops, "continuous", "stop")
        entries = tuple(
            LegendEntry(cls._number(value, format), color, value)
            for value, color in zip(values, colors)
        )
        return cls(entries, "continuous", title, units, format, orientation)

    @staticmethod
    def _checked(
        colors: Optional[Sequence[str]], wanted: int, kind: str, noun: str
    ) -> List[str]:
        """Return `colors` as a list, refusing a count that would misdescribe the picture.

        Args:
            colors: The colours the caller passed, or ``None``.
            wanted: How many are needed.
            kind: The legend kind, for the message.
            noun: What the colours correspond to — ``"class"`` or ``"stop"``.

        Returns:
            The colours.

        Raises:
            ValueError: if none were given, or the wrong number.
        """
        if colors is None:
            raise ValueError(
                f"a {kind} legend needs the colours the layer was drawn with, one per {noun}; "
                "they come from the renderer's colormap, not from the scale"
            )
        listed = list(colors)
        if len(listed) != wanted:
            plural = "classes" if noun == "class" else f"{noun}s"
            raise ValueError(
                f"a {kind} legend needs one colour per {noun}; got {len(listed)} colours "
                f"for {wanted} {plural}"
            )
        return listed

    @staticmethod
    def _number(value: float, spec: Optional[str]) -> str:
        """Format one number for a label.

        Args:
            value: The number.
            spec: A format spec, or ``None`` for `str`.

        Returns:
            The formatted text.
        """
        return format(value, spec) if spec else str(value)

    @classmethod
    def _range_label(cls, low: float, high: float, spec: Optional[str]) -> str:
        """Return the ``low – high`` label a graduated class is shown with.

        Args:
            low: Lower edge.
            high: Upper edge.
            spec: Format spec for the numbers.

        Returns:
            The label.
        """
        return f"{cls._number(low, spec)} – {cls._number(high, spec)}"

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a tier stores or serialises.

        Every field is included, so the result is complete rather than the subset a particular tier happens
        to read. It is not paired with a `from_dict`: a legend is *derived* from a `Scale`, and rebuilding one
        from a dict would be a second way to construct it — the one thing :meth:`from_scale` exists to
        prevent. Round-tripping belongs to whatever holds the scale, not to the legend.

        Returns:
            The legend as ``kind`` / ``values`` / ``colors`` plus the labels — the shape the web tier already
            keeps in ``last_legend``, so it can derive that from this rather than assembling it three times.

        Examples:
            - The three keys a tier reads back:
                ```python
                >>> from digitalearth.base.spec import LegendSpec, Scale
                >>> payload = LegendSpec.from_scale(Scale.categorical(["a"], ["#f00"])).to_dict()
                >>> payload["kind"], payload["values"], payload["colors"]
                ('categorical', ['a'], ['#f00'])

                ```
        """
        return {
            "kind": self.kind,
            "title": self.title,
            "units": self.units,
            "format": self.format,
            "orientation": self.orientation,
            "labels": [entry.label for entry in self.entries],
            "values": [entry.value for entry in self.entries],
            "colors": [entry.color for entry in self.entries],
        }
