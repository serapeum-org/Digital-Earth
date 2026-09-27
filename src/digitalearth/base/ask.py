"""The not-passed sentinel, and the record of what a caller actually asked a builder for.

`Symbology.encodings` is meant to carry **what a caller asked for**, so style can cross tiers (#328). A
keyword defaulted in a signature destroys that before anything can record it: ``size: float = 5.0`` makes a
deliberate ``points(size=5.0)`` and a bare ``points(features)`` the same call by the time the builder writes
its paint, so the figure holds a resolved value and no evidence of an ask.

Both 2-D tiers reconstructed the ask afterwards, by subtracting tables of measured defaults
(``UNASKED_PAINT``/``UNASKED_PROPS`` on the web tier, ``UNASKED_STYLE`` on the interactive one). That works
in one direction only: a caller who asks for exactly their tier's default publishes nothing for that channel,
because subtraction cannot tell the two apart — and a table row is tied to nothing, so a builder whose
default changes silently invalidates it (#334).

This module replaces the subtraction with a **record**:

* :data:`UNSET` is the default a style keyword carries instead of its value, so a builder can see the
  difference between "not passed" and "passed the same number I would have used".
* :class:`Ask` resolves each keyword against the tier's default and remembers the keys it was actually given,
  under the spelling that tier records its style in.
* :data:`ASKED_PROP` is where that record lives on the layer's `Symbology`, and :func:`asked_style` reads it
  back. The lift (:func:`~digitalearth.base.spec.style.asked_constants`) publishes those keys and no others.

The record is **fail-closed**, which is the property the tables did not have: a builder that records no ask
publishes no channel. A missing row used to mean a default published *as* caller intent; a missing record now
means an ask goes unpublished, which under-describes a layer instead of mis-describing it.

Nothing here knows a renderer, and nothing here is a style vocabulary — the channel a key drives is the
tier's own mapping, and the vocabulary is :mod:`digitalearth.base.spec.style`.
"""

from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    TypeAlias,
    TypeVar,
    Union,
)

__all__ = [
    "ASKED_PROP",
    "UNSET",
    "Ask",
    "Maybe",
    "Unset",
    "asked_record",
    "asked_style",
]

T = TypeVar("T")


class Unset:
    """The type of :data:`UNSET` — "the caller did not pass this keyword".

    A class rather than a bare ``object()`` so the absence is a *type* a signature can be annotated with
    (see :data:`Maybe`) and a checker can narrow on, and so ``help()`` and a traceback spell it as something
    a reader recognises rather than as ``<object object at 0x…>``.

    Examples:
        - It reads as itself wherever a signature's default is printed:
            ```python
            >>> from digitalearth.base.ask import UNSET
            >>> repr(UNSET)
            '<unset>'

            ```
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the placeholder spelling used in signatures and messages.

        Returns:
            ``"<unset>"``.
        """
        return "<unset>"


#: "Not passed." The one instance every builder's style keyword defaults to, so absence is identity rather
#: than a value that could collide with the caller's: `None`, `0.0` and `""` are all things a caller may
#: legitimately ask for, and the tier's own default certainly is.
UNSET: Unset = Unset()

#: A style keyword's annotation once its default is :data:`UNSET` — ``Maybe[float]`` is "a float, or not
#: passed".
#:
#: One alias rather than a widened annotation per keyword, so the two dozen signatures this touches say the
#: same thing the same way, and so the sentinel can never be *absent* from an annotation whose default is
#: `UNSET` without a checker noticing. Resolving through :class:`Ask` narrows it back to `T`, which is what
#: keeps the body of a builder typed exactly as it was.
Maybe: TypeAlias = Union[T, Unset]

#: The `Symbology.props` key a layer's record of the caller's asks lives under.
#:
#: A **list of key names**, in the spelling the tier records its style in — MapLibre paint properties on the
#: web tier (``"circle-radius"``), HoloViews keywords on the interactive one (``"size"``) — and not the
#: values, which are already in the style the builder wrote. Names only, for two reasons: the published
#: value is then always the value the layer is drawn with and cannot drift from it, and the figure carries no
#: second copy of the style. A list rather than a tuple or a set because a list is what
#: :func:`~digitalearth.base.spec._serial.travels_in_a_figure` carries; `Symbology` freezes it to a tuple on
#: the way in, which :func:`asked_style` reads back either way.
ASKED_PROP: str = "asked"


class Ask:
    """One builder call's style keywords: resolved to the tier's defaults, with the asks remembered.

    Built fresh per call, used once per style keyword, and handed to the description as :attr:`named`. The
    two jobs are deliberately one object: a builder cannot resolve a keyword without the record seeing it,
    which is what stops the record from drifting behind a signature the way a table of measured defaults
    did.

    Examples:
        - A keyword the caller passed resolves to their value and is recorded; one they left alone resolves
          to the tier's default and is not:
            ```python
            >>> from digitalearth.base.ask import Ask, UNSET
            >>> ask = Ask()
            >>> ask("circle-radius", 12.0, 5.0), ask("circle-opacity", UNSET, 0.9)
            (12.0, 0.9)
            >>> ask.named
            ['circle-radius']

            ```
        - Asking for exactly the tier's own default is an ask like any other, which is the whole point:
            ```python
            >>> from digitalearth.base.ask import Ask
            >>> ask = Ask()
            >>> ask("circle-radius", 5.0, 5.0)
            5.0
            >>> ask.named
            ['circle-radius']

            ```
    """

    __slots__ = ("_named",)

    def __init__(self) -> None:
        """Start with nothing asked for."""
        self._named: List[str] = []

    def __call__(self, key: str, value: Maybe[T], default: T) -> T:
        """Resolve one style keyword, recording it when the caller passed it.

        Args:
            key: The key the tier records this style under — the MapLibre paint property, or the flat
                property name, or the HoloViews keyword. Not the public keyword unless the two agree: it is
                what the lift looks the channel up by, so it has to be the spelling the style itself is
                keyed with.
            value: What the caller left in the signature — their value, or :data:`UNSET`.
            default: What this tier draws the channel at when nobody asked.

        Returns:
            `value` when the caller passed it, else `default`. The return is `T`, so the rest of the
            builder's body stays typed as it was before the sentinel widened the signature.
        """
        if isinstance(value, Unset):
            return default
        self._named.append(key)
        return value

    def always(self, key: str) -> None:
        """Record a key the builder cannot default, so any value under it is the caller's.

        The case :meth:`__call__` cannot express: a required argument (``extrusion(height=)``) and a
        keyword whose "not passed" is already spelled `None` (``contours(color=None)``) have no default to
        resolve against, and are an ask whenever they carry anything at all.

        Args:
            key: The key the tier records that style under.

        Examples:
            - A required argument is an ask by construction:
                ```python
                >>> from digitalearth.base.ask import Ask
                >>> ask = Ask()
                >>> ask.always("fill-extrusion-height")
                >>> ask.named
                ['fill-extrusion-height']

                ```
        """
        self._named.append(key)

    @property
    def named(self) -> List[str]:
        """The keys the caller asked for, sorted and deduplicated.

        Returns:
            A new `list` each time, so a description cannot be written through, sorted so one call records
            one thing however its builder happened to resolve the keywords, and deduplicated because a key
            a builder resolves twice — a filled contour's outline and its fill — is still one ask.

        Examples:
            - The order the keywords were resolved in is not the order recorded:
                ```python
                >>> from digitalearth.base.ask import Ask
                >>> ask = Ask()
                >>> _ = ask("line-width", 3.0, 1.5), ask("line-color", "#ff0000", "#3388ff")
                >>> ask.named
                ['line-color', 'line-width']

                ```
        """
        return sorted(set(self._named))

    @property
    def record(self) -> Dict[str, List[str]]:
        """The `Symbology.props` entry this call contributes.

        Returns:
            What :func:`asked_record` returns for :attr:`named` — a one-key mapping, or an empty one when
            the caller styled nothing. Spread into the props a builder writes (``**ask.record``) so the
            record is one token at every call site rather than a conditional each builder spells its own
            way.

        Examples:
            - A styled call carries the record; an unstyled one carries no key at all:
                ```python
                >>> from digitalearth.base.ask import Ask, UNSET
                >>> styled = Ask()
                >>> _ = styled("opacity", 0.5, 1.0)
                >>> styled.record
                {'asked': ['opacity']}
                >>> bare = Ask()
                >>> _ = bare("opacity", UNSET, 1.0)
                >>> bare.record
                {}

                ```
        """
        return asked_record(self._named)


def asked_record(asked: Iterable[str]) -> Dict[str, List[str]]:
    """Return the `Symbology.props` entry that records these asks, or nothing when there are none.

    The write side of :data:`ASKED_PROP`, so a builder that resolves its style somewhere other than where it
    describes it — a paint dict built in one method and recorded by another — records it the same way as one
    that does both in a line.

    Args:
        asked: The keys the caller named, in the tier's own spelling. Sorted and deduplicated here, so the
            record does not depend on the order a builder happened to resolve its keywords in.

    Returns:
        ``{ASKED_PROP: [...]}`` when anything was asked for, else ``{}`` — **not** an empty list. A layer
        nobody styled therefore records no key at all, which keeps an unstyled description exactly what it
        was before the record existed.

    Examples:
        - Nothing asked for records nothing, so an unstyled layer's description is unchanged:
            ```python
            >>> from digitalearth.base.ask import asked_record
            >>> asked_record(())
            {}
            >>> asked_record(["line-width", "line-color", "line-width"])
            {'asked': ['line-color', 'line-width']}

            ```
    """
    named = sorted(set(asked))
    return {ASKED_PROP: named} if named else {}


def asked_style(props: Mapping[str, Any]) -> FrozenSet[str]:
    """Return the style keys a layer's description says the caller named.

    The read side of :data:`ASKED_PROP`, shared so the two tiers cannot disagree about how the record is
    spelled or about what a layer carrying no record means.

    Args:
        props: A layer's recorded `Symbology.props`.

    Returns:
        The key names, as a frozenset. Empty for a description that records none — which is every layer
        nobody styled, and also any layer described by hand, so a `Symbology` read on its own publishes
        nothing rather than publishing whatever style it happens to hold.

    Examples:
        - A record survives `Symbology`'s freeze, which turns the recorded list into a tuple:
            ```python
            >>> from digitalearth.base.ask import asked_style
            >>> from digitalearth.base.spec import Symbology
            >>> sorted(asked_style(Symbology(props={"asked": ["size", "opacity"]}).props))
            ['opacity', 'size']

            ```
        - A description with no record has asked for nothing:
            ```python
            >>> from digitalearth.base.ask import asked_style
            >>> asked_style({"paint": {"circle-radius": 5.0}})
            frozenset()

            ```
    """
    return frozenset(str(key) for key in (props.get(ASKED_PROP) or ()))
