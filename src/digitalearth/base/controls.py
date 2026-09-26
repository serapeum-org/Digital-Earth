"""controls — what a layer control may expose, and where it may sit, in one vocabulary for every tier.

``layer_control`` is a Tier-2 name: not every tier has one, but the tiers that do must mean the same thing by
it. They did not. The web tier took ``(position, layer_ids, theme)`` and the interactive tier
``(opacity, reorder, basemap_switch)`` — not one parameter in common, so no caller could add a layer control
without knowing which backend it was talking to (#264). The settled signature is the layers to include, the
position, and the controls to expose; this module holds the last two, because they are the parts both tiers
have to agree about *word for word*:

* :data:`LAYER_CONTROLS` — the control names, so ``"basemap"`` is spelled one way rather than being
  ``basemap_switch=True`` on one tier and absent on the other;
* :data:`CONTROL_POSITIONS` — the four corners, which are MapLibre's names and now every tier's.

A tier says which of the controls it can actually build and :func:`resolved_controls` refuses the rest **by
name**, which is the house rule these two tiers arrived at the hard way: a flag accepted and quietly ignored
is how ``reorder=`` and ``basemap_switch=`` came to look implemented (#242, #244).

Pure standard library, so it sits in ``base/`` without threatening the engine-neutrality guard
(``tests/test_base_is_engine_neutral.py``).
"""

from typing import Sequence, Tuple

__all__ = [
    "CONTROL_POSITIONS",
    "LAYER_CONTROLS",
    "REQUIRED_CONTROL",
    "check_control_position",
    "resolved_controls",
]

#: The corners a control may be anchored to. MapLibre's four, which the web tier has always used and which
#: the interactive tier now reads as "which side of the map the widget column sits on, and which end of it".
CONTROL_POSITIONS: Tuple[str, ...] = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
)

#: The controls a layer control may expose, as the one name each goes by. ``"reorder"`` is deliberately
#: **not** here: no tier can reorder layers (the registry has no stable per-layer handle to reorder by), so
#: listing it would advertise a control every tier would then have to refuse. It stays the interactive tier's
#: own ``reorder=`` flag, which says ``NotImplementedError`` in those words.
LAYER_CONTROLS: Tuple[str, ...] = ("visibility", "opacity", "basemap")

#: The one control that is not optional. A layer control whose per-layer toggle was dropped is a box in the
#: corner with nothing to switch — which is the thing ``layer_control`` exists to add.
REQUIRED_CONTROL = "visibility"


def check_control_position(position: str) -> None:
    """Refuse anything but the four legal corners.

    Args:
        position: The requested corner.

    Raises:
        ValueError: when ``position`` is not one of :data:`CONTROL_POSITIONS`. Refused here rather than
            passed on, because a browser silently ignores a corner it does not recognise — so the control
            lands in the default corner and nothing says why.

    Examples:
        - A corner passes through silently:
            ```python
            >>> from digitalearth.base.controls import check_control_position
            >>> check_control_position("bottom-left")

            ```
        - Anything else names the four:
            ```python
            >>> from digitalearth.base.controls import check_control_position
            >>> check_control_position("middle")                 # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: unknown control position 'middle'; choose one of ['top-left', ...]

            ```
    """
    if position not in CONTROL_POSITIONS:
        raise ValueError(
            f"unknown control position {position!r}; choose one of {list(CONTROL_POSITIONS)}"
        )


def resolved_controls(
    controls: Sequence[str], *, offered: Sequence[str], caller: str
) -> Tuple[str, ...]:
    """Return the controls to build, refusing a name no tier has and one *this* tier cannot draw.

    Args:
        controls: The control names the caller asked for, in the order they should be laid out. Duplicates
            are collapsed rather than building the same widget twice.
        offered: The controls this tier can actually build, for the refusal to quote back. A subset of
            :data:`LAYER_CONTROLS`.
        caller: The public method the names were written on (e.g. ``"WebMap.layer_control()"``), so every
            message says *where* the offending call is.

    Returns:
        The requested names, de-duplicated, in the caller's own order.

    Raises:
        ValueError: when a name is not in :data:`LAYER_CONTROLS` (a typo, which would otherwise build
            nothing and say nothing); when :data:`REQUIRED_CONTROL` was left out; or when a name is a real
            control this tier has no widget for. The last is the case worth refusing loudly: accepting it
            and building nothing is precisely how two of this tier's own flags came to look implemented.

    Examples:
        - A tier that can build all three honours all three:
            ```python
            >>> from digitalearth.base.controls import LAYER_CONTROLS, resolved_controls
            >>> resolved_controls(
            ...     ["visibility", "opacity"], offered=LAYER_CONTROLS, caller="Map.layer_control()"
            ... )
            ('visibility', 'opacity')

            ```
        - A control the tier cannot draw is refused, with what it *can* draw named:
            ```python
            >>> from digitalearth.base.controls import resolved_controls
            >>> resolved_controls(                               # doctest: +ELLIPSIS
            ...     ["visibility", "opacity"], offered=("visibility",), caller="WebMap.layer_control()"
            ... )
            Traceback (most recent call last):
                ...
            ValueError: WebMap.layer_control() cannot build the controls ['opacity']: this tier offers ...

            ```
    """
    named = tuple(dict.fromkeys(controls))
    unknown = [name for name in named if name not in LAYER_CONTROLS]
    if unknown:
        raise ValueError(
            f"{caller} was given controls={unknown}, which name no control; choose from "
            f"{list(LAYER_CONTROLS)}"
        )
    if REQUIRED_CONTROL not in named:
        raise ValueError(
            f"{caller} needs {REQUIRED_CONTROL!r} among its controls: a layer control without a per-layer "
            f"toggle has nothing to switch. Got controls={list(named)}"
        )
    unbuildable = [name for name in named if name not in tuple(offered)]
    if unbuildable:
        raise ValueError(
            f"{caller} cannot build the controls {unbuildable}: this tier offers {list(offered)}. Drop "
            "them, or build the same map on a tier that has them."
        )
    return named
