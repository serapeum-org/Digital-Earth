"""Package-level options a caller sets to steer digitalearth without touching a backend's own config (#573).

digitalearth's contract is that a caller uses *digitalearth's* API, not the config of the backend that happens
to render — the same way they never touch Bokeh, MapLibre or PyVista globals directly. This module is the one
place that holds such a decision as a digitalearth setting: a small, validated key→value store, read by the
tier that owns each option's meaning.

The surface is deliberately tiny — :func:`set_option` and :func:`get_option`, keyed by a dotted name
(``"three_d.off_screen"``) so the namespace extends to future options without new top-level functions. The
store is engine-neutral: it imports no renderer and only holds plain values; the backend that reads an option
decides what it means. The first option is ``three_d.off_screen`` (``True`` / ``False`` / ``"auto"``, default
``"auto"``), which the 3-D tier reads to decide whether PyVista renders off-screen, so a notebook embed works
without the caller ever setting ``pyvista.OFF_SCREEN``.

Examples:
    - Read the default, then override it and read it back:
        ```python
        >>> import digitalearth
        >>> digitalearth.get_option("three_d.off_screen")
        'auto'
        >>> digitalearth.set_option("three_d.off_screen", True)
        >>> digitalearth.get_option("three_d.off_screen")
        True
        >>> digitalearth.set_option("three_d.off_screen", "auto")

        ```
"""

from collections.abc import Callable
from typing import Any

__all__ = ["get_option", "set_option"]


def _valid_off_screen(value: Any) -> bool:
    """Whether ``value`` is an accepted ``three_d.off_screen`` setting (a bool, or ``"auto"``)."""
    return isinstance(value, bool) or value == "auto"


def _valid_web_inline(value: Any) -> bool:
    """Whether ``value`` is an accepted ``web.inline`` setting (``"widget"``, ``"static"`` or ``"auto"``)."""
    return value in ("widget", "static", "auto")


#: Each option's validator, default, and a remedy phrase naming its accepted values (for the invalid-value
#: error). The dotted name is the public key; a new option is one row here plus the tier code that reads it —
#: no new top-level function.
_REGISTRY: dict[str, tuple[Callable[[Any], bool], Any, str]] = {
    "three_d.off_screen": (_valid_off_screen, "auto", "True, False or 'auto'"),
    "web.inline": (_valid_web_inline, "auto", "'widget', 'static' or 'auto'"),
}

#: The live values, seeded from the registry's defaults. :func:`set_option` writes here; :func:`get_option`
#: reads here. Kept module-level (not a class) so the whole package shares one store.
_VALUES: dict[str, Any] = {name: default for name, (_, default, _remedy) in _REGISTRY.items()}


def _known() -> str:
    """A comma-separated list of the defined option names, for an error message."""
    return ", ".join(sorted(_REGISTRY))


def set_option(name: str, value: Any) -> None:
    """Set a digitalearth package option.

    Args:
        name: The dotted option name, e.g. ``"three_d.off_screen"``.
        value: The new value. Each option validates its own: ``three_d.off_screen`` takes ``True``, ``False``
            or ``"auto"``.

    Raises:
        ValueError: when ``name`` is not a defined option (the message lists the known names), or when
            ``value`` is not valid for that option.

    Examples:
        - Set the 3-D tier to always render off-screen (so a notebook embed never opens a window):
            ```python
            >>> import digitalearth
            >>> digitalearth.set_option("three_d.off_screen", True)
            >>> digitalearth.get_option("three_d.off_screen")
            True
            >>> digitalearth.set_option("three_d.off_screen", "auto")

            ```
        - An unknown option is refused, naming what is available:
            ```python
            >>> import digitalearth
            >>> digitalearth.set_option("three_d.nope", True)
            Traceback (most recent call last):
                ...
            ValueError: 'three_d.nope' is not a digitalearth option; known options: three_d.off_screen, web.inline

            ```
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"{name!r} is not a digitalearth option; known options: {_known()}"
        )
    validator, _, remedy = _REGISTRY[name]
    if not validator(value):
        raise ValueError(
            f"{value!r} is not a valid value for option {name!r}; {name!r} takes {remedy}"
        )
    _VALUES[name] = value


def get_option(name: str) -> Any:
    """Return the current value of a digitalearth package option.

    Args:
        name: The dotted option name, e.g. ``"three_d.off_screen"``.

    Returns:
        The option's current value (its default until :func:`set_option` changes it).

    Raises:
        ValueError: when ``name`` is not a defined option — a typo returns no silent ``None``; the message
            lists the known names.

    Examples:
        - The off-screen option reads back its default before anything sets it:
            ```python
            >>> import digitalearth
            >>> digitalearth.get_option("three_d.off_screen")
            'auto'

            ```
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"{name!r} is not a digitalearth option; known options: {_known()}"
        )
    return _VALUES[name]


def _reset_options() -> None:
    """Reset every option to its registered default (internal: used for test isolation).

    The store is process-wide configuration, set once and read where it matters (matplotlib's ``rcParams`` and
    pandas' options work the same way), so it is not thread-safe and a program configures it from one thread.
    This restores the defaults in one call so a test cannot leak a setting into the next. It is intentionally
    private: the public surface is :func:`set_option` / :func:`get_option` only.
    """
    _VALUES.clear()
    _VALUES.update({name: default for name, (_, default, _remedy) in _REGISTRY.items()})
