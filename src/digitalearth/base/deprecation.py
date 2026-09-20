"""deprecation — the one rule every backend follows when a parameter is renamed.

A rename is a promise to two people at once: the caller who already wrote the old spelling (their call must
keep working) and the caller who reads the new signature (the old name must not quietly linger). Both tiers of
that promise are kept by :func:`renamed_parameter`, which is the *only* place in the package where a renamed
parameter is resolved.

One helper, not four, because four hand-written copies is exactly how the contract drifted: the tiers had each
grown their own resolver and they disagreed on the case that matters — a caller passing **both** spellings got
a ``TypeError`` on static, a ``ValueError`` on 3-D, the new value on interactive and the *old* value on web.
Two names for one parameter is a caller error, and Python already raises ``TypeError`` for an argument given
twice, so that is what this raises — naming both spellings and the one to keep, at the call site, instead of
hiding the mistake in someone's output.

The module is pure standard library (``warnings`` only), so it sits in ``base/`` without threatening the
engine-neutrality guard (``tests/test_base_is_engine_neutral.py``).

**The call shape it expects.** The new parameter takes ``None`` as its "was it passed?" sentinel and the
deprecated one does too; the *real* default is handed to this function as ``default`` rather than written into
the signature, because a signature default is indistinguishable from a value the caller typed — and telling
those apart is the whole job here. Document the effective default in the parameter's docstring line.
"""

import warnings
from contextvars import ContextVar
from typing import Any, Callable, Optional

#: How many frames a deprecated method alias has added between the caller and the code that is running. A
#: renamed method forwards to its replacement, so a parameter rename resolved *inside* that replacement is one
#: frame further from the caller than it would otherwise be — and a warning pointing at this module instead of
#: at the call is one nobody can act on. The alias counts itself here; `renamed_parameter` adds it to its own
#: stacklevel, which is the only place the two helpers need to know about each other.
_ALIAS_DEPTH: ContextVar = ContextVar("digitalearth_alias_depth", default=0)

#: What the warning says about when the old spelling stops working. A phrase rather than a version because CI
#: (commitizen) owns the version number: naming a release here would either go stale or have to be bumped by
#: hand in every call site. Pass ``removed_in="0.12.0"`` once a removal is actually scheduled.
REMOVED_IN = "a future release"


def renamed_method(
    *,
    new: str,
    old: str,
    owner: str,
    removed_in: str = REMOVED_IN,
) -> Callable[..., Any]:
    """Build the alias that keeps a renamed method working, warning on the caller's line.

    The method counterpart of :func:`renamed_parameter`, and for the same reason: the one hand-written alias in
    the package (`WebMap.to_gif`) had its own message shape, and a second hand-written one would have had a
    third. A rename is a promise to two callers — the one who already wrote the old name, and the one reading
    the new signature — and this keeps both.

    Args:
        new: The method that does the work now, looked up on the instance when the alias is called, so a
            subclass that overrides it is the one that runs.
        old: The deprecated spelling, named in the warning.
        owner: The class the methods live on, for the message — `"WebMap"`, `"Scene3D"`.
        removed_in: When the old spelling stops working. A phrase by default, since CI owns the version
            number; pass a real version once a removal is scheduled.

    Returns:
        A function to assign as the old name. It forwards every argument to `new` and returns what that
        returns, so a chainable builder stays chainable.

    Warns:
        DeprecationWarning: naming the old spelling, the new one and when the old one goes. The warning points
            at the caller's line (`stacklevel=2`), not at this module, because that is the line to change.

    Examples:
        - The alias forwards, and says what to write instead:
            ```python
            >>> import warnings
            >>> from digitalearth.base.deprecation import renamed_method
            >>> class Map:
            ...     def set_bounds(self, bounds):
            ...         return f"framed on {bounds}"
            ...     fit_bounds = renamed_method(new="set_bounds", old="fit_bounds", owner="Map")
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     Map().fit_bounds([0, 0, 1, 1])
            'framed on [0, 0, 1, 1]'
            >>> print(caught[0].message)
            Map.fit_bounds() is deprecated and will be removed in a future release; use Map.set_bounds() instead

            ```
        - Keywords travel too, so a caller's whole call keeps working:
            ```python
            >>> import warnings
            >>> from digitalearth.base.deprecation import renamed_method
            >>> class Map:
            ...     def save_animation(self, path, *, fps=10):
            ...         return f"{path} at {fps} fps"
            ...     animate = renamed_method(new="save_animation", old="animate", owner="Map")
            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("ignore")
            ...     Map().animate("out.gif", fps=24)
            'out.gif at 24 fps'

            ```
    """

    def alias(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Warn, then call the method this name was renamed to.

        Args:
            self: The instance the alias was called on.
            *args: Passed straight through.
            **kwargs: Passed straight through.

        Returns:
            Whatever the new method returns.
        """
        warnings.warn(
            f"{owner}.{old}() is deprecated and will be removed in {removed_in}; "
            f"use {owner}.{new}() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # The forwarding call is a frame the caller did not write, so a parameter rename resolved inside the
        # new method counts it and still points at the caller's line.
        token = _ALIAS_DEPTH.set(_ALIAS_DEPTH.get() + 1)
        try:
            return getattr(self, new)(*args, **kwargs)
        finally:
            _ALIAS_DEPTH.reset(token)

    alias.__name__ = old
    alias.__qualname__ = f"{owner}.{old}"
    alias.__doc__ = (
        f"Deprecated spelling of :meth:`{new}`; it forwards there and warns.\n\n"
        f"        Returns:\n            Whatever :meth:`{new}` returns.\n"
    )
    return alias


def renamed_parameter(
    *,
    new: str,
    value: Any,
    old: str,
    alias: Any,
    caller: str,
    default: Any = None,
    convert: Optional[Callable[[Any], Any]] = None,
    removed_in: str = REMOVED_IN,
    stacklevel: int = 3,
) -> Any:
    """Resolve a renamed parameter, warning on the old spelling and refusing both at once.

    Args:
        new: The parameter's current name — the one the caller should be writing.
        value: What the caller passed under ``new``; ``None`` means "not passed".
        old: The deprecated spelling, named in both the warning and the error.
        alias: What the caller passed under ``old``; ``None`` means "not passed".
        caller: The public method the keywords were written on (e.g. ``"Map.scatter()"``), so both messages
            say *where* the offending call is even when the stack does not survive a wrapper.
        default: The new parameter's real default — returned when neither spelling was given. Keep the
            signature's default at ``None`` and put the effective one here; see the module docstring.
        convert: Optional callable turning the old value into the new parameter's units. A pure rename passes
            the value straight through; web's ``duration=`` (seconds per frame) becomes ``fps`` this way, so
            an old call still produces the animation it always did.
        removed_in: The release the old spelling stops working in, as the warning phrases it.
        stacklevel: Frames to skip so the warning points at the **user's** call, not at this helper. ``3`` is
            right when a public method calls this directly (helper → method → user); pass ``4`` when a private
            resolver sits in between (helper → resolver → method → user).

    Returns:
        The value to use for ``new``: the caller's ``value`` when only the new spelling was given, the
        converted ``alias`` when only the old one was, and ``default`` when neither was.

    Raises:
        TypeError: when both spellings are passed. They name one parameter, so two values for it cannot both
            be honoured, and preferring either silently hides the mistake.

    Warns:
        DeprecationWarning: when ``old`` was the spelling used, naming ``new`` as its replacement.

    Examples:
        - Only the new spelling: the value passes through, silently.
            ```python
            >>> import warnings
            >>> from digitalearth.base.deprecation import renamed_parameter
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     renamed_parameter(
            ...         new="size", value=9.0, old="radius", alias=None,
            ...         caller="WebMap.points()", default=5.0,
            ...     )
            9.0
            >>> caught
            []

            ```
        - Neither spelling: the documented default, still silently.
            ```python
            >>> from digitalearth.base.deprecation import renamed_parameter
            >>> renamed_parameter(
            ...     new="size", value=None, old="radius", alias=None,
            ...     caller="WebMap.points()", default=5.0,
            ... )
            5.0

            ```
        - The old spelling alone: it still works, and the warning names what to write instead.
            ```python
            >>> import warnings
            >>> from digitalearth.base.deprecation import renamed_parameter
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     value = renamed_parameter(
            ...         new="size", value=None, old="radius", alias=9.0,
            ...         caller="WebMap.points()", default=5.0,
            ...     )
            >>> value, str(caught[0].message)
            (9.0, 'WebMap.points(): radius= is deprecated and will be removed in a future release; use size= instead')

            ```
        - A unit change is converted, never reinterpreted:
            ```python
            >>> import warnings
            >>> from digitalearth.base.deprecation import renamed_parameter
            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("ignore")
            ...     renamed_parameter(
            ...         new="fps", value=None, old="duration", alias=0.5,
            ...         caller="WebMap.animate()", default=3.0,
            ...         convert=lambda seconds: 1.0 / float(seconds),
            ...     )
            2.0

            ```
        - Both spellings at once: a caller error, reported as one.
            ```python
            >>> from digitalearth.base.deprecation import renamed_parameter
            >>> renamed_parameter(
            ...     new="fps", value=4.0, old="framerate", alias=9.0,
            ...     caller="orbit()", default=3.0,
            ... )
            Traceback (most recent call last):
                ...
            TypeError: orbit() got both fps= and the deprecated framerate=; they name one parameter, so pass only fps=

            ```
    """
    if alias is None:
        return default if value is None else value
    if value is not None:
        raise TypeError(
            f"{caller} got both {new}= and the deprecated {old}=; they name one parameter, "
            f"so pass only {new}="
        )
    warnings.warn(
        f"{caller}: {old}= is deprecated and will be removed in {removed_in}; use {new}= instead",
        DeprecationWarning,
        # Plus whatever a deprecated method alias put between the caller and here (see `_ALIAS_DEPTH`).
        stacklevel=stacklevel + _ALIAS_DEPTH.get(),
    )
    return alias if convert is None else convert(alias)
