"""Tests for :mod:`digitalearth.base.deprecation` — where a renamed-parameter warning is *attributed*.

A ``DeprecationWarning`` is only actionable if it names the line the caller has to edit. ``warnings.warn``
decides that from ``stacklevel``, a hand-counted number of frames, so every frame between the helper and the
user has to be accounted for — and the round-2 review found the counts were measured for direct calls only
(**M8**, **M9**): each in-repo alias adds a frame nobody counted, and the warning then lands *inside* the
package, on a file the caller does not own.

These tests assert the **filename** the warning is attributed to, not merely that one was raised: a warning
pointing at ``vector.py`` or ``export.py`` still matches ``pytest.warns(DeprecationWarning)`` while telling the
user nothing.
"""

import warnings

import pytest

from digitalearth.base.deprecation import renamed_parameter

HERE = __file__


def _public_method(size=None, radius=None):
    """Stand in for a public method that resolves a renamed parameter itself.

    Args:
        size: The current spelling.
        radius: The deprecated spelling.

    Returns:
        The resolved value.
    """
    return renamed_parameter(
        new="size",
        value=size,
        old="radius",
        alias=radius,
        caller="Fake.points()",
        default=5.0,
    )


def _private_resolver(size=None, radius=None):
    """Stand in for a private resolver sitting between the helper and the public method.

    Args:
        size: The current spelling.
        radius: The deprecated spelling.

    Returns:
        The resolved value.
    """
    return renamed_parameter(
        new="size",
        value=size,
        old="radius",
        alias=radius,
        caller="Fake.points()",
        default=5.0,
        stacklevel=4,
    )


def _method_with_a_resolver(size=None, radius=None):
    """Stand in for the public method that delegates to :func:`_private_resolver`.

    Args:
        size: The current spelling.
        radius: The deprecated spelling.

    Returns:
        The resolved value.
    """
    return _private_resolver(size=size, radius=radius)


def _warn_from(call, **kwargs):
    """Run ``call`` and return the single warning it emitted.

    Args:
        call: The callable to run under a recording filter.
        **kwargs: Keyword arguments for ``call``.

    Returns:
        The recorded :class:`warnings.WarningMessage`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(**kwargs)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, [str(w.message) for w in caught]
    return deprecations[0]


class TestTheWarningLandsOnTheCallersLine:
    """The documented stacklevel contract, asserted by filename rather than by counting frames by eye."""

    def test_the_default_points_at_the_caller_of_a_public_method(self):
        """``stacklevel=3`` is right when a public method calls the helper directly.

        Test scenario:
            helper -> method -> user. The warning must be attributed to *this* file, the one holding the
            offending keyword, and never to ``deprecation.py``.
        """
        recorded = _warn_from(_public_method, radius=9.0)
        assert recorded.filename == HERE, recorded.filename

    def test_four_is_right_when_a_private_resolver_sits_in_between(self):
        """``stacklevel=4`` is right when a private resolver adds a frame.

        Test scenario:
            helper -> resolver -> method -> user. This is the shape the static tier's
            ``_resolve_marker_size`` has, and the reason the parameter exists at all.
        """
        recorded = _warn_from(_method_with_a_resolver, radius=9.0)
        assert recorded.filename == HERE, recorded.filename

    def test_the_message_names_both_spellings_and_the_call(self):
        """The text has to carry the call site too, because a wrapper can still swallow the stack."""
        recorded = _warn_from(_public_method, radius=9.0)
        message = str(recorded.message)
        assert "Fake.points()" in message and "radius=" in message, message
        assert "use size= instead" in message, message

    def test_both_spellings_at_once_is_a_type_error_and_warns_about_neither(self):
        """Two names for one parameter is a caller error, not something to resolve silently."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match="pass only size="):
                _public_method(size=4.0, radius=9.0)
        assert caught == [], [str(w.message) for w in caught]


class TestTheStaticTierEntryPointsAreAttributedCorrectly:
    """The direct static-tier call site the review measured as correct, pinned so it stays correct."""

    def test_grid_points_points_at_the_caller(self, dataset):
        """``Map.grid_points(ds, point_size=8)`` attributes its warning to the caller's file.

        Args:
            dataset: The raster to scatter.

        Test scenario:
            ``grid_points`` is the direct call site ``_resolve_marker_size``'s ``stacklevel=4`` was counted
            for, so it is the control for the alias case ``Map.point_cloud`` (review M8).
        """
        import matplotlib

        matplotlib.use("Agg")
        from digitalearth.static import Map

        scene = Map(crs=dataset.epsg)
        recorded = _warn_from(lambda: scene.grid_points(dataset, point_size=8))
        assert recorded.filename == HERE, recorded.filename
        assert "Map.grid_points()" in str(recorded.message), str(recorded.message)
