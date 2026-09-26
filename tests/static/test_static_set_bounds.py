"""The static tier's framing method, under the Core name (order 27a, #265).

The Core contract calls it ``set_bounds``; the web and 3-D tiers already do, web's own ``fit_bounds`` having
become a deprecated alias when its seam landed. The static tier called it ``set_extent`` and returned
``None``, so a caller writing one script against two tiers had to spell the same intent two ways and could
not chain it on one of them.

``set_extent`` is a live alias now, and it was deliberately **not** in ``PLANNED_RENAMES``: the two are not
the same method under two names, because the Core declares ``set_bounds(padding=)`` and a ``None`` that fits
the data, and when 27a landed this tier had neither. Order 27a takes the *name*; the auto-framing behind it
is the framing order, which has since built both — see ``tests/static/test_static_auto_framing.py``. What
this module holds is the part 27a is responsible for: that the Core name frames the axes, that it chains,
and that the old spelling still frames them identically.
"""

import warnings

import pytest

from digitalearth.base.spec import Bounds
from digitalearth.static import Map

#: A rectangle in a CRS that is not the display one, so the reprojection path is exercised rather than
#: skipped. ``Bounds`` is the form the method exists for: it states the ordering and the CRS instead of
#: leaving both to position and assumption.
ELSEWHERE = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)

#: The same frame as a bare sequence, in matplotlib's own ordering and in the display CRS.
HERE = [0.0, 100.0, 0.0, 50.0]


@pytest.fixture
def drawn():
    """Yield an empty Web-Mercator map, closed on the way out.

    Yields:
        The tier's ``Map``.
    """
    scene = Map(crs=3857)
    yield scene
    scene.close()


def _limits(scene):
    """Return the axes limits a frame actually produced.

    Args:
        scene: The map to read.

    Returns:
        ``((xmin, xmax), (ymin, ymax))`` as plain floats — what matplotlib was left holding, which is the
        only evidence that a framing call framed anything.
    """
    return (
        tuple(float(value) for value in scene.ax.get_xlim()),
        tuple(float(value) for value in scene.ax.get_ylim()),
    )


class TestTheCoreNameFramesTheAxes:
    """``set_bounds`` is the name the contract declares, and it has to do the work."""

    def test_a_rectangle_in_another_crs_is_reprojected(self, drawn):
        """The frame lands where the data is, not where its numbers would be read literally.

        Args:
            drawn: The map under test.
        """
        drawn.set_bounds(ELSEWHERE)
        assert round(_limits(drawn)[0][1]) == 111319, _limits(drawn)

    def test_a_bare_sequence_is_taken_in_the_display_crs(self, drawn):
        """The ordering that was this method's contract keeps working under the new name.

        Args:
            drawn: The map under test.
        """
        drawn.set_bounds(HERE)
        assert _limits(drawn) == ((0.0, 100.0), (0.0, 50.0)), _limits(drawn)

    def test_it_chains(self, drawn):
        """The Core declares ``returns="self"``, which is what makes a one-line setup possible.

        Args:
            drawn: The map under test.

        Test scenario:
            ``set_extent`` returned ``None``, so `Map(...).set_extent(bbox).coastlines()` raised
            ``AttributeError`` on the tier that draws most of the package's figures while the same line
            worked on web. Asserted as identity rather than truthiness: a method returning any object would
            satisfy "not None" and still break a chain.
        """
        assert drawn.set_bounds(HERE) is drawn

    def test_a_sequence_of_the_wrong_length_is_refused_by_the_new_name(self, drawn):
        """The refusal has to name the method the caller wrote, not the one it used to be.

        Args:
            drawn: The map under test.
        """
        with pytest.raises(ValueError) as refused:
            drawn.set_bounds([0.0, 1.0, 2.0])
        assert "set_bounds needs exactly 4 values" in str(refused.value), refused.value


class TestTheOldSpellingStillFramesTheSameWay:
    """``set_extent`` is a promise to every script already written against this tier."""

    @pytest.mark.parametrize("frame", (ELSEWHERE, HERE), ids=("bounds", "sequence"))
    def test_the_alias_frames_the_axes_identically(self, frame):
        """Both spellings must leave matplotlib holding the same limits.

        Args:
            frame: The rectangle to frame on, in each accepted form.

        Test scenario:
            Compared on what the axes were left with rather than on the return value: an alias forwarding to
            the wrong method would still hand back a ``Map``.
        """
        under_core = Map(crs=3857)
        under_old = Map(crs=3857)
        try:
            under_core.set_bounds(frame)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                under_old.set_extent(frame)
            assert _limits(under_old) == _limits(under_core), (
                f"set_extent framed {_limits(under_old)} and set_bounds framed {_limits(under_core)}"
            )
        finally:
            under_core.close()
            under_old.close()

    def test_the_alias_warns_and_names_the_core_spelling(self, drawn):
        """The warning has to tell the caller what to write instead.

        Args:
            drawn: The map under test.
        """
        with pytest.warns(DeprecationWarning) as caught:
            drawn.set_extent(HERE)
        assert "Map.set_bounds()" in str(caught[0].message), caught[0].message

    def test_the_warning_points_at_the_callers_own_line(self, drawn):
        """A warning blaming a line inside the package is one nobody can act on.

        Args:
            drawn: The map under test.
        """
        with pytest.warns(DeprecationWarning) as caught:
            drawn.set_extent(HERE)
        assert caught[0].filename == __file__, caught[0].filename


class TestNoInternalCallerGoesThroughTheAlias:
    """Two methods on this tier frame the axes by calling the framing method, and both had to move.

    ``set_domain`` resolves a named region and frames on it; the globe frame re-applies the limits it
    computed. Either left on ``set_extent`` would warn a caller who wrote neither.
    """

    def test_set_domain_does_not_warn(self, drawn):
        """A named region frames the axes without deprecating anything.

        Args:
            drawn: The map under test.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            drawn.set_domain("europe")
        blamed = [
            str(record.message)
            for record in caught
            if issubclass(record.category, DeprecationWarning)
            and "set_extent" in str(record.message)
        ]
        assert blamed == [], f"set_domain reaches the deprecated spelling: {blamed}"

    def test_a_globe_frame_does_not_warn(self):
        """The globe frame re-applies its own limits, and must do it under the Core name."""
        from digitalearth.static import projections

        scene = Map(crs=projections.orthographic(lon=4.5, lat=53.3), globe=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DeprecationWarning)
                scene.set_global()
            blamed = [
                str(record.message)
                for record in caught
                if issubclass(record.category, DeprecationWarning)
                and "set_extent" in str(record.message)
            ]
        finally:
            scene.close()
        assert blamed == [], (
            f"the globe frame reaches the deprecated spelling: {blamed}"
        )
