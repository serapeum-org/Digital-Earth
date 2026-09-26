"""The static tier's framing method, under the Core name (order 27a, #265).

The Core contract calls it ``set_bounds``, and every tier answers to it. The static tier used to spell the
same intent differently and returned ``None``, so a caller writing one script against two tiers had to write
the same intent two ways and could not chain it on one of them. That spelling is gone rather than deprecated —
nothing here is released — so what is left to hold is the method itself.

Taking the *name* was order 27a's half; the auto-framing behind it — ``padding=`` and the ``None`` that fits
the data — is the framing order, which has since built both (see ``tests/static/test_static_auto_framing.py``).
What this module holds is that the Core name frames the axes, in either accepted form, and that it chains.
"""

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
            The spelling this tier used to carry returned ``None``, so
            ``Map(...).set_bounds(bbox).coastlines()`` raised ``AttributeError`` on the tier that draws most
            of the package's figures while the same line worked on web. Asserted as identity rather than
            truthiness: a method returning any object would satisfy "not None" and still break a chain.
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

    def test_the_two_accepted_forms_frame_the_axes_the_same_way(self):
        """A ``Bounds`` in the display CRS and the bare sequence of its own numbers must agree.

        Test scenario:
            The reprojecting branch and the literal one are different code paths to one frame. Given a
            ``Bounds`` already in the display CRS, reprojection is the identity, so the two must leave
            matplotlib holding the same limits — measured on the axes rather than on the return value,
            since a method forwarding to the wrong place would still hand back a ``Map``.
        """
        west, east, south, north = HERE
        from_bounds = Map(crs=3857)
        from_sequence = Map(crs=3857)
        try:
            from_bounds.set_bounds(Bounds(west, south, east, north, crs=3857))
            from_sequence.set_bounds(HERE)
            assert _limits(from_bounds) == _limits(from_sequence), (
                f"a Bounds framed {_limits(from_bounds)} and a sequence framed "
                f"{_limits(from_sequence)}"
            )
        finally:
            from_bounds.close()
            from_sequence.close()
