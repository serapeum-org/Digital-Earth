"""The static tier's framing method, under the Core name (order 27a, #265).

The Core contract calls it ``set_bounds``, and every tier answers to it. The static tier used to spell the
same intent differently and returned ``None``, so a caller writing one script against two tiers had to write
the same intent two ways and could not chain it on one of them. That spelling is gone rather than deprecated —
nothing here is released — so what is left to hold is the method itself.

Taking the *name* was order 27a's half; the auto-framing behind it — ``padding=`` and the ``None`` that fits
the data — is the framing order, which has since built both (see ``tests/static/test_static_auto_framing.py``).
What this module holds is that the Core name frames the axes, in either accepted form, and that it chains.

It also holds the **spelling of the argument**, which the rename got wrong and nothing caught (review H2). The
method kept ``set_extent``'s parameter name (``bbox``) and ``set_extent``'s matplotlib ordering
(``[xmin, xmax, ymin, ymax]``), while the other two tiers read ``bounds=(west, south, east, north)`` — so one
sequence framed two different rectangles and a keyword call ported in neither direction. The package's own
rectangle type is ``Bounds(xmin, ymin, xmax, ymax)`` and its ``from_bbox`` takes that same order, so bbox order
is what a bare sequence means here now. :class:`TestTheArgumentIsSpelledLikeTheRestOfThePackage` is the half
that fails on the old spelling.
"""

import pytest

from digitalearth.base.spec import Bounds
from digitalearth.static import Map

#: A rectangle in a CRS that is not the display one, so the reprojection path is exercised rather than
#: skipped. ``Bounds`` is the form the method exists for: it states the ordering and the CRS instead of
#: leaving both to position and assumption.
ELSEWHERE = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)

#: The same frame as a bare sequence, in bbox order — ``(west, south, east, north)`` — and in the display CRS.
#: That is the order ``Bounds`` itself holds its four numbers in, and the order the other two tiers read.
HERE = [0.0, 0.0, 100.0, 50.0]

#: The axes limits `HERE` has to produce, as matplotlib reports them: ``((xmin, xmax), (ymin, ymax))``.
HERE_LIMITS = ((0.0, 100.0), (0.0, 50.0))


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
        """A four-value sequence is read as ``(west, south, east, north)`` and framed as it is.

        Args:
            drawn: The map under test.
        """
        drawn.set_bounds(HERE)
        assert _limits(drawn) == HERE_LIMITS, _limits(drawn)

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
        west, south, east, north = HERE
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


class TestTheArgumentIsSpelledLikeTheRestOfThePackage:
    """One Core name has to take one argument, spelled one way and read in one order (review H2).

    The rename took the name and left ``set_extent``'s argument behind it: the parameter stayed ``bbox`` and
    the sequence stayed matplotlib-ordered, while the interactive and web tiers both read
    ``bounds=(west, south, east, north)``. ``set_bounds([0, 10, 0, 50])`` therefore framed x 0→10 / y 0→50
    here and x 0→0 — a degenerate frame, accepted in silence — on the interactive tier, and a keyword call
    written for either tier raised ``TypeError`` on the other.
    """

    def test_the_sequence_is_read_in_bbox_order(self, drawn):
        """``[0, 10, 0, 50]`` is west/south/east/north, so it frames x 0→10 and y 0→50.

        Args:
            drawn: The map under test.

        Test scenario:
            The literal the review reproduced with, chosen because the two orders disagree about it loudly:
            read as matplotlib's it is x 0→10 / y 0→50, read as bbox it is x 0→0 / y 10→50. Asserted on the
            axes rather than on the viewport, because the axes are what the figure is drawn against.
        """
        drawn.set_bounds([0.0, 0.0, 10.0, 50.0])
        assert _limits(drawn) == ((0.0, 10.0), (0.0, 50.0)), _limits(drawn)

    def test_the_matplotlib_ordering_is_gone_rather_than_also_accepted(self, drawn):
        """The losing order is not a second spelling, and a sequence written in it lands somewhere else.

        Args:
            drawn: The map under test.

        Test scenario:
            An ordering cannot be guessed from four numbers, so accepting both would be the defect with a
            second branch. ``[0, 10, 20, 30]`` is a valid rectangle under either reading and a *different*
            one under each — matplotlib's is x 0→10 / y 20→30, bbox order is x 0→20 / y 10→30 — so the frame
            says which of the two survived without a degenerate box muddying the answer.
        """
        drawn.set_bounds([0.0, 10.0, 20.0, 30.0])
        assert _limits(drawn) == ((0.0, 20.0), (10.0, 30.0)), _limits(drawn)

    def test_a_flipped_literal_is_read_in_bbox_order_too(self, drawn):
        """A backwards pair does not make the four numbers order-agnostic.

        Args:
            drawn: The map under test.

        Test scenario:
            The flipped-axis contract was preserved across the re-order, and the two literals that prove it
            (``[10, 0, 0, 10]`` and ``[0, 10, 10, 0]``, in `test_static_auto_framing.py`) hold **equal values**
            in positions 2 and 3 — the two the re-order swaps — so they read the same either way and say
            nothing about the ordering (review R2-L6). ``[10, 0, 5, 10]`` is the flipped literal whose middle
            pair differs: bbox order frames x 10→5 / y 0→10, the retired matplotlib order framed x 10→0 /
            y 5→10. Read off the axes, because the inversion is the axes' and not the region's.
        """
        drawn.set_bounds([10.0, 0.0, 5.0, 10.0])
        assert _limits(drawn) == ((10.0, 5.0), (0.0, 10.0)), _limits(drawn)

    def test_the_keyword_is_bounds(self, drawn):
        """A keyword call written against any tier reaches this one.

        Args:
            drawn: The map under test.

        Test scenario:
            ``bbox=`` was the old parameter name, so ``set_bounds(bounds=...)`` raised ``TypeError`` here and
            ``set_bounds(bbox=...)`` raised it on the other two. Passed by keyword on purpose: the positional
            form would pass whatever the parameter were called.
        """
        drawn.set_bounds(bounds=HERE)
        assert _limits(drawn) == HERE_LIMITS, _limits(drawn)

    def test_the_old_keyword_is_refused(self, drawn):
        """``bbox=`` is deleted, not aliased — nothing here is released.

        Args:
            drawn: The map under test.

        Test scenario:
            An alias would keep the two orderings alive under two names, which is the state this replaces.
            The refusal is Python's own, so it names the parameter the caller wrote.
        """
        with pytest.raises(TypeError) as refused:
            drawn.set_bounds(bbox=HERE)
        assert "bbox" in str(refused.value), refused.value

    def test_the_refusal_names_the_order_it_reads(self, drawn):
        """A wrong-length sequence has to say which four values were wanted.

        Args:
            drawn: The map under test.

        Test scenario:
            The message taught the matplotlib ordering — ``[xmin, xmax, ymin, ymax]`` — which was the
            clearest place the wrong order was documented as the contract.
        """
        with pytest.raises(ValueError) as refused:
            drawn.set_bounds([0.0, 1.0, 2.0])
        assert "(west, south, east, north)" in str(refused.value), refused.value
