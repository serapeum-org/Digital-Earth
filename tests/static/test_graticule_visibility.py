"""A graticule built hidden has to be *drawn* hidden, on a globe as on a flat map (#333).

The graticule is the one static layer whose artists are not made by its own drawer. ``draw_graticule``
computes the polylines and hands them to the map; the line artists themselves appear later, when
:meth:`~digitalearth.static.maps.projection.ProjectionMixin._apply_frame` puts the globe frame on. The
funnel that applies a layer's ``visible`` flag — ``Renderer.draw_layer`` — has therefore already run and
found nothing on the axes to toggle, so the flag reached no artist and the grid was drawn over a figure that
described it as hidden.

The claims here are read off the **artists**, not off the description or off ``is_visible``: the whole defect
was that those two disagreed with what matplotlib was left holding, so a test that asked the description
would have passed throughout.
"""

import pytest

from digitalearth.static import Map

#: An equal-area world projection with a limb, so ``globe=True`` really applies a frame and the frame really
#: draws the graticule. Named rather than repeated, because every test here needs the same one.
WORLD = "ESRI:54030"


@pytest.fixture
def globe():
    """Yield a globe map, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` whose frame draws a graticule.
    """
    scene = Map(crs=WORLD, globe=True)
    yield scene
    scene.close()


def _drawn_graticule(scene):
    """Return the graticule's described flag and the visibility its artists actually carry.

    Args:
        scene: A rendered globe map with one graticule.

    Returns:
        ``(described, drawn_flags)`` — the layer's ``visible`` as the figure states it, and one flag per
        matplotlib artist the frame drew for it. The artists are the evidence: reading
        ``Renderer.is_visible`` instead would have answered from the same record the defect corrupted.
    """
    layer = scene.figure_spec.layers.layers[-1]
    drawn = scene._renderer.drawn[layer.id]
    return layer.visible, [artist.get_visible() for artist in drawn.artists]


class TestTheFrameHonoursTheDescribedFlag:
    """What the figure says about a graticule and what the axes holds have to be the same thing."""

    def test_a_graticule_created_hidden_puts_no_visible_line_on_the_axes(self, globe):
        """The creating call, which is where #333 was measured.

        Args:
            globe: A globe map whose frame draws the grid.

        Test scenario:
            ``graticule(visible=False)`` on a fresh map creates the layer rather than replacing one, so
            nothing here depends on a second call. The frame is applied by ``render()``, which is the step
            that puts the lines on the axes — and the step after which the flag had never been applied.
        """
        globe.graticule(spacing=30, visible=False)
        globe.render()
        described, flags = _drawn_graticule(globe)
        assert flags, (
            "the frame drew no graticule lines, so this proves nothing about hiding them"
        )
        assert (described, set(flags)) == (False, {False}), (described, flags)

    def test_a_graticule_created_visible_is_still_drawn(self, globe):
        """The control: hiding the hidden one must not hide every one.

        Args:
            globe: A globe map whose frame draws the grid.
        """
        globe.graticule(spacing=30)
        globe.render()
        described, flags = _drawn_graticule(globe)
        assert (described, set(flags)) == (True, {True}), (described, flags)

    def test_a_replacing_call_that_hides_it_hides_the_lines_too(self, globe):
        """A second call replaces the one layer, and its flag has to reach the frame as well.

        Args:
            globe: A globe map whose frame draws the grid.

        Test scenario:
            The replacing call is the one shape of this method that was never in doubt for the
            *description* — it keeps the id and takes the new flag — so it is the shape that shows the
            defect was in the drawing rather than in the describing.
        """
        globe.graticule(spacing=30)
        globe.graticule(spacing=15, visible=False)
        globe.render()
        described, flags = _drawn_graticule(globe)
        assert (described, set(flags)) == (False, {False}), (described, flags)

    def test_the_globe_outline_is_not_hidden_with_the_grid(self, globe):
        """Hiding the graticule must leave the projection boundary on the axes.

        Args:
            globe: A globe map whose frame draws the grid.

        Test scenario:
            The frame draws the boundary patch and the grid in one call, and the layer owns only the grid.
            A fix that hid *everything the frame added* would take the globe's outline with it, which is the
            failure the artist-splitting in ``_apply_frame`` exists to prevent — so the patch it returns is
            checked here rather than assumed.
        """
        globe.graticule(spacing=30, visible=False)
        patch = globe._apply_frame()
        assert patch.get_visible() is True, (
            "the boundary is the frame's, not the graticule's"
        )

    def test_the_renderer_agrees_with_the_axes(self, globe):
        """``is_visible`` is the read-back a switcher uses, so it has to answer from hidden artists.

        Args:
            globe: A globe map whose frame draws the grid.
        """
        globe.graticule(spacing=30, visible=False)
        layer_id = globe.figure_spec.layers.layers[-1].id
        globe.render()
        assert globe._renderer.is_visible(layer_id) is False, globe._renderer.drawn[
            layer_id
        ]
