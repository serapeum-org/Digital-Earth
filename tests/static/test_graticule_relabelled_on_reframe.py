"""A labelled graticule is re-labelled when the view moves under it (R2-M2, R2-L14).

Degree labels are placed against the view the ``graticule()`` call can see, and a flat map has no
render-time pass to revisit them — only a globe does, through ``_apply_frame``. So a grid built before the
framing call kept the labels of a view the figure no longer shows. Measured on the parent (``23d25629``),
on ``Map(crs=4326)``, counting the ``Text`` artists the layer owns and testing each one's position against
the axes limits:

```
graticule(30) THEN set_domain('europe')
    xlim=[-25.0, 45.0] ylim=[34.0, 72.0]  drawn={'LineCollection': 1, 'Text': 18}
    labels inside axes (0): []
    labels OUTSIDE axes (18): ['180°','150°W','120°W','90°W','60°W','30°W','0°','30°E','60°E','90°E',
                               '120°E','150°E','180°','60°S','30°S','0°','30°N','60°N']
    warnings: none
set_domain('europe') THEN graticule(30)
    labels inside axes (3): ['0°', '30°E', '60°N']
set_domain('europe') THEN graticule(30) THEN set_global()
    xlim=[-180.0, 180.0] ylim=[-90.0, 90.0]  drawn={'LineCollection': 1, 'Text': 3}
    labels inside axes (3): ['0°', '30°E', '60°N']
```

Two shapes, one cause. The first renders a bare grid — all 18 labels sit outside the axes, and off the
canvas entirely, since each is drawn with ``clip_on=False`` — while the record and the ``FigureSpec`` both
still say the layer carries 18. The third is the one that is visually *wrong* rather than merely empty: a
world map at 30 degrees carrying three labels floating mid-Atlantic. Neither warned.

The labels are the **only** thing this tier positions against the call-time view. The polylines span the
world whatever the map is looking at (``projections.graticule`` reads the display CRS and the two steps, not
the limits), and the globe's boundary ring and limits come from ``_frame()``, which is keyed on the CRS and
applied at render time. That is asserted below rather than asserted in prose, by counting the collection's
segments across the same reframe that moves the labels.

Every claim is read off the **axes** — the artists' own positions against the limits — and never off the
description, because the defect was exactly that the description was right while the figure was not.
"""

import warnings

import pytest
from matplotlib.collections import LineCollection
from matplotlib.text import Text

from digitalearth.static import Map

#: The step every grid here is cut at, in degrees.
STEP = 30.0

#: A domain the measurements above were taken on. Its view holds three of the 18 lines a 30-degree grid
#: draws, which is what makes the two orders tell each other apart.
DOMAIN = "europe"

#: The degrees the ``europe`` view holds at :data:`STEP`, in the order the labels are placed (meridians
#: first). Written out rather than read off a map, so neither order is compared with itself.
EUROPE_DEGREES = ["0°", "30°E", "60°N"]

#: The degrees a world view holds at :data:`STEP` — 13 meridians and 5 parallels, both datelines labelled.
WORLD_DEGREE_COUNT = 18


def _labels(scene):
    """Return the degree labels the graticule layer owns, as the strings they draw.

    Args:
        scene: A map whose graticule has been drawn.

    Returns:
        The text of every :class:`~matplotlib.text.Text` artist the layer holds, in placement order.
    """
    drawn = scene._renderer.drawn["graticule-1"]
    return [artist.get_text() for artist in drawn.artists if isinstance(artist, Text)]


def _labels_inside(scene):
    """Return the layer's degree labels whose anchor the axes is actually showing.

    Args:
        scene: A map whose graticule has been drawn.

    Returns:
        The text of each label positioned inside the current axes limits. Read off the artists' own
        positions rather than off the window the placement used, which is the half that was wrong.
    """
    xmin, xmax = sorted(float(value) for value in scene.ax.get_xlim())
    ymin, ymax = sorted(float(value) for value in scene.ax.get_ylim())
    drawn = scene._renderer.drawn["graticule-1"]
    inside = []
    for artist in drawn.artists:
        if not isinstance(artist, Text):
            continue
        x, y = artist.get_position()
        if xmin <= x <= xmax and ymin <= y <= ymax:
            inside.append(artist.get_text())
    return inside


def _segments(scene):
    """Return how many polylines the graticule layer's one collection holds.

    Args:
        scene: A map whose graticule has been drawn.

    Returns:
        The segment count of the single :class:`~matplotlib.collections.LineCollection` the layer owns.
    """
    drawn = scene._renderer.drawn["graticule-1"]
    (collection,) = [
        artist for artist in drawn.artists if isinstance(artist, LineCollection)
    ]
    return len(collection.get_segments())


@pytest.fixture
def narrowed():
    """Yield a world grid that a later ``set_domain`` has narrowed the view of, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` built in the order the defect needs: the labelled grid
        first, on an unframed axes that is labelled for the whole world, then the framing call.
    """
    scene = Map(crs=4326)
    scene.graticule(spacing=STEP).set_domain(DOMAIN)
    yield scene
    scene.close()


@pytest.fixture
def widened():
    """Yield a grid built on a narrow view and then framed on the world, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` built the other way round — the variant that was visually
        wrong rather than empty, a world map carrying three labels mid-ocean.
    """
    scene = Map(crs=4326)
    scene.set_domain(DOMAIN).graticule(spacing=STEP).set_global()
    yield scene
    scene.close()


class TestANarrowedViewIsLabelledForTheViewItNowShows:
    """``graticule()`` then ``set_domain()``: the order the tier's own docs chain in."""

    def test_every_label_the_layer_owns_is_inside_the_axes(self, narrowed):
        """The figure must not carry a degree the reader cannot see.

        Args:
            narrowed: A world grid whose view was narrowed afterwards.

        Test scenario:
            Measured 18 labels owned and 0 of them inside the axes before this — the grid rendered bare,
            with every degree off the canvas, and nothing said so.
        """
        assert _labels_inside(narrowed) == _labels(narrowed), (
            f"{len(_labels(narrowed)) - len(_labels_inside(narrowed))} of the layer's labels sit outside "
            f"the view: {sorted(set(_labels(narrowed)) - set(_labels_inside(narrowed)))}"
        )

    def test_the_degrees_are_the_ones_the_new_view_holds(self, narrowed):
        """And they are the right three, not merely three that happen to be inside.

        Args:
            narrowed: A world grid whose view was narrowed afterwards.

        Test scenario:
            :data:`EUROPE_DEGREES` is written out, so this is not the two build orders compared with each
            other: it is what a 30-degree grid holds inside ``[-25, 45] x [34, 72]``.
        """
        assert _labels(narrowed) == EUROPE_DEGREES, (
            f"the narrowed view should be labelled {EUROPE_DEGREES}; got {_labels(narrowed)}"
        )

    def test_the_grid_itself_is_not_recut_by_the_reframe(self, narrowed):
        """The lines span the world whatever the view is, so only the labels may move.

        Args:
            narrowed: A world grid whose view was narrowed afterwards.

        Test scenario:
            This is the half of the layer that is *not* positioned against the call-time view, asserted
            rather than argued: the collection still holds all 18 polylines after the reframe.
        """
        assert _segments(narrowed) == WORLD_DEGREE_COUNT, (
            f"the reframe re-cut the grid to {_segments(narrowed)} lines"
        )

    def test_the_layer_still_owns_exactly_one_collection(self, narrowed):
        """A re-label must replace the drawing, not stack a second grid under it.

        Args:
            narrowed: A world grid whose view was narrowed afterwards.

        Test scenario:
            ``_on_flat_axes`` reuses the collection it already drew; two collections would mean the
            reframe drew a whole second grid and left the first one on the axes.
        """
        collections = [
            artist
            for artist in narrowed._renderer.drawn["graticule-1"].artists
            if isinstance(artist, LineCollection)
        ]
        assert len(collections) == 1, (
            f"the reframe left {len(collections)} collections on the layer"
        )

    def test_nothing_else_is_left_on_the_axes(self, narrowed):
        """The old labels are off the axes, not merely absent from the record.

        Args:
            narrowed: A world grid whose view was narrowed afterwards.

        Test scenario:
            The replaced labels are removed artist by artist, so an axes still holding 18 ``Text`` while
            the layer owns 3 would be the record disagreeing with the figure the other way round.
        """
        assert len(narrowed.ax.texts) == len(EUROPE_DEGREES), (
            f"the axes holds {len(narrowed.ax.texts)} labels for a layer owning {len(EUROPE_DEGREES)}"
        )


class TestAWidenedViewIsLabelledForTheWorld:
    """``set_domain()`` then ``graticule()`` then ``set_global()``: the visually wrong variant."""

    def test_the_world_view_carries_every_degree_it_holds(self, widened):
        """A world map at 30 degrees has 18 labels, not the 3 its first framing allowed.

        Args:
            widened: A narrow grid that was then framed on the world.

        Test scenario:
            Measured 3 before this, floating mid-Atlantic at the latitude the old window's inset put
            them at — a figure that looked labelled and was not.
        """
        assert len(_labels(widened)) == WORLD_DEGREE_COUNT, (
            f"a world view should carry {WORLD_DEGREE_COUNT} degrees; got {_labels(widened)}"
        )

    def test_the_two_build_orders_agree(self, widened):
        """The order a caller writes the two calls in must stop mattering.

        Args:
            widened: A narrow grid that was then framed on the world.

        Test scenario:
            The other side is built independently — one call, on a map framed globally from the start —
            rather than read back off the same figure, so this is not a value compared with itself.
        """
        once = Map(crs=4326)
        once.set_global().graticule(spacing=STEP)
        expected = _labels(once)
        once.close()
        assert _labels(widened) == expected, (
            f"built in two steps: {_labels(widened)}; built in one: {expected}"
        )


class TestTheReframeIsSilentWhenItHasNothingToReport:
    """A re-label that places every degree it asks for must not warn about anything."""

    def test_no_warning_is_raised_by_the_narrowing_reframe(self):
        """The placement warning names degrees it could not place, and there are none here.

        Test scenario:
            The narrowed window *asks* for only the three lines it holds, so nothing is dropped. A
            warning here would mean the re-label is asking for degrees the view does not hold.
        """
        scene = Map(crs=4326)
        scene.graticule(spacing=STEP)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene.set_domain(DOMAIN)
        messages = [str(record.message) for record in caught]
        scene.close()
        assert messages == [], f"the reframe warned: {messages}"


class TestWhatTheReframeLeavesAlone:
    """The three cases a re-label must not touch: a bare grid, a hidden one, and a globe."""

    def test_a_bare_grid_gains_no_labels(self):
        """``labels=False`` is a deliberate choice, and a reframe is not a request to undo it.

        Test scenario:
            The same rule a *replacing* ``graticule()`` call follows: a call that is not about the labels
            does not bring them back.
        """
        scene = Map(crs=4326)
        scene.graticule(spacing=STEP, labels=False).set_domain(DOMAIN)
        texts = len(scene.ax.texts)
        scene.close()
        assert texts == 0, f"the reframe put {texts} labels on a bare grid"

    def test_a_hidden_grid_stays_hidden(self):
        """A reframe must not put a grid the caller hid back on screen.

        Test scenario:
            The re-label goes through the renderer, which applies the described ``visible`` flag after
            every draw — so this holds by the same route a replacing call's does (#327).
        """
        scene = Map(crs=4326)
        scene.graticule(spacing=STEP, visible=False).set_domain(DOMAIN)
        visible = [artist for artist in scene.ax.texts if artist.get_visible()]
        scene.close()
        assert visible == [], f"the reframe made {len(visible)} hidden labels visible"

    def test_a_globe_is_framed_without_gaining_a_label(self):
        """A globe's grid carries no degrees at all, and a framing call cannot give it any.

        Test scenario:
            Its lines are drawn by the projection frame at render time, and ``labels=True`` is refused
            there outright — so the re-label has to leave the globe path alone.
        """
        scene = Map(crs=4326, globe=True)
        scene.graticule(spacing=STEP).set_global()
        scene.render()
        texts = len(scene.ax.texts)
        scene.close()
        assert texts == 0, f"framing a globe drew {texts} degree labels"

    def test_an_unframed_grid_framed_on_the_world_is_unchanged(self):
        """The one view change that was already a no-op must stay one.

        Test scenario:
            ``window()`` answers the whole world for unframed limits, so a grid built on a bare axes is
            already labelled for the world and ``set_global()`` cannot change it. That is why the one
            existing test pairing the two calls never saw the defect.
        """
        scene = Map(crs=4326)
        scene.graticule(spacing=STEP).set_global()
        drawn = len(_labels(scene))
        scene.close()
        assert drawn == WORLD_DEGREE_COUNT, (
            f"the world grid should still carry {WORLD_DEGREE_COUNT} degrees; got {drawn}"
        )
