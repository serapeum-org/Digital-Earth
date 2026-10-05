"""``graticule()`` has to put lines on the axes on every frame, and label them where it can (#221).

Measured on this branch's parent (7e26ca42), on ``Map(crs=4326)``, counting the artists the graticule layer
owns and the ``Text`` artists on the axes:

```
globe=False -> artists=0   labels=0   layer=['g']
globe=True  -> artists=18  labels=0   layer=['g']
```

So a **flat** map — which is the default — registered a graticule layer and drew nothing at all, because
the only consumer of the computed lines is
:meth:`~digitalearth.static.maps.projection.ProjectionMixin._apply_frame`, whose first statement returns
early off a globe. A globe did draw its meridians, through that frame. Neither frame ever drew a degree
label: ``grep -rn "°" src/digitalearth/static`` found none.

Every claim here is read off the **artists and the axes' texts**, never off the description or off
``is_visible``: the defect was exactly that the figure described a layer matplotlib was not holding, so a
test that asked the figure would have passed throughout.
"""

import warnings

import pytest
from matplotlib.collections import LineCollection
from matplotlib.text import Text

from digitalearth.static import Map, projections

#: The two steps every test here draws at, in degrees. 30 is also
#: :data:`~digitalearth.static.maps.projection.DEFAULT_GRATICULE_STEP`, so the counts below are the ones a
#: caller who names no spacing gets.
STEP = 30.0

#: A window, as ``(west, south, east, north)`` in EPSG:4326, that holds the meridians at -30/0/30 and the
#: parallels at 0/30/60 and **no others** — its edges sit 5 degrees clear of the next line either way, so
#: which lines fall inside it is not a question about float equality at the boundary.
WINDOW = [-35.0, -5.0, 35.0, 65.0]


def _grid_lines(scene):
    """Return the line collections the graticule layer owns on this scene.

    Args:
        scene: A map whose graticule has been drawn.

    Returns:
        Every :class:`~matplotlib.collections.LineCollection` among the artists the layer holds — the
        evidence that meridians reached the axes, rather than the description that says they did.
    """
    layer = scene.figure_spec.layers.layers[-1]
    drawn = scene._renderer.drawn[layer.id]
    return [artist for artist in drawn.artists if isinstance(artist, LineCollection)]


def _label_texts(scene):
    """Return the degree labels the graticule layer owns, as the strings they draw.

    Args:
        scene: A map whose graticule has been drawn.

    Returns:
        The text of every :class:`~matplotlib.text.Text` artist the layer holds, in the order it was added.
    """
    layer = scene.figure_spec.layers.layers[-1]
    drawn = scene._renderer.drawn[layer.id]
    return [artist.get_text() for artist in drawn.artists if isinstance(artist, Text)]


class TestAFlatMapDrawsTheGrid:
    """The default frame has to draw the meridians and parallels it was asked for."""

    def test_the_layer_owns_one_line_collection(self):
        """A flat graticule is one collection, not nothing and not one artist per line.

        Test scenario:
            ``Map(crs=4326)`` is the default, non-globe frame — the one that drew nothing. The lines go on
            as a single ``LineCollection`` because the grid is one layer, which is also what #221 asked for.
        """
        canvas = Map(crs=4326)
        canvas.graticule(spacing=STEP)
        collections = _grid_lines(canvas)
        canvas.close()
        assert len(collections) == 1, collections

    def test_it_holds_one_segment_per_meridian_and_parallel(self):
        """Every line asked for is on the axes, counted independently of what the builder produced.

        Test scenario:
            EPSG:4326 places every lon/lat pair finitely, so no line is split at a limb: a 30-degree grid
            is the 13 meridians from -180 to 180 and the 5 parallels from -60 to 60. The expected count is
            built from those two ranges rather than read back off the map, so this is not a comparison of
            the builder with itself.
        """
        expected = len(range(-180, 181, 30)) + len(range(-60, 61, 30))
        canvas = Map(crs=4326)
        canvas.graticule(spacing=STEP)
        (collection,) = _grid_lines(canvas)
        drawn = len(collection.get_segments())
        canvas.close()
        assert drawn == expected, (drawn, expected)

    def test_a_second_call_leaves_one_grid_on_the_axes(self):
        """A replacing call replaces the lines as well as the description.

        Test scenario:
            ``graticule()`` describes one layer however often it is called, and ``Renderer.draw_layer``
            does not take a previous drawing off — so the second call had to remove what the first put on,
            or the axes would hold two grids for the one layer the figure names.
        """
        canvas = Map(crs=4326)
        canvas.graticule(spacing=STEP)
        canvas.graticule(spacing=15.0)
        on_axes = [
            artist
            for artist in canvas.ax.collections
            if isinstance(artist, LineCollection)
        ]
        canvas.close()
        assert len(on_axes) == 1, on_axes

    def test_the_view_the_caller_framed_is_not_moved_by_the_grid(self):
        """A world-wide grid must not pull a regional view back out to the world.

        Test scenario:
            The graticule spans the globe whatever the map is looking at, so adding it to a framed axes is
            the case every other decoration guards with ``_preserve_view``.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds(WINDOW)
        canvas.graticule(spacing=STEP)
        held = [float(value) for value in canvas.ax.get_xlim()]
        canvas.close()
        assert held == [WINDOW[0], WINDOW[2]], held


class TestDegreeLabels:
    """A map with no coordinate reference is unpublishable; the lines have to carry their degrees."""

    def test_every_line_in_view_is_labelled_with_its_degree_and_hemisphere(self):
        """The label set is the lines the window holds, spelled as the other tiers spell them.

        Test scenario:
            :data:`WINDOW` holds the meridians at -30/0/30 and the parallels at 0/30/60. The expected
            strings are written out here rather than derived from the implementation, and they are the
            format the web tier already ships (``f"{abs(value):g}°{suffix}"``), so one spacing reads the
            same way on both tiers. The zero line belongs to no hemisphere, and the prime meridian and the
            equator share the one spelling.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds(WINDOW)
        canvas.graticule(spacing=STEP)
        labelled = set(_label_texts(canvas))
        canvas.close()
        assert labelled == {"30°W", "0°", "30°E", "30°N", "60°N"}, labelled

    def test_both_the_meridian_and_the_parallel_at_zero_are_drawn(self):
        """The two zero lines share a spelling, so the count is what shows both were labelled.

        Test scenario:
            Three meridians and three parallels fall inside :data:`WINDOW`, which is six labels — one of
            which reads ``0°`` twice over. A set cannot tell those two apart, so the count is asserted
            beside it.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds(WINDOW)
        canvas.graticule(spacing=STEP)
        drawn = _label_texts(canvas)
        canvas.close()
        assert drawn.count("0°") == 2, drawn

    def test_labels_sit_inside_the_view_they_were_placed_for(self):
        """A label outside the axes is no label at all.

        Test scenario:
            Each degree is placed on its own line, just inside the lower/left edge of the window the call
            saw — so every label's position has to lie within the framed rectangle.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds(WINDOW)
        canvas.graticule(spacing=STEP)
        layer = canvas.figure_spec.layers.layers[-1]
        positions = [
            artist.get_position()
            for artist in canvas._renderer.drawn[layer.id].artists
            if isinstance(artist, Text)
        ]
        canvas.close()
        outside = [
            (x, y)
            for x, y in positions
            if not (WINDOW[0] <= x <= WINDOW[2] and WINDOW[1] <= y <= WINDOW[3])
        ]
        assert len(positions) == 6, positions
        assert outside == [], outside

    def test_labels_false_draws_the_lines_and_no_text(self):
        """Turning the degrees off is not turning the grid off.

        Test scenario:
            ``labels=False`` is the web tier's spelling for the same choice, so the keyword means one thing
            across the tiers.
        """
        canvas = Map(crs=4326)
        canvas.graticule(spacing=STEP, labels=False)
        texts = _label_texts(canvas)
        collections = _grid_lines(canvas)
        canvas.close()
        assert texts == [], texts
        assert len(collections) == 1, collections

    def test_the_figure_records_whether_it_was_labelled(self):
        """A figure read back elsewhere has to draw the same grid, labels included.

        Test scenario:
            The labels are drawn from the description, so the choice travels in it — otherwise a stored
            figure would come back labelled when it was not.
        """
        canvas = Map(crs=4326)
        canvas.graticule(spacing=STEP, labels=False)
        recorded = canvas.figure_spec.layers.layers[-1].symbology.props["labels"]
        canvas.close()
        assert recorded is False, recorded


class TestWhatCannotBePlacedIsRefusedByName:
    """A degree label on a globe's limb is cleopatra's to place; asking for one here is refused, not dropped."""

    def test_labels_on_a_globe_are_refused_naming_the_frame(self):
        """The refusal says it is the globe frame that cannot carry them.

        Test scenario:
            ``apply_projection_frame`` turns the axes off (``set_axis_off``) and places no text, so there
            is no edge to hang a degree on — the upstream ask recorded in #221. Refusing names it; drawing
            the lines and silently dropping the labels is the failure this row exists to end.
        """
        canvas = Map(crs=projections.orthographic(0, 0), globe=True)
        with pytest.raises(ValueError, match="globe"):
            canvas.graticule(spacing=STEP, labels=True)
        canvas.close()

    def test_a_globe_that_asks_for_no_labels_still_draws_its_meridians(self):
        """The control: the refusal must not cost the globe its grid.

        Test scenario:
            The globe's lines are drawn by the projection frame rather than by the drawer, so they only
            exist after ``render()``. That path is untouched by this row and is checked here so a
            regression in it cannot hide behind the flat-map claims.
        """
        canvas = Map(crs=projections.orthographic(0, 0), globe=True)
        canvas.graticule(spacing=STEP)
        canvas.render()
        layer = canvas.figure_spec.layers.layers[-1]
        owned = len(canvas._renderer.drawn[layer.id].artists)
        canvas.close()
        assert owned > 0, "the projection frame drew the globe no graticule"


class TestHidingReachesTheLabels:
    """A grid built hidden has to be drawn hidden on a flat map too (#333's claim, one frame over)."""

    def test_a_hidden_flat_graticule_puts_nothing_visible_on_the_axes(self):
        """Lines and degrees alike answer to the layer's flag.

        Test scenario:
            A flat graticule's artists are made by its own drawer, so ``Renderer.draw_layer`` applies the
            flag to them the moment it returns — the funnel that a globe's artists arrive too late for.
        """
        canvas = Map(crs=4326)
        canvas.graticule(spacing=STEP, visible=False)
        layer = canvas.figure_spec.layers.layers[-1]
        drawn = canvas._renderer.drawn[layer.id]
        flags = {artist.get_visible() for artist in drawn.artists}
        canvas.close()
        assert flags == {False}, flags


#: A square window in EPSG:3413 (NSIDC Arctic Polar Stereographic North), 6000 km on a side, whose lon/lat
#: envelope spans every longitude while the square itself holds only four of the meridians' anchors — the
#: case a label has to be dropped in.
ARCTIC_SQUARE = [-3.0e6, -3.0e6, 3.0e6, 3.0e6]

#: The degree sign the labels are written with, spelled as an escape so this file stays ASCII-safe.
DEGREE = "°"


class TestTheDegreeAtTheDateline:
    """A line whose degree has no hemisphere: ``180`` belongs to neither side."""

    def test_the_dateline_is_labelled_without_a_hemisphere(self):
        """``180`` is the same meridian east and west, so its label carries no letter.

        Test scenario:
            The module's window never reaches it; a map framed on the whole world does. ``180E`` and
            ``180W`` name one line, so a hemisphere letter there would assert a side that does not exist
            — and the equator, already covered, is the other value this rule has to hold for.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds([-180.0, -90.0, 180.0, 90.0])
        canvas.graticule(spacing=STEP)
        drawn = _label_texts(canvas)
        canvas.close()
        assert f"180{DEGREE}" in drawn, (
            f"the dateline should be labelled without a hemisphere; got {drawn}"
        )

    def test_the_eastern_and_western_dateline_read_the_same(self):
        """Both ends of a world-framed view are labelled, and identically.

        Test scenario:
            ``lines_within`` returns -180 and 180 as two lines, one per edge of the view. They are the
            same meridian, so the two labels have to read the same — if the hemisphere rule used the sign
            rather than the magnitude they would read ``180W`` and ``180E``.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds([-180.0, -90.0, 180.0, 90.0])
        canvas.graticule(spacing=STEP)
        drawn = _label_texts(canvas)
        canvas.close()
        assert drawn.count(f"180{DEGREE}") == 2, (
            f"both edges of a world view should carry the same dateline label; got {drawn}"
        )


class TestAnUnframedMapIsLabelledForTheWholeWorld:
    """``graticule()`` before anything framed the view still labels its lines."""

    def test_the_lines_of_an_unframed_map_are_labelled(self):
        """A map nobody framed is about to be autoscaled to the grid, so no label is dropped.

        Test scenario:
            An unframed axes still holds matplotlib's unit square, and a label is placed in lon/lat and
            then projected — so judging "is it inside the view" against that square would drop every
            label on the most ordinary call there is, ``Map(crs=4326).graticule()``.
        """
        canvas = Map(crs=4326)
        canvas.graticule(spacing=60.0)
        drawn = _label_texts(canvas)
        canvas.close()
        assert f"60{DEGREE}E" in drawn, (
            f"an unframed map's lines should still be labelled; got {drawn}"
        )

    def test_an_unframed_map_still_draws_its_lines(self):
        """And the grid itself is there, so the labels are not standing alone.

        Test scenario:
            The claim above is about the labels; this is the other half, so a regression that dropped the
            lines and kept the text could not pass both.
        """
        canvas = Map(crs=4326)
        canvas.graticule(spacing=60.0)
        collections = _grid_lines(canvas)
        canvas.close()
        assert len(collections) == 1, (
            f"an unframed map should own one line collection; got {len(collections)}"
        )


class TestALabelWithNoPlaceInTheViewIsDroppedAndSaidSo:
    """What happens to a degree whose anchor the display CRS puts outside the frame."""

    def test_a_label_outside_the_framed_view_is_dropped_with_a_warning(self):
        """The dropped degrees are named, so a thinned grid is explained rather than mysterious.

        Test scenario:
            A polar-stereographic square's lon/lat envelope runs the whole way round the pole, so
            ``lines_within`` asks for every meridian — but most of their anchors project outside the
            square itself. Drawing them would put text beyond the axes; dropping them silently would
            leave a grid whose lines outnumber its labels for no stated reason.
        """
        canvas = Map(crs=3413)
        canvas.set_bounds(ARCTIC_SQUARE)
        with pytest.warns(UserWarning, match="could not place"):
            canvas.graticule(spacing=STEP)
        canvas.close()

    def test_the_labels_that_could_be_placed_are_still_drawn(self):
        """Dropping some is not dropping all — the grid keeps the degrees it can place.

        Test scenario:
            The refusal is per label, so the meridians whose anchors do land inside the square are drawn.
            A refusal that returned nothing would leave a polar map unlabelled altogether.
        """
        canvas = Map(crs=3413)
        canvas.set_bounds(ARCTIC_SQUARE)
        with pytest.warns(UserWarning):
            canvas.graticule(spacing=STEP)
        drawn = _label_texts(canvas)
        canvas.close()
        assert f"90{DEGREE}W" in drawn, (
            f"a placeable degree should still be drawn; got {drawn}"
        )

    def test_every_drawn_label_is_inside_the_axes(self):
        """What survives the drop is inside the view, which is the point of dropping anything.

        Test scenario:
            Read off the artists' own positions against the axes limits rather than off the count, so a
            change that kept the warning and drew the labels anyway fails here.
        """
        canvas = Map(crs=3413)
        canvas.set_bounds(ARCTIC_SQUARE)
        with pytest.warns(UserWarning):
            canvas.graticule(spacing=STEP)
        xmin, xmax = canvas.ax.get_xlim()
        ymin, ymax = canvas.ax.get_ylim()
        layer = canvas.figure_spec.layers.layers[-1]
        placed = [
            artist.get_position()
            for artist in canvas._renderer.drawn[layer.id].artists
            if isinstance(artist, Text)
        ]
        canvas.close()
        outside = [
            (x, y) for x, y in placed if not (xmin <= x <= xmax and ymin <= y <= ymax)
        ]
        assert outside == [], (
            f"every drawn label should be inside the view; {outside} are not"
        )


class TestAViewWithNoLineInItIsLabelledWithNothing:
    """Two different "no labels" answers, told apart: nothing asked for, and nothing placeable."""

    def test_a_window_holding_no_line_draws_no_label(self):
        """A one-degree window at a 30-degree spacing holds no meridian, so there is nothing to label.

        Test scenario:
            ``lines_within`` returns an empty list, and the reader stops there rather than reprojecting
            an empty anchor list — which is what a bare ``reproject_coordinates([], [])`` would be asked
            to do.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds([1.0, 1.0, 2.0, 2.0])
        canvas.graticule(spacing=STEP)
        drawn = _label_texts(canvas)
        canvas.close()
        assert drawn == [], (
            f"a window holding no line should draw no label; got {drawn}"
        )

    def test_a_window_holding_no_line_is_not_warned_about(self):
        """And it is not a warning: asking for a spacing coarser than the view is not an error.

        Test scenario:
            The warnings this reader raises are both about a label it *wanted* to place. A view that
            asked for none is an ordinary zoom, and warning there would fire on every close-up.
        """
        canvas = Map(crs=4326)
        canvas.set_bounds([1.0, 1.0, 2.0, 2.0])
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            canvas.graticule(spacing=STEP)
        canvas.close()
        raised = [
            str(record.message)
            for record in seen
            if "graticule()" in str(record.message)
        ]
        assert raised == [], (
            f"a view with no line in it should warn about nothing; got {raised}"
        )

    def test_a_view_with_no_longitude_at_all_warns_that_there_is_no_window(self):
        """An orthographic view framed entirely off the limb has no lon/lat to place a degree in.

        Test scenario:
            Every sample of the view reprojects to a non-finite lon/lat, so there is no window — a
            different failure from "no line in the window" above, and it says so: an unlabelled grid on a
            frame the caller believes is over the earth is worth one warning.
        """
        canvas = Map(crs=projections.orthographic(0.0, 0.0))
        canvas.set_bounds([2.0e7, 2.0e7, 3.0e7, 3.0e7])
        with pytest.warns(UserWarning, match="no window to place them in"):
            canvas.graticule(spacing=STEP)
        canvas.close()

    def test_that_view_still_draws_its_lines(self):
        """The grid is still drawn — only the labels were impossible.

        Test scenario:
            The labels are computed before the first artist reaches the axes precisely so that this
            warning does not cost the lines; a reader that raised instead of warning would lose them.
        """
        canvas = Map(crs=projections.orthographic(0.0, 0.0))
        canvas.set_bounds([2.0e7, 2.0e7, 3.0e7, 3.0e7])
        with pytest.warns(UserWarning):
            canvas.graticule(spacing=STEP)
        collections = _grid_lines(canvas)
        canvas.close()
        assert len(collections) == 1, (
            f"the unlabelled view should still own one line collection; got {len(collections)}"
        )
