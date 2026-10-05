"""Pick / click callbacks on the matplotlib canvas (ST-25).

A saved PNG has no pointer over it, which is why this tier declares no tooltip and no layer switcher. A
figure on a **live** canvas is a different thing, and the two halves it needs were already there before
this row: `fig.canvas.mpl_connect` is matplotlib's own, and the renderer already knows which artists each
layer put on the axes. Measured on this branch before anything was written:

```text
m.field(...);  m._renderer.drawn -> {'r': DrawnLayer(artist=AxesImage, ...)}
hasattr(Map, "on_pick")           -> False
```

So the plumbing existed and the public gesture did not. What this file pins is the gesture: a callback
hears a **layer id**, never a matplotlib artist, and the id it hears is the topmost layer under the
pointer — never a hidden one, and never one that has been removed.
"""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.text import Text
from pyramids.dataset import Dataset, GeoReference
from shapely.geometry import Point

from digitalearth.static import Map
from digitalearth.static.renderer import Renderer
from digitalearth.static.scene import Pick

#: A 2x2 lat/lon geo-reference, so a field can be drawn without reading a file.
GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)


def _raster():
    """Return a 2x2 in-memory raster covering lon 0-2, lat 0-2.

    Returns:
        A pyramids ``Dataset``.
    """
    return Dataset.from_array(
        arr=np.array([[1.0, 2.0], [3.0, 4.0]]), geo_ref=GEO, no_data_value=-9999.0
    )


def _two_labels():
    """Return two labelled points inside the raster's extent, for the multi-artist case.

    Returns:
        A ``GeoDataFrame`` with an ``nm`` column and two point geometries.
    """
    return gpd.GeoDataFrame(
        {"nm": ["a", "b"], "geometry": [Point(0.3, 1.5), Point(0.7, 0.5)]},
        crs="EPSG:4326",
    )


def _click(scene, x, y):
    """Drive one click at data coordinates ``(x, y)`` through a scene's canvas.

    The canvas is drawn first because a hit test on a ``Text`` needs a renderer to measure it with, which
    is exactly what a GUI backend has already done by the time a user clicks.

    Args:
        scene: The scene to click on.
        x: The x coordinate, in the axes' data coordinates.
        y: The y coordinate.
    """
    scene.fig.canvas.draw()
    px, py = scene.ax.transData.transform((x, y))
    event = MouseEvent("button_press_event", scene.fig.canvas, px, py, button=1)
    scene.fig.canvas.callbacks.process("button_press_event", event)


def _click_at_pixel(scene, px, py):
    """Drive one click at *display* pixels, for a point that is not over the axes at all.

    Args:
        scene: The scene to click on.
        px: The x pixel.
        py: The y pixel.
    """
    scene.fig.canvas.draw()
    event = MouseEvent("button_press_event", scene.fig.canvas, px, py, button=1)
    scene.fig.canvas.callbacks.process("button_press_event", event)


@pytest.fixture
def picked():
    """An empty list a registered callback appends each :class:`Pick` to.

    Yields:
        The list.
    """
    return []


@pytest.fixture
def overlapping(picked):
    """A map with a field and a text label on the same spot, listening for picks.

    The label is painted over the field — measured z-orders: the ``AxesImage`` has 0 and the ``Text`` has
    3 — so the two make the overlap case without either having to be given a `zorder=`.

    Args:
        picked: The list the callback appends to.

    Yields:
        The map, carrying `f` (the field) and `t` (the label). Closed afterwards.
    """
    m = Map(crs=4326, globe=False)
    m.field(_raster(), name="f")
    m.text(0.5, 1.5, "here", name="t")
    m.on_pick(picked.append)
    yield m
    m.close()


class TestAPickReportsALayerId:
    """What a callback is handed, and that it is the id rather than the engine's object."""

    def test_a_click_on_a_layer_reports_its_id(self, picked):
        """A click inside a field names the layer the caller named.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            The whole row in one line: matplotlib hands over an ``AxesImage``, and what reaches the
            callback is `'f'` — the id `layer_ids` lists and `remove_layer` takes off.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(picked.append)
            _click(m, 0.5, 1.5)
            assert [pick.layer_id for pick in picked] == ["f"]

    def test_the_pick_is_a_pick(self, picked):
        """The callback is handed a :class:`Pick`, not a raw matplotlib event.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            The value is the public surface of this gesture; the event it was built from is reachable on
            it for what it does not carry.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(picked.append)
            _click(m, 0.5, 1.5)
            assert isinstance(picked[0], Pick)

    def test_the_pick_carries_the_data_coordinates(self, picked):
        """The click's position arrives in the axes' own data coordinates.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            A caller picking a cell wants to look its value up, which needs coordinates in the CRS the
            figure is drawn in — not the display pixels matplotlib delivered.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(picked.append)
            _click(m, 0.5, 1.5)
            assert (picked[0].x, picked[0].y) == pytest.approx((0.5, 1.5), abs=1e-6)

    def test_the_pick_carries_the_event_it_was_built_from(self, picked):
        """The matplotlib event is the escape hatch for what the value leaves out.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            Which button, a double click and the pixel position are matplotlib's to report, and a caller
            who needs them should not have to connect a second handler to get them.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(picked.append)
            _click(m, 0.5, 1.5)
            assert picked[0].event.name == "button_press_event"

    def test_on_pick_hands_the_scene_back(self, picked):
        """`on_pick` is chainable, like every other figure-level call here.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            Registering a callback is decoration, and decoration on this tier reads as one expression.
        """
        with Map(crs=4326, globe=False) as m:
            assert m.on_pick(picked.append) is m

    def test_every_registered_callback_hears_one_click(self):
        """Two callbacks both hear the same pick.

        Test scenario:
            A pick is a notification rather than a handler slot, so registering a second one must not
            displace the first — unlike the axes' single legend slot, where one key does displace another.
        """
        first, second = [], []
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(first.append).on_pick(second.append)
            _click(m, 0.5, 1.5)
            assert (len(first), len(second)) == (1, 1)

    def test_a_callback_that_is_not_callable_is_refused(self):
        """`on_pick` refuses at the registration rather than at the first click.

        Test scenario:
            The caller is at the registration; by the time a click arrives they are not, and a
            ``TypeError`` from inside a canvas handler is swallowed by the backend.
        """
        with Map(crs=4326, globe=False) as m:
            with pytest.raises(TypeError, match="needs something to call"):
                m.on_pick("not a callback")


class TestARaisingCallbackDoesNotStarveTheOthers:
    """One misbehaving handler must not take the whole gesture down (R2-L1).

    `_deliver_pick` reasons carefully about iterating over a copy so a handler that calls `off_pick` does
    not skip its neighbour, and a handler that *raises* skipped it just as effectively: measured on the
    branch before the guard, the `RuntimeError` propagated out of the canvas dispatch and the handler
    registered after it ran 0 times.
    """

    @staticmethod
    def _boom(_pick):
        """Fail the way a caller's handler fails.

        Args:
            _pick: The pick delivered, unused.

        Raises:
            RuntimeError: always, which is the point.
        """
        raise RuntimeError("handler blew up")

    @pytest.fixture
    def after_a_raising_handler(self):
        """Click one layer with a raising handler registered ahead of a recording one.

        Returns:
            The list the second handler appends to, which is empty when it was starved.
        """
        seen = []
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(self._boom).on_pick(lambda pick: seen.append("second"))
            with pytest.warns(UserWarning):
                _click(m, 0.5, 1.5)
        return seen

    def test_the_handler_after_the_raising_one_still_hears_the_pick(
        self, after_a_raising_handler
    ):
        """The second handler is called although the first raised.

        Args:
            after_a_raising_handler: What the second handler recorded.

        Test scenario:
            Read as the list the second handler filled rather than as "no exception escaped": a dispatch
            that swallowed the failure and then stopped early would pass the weaker check.
        """
        assert after_a_raising_handler == ["second"], (
            f"the second handler should still have heard the pick; got {after_a_raising_handler}"
        )

    def test_the_failure_is_reported_rather_than_swallowed(self):
        """The warning names the callback and carries the exception it raised.

        Test scenario:
            Continuing quietly would turn a broken handler into a figure that merely ignores clicks, so
            the repr of the callback and the message of its exception both have to reach the caller.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(self._boom)
            with pytest.warns(UserWarning, match="handler blew up"):
                _click(m, 0.5, 1.5)

    def test_a_raising_handler_does_not_stop_the_next_click(self):
        """The gesture survives the failure: a second click is delivered too.

        Test scenario:
            A dispatch that disconnected itself on the first failure would satisfy the two checks above
            and still leave the figure dead to the click after.
        """
        seen = []
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(self._boom).on_pick(lambda pick: seen.append("again"))
            # Two clicks, so `pytest.warns` is the wrong manager here: a block holding two
            # warning-raising calls is what `tests/base/test_refusal_blocks.py` reports (S9088).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _click(m, 0.5, 1.5)
                _click(m, 0.5, 1.5)
        assert len(seen) == 2, (
            f"both clicks should have been delivered; got {len(seen)}"
        )


class TestWhichLayerAPickReports:
    """Overlap, hiding, removal, and the layer that owns more than one artist."""

    def test_the_topmost_layer_is_the_one_reported(self, overlapping, picked):
        """Two layers under the pointer report the one painted last.

        Args:
            overlapping: A map with a field under a text label, listening for picks.
            picked: The list the callback appends to.

        Test scenario:
            A pick has to agree with what the reader sees, and what they see on top is the label — the
            measured z-orders are 0 for the field's image and 3 for the text.
        """
        _click(overlapping, 0.5, 1.5)
        assert picked[0].layer_id == "t"

    def test_every_hit_is_listed_topmost_first(self, overlapping, picked):
        """The pick also carries what lies under the layer it reports.

        Args:
            overlapping: A map with a field under a text label, listening for picks.
            picked: The list the callback appends to.

        Test scenario:
            A caller who wants the field's value under a clicked label should not have to hit-test the
            figure a second time to find out it is there.
        """
        _click(overlapping, 0.5, 1.5)
        assert picked[0].hits == ("t", "f")

    def test_the_reported_layer_is_the_first_hit(self, overlapping, picked):
        """`layer_id` and `hits[0]` are one answer, not two.

        Args:
            overlapping: A map with a field under a text label, listening for picks.
            picked: The list the callback appends to.

        Test scenario:
            Two fields that could disagree would make "which layer was picked" an ambiguous question;
            `hits[0]` is where the answer comes from.
        """
        _click(overlapping, 0.5, 1.5)
        assert picked[0].layer_id == picked[0].hits[0]

    def test_a_hidden_layer_is_not_reported(self, overlapping, picked):
        """Hiding the top layer hands the pick to the one below it.

        Args:
            overlapping: A map with a field under a text label, listening for picks.
            picked: The list the callback appends to.

        Test scenario:
            The case that cannot be left to matplotlib: measured, an artist hidden with
            `set_visible(False)` still answers `True` to `contains` for a point inside it, so a pick that
            trusted the hit test alone would report a layer the reader cannot see.
        """
        overlapping.set_visible("t", False)
        _click(overlapping, 0.5, 1.5)
        assert picked[0].layer_id == "f"

    def test_a_hidden_layer_is_not_even_listed(self, overlapping, picked):
        """A hidden layer is absent from `hits`, not merely demoted in it.

        Args:
            overlapping: A map with a field under a text label, listening for picks.
            picked: The list the callback appends to.

        Test scenario:
            `hits` is "what is under the pointer" as the reader sees it, so a layer that is not drawn is
            not in it at all.
        """
        overlapping.set_visible("t", False)
        _click(overlapping, 0.5, 1.5)
        assert picked[0].hits == ("f",)

    def test_a_removed_layer_reports_no_stale_id(self, overlapping, picked):
        """A layer taken off after the callback was registered is never named again.

        Args:
            overlapping: A map with a field under a text label, listening for picks.
            picked: The list the callback appends to.

        Test scenario:
            The registration happens once and the figure goes on changing. The hit test runs against what
            the renderer holds at the moment of the click, so a removed layer cannot be reported — a
            lookup table built at registration time would have gone on naming it.
        """
        overlapping.remove_layer("t")
        _click(overlapping, 0.5, 1.5)
        assert picked[0].layer_id == "f"

    def test_removing_every_layer_leaves_nothing_to_pick(self, overlapping, picked):
        """With both layers gone the click reports nothing at all.

        Args:
            overlapping: A map with a field under a text label, listening for picks.
            picked: The list the callback appends to.

        Test scenario:
            A pick is a pick *of a layer*; with no layer under the pointer there is no id to hand over, so
            the callback is not called rather than called with `None`.
        """
        overlapping.remove_layer("t").remove_layer("f")
        _click(overlapping, 0.5, 1.5)
        assert picked == []

    def test_a_click_on_no_layer_reports_nothing(self, picked):
        """A click inside the axes but outside every layer calls nothing.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            The field covers lon/lat 0-2; the view is framed to it, so a point well outside it is still
            inside the axes once the limits are widened — which is what makes this a *miss* rather than a
            click off the figure.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(picked.append)
            m.ax.set_xlim(-10.0, 10.0)
            m.ax.set_ylim(-10.0, 10.0)
            _click(m, -8.0, -8.0)
            assert picked == []

    def test_a_click_off_the_axes_reports_nothing(self, picked):
        """A click outside the axes is not this scene's to report.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            A figure holds more than the map — a colorbar lives on an axes of its own — so the gate is
            "is this my axes?", and the figure's bottom-left corner pixel is nobody's.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(picked.append)
            _click_at_pixel(m, 1.0, 1.0)
            assert picked == []

    def test_an_artist_maps_back_to_the_layer_that_drew_it(self):
        """The renderer answers which layer owns one artist, by identity.

        Test scenario:
            The lookup the whole gesture rests on, asked directly: a pick hands over an artist and the
            caller wants an id.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            artist = m._renderer.drawn["f"].artist
            assert m._renderer.layer_of(artist) == "f"

    def test_an_artist_no_layer_owns_maps_to_nothing(self):
        """An artist the figure never drew answers `None`.

        Test scenario:
            Drawing straight onto `ax` is the documented escape hatch and is outside the description by
            design, so a click on one of those artists names no layer rather than guessing one.
        """
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            assert m._renderer.layer_of(Text(0.0, 0.0, "mine")) is None

    def test_the_multi_artist_case_really_owns_more_than_one_artist(self):
        """A `labels` layer puts one artist per feature on the axes.

        Test scenario:
            The premise of the test below, pinned so it cannot pass vacuously. Measured: two labelled
            points draw two ``Annotation`` artists for one layer, which is the shape `DrawnLayer.artists`
            is a tuple for — a limb-split coastline on a globe is the extreme of it, at 69 ``Line2D``.
        """
        with Map(crs=4326, globe=False) as m:
            m.labels(_two_labels(), column="nm", name="lb")
            assert len(m._renderer.drawn["lb"].artists) == 2

    def test_every_artist_of_one_layer_maps_to_that_layer(self):
        """A layer owning several artists maps all of them back to its one id.

        Test scenario:
            The lookup cannot stop at the layer's first artist: a click lands on whichever one is under
            the pointer, and they are one layer to the caller who named it.
        """
        with Map(crs=4326, globe=False) as m:
            m.labels(_two_labels(), column="nm", name="lb")
            m.fig.canvas.draw()
            artists = m._renderer.drawn["lb"].artists
            owners = {m._renderer.layer_of(artist) for artist in artists}
            assert owners == {"lb"}

    def test_a_click_on_the_second_of_a_layers_artists_names_that_layer(self, picked):
        """Clicking the second label reports the label layer, not a second id.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            The multi-artist case driven through the real gesture rather than through the lookup: the
            point clicked is the second feature's, and the id handed over is the layer's.
        """
        with Map(crs=4326, globe=False) as m:
            m.labels(_two_labels(), column="nm", name="lb")
            m.on_pick(picked.append)
            _click(m, 0.7, 0.5)
            assert [pick.layer_id for pick in picked] == ["lb"]


class TestTakingACallbackOff:
    """`off_pick`, and what a closed figure delivers."""

    def test_off_pick_stops_one_callback(self):
        """A callback taken off hears nothing more, and the other still does.

        Test scenario:
            Taking one listener off must not silence the rest, which is what makes the handler list a
            list rather than a slot.
        """
        kept, dropped = [], []
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(kept.append).on_pick(dropped.append).off_pick(dropped.append)
            _click(m, 0.5, 1.5)
            assert (len(kept), len(dropped)) == (1, 0)

    def test_off_pick_with_no_argument_stops_every_callback(self):
        """`off_pick()` is the teardown: nothing is delivered afterwards.

        Test scenario:
            A caller closing a session down should not have to remember which callbacks they registered.
        """
        first, second = [], []
        with Map(crs=4326, globe=False) as m:
            m.field(_raster(), name="f")
            m.on_pick(first.append).on_pick(second.append).off_pick()
            _click(m, 0.5, 1.5)
            assert first + second == []

    def test_off_pick_hands_the_scene_back(self, picked):
        """`off_pick` is chainable too.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            The pair reads as one expression, as `set_visible` and `remove_layer` do.
        """
        with Map(crs=4326, globe=False) as m:
            assert m.on_pick(picked.append).off_pick(picked.append) is m

    def test_off_pick_refuses_a_callback_it_never_had(self, picked):
        """Taking off something that was never registered is refused by name.

        Args:
            picked: The list the callback appends to.

        Test scenario:
            A silent no-op here would look exactly like a callback that refused to go away — the reason
            `remove_layer` refuses an unknown id rather than ignoring it.
        """
        with Map(crs=4326, globe=False) as m:
            m.on_pick(picked.append)
            with pytest.raises(ValueError, match="is not registered on this figure"):
                m.off_pick(print)

    def test_a_callback_may_take_itself_off_while_being_called(self):
        """A "pick once" callback can unregister itself from inside the call.

        Test scenario:
            The handler list is walked over a copy, so a callback that mutates it mid-delivery does not
            skip its neighbour or raise from the loop. The second click proves it really came off.
        """
        seen = []

        with Map(crs=4326, globe=False) as m:

            def once(pick):
                """Record the pick and stop listening.

                Args:
                    pick: The pick delivered.
                """
                seen.append(pick.layer_id)
                m.off_pick(once)

            m.field(_raster(), name="f")
            m.on_pick(once)
            _click(m, 0.5, 1.5)
            _click(m, 0.5, 1.5)
            assert seen == ["f"]

    def test_a_closed_scene_delivers_nothing(self, picked):
        """`close` takes the handler off the canvas with the figure.

        Test scenario:
            The canvas outlives `plt.close` as a Python object, so a handler left connected would go on
            answering for a scene whose data has just been let go.
        """
        m = Map(crs=4326, globe=False)
        m.field(_raster(), name="f")
        m.on_pick(picked.append)
        m.fig.canvas.draw()
        px, py = m.ax.transData.transform((0.5, 1.5))
        click = MouseEvent("button_press_event", m.fig.canvas, px, py, button=1)
        m.close()
        m.fig.canvas.callbacks.process("button_press_event", click)
        assert picked == []


class _Unhittable:
    """A visible artist whose hit test raises, as one drawn without a renderer does.

    ``Artist.contains`` needs a renderer to measure some artists against — a ``Text`` is the one in this
    file — and raises ``RuntimeError`` when there is none. A plugin's artist can raise for its own
    reasons. Either way the pick has to answer "not this one" rather than take the whole click down.
    """

    def __init__(self, error):
        self._error = error
        #: How many times the hit test was asked.
        self.asked = 0

    def get_visible(self):
        """Report the artist as visible, so the hit test is reached at all.

        Returns:
            ``True``, always.
        """
        return True

    def contains(self, event):
        """Refuse to answer whether the event is over this artist.

        Args:
            event: The click, ignored.

        Raises:
            Exception: The error this stub was built with.
        """
        self.asked += 1
        raise self._error


class TestAnArtistThatCannotBeHitTested:
    """A hit test that raises is not a pick, and not a crash either."""

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("no renderer to measure against"),
            AttributeError("contains is not implemented"),
            TypeError("contains got an event it cannot read"),
            ValueError("the artist has no path to test"),
        ],
    )
    def test_an_artist_whose_hit_test_raises_is_not_picked(self, error):
        """Each refusal a hit test can raise is read as "not under the pointer".

        Args:
            error: The exception the artist's ``contains`` raises.

        Test scenario:
            The pick walks every drawn artist, so one that cannot answer must not end the walk — a
            callback would then never hear about the layer *under* it. The four types are the ones the
            walk catches; anything else is a real fault and is left to propagate.
        """
        assert Renderer._touches(_Unhittable(error), object()) is False, (
            f"an artist raising {type(error).__name__} must not count as picked"
        )

    def test_the_hit_test_was_really_asked(self):
        """The answer above is a caught refusal, not a short circuit before the call.

        Test scenario:
            A reader that returned ``False`` without asking would pass every case above while quietly
            picking nothing at all, on any artist.
        """
        artist = _Unhittable(RuntimeError("no renderer to measure against"))
        Renderer._touches(artist, object())
        assert artist.asked == 1, (
            f"the hit test should have been asked once; it was asked {artist.asked} time(s)"
        )

    def test_an_invisible_artist_is_not_even_asked(self):
        """A hidden artist is answered from its flag, without a hit test.

        Test scenario:
            The other order this could have been written in. Asking first would make a hidden layer's
            pick depend on whether its artist happens to be hit-testable, which is not the rule — the
            flag decides.
        """
        artist = _Unhittable(RuntimeError("no renderer to measure against"))
        artist.get_visible = lambda: False
        Renderer._touches(artist, object())
        assert artist.asked == 0, (
            f"a hidden artist should not be hit-tested; it was asked {artist.asked} time(s)"
        )
