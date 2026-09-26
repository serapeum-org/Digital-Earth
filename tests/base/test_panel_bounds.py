"""``PanelSpec.bounds_of`` — the region a panel's own layers cover, in the CRS the panel draws in.

Auto-framing is the same question on every tier: *what do the layers I show cover, and in which CRS?* The
answer is not the figure's — a two-panel figure frames each panel on its own layers — and it is not the
tier's either, because ``Bounds.union`` refuses two rectangles in different CRSs and only the panel knows
which one they have to meet in.

So the composition lives here, on the value that holds both halves, and each tier supplies only what it can
measure: one ``Bounds`` per layer, from the artists or elements it actually drew. Nothing here is new
geometry — :meth:`~digitalearth.base.spec.bounds.Bounds.union` and
:meth:`~digitalearth.base.spec.bounds.Bounds.padded` already existed; this is what orders them.
"""

import pytest

from digitalearth.base.spec import Bounds, Camera, PanelSpec, Viewport

#: Two rectangles in Web Mercator that overlap in x and are disjoint in y, so a union that dropped either
#: one, or that took a corner from the wrong rectangle, cannot produce the expected answer by accident.
LEFT = Bounds(0.0, 0.0, 10.0, 10.0, crs=3857)
RIGHT = Bounds(5.0, 20.0, 30.0, 40.0, crs=3857)


class TestThePanelUnionsItsOwnLayers:
    """A panel frames on the layers it shows, and on no others."""

    def test_two_layers_frame_on_the_rectangle_that_holds_both(self):
        """The union, which is what fitting a figure to its data means."""
        panel = PanelSpec("main", Viewport(3857), layers=("a", "b"))
        assert panel.bounds_of({"a": LEFT, "b": RIGHT}).as_bbox() == [
            0.0,
            0.0,
            30.0,
            40.0,
        ]

    def test_a_layer_the_panel_does_not_show_is_not_framed_on(self):
        """The whole reason this is the panel's method rather than the figure's.

        Test scenario:
            A linked-view figure holds one tree and two panels, each naming a subset of it. A panel that
            framed on the tree would show a region containing data it does not draw — which is the bug a
            figure-level union would have, and the one a panel-level union cannot.
        """
        panel = PanelSpec("left", Viewport(3857), layers=("a",))
        assert panel.bounds_of({"a": LEFT, "b": RIGHT}).as_bbox() == LEFT.as_bbox()

    def test_a_layer_with_no_measured_extent_is_skipped(self):
        """A tier that could not measure one layer still frames on the rest.

        Test scenario:
            Not every layer has an extent to give: a text label carries a point, a graticule is computed
            from the CRS, and a tier may simply not have drawn one yet. The mapping is therefore read as
            "what is known", not as "one entry per layer" — a missing key and an explicit ``None`` both
            mean the same thing, and neither is an error.
        """
        panel = PanelSpec("main", Viewport(3857), layers=("a", "b", "c"))
        assert panel.bounds_of({"a": LEFT, "b": None}).as_bbox() == LEFT.as_bbox()

    def test_a_panel_with_nothing_measured_has_no_bounds(self):
        """``None``, so a caller can tell "nothing to frame on" from "framed on nothing"."""
        panel = PanelSpec("main", Viewport(3857), layers=("a",))
        assert panel.bounds_of({}) is None

    def test_the_order_the_extents_arrive_in_does_not_change_the_answer(self):
        """A union is commutative, and a dict's order must not leak into the frame.

        Test scenario:
            Built the two ways round rather than compared to itself, so this is a claim about the method
            and not the tautology ``X == X``.
        """
        panel = PanelSpec("main", Viewport(3857), layers=("a", "b"))
        forwards = panel.bounds_of({"a": LEFT, "b": RIGHT})
        backwards = panel.bounds_of({"b": RIGHT, "a": LEFT})
        assert forwards.as_bbox() == backwards.as_bbox()


class TestTheViewsCrsIsWhatTheyMeetIn:
    """Reprojection is the panel's job, because the view's CRS is the only one that is right."""

    def test_an_extent_in_another_crs_is_reprojected_into_the_view(self):
        """A tier may measure a layer in the data's CRS; the frame is in the display CRS.

        Test scenario:
            A degree rectangle unioned into a metre one without reprojection would place the frame within
            a few tens of metres of the origin — off the coast of Africa — rather than around the data.
            ``Bounds.union`` refuses that outright, so the reprojection has to happen before it.
        """
        panel = PanelSpec("main", Viewport(3857), layers=("here",))
        framed = panel.bounds_of({"here": Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)})
        assert (framed.crs, round(framed.xmax)) == (3857, 111319), framed.as_bbox()

    def test_a_view_with_no_crs_unions_the_extents_as_they_are(self):
        """A chart panel places nothing, so there is nothing to reproject into.

        Test scenario:
            A bare :class:`Viewport` has a CRS, but a :class:`Camera` need not — and a panel whose view
            names none cannot convert. It unions what it is given instead of inventing a target, which is
            also what keeps a CRS-less pair of rectangles (a chart's) working.
        """
        panel = PanelSpec("3d", Camera((0.0, -10.0, 5.0)), layers=("a", "b"))
        plain = (
            Bounds(0.0, 0.0, 1.0, 1.0, crs=None),
            Bounds(2.0, 2.0, 3.0, 3.0, crs=None),
        )
        assert panel.bounds_of({"a": plain[0], "b": plain[1]}).as_bbox() == [
            0.0,
            0.0,
            3.0,
            3.0,
        ]

    def test_extents_that_cannot_meet_are_refused_by_name(self):
        """Two CRSs and no view CRS to convert into is a caller error, not a silent reinterpretation."""
        panel = PanelSpec("3d", Camera((0.0, -10.0, 5.0)), layers=("a", "b"))
        extents = {
            "a": Bounds(0.0, 0.0, 1.0, 1.0, crs=4326),
            "b": Bounds(0.0, 0.0, 1.0, 1.0, crs=3857),
        }
        with pytest.raises(ValueError, match="one CRS"):
            panel.bounds_of(extents)


class TestPadding:
    """Breathing room, as a fraction of the span the layers cover."""

    def test_a_fraction_grows_the_frame_on_every_side(self):
        """A tenth of the span, which is what a map fitted to its data usually wants."""
        panel = PanelSpec("main", Viewport(3857), layers=("a",))
        padded = panel.bounds_of({"a": LEFT}, padding=0.1)
        assert padded.as_bbox() == [-1.0, -1.0, 11.0, 11.0]

    def test_no_padding_is_the_default(self):
        """The frame is the data's own rectangle unless a caller asks for room around it."""
        panel = PanelSpec("main", Viewport(3857), layers=("a",))
        assert panel.bounds_of({"a": LEFT}).as_bbox() == LEFT.as_bbox()

    def test_padding_is_applied_once_to_the_union_and_not_per_layer(self):
        """Padding each layer and then unioning would grow the frame by the widest layer's span.

        Test scenario:
            The two rectangles span 30 by 40, so a tenth of the union is 3 by 4. Padding ``LEFT`` on its
            own would add 1 by 1, and a per-layer pass would therefore leave the union's lower corner at
            ``-1``. Reading the corner is what tells the two implementations apart.
        """
        panel = PanelSpec("main", Viewport(3857), layers=("a", "b"))
        padded = panel.bounds_of({"a": LEFT, "b": RIGHT}, padding=0.1)
        assert padded.as_bbox() == [-3.0, -4.0, 33.0, 44.0]

    def test_a_padding_that_would_invert_the_frame_is_refused(self):
        """``Bounds.padded`` owns the refusal, and its message has to survive the delegation."""
        panel = PanelSpec("main", Viewport(3857), layers=("a",))
        with pytest.raises(ValueError, match="would invert"):
            panel.bounds_of({"a": LEFT}, padding=-0.9)


class TestWhatItRefuses:
    """A mapping is what this takes; anything else is a caller passing the wrong thing."""

    def test_a_non_mapping_is_named(self):
        """A list of rectangles has no layer ids in it, so it can only be a mistake."""
        panel = PanelSpec("main", Viewport(3857), layers=("a",))
        with pytest.raises(TypeError, match="bounds_of"):
            panel.bounds_of([LEFT])

    def test_an_extent_that_is_not_a_rectangle_is_named_with_its_layer(self):
        """The layer id is in the message, because a figure has many and only one is wrong."""
        panel = PanelSpec("main", Viewport(3857), layers=("a",))
        with pytest.raises(TypeError, match="'a'"):
            panel.bounds_of({"a": (0.0, 0.0, 1.0, 1.0)})
