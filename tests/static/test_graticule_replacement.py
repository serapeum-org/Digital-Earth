"""A replacing ``graticule()`` call keeps every option it does not name (round 1, M7).

``graticule()`` describes one layer however often it is called, so the second call is a *restyle* of the
first. ``visible`` was already held that way and says so in the method's own docstring — *"a call that does
not name it is asking for a different spacing, not for a hidden grid to come back"* — while ``labels`` and
the two steps fell back to their construction defaults instead. Measured on the parent commit
(``57ccd577``), on ``Map(crs=4326)`` framed on a window:

```
labels   first (0 texts, props False)  after graticule(spacing=60) -> (3 texts, props True)
lon/lat  first (10.0, 45.0)            after graticule(lon_step=20) -> (20.0, 30.0)
visible  first False                   after graticule(spacing=60) -> False
name     first ['grid']                after graticule(name='other') -> ['grid'], warned
```

So two of the five were lost and two were held. The claims below are read off the **axes' texts** and the
layer's stored props together: the defect put the labels back on the figure *and* in the description, so
either alone would have caught it, and both together say the two agree.
"""

import pytest

from digitalearth.static import Map

#: The window every test here frames on, as ``(west, south, east, north)`` in EPSG:4326. Wide enough to hold
#: several lines of a 30-degree grid, so a labelled grid has labels to count.
WINDOW = [-35.0, -5.0, 35.0, 65.0]


@pytest.fixture
def framed():
    """Yield a flat map framed on :data:`WINDOW`, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` ready for its first ``graticule()`` call.
    """
    scene = Map(crs=4326)
    scene.set_bounds(WINDOW)
    yield scene
    scene.close()


def _props(scene):
    """Return the props the described graticule layer carries.

    Args:
        scene: A map whose graticule has been drawn.

    Returns:
        The layer's symbology props as a plain dict — ``via``, ``lon_step``, ``lat_step`` and ``labels``.
    """
    return dict(scene._layer_tree.get(scene._graticule_id).symbology.props)


class TestARestylingCallKeepsTheLabelChoice:
    """``labels=False`` is a choice about the grid, not about the call that last touched it."""

    def test_a_replacing_call_leaves_a_bare_grid_bare(self, framed):
        """Restyling the spacing of an unlabelled grid does not put its degrees back.

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            The defect M7 records: ``_labels_asked(None)`` answered ``True`` on any flat map, so the
            second call relabelled a grid the caller had deliberately drawn bare — and did it silently.
        """
        framed.graticule(spacing=30.0, labels=False)
        framed.graticule(spacing=60.0)
        assert len(framed.ax.texts) == 0, (
            "a bare grid should stay bare; the axes holds "
            f"{[text.get_text() for text in framed.ax.texts]}"
        )

    def test_the_description_agrees_that_it_is_still_bare(self, framed):
        """The figure says what the axes holds, so a stored figure redraws the bare grid.

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            The labels are drawn *from* the description, so a restored ``labels=True`` would also come
            back labelled on any other tier the figure is drawn on.
        """
        framed.graticule(spacing=30.0, labels=False)
        framed.graticule(spacing=60.0)
        assert _props(framed)["labels"] is False, _props(framed)

    def test_a_replacing_call_can_still_ask_for_the_labels_back(self, framed):
        """Sticky is not stuck: naming ``labels=True`` labels the grid.

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            The option is inherited only when the call does not name it, which is the same rule
            ``visible`` follows.
        """
        framed.graticule(spacing=30.0, labels=False)
        framed.graticule(spacing=30.0, labels=True)
        assert len(framed.ax.texts) > 0, "labels=True should put the degrees back"

    def test_a_creating_call_is_still_labelled_by_default(self, framed):
        """With no grid to inherit from, ``labels=None`` is still "label what can be labelled".

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            The inheritance has to come from a layer that exists; the first call has none, and its
            default is the one every example in the docs is written against.
        """
        framed.graticule(spacing=30.0)
        assert _props(framed)["labels"] is True, _props(framed)


class TestARestylingCallKeepsTheStepItWasNotGiven:
    """The same rule for the two steps, which were lost the same way as the labels."""

    def test_naming_one_step_leaves_the_other_where_it_was(self, framed):
        """``graticule(lon_step=20)`` after ``lat_step=45`` keeps the parallels at 45.

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            ``_GraticuleSteps.asked`` fell back to ``DEFAULT_GRATICULE_STEP`` for the step the caller did
            not name, so restyling the meridians of a grid silently re-cut its parallels at 30.
        """
        framed.graticule(lon_step=10.0, lat_step=45.0)
        framed.graticule(lon_step=20.0)
        held = (_props(framed)["lon_step"], _props(framed)["lat_step"])
        assert held == (20.0, 45.0), (
            f"the unnamed step should be the one the grid carried; got {held}"
        )

    def test_spacing_still_sets_both_steps(self, framed):
        """``spacing`` outranks what the layer carries, because it names both steps.

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            The inheritance must not make ``spacing`` a half-measure: it is the spelling for a square
            grid, so it replaces both of the steps the grid had.
        """
        framed.graticule(lon_step=10.0, lat_step=45.0)
        framed.graticule(spacing=60.0)
        held = (_props(framed)["lon_step"], _props(framed)["lat_step"])
        assert held == (60.0, 60.0), f"spacing should set both steps; got {held}"

    def test_a_creating_call_still_falls_back_to_the_default_step(self, framed):
        """The first call has nothing to inherit, so it draws at the default 30 degrees.

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            The published default of the two keyword arguments, which the inheritance must not move.
        """
        framed.graticule()
        held = (_props(framed)["lon_step"], _props(framed)["lat_step"])
        assert held == (30.0, 30.0), (
            f"a creating call should draw at the default; got {held}"
        )


class TestWhatAReplacingCallStillDoesNotInherit:
    """The two options a replacement deliberately does not take from the layer it replaces."""

    def test_a_globe_still_refuses_labels_whatever_its_grid_carried(self):
        """The frame outranks the inheritance: a globe cannot carry degrees at all.

        Test scenario:
            The inherited value is read for a flat frame only. A globe's grid always carries
            ``labels=False``, and inheriting it must not turn ``labels=True`` from a refusal into a
            silently dropped argument.
        """
        globe = Map(crs=4326, globe=True)
        globe.graticule(spacing=30.0)
        with pytest.raises(ValueError, match=r"graticule\(labels=True\)"):
            globe.graticule(spacing=60.0, labels=True)
        globe.close()

    def test_a_hidden_grid_restyled_stays_hidden(self, framed):
        """``visible`` was already held this way; it is pinned here beside the two that were not.

        Args:
            framed: A flat map framed on the window.

        Test scenario:
            The contract M7's fix generalises — one rule for every option a replacing call does not
            name, rather than three answers across five arguments.
        """
        framed.graticule(spacing=30.0, visible=False)
        framed.graticule(spacing=60.0)
        assert not framed._layer_tree.is_visible(framed._graticule_id), (
            "a hidden grid should stay hidden through a restyle"
        )
