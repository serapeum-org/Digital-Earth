"""Tests for digitalearth.static.figure — grid() multi-panel layout + shared_colorbar (RP.8)."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

from digitalearth.static import Map, facet, figure, grid, shared_colorbar


@pytest.fixture
def closed_figures():
    """Close every figure the test leaves behind.

    Yields:
        None: the fixture is teardown only, so a panel grid built here cannot leak a figure into the next
        test (#382 — nine doctests already do).
    """
    yield
    plt.close("all")


class TestGrid:
    """Tests for grid()."""

    def test_returns_one_map_per_cell_sharing_a_figure(self):
        """grid(2, 2) returns four Maps that all share the one created figure."""
        fig, maps = grid(2, 2, crs=4326)
        assert len(maps) == 4, f"expected 4 panels, got {len(maps)}"
        assert all(isinstance(m, Map) for m in maps), "every panel must be a Map"
        assert all(m.fig is fig for m in maps), "all panels must share the figure"
        assert len({id(m.ax) for m in maps}) == 4, "each panel must have its own axes"

    def test_row_major_order_and_independent_drawing(self):
        """Panels are row-major and draw independently (titles land on the right axes)."""
        fig, maps = grid(1, 3, crs=4326)
        for i, m in enumerate(maps):
            m.set_title(f"p{i}")
        assert [m.ax.get_title() for m in maps] == ["p0", "p1", "p2"]

    def test_single_cell_grid(self):
        """grid(1, 1) returns a one-element list (np.atleast_1d handles the scalar axes)."""
        fig, maps = grid(1, 1, crs=4326)
        assert len(maps) == 1
        assert maps[0].fig is fig

    def test_globe_panels(self):
        """globe=True makes every panel a globe Map."""
        from digitalearth.static import projections

        fig, maps = grid(1, 2, crs=projections.orthographic(0, 0), globe=True)
        assert all(m.globe is True for m in maps), "all panels should be globes"

    def test_draw_data_on_each_panel(self, dataset):
        """Each panel renders its own data on its own axes."""
        fig, maps = grid(1, 2, crs=dataset.epsg)
        for m in maps:
            m.field(dataset)
        assert all(len(m.ax.images) == 1 for m in maps), (
            "each panel should hold its own image"
        )


class TestSharedColorbar:
    """Tests for shared_colorbar()."""

    def test_spans_given_panels(self, dataset):
        """shared_colorbar adds one colorbar axes for the supplied panels."""
        fig, maps = grid(1, 2, crs=dataset.epsg)
        im = maps[0].field(dataset).artist()
        n_axes_before = len(fig.axes)
        cbar = shared_colorbar(fig, im, maps, label="value")
        assert cbar.ax in fig.axes, "a colorbar axes should be added to the figure"
        assert len(fig.axes) == n_axes_before + 1, "exactly one colorbar axes added"
        assert cbar.ax.get_ylabel() == "value", "label should be applied"

    def test_spans_all_axes_when_maps_none(self, dataset):
        """With maps=None the colorbar steals space from every axes in the figure."""
        fig, maps = grid(1, 2, crs=dataset.epsg)
        im = maps[0].field(dataset).artist()
        cbar = shared_colorbar(fig, im)
        assert cbar.ax in fig.axes

    def test_a_none_mappable_adds_no_bar(self):
        """A layer that drew nothing has no colour scale, so it gets no colorbar rather than a TypeError.

        Test scenario:
            shared_colorbar is the documented consumer of what the layer methods return, and those can now
            hand back None when the data lies outside the display CRS. fig.colorbar(None) raises
            "TypeError: 'NoneType' object is not callable" deep in matplotlib.
        """
        from digitalearth.static.figure import grid, shared_colorbar

        fig, maps = grid(1, 1, figsize=(4, 4))
        assert shared_colorbar(fig, None, maps) is None, (
            "a None mappable should add no bar rather than raising"
        )
        assert not fig.axes[0].collections, "no colorbar axes should have been added"


def _x_group(panel):
    """Return the axes sharing ``panel``'s x axis, as a set of ids.

    Args:
        panel: A ``Map`` from :func:`grid`.

    Returns:
        The ids of every axes in its x sharing group — just its own when nothing is shared.
    """
    return {id(ax) for ax in panel.ax.get_shared_x_axes().get_siblings(panel.ax)}


def _y_group(panel):
    """Return the axes sharing ``panel``'s y axis, as a set of ids.

    Args:
        panel: A ``Map`` from :func:`grid`.

    Returns:
        The ids of every axes in its y sharing group — just its own when nothing is shared.
    """
    return {id(ax) for ax in panel.ax.get_shared_y_axes().get_siblings(panel.ax)}


class TestGridSharedAxes:
    """grid(sharex=/sharey=) — panels on one pair of axis scales."""

    def test_panels_share_nothing_by_default(self, closed_figures):
        """With no sharing asked for, each panel's sharing group is itself alone.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            Sharing is opt-in: the four panels of a default grid must stay independent, so a caller who
            draws a different region on each one keeps four separate scales.
        """
        _fig, maps = grid(2, 2, crs=4326)
        groups = [len(_x_group(panel)) for panel in maps]
        assert groups == [1, 1, 1, 1], (
            f"expected no x sharing, got group sizes {groups}"
        )

    def test_sharex_all_puts_every_panel_in_one_x_group(self, closed_figures):
        """``sharex="all"`` links all four panels' x axes into one group.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The global form of matplotlib's own vocabulary. Every panel's sharing group must hold every
            panel's axes, which is built here from the panels themselves rather than counted.
        """
        _fig, maps = grid(2, 2, crs=4326, sharex="all")
        everyone = {id(panel.ax) for panel in maps}
        assert _x_group(maps[0]) == everyone, (
            f"sharex='all' should group all four axes; got {len(_x_group(maps[0]))}"
        )

    def test_sharex_col_links_a_column_and_leaves_the_row_apart(self, closed_figures):
        """``sharex="col"`` groups the panels of one column, not of one row.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The per-column form. Panels are row-major, so in a 2x2 the first column is indices 0 and 2 —
            the expected group is built from those positions, independently of what the sharing reports.
        """
        _fig, maps = grid(2, 2, crs=4326, sharex="col")
        first_column = {id(maps[0].ax), id(maps[2].ax)}
        assert _x_group(maps[0]) == first_column, (
            "sharex='col' should group the panels of a column (row-major indices 0 and 2)"
        )

    def test_sharey_row_links_a_row_and_leaves_the_column_apart(self, closed_figures):
        """``sharey="row"`` groups the panels of one row, not of one column.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The per-row form, on the other axis. In a 2x2 the first row is row-major indices 0 and 1.
        """
        _fig, maps = grid(2, 2, crs=4326, sharey="row")
        first_row = {id(maps[0].ax), id(maps[1].ax)}
        assert _y_group(maps[0]) == first_row, (
            "sharey='row' should group the panels of a row (row-major indices 0 and 1)"
        )

    @pytest.mark.parametrize(
        ("asked", "shared"), [(True, True), (False, False), ("none", False)]
    )
    def test_the_boolean_spellings_mean_all_and_none(
        self, asked, shared, closed_figures
    ):
        """``True``/``False``/``"none"`` mean what they mean in ``plt.subplots``.

        Args:
            asked: The value passed as ``sharex``.
            shared: Whether every panel is expected to end up in one group.
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The vocabulary is matplotlib's, so the two boolean spellings have to be the aliases of
            ``"all"`` and ``"none"`` they are there rather than something a caller has to learn here.
        """
        _fig, maps = grid(1, 2, crs=4326, sharex=asked)
        together = _x_group(maps[0]) == {id(panel.ax) for panel in maps}
        assert together is shared, f"sharex={asked!r} should share={shared}"

    def test_shared_x_hides_the_inner_rows_tick_labels(self, closed_figures):
        """Under ``sharex="all"`` only the bottom row keeps x tick labels.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            matplotlib drops the redundant labels itself (``Axes._label_outer_xaxis``), and a caller needs
            to know the ticks move rather than merely linking. Measured in a 2x2: the top row's
            ``labelbottom`` goes False, the bottom row's stays True.
        """
        _fig, maps = grid(2, 2, crs=4326, sharex="all")
        labelled = [
            panel.ax.xaxis.get_tick_params(which="major").get("labelbottom")
            for panel in maps
        ]
        assert labelled == [False, False, True, True], (
            f"only the bottom row should keep x tick labels; got {labelled}"
        )

    def test_shared_y_per_row_hides_the_inner_columns_tick_labels(self, closed_figures):
        """Under ``sharey="row"`` only the left column keeps y tick labels.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The y counterpart, and the asymmetry worth pinning: labels are dropped where sharing makes
            them redundant *along that axis*, so ``"row"`` drops them for y where ``"col"`` does for x.
        """
        _fig, maps = grid(2, 2, crs=4326, sharey="row")
        labelled = [
            panel.ax.yaxis.get_tick_params(which="major").get("labelleft")
            for panel in maps
        ]
        assert labelled == [True, False, True, False], (
            f"only the left column should keep y tick labels; got {labelled}"
        )

    def test_framing_one_panel_frames_its_shared_siblings(self, closed_figures):
        """A shared panel's limits are its siblings' limits — framing one frames them all.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            This is what sharing is *for*: two panels of the same region read against each other. The
            second panel is framed by nothing of its own, so its limits can only come from the first.
        """
        _fig, maps = grid(1, 2, crs=4326, sharex="all", sharey="all")
        maps[0].set_bounds([2.0, 3.0, 8.0, 9.0])
        assert [float(v) for v in maps[1].ax.get_xlim()] == [2.0, 8.0], (
            f"the sibling should hold the framed x limits; got {maps[1].ax.get_xlim()}"
        )

    def test_framing_one_panel_leaves_an_unshared_panel_alone(self, closed_figures):
        """Without sharing the second panel keeps matplotlib's own default limits.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The other half of the claim above — the linkage is the opt-in and not something grid() does
            anyway. An unframed, undrawn axes holds the unit square.
        """
        _fig, maps = grid(1, 2, crs=4326)
        maps[0].set_bounds([2.0, 3.0, 8.0, 9.0])
        assert [float(v) for v in maps[1].ax.get_xlim()] == [0.0, 1.0], (
            f"an unshared sibling should be untouched; got {maps[1].ax.get_xlim()}"
        )

    def test_an_unknown_sharing_word_is_refused_by_matplotlib(self, closed_figures):
        """A value outside the vocabulary is refused, and the refusal names ``sharex``.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            grid() forwards the word rather than re-validating it, so the caller gets matplotlib's own
            message listing the supported values — the point of not inventing a second vocabulary.
        """
        with pytest.raises(ValueError, match="not a valid value for sharex"):
            grid(2, 2, crs=4326, sharex="both")


class TestGridFigureTitle:
    """grid(suptitle=) — one title over the panels."""

    def test_no_suptitle_by_default(self, closed_figures):
        """A grid built without one carries no figure-level text.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The title is opt-in, and ``fig.texts`` is where ``suptitle`` lands — so it must be empty until
            one is asked for.
        """
        fig, _maps = grid(1, 2, crs=4326)
        assert [text.get_text() for text in fig.texts] == [], (
            "a grid with no suptitle should carry no figure text"
        )

    def test_suptitle_titles_the_figure(self, closed_figures):
        """``suptitle=`` puts the text on the figure.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            One title over several panels is figure-level, which is matplotlib's ``Figure.suptitle`` and
            not any panel's ``set_title``.
        """
        fig, _maps = grid(1, 2, crs=4326, suptitle="rainfall, 2020")
        assert [text.get_text() for text in fig.texts] == ["rainfall, 2020"], (
            f"the figure title should be the one asked for; got {fig.texts}"
        )

    def test_the_figure_title_is_not_a_panel_title(self, closed_figures):
        """A figure title leaves every panel's own title empty.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The two titles are different places. A caller titling the figure must still be free to title
            each panel (which is what facet() does), so suptitle must not write any axes title.
        """
        _fig, maps = grid(1, 2, crs=4326, suptitle="rainfall, 2020")
        titles = [panel.ax.get_title() for panel in maps]
        assert titles == ["", ""], f"panel titles should be untouched; got {titles}"

    def test_a_panel_title_sits_beside_the_figure_title(self, closed_figures):
        """Both titles can be set, and neither replaces the other.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The combination is the normal one for small multiples — a shared heading plus one caption per
            panel — so the figure text and the axes title are read back together.
        """
        fig, maps = grid(1, 2, crs=4326, suptitle="whole")
        maps[0].set_title("left")
        assert (fig.texts[0].get_text(), maps[0].ax.get_title()) == ("whole", "left"), (
            "the figure title and the panel title should both stand"
        )

    def test_the_figure_title_is_described(self, closed_figures):
        """The heading a grid draws is carried in the figure's description, not only on the canvas.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            ST-18's premise is that a figure written down lost its heading. ``set_title`` records the
            panel's; this is the figure's, which is what ``FigureSpec.title`` holds and what
            :meth:`~digitalearth.static.map.Map.draw_figure` prefers when it restores one (M4).
        """
        _fig, maps = grid(1, 2, crs=4326, suptitle="rainfall, 2020")
        assert maps[0].figure_spec.title == "rainfall, 2020", (
            f"the grid's heading should be described; got {maps[0].figure_spec.title!r}"
        )

    def test_a_heading_set_on_the_figure_afterwards_is_described(self, closed_figures):
        """A ``fig.suptitle`` call of the caller's own is described as the figure's heading too.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            ``grid``'s own docstring sends a caller who wants to style the heading to ``fig.suptitle``, so
            the description reads the figure rather than remembering what ``grid`` was passed — one answer
            to "what heading does this figure carry", whoever wrote it.
        """
        fig, maps = grid(1, 1, crs=4326)
        fig.suptitle("styled by hand")
        assert maps[0].figure_spec.title == "styled by hand", (
            f"a heading set on the figure should be described; got {maps[0].figure_spec.title!r}"
        )

    def test_the_described_heading_is_restored_from_the_description(
        self, closed_figures
    ):
        """A panel's description, drawn again, comes back carrying the heading.

        Args:
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            The round trip is what the record is for: ``draw_figure`` restores
            ``figure.title or figure.panels[0].title``, so a described heading reaches the new figure while
            an undescribed one was lost between the two.
        """
        _fig, maps = grid(1, 1, crs=4326, suptitle="rainfall, 2020")
        assert (
            Map.from_figure(maps[0].figure_spec).ax.get_title() == "rainfall, 2020"
        ), "the heading should survive a round trip through the description"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_heading_draws_none(self, blank, closed_figures):
        """A blank ``suptitle`` is the request "no heading", and nothing is drawn for it.

        Args:
            blank: A way of asking for no heading.
            closed_figures: Teardown fixture closing the figure.

        Test scenario:
            ``set_title`` already reads ``None``, ``""`` and ``"   "`` as one request (ST-18), while
            ``grid`` guarded only on ``is not None`` and put an empty ``Text`` on the figure — two
            answers to one question on the same tier (L4).
        """
        fig, _maps = grid(1, 2, crs=4326, suptitle=blank)
        assert [text.get_text() for text in fig.texts] == [], (
            f"suptitle={blank!r} should draw no figure text; got {fig.texts}"
        )


class TestWhatFacetReadsTheSharedMappableThrough:
    """The shared colorbar is keyed through the public accessor, not the renderer's private record."""

    def test_the_module_reads_no_private_drawn_record(self):
        """``figure.py`` reaches a drawn artist through ``Map.artist`` and nowhere else.

        Test scenario:
            ``Map.artist(layer_id)`` is the accessor this row added so callers stopped reaching into
            ``_renderer.drawn`` — and ``facet``, in the same package, still read the private record to key
            its shared colorbar (L9). The source is the only place that invariant can be stated: both
            spellings return the same object, so no behaviour distinguishes them.
        """
        source = inspect.getsource(figure)
        assert "_renderer.drawn" not in source, (
            "digitalearth.static.figure should read a drawn artist through Map.artist()"
        )

    def test_the_shared_bar_is_keyed_to_the_first_panel_that_drew(self, dataset):
        """The bar represents the artist the first drawn panel hands back.

        Args:
            dataset: The raster fixture, read once per test session.

        Test scenario:
            The accessor swap must key the bar to the same mappable the private record held: the colorbar
            carries the colormap of the first panel's own artist, which is what one shared scale over the
            small multiples means.
        """
        _fig, maps = facet([dataset, dataset], crs=dataset.epsg, colorbar=True)
        drawn = maps[0].artist(maps[0].layer_ids[-1])
        assert drawn.colorbar.mappable is drawn, (
            f"the shared bar should be keyed to the first panel's artist; got {drawn.colorbar}"
        )
