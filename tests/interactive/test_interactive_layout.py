"""Multi-panel layouts and linked views across interactive maps (IN-7).

``panels``/``tabs``/``swipe`` compose several maps — the across-scenes composition the static tier's
``grid``/``shared_colorbar`` are for — so comparing, small-multiplying and swiping maps is possible where
the tier previously only overlaid one map's layers with ``*``.
"""

import pytest

hv = pytest.importorskip("holoviews")
pytest.importorskip("panel")

import panel as pn  # noqa: E402

from digitalearth.interactive import (  # noqa: E402
    InteractiveMap,
    grid,
    linked_views,
    panels,
    swipe,
    tabs,
)


@pytest.fixture
def maps(dataset):
    """Return two Web-Mercator maps, each carrying one raster layer.

    Args:
        dataset: The raster fixture.

    Returns:
        A tuple of two ``InteractiveMap`` instances.
    """
    return InteractiveMap().field(dataset), InteractiveMap().field(dataset)


class TestPanels:
    """``panels`` lays maps out as a HoloViews Layout."""

    def test_two_maps_make_a_two_panel_layout(self, maps):
        """Two maps compose into a ``Layout`` of two panels.

        Args:
            maps: The two-map fixture.
        """
        layout = panels(*maps)
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2

    def test_cols_wraps_into_a_grid(self, dataset):
        """``cols`` wraps the row into a grid that many panels wide.

        Args:
            dataset: The raster fixture.
        """
        four = [InteractiveMap().field(dataset) for _ in range(4)]
        layout = panels(*four, cols=2)
        assert isinstance(layout, hv.Layout)

    def test_a_single_panel_is_refused(self, maps):
        """One panel is a plain render, not a layout.

        Args:
            maps: The two-map fixture (one used).
        """
        with pytest.raises(ValueError, match="two or more"):
            panels(maps[0])


class TestTabsAndSwipe:
    """``tabs`` and ``swipe`` are the Panel-hosted multi-panel views."""

    def test_tabs_makes_one_tab_per_map(self, maps):
        """Each named map lands on its own tab.

        Args:
            maps: The two-map fixture.
        """
        layout = tabs(rain=maps[0], temp=maps[1])
        assert isinstance(layout, pn.Tabs)
        assert len(layout) == 2

    def test_tabs_needs_at_least_one_map(self):
        """An empty tab set is refused."""
        with pytest.raises(ValueError, match="at least one"):
            tabs()

    def test_swipe_composes_two_maps(self, maps):
        """``swipe`` returns a Panel ``Swipe`` of the two maps.

        Args:
            maps: The two-map fixture.
        """
        assert isinstance(swipe(*maps), pn.layout.Swipe)


class TestLinkedViews:
    """``linked_views`` wires a declarative RangeToolLink overview->detail (IN-7 #433)."""

    def test_returns_a_two_panel_layout(self, maps):
        """detail + overview compose into a two-panel Layout.

        Args:
            maps: The two-map fixture.
        """
        detail, overview = maps
        layout = linked_views(detail, overview)
        assert isinstance(layout, hv.Layout), f"got {type(layout)}"
        assert len(layout) == 2, f"expected two panels, got {len(layout)}"

    def test_registers_a_rangetoollink(self, maps):
        """The overview becomes a RangeToolLink source — the declarative link, not implicit shared axes.

        Args:
            maps: The two-map fixture.
        """
        from holoviews.plotting.links import Link, RangeToolLink

        detail, overview = maps
        before = sum(len(v) for v in Link.registry.values())
        linked_views(detail, overview)
        links = [lk for lks in Link.registry.values() for lk in lks]
        assert any(isinstance(lk, RangeToolLink) for lk in links), (
            "a RangeToolLink must be registered"
        )
        assert sum(len(v) for v in Link.registry.values()) > before, (
            "a link must have been added"
        )

    def test_unknown_axes_is_refused(self, maps):
        """An axes spec other than both/x/y is refused.

        Args:
            maps: The two-map fixture.
        """
        detail, overview = maps
        with pytest.raises(ValueError, match="unknown axes"):
            linked_views(detail, overview, axes="z")


class TestGrid:
    """``grid`` lays maps out as a linked GridSpace of small multiples (IN-7 #433)."""

    def test_four_maps_make_a_four_cell_grid(self, dataset):
        """Four maps fill a 2x2 GridSpace.

        Args:
            dataset: The raster fixture.
        """
        four = [InteractiveMap().field(dataset) for _ in range(4)]
        gs = grid(*four, cols=2)
        assert isinstance(gs, hv.GridSpace), f"got {type(gs)}"
        assert set(gs.keys()) == {(0, 0), (1, 0), (0, 1), (1, 1)}, (
            f"four maps at cols=2 must fill a 2x2 grid, got {sorted(gs.keys())}"
        )

    def test_a_single_map_is_refused(self, maps):
        """One map is a plain render, not a grid.

        Args:
            maps: The two-map fixture (one used).
        """
        with pytest.raises(ValueError, match="two or more"):
            grid(maps[0])

    def test_cols_below_one_is_refused(self, maps):
        """A grid narrower than one column is refused (IN-7 #433).

        Args:
            maps: The two-map fixture.
        """
        with pytest.raises(ValueError, match="at least 1"):
            grid(*maps, cols=0)
