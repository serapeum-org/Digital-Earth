"""Multi-panel layouts and linked views across interactive maps (IN-7).

``panels``/``tabs``/``swipe`` compose several maps — the across-scenes composition the static tier's
``grid``/``shared_colorbar`` are for — so comparing, small-multiplying and swiping maps is possible where
the tier previously only overlaid one map's layers with ``*``.
"""

import pytest

hv = pytest.importorskip("holoviews")
pytest.importorskip("panel")

import panel as pn  # noqa: E402

from digitalearth.interactive import InteractiveMap, panels, swipe, tabs  # noqa: E402


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
