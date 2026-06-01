"""Tests for digitalearth.scene.figure — grid() multi-panel layout + shared_colorbar (RP.8)."""
import numpy as np
import pytest

from digitalearth.scene import Map, grid, shared_colorbar


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
        assert len(maps) == 1 and maps[0].fig is fig

    def test_globe_panels(self):
        """globe=True makes every panel a globe Map."""
        from digitalearth.scene import projections

        fig, maps = grid(1, 2, crs=projections.orthographic(0, 0), globe=True)
        assert all(m.globe is True for m in maps), "all panels should be globes"

    def test_draw_data_on_each_panel(self, dataset):
        """Each panel renders its own data on its own axes."""
        fig, maps = grid(1, 2, crs=dataset.epsg)
        for m in maps:
            m.imshow(dataset)
        assert all(len(m.ax.images) == 1 for m in maps), "each panel should hold its own image"


class TestSharedColorbar:
    """Tests for shared_colorbar()."""

    def test_spans_given_panels(self, dataset):
        """shared_colorbar adds one colorbar axes for the supplied panels."""
        fig, maps = grid(1, 2, crs=dataset.epsg)
        im = maps[0].imshow(dataset)
        n_axes_before = len(fig.axes)
        cbar = shared_colorbar(fig, im, maps, label="value")
        assert cbar.ax in fig.axes, "a colorbar axes should be added to the figure"
        assert len(fig.axes) == n_axes_before + 1, "exactly one colorbar axes added"
        assert cbar.ax.get_ylabel() == "value", "label should be applied"

    def test_spans_all_axes_when_maps_none(self, dataset):
        """With maps=None the colorbar steals space from every axes in the figure."""
        fig, maps = grid(1, 2, crs=dataset.epsg)
        im = maps[0].imshow(dataset)
        cbar = shared_colorbar(fig, im)
        assert cbar.ax in fig.axes
