"""``facet`` — a raster stack as small multiples on one shared colour scale (ST-8, #223).

Small multiples are read against each other, so every panel has to colour one value the same way: the scale
is resolved **once over the whole stack**, not per panel. These pin that, the panel layout (``col_wrap``),
the per-panel titles and the one colorbar spanning the panels. Expected limits and class edges are computed
from the arrays the test builds, never through the code under test.
"""

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map, facet

#: Four 4 x 4 frames whose values climb frame by frame, so a per-panel scale would differ on every panel.
FRAMES = [np.arange(16, dtype="float64").reshape(4, 4) + 10.0 * i for i in range(4)]

#: The lon/lat placement every frame shares.
GEO = GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)


def _dataset(arr: np.ndarray) -> Dataset:
    """Wrap one array as a lon/lat dataset.

    Args:
        arr: A 2-D frame, or a 3-D ``(bands, rows, cols)`` stack.

    Returns:
        A pyramids `Dataset`.
    """
    return Dataset.from_array(arr=arr, geo_ref=GEO, no_data_value=-9999.0)


@pytest.fixture
def stack():
    """Four single-band frames.

    Returns:
        A list of pyramids `Dataset`.
    """
    return [_dataset(arr) for arr in FRAMES]


def _colorbars(fig) -> list:
    """Return the colorbar axes on a figure.

    Args:
        fig: The figure.

    Returns:
        The axes matplotlib created for a colorbar.
    """
    return [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]


class TestOneSharedScale:
    """Every panel colours one value the same way."""

    def test_one_map_per_frame_on_one_figure(self, stack):
        """Four frames, four panels, one figure.

        Args:
            stack: The frames.
        """
        fig, maps = facet(stack, crs=4326)
        assert len(maps) == 4, len(maps)
        assert all(isinstance(m, Map) and m.fig is fig for m in maps), maps

    def test_every_panel_shares_the_stack_s_limits(self, stack):
        """Each panel's colour limits are the whole stack's minimum and maximum.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326)
        expected = (
            float(min(a.min() for a in FRAMES)),
            float(max(a.max() for a in FRAMES)),
        )
        clims = [m.layers[-1][1].get_clim() for m in maps]
        assert clims == [expected] * 4, clims

    def test_explicit_limits_win(self, stack):
        """A caller's ``vmin``/``vmax`` are the shared scale.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326, vmin=5.0, vmax=25.0)
        clims = {m.layers[-1][1].get_clim() for m in maps}
        assert clims == {(5.0, 25.0)}, clims

    def test_a_scheme_is_cut_once_over_the_stack(self, stack):
        """``scheme=``/``k=`` classify every panel with the same edges, cut over all the frames.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326, scheme="equal_interval", k=4)
        lo, hi = min(a.min() for a in FRAMES), max(a.max() for a in FRAMES)
        expected = np.linspace(lo, hi, 5)
        for m in maps:
            norm = m.layers[-1][1].norm
            assert isinstance(norm, BoundaryNorm), type(norm)
            assert np.allclose(norm.boundaries, expected), norm.boundaries

    def test_filled_contours_share_the_levels_given(self, stack):
        """``kind="contourf"`` takes explicit levels, which every panel draws.

        Args:
            stack: The frames.
        """
        levels = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        _, maps = facet(stack, crs=4326, kind="contourf", levels=levels)
        for m in maps:
            assert list(m.layers[-1][1].levels) == levels, m.layers[-1][1].levels

    def test_filled_contours_without_levels_are_refused(self, stack):
        """Each panel would pick its own levels from its own frame, which is the bug facets exist to avoid.

        Args:
            stack: The frames.
        """
        with pytest.raises(ValueError, match="levels"):
            facet(stack, crs=4326, kind="contourf")

    def test_each_panel_describes_its_layer(self, stack):
        """A panel is an ordinary ``Map``: its frame is a described raster layer.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326)
        kinds = [[m.figure_spec.layers.get(i).kind for i in m.layer_ids] for m in maps]
        assert kinds == [["raster"]] * 4, kinds


class TestTheLayout:
    """One row by default; ``col_wrap`` folds it."""

    def test_one_row_by_default(self, stack):
        """Four frames side by side.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326)
        geometry = {m.ax.get_subplotspec().get_geometry()[:2] for m in maps}
        assert geometry == {(1, 4)}, geometry

    def test_col_wrap_folds_the_row(self, stack):
        """``col_wrap=3`` puts four frames on two rows of three, the last slots hidden.

        Args:
            stack: The frames.
        """
        fig, maps = facet(stack, crs=4326, col_wrap=3)
        geometry = {m.ax.get_subplotspec().get_geometry()[:2] for m in maps}
        assert geometry == {(2, 3)}, geometry
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        hidden = [ax for ax in panels if not ax.get_visible()]
        assert (len(panels), len(hidden)) == (6, 2), (len(panels), len(hidden))

    def test_a_col_wrap_below_one_is_refused(self, stack):
        """Zero columns is no layout.

        Args:
            stack: The frames.
        """
        with pytest.raises(ValueError, match="col_wrap"):
            facet(stack, crs=4326, col_wrap=0)

    def test_an_empty_stack_is_refused(self):
        """No frames, no panels."""
        with pytest.raises(ValueError, match="no frames"):
            facet([], crs=4326)


class TestTitlesAndKey:
    """Each panel says which frame it is; one bar keys them all."""

    def test_panels_are_titled_by_index(self, stack):
        """With no labels, a panel is titled ``"<col> = <index>"``.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326, col="time")
        titles = [m.ax.get_title() for m in maps]
        assert titles == ["time = 0", "time = 1", "time = 2", "time = 3"], titles

    def test_panels_are_titled_by_label(self, stack):
        """``labels=`` names each frame.

        Args:
            stack: The frames.
        """
        _, maps = facet(
            stack, crs=4326, col="month", labels=["Jan", "Feb", "Mar", "Apr"]
        )
        titles = [m.ax.get_title() for m in maps]
        assert titles == ["month = Jan", "month = Feb", "month = Mar", "month = Apr"], (
            titles
        )

    def test_labels_that_do_not_number_the_frames_are_refused(self, stack):
        """Three labels for four frames would title a panel with the wrong frame's name.

        Args:
            stack: The frames.
        """
        with pytest.raises(ValueError, match="labels"):
            facet(stack, crs=4326, labels=["Jan", "Feb", "Mar"])

    def test_one_colorbar_spans_the_panels(self, stack):
        """Exactly one bar, labelled as asked.

        Args:
            stack: The frames.
        """
        fig, _ = facet(stack, crs=4326, cbar_label="mm")
        bars = _colorbars(fig)
        assert len(bars) == 1, len(bars)
        assert bars[0].get_ylabel() == "mm", bars[0].get_ylabel()

    def test_the_bar_can_be_left_off(self, stack):
        """``colorbar=False`` draws none.

        Args:
            stack: The frames.
        """
        fig, _ = facet(stack, crs=4326, colorbar=False)
        assert _colorbars(fig) == [], _colorbars(fig)


class TestABandStack:
    """A multi-band dataset facets over its bands."""

    def test_one_panel_per_band_on_one_scale(self):
        """Three bands, three panels titled by band number, one shared scale."""
        cube = np.stack(FRAMES[:3])
        _, maps = facet(_dataset(cube), crs=4326)
        titles = [m.ax.get_title() for m in maps]
        clims = {m.layers[-1][1].get_clim() for m in maps}
        assert titles == ["band = 1", "band = 2", "band = 3"], titles
        assert clims == {(float(cube.min()), float(cube.max()))}, clims
