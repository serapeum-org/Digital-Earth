"""Tests for static.figure.quickpanels — one auto-captioned panel per field (gap G)."""

import matplotlib.pyplot as plt
import pytest

from digitalearth.static import Map, quickpanels


class TestQuickpanels:
    """quickpanels(fields) lays out one independently-styled, auto-titled panel per field."""

    def test_one_panel_per_field(self, dataset):
        """Three fields produce three visible panels, each a Map."""
        fig, maps = quickpanels([dataset, dataset, dataset], cols=2)
        assert len(maps) >= 3
        assert all(isinstance(m, Map) for m in maps)
        visible = [m for m in maps if m.ax.get_visible()]
        assert len(visible) == 3, "exactly the three fields should be drawn"
        plt.close(fig)

    def test_cols_controls_the_layout(self, dataset):
        """`cols` sets the column count; rows follow from the field count."""
        fig, maps = quickpanels([dataset] * 4, cols=2)
        # 4 fields, 2 cols -> 2 rows -> 4 cells, all used
        assert len(maps) == 4
        plt.close(fig)

    def test_each_panel_is_titled(self, dataset):
        """Every drawn panel gets a non-empty auto title from its metadata."""
        fig, maps = quickpanels([dataset, dataset])
        assert all(m.ax.get_title() for m in maps[:2]), (
            "each panel should be auto-titled"
        )
        plt.close(fig)

    def test_titles_override_the_captions(self, dataset):
        """An explicit titles= list replaces the auto captions, in order."""
        fig, maps = quickpanels([dataset, dataset], titles=["Left", "Right"])
        assert maps[0].ax.get_title() == "Left"
        assert maps[1].ax.get_title() == "Right"
        plt.close(fig)

    def test_suptitle_sets_a_figure_heading(self, dataset):
        """suptitle headings the figure, distinct from the per-panel titles."""
        fig, maps = quickpanels([dataset, dataset], suptitle="A pair of fields")
        assert (
            fig._suptitle is not None and fig._suptitle.get_text() == "A pair of fields"
        )
        plt.close(fig)

    def test_a_trailing_empty_cell_is_hidden(self, dataset):
        """Three fields in a 2-column grid leave one empty cell, which is hidden."""
        fig, maps = quickpanels([dataset, dataset, dataset], cols=2)
        assert len(maps) == 4, "2x2 grid holds the three fields plus one spare"
        assert maps[3].ax.get_visible() is False, "the spare cell is hidden"
        plt.close(fig)

    def test_colorbar_can_be_turned_off(self, dataset):
        """colorbar=False draws the panels without a colour key."""
        fig, maps = quickpanels([dataset, dataset], colorbar=False)
        assert len(maps) == 2
        plt.close(fig)

    def test_a_bad_kind_is_refused(self, dataset):
        """A render that is not a panel kind is refused by name."""
        with pytest.raises(ValueError, match="not a panel render"):
            quickpanels([dataset, dataset], kind="sankey")

    def test_no_fields_is_refused(self):
        """An empty field list raises rather than building an empty figure."""
        with pytest.raises(ValueError, match="at least one field"):
            quickpanels([])

    def test_titles_length_must_match(self, dataset):
        """A titles list of the wrong length is refused, like animate's."""
        with pytest.raises(ValueError, match="one entry per field"):
            quickpanels([dataset, dataset], titles=["only one"])
