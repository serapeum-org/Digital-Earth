"""Tests for static.figure.quickpanels — one auto-captioned panel per field (gap G)."""

import matplotlib.pyplot as plt
import pytest

from digitalearth.static import Map, quickpanels
from digitalearth.static.panels import figure


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
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "A pair of fields"
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

    def test_caller_filled_overrides_the_fixed_contour_kwarg(self, dataset):
        """kind="contourf" with an explicit filled= overrides the fixed value instead of colliding.

        Test scenario:
            _FACET_KINDS["contourf"] fixes filled=True; forwarding **style raw alongside it raised
            "multiple values for keyword 'filled'". The caller's value must win, as the docstring says.
        """
        fig, maps = quickpanels([dataset, dataset], kind="contourf", filled=False)
        assert len(maps) == 2
        plt.close(fig)

    def test_figure_is_closed_if_a_panel_raises(self, dataset, mocker):
        """A render error mid-loop closes the figure rather than leaking it.

        Test scenario:
            grid() builds the figure before the draw loop; an unguarded loop that raises would leave the
            figure open in plt.get_fignums(). The loop must close it on the error path and re-raise.
        """
        from digitalearth.static import Map

        mocker.patch.object(Map, "field", side_effect=RuntimeError("boom"))
        before = set(plt.get_fignums())
        with pytest.raises(RuntimeError, match="boom"):
            quickpanels([dataset, dataset])
        assert set(plt.get_fignums()) == before, (
            "quickpanels leaked a figure on the error path"
        )

    def test_metadataless_fields_get_distinct_titles(self, dataset):
        """Two datasets that both resolve to 'Band 1' are disambiguated, not left identical.

        Test scenario:
            quickpanels is documented as laying out distinct fields; a bare single-band Dataset has no CF
            metadata, so caption() names it 'Band 1' for both panels. The auto-titles must be made unique so
            the panels are distinguishable, rather than both reading 'Band 1'.
        """
        fig, maps = quickpanels([dataset, dataset])
        first, second = maps[0].ax.get_title(), maps[1].ax.get_title()
        assert first != second, f"panels should be distinguishable; both are {first!r}"
        plt.close(fig)

    def test_caption_not_computed_when_titles_given_and_no_colorbar(
        self, dataset, mocker
    ):
        """With explicit titles and no colorbar, the metadata caption is never computed.

        Test scenario:
            A caller passing titles= for every panel with colorbar=False uses neither cap.title nor
            cap.label, so the get_source + caption work is wasted; it must be skipped.
        """
        mocker.patch(
            "digitalearth.static.panels.figure.caption",
            side_effect=AssertionError("caption computed though it was discarded"),
        )
        fig, maps = quickpanels([dataset, dataset], titles=["A", "B"], colorbar=False)
        assert [m.ax.get_title() for m in maps] == ["A", "B"]
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


class TestTitlesFrom:
    """titles_from keys each panel by a metadata field of its own source (e.g. a GRIB forecast step)."""

    def test_titles_come_from_the_named_metadata_key(self, grib_stack):
        """A forecast stack titled by step reads each band's own step, not the full caption."""
        fig, maps = quickpanels(grib_stack, cols=3, titles_from="step")
        assert [m.ax.get_title() for m in maps[:3]] == ["21600", "43200", "64800"]
        plt.close(fig)

    def test_missing_key_falls_back_to_the_caption_title(self, dataset):
        """Fields lacking the key fall back to their caption title, never a bare leading-space number."""
        fig, maps = quickpanels([dataset, dataset], titles_from="no_such_key")
        titles = [m.ax.get_title() for m in maps[:2]]
        assert all(titles), f"titles should be non-empty, got {titles}"
        assert not any(t.startswith(" ") for t in titles), (
            f"no leading-space titles: {titles}"
        )
        assert titles[0] != titles[1], (
            f"colliding fallbacks must stay disambiguated, got {titles}"
        )
        plt.close(fig)

    def test_titles_and_titles_from_are_mutually_exclusive(self, dataset):
        """Passing both titles= and titles_from= is refused, since each sets the panel titles."""
        with pytest.raises(ValueError, match="not both"):
            quickpanels([dataset, dataset], titles=["a", "b"], titles_from="band")

    def test_missing_key_fallback_computes_caption_lazily_without_colorbar(
        self, dataset
    ):
        """With colorbar off (no captions pre-built), a missing-key panel still falls back to its caption.

        Test scenario:
            colorbar=False builds no captions list, so the missing-key fallback must compute the caption
            lazily rather than index a list that is None — the title is still the non-blank caption title.
        """
        fig, maps = quickpanels(
            [dataset, dataset], titles_from="no_such_key", colorbar=False
        )
        titles = [m.ax.get_title() for m in maps[:2]]
        assert all(titles), f"titles should be non-empty, got {titles}"
        assert not any(t.startswith(" ") for t in titles), (
            f"no leading-space titles: {titles}"
        )
        plt.close(fig)

    def test_missing_key_fallback_reuses_the_built_captions(self, dataset, mocker):
        """With colorbar on, a missing-key fallback reuses the captions already built, not a recompute.

        Test scenario:
            colorbar=True builds a caption per field for the labels; the titles_from missing-key fallback must
            reuse that list, not call caption() a second time per field. Two fields missing the key => two
            caption() calls total (the label build), not four.
        """
        spy = mocker.spy(figure, "caption")
        fig, maps = quickpanels(
            [dataset, dataset], titles_from="no_such_key", colorbar=True
        )
        assert spy.call_count == 2, (
            f"caption should run once per field, got {spy.call_count}"
        )
        plt.close(fig)


class TestUniqueTitles:
    """_unique_titles numbers only the repeats, leaving unique titles untouched."""

    def test_repeats_are_numbered_uniques_untouched(self):
        """A colliding title is suffixed per occurrence; a title appearing once is returned as-is."""
        from digitalearth.static.panels.figure import _unique_titles

        assert _unique_titles(["a", "a", "b"]) == ["a (1)", "a (2)", "b"]
        assert _unique_titles(["x", "y", "z"]) == ["x", "y", "z"]
