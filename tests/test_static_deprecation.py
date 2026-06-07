"""Tests for the StaticGlyph deprecation (PD-1 / L-2)."""
import pytest

from digitalearth.static import StaticGlyph


class TestStaticGlyphDeprecation:
    """StaticGlyph entry points emit a DeprecationWarning while still working."""

    def test_init_warns(self):
        """Constructing StaticGlyph emits a DeprecationWarning pointing at Map/quickmap.

        Test scenario:
            ``StaticGlyph()`` warns; the message names the modern replacement.
        """
        with pytest.warns(DeprecationWarning, match="StaticGlyph is deprecated"):
            StaticGlyph()

    def test_plot_warns_and_still_renders(self, dataset):
        """StaticGlyph.plot warns but still returns a figure/axes.

        Test scenario:
            Calling the legacy plot path emits the warning and produces a (fig, ax) pair.
        """
        with pytest.warns(DeprecationWarning, match="digitalearth.Map"):
            fig, ax = StaticGlyph.plot(dataset)
        assert fig is not None and ax is not None, "legacy plot should still render"

    def test_message_recommends_modern_api(self):
        """The deprecation message recommends quickmap / Map.

        Test scenario:
            The shared message mentions both ``quickmap`` and ``Map`` so users know where to go.
        """
        from digitalearth.static import _DEPRECATION_MSG

        assert "quickmap" in _DEPRECATION_MSG and "Map" in _DEPRECATION_MSG, (
            f"message should point to the modern API: {_DEPRECATION_MSG!r}"
        )
