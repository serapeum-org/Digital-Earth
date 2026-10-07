"""Theme support for the interactive tier (IN-18).

A dark-mode map used to be expressible only by threading ``bgcolor`` through ``**opts`` at every call.
``theme()`` records one Bokeh theme on the map and applies it to the renderer whenever the map composes.
"""

import pytest

pytest.importorskip("geoviews")

import holoviews as hv  # noqa: E402

from digitalearth.interactive import InteractiveMap  # noqa: E402


class TestTheme:
    """A theme is recorded, validated, applied on render, and chains like every other builder."""

    def test_theme_is_a_chaining_builder(self):
        """``theme()`` returns the map, so it sits in a builder chain."""
        m = InteractiveMap()
        assert m.theme("dark_minimal") is m

    def test_a_builtin_name_reaches_the_renderer_on_render(self):
        """Rendering a themed map sets the theme on HoloViews' Bokeh renderer."""
        hv.renderer("bokeh")
        m = InteractiveMap().add_layer(hv.Points([(0.0, 0.0)])).theme("dark_minimal")
        m.render()
        assert hv.Store.renderers["bokeh"].theme == "dark_minimal"

    def test_a_theme_instance_is_accepted(self):
        """A ``bokeh.themes.Theme`` built elsewhere is stored as-is (not only built-in names)."""
        from bokeh.themes import built_in_themes

        theme = built_in_themes["night_sky"]
        assert InteractiveMap().theme(theme)._theme is theme

    def test_an_unknown_name_is_refused(self):
        """A typo'd theme name is refused, naming the built-ins."""
        with pytest.raises(ValueError, match="unknown Bokeh theme"):
            InteractiveMap().theme("darkmode")

    def test_theme_none_clears_a_previously_set_theme(self):
        """M1 — ``theme(None)`` restores the default; it does not leave the last theme in force."""
        m = InteractiveMap().add_layer(hv.Points([(0.0, 0.0)]))
        m.theme("dark_minimal").render()
        m.theme(None).render()
        assert hv.Store.renderers["bokeh"].theme != "dark_minimal"

    def test_a_themeless_map_does_not_inherit_a_prior_theme(self):
        """M1 — a map with no theme rendered after a themed one is not drawn under the leaked theme."""
        InteractiveMap().add_layer(hv.Points([(0.0, 0.0)])).theme("night_sky").render()
        InteractiveMap().add_layer(hv.Points([(1.0, 1.0)])).render()
        assert hv.Store.renderers["bokeh"].theme != "night_sky"
