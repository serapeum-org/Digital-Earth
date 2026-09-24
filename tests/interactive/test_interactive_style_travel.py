"""Which spelling of a declared style keyword a saved figure carries (review R2-M13).

A keyword this tier **declares** — :data:`~digitalearth.interactive.style_fold.INTERACTIVE_STYLE_SCHEMA`
names `clim` "Colour limits, as (low, high)" — has to survive being written down, in the spelling its own
documentation gives. A tuple is exactly what
:func:`~digitalearth.base.spec._serial.travels_in_a_figure` refuses, so the documented form was held beside
the layer and dropped from every saved figure while the undocumented list form travelled: two spellings of
one intent behaving differently across a save and reload.
"""

import pytest

pytest.importorskip(
    "holoviews", reason="the interactive tier needs the interactive environment"
)
pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)

from digitalearth.interactive import InteractiveMap  # noqa: E402

DEM = "examples/data/acc4000.tif"


def _dem():
    """Return the raster the image builder draws.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(DEM)


def _recorded(clim):
    """Return what a figure records for one spelling of `clim`.

    Args:
        clim: The colour limits, as the caller wrote them.

    Returns:
        The recorded `props["clim"]`.
    """
    drawn = InteractiveMap()
    try:
        drawn.image(_dem(), clim=clim)
        figure = drawn.figure_spec
        return figure.layers.get(figure.layers.ids[-1]).symbology.props.get("clim")
    finally:
        drawn.close()


class TestTheDocumentedSpellingIsTheOneThatTravels:
    """`clim` is declared as a pair, and a pair is what a figure never carried."""

    @pytest.mark.parametrize("spelling", [(0.0, 100.0), [0.0, 100.0]])
    def test_either_spelling_of_the_colour_limits_reaches_the_figure(self, spelling):
        """Both forms of one ask have to be written down, and written down the same way.

        Args:
            spelling: The colour limits as a tuple or as a list.

        Test scenario:
            Measured before the fix: `image(dem, clim=(0.0, 100.0))` recorded `props['clim'] = None` — the
            documented spelling, dropped from every saved figure — while `clim=[0.0, 100.0]` recorded the
            pair and handed HoloViews a list rather than the tuple its own docstring names. A figure is
            normalised to one spelling now, so a save and reload cannot change which one the engine sees.
        """
        assert _recorded(spelling) == (0.0, 100.0), (
            f"image(clim={spelling!r}) recorded {_recorded(spelling)!r}; a declared keyword has to travel"
        )

    def test_declining_the_colour_limits_still_records_nothing(self):
        """`None` is a caller declining a keyword, and must not become a pair.

        Test scenario:
            The normalisation must not turn "auto-scale" into a recorded limit, which would freeze the
            colour range of every reloaded figure at whatever the first draw happened to compute.
        """
        assert _recorded(None) is None, (
            f"image(clim=None) recorded {_recorded(None)!r}; None is a decline, not a pair"
        )

    def test_the_engine_is_handed_the_pair_its_own_option_is_declared_with(self):
        """HoloViews reads `clim` as a `(low, high)` tuple, so that is what the drawer must hand it.

        Test scenario:
            The figure carries the JSON-safe spelling and the drawer restores the engine's — otherwise the
            list form reached HoloViews as a list, which is a second undocumented spelling of one option.
        """
        import holoviews as hv

        drawn = InteractiveMap()
        try:
            drawn.image(_dem(), clim=[0.0, 100.0])
            applied = hv.Store.lookup_options("bokeh", drawn.layers[-1], "plot").kwargs
        finally:
            drawn.close()
        assert applied.get("clim") == (0.0, 100.0), (
            f"the drawer handed HoloViews clim={applied.get('clim')!r}"
        )
