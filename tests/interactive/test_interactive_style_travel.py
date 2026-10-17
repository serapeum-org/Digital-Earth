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
        drawn.field(_dem(), limits=clim)
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
            drawn.field(_dem(), limits=[0.0, 100.0])
            applied = hv.Store.lookup_options("bokeh", drawn.layers[-1], "plot").kwargs
        finally:
            drawn.close()
        assert applied.get("clim") == (0.0, 100.0), (
            f"the drawer handed HoloViews clim={applied.get('clim')!r}"
        )


def _cube():
    """Return the collection the time-slider builder draws.

    Returns:
        A two-member pyramids `DatasetCollection`, both members the committed raster.
    """
    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files([DEM] * 2)


def _cube_recorded(clim):
    """Return what a figure records for one spelling of `timecube`'s `clim`.

    Args:
        clim: The frozen colour limits, as the caller wrote them.

    Returns:
        The recorded `props["clim"]`.
    """
    drawn = InteractiveMap()
    try:
        drawn.timecube(_cube(), clim=clim)
        figure = drawn.figure_spec
        return figure.layers.get(figure.layers.ids[-1]).symbology.props.get("clim")
    finally:
        drawn.close()


class TestTheTimeSliderCubeCarriesItTheSameWay:
    """`timecube` declares the same `clim` and dropped it the same way (review R2-M13).

    The finding was written about `image`; `timecube` takes the identical keyword, with the identical
    "Frozen ``(vmin, vmax)``" wording, through a builder that records it with a bare `describe`. Fixing one
    and leaving the other is how the divergence would have come straight back — and the stakes are higher
    here, because a dropped `clim` on a cube does not merely re-scale: it sends the drawer back to
    `_global_clim`, which rescans the members to invent one, so a reloaded figure both looks different and
    pays for the scan the caller passed `clim` to avoid.
    """

    @pytest.mark.parametrize("spelling", [(0.0, 100.0), [0.0, 100.0]])
    def test_either_spelling_of_the_frozen_limits_reaches_the_figure(self, spelling):
        """Both forms of one ask have to be written down, and written down the same way.

        Args:
            spelling: The colour limits as a tuple or as a list.

        Test scenario:
            Measured before the fix: `timecube(cube, clim=(0.0, 100.0))` recorded `props['clim'] = None`
            and held the pair beside the layer, while `clim=[0.0, 100.0]` recorded `(0.0, 100.0)` and held
            nothing — the documented spelling being the one that did not survive a save.
        """
        recorded = _cube_recorded(spelling)
        assert recorded == (0.0, 100.0), (
            f"timecube(clim={spelling!r}) recorded {recorded!r}; a declared keyword has to travel"
        )

    def test_declining_the_frozen_limits_still_records_nothing(self):
        """`None` is a caller asking for the scanned range, and must not become a recorded pair.

        Test scenario:
            Recording a pair here would freeze every reloaded cube at whatever the first draw's scan of at
            most `DEFAULT_CLIM_SCAN_CAP` members happened to compute — a sampled answer written down as if
            the caller had chosen it.
        """
        recorded = _cube_recorded(None)
        assert recorded is None, (
            f"timecube(clim=None) recorded {recorded!r}; None asks for the scanned range"
        )

    def test_the_engine_is_handed_the_pair_its_own_option_is_declared_with(self):
        """HoloViews reads `clim` as a `(low, high)` tuple, so that is what the drawer must hand it.

        Test scenario:
            Measured before the fix, the list spelling reached HoloViews as `[0.0, 100.0]`. The frame is
            drawn through a `DynamicMap`, so the option is read off a materialised frame rather than off
            the map — reading the map alone would report nothing and pass either way.
        """
        import holoviews as hv

        drawn = InteractiveMap()
        try:
            drawn.timecube(_cube(), clim=[0.0, 100.0])
            element = drawn.layers[-1]
            frame = element[list(element.kdims[0].values)[0]]
            applied = hv.Store.lookup_options("bokeh", frame, "plot").kwargs
        finally:
            drawn.close()
        assert applied.get("clim") == (0.0, 100.0), (
            f"the drawer handed HoloViews clim={applied.get('clim')!r}"
        )


def _recorded_opt(clim):
    """Return what a figure records for a `clim` passed through `**opts`, not the named `limits` param.

    Args:
        clim: The colour limits, as the caller wrote them.

    Returns:
        The recorded `props["opts"]["clim"]`, or `None` when nothing was recorded.
    """
    drawn = InteractiveMap()
    try:
        drawn.field(_dem(), clim=clim)
        figure = drawn.figure_spec
        props = figure.layers.get(figure.layers.ids[-1]).symbology.props
        return (props.get("opts") or {}).get("clim")
    finally:
        drawn.close()


class TestAPairPassedThroughOptsAlsoTravels:
    """A `**opts` pair travels like the named `limits`, rather than being held and lost on save (#332 M1)."""

    @pytest.mark.parametrize("spelling", [(0.0, 50.0), [0.0, 50.0]])
    def test_a_clim_pair_through_opts_reaches_the_figure(self, spelling):
        """`field(clim=(lo, hi))` via `**opts` records the pair instead of dropping it on save.

        Args:
            spelling: The colour limits as a tuple or as a list.

        Test scenario:
            `clim` was renamed to `limits`, so the old spelling now falls through `**opts`. A tuple is what
            `travels_in_a_figure` refuses, so it was held beside the layer and silently lost from every
            saved figure, while the list form travelled — the same R2-M13 split the named parameter was
            introduced to close, reappearing on the raw opts bag. `describe_opts` now normalises a
            travelling pair to a list, so both spellings become one recorded value.
        """
        assert _recorded_opt(spelling) == (0.0, 50.0), (
            f"field(clim={spelling!r}) via **opts recorded {_recorded_opt(spelling)!r}; a pair must travel"
        )

    def test_describe_opts_holds_a_nested_pair_rather_than_flattening_it(self):
        """`describe_opts` lifts only a pair whose list form travels; a nested pair stays held (#322 survives).

        Test scenario:
            The normalisation must not re-type a nested pair. A flat numeric pair (`clim`) is recorded as a
            list so it travels; a dash pattern `(0, (5, 5))` — which JSON would re-type to `[0, [5, 5]]`, a
            value the engine refuses — stays held beside the layer and out of the described bag, the #322
            guard the opts-level fix must not undo. Checked on `describe_opts` directly, since the builders'
            opts validation would reject an element-inappropriate keyword before the travel split is reached.
        """
        from digitalearth.interactive.base import OPTS_KEY, describe_opts

        held: dict = {}
        described = describe_opts(held, {"clim": (0.0, 50.0), "line_dash": (0, (5, 5))})
        assert described.get("clim") == [0.0, 50.0], (
            f"a flat pair must be recorded as a travelling list: {described!r}"
        )
        assert "line_dash" not in described, (
            f"a nested pair must not be written down: {described!r}"
        )
        assert held[OPTS_KEY]["line_dash"] == (0, (5, 5)), (
            f"a nested pair must be held beside the layer: {held!r}"
        )
