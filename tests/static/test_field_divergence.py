"""Robust percentile limits, and a centre a diverging field is actually drawn around (ST-3).

Two halves of one row, both about a colour decision the tier made without declaring it.

**`robust=True` already worked** — cleopatra clips the limits to the 2nd/98th percentile, xarray-style, so one
outlier stops flattening the rest of the field. It reached the glyph as an *undeclared* keyword: in no
signature, in no schema, pinned by no test. A rename upstream would have turned it into a silently ignored
keyword and a figure drawn at the wrong limits. The first class below declares it and pins the numbers against
percentiles computed here, so the two cannot drift.

**`center=` was worse than undeclared.** cleopatra symmetrises the limits around it *and* swaps in a diverging
colormap when the caller named none — but this tier resolves `cmap` on every field render (through
`auto_cmap`, so a variable's own colours win), which makes `cmap` always explicit from the glyph's point of
view and suppresses that swap. So `Map.field(anomaly, center=0.0)` drew symmetric limits through a
**sequential** ramp: the neutral centre the symmetry exists for was never there. The field path now builds the
diverging ramp itself, from the resolved ramp's two ends through `cleopatra.styling.perceptual.make_diverging`,
and only when the band really straddles the centre — a ramp centred outside the data draws every value in one
arm.

Divergence stays **opt-in**: nothing happens unless `center=` or `color_scale="midpoint", midpoint=` is on the
call, and a caller who names a colormap keeps it. The two notebooks that already draw a midpoint scale pass
`cmap='RdBu_r'`, so they are in that second case and nothing they draw moves.
"""

import logging

import numpy as np
import pytest
from cleopatra.styling.perceptual import srgb_to_lab
from matplotlib import colormaps

from digitalearth.static import Map
from digitalearth.static.maps.raster import FieldColors
from digitalearth.static.render_compat import (
    CENTER_KEY,
    ROBUST_KEY,
    STATIC_STYLE_SCHEMA,
)

#: A 0-99 ramp with one wild outlier, so the robust limits and the full range are far apart.
OUTLIER_GRID = np.arange(100.0).reshape(10, 10)
OUTLIER_GRID[0, 0] = 10000.0

#: An anomaly field: values on both sides of zero, and not symmetric about it.
ANOMALY = np.array([[-2.0, -1.0, 0.0], [1.0, 3.0, 6.0]])

#: The same shape of field with no negative arm — the case a diverging ramp must decline.
POSITIVE = np.array([[12.0, 40.0], [60.0, 88.0]])

#: The band issue #390 measures: running -3..8, so a centre of 100 is off-band and a centre of 0 straddles.
ANOMALY_WIDE = np.array([[-3.0, 0.0], [4.0, 8.0]])


def _lightness(rgba):
    """Return the CIE L* of an RGBA colour, through cleopatra's own converter.

    Args:
        rgba: An RGBA tuple as a mappable's ``to_rgba`` returns it.

    Returns:
        The lightness, as a float. A diverging ramp peaks in lightness at its centre, which is the property
        these tests check instead of comparing against a second call that builds the same ramp again.
    """
    return float(srgb_to_lab(tuple(rgba)[:3])[0])


class TestRobustIsDeclared:
    """The percentile limits are part of the declared vocabulary, not a keyword that happens to pass."""

    def test_robust_is_a_declared_key(self):
        """The schema lists it, so a caller can find it."""
        assert ROBUST_KEY in STATIC_STYLE_SCHEMA.names(), sorted(
            STATIC_STYLE_SCHEMA.names()
        )

    def test_robust_says_what_it_controls(self):
        """The description names the percentiles, which is the whole content of the option."""
        doc = STATIC_STYLE_SCHEMA.keys[ROBUST_KEY].doc
        assert "percentile" in doc, doc

    def test_robust_drives_no_channel(self):
        """It decides the domain the ramp spans, not a visual variable of the layer."""
        assert STATIC_STYLE_SCHEMA.keys[ROBUST_KEY].channel is None, (
            STATIC_STYLE_SCHEMA.keys[ROBUST_KEY]
        )

    def test_a_misspelt_robust_is_nameable(self):
        """A near miss has an answer to raise with."""
        assert STATIC_STYLE_SCHEMA.suggest("robus") == "robust", (
            STATIC_STYLE_SCHEMA.suggest("robus")
        )


class TestRobustLimits:
    """What `robust=True` draws, pinned against percentiles computed here."""

    def test_the_full_range_is_what_an_outlier_gives_by_default(self):
        """Without it, one cell at 10000 sets the top of the ramp and flattens the other 99.

        Test scenario:
            The before half. `min`/`max` are read off the data independently, so the claim is about the
            drawn limits rather than about a second render.
        """
        with Map(globe=False) as canvas:
            canvas.field(OUTLIER_GRID)
            drawn = canvas.artist().get_clim()
        assert drawn == (float(OUTLIER_GRID.min()), float(OUTLIER_GRID.max())), (
            f"the default limits must be the data's own range; got {drawn}"
        )

    def test_robust_clips_to_the_second_and_ninety_eighth_percentile(self):
        """The limits are the percentiles, computed here with numpy rather than read back."""
        expected = (
            float(np.nanpercentile(OUTLIER_GRID, 2.0)),
            float(np.nanpercentile(OUTLIER_GRID, 98.0)),
        )
        with Map(globe=False) as canvas:
            canvas.field(OUTLIER_GRID, robust=True)
            drawn = canvas.artist()
            clim = drawn.get_clim()
        assert clim == pytest.approx(expected), (
            f"robust limits must be the 2nd/98th percentiles {expected}; got {clim}"
        )

    def test_the_outlier_is_then_above_the_limit(self):
        """Which is the point: the outlier is clipped rather than setting the scale."""
        with Map(globe=False) as canvas:
            canvas.field(OUTLIER_GRID, robust=True)
            top = canvas.artist().get_clim()[1]
        assert top < float(OUTLIER_GRID.max()), (
            f"the robust top must sit below the outlier; got {top}"
        )


class TestTheCentreIsReadOntoTheColourDecisions:
    """`FieldColors` is where the two spellings of a stated centre become one value."""

    def test_the_center_keyword_is_read(self):
        """`center=` states the centre directly."""
        assert FieldColors.stated_on({CENTER_KEY: 0.0}).center == 0.0, (
            "center= must be read onto the colour decisions"
        )

    def test_a_midpoint_colour_scale_states_one_too(self):
        """`color_scale="midpoint", midpoint=` is the other spelling of the same thing."""
        stated = FieldColors.stated_on({"color_scale": "midpoint", "midpoint": 2.5})
        assert stated.center == 2.5, (
            f"a midpoint colour scale must state its centre; got {stated.center}"
        )

    def test_a_midpoint_without_a_midpoint_scale_states_nothing(self):
        """On its own it is a field of the colour group that nothing reads.

        Test scenario:
            `midpoint=` without `color_scale="midpoint"` leaves cleopatra's norm linear, so treating it as a
            divergence request would hand a diverging ramp to a plainly-scaled layer.
        """
        assert FieldColors.stated_on({"midpoint": 2.5}).center is None, (
            "a midpoint with no midpoint scale must state no centre"
        )

    def test_the_centre_is_left_in_the_drawing_options(self):
        """Unlike the extreme colours, it is cleopatra's keyword and has to reach the glyph.

        Test scenario:
            Symmetrising the limits around the centre is cleopatra's job; only the diverging *ramp* is this
            tier's to supply. Popping the centre would silently drop the symmetry.
        """
        opts = {CENTER_KEY: 0.0, "cmap": "viridis"}
        FieldColors.stated_on(opts)
        assert opts == {CENTER_KEY: 0.0, "cmap": "viridis"}, opts


class TestTheCentreIsDeclared:
    """`center=` is the opt-in that asks for divergence."""

    def test_center_is_a_declared_key(self):
        """The schema lists it."""
        assert CENTER_KEY in STATIC_STYLE_SCHEMA.names(), sorted(
            STATIC_STYLE_SCHEMA.names()
        )

    def test_center_says_what_it_controls(self):
        """The description says it is the value a diverging scale is built around."""
        doc = STATIC_STYLE_SCHEMA.keys[CENTER_KEY].doc
        assert "diverging" in doc, doc

    def test_center_drives_no_channel(self):
        """Like `norm` and the extreme colours, it shapes the ramp rather than being one."""
        assert STATIC_STYLE_SCHEMA.keys[CENTER_KEY].channel is None, (
            STATIC_STYLE_SCHEMA.keys[CENTER_KEY]
        )


class TestADivergingFieldIsDrawnAroundItsCentre:
    """A stated centre gets symmetric limits *and* a ramp whose neutral colour lands on it."""

    def test_the_limits_are_symmetric_about_the_centre(self):
        """The arms are equal, so one unit of departure is one step of colour on either side."""
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, center=0.0)
            low, high = canvas.artist().get_clim()
        assert low == -high, (
            f"the limits must be symmetric about zero; got {(low, high)}"
        )

    def test_the_limits_reach_the_furthest_value(self):
        """And they are as wide as the data's furthest departure, not wider."""
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, center=0.0)
            high = canvas.artist().get_clim()[1]
        assert high == float(np.abs(ANOMALY).max()), (
            f"the arms must reach the furthest departure; got {high}"
        )

    def test_the_centre_value_is_drawn_in_the_ramps_lightest_colour(self):
        """A diverging ramp peaks in lightness at its centre; a sequential one does not.

        Test scenario:
            This is the half that was missing. Before, the same call drew symmetric limits through
            *viridis*, whose lightest colour is at the top of the ramp — so the value the symmetry was built
            around was a mid-green indistinguishable from any other.
        """
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, center=0.0)
            artist = canvas.artist()
            middle = _lightness(artist.to_rgba(0.0))
            top = _lightness(artist.to_rgba(artist.get_clim()[1]))
        assert middle > top, (
            f"the centre must be lighter than the ramp's top; got L* {middle} vs {top}"
        )

    def test_and_lighter_than_the_bottom_of_the_ramp_too(self):
        """Both arms darken away from the centre, which is what makes the two readable apart."""
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, center=0.0)
            artist = canvas.artist()
            middle = _lightness(artist.to_rgba(0.0))
            bottom = _lightness(artist.to_rgba(artist.get_clim()[0]))
        assert middle > bottom, (
            f"the centre must be lighter than the ramp's bottom; got L* {middle} vs {bottom}"
        )

    def test_a_midpoint_colour_scale_gets_the_same_ramp(self):
        """The other spelling of a stated centre — the one the docs already use — is honoured too.

        Test scenario:
            `color_scale="midpoint", midpoint=0` keeps the asymmetric domain and moves the colour centre
            instead (cleopatra's `MidpointNormalize`). It needs the diverging ramp just as much, and for the
            same reason: without one there is no neutral colour for the midpoint to land on.
        """
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, color_scale="midpoint", midpoint=0.0)
            artist = canvas.artist()
            middle = _lightness(artist.to_rgba(0.0))
            top = _lightness(artist.to_rgba(artist.get_clim()[1]))
        assert middle > top, (
            f"a midpoint scale must also get a diverging ramp; got L* {middle} vs {top}"
        )

    def test_the_midpoint_scale_keeps_its_asymmetric_domain(self):
        """The two spellings are not the same request, and this one is not turned into the other."""
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, color_scale="midpoint", midpoint=0.0)
            clim = canvas.artist().get_clim()
        assert clim == (float(ANOMALY.min()), float(ANOMALY.max())), (
            f"a midpoint scale must keep the data's own limits; got {clim}"
        )


class TestACallersOwnColormapIsNeverReplaced:
    """Divergence supplies a ramp only where the tier picked one itself."""

    def test_a_named_colormap_is_what_is_drawn(self):
        """`cmap='viridis'` with a centre draws viridis, symmetrised.

        Test scenario:
            The two notebooks that already draw a midpoint scale pass `cmap='RdBu_r'`. Replacing a named
            colormap would have restyled them, which is why the substitution is conditional.
        """
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, center=0.0, cmap="viridis")
            artist = canvas.artist()
            drawn = tuple(artist.to_rgba(0.0))
        assert drawn == tuple(colormaps["viridis"](0.5)), (
            f"the named colormap's own middle colour must be drawn; got {drawn}"
        )

    def test_a_built_colormap_is_what_is_drawn(self):
        """And a colormap object the caller built themselves, likewise."""
        built = colormaps["plasma"]
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY, center=0.0, cmap=built)
            artist = canvas.artist()
            drawn = tuple(artist.to_rgba(0.0))
        assert drawn == tuple(built(0.5)), (
            f"the caller's own colormap must be drawn; got {drawn}"
        )


class TestACentreOutsideTheDataIsDeclined:
    """Detection is the half that makes the offer safe to accept."""

    def test_the_sequential_ramp_is_kept(self, caplog):
        """A band entirely above zero keeps the ramp the tier resolved.

        Args:
            caplog: pytest's log capture.

        Test scenario:
            Centred at zero, a ramp over 12..88 would draw every value in its upper arm — worse than the
            sequential ramp it replaced. cleopatra still symmetrises the limits (that is its keyword), so
            the top of the domain is where the resolved ramp's own top colour lands.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(POSITIVE, center=0.0)
                artist = canvas.artist()
                drawn = tuple(artist.to_rgba(artist.get_clim()[1]))
        assert drawn == tuple(colormaps["viridis"](1.0)), (
            f"a declined centre must leave the sequential ramp in place; got {drawn}"
        )

    def test_the_refusal_is_logged_with_the_centre_and_the_domain(self, caplog):
        """A silent decline would read as the feature not working.

        Args:
            caplog: pytest's log capture.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(POSITIVE, center=0.0)
        assert "12.0" in caplog.text, (
            f"the message must name the domain it measured; got {caplog.text!r}"
        )

    def test_the_declined_centre_still_widens_the_colour_domain(self, caplog):
        """Only the *ramp* is declined: `center=` is cleopatra's keyword and still symmetrises the limits.

        Args:
            caplog: pytest's log capture, so the decline does not reach the test report.

        Test scenario:
            A band running 12..88 centred at zero draws through ``(-88.0, 88.0)`` — the data then occupies
            the top 43% of the ramp. That is the half of the request that *is* honoured, and the message
            and the reference page have to say so rather than reading as "the centre was ignored".
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(POSITIVE, center=0.0)
                drawn = canvas.artist().get_clim()
        assert drawn == (-88.0, 88.0), (
            f"a declined centre must still symmetrise the limits on it; got {drawn}"
        )

    def test_the_message_says_the_limits_are_still_symmetrised(self, caplog):
        """A decline that is only half a decline has to name the half it is not.

        Args:
            caplog: pytest's log capture.

        Test scenario:
            The message said the scale "is not drawn" and the sequential colormap "is kept", which a
            reader takes to mean the centre was ignored — while the colour domain silently went from
            ``(12.0, 88.0)`` to ``(-88.0, 88.0)``, a far larger change than the ramp swap it describes.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(POSITIVE, center=0.0)
        assert "symmetris" in caplog.text, (
            f"the message must say the limits are still symmetrised on the centre; got {caplog.text!r}"
        )

    def test_nothing_is_logged_when_the_centre_is_inside_the_data(self, caplog):
        """The ordinary case is quiet.

        Args:
            caplog: pytest's log capture.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(ANOMALY, center=0.0)
        assert "diverging" not in caplog.text, (
            f"a straddled centre must log nothing; got {caplog.text!r}"
        )


class TestANamedColormapWithAnOffBandCentreIsAnnounced:
    """The gap #390 closes: an off-band ``center=`` under a named cmap widened the limits in silence."""

    def test_an_off_band_centre_under_a_named_cmap_warns(self, caplog):
        """`field(center=100, cmap='RdBu_r')` must say the diverging ramp was declined.

        Args:
            caplog: pytest's log capture.

        Test scenario:
            On a named colormap the decline used to be skipped, so an unreachable centre widened the
            colour limits with no signal. The warning now fires whichever colormap was named, because what
            moved is the *limits*, not the ramp.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(ANOMALY_WIDE, center=100.0, cmap="RdBu_r")
        assert "is not drawn" in caplog.text, (
            f"an off-band centre under a named cmap must be announced; got {caplog.text!r}"
        )

    def test_the_message_does_not_call_the_kept_cmap_sequential(self, caplog):
        """The kept colormap is the caller's named one, which may be diverging — so not "sequential".

        Args:
            caplog: pytest's log capture.

        Test scenario:
            `field(center=100, cmap="RdBu_r")` keeps `RdBu_r`, a diverging map. The word "sequential"
            was accurate only in the no-cmap case where the tier resolved a sequential ramp itself; under
            a named diverging cmap it contradicts what the caller passed.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(ANOMALY_WIDE, center=100.0, cmap="RdBu_r")
        assert "sequential" not in caplog.text, (
            f"the kept colormap is the caller's (here diverging) one, not sequential; got {caplog.text!r}"
        )

    def test_it_is_announced_exactly_once(self, caplog):
        """One call, one line — not one per internal draw pass.

        Args:
            caplog: pytest's log capture.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(ANOMALY_WIDE, center=100.0, cmap="RdBu_r")
        assert caplog.text.count("is not drawn") == 1, (
            f"the decline must be said once per call; got {caplog.text!r}"
        )

    def test_the_limits_are_unchanged_by_the_warning(self, caplog):
        """The behaviour the warning describes is the one that already drew: the limits do not move.

        Args:
            caplog: pytest's log capture.

        Test scenario:
            cleopatra symmetrises ``(-3, 8)`` on a centre of 100 to ``(-3, 203)``. The warning announces
            that widening; it does not change it.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(ANOMALY_WIDE, center=100.0, cmap="RdBu_r")
                clim = canvas.artist().get_clim()
        assert clim == (-3.0, 203.0), (
            f"the limits must be symmetrised on the centre, unchanged; got {clim}"
        )

    def test_a_straddled_centre_under_a_named_cmap_stays_silent(self, caplog):
        """A centre the data straddles is the ordinary case, and says nothing.

        Args:
            caplog: pytest's log capture.

        Test scenario:
            The ramp is reachable, so there is nothing to announce. The limits symmetrise to ``(-8, 8)``,
            which the sibling test pins; here the claim is only the silence.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(ANOMALY_WIDE, center=0.0, cmap="RdBu_r")
        assert "is not drawn" not in caplog.text, (
            f"a straddled centre under a named cmap must stay silent; got {caplog.text!r}"
        )

    def test_a_straddled_centre_under_a_named_cmap_still_symmetrises(self, caplog):
        """And it still symmetrises the limits, cleopatra's keyword doing its own job.

        Args:
            caplog: pytest's log capture.
        """
        with caplog.at_level(logging.WARNING):
            with Map(globe=False) as canvas:
                canvas.field(ANOMALY_WIDE, center=0.0, cmap="RdBu_r")
                clim = canvas.artist().get_clim()
        assert clim == (-8.0, 8.0), (
            f"a straddled centre must symmetrise to the furthest arm; got {clim}"
        )


class TestNothingShiftsWithoutACentre:
    """A field with no stated centre is drawn exactly as it was."""

    def test_the_resolved_ramp_is_untouched(self):
        """No centre, no substitution: the tier's own default ramp draws the field."""
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY)
            artist = canvas.artist()
            drawn = tuple(artist.to_rgba(artist.get_clim()[0]))
        assert drawn == tuple(colormaps["viridis"](0.0)), (
            f"an uncentred field must keep the default ramp; got {drawn}"
        )

    def test_the_limits_are_the_data_range(self):
        """And the limits are measured, not symmetrised."""
        with Map(globe=False) as canvas:
            canvas.field(ANOMALY)
            clim = canvas.artist().get_clim()
        assert clim == (float(ANOMALY.min()), float(ANOMALY.max())), (
            f"an uncentred field must keep the data's range; got {clim}"
        )


class TestAMidpointBelongingToAnotherColourScale:
    """``midpoint=`` is read as a centre only for the colour scale it belongs to."""

    def test_a_midpoint_on_a_linear_scale_states_no_centre(self):
        """A midpoint carried beside a *linear* scale is not a diverging centre.

        Test scenario:
            ``midpoint`` is cleopatra's keyword for the ``ColorScale.MIDPOINT`` scale, so it only names a
            centre when that scale is the one in force. A caller who switched ``color_scale`` to
            ``"linear"`` and left the old ``midpoint=`` on the call has asked for no divergence — reading
            it anyway would resymmetrise their limits and swap their ramp.
        """
        stated = FieldColors.stated_on({"color_scale": "linear", "midpoint": 2.5})
        assert stated.center is None, (
            f"a midpoint beside a linear scale should state no centre; got {stated.center}"
        )

    def test_an_explicit_centre_still_wins_over_the_scale(self):
        """``center=`` is read whatever ``color_scale`` says, because it is this tier's own keyword.

        Test scenario:
            The reverse of the case above, so the two together pin which of the two spellings decides.
            ``center=`` is declared in this tier's schema and read first; ``midpoint`` is cleopatra's and
            read only through its own scale.
        """
        stated = FieldColors.stated_on(
            {CENTER_KEY: 1.5, "color_scale": "linear", "midpoint": 2.5}
        )
        assert stated.center == 1.5, (
            f"an explicit centre should be the one read; got {stated.center}"
        )


class TestTheRampIsBuiltFromAColormapObjectToo:
    """The two arms are taken off the resolved ramp, which is not always a name."""

    def test_a_colormap_object_is_diverged_rather_than_looked_up(self):
        """A resolved ramp handed over as an object is used directly, not indexed by name.

        Test scenario:
            ``auto_cmap`` resolves a *name* for most variables, but a caller's built colormap and a
            registry object both arrive here as a ``Colormap``. ``colormaps[cmap]`` on one of those raises
            ``TypeError``, so the two cases are told apart — and the built ramp is named after the one it
            came from, which is what is read back here.
        """
        diverged = FieldColors(center=0.0).ramp_over(
            colormaps["viridis"], np.array([-1.0, 1.0]), None
        )
        assert diverged.name == "viridis-diverging", (
            f"the ramp should be built from the object it was handed; got {diverged.name!r}"
        )

    def test_the_built_ramp_is_lightest_at_its_centre(self):
        """And it is a real diverging ramp, not merely a renamed sequential one.

        Test scenario:
            The property a diverging map is read by. Measured through cleopatra's own Lab conversion
            against both ends, so a ramp that kept viridis' monotone lightness fails.
        """
        diverged = FieldColors(center=0.0).ramp_over(
            colormaps["viridis"], np.array([-1.0, 1.0]), None
        )
        middle, bottom, top = (
            _lightness(diverged(0.5)),
            _lightness(diverged(0.0)),
            _lightness(diverged(1.0)),
        )
        assert middle > max(bottom, top), (
            f"the centre ({middle}) should be lighter than both ends ({bottom}, {top})"
        )


class TestCenteredLimitsWithoutACentre:
    """`centered_limits` only widens when a centre is stated; with none it hands the limits straight back.

    The animation path bakes the symmetric domain in once over the stack, so it asks `centered_limits` of
    every field — including the ones the caller left centre-less. That branch has to return the band it was
    given, or a frame with no `center=` would be widened to something the clip never measured.
    """

    def test_no_centre_returns_the_limits_unchanged(self):
        """With `center=None` the band is handed back as it came in.

        Test scenario:
            The ``self.center is None`` arm of `centered_limits`: the symmetrising formula is cleopatra's and
            is reached only through a stated centre, so a centre-less `FieldColors` must not move the limits.
        """
        held = FieldColors().centered_limits(-3.0, 8.0)
        assert held == (-3.0, 8.0), (
            f"a centre-less call must not widen the band; got {held}"
        )
