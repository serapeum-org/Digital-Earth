"""The static field path takes its colormap's extremes from the layer's scale (ST-10).

Three colours decide what a field does with a value it cannot place: the nodata cell, the value below `vmin`
and the value above `vmax`. The tier stated **none** of them. A nodata cell was drawn with matplotlib's
default "bad" colour, which is fully transparent, so it was indistinguishable from no data at all; and a
clipped value took the colormap's own end colour, so a field clipped at `vmax` looked exactly like one that
really peaks there. The one place the tier *did* state a colour — the classified vector path — hard-coded
`with_extremes(bad=MISSING_COLOR)`, which no caller could change.

These pin the declared form: `missing=`, `over=` and `under=` on a field render, routed through
:class:`~digitalearth.base.spec.scale.Scale`'s extreme colours, applied to the drawn artist's colormap, and
published on the layer's colour encoding so the figure's *description* carries them too.

Nothing shifts for a caller who states none: the colormap is then left exactly as `auto_cmap` resolved it,
which `test_a_nodata_cell_is_still_blank_when_nothing_is_stated` is here to keep true.
"""

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgba

from digitalearth.static import Map
from digitalearth.static.maps.raster import FieldColors
from digitalearth.static.render_compat import EXTREME_KEYS, STATIC_STYLE_SCHEMA

MISSING = "#cccccc"
OVER = "#ff0000"
UNDER = "#0000ff"

#: A ramp with a hole in it, so "the colour of a cell that has no value" is a question the figure asks.
GAPPY = np.array([[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0]])

#: A ramp wider than the limits the tests clip it to, so both ends fall outside the domain.
WIDE = np.array([[1.0, 4.0], [7.0, 9.0]])


class TestTheExtremeColoursAreDeclared:
    """A caller can ask what the tier accepts, and a typo has an answer."""

    @pytest.mark.parametrize("key", EXTREME_KEYS)
    def test_the_key_is_declared(self, key):
        """Each of the three appears in the schema.

        Args:
            key: One of the extreme-colour keywords.
        """
        assert key in STATIC_STYLE_SCHEMA.names(), sorted(STATIC_STYLE_SCHEMA.names())

    @pytest.mark.parametrize("key", EXTREME_KEYS)
    def test_the_key_says_what_it_colours(self, key):
        """The declaration carries a description, which is what a discovery surface shows.

        Args:
            key: One of the extreme-colour keywords.
        """
        assert STATIC_STYLE_SCHEMA.keys[key].doc.strip(), STATIC_STYLE_SCHEMA.keys[key]

    @pytest.mark.parametrize("key", EXTREME_KEYS)
    def test_the_key_drives_no_channel(self, key):
        """These colour what the ramp cannot, so they are not a visual channel of the layer.

        Args:
            key: One of the extreme-colour keywords.

        Test scenario:
            The same reason `color` (the ColorScaling group) and `norm` declare none: the layer's colour
            comes from `cmap` plus the data, and these three say what happens off the ends of that.
        """
        assert STATIC_STYLE_SCHEMA.keys[key].channel is None, STATIC_STYLE_SCHEMA.keys[
            key
        ]

    def test_a_misspelt_extreme_is_nameable(self):
        """A near miss now has an answer to raise with."""
        assert STATIC_STYLE_SCHEMA.suggest("mising") == "missing", (
            STATIC_STYLE_SCHEMA.suggest("mising")
        )

    def test_the_colours_are_read_off_the_drawing_options(self):
        """`FieldColors` is what the stated colours become, in one value rather than three keywords.

        Test scenario:
            The decision is asked at two points of one draw — the keywords must leave `opts` before the
            glyph is built, and the colours can only be applied once the artist exists. A value read at the
            first and asked at the second is what keeps those from drifting.
        """
        assert FieldColors.stated_on({"missing": MISSING}).missing == MISSING, (
            "the stated colour must be read onto the value object"
        )

    def test_the_colours_are_taken_out_of_the_drawing_options(self):
        """They are the tier's own keywords, so they must not travel on to the glyph.

        Test scenario:
            cleopatra validates its keyword list and refuses anything outside it, so a key left in `opts`
            fails the whole render with "The given keyword argument:missing is not correct". Reading them
            off `opts` removes them, which is what makes them the tier's.
        """
        opts = {"missing": MISSING, "cmap": "viridis"}
        FieldColors.stated_on(opts)
        assert opts == {"cmap": "viridis"}, opts

    def test_a_call_that_states_nothing_says_so(self):
        """Which is the signal to leave the resolved colormap exactly as it is."""
        assert not FieldColors.stated_on({"cmap": "viridis"}).states_extremes, (
            "a call with no stated colour must report none"
        )


class TestNothingShiftsWhenNothingIsStated:
    """The colormap is untouched unless a colour was asked for."""

    def test_a_nodata_cell_is_still_blank_when_nothing_is_stated(self):
        """matplotlib's transparent "bad" colour stays the default.

        Test scenario:
            The before half of this row. Painting nodata grey by default would repaint every existing
            figure's gaps, so the capability is opt-in and this is the pin that keeps it so.
        """
        with Map(globe=False) as canvas:
            canvas.field(GAPPY)
            artist = canvas.artist()
            drawn = artist.get_cmap().get_bad()
        assert tuple(drawn) == (0.0, 0.0, 0.0, 0.0), (
            f"an unstated nodata colour must stay fully transparent; got {tuple(drawn)}"
        )

    def test_a_value_above_the_limit_still_takes_the_ramps_end(self):
        """Without `over=`, the top of the ramp is what an out-of-range value gets."""
        with Map(globe=False) as canvas:
            canvas.field(WIDE, vmin=2.0, vmax=6.0)
            artist = canvas.artist()
            above = tuple(artist.to_rgba(9.0))
            top = tuple(artist.to_rgba(6.0))
        assert above == top, (
            f"an unstated over colour must leave the ramp's end in place; got {above} vs {top}"
        )


class TestTheStatedColoursAreDrawn:
    """What the caller states is what the artist colours with."""

    def test_missing_paints_the_nodata_cells(self):
        """`missing=` is the colour a cell with no value is drawn in."""
        with Map(globe=False) as canvas:
            canvas.field(GAPPY, missing=MISSING)
            artist = canvas.artist()
            drawn = tuple(artist.get_cmap().get_bad())
        assert drawn == to_rgba(MISSING), (
            f"the stated nodata colour must reach the colormap; got {drawn}"
        )

    def test_under_paints_a_value_below_the_lower_limit(self):
        """A value under `vmin` is drawn in the stated colour, not the ramp's bottom."""
        with Map(globe=False) as canvas:
            canvas.field(WIDE, vmin=2.0, vmax=6.0, under=UNDER)
            artist = canvas.artist()
            drawn = tuple(artist.to_rgba(1.0))
        assert drawn == to_rgba(UNDER), (
            f"the stated under colour must reach the drawn artist; got {drawn}"
        )

    def test_over_paints_a_value_above_the_upper_limit(self):
        """And a value over `vmax` likewise."""
        with Map(globe=False) as canvas:
            canvas.field(WIDE, vmin=2.0, vmax=6.0, over=OVER)
            artist = canvas.artist()
            drawn = tuple(artist.to_rgba(9.0))
        assert drawn == to_rgba(OVER), (
            f"the stated over colour must reach the drawn artist; got {drawn}"
        )

    def test_the_ramp_itself_is_unchanged(self):
        """Stating the extremes does not restyle the values that *are* in range.

        Test scenario:
            `with_extremes` copies the colormap rather than rebuilding it, so an in-range value has to come
            out exactly as the named colormap draws it — read here off `viridis` directly rather than off a
            second render, so the two sides are not the same call twice.
        """
        with Map(globe=False) as canvas:
            canvas.field(WIDE, cmap="viridis", vmin=2.0, vmax=6.0, over=OVER)
            artist = canvas.artist()
            middle = tuple(artist.to_rgba(4.0))
        assert middle == tuple(colormaps["viridis"](0.5)), (
            f"an in-range value must keep the colormap's own colour; got {middle}"
        )

    def test_a_classified_field_honours_them_too(self):
        """A `scheme=` render draws through a built class lookup, and the extremes still reach it.

        Test scenario:
            The classified path replaces `cmap` with a `ListedColormap` of the class colours, so a fold
            applied only to the continuous path would silently do nothing for exactly the layers the
            hard-coded `with_extremes(bad=...)` was written for.
        """
        with Map(globe=False) as canvas:
            canvas.field(GAPPY, scheme="equal_interval", k=3, missing=MISSING)
            artist = canvas.artist()
            drawn = tuple(artist.get_cmap().get_bad())
        assert drawn == to_rgba(MISSING), (
            f"a classified field must honour the stated nodata colour; got {drawn}"
        )


class TestACallersOwnColormapKeepsItsExtremes:
    """A colormap handed in already carrying extremes is not flattened by the fold."""

    def test_an_extreme_the_caller_built_in_survives(self):
        """`over` set on the colormap itself stays, when only `missing` is stated as a keyword.

        Test scenario:
            This is the reason the fold states `None` for an unasked extreme instead of a default: the
            caller's colormap is the more specific statement, and `Colormap.with_extremes` copies the
            extremes it is not given.
        """
        built = colormaps["viridis"].with_extremes(over=OVER)
        with Map(globe=False) as canvas:
            canvas.field(WIDE, cmap=built, vmin=2.0, vmax=6.0, missing=MISSING)
            artist = canvas.artist()
            drawn = tuple(artist.to_rgba(9.0))
        assert drawn == to_rgba(OVER), (
            f"the colormap's own over colour must survive; got {drawn}"
        )

    def test_and_the_keyword_is_applied_beside_it(self):
        """The stated colour lands on the same colormap, so both are in force."""
        built = colormaps["viridis"].with_extremes(over=OVER)
        with Map(globe=False) as canvas:
            canvas.field(WIDE, cmap=built, vmin=2.0, vmax=6.0, missing=MISSING)
            artist = canvas.artist()
            drawn = tuple(artist.get_cmap().get_bad())
        assert drawn == to_rgba(MISSING), (
            f"the stated nodata colour must apply beside the built-in one; got {drawn}"
        )

    def test_a_stated_colour_wins_over_the_colormaps_own(self):
        """When both name the same extreme, the keyword is the later statement."""
        built = colormaps["viridis"].with_extremes(over="#00ff00")
        with Map(globe=False) as canvas:
            canvas.field(WIDE, cmap=built, vmin=2.0, vmax=6.0, over=OVER)
            artist = canvas.artist()
            drawn = tuple(artist.to_rgba(9.0))
        assert drawn == to_rgba(OVER), (
            f"the keyword must override the colormap's own over colour; got {drawn}"
        )


class TestTheLayerPublishesWhatItWasDrawnWith:
    """The figure's description carries the three colours, so another tier reads the same semantics."""

    def test_the_published_scale_states_the_colours(self):
        """The layer's colour encoding carries them on its scale."""
        with Map(globe=False) as canvas:
            canvas.field(WIDE, name="grid", vmin=2.0, vmax=6.0, over=OVER, under=UNDER)
            scale = (
                canvas.figure_spec.layers.get("grid").symbology.encoding("color").scale
            )
        assert scale.extremes() == {"over": OVER, "under": UNDER}, (
            f"the published scale must state what was drawn; got {scale.extremes()}"
        )

    def test_the_published_scale_keeps_the_limits_it_was_drawn_at(self):
        """Stating the extremes does not disturb the domain the key is labelled with."""
        with Map(globe=False) as canvas:
            canvas.field(WIDE, name="grid", vmin=2.0, vmax=6.0, over=OVER)
            scale = (
                canvas.figure_spec.layers.get("grid").symbology.encoding("color").scale
            )
        assert scale.as_limits() == (2.0, 6.0), (
            f"the published domain must be the drawn one; got {scale.as_limits()}"
        )

    def test_a_layer_that_states_none_publishes_none(self):
        """An unstated figure publishes an unstated scale, so nothing is invented."""
        with Map(globe=False) as canvas:
            canvas.field(WIDE, name="grid", vmin=2.0, vmax=6.0)
            scale = (
                canvas.figure_spec.layers.get("grid").symbology.encoding("color").scale
            )
        assert scale.extremes() == {}, (
            f"an unstated layer must publish no extreme colours; got {scale.extremes()}"
        )
