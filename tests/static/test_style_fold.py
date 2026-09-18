"""The colour group, built by cleopatra's own builder rather than by a copy of it (DE-43, #302).

`render_compat` mapped six flat keys onto `ColorScaling` fields by hand, and the copy had drifted: the
`equalize` scale's `samples` had no flat key, so `Map.imshow(ds, color_scale="equalize", samples=64)` raised
cleopatra's *"the 'samples' option moved onto a grouped parameter object"* — advice with no way to follow it
from here. `ColorScaling.from_options` is the builder that does not drift.

These cover the three rules that are this tier's own: the friendly spellings survive the swap, no colour
keyword still means no colour group (so a caller's `norm=` is not silently overridden), and a built group
beside a flat key is refused rather than forwarded.
"""

import matplotlib
import pytest
from cleopatra.styling.scaling import ColorScale, ColorScaling

from digitalearth.static.render_compat import group_render_kwargs
from digitalearth.static.style_fold import (
    COLOR_SCALE_ALIASES,
    coerce_color_scale,
    fold_color_scaling,
)

RASTER = "examples/data/acc4000.tif"

#: One knob per `ColorScaling` field a flat keyword reaches, beside the group field it lands on. `midpoint`
#: is the flat spelling of `center`; `samples` is covered on its own, because it is the one answer the fold
#: deliberately changes.
KNOBS = {
    "gamma": (0.3, "gamma"),
    "line_threshold": (2.0, "line_threshold"),
    "line_scale": (0.5, "line_scale"),
    "bounds": ([0.0, 1.0, 2.0], "bounds"),
    "midpoint": (4.0, "center"),
}


@pytest.fixture(scope="module")
def raster():
    """Return the small accumulation raster the render tests draw.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(RASTER)


class TestTheFriendlySpellings:
    """`sym_log` and `boundary` are this tier's documented names; upstream knows neither."""

    @pytest.mark.parametrize(
        "written, expected",
        [
            ("linear", ColorScale.LINEAR),
            ("power", ColorScale.POWER),
            ("lognorm", ColorScale.LOGNORM),
            ("sym_log", ColorScale.SYM_LOGNORM),
            ("symlog", ColorScale.SYM_LOGNORM),
            ("sym_lognorm", ColorScale.SYM_LOGNORM),
            ("sym-lognorm", ColorScale.SYM_LOGNORM),
            ("boundary", ColorScale.BOUNDARY_NORM),
            ("boundary_norm", ColorScale.BOUNDARY_NORM),
            ("MidPoint", ColorScale.MIDPOINT),
            ("equalize", ColorScale.EQUALIZE),
            (ColorScale.POWER, ColorScale.POWER),
        ],
    )
    def test_a_spelling_reaches_the_scale_it_names(self, written, expected):
        """Every alias, the exact enum values, a mixed case and a member itself.

        Args:
            written: What a caller wrote.
            expected: The scale it names.

        Test scenario:
            `from_options` validates through `ColorScale(raw)`, which ignores case but does not map `_` to
            `-`. Without this step five of these spellings would start raising.
        """
        assert coerce_color_scale(written) is expected, written

    def test_an_unknown_scale_is_refused_with_the_spellings_there_are(self):
        """A misspelling is answered with the vocabulary, not with a render-time attribute error."""
        with pytest.raises(ValueError, match="not a recognised colour scale"):
            coerce_color_scale("log")

    def test_the_newest_scale_is_among_them(self):
        """`equalize` was missing from the list the error prints, and from the schema's description."""
        with pytest.raises(ValueError, match="equalize"):
            coerce_color_scale("nope")


class TestTheGroupIsBuiltUpstream:
    """The builder's answer is the hand-rolled fold's answer, knob for knob."""

    @pytest.mark.parametrize("written, scale", sorted(COLOR_SCALE_ALIASES.items()))
    @pytest.mark.parametrize("knob", sorted(KNOBS))
    def test_every_spelling_against_every_knob_folds_as_it_did(
        self, written, knob, scale
    ):
        """The group the fold builds is the group the field map used to build.

        Args:
            written: The `color_scale=` spelling.
            knob: The flat keyword passed beside it.
            scale: The scale that spelling names.

        Test scenario:
            The old fold called `ColorScaling(**members)` with only the members the caller passed, and the
            dataclass filled the rest. `from_options` fills them from `_SCALE_DEFAULTS` instead — the same
            values, but from a second table, which is exactly the kind of agreement worth pinning.
        """
        value, field = KNOBS[knob]
        folded = group_render_kwargs({"color_scale": written, knob: value})
        assert folded["color"] == ColorScaling(kind=scale, **{field: value}), folded[
            "color"
        ]

    def test_a_knob_on_its_own_still_builds_a_group(self):
        """`gamma=` with no `color_scale=` is a colour keyword, so it folds — onto the default linear scale."""
        folded = group_render_kwargs({"gamma": 0.2})
        assert folded["color"] == ColorScaling(gamma=0.2), folded["color"]

    def test_the_equalize_resolution_is_the_one_answer_that_changes(self):
        """`samples` had no flat key, so the value a caller passed never reached the group."""
        folded = group_render_kwargs({"color_scale": "equalize", "samples": 64})
        assert folded["color"].samples == 64, folded["color"]

    def test_nothing_about_colour_builds_no_group(self):
        """`from_options({})` is a *linear* scale, and passing one would override the caller's own norm."""
        folded = group_render_kwargs({"cmap": "viridis", "add_colorbar": False})
        assert "color" not in folded, folded

    def test_a_built_group_passes_through_untouched(self):
        """A caller who built the group themselves gets the object they built, not a copy of it."""
        built = ColorScaling.power(gamma=0.7)
        folded = group_render_kwargs({"color": built})
        assert folded["color"] is built, folded["color"]

    def test_a_built_group_beside_a_flat_key_is_refused(self):
        """Both spellings style the same thing, and forwarding them produced advice already followed."""
        with pytest.raises(ValueError, match="style the same thing"):
            fold_color_scaling({"color": ColorScaling.power(), "gamma": 0.3})

    def test_the_refusal_names_both_spellings(self):
        """An error a caller can act on says which two keywords to choose between."""
        with pytest.raises(ValueError) as raised:
            fold_color_scaling({"color": ColorScaling.power(), "midpoint": 2.0})
        message = str(raised.value)
        assert "color=" in message, message
        assert "midpoint" in message, message

    def test_a_glyph_without_a_colour_parameter_leaves_the_keys_alone(self):
        """`accepted=` is what keeps a group off a glyph whose `plot()` cannot take it."""
        folded = group_render_kwargs({"gamma": 0.2}, accepted={"contour"})
        assert folded == {"gamma": 0.2}, folded


class TestWhatIsDrawn:
    """The two renders the fold changes, and the one it must not."""

    def test_an_equalized_field_is_drawn_at_the_resolution_asked_for(self, raster):
        """The keyword reaches the glyph rather than raising on the way.

        Args:
            raster: The accumulation raster.
        """
        from digitalearth.static import Map

        with Map() as canvas:
            canvas.imshow(raster, color_scale="equalize", samples=64)
            glyph = canvas.layers[-1][0]
        assert glyph.default_options["samples"] == 64, glyph.default_options["samples"]

    def test_a_caller_s_own_norm_is_still_in_force(self, raster):
        """The reason an empty colour group is not built: `to_options()` resets `norm` to `None`.

        Args:
            raster: The accumulation raster.
        """
        from digitalearth.static import Map

        with Map() as canvas:
            canvas.imshow(raster, norm=matplotlib.colors.LogNorm())
            drawn = canvas.layers[-1][1].norm
        assert isinstance(drawn, matplotlib.colors.LogNorm), type(drawn).__name__
