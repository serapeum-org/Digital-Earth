"""The static tier's declared style vocabulary, and the one place it folds (DE-15, #274).

26 style keys reached cleopatra through `**kwargs` with no signature anywhere: a caller could not ask what a
builder accepts, and a typo was a silently ignored keyword rather than an error. These cover the declaration
that replaces that, and the route/fold pair that is now the only translation between a declared style and the
flat form the glyphs take.

There are 27 today: `samples` joined them with #302, which is the count below and the reason for it.
"""

import pytest

from digitalearth.base.spec import Encoding, Symbology
from digitalearth.static.render_compat import (
    FLAT_STYLE_KEYS,
    MARKER_SIZE_KEY,
    STATIC_STYLE_SCHEMA,
    fold_symbology,
    relocate_flat_style,
    route_flat_style,
)


class TestTheDeclaredSchema:
    """The style keys that appeared in no signature anywhere, written down."""

    def test_every_key_cleopatra_rejects_is_declared(self):
        """The declaration covers the whole flat surface, plus the marker-size spelling.

        Test scenario:
            This is what stops the keys going undeclared again. A key added upstream — a new `ColorScaling`
            field, say — lands in `FLAT_STYLE_KEYS` automatically but would carry no description and no
            channel, so it has to fail here rather than quietly rejoin the soup.
        """
        assert set(STATIC_STYLE_SCHEMA.names()) == FLAT_STYLE_KEYS | {
            MARKER_SIZE_KEY
        }, "every flat style key must be declared, and nothing else"

    def test_the_undeclared_count_is_the_one_the_plan_names(self):
        """27 flat members, beside the 6 typed group parameters they fold into.

        Test scenario:
            The number the refactor plan and #274 are written against was 26. `samples` — the `equalize`
            scale's resolution — is the 27th: it was a `ColorScaling` field with no flat key here, so
            `color_scale="equalize", samples=64` reached cleopatra as a stray keyword and raised. #302
            declared it. Pinning the count lets a later reader check the claim rather than take it on trust.
        """
        group_params = {"color", "contour", "data_style", "classify", "cells", "points"}
        assert len(FLAT_STYLE_KEYS - group_params) == 27, (
            "the flat members are the 26 keys DE-15 exists to declare, plus #302's `samples`"
        )

    def test_every_key_says_what_it_controls(self):
        """A declaration with no description is not much of one.

        Test scenario:
            `doc` is the text a discovery surface shows. An empty one lists a key a caller still cannot use.
        """
        undocumented = [
            key.name for key in STATIC_STYLE_SCHEMA.keys.values() if not key.doc.strip()
        ]
        assert not undocumented, (
            f"these keys are declared without a description: {undocumented}"
        )

    def test_a_typo_can_now_be_answered(self):
        """The thing a schema buys: a misspelt key has a nearest match.

        Test scenario:
            `hilshade=True` used to be a silently ignored keyword and an unshaded map. Nothing raises on it
            yet — that is `IN-13` — but the answer now exists to raise with.
        """
        assert STATIC_STYLE_SCHEMA.suggest("hilshade") == "hillshade", (
            "a near miss on a declared key must be nameable"
        )


class TestRoutingAndFolding:
    """`render_compat` is the only place a declared style becomes cleopatra's flat form."""

    def test_flat_kwargs_route_to_channels_and_properties(self):
        """A declared keyword finds its home; everything else comes back.

        Test scenario:
            `alpha` is layer opacity, `hillshade` is a static property, and `cmap` is not styling at all —
            it is a constructor option the glyph still takes. All three travel in one kwargs dict.
        """
        sym, rest = route_flat_style(
            {"alpha": 0.4, "hillshade": True, "cmap": "viridis"}
        )
        assert sym.encoding("opacity").resolve() == 0.4, (
            "alpha must drive the opacity channel"
        )
        assert dict(sym.props) == {"hillshade": True}, "hillshade is a static property"
        assert rest == {"cmap": "viridis"}, "a non-styling key is handed back untouched"

    def test_routing_and_folding_round_trip(self):
        """What routes in folds back out unchanged.

        Test scenario:
            The two halves are the seam Wave 3's `LayerSpec` plugs into. If they disagreed, a style would
            change meaning just by being described.
        """
        flat = {"alpha": 0.4, "scheme": "quantiles", "k": 4, "hillshade": True}
        sym, rest = route_flat_style(flat)
        folded, unsupported = fold_symbology(sym)
        assert not rest, "this style is entirely expressible"
        assert not unsupported, "this style is entirely expressible"
        assert folded == flat, f"the fold must reproduce what was routed; got {folded}"

    def test_a_channel_the_flat_surface_cannot_express_is_reported_not_raised(self):
        """An unsupported channel comes back as a value, so a caller can degrade.

        Test scenario:
            `height` is declared because #199 asks for extrusion, and the static tier has no keyword for it.
            Raising would make a shared symbology unusable on matplotlib rather than merely flatter.
        """
        folded, unsupported = fold_symbology(Symbology.of(height=30.0, opacity=0.5))
        assert "height" in unsupported, "the channel with no keyword must be reported"
        assert folded == {"alpha": 0.5}, "and the rest must still fold"

    def test_a_data_driven_channel_is_reported_with_its_field(self):
        """The flat kwargs take a constant; a field-driven channel says so by name.

        Test scenario:
            cleopatra's glyphs colour a `values=` array themselves, so resolving a data-driven channel is the
            builder's job, not the fold's. The message has to name the field or the caller cannot tell which
            of their columns went unused.
        """
        driven = Symbology.of(opacity=Encoding.by_field("opacity", "confidence"))
        _, unsupported = fold_symbology(driven)
        assert "confidence" in unsupported["opacity"], (
            f"the reason must name the field; got {unsupported['opacity']!r}"
        )

    def test_the_color_key_is_the_scaling_group_not_the_colour_channel(self):
        """`color` names two different things, and the schema is explicit about which one it declares.

        Test scenario:
            cleopatra's `plot(color=...)` takes a ColorScaling group — how values are *scaled* onto a ramp —
            while `CHANNELS` declares a `color` channel for the colour itself. The names collide and cannot
            both change: the flat key has to match cleopatra's kwarg. So the collision is pinned here rather
            than left for a reader to trip over, along with the fact that the static tier has no flat keyword
            for a constant layer colour at all — colour comes from `cmap` plus the data values.
        """
        assert STATIC_STYLE_SCHEMA.keys["color"].channel is None, (
            "the flat color key is the scaling group, so it must declare no channel"
        )
        sym, rest = route_flat_style({"color": "a-colorscaling-object"})
        assert dict(sym.props) == {"color": "a-colorscaling-object"}, (
            "it must route as a static property, not as the colour channel"
        )
        assert not sym.encodings, "and nowhere else"
        assert not rest, "and nowhere else"

    def test_asking_to_fold_the_colour_channel_explains_why_it_cannot(self):
        """The unsupported reason names the actual reason, not just the absence.

        Test scenario:
            "no keyword for 'color'" on the one channel the vocabulary calls first-class reads as an
            oversight. It is not: a matplotlib layer is coloured by a colormap over its values, and the flat
            `color=` key is a different concept wearing the same name.
        """
        _, unsupported = fold_symbology(Symbology.of(color="#f00"))
        assert "cmap" in unsupported["color"], (
            f"the reason must explain the two-way collision; got {unsupported['color']!r}"
        )
        assert "ColorScaling" in unsupported["color"], (
            f"the reason must explain the two-way collision; got {unsupported['color']!r}"
        )


class TestTheMarkerSizeChannel:
    """The channel beyond colour, folded here instead of in the builders."""

    def test_the_size_channel_folds_onto_the_constructor_spelling(self):
        """`size=` becomes the `point_size` a cleopatra point glyph takes.

        Test scenario:
            This decision used to live in `static/maps/vector.py`, called from two builders. It is a
            style-key fold like the other 33, so moving it here leaves no builder holding a mapping of its
            own.
        """
        opts = {"size": 12, "cmap": "viridis"}
        plot_style = relocate_flat_style(opts, marker_size_for="Map.scatter()")
        assert opts == {"cmap": "viridis", "point_size": 12}, (
            "the size channel must reach the constructor as point_size"
        )
        assert "point_size" not in plot_style, (
            "and must not also travel to plot(), where it would style a point overlay"
        )

    def test_the_older_spelling_still_works_and_warns(self):
        """`point_size=` keeps working for one release.

        Test scenario:
            The rename promise the package keeps everywhere: the old spelling works, warns, and names the
            method the user actually called.
        """
        opts = {"point_size": 9}
        with pytest.warns(DeprecationWarning, match="Map.scatter"):
            relocate_flat_style(opts, marker_size_for="Map.scatter()")
        assert opts == {"point_size": 9}, "the old spelling must still reach the glyph"

    def test_both_spellings_at_once_is_refused(self):
        """A contradictory call is an error rather than a coin flip.

        Test scenario:
            Picking one silently would draw markers at a size the caller did not ask for, with no signal.
        """
        with pytest.raises(TypeError):
            relocate_flat_style(
                {"size": 12, "point_size": 9}, marker_size_for="Map.scatter()"
            )

    def test_a_glyph_that_is_not_a_point_glyph_keeps_the_old_behaviour(self):
        """Without `marker_size_for`, `point_size` stays on its way to the point overlay.

        Test scenario:
            A raster field's `point_size` styles the markers overlaid on it, not the field — so relocating it
            to the constructor would move it onto the wrong artist.
        """
        opts = {"point_size": 9, "points": [(0, 0)]}
        plot_style = relocate_flat_style(opts)
        assert plot_style["point_size"] == 9, (
            "the overlay's marker size must reach plot()"
        )
        assert opts == {}, "and nothing styling must be left on the constructor"
