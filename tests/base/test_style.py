"""`Symbology` and `StyleSchema` — declaring the style keys instead of passing soup (DE-15, #274).

26 style keys reached the renderers through `**kwargs` with no signature anywhere: undiscoverable, and a typo
in one was a silently ignored keyword rather than an error. These cover the declaration and the routing that
replaces them.
"""

import pytest

from digitalearth.base.spec import Encoding, Scale, StyleKey, StyleSchema, Symbology


class TestSymbology:
    """The look of one layer, as a value."""

    def test_channels_are_reachable_by_name(self):
        """A renderer asks what drives a channel, rather than searching keywords.

        Test scenario:
            The one question a renderer has. Answering it by scanning a kwargs dict for whichever spelling
            that builder used is the thing being replaced.
        """
        sym = Symbology.of(color="#f00", opacity=0.5)
        assert sym.encoding("color").resolve() == "#f00", (
            "the channel must be reachable"
        )
        assert sym.encoding("width") is None, (
            "an unstyled channel answers None, so the renderer uses its own default"
        )

    def test_an_encoding_filed_under_the_wrong_channel_is_refused(self):
        """The key and the encoding's channel must agree.

        Test scenario:
            A mismatch makes `encoding("color")` answer with something that styles size — which would draw
            silently, in the wrong channel.
        """
        with pytest.raises(ValueError, match="drives channel"):
            Symbology(encodings={"color": Encoding.constant("size", 6)})

    def test_a_field_driven_encoding_can_be_passed_whole(self):
        """`of()` takes an already-built encoding as well as a constant.

        Test scenario:
            Otherwise the convenient constructor would be the constants-only one and every data-driven layer
            would build the dict by hand.
        """
        sym = Symbology.of(color=Encoding.by_field("color", "class"))
        assert sym.encoding("color").field == "class", (
            "a built encoding must be kept as given"
        )

    def test_the_mappings_cannot_be_edited_after_construction(self):
        """Frozen means frozen, including the dicts passed in.

        Test scenario:
            A frozen dataclass still holds a mutable `dict`; a caller keeping a reference to what they passed
            could edit a symbology a renderer already holds.
        """
        passed = {"hillshade": True}
        sym = Symbology(props=passed)
        passed["hillshade"] = False
        assert sym.props["hillshade"] is True, (
            "the symbology must not track the caller's dict"
        )
        with pytest.raises(TypeError):
            sym.props["hillshade"] = False

    def test_a_symbology_can_key_a_cache(self):
        """`Symbology` hashes when its properties do.

        Test scenario:
            `Bounds`, `Scale`, `Selection` and `Encoding` all hash. A symbology that did not would be the odd
            one out the moment a cache or a set is keyed on a layer's style.
        """
        assert len({Symbology.of(color="#f00"), Symbology.of(color="#f00")}) == 1, (
            "two equal symbologies must collapse to one entry"
        )

    def test_a_property_value_is_not_deep_frozen(self):
        """Only the mapping is frozen; what a caller put in it stays theirs.

        Test scenario:
            Pinned deliberately rather than claimed away. Deep-freezing arbitrary style values is not
            realistic, so the docstring states the limit and this is the behaviour it states.
        """
        levels = [1.0, 2.0]
        sym = Symbology().with_props(levels=levels)
        levels.append(3.0)
        assert sym.props["levels"] == [1.0, 2.0, 3.0], (
            "a mutable property value is shared with the caller, by documented design"
        )


class TestMerging:
    """Defaults, themes and layer-kind styling, laid under what a caller asked for."""

    def test_the_caller_wins_per_channel(self):
        """Naming one channel does not discard the defaults for the others.

        Test scenario:
            Merging whole objects instead of per key is the bug this shape prevents: a caller who sets a
            colour would silently lose the theme's opacity.
        """
        merged = Symbology.of(color="#f00").merged_over(
            Symbology.of(color="#000", opacity=0.8)
        )
        assert merged.encoding("color").resolve() == "#f00", "the caller's channel wins"
        assert merged.encoding("opacity").resolve() == 0.8, "the default's survives"

    def test_properties_merge_the_same_way(self):
        """Static properties follow the same per-key rule as channels.

        Test scenario:
            Two merge rules in one type would be a coin flip at every call site.
        """
        merged = (
            Symbology()
            .with_props(k=7)
            .merged_over(Symbology().with_props(k=5, scheme="quantiles"))
        )
        assert dict(merged.props) == {"k": 7, "scheme": "quantiles"}, (
            "properties must merge key by key"
        )

    def test_merging_leaves_both_operands_alone(self):
        """A merge returns a new value.

        Test scenario:
            A theme is shared between layers; merging a caller's style over it must not edit the theme every
            other layer is about to read.
        """
        theme = Symbology.of(color="#000")
        Symbology.of(color="#f00").merged_over(theme)
        assert theme.encoding("color").resolve() == "#000", (
            "the defaults must be unchanged"
        )


class TestTheSchema:
    """What a builder accepts, written down."""

    def test_a_channel_key_becomes_an_encoding_and_a_plain_key_a_property(self):
        """Routing puts each declared keyword where it belongs.

        Test scenario:
            The half `prepare_plot_kwargs` never had: it can reject a keyword, but it cannot say which
            channel owns one.
        """
        schema = StyleSchema.of(
            StyleKey("alpha", "Opacity.", channel="opacity"),
            StyleKey("hillshade", "Shade the surface."),
        )
        sym, leftover = schema.route({"alpha": 0.5, "hillshade": True})
        assert sym.encoding("opacity").resolve() == 0.5, (
            "a channel key must become an encoding"
        )
        assert dict(sym.props) == {"hillshade": True}, (
            "a plain key must become a property"
        )
        assert leftover == {}, "nothing declared should be left over"

    def test_an_undeclared_key_is_handed_back_rather_than_swallowed(self):
        """Routing returns leftovers; it does not decide they are errors.

        Test scenario:
            The styling schema is not everything a builder accepts — `cmap`, `add_colorbar` and the glyph's
            own constructor options travel in the same dict. Raising here would refuse valid calls.
        """
        schema = StyleSchema.of(StyleKey("alpha", "Opacity.", channel="opacity"))
        _, leftover = schema.route({"alpha": 0.5, "cmap": "viridis"})
        assert leftover == {"cmap": "viridis"}, (
            "an unknown key must come back to the caller"
        )

    def test_a_declined_key_is_not_a_binding(self):
        """`None` means the caller did not ask, not "bind this channel to nothing".

        Test scenario:
            Builders pass their own defaults through as `None`; turning those into encodings would make every
            layer declare every channel and override the renderer's defaults with nulls.
        """
        schema = StyleSchema.of(StyleKey("alpha", "Opacity.", channel="opacity"))
        sym, leftover = schema.route({"alpha": None})
        assert not sym.encodings and not leftover, "a None must route nowhere"

    def test_the_nearest_declared_key_is_offered_for_a_typo(self):
        """A misspelling can be answered instead of silently ignored.

        Test scenario:
            This is the mechanism `IN-13` ("option validation with did-you-mean") is waiting on; without a
            schema there was nothing to compare a name against.
        """
        schema = StyleSchema.of(StyleKey("hillshade", "Shade the surface."))
        assert schema.suggest("hillshde") == "hillshade", "a near miss must be named"

    def test_something_unlike_every_key_is_not_guessed_at(self):
        """No suggestion beats a wrong one.

        Test scenario:
            Pointing a caller at a key that was never their intent costs more than saying nothing — they will
            go and read the wrong documentation.
        """
        schema = StyleSchema.of(StyleKey("hillshade", "Shade the surface."))
        assert schema.suggest("zzzzzz") is None, "a distant name must not be guessed at"

    def test_a_key_naming_an_undeclared_channel_is_refused(self):
        """The declaration is checked against the channel table when it is written.

        Test scenario:
            A key pointing at a channel that does not exist routes into a symbology nothing can read; failing
            at import is the only time anyone would notice.
        """
        with pytest.raises(ValueError, match="not declared"):
            StyleKey("alpha", "Opacity.", channel="opacty")

    def test_declaring_a_key_twice_is_refused(self):
        """Two rows for one keyword means one of them is dead.

        Test scenario:
            Which wins would depend on the order the rows happen to be written in, and the losing row's
            documentation would still be shown.
        """
        with pytest.raises(ValueError, match="declared twice"):
            StyleSchema.of(StyleKey("alpha", "One."), StyleKey("alpha", "Two."))

    def test_the_schema_answers_what_a_builder_accepts(self):
        """`names()` is the discoverability that did not exist.

        Test scenario:
            The question a caller — and any tooling — could not ask before, because the keys appeared in no
            signature anywhere.
        """
        schema = StyleSchema.of(
            StyleKey("alpha", "Opacity.", channel="opacity"),
            StyleKey("hillshade", "Shade the surface."),
        )
        assert schema.names() == ("alpha", "hillshade"), (
            "every declared key must be listed"
        )
        assert schema.channels() == ("opacity",), "and the channels it can drive"


class TestRouteAndResolveTogether:
    """The two types in the shape a layer will use them."""

    def test_a_routed_style_resolves_per_datum(self):
        """Flat kwargs in, per-feature visual values out, with nothing engine-specific in between.

        Test scenario:
            The end-to-end claim of DE-15: a caller's keyword becomes a declared channel, and that channel
            answers for each feature — without any tier having written a mapping of its own.
        """
        schema = StyleSchema.of(StyleKey("alpha", "Opacity.", channel="opacity"))
        sym, _ = schema.route({"alpha": 0.25})
        assert sym.encoding("opacity").resolve([1, 2]) == [0.25, 0.25], (
            "a routed constant must resolve for every datum"
        )

    def test_a_field_driven_channel_survives_a_merge(self):
        """Merging keeps the encoding whole, scale and all.

        Test scenario:
            A theme laid under a data-driven colour must not flatten it back to a constant.
        """
        driven = Symbology.of(
            color=Encoding.by_field("color", "v", scale=Scale.from_limits(0.0, 4.0))
        )
        merged = driven.merged_over(Symbology.of(color="#000", size=6))
        assert merged.encoding("color").resolve([2.0]) == [0.5], (
            "the data-driven colour must survive"
        )
        assert merged.encoding("size").resolve() == 6, (
            "and the default size fill the gap"
        )
