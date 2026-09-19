"""Folding a `Symbology` into HoloViews options, checked against the engine (IN-MC, #298).

One fold existed — static's — and nothing called it. The interactive tier spelled style in HoloViews' keywords
at nineteen call sites, and `.opts()` checked them only after the element was built, only in HoloViews' names,
and across whichever backends happened to be registered. These cover the fold that replaces that, and they are
deliberately checked against the installed HoloViews rather than against a table of what it once accepted.
"""

import subprocess
import sys

import pytest

from digitalearth.base.spec import CHANNELS, Encoding, Scale, Symbology

pytest.importorskip("holoviews")
# Imported for its side effect: geoviews registers WMTS and Feature into holoviews' option store, and the
# tier draws both, so the table is checked against a store that knows them.
pytest.importorskip("geoviews")

import holoviews as hv  # noqa: E402

from digitalearth.interactive.style_fold import (  # noqa: E402
    CHANNEL_OPTIONS,
    INTERACTIVE_STYLE_SCHEMA,
    UNEXPRESSIBLE,
    allowed_options,
    fold_symbology,
    route_flat_style,
)

#: Every element type the tier's builders emit, as HoloViews names them.
BUILT_ELEMENTS = (
    "Image",
    "RGB",
    "QuadMesh",
    "Points",
    "Path",
    "Polygons",
    "Contours",
    "HexTiles",
    "VectorField",
    "Rectangles",
    "Labels",
    "WMTS",
    "TriMesh",
)


class TestTheTableIsPinnedToTheEngine:
    """The channel table is a claim about HoloViews, so HoloViews is asked."""

    @pytest.mark.parametrize("entry", CHANNEL_OPTIONS, ids=lambda e: e.channel)
    def test_each_channel_folds_to_an_option_its_elements_accept(self, entry):
        """A channel the table says an element takes is one the engine accepts, in the group named.

        Args:
            entry: The channel's row in the table.
        """
        for element in sorted(entry.elements):
            options = allowed_options(element)
            assert entry.option in options[entry.group], (
                f"{element} does not take {entry.option!r} as a {entry.group} option"
            )

    @pytest.mark.parametrize("entry", CHANNEL_OPTIONS, ids=lambda e: e.channel)
    def test_each_channel_is_absent_where_the_table_says_so(self, entry):
        """An element the table leaves out is one that really cannot take the option.

        Args:
            entry: The channel's row in the table.

        Test scenario:
            The half that rots silently: a table that under-claims makes a tier look less capable than it is,
            and nothing but the engine can say which side is right.
        """
        missing = [
            element
            for element in BUILT_ELEMENTS
            if element not in entry.elements
            and entry.option in allowed_options(element)[entry.group]
        ]
        assert missing == [], (
            f"{entry.channel!r} is declared unavailable on {missing}, which do take {entry.option!r}"
        )

    def test_every_channel_is_either_folded_or_explained(self):
        """No channel is simply forgotten: it folds, or it says why it cannot."""
        covered = {entry.channel for entry in CHANNEL_OPTIONS} | set(UNEXPRESSIBLE)
        assert covered == set(CHANNELS), (
            f"channels neither folded nor explained: {sorted(set(CHANNELS) - covered)}"
        )

    def test_every_declared_keyword_is_one_the_engine_knows(self):
        """A schema keyword nothing accepts would be a promise the tier cannot keep."""
        somewhere = set()
        for element in BUILT_ELEMENTS:
            for keys in allowed_options(element).values():
                somewhere |= set(keys)
        unknown = [
            key for key in INTERACTIVE_STYLE_SCHEMA.names() if key not in somewhere
        ]
        assert unknown == [], f"declared keywords no element accepts: {unknown}"


class TestFoldingAChannel:
    """What a declared style becomes for one element."""

    def test_a_constant_folds_onto_the_element_s_own_keyword(self):
        """Opacity and size are `alpha` and `size` on a point layer."""
        grouped, unsupported = fold_symbology(
            Symbology.of(opacity=0.5, size=6), "Points"
        )
        assert grouped["style"] == {"alpha": 0.5, "size": 6}, grouped
        assert unsupported == {}, unsupported

    def test_a_channel_the_element_cannot_express_is_reported(self):
        """A polygon has no marker size, and the reason says why rather than dropping it."""
        _, unsupported = fold_symbology(Symbology.of(opacity=0.5, size=6), "Polygons")
        assert "marker" in unsupported["size"], unsupported

    def test_a_field_driven_colour_becomes_a_column_and_its_limits(self):
        """HoloViews colours by a dimension, so the field is the option and the scale is the range."""
        encoding = Encoding.by_field(
            "color", "pop", scale=Scale.from_limits(0.0, 100.0)
        )
        grouped, _ = fold_symbology(Symbology.of(color=encoding), "Points")
        assert grouped["style"]["color"] == "pop", grouped
        assert grouped["plot"]["clim"] == (0.0, 100.0), grouped

    def test_a_classified_colour_adds_its_class_edges(self):
        """A scheme's breaks are the `color_levels` Bokeh draws the classes with."""
        scale = Scale(0.0, 10.0, scheme="quantiles", breaks=(0.0, 4.0, 10.0))
        encoding = Encoding.by_field("color", "pop", scale=scale)
        grouped, _ = fold_symbology(Symbology.of(color=encoding), "Polygons")
        assert grouped["plot"]["color_levels"] == [0.0, 4.0, 10.0], grouped

    def test_a_colour_field_with_no_scale_folds_to_the_column_alone(self):
        """A column with no scale is a colour HoloViews ranges itself, so no limits are written."""
        encoding = Encoding.by_field("color", "landcover")
        grouped, _ = fold_symbology(Symbology.of(color=encoding), "Points")
        assert grouped["style"]["color"] == "landcover", grouped
        assert "clim" not in grouped["plot"], grouped

    def test_a_field_driven_size_is_refused_rather_than_guessed(self):
        """HoloViews takes a number for a size; a column would silently draw one marker size."""
        driven = Symbology.of(size=Encoding.by_field("size", "pop"))
        with pytest.raises(ValueError, match="resolve it and pass the values"):
            fold_symbology(driven, "Points")

    def test_a_tooltip_becomes_the_hover_tool(self):
        """Per-layer interaction is a channel (#292), and on this tier it is a Bokeh tool."""
        grouped, _ = fold_symbology(Symbology.of(tooltip=("pop",)), "Points")
        assert grouped["plot"]["tools"] == ["hover"], grouped

    @pytest.mark.parametrize("channel", sorted(UNEXPRESSIBLE))
    def test_a_channel_holoviews_has_no_option_for_is_explained(self, channel):
        """`height` and `text` are the two the vocabulary carries and this engine cannot style.

        Args:
            channel: The channel under test.
        """
        _, unsupported = fold_symbology(
            Symbology.of(**{channel: 3.0 if channel == "height" else "label"}), "Points"
        )
        assert channel in unsupported, unsupported


class TestCheckingAgainstTheEngine:
    """Every key the fold produces is checked before anything is built."""

    def test_a_raw_option_the_element_takes_passes_through(self):
        """A caller's own HoloViews option is not in the way of the fold."""
        grouped, _ = fold_symbology(Symbology(props={"cmap": "magma"}), "Image")
        assert grouped["style"]["cmap"] == "magma", grouped

    def test_a_plot_option_lands_in_the_plot_group(self):
        """One layer's style spans two groups, and HoloViews validates them separately."""
        grouped, _ = fold_symbology(Symbology(props={"colorbar": True}), "Image")
        assert grouped["plot"]["colorbar"] is True, grouped

    def test_an_option_of_another_backend_is_refused(self):
        """`interpolation` is matplotlib's; on Bokeh it was accepted and then ignored.

        Test scenario:
            The measured defect: `_styled` called `.opts()` with no backend, so once matplotlib was registered
            a matplotlib-only option passed the check and drew nothing.
        """
        matplotlib_only = Symbology(props={"interpolation": "nearest"})
        with pytest.raises(ValueError, match="interpolation"):
            fold_symbology(matplotlib_only, "Image")

    def test_the_refusal_suggests_what_was_meant(self):
        """The did-you-mean `.opts()` gives, but before an element exists."""
        misspelt = Symbology(props={"cmpa": "viridis"})
        with pytest.raises(ValueError, match=r"did you mean \['cmap'\]"):
            fold_symbology(misspelt, "Image")

    def test_an_unknown_element_is_refused_by_name(self):
        """A typo in the element type is answered, not raised from inside HoloViews."""
        with pytest.raises(KeyError, match="no element type 'Imgae'"):
            allowed_options("Imgae")


class TestRoutingFlatKeywords:
    """The builders' own keywords, split into channels and engine options."""

    def test_a_channel_keyword_becomes_an_encoding(self):
        """`alpha=` is the opacity channel, whatever HoloViews calls it."""
        symbology, leftover = route_flat_style({"alpha": 0.4, "cmap": "viridis"})
        assert symbology.encoding("opacity").resolve() == 0.4, symbology
        assert dict(symbology.props) == {"cmap": "viridis"}, symbology.props
        assert leftover == {}, "a declared keyword is not left over"

    def test_a_misspelt_keyword_is_refused_with_a_suggestion(self):
        """A keyword nothing declares is refused when it is folded, before an element is built.

        Test scenario:
            Routing cannot tell a misspelling from a raw HoloViews option — only the element can — so the
            keyword is carried to the fold, which names both what the engine accepts and what the tier
            declares.
        """
        symbology, _ = route_flat_style({"alhpa": 0.4})
        with pytest.raises(ValueError, match=r"'alhpa'.*did you mean \['alpha'\]"):
            fold_symbology(symbology, "Points")

    @pytest.mark.parametrize(
        "keyword",
        [
            "cmap",
            "clim",
            "alpha",
            "colorbar",
            "clabel",
            "color",
            "color_levels",
            "size",
            "fill_alpha",
            "line_color",
            "edge_color",
            "edge_cmap",
            "tools",
        ],
    )
    def test_every_keyword_the_builders_write_is_declared(self, keyword):
        """A keyword in use today must route, or #300 could not swap `_styled` out.

        Args:
            keyword: The builder keyword under test.
        """
        assert keyword in INTERACTIVE_STYLE_SCHEMA.names(), (
            f"{keyword!r} is written by a builder but not declared"
        )


class TestTheImportCost:
    """The schema is read to answer what the tier accepts, which must not load the engine."""

    def test_the_module_loads_without_holoviews(self):
        """A fresh interpreter imports the fold and no HoloViews with it.

        Test scenario:
            Run in a subprocess, since this session has HoloViews loaded already.
        """
        code = (
            "import sys; import digitalearth.interactive.style_fold as fold;"
            "print('holoviews' in sys.modules, len(fold.CHANNEL_OPTIONS) > 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False True", result.stdout or result.stderr


class TestTheOtherOptionGroups:
    """Review M15/M16 — a read stays a read, and a standard option gets an answer."""

    @pytest.mark.parametrize("option", ["framewise", "axiswise"])
    def test_a_norm_group_option_is_folded_rather_than_raised(self, option):
        """HoloViews sorts these under `norm`, and the fold only had `style` and `plot` to put them in.

        Args:
            option: The option under test.

        Test scenario:
            `grouped[_group_of(...)][key]` raised a bare `KeyError('norm')` — no message, no did-you-mean,
            which is the whole point of the module.
        """
        grouped, _ = fold_symbology(Symbology(props={option: True}), "Image")
        assert grouped["norm"] == {option: True}, grouped

    def test_a_tooltip_that_names_fields_reports_that_it_cannot_limit_them(self):
        """The hover tool reads the element's own dimensions; the named columns go nowhere.

        Test scenario:
            The fields were dropped in silence. The module's contract is that anything it cannot express is
            reported rather than discarded (review L17), so they are — while the tool is still added.
        """
        grouped, unsupported = fold_symbology(
            Symbology.of(tooltip=("pop", "area")), "Points"
        )
        assert grouped["plot"]["tools"] == ["hover"], grouped
        assert "cannot be limited to" in unsupported["tooltip.fields"], unsupported

    @pytest.mark.parametrize(
        "value, named",
        [
            (("pop", "area"), ["pop", "area"]),
            ("pop", ["pop"]),
            (1, ["1"]),
            (True, ["True"]),
        ],
    )
    def test_the_named_fields_are_read_however_they_were_written(self, value, named):
        """The channel is plain text, so nothing stops one column being written as a bare string.

        Args:
            value: What the caller put on the channel.
            named: The columns the report should name.

        Test scenario:
            Listing the value directly spelled a string out letter by letter — `['p', 'o', 'p']` — and
            raised `TypeError: 'int' object is not iterable` on a number, from inside the fold (review M9).
        """
        _, unsupported = fold_symbology(Symbology.of(tooltip=value), "Points")
        assert f"{named}" in unsupported["tooltip.fields"], unsupported[
            "tooltip.fields"
        ]

    def test_a_tooltip_that_names_none_reports_nothing(self):
        """Asking for hover without naming columns is exactly what the tool does, so nothing is lost."""
        grouped, unsupported = fold_symbology(Symbology.of(tooltip=()), "Points")
        assert grouped["plot"]["tools"] == ["hover"], grouped
        assert "tooltip.fields" not in unsupported, unsupported

    def test_the_usual_groups_are_always_there(self):
        """A caller reads `style` and `plot` without asking whether they exist."""
        grouped, _ = fold_symbology(Symbology(props={"cmap": "magma"}), "Image")
        assert set(grouped) == {"style", "plot"}, grouped

    def test_asking_what_an_element_takes_does_not_switch_the_backend(self):
        """`allowed_options` is a question, and it was changing the answer to every later one.

        Test scenario:
            `hv.extension(backend)` is `Store.set_current_backend`, so one cross-backend validation left the
            whole process rendering through matplotlib (review M15).
        """
        held = hv.Store.current_backend
        try:
            hv.extension("bokeh")
            allowed_options("Image", backend="matplotlib")
            assert hv.Store.current_backend == "bokeh", hv.Store.current_backend
        finally:
            # The session's backend is shared state; a test that sets it puts it back (review N6).
            hv.Store.set_current_backend(held)
