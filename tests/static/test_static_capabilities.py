"""What the matplotlib tier declares it can draw (U-1, #294; the static seam's part of #303).

Its answer used to be one hand-written row in `api.py` — six keyword names, the last row not derived from a
tier's own declaration — and everything beyond those six was a refusal raised wherever a builder reached it.
These cover the declaration that replaces it, and the two questions it is easiest to get wrong: which kinds
this tier actually draws, and which channels its flat keywords can carry.
"""

import ast
import pathlib

import pytest

from digitalearth.base.capabilities import Capabilities
from digitalearth.base.registry import kind_info, kinds
from digitalearth.static.capabilities import CAPABILITIES

#: Kinds whose registered description names a builder on another tier only. Declaring one here would claim a
#: layer this tier cannot draw.
ELSEWHERE = (
    "terrain",
    "volume",
    "isosurface",
    "model",
    "extrusion",
    "clusters",
    "labels",
    "point_cloud",
    "custom:holoviews",
    "custom:maplibre",
    "custom:pyvista",
)


class TestTheDeclaration:
    """The tier's own answer, as data."""

    def test_the_declaration_is_this_backend(self):
        """The row is named as `quickmap(backend=...)` spells it — `matplotlib`, not `static`."""
        assert CAPABILITIES.backend == "matplotlib", CAPABILITIES.backend

    def test_the_declaration_is_a_capabilities_value(self):
        """The shared type, so one support matrix can be built from every tier's row."""
        assert isinstance(CAPABILITIES, Capabilities), type(CAPABILITIES)

    def test_the_channels_are_the_ones_the_style_schema_declares(self):
        """Two, and they are not chosen here: the schema is where a keyword says what it drives."""
        from digitalearth.static.render_compat import STATIC_STYLE_SCHEMA

        declared = {
            key.channel for key in STATIC_STYLE_SCHEMA.keys.values() if key.channel
        }
        assert CAPABILITIES.channels == declared, (
            f"declared {sorted(CAPABILITIES.channels)}, schema drives {sorted(declared)}"
        )

    def test_no_channel_is_driven_by_a_field(self):
        """The flat keywords carry a constant, which is what `fold_symbology` says when asked for more.

        Test scenario:
            A field-driven channel is not folded into a keyword here — the builder resolves it to values and
            passes those — so `data_driven` is empty rather than a copy of `channels`.
        """
        from digitalearth.base.spec import Encoding, Symbology
        from digitalearth.static.render_compat import fold_symbology

        driven = Symbology.of(opacity=Encoding.by_field("opacity", "confidence"))
        _, unsupported = fold_symbology(driven)
        assert CAPABILITIES.data_driven == frozenset(), CAPABILITIES.data_driven
        assert "opacity" in unsupported, unsupported

    def test_every_kind_a_builder_draws_is_declared(self, dataset, points):
        """A builder drawing an undeclared kind would make the declaration a lie.

        Args:
            dataset: A raster to draw.
            points: Features to draw.

        Test scenario:
            Read off what the builders actually record, rather than by matching the word "static" in the
            registry's free-text descriptions — which is a spelling, not a fact about this tier (review N4).
            The registry's own credit line is checked separately, as a second opinion.
        """
        from pyramids.feature import FeatureCollection

        from digitalearth.static import Map

        # The tier does not describe its layers yet — that is #303's renderer half — so the kinds are read
        # from the registry, which names the builder that draws each one. Every builder listed here is
        # called, so a kind whose builder disappears fails on the call rather than on the table.
        builders = {
            "raster": lambda canvas: canvas.field(dataset),
            "contours": lambda canvas: canvas.contours(dataset),
            "filled_contours": lambda canvas: canvas.contours(dataset, filled=True),
            "mesh": lambda canvas: canvas.pcolormesh(dataset),
            "points": lambda canvas: canvas.points(FeatureCollection(points)),
            "coastlines": lambda canvas: canvas.coastlines(),
            "text": lambda canvas: canvas.text(0.0, 0.0, "here"),
            "graticule": lambda canvas: canvas.graticule(),
        }
        with Map() as canvas:
            for draw in builders.values():
                draw(canvas)
        undeclared = sorted(set(builders) - CAPABILITIES.kinds)
        assert undeclared == [], f"these are drawn and not declared: {undeclared}"

    def test_the_registry_credits_nothing_to_this_tier_that_it_does_not_declare(self):
        """A second opinion: the registry describes each kind by the builders that draw it.

        Test scenario:
            `kind_info("flow").doc` reads "flows between places — static sankey", so `flow` has to be
            declared. This is free text, so it is a cross-check rather than the check.
        """
        credited = {name for name in kinds() if "static " in kind_info(name).doc}
        assert credited <= CAPABILITIES.kinds, sorted(credited - CAPABILITIES.kinds)

    def test_the_decorations_are_declared_too(self):
        """A coastline is a layer this tier draws, so it is a kind rather than a feature."""
        assert {"coastlines", "borders", "land", "graticule", "text"} <= (
            CAPABILITIES.kinds
        ), sorted(CAPABILITIES.kinds)

    @pytest.mark.parametrize("name", ELSEWHERE)
    def test_a_kind_another_tier_draws_is_not_claimed(self, name):
        """A declaration that claims everything answers nothing.

        Args:
            name: A registered kind whose builders live on another tier.

        Test scenario:
            `point_cloud` is the trap: `Map.point_cloud` exists, but it is an alias of `grid_points` and
            draws the `points` kind. The `point_cloud` *kind* is the 3-D and web tiers' positioned cloud.
        """
        assert name not in CAPABILITIES.kinds, name

    def test_the_schemes_are_the_shared_classifier_s(self):
        """One `scheme`/`k` pair paints the same classes on every tier (contract C4)."""
        from digitalearth.interactive.capabilities import (
            CAPABILITIES as INTERACTIVE,
        )

        assert CAPABILITIES.schemes == INTERACTIVE.schemes, sorted(
            CAPABILITIES.schemes.symmetric_difference(INTERACTIVE.schemes)
        )

    def test_the_vector_export_is_this_tier_s_alone(self):
        """A figure written as PDF or SVG is what matplotlib has and a canvas does not."""
        from digitalearth.api import _DECLARATIONS

        others = [
            backend
            for backend, declaration in _DECLARATIONS.items()
            if backend != "matplotlib" and declaration.supports("export_vector")
        ]
        assert CAPABILITIES.supports("export_vector"), sorted(CAPABILITIES.features)
        assert not others, others

    @pytest.mark.parametrize(
        "name, phrase",
        [
            ("tooltip", "no pointer"),
            ("height", "flat"),
            ("export_html", "not as a page"),
            ("layer_switcher", "drawn once"),
        ],
    )
    def test_what_is_missing_says_why(self, name, phrase):
        """Saying "no" and saying "not yet" are different answers; only a reason tells them apart.

        Args:
            name: What the tier does not have.
            phrase: Part of the reason it gives.
        """
        assert phrase in (CAPABILITIES.reason(name) or ""), CAPABILITIES.reason(name)

    def test_the_declaration_reads_as_data(self):
        """It is a table, not code: nothing but the shared type is imported to build it.

        Test scenario:
            The other tiers pin this by importing their `capabilities` in a subprocess and checking the
            engine stayed out of `sys.modules`. That cannot be done here — `digitalearth.static.__init__`
            builds `Map`, so matplotlib is loaded before this module is reached — so the module's own
            imports are read instead.
        """
        import digitalearth.static.capabilities as module

        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        imported = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert imported == ["digitalearth.base.capabilities"], imported


class TestTheDispatcherReadsIt:
    """`api.py` no longer writes a row of its own."""

    def test_the_row_is_the_one_it_always_was(self):
        """The six keywords `quickmap` honoured on this backend, now derived rather than written."""
        from digitalearth.api import BACKEND_CAPABILITIES

        assert sorted(BACKEND_CAPABILITIES["matplotlib"]) == [
            "basemap",
            "coastlines",
            "colorbar",
            "crs",
            "domain",
            "kind",
        ], BACKEND_CAPABILITIES["matplotlib"]

    def test_every_backend_s_row_is_derived(self):
        """The hand-written row was the last one; the table is now one declaration per tier."""
        from digitalearth.api import _DECLARATIONS, BACKEND_CAPABILITIES

        assert set(BACKEND_CAPABILITIES) == set(_DECLARATIONS), sorted(
            set(BACKEND_CAPABILITIES).symmetric_difference(_DECLARATIONS)
        )

    def test_this_tier_is_the_only_one_that_frames_on_a_region(self):
        """`domain=` is what the row has that no other row does, and that is why it is checked."""
        from digitalearth.api import BACKEND_CAPABILITIES

        framing = sorted(
            backend for backend, row in BACKEND_CAPABILITIES.items() if "domain" in row
        )
        assert framing == ["matplotlib"], framing
