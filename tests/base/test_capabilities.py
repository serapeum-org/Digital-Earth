"""What a backend can draw, declared rather than discovered (U-1, #294).

`api.py` kept one table of the `quickmap` keywords each backend honours, and every other answer lived where it
was refused: the web tier's CRS check, the interactive tier's Web-Mercator guard, a channel table read by one
function. These cover the value type each tier's declaration is written in — the vocabularies it is checked
against, and the two outcomes a missing capability has.
"""

import json
import subprocess
import sys

import pytest

from digitalearth.base.capabilities import (
    FEATURES,
    Capabilities,
    CapabilityError,
    is_feature,
)
from digitalearth.base.custom import custom_kind
from digitalearth.base.registry import furniture_kinds
from digitalearth.base.spec import CHANNELS


class TestTheVocabularies:
    """Every name is checked against the table it belongs to, where the declaration is written."""

    def test_a_kind_nobody_registered_is_refused(self):
        """A typo would otherwise read as "this tier cannot draw rasters"."""
        with pytest.raises(ValueError, match="layer kinds nobody registered"):
            Capabilities("flat", kinds={"rastre"})

    def test_a_custom_kind_is_declarable(self):
        """A tier that holds a caller's own objects says which engine it holds them for (#293)."""
        declared = Capabilities("3d", kinds={custom_kind("pyvista")})
        assert "custom:pyvista" in declared.kinds, declared.kinds

    def test_a_channel_outside_the_table_is_refused(self):
        """Channels are a closed vocabulary, so a declaration cannot invent one."""
        with pytest.raises(ValueError, match="channels that are not declared"):
            Capabilities("flat", channels={"glow"})

    def test_the_tooltip_channel_is_declarable(self):
        """The channel #292 added is one a tier can claim."""
        assert "tooltip" in CHANNELS, sorted(CHANNELS)
        declared = Capabilities("web", channels={"tooltip"})
        assert declared.supports("tooltip"), declared.to_dict()

    def test_a_data_driven_channel_must_also_be_a_channel(self):
        """A channel the tier cannot set at all cannot be set from a field."""
        with pytest.raises(ValueError, match="does not list them as channels"):
            Capabilities("flat", channels={"color"}, data_driven={"size"})

    def test_furniture_is_declared_as_a_feature(self):
        """#292's furniture kinds are features under their own names."""
        declared = Capabilities("web", features={"scale_bar", "navigation"})
        assert declared.supports("scale_bar"), declared.to_dict()

    def test_every_registered_furniture_kind_is_a_feature(self):
        """The furniture registry is the growth axis; the feature table does not repeat it."""
        not_declarable = [name for name in furniture_kinds() if not is_feature(name)]
        assert not_declarable == [], (
            f"furniture no tier could declare: {not_declarable}"
        )

    def test_a_feature_nobody_declared_is_refused(self):
        """The message names both vocabularies, since a feature may come from either."""
        with pytest.raises(ValueError, match="neither declared nor furniture"):
            Capabilities("flat", features={"teleport"})

    def test_a_scheme_must_be_a_name(self):
        """Schemes come from the classifier, so only the spelling can be checked here."""
        with pytest.raises(ValueError, match="is not a name"):
            Capabilities("flat", schemes={"Quantiles"})

    def test_a_bare_string_is_refused_rather_than_split_into_letters(self):
        """`frozenset("crs")` would be three one-letter capabilities."""
        with pytest.raises(ValueError, match="must be a collection of names"):
            Capabilities("flat", features="domain")

    def test_a_backend_with_no_name_is_refused(self):
        """A declaration nobody can look up by backend name is useless."""
        with pytest.raises(ValueError, match="backend must be a non-empty name"):
            Capabilities("")

    def test_every_feature_in_the_table_documents_itself(self):
        """The table is the readable half of the support matrix."""
        undocumented = [name for name, doc in FEATURES.items() if not doc.strip()]
        assert undocumented == [], f"features with no description: {undocumented}"


class TestWhatIsDeliberatelyMissing:
    """`absent` is what separates a decision from a gap nobody has filled."""

    def test_an_absent_capability_carries_its_reason(self):
        """The reason is what a caller is shown, so it says what the tier does instead."""
        declared = Capabilities("3d", absent={"domain": "a scene has no extent to set"})
        assert declared.reason("domain") == "a scene has no extent to set", (
            declared.absent
        )

    def test_a_capability_nobody_mentioned_has_no_reason(self):
        """Silence is not an explanation, and does not pretend to be one."""
        assert Capabilities("3d").reason("domain") is None, "an undeclared absence"

    def test_a_reason_is_required(self):
        """An entry with no reason is the gap this field exists to rule out."""
        with pytest.raises(ValueError, match="needs a reason"):
            Capabilities("flat", absent={"domain": "   "})

    def test_claiming_and_disclaiming_one_name_is_refused(self):
        """ "We have it and we deliberately do not" is not an answer."""
        with pytest.raises(ValueError, match="as absent and as supported"):
            Capabilities(
                "flat", features={"colorbar"}, absent={"colorbar": "no colour key"}
            )

    def test_a_name_that_is_not_a_string_is_refused(self):
        """A number in a capability set would compare equal to nothing a caller can ask for."""
        with pytest.raises(ValueError, match="must be names"):
            Capabilities("flat", features={1})

    def test_absent_must_be_a_mapping_of_reasons(self):
        """A list of names would be an absence with no explanation, which is the point of the field."""
        with pytest.raises(ValueError, match="absent must be a mapping"):
            Capabilities("flat", absent=["domain"])

    def test_absent_must_be_keyed_by_name(self):
        """A reason filed under a number could never be looked up."""
        with pytest.raises(ValueError, match="absent must be keyed by name"):
            Capabilities("flat", absent={1: "no extent"})

    def test_the_declaration_cannot_be_edited_afterwards(self):
        """A frozen value type: a tier's answer is not something a caller can rewrite."""
        declared = Capabilities("flat", absent={"domain": "no extent"})
        with pytest.raises(TypeError):
            declared.absent["domain"] = "changed"


class TestAskingForSomething:
    """One question for a kind, a channel, a scheme or a feature."""

    @pytest.fixture
    def declared(self):
        """Return a small declaration covering each kind of capability.

        Returns:
            The declaration.
        """
        return Capabilities(
            "flat",
            kinds={"raster"},
            channels={"color"},
            schemes={"categorical"},
            features={"colorbar"},
            absent={"domain": "it has no extent to set"},
        )

    @pytest.mark.parametrize(
        "name, supported",
        [
            ("raster", True),
            ("color", True),
            ("categorical", True),
            ("colorbar", True),
            ("domain", False),
            ("graticule", False),
        ],
    )
    def test_support_is_one_question(self, declared, name, supported):
        """A caller asks the same way whatever sort of capability it is.

        Args:
            declared: The declaration under test.
            name: The capability asked about.
            supported: Whether it is declared.
        """
        assert declared.supports(name) is supported, name

    def test_a_supported_capability_passes_silently(self, declared):
        """`require` is a guard, not a getter.

        Args:
            declared: The declaration under test.
        """
        assert declared.require("raster", caller="quickmap") is None, "no refusal"

    def test_an_absent_capability_is_refused_with_its_reason(self, declared):
        """The caller is told what the tier does instead, not only that it refused.

        Args:
            declared: The declaration under test.
        """
        with pytest.raises(CapabilityError, match="it has no extent to set"):
            declared.require("domain", caller="quickmap")

    def test_an_undeclared_capability_is_refused_without_one(self, declared):
        """Nothing is invented for a capability the tier never mentioned.

        Args:
            declared: The declaration under test.
        """
        with pytest.raises(CapabilityError) as refusal:
            declared.require("graticule", caller="quickmap")
        assert str(refusal.value).endswith("does not support"), str(refusal.value)

    def test_the_refusal_names_the_caller(self, declared):
        """A figure with a dozen builders needs to say which one asked.

        Args:
            declared: The declaration under test.
        """
        with pytest.raises(CapabilityError, match="Map.graticule needs"):
            declared.require("graticule", caller="Map.graticule")

    def test_the_refusal_is_a_value_error(self, declared):
        """A caller catching `ValueError` around `quickmap` keeps catching it.

        Args:
            declared: The declaration under test.
        """
        assert issubclass(CapabilityError, ValueError), CapabilityError.__mro__


class TestTheSupportMatrix:
    """`to_dict` is what the docs' support matrix is built from (U-5)."""

    def test_every_field_is_written_sorted(self):
        """Two runs write the same table, so a docs diff shows a real change."""
        stored = Capabilities("flat", kinds={"raster", "points"}).to_dict()
        assert stored["kinds"] == ["points", "raster"], stored

    def test_the_table_row_survives_json(self):
        """The matrix is generated from JSON, so nothing in a row may be a set."""
        declared = Capabilities(
            "web",
            kinds={"raster"},
            channels={"color"},
            data_driven={"color"},
            schemes={"categorical"},
            features={"scale_bar"},
            absent={"domain": "the map pans"},
        )
        assert json.loads(json.dumps(declared.to_dict())) == declared.to_dict(), (
            declared.to_dict()
        )


class TestTheImportCost:
    """A declaration is read before a backend is chosen, so reading one must load no engine."""

    def test_the_value_type_loads_no_renderer(self):
        """A fresh interpreter imports the capabilities module and no engine with it.

        Test scenario:
            Contract C11: `quickmap` refuses an unsupported keyword before importing a backend, and it will do
            that by reading declarations. A declaration that dragged in pyvista would make the refusal cost the
            import it exists to avoid. Run in a subprocess, since this session has the engines loaded already.
        """
        code = (
            "import sys; import digitalearth.base.capabilities as c;"
            "print([e for e in ('pyvista', 'maplibre', 'holoviews', 'matplotlib.pyplot')"
            " if e in sys.modules], c.Capabilities('flat').backend)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "[] flat", result.stdout or result.stderr
