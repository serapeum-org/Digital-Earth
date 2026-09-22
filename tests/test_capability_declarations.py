"""What the four declarations say together, and what the dispatcher does with them (U-1, #294).

Each tier's own declaration test lives with that tier — `tests/three_d/test_capabilities3d.py`,
`tests/web/test_web_capabilities.py`, `tests/interactive/test_interactive_capabilities.py` — because a row
can only be checked against the engine that implements it, and only that tier's environment has the engine.
Three things are left over, and they belong here:

* **the static tier's own row.** It needs no optional engine, so it is checked in the default environment.
  Its channel and builder halves are in `tests/static/test_static_capabilities.py` and
  `tests/static/test_static_seam.py`; what those do not assert is the direction that catches a lying
  declaration — a kind claimed with nothing drawing it — so that is here.
* **contract C4 across all four.** "One `scheme`/`k` pair paints the same classes on every tier" is a
  statement about the four rows together, which no single tier's test can make.
* **how `api.py` refuses.** Both gates now decide from the declarations and raise the declared
  :class:`~digitalearth.base.capabilities.CapabilityError`, carrying the reason the tier gave. The wording
  of the first clause is pinned by `tests/test_quickplot.py`; what is pinned here is that the reason is
  there at all, and that the refusal is still a `ValueError` for anyone catching one.
"""

import pytest

from digitalearth.api import _DECLARATIONS, _KEYWORD_CAPABILITIES
from digitalearth.base.capabilities import FEATURES, CapabilityError
from digitalearth.base.registry import furniture_kinds, kinds
from digitalearth.base.spec import CHANNELS, Scale

#: The four tiers, as `quickmap(backend=...)` spells them. Read off the dispatcher rather than written out,
#: so a fifth tier joins these checks by being declared rather than by being added here.
BACKENDS = sorted(_DECLARATIONS)

#: Values with enough spread for any classifier to cut three classes out of.
SAMPLE = list(range(100))


class TestEveryTierClassifiesTheSameWay:
    """Contract C4 — a `scheme`/`k` pair means the same thing whichever backend draws it."""

    def test_the_four_rows_declare_one_set_of_schemes(self):
        """A scheme on one tier and not another would make `scheme="quantiles"` backend-dependent.

        Test scenario:
            C4 is the claim that the tiers classify identically, and they can only do that if they offer
            the same names. Comparing every row against the static one — rather than pairwise — names the
            offending backend in the message.
        """
        reference = _DECLARATIONS["matplotlib"].schemes
        disagreeing = {
            backend: sorted(declaration.schemes.symmetric_difference(reference))
            for backend, declaration in _DECLARATIONS.items()
            if declaration.schemes != reference
        }
        assert disagreeing == {}, f"these tiers offer other schemes: {disagreeing}"

    @pytest.mark.parametrize(
        "scheme", sorted(_DECLARATIONS["matplotlib"].schemes - {"categorical"})
    )
    def test_every_declared_scheme_is_one_the_shared_classifier_knows(self, scheme):
        """The classifier behind every tier is one function, and it raises for a name it does not know.

        Args:
            scheme: The declared scheme under test.

        Test scenario:
            `Scale.breaks_of` is the seam `interactive`, `three_d` and `web` all cut classes through, so a
            scheme declared here and unknown there is a name a caller can read off the support matrix and
            never use.
        """
        assert len(Scale.breaks_of(SAMPLE, scheme, 3)) >= 2, scheme

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_the_nominal_scheme_is_declared_everywhere(self, backend):
        """`categorical` is not the classifier's; it is the nominal path, and C4 pins it on every tier.

        Args:
            backend: The tier under test.
        """
        assert "categorical" in _DECLARATIONS[backend].schemes, backend

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_the_continuous_ramp_is_not_a_declared_scheme(self, backend):
        """C4 makes `scheme=None` the continuous default everywhere, so it names no scheme.

        Args:
            backend: The tier under test.

        Test scenario:
            A row that listed `"none"` or `"continuous"` would turn the absence of a classification into a
            classification, and a caller reading the matrix would pass it as a string.
        """
        listed = sorted(
            _DECLARATIONS[backend].schemes.intersection({"none", "None", "continuous"})
        )
        assert listed == [], (
            f"{backend} names the continuous ramp as a scheme: {listed}"
        )


class TestTheStaticTierDeclaresWhatItDraws:
    """The matplotlib row, checked in the direction that catches a claim with nothing behind it.

    `tests/static/test_static_capabilities.py` reads the builders and asserts everything they draw is
    declared; `tests/static/test_static_seam.py` asserts each builder records its registered kind. Neither
    can fail for a kind that is declared and drawn by nothing, which is the shape a capability lie takes.
    """

    def test_the_declaration_and_the_drawer_table_are_the_same_list(self):
        """This tier draws every kind it declares from a description, so the two lists are one.

        Test scenario:
            A kind added to the declaration has no drawer and lands on one side; a kind dropped from the
            declaration while its drawer stays lands on the other. The symmetric difference names which.
        """
        from digitalearth.static.capabilities import CAPABILITIES
        from digitalearth.static.renderer import DRAWN_KINDS

        difference = sorted(CAPABILITIES.kinds.symmetric_difference(DRAWN_KINDS))
        assert difference == [], f"{difference} is in one list and not the other"

    def test_every_declared_kind_resolves_to_a_drawer(self):
        """A declared kind the renderer could not draw would raise where a message belongs."""
        from digitalearth.static.capabilities import CAPABILITIES
        from digitalearth.static.renderer import drawer_for

        unresolved = []
        for kind in sorted(CAPABILITIES.kinds):
            try:
                drawer_for(kind)
            except KeyError:
                unresolved.append(kind)
        assert unresolved == [], f"{unresolved} are declared with no drawer"


class TestTheKeywordMapNamesRealCapabilities:
    """`quickmap`'s keywords are translated into the vocabulary the tiers declare in, so it has to be it."""

    @pytest.mark.parametrize(
        ("keyword", "capability"),
        sorted(
            (keyword, capability)
            for keyword, capabilities in _KEYWORD_CAPABILITIES.items()
            for capability in capabilities
        ),
    )
    def test_each_capability_a_keyword_maps_to_is_a_declarable_name(
        self, keyword, capability
    ):
        """A typo here is silent: every tier answers "no" and the keyword is refused everywhere.

        Args:
            keyword: The `quickmap` keyword.
            capability: The capability name it is translated into.

        Test scenario:
            `Capabilities` checks its own names at construction, but nothing checked the *other* side of
            the translation. `raster_render` for `raster_renderer` would leave `kind=` refused on all four
            backends, with each refusal correctly reporting that no tier declares it.
        """
        declarable = (
            set(FEATURES) | set(furniture_kinds()) | set(CHANNELS) | set(kinds())
        )
        assert capability in declarable, (
            f"{keyword}= maps to {capability!r}, which is not a capability"
        )

    @pytest.mark.parametrize(
        ("keyword", "capability"),
        sorted(
            (keyword, capability)
            for keyword, capabilities in _KEYWORD_CAPABILITIES.items()
            for capability in capabilities
        ),
    )
    def test_each_capability_a_keyword_maps_to_is_declared_or_refused_somewhere(
        self, keyword, capability
    ):
        """A capability no tier has an opinion on means a keyword no tier can explain.

        Args:
            keyword: The `quickmap` keyword.
            capability: The capability name it is translated into.

        Test scenario:
            A refusal reads its reason from `absent`, so a capability that is neither supported anywhere
            nor declared absent anywhere is refused with no reason on every backend — the hand-written
            silence #294 replaced.
        """
        answered = [
            backend
            for backend, declaration in _DECLARATIONS.items()
            if declaration.supports(capability) or declaration.reason(capability)
        ]
        assert answered != [], (
            f"no tier says anything about {capability!r} ({keyword}=)"
        )


class TestTheDispatcherRefusesFromTheDeclarations:
    """Both gates decide from the rows, say why in the tier's words, and raise the declared error."""

    def test_a_keyword_refusal_is_the_declared_error(self, dataset):
        """`CapabilityError` is what the declarations raise, so the dispatcher raises it too.

        Args:
            dataset: The raster to draw — never drawn, since the refusal precedes the build.
        """
        from digitalearth import quickmap

        with pytest.raises(CapabilityError):
            quickmap(dataset, backend="3d", domain="europe")

    def test_a_keyword_refusal_is_still_a_value_error(self, dataset):
        """Callers catching `ValueError` around `quickmap` keep catching this one.

        Args:
            dataset: The raster to draw.

        Test scenario:
            `CapabilityError` subclasses `ValueError` for exactly this reason. A refusal that stopped being
            one would break every caller written against the behaviour C11 shipped.
        """
        from digitalearth import quickmap

        with pytest.raises(ValueError):
            quickmap(dataset, backend="3d", domain="europe")

    @pytest.mark.parametrize(
        ("backend", "phrase"),
        [("3d", "framed by its camera"), ("web", "pans and zooms")],
    )
    def test_a_keyword_refusal_carries_the_tier_s_own_reason(
        self, dataset, backend, phrase
    ):
        """The sentence comes from that tier's `absent`, not from a table in `api.py`.

        Args:
            dataset: The raster to draw.
            backend: The tier that cannot frame on a region.
            phrase: Part of the reason it declared for having no `domain`.
        """
        from digitalearth import quickmap

        with pytest.raises(CapabilityError, match=phrase):
            quickmap(dataset, backend=backend, domain="europe")

    @pytest.mark.parametrize("backend", ["3d", "web"])
    def test_a_renderer_wrapper_refusal_is_the_declared_error(self, dataset, backend):
        """The four `imshow`/`contourf`/`contour`/`pcolormesh` wrappers refuse the same way.

        Args:
            dataset: The raster to draw.
            backend: A tier with no renderer selector.
        """
        from digitalearth.api import imshow

        with pytest.raises(CapabilityError):
            imshow(dataset, backend=backend)

    @pytest.mark.parametrize(
        ("backend", "phrase"),
        [
            ("3d", "a surface, a volume or a globe"),
            ("web", "drawn as an image and contours are their own builder"),
        ],
    )
    def test_a_renderer_wrapper_refusal_carries_the_tier_s_own_reason(
        self, dataset, backend, phrase
    ):
        """A wrapper is the one gate reached without naming `kind=`, so it most needs the reason.

        Args:
            dataset: The raster to draw.
            backend: A tier with no renderer selector.
            phrase: Part of the reason it declared for having no `raster_renderer`.

        Test scenario:
            The wrapper injects `kind="imshow"` itself, so its message may not name that keyword — it told
            the caller only that the renderer was missing. The declared reason is what turns that into an
            answer: a 3-D raster is a surface or a volume, chosen by calling a different builder.
        """
        from digitalearth.api import imshow

        with pytest.raises(CapabilityError, match=phrase):
            imshow(dataset, backend=backend)
