"""Tests for :mod:`digitalearth.base.bigdata` — the shared big-data cutoff and the rule that validates it.

Round-2 review **L6**: ``validate_big_data_threshold`` ran the caller's value through ``int()``, which
*truncated* a fraction (``1.9`` became ``1``, cutting over one row earlier than asked) and let every other
bad value escape as a bare ``ValueError``/``TypeError`` from ``int`` itself, naming neither the parameter nor
the call. A cutoff is a count of rows, so anything that is not one is now refused by name.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from digitalearth.base.bigdata import (
    DEFAULT_BIG_DATA_THRESHOLD,
    validate_big_data_threshold,
)

CALLER = "InteractiveMap.points()"


class TestAWholeRowCountIsAccepted:
    """The values that really are a row count still pass, unchanged."""

    @pytest.mark.parametrize(
        ("given", "expected"), [(1000, 1000), (0, 0), (1000.0, 1000)]
    )
    def test_a_whole_count_comes_back_as_an_int(self, given, expected):
        """An integral value is the count it names, whether it was written as an ``int`` or a ``float``.

        Args:
            given: The cutoff as the caller wrote it.
            expected: The count it means.

        Test scenario:
            ``1000.0`` is a whole number of rows, so refusing it would be pedantry — the rule is about
            values that are *not* counts, not about which literal spelling was used.
        """
        assert validate_big_data_threshold(given, caller=CALLER) == expected

    def test_the_shared_default_survives_the_check(self):
        """The package default is itself a legal cutoff — the guard must never reject the fallback."""
        assert (
            validate_big_data_threshold(DEFAULT_BIG_DATA_THRESHOLD, caller=CALLER)
            == DEFAULT_BIG_DATA_THRESHOLD
        )


class TestAValueThatIsNotARowCountIsRefused:
    """L6 — a cutoff that cannot be a row count is named, not coerced into one."""

    def test_a_fraction_is_refused_instead_of_truncated(self):
        """``1.9`` used to become ``1`` silently, cutting over one row earlier than the caller asked.

        Test scenario:
            This is the actual defect: no error, no warning, and a routing decision one row off. The
            message has to name the keyword and the call, because the truncation is invisible in the
            output.
        """
        with pytest.raises(ValueError) as excinfo:
            validate_big_data_threshold(1.9, caller=CALLER)
        message = str(excinfo.value)
        assert message.startswith(CALLER), message
        assert "whole number of rows" in message, (
            f"the message must say what a valid cutoff is, got {message!r}"
        )
        assert "1.9" in message, (
            f"the message must quote the value it refused, got {message!r}"
        )

    @pytest.mark.parametrize(
        "given",
        [float("nan"), float("inf"), float("-inf"), "abc", None, True, False, "1000"],
    )
    def test_a_non_count_is_refused_with_the_parameter_named(self, given):
        """Every non-count answers the same way, naming ``big_data_threshold`` and the call.

        Args:
            given: A value that is not a whole number of rows.

        Test scenario:
            ``int("abc")`` and ``int(nan)`` raised a bare ``ValueError``, ``int(None)`` a ``TypeError`` and
            ``int(inf)`` an ``OverflowError`` — three different classes, none of them mentioning the
            keyword, and only the negative case was documented in ``Raises:``. ``True`` is in the list
            because ``int(True)`` is ``1``: a flag silently became a one-row cutoff.
        """
        with pytest.raises(ValueError) as excinfo:
            validate_big_data_threshold(given, caller=CALLER)
        message = str(excinfo.value)
        assert message.startswith(CALLER), message
        assert "whole number of rows" in message, message

    def test_a_negative_cutoff_keeps_its_own_message(self):
        """A negative cutoff is a different mistake (an "unlimited" sentinel) and keeps its own wording."""
        with pytest.raises(ValueError, match="must not be negative"):
            validate_big_data_threshold(-1, caller=CALLER)


#: Every tier that routes a layer by size, as `(module, facade)`. Written down rather than discovered, and
#: `TestEveryTierWithTheCutoffIsListed` is what refuses a tier that gained the cutoff without joining it.
CUTOFF_TIERS = (
    ("digitalearth.web", "WebMap"),
    ("digitalearth.interactive", "InteractiveMap"),
    ("digitalearth.three_d", "Scene3D"),
)

#: The Core name for the cutoff, and the suffix every spelling of it carries.
CUTOFF = "big_data_threshold"
CUTOFF_SUFFIX = "_threshold"


def _cutoff_builders(facade) -> dict:
    """Return every public builder on a facade that takes a size cutoff, and the names it takes it under.

    Args:
        facade: The tier's facade class.

    Returns:
        `{builder: [parameter, ...]}` for the builders with at least one `*_threshold` parameter. Read off
        signatures, so it costs no engine: all three facades import without theirs, which is what lets one
        check ask all three tiers in the `main` matrix.
    """
    found = {}
    for name in dir(facade):
        if name.startswith("_"):
            continue
        held = getattr(facade, name, None)
        if not callable(held):
            continue
        try:
            parameters = inspect.signature(held).parameters
        except (
            TypeError,
            ValueError,
        ):  # pragma: no cover - a C-level callable has no signature
            continue
        spellings = [
            parameter for parameter in parameters if parameter.endswith(CUTOFF_SUFFIX)
        ]
        if spellings:
            found[name] = spellings
    return found


def _renamed_onto_the_cutoff(module: str) -> set:
    """Return the deprecated spellings a tier resolves onto the Core cutoff name.

    Read off the tier's own sources rather than off a builder's, because a builder need not apply the rename
    rule itself: the interactive tier's three hand both spellings to one resolver, which is the arrangement
    `renamed_parameter` exists to encourage. What has to be true is that *somewhere in the tier* each extra
    spelling is named as the `old=` of a `renamed_parameter` whose `new=` is the Core name.

    Args:
        module: The tier's package, e.g. `"digitalearth.web"`.

    Returns:
        The `old=` spellings found, as a set. Empty for a tier that deprecates nothing, which is what the
        clause allows.
    """
    package = Path(importlib.import_module(module).__file__).parent
    found = set()
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "renamed_parameter" not in text or CUTOFF not in text:
            continue
        for tree in (ast.parse(text),):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                named = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if named != "renamed_parameter":
                    continue
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"new", "old"}
                }
                new = keywords.get("new")
                old = keywords.get("old")
                if not isinstance(new, ast.Constant) or new.value != CUTOFF:
                    continue
                if isinstance(old, ast.Constant):
                    found.add(old.value)
    return found


def _cutoff_facade(module: str, name: str):
    """Return a tier's facade class, skipping the test when the package will not import.

    Args:
        module: The tier's package.
        name: The facade's class name.

    Returns:
        The class.
    """
    return getattr(pytest.importorskip(module), name)


class TestTheCutoffIsSpelledOneWayWhereverItAppears:
    """Contract C8, held across every tier that has the cutoff — including the two with no alias.

    The clause used to end "with one deprecated alias", which described the **interactive** tier's history as
    though it were a universal rule. Measured: `InteractiveMap` really does carry a prior spelling
    (`rasterize_threshold`, on `points`, `polygons` and `trimesh`), and `WebMap` and `Scene3D` never had one
    — so neither could keep the clause without inventing a keyword purely in order to deprecate it. The
    clause makes the alias conditional on there *having been* another spelling, and this is what holds the
    conditional form: the Core name everywhere, `None` as the per-call sentinel, and any second spelling
    resolved through `renamed_parameter` naming the Core one.
    """

    @pytest.mark.parametrize(("module", "name"), CUTOFF_TIERS, ids=lambda value: value)
    def test_every_builder_takes_the_cutoff_under_the_core_name(self, module, name):
        """A builder that routes by size takes `big_data_threshold`.

        Args:
            module: The tier's package.
            name: The facade's class name.
        """
        facade = _cutoff_facade(module, name)
        builders = _cutoff_builders(facade)
        assert builders != {}, f"{name} has no builder taking a size cutoff"
        wrong = {
            builder: spellings
            for builder, spellings in builders.items()
            if CUTOFF not in spellings
        }
        assert wrong == {}, (
            f"{name} routes by size under another name only: {wrong}; the cutoff is spelled {CUTOFF!r} on "
            "every tier"
        )

    @pytest.mark.parametrize(("module", "name"), CUTOFF_TIERS, ids=lambda value: value)
    def test_the_per_call_override_defers_to_the_map_when_unset(self, module, name):
        """`None` is the sentinel that means "use the map's attribute" — the clause's second reach.

        Args:
            module: The tier's package.
            name: The facade's class name.

        Test scenario:
            A signature default other than `None` would make the per-call override indistinguishable from
            the map's own setting: the builder could not tell a caller's value from its own default, which is
            the distinction the shared resolver exists to keep.
        """
        facade = _cutoff_facade(module, name)
        defaults = {
            builder: inspect.signature(getattr(facade, builder))
            .parameters[CUTOFF]
            .default
            for builder in _cutoff_builders(facade)
        }
        wrong = {
            builder: default
            for builder, default in defaults.items()
            if default is not None
        }
        assert wrong == {}, (
            f"{name} defaults the per-call cutoff to a value rather than to the map: {wrong}"
        )

    @pytest.mark.parametrize(("module", "name"), CUTOFF_TIERS, ids=lambda value: value)
    def test_any_second_spelling_is_a_deprecated_alias_for_it(self, module, name):
        """The conditional half: a second spelling must forward, and having none is keeping the clause.

        Args:
            module: The tier's package.
            name: The facade's class name.

        Test scenario:
            The check the old wording could not have. It passes for a tier with one alias (interactive's
            `rasterize_threshold`) **and** for a tier with none (web, 3-D). Demanding exactly one, as "with
            one deprecated alias" read, fails on two of the three — which is why the clause was the thing to
            fix rather than the tiers. The alias is checked by reading the builder's source for the shared
            resolver, the same way `tests/test_contract_names.py` checks a renamed *method* is a shim rather
            than a second copy of the body.

            Two readings, because a builder need not resolve the pair itself: the interactive tier's three
            hand both spellings to `_resolve_big_data_threshold`, which is the one place the rename rule is
            applied. So the builder must *pass the spelling on*, and the tier must resolve it through
            `renamed_parameter` naming the Core name. Deleting that resolution — which would make the old
            spelling work silently, the exact thing a deprecation exists to prevent — fails this.
        """
        facade = _cutoff_facade(module, name)
        resolved = _renamed_onto_the_cutoff(module)
        unforwarded = {}
        for builder, spellings in _cutoff_builders(facade).items():
            source = inspect.getsource(getattr(facade, builder))
            for spelling in spellings:
                if spelling == CUTOFF:
                    continue
                if spelling not in resolved:
                    unforwarded[f"{builder}({spelling}=)"] = "nothing deprecates it"
                elif spelling not in source:
                    unforwarded[f"{builder}({spelling}=)"] = "the builder drops it"
        assert unforwarded == {}, (
            f"{name} takes a second spelling of the cutoff that is not a deprecated alias of it: "
            f"{unforwarded}"
        )


class TestEveryTierWithTheCutoffIsListed:
    """`CUTOFF_TIERS` is written down, so something has to refuse a tier that is missing from it."""

    #: Every tier's facade, including the one with no size route, so the walk below can ask it.
    FACADES = {
        "digitalearth.static": "Map",
        "digitalearth.web": "WebMap",
        "digitalearth.interactive": "InteractiveMap",
        "digitalearth.three_d": "Scene3D",
    }

    def test_no_shipped_tier_routes_by_size_unlisted(self):
        """A tier that gained the cutoff would otherwise be held to none of the checks above.

        Test scenario:
            The static tier has no size route at all — it draws every row it is given — so it is
            legitimately absent from `CUTOFF_TIERS`. Discovered by walking the packages that ship a
            capability table rather than by trusting that, because "legitimately absent" is exactly the claim
            that goes stale.
        """
        root = Path(__file__).resolve().parents[2] / "src" / "digitalearth"
        listed = {module for module, _ in CUTOFF_TIERS}
        unlisted = []
        for path in sorted(root.glob("*/capabilities.py")):
            module = f"digitalearth.{path.parent.name}"
            if module in listed or module not in self.FACADES:
                continue
            facade = getattr(importlib.import_module(module), self.FACADES[module])
            if _cutoff_builders(facade):
                unlisted.append(module)
        assert unlisted == [], (
            f"{unlisted} route a layer by size and are not in CUTOFF_TIERS, so contract C8 is asserted of "
            "them nowhere"
        )
