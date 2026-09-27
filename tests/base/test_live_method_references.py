"""No prose in the package may name a method its receiver does not have (R2-M5, R2-M6).

A message or a cross-reference naming a deleted method is worse than none: it sends the reader to an
`AttributeError`. This branch's central act was deleting eighteen method spellings, and the only guard against
the class of defect that creates — `tests/three_d/test_capabilities3d.py`'s
`TestNoMessageNamesAMethodTheTierLost` — globbed ``src/digitalearth/three_d/*.py`` alone. Round 2 then found
five such sites, four of them outside that tree and one in ``base/``, which no guard read at all (R2-M6).

Two checks, because a reference comes in two shapes and they resolve differently:

1. **A dotted reference** — ``WebMap.animate`` — carries its receiver, so it is resolved on that class. This is
   the shape the finding turns on: ``terrain``, ``globe`` and ``animate`` were *web*-tier aliases that were
   deleted, while ``Scene3D.terrain``, ``Scene3D.globe``, ``Map.animate`` and ``TexturedGlobe.animate`` are
   current methods sharing the spelling. Matching on the bare name would clear a dead reference and condemn
   four live ones, so the receiver is what decides.
2. **A bare** ``:meth:`name` `` has no receiver, so it is resolved against every class in the package. That is
   looser than the 3-D guard's per-tier reading, and deliberately: a name that exists on some class is at least
   a name a reader can look up, and the 3-D guard keeps its tighter per-tier version of the same check.

Receivers are resolved by import rather than by AST: measured, every module under ``digitalearth`` imports in
the lean ``dev`` environment — the engines are reached inside the functions that need them — so `hasattr` can
answer for all four tiers from the one environment that runs the whole matrix. A **class name spelled by more
than one tier** (``DecorationMixin``, ``Renderer``, ``RasterMixin``, …) is ambiguous and is skipped, with
:class:`TestTheResolverSeesWhatItClaimsTo` holding that the skipping stays narrow rather than quietly total.

**Nothing is exempted.** This module was added with a two-entry exception list, because the sites it found
outside its own tree — ``static/textured_globe.py``'s bare ``save_gif`` and ``static/maps/raster.py``'s
``Scene._reproject``/``Scene._prepare`` — belonged to work another branch owned, and a guard that went red the
moment one was fixed would have made that fix require an edit here. Both are fixed, so the list is gone rather
than left empty: the checks below are the whole package, with no module standing outside them.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import digitalearth

#: The package tree every source file is read from.
PACKAGE_ROOT = Path(digitalearth.__file__).resolve().parent

#: A cross-reference or a prose mention carrying its receiver. The receiver has to start with a capital, which
#: keeps ``np.asarray`` and ``self.foo`` out; a dotted path naming modules first
#: (``digitalearth.three_d.base.Scene3DBase.vertical_exaggeration``) is excluded by the lookbehind, since it
#: resolves through the full path rather than through a bare class name.
DOTTED_REFERENCE = re.compile(r"`~?([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)`")

#: A bare cross-reference, as opposed to a dotted one, which the pattern above takes.
BARE_REFERENCE = re.compile(r":meth:`~?([A-Za-z_][A-Za-z0-9_]*)`")


#: Attribute names set on an instance rather than declared on the class, collected from the package's own
#: ``self.x = ...`` statements. `hasattr` cannot see one on the class, and ``Scene.ax`` is a perfectly good
#: reference to one, so they count as live.
SELF_ASSIGNMENT = re.compile(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=\n]+)?=[^=]")


def _package_modules() -> list:
    """Return every importable module in the package, the root included.

    Returns:
        The dotted module names, sorted.
    """
    found = [digitalearth.__name__]
    found += [
        info.name
        for info in pkgutil.walk_packages(
            digitalearth.__path__, f"{digitalearth.__name__}."
        )
    ]
    return sorted(found)


def _classes_by_name() -> tuple:
    """Return the package's classes, by the bare name a docstring writes them under.

    Returns:
        tuple: ``(unique, ambiguous, every)`` — the names carried by exactly one class, mapped to it; the names
        more than one tier spells the same way, which a bare receiver cannot pick between; and every class
        found, which is what a *receiverless* reference is resolved against.
    """
    seen: dict = {}
    for name in _package_modules():
        module = importlib.import_module(name)
        for class_name, held in inspect.getmembers(module, inspect.isclass):
            if not held.__module__.startswith(digitalearth.__name__):
                continue
            seen.setdefault(class_name, [])
            if held not in seen[class_name]:
                seen[class_name].append(held)
    unique = {name: held[0] for name, held in seen.items() if len(held) == 1}
    ambiguous = {name for name, held in seen.items() if len(held) > 1}
    every = [held for classes in seen.values() for held in classes]
    return unique, ambiguous, every


def _members_of(held: type) -> set:
    """Return every attribute name `held` answers to, instance attributes included.

    Args:
        held: The class a reference named.

    Returns:
        `dir()` plus, for each of its in-package bases, the annotated names and the ``self.x`` targets in its
        source — so an attribute a constructor sets counts as live.
    """
    names = set(dir(held))
    for base in held.__mro__:
        if not getattr(base, "__module__", "").startswith(digitalearth.__name__):
            continue
        names |= set(getattr(base, "__annotations__", {}) or {})
        try:
            source = inspect.getsource(base)
        except OSError:  # pragma: no cover - every in-package class has a source file
            continue
        names |= set(SELF_ASSIGNMENT.findall(source))
    return names


def _sources() -> list:
    """Return every source file in the package, as `(path, text)` pairs.

    Returns:
        One pair per ``.py`` file under ``src/digitalearth``, at any depth — which is what widens the 3-D
        guard's single-tier glob to the whole package.
    """
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    ]


@pytest.fixture(scope="module")
def resolver() -> tuple:
    """The class table every reference below is resolved against.

    Returns:
        tuple: ``(unique, ambiguous, live)`` — the unambiguous classes, the names too many tiers share, and the
        union of every member name in the package, for the bare references.
    """
    unique, ambiguous, every = _classes_by_name()
    # Every class, not only the unambiguously named ones: a bare reference has no receiver to disambiguate, and
    # `Renderer._arrange` is spelled the same way by two tiers, so restricting this union to `unique` reported
    # nine live private helpers as dangling.
    live: set = set()
    for held in every:
        live |= _members_of(held)
    # Module-level functions too: a bare reference has no receiver, so `_would_draw` written as a method
    # reference still names something a reader can look up. Using the method role for a function is a docs
    # nit; a name that resolves to nothing at all is the defect this half is for.
    for name in _package_modules():
        module = importlib.import_module(name)
        live |= {
            found
            for found, held in inspect.getmembers(module, inspect.isfunction)
            if held.__module__.startswith(digitalearth.__name__)
        }
    return unique, ambiguous, live


class TestNoProseNamesAMethodItsReceiverLost:
    """The guard R2-M6 asks for: the 3-D tier's check, over the whole package."""

    def test_no_dotted_reference_names_a_member_its_class_does_not_have(self, resolver):
        """A reference has to be a member of the class it names, whatever other tier has the spelling.

        Args:
            resolver: The class table.
        """
        unique, _, _ = resolver
        dangling = []
        for path, text in _sources():
            for number, line in enumerate(text.splitlines(), start=1):
                for class_name, member in DOTTED_REFERENCE.findall(line):
                    held = unique.get(class_name)
                    if held is None:
                        continue
                    if member not in _members_of(held):
                        dangling.append(
                            f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{number}: "
                            f"{class_name}.{member}"
                        )
        assert dangling == [], (
            f"these references name a member their own class does not have: {dangling}"
        )

    def test_no_bare_cross_reference_names_a_method_no_class_has(self, resolver):
        """A dangling cross-reference renders as plain text, so nothing but this says it points nowhere.

        Args:
            resolver: The class table.
        """
        _, _, live = resolver
        dangling = []
        for path, text in _sources():
            for number, line in enumerate(text.splitlines(), start=1):
                for name in BARE_REFERENCE.findall(line):
                    if name not in live:
                        dangling.append(
                            f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{number}: {name}"
                        )
        assert dangling == [], (
            f"these cross-references name a method no class in the package has: {dangling}"
        )


class TestTheResolverSeesWhatItClaimsTo:
    """A guard is worth what it opened, so what it opened is measured rather than trusted."""

    def test_every_module_in_the_package_imports_in_the_lean_environment(self):
        """The premise behind resolving by import: no tier's module needs its engine to be read.

        Test scenario:
            If one did, the checks above would silently stop covering that tier — the failure mode this module
            exists to close. Asserted here so the resolver's reach is a measurement.
        """
        unreadable = {}
        for name in _package_modules():
            try:
                importlib.import_module(name)
            except Exception as error:  # pragma: no cover - the assertion is the report
                unreadable[name] = f"{type(error).__name__}: {error}"
        assert unreadable == {}, (
            "these modules cannot be imported in this environment, so nothing resolves against them: "
            f"{unreadable}"
        )

    @pytest.mark.parametrize(
        "tier", ["base", "static", "interactive", "three_d", "web"]
    )
    def test_the_scan_reaches_every_tier(self, tier):
        """The 3-D guard globbed one tier, which is why five sites in the others survived a rename sweep.

        Args:
            tier: The subpackage that must appear among the scanned files.
        """
        scanned = {
            path.relative_to(PACKAGE_ROOT).parts[0]
            for path, _ in _sources()
            if len(path.relative_to(PACKAGE_ROOT).parts) > 1
        }
        assert tier in scanned, (
            f"the scan never reaches {tier}/; it read {sorted(scanned)}"
        )

    def test_the_scan_reaches_a_nested_module(self):
        """``rglob`` rather than ``glob``: ``base/spec/`` and ``static/maps/`` are two levels down.

        Test scenario:
            The 3-D guard used a one-level glob, which would have read none of them. ``static/maps/`` holds the
            five capability mixins, so a one-level scan misses most of the default tier.
        """
        nested = [
            path
            for path, _ in _sources()
            if len(path.relative_to(PACKAGE_ROOT).parts) > 2
        ]
        assert nested != [], "the scan reads only the top level of each tier"

    def test_only_the_names_more_than_one_tier_spells_are_skipped(self, resolver):
        """An ambiguous receiver is skipped, so the set of them must stay small and named.

        Args:
            resolver: The class table.

        Test scenario:
            Skipping is this guard's one blind spot, and a blind spot that grows quietly is how a guard stops
            being one. The facades a reference actually names must not be among them.
        """
        _, ambiguous, _ = resolver
        facades = {
            "Map",
            "WebMap",
            "InteractiveMap",
            "Scene3D",
            "Scene",
            "TexturedGlobe",
        }
        shared = sorted(facades & ambiguous)
        assert shared == [], (
            f"{shared} name more than one class in the package, so a reference to one resolves against "
            "whichever was imported first"
        )
