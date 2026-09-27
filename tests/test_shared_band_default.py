"""A ``band`` default is the shared constant, never the number written out again (#342).

``base/spec/selection.py`` declares :data:`~digitalearth.base.spec.DEFAULT_BAND` — "the band a builder reads
when the caller names none" — and ``base/stretch.py`` declares
:data:`~digitalearth.base.stretch.DEFAULT_COMPOSITE_BANDS` for the three-band case. Measured on
``feat/wave7-orders-27a-23-26-25``, 37 signatures across 23 modules wrote ``band: int = 1`` anyway while three
modules read the constant, so the shared default was spelled out far more often than it was referred to.

Nothing was wrong at run time — the literal and the constant are the same number, which is exactly why nothing
caught it and why no test can catch it by calling a builder. The cost is deferred: pyramids counts bands from
1, and the day that assumption or the default changes, a literal is an edit in every file that made its own
copy, with no name for a tool to follow. This guard is the tool.

**The rule.** A parameter named ``band`` or ``bands`` may not default to a number written out — an integer, or
a sequence of them. When it defaults to a name, that name must be the shared constant, and the module must
either *import* it or be the one that declares it. Where the name is bound is the load-bearing half:
``DEFAULT_BAND`` is ``1`` and CPython caches small integers, so a module-local ``DEFAULT_BAND = 1`` would
satisfy both ``==`` and ``is`` against the real one. Only the binding tells a shared constant from a private
copy of today's value.

**What is deliberately not an offence.** A ``band`` default that is not a number is not a band index at all
and is left alone: ``web/base.py``'s ``add_layer(band: str = "data")`` names a *field* of a GeoJSON feature,
``three_d/base.py``'s ``add_mesh(band: Any = None)`` means "unset", and ``base/types.py``'s Protocol writes
``band: int = ...``. A parameter that is not a band keeps its literals — ``level: int = 1`` is none of this
guard's business, and :class:`TestTheRuleItself` holds it to that so the check cannot grow into "no integer
default anywhere".

**Scope is the package, not the tests.** ``tests/`` is full of stub readers that mimic pyramids' own
``read_array(band=0)`` — 0-based, because that is the signature they stand in for. They are doubles of an
upstream API rather than builders of ours, so converting them would make them wrong.

Read with :mod:`ast` rather than by importing, so this runs in the lean ``dev`` environment, sees the four
engine-backed tiers without any of their engines, and reports the module and line rather than a traceback.
**Nothing is exempted**: there is no allow-list here, and the one module that could have needed one does not.
``base/arrays.py`` could not import ``base/spec/`` at all while ``base/spec/scale.py`` imported ``finite``
from it — the cycle was real and measured — so that edge was deferred into ``Scale._limits`` rather than
writing ``base/arrays.py`` an exemption. ``base/spec/`` is the lower of the two layers; it now imports nothing
from ``base/arrays.py`` at module scope, and every module in the package can read the shared default.
"""

import ast
from pathlib import Path

import pytest

import digitalearth

#: The package tree every module is read from.
SRC = Path(digitalearth.__file__).resolve().parent

#: For each band-ish parameter: the shared constant its default must name, and the modules that name may be
#: bound from. The plural is here because it is the same finding — ``bands=(1, 2, 3)`` was the composite's copy
#: of ``DEFAULT_COMPOSITE_BANDS`` — and one rule covering both keeps the pair from drifting apart again.
SHARED_DEFAULTS = {
    "band": (
        "DEFAULT_BAND",
        ("digitalearth.base.spec", "digitalearth.base.spec.selection"),
    ),
    "bands": ("DEFAULT_COMPOSITE_BANDS", ("digitalearth.base.stretch",)),
}


def _modules() -> list:
    """Every Python module in the package, ``__pycache__`` excluded.

    Returns:
        The paths, sorted, so the parametrize ids are stable between runs.
    """
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _module_id(path: Path) -> str:
    """Path-based, collision-free parametrize id (the package has several ``__init__.py`` files).

    Args:
        path: A module under :data:`SRC`.

    Returns:
        Its path relative to the package root, with forward slashes on every platform.
    """
    return str(path.relative_to(SRC)).replace("\\", "/")


def _package_of(module: Path) -> list:
    """The dotted package parts a relative import in `module` resolves against.

    Args:
        module: A module under :data:`SRC`.

    Returns:
        The parts of its own package — dropping the final one, which is right for both shapes: it strips the
        module name from ``base/arrays.py`` and ``__init__`` from ``base/__init__.py``, leaving
        ``digitalearth.base`` either way.
    """
    return list(module.relative_to(SRC.parent).with_suffix("").parts)[:-1]


def _dotted(module: Path) -> str:
    """The dotted name `module` is imported under.

    Args:
        module: A module under :data:`SRC`.

    Returns:
        Its importable name, with ``__init__`` collapsed onto its package — which is what lets the module
        *declaring* a shared constant satisfy the same rule as the modules importing it.
    """
    parts = list(module.relative_to(SRC.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _bound_names(tree: ast.AST, module: Path) -> dict:
    """Map each name imported into `module` to the dotted module it was taken from.

    Args:
        tree: The parsed module.
        module: Its path, used to resolve a relative import against its own package.

    Returns:
        ``{bound name: source module}``, covering lazy imports inside functions as well as module-level ones.
        A relative spelling is resolved, so ``from ...base.spec import DEFAULT_BAND`` reports the same source
        module as its absolute form.
    """
    package = _package_of(module)
    bound = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            head = (
                package[: len(package) - (node.level - 1)]
                if node.level > 1
                else package
            )
            source = ".".join(head + ([node.module] if node.module else []))
        else:
            source = node.module or ""
        for alias in node.names:
            bound[alias.asname or alias.name] = source
    return bound


def _band_defaults(tree: ast.AST) -> list:
    """Every band-ish parameter in `tree` that carries a default.

    Args:
        tree: A parsed module, or any AST holding function definitions.

    Returns:
        One ``(line, function, parameter, default node)`` tuple per parameter named in
        :data:`SHARED_DEFAULTS`, positional-only and keyword-only included.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        spec = node.args
        positional = spec.posonlyargs + spec.args
        padding = [None] * (len(positional) - len(spec.defaults))
        pairs = list(zip(positional, padding + list(spec.defaults)))
        pairs += list(zip(spec.kwonlyargs, spec.kw_defaults))
        for arg, default in pairs:
            if default is not None and arg.arg in SHARED_DEFAULTS:
                found.append((default.lineno, node.name, arg.arg, default))
    return found


def _is_number_written_out(node: ast.AST) -> bool:
    """Whether `node` is a band index spelled as a number rather than named.

    Args:
        node: The default expression of a band-ish parameter.

    Returns:
        ``True`` for an integer constant and for a tuple or list of them — the two shapes a copy of the shared
        default takes. ``bool`` is excluded: ``True`` is an ``int`` in Python, but a band default of ``True``
        is a different defect and :class:`~digitalearth.base.spec.Selection` already refuses it by name.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and all(
            _is_number_written_out(item) for item in node.elts
        )
    return False


def _offences(tree: ast.AST) -> list:
    """The band-ish defaults in `tree` that write a number out instead of naming the shared constant.

    Args:
        tree: A parsed module, or any AST holding function definitions.

    Returns:
        One ``"line N: func(param=default)"`` string per offence, ready to put in a failure message.
    """
    return [
        f"line {line}: {func}({param}={ast.unparse(node)})"
        for line, func, param, node in _band_defaults(tree)
        if _is_number_written_out(node)
    ]


def _parsed(module: Path) -> ast.AST:
    """Parse `module`, naming the file so a syntax error points at it.

    Args:
        module: A module under :data:`SRC`.

    Returns:
        Its AST.
    """
    return ast.parse(module.read_text(encoding="utf-8"), filename=str(module))


def test_the_scan_reads_the_whole_package():
    """Guard against every check below passing because the walk found nothing to read."""
    found = _modules()
    assert len(found) > 50, (
        f"expected the whole package under {SRC}, found {len(found)} modules"
    )


def test_the_scan_reads_every_tier():
    """``base/`` and all four backends must be in what the walk collects, not just the one being edited.

    Test scenario:
        The finding was that *every* tier had made its own copy of the default, so a guard that read one
        subtree would have cleared the other four. The tiers are asserted by name because a walk that
        silently stopped at ``base/`` would still satisfy the count above.
    """
    tiers = {
        p.relative_to(SRC).parts[0]
        for p in _modules()
        if len(p.relative_to(SRC).parts) > 1
    }
    for tier in ("base", "static", "interactive", "web", "three_d"):
        assert tier in tiers, (
            f"the walk never reaches {tier}/, so its band defaults go unchecked"
        )


def test_the_scan_finds_band_defaults_to_check():
    """The walk must actually collect band defaults, or the rule is being applied to an empty list."""
    collected = sum(len(_band_defaults(_parsed(module))) for module in _modules())
    assert collected > 20, (
        f"expected the package's band defaults, collected {collected}"
    )


@pytest.mark.parametrize("module", _modules(), ids=_module_id)
def test_no_band_default_is_a_number_written_out(module: Path):
    """No ``band``/``bands`` parameter may default to a number instead of the shared constant.

    Args:
        module: A module in the package.
    """
    offences = _offences(_parsed(module))
    assert not offences, (
        f"{module.relative_to(SRC.parent)} writes the band default out: {offences}. Import the shared "
        "constant instead — DEFAULT_BAND from digitalearth.base.spec, or DEFAULT_COMPOSITE_BANDS from "
        "digitalearth.base.stretch — so the default has one home."
    )


@pytest.mark.parametrize("module", _modules(), ids=_module_id)
def test_a_named_band_default_is_the_shared_constant(module: Path):
    """A band default that is a bare name must be the shared constant, bound from the module that owns it.

    Args:
        module: A module in the package.
    """
    tree = _parsed(module)
    bound = _bound_names(tree, module)
    own = _dotted(module)
    for line, func, param, node in _band_defaults(tree):
        if not isinstance(node, ast.Name):
            continue
        expected, sources = SHARED_DEFAULTS[param]
        where = f"{module.relative_to(SRC.parent)}:{line} {func}({param}={node.id})"
        assert node.id == expected, (
            f"{where} names {node.id}, not the shared {expected}"
        )
        origin = own if own in sources else bound.get(node.id)
        assert origin in sources, (
            f"{where} does not take {expected} from {' or '.join(sources)} — a module-local copy holding "
            "today's value would pass both == and is against the real constant, so the binding is the check."
        )


class TestTheRuleItself:
    """The rule, exercised on written-out snippets, so a future edit cannot quietly narrow or widen it."""

    def test_a_reintroduced_literal_is_reported(self):
        """The 34th site coming back as a 35th is the whole reason this module exists.

        Test scenario:
            The exact spelling that was removed, parsed on its own. Reported rather than merely counted, so
            the failure message can name the line.
        """
        found = _offences(
            ast.parse("def field(data, *, band: int = 1):\n    return band\n")
        )
        assert found == ["line 1: field(band=1)"], (
            f"a bare band literal went unreported: {found}"
        )

    def test_a_composite_tuple_written_out_is_reported(self):
        """``bands`` is the same finding in its plural form, so the rule reads a sequence of numbers too."""
        found = _offences(
            ast.parse("def rgb(data, bands=(3, 2, 1)):\n    return bands\n")
        )
        assert found == ["line 1: rgb(bands=(3, 2, 1))"], (
            f"a written-out composite went unreported: {found}"
        )

    def test_a_keyword_only_band_is_reached(self):
        """Most of the converted sites were keyword-only, so the walk must read past the ``*``."""
        found = _offences(
            ast.parse("def globe(self, *, cmap=None, band=2):\n    return band\n")
        )
        assert found == ["line 1: globe(band=2)"], (
            f"a keyword-only band was skipped: {found}"
        )

    @pytest.mark.parametrize(
        "source",
        [
            "from digitalearth.base.spec import DEFAULT_BAND\ndef field(data, band: int = DEFAULT_BAND): ...",
            "from digitalearth.base.stretch import DEFAULT_COMPOSITE_BANDS\n"
            "def rgb(data, bands=DEFAULT_COMPOSITE_BANDS): ...",
        ],
        ids=["single band", "composite"],
    )
    def test_naming_the_shared_constant_is_accepted(self, source: str):
        """The shape the package was converted to must pass, or the guard fails the fixed tree.

        Args:
            source: A snippet defaulting to one of the shared constants.
        """
        assert _offences(ast.parse(source)) == [], (
            f"the shared constant was reported as an offence: {source}"
        )

    @pytest.mark.parametrize(
        "source",
        [
            'def add_layer(self, *, band: str = "data"): ...',
            "def add_mesh(self, band=None): ...",
            "def read_array(self, band: int = ...): ...",
            "def barbs(self, band=True): ...",
        ],
        ids=["a field name", "unset", "a Protocol stub", "a bool"],
    )
    def test_a_default_that_is_not_a_number_is_left_alone(self, source: str):
        """A ``band`` default that is not a number is not a band index, so the rule has nothing to say.

        Args:
            source: A snippet whose ``band`` default means something other than an index.
        """
        assert _offences(ast.parse(source)) == [], (
            f"a non-numeric band default was reported: {source}"
        )

    @pytest.mark.parametrize(
        "source",
        [
            "def globe(self, level: int = 1): ...",
            "def animate(self, fps: int = 5): ...",
            "def field(self, band_count: int = 3): ...",
        ],
        ids=["level", "fps", "a different parameter"],
    )
    def test_a_parameter_that_is_not_a_band_keeps_its_literal(self, source: str):
        """This is a guard about one shared default, not a ban on integer defaults.

        Args:
            source: A snippet whose integer default belongs to a parameter this rule does not cover.
        """
        assert _offences(ast.parse(source)) == [], (
            f"an unrelated integer default was reported: {source}"
        )

    def test_a_locally_defined_constant_is_not_bound_from_the_shared_module(self):
        """The binding is what the name check turns on, since the values are indistinguishable.

        Test scenario:
            A module writing its own ``DEFAULT_BAND = 1`` passes ``==`` and ``is`` against the real constant
            — small integers are cached — so identity cannot tell them apart. What can is that nothing
            imported the name, which is what :func:`_bound_names` reports.
        """
        tree = ast.parse(
            "DEFAULT_BAND = 1\ndef field(data, band: int = DEFAULT_BAND): ...\n"
        )
        bound = _bound_names(tree, SRC / "static" / "maps" / "raster.py")
        assert "DEFAULT_BAND" not in bound, (
            f"a module-local assignment was read as an import: {bound}"
        )

    def test_a_relative_import_resolves_to_the_same_module(self):
        """A relative spelling of the import must satisfy the rule exactly as the absolute one does.

        Test scenario:
            ``static/maps/raster.py`` reaching ``digitalearth.base.spec`` writes three dots. Resolving it is
            what keeps the check from failing a module that did nothing wrong.
        """
        tree = ast.parse("from ...base.spec import DEFAULT_BAND\n")
        bound = _bound_names(tree, SRC / "static" / "maps" / "raster.py")
        assert bound["DEFAULT_BAND"] == "digitalearth.base.spec", (
            f"a relative import resolved to {bound['DEFAULT_BAND']!r}"
        )

    def test_the_declaring_module_needs_no_import(self):
        """``base/spec/selection.py`` defines ``DEFAULT_BAND``, so its own use of it is not an unbound name.

        Test scenario:
            :meth:`~digitalearth.base.spec.Selection.of` defaults to the constant it declares two lines
            above. A rule that demanded an import would fail the one module that could not possibly have one.
        """
        declaring = SRC / "base" / "spec" / "selection.py"
        assert _dotted(declaring) in SHARED_DEFAULTS["band"][1], (
            f"{_dotted(declaring)} is not among the modules a band default may be bound from"
        )


class TestTheSharedDefaultIsUsable:
    """What the constant must remain, as opposed to what it currently is.

    Its *value* is pinned behaviourally in ``tests/base/test_selection.py``, where ``Selection()`` is asserted
    to read band 1. Pinning it again here would defeat the point of the conversion — the value is meant to be
    changeable in one place. What must hold whatever it becomes is that it is still a band a reader can read.
    """

    def test_the_default_is_a_whole_number(self):
        """A band index reaches GDAL, which takes an ``int`` and nothing else."""
        from digitalearth.base.spec import DEFAULT_BAND

        assert isinstance(DEFAULT_BAND, int), (
            f"DEFAULT_BAND is a {type(DEFAULT_BAND).__name__}"
        )

    def test_the_default_is_not_a_bool(self):
        """``True`` is an ``int`` in Python and would read as band 1 by accident rather than by decision."""
        from digitalearth.base.spec import DEFAULT_BAND

        assert not isinstance(DEFAULT_BAND, bool), (
            "DEFAULT_BAND is a bool, which reads as a band by accident"
        )

    def test_the_default_is_a_one_based_band(self):
        """Bands are 1-based across this package's public surface, so 0 is never a readable default."""
        from digitalearth.base.spec import DEFAULT_BAND

        assert DEFAULT_BAND >= 1, (
            f"DEFAULT_BAND is {DEFAULT_BAND}, which no 1-based reader can read"
        )

    def test_a_selection_accepts_the_default(self):
        """The constant must survive the validation every selection goes through.

        Test scenario:
            :class:`~digitalearth.base.spec.Selection` refuses a band below 1 and a non-integer, so building
            one from the constant is the cheapest proof that a changed value is still a legal band rather
            than a number that only looks like one.
        """
        from digitalearth.base.spec import DEFAULT_BAND, Selection

        assert Selection.of(DEFAULT_BAND).first_band == DEFAULT_BAND, (
            "Selection does not accept DEFAULT_BAND as a band"
        )
