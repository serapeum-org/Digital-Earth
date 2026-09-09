"""``digitalearth.base`` must never reach a rendering engine — directly or through a backend.

``base/`` holds the logic shared by every backend (static/matplotlib, interactive/HoloViz, 3-D/PyVista and
web/MapLibre). The moment one of those engines is reachable from there, the shared layer stops being shared: an
``import digitalearth.base.symbology`` would drag a renderer into the web backend, and the split that this
subpackage exists to express quietly rots back into the old flat layout.

Three things are rejected, and the second is the one that actually bites in practice:

1. **A rendering engine imported directly** — cleopatra, pyvista, holoviews, maplibre, and the rest of
   :data:`BANNED_PACKAGES`.
2. **A sibling backend imported at all** — ``digitalearth.static``/``interactive``/``web``/``three_d``. This is
   how the coupling actually forms: one ``from digitalearth.static.figures import fig_of`` passes an
   engine-only check while dragging ``matplotlib.pyplot``, cleopatra and the whole static backend into every
   consumer of ``base/``. It is precisely the shape this restructure untangled (``interactive/charts.py``
   reaching into ``charts.py``; ``_arrays.fig_of`` blocking the split), so it is the shape most likely to
   return.
3. **Rendering surface from an otherwise-allowed package** — ``matplotlib.pyplot``/``figure``/``axes``,
   ``mpl_toolkits``, ``pylab``.

**The matplotlib carve-out.** ``matplotlib.colors`` / ``matplotlib.colormaps`` / ``matplotlib.cm`` are the
de-facto standard *colour* library — a named-colormap registry and hex conversion that return plain data, not
figures. :func:`digitalearth.base.symbology.categorical_colors` uses them to turn ``"tab10"`` into ``#rrggbb``
strings, which is what keeps the web, interactive and static backends colouring one ``cmap`` identically; that
is the whole reason the function is shared. So colour is allowed and *rendering* is not.

Relative imports are resolved against the module's own package before being tested, so ``from ..static.figures
import fig_of`` is caught exactly like its absolute spelling.

This walks every module under ``base/`` with :mod:`ast` — no importing, so it runs in the lean ``dev``
environment where none of the optional engines is installed, and it sees lazy imports inside functions.
**Limitation:** it reads ``import``/``from`` statements. A string-based import
(``importlib.import_module("cleopatra")``) is flagged separately by
:func:`test_no_dynamic_imports_in_base`, which is a blunter check but closes the obvious hole.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
BASE = SRC / "digitalearth" / "base"

#: Rendering engines and their helpers. None of these may be imported from ``digitalearth.base`` at all.
#: ``mpl_toolkits`` and ``pylab`` matter as much as the obvious ones: both ship with matplotlib (so they are
#: always importable) and both are pure rendering surface, which would defeat the colour carve-out below.
BANNED_PACKAGES = {
    "altair",
    "bokeh",
    "cleopatra",
    "datashader",
    "folium",
    "geoviews",
    "geovista",
    "holoviews",
    "hvplot",
    "ipyleaflet",
    "lonboard",
    "maplibre",
    "mpl_toolkits",
    "panel",
    "plotly",
    "pydeck",
    "pylab",
    "pyvista",
    "trame",
    "vtk",
}

#: Sibling backends. ``base/`` sits *below* these, so importing one inverts the layering and pulls that
#: backend's engine in behind it. ``digitalearth.base`` and ``digitalearth.ops`` are fine.
BANNED_SUBPACKAGES = {
    "digitalearth.static",
    "digitalearth.interactive",
    "digitalearth.web",
    "digitalearth.three_d",
}

#: The only matplotlib surface allowed in ``base/`` — colour data, never figures. See the module docstring.
ALLOWED_MATPLOTLIB = {"matplotlib.colors", "matplotlib.colormaps", "matplotlib.cm"}


def _modules() -> list[Path]:
    """Every Python module under ``base/``, ``__pycache__`` excluded."""
    return sorted(p for p in BASE.rglob("*.py") if "__pycache__" not in p.parts)


def _module_id(path: Path) -> str:
    """Path-based, collision-free parametrize id (``base/`` has three ``__init__.py`` files)."""
    return str(path.relative_to(BASE)).replace("\\", "/")


def _package_of(module: Path) -> list[str]:
    """The dotted package parts a relative import in ``module`` resolves against.

    Dropping the final part covers both shapes: for ``base/symbology.py`` it strips the module name to leave
    ``digitalearth.base``, and for ``base/__init__.py`` it strips ``__init__`` to leave the same thing — which
    is right, because a relative import written in a package's ``__init__`` is already anchored at it.
    """
    return list(module.relative_to(SRC).with_suffix("").parts)[:-1]


def _imported_targets(tree: ast.AST, module: Path) -> set[str]:
    """Every module path imported in ``tree``, including lazy imports inside functions.

    Relative imports are resolved against ``module``'s own package, so ``from ..static import x`` is reported
    as ``digitalearth.static``. ``from matplotlib import colormaps`` is reported as both ``matplotlib`` and
    ``matplotlib.colormaps`` so the carve-out can tell it from a bare ``import matplotlib``.
    """
    package = _package_of(module)
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = (
                    package[: len(package) - (node.level - 1)]
                    if node.level > 1
                    else package
                )
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if not prefix:
                continue
            targets.add(prefix)
            targets.update(f"{prefix}.{alias.name}" for alias in node.names)
    return targets


def _is_under(target: str, allowed: set[str]) -> bool:
    """True when ``target`` is one of ``allowed`` or a name imported out of one."""
    return any(target == item or target.startswith(f"{item}.") for item in allowed)


def _offenders(targets: set[str]) -> set[str]:
    """The subset of ``targets`` that ``base/`` is not allowed to import."""
    bad = set()
    allowed_matplotlib_present = any(_is_under(t, ALLOWED_MATPLOTLIB) for t in targets)
    for target in targets:
        root = target.split(".")[0]
        if root in BANNED_PACKAGES:
            bad.add(target)
        elif _is_under(target, BANNED_SUBPACKAGES):
            bad.add(target)
        elif root == "matplotlib" and not _is_under(target, ALLOWED_MATPLOTLIB):
            # `from matplotlib import colormaps` yields both "matplotlib" and "matplotlib.colormaps";
            # the bare parent is only an offence when nothing allowed was taken from it.
            if target != "matplotlib" or not allowed_matplotlib_present:
                bad.add(target)
    return bad


def _dynamic_import_literals(tree: ast.AST) -> set[str]:
    """String literals passed to ``importlib.import_module`` / ``__import__`` — imports the walk cannot see."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
        if name in {"import_module", "__import__"} and isinstance(
            node.args[0], ast.Constant
        ):
            if isinstance(node.args[0].value, str):
                found.add(node.args[0].value)
    return found


def test_base_package_is_populated():
    """Guard against the walk silently passing because it found nothing."""
    assert len(_modules()) > 5, (
        f"expected several modules under {BASE}, found {len(_modules())}"
    )


@pytest.mark.parametrize("module", _modules(), ids=_module_id)
def test_module_imports_no_rendering_engine(module: Path):
    """No module under ``base/`` may import a rendering engine or a sibling backend."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    offenders = _offenders(_imported_targets(tree, module))
    assert not offenders, (
        f"{module.relative_to(SRC)} imports {sorted(offenders)}. "
        "base/ is engine-neutral shared logic — move the engine-specific part into the backend that needs it. "
        "Only matplotlib's colour registry is exempt; see this module's docstring."
    )


@pytest.mark.parametrize("module", _modules(), ids=_module_id)
def test_no_dynamic_imports_in_base(module: Path):
    """A string-based import would slip past the AST walk, so ``base/`` may not use one at all."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    dynamic = _dynamic_import_literals(tree)
    assert not dynamic, (
        f"{module.relative_to(SRC)} imports {sorted(dynamic)} dynamically. The neutrality guard reads import "
        "statements, so a string-based import would not be checked — import it normally instead."
    )


class TestOffenderRules:
    """The rules themselves, so a future edit cannot quietly widen what base/ may import."""

    def test_matplotlib_rendering_surface_is_rejected(self):
        """The colour carve-out must not become 'matplotlib is fine in base/'."""
        assert _offenders({"matplotlib.pyplot"}) == {"matplotlib.pyplot"}
        assert _offenders({"matplotlib.figure", "matplotlib.axes"}) == {
            "matplotlib.figure",
            "matplotlib.axes",
        }
        assert _offenders({"matplotlib"}) == {"matplotlib"}

    def test_matplotlib_colour_registry_is_allowed(self):
        """The three colour modules, and names taken out of them, are the whole carve-out."""
        assert _offenders({"matplotlib", "matplotlib.colormaps"}) == set()
        assert _offenders({"matplotlib.colors", "matplotlib.colors.to_hex"}) == set()

    @pytest.mark.parametrize(
        "target",
        [
            "digitalearth.static",
            "digitalearth.static.figures",
            "digitalearth.interactive.charts",
            "digitalearth.web.base",
            "digitalearth.three_d.globe",
        ],
    )
    def test_sibling_backends_are_rejected(self, target):
        """Importing a backend inverts the layering and pulls its engine in behind it.

        Args:
            target: A dotted path into one of the four backend subpackages.
        """
        assert _offenders({target}) == {target}, (
            f"{target} should be rejected from base/"
        )

    @pytest.mark.parametrize(
        "target",
        ["digitalearth.base.arrays", "digitalearth.ops.plugins", "numpy", "pandas"],
    )
    def test_neutral_dependencies_are_allowed(self, target):
        """``base/`` may import itself, ops, and plain data libraries.

        Args:
            target: A dotted path that must remain importable from ``base/``.
        """
        assert _offenders({target}) == set(), f"{target} should be allowed in base/"

    @pytest.mark.parametrize(
        "engine", ["mpl_toolkits", "pylab", "vtk", "plotly", "folium", "cleopatra"]
    )
    def test_every_banned_engine_is_rejected(self, engine):
        """Each renderer on the deny-list is actually rejected.

        Args:
            engine: A top-level package name from :data:`BANNED_PACKAGES`.
        """
        assert _offenders({f"{engine}.thing"}) == {f"{engine}.thing"}, (
            f"{engine} should be banned in base/"
        )

    def test_relative_imports_are_resolved_and_checked(self):
        """``from ..static.figures import fig_of`` must be caught like its absolute spelling."""
        source = "from ..static.figures import fig_of\n"
        module = BASE / "symbology.py"
        targets = _imported_targets(ast.parse(source), module)
        assert "digitalearth.static.figures" in targets, (
            f"relative import not resolved: {sorted(targets)}"
        )
        assert _offenders(targets), "a relative backend import must be rejected"

    def test_relative_sibling_import_inside_base_is_allowed(self):
        """A relative import that stays inside ``base/`` resolves and is permitted."""
        targets = _imported_targets(
            ast.parse("from .arrays import finite\n"), BASE / "chartdata.py"
        )
        assert "digitalearth.base.arrays" in targets, (
            f"expected base sibling, got {sorted(targets)}"
        )
        assert _offenders(targets) == set(), (
            "a base-internal relative import must be allowed"
        )

    @pytest.mark.parametrize(
        "module, expected",
        [
            (BASE / "symbology.py", ["digitalearth", "base"]),
            (BASE / "__init__.py", ["digitalearth", "base"]),
            (BASE / "sources" / "__init__.py", ["digitalearth", "base", "sources"]),
            (BASE / "sources" / "source.py", ["digitalearth", "base", "sources"]),
        ],
    )
    def test_relative_imports_anchor_at_the_right_package(self, module, expected):
        """A module and its package's ``__init__`` anchor relative imports at the same place.

        Args:
            module: The file a relative import is written in.
            expected: The dotted package parts it should resolve against.

        Test scenario:
            Getting this wrong by one level would silently mis-resolve every relative import — and, because a
            mis-resolved path matches no rule, would fail *open* rather than loudly.
        """
        assert _package_of(module) == expected, (
            f"{module.name} anchors at {_package_of(module)}"
        )

    def test_deep_relative_import_climbs_out_of_base(self):
        """``from ..static import x`` in a nested module still resolves to the backend and is rejected."""
        targets = _imported_targets(
            ast.parse("from ...static import figures\n"), BASE / "sources" / "source.py"
        )
        assert "digitalearth.static" in targets, (
            f"expected digitalearth.static, got {sorted(targets)}"
        )
        assert _offenders(targets), (
            "a backend reached by a deep relative import must be rejected"
        )

    def test_dynamic_import_literals_are_detected(self):
        """Both string-import spellings are picked up."""
        source = 'import importlib\nimportlib.import_module("cleopatra")\n__import__("matplotlib.pyplot")\n'
        assert _dynamic_import_literals(ast.parse(source)) == {
            "cleopatra",
            "matplotlib.pyplot",
        }
