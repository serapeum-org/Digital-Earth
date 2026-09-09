"""``digitalearth.base`` must never import a rendering engine.

``base/`` holds the logic shared by every backend (static/matplotlib, interactive/HoloViz, 3-D/PyVista and
web/MapLibre). The moment one of those engines is imported there, the shared layer stops being shared: an
``import digitalearth.base.symbology`` would drag a renderer into the web backend, and the split that this
subpackage exists to express quietly rots back into the old flat layout.

**The matplotlib carve-out.** ``matplotlib.colors`` / ``matplotlib.colormaps`` / ``matplotlib.cm`` are the
de-facto standard *colour* library — a named-colormap registry and hex conversion that return plain data, not
figures. :func:`digitalearth.base.symbology.categorical_colors` uses them to turn ``"tab10"`` into ``#rrggbb``
strings, which is what keeps the web, interactive and static backends colouring one ``cmap`` identically; that
is the whole reason the function is shared. So colour is allowed and *rendering* is not: ``pyplot``,
``figure``, ``axes`` and the backends are banned, as is a bare ``import matplotlib``.

This test walks every module under ``base/`` with :mod:`ast` — no importing, so it runs in the lean ``dev``
environment where none of the optional engines is installed, and it catches lazy function-level imports too.
"""

import ast
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1] / "src" / "digitalearth" / "base"

#: Rendering engines and their helpers. None of these may be imported from ``digitalearth.base`` at all.
BANNED_PACKAGES = {
    "cleopatra",
    "pyvista",
    "geovista",
    "trame",
    "holoviews",
    "geoviews",
    "hvplot",
    "panel",
    "datashader",
    "bokeh",
    "maplibre",
    "lonboard",
}

#: The only matplotlib surface allowed in ``base/`` — colour data, never figures. See the module docstring.
ALLOWED_MATPLOTLIB = {"matplotlib.colors", "matplotlib.colormaps", "matplotlib.cm"}


def _modules() -> list[Path]:
    """Every Python module under ``base/``, ``__pycache__`` excluded."""
    return sorted(p for p in BASE.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_targets(tree: ast.AST) -> set[str]:
    """Every module path imported in ``tree``, including lazy imports inside functions.

    ``from matplotlib import colormaps`` is reported as ``matplotlib.colormaps`` so the carve-out can tell it
    apart from a bare ``import matplotlib``.
    """
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.add(node.module)
            targets.update(f"{node.module}.{alias.name}" for alias in node.names)
    return targets


def _is_allowed_matplotlib(target: str) -> bool:
    """True for an allowed colour module or a name imported out of one (``matplotlib.colors.to_hex``)."""
    return any(target == allowed or target.startswith(f"{allowed}.") for allowed in ALLOWED_MATPLOTLIB)


def _offenders(targets: set[str]) -> set[str]:
    """The subset of ``targets`` that ``base/`` is not allowed to import."""
    bad = set()
    allowed_present = any(_is_allowed_matplotlib(t) for t in targets)
    for target in targets:
        root = target.split(".")[0]
        if root in BANNED_PACKAGES:
            bad.add(target)
        elif root == "matplotlib" and not _is_allowed_matplotlib(target):
            # `from matplotlib import colormaps` yields both "matplotlib" and "matplotlib.colormaps";
            # the bare parent is only an offence when nothing allowed was taken from it.
            if target != "matplotlib" or not allowed_present:
                bad.add(target)
    return bad


def test_base_package_is_populated():
    """Guard against the walk silently passing because it found nothing."""
    assert len(_modules()) > 5, f"expected several modules under {BASE}, found {len(_modules())}"


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_module_imports_no_rendering_engine(module: Path):
    """No module under ``base/`` may import a rendering engine."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    offenders = _offenders(_imported_targets(tree))
    assert not offenders, (
        f"{module.relative_to(BASE.parents[2])} imports {sorted(offenders)}. "
        "base/ is engine-neutral shared logic — move the engine-specific part into the backend that needs it. "
        "Only matplotlib's colour registry is exempt; see this module's docstring."
    )


def test_matplotlib_rendering_surface_is_still_rejected():
    """The carve-out must not widen into 'matplotlib is fine in base/'."""
    assert _offenders({"matplotlib.pyplot"}) == {"matplotlib.pyplot"}
    assert _offenders({"matplotlib.figure", "matplotlib.axes"}) == {"matplotlib.figure", "matplotlib.axes"}
    assert _offenders({"matplotlib"}) == {"matplotlib"}
    assert _offenders({"matplotlib", "matplotlib.colormaps"}) == set()
