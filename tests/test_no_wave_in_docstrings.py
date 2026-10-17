"""No user-visible docstring names a wave number — a wave renumbers, an order does not (#317, #338).

`tests/base/test_contract.py` already refuses a wave number inside the contract *data* — a `PENDING` reason or
a `Method.builds_in`. That guard reads `contract.py` alone, and never the prose a reader meets first: the
module docstrings and the public class/function docstrings across `src/digitalearth`. #338 found six of those
naming a wave — `DataRef`'s "Wave 2", `render_compat`'s "Wave-1"/"Wave 3", and more — the exact rot #317 named
(a number that keeps its digits and loses its meaning the moment a wave is inserted ahead of it and the rest
renumber), sitting in the one place the contract-data guard cannot see. This scans that prose and refuses it.
"""

import ast
import re
from pathlib import Path

import digitalearth

#: A wave reference in prose. `tests/base/test_contract.py` uses `\bwave\s+\d+`; this widens the separator to
#: `[\s-]+` so the hyphen form (`Wave-1`, `Wave-0`) the codebase also grew is caught alongside the space form.
WAVE_REFERENCE = re.compile(r"\bwaves?[\s-]+\d+", re.IGNORECASE)

#: The package tree whose prose is scanned — resolved from the import so the scan does not depend on the cwd.
_PACKAGE_ROOT = Path(digitalearth.__file__).parent


def _documented_nodes(tree):
    """Yield the module node and every public class/function in it — the nodes a user reads a docstring on.

    Args:
        tree: The parsed module.

    Yields:
        The `ast.Module`, then each `ClassDef`/`FunctionDef`/`AsyncFunctionDef` whose name does not start with
        an underscore, at any nesting depth. A private symbol is skipped: its docstring is not user-facing.
    """
    yield tree
    public_defs = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, public_defs) and not node.name.startswith("_"):
            yield node


def _waves_named_in_docstrings():
    """Return every docstring across the package that names a wave number, as `[(path, symbol)]`.

    Returns:
        One row per offending docstring — the path relative to the package root and the symbol it documents
        (`<module>` for a module docstring) — empty for a package whose prose names no wave number.
    """
    offenders = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _documented_nodes(tree):
            doc = ast.get_docstring(node)
            if doc and WAVE_REFERENCE.search(doc):
                symbol = "<module>" if isinstance(node, ast.Module) else node.name
                offenders.append((path.relative_to(_PACKAGE_ROOT).as_posix(), symbol))
    return offenders


def test_no_docstring_names_a_wave():
    """A wave number in user-visible prose rots exactly as one in the contract data does (#317, #338)."""
    offenders = _waves_named_in_docstrings()
    listed = "; ".join(f"{path} -> {symbol}" for path, symbol in offenders)
    assert offenders == [], (
        f"a user-visible docstring names a wave, which renumbers when the plan moves: {listed}. "
        "Name the roadmap order that builds it, or use phrasing that carries no wave number."
    )
