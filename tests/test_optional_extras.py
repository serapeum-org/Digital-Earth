"""Guards on the project's dependency declarations — that each backend extra installs a working backend.

An extra is only exercised at install time, so a dependency that is missing (or newly wrong for an upgraded
renderer) surfaces as a runtime `ImportError` in a user's environment rather than as a failing test. These
checks read the declared requirements and assert the properties that a broken extra would violate.

The concrete regression behind this file is #158: the `3d` extra hand-listed pyvista's trame packages
(`trame`, `trame-vtk`, `trame-vuetify`) instead of asking pyvista for them. That list mirrored pyvista 0.48's
`jupyter` extra and drifted twice — it never included `nest-asyncio2`, which `Plotter.export_html` needs on
every version, and pyvista 0.49 moved trame support out to `trame-pyvista`, which the frozen list could not
know about. Depending on `pyvista[jupyter]` makes the set version-correct by construction.
"""
import pathlib
import tomllib

from packaging.requirements import Requirement

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Package-name prefixes that belong to pyvista's own `jupyter` extra and must not be restated by us. Re-pinning
#: any of them anywhere in the project re-creates the drift #158 describes, so every table is checked, not just
#: the `3d` extra.
PYVISTA_OWNED_PREFIXES = ("trame",)


def _pyproject() -> dict:
    """Return the parsed pyproject.toml (parsed per call, so no caller can mutate a shared copy)."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirements(extra: str) -> list:
    """Return one extra's requirements, parsed."""
    return [Requirement(spec) for spec in _pyproject()["project"]["optional-dependencies"][extra]]


def _every_declared_package() -> dict:
    """Return every declared package name, mapped to the table it was declared in.

    Covers the base `[project].dependencies`, every `[project.optional-dependencies]` extra, and the pixi
    dependency tables (`[tool.pixi.pypi-dependencies]`, `[tool.pixi.dependencies]` and any per-feature
    equivalents) — the four places a stray pin could reappear.
    """
    data = _pyproject()
    found = {}
    for spec in data["project"].get("dependencies", []):
        found[Requirement(spec).name] = "[project].dependencies"
    for extra, specs in data["project"].get("optional-dependencies", {}).items():
        for spec in specs:
            found[Requirement(spec).name] = f"[project.optional-dependencies].{extra!r}"
    pixi = data.get("tool", {}).get("pixi", {})
    tables = [("[tool.pixi.%s]" % key, pixi.get(key, {})) for key in ("dependencies", "pypi-dependencies")]
    for feature, body in pixi.get("feature", {}).items():
        for key in ("dependencies", "pypi-dependencies"):
            tables.append((f"[tool.pixi.feature.{feature}.{key}]", body.get(key, {})))
    for label, table in tables:
        for name in table:
            found[name] = label
    return found


def test_3d_extra_takes_pyvistas_trame_stack_from_pyvista():
    """The `3d` extra must pull pyvista's export stack via `pyvista[jupyter]`, not a hand-written list.

    `Scene3DBase.export_html` runs on trame/vtk.js, and which packages provide that differs by pyvista
    version. Requesting pyvista's own `jupyter` extra is what keeps the set correct across an upgrade.
    """
    pyvista = [req for req in _requirements("3d") if req.name == "pyvista"]
    assert pyvista, "the `3d` extra must declare pyvista"
    assert "jupyter" in pyvista[0].extras, (
        "the `3d` extra must request `pyvista[jupyter]` — that extra is pyvista's trame/vtk.js export stack "
        "(#158). Without it `export_html` is missing nest-asyncio2, and trame-pyvista on pyvista >=0.49."
    )


def test_no_dependency_table_hand_lists_the_trame_stack():
    """No `trame*` pin anywhere in the project — pyvista's own extra owns those versions.

    Re-listing them is what froze the set to pyvista 0.48's shape and broke `export_html` on 0.49. Checking
    only the `3d` extra would miss the same pin reappearing in the base dependencies, in another extra, or in
    a pixi table, so every declaration site is scanned.
    """
    restated = {
        name: table
        for name, table in _every_declared_package().items()
        if any(name.startswith(prefix) for prefix in PYVISTA_OWNED_PREFIXES)
    }
    assert not restated, (
        f"these tables restate pyvista's trame stack: {restated}. Those versions are pyvista's to pick — let "
        "`pyvista[jupyter]` supply them (#158)."
    )
