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
from packaging.utils import canonicalize_name

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Canonical package names that belong to pyvista's own `jupyter` extra and must not be restated by us. Matched
#: as prefixes against canonicalised names, so `Trame-VTK` and `trame_vtk` are caught alongside `trame-vtk`.
PYVISTA_OWNED_PREFIXES = ("trame",)


def _pyproject() -> dict:
    """Read and parse the project's pyproject.toml.

    Parsed per call rather than cached, so no caller can mutate a copy the next one will read.

    Returns:
        The parsed document.
    """
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirements(extra: str) -> list:
    """Return one extra's requirements, parsed.

    Args:
        extra: Name of the key in `[project.optional-dependencies]`.

    Returns:
        The extra's requirements as `packaging.requirements.Requirement` objects.
    """
    return [
        Requirement(spec)
        for spec in _pyproject()["project"]["optional-dependencies"][extra]
    ]


def _declaration_sites(data: dict = None) -> dict:
    """Return every declared package, canonicalised, mapped to the list of tables declaring it.

    Covers all five places a pin can live: `[project].dependencies`, every `[project.optional-dependencies]`
    extra, `[dependency-groups]` (which is where pixi derives the `dev`, `docs` and `notebook` features, so a
    pin there reaches every environment), and the pixi dependency tables including their per-feature and
    per-platform variants.

    Args:
        data: A parsed pyproject document. Defaults to this project's, which is what the guards read;
            tests pass a synthetic one to exercise table shapes the repo does not currently use.

    Returns:
        Mapping of canonical package name to the tables it appears in.
    """
    data = _pyproject() if data is None else data
    sites: dict = {}

    def record(name: str, table: str) -> None:
        sites.setdefault(canonicalize_name(name), []).append(table)

    for spec in data["project"].get("dependencies", []):
        record(Requirement(spec).name, "[project].dependencies")
    for extra, specs in data["project"].get("optional-dependencies", {}).items():
        for spec in specs:
            record(Requirement(spec).name, f"[project.optional-dependencies].{extra!r}")
    for group, specs in data.get("dependency-groups", {}).items():
        for spec in specs:
            if isinstance(spec, str):
                record(Requirement(spec).name, f"[dependency-groups].{group!r}")

    def walk_pixi(body: dict, label: str) -> None:
        """Record a pixi table's dependency keys, recursing through its per-platform `target` tables."""
        for key in ("dependencies", "pypi-dependencies"):
            for name in body.get(key, {}):
                record(name, f"{label}.{key}")
        for platform, target in body.get("target", {}).items():
            walk_pixi(target, f"{label}.target.{platform}")

    pixi = data.get("tool", {}).get("pixi", {})
    walk_pixi(pixi, "[tool.pixi]")
    for feature, body in pixi.get("feature", {}).items():
        walk_pixi(body, f"[tool.pixi.feature.{feature}]")
    return sites


def test_3d_extra_takes_pyvistas_trame_stack_from_pyvista():
    """The `3d` extra must pull pyvista's export stack via `pyvista[jupyter]`, not a hand-written list.

    `Scene3DBase.export_html` runs on trame/vtk.js, and which packages provide that differs by pyvista
    version. Requesting pyvista's own `jupyter` extra is what keeps the set correct across an upgrade.
    """
    pyvista = [
        req for req in _requirements("3d") if canonicalize_name(req.name) == "pyvista"
    ]
    assert pyvista, "the `3d` extra must declare pyvista"
    assert "jupyter" in pyvista[0].extras, (
        "the `3d` extra must request `pyvista[jupyter]` — that extra is pyvista's trame/vtk.js export stack "
        "(#158). Without it `export_html` is missing nest-asyncio2, and trame-pyvista on pyvista >=0.49."
    )


def test_no_dependency_table_hand_lists_the_trame_stack():
    """No `trame*` pin anywhere in the project — pyvista's own extra owns those versions.

    Re-listing them is what froze the set to pyvista 0.48's shape and broke `export_html` on 0.49. Checking
    only the `3d` extra would miss the same pin reappearing in the base dependencies, in another extra, in a
    dependency group (which reaches every pixi environment) or in a pixi table, so every site is scanned and
    names are canonicalised first — `Trame-VTK` resolves to the same distribution as `trame-vtk`.
    """
    restated = {
        name: tables
        for name, tables in _declaration_sites().items()
        if any(name.startswith(prefix) for prefix in PYVISTA_OWNED_PREFIXES)
    }
    assert not restated, (
        f"these tables restate pyvista's trame stack: {restated}. Those versions are pyvista's to pick — let "
        "`pyvista[jupyter]` supply them (#158)."
    )


def test_declaration_sites_reads_every_table_shape():
    """`_declaration_sites` reaches per-platform pixi tables and skips a dependency group's include entries.

    Args:
        None. A synthetic document is passed straight to the helper, covering the table shapes this repo does
        not currently use so the scan is proven against them before one appears.

    Test scenario:
        A pin is findable from all five sites, `[tool.pixi.feature.<f>.target.<platform>]` included, and names
        are canonicalised on the way in. A `[dependency-groups]` entry can also be a table
        (`{include-group = ...}`) rather than a requirement string; those carry no package name and must be
        skipped rather than crash the scan.
    """
    document = {
        "project": {
            "dependencies": ["Base_Pkg >=1"],
            "optional-dependencies": {"extra": ["Extra.Pkg >=1"]},
        },
        "dependency-groups": {"dev": ["Group-Pkg >=1", {"include-group": "other"}]},
        "tool": {
            "pixi": {
                "pypi-dependencies": {"Root_Pkg": "*"},
                "feature": {
                    "f": {"target": {"win-64": {"dependencies": {"Target-Pkg": "*"}}}}
                },
            }
        },
    }
    sites = _declaration_sites(document)
    assert set(sites) == {
        "base-pkg",
        "extra-pkg",
        "group-pkg",
        "root-pkg",
        "target-pkg",
    }, f"every table must be scanned and its names canonicalised, got {sorted(sites)}"
    assert sites["target-pkg"] == [
        "[tool.pixi.feature.f].target.win-64.dependencies"
    ], f"a per-platform pin must name its own table, got {sites['target-pkg']}"
    assert sites["group-pkg"] == ["[dependency-groups].'dev'"], (
        f"a dependency-group pin must be recorded, got {sites['group-pkg']}"
    )


def test_declaration_sites_records_every_table_a_package_appears_in():
    """A package pinned twice reports both tables, not just the last one seen.

    Args:
        None. The synthetic document pins one package in two places.

    Test scenario:
        Overwriting would hide one of the sites a maintainer has to edit to remove the pin.
    """
    document = {
        "project": {
            "dependencies": ["dup >=1"],
            "optional-dependencies": {"extra": ["dup >=1"]},
        }
    }
    assert _declaration_sites(document)["dup"] == [
        "[project].dependencies",
        "[project.optional-dependencies].'extra'",
    ], "both declaration sites must be reported"
