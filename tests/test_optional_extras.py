"""Guards on `[project.optional-dependencies]` — that each backend extra installs a working backend.

An extra is only exercised at install time, so a dependency that is missing (or newly wrong for an upgraded
renderer) surfaces as a runtime `ImportError` in a user's environment rather than as a failing test. These
checks read the declared requirements and assert the properties that a broken extra would violate.

The concrete regression behind this file is #158: the `3d` extra hand-listed pyvista's trame packages
(`trame`, `trame-vtk`, `trame-vuetify`) instead of asking pyvista for them. That list mirrored pyvista 0.48's
`jupyter` extra and drifted twice — it never included `nest-asyncio2`, which `Plotter.export_html` needs on
every version, and pyvista 0.49 moved trame support out to `trame-pyvista`, which the frozen list could not
know about. Depending on `pyvista[jupyter]` makes the set version-correct by construction.
"""
import functools
import pathlib
import tomllib

from packaging.requirements import Requirement

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Renderer stacks that a backend extra must obtain from its renderer's own extras rather than restate. Maps
#: the distribution that owns the stack to the package-name prefixes that belong to it.
OWNED_STACKS = {"pyvista": ("trame",)}


@functools.cache
def _extras() -> dict:
    """Return the parsed `[project.optional-dependencies]` table."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _requirements(extra: str) -> list:
    """Return the extra's requirements, parsed."""
    return [Requirement(spec) for spec in _extras()[extra]]


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


def test_3d_extra_does_not_hand_list_the_trame_stack():
    """No bare `trame*` requirement in the `3d` extra — pyvista's own extra owns those versions.

    Re-listing them here is what froze the set to pyvista 0.48's shape and broke `export_html` on 0.49.
    """
    restated = sorted(
        req.name
        for req in _requirements("3d")
        if any(req.name.startswith(prefix) for prefix in OWNED_STACKS["pyvista"])
    )
    assert not restated, (
        f"the `3d` extra restates pyvista's trame stack: {restated}. Those versions are pyvista's to pick — "
        "let `pyvista[jupyter]` supply them (#158)."
    )
