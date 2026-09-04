"""Guard: every pixi task the CI workflow and the pre-commit hooks invoke is actually defined.

`pixi run -e <env> <name>` accepts either a task from `[tool.pixi.tasks]` or a bare executable in the
environment, so a task that is renamed or deleted fails only at run time — in CI, or in a hook the author
skipped. That is the drift this file exists to catch: it resolves every reference in
`.github/workflows/tests.yml` and `.pre-commit-config.yaml` against the task table, ignoring the handful of
references that are executables rather than tasks.
"""
import pathlib
import re
import tomllib

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Names invoked through `pixi run` that are executables in the environment, not project tasks. Anything
#: outside this set is required to be a defined task.
EXECUTABLES = {"pip", "playwright", "pytest", "python"}

#: `pixi run [--frozen] -e <env> <name>`, capturing the environment and the name being run.
INVOCATION = re.compile(r"pixi run (?:--frozen )?-e (?P<env>[a-z0-9]+) (?P<name>[a-z0-9][a-z0-9-]*)")


def _tasks() -> dict:
    """Return the `[tool.pixi.tasks]` table from pyproject.toml."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["pixi"]["tasks"]


def _environments() -> set:
    """Return the environment names declared in `[tool.pixi.environments]`."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return set(tomllib.load(handle)["tool"]["pixi"]["environments"])


def _invocations(relative_path: str) -> list:
    """Return the (env, name) pairs invoked through `pixi run` in the given repo-relative file."""
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    return [(m.group("env"), m.group("name")) for m in INVOCATION.finditer(text)]


class TestPixiTaskReferences:
    """Tests that CI and hook invocations resolve against the declared pixi tasks and environments."""

    def test_workflow_task_references_are_defined(self):
        """Every non-executable name the CI workflow runs through pixi is a defined task.

        Test scenario:
            Each `pixi run -e <env> <name>` in .github/workflows/tests.yml either names an executable
            (pytest, pip, playwright, python) or a task in [tool.pixi.tasks]. A renamed or deleted task
            would otherwise surface only as a CI failure.
        """
        tasks = _tasks()
        missing = sorted({name for _, name in _invocations(".github/workflows/tests.yml")
                          if name not in EXECUTABLES and name not in tasks})
        assert not missing, f"workflow runs undefined pixi task(s): {missing}; defined: {sorted(tasks)}"

    def test_pre_commit_task_references_are_defined(self):
        """Every non-executable name the pre-commit hooks run through pixi is a defined task.

        Test scenario:
            Each `pixi run --frozen -e <env> <name>` in .pre-commit-config.yaml resolves the same way as
            the workflow references, so a task rename cannot silently disable a hook.
        """
        tasks = _tasks()
        missing = sorted({name for _, name in _invocations(".pre-commit-config.yaml")
                          if name not in EXECUTABLES and name not in tasks})
        assert not missing, f"pre-commit runs undefined pixi task(s): {missing}; defined: {sorted(tasks)}"

    def test_referenced_environments_are_declared(self):
        """Every environment CI and the hooks activate is declared in [tool.pixi.environments].

        Test scenario:
            `pixi run -e <env>` against an undeclared environment fails at run time; assert each env named
            in the workflow and the hook config exists.
        """
        declared = _environments()
        used = {env for env, _ in _invocations(".github/workflows/tests.yml")}
        used |= {env for env, _ in _invocations(".pre-commit-config.yaml")}
        undeclared = sorted(used - declared)
        assert not undeclared, f"undeclared pixi environment(s): {undeclared}; declared: {sorted(declared)}"

    def test_the_checks_added_for_the_migration_are_wired_up(self):
        """The lint and doctests tasks exist and are both invoked by CI.

        Test scenario:
            These two tasks are the gates that cover code CI cannot otherwise reach — undefined names in
            notebook cells that never execute, and the doctests, which `main` does not collect. A task left
            defined but unreferenced would silently stop gating anything.
        """
        tasks = _tasks()
        invoked = {name for _, name in _invocations(".github/workflows/tests.yml")}
        for name in ("lint", "doctests"):
            assert name in tasks, f"the {name!r} task is not defined in [tool.pixi.tasks]"
            assert name in invoked, f"the {name!r} task is defined but the CI workflow never runs it"
