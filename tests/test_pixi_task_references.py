"""Guard: every pixi task the CI workflows and the pre-commit hooks invoke is actually defined.

`pixi run -e <env> <name>` accepts either a task from `[tool.pixi.tasks]` or a bare executable in the
environment, so a task that is renamed or deleted fails only at run time — in CI, or in a hook the author
skipped. That is the drift this file exists to catch.

Invocations are read from the **parsed** YAML (a step's `run:` script, a hook's `entry:`), never from the raw
file text: a `pixi run …` mention inside a comment would otherwise satisfy these assertions while the real
step was gone. Matrix environments (`-e ${{ matrix.environment }}`) are expanded from the job's own matrix,
so the py311/py312/py313 legs are covered too. Every set-based assertion is paired with a check that the
scan actually found something, so none of them can pass vacuously.
"""
import functools
import pathlib
import re
import tomllib

import pytest
import yaml

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Names invoked through `pixi run` that are executables in the environment, not project tasks. Anything
#: outside this set is required to be a defined task. `python` is used by pure-wheel-test.yml, `pip` and
#: `playwright` by the deck-smoke job, and `pytest` by that job and the pre-commit hooks.
EXECUTABLES = {"pip", "playwright", "pytest", "python"}

#: `pixi run [--frozen] -e <env> <name>`, where <env> is a literal or a `${{ matrix.* }}` expression.
INVOCATION = re.compile(
    r"pixi run (?:--frozen )?-e (?P<env>\$\{\{[^}]*\}\}|[a-z0-9]+) (?P<name>[a-z0-9][a-z0-9-]*)"
)


@functools.lru_cache(maxsize=None)
def _pyproject() -> dict:
    """Return the parsed pyproject.toml (cached — every helper here reads from it)."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _tasks() -> dict:
    """Return the `[tool.pixi.tasks]` table."""
    return _pyproject()["tool"]["pixi"]["tasks"]


def _environments() -> set:
    """Return the environment names declared in `[tool.pixi.environments]`."""
    return set(_pyproject()["tool"]["pixi"]["environments"])


def _expand(env: str, matrix: dict) -> list:
    """Return the concrete environment names an `-e` argument can take.

    A literal yields itself; a `${{ matrix.<key> }}` expression yields that key's values from the job's
    matrix, so a matrix leg is checked against every environment it actually runs in.
    """
    if not env.startswith("${{"):
        return [env]
    key = env.strip("${} ").split(".")[-1]
    return [str(v) for v in matrix.get(key, [])]


def _workflow_invocations() -> list:
    """Return (env, name) pairs from every `run:` script in every workflow under .github/workflows."""
    found = []
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in (document.get("jobs") or {}).values():
            matrix = ((job.get("strategy") or {}).get("matrix")) or {}
            for step in job.get("steps") or []:
                for match in INVOCATION.finditer(step.get("run") or ""):
                    for env in _expand(match.group("env"), matrix):
                        found.append((env, match.group("name")))
    return found


def _hook_invocations() -> list:
    """Return (env, name) pairs from every hook `entry:` in .pre-commit-config.yaml."""
    document = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    found = []
    for repo in document.get("repos") or []:
        for hook in repo.get("hooks") or []:
            for match in INVOCATION.finditer(hook.get("entry") or ""):
                found.append((match.group("env"), match.group("name")))
    return found


class TestPixiTaskReferences:
    """Tests that CI and hook invocations resolve against the declared pixi tasks and environments."""

    def test_the_scan_finds_the_invocations(self):
        """The parsers return a realistic number of invocations from both sources.

        Test scenario:
            Every other test here asserts that a set of unresolved names is empty, which an empty scan
            would satisfy for the wrong reason. Pin a floor on what the scan must find, so a change to the
            invocation syntax or the YAML shape fails loudly instead of silently disabling this file.
        """
        workflow, hooks = _workflow_invocations(), _hook_invocations()
        assert len(workflow) >= 10, f"expected the workflows to invoke >=10 pixi commands, found {workflow}"
        assert len(hooks) >= 3, f"expected the hooks to invoke >=3 pixi commands, found {hooks}"

    @pytest.mark.parametrize("source", ["workflows", "hooks"])
    def test_task_references_are_defined(self, source):
        """Every non-executable name run through pixi is a task defined in pyproject.

        Args:
            source: Which config to scan — the CI workflows or the pre-commit hooks.

        Test scenario:
            Each `pixi run -e <env> <name>` names either an executable (pytest, pip, playwright) or a task
            in [tool.pixi.tasks]. A renamed or deleted task would otherwise surface only at run time.
        """
        tasks = _tasks()
        pairs = _workflow_invocations() if source == "workflows" else _hook_invocations()
        missing = sorted({name for _, name in pairs if name not in EXECUTABLES and name not in tasks})
        assert not missing, f"{source} run undefined pixi task(s): {missing}; defined: {sorted(tasks)}"

    @pytest.mark.parametrize("source", ["workflows", "hooks"])
    def test_referenced_environments_are_declared(self, source):
        """Every environment activated through pixi is declared in [tool.pixi.environments].

        Args:
            source: Which config to scan — the CI workflows or the pre-commit hooks.

        Test scenario:
            `pixi run -e <env>` against an undeclared environment fails at run time. Matrix legs are
            expanded first, so py311/py312/py313 are checked as well as the literal names.
        """
        declared = _environments()
        pairs = _workflow_invocations() if source == "workflows" else _hook_invocations()
        undeclared = sorted({env for env, _ in pairs} - declared)
        assert not undeclared, f"{source} use undeclared pixi env(s): {undeclared}; declared: {sorted(declared)}"

    @pytest.mark.parametrize("task", ["lint", "doctests"])
    def test_the_gates_added_for_the_migration_are_wired_up(self, task):
        """Each verification gate is defined and invoked by a real workflow step.

        Args:
            task: The pixi task expected to be both defined and run by CI.

        Test scenario:
            These two tasks cover code CI cannot otherwise reach — undefined names in notebook cells that
            never execute, and the doctests, which the `main` task does not collect. Because the scan reads
            parsed `run:` scripts, deleting the step stops satisfying this even though the workflow still
            mentions the task in a comment.
        """
        assert task in _tasks(), f"the {task!r} task is not defined in [tool.pixi.tasks]"
        invoked = {name for _, name in _workflow_invocations()}
        assert task in invoked, f"the {task!r} task is defined but no workflow step runs it"
