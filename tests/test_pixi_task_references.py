"""Guards against drift between the CI workflows, the pre-commit hooks and the project's own config.

Four kinds of drift are covered: a task reference that no longer resolves, the notebook skip-list falling out
of step with the hook regex that is documented as mirroring it, a composite gate that quietly stops fanning
out to one of the checks it aggregates, and the mypy baseline growing instead of shrinking.

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
import shlex
import tomllib

import pytest
import yaml
from _pytest.pathlib import fnmatch_ex

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
    document = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
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
        assert len(workflow) >= 10, (
            f"expected the workflows to invoke >=10 pixi commands, found {workflow}"
        )
        assert len(hooks) >= 3, (
            f"expected the hooks to invoke >=3 pixi commands, found {hooks}"
        )

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
        pairs = (
            _workflow_invocations() if source == "workflows" else _hook_invocations()
        )
        missing = sorted(
            {name for _, name in pairs if name not in EXECUTABLES and name not in tasks}
        )
        assert not missing, (
            f"{source} run undefined pixi task(s): {missing}; defined: {sorted(tasks)}"
        )

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
        pairs = (
            _workflow_invocations() if source == "workflows" else _hook_invocations()
        )
        undeclared = sorted({env for env, _ in pairs} - declared)
        assert not undeclared, (
            f"{source} use undeclared pixi env(s): {undeclared}; declared: {sorted(declared)}"
        )

    @pytest.mark.parametrize(
        "task", ["lint-names", "lint-format", "lint-imports", "mypy", "doctests"]
    )
    def test_the_gates_added_for_the_migration_are_wired_up(self, task):
        """Each verification gate is defined and invoked by a real workflow step.

        Args:
            task: The pixi task expected to be both defined and run by CI.

        Test scenario:
            These cover code the `main` task cannot reach — undefined names in notebook cells that never
            execute, formatting and import order, the type checker, and the doctests, which `main` does not
            collect. The three ruff gates are pinned individually rather than through the aggregate `lint`
            task: pixi's `depends-on` stops at the first failing subtask, so running them as one CI step
            would let a formatting slip hide the import-order result. Because the scan reads parsed `run:`
            scripts, deleting a step stops satisfying this even though the workflow still names the task in
            a comment.
        """
        assert task in _tasks(), (
            f"the {task!r} task is not defined in [tool.pixi.tasks]"
        )
        invoked = {name for _, name in _workflow_invocations()}
        assert task in invoked, (
            f"the {task!r} task is defined but no workflow step runs it"
        )


def _notebooks() -> list:
    """Return every example notebook, excluding checkpoint copies."""
    return sorted(
        p
        for p in (ROOT / "docs" / "examples").glob("**/*.ipynb")
        if ".ipynb_checkpoints" not in p.as_posix()
    )


def _task_skipped() -> set:
    """Return the notebooks the `notebooks` task's --ignore-glob arguments exclude.

    Matching goes through pytest's own `fnmatch_ex`, the function that implements `--ignore-glob`, so this
    reflects what pytest actually does rather than a second guess at its glob semantics.
    """
    globs = [
        arg.split("=", 1)[1]
        for arg in shlex.split(_tasks()["notebooks"]["cmd"])
        if arg.startswith("--ignore-glob=")
    ]
    return {p for p in _notebooks() if any(fnmatch_ex(glob, p) for glob in globs)}


def _hook_excluded() -> set:
    """Return the notebooks the pre-commit `notebook-check` hook's exclude regex filters out."""
    document = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    hook = next(
        h
        for repo in document["repos"]
        for h in repo.get("hooks") or []
        if h.get("id") == "notebook-check"
    )
    pattern = re.compile(hook["exclude"])
    return {p for p in _notebooks() if pattern.match(p.relative_to(ROOT).as_posix())}


class TestNotebookSkipListsMirror:
    """Tests that the notebook skip-list and the hook regex that documents itself as mirroring it agree."""

    def test_the_notebook_scan_finds_notebooks(self):
        """The gallery scan finds notebooks and the task declares ignore-globs.

        Test scenario:
            The mirror assertion compares two sets; both would be empty, and equal, if the scan or the
            task's argument list stopped being found. Pin a floor under each.
        """
        notebooks = _notebooks()
        globs = [
            a
            for a in shlex.split(_tasks()["notebooks"]["cmd"])
            if a.startswith("--ignore-glob=")
        ]
        assert len(notebooks) >= 40, (
            f"expected >=40 example notebooks, found {len(notebooks)}"
        )
        assert len(globs) >= 5, (
            f"expected the notebooks task to declare >=5 ignore-globs, found {globs}"
        )

    def test_hook_regex_excludes_exactly_what_the_task_skips(self):
        """The hook's exclude regex filters exactly the notebooks the task's ignore-globs skip.

        Test scenario:
            The hook's comment states it mirrors the `notebooks` task's ignore-globs, but the two are
            written in different languages — shell globs against a regex — so nothing but this test keeps
            them in step. A notebook in one set and not the other is checked by only one of the two gates.
        """
        task_only = sorted(
            p.relative_to(ROOT).as_posix() for p in _task_skipped() - _hook_excluded()
        )
        hook_only = sorted(
            p.relative_to(ROOT).as_posix() for p in _hook_excluded() - _task_skipped()
        )
        assert not task_only, (
            f"the task skips these but the pre-commit hook still checks them: {task_only}"
        )
        assert not hook_only, (
            f"the hook skips these but the notebooks task still runs them: {hook_only}"
        )


class TestRuffPinsAgree:
    """Tests that the ruff the lint task resolves is the ruff the pre-commit hook runs."""

    def test_dev_extra_pin_matches_the_pre_commit_rev(self):
        """The `ruff ==X` pin in the dev extra equals the ruff-pre-commit `rev: vX`.

        Test scenario:
            The `lint` task resolves ruff from the `dev` dependency group; the hooks resolve it from their
            own rev. They must agree, or a contributor's hook and CI can disagree about the same file. The
            two pins live in different files and were kept in step only by a comment saying to bump them.
        """
        dev = _pyproject()["dependency-groups"]["dev"]
        pinned = [d for d in dev if d.replace(" ", "").startswith("ruff==")]
        assert len(pinned) == 1, (
            f"expected exactly one pinned ruff in the dev extra, found {pinned}"
        )
        pyproject_version = pinned[0].split("==")[1].strip()

        document = yaml.safe_load(
            (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        )
        revs = [
            repo["rev"]
            for repo in document["repos"]
            if "ruff-pre-commit" in repo.get("repo", "")
        ]
        assert len(revs) == 1, (
            f"expected exactly one ruff-pre-commit repo, found {revs}"
        )
        hook_version = revs[0].lstrip("v")

        assert pyproject_version == hook_version, (
            f"ruff pins disagree: pyproject dev extra has {pyproject_version!r}, "
            f".pre-commit-config.yaml has {revs[0]!r}"
        )


#: The checks the composite `lint` task must keep fanning out to. Splitting it into three (#171) is what put
#: the formatter and the import sorter behind a gate; dropping one from `depends-on` would un-gate it in
#: silence, because `pixi run -e dev lint` would still exit 0.
LINT_GATES = ["lint-names", "lint-format", "lint-imports"]

#: `(task, tokens its command must contain)` for the gates that only work if they are pointed at the right
#: trees in the right mode. `--check` is what makes the formatter a gate rather than a rewrite.
GATE_ARGUMENTS = [
    ("lint-format", ["ruff", "format", "--check", "src", "tests"]),
    ("lint-imports", ["ruff", "check", "--select", "I", "src", "tests"]),
    ("lint-names", ["ruff", "check", "--select", "F821", "src", "tests", "docs"]),
]

#: Number of modules in the mypy baseline when it was recorded at #172. The table documents itself as a
#: ratchet that may shrink and must never grow, which is only true if something counts.
MYPY_BASELINE_CEILING = 35


def _mypy() -> dict:
    """Return the `[tool.mypy]` table, including its `overrides` array."""
    return _pyproject()["tool"]["mypy"]


def _mypy_overrides() -> list:
    """Return the `[[tool.mypy.overrides]]` entries — the per-module baseline."""
    return _mypy().get("overrides") or []


class TestLintGateFansOut:
    """Tests that the composite `lint` task still runs all three of the checks it aggregates."""

    def test_lint_depends_on_every_gate(self):
        """`lint` fans out to the name check, the formatter and the import sorter.

        Test scenario:
            CI and the pre-commit hook both invoke `lint`, never the three parts. It carries no `cmd` of its
            own, so a gate dropped from `depends-on` disappears from every caller at once while the gate
            still reports success — the shape of the drift #171 was filed for.
        """
        lint = _tasks()["lint"]
        assert lint.get("depends-on") == LINT_GATES, (
            f"`lint` must depend on {LINT_GATES}, found {lint.get('depends-on')}"
        )
        assert "cmd" not in lint, (
            "`lint` should stay a pure aggregator; a `cmd` on it would run outside the fan-out"
        )

    @pytest.mark.parametrize(
        "task, tokens", GATE_ARGUMENTS, ids=[name for name, _ in GATE_ARGUMENTS]
    )
    def test_gate_command_covers_the_intended_trees(self, task, tokens):
        """Each lint gate runs the intended tool, in check mode, over the intended trees.

        Args:
            task: The pixi task under test.
            tokens: Command-line tokens its `cmd` must contain.

        Test scenario:
            A gate narrowed to `src` would stop covering `tests/`, and a formatter invoked without `--check`
            would rewrite files and exit 0 instead of failing — both leave a green CI over a drifting tree.
        """
        command = shlex.split(_tasks()[task]["cmd"])
        missing = [token for token in tokens if token not in command]
        assert not missing, f"the {task!r} command is missing {missing}: {command}"

    def test_every_depends_on_target_is_a_defined_task(self):
        """Every name in any task's `depends-on` resolves to another declared task.

        Test scenario:
            A dependency naming a task that was renamed or deleted fails only when that task is run. This
            walks the whole table rather than just `lint`, so the same slip is caught anywhere.
        """
        tasks = _tasks()
        dangling = sorted(
            {
                f"{name} -> {dependency}"
                for name, body in tasks.items()
                if isinstance(body, dict)
                for dependency in body.get("depends-on") or []
                if dependency not in tasks
            }
        )
        assert not dangling, f"tasks depend on undefined task(s): {dangling}"

    def test_the_type_check_gate_is_wired_up(self):
        """The `mypy` task is defined and a real workflow step runs it.

        Test scenario:
            mypy was configured and hooked into pre-commit long before it could pass, so the hook was
            routinely skipped. It is clean as of #172 and CI now has a `Type-check` step; without this the
            step could be dropped and the baseline below would go on being maintained for nothing.
        """
        assert "mypy" in _tasks(), "the 'mypy' task is not defined in [tool.pixi.tasks]"
        invoked = {name for _, name in _workflow_invocations()}
        assert "mypy" in invoked, (
            "the 'mypy' task is defined but no workflow step runs it"
        )


class TestMypyBaseline:
    """Tests for the per-module mypy baseline in `[[tool.mypy.overrides]]`."""

    def test_the_baseline_is_read_and_is_not_empty(self):
        """The overrides array parses and holds entries.

        Test scenario:
            Every assertion below iterates the array; an empty or mis-keyed one would pass them all while
            checking nothing.
        """
        assert _mypy_overrides(), (
            "[[tool.mypy.overrides]] should hold the per-module baseline"
        )

    def test_the_baseline_has_not_grown(self):
        """The baseline holds no more modules than it did when it was recorded.

        Test scenario:
            The table calls itself a ratchet that may shrink and must never grow, but nothing enforced that.
            Draining an entry is meant to be routine; adding one silently exempts a module from the type
            gate. Lower this ceiling as entries are drained.
        """
        overrides = _mypy_overrides()
        assert len(overrides) <= MYPY_BASELINE_CEILING, (
            f"the mypy baseline grew to {len(overrides)} modules (ceiling {MYPY_BASELINE_CEILING}); "
            "fix the new errors instead of baselining them, or justify raising the ceiling"
        )

    def test_every_entry_names_a_module_that_exists(self):
        """Each baselined module resolves to a real file or package under `src/`.

        Test scenario:
            An entry left behind by a rename silences nothing and hides how much debt is really left. The
            backend restructure moved every module in the package, which is exactly when this rots.
        """
        stale = []
        for override in _mypy_overrides():
            module = override["module"]
            path = ROOT / "src" / pathlib.Path(*module.split("."))
            if not (
                path.with_suffix(".py").exists() or (path / "__init__.py").exists()
            ):
                stale.append(module)
        assert not stale, (
            f"the mypy baseline names module(s) that no longer exist: {stale}"
        )

    def test_no_entry_silences_a_whole_subtree(self):
        """No baseline entry uses a wildcard module pattern.

        Test scenario:
            The table's own rationale is that a blanket disable would also silence these codes in code
            written tomorrow. `module = "digitalearth.web.*"` reintroduces exactly that, one line at a time,
            and reads almost identically to a legitimate per-module entry.
        """
        wildcards = sorted(
            override["module"]
            for override in _mypy_overrides()
            if "*" in override["module"]
        )
        assert not wildcards, (
            f"the baseline must stay per-module; wildcard entries silence future code too: {wildcards}"
        )

    def test_every_entry_lists_specific_error_codes(self):
        """Each entry disables a non-empty list of named error codes and nothing broader.

        Test scenario:
            `disable_error_code` names what is owed; `ignore_errors` or `follow_imports = "skip"` would turn
            a numbered debt into a blanket exemption that no longer shrinks as the module improves.
        """
        offenders = []
        for override in _mypy_overrides():
            codes = override.get("disable_error_code")
            extra = set(override) - {"module", "disable_error_code"}
            if not codes or not isinstance(codes, list) or extra:
                offenders.append(
                    f"{override['module']}: codes={codes!r} extra_keys={sorted(extra)}"
                )
        assert not offenders, (
            f"every baseline entry must disable a specific list of codes and nothing else: {offenders}"
        )

    def test_the_global_settings_do_not_disable_anything(self):
        """`[tool.mypy]` itself disables no error codes and ignores no errors.

        Test scenario:
            The whole point of a per-module baseline is that new modules are checked in full. A global
            `disable_error_code` (or `ignore_errors`) would exempt every module at once — including ones
            with no entry here — and would make the per-module list look stricter than the project is.
        """
        settings = _mypy()
        assert "disable_error_code" not in settings, (
            "[tool.mypy] must not disable error codes globally; baseline them per module instead"
        )
        assert not settings.get("ignore_errors", False), (
            "[tool.mypy] must not set ignore_errors"
        )
