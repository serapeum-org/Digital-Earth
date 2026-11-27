"""Guards the release-ref artifact's role as the *publish token* for the downstream release consumers.

`github-release` hands the released commit to `pypi-release` and to `github-pages-mkdocs`'s deploy/gate
jobs through an uploaded artifact named `release-ref`. Those consumers are triggered by `workflow_run`,
whose payload cannot carry the dispatch inputs — so the artifact is the only channel that crosses the
boundary. Two invariants follow, and both were once missing:

**A draft uploads no token (#568).** `workflow_dispatch`'s `draft` input only sets the visibility of the
GitHub Release *page*: the version bump and the tag are pushed before it is consulted, so the release job
still concludes `success`. A consumer gated on that success therefore published to PyPI (irreversible) and
moved the `latest` docs alias for a release the maintainer had asked to keep unpublished. The fix is to
withhold the artifact on a draft, which is what the consumers now read as "nothing to publish".

**A missing token is not a failure (#567, #568).** Once a draft legitimately uploads nothing, a consumer
that hard-errors on the absent artifact just moves the problem — `actions/download-artifact` fails the step
when the named artifact does not exist, and no input softens it. So each download is `continue-on-error`
and each resolve step tests for the file before reading it. A *present but empty* ref must still hard-error:
that is corruption, not a draft, and it would otherwise publish the default branch.

**Consumers are guarded on the upstream's success (#567).** A release that fails before the upload also
leaves no artifact, and `workflow_run: types: [completed]` fires on a failure too. A consumer without a
`conclusion == 'success'` guard runs anyway and reddens its own workflow for a failure that has nothing to
do with it.

These are read from the **parsed** YAML, so a `${{ }}` or a path inside a comment cannot satisfy a guard.
"""

import pathlib

import pytest
import yaml

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where GitHub reads workflow definitions from.
WORKFLOWS = ROOT / ".github" / "workflows"

#: The artifact that carries the released commit across the `workflow_run` boundary.
TOKEN = "release-ref"

#: The file inside that artifact, and the path the consumers read it from.
TOKEN_FILE = "release-ref.txt"

#: The guard that keeps a consumer from running on a failed / cancelled upstream release.
SUCCESS_GUARD = "workflow_run.conclusion == 'success'"


def _workflow_files() -> list:
    """Return every workflow definition in the repository.

    Returns:
        The `.yml` / `.yaml` files under `.github/workflows`, sorted by name.
    """
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _steps(job: dict) -> list:
    """Return a job's steps, tolerating a `uses:` job that declares none.

    Args:
        job: One entry of a workflow's `jobs` mapping.

    Returns:
        The job's steps, as mappings.
    """
    steps = job.get("steps") or []
    return [step for step in steps if isinstance(step, dict)]


def _jobs(path: pathlib.Path) -> list:
    """Return a workflow's jobs, paired with the file they came from.

    Args:
        path: The workflow file.

    Returns:
        `(job_id, job)` for each mapping entry of the workflow's `jobs`.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    jobs = document.get("jobs") or {}
    return [(job_id, job) for job_id, job in jobs.items() if isinstance(job, dict)]


def _downloads_token(step: dict) -> bool:
    """Whether a step fetches the `release-ref` artifact.

    Args:
        step: One step of a job.

    Returns:
        `True` for a `download-artifact` step naming the publish token.
    """
    uses = step.get("uses") or ""
    name = (step.get("with") or {}).get("name")
    return "download-artifact" in uses and name == TOKEN


def _reads_token_file(step: dict) -> bool:
    """Whether a step's script *reads* the token file, rather than writing it.

    Args:
        step: One step of a job.

    Returns:
        `True` when the script `cat`s the token file the download placed on disk.
    """
    script = step.get("run")
    return isinstance(script, str) and TOKEN_FILE in script and "cat " in script


def _token_consumers() -> list:
    """Return every job in the repository that downloads the publish token.

    Returns:
        `(filename, job_id, job)` for each such job.
    """
    return [
        (path.name, job_id, job)
        for path in _workflow_files()
        for job_id, job in _jobs(path)
        if any(_downloads_token(step) for step in _steps(job))
    ]


def _producer_steps() -> list:
    """Return the steps that create and upload the publish token.

    The producer is whichever job writes the token file and uploads it as an artifact — located by
    behaviour rather than by workflow name, so renaming `github-release.yml` cannot silently empty
    this suite.

    Returns:
        `(filename, job_id, step)` for each producing step.
    """
    found = []
    for path in _workflow_files():
        for job_id, job in _jobs(path):
            for step in _steps(job):
                uses = step.get("uses") or ""
                script = step.get("run")
                uploads = (
                    "upload-artifact" in uses
                    and (step.get("with") or {}).get("name") == TOKEN
                )
                writes = isinstance(script, str) and f"> {TOKEN_FILE}" in script
                if uploads or writes:
                    found.append((path.name, job_id, step))
    return found


def _step_label(filename: str, job_id: str, step: dict) -> str:
    """Name a step the way a failure message should.

    Args:
        filename: The workflow file's name.
        job_id: The job the step belongs to.
        step: The step itself.

    Returns:
        A `file / job / step` label.
    """
    return f"{filename} / job {job_id} / {step.get('name', step.get('uses', '?'))}"


def _producer_params() -> list:
    """Return the producing steps as parametrize cases with readable identifiers.

    Returns:
        One `pytest.param` per producing step, identified by workflow, job and step name.
    """
    return [
        pytest.param(
            filename,
            job_id,
            step,
            id=f"{filename}:{job_id}:{step.get('name', step.get('uses', '?'))}",
        )
        for filename, job_id, step in _producer_steps()
    ]


def _consumer_params() -> list:
    """Return the consuming jobs as parametrize cases with readable identifiers.

    Returns:
        One `pytest.param` per token-consuming job, identified by workflow and job.
    """
    return [
        pytest.param(filename, job_id, job, id=f"{filename}:{job_id}")
        for filename, job_id, job in _token_consumers()
    ]


class TestTheSuiteIsNotVacuous:
    """Every guard below is a filter; an empty repository would satisfy all of them."""

    def test_the_workflow_directory_is_where_it_is_expected(self):
        """So the whole file cannot pass by scanning nothing."""
        assert WORKFLOWS.is_dir(), f"no workflow directory at {WORKFLOWS}"
        assert _workflow_files(), f"no workflow files under {WORKFLOWS}"

    def test_the_token_is_produced_somewhere(self):
        """A repository that stopped uploading the token would pass the producer guards vacuously."""
        assert _producer_steps(), (
            f"no step writes or uploads the {TOKEN!r} artifact — the producer guards would pass "
            "vacuously. If the release handoff was redesigned, retire this suite deliberately."
        )

    def test_the_token_has_consumers(self):
        """Likewise for the consumer guards."""
        consumers = _token_consumers()
        assert len(consumers) >= 2, (
            f"expected the publish token to be downloaded by several jobs, found {consumers}"
        )


class TestADraftReleaseUploadsNoPublishToken:
    """#568 — withholding the artifact is how a draft stops the downstream publish chain."""

    @pytest.mark.parametrize(("filename", "job_id", "step"), _producer_params())
    def test_the_producer_steps_are_gated_on_the_draft_input(
        self, filename, job_id, step
    ):
        """Both producing steps skip when the dispatch asked for a draft.

        Args:
            filename: The workflow file the step came from.
            job_id: The job it belongs to.
            step: The step under test.

        Test scenario:
            A `workflow_dispatch` with `draft: true` bumps the version, pushes the tag and succeeds —
            `draft` only reaches the GitHub Release page's visibility. With the artifact uploaded anyway,
            the success-gated consumers published to PyPI and moved the `latest` docs alias for a release
            that was never meant to go live. Skipping these two steps is the whole fix: no token, no
            publish.
        """
        condition = str(step.get("if") or "")
        assert "inputs.draft" in condition, (
            f"{_step_label(filename, job_id, step)} runs on a draft release, so a draft still hands "
            f"the downstream consumers a publish token. Gate it on `inputs.draft != true`. Found "
            f"if: {condition!r}"
        )


class TestEveryTokenConsumerToleratesAMissingToken:
    """#567 / #568 — a draft or a failed release leaves no token, and that is not an error."""

    @pytest.mark.parametrize(("filename", "job_id", "job"), _consumer_params())
    def test_the_consumer_job_is_guarded_on_the_upstream_succeeding(
        self, filename, job_id, job
    ):
        """A failed or cancelled release must not start the consumer at all.

        Args:
            filename: The workflow file the job came from.
            job_id: The job under test.
            job: The job itself.

        Test scenario:
            `workflow_run: types: [completed]` fires whatever the conclusion. `build-strict` shipped
            without a job-level `if:`, so a release that failed before the upload — at the admin check,
            the wheel matrix, commitizen — started the docs gate anyway, whose download then errored and
            reddened "Deploy MkDocs" for a failure that had nothing to do with docs.
        """
        condition = str(job.get("if") or "")
        assert SUCCESS_GUARD in condition, (
            f"{filename} / job {job_id} downloads the publish token but is not guarded on the "
            f"upstream release succeeding; add `{SUCCESS_GUARD}` to its job-level `if:`. Found "
            f"if: {condition!r}"
        )

    @pytest.mark.parametrize(("filename", "job_id", "job"), _consumer_params())
    def test_the_download_step_does_not_fail_on_a_missing_token(
        self, filename, job_id, job
    ):
        """`download-artifact` hard-errors on an absent artifact, so the step absorbs it.

        Args:
            filename: The workflow file the job came from.
            job_id: The job under test.
            job: The job itself.

        Test scenario:
            `actions/download-artifact` throws `Artifact 'release-ref' not found` and fails the step; no
            input softens it (`digest-mismatch` covers hashes of artifacts that *were* downloaded). Since
            a draft now deliberately uploads nothing, the step must continue instead.
        """
        offenders = [
            _step_label(filename, job_id, step)
            for step in _steps(job)
            if _downloads_token(step) and step.get("continue-on-error") is not True
        ]
        assert offenders == [], (
            "a missing publish token fails these download steps, so a draft release would red the run "
            "it was meant to skip. Add `continue-on-error: true`:\n  "
            + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize(("filename", "job_id", "job"), _consumer_params())
    def test_the_resolve_step_tests_for_the_token_before_reading_it(
        self, filename, job_id, job
    ):
        """With `continue-on-error` above, the bare `cat` becomes the new failure point.

        Args:
            filename: The workflow file the job came from.
            job_id: The job under test.
            job: The job itself.

        Test scenario:
            The resolve scripts run under `set -euo pipefail`, so `REF="$(cat .release-ref/release-ref.txt)"`
            on an absent file exits non-zero and fails the job — relocating the red rather than removing
            it. Each script must branch on the file existing.
        """
        readers = [step for step in _steps(job) if _reads_token_file(step)]
        assert readers, (
            f"{filename} / job {job_id} downloads the publish token but no step reads it; this guard "
            "cannot check what it cannot find."
        )
        offenders = [
            _step_label(filename, job_id, step)
            for step in readers
            if f"-f .release-ref/{TOKEN_FILE}" not in step["run"]
        ]
        assert offenders == [], (
            "these steps read the publish token without testing that it exists, so a draft release "
            f"fails on the `cat` under `set -e`. Branch on `[ -f .release-ref/{TOKEN_FILE} ]`:\n  "
            + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize(("filename", "job_id", "job"), _consumer_params())
    def test_a_present_but_empty_token_still_hard_errors(self, filename, job_id, job):
        """Corruption is not a draft, and it must never fall through to the default branch.

        Args:
            filename: The workflow file the job came from.
            job_id: The job under test.
            job: The job itself.

        Test scenario:
            The missing-token path is deliberately soft. An artifact that exists but holds an empty ref is
            a different failure: `checkout` would silently take the default branch and the job would
            publish pre-bump `main`. That guard predates this change and must survive it.
        """
        offenders = [
            _step_label(filename, job_id, step)
            for step in _steps(job)
            if _reads_token_file(step) and "::error::" not in step["run"]
        ]
        assert offenders == [], (
            "these steps no longer fail on a present-but-empty publish token, which would publish the "
            "default branch instead of the released commit:\n  "
            + "\n  ".join(offenders)
        )
