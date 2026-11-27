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

Producers and consumers are located by **behaviour** — which step writes or uploads the artifact, which
downloads or reads it — rather than by workflow name, so renaming a file cannot quietly empty the suite.
That makes the two predicates load-bearing, and `TestTheDiscriminatorsTellProducersFromConsumers` pins
them: the producer's upload carries the *same* `name: release-ref` as the consumers' downloads, and the two
owe opposite duties, so a predicate that over-matched would demand of one step what only the other can do.
`TestTheSuiteIsNotVacuous` covers the other direction, where a predicate that stopped matching would leave
every guard passing over an empty set.
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


def _resolve_step(job: dict) -> dict | None:
    """Return the step that reads the token and decides what the job does about it.

    Args:
        job: One entry of a workflow's `jobs` mapping.

    Returns:
        The resolve step, or `None` for a job that downloads the token but never reads it.
    """
    for step in _steps(job):
        if _reads_token_file(step):
            return step
    return None


def _skips_on_a_missing_token(job: dict) -> bool:
    """Whether this consumer answers a missing token by skipping its remaining steps.

    Two mechanisms are in use, deliberately. `pypi-release`'s publish and `deploy-release` have nothing
    useful to do without a released commit, so they emit a `skip=true` output and gate every later step
    on it. `build-strict` does — gating the default tree is a meaningful action — so it assigns `REF=""`
    and carries on. Which mechanism a job uses decides which guard applies to it.

    Args:
        job: One entry of a workflow's `jobs` mapping.

    Returns:
        `True` when the job's resolve step emits a skip signal rather than an empty-ref fallback.
    """
    step = _resolve_step(job)
    return step is not None and "skip=true" in step["run"]


def _steps_after_resolve(job: dict) -> list:
    """Return the steps a job runs once the token has been resolved.

    These are the steps that act — checkout, publish, deploy, prune — and the ones a skip has to stop.

    Args:
        job: One entry of a workflow's `jobs` mapping.

    Returns:
        Every step following the resolve step, in order.
    """
    steps = _steps(job)
    resolve = _resolve_step(job)
    if resolve is None:
        return []
    return steps[steps.index(resolve) + 1 :]


class TestADraftActuallyStopsThePublish:
    """The gating is the half of the fix that does the work, and it needs its own guards.

    The other consumer tests assert that a missing token is *tolerated* — the download continues, the
    resolve step does not die on the `cat`. None of that prevents anything: a job that tolerates the
    absence and then publishes regardless is the original defect with extra steps. What stops the
    publish is the `skip=true` output and the `if: steps.ref.outputs.skip != 'true'` gate on every step
    after it, so those are pinned here.

    Written after a review mutation-tested the first version of this suite and found six mutants
    surviving green, among them the removal of all five `deploy-release` gates plus the `skip=true`
    output — together a clean revert of the consumer side of the fix.
    """

    @pytest.mark.parametrize(("filename", "job_id", "job"), _consumer_params())
    def test_the_resolve_step_emits_a_skip_signal_or_an_empty_ref_fallback(
        self, filename, job_id, job
    ):
        """Each consumer answers a missing token by one of the two mechanisms, not by neither.

        Args:
            filename: The workflow file the job came from.
            job_id: The job under test.
            job: The job itself.

        Test scenario:
            Deleting `echo "skip=true" >> "$GITHUB_OUTPUT"` left the earlier suite green while making
            every downstream gate read `'' != 'true'` — true — so the job published a checkout of the
            default branch. A job must therefore either emit the skip signal into `$GITHUB_OUTPUT`, or
            be the fallback kind that assigns an empty `REF` and legitimately carries on.
        """
        step = _resolve_step(job)
        assert step is not None, (
            f"{filename} / job {job_id} downloads the token but no step reads it"
        )
        script = step["run"]
        emits_skip = "skip=true" in script and "$GITHUB_OUTPUT" in script
        falls_back = 'REF=""' in script
        assert emits_skip or falls_back, (
            f"{_step_label(filename, job_id, step)} neither emits `skip=true` into $GITHUB_OUTPUT nor "
            "falls back to an empty REF, so a missing token leaves the job acting on an unresolved ref"
        )

    @pytest.mark.parametrize(("filename", "job_id", "job"), _consumer_params())
    def test_every_step_after_the_resolve_is_gated_on_the_skip(
        self, filename, job_id, job
    ):
        """A skip that nothing is gated on does not skip anything.

        Args:
            filename: The workflow file the job came from.
            job_id: The job under test.
            job: The job itself.

        Test scenario:
            This is the finding that prompted the class. Removing the gate from `Build and publish to
            PyPI`, from its checkout, or from all five `deploy-release` steps each left the earlier
            suite at 35/35 green — and the first two are worse than the original bug: with `skip=true`
            set but the gate gone, `ref:` is empty and `actions/checkout` takes the default branch, so
            the job publishes pre-bump `main` to PyPI.

            Jobs using the empty-ref fallback (`build-strict`) are exempt by construction: they have no
            skip output, and gating the default tree is the action they are meant to take.
        """
        if not _skips_on_a_missing_token(job):
            pytest.skip(f"{job_id} uses the empty-ref fallback, not the skip mechanism")
        resolve_id = _resolve_step(job).get("id")
        assert resolve_id, (
            f"{filename} / job {job_id}'s resolve step has no `id:`, so nothing can gate on its output"
        )
        gate = f"steps.{resolve_id}.outputs.skip"
        offenders = [
            _step_label(filename, job_id, step)
            for step in _steps_after_resolve(job)
            if gate not in str(step.get("if") or "")
        ]
        assert offenders == [], (
            f"these steps run even when the publish token is absent, so a draft release still acts. "
            f"Gate each on `{gate} != 'true'`:\n  " + "\n  ".join(offenders)
        )

    def test_the_docs_gate_still_runs_on_every_non_release_event(self):
        """The success guard must not cost the repository its pull-request docs gate.

        Test scenario:
            `build-strict` is the `--strict` docs build every PR is gated on, and the success guard
            added for #567 is only the *second* arm of its condition. The first arm,
            `github.event_name != 'workflow_run'`, is what keeps it running on `push`, `pull_request`
            and `release`. Dropping that prefix leaves a condition that is false on every one of those
            events, so the docs gate would silently stop running on all pull requests while a suite
            asserting only the `success` clause stayed green.
        """
        document = yaml.safe_load(
            (WORKFLOWS / "github-pages-mkdocs.yml").read_text(encoding="utf-8")
        )
        condition = " ".join(str(document["jobs"]["build-strict"]["if"]).split())
        assert "github.event_name != 'workflow_run'" in condition, (
            "build-strict no longer admits non-workflow_run events, so the --strict docs gate does not "
            f"run on pull requests. Found if: {condition!r}"
        )
        assert SUCCESS_GUARD in condition, (
            f"build-strict lost its success guard (#567). Found if: {condition!r}"
        )


class TestTheDiscriminatorsTellProducersFromConsumers:
    """The guards below are only as good as the two predicates that decide what they apply to.

    Both predicates are exercised by the real workflows only on their *matching* side: while every
    guard passes, the rejecting branches and the failure-message path never run. A predicate that
    quietly over-matched would then demand of the producer what only a consumer owes — withholding the
    token AND absorbing its absence are opposite duties — and the contradiction would surface as a
    confusing failure on an unrelated change rather than here.
    """

    @pytest.mark.parametrize(
        ("step", "expected", "because"),
        [
            (
                {"uses": "actions/download-artifact@v8", "with": {"name": TOKEN}},
                True,
                "a download of the token is a consumer",
            ),
            (
                {"uses": "actions/upload-artifact@v7", "with": {"name": TOKEN}},
                False,
                "the producer's upload names the same artifact and must not be read as a consumer",
            ),
            (
                {"uses": "actions/download-artifact@v8", "with": {"name": "site-pr-1"}},
                False,
                "another artifact's download is unrelated",
            ),
            (
                {"uses": "actions/download-artifact@v8"},
                False,
                "a download with no name selects by pattern, not by this token",
            ),
            (
                {"run": f"cat .release-ref/{TOKEN_FILE}"},
                False,
                "a script is not a download step",
            ),
        ],
    )
    def test_only_a_download_of_the_token_counts_as_a_consumer(
        self, step, expected, because
    ):
        """`_downloads_token` keys off both the action and the artifact name.

        Args:
            step: A synthetic step.
            expected: Whether it should be read as consuming the token.
            because: Why, quoted back in the failure message.

        Test scenario:
            The decisive case is the producer's `upload-artifact` step, which carries the *same*
            `name: release-ref`. Matching on the name alone would class it as a consumer and demand
            `continue-on-error` on the very step whose job is to withhold the artifact.
        """
        assert _downloads_token(step) is expected, (
            f"_downloads_token({step}) should be {expected}: {because}"
        )

    @pytest.mark.parametrize(
        ("step", "expected", "because"),
        [
            (
                {"run": f'REF="$(cat .release-ref/{TOKEN_FILE})"'},
                True,
                "the consumers' resolve step reads the token",
            ),
            (
                {"run": f"git rev-parse HEAD > {TOKEN_FILE}"},
                False,
                "the producer WRITES the token; it has no file to test for",
            ),
            (
                {"uses": "actions/download-artifact@v8", "with": {"name": TOKEN}},
                False,
                "a `uses` step runs no script",
            ),
            (
                {"run": "echo no token here"},
                False,
                "an unrelated script does not touch the token",
            ),
        ],
    )
    def test_only_a_script_that_reads_the_token_counts(self, step, expected, because):
        """`_reads_token_file` must not mistake the producer's write for a read.

        Args:
            step: A synthetic step.
            expected: Whether it should be read as reading the token file.
            because: Why, quoted back in the failure message.

        Test scenario:
            The producer's `git rev-parse HEAD > release-ref.txt` mentions the same filename. Were it
            read as a reader, the resolve guard would demand a `[ -f … ]` test on a step that creates
            the file.
        """
        assert _reads_token_file(step) is expected, (
            f"_reads_token_file({step}) should be {expected}: {because}"
        )

    def test_the_producing_and_consuming_steps_are_disjoint_in_this_repository(self):
        """No real step is asked to both withhold the token and tolerate its absence.

        Test scenario:
            Run against the actual workflows: the producer's steps and the consumers' download steps
            must share no member. Overlap would mean the two guard families make contradictory demands
            of one step, which no edit could satisfy.
        """
        producing = {
            (filename, job_id, step.get("name") or step.get("uses"))
            for filename, job_id, step in _producer_steps()
        }
        consuming = {
            (filename, job_id, step.get("name") or step.get("uses"))
            for filename, job_id, job in _token_consumers()
            for step in _steps(job)
            if _downloads_token(step)
        }
        assert producing, "no producing steps found; the comparison would be vacuous"
        assert consuming, "no consuming steps found; the comparison would be vacuous"
        assert producing.isdisjoint(consuming), (
            "these steps are read as both producing and consuming the publish token, so the guards "
            f"would demand opposite things of them: {sorted(producing & consuming)}"
        )

    @pytest.mark.parametrize(
        ("step", "tail"),
        [
            ({"name": "Resolve the ref"}, "Resolve the ref"),
            (
                {
                    "name": "Download the release ref from the github-release run",
                    "uses": "actions/download-artifact@3e5f45b2",
                },
                "Download the release ref from the github-release run",
            ),
            ({"uses": "actions/checkout@v5"}, "actions/checkout@v5"),
            ({}, "?"),
        ],
    )
    def test_a_step_is_labelled_by_name_then_by_action(self, step, tail):
        """The label a tripped guard prints must locate the step for whoever has to fix it.

        Args:
            step: A synthetic step.
            tail: The identifier expected at the end of the label.

        Test scenario:
            This path runs only when a guard fails, so it is otherwise unexercised; a broken label
            would be discovered at the worst moment. The load-bearing case is a step carrying BOTH
            keys — every download step here does — where the human-readable `name` must win over the
            pinned action SHA. A `uses`-only step falls back to the action, and a step with neither
            still has to produce something.
        """
        label = _step_label("pypi-release.yml", "publish", step)
        assert label == f"pypi-release.yml / job publish / {tail}", (
            f"unexpected label for {step}: {label!r}"
        )

    @pytest.mark.parametrize(
        ("job", "expected"),
        [
            ({"steps": None}, []),
            ({}, []),
            ({"steps": ["not-a-mapping"]}, []),
            ({"steps": [{"run": "true"}, "stray"]}, [{"run": "true"}]),
        ],
    )
    def test_degenerate_jobs_yield_no_phantom_steps(self, job, expected):
        """`_steps` tolerates a job that declares none without inventing one.

        Args:
            job: A synthetic job.
            expected: The steps that should be recovered from it.

        Test scenario:
            A `uses:` job (a reusable-workflow call such as `pure-wheel-test`) has no `steps` key at
            all, and GitHub accepts a list that YAML parses with a non-mapping entry. Neither may crash
            the scan or contribute a step the guards would then judge.
        """
        assert _steps(job) == expected, f"_steps({job}) returned {_steps(job)}"


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
    def test_the_resolve_step_validates_the_ref_is_a_commit_sha(
        self, filename, job_id, job
    ):
        """A non-empty ref is not a *valid* ref, and the difference publishes the wrong tree.

        Args:
            filename: The workflow file the job came from.
            job_id: The job under test.
            job: The job itself.

        Test scenario:
            `[ -n "$REF" ]` admits a blank-but-non-empty token such as `"   "`: `$(cat …)` strips the
            trailing newline but not interior blanks, so the guard passes with `REF="   "`. That value
            reaches `actions/checkout`, whose `getInput` trims whitespace, so checkout receives an empty
            ref and silently takes the **default branch** — the exact outcome the guard's own message
            promises to refuse. Both consumers were measured returning exit 0 on such a token. The
            producer writes `git rev-parse HEAD`, so the ref is always 40 hex characters; the guard must
            check that shape, not mere non-emptiness.
        """
        readers = [step for step in _steps(job) if _reads_token_file(step)]
        assert readers, (
            f"{filename} / job {job_id} downloads the publish token but no step reads it"
        )
        offenders = [
            _step_label(filename, job_id, step)
            for step in readers
            if "${#REF}" not in step["run"] or "-eq 40" not in step["run"]
        ]
        assert offenders == [], (
            "these steps accept any non-empty release ref, so a blank token resolves to the default "
            "branch. Validate the 40-hex-character SHA the producer writes:\n  "
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
