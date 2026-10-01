"""Unit tests for :func:`tests.ratchet.assert_matches_committed`, the shared U-5 ratchet comparison.

The three U-5 ratchets (API snapshot, support matrix, known-deviations allowlist) only ever call this on a green
run, where the committed artifact matches — so the branch that *fires* when they drift, which is the whole point
of a ratchet, is never exercised by them. These tests drive both branches directly: the match path returns
quietly, and the mismatch path raises with the unified diff, the artifact's name, and the regenerate command.
"""

from pathlib import Path

import pytest

from tests.ratchet import assert_matches_committed


class TestAssertMatchesCommitted:
    """Both branches of :func:`tests.ratchet.assert_matches_committed`."""

    @pytest.fixture
    def committed(self, tmp_path: Path) -> Path:
        """Write a two-line committed artifact and return its path.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Returns:
            The path to a committed file holding ``"alpha\\nbeta\\n"``.
        """
        path = tmp_path / "artifact.json"
        path.write_text("alpha\nbeta\n", encoding="utf-8")
        return path

    def test_matching_text_passes_quietly(self, committed: Path):
        """Equal current and committed text returns ``None`` and raises nothing.

        Args:
            committed: The committed-artifact fixture.

        Test scenario:
            The green-run case: the live value equals the committed file, so the ratchet is satisfied and the
            call is a no-op.
        """
        result = assert_matches_committed(
            "alpha\nbeta\n", committed, what="thing", regenerate="python -m tests.x"
        )
        assert result is None, (
            f"a matching comparison should return None, got {result!r}"
        )

    def test_drift_raises_with_diff_name_and_regenerate_command(self, committed: Path):
        """Differing text raises ``AssertionError`` carrying the diff, the file name and the regen command.

        Args:
            committed: The committed-artifact fixture.

        Test scenario:
            The drift case the ratchet exists for: the live value differs from the committed file, so the call
            must fail, and the message must let a reviewer both see what changed (a unified diff of the two
            lines) and know how to accept it on purpose (the file name and the regenerate command).
        """
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_committed(
                "alpha\ngamma\n",
                committed,
                what="public API surface",
                regenerate="python -m tests.test_api_snapshot",
            )
        message = str(exc_info.value)
        assert "public API surface" in message, (
            f"the 'what' phrase should be in the message: {message}"
        )
        assert committed.name in message, (
            f"the artifact name should be in the message: {message}"
        )
        assert "python -m tests.test_api_snapshot" in message, (
            f"the regenerate command should be in the message: {message}"
        )
        assert "-beta" in message and "+gamma" in message, (
            f"the unified diff should show the changed lines: {message}"
        )

    def test_a_missing_trailing_newline_is_a_drift(self, committed: Path):
        """Text identical but for a missing trailing newline is caught as drift.

        Args:
            committed: The committed-artifact fixture (ends with a newline).

        Test scenario:
            The comparison is exact apart from line-ending normalisation on read (``read_text`` maps
            CRLF/CR to LF), so a CRLF-vs-LF-only difference compares equal — but a serialiser that *dropped*
            its trailing newline is still caught rather than silently accepted.
        """
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_committed(
                "alpha\nbeta", committed, what="thing", regenerate="python -m tests.x"
            )
        assert "thing" in str(exc_info.value), (
            f"the 'what' phrase should be in the message: {exc_info.value}"
        )
