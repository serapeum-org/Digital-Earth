"""Tests for :mod:`digitalearth.base.bigdata` — the shared big-data cutoff and the rule that validates it.

Round-2 review **L6**: ``validate_big_data_threshold`` ran the caller's value through ``int()``, which
*truncated* a fraction (``1.9`` became ``1``, cutting over one row earlier than asked) and let every other
bad value escape as a bare ``ValueError``/``TypeError`` from ``int`` itself, naming neither the parameter nor
the call. A cutoff is a count of rows, so anything that is not one is now refused by name.
"""

import pytest

from digitalearth.base.bigdata import (
    DEFAULT_BIG_DATA_THRESHOLD,
    validate_big_data_threshold,
)

CALLER = "InteractiveMap.points()"


class TestAWholeRowCountIsAccepted:
    """The values that really are a row count still pass, unchanged."""

    @pytest.mark.parametrize(
        ("given", "expected"), [(1000, 1000), (0, 0), (1000.0, 1000)]
    )
    def test_a_whole_count_comes_back_as_an_int(self, given, expected):
        """An integral value is the count it names, whether it was written as an ``int`` or a ``float``.

        Args:
            given: The cutoff as the caller wrote it.
            expected: The count it means.

        Test scenario:
            ``1000.0`` is a whole number of rows, so refusing it would be pedantry — the rule is about
            values that are *not* counts, not about which literal spelling was used.
        """
        assert validate_big_data_threshold(given, caller=CALLER) == expected

    def test_the_shared_default_survives_the_check(self):
        """The package default is itself a legal cutoff — the guard must never reject the fallback."""
        assert (
            validate_big_data_threshold(DEFAULT_BIG_DATA_THRESHOLD, caller=CALLER)
            == DEFAULT_BIG_DATA_THRESHOLD
        )


class TestAValueThatIsNotARowCountIsRefused:
    """L6 — a cutoff that cannot be a row count is named, not coerced into one."""

    def test_a_fraction_is_refused_instead_of_truncated(self):
        """``1.9`` used to become ``1`` silently, cutting over one row earlier than the caller asked.

        Test scenario:
            This is the actual defect: no error, no warning, and a routing decision one row off. The
            message has to name the keyword and the call, because the truncation is invisible in the
            output.
        """
        with pytest.raises(ValueError) as excinfo:
            validate_big_data_threshold(1.9, caller=CALLER)
        message = str(excinfo.value)
        assert message.startswith(CALLER), message
        assert "whole number of rows" in message and "1.9" in message, message

    @pytest.mark.parametrize(
        "given",
        [float("nan"), float("inf"), float("-inf"), "abc", None, True, False, "1000"],
    )
    def test_a_non_count_is_refused_with_the_parameter_named(self, given):
        """Every non-count answers the same way, naming ``big_data_threshold`` and the call.

        Args:
            given: A value that is not a whole number of rows.

        Test scenario:
            ``int("abc")`` and ``int(nan)`` raised a bare ``ValueError``, ``int(None)`` a ``TypeError`` and
            ``int(inf)`` an ``OverflowError`` — three different classes, none of them mentioning the
            keyword, and only the negative case was documented in ``Raises:``. ``True`` is in the list
            because ``int(True)`` is ``1``: a flag silently became a one-row cutoff.
        """
        with pytest.raises(ValueError) as excinfo:
            validate_big_data_threshold(given, caller=CALLER)
        message = str(excinfo.value)
        assert message.startswith(CALLER), message
        assert "whole number of rows" in message, message

    def test_a_negative_cutoff_keeps_its_own_message(self):
        """A negative cutoff is a different mistake (an "unlimited" sentinel) and keeps its own wording."""
        with pytest.raises(ValueError, match="must not be negative"):
            validate_big_data_threshold(-1, caller=CALLER)
