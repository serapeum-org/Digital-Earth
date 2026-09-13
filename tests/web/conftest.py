"""Shared fixtures for the web-tier tests.

The tier logs through **loguru**, which pytest's ``caplog`` does not see: loguru keeps its own handler
list rather than routing through :mod:`logging`. Every test that proves a skip was announced therefore
attaches a loguru sink instead, and :func:`warning_log` is that sink, shared so the C7 skip-and-warn tests
across the tier all read the same.
"""

import pytest


@pytest.fixture
def warning_log() -> list[str]:
    """Collect the tier's loguru warnings for the duration of one test.

    Yields:
        A list that grows with every ``WARNING``-or-worse record emitted while the test runs, as strings
        (loguru hands a sink its formatted message, which is what these assertions match on).
    """
    from loguru import logger

    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink)
