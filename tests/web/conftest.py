"""Shared fixtures for the web-tier tests.

The tier logs through **loguru**, which pytest's ``caplog`` does not see: loguru keeps its own handler
list rather than routing through :mod:`logging`. Every test that proves a skip was announced therefore
attaches a loguru sink instead, and :func:`warning_log` is that sink, shared so the C7 skip-and-warn tests
across the tier all read the same.
"""

from collections.abc import Iterator

import pytest


@pytest.fixture
def warning_log() -> Iterator[list[str]]:
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


@pytest.fixture
def mercator_rgb():
    """Return a three-band raster in EPSG:3857 whose warp to EPSG:4326 reshapes the grid.

    `examples/data/acc4000.tif` warps from EPSG:32618 to EPSG:4326 without changing its shape or its values,
    so a builder that drew the unwarped pixels on the warped corners passed every test written on it (review
    H1). These 6 x 8 cells of ~1.1 km near Amsterdam warp to a 4 x 9 grid, so the two can be told apart.

    Returns:
        A pyramids `Dataset` with three bands of distinct values and a `-9999` NoData value.
    """
    import numpy as np
    from pyramids.dataset import Dataset, GeoReference

    values = np.random.default_rng(7).random((3, 6, 8)).astype("float32") * 100
    return Dataset.from_array(
        arr=values,
        geo_ref=GeoReference(
            geo=(545000.0, 1100.0, 0.0, 6868000.0, 0.0, -1100.0), epsg=3857
        ),
        no_data_value=-9999.0,
    )


@pytest.fixture
def warp_counter(monkeypatch) -> dict:
    """Count the display warps a build performs, through both of the tier's warp paths.

    A raster reaches the display CRS either as a whole dataset (`WebMapBase._to_display_raster`, which calls
    the name `reproject` imported into :mod:`digitalearth.web.base`) or as one band
    (:func:`digitalearth.base.display.to_display_source`, which calls its own module's `reproject`). Both
    names are wrapped, so a warp repeated on either path is counted.

    Args:
        monkeypatch: pytest's patcher, which restores both names after the test.

    Returns:
        A dict whose `"warps"` entry counts the calls made so far.
    """
    from digitalearth.base import display
    from digitalearth.web import base as web_base

    counted = {"warps": 0}
    real = display.reproject

    def counting(data, crs):
        """Count one warp, then perform it.

        Args:
            data: The dataset being warped.
            crs: The CRS it is warped to.

        Returns:
            The warped dataset.
        """
        counted["warps"] += 1
        return real(data, crs)

    monkeypatch.setattr(display, "reproject", counting)
    monkeypatch.setattr(web_base, "reproject", counting)
    return counted
