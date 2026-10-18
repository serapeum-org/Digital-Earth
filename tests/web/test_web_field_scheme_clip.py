"""`WebMap.field(scheme=, limits=)` confines a graduated raster to the value window (#378).

The web tier spells the colour limits `limits=` (the cross-tier contract keyword, #299) or legacy
`vmin`/`vmax`; either way a graduated scheme now cuts its classes inside that window so a few outliers cannot
stretch them. A categorical scheme has no window to cut, so the limits pass through as the colour domain and
raise nothing. Edges are computed independently with numpy over the in-range cells.
"""

import numpy as np
import pytest

pytest.importorskip("maplibre", reason="web tier engine")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.web import WebMap  # noqa: E402

OUTLIER = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 999.0]])


def _band(values: np.ndarray):
    """Wrap ``values`` as a lon/lat source the web tier draws without warping."""
    rows, columns = values.shape
    return get_source(
        values,
        x=np.linspace(10.0, 10.0 + 0.1 * (columns - 1), columns),
        y=np.linspace(50.0, 50.0 - 0.1 * (rows - 1), rows),
        crs=4326,
    )


class TestWebFieldSchemeClip:
    """`limits=` confines a graduated field and is a harmless domain for a categorical one."""

    def test_limits_confine_a_graduated_field(self):
        """`scheme="equal_interval", limits=(0, 4)` keys edges 0, 2, 4, ignoring the 999 outlier."""
        web_map = WebMap().field(
            _band(OUTLIER), scheme="equal_interval", k=2, limits=(0.0, 4.0), name="r"
        )
        assert web_map.last_breaks == pytest.approx([0.0, 2.0, 4.0]), (
            web_map.last_breaks
        )

    def test_without_limits_the_outlier_stretches_the_classes(self):
        """The same field with no limits keys an edge at 999 — the ignored-limits bug, reproduced."""
        web_map = WebMap().field(_band(OUTLIER), scheme="equal_interval", k=2, name="r")
        assert web_map.last_breaks[-1] == 999.0, web_map.last_breaks

    def test_categorical_with_limits_keys_every_code_without_raising(self):
        """Categorical codes have no window to cut, so the limits touch nothing and all codes are keyed."""
        web_map = WebMap().field(
            _band(np.array([[1.0, 2.0], [2.0, 5.0]])),
            scheme="categorical",
            limits=(0.0, 1.0),
            name="c",
        )
        assert web_map.last_breaks == [1, 2, 5], web_map.last_breaks
