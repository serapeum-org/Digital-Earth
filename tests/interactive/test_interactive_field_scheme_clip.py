"""`InteractiveMap.field(scheme=, clim=)` confines a graduated raster to the value window (#378).

The HoloViz tier spells the colour limits `clim=(low, high)`; under a graduated scheme they now cut the
classes inside that window so outliers cannot stretch them. A categorical scheme has no window to cut, so the
`clim` passes through as the colour domain and raises nothing. Edges are computed independently with numpy.
"""

import numpy as np
import pytest

pytest.importorskip("holoviews")
pytest.importorskip("geoviews")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.interactive import InteractiveMap  # noqa: E402

OUTLIER = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 999.0]])


def _band(values: np.ndarray):
    """Wrap ``values`` as a lon/lat source."""
    rows, columns = values.shape
    return get_source(
        values,
        x=np.linspace(10.0, 10.0 + 0.1 * (columns - 1), columns),
        y=np.linspace(50.0, 50.0 - 0.1 * (rows - 1), rows),
        crs=4326,
    )


def _scale(interactive_map: InteractiveMap, name: str):
    """Return the recorded colour scale of the named layer."""
    return (
        interactive_map.figure_spec.layers.get(name).symbology.encoding("color").scale
    )


class TestInteractiveFieldSchemeClip:
    """`clim=` confines a graduated field and is a harmless domain for a categorical one."""

    def test_clim_confines_a_graduated_field(self):
        """`scheme="equal_interval", clim=(0, 4)` records edges 0, 2, 4, ignoring the 999 outlier."""
        m = InteractiveMap(crs=4326).field(
            _band(OUTLIER), scheme="equal_interval", k=2, clim=(0.0, 4.0), name="r"
        )
        assert list(_scale(m, "r").breaks) == pytest.approx([0.0, 2.0, 4.0]), list(
            _scale(m, "r").breaks
        )

    def test_without_clim_the_outlier_stretches_the_classes(self):
        """The same field with no `clim` records an edge at 999 — the ignored-limits bug, reproduced."""
        m = InteractiveMap(crs=4326).field(
            _band(OUTLIER), scheme="equal_interval", k=2, name="r"
        )
        assert list(_scale(m, "r").breaks)[-1] == 999.0, list(_scale(m, "r").breaks)

    def test_categorical_with_clim_keys_every_code_without_raising(self):
        """Categorical codes have no window to cut, so the `clim` touches nothing and all codes are keyed."""
        m = InteractiveMap(crs=4326).field(
            _band(np.array([[1.0, 2.0], [2.0, 5.0]])),
            scheme="categorical",
            clim=(0.0, 1.0),
            name="c",
        )
        assert list(_scale(m, "c").categories) == [1, 2, 5], list(
            _scale(m, "c").categories
        )
