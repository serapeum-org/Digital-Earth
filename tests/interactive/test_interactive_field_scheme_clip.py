"""`InteractiveMap.field(scheme=, limits=)` pins a graduated raster to the value window (#378).

The HoloViz tier spells the colour limits `limits=(low, high)` (the Core channel keyword, #332); under a
graduated scheme they pin the classes to that window so an outlier lands in an end class instead of stretching
them. A categorical scheme has no window to pin, so the `limits` pass through as the colour domain and raise
nothing. Edges are computed independently with numpy.
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
    """`limits=` pins a graduated field to the window and is a harmless domain for a categorical one."""

    def test_limits_pin_a_graduated_field_to_the_window(self):
        """`scheme="equal_interval", limits=(0, 4)` records edges 0, 2, 4, with the 999 outlier clamped."""
        m = InteractiveMap(crs=4326).field(
            _band(OUTLIER), scheme="equal_interval", k=2, limits=(0.0, 4.0), name="r"
        )
        assert list(_scale(m, "r").breaks) == pytest.approx([0.0, 2.0, 4.0]), list(
            _scale(m, "r").breaks
        )

    def test_without_limits_the_outlier_stretches_the_classes(self):
        """The same field with no `limits` records an edge at 999 — the ignored-limits bug, reproduced."""
        m = InteractiveMap(crs=4326).field(
            _band(OUTLIER), scheme="equal_interval", k=2, name="r"
        )
        assert list(_scale(m, "r").breaks)[-1] == 999.0, list(_scale(m, "r").breaks)

    def test_categorical_with_limits_keys_every_code_without_raising(self):
        """Categorical codes have no window to pin, so the `limits` touch nothing and all codes are keyed."""
        m = InteractiveMap(crs=4326).field(
            _band(np.array([[1.0, 2.0], [2.0, 5.0]])),
            scheme="categorical",
            limits=(0.0, 1.0),
            name="c",
        )
        assert list(_scale(m, "c").categories) == [1, 2, 5], list(
            _scale(m, "c").categories
        )
