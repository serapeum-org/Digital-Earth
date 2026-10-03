"""``InteractiveMap.field(..., scheme=, k=)`` — a raster coloured class by class on the HoloViz tier.

Expected values are computed independently of the classifier; cross-tier agreement is pinned in
``tests/test_cross_tier_raster_classes.py``.
"""

import numpy as np
import pytest

hv = pytest.importorskip("holoviews")
pytest.importorskip("geoviews")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.base.symbology import (  # noqa: E402
    categorical_colors,
    resolve_categorical_cmap,
    sample_cmap,
)
from digitalearth.interactive import InteractiveMap  # noqa: E402

CODES = np.array([[1.0, 2.0, 5.0], [5.0, 1.0, 2.0]])


def _band(values: np.ndarray):
    """Wrap ``values`` as a lon/lat source."""
    rows, columns = values.shape
    return get_source(
        values,
        x=np.linspace(10.0, 10.0 + 0.1 * (columns - 1), columns),
        y=np.linspace(50.0, 50.0 - 0.1 * (rows - 1), rows),
        crs=4326,
    )


def _options(element, group: str) -> dict:
    """Return the Bokeh options of ``group`` HoloViews resolved for ``element``."""
    return hv.Store.lookup_options("bokeh", element, group).kwargs


class TestClassifiedInteractiveField:
    """A ``scheme`` steps the image's colours at its class edges and records the classes."""

    def test_codes_step_at_half_steps_in_the_shared_palette(self):
        """Each code is a class of its own, the bar steps between codes and is ticked at them.

        Test scenario:
            Codes 1, 2, 5: colours are the categorical palette, ``color_levels`` the half-steps
            ``0.5, 1.5, 3.5, 5.5``, ``clim`` their span, and the colour bar's ticks the codes.
        """
        m = InteractiveMap(crs=4326).field(
            _band(CODES), scheme="categorical", name="cover"
        )
        element = m.layers[0]
        style, plot = _options(element, "style"), _options(element, "plot")
        palette = categorical_colors([1, 2, 5], resolve_categorical_cmap(None))[1]
        assert list(style["cmap"]) == list(palette), f"cmap {style['cmap']}"
        assert list(plot["color_levels"]) == [0.5, 1.5, 3.5, 5.5], (
            f"levels {plot['color_levels']}"
        )
        assert tuple(plot["clim"]) == (0.5, 5.5), f"clim {plot['clim']}"
        assert list(plot["colorbar_opts"]["ticker"].ticks) == [1, 2, 5], (
            "the bar should be ticked at the codes"
        )
        scale = m.figure_spec.layers.get("cover").symbology.encoding("color").scale
        assert list(scale.categories) == [1, 2, 5], f"categories {scale.categories}"

    def test_graduated_classes_step_at_their_edges(self):
        """Quantile classes take ``cmap``'s samples and step at the quantiles.

        Test scenario:
            ``0..11`` in four quantile classes of Set2; the levels are ``numpy``'s quantiles.
        """
        values = np.arange(12.0).reshape(3, 4)
        m = InteractiveMap(crs=4326).field(
            _band(values), scheme="quantiles", k=4, cmap="Set2", name="ramp"
        )
        element = m.layers[0]
        expected = list(np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0]))
        assert list(_options(element, "style")["cmap"]) == sample_cmap("Set2", 4), (
            "cmap should be Set2's samples"
        )
        assert list(_options(element, "plot")["color_levels"]) == pytest.approx(
            expected
        ), "levels are quantiles"
        recorded = m.figure_spec.layers.get("ramp").symbology.props
        assert (recorded["scheme"], recorded["k"]) == ("quantiles", 4), (
            f"recorded {recorded['scheme']}"
        )

    def test_without_a_scheme_the_ramp_is_continuous(self):
        """No ``scheme`` leaves the image a continuous ramp with no colour levels.

        Test scenario:
            The default must not step: ``color_levels`` stays unset.
        """
        m = InteractiveMap(crs=4326).field(_band(CODES), name="plain")
        assert "color_levels" not in _options(m.layers[0], "plot"), (
            "an unclassified field should not step"
        )

    def test_a_fractional_band_is_refused_as_categorical(self):
        """``scheme="categorical"`` on a magnitude is refused before anything is added.

        Test scenario:
            ``0.5`` is no code; the map stays empty.
        """
        m = InteractiveMap(crs=4326)
        band = _band(np.array([[0.5, 1.0], [2.0, 3.0]]))
        with pytest.raises(ValueError, match="integer class codes"):
            m.field(band, scheme="categorical")
        assert m.layer_ids == [], f"a refused layer was added: {m.layer_ids}"
